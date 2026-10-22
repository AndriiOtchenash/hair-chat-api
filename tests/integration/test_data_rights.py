import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy import func, select

from app.core.auth0_mgmt import Auth0Management, get_auth0_management
from app.core.security import get_token_verifier
from app.db.models import (
    Consent,
    ConsentType,
    Conversation,
    LLMCall,
    LLMCallStatus,
    Message,
    MessageRole,
    User,
)
from app.db.session import get_sessionmaker
from app.main import create_app
from tests.auth_fakes import FakeAuth0, SigningKey, bearer, mint

KEY = SigningKey("key-1")


class Auth0Down:
    """Management API answering 503 to everything; counts the attempts."""

    def __init__(self) -> None:
        self.requests = 0

    def handle(self, _: httpx.Request) -> httpx.Response:
        self.requests += 1
        return httpx.Response(503)


@pytest.fixture
def auth0() -> Auth0Down:
    return Auth0Down()


@pytest.fixture
async def client(app_settings_env: None, auth0: Auth0Down) -> AsyncIterator[AsyncClient]:
    app = create_app()
    verifier = FakeAuth0(KEY).verifier()
    management = Auth0Management(
        "test-tenant.auth0.test",
        "m2m",
        SecretStr("secret"),
        httpx.AsyncClient(transport=httpx.MockTransport(auth0.handle)),
    )
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    app.dependency_overrides[get_auth0_management] = lambda: management
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await verifier.aclose()
    await management.aclose()


@dataclass
class Seeded:
    auth: dict[str, str]
    user_id: uuid.UUID
    conversation_ids: set[uuid.UUID]
    message_ids: set[uuid.UUID]
    call_id: uuid.UUID
    marker: str


async def _seed(client: AsyncClient, name: str) -> Seeded:
    """A user with a revoked old consent, current consents, two conversations (one with a
    summary) and an answer linked to its llm_call."""
    auth = bearer(mint(KEY, sub=f"auth0|{name}-{uuid.uuid4()}"))
    user_id = uuid.UUID((await client.get("/me", headers=auth)).json()["id"])
    marker = f"{name}-{uuid.uuid4()}"
    grant = {"types": [t.value for t in ConsentType], "version": "2026-10-v1"}
    assert (await client.post("/consents", headers=auth, json=grant)).status_code < 300
    async with get_sessionmaker()() as session:
        session.add(
            Consent(
                user_id=user_id,
                type=ConsentType.AI_PROCESSING,
                version="old",
                revoked_at=func.now(),
            )
        )
        conversations = [Conversation(user_id=user_id, title=marker) for _ in range(2)]
        session.add_all(conversations)
        await session.flush()
        question = Message(
            conversation_id=conversations[0].id,
            user_id=user_id,
            role=MessageRole.USER,
            content=f"question {marker}",
            moderation={"provider": "openai", "model": "m", "flagged": False, "action": "allow"},
        )
        answer = Message(
            conversation_id=conversations[0].id,
            user_id=user_id,
            role=MessageRole.ASSISTANT,
            content=f"answer {marker}",
            provider="fake",
            model="m1",
        )
        session.add_all([question, answer])
        await session.flush()
        call = LLMCall(message_id=answer.id, provider="fake", model="m1", status=LLMCallStatus.OK)
        session.add(call)
        conversations[0].summary = f"summary {marker}"
        conversations[0].summary_upto_message_id = question.id
        await session.commit()
        return Seeded(
            auth,
            user_id,
            {c.id for c in conversations},
            {question.id, answer.id},
            call.id,
            marker,
        )


async def _rows(user_id: uuid.UUID, call_id: uuid.UUID) -> dict[str, int]:
    async with get_sessionmaker()() as session:
        counts = {
            model.__tablename__: await session.scalar(
                select(func.count()).select_from(model).where(column == user_id)
            )
            or 0
            for model, column in (
                (User, User.id),
                (Consent, Consent.user_id),
                (Conversation, Conversation.user_id),
                (Message, Message.user_id),
            )
        }
        counts["llm_calls"] = (
            await session.scalar(select(func.count()).where(LLMCall.id == call_id)) or 0
        )
    return counts


async def test_export_contains_exactly_own_data(client: AsyncClient) -> None:
    alice = await _seed(client, "alice")
    bob = await _seed(client, "bob")

    response = await client.get("/me/export", headers=alice.auth)

    assert response.status_code == 200, response.text
    assert response.headers["content-disposition"].startswith(
        'attachment; filename="hair-chat-export-'
    )
    body = response.json()
    assert body["user"]["id"] == str(alice.user_id)
    assert {uuid.UUID(c["id"]) for c in body["conversations"]} == alice.conversation_ids
    messages = [m for c in body["conversations"] for m in c["messages"]]
    assert {uuid.UUID(m["id"]) for m in messages} == alice.message_ids
    assert {(m["role"], m["provider"], m["model"], m["moderation"]) for m in messages} == {
        ("user", None, None, "allow"),
        ("assistant", "fake", "m1", None),
    }
    assert f"summary {alice.marker}" in {c["summary"] for c in body["conversations"]}
    assert sorted((c["version"], c["revoked_at"] is None) for c in body["consents"]) == sorted(
        [("old", False)] + [("2026-10-v1", True)] * len(ConsentType)
    )
    assert bob.marker not in response.text
    assert str(bob.user_id) not in response.text


async def test_export_still_works_after_consents_are_revoked(client: AsyncClient) -> None:
    alice = await _seed(client, "alice")
    for t in ConsentType:
        assert (await client.delete(f"/consents/{t.value}", headers=alice.auth)).status_code == 204

    response = await client.get("/me/export", headers=alice.auth)

    assert response.status_code == 200, response.text
    assert all(c["revoked_at"] is not None for c in response.json()["consents"])
    assert alice.marker in response.text


async def test_delete_cascades_everything_and_leaves_other_user_alone(
    client: AsyncClient, auth0: Auth0Down
) -> None:
    alice = await _seed(client, "alice")
    bob = await _seed(client, "bob")
    bob_before = await _rows(bob.user_id, bob.call_id)

    response = await client.request("DELETE", "/me", headers=alice.auth, json={"confirm": "DELETE"})

    # The Management API is down (503): our deletion still went through.
    assert response.status_code == 204, response.text
    assert auth0.requests == 1
    assert await _rows(alice.user_id, alice.call_id) == {
        "users": 0,
        "consents": 0,
        "conversations": 0,
        "messages": 0,
        "llm_calls": 0,
    }
    assert await _rows(bob.user_id, bob.call_id) == bob_before
    assert bob_before["messages"] == 2 and bob_before["llm_calls"] == 1


@pytest.mark.parametrize("body", [{"confirm": "delete"}, {}])
async def test_delete_with_wrong_confirm_is_422_and_deletes_nothing(
    client: AsyncClient, body: dict[str, str]
) -> None:
    alice = await _seed(client, "alice")
    before = await _rows(alice.user_id, alice.call_id)

    response = await client.request("DELETE", "/me", headers=alice.auth, json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert await _rows(alice.user_id, alice.call_id) == before
