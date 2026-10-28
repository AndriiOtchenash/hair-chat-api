import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from app.core.security import get_token_verifier
from app.db.models import (
    ConsentType,
    Conversation,
    LLMCall,
    LLMCallStatus,
    Message,
)
from app.db.repositories.conversations import SqlConversationRepository
from app.db.session import get_sessionmaker
from app.main import create_app
from tests.auth_fakes import FakeAuth0, SigningKey, bearer, mint

KEY = SigningKey("key-1")
DELETE_ALL = {"confirm": "DELETE_ALL"}


@pytest.fixture
async def client(app_settings_env: None) -> AsyncIterator[AsyncClient]:
    app = create_app()
    verifier = FakeAuth0(KEY).verifier()
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await verifier.aclose()


@dataclass
class Account:
    auth: dict[str, str]
    user_id: uuid.UUID


async def _account(client: AsyncClient) -> Account:
    """A new user with every current consent."""
    auth = bearer(mint(KEY, sub=f"auth0|{uuid.uuid4()}"))
    user_id = uuid.UUID((await client.get("/me", headers=auth)).json()["id"])
    grant = {"types": [t.value for t in ConsentType], "version": "2026-10-v1"}
    assert (await client.post("/consents", headers=auth, json=grant)).status_code < 300
    return Account(auth, user_id)


async def _answered_conversation(client: AsyncClient, account: Account) -> uuid.UUID:
    """A conversation (through the API) with a question and an answer linked to its llm_call."""
    created = await client.post("/conversations", headers=account.auth)
    conversation_id = uuid.UUID(created.json()["id"])
    async with get_sessionmaker()() as session:
        repo = SqlConversationRepository(session)
        await repo.add_user_message(account.user_id, conversation_id, "question")
        call = LLMCall(provider="fake", model="m1", status=LLMCallStatus.OK)
        session.add(call)
        await session.commit()
        await repo.add_assistant_message(
            account.user_id,
            conversation_id,
            "answer",
            provider="fake",
            model="m1",
            tokens_in=1,
            tokens_out=1,
            llm_call_ids=[call.id],
        )
    return conversation_id


async def _counts(conversation_ids: set[uuid.UUID]) -> tuple[int, int, int]:
    """Rows left: conversations, their messages, llm_calls linked to those messages."""
    async with get_sessionmaker()() as session:
        conversations = await session.scalar(
            select(func.count()).where(Conversation.id.in_(conversation_ids))
        )
        messages = await session.scalar(
            select(func.count()).where(Message.conversation_id.in_(conversation_ids))
        )
        calls = await session.scalar(
            select(func.count())
            .select_from(LLMCall)
            .join(Message, LLMCall.message_id == Message.id)
            .where(Message.conversation_id.in_(conversation_ids))
        )
        return conversations or 0, messages or 0, calls or 0


async def _llm_call_ids(conversation_id: uuid.UUID) -> set[uuid.UUID]:
    async with get_sessionmaker()() as session:
        rows = await session.scalars(
            select(LLMCall.id)
            .join(Message, LLMCall.message_id == Message.id)
            .where(Message.conversation_id == conversation_id)
        )
        return set(rows)


async def _revoke_all(client: AsyncClient, account: Account) -> None:
    for consent_type in ConsentType:
        response = await client.delete(f"/consents/{consent_type.value}", headers=account.auth)
        assert response.status_code == 204
    chat = await client.get("/conversations", headers=account.auth)
    assert chat.json()["code"] == "consent_required"


async def test_delete_cascades_to_messages_and_their_llm_calls(client: AsyncClient) -> None:
    alice = await _account(client)
    doomed = await _answered_conversation(client, alice)
    kept = await _answered_conversation(client, alice)
    calls_before = await _llm_call_ids(doomed)

    response = await client.delete(f"/conversations/{doomed}", headers=alice.auth)

    assert response.status_code == 204
    assert await _counts({doomed}) == (0, 0, 0)
    assert await _counts({kept}) == (1, 2, 1)
    async with get_sessionmaker()() as session:
        left = await session.scalar(select(func.count()).where(LLMCall.id.in_(calls_before)))
    assert left == 0


async def test_someone_elses_conversation_is_404_and_untouched(client: AsyncClient) -> None:
    alice, bob = await _account(client), await _account(client)
    bobs = await _answered_conversation(client, bob)

    deleted = await client.delete(f"/conversations/{bobs}", headers=alice.auth)
    renamed = await client.patch(
        f"/conversations/{bobs}", json={"title": "mine"}, headers=alice.auth
    )

    assert (deleted.status_code, renamed.status_code) == (404, 404)
    assert await _counts({bobs}) == (1, 2, 1)
    async with get_sessionmaker()() as session:
        assert await session.scalar(select(Conversation.title).where(Conversation.id == bobs)) == (
            "question"
        )


async def test_delete_all_removes_every_own_conversation_and_no_one_elses(
    client: AsyncClient,
) -> None:
    alice, bob = await _account(client), await _account(client)
    alices = {await _answered_conversation(client, alice) for _ in range(3)}
    bobs = await _answered_conversation(client, bob)

    response = await client.request("DELETE", "/conversations", json=DELETE_ALL, headers=alice.auth)

    assert response.status_code == 204
    assert await _counts(alices) == (0, 0, 0)
    assert await _counts({bobs}) == (1, 2, 1)
    listed = await client.get("/conversations", headers=alice.auth)
    assert listed.json()["items"] == []


async def test_delete_and_rename_work_after_consents_are_revoked(client: AsyncClient) -> None:
    alice = await _account(client)
    renamed_id = await _answered_conversation(client, alice)
    deleted_id = await _answered_conversation(client, alice)
    rest = {await _answered_conversation(client, alice) for _ in range(2)}
    await _revoke_all(client, alice)

    renamed = await client.patch(
        f"/conversations/{renamed_id}", json={"title": " Новое имя "}, headers=alice.auth
    )
    deleted = await client.delete(f"/conversations/{deleted_id}", headers=alice.auth)
    remaining = {renamed_id, *rest}
    assert await _counts(remaining) == (3, 6, 3)
    deleted_all = await client.request(
        "DELETE", "/conversations", json=DELETE_ALL, headers=alice.auth
    )

    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Новое имя"
    assert (deleted.status_code, deleted_all.status_code) == (204, 204)
    assert await _counts({deleted_id, *remaining}) == (0, 0, 0)


async def test_first_message_keeps_a_title_set_by_the_user(client: AsyncClient) -> None:
    alice = await _account(client)
    created = await client.post("/conversations", headers=alice.auth)
    conversation_id = uuid.UUID(created.json()["id"])
    await client.patch(
        f"/conversations/{conversation_id}", json={"title": "Окрашивание"}, headers=alice.auth
    )

    async with get_sessionmaker()() as session:
        await SqlConversationRepository(session).add_user_message(
            alice.user_id, conversation_id, "first question"
        )

    listed = await client.get("/conversations", headers=alice.auth)
    assert [c["title"] for c in listed.json()["items"]] == ["Окрашивание"]
