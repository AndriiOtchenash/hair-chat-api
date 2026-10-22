import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.account import get_account_service
from app.core.auth0_mgmt import Auth0ManagementError
from app.core.consents import get_consent_repository
from app.core.security import get_token_verifier, get_user_repository
from app.db.models import ConsentType
from app.main import create_app
from app.services.account import AccountService
from tests.auth_fakes import FakeAuth0, SigningKey, bearer, mint
from tests.fakes import (
    FakeAccountRepository,
    FakeConsentRepository,
    FakeConversationRepository,
    FakeIdentityProvider,
    FakeUserRepository,
)

KEY = SigningKey("key-1")
ALICE = bearer(mint(KEY, sub="auth0|alice"))
BOB = bearer(mint(KEY, sub="auth0|bob"))
NOW = datetime(2026, 10, 22, 23, 59, tzinfo=UTC)


@dataclass
class World:
    client: AsyncClient
    users: FakeUserRepository
    consents: FakeConsentRepository
    conversations: FakeConversationRepository
    identities: FakeIdentityProvider


async def _world(identities: FakeIdentityProvider | None) -> AsyncIterator[World]:
    app = create_app()
    verifier = FakeAuth0(KEY).verifier()
    users, consents, conversations = (
        FakeUserRepository(),
        FakeConsentRepository(),
        FakeConversationRepository(),
    )
    account = FakeAccountRepository(users, consents, conversations)
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    app.dependency_overrides[get_user_repository] = lambda: users
    app.dependency_overrides[get_consent_repository] = lambda: consents
    app.dependency_overrides[get_account_service] = lambda: AccountService(
        account, identities, clock=lambda: NOW
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield World(c, users, consents, conversations, identities or FakeIdentityProvider())
    await verifier.aclose()


@pytest.fixture
async def world() -> AsyncIterator[World]:
    async for w in _world(FakeIdentityProvider()):
        yield w


async def _seed(world: World, auth: dict[str, str], text: str) -> None:
    """Creates the user with a revoked and an active consent and one answered conversation."""
    me = (await world.client.get("/me", headers=auth)).json()
    user = next(u for u in world.users.by_sub.values() if str(u.id) == me["id"])
    await world.consents.grant(user.id, [ConsentType.AI_PROCESSING], "old")
    await world.consents.revoke(user.id, ConsentType.AI_PROCESSING)
    await world.consents.grant(user.id, list(ConsentType), "2026-10-v1")
    conversation = await world.conversations.create(user.id)
    await world.conversations.add_user_message(
        user.id, conversation.id, text, moderation={"action": "soft", "categories": ["self-harm"]}
    )
    await world.conversations.add_assistant_message(
        user.id,
        conversation.id,
        f"answer to {text}",
        provider="fake",
        model="m1",
        tokens_in=1,
        tokens_out=2,
        llm_call_ids=[],
    )


async def test_export_is_a_json_download_of_own_data_only(world: World) -> None:
    await _seed(world, ALICE, "alice's hair")
    await _seed(world, BOB, "bob's hair")

    response = await world.client.get("/me/export", headers=ALICE)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    assert response.headers["content-disposition"] == (
        'attachment; filename="hair-chat-export-2026-10-22.json"'
    )
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["format_version"] == 1
    assert body["user"]["auth_sub"] == "auth0|alice"
    consents = sorted((c["type"], c["version"], c["revoked_at"] is None) for c in body["consents"])
    assert consents == sorted(
        [("ai_processing", "old", False), *((t.value, "2026-10-v1", True) for t in ConsentType)]
    )
    [conversation] = body["conversations"]
    assert conversation["title"] == "alice's hair"
    assert [
        (m["role"], m["content"], m["provider"], m["model"], m["moderation"])
        for m in conversation["messages"]
    ] == [
        ("user", "alice's hair", None, None, "soft"),
        ("assistant", "answer to alice's hair", "fake", "m1", None),
    ]
    assert "bob" not in response.text
    assert "categories" not in response.text


async def test_export_works_without_any_consent(world: World) -> None:
    response = await world.client.get("/me/export", headers=ALICE)

    assert response.status_code == 200, response.text
    assert response.json()["conversations"] == []


async def test_delete_removes_only_own_account_and_calls_auth0(world: World) -> None:
    await _seed(world, ALICE, "alice's hair")
    await _seed(world, BOB, "bob's hair")
    alice_id = (await world.client.get("/me", headers=ALICE)).json()["id"]

    response = await world.client.request(
        "DELETE", "/me", headers=ALICE, json={"confirm": "DELETE"}
    )

    assert response.status_code == 204, response.text
    assert response.content == b""
    assert set(world.users.by_sub) == {"auth0|bob"}
    assert {m.content for m in world.conversations.messages} == {
        "bob's hair",
        "answer to bob's hair",
    }
    assert world.identities.deleted == ["auth0|alice"]
    # A still-valid token meets a fresh, empty account (ADR 0006).
    after = (await world.client.get("/me", headers=ALICE)).json()
    assert after["id"] != alice_id


@pytest.mark.parametrize(
    "body", [{}, {"confirm": "delete"}, {"confirm": "yes"}, {"confirm": None}, None]
)
async def test_delete_without_exact_confirm_is_422_and_keeps_data(
    world: World, body: dict[str, object] | None
) -> None:
    await _seed(world, ALICE, "alice's hair")

    response = await world.client.request("DELETE", "/me", headers=ALICE, json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert set(world.users.by_sub) == {"auth0|alice"}
    assert len(world.conversations.messages) == 2
    assert world.identities.deleted == []


@pytest.mark.parametrize(("method", "path"), [("GET", "/me/export"), ("DELETE", "/me")])
async def test_without_token_is_401(world: World, method: str, path: str) -> None:
    response = await world.client.request(method, path, json={"confirm": "DELETE"})

    assert response.status_code == 401


async def test_auth0_failure_does_not_undo_deletion(caplog: pytest.LogCaptureFixture) -> None:
    failing = FakeIdentityProvider(Auth0ManagementError("delete user: HTTP 503"))
    async for world in _world(failing):
        await _seed(world, ALICE, "alice's hair")
        with caplog.at_level(logging.INFO):
            response = await world.client.request(
                "DELETE", "/me", headers=ALICE, json={"confirm": "DELETE"}
            )

        assert response.status_code == 204
        assert world.users.by_sub == {}
        assert world.conversations.messages == []
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "auth_sub=auth0|alice" in errors[0].getMessage()
    assert "HTTP 503" in errors[0].getMessage()
    assert "alice's hair" not in caplog.text


async def test_auth0_not_configured_deletes_locally_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async for world in _world(None):
        await _seed(world, ALICE, "alice's hair")
        with caplog.at_level(logging.INFO):
            response = await world.client.request(
                "DELETE", "/me", headers=ALICE, json={"confirm": "DELETE"}
            )

        assert response.status_code == 204
        assert world.users.by_sub == {}
    assert any(
        r.levelno == logging.WARNING and "deleted locally only" in r.getMessage()
        for r in caplog.records
    )
