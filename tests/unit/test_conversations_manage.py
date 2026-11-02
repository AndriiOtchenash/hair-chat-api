import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.chat import get_conversation_locks, get_conversation_repository
from app.core.consents import get_consent_repository
from app.core.security import get_token_verifier, get_user_repository
from app.main import create_app
from app.services.chat_service import ConversationLocks
from tests.auth_fakes import FakeAuth0, SigningKey, bearer, mint
from tests.fakes import FakeConsentRepository, FakeConversationRepository, FakeUserRepository

KEY = SigningKey("key-1")
ALICE = bearer(mint(KEY, sub="auth0|alice"))
BOB = bearer(mint(KEY, sub="auth0|bob"))
MISSING = "01890000-0000-7000-8000-00000000dead"


@dataclass
class World:
    client: AsyncClient
    users: FakeUserRepository
    conversations: FakeConversationRepository
    locks: ConversationLocks

    async def user_id(self, auth: dict[str, str]) -> uuid.UUID:
        """Creates the user (no consents) through GET /me."""
        return uuid.UUID((await self.client.get("/me", headers=auth)).json()["id"])

    async def conversation(self, auth: dict[str, str], first_message: str | None = None) -> str:
        user_id = await self.user_id(auth)
        conversation = await self.conversations.create(user_id)
        if first_message is not None:
            await self.conversations.add_user_message(user_id, conversation.id, first_message)
        return str(conversation.id)


@pytest.fixture
async def world() -> AsyncIterator[World]:
    """No consents are granted: these endpoints must work without them."""
    app = create_app()
    verifier = FakeAuth0(KEY).verifier()
    users, conversations, locks = (
        FakeUserRepository(),
        FakeConversationRepository(),
        ConversationLocks(),
    )
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    app.dependency_overrides[get_user_repository] = lambda: users
    app.dependency_overrides[get_consent_repository] = FakeConsentRepository
    app.dependency_overrides[get_conversation_repository] = lambda: conversations
    app.dependency_overrides[get_conversation_locks] = lambda: locks
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield World(c, users, conversations, locks)
    await verifier.aclose()


# --- PATCH /conversations/{id} -------------------------------------------------------------


async def test_rename_strips_the_title_and_returns_the_conversation(world: World) -> None:
    conversation_id = await world.conversation(ALICE)

    response = await world.client.patch(
        f"/conversations/{conversation_id}", json={"title": "  Сухие кончики  "}, headers=ALICE
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["id"], body["title"]) == (conversation_id, "Сухие кончики")


async def test_rename_accepts_120_chars(world: World) -> None:
    conversation_id = await world.conversation(ALICE)

    response = await world.client.patch(
        f"/conversations/{conversation_id}", json={"title": "x" * 120}, headers=ALICE
    )

    assert response.json()["title"] == "x" * 120


@pytest.mark.parametrize("body", [{"title": ""}, {"title": "   "}, {"title": "x" * 121}, {}])
async def test_rename_rejects_empty_or_too_long_title(world: World, body: dict[str, str]) -> None:
    conversation_id = await world.conversation(ALICE, "first")

    response = await world.client.patch(
        f"/conversations/{conversation_id}", json=body, headers=ALICE
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert world.conversations.conversations[uuid.UUID(conversation_id)].title == "first"


@pytest.mark.parametrize("target", ["someone_elses", "missing"])
async def test_rename_of_someone_elses_or_missing_conversation_is_404(
    world: World, target: str
) -> None:
    bobs = await world.conversation(BOB, "bob's question")
    conversation_id = bobs if target == "someone_elses" else MISSING

    response = await world.client.patch(
        f"/conversations/{conversation_id}", json={"title": "mine now"}, headers=ALICE
    )

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert world.conversations.conversations[uuid.UUID(bobs)].title == "bob's question"


async def test_first_message_does_not_replace_a_title_set_by_the_user(world: World) -> None:
    alice = await world.user_id(ALICE)
    conversation_id = await world.conversation(ALICE)
    await world.client.patch(
        f"/conversations/{conversation_id}", json={"title": "Окрашивание"}, headers=ALICE
    )

    await world.conversations.add_user_message(alice, uuid.UUID(conversation_id), "first")

    assert world.conversations.conversations[uuid.UUID(conversation_id)].title == "Окрашивание"


# --- DELETE /conversations/{id} ------------------------------------------------------------


async def test_delete_removes_the_conversation_with_its_messages(world: World) -> None:
    conversation_id = await world.conversation(ALICE, "question")
    kept = await world.conversation(ALICE, "another")

    response = await world.client.delete(f"/conversations/{conversation_id}", headers=ALICE)

    assert response.status_code == 204
    assert set(world.conversations.conversations) == {uuid.UUID(kept)}
    assert [m.content for m in world.conversations.messages] == ["another"]


@pytest.mark.parametrize("target", ["someone_elses", "missing"])
async def test_delete_of_someone_elses_or_missing_conversation_is_404(
    world: World, target: str
) -> None:
    bobs = await world.conversation(BOB, "bob's question")
    conversation_id = bobs if target == "someone_elses" else MISSING

    response = await world.client.delete(f"/conversations/{conversation_id}", headers=ALICE)

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert uuid.UUID(bobs) in world.conversations.conversations


async def test_delete_while_an_answer_is_generated_is_409(world: World) -> None:
    conversation_id = await world.conversation(ALICE, "question")

    with world.locks.hold(uuid.UUID(conversation_id)):
        response = await world.client.delete(f"/conversations/{conversation_id}", headers=ALICE)

    assert response.status_code == 409
    assert response.json()["code"] == "conversation_busy"
    assert uuid.UUID(conversation_id) in world.conversations.conversations


async def test_someone_elses_busy_conversation_is_404_not_409(world: World) -> None:
    bobs = await world.conversation(BOB, "bob's question")

    with world.locks.hold(uuid.UUID(bobs)):
        response = await world.client.delete(f"/conversations/{bobs}", headers=ALICE)

    assert response.status_code == 404


async def test_delete_frees_the_generation_slot(world: World) -> None:
    conversation_id = uuid.UUID(await world.conversation(ALICE, "question"))

    await world.client.delete(f"/conversations/{conversation_id}", headers=ALICE)

    assert not world.locks.is_busy(conversation_id)


# --- DELETE /conversations -----------------------------------------------------------------


async def test_delete_all_removes_only_own_conversations(world: World) -> None:
    await world.conversation(ALICE, "one")
    await world.conversation(ALICE, "two")
    bobs = await world.conversation(BOB, "bob's question")

    response = await world.client.request(
        "DELETE", "/conversations", json={"confirm": "DELETE_ALL"}, headers=ALICE
    )

    assert response.status_code == 204
    assert set(world.conversations.conversations) == {uuid.UUID(bobs)}
    assert [m.content for m in world.conversations.messages] == ["bob's question"]


async def test_delete_all_without_conversations_is_204(world: World) -> None:
    response = await world.client.request(
        "DELETE", "/conversations", json={"confirm": "DELETE_ALL"}, headers=ALICE
    )

    assert response.status_code == 204


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"json": {}}, {"json": {"confirm": "DELETE"}}, {"json": {"confirm": "delete_all"}}],
)
async def test_delete_all_without_confirmation_is_422(
    world: World, kwargs: dict[str, object]
) -> None:
    await world.conversation(ALICE, "one")

    response = await world.client.request("DELETE", "/conversations", headers=ALICE, **kwargs)  # type: ignore[arg-type]

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert len(world.conversations.conversations) == 1


async def test_delete_all_while_one_is_generating_is_409_and_deletes_nothing(
    world: World,
) -> None:
    first = await world.conversation(ALICE, "one")
    await world.conversation(ALICE, "two")

    with world.locks.hold(uuid.UUID(first)):
        response = await world.client.request(
            "DELETE", "/conversations", json={"confirm": "DELETE_ALL"}, headers=ALICE
        )

    assert response.status_code == 409
    assert response.json()["code"] == "conversation_busy"
    assert len(world.conversations.conversations) == 2
    assert not any(
        world.locks.is_busy(c) for c in world.conversations.conversations if str(c) != first
    )


async def test_deletes_log_ids_and_counts_only(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    conversation_id = await world.conversation(ALICE, "секретный вопрос про кожу головы")
    await world.conversation(ALICE, "второй вопрос")
    caplog.set_level(logging.INFO)

    await world.client.delete(f"/conversations/{conversation_id}", headers=ALICE)
    await world.client.request(
        "DELETE", "/conversations", json={"confirm": "DELETE_ALL"}, headers=ALICE
    )

    logged = caplog.text
    assert conversation_id in logged
    assert "count=1" in logged
    assert "вопрос" not in logged
