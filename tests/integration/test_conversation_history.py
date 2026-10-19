"""History after the summary point and the conditional summary save, against Postgres."""

import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Message
from app.db.repositories.conversations import SqlConversationRepository
from app.db.repositories.users import SqlUserRepository
from app.db.session import get_sessionmaker


@pytest.fixture
async def session(app_settings_env: None) -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as s:
        yield s


async def user_id(session: AsyncSession) -> uuid.UUID:
    return (await SqlUserRepository(session).get_or_create_by_auth_sub(f"t|{uuid.uuid4()}")).id


async def dialog(repo: SqlConversationRepository, owner: uuid.UUID, count: int) -> list[Message]:
    conversation = await repo.create(owner)
    return [await repo.add_user_message(owner, conversation.id, f"m{i}") for i in range(count)]


async def test_history_after_a_message_in_both_orders(session: AsyncSession) -> None:
    repo = SqlConversationRepository(session)
    owner = await user_id(session)
    messages = await dialog(repo, owner, 6)
    conversation_id = messages[0].conversation_id

    newest = await repo.history(
        owner, conversation_id, after=messages[1].id, limit=3, newest_first=True
    )
    oldest = await repo.history(
        owner, conversation_id, after=messages[1].id, limit=3, newest_first=False
    )
    everything = await repo.history(
        owner, conversation_id, after=None, limit=10, newest_first=False
    )

    assert [m.content for m in newest] == ["m5", "m4", "m3"]
    assert [m.content for m in oldest] == ["m2", "m3", "m4"]
    assert [m.content for m in everything] == [f"m{i}" for i in range(6)]


async def test_history_and_summary_of_someone_elses_conversation(session: AsyncSession) -> None:
    repo = SqlConversationRepository(session)
    owner, stranger = await user_id(session), await user_id(session)
    messages = await dialog(repo, owner, 3)
    conversation_id = messages[0].conversation_id

    assert (
        await repo.history(stranger, conversation_id, after=None, limit=10, newest_first=True) == []
    )
    assert not await repo.save_summary(
        stranger, conversation_id, "stolen", upto_message_id=messages[1].id, expected_upto=None
    )
    conversation = await repo.get(owner, conversation_id)
    assert conversation is not None and conversation.summary is None


async def test_summary_is_saved_only_over_the_expected_one(session: AsyncSession) -> None:
    repo = SqlConversationRepository(session)
    owner = await user_id(session)
    messages = await dialog(repo, owner, 4)
    conversation_id = messages[0].conversation_id

    assert await repo.save_summary(
        owner, conversation_id, "first", upto_message_id=messages[1].id, expected_upto=None
    )
    # A run that started before "first" was saved is refused.
    assert not await repo.save_summary(
        owner, conversation_id, "stale", upto_message_id=messages[0].id, expected_upto=None
    )
    assert await repo.save_summary(
        owner,
        conversation_id,
        "second",
        upto_message_id=messages[2].id,
        expected_upto=messages[1].id,
    )

    conversation = await repo.get(owner, conversation_id)
    assert conversation is not None
    await session.refresh(conversation)
    assert (conversation.summary, conversation.summary_upto_message_id) == (
        "second",
        messages[2].id,
    )
