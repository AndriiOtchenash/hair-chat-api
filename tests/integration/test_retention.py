import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, func, select

from app.db.base import new_id
from app.db.models import Conversation, LLMCall, LLMCallStatus, Message, MessageRole, User
from app.db.repositories.retention import SqlRetentionRepository
from app.db.session import get_sessionmaker
from app.services.retention import purge

NOW = datetime.now(UTC)
DAY = timedelta(days=1)


@pytest.fixture
async def repo(app_settings_env: None) -> SqlRetentionRepository:
    # Other tests share the container: start from no conversations so counts are exact.
    async with get_sessionmaker()() as session:
        await session.execute(delete(Conversation))
        await session.execute(delete(LLMCall))
        await session.commit()
    return SqlRetentionRepository(get_sessionmaker())


async def _conversation(user_id: uuid.UUID, age: timedelta, messages: int) -> uuid.UUID:
    async with get_sessionmaker()() as session:
        conversation = Conversation(user_id=user_id, last_message_at=NOW - age)
        session.add(conversation)
        await session.flush()
        for i in range(messages):
            message = Message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.USER,
                content=f"m{i}",
            )
            session.add(message)
            await session.flush()
            session.add(
                LLMCall(message_id=message.id, provider="fake", model="m", status=LLMCallStatus.OK)
            )
        await session.commit()
        return conversation.id


async def _orphan_call(age: timedelta) -> uuid.UUID:
    async with get_sessionmaker()() as session:
        call = LLMCall(
            id=new_id(),
            provider="fake",
            model="m",
            status=LLMCallStatus.ERROR,
            created_at=NOW - age,
        )
        session.add(call)
        await session.commit()
        return call.id


async def _user() -> uuid.UUID:
    async with get_sessionmaker()() as session:
        user = User(auth_sub=f"auth0|{uuid.uuid4()}")
        session.add(user)
        await session.commit()
        return user.id


async def _existing(
    model: type[Conversation] | type[LLMCall], ids: list[uuid.UUID]
) -> set[uuid.UUID]:
    async with get_sessionmaker()() as session:
        return set(await session.scalars(select(model.id).where(model.id.in_(ids))))


async def _message_count() -> int:
    async with get_sessionmaker()() as session:
        return await session.scalar(select(func.count()).select_from(Message)) or 0


async def test_purge_deletes_only_old_conversations_with_their_messages(
    repo: SqlRetentionRepository,
) -> None:
    alice, bob = await _user(), await _user()
    old = [await _conversation(alice, 31 * DAY, 2), await _conversation(bob, 90 * DAY, 3)]
    fresh = [await _conversation(alice, 29 * DAY, 4), await _conversation(bob, DAY, 1)]
    old_orphan, fresh_orphan = await _orphan_call(31 * DAY), await _orphan_call(DAY)

    result = await purge(repo, now=NOW, days=30, dry_run=False, batch_size=1)

    assert (result.conversations, result.messages, result.orphan_llm_calls) == (2, 5, 1)
    assert await _existing(Conversation, old + fresh) == set(fresh)
    assert await _message_count() == 5
    assert await _existing(LLMCall, [old_orphan, fresh_orphan]) == {fresh_orphan}
    async with get_sessionmaker()() as session:
        linked = await session.scalar(
            select(func.count()).select_from(LLMCall).where(LLMCall.message_id.is_not(None))
        )
    assert linked == 5  # the llm_calls of deleted messages went with them

    again = await purge(repo, now=NOW, days=30, dry_run=False)
    assert (again.conversations, again.messages, again.orphan_llm_calls) == (0, 0, 0)


async def test_dry_run_deletes_nothing(repo: SqlRetentionRepository) -> None:
    alice = await _user()
    old = await _conversation(alice, 31 * DAY, 2)
    orphan = await _orphan_call(40 * DAY)

    result = await purge(repo, now=NOW, days=30, dry_run=True)

    assert (result.conversations, result.messages, result.orphan_llm_calls) == (1, 2, 1)
    assert await _existing(Conversation, [old]) == {old}
    assert await _existing(LLMCall, [orphan]) == {orphan}
    assert await _message_count() == 2


async def test_parallel_runs_split_the_work_without_double_counting(
    repo: SqlRetentionRepository,
) -> None:
    alice = await _user()
    old = [await _conversation(alice, 31 * DAY, 1) for _ in range(12)]

    results = await asyncio.gather(
        *(purge(repo, now=NOW, days=30, dry_run=False, batch_size=2) for _ in range(3))
    )

    assert sum(r.conversations for r in results) == 12
    assert sum(r.messages for r in results) == 12
    assert await _existing(Conversation, old) == set()
