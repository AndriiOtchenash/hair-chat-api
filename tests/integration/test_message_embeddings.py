"""Conversation index against real Postgres: which answers are pending, the question each
gets, writes and re-embedding, cascades. Fake embeddings only (rule 1)."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, select, text

from app.db.models import Conversation, Message, MessageEmbedding, MessageRole, User
from app.db.repositories.message_embeddings import SqlMessageEmbeddingRepository
from app.db.session import get_sessionmaker
from app.rag.conversation_index import index_conversations
from tests.llm_fakes import FakeEmbedder

T0 = datetime(2026, 11, 1, tzinfo=UTC)


@pytest.fixture
async def clean(app_settings_env: None) -> AsyncIterator[None]:
    async with get_sessionmaker()() as session:
        await session.execute(delete(MessageEmbedding))
        await session.execute(delete(Message))
        await session.commit()
    yield


async def conversation(*turns: tuple[MessageRole, str, dict[str, str] | None]) -> list[Message]:
    """One user, one conversation; the turns a minute apart, in order."""
    async with get_sessionmaker()(expire_on_commit=False) as session:
        user = User(auth_sub=f"auth0|{uuid.uuid4()}")
        session.add(user)
        await session.flush()
        conv = Conversation(user_id=user.id)
        session.add(conv)
        await session.flush()
        messages = [
            Message(
                conversation_id=conv.id,
                user_id=user.id,
                role=role,
                content=content,
                moderation=moderation,
                created_at=T0 + timedelta(minutes=i),
            )
            for i, (role, content, moderation) in enumerate(turns)
        ]
        session.add_all(messages)
        await session.commit()
        return messages


def u(
    content: str, moderation: dict[str, str] | None = None
) -> tuple[MessageRole, str, dict[str, str] | None]:
    return (MessageRole.USER, content, moderation)


def a(content: str) -> tuple[MessageRole, str, None]:
    return (MessageRole.ASSISTANT, content, None)


async def pending(model: str = "fake-embedding") -> dict[uuid.UUID, str | None]:
    async with get_sessionmaker()() as session:
        rows = await SqlMessageEmbeddingRepository(session).pending(
            model=model, after=None, limit=100
        )
    return {row.message_id: row.question for row in rows}


async def run(embedder: FakeEmbedder) -> None:
    async with get_sessionmaker()() as session:
        await index_conversations(SqlMessageEmbeddingRepository(session), embedder, max_chars=1000)


async def stored() -> dict[uuid.UUID, MessageEmbedding]:
    async with get_sessionmaker()() as session:
        return {row.message_id: row for row in await session.scalars(select(MessageEmbedding))}


async def test_each_answer_is_paired_with_the_latest_user_message_before_it(clean: None) -> None:
    first = await conversation(
        u("Как часто мыть?"),
        a("По мере загрязнения."),
        u("без ответа"),
        u("А маски?"),
        a("Раз в неделю."),
    )
    other = await conversation(u("Чужой вопрос"))
    orphan = await conversation(a("Здравствуйте!"))

    assert await pending() == {
        first[1].id: "Как часто мыть?",
        first[4].id: "А маски?",
        orphan[0].id: None,
    }
    assert other  # its question never leaks into another conversation's pair


async def test_index_writes_rows_and_rebuilds_only_another_model(clean: None) -> None:
    messages = await conversation(u("Q1"), a("A1"), u("Q2", {"action": "block"}), a("A2"))

    await run(FakeEmbedder())

    rows = await stored()
    assert set(rows) == {messages[1].id}  # the blocked question's answer is skipped
    row = rows[messages[1].id]
    assert (row.user_id, row.conversation_id, row.embedding_model) == (
        messages[1].user_id,
        messages[1].conversation_id,
        "fake-embedding",
    )
    assert set(await pending()) == {messages[3].id}
    assert set(await pending("new-model")) == {messages[1].id, messages[3].id}

    await run(FakeEmbedder("new-model"))
    assert {r.embedding_model for r in (await stored()).values()} == {"new-model"}


async def test_rows_go_with_their_answer_conversation_and_user(clean: None) -> None:
    one = await conversation(u("Q"), a("A"))
    two = await conversation(u("Q"), a("A"))
    three = await conversation(u("Q"), a("A"))
    await run(FakeEmbedder())
    assert len(await stored()) == 3

    async with get_sessionmaker()() as session:
        await session.execute(text("DELETE FROM messages WHERE id = :id"), {"id": one[1].id})
        await session.execute(
            text("DELETE FROM conversations WHERE id = :id"), {"id": two[1].conversation_id}
        )
        await session.execute(text("DELETE FROM users WHERE id = :id"), {"id": three[1].user_id})
        await session.commit()

    assert await stored() == {}


async def test_save_leaves_out_answers_deleted_after_pending(clean: None) -> None:
    messages = await conversation(u("Q"), a("A"))
    async with get_sessionmaker()() as session:
        repo = SqlMessageEmbeddingRepository(session)
        [answer] = await repo.pending(model="m", after=None, limit=10)
        await session.execute(delete(Message).where(Message.id == messages[1].id))
        await session.commit()

        assert await repo.save([(answer, [0.1] * 768)], model="m") == 0
    assert await stored() == {}
