"""Conversation index against real Postgres: which answers are pending, the question and title
each gets, consents, writes and re-embedding (model, format, rename), cascades. Fake embeddings
only (rule 1)."""

import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Consent,
    ConsentType,
    Conversation,
    Message,
    MessageEmbedding,
    MessageRole,
    User,
)
from app.db.repositories.consents import SqlConsentRepository
from app.db.repositories.conversations import SqlConversationRepository
from app.db.repositories.message_embeddings import PendingAnswer, SqlMessageEmbeddingRepository
from app.db.session import get_sessionmaker
from app.rag.conversation_index import (
    CONVERSATION_INDEX_FORMAT_VERSION,
    index_conversations,
    index_text,
)
from tests.llm_fakes import FakeEmbedder

T0 = datetime(2026, 11, 1, tzinfo=UTC)
VERSION = "test-v1"  # consents of this version count as valid here
FORMAT = CONVERSATION_INDEX_FORMAT_VERSION


def repository(session: AsyncSession) -> SqlMessageEmbeddingRepository:
    return SqlMessageEmbeddingRepository(session, consent_version=VERSION)


@pytest.fixture
async def clean(app_settings_env: None) -> AsyncIterator[None]:
    async with get_sessionmaker()() as session:
        await session.execute(delete(MessageEmbedding))
        await session.execute(delete(Message))
        await session.commit()
    yield


async def conversation(
    *turns: tuple[MessageRole, str, dict[str, str] | None],
    title: str | None = None,
    consents: Sequence[ConsentType] = tuple(ConsentType),
    user_id: uuid.UUID | None = None,
) -> list[Message]:
    """One conversation (of a new user with `consents` of VERSION, unless `user_id`); the
    turns a minute apart, in order."""
    async with get_sessionmaker()(expire_on_commit=False) as session:
        if user_id is None:
            user = User(auth_sub=f"auth0|{uuid.uuid4()}")
            session.add(user)
            await session.flush()
            user_id = user.id
            await SqlConsentRepository(session).grant(user_id, consents, VERSION)
        conv = Conversation(user_id=user_id, title=title)
        session.add(conv)
        await session.flush()
        messages = [
            Message(
                conversation_id=conv.id,
                user_id=user_id,
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


async def pending_rows(model: str = "fake-embedding", **scope: uuid.UUID) -> list[PendingAnswer]:
    async with get_sessionmaker()() as session:
        return await repository(session).pending(
            model=model, index_format=FORMAT, after=None, limit=100, **scope
        )


async def pending(model: str = "fake-embedding") -> dict[uuid.UUID, str | None]:
    return {row.message_id: row.question for row in await pending_rows(model)}


async def run(embedder: FakeEmbedder) -> None:
    async with get_sessionmaker()() as session:
        await index_conversations(repository(session), embedder, max_chars=1000)


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
        repo = repository(session)
        [answer] = await repo.pending(model="m", index_format=FORMAT, after=None, limit=10)
        await session.execute(delete(Message).where(Message.id == messages[1].id))
        await session.commit()

        assert await repo.save([(answer, [0.1] * 768)], model="m", index_format=FORMAT) == 0
    assert await stored() == {}


async def test_pending_carries_the_title_and_whether_the_user_has_every_consent(
    clean: None,
) -> None:
    full = await conversation(u("Q"), a("A"), title="Мытьё")
    partial = await conversation(u("Q"), a("A"), consents=[ConsentType.AGE_18])
    revoked = await conversation(u("Q"), a("A"))
    outdated = await conversation(u("Q"), a("A"))
    async with get_sessionmaker()() as session:
        await SqlConsentRepository(session).revoke(revoked[1].user_id, ConsentType.AI_PROCESSING)
        await session.execute(
            update(Consent)
            .where(Consent.user_id == outdated[1].user_id)
            .values(version="old-version")
        )
        await session.commit()

    rows = {row.message_id: row for row in await pending_rows()}

    assert (rows[full[1].id].title, rows[full[1].id].consented) == ("Мытьё", True)
    assert rows[partial[1].id].title is None
    assert not rows[partial[1].id].consented
    assert not rows[revoked[1].id].consented
    assert not rows[outdated[1].id].consented

    await run(FakeEmbedder())
    assert set(await stored()) == {full[1].id}


async def test_the_index_embeds_the_titled_pair_and_stores_the_format(clean: None) -> None:
    messages = await conversation(u("Как часто?"), a("Раз в два дня."), title="Мытьё")
    embedder = FakeEmbedder()

    await run(embedder)

    assert embedder.embedded == [
        index_text("Мытьё", "Как часто?", "Раз в два дня.", max_chars=1000)
    ]
    assert (await stored())[messages[1].id].index_format == FORMAT


async def test_rows_of_another_format_or_stale_are_pending_again(clean: None) -> None:
    one = await conversation(u("Q"), a("A"))
    two = await conversation(u("Q"), a("A"))
    await run(FakeEmbedder())
    assert await pending() == {}

    async with get_sessionmaker()() as session:
        await session.execute(
            update(MessageEmbedding)
            .where(MessageEmbedding.message_id == one[1].id)
            .values(index_format="1")
        )
        await session.execute(
            update(MessageEmbedding)
            .where(MessageEmbedding.message_id == two[1].id)
            .values(index_format=None)
        )
        await session.commit()

    assert set(await pending()) == {one[1].id, two[1].id}
    await run(FakeEmbedder())
    assert {r.index_format for r in (await stored()).values()} == {FORMAT}


async def test_a_rename_marks_only_that_conversations_rows_stale(clean: None) -> None:
    renamed = await conversation(u("Q1"), a("A1"), u("Q2"), a("A2"), title="Старое")
    user_id = renamed[0].user_id
    other_of_same_user = await conversation(u("Q"), a("A"), user_id=user_id)
    await run(FakeEmbedder())

    async with get_sessionmaker()() as session:
        await SqlConversationRepository(session).rename(
            user_id, renamed[0].conversation_id, "Новое"
        )

    rows = await stored()
    assert rows[renamed[1].id].index_format is None
    assert rows[renamed[3].id].index_format is None
    assert rows[other_of_same_user[1].id].index_format == FORMAT
    # Still searchable meanwhile: the vector stays, only its format is unknown.
    assert len(rows) == 3

    scoped = await pending_rows(user_id=user_id, conversation_id=renamed[0].conversation_id)
    assert {r.message_id for r in scoped} == {renamed[1].id, renamed[3].id}
    assert {r.title for r in scoped} == {"Новое"}


async def test_someone_elses_rename_marks_nothing(clean: None) -> None:
    mine = await conversation(u("Q"), a("A"), title="Моё")
    stranger = await conversation(u("Q"), a("A"))
    await run(FakeEmbedder())

    async with get_sessionmaker()() as session:
        assert (
            await SqlConversationRepository(session).rename(
                stranger[0].user_id, mine[0].conversation_id, "Чужое"
            )
            is None
        )

    assert (await stored())[mine[1].id].index_format == FORMAT


async def test_the_conversation_scope_is_one_conversation_of_that_user(clean: None) -> None:
    mine = await conversation(u("Q"), a("A"))
    other = await conversation(u("Q"), a("A"))

    assert (
        await pending_rows(user_id=other[0].user_id, conversation_id=mine[0].conversation_id) == []
    )
    assert [
        r.message_id
        for r in await pending_rows(
            user_id=mine[0].user_id, conversation_id=mine[0].conversation_id
        )
    ] == [mine[1].id]
