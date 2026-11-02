"""Conversation index against real Postgres: which answers are pending, the question each gets,
consents, small talk, writes and re-embedding (model, format), a rename leaving the index alone,
cascades. Fake embeddings only (rule 1)."""

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
    IndexReport,
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


async def pending_rows(
    model: str = "fake-embedding", min_answer_chars: int = 0, **scope: uuid.UUID
) -> list[PendingAnswer]:
    async with get_sessionmaker()() as session:
        return await repository(session).pending(
            model=model,
            index_format=FORMAT,
            after=None,
            limit=100,
            min_answer_chars=min_answer_chars,
            **scope,
        )


async def pending(model: str = "fake-embedding") -> dict[uuid.UUID, str | None]:
    return {row.message_id: row.question for row in await pending_rows(model)}


async def run(embedder: FakeEmbedder, min_answer_chars: int = 0) -> IndexReport:
    async with get_sessionmaker()() as session:
        return await index_conversations(
            repository(session), embedder, max_chars=1000, min_answer_chars=min_answer_chars
        )


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


async def test_pending_carries_whether_the_user_has_every_consent(clean: None) -> None:
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

    assert rows[full[1].id].consented
    assert not rows[partial[1].id].consented
    assert not rows[revoked[1].id].consented
    assert not rows[outdated[1].id].consented

    await run(FakeEmbedder())
    assert set(await stored()) == {full[1].id}


async def test_the_index_embeds_the_untitled_pair_and_stores_the_format(clean: None) -> None:
    messages = await conversation(u("Как часто?"), a("Раз в два дня."), title="Мытьё")
    embedder = FakeEmbedder()

    await run(embedder)

    assert embedder.embedded == [index_text("Как часто?", "Раз в два дня.", max_chars=1000)]
    assert embedder.embedded[0].startswith("title: none | ")
    assert (await stored())[messages[1].id].index_format == FORMAT == "3"


async def set_format(message_id: uuid.UUID, index_format: str | None) -> None:
    async with get_sessionmaker()() as session:
        await session.execute(
            update(MessageEmbedding)
            .where(MessageEmbedding.message_id == message_id)
            .values(index_format=index_format)
        )
        await session.commit()


async def test_rows_of_another_format_or_stale_are_pending_again(clean: None) -> None:
    v1, v2, stale = [await conversation(u("Q"), a("A")) for _ in range(3)]
    await run(FakeEmbedder())
    assert await pending() == {}

    await set_format(v1[1].id, "1")
    await set_format(v2[1].id, "2")  # titled, before format 3
    await set_format(stale[1].id, None)

    assert set(await pending()) == {v1[1].id, v2[1].id, stale[1].id}
    await run(FakeEmbedder())
    assert {r.index_format for r in (await stored()).values()} == {FORMAT}


async def test_small_talk_is_pending_only_while_it_has_a_row(clean: None) -> None:
    long_enough = await conversation(u("Что с перхотью?"), a("ж" * 200))
    short = await conversation(u("hallo"), a(" " + "ж" * 199 + "\n\t"))  # stripped: 199
    v2_greeting = await conversation(u("hallo"), a("Здравствуйте!"))
    await run(FakeEmbedder())  # everything indexed, as before format 3
    await set_format(v2_greeting[1].id, "2")
    await set_format(long_enough[1].id, "2")

    rows = {r.message_id: r for r in await pending_rows(min_answer_chars=200)}
    # short has a current row: pending, to be dropped; the v2 greeting too.
    assert {m: r.small_talk for m, r in rows.items()} == {
        long_enough[1].id: False,
        short[1].id: True,
        v2_greeting[1].id: True,
    }

    embedder = FakeEmbedder()
    report = await run(embedder, min_answer_chars=200)

    assert (report.indexed, report.skipped_small_talk) == (1, 2)
    assert set(await stored()) == {long_enough[1].id}
    assert not any("hallo" in t for t in embedder.embedded)
    # No row and short: never pending again, the next run reads nothing.
    assert await pending_rows(min_answer_chars=200) == []
    assert (await run(FakeEmbedder(), min_answer_chars=200)).seen == 0
    # A lower threshold brings them back.
    assert set(await pending()) == {short[1].id, v2_greeting[1].id}


async def test_a_rename_does_not_touch_the_index(clean: None) -> None:
    renamed = await conversation(u("Q1"), a("A1"), u("Q2"), a("A2"), title="Старое")
    user_id = renamed[0].user_id
    await run(FakeEmbedder())
    before = {
        m: (list(r.embedding), r.embedding_model, r.index_format, r.created_at)
        for m, r in (await stored()).items()
    }

    async with get_sessionmaker()() as session:
        await SqlConversationRepository(session).rename(
            user_id, renamed[0].conversation_id, "Новое"
        )

    after = {
        m: (list(r.embedding), r.embedding_model, r.index_format, r.created_at)
        for m, r in (await stored()).items()
    }
    assert after == before and len(after) == 2
    assert await pending_rows(user_id=user_id, conversation_id=renamed[0].conversation_id) == []


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
