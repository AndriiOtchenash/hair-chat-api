"""Conversation search SQL against real Postgres/pgvector: threshold, one result per
conversation, the question shown, isolation by user_id and embedding_model, the ILIKE fallback.
Crafted vectors only (rule 1)."""

import math
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete

from app.db.models import Conversation, Message, MessageEmbedding, MessageRole, User
from app.db.repositories.conversation_search import SearchRow, SqlConversationSearchRepository
from app.db.session import get_sessionmaker
from tests.llm_fakes import unit_vector

T0 = datetime(2026, 11, 1, tzinfo=UTC)
MODEL = "fake-embedding"
QUERY = unit_vector(1.0)


def at_distance(distance: float) -> list[float]:
    """A unit vector at this cosine distance from QUERY."""
    similarity = 1 - distance
    return unit_vector(similarity, math.sqrt(1 - similarity**2))


@pytest.fixture
async def clean(app_settings_env: None) -> AsyncIterator[None]:
    async with get_sessionmaker()() as session:
        await session.execute(delete(MessageEmbedding))
        await session.execute(delete(Message))
        await session.commit()
    yield


async def new_user() -> uuid.UUID:
    async with get_sessionmaker()() as session:
        user = User(auth_sub=f"auth0|{uuid.uuid4()}")
        session.add(user)
        await session.commit()
        return user.id


async def conversation(
    user_id: uuid.UUID,
    *turns: tuple[str, str, float | None],
    title: str | None = None,
    model: str = MODEL,
    start: datetime = T0,
) -> list[Message]:
    """(question, answer, distance of the answer's vector or None = not indexed), a minute
    apart from `start`."""
    async with get_sessionmaker()(expire_on_commit=False) as session:
        conv = Conversation(user_id=user_id, title=title)
        session.add(conv)
        await session.flush()
        messages: list[Message] = []
        for i, (question, answer, distance) in enumerate(turns):
            pair = [
                Message(
                    conversation_id=conv.id,
                    user_id=user_id,
                    role=role,
                    content=content,
                    created_at=start + timedelta(minutes=2 * i + offset),
                )
                for offset, (role, content) in enumerate(
                    [(MessageRole.USER, question), (MessageRole.ASSISTANT, answer)]
                )
            ]
            session.add_all(pair)
            await session.flush()
            if distance is not None:
                session.add(
                    MessageEmbedding(
                        message_id=pair[1].id,
                        user_id=user_id,
                        conversation_id=conv.id,
                        embedding=at_distance(distance),
                        embedding_model=model,
                    )
                )
            messages.extend(pair)
        await session.commit()
        return messages


async def nearest(
    user_id: uuid.UUID, *, max_distance: float = 0.4, limit: int = 10, model: str = MODEL
) -> list[SearchRow]:
    async with get_sessionmaker()() as session:
        return await SqlConversationSearchRepository(session).nearest(
            user_id, QUERY, model=model, max_distance=max_distance, limit=limit
        )


async def containing(user_id: uuid.UUID, text: str, limit: int = 10) -> list[SearchRow]:
    async with get_sessionmaker()() as session:
        return await SqlConversationSearchRepository(session).containing(user_id, text, limit=limit)


async def test_only_rows_within_the_threshold_closest_first(clean: None) -> None:
    user = await new_user()
    near = await conversation(user, ("Как часто мыть?", "Раз в два дня.", 0.1), title="Мытьё")
    middle = await conversation(user, ("А маски?", "Раз в неделю.", 0.3))
    await conversation(user, ("Погода?", "Не знаю.", 0.5))

    rows = await nearest(user)

    assert [(r.conversation_id, r.title, r.question) for r in rows] == [
        (near[0].conversation_id, "Мытьё", "Как часто мыть?"),
        (middle[0].conversation_id, None, "А маски?"),
    ]
    assert [r.distance for r in rows] == [pytest.approx(0.1), pytest.approx(0.3)]
    assert (rows[0].message_id, rows[0].created_at) == (near[0].id, near[0].created_at)
    assert len(await nearest(user, limit=1)) == 1


async def test_one_result_per_conversation_its_best_answer_with_that_answers_question(
    clean: None,
) -> None:
    user = await new_user()
    messages = await conversation(
        user,
        ("Первый вопрос", "Первый ответ", 0.3),
        ("Лучший вопрос", "Лучший ответ", 0.05),
        ("Третий вопрос", "Третий ответ", 0.2),
    )

    [row] = await nearest(user)

    assert (row.message_id, row.question) == (messages[2].id, "Лучший вопрос")
    assert row.distance == pytest.approx(0.05)


async def test_someone_elses_rows_and_rows_of_another_model_are_never_found(
    clean: None,
) -> None:
    alice, bob = await new_user(), await new_user()
    await conversation(bob, ("Вопрос Боба", "Ответ", 0.0))
    await conversation(alice, ("Старая модель", "Ответ", 0.0), model="old-model")

    assert await nearest(alice) == []
    assert [r.question for r in await nearest(alice, model="old-model")] == ["Старая модель"]
    assert [r.question for r in await nearest(bob)] == ["Вопрос Боба"]


async def test_text_search_matches_own_questions_case_insensitively_newest_per_conversation(
    clean: None,
) -> None:
    alice, bob = await new_user(), await new_user()
    first = await conversation(
        alice,
        ("Что делать с ПЕРХОТЬЮ?", "Про перхоть в ответе", None),
        ("А перхоть вернётся?", "Может.", None),
    )
    second = await conversation(
        alice, ("Перхоть у ребёнка", "К врачу.", None), start=T0 + timedelta(hours=1)
    )
    await conversation(
        alice, ("Сухие кончики", "Только в ответе: перхоть", None), start=T0 + timedelta(hours=2)
    )
    await conversation(bob, ("Моя перхоть", "Ответ", None))

    rows = await containing(alice, "перхот")

    # Newest conversation's match first; in `first` only its newest matching question.
    assert [(r.conversation_id, r.message_id, r.question) for r in rows] == [
        (second[0].conversation_id, second[0].id, "Перхоть у ребёнка"),
        (first[0].conversation_id, first[2].id, "А перхоть вернётся?"),
    ]
    assert all(r.distance is None for r in rows)
    assert len(await containing(alice, "перхот", limit=1)) == 1


async def test_text_search_treats_wildcards_literally(clean: None) -> None:
    user = await new_user()
    await conversation(user, ("Скидка 100% на маску_для волос", "Ок", None))
    await conversation(user, ("Скидка 1000 на маскуXдля волос", "Ок", None))

    assert [r.question for r in await containing(user, "100%")] == [
        "Скидка 100% на маску_для волос"
    ]
    assert [r.question for r in await containing(user, "маску_для")] == [
        "Скидка 100% на маску_для волос"
    ]
    assert await containing(user, "\\") == []
