"""Search over one user's own conversations: by vector (message_embeddings) or by text.

Every query is scoped to `user_id` on each table it reads (rule 5): the index rows, the
conversation, the answer and the question. One result per conversation, its best match.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import and_, select, true, tuple_
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models import Conversation, Message, MessageEmbedding, MessageRole

LIKE_ESCAPE = "\\"


@dataclass(frozen=True, slots=True)
class SearchRow:
    conversation_id: uuid.UUID
    title: str | None
    message_id: uuid.UUID  # the user message (question) shown as the snippet
    question: str
    created_at: datetime  # of that question
    distance: float | None  # cosine distance; None in a text search


def like_pattern(text: str) -> str:
    """`%text%` for ILIKE with %, _ and the escape character matched literally."""
    for char in (LIKE_ESCAPE, "%", "_"):
        text = text.replace(char, LIKE_ESCAPE + char)
    return f"%{text}%"


class SqlConversationSearchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def nearest(
        self,
        user_id: uuid.UUID,
        vector: Sequence[float],
        *,
        model: str,
        max_distance: float,
        limit: int,
    ) -> list[SearchRow]:
        """Per conversation its answer nearest to `vector` within `max_distance`, closest
        first, with the question before that answer. Exact search (no ANN index): the rows of
        one user only, of `model` only (rule 4)."""
        distance = MessageEmbedding.embedding.cosine_distance(vector)
        best = (
            select(
                MessageEmbedding.conversation_id,
                MessageEmbedding.message_id,
                distance.label("distance"),
            )
            .where(
                MessageEmbedding.user_id == user_id,
                MessageEmbedding.embedding_model == model,
                distance <= max_distance,
            )
            .ext(distinct_on(MessageEmbedding.conversation_id))
            .order_by(MessageEmbedding.conversation_id, distance, MessageEmbedding.message_id)
            .subquery("best")
        )
        answer = aliased(Message, name="answer")
        question = aliased(Message, name="question")
        previous = (
            select(question.id, question.content, question.created_at)
            .where(
                question.conversation_id == answer.conversation_id,
                question.user_id == user_id,
                question.role == MessageRole.USER,
                tuple_(question.created_at, question.id) < tuple_(answer.created_at, answer.id),
            )
            .order_by(question.created_at.desc(), question.id.desc())
            .limit(1)
            .lateral("previous")
        )
        query = (
            select(
                best.c.conversation_id,
                Conversation.title,
                previous.c.id,
                previous.c.content,
                previous.c.created_at,
                best.c.distance,
            )
            .join(
                Conversation,
                and_(Conversation.id == best.c.conversation_id, Conversation.user_id == user_id),
            )
            .join(answer, and_(answer.id == best.c.message_id, answer.user_id == user_id))
            .join(previous, true())
            .order_by(best.c.distance, best.c.conversation_id)
            .limit(limit)
        )
        return [SearchRow(*row) for row in await self._session.execute(query)]

    async def containing(self, user_id: uuid.UUID, text: str, *, limit: int) -> list[SearchRow]:
        """Per conversation the newest user message containing `text` (ILIKE, literal), newest
        conversations' matches first."""
        matched = (
            select(Message.conversation_id, Message.id, Message.content, Message.created_at)
            .where(
                Message.user_id == user_id,
                Message.role == MessageRole.USER,
                Message.content.ilike(like_pattern(text), escape=LIKE_ESCAPE),
            )
            .ext(distinct_on(Message.conversation_id))
            .order_by(Message.conversation_id, Message.created_at.desc(), Message.id.desc())
            .subquery("matched")
        )
        query = (
            select(
                matched.c.conversation_id,
                Conversation.title,
                matched.c.id,
                matched.c.content,
                matched.c.created_at,
            )
            .join(
                Conversation,
                and_(Conversation.id == matched.c.conversation_id, Conversation.user_id == user_id),
            )
            .order_by(matched.c.created_at.desc(), matched.c.id.desc())
            .limit(limit)
        )
        return [
            SearchRow(conversation_id, title, message_id, question, created_at, None)
            for conversation_id, title, message_id, question, created_at in (
                await self._session.execute(query)
            )
        ]
