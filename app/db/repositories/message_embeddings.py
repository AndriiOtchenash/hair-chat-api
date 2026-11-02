"""message_embeddings: what the conversation index still lacks, and writes of its vectors.

Backfill across all users on purpose (like app.db.repositories.retention): the rows carry
user_id and conversation_id of their answer, and the question is looked up in the same
conversation of the same user. Searches (later) must filter by user_id (rule 5).
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, or_, select, true, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models import Message, MessageEmbedding, MessageRole


@dataclass(frozen=True, slots=True)
class PendingAnswer:
    """An answer without a vector of the current model, with the user message before it."""

    message_id: uuid.UUID
    user_id: uuid.UUID
    conversation_id: uuid.UUID
    answer: str
    question: str | None  # None: no user message before the answer in its conversation
    question_moderation: dict[str, Any] | None


class SqlMessageEmbeddingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def pending(
        self, *, model: str, after: uuid.UUID | None, limit: int
    ) -> list[PendingAnswer]:
        """Answers with no row in message_embeddings or a row of another model, by id (UUID v7,
        so oldest first) after `after`: keyset paging, so skipped answers are not read again
        within one run."""
        question = aliased(Message)
        previous = (
            select(question.content, question.moderation)
            .where(
                question.conversation_id == Message.conversation_id,
                question.user_id == Message.user_id,
                question.role == MessageRole.USER,
                tuple_(question.created_at, question.id) < tuple_(Message.created_at, Message.id),
            )
            .order_by(question.created_at.desc(), question.id.desc())
            .limit(1)
            .lateral("previous")
        )
        query = (
            select(
                Message.id,
                Message.user_id,
                Message.conversation_id,
                Message.content,
                previous.c.content,
                previous.c.moderation,
            )
            .outerjoin(previous, true())
            .outerjoin(MessageEmbedding, MessageEmbedding.message_id == Message.id)
            .where(
                Message.role == MessageRole.ASSISTANT,
                or_(
                    MessageEmbedding.message_id.is_(None),
                    MessageEmbedding.embedding_model != model,
                ),
            )
            .order_by(Message.id)
            .limit(limit)
        )
        if after is not None:
            query = query.where(Message.id > after)
        rows = await self._session.execute(query)
        return [PendingAnswer(*row) for row in rows]

    async def save(
        self, rows: Sequence[tuple[PendingAnswer, Sequence[float]]], *, model: str
    ) -> int:
        """Insert or replace the vectors in one transaction; returns how many were written.

        Answers deleted since `pending` (account or conversation delete, retention) are left
        out: the KEY SHARE lock keeps the rest from being deleted until the commit.
        """
        if not rows:
            return 0
        ids = [answer.message_id for answer, _ in rows]
        alive = set(
            await self._session.scalars(
                select(Message.id).where(Message.id.in_(ids)).with_for_update(key_share=True)
            )
        )
        values = [
            {
                "message_id": answer.message_id,
                "user_id": answer.user_id,
                "conversation_id": answer.conversation_id,
                "embedding": list(vector),
                "embedding_model": model,
            }
            for answer, vector in rows
            if answer.message_id in alive
        ]
        if values:
            statement = insert(MessageEmbedding).values(values)
            await self._session.execute(
                statement.on_conflict_do_update(
                    index_elements=[MessageEmbedding.message_id],
                    set_={
                        "embedding": statement.excluded.embedding,
                        "embedding_model": statement.excluded.embedding_model,
                        "created_at": func.now(),
                    },
                )
            )
        await self._session.commit()
        return len(values)
