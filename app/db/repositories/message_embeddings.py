"""message_embeddings: what the conversation index still lacks, and writes of its vectors.

Backfill across all users on purpose (like app.db.repositories.retention), or one conversation of
one user (ConversationIndexer): the rows carry user_id and conversation_id of their answer, and
the question is looked up in the same conversation of the same user. Searches
(app.db.repositories.conversation_search) filter by user_id (rule 5).
"""

import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, delete, distinct, func, or_, select, true, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from app.db.models import (
    Consent,
    ConsentType,
    Conversation,
    Message,
    MessageEmbedding,
    MessageRole,
)


@dataclass(frozen=True, slots=True)
class PendingAnswer:
    """An answer without a current vector (model and format), with the user message before it;
    or a small-talk answer that still has a row (to be deleted)."""

    message_id: uuid.UUID
    user_id: uuid.UUID
    conversation_id: uuid.UUID
    answer: str
    question: str | None  # None: no user message before the answer in its conversation
    question_moderation: dict[str, Any] | None
    # The user has every consent of the current version (CONSENT_VERSION), like
    # require_consents: without them nothing of theirs is embedded.
    consented: bool
    # Shorter than min_answer_chars (stripped): not indexed, its row (if any) is deleted.
    small_talk: bool = False


class SqlMessageEmbeddingRepository:
    def __init__(self, session: AsyncSession, *, consent_version: str) -> None:
        self._session = session
        self._consent_version = consent_version

    async def pending(
        self,
        *,
        model: str,
        index_format: str,
        after: uuid.UUID | None,
        limit: int,
        user_id: uuid.UUID | None = None,
        conversation_id: uuid.UUID | None = None,
        min_answer_chars: int = 0,
    ) -> list[PendingAnswer]:
        """Answers with no row in message_embeddings or a row of another model or format
        (NULL = stale), by id (UUID v7, so oldest first) after `after`: keyset paging, so
        skipped answers are not read again within one run. `user_id` + `conversation_id`
        limit it to one conversation of that user.

        An answer shorter than `min_answer_chars` (whitespace stripped) is small talk: it comes
        only while it still has a row (any model or format), for `drop`; without one it is
        never pending, so the backfill does not read it again on every run."""
        if (user_id is None) != (conversation_id is None):
            raise ValueError("user_id and conversation_id go together")
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
        consented = (
            select(func.count(distinct(Consent.type)))
            .where(
                Consent.user_id == Message.user_id,
                Consent.revoked_at.is_(None),
                Consent.version == self._consent_version,
            )
            .scalar_subquery()
        ) == len(ConsentType)
        # btrim strips spaces only by default; these are what str.strip() strips in practice.
        small_talk = func.char_length(func.btrim(Message.content, " \t\r\n")) < min_answer_chars
        outdated = or_(
            MessageEmbedding.message_id.is_(None),
            MessageEmbedding.embedding_model != model,
            MessageEmbedding.index_format.is_distinct_from(index_format),
        )
        query = (
            select(
                Message.id,
                Message.user_id,
                Message.conversation_id,
                Message.content,
                previous.c.content,
                previous.c.moderation,
                consented,
                small_talk,
            )
            .join(
                Conversation,
                and_(
                    Conversation.id == Message.conversation_id,
                    Conversation.user_id == Message.user_id,
                ),
            )
            .outerjoin(previous, true())
            .outerjoin(MessageEmbedding, MessageEmbedding.message_id == Message.id)
            .where(
                Message.role == MessageRole.ASSISTANT,
                or_(
                    and_(~small_talk, outdated),
                    and_(small_talk, MessageEmbedding.message_id.is_not(None)),
                ),
            )
            .order_by(Message.id)
            .limit(limit)
        )
        if after is not None:
            query = query.where(Message.id > after)
        if user_id is not None:
            query = query.where(
                Message.user_id == user_id, Message.conversation_id == conversation_id
            )
        rows = await self._session.execute(query)
        return [PendingAnswer(*row) for row in rows]

    async def save(
        self,
        rows: Sequence[tuple[PendingAnswer, Sequence[float]]],
        *,
        model: str,
        index_format: str,
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
                "index_format": index_format,
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
                        "index_format": statement.excluded.index_format,
                        "created_at": func.now(),
                    },
                )
            )
        await self._session.commit()
        return len(values)

    async def drop(self, message_ids: Sequence[uuid.UUID]) -> int:
        """Delete the rows of these answers (small talk) in one transaction; returns how many
        there were."""
        if not message_ids:
            return 0
        deleted = await self._session.scalars(
            delete(MessageEmbedding)
            .where(MessageEmbedding.message_id.in_(message_ids))
            .returning(MessageEmbedding.message_id)
        )
        count = len(deleted.all())
        await self._session.commit()
        return count


def sql_message_embedding_repositories(
    sessionmaker: async_sessionmaker[AsyncSession], *, consent_version: str
) -> Callable[[], AbstractAsyncContextManager[SqlMessageEmbeddingRepository]]:
    """A repository on its own session per use: for background tasks, after the request's
    session is closed."""

    @asynccontextmanager
    async def open_repository() -> AsyncIterator[SqlMessageEmbeddingRepository]:
        async with sessionmaker() as session:
            yield SqlMessageEmbeddingRepository(session, consent_version=consent_version)

    return open_repository
