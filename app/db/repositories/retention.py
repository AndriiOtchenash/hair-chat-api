from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, LLMCall, Message


@dataclass(frozen=True, slots=True)
class PurgeCounts:
    conversations: int = 0
    messages: int = 0


class RetentionRepository(Protocol):
    """Crosses users on purpose: retention applies to every account alike."""

    async def count_expired(self, cutoff: datetime) -> PurgeCounts:
        """Conversations with last_message_at < cutoff and their messages."""
        ...

    async def delete_expired_batch(self, cutoff: datetime, limit: int) -> PurgeCounts:
        """Deletes at most `limit` such conversations (cascading to messages) in one transaction.

        Rows locked by a concurrent run are skipped, so parallel runs never count a
        conversation twice. A zero result means nothing is left for this run.
        """
        ...

    async def count_orphan_llm_calls(self, cutoff: datetime) -> int:
        """llm_calls never linked to a message (summaries, failed generations), created < cutoff."""
        ...

    async def delete_orphan_llm_calls(self, cutoff: datetime) -> int: ...


class SqlRetentionRepository:
    """Opens a short session per batch: a long run never holds one big transaction."""

    def __init__(self, sessions: Callable[[], AbstractAsyncContextManager[AsyncSession]]) -> None:
        self._sessions = sessions

    async def count_expired(self, cutoff: datetime) -> PurgeCounts:
        async with self._sessions() as session:
            conversations = await session.scalar(
                select(func.count()).where(Conversation.last_message_at < cutoff)
            )
            messages = await session.scalar(
                select(func.count())
                .select_from(Message)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(Conversation.last_message_at < cutoff)
            )
        return PurgeCounts(conversations or 0, messages or 0)

    async def delete_expired_batch(self, cutoff: datetime, limit: int) -> PurgeCounts:
        async with self._sessions() as session, session.begin():
            # FOR UPDATE also blocks a chat message being added meanwhile (its FK check needs
            # a KEY SHARE lock), so the message count below is exact.
            ids = list(
                await session.scalars(
                    select(Conversation.id)
                    .where(Conversation.last_message_at < cutoff)
                    .order_by(Conversation.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            if not ids:
                return PurgeCounts()
            messages = await session.scalar(
                select(func.count()).where(Message.conversation_id.in_(ids))
            )
            await session.execute(
                delete(Conversation)
                .where(Conversation.id.in_(ids))
                .execution_options(synchronize_session=False)
            )
        return PurgeCounts(len(ids), messages or 0)

    async def count_orphan_llm_calls(self, cutoff: datetime) -> int:
        async with self._sessions() as session:
            count = await session.scalar(
                select(func.count()).where(
                    LLMCall.message_id.is_(None), LLMCall.created_at < cutoff
                )
            )
        return count or 0

    async def delete_orphan_llm_calls(self, cutoff: datetime) -> int:
        async with self._sessions() as session, session.begin():
            deleted = await session.scalars(
                delete(LLMCall)
                .where(LLMCall.message_id.is_(None), LLMCall.created_at < cutoff)
                .returning(LLMCall.id)
                .execution_options(synchronize_session=False)
            )
            return len(list(deleted))
