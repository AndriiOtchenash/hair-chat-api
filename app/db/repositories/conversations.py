import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

from sqlalchemy import delete, exists, func, insert, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Conversation, LLMCall, Message, MessageRole
from app.db.pagination import Cursor, Page

TITLE_MAX_CHARS = 60


class ConversationRepository(Protocol):
    """Conversations and messages of one user: every method is scoped by `user_id` (rule 5)."""

    async def create(self, user_id: uuid.UUID) -> Conversation: ...

    async def get(self, user_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation | None: ...

    async def list_page(
        self, user_id: uuid.UUID, limit: int, cursor: Cursor | None
    ) -> Page[Conversation]:
        """Most recently active first (last_message_at desc)."""
        ...

    async def ids(self, user_id: uuid.UUID) -> list[uuid.UUID]: ...

    async def rename(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, title: str
    ) -> Conversation | None:
        """The renamed conversation, None if it is missing or someone else's."""
        ...

    async def delete(self, user_id: uuid.UUID, conversation_ids: Sequence[uuid.UUID]) -> int:
        """Hard-deletes these of the user's conversations; messages and their llm_calls go by
        ON DELETE CASCADE. Returns how many conversations were deleted."""
        ...

    async def list_messages_page(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, limit: int, cursor: Cursor | None
    ) -> Page[Message]:
        """Newest first; the cursor leads to older messages."""
        ...

    async def history(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        *,
        after: uuid.UUID | None,
        limit: int,
        newest_first: bool,
    ) -> list[Message]:
        """Messages after message `after` (all if None), at most `limit`.

        newest_first: the newest `limit` of them, newest first; otherwise the oldest, oldest first.
        """
        ...

    async def save_summary(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        summary: str,
        *,
        upto_message_id: uuid.UUID,
        expected_upto: uuid.UUID | None,
    ) -> bool:
        """Stores the summary of every message up to `upto_message_id`, only if the summary
        still ends at `expected_upto` (a concurrent summarizer wins otherwise). True if saved."""
        ...

    async def add_user_message(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        content: str,
        *,
        moderation: dict[str, Any] | None = None,
    ) -> Message:
        """Saves the message; the first message of a conversation also sets its title.

        moderation: the check's outcome for messages.moderation (never scores or text).
        """
        ...

    async def add_assistant_message(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        content: str,
        *,
        provider: str,
        model: str,
        tokens_in: int | None,
        tokens_out: int | None,
        llm_call_ids: Sequence[uuid.UUID],
    ) -> Message:
        """Saves the answer and links the llm_calls of its generation to it."""
        ...


class SqlConversationRepository:
    """Commits after every write, so no connection is held while an answer streams."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, user_id: uuid.UUID) -> Conversation:
        conversation: Conversation | None = await self._session.scalar(
            insert(Conversation).values(user_id=user_id).returning(Conversation)
        )
        await self._session.commit()
        assert conversation is not None
        return conversation

    async def get(self, user_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation | None:
        return await self._session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.user_id == user_id
            )
        )

    async def list_page(
        self, user_id: uuid.UUID, limit: int, cursor: Cursor | None
    ) -> Page[Conversation]:
        query = select(Conversation).where(Conversation.user_id == user_id)
        if cursor is not None:
            query = query.where(
                tuple_(Conversation.last_message_at, Conversation.id) < (cursor.at, cursor.id)
            )
        rows = list(
            await self._session.scalars(
                query.order_by(Conversation.last_message_at.desc(), Conversation.id.desc()).limit(
                    limit + 1
                )
            )
        )
        return _page(rows, limit, lambda c: Cursor(c.last_message_at, c.id))

    async def ids(self, user_id: uuid.UUID) -> list[uuid.UUID]:
        return list(
            await self._session.scalars(
                select(Conversation.id).where(Conversation.user_id == user_id)
            )
        )

    async def rename(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, title: str
    ) -> Conversation | None:
        conversation: Conversation | None = await self._session.scalar(
            update(Conversation)
            .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
            .values(title=title)
            .returning(Conversation)
        )
        await self._session.commit()
        return conversation

    async def delete(self, user_id: uuid.UUID, conversation_ids: Sequence[uuid.UUID]) -> int:
        if not conversation_ids:
            return 0
        deleted = await self._session.scalars(
            delete(Conversation)
            .where(Conversation.user_id == user_id, Conversation.id.in_(conversation_ids))
            .returning(Conversation.id)
        )
        count = len(deleted.all())
        await self._session.commit()
        return count

    async def list_messages_page(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, limit: int, cursor: Cursor | None
    ) -> Page[Message]:
        query = select(Message).where(
            Message.conversation_id == conversation_id, Message.user_id == user_id
        )
        if cursor is not None:
            query = query.where(tuple_(Message.created_at, Message.id) < (cursor.at, cursor.id))
        rows = list(
            await self._session.scalars(
                query.order_by(Message.created_at.desc(), Message.id.desc()).limit(limit + 1)
            )
        )
        return _page(rows, limit, lambda m: Cursor(m.created_at, m.id))

    async def history(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        *,
        after: uuid.UUID | None,
        limit: int,
        newest_first: bool,
    ) -> list[Message]:
        query = select(Message).where(
            Message.conversation_id == conversation_id, Message.user_id == user_id
        )
        if after is not None:
            boundary = (
                await self._session.execute(
                    select(Message.created_at, Message.id).where(
                        Message.id == after, Message.user_id == user_id
                    )
                )
            ).one_or_none()
            if boundary is not None:
                query = query.where(
                    tuple_(Message.created_at, Message.id) > (boundary.created_at, boundary.id)
                )
        order = (
            (Message.created_at.desc(), Message.id.desc())
            if newest_first
            else (Message.created_at, Message.id)
        )
        return list(await self._session.scalars(query.order_by(*order).limit(limit)))

    async def save_summary(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        summary: str,
        *,
        upto_message_id: uuid.UUID,
        expected_upto: uuid.UUID | None,
    ) -> bool:
        saved = await self._session.scalar(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.summary_upto_message_id.is_not_distinct_from(expected_upto),
            )
            .values(summary=summary, summary_upto_message_id=upto_message_id)
            .returning(Conversation.id)
        )
        await self._session.commit()
        return saved is not None

    async def add_user_message(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        content: str,
        *,
        moderation: dict[str, Any] | None = None,
    ) -> Message:
        has_messages = exists().where(Message.conversation_id == conversation_id)
        await self._session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.title.is_(None),
                ~has_messages,
            )
            .values(title=content[:TITLE_MAX_CHARS])
        )
        message = await self._insert_message(
            user_id, conversation_id, MessageRole.USER, content, moderation=moderation
        )
        await self._session.commit()
        return message

    async def add_assistant_message(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        content: str,
        *,
        provider: str,
        model: str,
        tokens_in: int | None,
        tokens_out: int | None,
        llm_call_ids: Sequence[uuid.UUID],
    ) -> Message:
        message = await self._insert_message(
            user_id,
            conversation_id,
            MessageRole.ASSISTANT,
            content,
            provider=provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        if llm_call_ids:
            # llm_calls has no user_id: the ids come from this very generation, never from input.
            await self._session.execute(
                update(LLMCall)
                .where(LLMCall.id.in_(llm_call_ids), LLMCall.message_id.is_(None))
                .values(message_id=message.id)
            )
        await self._session.commit()
        return message

    async def _insert_message(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        role: MessageRole,
        content: str,
        **fields: str | int | dict[str, Any] | None,
    ) -> Message:
        message: Message | None = await self._session.scalar(
            insert(Message)
            .values(
                user_id=user_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                **fields,
            )
            .returning(Message)
        )
        assert message is not None
        await self._session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
            .values(last_message_at=func.now())
        )
        return message


# Opens a repository on its own session: for work that outlives the request (background tasks).
ConversationRepositoryFactory = Callable[[], AbstractAsyncContextManager[ConversationRepository]]


def sql_conversation_repositories(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> ConversationRepositoryFactory:
    @asynccontextmanager
    async def open_repository() -> AsyncIterator[ConversationRepository]:
        async with sessionmaker() as session:
            yield SqlConversationRepository(session)

    return open_repository


def _page[T](rows: list[T], limit: int, cursor_of: Callable[[T], Cursor]) -> Page[T]:
    """`rows` were fetched with limit + 1: an extra row means there is a next page."""
    if len(rows) <= limit:
        return Page(rows, None)
    items = rows[:limit]
    return Page(items, cursor_of(items[-1]).encode())
