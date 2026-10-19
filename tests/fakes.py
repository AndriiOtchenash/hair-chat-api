"""In-memory stand-ins for the repositories, for unit tests that should not need Postgres."""

import uuid
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from app.db.base import new_id
from app.db.models import (
    Consent,
    ConsentType,
    Conversation,
    Locale,
    Message,
    MessageRole,
    User,
    UserRole,
)
from app.db.pagination import Cursor, Page
from app.db.repositories.conversations import (
    TITLE_MAX_CHARS,
    ConversationRepository,
    ConversationRepositoryFactory,
)
from app.services.prompts import SystemPrompt


class StaticPrompt:
    """PromptProvider with a short fixed prompt: budget tests must not depend on the real one."""

    def __init__(self, text: str = "You are a hair-care assistant.", version: str = "test_v9"):
        self.prompt = SystemPrompt(text, version)

    async def get_active(self) -> SystemPrompt:
        return self.prompt


class FakeUserRepository:
    def __init__(self) -> None:
        self.by_sub: dict[str, User] = {}

    async def get_or_create_by_auth_sub(self, auth_sub: str) -> User:
        if auth_sub not in self.by_sub:
            self.by_sub[auth_sub] = User(
                id=new_id(),
                auth_sub=auth_sub,
                role=UserRole.USER,
                locale=Locale.EN,
                created_at=datetime.now(UTC),
            )
        return self.by_sub[auth_sub]


class FakeConsentRepository:
    def __init__(self) -> None:
        self.rows: list[Consent] = []

    async def list_active(self, user_id: uuid.UUID) -> list[Consent]:
        return [c for c in self.rows if c.user_id == user_id and c.revoked_at is None]

    async def grant(self, user_id: uuid.UUID, types: Iterable[ConsentType], version: str) -> None:
        active = {(c.type, c.version) for c in await self.list_active(user_id)}
        for consent_type in set(types):
            if (consent_type, version) not in active:
                self.rows.append(
                    Consent(
                        id=new_id(),
                        user_id=user_id,
                        type=consent_type,
                        version=version,
                        granted_at=datetime.now(UTC),
                    )
                )

    async def revoke(self, user_id: uuid.UUID, consent_type: ConsentType) -> None:
        for c in await self.list_active(user_id):
            if c.type == consent_type:
                c.revoked_at = datetime.now(UTC)


class FakeConversationRepository:
    """Same contract as SqlConversationRepository (user scoping included), kept in memory.

    Timestamps tick by 1 ms per write so ordering is deterministic. Pagination is covered by
    the integration tests against Postgres; here pages are plain slices.
    """

    def __init__(self) -> None:
        self.conversations: dict[uuid.UUID, Conversation] = {}
        self.messages: list[Message] = []
        # llm_call_id -> message_id, as the UPDATE of llm_calls would leave it
        self.linked_calls: dict[uuid.UUID, uuid.UUID] = {}
        self._now = datetime(2026, 1, 1, tzinfo=UTC)

    def _tick(self) -> datetime:
        self._now += timedelta(milliseconds=1)
        return self._now

    def _owned(self, user_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation | None:
        conversation = self.conversations.get(conversation_id)
        return conversation if conversation and conversation.user_id == user_id else None

    def of(self, conversation_id: uuid.UUID) -> list[Message]:
        return [m for m in self.messages if m.conversation_id == conversation_id]

    async def create(self, user_id: uuid.UUID) -> Conversation:
        now = self._tick()
        conversation = Conversation(
            id=new_id(),
            user_id=user_id,
            title=None,
            summary=None,
            summary_upto_message_id=None,
            created_at=now,
            last_message_at=now,
        )
        self.conversations[conversation.id] = conversation
        return conversation

    async def get(self, user_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation | None:
        return self._owned(user_id, conversation_id)

    async def list_page(
        self, user_id: uuid.UUID, limit: int, cursor: Cursor | None
    ) -> Page[Conversation]:
        rows = sorted(
            (c for c in self.conversations.values() if c.user_id == user_id),
            key=lambda c: (c.last_message_at, c.id),
            reverse=True,
        )
        return Page(rows[:limit], None)

    async def list_messages_page(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, limit: int, cursor: Cursor | None
    ) -> Page[Message]:
        rows = [m for m in reversed(self.of(conversation_id)) if m.user_id == user_id]
        return Page(rows[:limit], None)

    async def history(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        *,
        after: uuid.UUID | None,
        limit: int,
        newest_first: bool,
    ) -> list[Message]:
        rows = [m for m in self.of(conversation_id) if m.user_id == user_id]
        boundary = next((i for i, m in enumerate(rows) if m.id == after), None)
        if boundary is not None:
            rows = rows[boundary + 1 :]
        return list(reversed(rows))[:limit] if newest_first else rows[:limit]

    async def save_summary(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        summary: str,
        *,
        upto_message_id: uuid.UUID,
        expected_upto: uuid.UUID | None,
    ) -> bool:
        conversation = self._owned(user_id, conversation_id)
        if conversation is None or conversation.summary_upto_message_id != expected_upto:
            return False
        conversation.summary = summary
        conversation.summary_upto_message_id = upto_message_id
        return True

    def repositories(self) -> ConversationRepositoryFactory:
        """Factory as the summarizer takes it, always handing out this repository."""

        @asynccontextmanager
        async def open_repository() -> AsyncIterator[ConversationRepository]:
            yield self

        return open_repository

    async def add_user_message(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, content: str
    ) -> Message:
        conversation = self._owned(user_id, conversation_id)
        assert conversation is not None
        if conversation.title is None and not self.of(conversation_id):
            conversation.title = content[:TITLE_MAX_CHARS]
        return self._add(conversation, MessageRole.USER, content)

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
        conversation = self._owned(user_id, conversation_id)
        assert conversation is not None
        message = self._add(
            conversation,
            MessageRole.ASSISTANT,
            content,
            provider=provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        for call_id in llm_call_ids:
            self.linked_calls.setdefault(call_id, message.id)
        return message

    def _add(
        self,
        conversation: Conversation,
        role: MessageRole,
        content: str,
        **fields: str | int | None,
    ) -> Message:
        now = self._tick()
        message = Message(
            id=new_id(),
            user_id=conversation.user_id,
            conversation_id=conversation.id,
            role=role,
            content=content,
            created_at=now,
            **fields,
        )
        self.messages.append(message)
        conversation.last_message_at = now
        return message
