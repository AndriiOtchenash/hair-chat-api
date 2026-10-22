import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Consent, Conversation, Message, User


@dataclass(frozen=True, slots=True)
class AccountData:
    """Everything stored about one user, besides the users row itself."""

    consents: list[Consent]
    conversations: list[Conversation]
    messages: list[Message]


class AccountRepository(Protocol):
    """Whole-account reads and deletes, scoped by `user_id` (rule 5)."""

    async def export(self, user_id: uuid.UUID) -> AccountData:
        """All consents (revoked and old versions too), conversations and messages, oldest first."""
        ...

    async def delete_user(self, user_id: uuid.UUID) -> bool:
        """Hard-deletes the user; Postgres cascades to everything else. True if a row was gone."""
        ...


class SqlAccountRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def export(self, user_id: uuid.UUID) -> AccountData:
        consents = await self._session.scalars(
            select(Consent)
            .where(Consent.user_id == user_id)
            .order_by(Consent.granted_at, Consent.id)
        )
        conversations = await self._session.scalars(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at, Conversation.id)
        )
        messages = await self._session.scalars(
            select(Message)
            .where(Message.user_id == user_id)
            .order_by(Message.conversation_id, Message.created_at, Message.id)
        )
        return AccountData(list(consents), list(conversations), list(messages))

    async def delete_user(self, user_id: uuid.UUID) -> bool:
        # ON DELETE CASCADE: consents, conversations, messages and, through messages, the
        # llm_calls linked to them (tests/integration/test_data_rights.py).
        deleted = await self._session.scalar(
            delete(User)
            .where(User.id == user_id)
            .returning(User.id)
            .execution_options(synchronize_session=False)
        )
        await self._session.commit()
        return deleted is not None
