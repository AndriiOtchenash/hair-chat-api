"""The user's data rights: export everything we store about them, delete the account (ADR 0006)."""

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from app.db.models import ConsentType, Locale, Message, MessageRole, User, UserRole
from app.db.repositories.account import AccountRepository

logger = logging.getLogger(__name__)


class IdentityProvider(Protocol):
    """Where the login lives (Auth0); raises on failure."""

    async def delete_user(self, auth_sub: str) -> None: ...


class ExportedUser(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    auth_sub: str
    email: str | None
    display_name: str | None
    locale: Locale
    role: UserRole
    created_at: datetime
    age_confirmed_at: datetime | None


class ExportedConsent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    type: ConsentType
    version: str
    granted_at: datetime
    revoked_at: datetime | None


class ExportedMessage(BaseModel):
    id: uuid.UUID
    role: MessageRole
    content: str
    created_at: datetime
    provider: str | None
    model: str | None
    # Moderation action (allow / soft) or status (unavailable / disabled); null for answers.
    moderation: str | None


class ExportedConversation(BaseModel):
    id: uuid.UUID
    title: str | None
    summary: str | None
    created_at: datetime
    last_message_at: datetime
    messages: list[ExportedMessage]


class AccountExport(BaseModel):
    format_version: Literal[1] = 1
    exported_at: datetime
    user: ExportedUser
    consents: list[ExportedConsent]
    conversations: list[ExportedConversation]


def _moderation(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    outcome = value.get("action") or value.get("status")
    return str(outcome) if outcome is not None else None


def _message(m: Message) -> ExportedMessage:
    return ExportedMessage(
        id=m.id,
        role=m.role,
        content=m.content,
        created_at=m.created_at,
        provider=m.provider,
        model=m.model,
        moderation=_moderation(m.moderation),
    )


class AccountService:
    def __init__(
        self,
        repo: AccountRepository,
        identities: IdentityProvider | None,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repo = repo
        self._identities = identities
        self._clock = clock

    async def export(self, user: User) -> AccountExport:
        data = await self._repo.export(user.id)
        by_conversation: dict[uuid.UUID, list[ExportedMessage]] = {}
        for m in data.messages:
            by_conversation.setdefault(m.conversation_id, []).append(_message(m))
        return AccountExport(
            exported_at=self._clock(),
            user=ExportedUser.model_validate(user),
            consents=[ExportedConsent.model_validate(c) for c in data.consents],
            conversations=[
                ExportedConversation(
                    id=c.id,
                    title=c.title,
                    summary=c.summary,
                    created_at=c.created_at,
                    last_message_at=c.last_message_at,
                    messages=by_conversation.get(c.id, []),
                )
                for c in data.conversations
            ],
        )

    async def delete(self, user: User) -> None:
        """Hard delete of all our data; committed before the identity provider is called."""
        await self._repo.delete_user(user.id)
        logger.info("account: deleted user %s", user.id)

    async def delete_identity(self, user_id: uuid.UUID, auth_sub: str) -> None:
        """Best effort, after `delete`: never raises, our deletion stays either way."""
        if self._identities is None:
            logger.warning(
                "account: Auth0 management not configured, user %s deleted locally only", user_id
            )
            return
        try:
            await self._identities.delete_user(auth_sub)
        except Exception as exc:
            # auth_sub is the only key left for a manual retry: our users row is gone.
            logger.error(
                "account: Auth0 user delete failed, retry by hand: user_id=%s auth_sub=%s error=%s",
                user_id,
                auth_sub,
                f"{type(exc).__name__}: {exc}",
            )
            return
        logger.info("account: Auth0 user deleted for user %s", user_id)
