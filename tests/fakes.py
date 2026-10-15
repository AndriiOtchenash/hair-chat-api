"""In-memory stand-ins for the repositories, for unit tests that should not need Postgres."""

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime

from app.db.base import new_id
from app.db.models import Consent, ConsentType, Locale, User, UserRole


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
