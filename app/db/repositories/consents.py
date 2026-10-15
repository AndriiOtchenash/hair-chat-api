import uuid
from collections.abc import Iterable
from typing import Protocol

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import new_id
from app.db.models import Consent, ConsentType, User


class ConsentRepository(Protocol):
    async def list_active(self, user_id: uuid.UUID) -> list[Consent]: ...

    async def grant(
        self, user_id: uuid.UUID, types: Iterable[ConsentType], version: str
    ) -> None: ...

    async def revoke(self, user_id: uuid.UUID, consent_type: ConsentType) -> None: ...


class SqlConsentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_active(self, user_id: uuid.UUID) -> list[Consent]:
        """Non-revoked consents of all versions, oldest first."""
        result = await self._session.execute(
            select(Consent)
            .where(Consent.user_id == user_id, Consent.revoked_at.is_(None))
            .order_by(Consent.granted_at, Consent.id)
        )
        return list(result.scalars())

    async def grant(self, user_id: uuid.UUID, types: Iterable[ConsentType], version: str) -> None:
        """Idempotent: an active consent of the same type and version is left as is.

        The partial unique index ix_consents_user_id_type_version_active is the ON CONFLICT
        arbiter, so concurrent grants cannot create duplicates either.
        Granting age_18 sets users.age_confirmed_at once (first confirmation is kept).
        """
        # Sorted: concurrent multi-row grants take index locks in the same order (no deadlock).
        unique_types = sorted(set(types))
        if not unique_types:
            return
        rows = [
            {"id": new_id(), "user_id": user_id, "type": t, "version": version}
            for t in unique_types
        ]
        await self._session.execute(
            insert(Consent)
            .values(rows)
            .on_conflict_do_nothing(
                index_elements=[Consent.user_id, Consent.type, Consent.version],
                index_where=Consent.revoked_at.is_(None),
            )
        )
        if ConsentType.AGE_18 in unique_types:
            await self._session.execute(
                update(User)
                .where(User.id == user_id, User.age_confirmed_at.is_(None))
                .values(age_confirmed_at=func.now())
            )
        await self._session.commit()

    async def revoke(self, user_id: uuid.UUID, consent_type: ConsentType) -> None:
        """Marks every active consent of this type (any version) revoked; rows are kept."""
        await self._session.execute(
            update(Consent)
            .where(
                Consent.user_id == user_id,
                Consent.type == consent_type,
                Consent.revoked_at.is_(None),
            )
            .values(revoked_at=func.now())
        )
        await self._session.commit()
