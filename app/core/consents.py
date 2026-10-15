"""require_consents: the gate in front of chat endpoints (CLAUDE.md rule 7)."""

from typing import Annotated

from fastapi import Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import api_error
from app.core.security import CurrentUser
from app.db.models import User
from app.db.repositories.consents import ConsentRepository, SqlConsentRepository
from app.db.session import get_session
from app.services.consents import missing_consents


def get_consent_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ConsentRepository:
    return SqlConsentRepository(session)


async def require_consents(
    user: CurrentUser,
    consents: Annotated[ConsentRepository, Depends(get_consent_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    """403 consent_required unless every ConsentType is granted for CONSENT_VERSION."""
    missing = missing_consents(await consents.list_active(user.id), settings.consent_version)
    if missing:
        raise api_error(
            status.HTTP_403_FORBIDDEN,
            "consent_required",
            "Missing consents: " + ", ".join(missing),
        )
    return user
