from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.consents import get_consent_repository
from app.core.errors import api_error
from app.core.security import CurrentUser
from app.db.models import ConsentType
from app.db.repositories.consents import ConsentRepository
from app.services.consents import consent_statuses

router = APIRouter(prefix="/consents", tags=["consents"])

Consents = Annotated[ConsentRepository, Depends(get_consent_repository)]
AppSettings = Annotated[Settings, Depends(get_settings)]


class ConsentItem(BaseModel):
    type: ConsentType
    granted: bool
    version: str | None
    granted_at: datetime | None


class ConsentsResponse(BaseModel):
    required_version: str
    items: list[ConsentItem]


class GrantConsentsRequest(BaseModel):
    types: list[ConsentType] = Field(min_length=1)
    version: str


async def _status(user: CurrentUser, consents: Consents, settings: Settings) -> ConsentsResponse:
    version = settings.consent_version
    statuses = consent_statuses(await consents.list_active(user.id), version)
    return ConsentsResponse(
        required_version=version,
        items=[ConsentItem(**vars(s)) for s in statuses],
    )


@router.get("")
async def get_consents(
    user: CurrentUser, consents: Consents, settings: AppSettings
) -> ConsentsResponse:
    return await _status(user, consents, settings)


@router.post("")
async def grant_consents(
    body: GrantConsentsRequest, user: CurrentUser, consents: Consents, settings: AppSettings
) -> ConsentsResponse:
    if body.version != settings.consent_version:
        raise api_error(
            status.HTTP_400_BAD_REQUEST,
            "consent_version_mismatch",
            f"Consent texts changed; current version is {settings.consent_version}",
        )
    await consents.grant(user.id, body.types, body.version)
    return await _status(user, consents, settings)


@router.delete("/{consent_type}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_consent(consent_type: ConsentType, user: CurrentUser, consents: Consents) -> None:
    await consents.revoke(user.id, consent_type)
