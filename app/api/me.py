import uuid
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from app.core.security import CurrentUser
from app.db.models import Locale, UserRole

router = APIRouter(tags=["users"])


class MeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: UserRole
    locale: Locale
    created_at: datetime


@router.get("/me")
async def me(user: CurrentUser) -> MeResponse:
    return MeResponse.model_validate(user)
