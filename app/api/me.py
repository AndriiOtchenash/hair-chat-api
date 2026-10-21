import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from app.core.security import CurrentUser, get_user_repository
from app.db.models import Locale, UserRole
from app.db.repositories.users import UserRepository

router = APIRouter(tags=["users"])

Users = Annotated[UserRepository, Depends(get_user_repository)]


class MeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: UserRole
    locale: Locale
    created_at: datetime


class UpdateMeRequest(BaseModel):
    locale: Locale


@router.get("/me")
async def me(user: CurrentUser) -> MeResponse:
    return MeResponse.model_validate(user)


# Protected, not consented: the client sets the language before the consent screen.
@router.patch("/me")
async def update_me(body: UpdateMeRequest, user: CurrentUser, users: Users) -> MeResponse:
    return MeResponse.model_validate(await users.set_locale(user.id, body.locale))
