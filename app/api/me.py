import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from pydantic import BaseModel, ConfigDict

from app.core.account import get_account_service
from app.core.security import CurrentUser, get_user_repository
from app.db.models import Locale, UserRole
from app.db.repositories.users import UserRepository
from app.services.account import AccountExport, AccountService

router = APIRouter(tags=["users"])

Users = Annotated[UserRepository, Depends(get_user_repository)]
Account = Annotated[AccountService, Depends(get_account_service)]


class MeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: UserRole
    locale: Locale
    created_at: datetime


class UpdateMeRequest(BaseModel):
    locale: Locale


class DeleteMeRequest(BaseModel):
    # Typed out by the user: anything else is a 422 and nothing is deleted.
    confirm: Literal["DELETE"]


@router.get("/me")
async def me(user: CurrentUser) -> MeResponse:
    return MeResponse.model_validate(user)


# Protected, not consented: the client sets the language before the consent screen.
@router.patch("/me")
async def update_me(body: UpdateMeRequest, user: CurrentUser, users: Users) -> MeResponse:
    return MeResponse.model_validate(await users.set_locale(user.id, body.locale))


# Data rights (ADR 0006): protected, not consented, so they keep working after a revoke.
@router.get(
    "/me/export",
    response_class=Response,
    responses={200: {"model": AccountExport, "description": "JSON file download"}},
)
async def export_me(user: CurrentUser, account: Account) -> Response:
    export = await account.export(user)
    filename = f"hair-chat-export-{export.exported_at:%Y-%m-%d}.json"
    return Response(
        export.model_dump_json(indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_me(
    body: DeleteMeRequest, user: CurrentUser, account: Account, background: BackgroundTasks
) -> None:
    user_id, auth_sub = user.id, user.auth_sub
    await account.delete(user)
    # After the response: Auth0 being slow or down must not hold or undo our deletion.
    background.add_task(account.delete_identity, user_id, auth_sub)
