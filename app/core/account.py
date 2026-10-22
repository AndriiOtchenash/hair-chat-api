"""FastAPI wiring of the data-rights endpoints (GET /me/export, DELETE /me)."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth0_mgmt import Auth0Management, get_auth0_management
from app.db.repositories.account import AccountRepository, SqlAccountRepository
from app.db.session import get_session
from app.services.account import AccountService


def get_account_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AccountRepository:
    return SqlAccountRepository(session)


def get_account_service(
    repo: Annotated[AccountRepository, Depends(get_account_repository)],
    auth0: Annotated[Auth0Management | None, Depends(get_auth0_management)],
) -> AccountService:
    return AccountService(repo, auth0)
