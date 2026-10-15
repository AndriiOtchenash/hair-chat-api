import logging
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import new_id
from app.db.models import User

logger = logging.getLogger(__name__)


class UserRepository(Protocol):
    async def get_or_create_by_auth_sub(self, auth_sub: str) -> User: ...


class SqlUserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create_by_auth_sub(self, auth_sub: str) -> User:
        """Return the user for an Auth0 `sub`, creating it on first sight.

        ON CONFLICT DO NOTHING + SELECT instead of SELECT-then-INSERT: two concurrent first
        requests both reach the INSERT, the second waits for the first to commit and then
        skips, so neither fails on the unique constraint. Commits immediately so the new row
        (and its lock) is released before the request does anything else.
        Email is not in the access token; it stays NULL until we fetch it from Auth0 later.
        """
        result = await self._session.execute(
            insert(User)
            .values(id=new_id(), auth_sub=auth_sub)
            .on_conflict_do_nothing(index_elements=[User.auth_sub])
            .returning(User.id)
        )
        created_id = result.scalar_one_or_none()
        user = (
            await self._session.execute(select(User).where(User.auth_sub == auth_sub))
        ).scalar_one()
        await self._session.commit()
        if created_id is not None:
            logger.info("users: created user %s", created_id)
        return user
