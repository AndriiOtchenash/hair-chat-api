from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from sqlalchemy import event, exc
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import ConnectionPoolEntry, PoolProxiedConnection

from app.core.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    settings = get_settings()
    engine = create_async_engine(
        settings.database_url,
        # Replaced by _ping_on_checkout: the built-in pre-ping lets a ping timeout through to
        # the request instead of reconnecting.
        pool_pre_ping=False,
        pool_recycle=settings.db_pool_recycle_s,
        connect_args={
            # asyncpg: per statement (the ping included) and per connect.
            "command_timeout": settings.db_command_timeout_s,
            "timeout": settings.db_command_timeout_s,
        },
    )
    event.listen(engine.sync_engine.pool, "checkout", _ping_on_checkout)
    return engine


def _ping_on_checkout(
    dbapi_connection: Any,
    connection_record: ConnectionPoolEntry,
    connection_proxy: PoolProxiedConnection,
) -> None:
    """Pessimistic disconnect handling: a pooled connection is pinged before every use.

    After a Fly suspend the pool comes back from the memory snapshot with sockets whose server
    side is gone: a ping on them fails at once (closed) or hangs until `command_timeout`. Either
    way the pool drops every connection opened before now and the checkout retries on a fresh
    one, so the request never sees the dead connection."""
    try:
        dbapi_connection.ping()
    except Exception as error:
        # Abort the socket now: a graceful close on a hung one would wait out its own timeout.
        connection_proxy.driver_connection.terminate()  # type: ignore[union-attr]
        raise exc.InvalidatePoolError() from error


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session
