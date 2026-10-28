"""Pooled connections that died while the app was idle (Fly suspend, Neon restart) must not
reach a request: the next query reconnects instead of failing or hanging."""

import asyncio
import contextlib
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.db.session import get_engine
from tests.integration.conftest import _clear_cached_singletons

POOL_SIZE = 3


async def _fill_pool() -> list[int]:
    """Open POOL_SIZE connections at once and return them to the pool; returns backend pids."""
    engine = get_engine()
    async with contextlib.AsyncExitStack() as stack:
        conns = [await stack.enter_async_context(engine.connect()) for _ in range(POOL_SIZE)]
        return [(await c.execute(text("SELECT pg_backend_pid()"))).scalar_one() for c in conns]


async def _query_pid() -> int:
    async with get_engine().connect() as conn:
        return (await conn.execute(text("SELECT pg_backend_pid()"))).scalar_one()


@pytest.mark.usefixtures("app_settings_env")
async def test_query_after_pool_connections_were_terminated_succeeds(database_url: str) -> None:
    old_pids = await _fill_pool()

    admin = create_async_engine(database_url, poolclass=NullPool)
    async with admin.connect() as conn:
        await conn.execute(
            text("SELECT pg_terminate_backend(pid) FROM unnest(CAST(:pids AS int[])) AS pid"),
            {"pids": old_pids},
        )
    await admin.dispose()
    # pg_terminate_backend only signals the backends; give them a moment to exit.
    await asyncio.sleep(0.5)

    for _ in range(POOL_SIZE + 1):
        assert await _query_pid() not in old_pids


class _FreezableProxy:
    """TCP proxy to Postgres whose existing connections can be frozen: sockets stay open, bytes
    stop flowing in both directions, exactly what pooled connections look like after a VM
    snapshot is restored. Connections opened after the freeze work normally."""

    def __init__(self, host: str, port: int) -> None:
        self._target = (host, port)
        self._pipes: list[asyncio.Task[None]] = []
        self._server: asyncio.Server | None = None
        self.port = 0

    async def __aenter__(self) -> "_FreezableProxy":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.freeze()
        assert self._server is not None
        self._server.close()

    def freeze(self) -> None:
        for pipe in self._pipes:
            pipe.cancel()
        self._pipes.clear()

    async def _handle(self, client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
        server_r, server_w = await asyncio.open_connection(*self._target)
        self._pipes += [
            asyncio.create_task(self._pipe(client_r, server_w)),
            asyncio.create_task(self._pipe(server_r, client_w)),
        ]

    @staticmethod
    async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # Cancelled = frozen: the writer is deliberately left open, nothing is closed.
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()


@pytest.fixture
async def frozen_proxy(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[_FreezableProxy]:
    url = make_url(database_url)
    assert url.host is not None and url.port is not None
    async with _FreezableProxy(url.host, url.port) as proxy:
        proxied = url.set(host="127.0.0.1", port=proxy.port)
        monkeypatch.setenv("DATABASE_URL", proxied.render_as_string(hide_password=False))
        monkeypatch.setenv("DB_COMMAND_TIMEOUT_S", "1")
        _clear_cached_singletons()
        yield proxy
        await get_engine().dispose()
        _clear_cached_singletons()


async def test_query_after_pool_connections_hung_reconnects_within_timeout(
    frozen_proxy: _FreezableProxy,
) -> None:
    old_pids = await _fill_pool()
    frozen_proxy.freeze()

    # The first query waits out one ping timeout; every later one that meets a stale connection
    # pays SQLAlchemy's 2 s graceful close of it. No request hangs beyond that.
    limit = get_settings().db_command_timeout_s + 3
    for _ in range(POOL_SIZE + 1):
        async with asyncio.timeout(limit):
            assert await _query_pid() not in old_pids
