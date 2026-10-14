import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from testcontainers.community.postgres import PostgresContainer

from app.core.config import get_settings
from app.db.session import get_engine, get_sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if Path(item.fspath).is_relative_to(Path(__file__).parent):
            item.add_marker(pytest.mark.integration)


def _clear_cached_singletons() -> None:
    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """Fresh pgvector Postgres, migrated with `alembic upgrade head` exactly as in deploys."""
    with PostgresContainer("pgvector/pgvector:pg16", driver="asyncpg") as pg:
        url = pg.get_connection_url()
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            env={**os.environ, "DATABASE_URL": url},
            check=True,
        )
        yield url


@pytest.fixture
async def app_settings_env(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Point the app's Settings at the test container for the duration of one test."""
    monkeypatch.setenv("DATABASE_URL", database_url)
    _clear_cached_singletons()
    yield
    await get_engine().dispose()
    _clear_cached_singletons()
