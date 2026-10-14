from collections.abc import AsyncIterator

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.session import get_session
from app.main import create_app

# Port 1 on loopback: nothing listens there, so the connection is refused immediately.
UNREACHABLE_DB_URL = "postgresql+asyncpg://user:pass@127.0.0.1:1/nodb"


async def test_health_reports_db_down_when_database_is_unreachable() -> None:
    engine = create_async_engine(UNREACHABLE_DB_URL)

    async def unreachable_session() -> AsyncIterator[AsyncSession]:
        async with async_sessionmaker(engine)() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_session] = unreachable_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    await engine.dispose()
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "down"}
