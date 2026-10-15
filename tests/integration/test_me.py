import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from app.core.security import get_token_verifier
from app.db.models import User
from app.db.session import get_sessionmaker
from app.main import create_app
from tests.auth_fakes import FakeAuth0, SigningKey, bearer, mint

KEY = SigningKey("key-1")


@pytest.fixture
async def client(app_settings_env: None) -> AsyncIterator[AsyncClient]:
    app = create_app()
    verifier = FakeAuth0(KEY).verifier()
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await verifier.aclose()


async def _count_users(auth_sub: str) -> int:
    async with get_sessionmaker()() as session:
        count = await session.scalar(
            select(func.count()).select_from(User).where(User.auth_sub == auth_sub)
        )
    return count or 0


async def test_first_me_creates_user_and_second_returns_same(client: AsyncClient) -> None:
    sub = f"auth0|{uuid.uuid4()}"

    first = await client.get("/me", headers=bearer(mint(KEY, sub=sub)))
    second = await client.get("/me", headers=bearer(mint(KEY, sub=sub)))

    assert first.status_code == 200, first.text
    assert first.json()["role"] == "user"
    assert first.json()["locale"] == "en"
    assert second.json() == first.json()
    assert await _count_users(sub) == 1


async def test_concurrent_first_requests_create_exactly_one_user(client: AsyncClient) -> None:
    sub = f"auth0|{uuid.uuid4()}"
    token = mint(KEY, sub=sub)

    responses = await asyncio.gather(*(client.get("/me", headers=bearer(token)) for _ in range(5)))

    assert [r.status_code for r in responses] == [200] * 5
    assert len({r.json()["id"] for r in responses}) == 1
    assert await _count_users(sub) == 1
