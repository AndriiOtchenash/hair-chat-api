from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.consents import get_consent_repository
from app.core.security import get_token_verifier, get_user_repository
from app.main import create_app
from tests.auth_fakes import FakeAuth0, SigningKey, bearer, mint
from tests.fakes import FakeConsentRepository, FakeUserRepository

KEY = SigningKey("key-1")
AUTH = bearer(mint(KEY, sub="auth0|alice"))


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = create_app()
    verifier = FakeAuth0(KEY).verifier()
    users = FakeUserRepository()
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    app.dependency_overrides[get_user_repository] = lambda: users
    app.dependency_overrides[get_consent_repository] = FakeConsentRepository
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await verifier.aclose()


async def test_patch_locale_returns_me_with_new_locale(client: AsyncClient) -> None:
    before = (await client.get("/me", headers=AUTH)).json()

    response = await client.patch("/me", headers=AUTH, json={"locale": "uk"})

    assert response.status_code == 200, response.text
    assert response.json() == {**before, "locale": "uk"}
    assert (await client.get("/me", headers=AUTH)).json()["locale"] == "uk"


async def test_patch_locale_works_without_consents(client: AsyncClient) -> None:
    # No consents granted in this app: the language is chosen before the consent screen.
    response = await client.patch("/me", headers=AUTH, json={"locale": "pl"})

    assert response.status_code == 200, response.text


@pytest.mark.parametrize("body", [{"locale": "de"}, {"locale": "UK"}, {"locale": None}, {}])
async def test_invalid_locale_is_422_and_keeps_old_one(
    client: AsyncClient, body: dict[str, object]
) -> None:
    response = await client.patch("/me", headers=AUTH, json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert (await client.get("/me", headers=AUTH)).json()["locale"] == "en"


async def test_patch_without_token_is_401(client: AsyncClient) -> None:
    response = await client.patch("/me", json={"locale": "uk"})

    assert response.status_code == 401
