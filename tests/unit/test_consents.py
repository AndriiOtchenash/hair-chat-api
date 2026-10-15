from collections.abc import AsyncIterator

import pytest
from fastapi import APIRouter, Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, get_settings
from app.core.consents import get_consent_repository, require_consents
from app.core.security import get_token_verifier, get_user_repository
from app.db.models import ConsentType, User
from app.main import create_app
from tests.auth_fakes import FakeAuth0, SigningKey, bearer, mint
from tests.fakes import FakeConsentRepository, FakeUserRepository

KEY = SigningKey("key-1")
VERSION = "2026-10-v1"
ALL_TYPES = ["age_18", "health_data", "ai_processing"]
AUTH = bearer(mint(KEY, sub="auth0|alice"))


@pytest.fixture
def users() -> FakeUserRepository:
    return FakeUserRepository()


@pytest.fixture
def consents() -> FakeConsentRepository:
    return FakeConsentRepository()


@pytest.fixture
async def client(
    users: FakeUserRepository, consents: FakeConsentRepository
) -> AsyncIterator[AsyncClient]:
    app: FastAPI = create_app()
    verifier = FakeAuth0(KEY).verifier()
    settings = Settings(_env_file=None, consent_version=VERSION)
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    app.dependency_overrides[get_user_repository] = lambda: users
    app.dependency_overrides[get_consent_repository] = lambda: consents
    app.dependency_overrides[get_settings] = lambda: settings

    gated = APIRouter(dependencies=[Depends(require_consents)])

    @gated.get("/gated")
    async def gated_endpoint() -> dict[str, str]:
        return {"ok": "yes"}

    app.include_router(gated)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await verifier.aclose()


async def alice(users: FakeUserRepository) -> User:
    return await users.get_or_create_by_auth_sub("auth0|alice")


async def test_no_consents_is_403_listing_all_missing_types(client: AsyncClient) -> None:
    response = await client.get("/gated", headers=AUTH)

    assert response.status_code == 403
    body = response.json()
    assert body["code"] == "consent_required"
    assert body["message"] == "Missing consents: age_18, health_data, ai_processing"


async def test_all_consents_granted_passes(client: AsyncClient) -> None:
    await client.post("/consents", headers=AUTH, json={"types": ALL_TYPES, "version": VERSION})

    response = await client.get("/gated", headers=AUTH)

    assert response.status_code == 200


async def test_partially_granted_lists_only_missing(client: AsyncClient) -> None:
    await client.post("/consents", headers=AUTH, json={"types": ["age_18"], "version": VERSION})

    response = await client.get("/gated", headers=AUTH)

    assert response.json()["message"] == "Missing consents: health_data, ai_processing"


async def test_revoked_consent_is_403(client: AsyncClient) -> None:
    await client.post("/consents", headers=AUTH, json={"types": ALL_TYPES, "version": VERSION})

    revoke = await client.delete("/consents/health_data", headers=AUTH)
    response = await client.get("/gated", headers=AUTH)

    assert revoke.status_code == 204
    assert response.status_code == 403
    assert response.json()["message"] == "Missing consents: health_data"


async def test_consents_for_old_version_are_403(
    client: AsyncClient, users: FakeUserRepository, consents: FakeConsentRepository
) -> None:
    await consents.grant((await alice(users)).id, list(ConsentType), "2025-01-v0")

    response = await client.get("/gated", headers=AUTH)

    assert response.status_code == 403
    assert response.json()["code"] == "consent_required"


async def test_get_consents_reports_status_per_type(
    client: AsyncClient, users: FakeUserRepository, consents: FakeConsentRepository
) -> None:
    user_id = (await alice(users)).id
    await consents.grant(user_id, [ConsentType.HEALTH_DATA], "2025-01-v0")
    await consents.grant(user_id, [ConsentType.AGE_18], VERSION)

    response = await client.get("/consents", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["required_version"] == VERSION
    items = {item["type"]: item for item in body["items"]}
    assert set(items) == set(ALL_TYPES)
    assert items["age_18"]["granted"] is True
    assert items["age_18"]["version"] == VERSION
    assert items["health_data"]["granted"] is False  # agreed to an outdated text
    assert items["health_data"]["version"] == "2025-01-v0"
    assert items["ai_processing"] == {
        "type": "ai_processing",
        "granted": False,
        "version": None,
        "granted_at": None,
    }


async def test_post_consents_returns_new_status(client: AsyncClient) -> None:
    response = await client.post(
        "/consents", headers=AUTH, json={"types": ["age_18", "age_18"], "version": VERSION}
    )

    assert response.status_code == 200
    granted = {i["type"] for i in response.json()["items"] if i["granted"]}
    assert granted == {"age_18"}


async def test_post_consents_with_other_version_is_400(
    client: AsyncClient, consents: FakeConsentRepository
) -> None:
    response = await client.post(
        "/consents", headers=AUTH, json={"types": ALL_TYPES, "version": "2025-01-v0"}
    )

    assert response.status_code == 400
    assert response.json()["code"] == "consent_version_mismatch"
    assert VERSION in response.json()["message"]
    assert consents.rows == []


@pytest.mark.parametrize(
    "body",
    [
        {"types": [], "version": VERSION},
        {"types": ["marketing"], "version": VERSION},
        {"types": ["age_18"]},
    ],
    ids=["empty-types", "unknown-type", "no-version"],
)
async def test_post_consents_invalid_body_is_422(
    client: AsyncClient, body: dict[str, object]
) -> None:
    response = await client.post("/consents", headers=AUTH, json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


async def test_revoke_unknown_type_is_422(client: AsyncClient) -> None:
    response = await client.delete("/consents/marketing", headers=AUTH)

    assert response.status_code == 422


async def test_revoke_without_active_consent_is_idempotent(client: AsyncClient) -> None:
    assert (await client.delete("/consents/age_18", headers=AUTH)).status_code == 204
    assert (await client.delete("/consents/age_18", headers=AUTH)).status_code == 204


async def test_consents_are_per_user(client: AsyncClient) -> None:
    await client.post("/consents", headers=AUTH, json={"types": ALL_TYPES, "version": VERSION})
    bob = bearer(mint(KEY, sub="auth0|bob"))

    response = await client.get("/gated", headers=bob)

    assert response.status_code == 403
