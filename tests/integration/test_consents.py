import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
from fastapi import APIRouter, Depends
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.core.consents import require_consents
from app.core.security import get_token_verifier
from app.db.models import Consent, ConsentType, User
from app.db.session import get_sessionmaker
from app.main import create_app
from tests.auth_fakes import FakeAuth0, SigningKey, bearer, mint

KEY = SigningKey("key-1")
V1 = "2026-10-v1"
V2 = "2027-01-v2"
ALL_TYPES = [t.value for t in ConsentType]


class VersionSwitch:
    """Lets a test change CONSENT_VERSION mid-way, as a redeploy with new texts would."""

    def __init__(self) -> None:
        self.settings = Settings(consent_version=V1)

    def set(self, version: str) -> None:
        self.settings = self.settings.model_copy(update={"consent_version": version})


@pytest.fixture
def version() -> VersionSwitch:
    return VersionSwitch()


@pytest.fixture
async def client(app_settings_env: None, version: VersionSwitch) -> AsyncIterator[AsyncClient]:
    app = create_app()
    verifier = FakeAuth0(KEY).verifier()
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    app.dependency_overrides[get_settings] = lambda: version.settings
    gated = APIRouter(dependencies=[Depends(require_consents)])
    gated.add_api_route("/gated", lambda: {"ok": "yes"}, methods=["GET"])
    app.include_router(gated)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await verifier.aclose()


@pytest.fixture
def auth() -> dict[str, str]:
    return bearer(mint(KEY, sub=f"auth0|{uuid.uuid4()}"))


async def _user(headers: dict[str, str], client: AsyncClient) -> User:
    user_id = (await client.get("/me", headers=headers)).json()["id"]
    async with get_sessionmaker()() as session:
        return (
            await session.execute(select(User).where(User.id == uuid.UUID(user_id)))
        ).scalar_one()


async def _rows(user_id: uuid.UUID) -> list[Consent]:
    async with get_sessionmaker()() as session:
        result = await session.execute(
            select(Consent).where(Consent.user_id == user_id).order_by(Consent.granted_at)
        )
        return list(result.scalars())


def _granted(body: dict[str, object]) -> set[str]:
    items = body["items"]
    assert isinstance(items, list)
    return {i["type"] for i in items if i["granted"]}


async def test_grant_is_idempotent(client: AsyncClient, auth: dict[str, str]) -> None:
    for _ in range(2):
        response = await client.post(
            "/consents", headers=auth, json={"types": ALL_TYPES, "version": V1}
        )
        assert response.status_code == 200, response.text

    user = await _user(auth, client)
    assert len(await _rows(user.id)) == 3
    assert _granted(response.json()) == set(ALL_TYPES)


async def test_concurrent_grants_create_one_row_per_type(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    await client.get("/me", headers=auth)  # user exists; the race under test is on consents

    responses = await asyncio.gather(
        *(
            client.post("/consents", headers=auth, json={"types": ALL_TYPES, "version": V1})
            for _ in range(5)
        )
    )

    assert [r.status_code for r in responses] == [200] * 5
    user = await _user(auth, client)
    assert sorted(c.type.value for c in await _rows(user.id)) == sorted(ALL_TYPES)


async def test_revoke_keeps_row_and_regrant_adds_new_one(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    await client.post("/consents", headers=auth, json={"types": ["health_data"], "version": V1})

    revoke = await client.delete("/consents/health_data", headers=auth)
    after_revoke = await client.get("/consents", headers=auth)
    await client.post("/consents", headers=auth, json={"types": ["health_data"], "version": V1})

    assert revoke.status_code == 204
    assert "health_data" not in _granted(after_revoke.json())
    rows = await _rows((await _user(auth, client)).id)
    assert len(rows) == 2
    assert rows[0].revoked_at is not None
    assert rows[1].revoked_at is None


async def test_age_consent_sets_age_confirmed_at_once(
    client: AsyncClient, auth: dict[str, str], version: VersionSwitch
) -> None:
    assert (await _user(auth, client)).age_confirmed_at is None

    await client.post("/consents", headers=auth, json={"types": ["age_18"], "version": V1})
    first = (await _user(auth, client)).age_confirmed_at
    await client.delete("/consents/age_18", headers=auth)
    version.set(V2)
    await client.post("/consents", headers=auth, json={"types": ["age_18"], "version": V2})

    assert first is not None
    assert (await _user(auth, client)).age_confirmed_at == first


async def test_new_consent_version_invalidates_old_consents(
    client: AsyncClient, auth: dict[str, str], version: VersionSwitch
) -> None:
    await client.post("/consents", headers=auth, json={"types": ALL_TYPES, "version": V1})
    version.set(V2)

    status = await client.get("/consents", headers=auth)
    stale = await client.post("/consents", headers=auth, json={"types": ALL_TYPES, "version": V1})
    regrant = await client.post("/consents", headers=auth, json={"types": ALL_TYPES, "version": V2})

    assert status.json()["required_version"] == V2
    assert _granted(status.json()) == set()
    assert {i["version"] for i in status.json()["items"]} == {V1}
    assert stale.status_code == 400
    assert stale.json()["code"] == "consent_version_mismatch"
    assert _granted(regrant.json()) == set(ALL_TYPES)
    assert len(await _rows((await _user(auth, client)).id)) == 6


async def test_require_consents_follows_consent_version(
    client: AsyncClient, auth: dict[str, str], version: VersionSwitch
) -> None:
    await client.post("/consents", headers=auth, json={"types": ALL_TYPES, "version": V1})

    before = await client.get("/gated", headers=auth)
    version.set(V2)
    after = await client.get("/gated", headers=auth)

    assert before.status_code == 200
    assert after.status_code == 403
    assert after.json()["code"] == "consent_required"
