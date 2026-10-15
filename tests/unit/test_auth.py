import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import jwt
import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient, Response

from app.core.security import get_token_verifier, get_user_repository, require_role
from app.db.models import UserRole
from app.main import create_app
from tests.auth_fakes import AUDIENCE, FakeAuth0, SigningKey, bearer, mint
from tests.fakes import FakeUserRepository

KEY = SigningKey("key-1")
ROTATED_KEY = SigningKey("key-2")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def auth0() -> FakeAuth0:
    return FakeAuth0(KEY)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def users() -> FakeUserRepository:
    return FakeUserRepository()


@pytest.fixture
async def client(
    auth0: FakeAuth0, clock: FakeClock, users: FakeUserRepository
) -> AsyncIterator[AsyncClient]:
    app: FastAPI = create_app()
    verifier = auth0.verifier(clock=clock)
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    app.dependency_overrides[get_user_repository] = lambda: users

    @app.get("/admin-only", dependencies=[Depends(require_role(UserRole.ADMIN))])
    async def admin_only() -> dict[str, str]:
        return {"ok": "yes"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await verifier.aclose()


def assert_error(response: Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    assert response.json()["code"] == code


def assert_unauthorized(response: Response) -> None:
    assert_error(response, 401, "invalid_token")
    assert response.headers["WWW-Authenticate"].startswith("Bearer")


async def test_valid_token_returns_current_user(
    client: AsyncClient, users: FakeUserRepository
) -> None:
    response = await client.get("/me", headers=bearer(mint(KEY, sub="auth0|alice")))

    assert response.status_code == 200
    user = users.by_sub["auth0|alice"]
    assert response.json() == {
        "id": str(user.id),
        "role": "user",
        "locale": "en",
        "created_at": user.created_at.isoformat().replace("+00:00", "Z"),
    }


async def test_missing_authorization_header_is_401_with_bearer_challenge(
    client: AsyncClient,
) -> None:
    response = await client.get("/me")

    assert_error(response, 401, "not_authenticated")
    assert response.headers["WWW-Authenticate"] == "Bearer"


async def test_non_bearer_scheme_is_401(client: AsyncClient) -> None:
    response = await client.get("/me", headers={"Authorization": "Basic dXNlcjpwYXNz"})

    assert_error(response, 401, "not_authenticated")


async def test_expired_token_is_401(client: AsyncClient) -> None:
    now = int(time.time())
    token = mint(KEY, iat=now - 600, exp=now - 120)

    assert_unauthorized(await client.get("/me", headers=bearer(token)))


async def test_token_expired_within_clock_leeway_is_accepted(client: AsyncClient) -> None:
    now = int(time.time())
    token = mint(KEY, iat=now - 600, exp=now - 30)

    response = await client.get("/me", headers=bearer(token))

    assert response.status_code == 200


@pytest.mark.parametrize(
    "claims",
    [
        {"aud": "https://some-other-api.test"},
        {"iss": "https://evil-tenant.auth0.test/"},
        {"exp": None},
        {"sub": None},
    ],
    ids=["foreign-aud", "foreign-iss", "no-exp", "no-sub"],
)
async def test_token_with_wrong_or_missing_claims_is_401(
    client: AsyncClient, claims: dict[str, object]
) -> None:
    assert_unauthorized(await client.get("/me", headers=bearer(mint(KEY, **claims))))


async def test_token_signed_by_another_key_with_known_kid_is_401(client: AsyncClient) -> None:
    impostor = SigningKey(KEY.kid)

    assert_unauthorized(await client.get("/me", headers=bearer(mint(impostor))))


async def test_tampered_payload_is_401(client: AsyncClient) -> None:
    header, _, signature = mint(KEY, sub="auth0|alice").split(".")
    other_payload = mint(KEY, sub="auth0|mallory").split(".")[1]

    response = await client.get("/me", headers=bearer(f"{header}.{other_payload}.{signature}"))

    assert_unauthorized(response)


async def test_hs256_token_is_401(client: AsyncClient) -> None:
    token = jwt.encode(
        {"sub": "auth0|alice", "aud": AUDIENCE},
        "x" * 32,
        algorithm="HS256",
        headers={"kid": KEY.kid},
    )

    assert_unauthorized(await client.get("/me", headers=bearer(token)))


async def test_garbage_token_is_401(client: AsyncClient) -> None:
    assert_unauthorized(await client.get("/me", headers=bearer("not-a-jwt")))


async def test_unknown_kid_reloads_jwks_once_after_key_rotation(
    client: AsyncClient, auth0: FakeAuth0, clock: FakeClock
) -> None:
    assert (await client.get("/me", headers=bearer(mint(KEY)))).status_code == 200
    assert auth0.jwks_fetches == 1

    auth0.published.append(ROTATED_KEY)
    clock.now += 61
    response = await client.get("/me", headers=bearer(mint(ROTATED_KEY)))

    assert response.status_code == 200
    assert auth0.jwks_fetches == 2


async def test_unknown_kid_reload_is_rate_limited_to_once_a_minute(
    client: AsyncClient, auth0: FakeAuth0, clock: FakeClock
) -> None:
    stranger = SigningKey("made-up-kid")
    await client.get("/me", headers=bearer(mint(KEY)))

    clock.now += 61
    for _ in range(5):
        assert_unauthorized(await client.get("/me", headers=bearer(mint(stranger))))
    assert auth0.jwks_fetches == 2

    clock.now += 61
    assert_unauthorized(await client.get("/me", headers=bearer(mint(stranger))))
    assert auth0.jwks_fetches == 3


async def test_jwks_is_cached_for_an_hour(
    client: AsyncClient, auth0: FakeAuth0, clock: FakeClock
) -> None:
    for _ in range(3):
        await client.get("/me", headers=bearer(mint(KEY)))
    assert auth0.jwks_fetches == 1

    clock.now += 3601
    await client.get("/me", headers=bearer(mint(KEY)))
    assert auth0.jwks_fetches == 2


async def test_jwks_unreachable_without_cached_keys_is_503(
    client: AsyncClient, auth0: FakeAuth0
) -> None:
    auth0.status_code = 500

    response = await client.get("/me", headers=bearer(mint(KEY)))

    assert_error(response, 503, "auth_unavailable")


async def test_stale_keys_keep_working_when_jwks_refresh_fails(
    client: AsyncClient, auth0: FakeAuth0, clock: FakeClock
) -> None:
    await client.get("/me", headers=bearer(mint(KEY)))
    auth0.status_code = 500
    clock.now += 3601

    response = await client.get("/me", headers=bearer(mint(KEY)))

    assert response.status_code == 200
    assert auth0.jwks_fetches == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"sub": "aBcD1234@clients", "gty": "client-credentials"},
        {"sub": "aBcD1234@clients"},
        {"sub": "auth0|alice", "gty": "client-credentials"},
    ],
    ids=["both", "sub-suffix", "gty-claim"],
)
async def test_client_credentials_token_is_403(
    client: AsyncClient, users: FakeUserRepository, overrides: dict[str, str]
) -> None:
    response = await client.get("/me", headers=bearer(mint(KEY, **overrides)))

    assert_error(response, 403, "user_token_required")
    assert users.by_sub == {}


async def test_deleted_account_is_403(client: AsyncClient, users: FakeUserRepository) -> None:
    user = await users.get_or_create_by_auth_sub("auth0|alice")
    user.deleted_at = datetime.now(UTC)

    response = await client.get("/me", headers=bearer(mint(KEY, sub="auth0|alice")))

    assert_error(response, 403, "account_deleted")


async def test_require_role_rejects_other_roles(client: AsyncClient) -> None:
    response = await client.get("/admin-only", headers=bearer(mint(KEY)))

    assert_error(response, 403, "forbidden")


async def test_require_role_allows_matching_role(
    client: AsyncClient, users: FakeUserRepository
) -> None:
    (await users.get_or_create_by_auth_sub("auth0|boss")).role = UserRole.ADMIN

    response = await client.get("/admin-only", headers=bearer(mint(KEY, sub="auth0|boss")))

    assert response.status_code == 200


async def test_auth_not_configured_is_503() -> None:
    app = create_app()
    app.dependency_overrides[get_token_verifier] = lambda: None
    app.dependency_overrides[get_user_repository] = FakeUserRepository

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/me", headers=bearer(mint(KEY)))

    assert_error(response, 503, "auth_not_configured")
