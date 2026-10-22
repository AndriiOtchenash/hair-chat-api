import json

import httpx
import pytest
from pydantic import SecretStr

from app.core.auth0_mgmt import Auth0Management, Auth0ManagementError

DOMAIN = "test-tenant.auth0.test"
SECRET = "m2m-client-secret"


class FakeManagementApi:
    """Auth0 token endpoint + DELETE /api/v2/users/{id}, served by httpx.MockTransport."""

    def __init__(self) -> None:
        self.token_requests: list[dict[str, object]] = []
        self.deleted: list[str] = []
        self.token_status = 200
        self.delete_status = 204
        self.raise_on_delete: Exception | None = None
        self.expires_in = 86400

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth/token":
            self.token_requests.append(json.loads(request.content))
            if self.token_status != 200:
                return httpx.Response(self.token_status, json={"error": SECRET})
            return httpx.Response(
                200, json={"access_token": "mgmt-token", "expires_in": self.expires_in}
            )
        assert request.method == "DELETE"
        assert request.headers["authorization"] == "Bearer mgmt-token"
        if self.raise_on_delete is not None:
            raise self.raise_on_delete
        self.deleted.append(request.url.raw_path.decode())
        return httpx.Response(self.delete_status)


@pytest.fixture
def api() -> FakeManagementApi:
    return FakeManagementApi()


def _client(api: FakeManagementApi, clock: list[float] | None = None) -> Auth0Management:
    now = clock if clock is not None else [0.0]
    return Auth0Management(
        DOMAIN,
        "m2m-client",
        SecretStr(SECRET),
        httpx.AsyncClient(transport=httpx.MockTransport(api.handle)),
        clock=lambda: now[0],
    )


async def test_delete_user_gets_token_and_deletes_url_encoded_sub(api: FakeManagementApi) -> None:
    client = _client(api)

    await client.delete_user("auth0|abc 1")

    assert api.token_requests == [
        {
            "grant_type": "client_credentials",
            "client_id": "m2m-client",
            "client_secret": SECRET,
            "audience": f"https://{DOMAIN}/api/v2/",
        }
    ]
    assert api.deleted == ["/api/v2/users/auth0%7Cabc%201"]
    await client.aclose()


async def test_token_is_cached_until_shortly_before_expiry(api: FakeManagementApi) -> None:
    clock = [0.0]
    client = _client(api, clock)
    api.expires_in = 3600

    await client.delete_user("auth0|a")
    clock[0] = 3500.0
    await client.delete_user("auth0|b")
    clock[0] = 3541.0  # within the 60 s margin
    await client.delete_user("auth0|c")

    assert len(api.token_requests) == 2
    await client.aclose()


async def test_missing_user_counts_as_deleted(api: FakeManagementApi) -> None:
    api.delete_status = 404

    await _client(api).delete_user("auth0|gone")


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_delete_failure_raises_with_status_only(api: FakeManagementApi, status: int) -> None:
    api.delete_status = status

    with pytest.raises(Auth0ManagementError, match=f"HTTP {status}"):
        await _client(api).delete_user("auth0|a")


async def test_token_failure_raises_without_the_response_body(api: FakeManagementApi) -> None:
    api.token_status = 401

    with pytest.raises(Auth0ManagementError) as info:
        await _client(api).delete_user("auth0|a")

    assert str(info.value) == "token: HTTP 401"
    assert SECRET not in str(info.value)


async def test_network_error_raises_management_error(api: FakeManagementApi) -> None:
    api.raise_on_delete = httpx.ConnectTimeout("timed out")

    with pytest.raises(Auth0ManagementError, match="ConnectTimeout"):
        await _client(api).delete_user("auth0|a")


async def test_rejected_token_is_fetched_again_next_time(api: FakeManagementApi) -> None:
    client = _client(api)
    api.delete_status = 401
    with pytest.raises(Auth0ManagementError):
        await client.delete_user("auth0|a")

    api.delete_status = 204
    await client.delete_user("auth0|a")

    assert len(api.token_requests) == 2
