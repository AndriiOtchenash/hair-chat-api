"""Auth0 Management API client: deletes the Auth0 account behind a user (DELETE /me, ADR 0006).

Client-credentials token for `https://{domain}/api/v2/`, cached until shortly before it expires.
"""

import time
from collections.abc import Callable
from functools import lru_cache
from urllib.parse import quote

import httpx
from pydantic import SecretStr

from app.core.config import get_settings

MGMT_HTTP_TIMEOUT_SECONDS = 5.0
TOKEN_EXPIRY_MARGIN_SECONDS = 60.0


class Auth0ManagementError(Exception):
    """The Management API call failed; the message is a status or exception class, no bodies."""


class Auth0Management:
    def __init__(
        self,
        domain: str,
        client_id: str,
        client_secret: SecretStr,
        http: httpx.AsyncClient,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._base = f"https://{domain}"
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = http
        self._clock = clock
        self._token: str | None = None
        self._token_expires_at = float("-inf")

    async def delete_user(self, auth_sub: str) -> None:
        """Deletes the Auth0 user; an already missing user (404) counts as deleted."""
        token = await self._access_token()
        try:
            response = await self._http.delete(
                f"{self._base}/api/v2/users/{quote(auth_sub, safe='')}",
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            raise Auth0ManagementError(type(exc).__name__) from exc
        if response.status_code == 401:
            self._token = None  # revoked or rotated: fetch a new one next time
        if response.status_code not in (200, 204, 404):
            raise Auth0ManagementError(f"delete user: HTTP {response.status_code}")

    async def _access_token(self) -> str:
        now = self._clock()
        if self._token is not None and now < self._token_expires_at:
            return self._token
        try:
            response = await self._http.post(
                f"{self._base}/oauth/token",
                json={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret.get_secret_value(),
                    "audience": f"{self._base}/api/v2/",
                },
            )
        except httpx.HTTPError as exc:
            raise Auth0ManagementError(type(exc).__name__) from exc
        if response.status_code != 200:
            raise Auth0ManagementError(f"token: HTTP {response.status_code}")
        try:
            data = response.json()
            token = data["access_token"]
            expires_in = float(data.get("expires_in", 0))
        except (ValueError, KeyError, TypeError) as exc:
            raise Auth0ManagementError(f"token: {type(exc).__name__}") from exc
        if not isinstance(token, str) or not token:
            raise Auth0ManagementError("token: no access_token")
        self._token = token
        self._token_expires_at = now + expires_in - TOKEN_EXPIRY_MARGIN_SECONDS
        return token

    async def aclose(self) -> None:
        await self._http.aclose()


@lru_cache
def get_auth0_management() -> Auth0Management | None:
    """Process-wide client; None when AUTH0_DOMAIN / AUTH0_MGMT_CLIENT_* are not all set."""
    settings = get_settings()
    if not (
        settings.auth0_domain
        and settings.auth0_mgmt_client_id
        and settings.auth0_mgmt_client_secret
    ):
        return None
    return Auth0Management(
        settings.auth0_domain,
        settings.auth0_mgmt_client_id,
        settings.auth0_mgmt_client_secret,
        httpx.AsyncClient(timeout=MGMT_HTTP_TIMEOUT_SECONDS),
    )
