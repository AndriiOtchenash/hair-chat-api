"""Auth0 access-token validation (RS256 via JWKS) and the FastAPI auth dependencies.

Roles come from our DB, never from token claims.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any

import httpx
import jwt
from fastapi import Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.errors import api_error
from app.db.models import User, UserRole
from app.db.repositories.users import SqlUserRepository, UserRepository
from app.db.session import get_session

logger = logging.getLogger(__name__)

ALGORITHMS = ["RS256"]  # whitelist: rejects alg=none and RS256->HS256 key confusion
CLOCK_LEEWAY_SECONDS = 60
JWKS_TTL_SECONDS = 3600.0
JWKS_MIN_REFRESH_INTERVAL_SECONDS = 60.0
JWKS_HTTP_TIMEOUT_SECONDS = 5.0


class InvalidTokenError(Exception):
    """Token is malformed, expired, for another audience/issuer, or badly signed."""


class JWKSUnavailableError(Exception):
    """Signing keys could not be fetched and none are cached."""


class JWKSCache:
    """Auth0 signing keys, cached for `ttl` seconds.

    An unknown `kid` (key rotation) triggers one extra fetch, but at most once per
    `min_refresh_interval`, so tokens with made-up kids cannot make us hammer Auth0.
    If a refresh fails while keys are cached, the stale keys keep working.
    """

    def __init__(
        self,
        url: str,
        http: httpx.AsyncClient,
        *,
        ttl: float = JWKS_TTL_SECONDS,
        min_refresh_interval: float = JWKS_MIN_REFRESH_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._url = url
        self._http = http
        self._ttl = ttl
        self._min_refresh_interval = min_refresh_interval
        self._clock = clock
        self._keys: dict[str, jwt.PyJWK] = {}
        self._expires_at = float("-inf")
        self._last_fetch_at = float("-inf")
        self._lock = asyncio.Lock()  # one fetch at a time; concurrent callers wait for it

    async def get_key(self, kid: str) -> jwt.PyJWK:
        async with self._lock:
            now = self._clock()
            if now >= self._expires_at:
                await self._refresh(now)
            elif not self._keys:
                raise JWKSUnavailableError("JWKS fetch failed recently")
            key = self._keys.get(kid)
            if key is None and now - self._last_fetch_at >= self._min_refresh_interval:
                await self._refresh(now)
                key = self._keys.get(kid)
        if key is None:
            raise InvalidTokenError("unknown kid")
        return key

    async def _refresh(self, now: float) -> None:
        self._last_fetch_at = now
        try:
            response = await self._http.get(self._url)
            response.raise_for_status()
            keys = _parse_jwks(response.json())
        except (httpx.HTTPError, ValueError, jwt.PyJWTError) as exc:
            logger.warning("auth: JWKS fetch failed: %s", type(exc).__name__)
            # Retry no sooner than min_refresh_interval; keep serving stale keys meanwhile.
            self._expires_at = now + self._min_refresh_interval
            if not self._keys:
                raise JWKSUnavailableError("JWKS fetch failed") from exc
            return
        logger.info("auth: JWKS loaded, kids=%s", sorted(keys))
        self._keys = keys
        self._expires_at = now + self._ttl

    async def aclose(self) -> None:
        await self._http.aclose()


def _parse_jwks(data: Any) -> dict[str, jwt.PyJWK]:
    if not isinstance(data, dict):
        raise ValueError("JWKS is not a JSON object")
    keys = jwt.PyJWKSet.from_dict(data).keys
    return {k.key_id: k for k in keys if k.key_id and k.public_key_use in (None, "sig")}


@dataclass(frozen=True)
class AccessToken:
    sub: str
    claims: dict[str, Any]

    @property
    def is_client_credentials(self) -> bool:
        """Machine-to-machine token (Auth0 client-credentials grant), not a person."""
        return self.claims.get("gty") == "client-credentials" or self.sub.endswith("@clients")


class TokenVerifier:
    def __init__(self, jwks: JWKSCache, *, issuer: str, audience: str) -> None:
        self._jwks = jwks
        self._issuer = issuer
        self._audience = audience

    async def verify(self, token: str) -> AccessToken:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise InvalidTokenError("malformed token") from exc
        kid = header.get("kid")
        if header.get("alg") not in ALGORITHMS or not isinstance(kid, str):
            raise InvalidTokenError("unsupported alg or missing kid")
        key = await self._jwks.get_key(kid)
        try:
            claims = jwt.decode(
                token,
                key=key.key,
                algorithms=ALGORITHMS,
                audience=self._audience,
                issuer=self._issuer,
                leeway=CLOCK_LEEWAY_SECONDS,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError as exc:
            raise InvalidTokenError(type(exc).__name__) from exc
        sub = claims["sub"]
        if not isinstance(sub, str) or not sub:
            raise InvalidTokenError("bad sub")
        return AccessToken(sub=sub, claims=claims)

    async def aclose(self) -> None:
        await self._jwks.aclose()


@lru_cache
def get_token_verifier() -> TokenVerifier | None:
    """Process-wide verifier (shares the JWKS cache); None when Auth0 is not configured (dev)."""
    settings = get_settings()
    if not (settings.auth0_domain and settings.auth0_audience):
        return None
    base = f"https://{settings.auth0_domain}"
    jwks = JWKSCache(
        f"{base}/.well-known/jwks.json",
        httpx.AsyncClient(timeout=JWKS_HTTP_TIMEOUT_SECONDS),
    )
    return TokenVerifier(jwks, issuer=f"{base}/", audience=settings.auth0_audience)


def get_user_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> UserRepository:
    return SqlUserRepository(session)


_bearer = HTTPBearer(auto_error=False)


async def current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    verifier: Annotated[TokenVerifier | None, Depends(get_token_verifier)],
    users: Annotated[UserRepository, Depends(get_user_repository)],
) -> User:
    if verifier is None:
        raise api_error(503, "auth_not_configured", "Authentication is not configured")
    if credentials is None:
        raise api_error(
            status.HTTP_401_UNAUTHORIZED,
            "not_authenticated",
            "Bearer token required",
            {"WWW-Authenticate": "Bearer"},
        )
    try:
        token = await verifier.verify(credentials.credentials)
    except InvalidTokenError as exc:
        logger.info("auth: token rejected: %s", exc)
        raise api_error(
            status.HTTP_401_UNAUTHORIZED,
            "invalid_token",
            "Invalid or expired token",
            {"WWW-Authenticate": 'Bearer error="invalid_token"'},
        ) from exc
    except JWKSUnavailableError as exc:
        # Not 401: the token may be fine, and a 401 would make the client log the user out.
        raise api_error(503, "auth_unavailable", "Authentication temporarily unavailable") from exc
    if token.is_client_credentials:
        raise api_error(
            status.HTTP_403_FORBIDDEN, "user_token_required", "A user access token is required"
        )
    user = await users.get_or_create_by_auth_sub(token.sub)
    if user.deleted_at is not None:
        raise api_error(status.HTTP_403_FORBIDDEN, "account_deleted", "Account has been deleted")
    return user


CurrentUser = Annotated[User, Depends(current_user)]


def require_role(*roles: UserRole) -> Callable[[User], Awaitable[User]]:
    """Dependency factory: `Depends(require_role(UserRole.ADMIN))` -> 403 for other roles."""
    allowed = frozenset(roles)

    async def dependency(user: CurrentUser) -> User:
        if user.role not in allowed:
            raise api_error(status.HTTP_403_FORBIDDEN, "forbidden", "Insufficient role")
        return user

    return dependency
