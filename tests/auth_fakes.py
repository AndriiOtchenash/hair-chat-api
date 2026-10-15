"""Test-only Auth0 stand-in: our own RSA keys, a JWKS served by httpx.MockTransport, token minting.

Nothing here talks to the network.
"""

import time
from typing import Any

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from app.core.security import JWKSCache, TokenVerifier

ISSUER = "https://test-tenant.auth0.test/"
AUDIENCE = "https://api.hair-chat.test"
JWKS_URL = "https://test-tenant.auth0.test/.well-known/jwks.json"


class SigningKey:
    def __init__(self, kid: str) -> None:
        self.kid = kid
        self.private = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def jwk(self) -> dict[str, Any]:
        jwk = RSAAlgorithm.to_jwk(self.private.public_key(), as_dict=True)
        return {**jwk, "kid": self.kid, "use": "sig", "alg": "RS256"}


class FakeAuth0:
    """Serves the JWKS for `published` keys and counts how often it was fetched."""

    def __init__(self, *published: SigningKey) -> None:
        self.published = list(published)
        self.jwks_fetches = 0
        self.status_code = 200

    def _handle(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == JWKS_URL
        self.jwks_fetches += 1
        if self.status_code != 200:
            return httpx.Response(self.status_code)
        return httpx.Response(200, json={"keys": [k.jwk() for k in self.published]})

    def verifier(self, clock: Any = time.monotonic) -> TokenVerifier:
        http = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))
        jwks = JWKSCache(JWKS_URL, http, clock=clock)
        return TokenVerifier(jwks, issuer=ISSUER, audience=AUDIENCE)


def mint(key: SigningKey, *, sub: str = "auth0|alice", **overrides: Any) -> str:
    """RS256 token valid for 5 minutes; pass claim=None to drop a claim."""
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": [AUDIENCE, "https://test-tenant.auth0.test/userinfo"],
        "sub": sub,
        "iat": now,
        "exp": now + 300,
        **overrides,
    }
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key.private, algorithm="RS256", headers={"kid": key.kid})


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
