"""Guard against forgotten auth: every route outside the allowlist must demand a token."""

import re
from collections.abc import AsyncIterator

import pytest
from fastapi.routing import APIRoute, iter_route_contexts
from httpx import ASGITransport, AsyncClient
from starlette.routing import Route

from app.core.security import get_token_verifier
from app.main import create_app
from tests.auth_fakes import FakeAuth0, SigningKey

PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}
DUMMY_PATH_PARAM = "01890000-0000-7000-8000-000000000000"

app = create_app()


def _protected_requests() -> list[tuple[str, str]]:
    requests = []
    # iter_route_contexts flattens included routers (app.routes holds them unexpanded)
    # and reports each route's effective path, i.e. with router prefixes applied.
    for route in iter_route_contexts(app.routes):
        if not isinstance(route.original_route, APIRoute | Route) or route.path is None:
            pytest.fail(f"{route!r}: unknown route type, extend this test to cover it")
        if route.path in PUBLIC_PATHS:
            continue
        path = re.sub(r"\{[^}]+\}", DUMMY_PATH_PARAM, route.path)
        requests += [(method, path) for method in sorted(route.methods or ()) if method != "HEAD"]
    return requests


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    verifier = FakeAuth0(SigningKey("key-1")).verifier()
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    await verifier.aclose()


def test_there_are_protected_routes_to_check() -> None:
    assert ("GET", "/me") in _protected_requests()


@pytest.mark.parametrize(("method", "path"), _protected_requests())
async def test_route_outside_allowlist_requires_token(
    client: AsyncClient, method: str, path: str
) -> None:
    response = await client.request(method, path)

    assert response.status_code == 401, f"{method} {path} is reachable without a token"
    assert response.json()["code"] == "not_authenticated"
