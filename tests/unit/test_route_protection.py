"""Guard against forgotten auth.

- every route outside the public allowlist must demand a token (401);
- every route on the consented level must, with a valid token but no consents, answer 403.
"""

import re
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import pytest
from fastapi import APIRouter
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute, RouteContext, iter_route_contexts
from httpx import ASGITransport, AsyncClient
from starlette.routing import Route

import app.main as main
from app.core.consents import get_consent_repository, require_consents
from app.core.security import get_token_verifier, get_user_repository
from tests.auth_fakes import FakeAuth0, SigningKey, bearer, mint
from tests.fakes import FakeConsentRepository, FakeUserRepository

PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}
DUMMY_PATH_PARAM = "01890000-0000-7000-8000-000000000000"
KEY = SigningKey("key-1")
# Chat routes are consented, except these data-rights ones: deleting or renaming own
# conversations keeps working after the consents are revoked (like /me/export, DELETE /me).
CONSENT_FREE_CONVERSATION_ROUTES = {
    ("PATCH", f"/conversations/{DUMMY_PATH_PARAM}"),
    ("DELETE", f"/conversations/{DUMMY_PATH_PARAM}"),
    ("DELETE", "/conversations"),
}

# The consented level has no routers yet (chat comes later). A probe router keeps the
# consent check below from silently testing nothing.
_probe = APIRouter()


@_probe.get("/__consent-probe")
async def _consent_probe() -> None: ...


def _build_app() -> Any:
    original = main.CONSENTED_ROUTERS
    main.CONSENTED_ROUTERS = (*original, _probe)
    try:
        return main.create_app()
    finally:
        main.CONSENTED_ROUTERS = original


app = _build_app()


def _calls(dependant: Dependant) -> Iterator[Callable[..., Any] | None]:
    yield dependant.call
    for sub in dependant.dependencies:
        yield from _calls(sub)


def _requires_consents(route: RouteContext) -> bool:
    dependant = getattr(route, "dependant", None)
    return dependant is not None and require_consents in set(_calls(dependant))


def _requests(*, consented_only: bool = False) -> list[tuple[str, str]]:
    requests = []
    # iter_route_contexts flattens included routers (app.routes holds them unexpanded)
    # and reports each route's effective path and dependencies, i.e. with routers applied.
    for route in iter_route_contexts(app.routes):
        if not isinstance(route.original_route, APIRoute | Route) or route.path is None:
            pytest.fail(f"{route!r}: unknown route type, extend this test to cover it")
        if route.path in PUBLIC_PATHS:
            continue
        if consented_only and not _requires_consents(route):
            continue
        path = re.sub(r"\{[^}]+\}", DUMMY_PATH_PARAM, route.path)
        requests += [(method, path) for method in sorted(route.methods or ()) if method != "HEAD"]
    return requests


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    verifier = FakeAuth0(KEY).verifier()
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    app.dependency_overrides[get_user_repository] = FakeUserRepository
    app.dependency_overrides[get_consent_repository] = FakeConsentRepository
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    await verifier.aclose()


def test_route_lists_cover_known_routes() -> None:
    assert ("GET", "/me") in _requests()
    assert ("GET", "/me") not in _requests(consented_only=True)
    assert ("GET", "/consents") not in _requests(consented_only=True)
    assert ("GET", "/__consent-probe") in _requests(consented_only=True)


def test_conversation_routes_need_consents_except_the_data_rights_ones() -> None:
    conversation_routes = {r for r in _requests() if r[1].startswith("/conversations")}
    consented = set(_requests(consented_only=True))

    assert CONSENT_FREE_CONVERSATION_ROUTES <= conversation_routes
    assert conversation_routes - consented == CONSENT_FREE_CONVERSATION_ROUTES


@pytest.mark.parametrize(("method", "path"), _requests())
async def test_route_outside_allowlist_requires_token(
    client: AsyncClient, method: str, path: str
) -> None:
    response = await client.request(method, path)

    assert response.status_code == 401, f"{method} {path} is reachable without a token"
    assert response.json()["code"] == "not_authenticated"


@pytest.mark.parametrize(("method", "path"), _requests(consented_only=True))
async def test_consented_route_requires_consents(
    client: AsyncClient, method: str, path: str
) -> None:
    response = await client.request(method, path, headers=bearer(mint(KEY)))

    assert response.status_code == 403, f"{method} {path} is reachable without consents"
    assert response.json()["code"] == "consent_required"
