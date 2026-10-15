from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI

from app.api import consents, health, me
from app.core.consents import require_consents
from app.core.errors import install_error_handlers
from app.core.security import current_user, get_token_verifier
from app.db.session import get_engine

# Token required (current_user).
PROTECTED_ROUTERS: tuple[APIRouter, ...] = (me.router, consents.router)
# Token + all consents for the current CONSENT_VERSION (CLAUDE.md rule 7): chat goes here.
CONSENTED_ROUTERS: tuple[APIRouter, ...] = ()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await get_engine().dispose()
    if (verifier := get_token_verifier()) is not None:
        await verifier.aclose()


def create_app() -> FastAPI:
    # No Swagger OAuth2 redirect page: tokens are pasted into "Authorize" as plain Bearer.
    app = FastAPI(title="hair-chat-api", lifespan=lifespan, swagger_ui_oauth2_redirect_url=None)
    install_error_handlers(app)

    # Public allowlist: /health plus FastAPI's /docs, /redoc, /openapi.json.
    # tests/unit/test_route_protection.py fails if anything else answers without a token.
    app.include_router(health.router)

    # Closed by default: every business router goes into one of the tuples above.
    # Dependencies are cached per request, so endpoints that also take CurrentUser do not
    # validate the token twice. Levels nest: consented routes get current_user from protected.
    consented = APIRouter(dependencies=[Depends(require_consents)])
    for router in CONSENTED_ROUTERS:
        consented.include_router(router)
    protected = APIRouter(dependencies=[Depends(current_user)])
    for router in PROTECTED_ROUTERS:
        protected.include_router(router)
    protected.include_router(consented)
    app.include_router(protected)
    return app


app = create_app()
