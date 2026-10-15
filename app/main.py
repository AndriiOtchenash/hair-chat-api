from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI

from app.api import health, me
from app.core.errors import install_error_handlers
from app.core.security import current_user, get_token_verifier
from app.db.session import get_engine


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

    # Closed by default: every business router goes here. current_user is cached per request,
    # so endpoints that also take CurrentUser do not validate the token twice.
    protected = APIRouter(dependencies=[Depends(current_user)])
    protected.include_router(me.router)
    app.include_router(protected)
    return app


app = create_app()
