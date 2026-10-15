from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, me
from app.core.security import get_token_verifier
from app.db.session import get_engine


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await get_engine().dispose()
    if (verifier := get_token_verifier()) is not None:
        await verifier.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="hair-chat-api", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(me.router)
    return app


app = create_app()
