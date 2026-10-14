import asyncio
import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

DB_PING_TIMEOUT_SECONDS = 2.0


class HealthResponse(BaseModel):
    status: Literal["ok"]
    db: Literal["ok", "down"]


@router.get("/health")
async def health(session: Annotated[AsyncSession, Depends(get_session)]) -> HealthResponse:
    # Liveness stays 200 even if the DB is unreachable; `db` reports its state separately.
    try:
        async with asyncio.timeout(DB_PING_TIMEOUT_SECONDS):
            await session.execute(text("SELECT 1"))
        db: Literal["ok", "down"] = "ok"
    except Exception as exc:
        logger.warning("health: db ping failed: %s", type(exc).__name__)
        db = "down"
    return HealthResponse(status="ok", db=db)
