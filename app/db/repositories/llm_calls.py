import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import LLMCall, LLMCallStatus

logger = logging.getLogger(__name__)

DEFAULT_WRITE_TIMEOUT_S = 2.0


@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    """One attempt at one provider. `error` is a class/code only, never request or answer text."""

    provider: str
    model: str
    latency_ms: int
    status: LLMCallStatus
    error: str | None = None
    fallback_from: str | None = None


class LLMCallRecorder(Protocol):
    async def record(self, call: LLMCallRecord) -> None:
        """Best effort: must never raise, a broken audit log must not break the answer."""
        ...


class SqlLLMCallRecorder:
    """Writes each attempt in its own short session, independent of any request session."""

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        timeout: float = DEFAULT_WRITE_TIMEOUT_S,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._timeout = timeout

    async def record(self, call: LLMCallRecord) -> None:
        try:
            # Bounded: an unreachable DB would otherwise stall the stream on connect.
            async with asyncio.timeout(self._timeout), self._sessionmaker() as session:
                session.add(
                    LLMCall(
                        provider=call.provider,
                        model=call.model,
                        latency_ms=call.latency_ms,
                        status=call.status,
                        error=call.error,
                        fallback_from=call.fallback_from,
                    )
                )
                await session.commit()
        except Exception as exc:
            logger.warning(
                "llm_calls write failed: provider=%s model=%s status=%s error=%s",
                call.provider,
                call.model,
                call.status.value,
                type(exc).__name__,
            )
