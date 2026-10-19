import logging

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import LLMCallStatus
from app.db.repositories.llm_calls import LLMCallRecord, SqlLLMCallRecorder


async def test_unreachable_database_does_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    # Port 1: nothing listens there, the connect fails fast.
    engine = create_async_engine("postgresql+asyncpg://u:p@127.0.0.1:1/db")
    recorder = SqlLLMCallRecorder(async_sessionmaker(engine), timeout=5)

    await recorder.record(LLMCallRecord("gemini", "flash", latency_ms=12, status=LLMCallStatus.OK))

    assert any("llm_calls write failed" in r.getMessage() for r in caplog.records)
    assert all(r.levelno <= logging.WARNING for r in caplog.records)
    await engine.dispose()


async def test_slow_database_is_bounded_by_timeout() -> None:
    # A blackhole address: connect hangs, the recorder's own timeout must cut it.
    engine = create_async_engine("postgresql+asyncpg://u:p@10.255.255.1:5432/db")
    recorder = SqlLLMCallRecorder(async_sessionmaker(engine), timeout=0.2)

    await recorder.record(
        LLMCallRecord("gemini", "flash", latency_ms=12, status=LLMCallStatus.ERROR, error="x")
    )
    await engine.dispose()
