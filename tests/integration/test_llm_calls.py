from sqlalchemy import select

from app.db.models import LLMCall, LLMCallStatus
from app.db.repositories.llm_calls import LLMCallRecord, SqlLLMCallRecorder
from app.db.session import get_sessionmaker
from app.llm.base import ProviderTimeout
from app.llm.resilient import ResilientLLM
from tests.llm_fakes import FakeLLMProvider, fail_before_first, success


async def test_fallback_attempts_are_written_to_llm_calls(app_settings_env: None) -> None:
    primary = FakeLLMProvider("gemini", "it-primary").script(fail_before_first(ProviderTimeout()))
    backup = FakeLLMProvider("groq", "it-backup").script(success("answer"))
    llm = ResilientLLM(
        [primary, backup], SqlLLMCallRecorder(get_sessionmaker()), first_token_timeout=1
    )

    _ = [event async for event in llm.stream([{"role": "user", "content": "q"}], 10)]

    async with get_sessionmaker()() as session:
        rows = (
            await session.scalars(
                select(LLMCall)
                .where(LLMCall.model.in_(["it-primary", "it-backup"]))
                .order_by(LLMCall.id)
            )
        ).all()
    assert [(r.provider, r.model, r.status, r.error, r.fallback_from) for r in rows] == [
        ("gemini", "it-primary", LLMCallStatus.FALLBACK, "timeout", None),
        ("groq", "it-backup", LLMCallStatus.OK, None, "gemini:it-primary"),
    ]
    assert all(r.message_id is None and r.latency_ms is not None for r in rows)


async def test_record_round_trips_enum_value(app_settings_env: None) -> None:
    await SqlLLMCallRecorder(get_sessionmaker()).record(
        LLMCallRecord("groq", "it-enum", latency_ms=5, status=LLMCallStatus.ERROR, error="x:1")
    )

    async with get_sessionmaker()() as session:
        raw = await session.scalar(select(LLMCall.status).where(LLMCall.model == "it-enum"))
    assert raw is LLMCallStatus.ERROR
