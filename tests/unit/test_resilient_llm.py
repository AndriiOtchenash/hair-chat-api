from collections.abc import Sequence
from dataclasses import asdict, replace

import pytest

from app.db.models import LLMCallStatus
from app.llm.base import (
    AllProvidersDown,
    ChatMessage,
    Done,
    LLMEvent,
    LLMRequestError,
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderTimeout,
    Restart,
    StreamBroken,
    StreamEnd,
    Token,
)
from app.llm.resilient import COOLDOWN_SECONDS, ResilientLLM
from tests.llm_fakes import (
    FailingLLMCallRecorder,
    FakeLLMCallRecorder,
    FakeLLMProvider,
    Scenario,
    fail_before_first,
    fail_mid_stream,
    hang,
    success,
    truncated,
)

MESSAGES: list[ChatMessage] = [{"role": "user", "content": "secret question about my scalp"}]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_llm(
    *providers: FakeLLMProvider,
    recorder: FakeLLMCallRecorder | FailingLLMCallRecorder | None = None,
    clock: FakeClock | None = None,
    first_token_timeout: float = 0.05,
) -> ResilientLLM:
    return ResilientLLM(
        providers,
        recorder or FakeLLMCallRecorder(),
        first_token_timeout=first_token_timeout,
        clock=clock or FakeClock(),
    )


async def collect(llm: ResilientLLM, messages: Sequence[ChatMessage] = MESSAGES) -> list[LLMEvent]:
    """Events with Done.call_ids blanked (random ids); test_done_carries_call_ids checks them."""
    return [
        replace(event, call_ids=()) if isinstance(event, Done) else event
        async for event in llm.stream(messages, max_tokens=100)
    ]


def text_of(events: list[LLMEvent]) -> str:
    """The answer as a client sees it: tokens after the last Restart."""
    text = ""
    for event in events:
        if isinstance(event, Restart):
            text = ""
        elif isinstance(event, Token):
            text += event.text
    return text


@pytest.fixture
def recorder() -> FakeLLMCallRecorder:
    return FakeLLMCallRecorder()


async def test_success_streams_tokens_then_done(recorder: FakeLLMCallRecorder) -> None:
    primary = FakeLLMProvider("gemini", "flash").script(success("Hel", "lo"))

    events = await collect(make_llm(primary, recorder=recorder))

    assert events == [
        Token("Hel", "gemini", "flash"),
        Token("lo", "gemini", "flash"),
        Done("gemini", "flash"),
    ]
    assert [(c.provider, c.model, c.status, c.error) for c in recorder.calls] == [
        ("gemini", "flash", LLMCallStatus.OK, None)
    ]


async def test_done_carries_call_ids_of_every_attempt(recorder: FakeLLMCallRecorder) -> None:
    primary = FakeLLMProvider("gemini", "flash").script(fail_before_first(ProviderTimeout()))
    backup = FakeLLMProvider("groq", "oss").script(success("answer"))

    events = [e async for e in make_llm(primary, backup, recorder=recorder).stream(MESSAGES, 100)]

    done = events[-1]
    assert isinstance(done, Done)
    assert done.call_ids == tuple(c.id for c in recorder.calls)
    assert len(set(done.call_ids)) == 2


async def test_done_reports_truncation_and_usage_of_answering_link() -> None:
    primary = FakeLLMProvider("gemini", "flash").script(truncated("cut"))

    events = await collect(make_llm(primary))

    assert events[-1] == Done("gemini", "flash", truncated=True, tokens_in=10, tokens_out=20)


async def test_not_truncated_when_finish_reason_is_stop() -> None:
    primary = FakeLLMProvider("gemini", "flash").script(
        success("all", end=StreamEnd(finish_reason="stop"))
    )

    events = await collect(make_llm(primary))

    assert events[-1] == Done("gemini", "flash", truncated=False)


async def test_stream_end_without_any_token_is_empty_response() -> None:
    primary = FakeLLMProvider("gemini", "flash").script(Scenario(end=StreamEnd("stop")))
    backup = FakeLLMProvider("groq", "oss").script(success("text"))
    recorder = FakeLLMCallRecorder()

    assert text_of(await collect(make_llm(primary, backup, recorder=recorder))) == "text"
    assert recorder.calls[0].error == "empty_response"


@pytest.mark.parametrize(
    "error",
    [
        ProviderConnectionError(),
        ProviderTimeout(),
        ProviderHTTPError(408, "APIStatusError"),
        ProviderHTTPError(429, "RateLimitError"),
        ProviderHTTPError(503, "InternalServerError"),
        ProviderHTTPError(401, "AuthenticationError"),
        ProviderHTTPError(403, "PermissionDeniedError"),
    ],
    ids=lambda e: e.code,
)
async def test_failure_before_first_token_falls_back(
    error: ProviderHTTPError, recorder: FakeLLMCallRecorder
) -> None:
    primary = FakeLLMProvider("gemini", "flash").script(fail_before_first(error))
    backup = FakeLLMProvider("groq", "oss").script(success("answer"))

    events = await collect(make_llm(primary, backup, recorder=recorder))

    assert events == [Token("answer", "groq", "oss"), Done("groq", "oss")]
    assert [(c.provider, c.status, c.error, c.fallback_from) for c in recorder.calls] == [
        ("gemini", LLMCallStatus.FALLBACK, error.code, None),
        ("groq", LLMCallStatus.OK, None, "gemini:flash"),
    ]


async def test_misconfigured_provider_is_logged_as_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    primary = FakeLLMProvider("gemini", "flash").script(
        fail_before_first(ProviderHTTPError(401, "AuthenticationError"))
    )

    await collect(make_llm(primary, FakeLLMProvider("groq", "oss")))

    failed = [r for r in caplog.records if "attempt failed" in r.getMessage()]
    assert [r.levelname for r in failed] == ["ERROR"]


async def test_first_token_timeout_falls_back(recorder: FakeLLMCallRecorder) -> None:
    primary = FakeLLMProvider("gemini", "flash").script(hang())
    backup = FakeLLMProvider("groq", "oss").script(success("late but fine"))

    events = await collect(make_llm(primary, backup, recorder=recorder))

    assert text_of(events) == "late but fine"
    assert recorder.calls[0].error == "first_token_timeout"
    assert primary.closed_streams == 1


async def test_break_mid_stream_restarts_with_full_answer_from_next(
    recorder: FakeLLMCallRecorder,
) -> None:
    primary = FakeLLMProvider("gemini", "flash").script(
        fail_mid_stream(StreamBroken(), "Wash ", "every")
    )
    backup = FakeLLMProvider("groq", "oss").script(success("Wash ", "twice ", "a week"))

    events = await collect(make_llm(primary, backup, recorder=recorder))

    assert events == [
        Token("Wash ", "gemini", "flash"),
        Token("every", "gemini", "flash"),
        Restart(reason="stream_broken"),
        Token("Wash ", "groq", "oss"),
        Token("twice ", "groq", "oss"),
        Token("a week", "groq", "oss"),
        Done("groq", "oss"),
    ]
    assert text_of(events) == "Wash twice a week"
    assert [c.status for c in recorder.calls] == [LLMCallStatus.FALLBACK, LLMCallStatus.OK]


async def test_400_is_not_retried(recorder: FakeLLMCallRecorder) -> None:
    primary = FakeLLMProvider("gemini", "flash").script(
        fail_before_first(ProviderHTTPError(400, "BadRequestError"))
    )
    backup = FakeLLMProvider("groq", "oss")

    with pytest.raises(LLMRequestError) as exc_info:
        await collect(make_llm(primary, backup, recorder=recorder))

    assert exc_info.value.code == "BadRequestError:400"
    assert backup.calls == 0
    assert [(c.status, c.error) for c in recorder.calls] == [
        (LLMCallStatus.ERROR, "BadRequestError:400")
    ]


async def test_invalid_key_reported_as_400_falls_back() -> None:
    primary = FakeLLMProvider("gemini", "flash").script(
        fail_before_first(ProviderHTTPError(400, "BadRequestError", invalid_api_key=True))
    )
    backup = FakeLLMProvider("groq", "oss").script(success("ok"))

    assert text_of(await collect(make_llm(primary, backup))) == "ok"


async def test_400_does_not_count_towards_cooldown() -> None:
    bad_request = ProviderHTTPError(400, "BadRequestError")
    primary = FakeLLMProvider("gemini", "flash").script(
        *[fail_before_first(bad_request)] * 3, success("fine")
    )
    llm = make_llm(primary)
    for _ in range(3):
        with pytest.raises(LLMRequestError):
            await collect(llm)

    assert text_of(await collect(llm)) == "fine"


async def test_all_providers_down(recorder: FakeLLMCallRecorder) -> None:
    primary = FakeLLMProvider("gemini", "flash").script(
        fail_before_first(ProviderHTTPError(429, "RateLimitError"))
    )
    backup = FakeLLMProvider("groq", "oss").script(fail_before_first(ProviderConnectionError()))

    with pytest.raises(AllProvidersDown) as exc_info:
        await collect(make_llm(primary, backup, recorder=recorder))

    assert exc_info.value.attempts == [
        ("gemini:flash", "RateLimitError:429"),
        ("groq:oss", "connection_error"),
    ]
    # The last attempt has nothing to fall back to.
    assert [c.status for c in recorder.calls] == [LLMCallStatus.FALLBACK, LLMCallStatus.ERROR]


async def test_last_link_breaking_mid_stream_raises_without_restart() -> None:
    only = FakeLLMProvider("gemini", "flash").script(fail_mid_stream(StreamBroken(), "part"))
    events: list[LLMEvent] = []

    with pytest.raises(AllProvidersDown):
        async for event in make_llm(only).stream(MESSAGES, max_tokens=100):
            events.append(event)

    assert events == [Token("part", "gemini", "flash")]


async def test_empty_answer_falls_back() -> None:
    primary = FakeLLMProvider("gemini", "flash").script(Scenario(tokens=()))
    backup = FakeLLMProvider("groq", "oss").script(success("text"))

    assert text_of(await collect(make_llm(primary, backup))) == "text"


async def test_cooldown_after_three_consecutive_failures() -> None:
    clock = FakeClock()
    recorder = FakeLLMCallRecorder()
    primary = FakeLLMProvider("gemini", "flash").script(
        *[fail_before_first(ProviderHTTPError(503, "InternalServerError"))] * 3
    )
    backup = FakeLLMProvider("groq", "oss")
    llm = make_llm(primary, backup, recorder=recorder, clock=clock)

    for _ in range(3):
        await collect(llm)
    assert primary.calls == 3

    # Fourth request: primary is skipped without a call, backup answers directly.
    clock.now += COOLDOWN_SECONDS - 1
    events = await collect(llm)
    assert primary.calls == 3
    assert events[-1] == Done("groq", "oss")
    assert recorder.calls[-1].fallback_from is None

    # After the cooldown the primary is tried again.
    clock.now += 1
    events = await collect(llm)
    assert primary.calls == 4
    assert events[-1] == Done("gemini", "flash")


async def test_success_resets_consecutive_failures() -> None:
    error = fail_before_first(ProviderTimeout())
    primary = FakeLLMProvider("gemini", "flash").script(
        error, error, success("ok"), error, error, success("ok")
    )
    llm = make_llm(primary, FakeLLMProvider("groq", "oss"))

    for _ in range(6):
        await collect(llm)

    assert primary.calls == 6  # never reached 3 in a row, never skipped


async def test_all_links_in_cooldown_raises_all_providers_down() -> None:
    only = FakeLLMProvider("gemini", "flash").script(*[fail_before_first(ProviderTimeout())] * 3)
    llm = make_llm(only)
    for _ in range(3):
        with pytest.raises(AllProvidersDown):
            await collect(llm)

    with pytest.raises(AllProvidersDown) as exc_info:
        await collect(llm)

    assert exc_info.value.attempts == [("gemini:flash", "cooldown")]
    assert only.calls == 3


async def test_context_budget_is_minimum_over_chain() -> None:
    llm = make_llm(
        FakeLLMProvider("gemini", "flash", max_context_tokens=1_048_576),
        FakeLLMProvider("groq", "oss", max_context_tokens=131_072),
    )

    assert llm.max_context_tokens == 131_072


async def test_records_contain_no_message_content(recorder: FakeLLMCallRecorder) -> None:
    primary = FakeLLMProvider("gemini", "flash").script(
        fail_mid_stream(StreamBroken(), "your scalp")
    )

    await collect(make_llm(primary, FakeLLMProvider("groq", "oss"), recorder=recorder))

    stored = " ".join(str(value) for call in recorder.calls for value in asdict(call).values())
    assert "scalp" not in stored


async def test_failing_recorder_does_not_break_the_answer() -> None:
    primary = FakeLLMProvider("gemini", "flash").script(fail_before_first(ProviderTimeout()))
    backup = FakeLLMProvider("groq", "oss").script(success("still here"))

    events = await collect(make_llm(primary, backup, recorder=FailingLLMCallRecorder()))

    assert text_of(events) == "still here"


async def test_consumer_stopping_early_closes_stream_and_is_not_a_failure(
    recorder: FakeLLMCallRecorder,
) -> None:
    primary = FakeLLMProvider("gemini", "flash").script(success("a", "b", "c"), success("again"))
    llm = make_llm(primary, recorder=recorder)

    stream = llm.stream(MESSAGES, max_tokens=100)
    assert await anext(stream) == Token("a", "gemini", "flash")
    await stream.aclose()

    assert primary.closed_streams == 1
    assert recorder.calls == []
    assert text_of(await collect(llm)) == "again"


def test_empty_chain_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        make_llm()
