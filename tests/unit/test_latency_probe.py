import io

import pytest

from app.core.config import Settings
from app.llm.base import LLMProvider, ProviderHTTPError, StreamEnd
from app.llm.registry import MODELS
from app.services.context_builder import system_message
from app.services.prompts import PROMPT_VERSION, FilePromptProvider
from scripts import latency_probe
from tests.llm_fakes import FakeLLMProvider, Scenario, fail_before_first, hang, success

LINK = "gemini:gemini-3.5-flash-lite"


def factory(provider: FakeLLMProvider) -> latency_probe.ProviderFactory:
    def make(settings: Settings, link: str, thinking: str) -> LLMProvider:
        return provider

    return make


async def run(provider: FakeLLMProvider, count: int, request_timeout: float = 5) -> tuple[int, str]:
    out = io.StringIO()
    code = await latency_probe.run(
        link=LINK,
        count=count,
        thinking="registry",
        request_timeout=request_timeout,
        kb=False,
        provider_factory=factory(provider),
        out=out,
    )
    return code, out.getvalue()


async def test_sends_the_real_prompt_and_question_and_prints_timings_and_stats() -> None:
    end = StreamEnd(finish_reason="stop", tokens_in=900, tokens_out=42)
    provider = FakeLLMProvider("gemini", "flash-lite").script(
        success("Мойте ", "голову", end=end), success("ok")
    )

    code, out = await run(provider, count=2)

    assert code == 0
    assert provider.calls == 2
    system, question = provider.received[0]
    prompt = await FilePromptProvider(PROMPT_VERSION).get_active()
    assert system == {"role": "system", "content": system_message(prompt, [])}
    assert question == {"role": "user", "content": latency_probe.DEFAULT_QUESTION}
    assert "#1  ttft" in out and "tokens_out 42" in out and "#2  ttft" in out
    assert "2/2 ok, 0 errors" in out
    assert "ttft : p50" in out and "total: p50" in out


async def test_errors_and_timeouts_are_reported_per_request_and_exit_1() -> None:
    provider = FakeLLMProvider("gemini", "flash-lite").script(
        fail_before_first(ProviderHTTPError(503, "InternalServerError")), hang(), success()
    )

    code, out = await run(provider, count=3, request_timeout=0.05)

    assert code == 1
    assert "#1" in out and "ERROR" in out
    assert "ERROR timeout after 0.05s" in out
    assert "1/3 ok, 2 errors" in out
    assert provider.closed_streams == 3


async def test_a_stream_without_text_is_an_error() -> None:
    provider = FakeLLMProvider("gemini", "flash-lite").script(Scenario(end=StreamEnd()))

    code, out = await run(provider, count=1)

    assert code == 1
    assert "ERROR empty_response" in out


def test_summary_counts_first_tokens_later_than_the_chat_timeout() -> None:
    samples = [
        latency_probe.Sample(ttft=2.0, total=4.0),
        latency_probe.Sample(ttft=9.0, total=12.0),
        latency_probe.Sample(error="timeout after 60s"),
    ]

    lines = latency_probe.summary(samples, first_token_timeout=8)

    assert lines[0] == "2/3 ok, 1 errors"
    assert lines[1] == "ttft : p50 5.50s  max 9.00s"
    assert lines[3].endswith(": 1")


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [
        ("registry", MODELS[LINK].reasoning_effort),
        ("default", None),
        ("off", "none"),
        ("low", "low"),
    ],
)
def test_thinking_maps_to_reasoning_effort(thinking: str, expected: str | None) -> None:
    assert latency_probe.reasoning_effort(thinking, LINK) == expected


def test_cli_rejects_a_model_missing_from_the_registry() -> None:
    with pytest.raises(SystemExit):
        latency_probe.main(["--link", "gemini:unknown"])
