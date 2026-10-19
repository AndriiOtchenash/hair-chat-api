import io
from collections.abc import Sequence

import pytest

from app.core.config import Settings
from app.llm import cli
from app.llm.base import ProviderHTTPError, StreamBroken
from app.llm.resilient import ResilientLLM
from tests.llm_fakes import (
    FakeLLMCallRecorder,
    FakeLLMProvider,
    fail_before_first,
    fail_mid_stream,
    success,
)


def factory(*providers: FakeLLMProvider) -> cli.LLMFactory:
    def make(settings: Settings, skip: Sequence[str]) -> ResilientLLM:
        chain = [p for p in providers if f"{p.name}:{p.model}" not in skip]
        return ResilientLLM(chain, FakeLLMCallRecorder(), first_token_timeout=1)

    return make


async def run(*providers: FakeLLMProvider, skip: Sequence[str] = ()) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = await cli.run("question", skip, 100, llm_factory=factory(*providers), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


async def test_streams_answer_and_names_the_model() -> None:
    code, out, err = await run(FakeLLMProvider("gemini", "flash").script(success("Hi", "!")))

    assert code == 0
    assert out == "Hi!\n\n— answered by gemini:flash\n"
    assert "chain: gemini:flash" in err


async def test_skip_shows_fallback_model() -> None:
    code, out, _ = await run(
        FakeLLMProvider("gemini", "flash"),
        FakeLLMProvider("groq", "oss").script(success("from groq")),
        skip=["gemini:flash"],
    )

    assert code == 0
    assert out.endswith("— answered by groq:oss\n")


async def test_restart_is_visible() -> None:
    _, out, _ = await run(
        FakeLLMProvider("gemini", "flash").script(fail_mid_stream(StreamBroken(), "par")),
        FakeLLMProvider("groq", "oss").script(success("full")),
    )

    assert out.startswith("par\n[restart: stream_broken]\nfull")


async def test_all_down_exits_1() -> None:
    code, _, err = await run(
        FakeLLMProvider("gemini", "flash").script(
            fail_before_first(ProviderHTTPError(503, "InternalServerError"))
        )
    )

    assert code == 1
    assert "gemini:flash=InternalServerError:503" in err


async def test_bad_request_exits_2() -> None:
    code, _, err = await run(
        FakeLLMProvider("gemini", "flash").script(
            fail_before_first(ProviderHTTPError(400, "BadRequestError"))
        )
    )

    assert code == 2
    assert "no fallback" in err


async def test_config_error_exits_2() -> None:
    def broken(settings: Settings, skip: Sequence[str]) -> ResilientLLM:
        raise ValueError("GROQ_API_KEY is not set")

    err = io.StringIO()
    code = await cli.run("q", [], 10, llm_factory=broken, out=io.StringIO(), err=err)

    assert code == 2
    assert "GROQ_API_KEY" in err.getvalue()


def test_parser_collects_repeated_skip() -> None:
    args = cli._parser().parse_args(["q", "--skip", "a:b", "--skip", "c:d"])

    assert args.skip == ["a:b", "c:d"]


def test_main_without_question_shows_usage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main([])

    assert "usage" in capsys.readouterr().err
