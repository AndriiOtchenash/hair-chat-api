"""Latency probe for one registry model: time to first token and total time over N requests.

    uv run python -m scripts.latency_probe --link PROVIDER:MODEL [-n 5]
        [--thinking registry|default|off|minimal|low|medium|high] [--timeout 60]
        [--prompt system_v6] [--question TEXT] [--no-kb]

Sends the chat's real context for one of our own questions (never user data): the prompt file,
the <knowledge> block from the docs/kb articles (embedded in memory like scripts.prompt_eval,
drafts included) and the question. Calls the provider directly, without ResilientLLM: no
LLM_FIRST_TOKEN_TIMEOUT_S cut-off, no fallback, no retries, so the real time to the first content
token is seen; --timeout bounds each request instead (LLM_READ_TIMEOUT_S still applies to the
wait for the response and between chunks: that one shows up as ProviderTimeout). Not run in CI.

--thinking sets the `reasoning_effort` sent to the OpenAI-compatible endpoint:
registry = what the chat sends now (app/llm/registry.py MODELS); default = nothing sent, the
provider's own default; off = "none"; the rest are sent as is. A value the model refuses shows up
as an HTTP error per request.
No database: llm_calls are not recorded. Exit code 1 if any request failed.
"""

import argparse
import asyncio
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TextIO

from openai.types.shared import ReasoningEffort

from app.core.config import Settings, get_settings
from app.llm.base import ChatMessage, LLMProvider, ProviderError, StreamEnd
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.registry import MODELS, PROVIDERS
from app.services.prompts import PROMPT_VERSION, FilePromptProvider
from scripts.prompt_eval import NO_KB, Case, knowledge_for, messages
from scripts.rag_eval import DEFAULT_KB

DEFAULT_QUESTION = "Как часто мыть голову, если у меня перхоть?"
THINKING = ("registry", "default", "off", "minimal", "low", "medium", "high")

ProviderFactory = Callable[[Settings, str, str], LLMProvider]


@dataclass
class Sample:
    ttft: float | None = None  # seconds to the first content token
    total: float | None = None  # seconds to the end of the stream
    tokens_out: int | None = None
    finish_reason: str | None = None
    error: str | None = None


def reasoning_effort(thinking: str, link: str) -> ReasoningEffort:
    if thinking == "registry":
        return MODELS[link].reasoning_effort
    if thinking == "default":
        return None
    if thinking == "off":
        return "none"
    return thinking  # type: ignore[return-value]


def make_provider(settings: Settings, link: str, thinking: str) -> LLMProvider:
    provider, _, model = link.partition(":")
    spec = PROVIDERS[provider]
    keys = {"gemini": settings.gemini_api_key, "groq": settings.groq_api_key}
    api_key = keys.get(provider)
    if api_key is None:
        raise SystemExit(f"{spec.api_key_env} is not set")
    return OpenAICompatibleProvider(
        name=provider,
        model=model,
        max_context_tokens=MODELS[link].max_context_tokens,
        base_url=spec.base_url,
        api_key=api_key.get_secret_value(),
        connect_timeout=settings.llm_connect_timeout_s,
        read_timeout=settings.llm_read_timeout_s,
        reasoning_effort=reasoning_effort(thinking, link),
    )


async def measure(
    provider: LLMProvider,
    chat: list[ChatMessage],
    max_tokens: int,
    request_timeout: float,
    clock: Callable[[], float] = time.monotonic,
) -> Sample:
    sample = Sample()
    started = clock()
    tokens = provider.stream(chat, max_tokens)
    try:
        async with asyncio.timeout(request_timeout):
            async for item in tokens:
                if isinstance(item, StreamEnd):
                    sample.tokens_out = item.tokens_out
                    sample.finish_reason = item.finish_reason
                elif sample.ttft is None:
                    sample.ttft = clock() - started
        if sample.ttft is None:
            sample.error = "empty_response"
    except TimeoutError:
        sample.error = f"timeout after {request_timeout:g}s"
    except ProviderError as exc:
        sample.error = f"{type(exc).__name__}: {exc.code}"
    finally:
        sample.total = clock() - started  # for a failed request: how long until it failed
        if (aclose := getattr(tokens, "aclose", None)) is not None:
            await aclose()
    return sample


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:6.2f}s"


def _stats(name: str, values: Sequence[float]) -> str:
    if not values:
        return f"{name}: -"
    return f"{name}: p50 {statistics.median(values):.2f}s  max {max(values):.2f}s"


def summary(samples: Sequence[Sample], first_token_timeout: float) -> list[str]:
    ok = [s for s in samples if s.error is None]
    ttfts = [s.ttft for s in ok if s.ttft is not None]
    totals = [s.total for s in ok if s.total is not None]
    late = sum(t > first_token_timeout for t in ttfts)
    return [
        f"{len(ok)}/{len(samples)} ok, {len(samples) - len(ok)} errors",
        _stats("ttft ", ttfts),
        _stats("total", totals),
        f"ttft over LLM_FIRST_TOKEN_TIMEOUT_S ({first_token_timeout:g}s): {late}",
    ]


async def run(
    *,
    link: str,
    count: int,
    thinking: str,
    request_timeout: float,
    prompt_version: str = PROMPT_VERSION,
    question: str = DEFAULT_QUESTION,
    kb: bool = True,
    provider_factory: ProviderFactory = make_provider,
    out: TextIO = sys.stdout,
) -> int:
    settings = get_settings()
    prompt = await FilePromptProvider(prompt_version).get_active()
    case = Case(id="probe", group="probe", lang="ru", text=question, expect=())
    context = (await knowledge_for([case], DEFAULT_KB))[0] if kb else NO_KB
    chat = messages(prompt, case, context.knowledge)
    provider = provider_factory(settings, link, thinking)
    print(
        f"{link}, thinking={thinking}, prompt {prompt.version}, kb_context: {context.describe()}, "
        f"max_tokens {settings.chat_max_answer_tokens}, timeout {request_timeout:g}s",
        file=out,
    )
    samples: list[Sample] = []
    try:
        for i in range(1, count + 1):
            sample = await measure(provider, chat, settings.chat_max_answer_tokens, request_timeout)
            samples.append(sample)
            line = f"#{i}  ttft {_fmt(sample.ttft)}  total {_fmt(sample.total)}"
            if sample.tokens_out is not None:
                line += f"  tokens_out {sample.tokens_out}"
            if sample.finish_reason not in (None, "stop"):
                line += f"  finish {sample.finish_reason}"
            if sample.error:
                line += f"  ERROR {sample.error}"
            print(line, file=out)
    finally:
        if (aclose := getattr(provider, "aclose", None)) is not None:
            await aclose()
    for line in summary(samples, settings.llm_first_token_timeout_s):
        print(line, file=out)
    return 1 if any(s.error for s in samples) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.latency_probe", description=__doc__)
    parser.add_argument("--link", required=True, metavar="PROVIDER:MODEL")
    parser.add_argument("-n", "--count", type=int, default=5)
    parser.add_argument("--thinking", choices=THINKING, default="registry")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    parser.add_argument("--prompt", default=PROMPT_VERSION, help="prompt file, e.g. system_v6")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--no-kb", action="store_true", help="no <knowledge>, no embeddings API")
    args = parser.parse_args(argv)
    if args.link not in MODELS:
        parser.error(f"unknown model {args.link}: not in app/llm/registry.py MODELS")
    if args.count < 1:
        parser.error("-n must be at least 1")
    return asyncio.run(
        run(
            link=args.link,
            count=args.count,
            thinking=args.thinking,
            request_timeout=args.timeout,
            prompt_version=args.prompt,
            question=args.question,
            kb=not args.no_kb,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
