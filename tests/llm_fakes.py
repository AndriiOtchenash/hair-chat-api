"""FakeLLMProvider with scripted scenarios and an in-memory llm_calls recorder (no real APIs)."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from app.db.repositories.llm_calls import LLMCallRecord
from app.llm.base import ChatMessage, ProviderError


@dataclass(frozen=True)
class Scenario:
    """What one call to the fake does: stream `tokens`, then optionally raise `error`.

    error with no tokens = fail before the first token; with tokens = fail mid-stream.
    hang = never produce the first token (for the first-token timeout).
    """

    tokens: tuple[str, ...] = ()
    error: ProviderError | None = None
    hang: bool = False


def success(*tokens: str) -> Scenario:
    return Scenario(tokens=tokens or ("ok",))


def fail_before_first(error: ProviderError) -> Scenario:
    return Scenario(error=error)


def fail_mid_stream(error: ProviderError, *tokens: str) -> Scenario:
    return Scenario(tokens=tokens or ("partial",), error=error)


def hang() -> Scenario:
    return Scenario(hang=True)


@dataclass
class FakeLLMProvider:
    name: str
    model: str
    scenarios: list[Scenario] = field(default_factory=list)
    max_context_tokens: int = 100_000
    calls: int = 0
    closed_streams: int = 0

    def script(self, *scenarios: Scenario) -> "FakeLLMProvider":
        self.scenarios.extend(scenarios)
        return self

    async def stream(self, messages: Sequence[ChatMessage], max_tokens: int) -> AsyncIterator[str]:
        self.calls += 1
        scenario = self.scenarios.pop(0) if self.scenarios else success()
        try:
            if scenario.hang:
                await asyncio.Event().wait()
            for token in scenario.tokens:
                await asyncio.sleep(0)
                yield token
            if scenario.error is not None:
                raise scenario.error
        finally:
            self.closed_streams += 1


class FakeLLMCallRecorder:
    def __init__(self) -> None:
        self.calls: list[LLMCallRecord] = []

    async def record(self, call: LLMCallRecord) -> None:
        self.calls.append(call)


class FailingLLMCallRecorder:
    """A recorder that breaks contract and raises, as a buggy implementation might."""

    async def record(self, call: LLMCallRecord) -> None:
        raise ConnectionError("db is down")
