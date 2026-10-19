"""FakeLLMProvider with scripted scenarios and an in-memory llm_calls recorder (no real APIs)."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from app.db.repositories.llm_calls import LLMCallRecord
from app.llm.base import ChatMessage, ProviderError, StreamEnd


@dataclass(frozen=True)
class Scenario:
    """What one call to the fake does: stream `tokens`, then optionally raise `error`.

    error with no tokens = fail before the first token; with tokens = fail mid-stream.
    hang = never produce the first token (for the first-token timeout).
    end = the StreamEnd sent after the tokens (finish_reason, usage) on success.
    stall = after the tokens, wait forever (a client can disconnect mid-answer).
    """

    tokens: tuple[str, ...] = ()
    error: ProviderError | None = None
    hang: bool = False
    end: StreamEnd | None = None
    stall: bool = False


def stall_after(*tokens: str) -> Scenario:
    return Scenario(tokens=tokens or ("partial",), stall=True)


def success(*tokens: str, end: StreamEnd | None = None) -> Scenario:
    return Scenario(tokens=tokens or ("ok",), end=end)


def truncated(*tokens: str) -> Scenario:
    """Answer cut off by max_tokens (finish_reason == "length")."""
    return success(*tokens, end=StreamEnd(finish_reason="length", tokens_in=10, tokens_out=20))


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
    # The messages of every call, to check the context sent to the model.
    received: list[list[ChatMessage]] = field(default_factory=list)

    def script(self, *scenarios: Scenario) -> "FakeLLMProvider":
        self.scenarios.extend(scenarios)
        return self

    async def stream(
        self, messages: Sequence[ChatMessage], max_tokens: int
    ) -> AsyncIterator[str | StreamEnd]:
        self.calls += 1
        self.received.append(list(messages))
        scenario = self.scenarios.pop(0) if self.scenarios else success()
        try:
            if scenario.hang:
                await asyncio.Event().wait()
            for token in scenario.tokens:
                await asyncio.sleep(0)
                yield token
            if scenario.stall:
                await asyncio.Event().wait()
            if scenario.error is not None:
                raise scenario.error
            if scenario.end is not None:
                yield scenario.end
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
