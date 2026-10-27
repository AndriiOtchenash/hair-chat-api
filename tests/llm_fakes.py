"""FakeLLMProvider, FakeModerator and FakeEmbedder with scripted scenarios, an in-memory llm_calls
recorder.

No real APIs (CLAUDE.md rule 1).
"""

import asyncio
import hashlib
import math
import random
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from app.db.repositories.llm_calls import LLMCallRecord
from app.llm.base import ChatMessage, ProviderError, StreamEnd
from app.llm.moderation import ModerationResult, ModerationUnavailable
from app.rag.embeddings import EmbeddingUnavailable


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


class FakeModerator:
    """Moderator with scripted verdicts: flag(...) / unavailable(...) apply to the next calls,
    then every message is clean. Records the texts it was asked about."""

    enabled = True

    def __init__(self, *verdicts: ModerationResult | ModerationUnavailable) -> None:
        self.verdicts = list(verdicts)
        self.checked: list[str] = []

    async def check(self, text: str) -> ModerationResult:
        self.checked.append(text)
        verdict = self.verdicts.pop(0) if self.verdicts else clean()
        if isinstance(verdict, ModerationUnavailable):
            raise verdict
        return verdict

    async def aclose(self) -> None:
        return None


def clean() -> ModerationResult:
    return ModerationResult("openai", "omni-moderation-test", flagged=False)


def flag(*categories: str, scores: dict[str, float] | None = None) -> ModerationResult:
    """Flagged `categories`; each scores 1.0 unless `scores` says otherwise."""
    return ModerationResult(
        "openai",
        "omni-moderation-test",
        flagged=True,
        categories=categories,
        scores={c: 1.0 for c in categories} | (scores or {}),
    )


EMBEDDING_DIM = 768


def unit_vector(*weights: float, dim: int = EMBEDDING_DIM) -> list[float]:
    """`weights` on the first axes, zeros elsewhere, normalized: crafted cosine similarities."""
    raw = [*weights, *([0.0] * (dim - len(weights)))]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def text_vector(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic pseudo-random unit vector of a text: same text, same vector."""
    rng = random.Random(hashlib.sha256(text.encode()).digest())
    return unit_vector(*(rng.uniform(-1, 1) for _ in range(dim)), dim=dim)


class FakeEmbedder:
    """Embedder with deterministic vectors: `vectors[text]` if given, else text_vector(text).

    fail: raise EmbeddingUnavailable on every call (`error` if given); hang: never answer;
    wait_for: answer only after this event is set (to prove calls run in parallel).
    """

    def __init__(
        self,
        model: str = "fake-embedding",
        vectors: dict[str, list[float]] | None = None,
        *,
        fail: bool = False,
        error: EmbeddingUnavailable | None = None,
        hang: bool = False,
        wait_for: asyncio.Event | None = None,
    ) -> None:
        self.model = model
        self.vectors = vectors or {}
        self.fail = fail or error is not None
        self.error = error
        self.hang = hang
        self.wait_for = wait_for
        self.calls = 0
        self.embedded: list[str] = []
        self.cancelled = False
        self.started = asyncio.Event()  # set on the first call

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        self.started.set()
        try:
            if self.hang:
                await asyncio.Event().wait()
            if self.wait_for is not None:
                await self.wait_for.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.fail:
            raise self.error or EmbeddingUnavailable("timeout")
        self.embedded.extend(texts)
        return [self.vectors.get(t) or text_vector(t) for t in texts]

    async def aclose(self) -> None:
        return None
