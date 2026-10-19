"""Fallback over the provider chain, emitting the SSE-shaped events Token / Restart / Done."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence

from app.db.models import LLMCallStatus
from app.db.repositories.llm_calls import LLMCallRecord, LLMCallRecorder
from app.llm.base import (
    AllProvidersDown,
    ChatMessage,
    Done,
    LLMEvent,
    LLMProvider,
    LLMRequestError,
    ProviderError,
    ProviderTimeout,
    Restart,
    Token,
    link_key,
)

logger = logging.getLogger(__name__)

COOLDOWN_AFTER_FAILURES = 3
COOLDOWN_SECONDS = 60.0

Clock = Callable[[], float]


class Cooldown:
    """After N consecutive failures a link is skipped for a fixed time, then gets a clean slate.

    Deliberately simpler than a circuit breaker: no failure-rate window, no half-open probes.
    State is per process (each worker learns on its own), which is fine for a fallback chain.
    """

    def __init__(
        self,
        clock: Clock,
        failures: int = COOLDOWN_AFTER_FAILURES,
        seconds: float = COOLDOWN_SECONDS,
    ) -> None:
        self._clock = clock
        self._threshold = failures
        self._seconds = seconds
        self._failures: dict[str, int] = {}
        self._until: dict[str, float] = {}

    def is_cooling(self, key: str) -> bool:
        until = self._until.get(key)
        if until is None:
            return False
        if self._clock() >= until:
            del self._until[key]
            return False
        return True

    def failure(self, key: str) -> None:
        count = self._failures.get(key, 0) + 1
        if count >= self._threshold:
            self._until[key] = self._clock() + self._seconds
            self._failures.pop(key, None)
            logger.warning("llm link in cooldown: link=%s seconds=%.0f", key, self._seconds)
        else:
            self._failures[key] = count

    def success(self, key: str) -> None:
        self._failures.pop(key, None)


class EmptyResponse(ProviderError):
    def __init__(self) -> None:
        super().__init__("empty_response")


class ResilientLLM:
    def __init__(
        self,
        providers: Sequence[LLMProvider],
        recorder: LLMCallRecorder,
        *,
        first_token_timeout: float,
        clock: Clock = time.monotonic,
    ) -> None:
        if not providers:
            raise ValueError("LLM chain is empty")
        self._providers = list(providers)
        self._recorder = recorder
        self._first_token_timeout = first_token_timeout
        self._clock = clock
        self._cooldown = Cooldown(clock)

    @property
    def providers(self) -> list[LLMProvider]:
        return list(self._providers)

    @property
    def max_context_tokens(self) -> int:
        """Context budget for the whole chain (CLAUDE.md rule 3): any link may end up answering."""
        return min(p.max_context_tokens for p in self._providers)

    async def aclose(self) -> None:
        for provider in self._providers:
            if (aclose := getattr(provider, "aclose", None)) is not None:
                await aclose()

    async def stream(
        self, messages: Sequence[ChatMessage], max_tokens: int
    ) -> AsyncIterator[LLMEvent]:
        """Full answer from the first link that manages it.

        Raises LLMRequestError (our request is bad, no fallback) or AllProvidersDown.
        If a link fails after streaming tokens, Restart is emitted right before the next link
        starts, so the client drops the partial answer and receives a complete one.
        """
        attempts: list[tuple[str, str]] = []
        fallback_from: str | None = None
        partial_reason: str | None = None  # set while a partial answer is out at the client

        for index, provider in enumerate(self._providers):
            key = link_key(provider.name, provider.model)
            if self._cooldown.is_cooling(key):
                logger.info("llm link skipped (cooldown): link=%s", key)
                attempts.append((key, "cooldown"))
                continue
            if partial_reason is not None:
                yield Restart(reason=partial_reason)
                partial_reason = None

            started = self._clock()
            tokens = provider.stream(messages, max_tokens)
            streamed = False
            try:
                try:
                    async with asyncio.timeout(self._first_token_timeout):
                        first = await anext(tokens)
                except TimeoutError:
                    raise ProviderTimeout("first_token_timeout") from None
                except StopAsyncIteration:
                    raise EmptyResponse() from None
                streamed = True
                yield Token(first, provider.name, provider.model)
                async for text in tokens:
                    yield Token(text, provider.name, provider.model)
            except ProviderError as err:
                latency_ms = self._elapsed_ms(started)
                if not err.retryable:
                    logger.error(
                        "llm request rejected: link=%s code=%s latency_ms=%d",
                        key,
                        err.code,
                        latency_ms,
                    )
                    await self._record(
                        provider, latency_ms, LLMCallStatus.ERROR, err, fallback_from
                    )
                    raise LLMRequestError(provider.name, provider.model, err.code) from err

                self._cooldown.failure(key)
                log = logger.error if err.misconfigured else logger.warning
                log(
                    "llm attempt failed: link=%s code=%s streamed=%s latency_ms=%d",
                    key,
                    err.code,
                    streamed,
                    latency_ms,
                )
                has_next = any(
                    not self._cooldown.is_cooling(link_key(p.name, p.model))
                    for p in self._providers[index + 1 :]
                )
                status = LLMCallStatus.FALLBACK if has_next else LLMCallStatus.ERROR
                await self._record(provider, latency_ms, status, err, fallback_from)
                attempts.append((key, err.code))
                fallback_from = key
                if streamed:
                    partial_reason = err.code
                continue
            finally:
                # Closes the HTTP stream on failure, early exit or client disconnect alike.
                if (aclose := getattr(tokens, "aclose", None)) is not None:
                    await aclose()

            latency_ms = self._elapsed_ms(started)
            self._cooldown.success(key)
            logger.info("llm answered: link=%s latency_ms=%d", key, latency_ms)
            await self._record(provider, latency_ms, LLMCallStatus.OK, None, fallback_from)
            yield Done(provider.name, provider.model)
            return

        raise AllProvidersDown(attempts)

    def _elapsed_ms(self, started: float) -> int:
        return int((self._clock() - started) * 1000)

    async def _record(
        self,
        provider: LLMProvider,
        latency_ms: int,
        status: LLMCallStatus,
        err: ProviderError | None,
        fallback_from: str | None,
    ) -> None:
        call = LLMCallRecord(
            provider=provider.name,
            model=provider.model,
            latency_ms=latency_ms,
            status=status,
            error=err.code if err else None,
            fallback_from=fallback_from,
        )
        try:
            await self._recorder.record(call)
        except Exception as exc:  # recorders are best effort; the answer matters more
            logger.warning("llm_calls record failed: error=%s", type(exc).__name__)
