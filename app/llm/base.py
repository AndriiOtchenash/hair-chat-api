"""LLM layer contracts: the provider protocol, provider errors and the stream events.

ResilientLLM depends only on what is defined here, never on the openai SDK, so fakes in tests
implement the same Protocol and raise the same errors as real providers.
"""

import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMProvider(Protocol):
    """One link of the chain: a single model at a single provider. Stateless (CLAUDE.md rule 2)."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def max_context_tokens(self) -> int: ...

    def stream(
        self, messages: Sequence[ChatMessage], max_tokens: int
    ) -> AsyncIterator["str | StreamEnd"]:
        """Yields text deltas, optionally a final StreamEnd; raises ProviderError on failure."""
        ...


def link_key(provider: str, model: str) -> str:
    """`provider:model`, the form used in LLM_CHAIN, --skip and logs."""
    return f"{provider}:{model}"


class ProviderError(Exception):
    """A provider call failed. `code` is safe to log and store: never request or response text.

    retryable: another provider may succeed (network, timeout, 408/429/5xx, 401/403, provider-side
    400s from `provider_side_400`).
    misconfigured: our side of this provider is wrong (key, permissions, region) -> logged as error.
    """

    def __init__(self, code: str, *, retryable: bool = True, misconfigured: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.misconfigured = misconfigured


class ProviderTimeout(ProviderError):
    def __init__(self, code: str = "timeout") -> None:
        super().__init__(code)


class ProviderConnectionError(ProviderError):
    def __init__(self, code: str = "connection_error") -> None:
        super().__init__(code)


class StreamBroken(ProviderError):
    """The connection dropped while the answer was streaming."""

    def __init__(self, code: str = "stream_broken") -> None:
        super().__init__(code)


# The request itself is bad (malformed, too large, invalid params): any provider would reject it.
NON_RETRYABLE_STATUSES = frozenset({400, 413, 422})
# This link is set up wrong (key, permissions, model name); the next link has its own setup.
MISCONFIGURED_STATUSES = frozenset({401, 403, 404})


def provider_side_400(status: int, body: object) -> str | None:
    """Reason when a 400 is the provider's setup, not our request (Gemini has no other status
    for these), so the next link may still answer:

    - invalid_api_key: "Please pass a valid API key" (INVALID_ARGUMENT);
    - location_unsupported: "User location is not supported for the API use."
      (FAILED_PRECONDITION), e.g. a Fly egress IP Google places outside supported regions.

    The body text is only matched here, never logged or stored.
    """
    if status != 400:
        return None
    text = str(body).lower()
    if "failed_precondition" in text or "location is not supported" in text:
        return "location_unsupported"
    if "api key" in text:
        return "invalid_api_key"
    return None


class ProviderHTTPError(ProviderError):
    def __init__(
        self, status: int, error_class: str = "HTTPError", *, reason: str | None = None
    ) -> None:
        # reason: a provider-side failure reported as 400 (`provider_side_400`): like 401/403.
        super().__init__(
            f"{error_class}:{status}" + (f":{reason}" if reason else ""),
            retryable=reason is not None or status not in NON_RETRYABLE_STATUSES,
            misconfigured=reason is not None or status in MISCONFIGURED_STATUSES,
        )
        self.status = status


class LLMRequestError(Exception):
    """Our request was rejected (400 and other non-retryable 4xx): no fallback, it is our bug."""

    def __init__(self, provider: str, model: str, code: str) -> None:
        super().__init__(f"{link_key(provider, model)}: {code}")
        self.provider = provider
        self.model = model
        self.code = code


class AllProvidersDown(Exception):
    """No link of the chain produced a full answer. `attempts` = [(provider:model, code)]."""

    def __init__(self, attempts: list[tuple[str, str]]) -> None:
        summary = ", ".join(f"{key}={code}" for key, code in attempts) or "no attempts"
        super().__init__(f"all LLM providers failed: {summary}")
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class StreamEnd:
    """Optional last item of a provider stream: why it stopped and the token usage, if known."""

    finish_reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class Restart:
    """The provider failed mid-stream; the client drops the partial answer (SSE `restart`)."""

    reason: str


@dataclass(frozen=True, slots=True)
class Done:
    provider: str
    model: str
    # finish_reason == "length": the answer hit max_tokens and is cut off.
    truncated: bool = False
    tokens_in: int | None = None
    tokens_out: int | None = None
    # llm_calls ids of every attempt of this generation, to link them to the saved message.
    call_ids: tuple[uuid.UUID, ...] = ()


LLMEvent = Token | Restart | Done
