"""One provider implementation for every OpenAI-compatible endpoint (Gemini, Groq, ...).

The openai SDK (v3) runs on httpx2 (maintained fork of httpx), so timeouts/clients are httpx2 types.
The provider only translates: our messages -> POST /chat/completions (stream=True), SSE chunks ->
text deltas, SDK/httpx failures -> ProviderError. Retries and fallback live in ResilientLLM.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import httpx2
import openai
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared import ReasoningEffort

from app.llm.base import (
    ChatMessage,
    ProviderConnectionError,
    ProviderError,
    ProviderHTTPError,
    ProviderTimeout,
    StreamBroken,
    StreamEnd,
)


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        name: str,
        model: str,
        max_context_tokens: int,
        base_url: str,
        api_key: str,
        connect_timeout: float,
        read_timeout: float,
        reasoning_effort: ReasoningEffort = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._max_context_tokens = max_context_tokens
        self._reasoning_effort = reasoning_effort
        # max_retries=0: the SDK would otherwise retry 429/5xx on its own and hide the failure
        # from ResilientLLM's fallback and cooldown. read_timeout also bounds stalls mid-stream.
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=httpx2.Timeout(read_timeout, connect=connect_timeout),
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    async def aclose(self) -> None:
        await self._client.close()

    async def stream(
        self, messages: Sequence[ChatMessage], max_tokens: int
    ) -> AsyncIterator[str | StreamEnd]:
        extra: dict[str, Any] = {}
        if self._reasoning_effort is not None:
            extra["reasoning_effort"] = self._reasoning_effort
        started = False
        finish_reason: str | None = None
        usage: CompletionUsage | None = None
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=cast(list[ChatCompletionMessageParam], list(messages)),
                max_tokens=max_tokens,
                stream=True,
                # Usage arrives in the stream (Gemini repeats a running total on every chunk,
                # Groq sends it once at the end): the last one seen wins.
                stream_options={"include_usage": True},
                **extra,
            )
            try:
                async for chunk in response:
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason
                    # Reasoning models may send thinking in other delta fields; only content counts.
                    if text := choice.delta.content:
                        started = True
                        yield text
            finally:
                await response.close()
            yield StreamEnd(
                finish_reason=finish_reason,
                tokens_in=usage.prompt_tokens if usage else None,
                tokens_out=usage.completion_tokens if usage else None,
            )
        except openai.APIStatusError as exc:
            raise ProviderHTTPError(
                exc.status_code, type(exc).__name__, invalid_api_key=_is_invalid_api_key(exc)
            ) from exc
        except openai.APITimeoutError as exc:
            raise (StreamBroken("stream_timeout") if started else ProviderTimeout()) from exc
        except openai.APIConnectionError as exc:
            raise (StreamBroken() if started else ProviderConnectionError()) from exc
        except openai.APIError as exc:
            # An `error` object inside the SSE stream (provider-side failure after HTTP 200).
            raise ProviderError("stream_error") from exc


def _is_invalid_api_key(exc: openai.APIStatusError) -> bool:
    """Gemini's OpenAI endpoint answers a bad key with 400 INVALID_ARGUMENT, no machine-readable
    reason, only the message "Please pass a valid API key". Without this check a broken Gemini
    key would count as our bad request and block the fallback. The text is only matched here,
    never logged or stored.
    """
    return exc.status_code == 400 and "api key" in str(exc.body).lower()
