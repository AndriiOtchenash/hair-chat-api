"""OpenAICompatibleProvider against a fake HTTP server (httpx2.MockTransport, the openai SDK's
transport layer): real SDK request building and SSE parsing, no network."""

import json
from collections.abc import AsyncIterator, Callable

import httpx2
import pytest

from app.llm.base import (
    ChatMessage,
    ProviderConnectionError,
    ProviderError,
    ProviderHTTPError,
    ProviderTimeout,
    StreamBroken,
)
from app.llm.openai_compatible import OpenAICompatibleProvider

BASE_URL = "https://llm.test/v1"
MESSAGES: list[ChatMessage] = [
    {"role": "system", "content": "be nice"},
    {"role": "user", "content": "hi"},
]

Handler = Callable[[httpx2.Request], httpx2.Response]


def sse_chunk(content: str | None, **delta_extra: str) -> bytes:
    delta: dict[str, str | None] = {"content": content, **delta_extra}
    body = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(body)}\n\n".encode()


DONE = b"data: [DONE]\n\n"


def sse_response(*chunks: bytes) -> httpx2.Response:
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content=b"".join(chunks)
    )


class BreaksAfterFirstChunk(httpx2.AsyncByteStream):
    def __init__(self, first: bytes, error: Exception) -> None:
        self._first = first
        self._error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first
        raise self._error

    async def aclose(self) -> None:
        pass


def provider(handler: Handler) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="groq",
        model="openai/gpt-oss-120b",
        max_context_tokens=131_072,
        base_url=BASE_URL,
        api_key="test-key",
        connect_timeout=1,
        read_timeout=1,
        reasoning_effort="low",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


async def collect(p: OpenAICompatibleProvider) -> list[str]:
    return [text async for text in p.stream(MESSAGES, max_tokens=50)]


async def test_streams_content_deltas_and_sends_openai_request() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return sse_response(
            sse_chunk(None, reasoning="thinking..."),  # reasoning is not part of the answer
            sse_chunk("Hel"),
            sse_chunk("lo"),
            sse_chunk(""),
            DONE,
        )

    assert await collect(provider(handler)) == ["Hel", "lo"]

    [request] = seen
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["stream"] is True
    assert body["max_tokens"] == 50
    assert body["reasoning_effort"] == "low"
    assert body["messages"] == MESSAGES


async def test_no_reasoning_effort_is_not_sent() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return sse_response(sse_chunk("x"), DONE)

    plain = OpenAICompatibleProvider(
        name="gemini",
        model="m",
        max_context_tokens=1,
        base_url=BASE_URL,
        api_key="k",
        connect_timeout=1,
        read_timeout=1,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    await collect(plain)

    assert "reasoning_effort" not in bodies[0]


@pytest.mark.parametrize(
    ("status", "code", "retryable", "misconfigured"),
    [
        (400, "BadRequestError:400", False, False),
        (401, "AuthenticationError:401", True, True),
        (403, "PermissionDeniedError:403", True, True),
        (404, "NotFoundError:404", True, True),
        (408, "APIStatusError:408", True, False),
        (429, "RateLimitError:429", True, False),
        (500, "InternalServerError:500", True, False),
        (503, "InternalServerError:503", True, False),
    ],
)
async def test_http_errors_map_to_provider_errors(
    status: int, code: str, retryable: bool, misconfigured: bool
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": {"message": "user text echoed back"}})

    with pytest.raises(ProviderHTTPError) as exc_info:
        await collect(provider(handler))

    err = exc_info.value
    assert (err.status, err.code, err.retryable, err.misconfigured) == (
        status,
        code,
        retryable,
        misconfigured,
    )
    assert "echoed" not in err.code


async def test_sdk_does_not_retry_on_its_own() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(429, json={"error": {"message": "slow down"}})

    with pytest.raises(ProviderHTTPError):
        await collect(provider(handler))

    assert calls == 1


async def test_connect_timeout() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectTimeout("too slow", request=request)

    with pytest.raises(ProviderTimeout):
        await collect(provider(handler))


async def test_connection_refused() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(ProviderConnectionError):
        await collect(provider(handler))


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (httpx2.ReadError("connection reset"), "stream_broken"),
        (httpx2.ReadTimeout("stalled"), "stream_timeout"),
    ],
)
async def test_failure_mid_stream_is_stream_broken(error: Exception, code: str) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=BreaksAfterFirstChunk(sse_chunk("partial"), error),
        )

    received: list[str] = []
    with pytest.raises(StreamBroken) as exc_info:
        async for text in provider(handler).stream(MESSAGES, max_tokens=50):
            received.append(text)

    assert received == ["partial"]
    assert exc_info.value.code == code


async def test_error_object_inside_stream() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        error = json.dumps({"error": {"message": "overloaded", "code": 503}})
        return sse_response(f"data: {error}\n\n".encode())

    with pytest.raises(ProviderError) as exc_info:
        await collect(provider(handler))

    assert exc_info.value.code == "stream_error"
    assert exc_info.value.retryable


async def test_gemini_invalid_key_400_is_misconfiguration_not_bad_request() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        # Real Gemini body for a bad key (2026-10-19).
        body = [
            {
                "error": {
                    "code": 400,
                    "message": "Please pass a valid API key",
                    "status": "INVALID_ARGUMENT",
                }
            }
        ]
        return httpx2.Response(400, json=body)

    with pytest.raises(ProviderHTTPError) as exc_info:
        await collect(provider(handler))

    err = exc_info.value
    assert err.code == "BadRequestError:400:invalid_api_key"
    assert err.retryable and err.misconfigured
