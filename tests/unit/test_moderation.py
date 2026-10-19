"""OpenAIModerator against a fake HTTP server (httpx2.MockTransport): real SDK request building
and response parsing, no network."""

import json
from collections.abc import Callable

import httpx2
import pytest

from app.llm.moderation import ModerationUnavailable, OpenAIModerator

BASE_URL = "https://moderation.test/v1"
CATEGORIES = (
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/instructions",
    "self-harm/intent",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
)

Handler = Callable[[httpx2.Request], httpx2.Response]


def moderation_response(*flagged: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "modr-1",
            "model": "omni-moderation-2024-09-26",
            "results": [
                {
                    "flagged": bool(flagged),
                    "categories": {c: c in flagged for c in CATEGORIES},
                    "category_scores": {c: 0.9 if c in flagged else 0.01 for c in CATEGORIES},
                    "category_applied_input_types": {c: ["text"] for c in CATEGORIES},
                }
            ],
        },
    )


def moderator(handler: Handler) -> OpenAIModerator:
    return OpenAIModerator(
        api_key="test-key",
        timeout=3,
        base_url=BASE_URL,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


async def test_clean_text_is_not_flagged_and_request_uses_omni_model() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return moderation_response()

    result = await moderator(handler).check("How often should I wash oily hair?")

    assert (result.provider, result.model, result.flagged, result.categories) == (
        "openai",
        "omni-moderation-2024-09-26",
        False,
        (),
    )
    assert requests[0].url == f"{BASE_URL}/moderations"
    assert json.loads(requests[0].content) == {
        "model": "omni-moderation-latest",
        "input": "How often should I wash oily hair?",
    }


async def test_flagged_categories_come_back_by_api_name() -> None:
    result = await moderator(lambda _: moderation_response("self-harm", "self-harm/intent")).check(
        "x"
    )

    assert result.flagged is True
    assert result.categories == ("self-harm", "self-harm/intent")
    assert result.scores["self-harm/intent"] == 0.9
    assert result.scores["hate"] == 0.01


@pytest.mark.parametrize(
    ("status", "code", "misconfigured"),
    [
        (429, "RateLimitError:429", False),
        (500, "InternalServerError:500", False),
        (503, "InternalServerError:503", False),
        (401, "AuthenticationError:401", True),
        (400, "BadRequestError:400", True),
    ],
)
async def test_http_errors_make_moderation_unavailable(
    status: int, code: str, misconfigured: bool
) -> None:
    with pytest.raises(ModerationUnavailable) as caught:
        await moderator(lambda _: httpx2.Response(status, json={"error": {}})).check("x")

    assert (caught.value.code, caught.value.misconfigured) == (code, misconfigured)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (httpx2.ReadTimeout("slow"), "timeout"),
        (httpx2.ConnectError("refused"), "connection_error"),
    ],
)
async def test_timeout_and_network_errors_make_moderation_unavailable(
    error: Exception, code: str
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise error

    with pytest.raises(ModerationUnavailable) as caught:
        await moderator(handler).check("x")

    assert (caught.value.code, caught.value.misconfigured) == (code, False)


async def test_sdk_does_not_retry_on_its_own() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(503, json={"error": {}})

    with pytest.raises(ModerationUnavailable):
        await moderator(handler).check("x")

    assert calls == 1
