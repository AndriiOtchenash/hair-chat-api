"""OpenAICompatibleEmbedder against a fake HTTP server (httpx2.MockTransport): real SDK request
building and response parsing, no network."""

import json
from collections.abc import Callable

import httpx2
import pytest

from app.rag.embeddings import MAX_BATCH, EmbeddingUnavailable, OpenAICompatibleEmbedder

BASE_URL = "https://embeddings.test/v1"
DIM = 4

Handler = Callable[[httpx2.Request], httpx2.Response]


def embeddings_response(
    count: int, *, dim: int = DIM, with_index: bool = False, start: float = 0.0
) -> httpx2.Response:
    # Gemini's OpenAI endpoint sends "index": null; OpenAI sends the position.
    data = [
        {"object": "embedding", "index": i if with_index else None, "embedding": [start + i] * dim}
        for i in range(count)
    ]
    return httpx2.Response(
        200, json={"object": "list", "model": "gemini-embedding-2", "data": data}
    )


def embedder(handler: Handler) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(
        model="gemini-embedding-2",
        dimensions=DIM,
        api_key="test-key",
        timeout=3,
        base_url=BASE_URL,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


async def test_request_asks_for_the_model_dimensions_and_floats() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return embeddings_response(2)

    vectors = await embedder(handler).embed(["перхоть", "dandruff"])

    assert vectors == [[0.0] * DIM, [1.0] * DIM]
    assert requests == [
        {
            "model": "gemini-embedding-2",
            "input": ["перхоть", "dandruff"],
            "dimensions": DIM,
            "encoding_format": "float",
        }
    ]


async def test_rows_with_index_are_put_in_input_order() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        response = embeddings_response(2, with_index=True)
        body = response.json()
        body["data"].reverse()
        return httpx2.Response(200, json=body)

    assert await embedder(handler).embed(["a", "b"]) == [[0.0] * DIM, [1.0] * DIM]


async def test_more_texts_than_a_batch_are_sent_in_several_requests() -> None:
    sizes: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        count = len(json.loads(request.content)["input"])
        sizes.append(count)
        return embeddings_response(count, start=float(sum(sizes) - count))

    vectors = await embedder(handler).embed([f"t{i}" for i in range(MAX_BATCH + 5)])

    assert sizes == [MAX_BATCH, 5]
    assert [v[0] for v in vectors] == [float(i) for i in range(MAX_BATCH + 5)]


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (httpx2.Response(429, json={"error": {"message": "slow down"}}), "RateLimitError:429"),
        (httpx2.Response(503, json={"error": {"message": "down"}}), "InternalServerError:503"),
        (httpx2.Response(400, json={"error": {"message": "bad key"}}), "BadRequestError:400"),
        (embeddings_response(1, dim=DIM + 1), "wrong_dimensions"),
        (embeddings_response(3), "wrong_count"),
    ],
)
async def test_failures_become_embedding_unavailable(response: httpx2.Response, code: str) -> None:
    with pytest.raises(EmbeddingUnavailable) as exc:
        await embedder(lambda request: response).embed(
            ["a", "b"] if code == "wrong_count" else ["a"]
        )

    assert exc.value.code == code


@pytest.mark.parametrize(
    ("error", "code"),
    [(httpx2.ReadTimeout("slow"), "timeout"), (httpx2.ConnectError("refused"), "connection_error")],
)
async def test_network_failures_become_embedding_unavailable(error: Exception, code: str) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise error

    with pytest.raises(EmbeddingUnavailable) as exc:
        await embedder(handler).embed(["a"])

    assert exc.value.code == code
