"""Conversation search: the service over an in-memory repository, and the endpoint.

Threshold, one-per-conversation and user isolation of the SQL itself: tests/integration.
"""

import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.chat import get_conversation_search
from app.core.consents import require_consents
from app.core.security import get_token_verifier, get_user_repository
from app.db.repositories.conversation_search import SearchRow, like_pattern
from app.main import create_app
from app.rag.embeddings import EmbeddingUnavailable
from app.services.conversation_search import (
    ConversationSearch,
    InvalidSearch,
    SearchMode,
    snippet,
)
from tests.auth_fakes import FakeAuth0, SigningKey, bearer, mint
from tests.fakes import FakeUserRepository
from tests.llm_fakes import FakeEmbedder

USER = uuid.UUID("01890000-0000-7000-8000-000000000001")
T0 = datetime(2026, 11, 1, tzinfo=UTC)


def row(question: str, distance: float | None, minutes: int = 0) -> SearchRow:
    return SearchRow(
        uuid.uuid4(), "Title", uuid.uuid4(), question, T0 + timedelta(minutes=minutes), distance
    )


@dataclass
class FakeSearchRepository:
    """Returns the scripted rows and records what it was asked."""

    semantic: list[SearchRow] = field(default_factory=list)
    text: list[SearchRow] = field(default_factory=list)
    nearest_calls: list[dict[str, object]] = field(default_factory=list)
    containing_calls: list[dict[str, object]] = field(default_factory=list)

    async def nearest(
        self,
        user_id: uuid.UUID,
        vector: Sequence[float],
        *,
        model: str,
        max_distance: float,
        limit: int,
    ) -> list[SearchRow]:
        self.nearest_calls.append(
            {"user_id": user_id, "model": model, "max_distance": max_distance, "limit": limit}
        )
        return self.semantic[:limit]

    async def containing(self, user_id: uuid.UUID, text: str, *, limit: int) -> list[SearchRow]:
        self.containing_calls.append({"user_id": user_id, "text": text, "limit": limit})
        return self.text[:limit]


def service(
    repo: FakeSearchRepository, embedder: FakeEmbedder | None = None, timeout: float = 1.0
) -> ConversationSearch:
    return ConversationSearch(repo, embedder or FakeEmbedder(), max_distance=0.4, timeout=timeout)


async def test_semantic_search_uses_the_threshold_model_and_user_and_scores_by_similarity() -> None:
    repo = FakeSearchRepository(semantic=[row("Как часто мыть?", 0.25), row("А маски?", 0.31)])
    embedder = FakeEmbedder("emb-model")

    result = await service(repo, embedder).search(USER, "  мытьё головы  ", 5)

    assert embedder.embedded == ["мытьё головы"]  # stripped
    assert repo.nearest_calls == [
        {"user_id": USER, "model": "emb-model", "max_distance": 0.4, "limit": 5}
    ]
    assert result.mode is SearchMode.SEMANTIC
    assert [(h.snippet, h.score, h.mode) for h in result.hits] == [
        ("Как часто мыть?", 0.75, SearchMode.SEMANTIC),
        ("А маски?", 0.69, SearchMode.SEMANTIC),
    ]
    assert repo.containing_calls == []


async def test_nothing_within_the_threshold_is_an_empty_semantic_result() -> None:
    result = await service(FakeSearchRepository()).search(USER, "борщ")

    assert (result.mode, result.hits) == (SearchMode.SEMANTIC, ())


@pytest.mark.parametrize(
    "embedder",
    [
        FakeEmbedder(error=EmbeddingUnavailable("APIConnectionError")),
        FakeEmbedder(hang=True),
        None,
    ],
    ids=["embeddings-down", "embeddings-too-slow", "no-embedder"],
)
async def test_without_embeddings_the_search_falls_back_to_text(
    embedder: FakeEmbedder | None,
) -> None:
    repo = FakeSearchRepository(text=[row("Про перхоть и шампунь", None)])
    search = ConversationSearch(repo, embedder, max_distance=0.4, timeout=0.05)

    result = await search.search(USER, " перхоть ", 10)

    assert repo.containing_calls == [{"user_id": USER, "text": "перхоть", "limit": 10}]
    assert repo.nearest_calls == []
    assert result.mode is SearchMode.TEXT
    [hit] = result.hits
    assert (hit.snippet, hit.score, hit.mode) == ("Про перхоть и шампунь", None, SearchMode.TEXT)


@pytest.mark.parametrize(
    ("query", "limit", "field"),
    [("a", 10, "query.q"), ("   a   ", 10, "query.q"), ("a" * 501, 10, "query.q")]
    + [("ok query", 0, "query.limit"), ("ok query", 21, "query.limit")],
)
async def test_invalid_query_or_limit_is_rejected_before_any_call(
    query: str, limit: int, field: str
) -> None:
    repo, embedder = FakeSearchRepository(), FakeEmbedder()

    with pytest.raises(InvalidSearch, match=field):
        await service(repo, embedder).search(USER, query, limit)

    assert embedder.calls == 0 and repo.nearest_calls == repo.containing_calls == []


async def test_the_log_has_mode_count_and_time_but_never_the_query(
    caplog: pytest.LogCaptureFixture,
) -> None:
    repo = FakeSearchRepository(text=[row("Секретный вопрос", None)])
    search = service(repo, FakeEmbedder(error=EmbeddingUnavailable("timeout")))

    with caplog.at_level(logging.INFO, logger="app"):
        await search.search(USER, "Секретный")

    assert "mode=text results=1 ms=" in caplog.text
    assert "code=timeout" in caplog.text
    assert "Секрет" not in caplog.text


def test_snippet_is_one_line_and_at_most_200_chars() -> None:
    assert snippet("  Как\n часто   мыть? ") == "Как часто мыть?"
    long = snippet("слово " * 100)
    assert long.endswith("…") and len(long) <= 201


def test_text_snippet_moves_to_a_match_beyond_the_first_200_chars() -> None:
    text = "а" * 300 + " ПЕРХОТЬ " + "б" * 300

    shown = snippet(text, "перхоть")

    assert "ПЕРХОТЬ" in shown
    assert shown.startswith("…") and shown.endswith("…")
    assert snippet("перхоть в начале " + "б" * 300, "перхоть").startswith("перхоть")


def test_like_pattern_matches_wildcards_literally() -> None:
    assert like_pattern("100%_a\\b") == "%100\\%\\_a\\\\b%"


# --- GET /conversations/search -------------------------------------------------------------

KEY = SigningKey("key-1")
ALICE = bearer(mint(KEY, sub="auth0|alice"))


@pytest.fixture
async def api() -> AsyncIterator[tuple[AsyncClient, FakeSearchRepository]]:
    app = create_app()
    verifier = FakeAuth0(KEY).verifier()
    users, repo = FakeUserRepository(), FakeSearchRepository()
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    app.dependency_overrides[get_user_repository] = lambda: users
    app.dependency_overrides[require_consents] = lambda: None  # covered by test_route_protection
    app.dependency_overrides[get_conversation_search] = lambda: service(repo)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, repo
    await verifier.aclose()


async def test_search_endpoint_returns_hits_with_mode(
    api: tuple[AsyncClient, FakeSearchRepository],
) -> None:
    client, repo = api
    hit = row("Как часто мыть?", 0.2)
    repo.semantic = [hit]

    response = await client.get("/conversations/search", params={"q": "мыть"}, headers=ALICE)

    assert response.status_code == 200
    assert response.json() == {
        "mode": "semantic",
        "items": [
            {
                "conversation_id": str(hit.conversation_id),
                "title": "Title",
                "message_id": str(hit.message_id),
                "snippet": "Как часто мыть?",
                "created_at": "2026-11-01T00:00:00Z",
                "score": 0.8,
                "mode": "semantic",
            }
        ],
    }
    assert repo.nearest_calls[0]["limit"] == 10  # default


@pytest.mark.parametrize(
    "params",
    [{}, {"q": "a"}, {"q": "a" * 501}, {"q": "  a  "}, {"q": "мыть", "limit": 21}]
    + [{"q": "мыть", "limit": 0}],
    ids=["missing", "short", "long", "short-after-strip", "limit-high", "limit-zero"],
)
async def test_search_endpoint_rejects_bad_parameters_without_echoing_them(
    api: tuple[AsyncClient, FakeSearchRepository], params: dict[str, object]
) -> None:
    client, _ = api

    response = await client.get("/conversations/search", params=params, headers=ALICE)

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert "aaa" not in body["message"]
