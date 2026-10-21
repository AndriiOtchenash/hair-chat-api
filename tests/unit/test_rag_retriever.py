import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from app.db.repositories.kb import KBHit
from app.rag.retriever import QueryEmbedding, RagKnowledge, RagQueryEmbedder, Source
from app.services.tokens import estimate_tokens
from tests.llm_fakes import FakeEmbedder

USER = uuid.UUID("01890000-0000-7000-8000-000000000001")
QUERY = QueryEmbedding([1.0, 0.0], "fake-embedding")


@dataclass
class Search:
    model: str
    include_drafts: bool
    limit: int


class StaticKB:
    """KBRepository returning `hits` (closest first) and recording each search."""

    def __init__(self, *hits: KBHit) -> None:
        self.hits = list(hits)
        self.searches: list[Search] = []

    async def search(
        self, vector: Sequence[float], *, model: str, include_drafts: bool, limit: int
    ) -> list[KBHit]:
        self.searches.append(Search(model, include_drafts, limit))
        return self.hits[:limit]


def hit(slug: str, similarity: float, content: str = "", title: str | None = None) -> KBHit:
    return KBHit(
        content or f"{slug} › section\ntext", title or slug.title(), slug, None, similarity
    )


def knowledge(
    kb: StaticKB,
    *,
    min_similarity: float = 0.6,
    max_tokens: int = 1500,
    include_drafts: bool = False,
    top_k: int = 4,
) -> RagKnowledge:
    return RagKnowledge(
        kb,
        top_k=top_k,
        min_similarity=min_similarity,
        max_tokens=max_tokens,
        include_drafts=include_drafts,
    )


async def test_hits_below_the_threshold_are_left_out() -> None:
    kb = StaticKB(hit("dandruff", 0.81), hit("scalp-types", 0.6), hit("hair-myths", 0.59))

    result = await knowledge(kb).knowledge(USER, "перхоть?", QUERY)

    assert result is not None
    assert [s.slug for s in result.sources] == ["dandruff", "scalp-types"]
    assert "hair-myths" not in result.text


async def test_nothing_above_the_threshold_means_no_knowledge() -> None:
    kb = StaticKB(hit("dandruff", 0.4))

    assert await knowledge(kb).knowledge(USER, "погода?", QUERY) is None


async def test_no_query_embedding_means_no_search() -> None:
    kb = StaticKB(hit("dandruff", 0.9))

    assert await knowledge(kb).knowledge(USER, "перхоть?", None) is None
    assert kb.searches == []


async def test_search_uses_top_k_the_query_model_and_the_drafts_setting() -> None:
    kb = StaticKB()

    await knowledge(kb, include_drafts=False, top_k=4).knowledge(USER, "q", QUERY)
    await knowledge(kb, include_drafts=True, top_k=2).knowledge(USER, "q", QUERY)

    assert kb.searches == [
        Search("fake-embedding", include_drafts=False, limit=4),
        Search("fake-embedding", include_drafts=True, limit=2),
    ]


async def test_passages_are_packed_into_the_token_budget_closest_first() -> None:
    long = "x" * 1400  # ~460 tokens
    kb = StaticKB(
        hit("a", 0.9, long), hit("b", 0.8, long), hit("c", 0.7, long), hit("d", 0.65, "short")
    )

    result = await knowledge(kb, max_tokens=1000).knowledge(USER, "q", QUERY)

    assert result is not None
    assert [s.slug for s in result.sources] == ["a", "b", "d"]  # c does not fit, d still does
    assert estimate_tokens(result.text) <= 1000 + 10  # separators between passages


async def test_sources_are_unique_per_article_in_order_of_relevance() -> None:
    kb = StaticKB(
        hit("dandruff", 0.9, title="Перхоть"),
        hit("choosing-shampoo", 0.8, title="Как выбрать шампунь"),
        hit("dandruff", 0.7, title="Перхоть"),
    )

    result = await knowledge(kb).knowledge(USER, "q", QUERY)

    assert result is not None
    assert result.sources == (
        Source("Перхоть", "dandruff", None),
        Source("Как выбрать шампунь", "choosing-shampoo", None),
    )
    assert result.text.count("dandruff › section") == 2
    assert "[dandruff]" not in result.text  # no ids for the model to cite


async def test_query_embedder_returns_the_vector_and_model() -> None:
    embedder = FakeEmbedder(vectors={"перхоть?": [0.6, 0.8]})

    query = await RagQueryEmbedder(embedder).embed_query("перхоть?")

    assert query == QueryEmbedding([0.6, 0.8], "fake-embedding")


async def test_embeddings_down_means_none_and_a_warning_without_the_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        query = await RagQueryEmbedder(FakeEmbedder(fail=True)).embed_query("секретный вопрос")

    assert query is None
    assert "embeddings unavailable" in caplog.text
    assert "секретный вопрос" not in caplog.text
