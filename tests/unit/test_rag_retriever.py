import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from app.db.repositories.kb import KBHit
from app.rag.embeddings import EmbeddingUnavailable
from app.rag.retriever import (
    SUMMARY_QUERY_MAX_TOKENS,
    QueryEmbedding,
    RagKnowledge,
    RagQueryEmbedder,
    Source,
    dialog_query,
    merge_hits,
)
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


async def test_misconfigured_embeddings_are_logged_as_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = EmbeddingUnavailable("BadRequestError:400:location_unsupported", misconfigured=True)

    with caplog.at_level(logging.WARNING):
        query = await RagQueryEmbedder(FakeEmbedder(error=error)).embed_query("вопрос")

    assert query is None
    [record] = caplog.records
    assert record.levelno == logging.ERROR
    assert "location_unsupported" in record.getMessage()


async def test_passages_are_numbered_for_the_kb_used_tag_one_source_each() -> None:
    kb = StaticKB(
        hit("dandruff", 0.9, title="Перхоть"),
        hit("choosing-shampoo", 0.8, title="Как выбрать шампунь"),
        hit("dandruff", 0.7, "dandruff › other\ntext", title="Перхоть"),
    )

    result = await knowledge(kb).knowledge(USER, "q", QUERY)

    assert result is not None
    assert result.text.startswith("[1] dandruff › section\n")
    assert "\n\n[2] choosing-shampoo › section\n" in result.text
    assert "\n\n[3] dandruff › other\n" in result.text
    assert [s.slug for s in result.passages] == ["dandruff", "choosing-shampoo", "dandruff"]


# --- dialog search: the latest message merged with the dialog query ---


class TwoQueryKB:
    """KBRepository answering the latest message's vector and the dialog vector differently."""

    def __init__(self, latest: list[KBHit], dialog: list[KBHit]) -> None:
        self.by_vector = {(1.0, 0.0): latest, (0.0, 1.0): dialog}
        self.searches = 0

    async def search(
        self, vector: Sequence[float], *, model: str, include_drafts: bool, limit: int
    ) -> list[KBHit]:
        self.searches += 1
        return self.by_vector[tuple(vector)][:limit]


DIALOG_QUERY = QueryEmbedding([1.0, 0.0], "fake-embedding", context_vector=[0.0, 1.0])


def test_dialog_query_is_the_summary_and_the_latest_message() -> None:
    query = dialog_query("А феном?", summary="Кудрявые волосы.", earlier_user=["Спасибо"])

    assert query == "Кудрявые волосы.\nА феном?"


def test_dialog_query_keeps_the_tail_of_a_long_summary() -> None:
    summary = " ".join(f"слово{i}" for i in range(2000))

    query = dialog_query("А феном?", summary=summary, earlier_user=())

    assert query is not None and query.endswith("слово1999\nА феном?")
    head = query.removesuffix("\nА феном?")
    assert estimate_tokens(head) <= SUMMARY_QUERY_MAX_TOKENS
    assert head.startswith("слово")  # cut at a word start
    assert "слово0 " not in head


def test_dialog_query_without_a_summary_is_the_last_three_user_messages() -> None:
    earlier = ["Привет", "У меня перхоть", "  ", "А если кожа сухая?"]

    query = dialog_query("А шампунь?", summary=None, earlier_user=earlier)

    assert query == "У меня перхоть\nА если кожа сухая?\nА шампунь?"


def test_no_dialog_query_on_the_first_message() -> None:
    assert dialog_query("Привет", summary=None, earlier_user=()) is None
    assert dialog_query("Привет", summary="  ", earlier_user=()) is None


def test_merge_keeps_the_two_best_of_the_latest_then_the_dialog_without_duplicates() -> None:
    latest = [hit("bleach", 0.8), hit("burn", 0.75), hit("bleach-2", 0.7)]
    dialog = [hit("burn", 0.7), hit("dandruff", 0.69), hit("washing", 0.65), hit("shampoo", 0.6)]

    merged = merge_hits(latest, dialog, 4)

    assert [h.slug for h in merged] == ["bleach", "burn", "dandruff", "washing"]


def test_merge_fills_from_the_latest_when_the_dialog_has_few_hits() -> None:
    latest = [hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]

    assert [h.slug for h in merge_hits(latest, [hit("d", 0.7)], 4)] == ["a", "b", "d", "c"]
    assert [h.slug for h in merge_hits([], [hit("d", 0.7)], 4)] == ["d"]


async def test_a_dialog_vector_adds_a_second_search_merged_with_the_first() -> None:
    kb = TwoQueryKB(
        latest=[hit("choosing-shampoo", 0.8), hit("washing", 0.7), hit("scalp", 0.65)],
        dialog=[hit("dandruff", 0.85), hit("off-topic", 0.5)],
    )

    result = await knowledge(kb).knowledge(USER, "А шампунь?", DIALOG_QUERY)

    assert kb.searches == 2
    assert result is not None
    # The dialog's hits below the threshold are left out like the latest's.
    assert [s.slug for s in result.passages] == ["choosing-shampoo", "washing", "dandruff", "scalp"]


async def test_query_embedder_embeds_the_dialog_query_in_the_same_call() -> None:
    embedder = FakeEmbedder(vectors={"А шампунь?": [1.0, 0.0], "Перхоть\nА шампунь?": [0.0, 1.0]})

    query = await RagQueryEmbedder(embedder).embed_query("А шампунь?", "Перхоть\nА шампунь?")

    assert query == DIALOG_QUERY
    assert embedder.calls == 1
