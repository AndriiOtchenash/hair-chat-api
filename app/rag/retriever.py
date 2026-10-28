"""Question -> knowledge-base passages for the <knowledge> block (KnowledgeSource of 1.9).

Two steps, so the chat can embed the question while moderation runs (ChatService.reply):
`QueryEmbedder.embed_query` (network, never raises) and `RagKnowledge.knowledge` (DB search with
that vector, threshold, packing into the token budget).
"""

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.db.repositories.kb import KBHit, KBRepository
from app.rag.embeddings import Embedder, EmbeddingUnavailable
from app.services.tokens import estimate_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueryEmbedding:
    vector: Sequence[float]
    model: str  # searched against kb_chunks.embedding_model (rule 4)


@dataclass(frozen=True, slots=True)
class Source:
    """An article that reached the context; sent to the client with `done`."""

    title: str
    slug: str
    url: str | None


@dataclass(frozen=True, slots=True)
class Knowledge:
    text: str
    # The article of each passage: passages[0] is [1] in the text (app.services.kb_used).
    passages: tuple[Source, ...]

    @property
    def sources(self) -> tuple[Source, ...]:
        """Every article in the text, once each, in passage order."""
        sources: dict[str, Source] = {}
        for source in self.passages:
            sources.setdefault(source.slug, source)
        return tuple(sources.values())


class QueryEmbedder(Protocol):
    async def embed_query(self, text: str) -> QueryEmbedding | None:
        """Never raises: None means no knowledge base for this reply."""
        ...


class RagQueryEmbedder:
    def __init__(self, embedder: Embedder, timeout: float | None = None) -> None:
        self._embedder = embedder
        # Whole call, on top of the HTTP client's per-phase timeouts: the answer waits for it.
        self._timeout = timeout

    async def embed_query(self, text: str) -> QueryEmbedding | None:
        """None when embeddings are down: the answer goes without the knowledge base (no
        fallback model, its vectors would not match the index)."""
        try:
            async with asyncio.timeout(self._timeout):
                vectors = await self._embedder.embed([text])
        except EmbeddingUnavailable as exc:
            log = logger.error if exc.misconfigured else logger.warning
            log("embeddings unavailable, answering without knowledge: code=%s", exc.code)
            return None
        except TimeoutError:
            logger.warning("embeddings unavailable, answering without knowledge: code=timeout")
            return None
        return QueryEmbedding(vectors[0], self._embedder.model)


class NoQueryEmbedder:
    """No embedding key (dev): no knowledge base."""

    async def embed_query(self, text: str) -> QueryEmbedding | None:
        return None


class RagKnowledge:
    def __init__(
        self,
        repo: KBRepository,
        *,
        top_k: int,
        min_similarity: float,
        max_tokens: int,
        include_drafts: bool,
    ) -> None:
        self._repo = repo
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._max_tokens = max_tokens
        self._include_drafts = include_drafts

    async def knowledge(
        self, user_id: uuid.UUID, question: str, query: QueryEmbedding | None
    ) -> Knowledge | None:
        if query is None:
            return None
        hits = await self._repo.search(
            query.vector,
            model=query.model,
            include_drafts=self._include_drafts,
            limit=self._top_k,
        )
        relevant = [h for h in hits if h.similarity >= self._min_similarity]
        packed = self._pack(relevant)
        logger.info(
            "knowledge: hits=%d relevant=%d used=%d best=%.3f",
            len(hits),
            len(relevant),
            len(packed),
            hits[0].similarity if hits else 0.0,
        )
        if not packed:
            return None
        text = "\n\n".join(_passage(number, hit) for number, hit in enumerate(packed, start=1))
        return Knowledge(text, tuple(Source(h.title, h.slug, h.url) for h in packed))

    def _pack(self, hits: list[KBHit]) -> list[KBHit]:
        """Closest first, within max_tokens; a passage that does not fit is skipped."""
        packed: list[KBHit] = []
        used = 0
        for hit in hits:
            cost = estimate_tokens(_passage(len(packed) + 1, hit))
            if used + cost > self._max_tokens:
                continue
            packed.append(hit)
            used += cost
        return packed


def _passage(number: int, hit: KBHit) -> str:
    # The chunk already starts with "Title › Section". No slug: the model must not cite it; the
    # number is only for the <kb_used> tag, which the app cuts out of the answer.
    return f"[{number}] {hit.content}"
