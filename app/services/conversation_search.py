"""Search over the user's own conversations (GET /conversations/search).

Semantic first: the query is embedded (chat embedder: EMBEDDING_TIMEOUT_S, no retries) and
compared with message_embeddings (app.rag.conversation_index) of this user and of the current
EMBEDDING_MODEL, kept within CONVERSATION_SEARCH_MAX_DISTANCE. When embeddings are down, too
slow or not configured (dev), the search falls back to a case-insensitive substring match over
the user's own messages (mode "text"), with the same result shape and `score` null.
One result per conversation, its best match. Logs: mode, result count, ms; never the query.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from app.db.repositories.conversation_search import SearchRow
from app.rag.embeddings import Embedder, EmbeddingUnavailable

logger = logging.getLogger(__name__)

MIN_QUERY_CHARS = 2
MAX_QUERY_CHARS = 500
DEFAULT_LIMIT = 10
MAX_LIMIT = 20
SNIPPET_CHARS = 200
# Text mode: characters kept before the match when the snippet does not start at the question.
SNIPPET_LEAD_CHARS = 60
ELLIPSIS = "…"


class SearchMode(StrEnum):
    SEMANTIC = "semantic"
    TEXT = "text"  # fallback: embeddings unavailable


class InvalidSearch(ValueError):
    """`field: reason`, safe to return (never echoes the query)."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    conversation_id: uuid.UUID
    title: str | None
    message_id: uuid.UUID  # the user's question the snippet comes from
    snippet: str
    created_at: datetime
    score: float | None  # cosine similarity (1 - distance), higher is closer; None in text mode
    mode: SearchMode


@dataclass(frozen=True, slots=True)
class SearchResult:
    mode: SearchMode
    hits: tuple[SearchHit, ...]


class ConversationSearchRepository(Protocol):
    async def nearest(
        self,
        user_id: uuid.UUID,
        vector: Sequence[float],
        *,
        model: str,
        max_distance: float,
        limit: int,
    ) -> list[SearchRow]: ...

    async def containing(self, user_id: uuid.UUID, text: str, *, limit: int) -> list[SearchRow]: ...


def snippet(text: str, match: str | None = None) -> str:
    """The question on one line, at most SNIPPET_CHARS (plus ellipses). In text mode the
    window moves so that the match is in it."""
    text = " ".join(text.split())
    start = 0
    if match is not None:
        position = text.casefold().find(" ".join(match.split()).casefold())
        if position + len(match) > SNIPPET_CHARS:
            start = max(0, position - SNIPPET_LEAD_CHARS)
    window = text[start : start + SNIPPET_CHARS]
    if start + SNIPPET_CHARS < len(text):
        window = window.rstrip() + ELLIPSIS
    return (ELLIPSIS + window.lstrip()) if start else window


def validate(query: str, limit: int) -> str:
    """The stripped query; raises InvalidSearch."""
    query = query.strip()
    if not MIN_QUERY_CHARS <= len(query) <= MAX_QUERY_CHARS:
        raise InvalidSearch(f"query.q: must be {MIN_QUERY_CHARS}..{MAX_QUERY_CHARS} characters")
    if not 1 <= limit <= MAX_LIMIT:
        raise InvalidSearch(f"query.limit: must be 1..{MAX_LIMIT}")
    return query


class ConversationSearch:
    def __init__(
        self,
        repo: ConversationSearchRepository,
        embedder: Embedder | None,
        *,
        max_distance: float,
        timeout: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._repo = repo
        self._embedder = embedder  # None: no GEMINI_API_KEY (dev), text search only
        self._max_distance = max_distance
        self._timeout = timeout
        self._clock = clock

    async def search(
        self, user_id: uuid.UUID, query: str, limit: int = DEFAULT_LIMIT
    ) -> SearchResult:
        """Raises InvalidSearch; never fails because of embeddings."""
        query = validate(query, limit)
        t0 = self._clock()
        vector = await self._embed(query)
        if vector is not None and self._embedder is not None:
            rows = await self._repo.nearest(
                user_id,
                vector,
                model=self._embedder.model,
                max_distance=self._max_distance,
                limit=limit,
            )
            result = SearchResult(
                SearchMode.SEMANTIC,
                tuple(_hit(row, SearchMode.SEMANTIC, snippet(row.question)) for row in rows),
            )
        else:
            rows = await self._repo.containing(user_id, query, limit=limit)
            result = SearchResult(
                SearchMode.TEXT,
                tuple(_hit(row, SearchMode.TEXT, snippet(row.question, query)) for row in rows),
            )
        logger.info(
            "conversation search: mode=%s results=%d ms=%d",
            result.mode,
            len(result.hits),
            (self._clock() - t0) * 1000,
        )
        return result

    async def _embed(self, query: str) -> Sequence[float] | None:
        """The query vector, or None: the search falls back to text."""
        if self._embedder is None:
            return None
        try:
            async with asyncio.timeout(self._timeout):
                [vector] = await self._embedder.embed([query])
        except EmbeddingUnavailable as exc:
            log = logger.error if exc.misconfigured else logger.warning
            log("conversation search: embeddings unavailable, text search: code=%s", exc.code)
            return None
        except TimeoutError:
            logger.warning("conversation search: embeddings unavailable, text search: code=timeout")
            return None
        return vector


def _hit(row: SearchRow, mode: SearchMode, text: str) -> SearchHit:
    score = None if row.distance is None else round(1.0 - row.distance, 4)
    return SearchHit(
        row.conversation_id, row.title, row.message_id, text, row.created_at, score, mode
    )
