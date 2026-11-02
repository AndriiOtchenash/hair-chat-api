"""Question -> knowledge-base passages for the <knowledge> block (KnowledgeSource of 1.9).

Two steps, so the chat can embed the question while moderation runs (ChatService.reply):
`QueryEmbedder.embed_query` (network, never raises) and `RagKnowledge.knowledge` (DB search with
that vector, threshold, packing into the token budget).

A follow-up ("А шампунь какой?") says little on its own, so the dialog is searched too
(`dialog_query`: the rolling summary + the latest message, else the last user messages), and the
two searches are merged so that a change of topic still gets the new topic's passages
(`merge_hits`). Measured with `scripts.prompt_eval --retrieval-only` (dialog cases).
"""

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.db.repositories.kb import KBHit, KBRepository
from app.rag.embeddings import Embedder, EmbeddingUnavailable
from app.services.tokens import CHARS_PER_TOKEN, SAFETY_MARGIN, estimate_tokens

logger = logging.getLogger(__name__)

# The summary part of the dialog query. gemini-embedding-2 takes 8192 input tokens and the summary
# is up to 1024 (SUMMARY_MAX_TOKENS), so this is about the vector, not the limit: about one KB
# chunk (~500 tokens), so the summary does not drown the latest message. Its tail is kept (newer).
SUMMARY_QUERY_MAX_TOKENS = 500
# Without a summary: the latest user message and the ones before it, this many in all.
DIALOG_USER_MESSAGES = 3
# Of the merged passages, at least this many best ones come from the latest message alone.
LATEST_KEEP = 2


@dataclass(frozen=True, slots=True)
class QueryEmbedding:
    vector: Sequence[float]
    model: str  # searched against kb_chunks.embedding_model (rule 4)
    # The dialog query (`dialog_query`), same model; None on the first message of a conversation.
    context_vector: Sequence[float] | None = None


@dataclass(frozen=True, slots=True)
class QueryFailed:
    """The embeddings call failed: no knowledge base for this reply, and the model is told so.

    `code` is EmbeddingUnavailable.code or "timeout": safe to log and store (no message text).
    """

    code: str


# None = no embedder configured (dev without a key): no knowledge base, nothing failed.
QueryResult = QueryEmbedding | QueryFailed | None


class RetrievalStatus(StrEnum):
    """How the knowledge-base search for a reply went; stored in messages.retrieval."""

    OK = "ok"  # passages found (they may still be dropped by the token budget)
    EMPTY = "empty"  # searched, nothing above KB_MIN_SIMILARITY
    ERROR = "error"  # embeddings down or too slow (QueryFailed)
    DISABLED = "disabled"  # no embedder (dev only: deployed envs require the key)


@dataclass(frozen=True, slots=True)
class Retrieval:
    status: RetrievalStatus
    error: str | None = None  # QueryFailed.code, only with ERROR


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


def dialog_query(latest: str, *, summary: str | None, earlier_user: Sequence[str]) -> str | None:
    """The search text for the dialog: summary (its tail) + the latest message, else the last
    DIALOG_USER_MESSAGES user messages (oldest first). None when there is nothing before it."""
    if summary and summary.strip():
        return f"{_tail(summary.strip(), SUMMARY_QUERY_MAX_TOKENS)}\n{latest}"
    earlier = [t for t in earlier_user if t.strip()][-(DIALOG_USER_MESSAGES - 1) :]
    return "\n".join([*earlier, latest]) if earlier else None


def _tail(text: str, max_tokens: int) -> str:
    """The end of `text` within max_tokens (app.services.tokens estimate), from a word start."""
    if estimate_tokens(text) <= max_tokens:
        return text
    cut = text[-int(max_tokens * CHARS_PER_TOKEN / SAFETY_MARGIN) :]
    word = cut.find(" ")
    return cut[word + 1 :] if word != -1 else cut


def merge_hits(latest: Sequence[KBHit], dialog: Sequence[KBHit], limit: int) -> list[KBHit]:
    """The best LATEST_KEEP of the latest message's hits, then the dialog's, then the rest of the
    latest's; no chunk twice, at most `limit`. The latest message keeps its own topic even when
    the dialog is about something else; the dialog fills in what a follow-up leaves unsaid."""
    merged: list[KBHit] = []
    seen: set[tuple[str, str]] = set()
    for hit in [*latest[:LATEST_KEEP], *dialog, *latest[LATEST_KEEP:]]:
        key = (hit.slug, hit.content)
        if key in seen or len(merged) == limit:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


class QueryEmbedder(Protocol):
    async def embed_query(self, text: str, context: str | None = None) -> QueryResult:
        """`text` is the latest message, `context` the dialog query (`dialog_query`), both in one
        call. Never raises: QueryFailed or None means no knowledge base for this reply."""
        ...


class RagQueryEmbedder:
    def __init__(self, embedder: Embedder, timeout: float | None = None) -> None:
        self._embedder = embedder
        # Whole call, on top of the HTTP client's per-phase timeouts: the answer waits for it.
        self._timeout = timeout

    async def embed_query(self, text: str, context: str | None = None) -> QueryResult:
        """QueryFailed when embeddings are down: the answer goes without the knowledge base (no
        fallback model, its vectors would not match the index)."""
        texts = [text] if context is None else [text, context]
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        outcome = "ok"
        try:
            async with asyncio.timeout(self._timeout):
                vectors = await self._embedder.embed(texts)
        except EmbeddingUnavailable as exc:
            outcome = exc.code
            log = logger.error if exc.misconfigured else logger.warning
            log("embeddings unavailable, answering without knowledge: code=%s", exc.code)
            return QueryFailed(exc.code)
        except TimeoutError:
            outcome = "timeout"
            logger.warning("embeddings unavailable, answering without knowledge: code=timeout")
            return QueryFailed("timeout")
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            # Timing diagnostics (staging timeouts): lengths only, never the text (rule 6).
            logger.info(
                "embed_query: outcome=%s elapsed_ms=%d n_texts=%d chars=%s",
                outcome,
                (loop.time() - t0) * 1000,
                len(texts),
                ",".join(str(len(t)) for t in texts),
            )
        context_vector = vectors[1] if context is not None else None
        return QueryEmbedding(vectors[0], self._embedder.model, context_vector)


class NoQueryEmbedder:
    """No embedding key (dev): no knowledge base."""

    async def embed_query(self, text: str, context: str | None = None) -> QueryResult:
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
        hits = await self._search(query.vector, query.model)
        relevant = [h for h in hits if h.similarity >= self._min_similarity]
        dialog: list[KBHit] = []
        if query.context_vector is not None:
            dialog = [
                h
                for h in await self._search(query.context_vector, query.model)
                if h.similarity >= self._min_similarity
            ]
            relevant = merge_hits(relevant, dialog, self._top_k)
        packed = self._pack(relevant)
        logger.info(
            "knowledge: hits=%d relevant=%d dialog=%d used=%d best=%.3f",
            len(hits),
            len(relevant),
            len(dialog),
            len(packed),
            hits[0].similarity if hits else 0.0,
        )
        if not packed:
            return None
        text = "\n\n".join(_passage(number, hit) for number, hit in enumerate(packed, start=1))
        return Knowledge(text, tuple(Source(h.title, h.slug, h.url) for h in packed))

    async def _search(self, vector: Sequence[float], model: str) -> list[KBHit]:
        return await self._repo.search(
            vector, model=model, include_drafts=self._include_drafts, limit=self._top_k
        )

    def _pack(self, hits: list[KBHit]) -> list[KBHit]:
        """In the given order (closest first), within max_tokens; a passage that does not fit is
        skipped."""
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
