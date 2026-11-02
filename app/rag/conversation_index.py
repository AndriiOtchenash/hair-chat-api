"""Vectors of the user's own conversations, searched by app.services.conversation_search.

The unit is one answer: the user message before it plus the answer, embedded in
gemini-embedding-2's asymmetric document format (ai.google.dev/gemini-api/docs/embeddings)
without a title: "title: none | text: Вопрос: ...\nОтвет: ...", cut to
CONVERSATION_INDEX_MAX_CHARS; the search embeds its query as "task: search result | query: ...".
No conversation title in the vector (format 3): in format 2 it filled short pairs, so a
greeting ("hallo") in a conversation about dandruff beat the real question there
(search_probe --pairs, staging 2026-11-02). So a rename does not touch the index.
Every row stores the model and CONVERSATION_INDEX_FORMAT_VERSION: a row of another model or
format version (NULL: before the column, or marked stale by a rename before format 3) counts as
missing and is rebuilt. One embedding model for the whole index (rule 4).
Small talk is not indexed: an answer shorter than CONVERSATION_INDEX_MIN_ANSWER_CHARS (a
greeting, "ok", "thanks") is never pending, and a row it still has (older format, or a higher
threshold since) is deleted by the next run and counted as skipped_small_talk.
Skipped too, and seen by every run: answers with no user message before them, those whose
question moderation blocked (ChatService never saves a blocked message, ADR 0004, so this only
guards a future change), and users without every valid consent (the chat would not serve them
either: nothing of theirs goes to the embeddings API). Soft (self-harm) questions are indexed:
they are the user's own data, answered and shown in the history like any other.
Writers: the backfill (`index_conversations`, app.cli.index_conversations) and
`ConversationIndexer`, a background job after each saved answer. The job is best effort: a
failure leaves the rows for the backfill. Logs carry counts, ids and codes, never text (rule 6).
"""

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from app.db.repositories.message_embeddings import PendingAnswer
from app.rag.embeddings import MAX_BATCH, Embedder, EmbeddingUnavailable
from app.services.moderation_policy import ModerationAction

logger = logging.getLogger(__name__)

# Stored in message_embeddings.index_format; a change here re-embeds every row on the next
# backfill. 1 = symmetric "Вопрос/Ответ" text (before the column, rows have NULL);
# 2 = asymmetric document/query prefixes with the conversation title; 3 = title always "none".
CONVERSATION_INDEX_FORMAT_VERSION = "3"

QUESTION_PREFIX = "Вопрос: "
ANSWER_PREFIX = "\nОтвет: "
# Google's placeholder for a document without a title: the conversation title is left out.
DOCUMENT_PREFIX = "title: none | text: "


def query_text(query: str) -> str:
    """gemini-embedding-2 query format for search (asymmetric retrieval)."""
    return f"task: search result | query: {query.strip()}"


class MessageEmbeddingRepository(Protocol):
    async def pending(
        self,
        *,
        model: str,
        index_format: str,
        after: uuid.UUID | None,
        limit: int,
        user_id: uuid.UUID | None = None,
        conversation_id: uuid.UUID | None = None,
        min_answer_chars: int = 0,
    ) -> Sequence[PendingAnswer]: ...

    async def drop(self, message_ids: Sequence[uuid.UUID]) -> int: ...

    async def save(
        self,
        rows: Sequence[tuple[PendingAnswer, Sequence[float]]],
        *,
        model: str,
        index_format: str,
    ) -> int: ...


MessageEmbeddingRepositoryFactory = Callable[
    [], AbstractAsyncContextManager[MessageEmbeddingRepository]
]


@dataclass
class IndexReport:
    seen: int = 0
    indexed: int = 0  # in a dry run: would be indexed
    skipped_small_talk: int = 0  # rows of short answers deleted (in a dry run: would be)
    skipped_blocked: int = 0
    skipped_no_question: int = 0
    skipped_no_consent: int = 0
    gone: int = 0  # deleted between reading and writing

    def summary(self) -> str:
        return (
            f"answers seen {self.seen}, indexed {self.indexed}, "
            f"skipped small talk {self.skipped_small_talk}, "
            f"skipped blocked {self.skipped_blocked}, "
            f"skipped without question {self.skipped_no_question}, "
            f"skipped without consent {self.skipped_no_consent}, gone {self.gone}"
        )


def pair_text(question: str, answer: str, *, max_chars: int) -> str:
    """The "question + answer" part of a row, at most `max_chars` long.

    When it is too long the answer is cut and the question kept: the search looks for what
    the user asked. The question (at most 4000 chars, MAX_CONTENT_CHARS) is cut only when it
    alone does not fit, which CONVERSATION_INDEX_MAX_CHARS (>= 200) makes a corner case.
    """
    question, answer = question.strip(), answer.strip()
    room = max(0, max_chars - len(QUESTION_PREFIX) - len(ANSWER_PREFIX))
    question = question[:room]
    answer = answer[: room - len(question)]
    return f"{QUESTION_PREFIX}{question}{ANSWER_PREFIX}{answer}"


def is_small_talk(answer: str, min_answer_chars: int) -> bool:
    """Too short to be worth a vector (a greeting, "ok"); the backfill's SQL
    (SqlMessageEmbeddingRepository.pending) decides the same."""
    return len(answer.strip()) < min_answer_chars


def index_text(question: str, answer: str, *, max_chars: int) -> str:
    """The text embedded for one answer (format 3), at most `max_chars` long."""
    pair = pair_text(question, answer, max_chars=max_chars - len(DOCUMENT_PREFIX))
    return f"{DOCUMENT_PREFIX}{pair}"


def is_blocked(moderation: dict[str, Any] | None) -> bool:
    """messages.moderation of a question that moderation blocked (see ChatService._moderate)."""
    return moderation is not None and moderation.get("action") == ModerationAction.BLOCK.value


async def index_conversations(
    repo: MessageEmbeddingRepository,
    embedder: Embedder,
    *,
    max_chars: int,
    min_answer_chars: int = 0,
    batch_size: int = MAX_BATCH,
    limit: int | None = None,
    dry_run: bool = False,
    user_id: uuid.UUID | None = None,
    conversation_id: uuid.UUID | None = None,
) -> IndexReport:
    """Embed the answers the index lacks, `batch_size` per embeddings call and transaction, so
    a failure keeps the batches already written and a re-run continues. `limit`: answers to
    look at in this run (skipped ones count too). `user_id` + `conversation_id`: one
    conversation only (ConversationIndexer). `min_answer_chars`: shorter answers are small
    talk, their leftover rows are deleted. Raises EmbeddingUnavailable."""
    report = IndexReport()
    after: uuid.UUID | None = None
    version = CONVERSATION_INDEX_FORMAT_VERSION
    while True:
        size = batch_size if limit is None else min(batch_size, limit - report.seen)
        if size <= 0:
            break
        batch = await repo.pending(
            model=embedder.model,
            index_format=version,
            after=after,
            limit=size,
            user_id=user_id,
            conversation_id=conversation_id,
            min_answer_chars=min_answer_chars,
        )
        if not batch:
            break
        after = batch[-1].message_id
        report.seen += len(batch)
        todo: list[PendingAnswer] = []
        small_talk: list[uuid.UUID] = []
        for answer in batch:
            if answer.small_talk:
                small_talk.append(answer.message_id)
            elif not answer.consented:
                report.skipped_no_consent += 1
            elif answer.question is None:
                report.skipped_no_question += 1
            elif is_blocked(answer.question_moderation):
                report.skipped_blocked += 1
            else:
                todo.append(answer)
        if dry_run:
            report.indexed += len(todo)
            report.skipped_small_talk += len(small_talk)
            continue
        if small_talk:
            report.skipped_small_talk += await repo.drop(small_talk)
        if not todo:
            continue
        vectors = await embedder.embed(
            [index_text(a.question or "", a.answer, max_chars=max_chars) for a in todo]
        )
        saved = await repo.save(
            list(zip(todo, vectors, strict=True)), model=embedder.model, index_format=version
        )
        report.indexed += saved
        report.gone += len(todo) - saved
        if conversation_id is None:
            logger.info("conversation index batch: indexed=%d total=%d", saved, report.indexed)
    return report


class ConversationIndexer:
    """Indexes what one conversation lacks after the chat saved an answer: that answer, plus
    any earlier one left behind.

    Own DB session, not the request's. `timeout` bounds the whole run (None: only the
    embedder's own HTTP timeout). Never raises: a failure is a WARNING with a safe code and
    the rows stay pending until app.cli.index_conversations picks them up.
    """

    def __init__(
        self,
        repositories: MessageEmbeddingRepositoryFactory,
        embedder: Embedder,
        *,
        max_chars: int,
        min_answer_chars: int,
        timeout: float | None,
    ) -> None:
        self._repositories = repositories
        self._embedder = embedder
        self._max_chars = max_chars
        self._min_answer_chars = min_answer_chars
        self._timeout = timeout

    async def run(self, user_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        try:
            async with asyncio.timeout(self._timeout), self._repositories() as repo:
                report = await index_conversations(
                    repo,
                    self._embedder,
                    max_chars=self._max_chars,
                    min_answer_chars=self._min_answer_chars,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
        except EmbeddingUnavailable as exc:
            code = exc.code
        except TimeoutError:
            code = "timeout"
        except Exception as exc:
            # Class name only: DB errors may carry the vector or text in their parameters.
            code = type(exc).__name__
        else:
            logger.info(
                "conversation index: conversation=%s indexed=%d skipped_no_consent=%d ms=%d",
                conversation_id,
                report.indexed,
                report.skipped_no_consent,
                (loop.time() - t0) * 1000,
            )
            return
        logger.warning(
            "conversation index failed, left for the backfill: conversation=%s code=%s",
            conversation_id,
            code,
        )
