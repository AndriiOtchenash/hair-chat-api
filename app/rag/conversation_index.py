"""Vectors of the user's own conversations, for a later search over them (iteration 1 of 3:
storage and backfill only).

The unit is one answer: the user message before it plus the answer, as
"Вопрос: ...\nОтвет: ...", cut to CONVERSATION_INDEX_MAX_CHARS. One embedding model for the
whole index (rule 4): a row of another model counts as missing and is rebuilt.
Skipped: answers with no user message before them, and those whose question moderation
blocked. ChatService never saves a blocked message (ADR 0004), so the second rule only guards
against a future change. Soft (self-harm) questions are indexed: they are the user's own data,
answered and shown in the history like any other.
Reports counts only: no text, no ids (rule 6).
"""

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.db.repositories.message_embeddings import PendingAnswer
from app.rag.embeddings import MAX_BATCH, Embedder
from app.services.moderation_policy import ModerationAction

logger = logging.getLogger(__name__)

QUESTION_PREFIX = "Вопрос: "
ANSWER_PREFIX = "\nОтвет: "


class MessageEmbeddingRepository(Protocol):
    async def pending(
        self, *, model: str, after: uuid.UUID | None, limit: int
    ) -> Sequence[PendingAnswer]: ...

    async def save(
        self, rows: Sequence[tuple[PendingAnswer, Sequence[float]]], *, model: str
    ) -> int: ...


@dataclass
class IndexReport:
    seen: int = 0
    indexed: int = 0  # in a dry run: would be indexed
    skipped_blocked: int = 0
    skipped_no_question: int = 0
    gone: int = 0  # deleted between reading and writing

    def summary(self) -> str:
        return (
            f"answers seen {self.seen}, indexed {self.indexed}, "
            f"skipped blocked {self.skipped_blocked}, "
            f"skipped without question {self.skipped_no_question}, gone {self.gone}"
        )


def pair_text(question: str, answer: str, *, max_chars: int) -> str:
    """The text embedded for one answer, at most `max_chars` long.

    When it is too long the answer keeps at least half of the room: the question is cut first
    if the answer is short, otherwise both are.
    """
    question, answer = question.strip(), answer.strip()
    room = max_chars - len(QUESTION_PREFIX) - len(ANSWER_PREFIX)
    question = question[: max(room - len(answer), room // 2)]
    answer = answer[: room - len(question)]
    return f"{QUESTION_PREFIX}{question}{ANSWER_PREFIX}{answer}"


def is_blocked(moderation: dict[str, Any] | None) -> bool:
    """messages.moderation of a question that moderation blocked (see ChatService._moderate)."""
    return moderation is not None and moderation.get("action") == ModerationAction.BLOCK.value


async def index_conversations(
    repo: MessageEmbeddingRepository,
    embedder: Embedder,
    *,
    max_chars: int,
    batch_size: int = MAX_BATCH,
    limit: int | None = None,
    dry_run: bool = False,
) -> IndexReport:
    """Embed the answers the index lacks, `batch_size` per embeddings call and transaction, so
    a failure keeps the batches already written and a re-run continues. `limit`: answers to
    look at in this run (skipped ones count too). Raises EmbeddingUnavailable."""
    report = IndexReport()
    after: uuid.UUID | None = None
    while True:
        size = batch_size if limit is None else min(batch_size, limit - report.seen)
        if size <= 0:
            break
        batch = await repo.pending(model=embedder.model, after=after, limit=size)
        if not batch:
            break
        after = batch[-1].message_id
        report.seen += len(batch)
        todo: list[PendingAnswer] = []
        for answer in batch:
            if answer.question is None:
                report.skipped_no_question += 1
            elif is_blocked(answer.question_moderation):
                report.skipped_blocked += 1
            else:
                todo.append(answer)
        if dry_run:
            report.indexed += len(todo)
            continue
        if not todo:
            continue
        vectors = await embedder.embed(
            [pair_text(a.question or "", a.answer, max_chars=max_chars) for a in todo]
        )
        saved = await repo.save(list(zip(todo, vectors, strict=True)), model=embedder.model)
        report.indexed += saved
        report.gone += len(todo) - saved
        logger.info("conversation index batch: indexed=%d total=%d", saved, report.indexed)
    return report
