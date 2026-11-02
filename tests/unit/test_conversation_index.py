"""Conversation index over an in-memory repository; the SQL side is in tests/integration."""

import uuid
from collections.abc import Sequence

import pytest

from app.cli import index_conversations as cli
from app.core.config import Settings
from app.db.base import new_id
from app.db.repositories.message_embeddings import PendingAnswer
from app.rag.conversation_index import index_conversations, pair_text
from app.rag.embeddings import EmbeddingUnavailable
from tests.llm_fakes import FakeEmbedder

ALLOWED = {"provider": "openai", "model": "m", "flagged": False, "action": "allow"}
BLOCKED = {"provider": "openai", "model": "m", "flagged": True, "action": "block"}
SOFT = {"provider": "openai", "model": "m", "flagged": True, "action": "soft"}


def answer(
    text: str = "Мойте мягким шампунем.",
    question: str | None = "Как часто мыть голову?",
    moderation: dict[str, object] | None = ALLOWED,
) -> PendingAnswer:
    return PendingAnswer(new_id(), uuid.uuid4(), uuid.uuid4(), text, question, moderation)


class MemoryRepo:
    """pending() = answers without a vector of `model`, by id after `after`, like the SQL."""

    def __init__(self, answers: Sequence[PendingAnswer]) -> None:
        self.answers = sorted(answers, key=lambda a: a.message_id)
        self.rows: dict[uuid.UUID, str] = {}  # message_id -> embedding_model
        self.deleted: set[uuid.UUID] = set()
        self.saves: list[int] = []

    async def pending(
        self, *, model: str, after: uuid.UUID | None, limit: int
    ) -> list[PendingAnswer]:
        return [
            a
            for a in self.answers
            if self.rows.get(a.message_id) != model
            and (after is None or a.message_id > after)
            and a.message_id not in self.deleted
        ][:limit]

    async def save(
        self, rows: Sequence[tuple[PendingAnswer, Sequence[float]]], *, model: str
    ) -> int:
        alive = [a for a, _ in rows if a.message_id not in self.deleted]
        for a in alive:
            self.rows[a.message_id] = model
        self.saves.append(len(alive))
        return len(alive)


def test_pair_text_is_question_then_answer() -> None:
    assert pair_text("  Как часто мыть?\n", " По мере загрязнения. ", max_chars=1000) == (
        "Вопрос: Как часто мыть?\nОтвет: По мере загрязнения."
    )


@pytest.mark.parametrize(
    ("question", "answer", "kept_question", "kept_answer"),
    [
        ("щ" * 10, "ж" * 500, 10, 82),
        ("щ" * 500, "ж" * 10, 82, 10),
        ("щ" * 500, "ж" * 500, 46, 46),
    ],
    ids=["long-answer-cut", "long-question-cut", "both-long-answer-keeps-half"],
)
def test_pair_text_is_cut_to_the_limit(
    question: str, answer: str, kept_question: int, kept_answer: int
) -> None:
    text = pair_text(question, answer, max_chars=108)  # 92 chars of room after the prefixes

    assert len(text) == 108
    assert text == f"Вопрос: {'щ' * kept_question}\nОтвет: {'ж' * kept_answer}"


async def test_indexes_pairs_and_skips_blocked_and_questionless_answers() -> None:
    ok, soft, blocked, orphan, unmoderated = (
        answer(),
        answer(question="Мне плохо", moderation=SOFT),
        answer(question="запрещённое", moderation=BLOCKED),
        answer(question=None, moderation=None),
        answer(question="Сухие кончики?", moderation={"status": "unavailable"}),
    )
    repo = MemoryRepo([ok, soft, blocked, orphan, unmoderated])
    embedder = FakeEmbedder()

    report = await index_conversations(repo, embedder, max_chars=1000)

    assert (report.seen, report.indexed, report.skipped_blocked, report.skipped_no_question) == (
        5,
        3,
        1,
        1,
    )
    assert set(repo.rows) == {ok.message_id, soft.message_id, unmoderated.message_id}
    assert set(repo.rows.values()) == {"fake-embedding"}
    assert pair_text(ok.question or "", ok.answer, max_chars=1000) in embedder.embedded
    assert not any("запрещённое" in t for t in embedder.embedded)


async def test_rerun_embeds_nothing_and_a_model_change_rebuilds_everything() -> None:
    repo = MemoryRepo([answer() for _ in range(3)])
    await index_conversations(repo, FakeEmbedder(), max_chars=1000)

    again = FakeEmbedder()
    assert (await index_conversations(repo, again, max_chars=1000)).indexed == 0
    assert again.calls == 0

    report = await index_conversations(repo, FakeEmbedder("new-model"), max_chars=1000)
    assert report.indexed == 3
    assert set(repo.rows.values()) == {"new-model"}


async def test_batches_and_limit() -> None:
    repo = MemoryRepo([answer() for _ in range(7)])
    embedder = FakeEmbedder()

    report = await index_conversations(repo, embedder, max_chars=1000, batch_size=3, limit=5)

    assert report.indexed == 5
    assert repo.saves == [3, 2]
    assert embedder.calls == 2


async def test_skipped_answers_do_not_stall_the_batches() -> None:
    # A full batch of skipped answers must not be read again forever (keyset paging).
    repo = MemoryRepo([answer(question=None) for _ in range(4)] + [answer()])

    report = await index_conversations(repo, FakeEmbedder(), max_chars=1000, batch_size=2)

    assert (report.seen, report.indexed, report.skipped_no_question) == (5, 1, 4)


async def test_dry_run_counts_without_embedding_or_writing() -> None:
    repo = MemoryRepo([answer(), answer(moderation=BLOCKED)])
    embedder = FakeEmbedder()

    report = await index_conversations(repo, embedder, max_chars=1000, dry_run=True)

    assert (report.indexed, report.skipped_blocked) == (1, 1)
    assert embedder.calls == 0 and repo.rows == {}


async def test_answers_deleted_meanwhile_are_counted_as_gone() -> None:
    kept, deleted = answer(), answer()

    class DeletingRepo(MemoryRepo):
        async def save(
            self, rows: Sequence[tuple[PendingAnswer, Sequence[float]]], *, model: str
        ) -> int:
            self.deleted.add(deleted.message_id)
            return await super().save(rows, model=model)

    repo = DeletingRepo([kept, deleted])
    report = await index_conversations(repo, FakeEmbedder(), max_chars=1000)

    assert (report.indexed, report.gone) == (1, 1)
    assert set(repo.rows) == {kept.message_id}


async def test_embedding_failure_keeps_written_batches() -> None:
    repo = MemoryRepo([answer() for _ in range(2)])
    await index_conversations(repo, FakeEmbedder(), max_chars=1000, batch_size=1, limit=1)

    with pytest.raises(EmbeddingUnavailable):
        await index_conversations(repo, FakeEmbedder(fail=True), max_chars=1000)
    assert len(repo.rows) == 1


def test_cli_needs_the_gemini_key(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(_env_file=None, gemini_api_key=None)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    assert cli.main(["--dry-run"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_cli_rejects_a_non_positive_limit() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--limit", "0"])
