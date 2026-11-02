"""Conversation index over an in-memory repository; the SQL side is in tests/integration."""

import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from app.cli import index_conversations as cli
from app.core.config import Settings
from app.db.base import new_id
from app.db.repositories.message_embeddings import PendingAnswer
from app.rag.conversation_index import (
    CONVERSATION_INDEX_FORMAT_VERSION,
    ConversationIndexer,
    index_conversations,
    index_text,
    is_small_talk,
    pair_text,
    query_text,
)
from app.rag.embeddings import EmbeddingUnavailable
from tests.llm_fakes import FakeEmbedder

ALLOWED = {"provider": "openai", "model": "m", "flagged": False, "action": "allow"}
BLOCKED = {"provider": "openai", "model": "m", "flagged": True, "action": "block"}
SOFT = {"provider": "openai", "model": "m", "flagged": True, "action": "soft"}


def answer(
    text: str = "Мойте мягким шампунем.",
    question: str | None = "Как часто мыть голову?",
    moderation: dict[str, object] | None = ALLOWED,
    *,
    consented: bool = True,
    user_id: uuid.UUID | None = None,
    conversation_id: uuid.UUID | None = None,
) -> PendingAnswer:
    return PendingAnswer(
        new_id(),
        user_id or uuid.uuid4(),
        conversation_id or uuid.uuid4(),
        text,
        question,
        moderation,
        consented,
    )


class MemoryRepo:
    """pending() = answers without a vector of `model` and `index_format` (None = stale), by id
    after `after`, of one conversation if given, like the SQL; small talk (shorter than
    `min_answer_chars`, stripped) only while it has a row, marked `small_talk`."""

    def __init__(self, answers: Sequence[PendingAnswer]) -> None:
        self.answers = sorted(answers, key=lambda a: a.message_id)
        # message_id -> (embedding_model, index_format)
        self.rows: dict[uuid.UUID, tuple[str, str | None]] = {}
        self.deleted: set[uuid.UUID] = set()
        self.saves: list[int] = []

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
    ) -> list[PendingAnswer]:
        found = []
        scope = (user_id, conversation_id)
        for a in self.answers:
            small = len(a.answer.strip()) < min_answer_chars
            wanted = (
                a.message_id in self.rows
                if small
                else self.rows.get(a.message_id) != (model, index_format)
            )
            if (
                wanted
                and (after is None or a.message_id > after)
                and (user_id is None or (a.user_id, a.conversation_id) == scope)
                and a.message_id not in self.deleted
            ):
                found.append(replace(a, small_talk=small))
        return found[:limit]

    async def save(
        self,
        rows: Sequence[tuple[PendingAnswer, Sequence[float]]],
        *,
        model: str,
        index_format: str,
    ) -> int:
        alive = [a for a, _ in rows if a.message_id not in self.deleted]
        for a in alive:
            self.rows[a.message_id] = (model, index_format)
        self.saves.append(len(alive))
        return len(alive)

    async def drop(self, message_ids: Sequence[uuid.UUID]) -> int:
        there = [m for m in message_ids if m in self.rows]
        for m in there:
            del self.rows[m]
        return len(there)

    def models(self) -> set[str]:
        return {model for model, _ in self.rows.values()}


def test_pair_text_is_question_then_answer() -> None:
    assert pair_text("  Как часто мыть?\n", " По мере загрязнения. ", max_chars=1000) == (
        "Вопрос: Как часто мыть?\nОтвет: По мере загрязнения."
    )


@pytest.mark.parametrize(
    ("question", "answer", "kept_question", "kept_answer"),
    [
        ("щ" * 10, "ж" * 500, 10, 82),
        ("щ" * 60, "ж" * 500, 60, 32),
        ("щ" * 500, "ж" * 10, 92, 0),
    ],
    ids=["long-answer-cut", "long-question-kept-answer-cut", "question-alone-too-long"],
)
def test_pair_text_is_cut_to_the_limit(
    question: str, answer: str, kept_question: int, kept_answer: int
) -> None:
    text = pair_text(question, answer, max_chars=108)  # 92 chars of room after the prefixes

    assert len(text) == 108
    assert text == f"Вопрос: {'щ' * kept_question}\nОтвет: {'ж' * kept_answer}"


def test_query_format() -> None:
    assert query_text("  перхоть ") == "task: search result | query: перхоть"


def test_format_3_is_the_untitled_pair() -> None:
    assert CONVERSATION_INDEX_FORMAT_VERSION == "3"
    assert index_text(" Как часто? ", "Раз в два дня.\n", max_chars=1000) == (
        "title: none | text: Вопрос: Как часто?\nОтвет: Раз в два дня."
    )


def test_index_text_cuts_the_answer_to_the_limit() -> None:
    text = index_text("щ" * 4000, "ж" * 20000, max_chars=16000)

    assert len(text) == 16000
    assert text.startswith(f"title: none | text: Вопрос: {'щ' * 4000}\nОтвет: ж")


@pytest.mark.parametrize(
    ("text", "small"),
    [("ж" * 199, True), ("ж" * 200, False), (f"  {'ж' * 199}\n ", True), ("", True)],
    ids=["one-below", "at-threshold", "whitespace-not-counted", "empty"],
)
def test_small_talk_is_shorter_than_the_threshold(text: str, small: bool) -> None:
    assert is_small_talk(text, 200) is small
    assert is_small_talk(text, 0) is False


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
    assert set(repo.rows.values()) == {("fake-embedding", CONVERSATION_INDEX_FORMAT_VERSION)}
    assert index_text(ok.question or "", ok.answer, max_chars=1000) in embedder.embedded
    assert not any("запрещённое" in t for t in embedder.embedded)


async def test_rerun_embeds_nothing_and_a_model_change_rebuilds_everything() -> None:
    repo = MemoryRepo([answer() for _ in range(3)])
    await index_conversations(repo, FakeEmbedder(), max_chars=1000)

    again = FakeEmbedder()
    assert (await index_conversations(repo, again, max_chars=1000)).indexed == 0
    assert again.calls == 0

    report = await index_conversations(repo, FakeEmbedder("new-model"), max_chars=1000)
    assert report.indexed == 3
    assert repo.models() == {"new-model"}


async def test_rows_of_another_format_version_or_stale_are_rebuilt() -> None:
    v1, v2, stale, current = answer(), answer(), answer(), answer()
    repo = MemoryRepo([v1, v2, stale, current])
    repo.rows = {
        v1.message_id: ("fake-embedding", "1"),
        v2.message_id: ("fake-embedding", "2"),  # titled
        stale.message_id: ("fake-embedding", None),  # before the column, or renamed in v2
        current.message_id: ("fake-embedding", CONVERSATION_INDEX_FORMAT_VERSION),
    }
    embedder = FakeEmbedder()

    report = await index_conversations(repo, embedder, max_chars=1000)

    assert report.indexed == 3
    assert len(embedder.embedded) == 3
    assert all(t.startswith("title: none | text: Вопрос: ") for t in embedder.embedded)
    assert set(repo.rows.values()) == {("fake-embedding", CONVERSATION_INDEX_FORMAT_VERSION)}


async def test_small_talk_is_not_indexed_and_its_old_rows_are_dropped() -> None:
    long_enough, below = answer("ж" * 200), answer("ж" * 199, question="hallo")
    v2_greeting = answer("Здравствуйте! Чем помочь?", question="hallo")
    repo = MemoryRepo([long_enough, below, v2_greeting])
    repo.rows = {v2_greeting.message_id: ("fake-embedding", "2")}
    embedder = FakeEmbedder()

    dry = await index_conversations(
        repo, embedder, max_chars=1000, min_answer_chars=200, dry_run=True
    )
    assert (dry.indexed, dry.skipped_small_talk) == (1, 1)
    assert v2_greeting.message_id in repo.rows and embedder.calls == 0

    report = await index_conversations(repo, embedder, max_chars=1000, min_answer_chars=200)

    assert (report.seen, report.indexed, report.skipped_small_talk) == (2, 1, 1)
    assert set(repo.rows) == {long_enough.message_id}
    assert not any("hallo" in t for t in embedder.embedded)
    assert "skipped small talk 1" in report.summary()

    # Not pending any more: the next run reads nothing.
    again = await index_conversations(repo, embedder, max_chars=1000, min_answer_chars=200)
    assert (again.seen, again.skipped_small_talk) == (0, 0)


async def test_a_threshold_change_heals_the_index_both_ways() -> None:
    short, medium = answer("ж" * 50), answer("ж" * 150)
    repo = MemoryRepo([short, medium])

    await index_conversations(repo, FakeEmbedder(), max_chars=1000, min_answer_chars=100)
    assert set(repo.rows) == {medium.message_id}

    raised = await index_conversations(repo, FakeEmbedder(), max_chars=1000, min_answer_chars=200)
    assert (raised.skipped_small_talk, set(repo.rows)) == (1, set())

    lowered = await index_conversations(repo, FakeEmbedder(), max_chars=1000, min_answer_chars=0)
    assert (lowered.indexed, set(repo.rows)) == (2, {short.message_id, medium.message_id})


async def test_users_without_consents_are_skipped_and_counted() -> None:
    consented, revoked = answer(), answer(text="Секретный ответ", consented=False)
    repo = MemoryRepo([consented, revoked])
    embedder = FakeEmbedder()

    report = await index_conversations(repo, embedder, max_chars=1000)

    assert (report.seen, report.indexed, report.skipped_no_consent) == (2, 1, 1)
    assert set(repo.rows) == {consented.message_id}
    assert not any("Секретный" in t for t in embedder.embedded)
    assert "skipped without consent 1" in report.summary()


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
            self,
            rows: Sequence[tuple[PendingAnswer, Sequence[float]]],
            *,
            model: str,
            index_format: str,
        ) -> int:
            self.deleted.add(deleted.message_id)
            return await super().save(rows, model=model, index_format=index_format)

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


def test_a_whole_question_is_kept_at_the_real_limit() -> None:
    question = "щ" * 4000  # MAX_CONTENT_CHARS of a user message
    text = pair_text(question, "ж" * 20000, max_chars=16000)

    assert len(text) == 16000
    assert text.startswith(f"Вопрос: {question}\nОтвет: ж")


# --- ConversationIndexer: one conversation, after a chat answer ---


class BrokenRepo(MemoryRepo):
    async def save(
        self,
        rows: Sequence[tuple[PendingAnswer, Sequence[float]]],
        *,
        model: str,
        index_format: str,
    ) -> int:
        raise ConnectionResetError("secret vector 0.1, 0.2 and text")


def indexer(
    repo: MemoryRepo, embedder: FakeEmbedder, timeout: float | None = 1.0
) -> ConversationIndexer:
    @asynccontextmanager
    async def repositories() -> AsyncIterator[MemoryRepo]:
        yield repo

    return ConversationIndexer(
        repositories, embedder, max_chars=1000, min_answer_chars=10, timeout=timeout
    )


async def test_conversation_indexer_writes_what_that_conversation_lacks_only() -> None:
    user, conv = uuid.uuid4(), uuid.uuid4()
    earlier, saved = (
        answer(user_id=user, conversation_id=conv),
        answer(user_id=user, conversation_id=conv),
    )
    elsewhere = answer(user_id=user)
    repo, embedder = MemoryRepo([earlier, saved, elsewhere]), FakeEmbedder()

    await indexer(repo, embedder).run(user, conv)

    assert set(repo.rows) == {earlier.message_id, saved.message_id}
    assert embedder.embedded == [
        index_text(a.question or "", a.answer, max_chars=1000) for a in (earlier, saved)
    ]


@pytest.mark.parametrize(
    ("embedder", "repo_class", "code"),
    [
        (FakeEmbedder(error=EmbeddingUnavailable("RateLimitError:429")), MemoryRepo, "429"),
        (FakeEmbedder(hang=True), MemoryRepo, "code=timeout"),
        (FakeEmbedder(), BrokenRepo, "code=ConnectionResetError"),
    ],
    ids=["embeddings-down", "embeddings-too-slow", "db-error"],
)
async def test_conversation_indexer_failure_is_a_warning_without_text_and_never_raises(
    embedder: FakeEmbedder,
    repo_class: type[MemoryRepo],
    code: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    saved = answer(text="Секретный ответ", question="Секретный вопрос")
    repo = repo_class([saved])

    with caplog.at_level(logging.INFO, logger="app"):
        await indexer(repo, embedder, timeout=0.05).run(saved.user_id, saved.conversation_id)

    assert repo.rows == {}  # left for the backfill CLI
    [record] = [r for r in caplog.records if r.name == "app.rag.conversation_index"]
    assert record.levelno == logging.WARNING
    assert code in record.getMessage() and str(saved.conversation_id) in record.getMessage()
    assert "Секрет" not in caplog.text and "secret" not in caplog.text


@pytest.mark.parametrize(
    "saved",
    [answer(question=None), answer(moderation=BLOCKED), answer("Привет!")],
    ids=["no-question", "blocked-question", "small-talk"],
)
async def test_conversation_indexer_skips_what_the_backfill_skips(saved: PendingAnswer) -> None:
    repo, embedder = MemoryRepo([saved]), FakeEmbedder()

    await indexer(repo, embedder).run(saved.user_id, saved.conversation_id)

    assert embedder.calls == 0 and repo.rows == {}


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
