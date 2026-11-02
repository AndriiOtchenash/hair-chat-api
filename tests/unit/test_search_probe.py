"""search_probe experiment CLI: gemini-embedding-2 prefixes and the three modes."""

import uuid

from app.cli.search_probe import Pair, document_text, gap, probe, query_text, render
from app.rag.conversation_index import pair_text
from tests.llm_fakes import FakeEmbedder


def test_query_format() -> None:
    assert query_text("  перхоть ") == "task: search result | query: перхоть"


def test_document_format() -> None:
    assert document_text("Сухие кончики", " Вопрос: что делать? ") == (
        "title: Сухие кончики | text: Вопрос: что делать?"
    )


def test_document_without_title_uses_none() -> None:
    assert document_text(None, "x") == "title: none | text: x"
    assert document_text("   ", "x") == "title: none | text: x"


def pair(title: str | None, question: str, answer: str, vector: list[float]) -> Pair:
    return Pair(uuid.uuid4(), uuid.uuid4(), title, question, answer, vector)


async def test_modes_embed_the_right_texts_and_rank() -> None:
    near, far = [1.0, 0.0], [0.0, 1.0]
    p1 = pair("Перхоть", "Что с перхотью?", "Длинный ответ.", far)
    p2 = pair(None, "Как сушить?", "Феном.", near)
    b1 = document_text("Перхоть", pair_text(p1.question, p1.answer, max_chars=1000))
    c1 = document_text("Перхоть", p1.question)
    embedder = FakeEmbedder(
        vectors={
            "перхоть": near,  # A: raw query, stored vectors -> p2 wins
            query_text("перхоть"): near,
            b1: near,  # B: p1 wins
            c1: [0.6, 0.8],  # C: p1 close but p2's C doc decides
            document_text(None, pair_text(p2.question, p2.answer, max_chars=1000)): far,
            document_text(None, p2.question): near,
        }
    )
    results = await probe([p1, p2], ["перхоть"], embedder, max_chars=1000)

    assert results[("перхоть", "A")][0].pair is p2
    assert results[("перхоть", "B")][0].pair is p1
    assert results[("перхоть", "C")][0].pair is p2
    assert gap(results[("перхоть", "B")]) == 1.0
    assert set(embedder.embedded) >= {"перхоть", query_text("перхоть"), b1, c1}

    table = render(results, ["перхоть"], top=5)
    assert "перхоть | B    | Перхоть" in table
    assert "gap top1-top2" in table
