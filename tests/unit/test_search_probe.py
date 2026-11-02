"""search_probe experiment CLI: the three modes over the index's shared formats.

The formats themselves are tested in test_conversation_index.py.
"""

import uuid

from app.cli.search_probe import Pair, gap, probe, render
from app.rag.conversation_index import document_text, index_text, query_text
from tests.llm_fakes import FakeEmbedder


def pair(title: str | None, question: str, answer: str, vector: list[float]) -> Pair:
    return Pair(uuid.uuid4(), uuid.uuid4(), title, question, answer, vector)


async def test_modes_embed_the_index_formats_and_rank() -> None:
    near, far = [1.0, 0.0], [0.0, 1.0]
    p1 = pair("Перхоть", "Что с перхотью?", "Длинный ответ.", far)
    p2 = pair(None, "Как сушить?", "Феном.", near)
    b1 = index_text("Перхоть", p1.question, p1.answer, max_chars=1000)
    c1 = document_text("Перхоть", p1.question)
    embedder = FakeEmbedder(
        vectors={
            query_text("перхоть"): near,  # A: stored vectors -> p2 wins
            b1: near,  # B: p1 wins
            c1: [0.6, 0.8],  # C: p2's question doc is closer
            index_text(None, p2.question, p2.answer, max_chars=1000): far,
            document_text(None, p2.question): near,
        }
    )
    results = await probe([p1, p2], ["перхоть"], embedder, max_chars=1000)

    assert results[("перхоть", "A")][0].pair is p2
    assert results[("перхоть", "B")][0].pair is p1
    assert results[("перхоть", "C")][0].pair is p2
    assert gap(results[("перхоть", "B")]) == 1.0
    assert "перхоть" not in embedder.embedded  # every mode queries in the index's format
    assert set(embedder.embedded) >= {query_text("перхоть"), b1, c1}

    table = render(results, ["перхоть"], top=5)
    assert "перхоть | B    | Перхоть" in table
    assert "gap top1-top2" in table
