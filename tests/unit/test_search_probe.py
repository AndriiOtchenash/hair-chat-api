"""search_probe experiment CLI: stored vectors (A) vs the current format re-embedded (B).

The formats themselves are tested in test_conversation_index.py.
"""

import uuid

from app.cli.search_probe import Pair, gap, probe, render
from app.rag.conversation_index import index_text, query_text
from tests.llm_fakes import FakeEmbedder


def pair(
    title: str | None,
    question: str,
    answer: str,
    vector: list[float],
    conversation_id: uuid.UUID | None = None,
) -> Pair:
    return Pair(uuid.uuid4(), conversation_id or uuid.uuid4(), title, question, answer, vector)


def b(p: Pair) -> str:
    return index_text(p.question, p.answer, max_chars=1000)


async def test_a_ranks_stored_vectors_and_b_the_current_format() -> None:
    near, far = [1.0, 0.0], [0.0, 1.0]
    p1 = pair("Перхоть", "Что с перхотью?", "Длинный ответ.", far)
    p2 = pair(None, "Как сушить?", "Феном.", near)
    embedder = FakeEmbedder(vectors={query_text("перхоть"): near, b(p1): near, b(p2): far})

    results = await probe([p1, p2], ["перхоть"], embedder, max_chars=1000)

    assert results[("перхоть", "A")][0].pair is p2
    assert results[("перхоть", "B")][0].pair is p1
    assert gap(results[("перхоть", "B")]) == 1.0
    assert sorted(embedder.embedded) == sorted([query_text("перхоть"), b(p1), b(p2)])
    assert not any("Перхоть" in t for t in embedder.embedded)  # no title in format 3

    table = render(results, ["перхоть"], top=5)
    assert "перхоть | B    | Перхоть" in table
    assert "| C    |" not in table and "| D    |" not in table
    assert "gap top1-top2" in table


async def test_b_leaves_out_small_talk_and_pairs_show_the_winner_inside() -> None:
    conv = uuid.uuid4()
    hallo = pair("Перхоть", "hallo", "Здравствуйте!", [1.0, 0.0], conv)
    real = pair("Перхоть", "Что делать с перхотью на коже головы?", "ж" * 50, [0.6, 0.8], conv)
    other = pair("Сушка", "Как сушить?", "ж" * 50, [0.0, 1.0])
    embedder = FakeEmbedder(
        vectors={query_text("перхоть"): [1.0, 0.0], b(real): [0.9, 0.1], b(other): [0.1, 0.9]}
    )

    results = await probe(
        [hallo, real, other], ["перхоть"], embedder, max_chars=1000, min_answer_chars=20
    )

    assert [h.pair for h in results[("перхоть", "A")]] == [hallo, real, other]  # as stored
    assert [h.pair for h in results[("перхоть", "B")]] == [real, other]
    assert b(hallo) not in embedder.embedded

    table = render(results, ["перхоть"], top=5, pairs=True).splitlines()
    a_rows = table[table.index(next(r for r in table if "| A    |" in r)) :][:3]
    assert [r.split("|")[2].strip() for r in a_rows[1:]] == ["hallo", real.question[:40]]
    b_rows = table[table.index(next(r for r in table if "| B    |" in r)) :][:2]
    assert b_rows[1].split("|")[2].strip() == real.question[:40]
    assert "hallo" not in render(results, ["перхоть"], top=5)
