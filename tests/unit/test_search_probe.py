"""search_probe experiment CLI: the four modes over the index's shared formats.

The formats themselves are tested in test_conversation_index.py.
"""

import uuid

from app.cli.search_probe import Pair, gap, probe, render, untitled_text
from app.rag.conversation_index import document_text, index_text, query_text
from tests.llm_fakes import FakeEmbedder


def pair(
    title: str | None,
    question: str,
    answer: str,
    vector: list[float],
    conversation_id: uuid.UUID | None = None,
) -> Pair:
    return Pair(uuid.uuid4(), conversation_id or uuid.uuid4(), title, question, answer, vector)


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
            untitled_text(p1, max_chars=1000): far,  # p2 has no title: its D text = its B text
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


def test_mode_d_is_the_index_format_without_the_title() -> None:
    p = pair("Перхоть", " hallo ", "Здравствуйте!", [1.0])

    assert (
        untitled_text(p, max_chars=1000)
        == "title: none | text: Вопрос: hallo\nОтвет: Здравствуйте!"
    )
    assert untitled_text(p, max_chars=1000) == index_text(
        None, "hallo", "Здравствуйте!", max_chars=1000
    )
    assert len(untitled_text(pair("T", "q", "a" * 500, [1.0]), max_chars=100)) == 100


async def test_mode_d_and_pairs_show_which_pair_wins_inside_a_conversation() -> None:
    """In B the title lifts the greeting above the real question; D drops the title."""
    conv = uuid.uuid4()
    hallo = pair("Перхоть", "hallo", "Здравствуйте!", [1.0, 0.0], conv)
    real = pair("Перхоть", "Что делать с перхотью на коже головы?", "Шампунь.", [1.0, 0.0], conv)
    embedder = FakeEmbedder(
        vectors={
            query_text("перхоть"): [1.0, 0.0],
            index_text("Перхоть", "hallo", "Здравствуйте!", max_chars=1000): [0.9, 0.1],
            index_text("Перхоть", real.question, real.answer, max_chars=1000): [0.5, 0.5],
            untitled_text(hallo, max_chars=1000): [0.1, 0.9],
            untitled_text(real, max_chars=1000): [0.8, 0.2],
            document_text("Перхоть", "hallo"): [0.0, 1.0],
            document_text("Перхоть", real.question): [0.0, 1.0],
        }
    )
    results = await probe([hallo, real], ["перхоть"], embedder, max_chars=1000)

    assert [h.pair for h in results[("перхоть", "B")]] == [hallo, real]  # every pair, best first
    assert [h.pair for h in results[("перхоть", "D")]] == [real, hallo]

    table = render(results, ["перхоть"], top=5, pairs=True).splitlines()
    rows_b = table[table.index(next(r for r in table if "| B    |" in r)) :][:3]
    assert "| B    | Перхоть" in rows_b[0]
    assert rows_b[1].split("|")[2].strip() == "hallo"
    assert rows_b[2].split("|")[2].strip() == "Что делать с перхотью на коже головы?"[:40]
    rows_d = table[table.index(next(r for r in table if "| D    |" in r)) :][:3]
    assert rows_d[1].split("|")[2].strip().startswith("Что делать с перхотью")
    assert rows_d[2].split("|")[2].strip() == "hallo"
    # one conversation -> one row per mode, no gap
    assert sum("| D    | Перхоть" in r for r in table) == 1
    assert "перхоть | D    | n/a" in table

    plain = render(results, ["перхоть"], top=5)
    assert "hallo" not in plain
