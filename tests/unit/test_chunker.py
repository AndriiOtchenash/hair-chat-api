from app.rag.chunker import chunk_markdown
from app.services.tokens import estimate_tokens

TITLE = "Перхоть"


def paragraph(word: str, sentences: int) -> str:
    return " ".join(f"{word} номер {i} про уход за волосами." for i in range(sentences))


def test_short_article_is_one_chunk_with_the_title_breadcrumb() -> None:
    body = (
        "# Перхоть\n\nВступление.\n\n## Уход\n\nМягкий шампунь.\n\n## Когда к врачу\n\nЕсли больно."
    )

    chunks = chunk_markdown(TITLE, body)

    assert len(chunks) == 1
    text = chunks[0].text
    assert text.startswith("Перхоть\n\nВступление.")
    assert "Уход\nМягкий шампунь." in text
    assert "Когда к врачу\nЕсли больно." in text
    assert text.count("# ") == 0  # markdown heading marks are not repeated inside chunks


def test_sections_that_do_not_fit_together_start_new_chunks_with_their_breadcrumb() -> None:
    body = f"## Уход\n\n{paragraph('уход', 25)}\n\n## Когда к врачу\n\n{paragraph('врач', 25)}"

    chunks = chunk_markdown(TITLE, body, target_tokens=300)

    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].text.startswith("Перхоть › Уход\n\n")
    assert chunks[-1].text.startswith("Перхоть › Когда к врачу\n\n")
    assert all(estimate_tokens(c.text) <= 300 for c in chunks)


def test_nested_headings_make_the_breadcrumb_path() -> None:
    body = (
        f"## Уход\n\n### Шампунь\n\n{paragraph('шампунь', 30)}\n\n## Другое\n\n{paragraph('x', 30)}"
    )

    chunks = chunk_markdown(TITLE, body, target_tokens=300)

    assert chunks[0].text.startswith("Перхоть › Уход › Шампунь\n\n")
    assert any(c.text.startswith("Перхоть › Другое\n\n") for c in chunks)


def test_long_section_is_split_with_overlap_and_stays_within_target() -> None:
    sentences = [f"Предложение {i} о мытье головы и выборе шампуня." for i in range(120)]
    body = "## Мытьё\n\n" + "\n\n".join(" ".join(sentences[i : i + 4]) for i in range(0, 120, 4))

    chunks = chunk_markdown(TITLE, body, target_tokens=200, overlap_tokens=30)

    assert len(chunks) > 3
    assert all(estimate_tokens(c.text) <= 200 for c in chunks)
    assert all(c.text.startswith("Перхоть › Мытьё\n\n") for c in chunks)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_sentence = previous.text.rsplit(". ", 1)[-1].strip()
        assert last_sentence in current.text  # the tail of one chunk opens the next
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sentences)  # nothing lost


def test_one_huge_paragraph_is_split_by_sentences() -> None:
    body = paragraph("длинный", 200)

    chunks = chunk_markdown(TITLE, body, target_tokens=150, overlap_tokens=20)

    assert len(chunks) > 5
    assert all(estimate_tokens(c.text) <= 150 for c in chunks)
