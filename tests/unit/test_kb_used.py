import pytest

from app.rag.retriever import Source
from app.services.kb_used import KbUsedFilter, used_sources

DANDRUFF = Source("Перхоть", "dandruff", None)
SHAMPOO = Source("Как выбрать шампунь", "choosing-shampoo", "https://example.test/shampoo")


def run(*pieces: str) -> tuple[list[str], str, set[int]]:
    """What the client gets piece by piece, the whole shown text, and the tag's ids."""
    tag = KbUsedFilter()
    shown = [tag.feed(p) for p in pieces]
    shown.append(tag.finish())
    return shown, "".join(shown), tag.ids


ANSWER = "Мойте мягким шампунем."


@pytest.mark.parametrize(
    ("pieces", "shown_text"),
    [
        ((ANSWER, '\n\n<kb_used ids="1,3"/>'), ANSWER),
        ((f"{ANSWER}\n\n<", 'kb_used ids="1', ',3"/>'), ANSWER),
        ((ANSWER, "\n\n<k", "b_us", 'ed ids="1,3"', "/>"), ANSWER),
        # One character at a time: the blank line was already sent before "<" came.
        (tuple(f'{ANSWER}\n\n<kb_used ids="1,3"/>'), f"{ANSWER}\n\n"),
    ],
)
def test_tag_split_between_pieces_never_reaches_the_client(
    pieces: tuple[str, ...], shown_text: str
) -> None:
    shown, text, ids = run(*pieces)

    assert text == shown_text
    assert not any("<" in s for s in shown)
    assert ids == {1, 3}


def test_text_after_the_tag_is_still_shown() -> None:
    _, text, ids = run('Ответ.\n<kb_used ids="2"/>', "\n")

    assert text == "Ответ.\n"
    assert ids == {2}


@pytest.mark.parametrize(
    "pieces",
    [
        ("Если pH < 5, ", "кожа головы ", "спокойнее."),
        ("Сравните: 3 <", " 5 и <b>жирный</b>."),
        ("Стрелка <", "-- и <kb", "ox> не тег."),
        ("Текст заканчивается на <",),
    ],
)
def test_a_less_than_sign_without_the_tag_loses_no_text(pieces: tuple[str, ...]) -> None:
    _, text, ids = run(*pieces)

    assert text == "".join(pieces)
    assert ids == set()


def test_no_tag_means_no_ids() -> None:
    _, text, ids = run("Просто ", "ответ без базы знаний.")

    assert text == "Просто ответ без базы знаний."
    assert ids == set()


def test_an_unfinished_tag_at_the_end_is_dropped() -> None:
    # The answer hit max_tokens in the middle of the tag.
    _, text, ids = run("Ответ.\n\n<kb_used ids=", '"1')

    assert text == "Ответ."
    assert ids == set()


@pytest.mark.parametrize(
    ("tag", "ids"),
    [
        ('<kb_used ids="1, x, 2,-3,, 1.5,٣"/>', {1, 2}),
        ("<kb_used ids=2,4/>", {2, 4}),
        ("<kb_used ids='3'/>", {3}),
        ("<kb_used/>", set()),
        ('<kb_used ids=""/>', set()),
    ],
)
def test_ids_that_are_not_numbers_are_ignored(tag: str, ids: set[int]) -> None:
    _, text, found = run("Ответ.", tag)

    assert text == "Ответ."
    assert found == ids


def test_sources_are_the_articles_of_the_named_passages_in_passage_order() -> None:
    passages = (DANDRUFF, SHAMPOO, DANDRUFF)

    assert used_sources({3, 2}, passages) == (SHAMPOO, DANDRUFF)
    assert used_sources({1, 3}, passages) == (DANDRUFF,)


def test_ids_out_of_range_are_ignored() -> None:
    assert used_sources({0, 2, 4, 99}, (DANDRUFF, SHAMPOO)) == (SHAMPOO,)
    assert used_sources({1}, ()) == ()
    assert used_sources(set(), (DANDRUFF,)) == ()
