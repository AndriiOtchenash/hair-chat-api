import pytest

from app.db.models import Locale
from app.services.reply_language import (
    OTHER,
    ReplyLanguage,
    detect_language,
    directive,
    reply_lang_of,
    resolve_reply_lang,
    strip_quoted,
)

POLISH = "Mam łupież, co robić?"
UKRAINIAN = "Після фарбування волосся стало сухим і ламким. Як його відновити?"
QUOTED_ENGLISH = (
    'В статье написано: "Dandruff is caused by Malassezia yeast overgrowth on the scalp, which '
    'feeds on sebum and irritates the skin." Это правда?'
)


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        (POLISH, Locale.PL),
        (UKRAINIAN, Locale.UK),
        ("Що робити з лупою?", Locale.UK),
        ("Как часто мыть голову?", Locale.RU),
        ("How often should I wash oily hair?", Locale.EN),
    ],
)
def test_a_message_surely_in_one_of_ours_gets_that_language(message: str, lang: Locale) -> None:
    # Whatever the locale and the language of the previous answer.
    assert resolve_reply_lang(message, Locale.EN, Locale.RU) == (lang, False)
    another = Locale.PL if lang is Locale.UK else Locale.UK
    assert resolve_reply_lang(message, None, another) == (lang, False)


@pytest.mark.parametrize(
    "message",
    ["hallo", "ok", "OK!", "thanks", "Привет", "Jaki szampon?", "👍", "🙏🙂", "42", ""],
)
def test_a_short_first_message_gets_the_interface_locale(message: str) -> None:
    assert resolve_reply_lang(message, None, Locale.RU) == (Locale.RU, False)
    assert resolve_reply_lang(message, None, Locale.PL) == (Locale.PL, False)


@pytest.mark.parametrize("message", ["hallo", "ok", "thanks", "👍"])
def test_a_short_message_keeps_the_language_of_the_latest_answer(message: str) -> None:
    assert resolve_reply_lang(message, Locale.PL, Locale.RU) == (Locale.PL, False)
    assert resolve_reply_lang(message, Locale.EN, Locale.RU) == (Locale.EN, False)


def test_an_unsure_message_is_treated_like_a_short_one() -> None:
    # Four English words, but lingua has no clear favourite: no guess.
    assert detect_language("How to stop frizz") is None
    assert resolve_reply_lang("How to stop frizz", Locale.UK, Locale.RU) == (Locale.UK, False)


@pytest.mark.parametrize(
    "message",
    [
        "Ich habe Schuppen",
        "Haarausfall, was tun?",
        "Tengo caspa, ¿qué hago?",
        "J'ai des pellicules, que faire ?",
        "Mám lupy, co mám dělat?",
    ],
)
def test_another_language_gets_the_locale_and_the_note(message: str) -> None:
    assert detect_language(message) == OTHER
    # Even mid-conversation: the note says which languages we do answer in.
    assert resolve_reply_lang(message, Locale.EN, Locale.RU) == (Locale.RU, True)


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("Привет! Mam łupież, co robić?", Locale.PL),
        ("Mam łupież, co robić? Спасибо!", Locale.PL),
        ("Wypadają mi włosy po porodzie. Thanks a lot, hi", Locale.PL),
        ("Mam łupież. Ответь, пожалуйста, по-русски", Locale.RU),
    ],
)
def test_a_mixed_message_gets_the_language_of_most_of_its_words(message: str, lang: Locale) -> None:
    assert resolve_reply_lang(message, Locale.EN, Locale.UK) == (lang, False)


def test_a_mixed_message_without_a_clear_majority_falls_back() -> None:
    # German 3 words, Russian 4: neither has MIN_SHARE of the words.
    message = "Ich habe Schuppen. Как часто мыть голову?"

    assert detect_language(message) is None
    assert resolve_reply_lang(message, Locale.PL, Locale.RU) == (Locale.PL, False)


@pytest.mark.parametrize(
    "message",
    [
        QUOTED_ENGLISH,
        "> Dandruff is a common condition of the scalp and face\nЭто правда, что перхоть заразна?",
        "Там пишут «Dandruff is caused by yeast on the scalp», это правда?",
        "Там пишут „Dandruff is caused by yeast on the scalp“, это правда?",
        "Посмотри ```print('wash your hair every single day please')``` что тут?",
    ],
)
def test_quoted_text_and_code_do_not_decide_the_language(message: str) -> None:
    assert resolve_reply_lang(message, Locale.EN, Locale.PL) == (Locale.RU, False)


def test_a_message_that_is_mostly_a_quote_falls_back() -> None:
    message = '"Dandruff is caused by Malassezia yeast overgrowth on the scalp." Да?'

    assert strip_quoted(message).split() == ["Да?"]
    assert resolve_reply_lang(message, None, Locale.UK) == (Locale.UK, False)


def test_the_language_of_a_saved_answer() -> None:
    answer = (
        "Najlepiej myć włosy wtedy, gdy stają się tłuste, nawet codziennie, łagodnym szamponem. "
        * 30
    )

    assert reply_lang_of(answer) is Locale.PL
    assert reply_lang_of("Schuppen sind ein sehr häufiges Problem der Kopfhaut.") is None
    assert reply_lang_of("Ок!") is None


def test_the_directive_lets_the_user_pick_another_of_ours_only() -> None:
    assert directive(ReplyLanguage(Locale.UK)) == (
        "Write the entire reply in Ukrainian unless the user explicitly asks for another of:"
        " Ukrainian, Polish, English, Russian. If the user asks for any other language, still"
        " write in Ukrainian and add one short, polite sentence, in Ukrainian, saying that you"
        " answer in Ukrainian, Polish, English and Russian."
    )


def test_the_directive_adds_the_note_for_a_message_in_another_language() -> None:
    assert directive(ReplyLanguage(Locale.RU, note_about_languages=True)) == (
        "Write the entire reply in Russian unless the user explicitly asks for another of:"
        " Ukrainian, Polish, English, Russian. Add one short, polite sentence, in Russian,"
        " saying that you answer in Ukrainian, Polish, English and Russian."
    )
