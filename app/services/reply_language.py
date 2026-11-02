"""The language of every reply is decided here, not by the model (ADR 0007).

`resolve_reply_lang` picks it from the new message, the language of the assistant's latest reply
and the user's interface locale; the context builder ends the system message with
`directive(...)`, one line that names it. The model never chooses: a prompt rule such as "reply in
the language of the message" let "hallo" get a German answer.

Detection is lingua (local n-gram models, no network, about a millisecond per message) over
`LANGUAGES`: the four we reply in, German, and the other languages our users are likely to write
in, so that a Czech or Spanish message is told apart as "other" instead of being forced onto the
closest of ours (lingua only ever answers with a language it was given). Models load on first use.

What is detected: the message without quoted text ("…", «…», „…“, lines starting with `>`, code),
split into single-language segments; each segment is classified on its own and the language with
most of the words wins, if it has a clear majority (MIN_SHARE). A greeting or a "спасибо" inside a
longer message does not decide anything.
"""

import re
from functools import cache
from typing import Literal, NamedTuple

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

from app.db.models import Locale

# How the directive names each reply language.
LANGUAGE_NAMES: dict[Locale, str] = {
    Locale.UK: "Ukrainian",
    Locale.PL: "Polish",
    Locale.EN: "English",
    Locale.RU: "Russian",
}
SUPPORTED: dict[Language, Locale] = {
    Language.UKRAINIAN: Locale.UK,
    Language.POLISH: Locale.PL,
    Language.ENGLISH: Locale.EN,
    Language.RUSSIAN: Locale.RU,
}
# Detected only to be recognised as "other". Cyrillic and Slavic neighbours keep uk/ru/pl honest.
OTHER_LANGUAGES = (
    Language.GERMAN,
    Language.FRENCH,
    Language.SPANISH,
    Language.ITALIAN,
    Language.PORTUGUESE,
    Language.DUTCH,
    Language.CZECH,
    Language.SLOVAK,
    Language.BELARUSIAN,
    Language.BULGARIAN,
    Language.TURKISH,
)
LANGUAGES = (*SUPPORTED, *OTHER_LANGUAGES)

OTHER: Literal["other"] = "other"
Detected = Locale | Literal["other"] | None

# Fewer words than this (quotes left out) is "short": a greeting, "ok", "thanks" cannot be told
# apart reliably ("hallo" is German, Dutch and nearly English), two words rarely better.
MIN_WORDS = 3
# Share of the words the winning language needs in a mixed message; less = unsure.
MIN_SHARE = 0.6
# lingua's confidences are relative over LANGUAGES (English often tops at 0.2-0.4), so "sure"
# means a clear lead over the runner-up, not a high value: "Це нормально?" is uk 0.47 / ru 0.50.
MIN_LEAD = 0.1

ANSWER_SAMPLE_CHARS = 1000

_WORD = re.compile(r"[^\W\d_]+")
# Quoted text says nothing about the user's own language: an English article quoted in a Russian
# question is still a Russian question.
_QUOTED = (
    re.compile(r"```.*?(?:```|$)", re.DOTALL),  # code blocks, an unclosed one to the end
    re.compile(r"`[^`\n]*`"),
    re.compile(r"^[ \t]*>.*$", re.MULTILINE),  # quoted lines
    re.compile(r'"[^"\n]*"'),
    re.compile(r"«[^»]*»"),
    re.compile(r"„[^“”]*[“”]"),
    re.compile(r"“[^”]*”"),
)


class ReplyLanguage(NamedTuple):
    lang: Locale
    # The message was in a language we do not support: say once which ones we do.
    note_about_languages: bool = False


@cache
def _detector() -> LanguageDetector:
    return LanguageDetectorBuilder.from_languages(*LANGUAGES).build()


def top_language(text: str) -> tuple[str, float, float] | None:
    """Best guess as (ISO 639-1 code, confidence, lead over the runner-up); None without letters."""
    if not _WORD.search(text):
        return None
    first, second = _detector().compute_language_confidence_values(text)[:2]
    code = first.language.iso_code_639_1.name.lower()
    return code, first.value, first.value - second.value


def strip_quoted(text: str) -> str:
    """The text without quotes, quoted lines and code (see _QUOTED)."""
    for pattern in _QUOTED:
        text = pattern.sub(" ", text)
    return text


def _segments(text: str) -> list[str]:
    found = _detector().detect_multiple_languages_of(text)
    return [text[r.start_index : r.end_index] for r in found] or [text]


def detect_language(text: str) -> Detected:
    """uk/pl/en/ru when sure, OTHER for another language, None when short or unsure.

    lingua's own segment labels are weaker than its confidence values (it called a whole Russian
    sentence Ukrainian), so the segments are only the split: each is classified again."""
    own = strip_quoted(text)
    if len(_WORD.findall(own)) < MIN_WORDS:
        return None
    words: dict[str | None, int] = {}
    for segment in _segments(own):
        guess = top_language(segment)
        lang = guess[0] if guess and guess[2] >= MIN_LEAD else None  # None = unsure
        words[lang] = words.get(lang, 0) + len(_WORD.findall(segment))
    lang, count = max(words.items(), key=lambda item: item[1])
    if lang is None or count < MIN_SHARE * sum(words.values()):
        return None
    return Locale(lang) if lang in set(Locale) else OTHER


def resolve_reply_lang(
    message: str, last_assistant_lang: Locale | None, locale: Locale
) -> ReplyLanguage:
    """Sure uk/pl/en/ru -> that; short/unsure -> the assistant's latest reply language, else the
    interface locale; any other language -> the locale, with the note about languages."""
    detected = detect_language(message)
    if detected is None:
        return ReplyLanguage(last_assistant_lang or locale)
    if detected == OTHER:
        return ReplyLanguage(locale, note_about_languages=True)
    return ReplyLanguage(detected)


def reply_lang_of(answer: str) -> Locale | None:
    """The language of a saved assistant reply, if it is one we reply in (from its start: a long
    answer costs milliseconds and says nothing more)."""
    detected = detect_language(answer[:ANSWER_SAMPLE_CHARS])
    return detected if isinstance(detected, Locale) else None


SUPPORTED_NAMES = "Ukrainian, Polish, English, Russian"


def directive(reply: ReplyLanguage) -> str:
    """The last line of the system message (system_v7 points the model at it).

    The user may still ask for another of our languages; a request for any other language
    gets our language and the note, like a message written in one."""
    name = LANGUAGE_NAMES[reply.lang]
    text = (
        f"Write the entire reply in {name} unless the user explicitly asks for another of:"
        f" {SUPPORTED_NAMES}."
    )
    note = (
        f"one short, polite sentence, in {name}, saying that you answer in Ukrainian, Polish,"
        " English and Russian"
    )
    if reply.note_about_languages:
        return f"{text} Add {note}."
    return f"{text} If the user asks for any other language, still write in {name} and add {note}."
