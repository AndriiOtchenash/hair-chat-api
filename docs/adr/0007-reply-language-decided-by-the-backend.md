# 0007 — The reply language is decided by the backend, not the model

Date: 2026-11-02 · Status: accepted

## Context
We answer in Ukrainian, Polish, English and Russian only. Up to system_v6 the prompt told the
model to "reply in the language of the user's latest message", and the model never saw the
user's interface locale. "hallo" with the Russian interface got a German answer.

A first fix (system_v7 drafts) put the locale into the prompt with conditional rules: a clear
uk/pl/en/ru message -> that language, a short one -> the conversation's or the interface
language, anything else -> the interface language plus a note. On the eval set the rules moved
the problem around instead of solving it: flash-lite answered "ok" after a Polish dialog in
Russian (Russian knowledge-base passages pulled it), a wording fix for that made both links
answer "thanks" after an English dialog in Russian, and "Ich habe Schuppen" got a Russian
answer that opened in German. Which language is "clear" is a classification
task; a small, deterministic classifier does it better and can be unit-tested.

## Decision
- `app.services.reply_language.resolve_reply_lang(message, last_assistant_lang, locale)` returns
  `(lang, note_about_languages)`:
  - message surely in uk/pl/en/ru -> that language;
  - short (< 3 words once quotes are left out) or unsure -> the language of the assistant's
    latest reply (detected from its text), else the interface locale (`users.locale`);
  - another language -> the locale, with a one-sentence note on the languages we support.
- Detection: quotes ("…", «…», „…“, `>` lines, code) are cut out first; the rest is split into
  single-language segments (lingua `detect_multiple_languages_of`), each segment is classified
  again with lingua's confidence values (its segment labels are weaker), and the language with
  >= 60 % of the words wins if its lead over the runner-up is >= 0.1. Greetings or a "спасибо"
  inside a longer message do not decide.
- The context builder ends the system message with one directive, after everything else
  (after the `<knowledge>` reminder and the summary): "Write the entire reply in {Lang} unless
  the user explicitly asks for another of: Ukrainian, Polish, English, Russian", plus either the
  note or the rule for a request in any other language (our language plus the note). The prompt
  (system_v7) has no language rules of its own beyond pointing at that line.
- The language is not stored: the last answer's language is detected again from its text
  (first 1000 characters, a few ms).

## Library: lingua-language-detector 2.2
Compared on our phrases (23 full sentences, 34 two-to-five-word phrases, uk/pl/en/ru/de and
es/fr/it/cs/nl/pt/sk as "other"):

| | wrong on short phrases | "other" detected | prod venv | RSS |
|---|---|---|---|---|
| lingua, 15 languages | 1 / 34 (uk/ru tie, now "unsure") | yes | +292 MB | ~15 MB |
| fast-langdetect (fastText lid.176 lite) | 7 / 34 ("What shampoo?" -> it, "Jaki szampon?" -> hu, "Який шампунь?" -> ru) | yes | +5 MB | ~35 MB |
| py3langid | worse on short text; "Привет" -> uk | only if given | +61 MB (numpy) | ~130 MB |
| langdetect | non-deterministic without a seed, weakest on short text | yes | +2.5 MB | — |

lingua is chosen for accuracy on short and mixed texts, the reason this ADR exists. Its cost is
size: the wheel is one 292 MB native library with the models of all 75 languages compiled in
(nothing to strip), so the prod virtualenv grows from 105 MB to 397 MB. Only the 15 languages
we build the detector with are loaded into memory, lazily. Fly's image limit and our deploy
time can take it; if image size ever matters more, fastText's full lid.176.bin (126 MB, would
have to be baked into the image) is the next candidate. No network at run time.

lingua only answers with a language it was given, so "other" needs other languages to win:
with uk/pl/en/ru/de alone, Czech came out as English (0.96). The detector knows German, French,
Spanish, Italian, Portuguese, Dutch, Czech, Slovak, Belarusian, Bulgarian and Turkish besides
our four; a message in a language outside all 15 lands on the closest one, usually as "other"
or "unsure", both of which give the interface language.

## Consequences
- The model can no longer drift into the user's language: wrong-language answers now come from
  the classifier, which is unit-tested (tests/unit/test_reply_language.py), not from the prompt.
- "Mid-conversation 'ok'" keeps the conversation's language; the first message of a
  conversation falls back to the interface language.
- The prompt eval prints the language the backend picked (`reply_lang`) and checks the answer's
  first sentence on its own, since a majority check missed an answer opening in German.
- Detection is CPU work in the request path: ~1 ms per message, ~7 ms for a long answer sample.
