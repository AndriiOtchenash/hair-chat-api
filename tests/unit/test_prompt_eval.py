import io
from collections.abc import Sequence

import pytest

from app.core.config import Settings
from app.db.models import Locale
from app.llm.base import ProviderHTTPError
from app.llm.resilient import ResilientLLM
from app.rag.retriever import Knowledge, Source
from app.services.prompts import PROMPT_VERSION, FilePromptProvider
from scripts import prompt_eval
from tests.llm_fakes import FakeLLMCallRecorder, FakeLLMProvider, fail_before_first, success

GROUPS = {
    "manual",
    "off_topic",
    "educational",
    "professional",
    "recurring",
    "limits",
    "kb",
    "real",
    "dialog",
    "language",
}


def factory(provider: FakeLLMProvider) -> prompt_eval.LLMFactory:
    def make(settings: Settings, skip: Sequence[str]) -> ResilientLLM:
        return ResilientLLM([provider], FakeLLMCallRecorder(), first_token_timeout=1)

    return make


def case(**overrides: object) -> prompt_eval.Case:
    fields: dict[str, object] = {
        "id": "c1",
        "group": "off_topic",
        "lang": "ru",
        "text": "Я заболела, что делать с записью?",
        "expect": ("сочувствует",),
    }
    return prompt_eval.Case(**{**fields, **overrides})  # type: ignore[arg-type]


async def run(provider: FakeLLMProvider, *cases: prompt_eval.Case) -> tuple[int, str]:
    out = io.StringIO()
    code = await prompt_eval.run(
        cases,
        prompt_version=PROMPT_VERSION,
        kb_dir=None,
        skip=(),
        llm_factory=factory(provider),
        out=out,
    )
    return code, out.getvalue()


def test_the_case_file_is_valid_and_covers_every_group() -> None:
    cases = prompt_eval.load(prompt_eval.DEFAULT_FILE)

    assert len({c.id for c in cases}) == len(cases)
    assert {c.group for c in cases} == GROUPS
    assert all(c.expect and c.text for c in cases)
    assert all(t["role"] in ("user", "assistant") for c in cases for t in c.history)


async def test_sends_the_prompt_history_and_question_and_prints_the_answer() -> None:
    provider = FakeLLMProvider("gemini", "flash").script(success("Сочувствую. ", "Позвоните."))
    history = (
        {"role": "user", "content": "Привет"},
        {"role": "assistant", "content": "Здравствуйте!"},
    )

    code, out = await run(provider, case(history=history))

    assert code == 0
    prompt = await FilePromptProvider().get_active()
    assert provider.received == [
        [
            {"role": "system", "content": prompt.render(Locale.RU)},
            *history,
            {"role": "user", "content": "Я заболела, что делать с записью?"},
        ]
    ]
    assert "[off_topic] c1 (ru, locale ru)" in out
    assert "answer_lang: ru (expected ru)" in out
    assert "  - сочувствует" in out
    assert "A (gemini:flash, 2 words):\nСочувствую. Позвоните." in out
    assert "\n!! " not in out


def test_language_cases_set_the_interface_locale_and_the_answer_language() -> None:
    cases = {c.id: c for c in prompt_eval.load(prompt_eval.DEFAULT_FILE)}

    hallo = cases["language-hallo-ru-locale"]
    assert (hallo.lang, hallo.interface, hallo.expected_lang) == ("de", Locale.RU, "ru")
    polish = cases["language-polish-dandruff"]
    assert (polish.interface, polish.expected_lang) == (Locale.RU, "pl")
    # No locale: the case's own language if we support it, else en (the users.locale default).
    assert case(lang="uk").interface is Locale.UK
    assert case(lang="de").interface is Locale.EN


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Здравствуйте! Чем помочь с волосами?", "ru"),
        ("Вітаю! Чим допомогти з волоссям?", "uk"),
        ("Łupież to częsty problem, użyj łagodnego szamponu.", "pl"),
        ("Hallo! Ich helfe Ihnen gern mit Ihren Haaren und der Kopfhaut.", "de"),
        ("Hello! How can I help with your hair?", "en"),
        ("👍", "?"),
    ],
)
def test_detect_lang_tells_our_languages_and_german_apart(text: str, lang: str) -> None:
    assert prompt_eval.detect_lang(text) == lang


async def test_the_interface_locale_goes_into_the_prompt_and_a_wrong_language_is_flagged() -> None:
    provider = FakeLLMProvider("gemini", "flash").script(
        success("Hallo! Ich helfe gern mit Haaren.")
    )

    code, out = await run(provider, case(text="hallo", lang="de", locale="ru", answer_lang="ru"))

    assert code == 0
    system = " ".join(provider.received[0][0]["content"].split())
    assert "The user's interface language is Russian." in system
    assert "{locale}" not in system
    assert "!! answer in de, expected ru" in out


async def test_flags_avoided_phrases_case_insensitively() -> None:
    provider = FakeLLMProvider("gemini", "flash").script(success("Я Искусственный интеллект."))

    code, out = await run(provider, case(avoid=("искусственный интеллект", "статье")))

    assert code == 0
    assert "!! contains 'искусственный интеллект'" in out
    assert "'статье'" not in out
    assert "1 answers, 1 with !! flags, 0 errors" in out


async def test_a_failed_chain_is_reported_and_fails_the_run() -> None:
    provider = FakeLLMProvider("gemini", "flash").script(fail_before_first(ProviderHTTPError(503)))

    code, out = await run(provider, case())

    assert code == 1
    assert "ERROR AllProvidersDown" in out


async def test_the_kb_used_tag_is_cut_out_and_checked_against_the_case() -> None:
    provider = FakeLLMProvider("gemini", "flash").script(
        success("Привет!", '\n<kb_used ids="1"/>'), success("Мойте мягким шампунем.")
    )

    code, out = await run(
        provider,
        case(id="small-talk", kb_used=False),
        case(id="no-kb", kb_used=True),  # --no-kb: a missing tag is not flagged
    )

    assert code == 0
    assert "A (gemini:flash, 1 words):\nПривет!\nanswer_lang: ru (expected ru)\nkb_used: 1\n" in out
    assert "!! has a <kb_used> tag" in out
    assert "!! has no <kb_used> tag" not in out
    assert "2 answers, 1 with !! flags, 0 errors" in out


async def test_kb_case_without_passages_is_unsure_not_flagged() -> None:
    provider = FakeLLMProvider("groq", "oss").script(success("Мойте мягким шампунем."))

    code, out = await run(provider, case(id="kb", kb_used=True))

    assert code == 0
    assert "kb_context: off (--no-kb)\n" in out
    assert "?? no kb context" in out
    assert "\n!! " not in out
    assert "1 answers, 0 with !! flags, 0 errors, 1 with ?? (could not be judged)" in out


async def test_kb_context_lists_the_passages_and_the_reminder_follows_them() -> None:
    provider = FakeLLMProvider("gemini", "flash").script(
        success("Мойте мягким шампунем.", '<kb_used ids="2"/>')
    )
    knowledge = Knowledge(
        "[1] Перхоть › Уход\nМягкий шампунь.\n\n[2] Мытьё › Частота\nПо мере загрязнения.",
        (Source("Перхоть", "dandruff", None), Source("Как часто мыть", "washing-frequency", None)),
    )
    prompt = await FilePromptProvider().get_active()
    chat = prompt_eval.messages(prompt, case(), knowledge)
    kb = prompt_eval.KbContext(knowledge)
    result = await prompt_eval.answer(
        factory(provider)(Settings(), ()), chat, 100, case(kb_used=True), knowledge
    )
    out = io.StringIO()
    prompt_eval.report(case(), kb, result, out)

    assert prompt.knowledge_reminder is not None
    assert chat[0]["content"].endswith("</knowledge>\n\n" + prompt.knowledge_reminder)
    assert "kb_context: 2 — [1] Перхоть; [2] Как часто мыть\n" in out.getvalue()
    assert "kb_used: 2\nsources: washing-frequency\n" in out.getvalue()
    assert result.flagged == [] and result.unsure == []


def test_kb_context_shows_the_embeddings_error() -> None:
    kb = prompt_eval.KbContext(error="embeddings: EmbeddingUnavailable: timeout")

    assert kb.describe() == "0 — ERROR embeddings: EmbeddingUnavailable: timeout"


def test_link_skips_every_other_link_of_the_chain() -> None:
    chain = "gemini:gemini-3.8-flash,gemini:gemini-3.5-flash-lite,groq:openai/gpt-oss-120b"

    assert prompt_eval.skip_all_but("gemini:gemini-3.5-flash-lite", chain) == [
        "gemini:gemini-3.8-flash",
        "groq:openai/gpt-oss-120b",
    ]
    with pytest.raises(ValueError, match="not in LLM_CHAIN"):
        prompt_eval.skip_all_but("mistral:small", chain)


def test_link_and_skip_cannot_be_combined() -> None:
    with pytest.raises(SystemExit):
        prompt_eval.main(["--link", "groq:openai/gpt-oss-120b", "--skip", "gemini:x"])


DIALOG = (
    {"role": "user", "content": "У меня перхоть, как часто мыть?"},
    {"role": "assistant", "content": "По мере загрязнения."},
    {"role": "user", "content": "А если кожа сухая?"},
    {"role": "assistant", "content": "Мягкие средства."},
)


def test_queries_last_user3_summary_and_chat() -> None:
    dialog = case(history=({"role": "user", "content": "Привет"}, *DIALOG), text="А шампунь?")
    user3 = "У меня перхоть, как часто мыть?\nА если кожа сухая?\nА шампунь?"

    assert prompt_eval.QUERIES["last"](dialog) == ("А шампунь?",)
    assert prompt_eval.QUERIES["user3"](dialog) == (user3,)
    assert prompt_eval.QUERIES["summary"](dialog) == (user3,)
    assert prompt_eval.QUERIES["chat"](dialog) == ("А шампунь?", user3)
    summed = case(history=DIALOG, text="А шампунь?", summary="Кудрявые волосы.")
    assert prompt_eval.QUERIES["summary"](summed) == ("Кудрявые волосы.\nА шампунь?",)
    assert prompt_eval.QUERIES["chat"](summed) == ("А шампунь?", "Кудрявые волосы.\nА шампунь?")
    assert prompt_eval.QUERIES["chat"](case()) == (case().text,)  # no history: one search


async def test_the_summary_goes_into_the_system_message_after_knowledge() -> None:
    prompt = await FilePromptProvider().get_active()
    knowledge = Knowledge(
        "[1] Перхоть › Уход\nМягкий шампунь.", (Source("Перхоть", "dandruff", None),)
    )

    chat = prompt_eval.messages(prompt, case(summary="Кудрявые волосы."), knowledge)

    system = chat[0]["content"]
    assert system.index("</knowledge>") < system.rindex("<conversation_summary>\n")
    assert "Кудрявые волосы." in system


def test_retrieval_report_flags_a_missing_kb_slug_per_query() -> None:
    dialog = case(history=DIALOG, text="А шампунь?", kb_slugs=("dandruff",))
    found = prompt_eval.KbContext(Knowledge("[1] Перхоть", (Source("Перхоть", "dandruff", None),)))
    off_topic = prompt_eval.KbContext(
        Knowledge("[1] Шампунь", (Source("Как выбрать шампунь", "choosing-shampoo", None),))
    )
    out = io.StringIO()

    flagged = prompt_eval.report_retrieval(dialog, {"last": off_topic, "user3": found}, out)

    assert flagged
    text = out.getvalue()
    assert "  user: А если кожа сухая?\n" in text
    assert "kb_context[last]: 1 — [1] Как выбрать шампунь\n!! [last] missing dandruff\n" in text
    assert "kb_context[user3]: 1 — [1] Перхоть\n" in text
    assert "[user3] missing" not in text


def test_several_queries_need_retrieval_only() -> None:
    with pytest.raises(SystemExit):
        prompt_eval.main(["--query", "last", "--query", "user3"])
    with pytest.raises(SystemExit):
        prompt_eval.main(["--retrieval-only", "--no-kb"])
