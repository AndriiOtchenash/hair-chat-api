import io
from collections.abc import Sequence

from app.core.config import Settings
from app.llm.base import ProviderHTTPError
from app.llm.resilient import ResilientLLM
from app.services.prompts import PROMPT_VERSION, FilePromptProvider
from scripts import prompt_eval
from tests.llm_fakes import FakeLLMCallRecorder, FakeLLMProvider, fail_before_first, success

GROUPS = {"manual", "off_topic", "educational", "professional", "recurring", "limits", "kb"}


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
            {"role": "system", "content": prompt.text},
            *history,
            {"role": "user", "content": "Я заболела, что делать с записью?"},
        ]
    ]
    assert "[off_topic] c1 (ru)" in out
    assert "  - сочувствует" in out
    assert "A (gemini:flash, 2 words):\nСочувствую. Позвоните." in out
    assert "\n!! " not in out


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
    assert "A (gemini:flash, 1 words):\nПривет!\nkb_used: 1\n" in out
    assert "!! has a <kb_used> tag" in out
    assert "!! has no <kb_used> tag" not in out
    assert "2 answers, 1 with !! flags, 0 errors" in out
