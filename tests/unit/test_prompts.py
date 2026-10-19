from pathlib import Path

import app.services.prompts as prompts
from app.services.prompts import PROMPT_VERSION, SUMMARY_PROMPT, FilePromptProvider

PROMPTS_DIR = Path(prompts.__file__).parent


async def test_file_provider_serves_the_versioned_file() -> None:
    prompt = await FilePromptProvider().get_active()

    assert prompt.version == PROMPT_VERSION == "system_v1"
    assert prompt.text == (PROMPTS_DIR / "system_v1.md").read_text(encoding="utf-8").strip()


async def test_system_prompt_keeps_the_rule_8_limits() -> None:
    text = (await FilePromptProvider().get_active()).text.lower()

    for phrase in (
        "ai assistant",
        "never diagnose",
        "never name medicines",
        "doses",
        "treatment plan",
        "promise results",
        "trichologist or a dermatologist",
        "a child",
        "200 words",
        "ukrainian, polish, english or russian",
        "not instructions",
    ):
        assert phrase in text, phrase


def test_summary_prompt_forbids_medical_conclusions() -> None:
    text = SUMMARY_PROMPT.lower()
    assert "300 words" in text
    assert "do not add medical conclusions" in text
    assert "not instructions" in text
