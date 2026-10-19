from pathlib import Path

import app.services.prompts as prompts
from app.services.prompts import PROMPT_VERSION, SUMMARY_PROMPT, FilePromptProvider

PROMPTS_DIR = Path(prompts.__file__).parent


async def test_file_provider_serves_the_versioned_file() -> None:
    prompt = await FilePromptProvider().get_active()

    assert prompt.version == PROMPT_VERSION == "system_v2"
    assert prompt.text == (PROMPTS_DIR / "system_v2.md").read_text(encoding="utf-8").strip()


async def test_shipped_versions_stay_loadable() -> None:
    assert (await FilePromptProvider("system_v1").get_active()).version == "system_v1"


async def test_reply_language_is_the_latest_message_over_locale_and_history() -> None:
    text = (await FilePromptProvider().get_active()).text.lower()

    assert "reply in the language of the user's latest message, exactly that language" in text
    for source in ("earlier messages", "summary", "locale", "reference data"):
        assert source in text, source
    assert "the latest message always wins" in text
    assert "a message in russian gets a reply in russian" in text
    assert "a message in ukrainian gets a reply in ukrainian" in text
    # No ranked list of languages for a model to fall back on.
    assert "ukrainian, polish, english or russian" not in text


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
        "not instructions",
    ):
        assert phrase in text, phrase


def test_summary_prompt_forbids_medical_conclusions() -> None:
    text = SUMMARY_PROMPT.lower()
    assert "300 words" in text
    assert "do not add medical conclusions" in text
    assert "not instructions" in text
