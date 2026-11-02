"""Prompts are versioned files: a change is a new `<name>_vN.md`, never an edit of a shipped one.

The system prompt is loaded through `PromptProvider`, so stage 2.7 can swap the file for versions
edited by admins/specialists in the DB (DbPromptProvider) without touching the context builder.
The summary prompt is internal and stays a file constant.

A system prompt file may end with a part after the `KNOWLEDGE_REMINDER_MARKER` line: the context
builder puts it right after the <knowledge> block (only when there is one), so the rule about
the passages is the last thing the model reads before the conversation.

From system_v7 on, the prompt names the interface language through the `LOCALE_PLACEHOLDER`
token, filled in per reply from users.locale (`SystemPrompt.render`); the model falls back to it
for short, ambiguous or unsupported-language messages. Older prompts have no placeholder.
"""

from dataclasses import dataclass
from importlib.resources import files
from typing import Protocol

from app.db.models import Locale

PROMPT_VERSION = "system_v7"
SUMMARY_PROMPT_VERSION = "summary_v1"
KNOWLEDGE_REMINDER_MARKER = "<!-- after knowledge -->"
LOCALE_PLACEHOLDER = "{locale}"
# How the prompt names each interface language (English, like the rest of the prompt).
LOCALE_LANGUAGES: dict[Locale, str] = {
    Locale.UK: "Ukrainian",
    Locale.PL: "Polish",
    Locale.EN: "English",
    Locale.RU: "Russian",
}


def _read(version: str) -> str:
    return files(__package__).joinpath(f"{version}.md").read_text(encoding="utf-8").strip()


SUMMARY_PROMPT = _read(SUMMARY_PROMPT_VERSION)


@dataclass(frozen=True, slots=True)
class SystemPrompt:
    text: str
    # Logged with every answer, so a reply can be traced to the prompt that produced it.
    version: str
    # Goes right after the <knowledge> block, see the module docstring.
    knowledge_reminder: str | None = None

    def render(self, locale: Locale) -> str:
        """The prompt text with the user's interface language in place of `LOCALE_PLACEHOLDER`."""
        return self.text.replace(LOCALE_PLACEHOLDER, LOCALE_LANGUAGES[locale])


class PromptProvider(Protocol):
    async def get_active(self) -> SystemPrompt:
        """The system prompt to use for the next reply (CLAUDE.md rule 8)."""
        ...


class FilePromptProvider:
    """The prompt shipped with the code: app/services/prompts/<version>.md, read once."""

    def __init__(self, version: str = PROMPT_VERSION) -> None:
        text, _, reminder = _read(version).partition(KNOWLEDGE_REMINDER_MARKER)
        self._prompt = SystemPrompt(text.strip(), version, reminder.strip() or None)

    async def get_active(self) -> SystemPrompt:
        return self._prompt
