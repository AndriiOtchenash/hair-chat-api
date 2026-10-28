"""Prompts are versioned files: a change is a new `<name>_vN.md`, never an edit of a shipped one.

The system prompt is loaded through `PromptProvider`, so stage 2.7 can swap the file for versions
edited by admins/specialists in the DB (DbPromptProvider) without touching the context builder.
The summary prompt is internal and stays a file constant.

A system prompt file may end with a part after the `KNOWLEDGE_REMINDER_MARKER` line: the context
builder puts it right after the <knowledge> block (only when there is one), so the rule about
the passages is the last thing the model reads before the conversation.
"""

from dataclasses import dataclass
from importlib.resources import files
from typing import Protocol

PROMPT_VERSION = "system_v6"
SUMMARY_PROMPT_VERSION = "summary_v1"
KNOWLEDGE_REMINDER_MARKER = "<!-- after knowledge -->"


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
