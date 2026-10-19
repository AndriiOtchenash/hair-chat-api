"""What the model sees for a reply. The full builder (profile, RAG, token budget) is item 1.7;
it replaces RecentMessagesContextBuilder behind the same ContextBuilder protocol."""

import uuid
from typing import Protocol

from app.db.models import MessageRole
from app.db.repositories.conversations import ConversationRepository
from app.llm.base import ChatMessage
from app.services.prompts import SYSTEM_PROMPT

HISTORY_MESSAGES = 20


class ContextBuilder(Protocol):
    async def build(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, max_context_tokens: int
    ) -> list[ChatMessage]:
        """Full context, rebuilt from the DB on every call (rule 2), ending with the user turn."""
        ...


class RecentMessagesContextBuilder:
    """Placeholder: system prompt + the last N messages. Ignores the token budget (1.7 will not)."""

    def __init__(self, repo: ConversationRepository, history: int = HISTORY_MESSAGES) -> None:
        self._repo = repo
        self._history = history

    async def build(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, max_context_tokens: int
    ) -> list[ChatMessage]:
        messages: list[ChatMessage] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for message in await self._repo.recent_messages(user_id, conversation_id, self._history):
            if message.role is MessageRole.USER:
                messages.append({"role": "user", "content": message.content})
            elif message.role is MessageRole.ASSISTANT:
                messages.append({"role": "assistant", "content": message.content})
        return messages
