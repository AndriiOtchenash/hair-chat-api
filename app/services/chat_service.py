"""Chat use cases: conversations, messages and one streamed reply per user message.

A reply is two steps so HTTP can answer 404/409 before the stream starts:
`async with service.reply(...)` checks the conversation, takes its generation slot and saves the
user message; `reply.events()` then streams the answer. Only a complete answer (Done) is saved:
a client that disconnects cancels the stream and leaves no partial assistant message behind.
"""

import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import aclosing, asynccontextmanager, contextmanager
from dataclasses import dataclass

from app.db.models import Conversation, Message
from app.db.pagination import Cursor, Page
from app.db.repositories.conversations import ConversationRepository
from app.llm.base import AllProvidersDown, Done, LLMRequestError, Restart, Token
from app.llm.resilient import ResilientLLM
from app.services.context_builder import ContextBuilder

logger = logging.getLogger(__name__)


class ConversationNotFound(Exception):
    """Missing or someone else's: callers answer both the same way (404)."""


class ConversationBusy(Exception):
    """An answer is already being generated in this conversation."""


class ConversationLocks:
    """One generation per conversation, in this process only.

    With several app instances a second request can land on another instance; that needs a
    shared lock (e.g. a Postgres advisory lock), see docs/adr/0003-chat-streaming.md.
    """

    def __init__(self) -> None:
        self._busy: set[uuid.UUID] = set()

    def is_busy(self, conversation_id: uuid.UUID) -> bool:
        return conversation_id in self._busy

    @contextmanager
    def hold(self, conversation_id: uuid.UUID) -> Iterator[None]:
        # No await between the check and the add: atomic within one event loop.
        if conversation_id in self._busy:
            raise ConversationBusy()
        self._busy.add(conversation_id)
        try:
            yield
        finally:
            self._busy.discard(conversation_id)


@dataclass(frozen=True, slots=True)
class ChatDone:
    message_id: uuid.UUID
    provider: str
    model: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class ChatError:
    code: str
    message: str


ChatEvent = Token | Restart | ChatDone | ChatError


class Reply:
    def __init__(
        self,
        service: "ChatService",
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_message_id: uuid.UUID,
    ) -> None:
        self._service = service
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.user_message_id = user_message_id
        self._events: AsyncGenerator[ChatEvent] | None = None

    def events(self) -> AsyncIterator[ChatEvent]:
        """Token* (Restart Token*)* then ChatDone, or ChatError. Never raises for LLM failures.

        Single use. Closed by `ChatService.reply` on exit, see `aclose`.
        """
        if self._events is not None:
            raise RuntimeError("Reply.events() can be consumed only once")
        self._events = self._generate()
        return self._events

    async def aclose(self) -> None:
        """Close the stream now rather than whenever the GC finalizes it.

        A consumer cancelled while the generator is parked at `yield` (e.g. the client stopped
        reading) leaves it suspended; closing it runs its `finally` blocks down to the
        provider's HTTP stream, so a disconnect releases the upstream connection immediately.
        """
        if self._events is None:
            return
        try:
            await self._events.aclose()
        except RuntimeError:  # still running in another task; it will finish on its own
            logger.warning("chat reply stream still running: conversation=%s", self.conversation_id)

    async def _generate(self) -> AsyncGenerator[ChatEvent]:
        service = self._service
        try:
            context = await service.context_builder.build(
                self.user_id, self.conversation_id, service.llm.max_context_tokens
            )
            answer: list[str] = []
            # aclosing: leaving `async for` early does not close the inner generator by itself.
            async with aclosing(service.llm.stream(context, service.max_answer_tokens)) as events:
                async for event in events:
                    match event:
                        case Token(text=text):
                            answer.append(text)
                            yield event
                        case Restart():
                            answer.clear()
                            yield event
                        case Done():
                            yield await self._save(event, "".join(answer))
        except AllProvidersDown:
            logger.warning(
                "chat reply failed: conversation=%s code=llm_unavailable", self.conversation_id
            )
            yield ChatError(
                "llm_unavailable", "The assistant is temporarily unavailable, please try again"
            )
        except LLMRequestError as exc:
            logger.error(
                "chat reply rejected: conversation=%s code=%s", self.conversation_id, exc.code
            )
            yield ChatError("llm_request_rejected", "The assistant could not process this request")
        except Exception as exc:
            # Class name only: DB errors may carry the message text in their parameters (rule 6).
            logger.error(
                "chat reply failed: conversation=%s error=%s",
                self.conversation_id,
                type(exc).__name__,
            )
            yield ChatError("internal_error", "Something went wrong, please try again")

    async def _save(self, done: Done, answer: str) -> ChatDone:
        message = await self._service.repo.add_assistant_message(
            self.user_id,
            self.conversation_id,
            answer,
            provider=done.provider,
            model=done.model,
            tokens_in=done.tokens_in,
            tokens_out=done.tokens_out,
            llm_call_ids=done.call_ids,
        )
        logger.info(
            "chat reply saved: conversation=%s message=%s provider=%s model=%s "
            "tokens_in=%s tokens_out=%s truncated=%s",
            self.conversation_id,
            message.id,
            done.provider,
            done.model,
            done.tokens_in,
            done.tokens_out,
            done.truncated,
        )
        return ChatDone(message.id, done.provider, done.model, done.truncated)


class ConversationService:
    """Reads and plain writes; needs no LLM, so listing works even when the chain is down."""

    def __init__(self, repo: ConversationRepository) -> None:
        self.repo = repo

    async def create(self, user_id: uuid.UUID) -> Conversation:
        return await self.repo.create(user_id)

    async def list(
        self, user_id: uuid.UUID, limit: int, cursor: Cursor | None
    ) -> Page[Conversation]:
        return await self.repo.list_page(user_id, limit, cursor)

    async def list_messages(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, limit: int, cursor: Cursor | None
    ) -> Page[Message]:
        await self.require(user_id, conversation_id)
        return await self.repo.list_messages_page(user_id, conversation_id, limit, cursor)

    async def require(self, user_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        if await self.repo.get(user_id, conversation_id) is None:
            raise ConversationNotFound()


class ChatService:
    def __init__(
        self,
        conversations: ConversationService,
        llm: ResilientLLM,
        context_builder: ContextBuilder,
        locks: ConversationLocks,
        *,
        max_answer_tokens: int,
    ) -> None:
        self.conversations = conversations
        self.repo = conversations.repo
        self.llm = llm
        self.context_builder = context_builder
        self.locks = locks
        self.max_answer_tokens = max_answer_tokens

    @asynccontextmanager
    async def reply(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, content: str
    ) -> AsyncIterator[Reply]:
        """Raises ConversationNotFound / ConversationBusy before anything is saved.

        The slot is held until the block exits, i.e. until the stream ends or is cancelled.
        """
        await self.conversations.require(user_id, conversation_id)
        with self.locks.hold(conversation_id):
            message = await self.repo.add_user_message(user_id, conversation_id, content)
            reply = Reply(self, user_id, conversation_id, message.id)
            try:
                yield reply
            finally:
                await reply.aclose()
