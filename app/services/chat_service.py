"""Chat use cases: conversations, messages and one streamed reply per user message.

A reply is two steps so HTTP can answer 404/409/422 before the stream starts:
`async with service.reply(...)` checks the conversation, takes its generation slot, moderates
and saves the user message; `reply.events()` then streams the answer. Only a complete answer
(Done) is saved: a client that disconnects cancels the stream and leaves no partial assistant
message behind.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import ExitStack, aclosing, asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from typing import Any

from app.db.models import Conversation, Message
from app.db.pagination import Cursor, Page
from app.db.repositories.conversations import ConversationRepository
from app.llm.base import AllProvidersDown, Done, LLMRequestError, Restart, Token
from app.llm.moderation import ModerationUnavailable, Moderator
from app.llm.resilient import ResilientLLM
from app.rag.retriever import NoQueryEmbedder, QueryEmbedder, QueryEmbedding, Source
from app.services.context_builder import Context, ContextBuilder
from app.services.kb_used import KbUsedFilter, used_sources
from app.services.moderation_policy import ModerationAction, decide

logger = logging.getLogger(__name__)


class ConversationNotFound(Exception):
    """Missing or someone else's: callers answer both the same way (404)."""


class ConversationBusy(Exception):
    """An answer is already being generated in this conversation."""


class MessageBlocked(Exception):
    """Moderation blocked the message (API code message_blocked); it is not saved."""

    def __init__(self, categories: tuple[str, ...]) -> None:
        super().__init__("message_blocked")
        self.categories = categories


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
    # Knowledge-base articles the answer says it used (its <kb_used> tag), see kb_used.py.
    sources: tuple[Source, ...] = ()


@dataclass(frozen=True, slots=True)
class ChatError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SafetyNotice:
    """Sent before the first token: the client shows a card with help lines for `kind`."""

    kind: str  # "self_harm"


SELF_HARM_NOTICE = SafetyNotice("self_harm")

ChatEvent = SafetyNotice | Token | Restart | ChatDone | ChatError


class Reply:
    def __init__(
        self,
        service: "ChatService",
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_message_id: uuid.UUID,
        safety_notice: SafetyNotice | None = None,
        query: QueryEmbedding | None = None,
    ) -> None:
        self._service = service
        self._query = query
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.user_message_id = user_message_id
        self.safety_notice = safety_notice
        self._events: AsyncGenerator[ChatEvent] | None = None
        # Set with ChatDone: the context left messages out, so the caller should run the
        # summarizer once the answer is delivered (never when the reply failed).
        self.summary_due = False

    def events(self) -> AsyncIterator[ChatEvent]:
        """SafetyNotice? Token* (Restart Token*)* then ChatDone, or ChatError.

        Never raises for LLM failures.

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
        if self.safety_notice is not None:
            yield self.safety_notice
        try:
            context = await service.context_builder.build(
                self.user_id, self.conversation_id, service.budget_tokens, self._query
            )
            answer: list[str] = []
            # Cuts the <kb_used> tag out of the stream: neither the client nor the DB sees it.
            tag = KbUsedFilter()
            stream = service.llm.stream(context.messages, service.max_answer_tokens)
            # aclosing: leaving `async for` early does not close the inner generator by itself.
            async with aclosing(stream) as events:
                async for event in events:
                    match event:
                        case Token(text=text):
                            if shown := tag.feed(text):
                                answer.append(shown)
                                yield replace(event, text=shown)
                        case Restart():
                            answer.clear()
                            tag = KbUsedFilter()
                            yield event
                        case Done():
                            if rest := tag.finish():
                                answer.append(rest)
                                yield Token(rest, event.provider, event.model)
                            sources = used_sources(tag.ids, context.passages)
                            done = await self._save(event, "".join(answer), context, sources)
                            self.summary_due = context.needs_summary
                            yield done
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

    async def _save(
        self, done: Done, answer: str, context: Context, sources: tuple[Source, ...]
    ) -> ChatDone:
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
            "chat reply saved: conversation=%s message=%s provider=%s model=%s prompt=%s "
            "context_tokens=%d tokens_in=%s tokens_out=%s truncated=%s passages=%d sources=%d",
            self.conversation_id,
            message.id,
            done.provider,
            done.model,
            context.prompt_version,
            context.tokens,
            done.tokens_in,
            done.tokens_out,
            done.truncated,
            len(context.passages),
            len(sources),
        )
        return ChatDone(message.id, done.provider, done.model, done.truncated, sources=sources)


class ConversationService:
    """Reads and plain writes; needs no LLM, so listing works even when the chain is down."""

    def __init__(self, repo: ConversationRepository, locks: ConversationLocks) -> None:
        self.repo = repo
        # The chat's generation slots: nothing is deleted while an answer is being generated.
        self.locks = locks

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

    async def rename(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, title: str
    ) -> Conversation:
        """A title set here is kept: the first message only titles an untitled conversation."""
        conversation = await self.repo.rename(user_id, conversation_id, title)
        if conversation is None:
            raise ConversationNotFound()
        return conversation

    async def delete(self, user_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        """Raises ConversationNotFound, then ConversationBusy while an answer is generated."""
        # Ownership first: a busy conversation of someone else is a 404, not a 409.
        await self.require(user_id, conversation_id)
        with self.locks.hold(conversation_id):
            if not await self.repo.delete(user_id, [conversation_id]):
                raise ConversationNotFound()
        logger.info("conversation deleted: conversation=%s", conversation_id)

    async def delete_all(self, user_id: uuid.UUID) -> int:
        """All or nothing: ConversationBusy if any of them is generating an answer."""
        conversation_ids = await self.repo.ids(user_id)
        with ExitStack() as held:
            # No await while taking the slots: all of them or none, atomically.
            for conversation_id in conversation_ids:
                held.enter_context(self.locks.hold(conversation_id))
            deleted = await self.repo.delete(user_id, conversation_ids)
        logger.info("conversations deleted: user=%s count=%d", user_id, deleted)
        return deleted


class ChatService:
    def __init__(
        self,
        conversations: ConversationService,
        llm: ResilientLLM,
        context_builder: ContextBuilder,
        locks: ConversationLocks,
        moderator: Moderator,
        *,
        max_answer_tokens: int,
        budget_tokens: int,
        violence_threshold: float = 0.9,
        query_embedder: QueryEmbedder | None = None,
    ) -> None:
        self.conversations = conversations
        self.repo = conversations.repo
        self.llm = llm
        self.context_builder = context_builder
        self.locks = locks
        self.moderator = moderator
        self.violence_threshold = violence_threshold
        self.query_embedder = query_embedder or NoQueryEmbedder()
        self.max_answer_tokens = max_answer_tokens
        # Input tokens per reply: app.services.tokens.context_budget.
        self.budget_tokens = budget_tokens

    @asynccontextmanager
    async def reply(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, content: str
    ) -> AsyncIterator[Reply]:
        """Raises ConversationNotFound / ConversationBusy / MessageBlocked before anything is saved.

        The slot is held until the block exits, i.e. until the stream ends or is cancelled.
        Moderation runs after the slot is taken, so a busy conversation costs no moderation call.
        The question is embedded for the knowledge base at the same time as it is moderated.
        """
        await self.conversations.require(user_id, conversation_id)
        with self.locks.hold(conversation_id):
            embedding = asyncio.create_task(self.query_embedder.embed_query(content))
            try:
                moderation = await self._moderate(content)
            except BaseException:
                embedding.cancel()  # blocked (or cancelled): its vector is not needed
                raise
            query = await embedding
            message = await self.repo.add_user_message(
                user_id, conversation_id, content, moderation=moderation
            )
            # The context builder reads the same "soft" mark from the saved message.
            soft = moderation.get("action") == ModerationAction.SOFT
            notice = SELF_HARM_NOTICE if soft else None
            reply = Reply(self, user_id, conversation_id, message.id, notice, query)
            try:
                yield reply
            finally:
                await reply.aclose()

    async def _moderate(self, content: str) -> dict[str, Any]:
        """Raises MessageBlocked; otherwise returns what to store in messages.moderation.

        Policy in app.services.moderation_policy. Fail-open (ADR 0004): if moderation is down
        the message goes through, marked "unavailable". Logs carry the action and categories
        only: no text, no scores, no user id (rule 6).
        """
        if not self.moderator.enabled:
            return {"status": "disabled"}
        try:
            result = await self.moderator.check(content)
        except ModerationUnavailable as exc:
            log = logger.error if exc.misconfigured else logger.warning
            log("moderation unavailable, message let through: code=%s", exc.code)
            return {"status": "unavailable"}
        decision = decide(result, violence_threshold=self.violence_threshold)
        if decision.categories:
            logger.info(
                "moderation flagged: action=%s categories=%s",
                decision.action,
                ",".join(decision.categories),
            )
        if decision.action is ModerationAction.BLOCK:
            raise MessageBlocked(decision.categories)
        stored: dict[str, Any] = {
            "provider": result.provider,
            "model": result.model,
            "flagged": result.flagged,
            "action": decision.action.value,
        }
        if decision.categories:
            stored["categories"] = list(decision.categories)
        return stored
