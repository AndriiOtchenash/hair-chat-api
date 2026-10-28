import asyncio
import logging
import uuid
from collections.abc import Sequence

import pytest

from app.db.models import MessageRole
from app.db.repositories.kb import KBHit
from app.llm.base import ProviderHTTPError, ProviderTimeout, Restart, StreamBroken, Token
from app.llm.moderation import ModerationResult, ModerationUnavailable, Moderator, NoopModerator
from app.llm.resilient import ResilientLLM
from app.rag.retriever import QueryEmbedder, RagKnowledge, RagQueryEmbedder, Source
from app.services.chat_service import (
    ChatDone,
    ChatError,
    ChatEvent,
    ChatService,
    ConversationBusy,
    ConversationLocks,
    ConversationNotFound,
    ConversationService,
    MessageBlocked,
    SafetyNotice,
)
from app.services.context_builder import (
    KNOWLEDGE_NOTE,
    KnowledgeSource,
    TokenBudgetContextBuilder,
)
from app.services.prompts import FilePromptProvider, PromptProvider
from tests.fakes import FakeConversationRepository, StaticPrompt
from tests.llm_fakes import (
    FakeEmbedder,
    FakeLLMCallRecorder,
    FakeLLMProvider,
    FakeModerator,
    Scenario,
    fail_before_first,
    fail_mid_stream,
    flag,
    hang,
    success,
    truncated,
)

USER = uuid.UUID("01890000-0000-7000-8000-000000000001")
OTHER = uuid.UUID("01890000-0000-7000-8000-000000000002")


class World:
    """ChatService over in-memory fakes; providers are scripted per test."""

    def __init__(
        self,
        *providers: FakeLLMProvider,
        budget: int = 6000,
        prompts: PromptProvider | None = None,
        moderator: Moderator | None = None,
        query_embedder: QueryEmbedder | None = None,
        knowledge: KnowledgeSource | None = None,
    ) -> None:
        self.repo = FakeConversationRepository()
        self.recorder = FakeLLMCallRecorder()
        self.providers = providers or (FakeLLMProvider("gemini", "flash"),)
        self.locks = ConversationLocks()
        self.moderator = moderator or FakeModerator()
        self.chat = ChatService(
            ConversationService(self.repo, self.locks),
            ResilientLLM(self.providers, self.recorder, first_token_timeout=0.05),
            TokenBudgetContextBuilder(
                self.repo, prompts or FilePromptProvider(), knowledge=knowledge
            ),
            self.locks,
            self.moderator,
            max_answer_tokens=100,
            budget_tokens=budget,
            query_embedder=query_embedder,
        )

    async def conversation(self, user: uuid.UUID = USER) -> uuid.UUID:
        return (await self.repo.create(user)).id

    async def send(
        self, conversation_id: uuid.UUID, content: str = "How often to wash?"
    ) -> list[ChatEvent]:
        async with self.chat.reply(USER, conversation_id, content) as reply:
            return [event async for event in reply.events()]


def roles_and_texts(world: World, conversation_id: uuid.UUID) -> list[tuple[MessageRole, str]]:
    return [(m.role, m.content) for m in world.repo.of(conversation_id)]


async def test_success_saves_both_messages_and_ends_with_done() -> None:
    world = World(FakeLLMProvider("gemini", "flash").script(success("Twice ", "a week")))
    conversation_id = await world.conversation()

    events = await world.send(conversation_id, "How often should I wash oily hair?")

    assert events[:2] == [Token("Twice ", "gemini", "flash"), Token("a week", "gemini", "flash")]
    done = events[-1]
    assert isinstance(done, ChatDone)
    assert (done.provider, done.model, done.truncated) == ("gemini", "flash", False)
    assert roles_and_texts(world, conversation_id) == [
        (MessageRole.USER, "How often should I wash oily hair?"),
        (MessageRole.ASSISTANT, "Twice a week"),
    ]
    assistant = world.repo.of(conversation_id)[-1]
    assert done.message_id == assistant.id
    assert (assistant.provider, assistant.model) == ("gemini", "flash")


async def test_first_message_sets_title_to_60_chars_later_ones_do_not() -> None:
    world = World()
    conversation_id = await world.conversation()

    await world.send(conversation_id, "x" * 70)
    await world.send(conversation_id, "another question")

    assert world.repo.conversations[conversation_id].title == "x" * 60


async def test_llm_calls_of_the_generation_are_linked_to_the_answer() -> None:
    world = World(
        FakeLLMProvider("gemini", "flash").script(fail_before_first(ProviderTimeout())),
        FakeLLMProvider("groq", "oss").script(success("answer")),
    )
    conversation_id = await world.conversation()

    events = await world.send(conversation_id)

    done = events[-1]
    assert isinstance(done, ChatDone)
    assert world.repo.linked_calls == {c.id: done.message_id for c in world.recorder.calls}
    assert len(world.recorder.calls) == 2


async def test_restart_saves_only_the_full_answer_of_the_next_provider() -> None:
    world = World(
        FakeLLMProvider("gemini", "flash").script(fail_mid_stream(StreamBroken(), "Wash ev")),
        FakeLLMProvider("groq", "oss").script(success("Wash ", "twice a week")),
    )
    conversation_id = await world.conversation()

    events = await world.send(conversation_id)

    assert Restart(reason="stream_broken") in events
    done = events[-1]
    assert isinstance(done, ChatDone) and done.provider == "groq"
    assert roles_and_texts(world, conversation_id)[-1] == (
        MessageRole.ASSISTANT,
        "Wash twice a week",
    )


async def test_all_providers_down_yields_llm_unavailable_and_keeps_user_message() -> None:
    world = World(
        FakeLLMProvider("gemini", "flash").script(
            fail_before_first(ProviderHTTPError(503, "InternalServerError"))
        )
    )
    conversation_id = await world.conversation()

    events = await world.send(conversation_id, "my question")

    assert events == [
        ChatError("llm_unavailable", "The assistant is temporarily unavailable, please try again")
    ]
    assert roles_and_texts(world, conversation_id) == [(MessageRole.USER, "my question")]


async def test_rejected_request_yields_llm_request_rejected() -> None:
    world = World(
        FakeLLMProvider("gemini", "flash").script(
            fail_before_first(ProviderHTTPError(400, "BadRequestError"))
        )
    )
    conversation_id = await world.conversation()

    [event] = await world.send(conversation_id)

    assert isinstance(event, ChatError) and event.code == "llm_request_rejected"


async def test_unexpected_failure_yields_internal_error(caplog: pytest.LogCaptureFixture) -> None:
    world = World()
    conversation_id = await world.conversation()

    async def broken_save(*args: object, **kwargs: object) -> None:
        raise RuntimeError("INSERT ... 'secret user text'")

    world.repo.add_assistant_message = broken_save  # type: ignore[method-assign,assignment]

    events = await world.send(conversation_id)

    assert isinstance(events[-1], ChatError) and events[-1].code == "internal_error"
    assert "secret user text" not in caplog.text


async def test_truncated_answer_is_flagged_and_saved_with_usage() -> None:
    world = World(FakeLLMProvider("gemini", "flash").script(truncated("cut ", "off")))
    conversation_id = await world.conversation()

    events = await world.send(conversation_id)

    done = events[-1]
    assert isinstance(done, ChatDone) and done.truncated
    assistant = world.repo.of(conversation_id)[-1]
    assert (assistant.content, assistant.tokens_in, assistant.tokens_out) == ("cut off", 10, 20)


async def test_client_cancel_mid_stream_saves_no_answer_and_frees_the_slot() -> None:
    provider = FakeLLMProvider("gemini", "flash").script(
        Scenario(tokens=("first", "second", "third")), success("next")
    )
    world = World(provider)
    conversation_id = await world.conversation()
    first_token = asyncio.Event()

    async def client() -> None:
        async with world.chat.reply(USER, conversation_id, "question") as reply:
            async for event in reply.events():
                if isinstance(event, Token):
                    first_token.set()
                    await asyncio.Event().wait()  # the client hangs, then disconnects

    task = asyncio.create_task(client())
    await first_token.wait()
    assert world.locks.is_busy(conversation_id)
    task.cancel()  # what Starlette does when the HTTP client goes away
    with pytest.raises(asyncio.CancelledError):
        await task

    assert roles_and_texts(world, conversation_id) == [(MessageRole.USER, "question")]
    assert provider.closed_streams == 1
    assert not world.locks.is_busy(conversation_id)
    assert world.recorder.calls == []  # a disconnect is not a provider failure
    # The conversation is usable again right away.
    assert isinstance((await world.send(conversation_id))[-1], ChatDone)


async def test_second_reply_while_generating_is_busy() -> None:
    world = World(FakeLLMProvider("gemini", "flash").script(hang()))
    conversation_id = await world.conversation()

    async with world.chat.reply(USER, conversation_id, "first"):
        with pytest.raises(ConversationBusy):
            async with world.chat.reply(USER, conversation_id, "second"):
                pass  # pragma: no cover

    # The rejected request saved nothing.
    assert roles_and_texts(world, conversation_id) == [(MessageRole.USER, "first")]
    assert not world.locks.is_busy(conversation_id)


async def test_other_conversations_are_not_blocked() -> None:
    world = World()
    first, second = await world.conversation(), await world.conversation()

    async with world.chat.reply(USER, first, "q"):
        assert isinstance((await world.send(second))[-1], ChatDone)


async def test_someone_elses_or_missing_conversation_is_not_found() -> None:
    world = World()
    foreign = await world.conversation(user=OTHER)

    for conversation_id in (foreign, uuid.uuid4()):
        with pytest.raises(ConversationNotFound):
            async with world.chat.reply(USER, conversation_id, "q"):
                pass  # pragma: no cover
        with pytest.raises(ConversationNotFound):
            await world.chat.conversations.list_messages(USER, conversation_id, 20, None)

    assert world.repo.of(foreign) == []


# With StaticPrompt (14 tokens) a 35-char message costs 16: 30 fits the prompt and one message.
TIGHT_BUDGET = 30
MESSAGE = "x" * 35


async def test_summary_is_due_after_an_answer_whose_context_left_messages_out() -> None:
    world = World(budget=TIGHT_BUDGET, prompts=StaticPrompt())
    conversation_id = await world.conversation()

    async with world.chat.reply(USER, conversation_id, MESSAGE) as reply:
        _ = [event async for event in reply.events()]
    assert not reply.summary_due  # everything fit

    async with world.chat.reply(USER, conversation_id, MESSAGE) as reply:
        events = [event async for event in reply.events()]
    assert isinstance(events[-1], ChatDone)
    assert reply.summary_due


async def test_no_summary_is_due_when_the_reply_failed() -> None:
    world = World(
        FakeLLMProvider("gemini", "flash").script(
            success(), fail_before_first(ProviderHTTPError(503, "InternalServerError"))
        ),
        budget=TIGHT_BUDGET,
        prompts=StaticPrompt(),
    )
    conversation_id = await world.conversation()
    await world.send(conversation_id, MESSAGE)

    async with world.chat.reply(USER, conversation_id, MESSAGE) as reply:
        events = [event async for event in reply.events()]

    assert isinstance(events[-1], ChatError)
    assert not reply.summary_due


# --- moderation (ADR 0004) ---


@pytest.mark.parametrize("category", ["self-harm", "self-harm/intent", "self-harm/instructions"])
async def test_self_harm_is_answered_softly_notice_first_and_saved(category: str) -> None:
    provider = FakeLLMProvider("gemini", "flash").script(success("care"))
    world = World(provider, prompts=StaticPrompt(), moderator=FakeModerator(flag(category)))
    conversation_id = await world.conversation()

    events = await world.send(conversation_id, "I can't take it anymore, my hair is falling out")

    assert events[0] == SafetyNotice("self_harm")
    assert events[1] == Token("care", "gemini", "flash")
    assert isinstance(events[-1], ChatDone)
    user_message = world.repo.of(conversation_id)[0]
    assert user_message.content == "I can't take it anymore, my hair is falling out"
    assert user_message.moderation == {
        "provider": "openai",
        "model": "omni-moderation-test",
        "flagged": True,
        "action": "soft",
        "categories": [category],
    }
    system = provider.received[0][0]["content"]
    assert system.startswith("You are a hair-care assistant.\n\n<moderation_signal>\n")
    assert "Possible crisis signal" in system


async def test_self_harm_wins_over_blocking_categories() -> None:
    world = World(moderator=FakeModerator(flag("self-harm/intent", "violence", "harassment")))
    conversation_id = await world.conversation()

    events = await world.send(conversation_id, "q")

    assert events[0] == SafetyNotice("self_harm")
    assert isinstance(events[-1], ChatDone)


async def test_safety_signal_is_only_for_the_flagged_message() -> None:
    provider = FakeLLMProvider("gemini", "flash")
    world = World(provider, moderator=FakeModerator(flag("self-harm")))
    conversation_id = await world.conversation()

    await world.send(conversation_id, "first")
    events = await world.send(conversation_id, "second")

    assert not any(isinstance(e, SafetyNotice) for e in events)
    assert "<moderation_signal>\n" not in provider.received[1][0]["content"]


async def test_notice_comes_before_the_error_when_the_chain_is_down() -> None:
    world = World(
        FakeLLMProvider("gemini", "flash").script(fail_before_first(ProviderTimeout())),
        moderator=FakeModerator(flag("self-harm")),
    )
    conversation_id = await world.conversation()

    events = await world.send(conversation_id, "q")

    assert events[0] == SafetyNotice("self_harm")
    assert isinstance(events[-1], ChatError)


@pytest.mark.parametrize(
    "verdict",
    [
        flag("harassment"),
        flag("hate"),
        flag("violence", scores={"violence": 0.9}),
        flag("violence/graphic", scores={"violence/graphic": 0.95}),
        flag("violence", "harassment/threatening", scores={"violence": 0.3}),
    ],
)
async def test_blocked_message_is_not_saved_and_frees_the_slot(verdict: ModerationResult) -> None:
    provider = FakeLLMProvider("gemini", "flash")
    world = World(provider, moderator=FakeModerator(verdict))
    conversation_id = await world.conversation()

    with pytest.raises(MessageBlocked) as caught:
        await world.send(conversation_id, "blocked text")

    assert caught.value.categories == verdict.categories
    assert world.repo.of(conversation_id) == []
    assert world.repo.conversations[conversation_id].title is None
    assert provider.calls == 0
    assert not world.locks.is_busy(conversation_id)


async def test_violence_below_threshold_is_allowed_and_kept_as_flagged() -> None:
    world = World(
        moderator=FakeModerator(
            flag("violence", "violence/graphic", scores={"violence": 0.5, "violence/graphic": 0.2})
        )
    )
    conversation_id = await world.conversation()

    events = await world.send(conversation_id, "I burned my scalp with bleach")

    assert not any(isinstance(e, SafetyNotice) for e in events)
    assert isinstance(events[-1], ChatDone)
    assert world.repo.of(conversation_id)[0].moderation == {
        "provider": "openai",
        "model": "omni-moderation-test",
        "flagged": True,
        "action": "allow",
        "categories": ["violence", "violence/graphic"],
    }


async def test_moderation_log_has_action_and_categories_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    world = World(moderator=FakeModerator(flag("harassment", scores={"harassment": 0.123456})))
    conversation_id = await world.conversation()

    with caplog.at_level("DEBUG"), pytest.raises(MessageBlocked):
        await world.send(conversation_id, "secret words here")

    assert "action=block categories=harassment" in caplog.text
    assert "secret words" not in caplog.text
    assert "0.123" not in caplog.text
    assert str(USER) not in caplog.text


async def test_clean_message_is_checked_saved_with_moderation_and_answered() -> None:
    moderator = FakeModerator()
    world = World(moderator=moderator)
    conversation_id = await world.conversation()

    events = await world.send(conversation_id, "How often should I wash oily hair?")

    assert isinstance(events[0], Token)
    assert isinstance(events[-1], ChatDone)
    assert moderator.checked == ["How often should I wash oily hair?"]
    assert world.repo.of(conversation_id)[0].moderation == {
        "provider": "openai",
        "model": "omni-moderation-test",
        "flagged": False,
        "action": "allow",
    }


@pytest.mark.parametrize(
    "error",
    [
        ModerationUnavailable("timeout"),
        ModerationUnavailable("InternalServerError:503"),
        ModerationUnavailable("RateLimitError:429"),
        ModerationUnavailable("connection_error"),
        ModerationUnavailable("AuthenticationError:401", misconfigured=True),
    ],
)
async def test_moderation_down_lets_the_message_through(error: ModerationUnavailable) -> None:
    world = World(moderator=FakeModerator(error))
    conversation_id = await world.conversation()

    events = await world.send(conversation_id, "question")

    assert isinstance(events[-1], ChatDone)
    assert world.repo.of(conversation_id)[0].moderation == {"status": "unavailable"}


async def test_moderation_timeout_is_logged_as_warning_misconfiguration_as_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    world = World(
        moderator=FakeModerator(
            ModerationUnavailable("timeout"),
            ModerationUnavailable("AuthenticationError:401", misconfigured=True),
        )
    )
    conversation_id = await world.conversation()

    with caplog.at_level("WARNING"):
        await world.send(conversation_id, "one")
        await world.send(conversation_id, "two")

    levels = [(r.levelname, r.getMessage()) for r in caplog.records if "moderation" in r.message]
    assert levels == [
        ("WARNING", "moderation unavailable, message let through: code=timeout"),
        ("ERROR", "moderation unavailable, message let through: code=AuthenticationError:401"),
    ]


async def test_without_moderator_messages_are_marked_disabled() -> None:
    world = World(moderator=NoopModerator())
    conversation_id = await world.conversation()

    await world.send(conversation_id, "question")

    assert world.repo.of(conversation_id)[0].moderation == {"status": "disabled"}


async def test_busy_conversation_is_not_sent_to_moderation() -> None:
    moderator = FakeModerator()
    world = World(moderator=moderator)
    conversation_id = await world.conversation()

    with world.locks.hold(conversation_id), pytest.raises(ConversationBusy):
        await world.send(conversation_id, "second")

    assert moderator.checked == []


# --- knowledge base (RAG) ---


class OneArticleKB:
    """KBRepository with one dandruff passage that matches any question."""

    async def search(
        self, vector: Sequence[float], *, model: str, include_drafts: bool, limit: int
    ) -> list[KBHit]:
        return [KBHit("Перхоть › Уход\nМягкий шампунь.", "Перхоть", "dandruff", None, 0.8)]


def rag_world(embedder: FakeEmbedder, moderator: Moderator | None = None) -> World:
    knowledge = RagKnowledge(
        OneArticleKB(), top_k=4, min_similarity=0.6, max_tokens=1500, include_drafts=False
    )
    return World(
        moderator=moderator,
        query_embedder=RagQueryEmbedder(embedder, timeout=0.05),
        knowledge=knowledge,
    )


async def test_done_lists_the_articles_that_were_in_the_context() -> None:
    world = rag_world(FakeEmbedder())
    conversation_id = await world.conversation()

    events = await world.send(conversation_id, "Что делать с перхотью?")

    done = events[-1]
    assert isinstance(done, ChatDone)
    assert done.sources == (Source("Перхоть", "dandruff", None),)
    system = world.providers[0].received[0][0]["content"]
    assert KNOWLEDGE_NOTE in system and "Мягкий шампунь." in system


@pytest.mark.parametrize("embedder", [FakeEmbedder(fail=True), FakeEmbedder(hang=True)])
async def test_embeddings_down_answer_goes_without_knowledge(
    embedder: FakeEmbedder, caplog: pytest.LogCaptureFixture
) -> None:
    world = rag_world(embedder)
    conversation_id = await world.conversation()

    with caplog.at_level(logging.WARNING):
        events = await world.send(conversation_id, "Что делать с перхотью?")

    done = events[-1]
    assert isinstance(done, ChatDone)
    assert done.sources == ()
    assert KNOWLEDGE_NOTE not in world.providers[0].received[0][0]["content"]
    assert "embeddings unavailable" in caplog.text
    assert "перхот" not in caplog.text


class WaitsForEmbedding:
    """Moderator that answers only once the embedding has started: proves they run together."""

    enabled = True

    def __init__(self, embedder: FakeEmbedder, verdict: ModerationResult | None = None) -> None:
        self.embedder = embedder
        self.verdict = verdict or ModerationResult("openai", "omni-moderation-test", flagged=False)

    async def check(self, text: str) -> ModerationResult:
        async with asyncio.timeout(1):
            await self.embedder.started.wait()
        return self.verdict

    async def aclose(self) -> None:
        return None


async def test_question_is_embedded_while_it_is_moderated() -> None:
    embedder = FakeEmbedder()
    world = rag_world(embedder, moderator=WaitsForEmbedding(embedder))
    conversation_id = await world.conversation()

    events = await world.send(conversation_id, "Что делать с перхотью?")

    assert isinstance(events[-1], ChatDone)
    assert embedder.embedded == ["Что делать с перхотью?"]


async def test_blocked_message_cancels_its_embedding() -> None:
    embedder = FakeEmbedder(hang=True)
    world = rag_world(embedder, moderator=WaitsForEmbedding(embedder, flag("harassment")))
    conversation_id = await world.conversation()

    with pytest.raises(MessageBlocked):
        await world.send(conversation_id, "blocked text")
    await asyncio.sleep(0)

    assert embedder.cancelled
    assert world.repo.of(conversation_id) == []
