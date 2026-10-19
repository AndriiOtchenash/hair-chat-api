import asyncio
import uuid

import pytest

from app.db.models import MessageRole
from app.llm.base import ProviderHTTPError, ProviderTimeout, Restart, StreamBroken, Token
from app.llm.moderation import ModerationUnavailable, Moderator, NoopModerator
from app.llm.resilient import ResilientLLM
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
)
from app.services.context_builder import TokenBudgetContextBuilder
from app.services.prompts import FilePromptProvider, PromptProvider
from tests.fakes import FakeConversationRepository, StaticPrompt
from tests.llm_fakes import (
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
    ) -> None:
        self.repo = FakeConversationRepository()
        self.recorder = FakeLLMCallRecorder()
        self.providers = providers or (FakeLLMProvider("gemini", "flash"),)
        self.locks = ConversationLocks()
        self.moderator = moderator or FakeModerator()
        self.chat = ChatService(
            ConversationService(self.repo),
            ResilientLLM(self.providers, self.recorder, first_token_timeout=0.05),
            TokenBudgetContextBuilder(self.repo, prompts or FilePromptProvider()),
            self.locks,
            self.moderator,
            max_answer_tokens=100,
            budget_tokens=budget,
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
async def test_self_harm_is_blocked_with_its_own_code_and_nothing_is_saved(category: str) -> None:
    provider = FakeLLMProvider("gemini", "flash")
    world = World(provider, moderator=FakeModerator(flag(category, "violence")))
    conversation_id = await world.conversation()

    with pytest.raises(MessageBlocked) as caught:
        await world.send(conversation_id, "blocked text")

    assert caught.value.code == "message_blocked_self_harm"
    assert world.repo.of(conversation_id) == []
    assert world.repo.conversations[conversation_id].title is None
    assert provider.calls == 0
    assert not world.locks.is_busy(conversation_id)


async def test_other_flagged_categories_are_blocked_as_message_blocked() -> None:
    world = World(moderator=FakeModerator(flag("harassment", "violence")))
    conversation_id = await world.conversation()

    with pytest.raises(MessageBlocked) as caught:
        await world.send(conversation_id, "blocked text")

    assert (caught.value.code, caught.value.categories) == (
        "message_blocked",
        ("harassment", "violence"),
    )
    assert world.repo.of(conversation_id) == []


async def test_block_log_has_code_and_categories_but_no_text_or_user(
    caplog: pytest.LogCaptureFixture,
) -> None:
    world = World(moderator=FakeModerator(flag("self-harm/intent")))
    conversation_id = await world.conversation()

    with caplog.at_level("DEBUG"), pytest.raises(MessageBlocked):
        await world.send(conversation_id, "secret words here")

    assert "code=message_blocked_self_harm categories=self-harm/intent" in caplog.text
    assert "secret words" not in caplog.text
    assert str(USER) not in caplog.text


async def test_clean_message_is_checked_saved_with_moderation_and_answered() -> None:
    moderator = FakeModerator()
    world = World(moderator=moderator)
    conversation_id = await world.conversation()

    events = await world.send(conversation_id, "How often should I wash oily hair?")

    assert isinstance(events[-1], ChatDone)
    assert moderator.checked == ["How often should I wash oily hair?"]
    user_message = world.repo.of(conversation_id)[0]
    assert user_message.moderation == {
        "provider": "openai",
        "model": "omni-moderation-test",
        "flagged": False,
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
