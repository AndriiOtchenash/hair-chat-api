import uuid

import pytest

from app.db.models import LLMCallStatus, Message
from app.llm.base import ProviderHTTPError
from app.services.summarizer import KEEP_LAST_MESSAGES, Summarizer
from app.services.tokens import message_tokens
from tests.fakes import FakeConversationRepository
from tests.llm_fakes import (
    FailingLLMCallRecorder,
    FakeLLMCallRecorder,
    FakeLLMProvider,
    fail_before_first,
    hang,
    success,
)

USER = uuid.UUID("01890000-0000-7000-8000-000000000001")
OTHER = uuid.UUID("01890000-0000-7000-8000-000000000002")
BUDGET = 900  # keeps a tail of at most 300 tokens
SECRET = "my scalp secret"


class World:
    def __init__(self, provider: FakeLLMProvider | None = None) -> None:
        self.repo = FakeConversationRepository()
        self.provider = provider or FakeLLMProvider("groq", "oss").script(success("Summary."))
        self.recorder = FakeLLMCallRecorder()
        self.summarizer = Summarizer(
            self.repo.repositories(),
            self.provider,
            self.recorder,
            budget_tokens=BUDGET,
            timeout=0.05,
        )

    async def conversation_with(
        self, count: int, chars: int = 350
    ) -> tuple[uuid.UUID, list[Message]]:
        conversation = await self.repo.create(USER)
        messages = []
        for i in range(count):
            text = f"{SECRET} {i:02d} " + "x" * chars
            if i % 2 == 0:
                messages.append(await self.repo.add_user_message(USER, conversation.id, text))
            else:
                messages.append(
                    await self.repo.add_assistant_message(
                        USER,
                        conversation.id,
                        text,
                        provider="gemini",
                        model="flash",
                        tokens_in=None,
                        tokens_out=None,
                        llm_call_ids=(),
                    )
                )
        return conversation.id, messages

    def summary_of(self, conversation_id: uuid.UUID) -> tuple[str | None, uuid.UUID | None]:
        conversation = self.repo.conversations[conversation_id]
        return conversation.summary, conversation.summary_upto_message_id


async def test_folds_old_messages_and_keeps_a_tail_within_a_third_of_the_budget() -> None:
    world = World()
    conversation_id, messages = await world.conversation_with(10)  # ~124 tokens each

    await world.summarizer.run(USER, conversation_id)

    summary, upto = world.summary_of(conversation_id)
    assert summary == "Summary."
    folded = next(i for i, m in enumerate(messages) if m.id == upto) + 1
    tail = messages[folded:]
    assert sum(message_tokens(m.content) for m in tail) <= BUDGET // 3
    assert len(tail) >= KEEP_LAST_MESSAGES
    # The model got the folded messages as data, and nothing of the tail.
    request = world.provider.received[0][1]["content"]
    assert request.startswith("<conversation>")
    assert all(m.content in request for m in messages[:folded])
    assert all(m.content not in request for m in tail)
    assert [(c.provider, c.model, c.status) for c in world.recorder.calls] == [
        ("groq", "oss", LLMCallStatus.OK)
    ]


async def test_the_next_run_merges_the_previous_summary_and_continues_after_it() -> None:
    world = World(FakeLLMProvider("groq", "oss").script(success("First."), success("Second.")))
    conversation_id, messages = await world.conversation_with(10)
    await world.summarizer.run(USER, conversation_id)
    _, first_upto = world.summary_of(conversation_id)
    for _ in range(3):
        await world.repo.add_user_message(USER, conversation_id, "y" * 700)

    await world.summarizer.run(USER, conversation_id)

    request = world.provider.received[1][1]["content"]
    assert "<previous_summary>\n" in request and "First." in request
    already_folded = [m for m in messages if m.id == first_upto][0]
    assert already_folded.content not in request
    assert world.summary_of(conversation_id)[0] == "Second."


async def test_the_last_exchange_always_stays_verbatim() -> None:
    world = World()
    conversation_id, messages = await world.conversation_with(3, chars=2000)

    await world.summarizer.run(USER, conversation_id)

    assert world.summary_of(conversation_id) == ("Summary.", messages[0].id)


async def test_nothing_to_fold_calls_no_model() -> None:
    world = World()
    conversation_id, _ = await world.conversation_with(2, chars=10)

    await world.summarizer.run(USER, conversation_id)

    assert world.provider.calls == 0
    assert world.summary_of(conversation_id) == (None, None)


@pytest.mark.parametrize(
    ("provider", "code"),
    [
        (
            FakeLLMProvider("groq", "oss").script(
                fail_before_first(ProviderHTTPError(503, "InternalServerError"))
            ),
            "InternalServerError:503",
        ),
        (FakeLLMProvider("groq", "oss").script(hang()), "timeout"),
        (FakeLLMProvider("groq", "oss").script(success("  ")), "empty_response"),
    ],
)
async def test_a_failed_summary_keeps_the_old_one_and_never_raises(
    provider: FakeLLMProvider, code: str, caplog: pytest.LogCaptureFixture
) -> None:
    world = World(provider)
    conversation_id, _ = await world.conversation_with(10)

    await world.summarizer.run(USER, conversation_id)

    assert world.summary_of(conversation_id) == (None, None)
    assert f"error={code}" in caplog.text
    assert SECRET not in caplog.text
    assert [(c.status, c.error) for c in world.recorder.calls] == [(LLMCallStatus.ERROR, code)]
    assert provider.closed_streams == 1


async def test_a_broken_recorder_does_not_stop_the_summary() -> None:
    world = World()
    world.summarizer = Summarizer(
        world.repo.repositories(), world.provider, FailingLLMCallRecorder(), budget_tokens=BUDGET
    )
    conversation_id, _ = await world.conversation_with(10)

    await world.summarizer.run(USER, conversation_id)

    assert world.summary_of(conversation_id)[0] == "Summary."


class RacingRepository(FakeConversationRepository):
    """Another run saves its summary while this run waits for the model."""

    def __init__(self) -> None:
        super().__init__()
        self.other_upto: uuid.UUID | None = None

    async def save_summary(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        summary: str,
        *,
        upto_message_id: uuid.UUID,
        expected_upto: uuid.UUID | None,
    ) -> bool:
        assert self.other_upto is not None
        await super().save_summary(
            user_id,
            conversation_id,
            "other run",
            upto_message_id=self.other_upto,
            expected_upto=None,
        )
        return await super().save_summary(
            user_id,
            conversation_id,
            summary,
            upto_message_id=upto_message_id,
            expected_upto=expected_upto,
        )


async def test_a_concurrent_summary_is_not_overwritten() -> None:
    world = World()
    world.repo = RacingRepository()
    world.summarizer = Summarizer(
        world.repo.repositories(), world.provider, world.recorder, budget_tokens=BUDGET
    )
    conversation_id, messages = await world.conversation_with(10)
    world.repo.other_upto = messages[5].id

    await world.summarizer.run(USER, conversation_id)

    assert world.summary_of(conversation_id) == ("other run", messages[5].id)


async def test_someone_elses_conversation_is_left_alone() -> None:
    world = World()
    conversation_id, _ = await world.conversation_with(10)

    await world.summarizer.run(OTHER, conversation_id)

    assert world.provider.calls == 0
    assert world.summary_of(conversation_id) == (None, None)
