import io
import uuid
from collections.abc import AsyncIterator

from app.cli.chat import converse
from app.db.models import MessageRole
from app.llm.base import ProviderHTTPError, StreamBroken
from tests.llm_fakes import FakeLLMProvider, fail_before_first, fail_mid_stream, success, truncated
from tests.unit.test_chat_service import USER, World


async def lines(*items: str) -> AsyncIterator[str]:
    for item in items:
        yield item


async def run(world: World, *inputs: str) -> tuple[uuid.UUID, str]:
    conversation_id = await world.conversation()
    out = io.StringIO()
    await converse(world.chat, USER, conversation_id, lines(*inputs), out)
    return conversation_id, out.getvalue()


async def test_dialog_prints_tokens_restart_and_model() -> None:
    world = World(
        FakeLLMProvider("gemini", "flash").script(
            success("Hi"), fail_mid_stream(StreamBroken(), "pa")
        ),
        FakeLLMProvider("groq", "oss").script(truncated("full")),
    )

    conversation_id, out = await run(world, "hello", "  ", "more", "/exit", "never sent")

    assert out == (
        "Hi\n— gemini:flash\n\npa\n[restart: stream_broken]\nfull\n— groq:oss (truncated)\n\n"
    )
    assert [m.content for m in world.repo.of(conversation_id) if m.role is MessageRole.USER] == [
        "hello",
        "more",
    ]


async def test_error_is_printed_and_dialog_goes_on() -> None:
    world = World(
        FakeLLMProvider("gemini", "flash").script(
            fail_before_first(ProviderHTTPError(503, "InternalServerError")), success("back")
        )
    )

    _, out = await run(world, "one", "two")

    assert "[error llm_unavailable:" in out
    assert out.endswith("back\n— gemini:flash\n\n")


class RecordingSummarizer:
    def __init__(self) -> None:
        self.runs: list[uuid.UUID] = []

    async def run(self, user_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        self.runs.append(conversation_id)


async def test_summarizer_runs_after_an_answer_whose_context_overflowed() -> None:
    world = World(budget=1200)
    summarizer = RecordingSummarizer()
    conversation_id = await world.conversation()
    out = io.StringIO()

    await converse(
        world.chat, USER, conversation_id, lines("short", *["x" * 400] * 3), out, summarizer
    )

    # The 4th answer is the first whose context had to leave messages out.
    assert summarizer.runs == [conversation_id]
