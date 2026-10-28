"""Chat over real HTTP: a live uvicorn server, Postgres, fake LLM providers.

httpx's ASGITransport buffers the whole response, so it can show neither streaming nor a client
that disconnects mid-answer; a real server on a local port can.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import fastapi.routing
import pytest
import uvicorn
from httpx import AsyncClient
from sqlalchemy import select

from app.core.chat import get_conversation_locks, get_llm, get_moderator, get_query_embedder
from app.core.config import get_settings
from app.core.security import get_token_verifier
from app.db.models import ConsentType, Conversation, LLMCall, Message, MessageRole
from app.db.repositories.conversations import SqlConversationRepository
from app.db.repositories.kb import SqlKBRepository
from app.db.repositories.llm_calls import SqlLLMCallRecorder
from app.db.session import get_sessionmaker
from app.llm.base import ProviderHTTPError, ProviderTimeout, StreamBroken
from app.llm.resilient import ResilientLLM
from app.main import create_app
from app.rag.chunker import Chunk
from app.rag.documents import parse_article
from app.rag.retriever import RagQueryEmbedder
from app.services.chat_service import ConversationLocks
from app.services.tokens import message_tokens
from tests.auth_fakes import FakeAuth0, SigningKey, bearer, mint
from tests.llm_fakes import (
    FakeEmbedder,
    FakeLLMProvider,
    FakeModerator,
    fail_before_first,
    fail_mid_stream,
    flag,
    hang,
    stall_after,
    success,
    text_vector,
)

KEY = SigningKey("key-1")


@dataclass
class Live:
    client: AsyncClient
    locks: ConversationLocks
    moderator: FakeModerator
    primary: FakeLLMProvider = field(default_factory=lambda: FakeLLMProvider("gemini", "flash"))
    backup: FakeLLMProvider = field(default_factory=lambda: FakeLLMProvider("groq", "oss"))


@pytest.fixture
async def live(app_settings_env: None) -> AsyncIterator[Live]:
    app = create_app()
    verifier = FakeAuth0(KEY).verifier()
    locks = ConversationLocks()
    moderator = FakeModerator()
    providers = (FakeLLMProvider("gemini", "flash"), FakeLLMProvider("groq", "oss"))
    llm = ResilientLLM(providers, SqlLLMCallRecorder(get_sessionmaker()), first_token_timeout=2)
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_conversation_locks] = lambda: locks
    app.dependency_overrides[get_moderator] = lambda: moderator
    # Never the real embeddings API (rule 1), even with GEMINI_API_KEY in the environment.
    query_embedder = RagQueryEmbedder(FakeEmbedder())
    app.dependency_overrides[get_query_embedder] = lambda: query_embedder

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
    )
    task = asyncio.create_task(server.serve())
    for _ in range(500):  # uvicorn exposes only a flag, no event, for "listening"
        if server.started:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("uvicorn did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    async with AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
        yield Live(client, locks, moderator, *providers)
    server.should_exit = True
    await task
    await verifier.aclose()


async def consented_user(live: Live) -> dict[str, str]:
    headers = bearer(mint(KEY, sub=f"auth0|{uuid.uuid4()}"))
    response = await live.client.post(
        "/consents",
        headers=headers,
        json={"types": [t.value for t in ConsentType], "version": get_settings().consent_version},
    )
    assert response.status_code == 200, response.text
    return headers


async def new_conversation(live: Live, headers: dict[str, str]) -> str:
    response = await live.client.post("/conversations", headers=headers)
    assert response.status_code == 201, response.text
    conversation_id: str = response.json()["id"]
    return conversation_id


@dataclass
class SSE:
    event: str
    data: dict[str, object]


async def read_events(lines: AsyncIterator[str], *, stop_after: int | None = None) -> list[SSE]:
    """Parses `event:`/`data:` blocks; `: ...` comments come back as event "comment"."""
    events: list[SSE] = []
    name, data = "message", ""
    async for line in lines:
        if line.startswith(":"):
            events.append(SSE("comment", {"text": line[1:].strip()}))
        elif line.startswith("event:"):
            name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data += line.removeprefix("data:").strip()
        elif line == "" and data:
            events.append(SSE(name, json.loads(data)))
            name, data = "message", ""
        if stop_after is not None and len(events) >= stop_after:
            break
    return events


async def send(
    live: Live, headers: dict[str, str], conversation_id: str, content: str
) -> list[SSE]:
    async with live.client.stream(
        "POST",
        f"/conversations/{conversation_id}/messages",
        headers=headers,
        json={"content": content},
    ) as response:
        assert response.status_code == 200, await response.aread()
        return await read_events(response.aiter_lines())


async def stored(conversation_id: str) -> list[Message]:
    async with get_sessionmaker()() as session:
        rows = await session.scalars(
            select(Message)
            .where(Message.conversation_id == uuid.UUID(conversation_id))
            .order_by(Message.created_at, Message.id)
        )
        return list(rows)


async def wait_until_free(live: Live, conversation_id: str) -> None:
    for _ in range(200):
        if not live.locks.is_busy(uuid.UUID(conversation_id)):
            return
        await asyncio.sleep(0.01)
    pytest.fail("generation slot was not released")


async def test_streams_tokens_then_done_and_persists_everything(live: Live) -> None:
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)
    live.primary.script(success("Привіт, ", "мийте ", "двічі на тиждень"))

    async with live.client.stream(
        "POST",
        f"/conversations/{conversation_id}/messages",
        headers=headers,
        json={"content": "  Як часто мити голову?  "},
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
        events = await read_events(response.aiter_lines())

    assert [e.event for e in events] == ["token", "token", "token", "done"]
    assert "".join(str(e.data["text"]) for e in events[:3]) == "Привіт, мийте двічі на тиждень"
    done = events[-1].data
    assert (done["provider"], done["model"], done["truncated"]) == ("gemini", "flash", False)
    assert done["sources"] == []  # no article of the fake embedder's model in the KB

    user_msg, assistant = await stored(conversation_id)
    assert (user_msg.role, user_msg.content) == (MessageRole.USER, "Як часто мити голову?")
    assert user_msg.moderation == {
        "provider": "openai",
        "model": "omni-moderation-test",
        "flagged": False,
        "action": "allow",
    }
    assert (assistant.role, assistant.content) == (
        MessageRole.ASSISTANT,
        "Привіт, мийте двічі на тиждень",
    )
    assert str(assistant.id) == done["message_id"]
    async with get_sessionmaker()() as session:
        calls = list(
            await session.scalars(select(LLMCall).where(LLMCall.message_id == assistant.id))
        )
    assert [(c.provider, c.model) for c in calls] == [("gemini", "flash")]

    listing = (await live.client.get("/conversations", headers=headers)).json()
    assert listing["items"][0]["title"] == "Як часто мити голову?"


async def test_blocked_message_is_422_before_the_stream_and_not_saved(live: Live) -> None:
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)
    live.moderator.verdicts.append(flag("harassment/threatening", "violence"))

    response = await live.client.post(
        f"/conversations/{conversation_id}/messages", headers=headers, json={"content": "bad"}
    )

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    assert response.json()["code"] == "message_blocked"
    assert set(response.json()) == {"code", "message"}
    assert await stored(conversation_id) == []
    assert live.primary.calls == 0
    await wait_until_free(live, conversation_id)


async def test_self_harm_gets_safety_notice_before_tokens_and_is_saved(live: Live) -> None:
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)
    live.moderator.verdicts.append(flag("self-harm/intent"))
    live.primary.script(success("Мені шкода, ", "що вам важко"))

    events = await send(live, headers, conversation_id, "волосся випадає, не хочу жити")

    assert [e.event for e in events] == ["safety_notice", "token", "token", "done"]
    assert events[0].data == {"kind": "self_harm"}
    user_msg, _ = await stored(conversation_id)
    assert user_msg.moderation == {
        "provider": "openai",
        "model": "omni-moderation-test",
        "flagged": True,
        "action": "soft",
        "categories": ["self-harm/intent"],
    }
    assert "<moderation_signal>\n" in live.primary.received[0][0]["content"]


async def test_restart_then_full_answer_from_backup_with_linked_attempts(live: Live) -> None:
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)
    live.primary.script(fail_mid_stream(StreamBroken(), "half "))
    live.backup.script(success("whole ", "answer"))

    events = await send(live, headers, conversation_id, "q")

    assert [e.event for e in events] == ["token", "restart", "token", "token", "done"]
    assert events[1].data == {"reason": "stream_broken"}
    assert events[-1].data["provider"] == "groq"
    assistant = (await stored(conversation_id))[-1]
    assert assistant.content == "whole answer"
    async with get_sessionmaker()() as session:
        calls = list(
            await session.scalars(select(LLMCall).where(LLMCall.message_id == assistant.id))
        )
    assert sorted(c.provider for c in calls) == ["gemini", "groq"]


async def test_all_providers_down_sends_error_event_and_keeps_user_message(live: Live) -> None:
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)
    live.primary.script(fail_before_first(ProviderHTTPError(503, "InternalServerError")))
    live.backup.script(fail_before_first(ProviderTimeout()))

    events = await send(live, headers, conversation_id, "anyone there?")

    assert [(e.event, e.data["code"]) for e in events] == [("error", "llm_unavailable")]
    assert set(events[0].data) == {"code", "message"}
    assert [(m.role, m.content) for m in await stored(conversation_id)] == [
        (MessageRole.USER, "anyone there?")
    ]


async def test_client_disconnect_cancels_generation_and_saves_no_answer(live: Live) -> None:
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)
    live.primary.script(stall_after("partial "), success("later answer"))

    async with live.client.stream(
        "POST",
        f"/conversations/{conversation_id}/messages",
        headers=headers,
        json={"content": "q"},
    ) as response:
        first = await read_events(response.aiter_lines(), stop_after=1)
        assert first[0].event == "token"
    # Leaving the block closes the connection mid-stream.

    await wait_until_free(live, conversation_id)
    assert [m.role for m in await stored(conversation_id)] == [MessageRole.USER]
    assert live.primary.closed_streams == 1

    events = await send(live, headers, conversation_id, "again")
    assert events[-1].event == "done"


async def test_second_message_while_generating_is_409(live: Live) -> None:
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)
    live.primary.script(stall_after("thinking "))
    url = f"/conversations/{conversation_id}/messages"

    async with live.client.stream("POST", url, headers=headers, json={"content": "1"}) as first:
        # Keep a reference: a dropped aiter_lines() generator gets finalized, which closes the
        # response, i.e. the client would disconnect (and correctly free the slot).
        lines = first.aiter_lines()
        await read_events(lines, stop_after=1)
        assert live.locks.is_busy(uuid.UUID(conversation_id))
        second = await live.client.post(url, headers=headers, json={"content": "2"})

    assert second.status_code == 409
    assert second.json()["code"] == "conversation_busy"
    assert [m.content for m in await stored(conversation_id)] == ["1"]
    await wait_until_free(live, conversation_id)


async def test_someone_elses_or_missing_conversation_is_404(live: Live) -> None:
    owner = await consented_user(live)
    stranger = await consented_user(live)
    foreign = await new_conversation(live, owner)

    for conversation_id in (foreign, str(uuid.uuid4())):
        url = f"/conversations/{conversation_id}/messages"
        listed = await live.client.get(url, headers=stranger)
        posted = await live.client.post(url, headers=stranger, json={"content": "hi"})
        for response in (listed, posted):
            assert response.status_code == 404, response.text
            assert response.json() == {"code": "not_found", "message": "Conversation not found"}

    assert await stored(foreign) == []
    assert (await live.client.get("/conversations", headers=stranger)).json()["items"] == []


@pytest.mark.parametrize("content", ["", "   \n ", "x" * 4001])
async def test_content_must_be_1_to_4000_chars_after_strip(live: Live, content: str) -> None:
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)

    response = await live.client.post(
        f"/conversations/{conversation_id}/messages", headers=headers, json={"content": content}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert "xxxxx" not in response.json()["message"]  # the input is never echoed
    assert await stored(conversation_id) == []


async def test_4000_chars_is_accepted(live: Live) -> None:
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)

    events = await send(live, headers, conversation_id, "y" * 4000 + "   ")

    assert events[-1].event == "done"
    assert len((await stored(conversation_id))[0].content) == 4000


async def test_conversations_paginate_by_last_activity(live: Live) -> None:
    headers = await consented_user(live)
    oldest, middle, newest = [await new_conversation(live, headers) for _ in range(3)]
    await send(live, headers, oldest, "bump")  # now the most recently active

    first = (await live.client.get("/conversations?limit=2", headers=headers)).json()
    second = (
        await live.client.get(
            f"/conversations?limit=2&cursor={first['next_cursor']}", headers=headers
        )
    ).json()

    assert [c["id"] for c in first["items"]] == [oldest, newest]
    assert [c["id"] for c in second["items"]] == [middle]
    assert second["next_cursor"] is None


async def test_messages_paginate_newest_first(live: Live) -> None:
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)
    for i in range(3):
        live.primary.script(success(f"a{i}"))
        await send(live, headers, conversation_id, f"q{i}")
    url = f"/conversations/{conversation_id}/messages"

    first = (await live.client.get(f"{url}?limit=4", headers=headers)).json()
    second = (
        await live.client.get(f"{url}?limit=4&cursor={first['next_cursor']}", headers=headers)
    ).json()

    assert [m["content"] for m in first["items"]] == ["a2", "q2", "a1", "q1"]
    assert [m["content"] for m in second["items"]] == ["a0", "q0"]
    assert second["next_cursor"] is None
    assert first["items"][0]["provider"] == "gemini"


@pytest.mark.parametrize("query", ["cursor=not-a-cursor", "limit=0", "limit=101"])
async def test_bad_pagination_params_are_422(live: Live, query: str) -> None:
    headers = await consented_user(live)

    response = await live.client.get(f"/conversations?{query}", headers=headers)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


async def test_idle_stream_sends_heartbeat_comments(
    live: Live, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FastAPI's keep-alive interval (15 s by default) is read at stream time.
    monkeypatch.setattr(fastapi.routing, "_PING_INTERVAL", 0.05)
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)
    live.primary.script(hang())

    async with live.client.stream(
        "POST",
        f"/conversations/{conversation_id}/messages",
        headers=headers,
        json={"content": "q"},
    ) as response:
        events = await read_events(response.aiter_lines(), stop_after=2)

    assert [(e.event, e.data["text"]) for e in events] == [("comment", "ping")] * 2
    await wait_until_free(live, conversation_id)


async def summary_of(conversation_id: str) -> tuple[str | None, uuid.UUID | None]:
    async with get_sessionmaker()() as session:
        conversation = await session.get(Conversation, uuid.UUID(conversation_id))
        assert conversation is not None
        return conversation.summary, conversation.summary_upto_message_id


async def test_long_conversation_is_summarized_after_the_answer_and_fits_the_budget(
    live: Live,
) -> None:
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)
    async with get_sessionmaker()() as session:
        conversation = await session.get(Conversation, uuid.UUID(conversation_id))
        assert conversation is not None
        repo = SqlConversationRepository(session)
        for i in range(20):  # 40 messages of ~270 tokens: ~10.8K, well over 6000
            await repo.add_user_message(conversation.user_id, conversation.id, f"q{i:02d} " * 200)
            await repo.add_assistant_message(
                conversation.user_id,
                conversation.id,
                f"a{i:02d} " * 200,
                provider="gemini",
                model="flash",
                tokens_in=None,
                tokens_out=None,
                llm_call_ids=(),
            )
    live.backup.script(success("SUMMARY: oily roots, washes daily."))  # the chain's last link

    events = await send(live, headers, conversation_id, "And what about dry ends?")
    assert events[-1].event == "done"
    for _ in range(200):  # the background task runs after the stream has ended
        summary, upto = await summary_of(conversation_id)
        if summary is not None:
            break
        await asyncio.sleep(0.02)
    assert summary == "SUMMARY: oily roots, washes daily."
    folded = await stored(conversation_id)
    upto_index = next(i for i, m in enumerate(folded) if m.id == upto)
    assert folded[0].content in live.backup.received[0][1]["content"]

    await send(live, headers, conversation_id, "Thanks!")

    context = live.primary.received[-1]
    assert (
        sum(message_tokens(m["content"]) for m in context) <= get_settings().context_budget_tokens
    )
    assert "SUMMARY: oily roots, washes daily." in context[0]["content"]
    sent = [m["content"] for m in context[1:]]
    assert all(m.content not in sent for m in folded[: upto_index + 1])
    assert sent[-1] == "Thanks!"


async def test_done_lists_the_knowledge_base_articles_used_for_the_answer(live: Live) -> None:
    question = "Що робити з лупою? (sse kb test)"
    article = parse_article(
        "sse-dandruff",
        "---\ntitle: Перхоть\nlanguage: ru\nstatus: draft\ntopics: [dandruff]\n"
        "license: internal\nurl: https://example.test/kb/dandruff\n---\n\n"
        "## Уход\n\nМягкий шампунь.\n",
    )
    async with get_sessionmaker()() as session:
        # The fake embedder maps the question to text_vector(question): similarity 1.0.
        await SqlKBRepository(session).replace(
            article,
            [Chunk(0, "Перхоть › Уход\n\nМягкий шампунь.")],
            [text_vector(question)],
            model="fake-embedding",
        )
    headers = await consented_user(live)
    conversation_id = await new_conversation(live, headers)
    live.primary.script(success("Мийте м'яким шампунем", '\n<kb_used ids="1"/>'))

    events = await send(live, headers, conversation_id, question)

    assert events[-1].event == "done"
    assert events[-1].data["sources"] == [
        {"title": "Перхоть", "slug": "sse-dandruff", "url": "https://example.test/kb/dandruff"}
    ]
    system = live.primary.received[0][0]["content"]
    assert "Мягкий шампунь." in system
    tokens = [e.data["text"] for e in events if e.event == "token"]
    assert "".join(tokens) == "Мийте м'яким шампунем"
