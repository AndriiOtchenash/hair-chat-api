from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import delete

from app.db.models import Conversation, Message, MessageRole, User
from app.db.session import get_sessionmaker
from scripts.rag_report import Group, fetch

NOW = datetime.now(UTC)


@pytest.fixture
async def clean(app_settings_env: None) -> None:
    # Other tests share the container: start from no messages so counts are exact.
    async with get_sessionmaker()() as session:
        await session.execute(delete(Conversation))
        await session.commit()


async def _answers(*answers: tuple[str, dict[str, Any] | None, timedelta]) -> None:
    async with get_sessionmaker()() as session:
        user = User(auth_sub=f"auth0|rag-report-{NOW.timestamp()}")
        session.add(user)
        await session.flush()
        conversation = Conversation(user_id=user.id)
        session.add(conversation)
        await session.flush()
        session.add(
            Message(
                conversation_id=conversation.id,
                user_id=user.id,
                role=MessageRole.USER,
                content="question",
                retrieval={"status": "ok", "passages": 9, "kb_used": [1]},  # not an answer
            )
        )
        for provider, retrieval, age in answers:
            session.add(
                Message(
                    conversation_id=conversation.id,
                    user_id=user.id,
                    role=MessageRole.ASSISTANT,
                    content="answer",
                    provider=provider,
                    model="m",
                    retrieval=retrieval,
                    created_at=NOW - age,
                )
            )
        await session.commit()


async def test_fetch_groups_answers_of_the_period_by_link_and_outcome(clean: None) -> None:
    ok_used = {"status": "ok", "passages": 2, "kb_used": [2]}
    ok_unused = {"status": "ok", "passages": 2, "kb_used": []}
    error = {"status": "error", "passages": 0, "kb_used": [], "error": "timeout"}
    hour, old = timedelta(hours=1), timedelta(days=8)
    await _answers(
        ("gemini", ok_used, hour),
        ("gemini", ok_used, hour),
        ("gemini", ok_unused, hour),
        ("groq", error, hour),
        ("groq", None, hour),  # saved before messages.retrieval existed
        ("gemini", error, old),  # outside the period
    )

    async with get_sessionmaker()() as session:
        groups = await fetch(session, NOW - timedelta(days=7))

    assert sorted(groups, key=lambda g: (g.link, g.status, g.kb_used)) == [
        Group("gemini:m", "ok", False, None, 1),
        Group("gemini:m", "ok", True, None, 2),
        Group("groq:m", "error", False, "timeout", 1),
        Group("groq:m", "unknown", False, None, 1),
    ]
