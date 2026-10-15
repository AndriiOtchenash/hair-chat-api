import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import (
    Consent,
    ConsentType,
    Conversation,
    LLMCall,
    LLMCallStatus,
    Message,
    MessageRole,
    User,
)
from tests.integration.conftest import PROJECT_ROOT


def _alembic(database_url: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=PROJECT_ROOT,
        env={**os.environ, "DATABASE_URL": database_url},
        capture_output=True,
        text=True,
    )


@pytest.fixture
async def session(database_url: str) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(database_url)
    async with async_sessionmaker(engine, expire_on_commit=False)() as s:
        yield s
    await engine.dispose()


def _user() -> User:
    return User(auth_sub=f"auth0|{uuid.uuid4()}")


def test_migrations_roundtrip_and_match_models(database_url: str) -> None:
    for args in (("downgrade", "base"), ("upgrade", "head"), ("check",)):
        result = _alembic(database_url, *args)
        assert result.returncode == 0, f"alembic {' '.join(args)} failed:\n{result.stderr}"


async def test_deleting_user_cascades_to_all_their_data(session: AsyncSession) -> None:
    user = _user()
    session.add(user)
    await session.flush()
    conversation = Conversation(user_id=user.id)
    session.add_all([conversation, Consent(user_id=user.id, type=ConsentType.AGE_18, version="1")])
    await session.flush()
    message = Message(
        conversation_id=conversation.id, user_id=user.id, role=MessageRole.USER, content="hi"
    )
    session.add(message)
    await session.flush()
    conversation.summary_upto_message_id = message.id  # exercise the cyclic FK too
    call = LLMCall(message_id=message.id, provider="fake", model="m", status=LLMCallStatus.OK)
    session.add(call)
    await session.commit()

    # Raw DELETE: the cascade must come from Postgres, not from the ORM.
    await session.execute(text("DELETE FROM users WHERE id = :id"), {"id": user.id})
    await session.commit()

    for model, column in (
        (Consent, Consent.user_id),
        (Conversation, Conversation.user_id),
        (Message, Message.user_id),
    ):
        count = await session.scalar(
            select(func.count()).select_from(model).where(column == user.id)
        )
        assert count == 0, f"{model.__tablename__} rows survived user deletion"
    assert await session.scalar(select(func.count()).where(LLMCall.id == call.id)) == 0


async def test_duplicate_auth_sub_is_rejected(session: AsyncSession) -> None:
    first = _user()
    session.add(first)
    await session.commit()

    session.add(User(auth_sub=first.auth_sub))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_kb_chunks_embedding_has_hnsw_cosine_index(session: AsyncSession) -> None:
    indexdefs = (
        await session.scalars(text("SELECT indexdef FROM pg_indexes WHERE tablename = 'kb_chunks'"))
    ).all()

    assert any("USING hnsw" in d and "(embedding vector_cosine_ops)" in d for d in indexdefs), (
        indexdefs
    )
