"""FastAPI wiring of the chat: process-wide LLM chain and locks, per-request services."""

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import api_error
from app.db.repositories.conversations import (
    ConversationRepository,
    SqlConversationRepository,
    sql_conversation_repositories,
)
from app.db.repositories.llm_calls import SqlLLMCallRecorder
from app.db.session import get_session, get_sessionmaker
from app.llm.registry import build_llm
from app.llm.resilient import ResilientLLM
from app.services.chat_service import ChatService, ConversationLocks, ConversationService
from app.services.context_builder import TokenBudgetContextBuilder
from app.services.prompts import FilePromptProvider, PromptProvider
from app.services.summarizer import Summarizer, SummaryRunner
from app.services.tokens import context_budget

logger = logging.getLogger(__name__)


@lru_cache
def get_conversation_locks() -> ConversationLocks:
    return ConversationLocks()


@lru_cache
def _llm() -> ResilientLLM:
    # One chain per process: cooldown state and HTTP connection pools are shared by requests.
    return build_llm(get_settings(), SqlLLMCallRecorder(get_sessionmaker()))


def get_llm() -> ResilientLLM:
    try:
        return _llm()
    except ValueError as exc:  # missing key / unknown model; message names settings only
        logger.error("llm not configured: %s", exc)
        raise api_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "llm_not_configured",
            "The assistant is not configured",
        ) from exc


async def close_llm() -> None:
    if _llm.cache_info().currsize:
        await _llm().aclose()
        _llm.cache_clear()


@lru_cache
def get_prompt_provider() -> PromptProvider:
    # Stage 2.7 swaps this for a DbPromptProvider with versions edited in the admin UI.
    return FilePromptProvider()


LLM = Annotated[ResilientLLM, Depends(get_llm)]


def get_context_budget(llm: LLM, settings: Annotated[Settings, Depends(get_settings)]) -> int:
    try:
        return context_budget(
            settings.context_budget_tokens, llm.max_context_tokens, settings.chat_max_answer_tokens
        )
    except ValueError as exc:
        logger.error("llm not configured: %s", exc)
        raise api_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "llm_not_configured",
            "The assistant is not configured",
        ) from exc


Budget = Annotated[int, Depends(get_context_budget)]


def get_summarizer(llm: LLM, budget: Budget) -> SummaryRunner:
    # Its own sessions: it runs as a background task, after the request session is closed.
    return Summarizer(
        sql_conversation_repositories(get_sessionmaker()),
        llm.providers[-1],  # the cheapest link of the chain
        SqlLLMCallRecorder(get_sessionmaker()),
        budget_tokens=budget,
    )


def get_conversation_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ConversationRepository:
    return SqlConversationRepository(session)


Conversations = Annotated[ConversationRepository, Depends(get_conversation_repository)]


def get_conversation_service(repo: Conversations) -> ConversationService:
    return ConversationService(repo)


def get_chat_service(
    repo: Conversations,
    llm: LLM,
    budget: Budget,
    prompts: Annotated[PromptProvider, Depends(get_prompt_provider)],
    locks: Annotated[ConversationLocks, Depends(get_conversation_locks)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatService:
    return ChatService(
        ConversationService(repo),
        llm,
        TokenBudgetContextBuilder(repo, prompts),
        locks,
        max_answer_tokens=settings.chat_max_answer_tokens,
        budget_tokens=budget,
    )
