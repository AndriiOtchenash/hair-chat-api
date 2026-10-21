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
from app.db.repositories.kb import SqlKBRepository
from app.db.repositories.llm_calls import SqlLLMCallRecorder
from app.db.session import get_session, get_sessionmaker
from app.llm.moderation import Moderator, NoopModerator, OpenAIModerator
from app.llm.registry import build_llm
from app.llm.resilient import ResilientLLM
from app.rag.embeddings import Embedder, OpenAICompatibleEmbedder
from app.rag.retriever import NoQueryEmbedder, QueryEmbedder, RagKnowledge, RagQueryEmbedder
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


def build_moderator(settings: Settings) -> Moderator:
    if settings.openai_api_key is None:  # dev only: Settings refuses prod without the key
        return NoopModerator()
    return OpenAIModerator(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.moderation_timeout_s,
    )


@lru_cache
def get_moderator() -> Moderator:
    # One client per process: its HTTP connection pool is shared by requests.
    return build_moderator(get_settings())


def build_embedder(settings: Settings) -> Embedder | None:
    """None without GEMINI_API_KEY: replies go without the knowledge base."""
    if settings.gemini_api_key is None:
        return None
    return OpenAICompatibleEmbedder(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        api_key=settings.gemini_api_key.get_secret_value(),
        timeout=settings.embedding_timeout_s,
    )


@lru_cache
def get_embedder() -> Embedder | None:
    # One client per process: its HTTP connection pool is shared by requests.
    embedder = build_embedder(get_settings())
    if embedder is None:
        logger.warning("GEMINI_API_KEY not set: answers go without the knowledge base")
    return embedder


def get_query_embedder(
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> QueryEmbedder:
    if embedder is None:
        return NoQueryEmbedder()
    return RagQueryEmbedder(embedder, timeout=settings.embedding_timeout_s)


def build_knowledge(session: AsyncSession, settings: Settings) -> RagKnowledge:
    return RagKnowledge(
        SqlKBRepository(session),
        top_k=settings.kb_top_k,
        min_similarity=settings.kb_min_similarity,
        max_tokens=settings.kb_max_tokens,
        include_drafts=settings.kb_drafts_searched,
    )


async def close_llm() -> None:
    if _llm.cache_info().currsize:
        await _llm().aclose()
        _llm.cache_clear()
    if get_moderator.cache_info().currsize:
        await get_moderator().aclose()
        get_moderator.cache_clear()
    if get_embedder.cache_info().currsize:
        embedder = get_embedder()
        if embedder is not None:
            await embedder.aclose()
        get_embedder.cache_clear()


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
    session: Annotated[AsyncSession, Depends(get_session)],
    repo: Conversations,
    llm: LLM,
    budget: Budget,
    prompts: Annotated[PromptProvider, Depends(get_prompt_provider)],
    locks: Annotated[ConversationLocks, Depends(get_conversation_locks)],
    moderator: Annotated[Moderator, Depends(get_moderator)],
    query_embedder: Annotated[QueryEmbedder, Depends(get_query_embedder)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatService:
    return ChatService(
        ConversationService(repo),
        llm,
        TokenBudgetContextBuilder(repo, prompts, knowledge=build_knowledge(session, settings)),
        locks,
        moderator,
        max_answer_tokens=settings.chat_max_answer_tokens,
        budget_tokens=budget,
        violence_threshold=settings.moderation_violence_threshold,
        query_embedder=query_embedder,
    )
