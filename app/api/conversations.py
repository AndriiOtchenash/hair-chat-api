import json
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.core.chat import get_chat_service, get_conversation_service
from app.core.errors import api_error
from app.core.security import CurrentUser
from app.db.models import MessageRole
from app.db.pagination import Cursor, InvalidCursor
from app.llm.base import Restart, Token
from app.services.chat_service import (
    ChatDone,
    ChatError,
    ChatEvent,
    ChatService,
    ConversationBusy,
    ConversationNotFound,
    ConversationService,
    Reply,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])

Conversations = Annotated[ConversationService, Depends(get_conversation_service)]
Chat = Annotated[ChatService, Depends(get_chat_service)]
Limit = Annotated[int, Query(ge=1, le=100)]
CursorParam = Annotated[str | None, Query(description="next_cursor of the previous page")]

MAX_CONTENT_CHARS = 4000


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    created_at: datetime
    last_message_at: datetime


class ConversationsPage(BaseModel):
    items: list[ConversationOut]
    next_cursor: str | None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    provider: str | None
    model: str | None
    created_at: datetime


class MessagesPage(BaseModel):
    """Newest first; follow next_cursor for older messages."""

    items: list[MessageOut]
    next_cursor: str | None


class SendMessageRequest(BaseModel):
    content: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_CONTENT_CHARS)
    ]


def _cursor(value: str | None) -> Cursor | None:
    if value is None:
        return None
    try:
        return Cursor.decode(value)
    except InvalidCursor:
        raise api_error(422, "validation_error", "query.cursor: invalid cursor") from None


def _not_found() -> Exception:
    return api_error(status.HTTP_404_NOT_FOUND, "not_found", "Conversation not found")


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(user: CurrentUser, conversations: Conversations) -> ConversationOut:
    return ConversationOut.model_validate(await conversations.create(user.id))


@router.get("")
async def list_conversations(
    user: CurrentUser,
    conversations: Conversations,
    limit: Limit = 20,
    cursor: CursorParam = None,
) -> ConversationsPage:
    page = await conversations.list(user.id, limit, _cursor(cursor))
    return ConversationsPage(
        items=[ConversationOut.model_validate(c) for c in page.items],
        next_cursor=page.next_cursor,
    )


@router.get("/{conversation_id}/messages")
async def list_messages(
    conversation_id: uuid.UUID,
    user: CurrentUser,
    conversations: Conversations,
    limit: Limit = 20,
    cursor: CursorParam = None,
) -> MessagesPage:
    try:
        page = await conversations.list_messages(user.id, conversation_id, limit, _cursor(cursor))
    except ConversationNotFound:
        raise _not_found() from None
    return MessagesPage(
        items=[MessageOut.model_validate(m) for m in page.items],
        next_cursor=page.next_cursor,
    )


async def start_reply(
    conversation_id: uuid.UUID, body: SendMessageRequest, user: CurrentUser, chat: Chat
) -> AsyncIterator[Reply]:
    """Runs before the response starts, so 404/409 are real HTTP statuses, not SSE events.

    Its teardown runs after the stream ends or the client disconnects: that frees the slot.
    """
    try:
        async with chat.reply(user.id, conversation_id, body.content) as reply:
            yield reply
    except ConversationNotFound:
        raise _not_found() from None
    except ConversationBusy:
        raise api_error(
            status.HTTP_409_CONFLICT,
            "conversation_busy",
            "An answer is already being generated in this conversation",
        ) from None


def _sse(event: ChatEvent) -> ServerSentEvent:
    data: dict[str, Any]
    match event:
        case Token(text=text):
            name, data = "token", {"text": text}
        case Restart(reason=reason):
            name, data = "restart", {"reason": reason}
        case ChatDone():
            name, data = (
                "done",
                {
                    "message_id": str(event.message_id),
                    "provider": event.provider,
                    "model": event.model,
                    "truncated": event.truncated,
                },
            )
        case ChatError(code=code, message=message):
            name, data = "error", {"code": code, "message": message}  # = ErrorBody
    # raw_data: keep Cyrillic as is instead of \uXXXX escapes (json.dumps keeps it one line).
    return ServerSentEvent(event=name, raw_data=json.dumps(data, ensure_ascii=False))


@router.post(
    "/{conversation_id}/messages",
    response_class=EventSourceResponse,
    responses={
        404: {"description": "Missing or someone else's conversation"},
        409: {"description": "conversation_busy: an answer is already streaming"},
    },
)
async def send_message(
    reply: Annotated[Reply, Depends(start_reply)],
) -> AsyncIterable[ServerSentEvent]:
    """Streams the answer: token* (restart token*)* then done, or error (CLAUDE.md SSE contract).

    FastAPI adds a `: ping` comment every 15 s while idle and the no-buffering headers.
    """
    async for event in reply.events():
        yield _sse(event)
