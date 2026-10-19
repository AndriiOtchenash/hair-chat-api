"""Terminal chat through the real ChatService: python -m app.cli.chat [--conversation ID]

Dev only. Uses the local DB (DATABASE_URL) and the real LLM chain. Acts as the dev user
`dev|local`, created on first run and given every consent of CONSENT_VERSION. This bypasses
Auth0 only here, in a process that already has DB access; the HTTP API has no such bypass.
Type a message and Enter; /exit or Ctrl-D quits.
"""

import argparse
import asyncio
import logging
import sys
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import TextIO

from app.core.config import Env, get_settings
from app.db.models import ConsentType
from app.db.repositories.consents import SqlConsentRepository
from app.db.repositories.conversations import SqlConversationRepository
from app.db.repositories.llm_calls import SqlLLMCallRecorder
from app.db.repositories.users import SqlUserRepository
from app.db.session import get_engine, get_sessionmaker
from app.llm.base import Restart, Token
from app.llm.registry import build_llm
from app.services.chat_service import (
    ChatDone,
    ChatError,
    ChatService,
    ConversationLocks,
    ConversationNotFound,
    ConversationService,
)
from app.services.context_builder import RecentMessagesContextBuilder

DEV_AUTH_SUB = "dev|local"
EXIT_COMMANDS = {"/exit", "/quit"}


async def converse(
    chat: ChatService,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    lines: AsyncIterator[str],
    out: TextIO,
) -> None:
    async for raw in lines:
        content = raw.strip()
        if content in EXIT_COMMANDS:
            return
        if not content:
            continue
        async with chat.reply(user_id, conversation_id, content) as reply:
            async for event in reply.events():
                match event:
                    case Token(text=text):
                        out.write(text)
                        out.flush()
                    case Restart(reason=reason):
                        out.write(f"\n[restart: {reason}]\n")
                    case ChatDone():
                        flag = " (truncated)" if event.truncated else ""
                        out.write(f"\n— {event.provider}:{event.model}{flag}\n\n")
                    case ChatError(code=code, message=message):
                        out.write(f"\n[error {code}: {message}]\n\n")


async def _stdin_lines(prompt: str) -> AsyncIterator[str]:
    while True:
        try:
            yield await asyncio.to_thread(input, prompt)
        except EOFError:
            return


async def _main(conversation: uuid.UUID | None) -> int:
    settings = get_settings()
    if settings.env is Env.PROD:
        print("app.cli.chat is dev-only and refuses to run with ENV=prod", file=sys.stderr)
        return 2
    try:
        llm = build_llm(settings, SqlLLMCallRecorder(get_sessionmaker()))
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        async with get_sessionmaker()() as session:
            user = await SqlUserRepository(session).get_or_create_by_auth_sub(DEV_AUTH_SUB)
            await SqlConsentRepository(session).grant(
                user.id, list(ConsentType), settings.consent_version
            )
            repo = SqlConversationRepository(session)
            conversations = ConversationService(repo)
            if conversation is None:
                conversation = (await conversations.create(user.id)).id
            else:
                await conversations.require(user.id, conversation)
            chat = ChatService(
                conversations,
                llm,
                RecentMessagesContextBuilder(repo),
                ConversationLocks(),
                max_answer_tokens=settings.chat_max_answer_tokens,
            )
            print(f"conversation {conversation} as {DEV_AUTH_SUB}; /exit to quit", file=sys.stderr)
            await converse(chat, user.id, conversation, _stdin_lines("> "), sys.stdout)
    except ConversationNotFound:
        print(f"conversation {conversation} not found for {DEV_AUTH_SUB}", file=sys.stderr)
        return 1
    finally:
        await llm.aclose()
        await get_engine().dispose()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli.chat", description=__doc__)
    parser.add_argument("--conversation", type=uuid.UUID, help="continue this conversation")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(_main(args.conversation))


if __name__ == "__main__":
    sys.exit(main())
