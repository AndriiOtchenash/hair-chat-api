"""Manual check of the real provider chain: python -m app.llm.cli "question" [--skip p:m ...]

Streams the answer to stdout and prints which link answered. Fallback warnings go to stderr.
Attempts are written to llm_calls when the DB is reachable (otherwise only a warning).
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Callable, Sequence
from typing import TextIO

from app.core.config import Settings, get_settings
from app.db.repositories.llm_calls import SqlLLMCallRecorder
from app.db.session import get_engine, get_sessionmaker
from app.llm.base import AllProvidersDown, ChatMessage, Done, LLMRequestError, Restart, Token
from app.llm.registry import build_llm
from app.llm.resilient import ResilientLLM

LLMFactory = Callable[[Settings, Sequence[str]], ResilientLLM]


def _real_llm(settings: Settings, skip: Sequence[str]) -> ResilientLLM:
    return build_llm(settings, SqlLLMCallRecorder(get_sessionmaker()), skip)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.llm.cli", description=__doc__)
    parser.add_argument("question")
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="PROVIDER:MODEL",
        help="drop this link from LLM_CHAIN (repeatable) to watch the fallback",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser


async def run(
    question: str,
    skip: Sequence[str],
    max_tokens: int,
    *,
    llm_factory: LLMFactory = _real_llm,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    try:
        llm = llm_factory(get_settings(), skip)
    except ValueError as exc:
        print(f"config error: {exc}", file=err)
        return 2
    chain = ", ".join(f"{p.name}:{p.model}" for p in llm.providers)
    print(f"chain: {chain} (context budget {llm.max_context_tokens} tokens)", file=err)
    messages: list[ChatMessage] = [{"role": "user", "content": question}]
    try:
        async for event in llm.stream(messages, max_tokens):
            match event:
                case Token(text=text):
                    out.write(text)
                    out.flush()
                case Restart(reason=reason):
                    out.write(f"\n[restart: {reason}]\n")
                case Done(provider=provider, model=model):
                    out.write(f"\n\n— answered by {provider}:{model}\n")
    except AllProvidersDown as exc:
        print(f"\n{exc}", file=err)
        return 1
    except LLMRequestError as exc:
        print(f"\nrequest rejected (no fallback): {exc}", file=err)
        return 2
    finally:
        await llm.aclose()
    return 0


async def _main(args: argparse.Namespace) -> int:
    try:
        return await run(args.question, args.skip, args.max_tokens)
    finally:
        await get_engine().dispose()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(_main(_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
