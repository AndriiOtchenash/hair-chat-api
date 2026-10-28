"""Prompt eval against the real LLM chain: tests/prompt_eval/cases.yaml -> answers to read.

    uv run python -m scripts.prompt_eval [--prompt system_v5] [--only ID ...] [--no-kb]
                                         [--skip PROVIDER:MODEL ...] [--file PATH]

Sends only our own questions from the YAML file, never user data. Not run in CI (real APIs, and
the verdict is a human one): each answer is printed with the case's `expect` list, the sources
that would go under it, and `!!` for every `avoid` substring found in it.
The system message is built like the chat builds it: the prompt file, then the <knowledge> block
from the docs/kb articles (embedded in memory by the same loader, chunker and RagKnowledge as
scripts.rag_eval), then the case's history and the question. --no-kb skips the embeddings API.
No database: llm_calls are not recorded. Exit code 1 only if the chain fails on a case.
"""

import argparse
import asyncio
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO

import yaml

from app.core.config import Settings, get_settings
from app.db.repositories.llm_calls import LLMCallRecord
from app.llm.base import AllProvidersDown, ChatMessage, Done, LLMRequestError, Restart, Token
from app.llm.registry import build_llm
from app.llm.resilient import ResilientLLM
from app.rag.retriever import Knowledge, RagKnowledge
from app.services.context_builder import KNOWLEDGE_NOTE, defuse_tags, wrap_data
from app.services.prompts import PROMPT_VERSION, FilePromptProvider
from scripts.rag_eval import DEFAULT_KB, embed_all

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "tests/prompt_eval/cases.yaml"
EVAL_USER = uuid.UUID(int=0)
RULE = "─" * 72

LLMFactory = Callable[[Settings, Sequence[str]], ResilientLLM]


@dataclass(frozen=True)
class Case:
    id: str
    group: str
    lang: str
    text: str
    expect: tuple[str, ...]
    avoid: tuple[str, ...] = ()
    history: tuple[ChatMessage, ...] = ()


@dataclass
class Answer:
    text: str = ""
    by: str = "-"
    restarts: int = 0
    truncated: bool = False
    error: str | None = None
    sources: tuple[str, ...] = ()
    flagged: list[str] = field(default_factory=list)


class NoRecorder:
    """llm_calls need the DB; the eval runs without one."""

    async def record(self, call: LLMCallRecord) -> None:
        return None


def _real_llm(settings: Settings, skip: Sequence[str]) -> ResilientLLM:
    return build_llm(settings, NoRecorder(), skip)


def _turn(row: dict[str, Any]) -> ChatMessage:
    role: Literal["user", "assistant"] = "assistant" if row["role"] == "assistant" else "user"
    return {"role": role, "content": row["content"]}


def load(path: Path) -> list[Case]:
    rows: list[dict[str, Any]] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Case(
            id=r["id"],
            group=r["group"],
            lang=r["lang"],
            text=r["text"],
            expect=tuple(r["expect"]),
            avoid=tuple(r.get("avoid") or ()),
            history=tuple(_turn(t) for t in r.get("history") or ()),
        )
        for r in rows
    ]


def messages(prompt: str, case: Case, knowledge: Knowledge | None) -> list[ChatMessage]:
    """Same shape as TokenBudgetContextBuilder: one system message, history, the question."""
    system = prompt
    if knowledge is not None:
        system = "\n\n".join([prompt, wrap_data("knowledge", knowledge.text, KNOWLEDGE_NOTE)])
    question: ChatMessage = {"role": "user", "content": case.text}
    turns: list[ChatMessage] = [
        {"role": t["role"], "content": defuse_tags(t["content"])} for t in [*case.history, question]
    ]
    return [{"role": "system", "content": system}, *turns]


async def knowledge_for(cases: Sequence[Case], kb_dir: Path) -> list[Knowledge | None]:
    settings = get_settings()
    kb, embeddings = await embed_all(kb_dir, [c.text for c in cases])
    rag = RagKnowledge(
        kb,
        top_k=settings.kb_top_k,
        min_similarity=settings.kb_min_similarity,
        max_tokens=settings.kb_max_tokens,
        include_drafts=True,
    )
    return [
        await rag.knowledge(EVAL_USER, c.text, e) for c, e in zip(cases, embeddings, strict=True)
    ]


async def answer(llm: ResilientLLM, chat: list[ChatMessage], max_tokens: int, case: Case) -> Answer:
    result = Answer()
    parts: list[str] = []
    try:
        async for event in llm.stream(chat, max_tokens):
            match event:
                case Token(text=text):
                    parts.append(text)
                case Restart():
                    parts.clear()
                    result.restarts += 1
                case Done(provider=provider, model=model, truncated=truncated):
                    result.by = f"{provider}:{model}"
                    result.truncated = truncated
    except (AllProvidersDown, LLMRequestError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    result.text = "".join(parts).strip()
    lowered = result.text.lower()
    result.flagged = [a for a in case.avoid if a.lower() in lowered]
    return result


def report(case: Case, result: Answer, out: TextIO) -> None:
    print(RULE, file=out)
    print(f"[{case.group}] {case.id} ({case.lang})", file=out)
    for turn in case.history:
        print(f"  {turn['role']}: {turn['content']}", file=out)
    print(f"Q: {case.text}", file=out)
    print("expect:", file=out)
    for item in case.expect:
        print(f"  - {item}", file=out)
    print(f"sources: {', '.join(result.sources) or '-'}", file=out)
    if result.error:
        print(f"ERROR {result.error}", file=out)
        return
    notes = [result.by, f"{len(result.text.split())} words"]
    if result.restarts:
        notes.append(f"{result.restarts} restarts")
    if result.truncated:
        notes.append("truncated")
    print(f"A ({', '.join(notes)}):", file=out)
    print(result.text, file=out)
    for found in result.flagged:
        print(f"!! contains {found!r}", file=out)


async def run(
    cases: Sequence[Case],
    *,
    prompt_version: str,
    kb_dir: Path | None,
    skip: Sequence[str],
    llm_factory: LLMFactory = _real_llm,
    out: TextIO = sys.stdout,
) -> int:
    settings = get_settings()
    prompt = await FilePromptProvider(prompt_version).get_active()
    knowledge = await knowledge_for(cases, kb_dir) if kb_dir else [None] * len(cases)
    llm = llm_factory(settings, skip)
    chain = ", ".join(f"{p.name}:{p.model}" for p in llm.providers)
    print(f"prompt {prompt.version}, chain {chain}, {len(cases)} cases", file=out)
    results: list[Answer] = []
    try:
        for case, found in zip(cases, knowledge, strict=True):
            result = await answer(
                llm, messages(prompt.text, case, found), settings.chat_max_answer_tokens, case
            )
            result.sources = tuple(s.slug for s in found.sources) if found else ()
            report(case, result, out)
            results.append(result)
    finally:
        await llm.aclose()
    errors = sum(r.error is not None for r in results)
    flagged = sum(bool(r.flagged) for r in results)
    print(RULE, file=out)
    print(f"{len(results)} answers, {flagged} with !! flags, {errors} errors", file=out)
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.prompt_eval", description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--prompt", default=PROMPT_VERSION, help="prompt file, e.g. system_v4")
    parser.add_argument("--only", action="append", default=[], metavar="ID")
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB)
    parser.add_argument("--no-kb", action="store_true", help="no <knowledge>, no embeddings API")
    parser.add_argument("--skip", action="append", default=[], metavar="PROVIDER:MODEL")
    args = parser.parse_args(argv)
    cases = load(args.file)
    if args.only:
        unknown = set(args.only) - {c.id for c in cases}
        if unknown:
            parser.error(f"unknown case ids: {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c.id in args.only]
    return asyncio.run(
        run(
            cases,
            prompt_version=args.prompt,
            kb_dir=None if args.no_kb else args.kb,
            skip=args.skip,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
