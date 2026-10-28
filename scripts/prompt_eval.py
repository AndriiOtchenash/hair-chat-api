"""Prompt eval against the real LLM chain: tests/prompt_eval/cases.yaml -> answers to read.

    uv run python -m scripts.prompt_eval [--prompt system_v5] [--only ID ...] [--no-kb]
                                         [--skip PROVIDER:MODEL ... | --link PROVIDER:MODEL]
                                         [--file PATH]

Sends only our own questions from the YAML file, never user data. Not run in CI (real APIs, and
the verdict is a human one): each answer is printed with the case's `expect` list, the passages
of its <knowledge> block (`kb_context`, or the embeddings/retrieval error that left it empty),
the passage numbers of its <kb_used> tag (cut out of the answer as the chat does), the sources
that would go under it, `!!` for every `avoid` substring found in it and `!!` when the tag is
there or missing against the case's `kb_used`. A case that expects the tag but got no passages
is marked `??` instead: its answer says nothing about the prompt.
The system message is built like the chat builds it: the prompt file, then the <knowledge> block
from the docs/kb articles (embedded in memory by the same loader, chunker and RagKnowledge as
scripts.rag_eval) followed by the prompt's knowledge reminder, then the case's history and the
question. --no-kb skips the embeddings API. --link runs one link of LLM_CHAIN, no fallback.
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
from app.llm.base import (
    AllProvidersDown,
    ChatMessage,
    Done,
    LLMRequestError,
    Restart,
    Token,
    link_key,
)
from app.llm.registry import build_llm, parse_chain
from app.llm.resilient import ResilientLLM
from app.rag.retriever import Knowledge, RagKnowledge
from app.services.context_builder import KNOWLEDGE_NOTE, DataBlock, defuse_tags, system_message
from app.services.kb_used import KbUsedFilter, used_sources
from app.services.prompts import PROMPT_VERSION, FilePromptProvider, SystemPrompt
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
    # Whether the answer should end with a <kb_used> tag; None = not checked.
    kb_used: bool | None = None


@dataclass(frozen=True)
class KbContext:
    """What a case's <knowledge> block held, or why there was none."""

    knowledge: Knowledge | None = None
    error: str | None = None  # embeddings or retrieval failed
    off: bool = False  # --no-kb

    def describe(self) -> str:
        if self.off:
            return "off (--no-kb)"
        if self.error:
            return f"0 — ERROR {self.error}"
        passages = self.knowledge.passages if self.knowledge else ()
        titles = "; ".join(f"[{n}] {s.title}" for n, s in enumerate(passages, start=1))
        return f"{len(passages)}" + (f" — {titles}" if titles else "")


NO_KB = KbContext(off=True)


@dataclass
class Answer:
    text: str = ""
    by: str = "-"
    restarts: int = 0
    truncated: bool = False
    error: str | None = None
    kb_ids: tuple[int, ...] = ()
    sources: tuple[str, ...] = ()
    flagged: list[str] = field(default_factory=list)
    # The case cannot be judged, e.g. it expects a <kb_used> tag but got no passages.
    unsure: list[str] = field(default_factory=list)


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
            kb_used=r.get("kb_used"),
        )
        for r in rows
    ]


def messages(prompt: SystemPrompt, case: Case, knowledge: Knowledge | None) -> list[ChatMessage]:
    """Same shape as TokenBudgetContextBuilder: one system message, history, the question."""
    blocks = [DataBlock("knowledge", knowledge.text, KNOWLEDGE_NOTE)] if knowledge else []
    system = system_message(prompt, blocks)
    question: ChatMessage = {"role": "user", "content": case.text}
    turns: list[ChatMessage] = [
        {"role": t["role"], "content": defuse_tags(t["content"])} for t in [*case.history, question]
    ]
    return [{"role": "system", "content": system}, *turns]


async def knowledge_for(cases: Sequence[Case], kb_dir: Path) -> list[KbContext]:
    """Never raises for API failures: the cases then run without passages, with the error."""
    settings = get_settings()
    try:
        kb, embeddings = await embed_all(kb_dir, [c.text for c in cases])
    except Exception as exc:
        return [KbContext(error=f"embeddings: {type(exc).__name__}: {exc}")] * len(cases)
    rag = RagKnowledge(
        kb,
        top_k=settings.kb_top_k,
        min_similarity=settings.kb_min_similarity,
        max_tokens=settings.kb_max_tokens,
        include_drafts=True,
    )
    contexts: list[KbContext] = []
    for case, embedding in zip(cases, embeddings, strict=True):
        try:
            contexts.append(KbContext(await rag.knowledge(EVAL_USER, case.text, embedding)))
        except Exception as exc:
            contexts.append(KbContext(error=f"retrieval: {type(exc).__name__}: {exc}"))
    return contexts


async def answer(
    llm: ResilientLLM,
    chat: list[ChatMessage],
    max_tokens: int,
    case: Case,
    knowledge: Knowledge | None = None,
) -> Answer:
    result = Answer()
    parts: list[str] = []
    tag = KbUsedFilter()
    try:
        async for event in llm.stream(chat, max_tokens):
            match event:
                case Token(text=text):
                    parts.append(tag.feed(text))
                case Restart():
                    parts.clear()
                    tag = KbUsedFilter()
                    result.restarts += 1
                case Done(provider=provider, model=model, truncated=truncated):
                    parts.append(tag.finish())
                    result.by = f"{provider}:{model}"
                    result.truncated = truncated
    except (AllProvidersDown, LLMRequestError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    result.text = "".join(parts).strip()
    result.kb_ids = tuple(sorted(tag.ids))
    passages = knowledge.passages if knowledge else ()
    result.sources = tuple(s.slug for s in used_sources(tag.ids, passages))
    lowered = result.text.lower()
    result.flagged = [f"contains {a!r}" for a in case.avoid if a.lower() in lowered]
    if case.kb_used and knowledge is None:
        # No passages to rely on: a missing tag is right, so it tells nothing about the prompt.
        result.unsure.append("no kb context")
    elif case.kb_used is not None and case.kb_used != bool(tag.ids):
        result.flagged.append("has a <kb_used> tag" if tag.ids else "has no <kb_used> tag")
    return result


def report(case: Case, kb: KbContext, result: Answer, out: TextIO) -> None:
    print(RULE, file=out)
    print(f"[{case.group}] {case.id} ({case.lang})", file=out)
    for turn in case.history:
        print(f"  {turn['role']}: {turn['content']}", file=out)
    print(f"Q: {case.text}", file=out)
    print("expect:", file=out)
    for item in case.expect:
        print(f"  - {item}", file=out)
    print(f"kb_context: {kb.describe()}", file=out)
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
    print(f"kb_used: {', '.join(map(str, result.kb_ids)) or '-'}", file=out)
    print(f"sources: {', '.join(result.sources) or '-'}", file=out)
    for flag in result.flagged:
        print(f"!! {flag}", file=out)
    for note in result.unsure:
        print(f"?? {note}", file=out)


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
    contexts = await knowledge_for(cases, kb_dir) if kb_dir else [NO_KB] * len(cases)
    llm = llm_factory(settings, skip)
    chain = ", ".join(f"{p.name}:{p.model}" for p in llm.providers)
    print(f"prompt {prompt.version}, chain {chain}, {len(cases)} cases", file=out)
    results: list[Answer] = []
    try:
        for case, kb in zip(cases, contexts, strict=True):
            chat = messages(prompt, case, kb.knowledge)
            result = await answer(llm, chat, settings.chat_max_answer_tokens, case, kb.knowledge)
            report(case, kb, result, out)
            results.append(result)
    finally:
        await llm.aclose()
    errors = sum(r.error is not None for r in results)
    flagged = sum(bool(r.flagged) for r in results)
    unsure = sum(bool(r.unsure) for r in results)
    print(RULE, file=out)
    summary = f"{len(results)} answers, {flagged} with !! flags, {errors} errors"
    if unsure:
        summary += f", {unsure} with ?? (could not be judged)"
    print(summary, file=out)
    return 1 if errors else 0


def skip_all_but(link: str, chain: str) -> list[str]:
    """--link: every other link of LLM_CHAIN is skipped, so nothing falls back."""
    keys = [link_key(provider, model) for provider, model in parse_chain(chain)]
    if link not in keys:
        raise ValueError(f"--link {link} is not in LLM_CHAIN ({', '.join(keys)})")
    return [k for k in keys if k != link]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.prompt_eval", description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--prompt", default=PROMPT_VERSION, help="prompt file, e.g. system_v4")
    parser.add_argument("--only", action="append", default=[], metavar="ID")
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB)
    parser.add_argument("--no-kb", action="store_true", help="no <knowledge>, no embeddings API")
    links = parser.add_mutually_exclusive_group()
    links.add_argument("--skip", action="append", default=[], metavar="PROVIDER:MODEL")
    links.add_argument(
        "--link", metavar="PROVIDER:MODEL", help="only this link of LLM_CHAIN, no fallback"
    )
    args = parser.parse_args(argv)
    skip: list[str] = args.skip
    if args.link:
        try:
            skip = skip_all_but(args.link, get_settings().llm_chain)
        except ValueError as exc:
            parser.error(str(exc))
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
            skip=skip,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
