"""Prompt eval against the real LLM chain: tests/prompt_eval/cases.yaml -> answers to read.

    uv run python -m scripts.prompt_eval [--prompt system_v5] [--only ID ...] [--no-kb]
                                         [--skip PROVIDER:MODEL ... | --link PROVIDER:MODEL]
                                         [--file PATH] [--query last|user3|summary ...]
                                         [--retrieval-only]

Sends only our own questions from the YAML file, never user data. Not run in CI (real APIs, and
the verdict is a human one): each answer is printed with the case's `expect` list, the passages
of its <knowledge> block (`kb_context`, or the embeddings/retrieval error that left it empty),
the passage numbers of its <kb_used> tag (cut out of the answer as the chat does), the sources
that would go under it, `!!` for every `avoid` substring found in it and `!!` when the tag is
there or missing against the case's `kb_used`. A case that expects the tag but got no passages
is marked `??` instead: its answer says nothing about the prompt.
Every answer gets `answer_lang` (a rough script/letter guess, `detect_lang`), flagged `!!` when it
differs from the case's `answer_lang` (default: its `lang`). The prompt names the case's
`locale` as the interface language (default: `lang` if we support it, else en, like users.locale).
The system message is built like the chat builds it: the prompt file, then the <knowledge> block
from the docs/kb articles (embedded in memory by the same loader, chunker and RagKnowledge as
scripts.rag_eval) followed by the prompt's knowledge reminder, then the case's history and the
question (a case's `summary` goes into <conversation_summary> like conversations.summary).
--no-kb skips the embeddings API. --link runs one link of LLM_CHAIN, no fallback.
--query picks what is embedded for the search (QUERIES; `chat`, the default, is what the chat
does: the latest message merged with the dialog query of app.rag.retriever).
--retrieval-only calls no LLM: it prints kb_context per --query (several may be given) and
`!!` when a passage of an article in the case's `kb_slugs` is missing.
No database: llm_calls are not recorded. Exit code 1 only if the chain fails on a case.
"""

import argparse
import asyncio
import re
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO

import yaml

from app.core.config import Settings, get_settings
from app.db.models import Locale
from app.db.repositories.kb import KBRepository
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
from app.rag.retriever import Knowledge, QueryEmbedding, RagKnowledge, dialog_query
from app.services.context_builder import (
    KNOWLEDGE_NOTE,
    SUMMARY_NOTE,
    DataBlock,
    defuse_tags,
    system_message,
)
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
    # Rolling summary of the earlier part of the conversation (conversations.summary).
    summary: str | None = None
    # Articles whose passages the search must find for `text`; !! when one is missing.
    kb_slugs: tuple[str, ...] = ()
    # The user's interface language (users.locale); None = `lang` if supported, else en.
    locale: str | None = None
    # Language the answer must be in; None = `lang`.
    answer_lang: str | None = None

    @property
    def interface(self) -> Locale:
        if self.locale is not None:
            return Locale(self.locale)
        return Locale(self.lang) if self.lang in set(Locale) else Locale.EN

    @property
    def expected_lang(self) -> str:
        return self.answer_lang or self.lang


_WORD = re.compile(r"[^\W\d_]+")
_GERMAN = {"ich", "und", "die", "der", "das", "nicht", "ist", "sie", "mit", "haare", "für"}


def detect_lang(text: str) -> str:
    """Rough guess for the eval report: uk | ru | pl | de | en, or ? for no letters.

    Cyrillic: і/ї/є/ґ vs ы/э/ъ/ё. Latin: Polish diacritics, German umlauts or common words."""
    words = _WORD.findall(text.lower())
    letters = "".join(words)
    cyrillic = sum("а" <= ch <= "я" or ch in "ёіїєґ" for ch in letters)
    if not letters:
        return "?"
    if cyrillic * 2 >= len(letters):
        uk = sum(ch in "іїєґ" for ch in letters)
        ru = sum(ch in "ыэъё" for ch in letters)
        return "uk" if uk > ru else "ru"
    if sum(ch in "ąęłśżźćń" for ch in letters) >= 2:
        return "pl"
    if sum(ch in "äöüß" for ch in letters) >= 2 or sum(w in _GERMAN for w in words) >= 2:
        return "de"
    return "en"


def _dialog(case: Case, *, summary: bool = True) -> str | None:
    earlier = [t["content"] for t in case.history if t["role"] == "user"]
    return dialog_query(case.text, summary=case.summary if summary else None, earlier_user=earlier)


# The texts embedded for the knowledge search, by --query name: one = a single search; two = the
# latest message merged with the dialog query (QueryEmbedding.context_vector, merge_hits).
QUERIES: dict[str, Callable[[Case], tuple[str, ...]]] = {
    "last": lambda c: (c.text,),  # the latest user message only (the chat before dialog search)
    "user3": lambda c: (_dialog(c, summary=False) or c.text,),  # the last 3 user messages
    "summary": lambda c: (_dialog(c) or c.text,),  # summary + latest, else user3
    "chat": lambda c: (c.text, d) if (d := _dialog(c)) else (c.text,),  # what the chat does
}


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
    lang: str = "?"  # detect_lang of the answer
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
            summary=r.get("summary"),
            kb_slugs=tuple(r.get("kb_slugs") or ()),
            locale=r.get("locale"),
            answer_lang=r.get("answer_lang"),
        )
        for r in rows
    ]


def messages(prompt: SystemPrompt, case: Case, knowledge: Knowledge | None) -> list[ChatMessage]:
    """Same shape as TokenBudgetContextBuilder: one system message, history, the question."""
    blocks = [DataBlock("knowledge", knowledge.text, KNOWLEDGE_NOTE)] if knowledge else []
    if case.summary:
        blocks.append(DataBlock("conversation_summary", case.summary, SUMMARY_NOTE))
    system = system_message(prompt, blocks, case.interface)
    question: ChatMessage = {"role": "user", "content": case.text}
    turns: list[ChatMessage] = [
        {"role": t["role"], "content": defuse_tags(t["content"])} for t in [*case.history, question]
    ]
    return [{"role": "system", "content": system}, *turns]


async def knowledge_for(
    cases: Sequence[Case], kb_dir: Path, query: str = "chat"
) -> list[KbContext]:
    """Never raises for API failures: the cases then run without passages, with the error."""
    return (await knowledge_by_query(cases, kb_dir, [query]))[query]


async def knowledge_by_query(
    cases: Sequence[Case], kb_dir: Path, queries: Sequence[str]
) -> dict[str, list[KbContext]]:
    """kb_context of every case for every QUERIES name; the KB is embedded once."""
    texts = {q: [QUERIES[q](c) for c in cases] for q in queries}
    unique = list(dict.fromkeys(t for per_case in texts.values() for ts in per_case for t in ts))
    try:
        kb, embeddings = await embed_all(kb_dir, unique)
    except Exception as exc:
        failed = [KbContext(error=f"embeddings: {type(exc).__name__}: {exc}")] * len(cases)
        return {q: failed for q in queries}
    by_text = dict(zip(unique, embeddings, strict=True))

    def query(ts: tuple[str, ...]) -> QueryEmbedding:
        first = by_text[ts[0]]
        context = by_text[ts[1]].vector if len(ts) > 1 else None
        return QueryEmbedding(first.vector, first.model, context)

    return {
        q: await retrieve(kb, [c.text for c in cases], [query(ts) for ts in texts[q]])
        for q in queries
    }


async def retrieve(
    kb: KBRepository, texts: Sequence[str], embeddings: Sequence[QueryEmbedding]
) -> list[KbContext]:
    settings = get_settings()
    rag = RagKnowledge(
        kb,
        top_k=settings.kb_top_k,
        min_similarity=settings.kb_min_similarity,
        max_tokens=settings.kb_max_tokens,
        include_drafts=True,
    )
    contexts: list[KbContext] = []
    for text, embedding in zip(texts, embeddings, strict=True):
        try:
            contexts.append(KbContext(await rag.knowledge(EVAL_USER, text, embedding)))
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
    result.lang = detect_lang(result.text)
    if result.text and result.lang != case.expected_lang:
        result.flagged.append(f"answer in {result.lang}, expected {case.expected_lang}")
    if case.kb_used and knowledge is None:
        # No passages to rely on: a missing tag is right, so it tells nothing about the prompt.
        result.unsure.append("no kb context")
    elif case.kb_used is not None and case.kb_used != bool(tag.ids):
        result.flagged.append("has a <kb_used> tag" if tag.ids else "has no <kb_used> tag")
    return result


def report(case: Case, kb: KbContext, result: Answer, out: TextIO) -> None:
    print(RULE, file=out)
    print(f"[{case.group}] {case.id} ({case.lang}, locale {case.interface})", file=out)
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
    print(f"answer_lang: {result.lang} (expected {case.expected_lang})", file=out)
    print(f"kb_used: {', '.join(map(str, result.kb_ids)) or '-'}", file=out)
    print(f"sources: {', '.join(result.sources) or '-'}", file=out)
    for flag in result.flagged:
        print(f"!! {flag}", file=out)
    for note in result.unsure:
        print(f"?? {note}", file=out)


def missing_slugs(case: Case, kb: KbContext) -> list[str]:
    found = {s.slug for s in kb.knowledge.passages} if kb.knowledge else set()
    return [slug for slug in case.kb_slugs if slug not in found]


def report_retrieval(case: Case, contexts: dict[str, KbContext], out: TextIO) -> bool:
    """kb_context per query; True when a `kb_slugs` article is missing from one of them."""
    print(RULE, file=out)
    print(f"[{case.group}] {case.id} ({case.lang})", file=out)
    if case.summary:
        print(f"  summary: {case.summary}", file=out)
    for turn in case.history:
        print(f"  {turn['role']}: {turn['content']}", file=out)
    print(f"Q: {case.text}", file=out)
    if case.kb_slugs:
        print(f"kb_slugs: {', '.join(case.kb_slugs)}", file=out)
    flagged = False
    for query, kb in contexts.items():
        print(f"kb_context[{query}]: {kb.describe()}", file=out)
        missing = missing_slugs(case, kb)
        if missing and not kb.error:
            print(f"!! [{query}] missing {', '.join(missing)}", file=out)
            flagged = True
    return flagged


async def run_retrieval(
    cases: Sequence[Case],
    *,
    kb_dir: Path,
    queries: Sequence[str],
    out: TextIO = sys.stdout,
) -> int:
    """--retrieval-only: the embeddings API only, no LLM. Exit code 1 if embeddings failed."""
    by_query = await knowledge_by_query(cases, kb_dir, queries)
    print(f"retrieval only, queries {', '.join(queries)}, {len(cases)} cases", file=out)
    flagged = 0
    for i, case in enumerate(cases):
        flagged += report_retrieval(case, {q: by_query[q][i] for q in queries}, out)
    errors = any(kb.error for contexts in by_query.values() for kb in contexts)
    print(RULE, file=out)
    print(f"{len(cases)} cases, {flagged} with !! flags", file=out)
    return 1 if errors else 0


async def run(
    cases: Sequence[Case],
    *,
    prompt_version: str,
    kb_dir: Path | None,
    skip: Sequence[str],
    query: str = "chat",
    llm_factory: LLMFactory = _real_llm,
    out: TextIO = sys.stdout,
) -> int:
    settings = get_settings()
    prompt = await FilePromptProvider(prompt_version).get_active()
    contexts = await knowledge_for(cases, kb_dir, query) if kb_dir else [NO_KB] * len(cases)
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
    parser.add_argument(
        "--query",
        action="append",
        choices=sorted(QUERIES),
        help="what is embedded for the search (default chat); several only with --retrieval-only",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true", help="kb_context per --query, no LLM calls"
    )
    links = parser.add_mutually_exclusive_group()
    links.add_argument("--skip", action="append", default=[], metavar="PROVIDER:MODEL")
    links.add_argument(
        "--link", metavar="PROVIDER:MODEL", help="only this link of LLM_CHAIN, no fallback"
    )
    args = parser.parse_args(argv)
    queries: list[str] = args.query or ["chat"]
    if args.retrieval_only and args.no_kb:
        parser.error("--retrieval-only needs the knowledge base, not --no-kb")
    if len(queries) > 1 and not args.retrieval_only:
        parser.error("several --query only with --retrieval-only")
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
    if args.retrieval_only:
        return asyncio.run(run_retrieval(cases, kb_dir=args.kb, queries=queries))
    return asyncio.run(
        run(
            cases,
            prompt_version=args.prompt,
            kb_dir=None if args.no_kb else args.kb,
            skip=skip,
            query=queries[0],
        )
    )


if __name__ == "__main__":
    sys.exit(main())
