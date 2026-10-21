"""Markdown article -> chunks of ~500 tokens, split along its headings.

Every chunk starts with its breadcrumb ("Title › Section"), so a passage keeps its topic both for
the embedding and for the model reading it. Adjacent short sections are merged; a section longer
than the target is split by paragraphs (then sentences) with ~50 tokens of overlap.
Token counts are the provider-neutral estimate of app.services.tokens.
"""

import re
from dataclasses import dataclass

from app.services.tokens import estimate_tokens

# Part of the content hash: a change here re-embeds every article on the next ingest.
CHUNKER_VERSION = "1"
TARGET_TOKENS = 500
OVERLAP_TOKENS = 50

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class _Section:
    path: tuple[str, ...]  # headings below the title, outermost first
    body: str


def chunk_markdown(
    title: str,
    body: str,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    texts: list[str] = []
    pending: list[_Section] = []

    def flush() -> None:
        if pending:
            texts.append(_render(title, pending))
            pending.clear()

    for section in _sections(title, body):
        candidate = _render(title, [*pending, section])
        if estimate_tokens(candidate) <= target_tokens:
            pending.append(section)
            continue
        flush()
        if estimate_tokens(_render(title, [section])) <= target_tokens:
            pending.append(section)
            continue
        head = _breadcrumb(title, section.path)
        budget = target_tokens - estimate_tokens(head) - 1
        texts.extend(f"{head}\n\n{part}" for part in _split(section.body, budget, overlap_tokens))
    flush()
    return [Chunk(i, text) for i, text in enumerate(texts)]


def _sections(title: str, body: str) -> list[_Section]:
    sections: list[_Section] = []
    path: list[tuple[int, str]] = []
    lines: list[str] = []

    def close() -> None:
        text = "\n".join(lines).strip()
        if text:
            sections.append(_Section(tuple(h for _, h in path), text))
        lines.clear()

    for line in body.splitlines():
        match = _HEADING.match(line)
        if match is None:
            lines.append(line)
            continue
        close()
        level, heading = len(match.group(1)), match.group(2)
        if level == 1 and heading.strip() == title.strip():
            continue  # the article's own title: already in every breadcrumb
        path = [(lvl, h) for lvl, h in path if lvl < level]
        path.append((level, heading))
    close()
    return sections


def _breadcrumb(title: str, path: tuple[str, ...]) -> str:
    return " › ".join((title, *path))


def _render(title: str, sections: list[_Section]) -> str:
    """One chunk of whole sections: the first one's breadcrumb, the others' headings inline."""
    first, *rest = sections
    parts = [_breadcrumb(title, first.path), first.body]
    for section in rest:
        parts.append(f"{section.path[-1]}\n{section.body}" if section.path else section.body)
    return "\n\n".join(parts)


def _split(text: str, budget: int, overlap: int) -> list[str]:
    """Pieces of at most ~budget tokens; each repeats the tail (~overlap tokens) of the last one."""
    units = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units = [s for unit in units for s in _fit(unit, budget)]
    pieces: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and estimate_tokens("\n\n".join([*current, unit])) > budget:
            pieces.append("\n\n".join(current))
            current = _tail(current, overlap)
            if estimate_tokens("\n\n".join([*current, unit])) > budget:
                current = []
        current.append(unit)
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def _fit(paragraph: str, budget: int) -> list[str]:
    """A paragraph too long for one piece becomes its sentences (words as a last resort)."""
    if estimate_tokens(paragraph) <= budget:
        return [paragraph]
    out: list[str] = []
    for sentence in _SENTENCE_END.split(paragraph):
        if estimate_tokens(sentence) <= budget:
            out.append(sentence)
            continue
        words: list[str] = []
        for word in sentence.split():
            if words and estimate_tokens(" ".join([*words, word])) > budget:
                out.append(" ".join(words))
                words = []
            words.append(word)
        if words:
            out.append(" ".join(words))
    return out


def _tail(units: list[str], overlap: int) -> list[str]:
    """The last units (whole sentences of the last one) worth at most ~overlap tokens."""
    sentences = _SENTENCE_END.split(units[-1])
    tail: list[str] = []
    for sentence in reversed(sentences):
        if estimate_tokens(" ".join([sentence, *tail])) > overlap:
            break
        tail.insert(0, sentence)
    return [" ".join(tail)] if tail else []
