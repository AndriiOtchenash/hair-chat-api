"""The `<kb_used ids="1,3"/>` tag: which numbered <knowledge> passages an answer relied on.

The prompt (system_v6) asks the model to end an answer based on passages with this tag. It never
reaches the client or the DB: `KbUsedFilter` cuts it out of the token stream, holding back the
tail of the stream while it may still be the start of the tag. `done.sources` then lists only the
articles of the passages named in it (`used_sources`); no tag means no sources.
"""

import re

from app.rag.retriever import Source

TAG_START = "<kb_used"
_TAG = re.compile(r"<kb_used\b([^>]*)>")
_IDS = re.compile(r"""ids\s*=\s*["']?([^"'/>]*)""")


class KbUsedFilter:
    """Feed it the answer's text pieces; it returns what may be shown, minus the tag.

    Whitespace right before a possible tag is held back with it, so the answer does not end
    with the blank line the model puts before the tag (unless that came in an earlier piece).
    A "<" that does not go on to "<kb_used" is ordinary text.
    """

    def __init__(self) -> None:
        self._pending = ""
        self.ids: set[int] = set()

    def feed(self, text: str) -> str:
        buf = self._pending + text
        shown: list[str] = []
        while (tag := _TAG.search(buf)) is not None:
            shown.append(buf[: tag.start()].rstrip())
            self.ids |= _parse_ids(tag.group(1))
            buf = buf[tag.end() :]
        hold = _hold_from(buf)
        shown.append(buf[:hold])
        self._pending = buf[hold:]
        return "".join(shown)

    def finish(self) -> str:
        """The held tail at the end of the stream: an unfinished tag (the answer was cut off) is
        dropped, anything else is text."""
        pending, self._pending = self._pending, ""
        rest = pending.lstrip()
        if rest.startswith("<") and len(rest) > 1:
            return ""
        return pending


def used_sources(ids: set[int], passages: tuple[Source, ...]) -> tuple[Source, ...]:
    """Articles of the passages [1]..[n] named in the tag, once each, in passage order (closest
    first). Ids out of range are ignored."""
    used: dict[str, Source] = {}
    for number, source in enumerate(passages, start=1):
        if number in ids:
            used.setdefault(source.slug, source)
    return tuple(used.values())


def _parse_ids(attrs: str) -> set[int]:
    found = _IDS.search(attrs)
    if found is None:
        return set()
    parts = (p.strip() for p in found.group(1).split(","))
    return {int(p) for p in parts if p.isascii() and p.isdigit()}


def _hold_from(buf: str) -> int:
    """Where the tail that may still become a tag starts (len(buf) if there is none)."""
    start = buf.find(TAG_START)
    if start == -1:
        # The longest end of buf that TAG_START begins with, e.g. "<kb_u".
        start = next(
            (
                i
                for i in range(max(0, len(buf) - len(TAG_START) + 1), len(buf))
                if TAG_START.startswith(buf[i:])
            ),
            len(buf),
        )
    while 0 < start < len(buf) and buf[start - 1].isspace():
        start -= 1
    return start
