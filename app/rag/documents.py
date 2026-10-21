"""Knowledge-base articles on disk: docs/kb/<slug>.md with a YAML frontmatter.

    ---
    title: Перхоть
    language: ru
    status: draft            # draft | reviewed; prod searches reviewed only
    topics: [dandruff, scalp]
    sources: []              # where the facts were checked; may be empty
    license: internal
    url: null                # optional public page of the article, sent to clients as sources[].url
    ---
    markdown body

README.md and files starting with `_` are not articles. See docs/kb/README.md.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.db.models import KBDocumentStatus
from app.rag.chunker import CHUNKER_VERSION

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class InvalidArticle(Exception):
    """A file in the KB directory is not a valid article; the message names the file."""


class Frontmatter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    language: str = Field(min_length=2)
    status: KBDocumentStatus
    topics: list[str] = Field(min_length=1)
    sources: list[str] = []
    license: str = Field(min_length=1)
    url: str | None = None


@dataclass(frozen=True, slots=True)
class Article:
    slug: str
    meta: Frontmatter
    body: str

    @property
    def content_hash(self) -> str:
        """What the chunks are made of: the same hash means the stored vectors are still valid.
        Status, topics, sources and url are not part of it, so a review costs no re-embedding."""
        payload = "\n".join([CHUNKER_VERSION, self.meta.title, self.body])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_article(slug: str, text: str) -> Article:
    if not _SLUG.match(slug):
        raise InvalidArticle(f"{slug}.md: file name must be lowercase-kebab-case")
    match = _FRONTMATTER.match(text.replace("\r\n", "\n"))
    if match is None:
        raise InvalidArticle(f"{slug}.md: no YAML frontmatter between --- lines")
    try:
        raw: Any = yaml.safe_load(match.group(1))
        meta = Frontmatter.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as exc:
        raise InvalidArticle(f"{slug}.md: invalid frontmatter: {exc}") from exc
    body = match.group(2).strip()
    if not body:
        raise InvalidArticle(f"{slug}.md: empty body")
    return Article(slug, meta, body)


def load_articles(directory: Path) -> list[Article]:
    if not directory.is_dir():
        raise InvalidArticle(f"{directory}: not a directory")
    paths = sorted(
        p for p in directory.glob("*.md") if p.name != "README.md" and not p.name.startswith("_")
    )
    return [parse_article(p.stem, p.read_text(encoding="utf-8")) for p in paths]
