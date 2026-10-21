from pathlib import Path

import pytest

from app.db.models import KBDocumentStatus
from app.rag.documents import InvalidArticle, load_articles, parse_article

KB_DIR = Path(__file__).resolve().parents[2] / "docs/kb"

ARTICLE = """---
title: Перхоть
language: ru
status: draft
topics: [dandruff]
sources: []
license: internal
---

## Уход

Мягкий шампунь.
"""


def test_frontmatter_and_body_are_parsed() -> None:
    article = parse_article("dandruff", ARTICLE)

    assert article.slug == "dandruff"
    assert article.meta.title == "Перхоть"
    assert article.meta.status is KBDocumentStatus.DRAFT
    assert article.meta.url is None
    assert article.body == "## Уход\n\nМягкий шампунь."


def test_hash_changes_with_the_text_but_not_with_the_status() -> None:
    article = parse_article("dandruff", ARTICLE)
    reviewed = parse_article("dandruff", ARTICLE.replace("status: draft", "status: reviewed"))
    edited = parse_article("dandruff", ARTICLE.replace("Мягкий", "Деликатный"))

    assert reviewed.content_hash == article.content_hash
    assert edited.content_hash != article.content_hash


@pytest.mark.parametrize(
    ("slug", "text", "error"),
    [
        ("dandruff", "## no frontmatter", "no YAML frontmatter"),
        ("dandruff", ARTICLE.replace("status: draft", "status: final"), "invalid frontmatter"),
        ("dandruff", ARTICLE.replace("license: internal\n", ""), "invalid frontmatter"),
        ("dandruff", ARTICLE.replace("title: Перхоть", "title: x\nauthor: me"), "invalid"),
        ("dandruff", ARTICLE.split("---\n\n")[0] + "---\n", "empty body"),
        ("Dandruff_1", ARTICLE, "kebab-case"),
    ],
)
def test_invalid_articles_are_rejected_with_the_file_name(slug: str, text: str, error: str) -> None:
    with pytest.raises(InvalidArticle, match=error) as exc:
        parse_article(slug, text)

    assert f"{slug}.md" in str(exc.value)


def test_shipped_knowledge_base_is_valid() -> None:
    articles = load_articles(KB_DIR)

    assert len(articles) >= 12
    assert all(a.meta.language == "ru" for a in articles)
    assert "README" not in {a.slug for a in articles}
