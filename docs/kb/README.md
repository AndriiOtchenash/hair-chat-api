# Knowledge base (RAG source)

Short cosmetic hair-care articles in Russian, our own wording. Loaded into `kb_documents` /
`kb_chunks` by `uv run python -m app.rag.ingest docs/kb` (idempotent; see app/rag/ingest.py).
This README and files starting with `_` are not ingested.

## Format

File name = slug (`lowercase-kebab-case.md`), sent to clients in `done.sources`.

```yaml
---
title: Перхоть            # shown to users in sources
language: ru
status: draft             # draft | reviewed; prod searches reviewed only (KB_INCLUDE_DRAFTS)
topics: [dandruff, scalp]
sources: []               # where facts were checked (may be empty)
license: internal
url: null                 # optional public page, sources[].url
---
```

Body: markdown with `##` sections (the chunker splits along headings, ~500 tokens per chunk).

## Rules for the text

Same as the system prompt: cosmetic advice only, no diagnoses, no medicines, no doses or
treatment plans; say when to see a trichologist or a doctor. No copied texts.

## Review

A specialist reads the article, then sets `status: reviewed` and re-runs the ingest (a status change
does not re-embed). Changing the text re-embeds that article only. After changing
`KB_MIN_SIMILARITY` or adding articles, re-run `uv run python -m scripts.rag_eval`.
