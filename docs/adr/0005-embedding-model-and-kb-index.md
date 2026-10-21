# 0005 — Knowledge base: one embedding model, 768 dimensions, reviewed articles in prod

Date: 2026-10-21 · Status: accepted

## Context
1.9 adds RAG: docs/kb articles are chunked, embedded and stored in `kb_chunks.embedding`
(`vector(768)`, HNSW, cosine). Each question is embedded the same way and the nearest chunks go
into the `<knowledge>` block. Vectors are comparable only within one model (and one dimension):
a vector of model B searched against an index of model A returns noise, not an error.

## Decision
- **Gemini `gemini-embedding-2`** via the OpenAI-compatible `/embeddings` endpoint, key
  `GEMINI_API_KEY` (the chat already sends user messages to Gemini, so no new processor;
  paid tier in prod). Picked from `GET /models` (2026-10-21: `gemini-embedding-001`,
  `gemini-embedding-2-preview`, `gemini-embedding-2`). A live check confirmed that
  `dimensions=768` returns 768 values, already unit-normalized; cross-lingual similarity of
  the same question in ru/en is 0.83, so uk/pl/en questions find the Russian articles.
  The schema stays `vector(768)`. `encoding_format="float"` is sent explicitly (the SDK asks
  for base64 by default).
- **One model for the whole index** (CLAUDE.md rule 4): `EMBEDDING_MODEL` in Settings, every
  chunk stores `embedding_model`, the search filters on it. Changing the model = re-running
  `python -m app.rag.ingest docs/kb`, which re-embeds every article whose chunks are of
  another model. Until then the old rows are simply not found (answers go without knowledge),
  never mixed in.
- **No fallback for embeddings.** Another provider's vectors would not match the index. When the
  query embedding fails or exceeds `EMBEDDING_TIMEOUT_S` (3 s, whole call) the answer goes
  without `<knowledge>` and a warning is logged. The embedding runs in parallel with
  moderation (ChatService.reply), so it adds no latency in the normal case.
- **Threshold** `KB_MIN_SIMILARITY` = 0.60, chosen on `scripts/rag_eval.py` (25 questions in
  ru/uk/pl/en, 4 off-topic): off-topic best 0.53, weakest right hit 0.64; 0.55–0.60 give
  hit@4 21/21 with no false finds, 0.65 already misses one Ukrainian question. Applied after
  `ORDER BY embedding <=> q LIMIT top_k` (a distance in WHERE would stop the HNSW walk).
  Top 4 chunks, packed into `KB_MAX_TOKENS` (1500).
- **Review gate:** `kb_documents.status` draft | reviewed. Prod searches reviewed only (Settings
  refuses `KB_INCLUDE_DRAFTS=true` with `ENV=prod`); dev searches drafts too by default.
- **Idempotent ingest:** `content_hash` = sha256(chunker version + title + body). Same hash and
  model -> metadata only; changed -> chunks replaced in one transaction; missing file ->
  document deleted (chunks cascade). Status is not in the hash, so a review costs no embedding.

## Consequences
- Every article shipped in 1.9 is `draft`: prod has an empty knowledge base until a specialist
  reviews articles and the ingest is re-run.
- The threshold is model-specific: a model change also means re-running the eval and
  re-choosing `KB_MIN_SIMILARITY`.
- `done.sources` lists only articles whose passages actually reached the context (not those
  dropped by the token budget).
