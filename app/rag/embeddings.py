"""Text -> vector, one model for the whole knowledge-base index (rule 4, ADR 0005).

There is no fallback: vectors of another model live in another space and cannot be compared with
the index. When embeddings are down the chat answers without the knowledge base instead.
Errors carry a safe code only, never the texts.
"""

from collections.abc import Sequence
from typing import Protocol

import httpx2
import openai

from app.llm.base import MISCONFIGURED_STATUSES, provider_side_400

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Inputs per /embeddings request; Gemini accepts up to 100.
MAX_BATCH = 100


class EmbeddingUnavailable(Exception):
    """The embedding call failed. `code` is safe to log (no request or response text).

    misconfigured: our side is wrong (key, permissions, region: 401/403/404 or a provider-side
    400) -> logged as error; it will not heal by itself.
    """

    def __init__(self, code: str, *, misconfigured: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.misconfigured = misconfigured


class Embedder(Protocol):
    @property
    def model(self) -> str:
        """Stored in kb_chunks.embedding_model and used to filter the search."""
        ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in order. Raises EmbeddingUnavailable."""
        ...

    async def aclose(self) -> None: ...


class OpenAICompatibleEmbedder:
    """POST /embeddings of an OpenAI-compatible endpoint (Gemini), with `dimensions`."""

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        api_key: str,
        timeout: float,
        max_retries: int = 0,
        base_url: str = GEMINI_OPENAI_BASE_URL,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        # Chat: max_retries=0, a retry would delay the answer; the ingest CLI may retry.
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=httpx2.Timeout(timeout),
            http_client=http_client,
        )

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.close()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), MAX_BATCH):
            vectors.extend(await self._batch(list(texts[start : start + MAX_BATCH])))
        return vectors

    async def _batch(self, texts: list[str]) -> list[list[float]]:
        try:
            # encoding_format: the SDK asks for base64 by default; plain floats work everywhere.
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimensions,
                encoding_format="float",
            )
        except openai.APIStatusError as exc:
            reason = provider_side_400(exc.status_code, exc.body)
            raise EmbeddingUnavailable(
                f"{type(exc).__name__}:{exc.status_code}" + (f":{reason}" if reason else ""),
                misconfigured=reason is not None or exc.status_code in MISCONFIGURED_STATUSES,
            ) from exc
        except openai.APITimeoutError as exc:
            raise EmbeddingUnavailable("timeout") from exc
        except openai.APIConnectionError as exc:
            raise EmbeddingUnavailable("connection_error") from exc
        except openai.APIError as exc:
            raise EmbeddingUnavailable("api_error") from exc
        # Gemini leaves `index` empty and answers in input order; sort only when it is set.
        rows = response.data
        if all(row.index is not None for row in rows):
            rows = sorted(rows, key=lambda row: row.index)
        if len(rows) != len(texts):
            raise EmbeddingUnavailable("wrong_count")
        if any(len(row.embedding) != self._dimensions for row in rows):
            raise EmbeddingUnavailable("wrong_dimensions")
        return [list(row.embedding) for row in rows]
