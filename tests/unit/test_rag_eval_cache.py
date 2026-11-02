from pathlib import Path

from scripts.rag_eval import EmbeddingCache
from tests.llm_fakes import FakeEmbedder


async def test_cached_texts_are_not_embedded_again_across_runs(tmp_path: Path) -> None:
    file = tmp_path / "cache" / "embeddings.json"
    embedder = FakeEmbedder(vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]})

    first = await EmbeddingCache(file, "m", 768).embed(embedder, ["a", "b", "a"])
    second = await EmbeddingCache(file, "m", 768).embed(embedder, ["b", "a"])

    assert first == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    assert second == [[0.0, 1.0], [1.0, 0.0]]
    assert embedder.calls == 1
    assert embedder.embedded == ["a", "b"]


async def test_another_model_or_dimensions_misses_the_cache(tmp_path: Path) -> None:
    file = tmp_path / "embeddings.json"
    embedder = FakeEmbedder()

    await EmbeddingCache(file, "m", 768).embed(embedder, ["a"])
    await EmbeddingCache(file, "other", 768).embed(embedder, ["a"])
    await EmbeddingCache(file, "m", 256).embed(embedder, ["a"])

    assert embedder.calls == 3
