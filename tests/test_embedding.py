"""Tests for embedding helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor import embedding as emb_mod
from chronovisor.embedding import cosine, embed_text, embed_texts, most_similar


@pytest.fixture()
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_dir = tmp_path / ".embeddings"
    monkeypatch.setattr(emb_mod, "_CACHE_DIR", cache_dir)
    return cache_dir


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical_vectors_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert cosine(v, v) == pytest.approx(1.0, abs=1e-9)

    def test_opposite_vectors_minus_one(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine(a, b) == pytest.approx(-1.0, abs=1e-9)

    def test_orthogonal_vectors_zero(self) -> None:
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-9)

    def test_zero_vector_returns_zero(self) -> None:
        assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine([1.0, 1.0], [0.0, 0.0]) == 0.0
        assert cosine([], [1.0]) == 0.0


# ---------------------------------------------------------------------------
# embed_text + cache
# ---------------------------------------------------------------------------


class TestEmbedTextCache:
    def test_first_call_hits_ollama_second_uses_cache(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def fake_ollama_embed(texts):
            calls["n"] += 1
            return [[0.1, 0.2, 0.3] for _ in texts]

        monkeypatch.setattr(emb_mod, "_ollama_embed", fake_ollama_embed)

        v1 = embed_text("hello")
        v2 = embed_text("hello")
        assert v1 == v2 == [0.1, 0.2, 0.3]
        assert calls["n"] == 1, "second call must hit the cache"

    def test_cache_miss_for_different_text(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def fake_ollama_embed(texts):
            calls["n"] += 1
            # Distinct vectors so the test catches accidental key collision.
            return [[float(len(t))] for t in texts]

        monkeypatch.setattr(emb_mod, "_ollama_embed", fake_ollama_embed)

        embed_text("foo")
        embed_text("bar-baz")
        assert calls["n"] == 2

    def test_corrupt_cache_falls_back_to_ollama(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = emb_mod._cache_path("hello")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not-json{{{")  # corrupt

        def fake_ollama_embed(texts):
            return [[1.0, 2.0]]

        monkeypatch.setattr(emb_mod, "_ollama_embed", fake_ollama_embed)
        assert embed_text("hello") == [1.0, 2.0]


# ---------------------------------------------------------------------------
# embed_texts batching
# ---------------------------------------------------------------------------


class TestEmbedTexts:
    def test_only_uncached_hit_ollama(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pre-warm the cache for "a".
        cache_for_a = emb_mod._cache_path("a")
        cache_for_a.parent.mkdir(parents=True, exist_ok=True)
        cache_for_a.write_text(json.dumps([9.0]))

        captured = {"requested": []}

        def fake_ollama_embed(texts):
            captured["requested"].extend(texts)
            return [[float(len(t))] for t in texts]

        monkeypatch.setattr(emb_mod, "_ollama_embed", fake_ollama_embed)

        results = embed_texts(["a", "bb", "ccc"])
        # Cached "a" returned without going to Ollama.
        assert results == [[9.0], [2.0], [3.0]]
        assert captured["requested"] == ["bb", "ccc"]

    def test_empty_input_no_ollama_call(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(_texts):
            raise AssertionError("must not call ollama for empty input")

        monkeypatch.setattr(emb_mod, "_ollama_embed", boom)
        assert embed_texts([]) == []


# ---------------------------------------------------------------------------
# most_similar
# ---------------------------------------------------------------------------


class TestMostSimilar:
    def test_returns_best_match_above_threshold(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Wire embeddings so:
        #   "query" -> [1, 0]
        #   "near"  -> [0.9, 0.1]   sim ~= 0.9939
        #   "far"   -> [0, 1]       sim = 0.0
        vectors = {
            "query": [1.0, 0.0],
            "near": [0.9, 0.1],
            "far": [0.0, 1.0],
        }

        def fake_ollama_embed(texts):
            return [vectors[t] for t in texts]

        monkeypatch.setattr(emb_mod, "_ollama_embed", fake_ollama_embed)

        result = most_similar("query", ["near", "far"], threshold=0.8)
        assert result is not None
        cand, sim = result
        assert cand == "near"
        assert sim > 0.99

    def test_returns_none_when_below_threshold(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vectors = {
            "query": [1.0, 0.0],
            "far": [0.0, 1.0],
        }
        monkeypatch.setattr(
            emb_mod, "_ollama_embed", lambda texts: [vectors[t] for t in texts]
        )
        assert most_similar("query", ["far"], threshold=0.5) is None

    def test_returns_none_for_empty_candidates(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            emb_mod,
            "_ollama_embed",
            lambda _t: (_ for _ in ()).throw(AssertionError("must not call")),
        )
        assert most_similar("query", [], threshold=0.5) is None
