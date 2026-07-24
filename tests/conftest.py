from __future__ import annotations

import pytest

from chronovisor.runtime_config import SearchEmbeddingConfig


@pytest.fixture(autouse=True)
def isolate_operator_raw_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit behavior independent from the operator's live rollout mode."""

    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "legacy")


@pytest.fixture(autouse=True)
def isolate_operator_search_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests from enqueueing work against the live semantic service."""

    from chronovisor import search

    monkeypatch.setattr(
        search,
        "load_search_embedding_config",
        lambda: SearchEmbeddingConfig(backend="legacy_ollama"),
    )
