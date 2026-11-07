from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from chronovisor.core import decision_mlx
from chronovisor.core.decision_mlx import (
    DecisionMLXBackend,
    DecisionModelConfig,
    decision_state,
    micro_batches,
)
from chronovisor.core.llm_config import (
    LLMConfigError,
    build_llm_runtime,
    parse_llm_config,
)
from chronovisor.core.llm_runtime import (
    RerankRequest,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)

SOURCE = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)


def _config(provider: dict) -> dict:
    return {
        "llm": {
            "providers": {"decision": provider},
            "roles": {
                "rerank": {
                    "capability": "rerank",
                    "provider": "decision",
                    "model": "startlux-decision-4b",
                }
            },
        }
    }


def test_config_builds_a_decision_backend_with_defaults() -> None:
    config = parse_llm_config(_config({"kind": "local-decision-mlx", "path": "~/m"}))
    backend = build_llm_runtime(config)._rerank["rerank"].backend

    assert isinstance(backend, DecisionMLXBackend)
    assert backend.config.path.endswith("/m") and not backend.config.path.startswith(
        "~"
    )
    assert (backend.config.max_chars, backend.config.batch_tokens) == (700, 8192)


@pytest.mark.parametrize(
    "provider",
    [
        {"kind": "local-decision-mlx"},
        {"kind": "local-decision-mlx", "path": ""},
        {"kind": "local-decision-mlx", "path": "/m", "device": "mps"},
    ],
)
def test_config_rejects_incomplete_or_unknown_decision_keys(provider) -> None:
    with pytest.raises(LLMConfigError):
        parse_llm_config(_config(provider))


def test_states_follow_the_purpose_and_cut_the_candidate() -> None:
    search = RerankRequest("dict sort", ("abcdef",), SOURCE)
    gate = RerankRequest(
        "and by value?",
        ("abcdef",),
        SOURCE,
        purpose="recall_gate",
        context="sort a dict",
    )

    assert decision_state(search, "abcdef", 3) == (
        "Search query:\ndict sort\n\nCandidate memory:\nabc"
    )
    assert decision_state(gate, "abcdef", 3) == (
        "Preceding user turn:\nsort a dict\n\nCurrent user turn:\nand by value?"
        "\n\nCandidate memory:\nabc"
    )
    no_context = RerankRequest("q", ("x",), SOURCE, purpose="recall_gate")
    assert decision_state(no_context, "x", 9).startswith("Current user turn:\nq")


def test_micro_batches_respect_the_padded_token_budget() -> None:
    groups = micro_batches([5, 100, 6, 50], budget=100)

    assert sorted(i for g in groups for i in g) == [0, 1, 2, 3]
    lengths = [5, 100, 6, 50]
    assert all(
        len(g) * max(lengths[i] for i in g) <= 100 or len(g) == 1 for g in groups
    )


def test_rerank_orders_by_score_on_the_owner_thread(monkeypatch) -> None:
    backend = DecisionMLXBackend(DecisionModelConfig(path="/m"))
    owner = ThreadPoolExecutor(1)
    backend._engine = type("Engine", (), {"_executor": owner})()
    seen: dict = {}

    def readout(states, question):
        seen.update(states=states, question=question)
        return [-1.0, 2.0, 2.0]

    monkeypatch.setattr(backend, "_readout", readout)
    result = backend.rerank(RerankRequest("q", ("a", "b", "c"), SOURCE), model="m")
    owner.shutdown()

    assert [(item.index, item.score) for item in result.items] == [
        (1, 2.0),
        (2, 2.0),
        (0, -1.0),
    ]
    assert seen["question"] == decision_mlx.QUESTIONS["search"]
    assert seen["states"][0].endswith("Candidate memory:\na")


def test_rerank_rejects_an_unknown_purpose() -> None:
    backend = DecisionMLXBackend(DecisionModelConfig(path="/m"))

    with pytest.raises(ValueError):
        backend.rerank(RerankRequest("q", ("a",), SOURCE, purpose="other"), model="m")
