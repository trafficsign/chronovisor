from __future__ import annotations

from chronovisor.recall.recall_confidence import (
    cluster_bootstrap_interval,
    cluster_rate_wilson_interval,
)


def test_anonymous_rows_never_count_as_independent_clusters() -> None:
    bound = cluster_bootstrap_interval(
        [{"score": 1.0}, {"score": 1.0}], value_key="score"
    )

    assert bound["valid"] is False
    assert bound["clusters"] == 0


def test_perfect_finite_cluster_rate_keeps_conservative_uncertainty() -> None:
    rows = [
        {"session_hash": f"session-{index}", "coverage": 1.0}
        for index in range(20)
    ]
    bound = cluster_rate_wilson_interval(
        rows, value_key="coverage", success_threshold=0.99
    )

    assert bound["valid"] is True
    assert bound["point"] == 1.0
    assert 0.0 < bound["lower"] < 1.0
