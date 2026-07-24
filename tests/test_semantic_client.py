from chronovisor.runtime_config import SearchEmbeddingConfig
from chronovisor.semantic_client import selected_for_rollout


def test_rollout_selection_is_stable_and_respects_modes() -> None:
    base = SearchEmbeddingConfig(backend="nemotron_service")
    assert not selected_for_rollout("query", base)
    assert selected_for_rollout(
        "query",
        SearchEmbeddingConfig(backend="nemotron_service", rollout_mode="on"),
    )
    assert not selected_for_rollout(
        "query",
        SearchEmbeddingConfig(
            backend="nemotron_service",
            rollout_mode="canary",
            canary_percent=0,
        ),
    )
    assert selected_for_rollout(
        "query",
        SearchEmbeddingConfig(
            backend="nemotron_service",
            rollout_mode="canary",
            canary_percent=100,
        ),
    )
