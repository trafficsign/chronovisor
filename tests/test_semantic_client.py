from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.core.semantic_client import selected_for_rollout


def test_rollout_selection_is_stable_and_respects_modes() -> None:
    base = SearchEmbeddingConfig()
    assert not selected_for_rollout("query", base)
    assert selected_for_rollout(
        "query",
        SearchEmbeddingConfig(enabled=True, rollout_mode="on"),
    )
    assert not selected_for_rollout(
        "query",
        SearchEmbeddingConfig(
            enabled=True,
            rollout_mode="canary",
            canary_percent=0,
        ),
    )
    assert selected_for_rollout(
        "query",
        SearchEmbeddingConfig(
            enabled=True,
            rollout_mode="canary",
            canary_percent=100,
        ),
    )
