from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_materialization as materialization
from chronovisor.recall import recall_distillation_store as store


@pytest.mark.parametrize(
    ("exception", "reason"),
    [
        (
            materialization.MaterializationSourceChanged(
                "canonical materialization projection changed"
            ),
            "materialization_source_changed",
        ),
        (
            materialization.MaterializationBudgetExhausted(
                "materialization cycle budget exhausted"
            ),
            "materialization_budget_exhausted",
        ),
    ],
)
def test_chunk_defers_verified_materialization_boundaries_without_failure_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception: materialization.MaterializationDeferred,
    reason: str,
) -> None:
    state_path = store.distillation_dir(tmp_path) / store.STATE_FILE
    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "worker_status": "error",
            "error_code": "distillation_run_failed",
            "worker_error_sha256": "a" * 64,
            "last_success_at": "2026-10-17T20:00:00Z",
            "worker_failure_count": 3,
            "last_failure_at": "2026-10-17T20:01:00Z",
        },
    )
    before_bytes = state_path.read_bytes()

    def defer(**_kwargs: object) -> dict[str, object]:
        raise exception

    monkeypatch.setattr(distill, "_run_distillation_chunk_impl", defer)
    result = distill.run_distillation_chunk(root=tmp_path, max_elapsed_seconds=60)

    assert result == {
        "status": "deferred",
        "processed": 0,
        "reason": reason,
        "atomic_progress_may_be_present": True,
        "model_calls_may_be_present": True,
    }
    preserved = distill._read_worker_state(tmp_path)
    assert preserved["worker_status"] == "error"
    assert preserved["error_code"] == "distillation_run_failed"
    assert preserved["worker_failure_count"] == 3
    assert state_path.read_bytes() == before_bytes


def test_chunk_keeps_generic_store_error_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = store.distillation_dir(tmp_path) / store.STATE_FILE
    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "worker_status": "ready",
            "last_success_at": "2026-10-17T20:00:00Z",
            "worker_failure_count": 0,
        },
    )

    def fail(**_kwargs: object) -> dict[str, object]:
        raise store.DistillationStoreError("ledger state is invalid")

    monkeypatch.setattr(distill, "_run_distillation_chunk_impl", fail)
    with pytest.raises(store.DistillationStoreError, match="ledger state"):
        distill.run_distillation_chunk(root=tmp_path, max_elapsed_seconds=60)

    state = distill._read_worker_state(tmp_path)
    assert state["worker_status"] == "error"
    assert state["error_code"] == "distillation_run_failed"
    assert state["worker_failure_count"] == 1
