from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation_progress as progress
from chronovisor.recall import recall_distillation_store as store


def _state_path(root: Path) -> Path:
    return store.distillation_dir(root) / store.STATE_FILE


def _write_worker_state(root: Path, **payload: object) -> None:
    store.write_sealed_state(
        _state_path(root),
        {"kind": "worker-state", **payload},
    )


def test_read_returns_none_when_attempt_receipt_is_absent(tmp_path: Path) -> None:
    assert progress.read(tmp_path) is None


def test_begin_creates_sealed_attempt_receipt(tmp_path: Path) -> None:
    token = progress.begin(tmp_path)

    receipt = progress.read(tmp_path)
    assert receipt is not None
    assert receipt["kind"] == progress.FORMAL_PROGRESS_KIND
    assert receipt["first_attempt_at"] == token["started_at"]
    assert receipt["last_attempt_at"] == token["started_at"]
    assert receipt["last_run_started_at"] == token["started_at"]
    assert receipt["last_outcome"] == "running"
    assert receipt["attempt_count"] == 1
    assert receipt["consecutive_deferrals"] == 0
    assert receipt["seal_sha256"]


def test_deferred_attempt_increments_only_defer_streak(tmp_path: Path) -> None:
    first = progress.begin(tmp_path)
    result = progress.finish(
        tmp_path,
        first,
        {"status": "deferred", "reason": "materialization_budget_exhausted"},
    )

    assert result["last_outcome"] == "deferred"
    assert result["last_defer_reason"] == "materialization_budget_exhausted"
    assert result["consecutive_deferrals"] == 1
    assert result["attempt_count"] == 1
    assert result["success_count"] == 0

    second = progress.begin(tmp_path)
    result = progress.finish(
        tmp_path,
        second,
        {"status": "deferred", "reason": "distillation_store_busy"},
    )

    assert result["last_outcome"] == "deferred"
    assert result["last_defer_reason"] == "distillation_store_busy"
    assert result["consecutive_deferrals"] == 2
    assert result["attempt_count"] == 2


def test_worker_busy_is_recorded_as_a_deferred_attempt(tmp_path: Path) -> None:
    first = progress.record_busy(tmp_path)
    assert first["last_outcome"] == "deferred"
    assert first["last_defer_reason"] == "worker_busy"
    assert first["consecutive_deferrals"] == 1
    assert first["attempt_count"] == 1
    assert first["last_run_started_at"] == ""

    second = progress.record_busy(tmp_path)
    assert second["last_outcome"] == "deferred"
    assert second["last_defer_reason"] == "worker_busy"
    assert second["consecutive_deferrals"] == 2
    assert second["attempt_count"] == 2


def test_data_reject_is_not_counted_as_operational_defer(tmp_path: Path) -> None:
    token = progress.begin(tmp_path)
    _write_worker_state(
        tmp_path,
        worker_error_stage="counterfactual",
        worker_error_kind="counterfactual_data_reject",
        last_failure_at=datetime.now(UTC).isoformat(),
    )

    result = progress.finish(
        tmp_path,
        token,
        {"status": "deferred", "reason": "counterfactual_invalid"},
    )

    assert result["last_outcome"] == "data_reject"
    assert result["consecutive_deferrals"] == 0
    assert result["last_defer_reason"] == ""


def test_success_requires_worker_state_watermark_to_advance(tmp_path: Path) -> None:
    before = datetime.now(UTC) - timedelta(minutes=1)
    _write_worker_state(tmp_path, last_success_at=before.isoformat())
    token = progress.begin(tmp_path)

    deferred = progress.finish(tmp_path, token, {"status": "ready"})
    assert deferred["last_outcome"] == "other"
    assert deferred["success_count"] == 0

    token = progress.begin(tmp_path)
    _write_worker_state(tmp_path, last_success_at=datetime.now(UTC).isoformat())
    succeeded = progress.finish(tmp_path, token, {"status": "deferred"})

    assert succeeded["last_outcome"] == "succeeded"
    assert succeeded["success_count"] == 1
    assert succeeded["consecutive_deferrals"] == 0
    assert succeeded["last_success_at"]


def test_busy_record_does_not_invalidate_concurrent_run_token(
    tmp_path: Path,
) -> None:
    token = progress.begin(tmp_path)
    busy = progress.record_busy(tmp_path)
    assert busy["last_outcome"] == "deferred"
    assert busy["last_run_started_at"] == token["started_at"]
    _write_worker_state(tmp_path, last_success_at=datetime.now(UTC).isoformat())

    succeeded = progress.finish(tmp_path, token, {"status": "ready"})

    assert succeeded["last_outcome"] == "succeeded"
    assert succeeded["consecutive_deferrals"] == 0
    assert succeeded["attempt_count"] == 2
    assert succeeded["last_defer_reason"] == ""


def test_finish_accepts_legacy_receipt_without_run_start_field(
    tmp_path: Path,
) -> None:
    token = progress.begin(tmp_path)
    receipt = progress.read(tmp_path)
    assert receipt is not None
    legacy = {
        key: value
        for key, value in receipt.items()
        if key not in {"schema", "namespace", "seal_sha256", "last_run_started_at"}
    }
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / progress.FORMAL_PROGRESS_FILE,
        legacy,
    )

    result = progress.finish(
        tmp_path,
        token,
        {"status": "deferred", "reason": "materialization_budget_exhausted"},
    )

    assert result["last_outcome"] == "deferred"
    assert result["consecutive_deferrals"] == 1


def test_stale_token_fails_closed_without_rewriting_receipt(tmp_path: Path) -> None:
    token = progress.begin(tmp_path)
    progress.begin(tmp_path)

    with pytest.raises(store.DistillationStoreError, match="token is stale"):
        progress.finish(tmp_path, token, {"status": "deferred"})

    receipt = progress.read(tmp_path)
    assert receipt is not None
    assert receipt["attempt_count"] == 2
    assert receipt["last_outcome"] == "running"
