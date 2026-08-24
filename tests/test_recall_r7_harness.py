from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_r7_harness_test", ROOT / "scripts" / "recall_r7_harness.py"
)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def _id(value: int) -> str:
    return f"{value:064x}"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fixture(*, synthetic: bool = True) -> dict[str, object]:
    baseline, candidate, lkg = _id(1), _id(2), _id(3)
    commit = "a" * 40
    identity = {
        "active_id": candidate,
        "baseline_id": baseline,
        "candidate_id": candidate,
        "lkg_id": lkg,
        "policy_id": candidate,
        "source_commit": commit,
    }
    receipts = {
        name: {"kind": f"recall-r7-{name}-receipt", "identity": identity}
        for name in ("runtime", "archive", "process", "health", "api", "dom", "policy")
    }
    receipt_digests = {name: _digest(value) for name, value in receipts.items()}
    now = datetime(2026, 8, 24, tzinfo=UTC)
    metrics = {
        "quality": {
            "baseline_successes": 450,
            "candidate_successes": 500,
            "total": 500,
        },
        "coverage": {"successes": 500, "total": 500},
        "abstain": {"baseline": 0, "candidate": 0, "total": 500},
        "latency": {
            "p95_ms": 100,
            "deadline_ms": 100,
            "deadline_breaches": 0,
            "timeout_count": 0,
            "total": 500,
        },
        "resource": {
            "worker_count": 1,
            "declared_max_workers": 1,
            "resource_violations": 0,
        },
        "integrity": {
            "anchor_retained": True,
            "blind_repeat": True,
            "feature_bytes_sha256": candidate,
            "negative_vetoes": 0,
            "order_swap": True,
        },
    }
    stages = []
    previous = None
    for index, (stage, percent) in enumerate(HARNESS.STAGES, 10):
        run_id = _id(index)
        pairs = [
            {
                "receipt_id": _id(1000 + pair),
                "host": "h1",
                "cohort": "c1",
                "run_id": run_id,
                "stage": stage,
            }
            for pair in range(500)
        ]
        started = now - timedelta(days=32 - 8 * (index - 10))
        stages.append(
            {
                "stage": stage,
                "rollout_percent": percent,
                "run_id": run_id,
                "stage_started_at": started.isoformat(),
                "observed_at": (started + timedelta(days=7)).isoformat(),
                "same_decision_paired_eligible": 500,
                "pairs": pairs,
                "declared_minimums": {"hosts": 1, "cohorts": 1},
                "hosts": ["h1"],
                "cohorts": ["c1"],
                "observation_mode": "paired",
                "identity_receipts": receipt_digests,
                "previous_stage_run_id": previous,
                "stage_reset": index != 10,
                "metrics": metrics,
            }
        )
        previous = run_id
    locked = {
        "schema": HARNESS.LOCKED_REPLAY_SCHEMA,
        "namespace": "recall-distillation",
        "placeholder": False,
        "replay_status": "complete",
        "synthetic_fixture": synthetic,
        "splits": {"train": 70, "validation": 15, "test": 15, "embargo": True},
        "boundaries": {
            "train_end": (now - timedelta(days=10)).isoformat(),
            "validation_start": (now - timedelta(days=9)).isoformat(),
            "validation_end": (now - timedelta(days=8)).isoformat(),
            "embargo_start": (now - timedelta(days=7)).isoformat(),
            "embargo_end": (now - timedelta(days=6)).isoformat(),
            "test_start": (now - timedelta(days=5)).isoformat(),
            "test_end": (now - timedelta(days=4)).isoformat(),
        },
        "probes": {"blind_repeat": True, "order_swap": True, "negative_veto": True},
        "feature_bytes_sha256": candidate,
        "identity_receipts": receipt_digests,
    }
    locked["seal_sha256"] = _digest(locked)
    failure = {
        "kind": "recall-r7-forced-failure-receipt",
        "deterministic_failure": True,
        "rolled_back": True,
        "learning_halted": True,
        "rollout_percent": 0,
        "active_id": lkg,
        "candidate_cleared": True,
        "candidate_id": candidate,
        "lkg_id": lkg,
        "quarantine_id": _id(99),
        "identity_receipts": receipt_digests,
    }
    return {
        "locked_replay": locked,
        "stages": stages,
        "forced_failure": failure,
        "receipts": receipts,
        "baseline_id": baseline,
        "candidate_id": candidate,
        "lkg_id": lkg,
        "source_commit": commit,
        "now": now,
    }


def _validate(fixture: dict[str, object]) -> dict[str, object]:
    return HARNESS.validate_bundle(**fixture)  # type: ignore[arg-type]


def test_happy_synthetic_validator_fixture_is_never_certification() -> None:
    result = _validate(_fixture())
    assert result["synthetic_fixture"] is True
    assert result["certification"] is False
    assert len(result["stages"]) == 4


@pytest.mark.parametrize(
    "path, message",
    [
        (("stages", 0, "metrics", "quality", "total"), "denominator"),
        (("stages", 0, "stage_started_at"), "seven-day"),
    ],
)
def test_insufficient_denominator_and_days_fail_closed(
    path: tuple[object, ...], message: str
) -> None:
    fixture = _fixture()
    target: object = fixture
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = 1 if path[-1] == "total" else fixture["now"].isoformat()  # type: ignore[index,union-attr]
    with pytest.raises(HARNESS.R7Error, match=message):
        _validate(fixture)


def test_placeholder_replay_and_duplicate_and_mixed_host_fail_closed() -> None:
    fixture = _fixture()
    fixture["locked_replay"]["placeholder"] = True  # type: ignore[index]
    unsigned = {
        key: value
        for key, value in fixture["locked_replay"].items()
        if key != "seal_sha256"
    }  # type: ignore[index]
    fixture["locked_replay"]["seal_sha256"] = _digest(unsigned)  # type: ignore[index]
    with pytest.raises(HARNESS.R7Error, match="placeholder"):
        _validate(fixture)
    fixture = _fixture()
    fixture["stages"][0]["pairs"][1]["receipt_id"] = fixture["stages"][0]["pairs"][0][
        "receipt_id"
    ]  # type: ignore[index]
    with pytest.raises(HARNESS.R7Error, match="duplicate"):
        _validate(fixture)
    fixture = _fixture()
    fixture["stages"][0]["pairs"][0]["host"] = "foreign"  # type: ignore[index]
    with pytest.raises(HARNESS.R7Error, match="mixed"):
        _validate(fixture)


def test_candidate_only_is_rejected_before_100_and_rollback_must_be_complete() -> None:
    fixture = _fixture()
    fixture["stages"][1]["observation_mode"] = "candidate_only_legacy_incumbent"  # type: ignore[index]
    with pytest.raises(HARNESS.R7Error, match="only at 100"):
        _validate(fixture)
    fixture = _fixture()
    fixture["forced_failure"]["learning_halted"] = False  # type: ignore[index]
    with pytest.raises(HARNESS.R7Error, match="rollback is incomplete"):
        _validate(fixture)


def test_identity_drift_and_path_safety_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture()
    fixture["receipts"]["api"]["identity"]["source_commit"] = "b" * 40  # type: ignore[index]
    with pytest.raises(HARNESS.R7Error, match="drift"):
        _validate(fixture)
    production, source = tmp_path / "production", tmp_path / "source"
    production.mkdir()
    source.mkdir()
    link = tmp_path / "link"
    link.symlink_to(production, target_is_directory=True)
    with pytest.raises(HARNESS.R7Error, match="symlink"):
        HARNESS._assert_paths(link, source, tmp_path / "output", [])
    with pytest.raises(HARNESS.R7Error, match="overlap"):
        HARNESS._assert_paths(production, source, production / "output", [])


def test_dirty_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    (source / "tracked.txt").write_text("one")
    subprocess.run(["git", "add", "tracked.txt"], cwd=source, check=True)
    subprocess.run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "base"],
        cwd=source,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert HARNESS._source_identity(source, commit) == {"source_commit": commit}
    (source / "tracked.txt").write_text("dirty")
    with pytest.raises(HARNESS.R7Error, match="dirty"):
        HARNESS._source_identity(source, commit)


def test_main_removes_temporary_clone_after_writing_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production, source, clone = (
        tmp_path / "production",
        tmp_path / "source",
        tmp_path / "clone",
    )
    production.mkdir()
    source.mkdir()
    clone.mkdir()
    receipt_paths = {
        name: tmp_path / f"{name}.json"
        for name in ("runtime", "archive", "process", "health", "api", "dom", "policy")
    }
    stages = [tmp_path / f"stage-{index}.json" for index in range(4)]
    monkeypatch.setattr(HARNESS.R0, "_clone", lambda *_args: (clone, True))
    monkeypatch.setattr(
        HARNESS, "_source_identity", lambda *_args: {"source_commit": "a" * 40}
    )
    monkeypatch.setattr(
        HARNESS, "_tree_state", lambda _path: {"files": 0, "state_sha256": "x"}
    )
    monkeypatch.setattr(HARNESS, "_read_json", lambda *_args: ({}, "x"))
    monkeypatch.setattr(
        HARNESS, "validate_bundle", lambda **_kwargs: {"synthetic_fixture": False}
    )
    store = SimpleNamespace(
        write_immutable=lambda output, payload, schema: (
            _id(88),
            output / "artifact.json",
            {"schema": schema},
        )
    )
    monkeypatch.setattr(
        HARNESS.R0, "_load", lambda _source: (None, None, store, None, None)
    )
    argv = [
        "--production-root",
        str(production),
        "--source-root",
        str(source),
        "--source-commit",
        "a" * 40,
        "--output",
        str(tmp_path / "output"),
        "--baseline-id",
        _id(1),
        "--candidate-id",
        _id(2),
        "--lkg-id",
        _id(3),
        "--locked-replay",
        str(tmp_path / "locked.json"),
        "--forced-failure-receipt",
        str(tmp_path / "failure.json"),
    ]
    for path in stages:
        argv.extend(("--stage-receipt", str(path)))
    for name, path in receipt_paths.items():
        argv.extend((f"--{name}-receipt", str(path)))
    assert HARNESS.main(argv) == 0
    assert not clone.exists()
