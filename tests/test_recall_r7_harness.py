from __future__ import annotations

import importlib.util
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_r7_harness_test", ROOT / "scripts" / "recall_r7_harness.py"
)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def _id(number: int) -> str:
    return f"{number:064x}"


def _sealed(value: dict[str, Any]) -> dict[str, Any]:
    value["seal_sha256"] = HARNESS._digest(
        {key: item for key, item in value.items() if key != "seal_sha256"}
    )
    return value


def _stage_seal(stage: dict[str, Any]) -> None:
    rows_sha = HARNESS._digest(stage["rows"])
    polls = stage["poll_history"]
    for poll in polls:
        poll["rows_sha256"] = rows_sha
        _sealed(poll)
    stage["stage_started_at"] = polls[0]["polled_at"]
    _sealed(stage)


def _fixture() -> dict[str, Any]:
    baseline, candidate, lkg, feature = _id(1), _id(2), _id(3), _id(4)
    source_commit = "a" * 40
    now = datetime(2026, 8, 24, tzinfo=UTC)
    artifacts: dict[str, dict[str, Any]] = {}
    for name, policy_id in (
        ("baseline", baseline),
        ("candidate", candidate),
        ("lkg", lkg),
    ):
        artifacts[name] = _sealed(
            {
                "schema": (
                    HARNESS.BASELINE_SCHEMA
                    if name == "baseline"
                    else HARNESS.POLICY_SCHEMA
                ),
                "namespace": "recall-distillation",
                "artifact_id": policy_id,
                "feature_bytes_sha256": feature,
            }
        )
    for name, policy_id in (
        ("active", candidate),
        ("candidate", candidate),
        ("lkg", lkg),
    ):
        artifacts[f"{name}_pointer"] = _sealed(
            {
                "schema": HARNESS.STORE_SCHEMA,
                "namespace": "recall-distillation",
                "kind": f"{name}-policy-pointer",
                "policy_id": policy_id,
            }
        )
    identity = {
        "active_id": candidate,
        "baseline_id": baseline,
        "candidate_id": candidate,
        "lkg_id": lkg,
        "policy_id": candidate,
        "source_commit": source_commit,
    }
    actual = {
        "runtime": {"runtime_commit": source_commit},
        "archive": {
            "archive_commit": source_commit,
            "direct_url": "https://example.invalid/archive",
        },
        "process": {
            "process_commit": source_commit,
            "pid": 42,
            "started_at": now.isoformat(),
        },
        "health": {"health_commit": source_commit, "status": "ok"},
        "api": {"api_commit": source_commit, "status": 200},
        "dom": {"dom_commit": source_commit, "status": "ready"},
        "policy": {
            "active_pointer": candidate,
            "candidate_pointer": candidate,
            "lkg_pointer": lkg,
        },
    }
    receipts = {
        name: _sealed(
            {
                "schema": f"chronovisor.recall-r7-{name}-receipt.v1",
                "namespace": "recall-distillation",
                "kind": f"recall-r7-{name}-receipt",
                "identity": identity,
                "actual": payload,
            }
        )
        for name, payload in actual.items()
    }
    refs = {name: HARNESS._digest(value) for name, value in receipts.items()} | {
        name: HARNESS._digest(value) for name, value in artifacts.items()
    }
    locked_rows = []
    for index in range(100):
        split = "train" if index < 70 else "validation" if index < 85 else "test"
        locked_rows.append(
            {
                "row_id": _id(1_000 + index),
                "split": split,
                "decision_sha256": _id(2_000 + index),
                "session_sha256": _id(3_000 + index),
                "query_sha256": _id(4_000 + index),
                "candidate_pool_sha256": _id(5_000 + index),
                "feature_bytes_sha256": feature,
                "timestamp": (now - timedelta(days=80)).isoformat(),
                "read_only": True,
                "route_probe": True,
                "ox_blind": True,
                "order_swap": True,
                "counterfactual": True,
                "negative_veto": False,
            }
        )
    locked = _sealed(
        {
            "schema": HARNESS.LOCKED_REPLAY_SCHEMA,
            "namespace": "recall-distillation",
            "kind": "locked-replay",
            "synthetic_fixture": False,
            "provenance": "production-immutable-locked-replay",
            "splits": {"train": 70, "validation": 15, "test": 15, "embargo": True},
            "rows": locked_rows,
            "identity_refs": refs,
        }
    )
    stages = []
    previous_run: str | None = None
    for stage_index, (stage_name, percent) in enumerate(HARNESS.STAGES):
        run_id = _id(10_000 + stage_index)
        start = now - timedelta(days=40 - stage_index * 8)
        end = start + timedelta(days=7)
        rows = [
            {
                "receipt_id": _id(20_000 + stage_index * 1_000 + row),
                "decision_sha256": _id(30_000 + stage_index * 1_000 + row),
                "session_sha256": _id(40_000 + stage_index * 1_000 + row),
                "query_sha256": _id(50_000 + stage_index * 1_000 + row),
                "candidate_pool_sha256": _id(60_000 + stage_index * 1_000 + row),
                "feature_bytes_sha256": feature,
                "observed_at": (start + timedelta(days=1 + row % 2)).isoformat(),
                "host": "host-a",
                "cohort": "cohort-a",
                "baseline_quality": row < 450,
                "candidate_quality": True,
                "baseline_covered": True,
                "candidate_covered": True,
                "baseline_abstained": False,
                "candidate_abstained": False,
                "candidate_score_ms": 180,
                "live_latency_ms": 300,
                "timed_out": False,
                "deadline_ms": 1_200,
                "worker_id": "worker-a",
                "resource_ok": True,
                "integrity_ok": True,
                "negative_veto": False,
            }
            for row in range(500)
        ]
        polls = [
            {
                "schema": "chronovisor.recall-r7-poll.v1",
                "namespace": "recall-distillation",
                "kind": "immutable-stage-poll",
                "artifact_id": _id(70_000 + stage_index * 2 + poll),
                "stage": stage_name,
                "run_id": run_id,
                "polled_at": (start if poll == 0 else end).isoformat(),
                "rows_sha256": "",
            }
            for poll in range(2)
        ]
        stage = {
            "schema": "chronovisor.recall-r7-stage-receipt.v1",
            "namespace": "recall-distillation",
            "kind": "stage-receipt",
            "stage": stage_name,
            "rollout_percent": percent,
            "run_id": run_id,
            "stage_started_at": start.isoformat(),
            "poll_history": polls,
            "rows": rows,
            "stratum_minimums": {"host": 500, "cohort": 500},
            "observation_mode": "paired",
            "identity_refs": refs,
            "locked_replay_sha256": HARNESS._digest(
                {key: item for key, item in locked.items() if key != "seal_sha256"}
            ),
            "feature_bytes_sha256": feature,
            "previous_run_id": previous_run,
            "legacy_incumbent_proof": None,
        }
        _stage_seal(stage)
        stages.append(stage)
        previous_run = run_id
    final = stages[-1]
    rollback = _sealed(
        {
            "schema": "chronovisor.recall-r7-forced-failure-receipt.v1",
            "namespace": "recall-distillation",
            "kind": "forced-failure-receipt",
            "run_id": final["run_id"],
            "stage": "100",
            "failure_at": (now - timedelta(days=8)).isoformat(),
            "stage_seal_sha256": final["seal_sha256"],
            "last_poll_seal_sha256": final["poll_history"][-1]["seal_sha256"],
            "identity_refs": refs,
            "deterministic_failure": True,
            "rolled_back": True,
            "learning_halted": True,
            "rollout_percent": 0,
            "rollback_state": {
                "active_policy_id": lkg,
                "candidate_policy_id": None,
                "lkg_policy_id": lkg,
            },
            "quarantine_id": _id(90_000),
            "rollback_receipt_id": _id(90_001),
            "rollback_receipt_sha256": _id(90_002),
        }
    )
    return {
        "locked_replay": locked,
        "stages": stages,
        "forced_failure": rollback,
        "receipts": receipts,
        "artifacts": artifacts,
        "baseline_id": baseline,
        "candidate_id": candidate,
        "lkg_id": lkg,
        "source_commit": source_commit,
        "now": now,
    }


def _validate(fixture: dict[str, Any]) -> dict[str, Any]:
    return HARNESS.validate_bundle(**fixture)


def test_happy_real_receipt_fixture_recomputes_all_fixed_gates() -> None:
    result = _validate(_fixture())
    assert result["certification"] is False
    assert (
        result["certification_reason"]
        == "trusted_active_host_cohort_inventory_unavailable"
    )
    assert result["stages"][-1]["metrics"]["candidate_score_p95_ms"] == 180
    assert result["stages"][-1]["metrics"]["live_p95_ms"] == 300


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("locked_replay", "synthetic_fixture"), True, "synthetic/source-only"),
        (("locked_replay", "provenance"), "synthetic-fixture", "synthetic/source-only"),
        (("stages", 0, "rows", 0, "candidate_score_ms"), 181, "fixed gate"),
        (("stages", 0, "rows", 0, "live_latency_ms"), 900, "fixed gate"),
        (("stages", 0, "rows", 0, "deadline_ms"), 1201, "hard deadline"),
    ],
)
def test_known_false_passes_are_rejected(
    path: tuple[object, ...], value: object, message: str
) -> None:
    fixture = _fixture()
    target: Any = fixture
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path[0] == "locked_replay":
        _sealed(fixture["locked_replay"])
    else:
        if path[-1] in {"candidate_score_ms", "live_latency_ms"}:
            for row in fixture["stages"][0]["rows"]:
                row[path[-1]] = value
        _stage_seal(fixture["stages"][0])
    with pytest.raises(HARNESS.R7Error, match=message):
        _validate(fixture)


def test_rejects_cross_stage_reuse_poll_shortcut_tamper_and_early_legacy() -> None:
    fixture = _fixture()
    fixture["stages"][1]["rows"][0]["receipt_id"] = fixture["stages"][0]["rows"][0][
        "receipt_id"
    ]
    _stage_seal(fixture["stages"][1])
    with pytest.raises(HARNESS.R7Error, match="reuse"):
        _validate(fixture)
    fixture = _fixture()
    for key in (
        "decision_sha256",
        "session_sha256",
        "query_sha256",
        "candidate_pool_sha256",
    ):
        fixture["stages"][0]["rows"][1][key] = fixture["stages"][0]["rows"][0][key]
    _stage_seal(fixture["stages"][0])
    with pytest.raises(HARNESS.R7Error, match="duplicate paired observation"):
        _validate(fixture)
    fixture = _fixture()
    timestamp = fixture["stages"][0]["rows"][0]["observed_at"]
    for row in fixture["stages"][0]["rows"]:
        row["observed_at"] = timestamp
    _stage_seal(fixture["stages"][0])
    with pytest.raises(HARNESS.R7Error, match="one synthetic timestamp"):
        _validate(fixture)
    fixture = _fixture()
    fixture["stages"][0]["poll_history"][1]["polled_at"] = fixture["stages"][0][
        "poll_history"
    ][0]["polled_at"]
    _stage_seal(fixture["stages"][0])
    with pytest.raises(HARNESS.R7Error, match="seven-day"):
        _validate(fixture)
    fixture = _fixture()
    fixture["receipts"]["api"]["actual"]["status"] = 500
    with pytest.raises(HARNESS.R7Error, match="seal mismatch"):
        _validate(fixture)
    fixture = _fixture()
    stage = fixture["stages"][1]
    stage["observation_mode"] = "candidate_only_legacy_incumbent"
    stage["legacy_incumbent_proof"] = _sealed(
        {
            "schema": "chronovisor.recall-r7-legacy-incumbent-proof.v1",
            "namespace": "recall-distillation",
            "kind": "legacy-incumbent-proof",
            "incumbent_unavailable": True,
            "run_id": stage["run_id"],
            "stage": "5",
        }
    )
    _stage_seal(stage)
    with pytest.raises(HARNESS.R7Error, match="only at 100"):
        _validate(fixture)


def test_identity_pointer_and_rollback_binding_fail_closed() -> None:
    fixture = _fixture()
    fixture["artifacts"]["candidate_pointer"]["policy_id"] = _id(99)
    _sealed(fixture["artifacts"]["candidate_pointer"])
    with pytest.raises(HARNESS.R7Error, match="identity drift"):
        _validate(fixture)
    fixture = _fixture()
    fixture["forced_failure"]["run_id"] = _id(99)
    _sealed(fixture["forced_failure"])
    with pytest.raises(HARNESS.R7Error, match="binding/state"):
        _validate(fixture)


def test_rejects_nonadjacent_stage_run_reuse_and_missing_trusted_roster() -> None:
    fixture = _fixture()
    reused = fixture["stages"][2]
    reused["run_id"] = fixture["stages"][0]["run_id"]
    for poll in reused["poll_history"]:
        poll["run_id"] = reused["run_id"]
    _stage_seal(reused)
    with pytest.raises(HARNESS.R7Error, match="globally reused"):
        _validate(fixture)
    result = _validate(_fixture())
    assert result["certification"] is False
    assert all(
        stage["reason"] == "trusted_active_host_cohort_inventory_unavailable"
        for stage in result["stages"]
    )


def test_100_percent_legacy_incumbent_requires_and_honors_sealed_proof() -> None:
    fixture = _fixture()
    stage = fixture["stages"][-1]
    stage["observation_mode"] = "candidate_only_legacy_incumbent"
    for row in stage["rows"]:
        row["baseline_quality"] = None
        row["baseline_covered"] = None
        row["baseline_abstained"] = None
    stage["legacy_incumbent_proof"] = _sealed(
        {
            "schema": "chronovisor.recall-r7-legacy-incumbent-proof.v1",
            "namespace": "recall-distillation",
            "kind": "legacy-incumbent-proof",
            "incumbent_unavailable": True,
            "run_id": stage["run_id"],
            "stage": "100",
        }
    )
    _stage_seal(stage)
    fixture["forced_failure"]["stage_seal_sha256"] = stage["seal_sha256"]
    fixture["forced_failure"]["last_poll_seal_sha256"] = stage["poll_history"][-1][
        "seal_sha256"
    ]
    _sealed(fixture["forced_failure"])
    assert _validate(fixture)["certification"] is False


def test_paths_dirty_source_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production, source = tmp_path / "production", tmp_path / "source"
    production.mkdir()
    source.mkdir()
    link = tmp_path / "link"
    link.symlink_to(production, target_is_directory=True)
    with pytest.raises(HARNESS.R7Error, match="symlink"):
        HARNESS._assert_paths(link, source, tmp_path / "output", [])
    with pytest.raises(HARNESS.R7Error, match="evidence input"):
        HARNESS._assert_paths(
            production,
            source,
            tmp_path / "output",
            [tmp_path / "output" / "locked.json"],
        )
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    (source / "x").write_text("x")
    subprocess.run(["git", "add", "x"], cwd=source, check=True)
    subprocess.run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "x"],
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
    (source / "x").write_text("dirty")
    with pytest.raises(HARNESS.R7Error, match="dirty"):
        HARNESS._source_identity(source, commit)
    clone = tmp_path / "clone"
    clone.mkdir()
    monkeypatch.setattr(HARNESS.R0, "_clone", lambda *_args: (clone, True))
    monkeypatch.setattr(HARNESS, "_clone_state", lambda _path: {"ok": True})
    monkeypatch.setattr(
        HARNESS,
        "_source_identity",
        lambda *_args: {
            "source_commit": "a" * 40,
            "source_tree_sha256": _id(4),
            "source_bytes_sha256": _id(5),
        },
    )
    monkeypatch.setattr(HARNESS, "_read_json", lambda *_args: {})
    monkeypatch.setattr(HARNESS, "validate_bundle", lambda **_kwargs: {"ok": True})
    store = SimpleNamespace(
        write_immutable=lambda output, payload, schema: (
            _id(1),
            output / "a.json",
            {"schema": schema},
        )
    )
    monkeypatch.setattr(
        HARNESS.R0, "_load", lambda _source: (None, None, store, None, None)
    )
    arguments = [
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
        str(tmp_path / "forced.json"),
    ]
    for index in range(4):
        arguments.extend(("--stage-receipt", str(tmp_path / f"stage-{index}.json")))
    for name in ("runtime", "archive", "process", "health", "api", "dom", "policy"):
        arguments.extend((f"--{name}-receipt", str(tmp_path / f"{name}.json")))
    for name in ("baseline", "candidate", "lkg"):
        arguments.extend((f"--{name}-artifact", str(tmp_path / f"{name}.json")))
    for name in ("active", "candidate", "lkg"):
        arguments.extend((f"--{name}-pointer", str(tmp_path / f"{name}-pointer.json")))
    assert HARNESS.main(arguments) == 1
    assert not clone.exists()


def test_source_identity_allows_ignored_environment_but_rejects_ignored_code(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "src" / "chronovisor").mkdir(parents=True)
    (source / "src" / "chronovisor" / "__init__.py").write_text("")
    (source / ".gitignore").write_text(
        ".venv/\nsrc/chronovisor/injected\nsrc/chronovisor/injected.*\n"
        "scripts/recall_r7_payload\nsrc/chronovisor/__pycache__/\n"
        "src/chronovisor/.pytest_cache/\nsrc/chronovisor/.mypy_cache/\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "x"],
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
    (source / ".venv").mkdir()
    (source / ".venv" / "cache.py").write_text("cache")
    for cache in ("__pycache__", ".pytest_cache", ".mypy_cache"):
        cache_dir = source / "src" / "chronovisor" / cache
        cache_dir.mkdir()
        (cache_dir / "cache.pyc").write_text("cache")
    clean = HARNESS._source_identity(source, commit)
    assert clean["source_commit"] == commit
    assert HARNESS._HEX.fullmatch(clean["source_tree_sha256"])
    for name in (
        "injected.py",
        "injected.pyc",
        "injected.so",
        "injected.dylib",
        "injected.pyd",
        "injected",
    ):
        path = source / "src" / "chronovisor" / name
        path.write_text("payload")
        with pytest.raises(HARNESS.R7Error, match="ignored protected source drift"):
            HARNESS._source_identity(source, commit)
        path.unlink()
    payload = source / "scripts" / "recall_r7_payload"
    payload.parent.mkdir()
    payload.write_text("payload")
    with pytest.raises(HARNESS.R7Error, match="ignored protected source drift"):
        HARNESS._source_identity(source, commit)
