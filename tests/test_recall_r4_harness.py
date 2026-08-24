from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_r4_harness_test_module", ROOT / "scripts" / "recall_r4_harness.py"
)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


def _git_source(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("source\n")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(
        ["git", "config", "user.email", "r4@example.invalid"], cwd=source, check=True
    )
    subprocess.run(["git", "config", "user.name", "r4"], cwd=source, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source, commit


def _receipt(payload: dict[str, object], index: int) -> dict[str, object]:
    unsigned = {
        "schema": HARNESS.RECEIPT_SCHEMA,
        "namespace": "recall-distillation",
        "receipt_id": HARNESS._sha256({"index": index, "payload": payload}),
        **payload,
    }
    return cast(dict[str, object], HARNESS._sealed(unsigned))


def _write_receipts(path: Path, receipts: list[dict[str, object]]) -> None:
    path.mkdir()
    # Keep each immutable receipt file below the harness's bounded input size;
    # production-sized local cohorts are intentionally represented by a set of
    # files rather than one unbounded JSON array.
    chunk_size = 100
    for index in range(0, len(receipts), chunk_size):
        chunk = receipts[index : index + chunk_size]
        (path / f"receipts-{index // chunk_size:03d}.json").write_text(
            json.dumps(chunk, sort_keys=True)
        )


def _local_rows(source: Path, commit: str) -> list[dict[str, object]]:
    tree = HARNESS._source_tree_digest(source)["tree_sha256"]
    rows: list[dict[str, object]] = []
    reasons = [
        "schema",
        "coverage",
        "capacity",
        "timeout",
        "preemption",
        "route_model_mismatch",
        "ok",
    ]
    for index, reason in enumerate(reasons):
        rally = f"rally-{index}"
        candidate = f"candidate-{index}"
        owner = HARNESS._expected_owner(rally, candidate)
        outcome_class = (
            "valid"
            if reason == "ok"
            else "deferred"
            if reason in HARNESS._DEFERRED_REASONS
            else "invalid"
        )
        outcome: dict[str, object] = {"class": outcome_class, "reason": reason}
        if outcome_class == "valid":
            outcome.update(schema_valid=True, coverage_valid=True)
        rows.append(
            {
                "profile": HARNESS.LOCAL_PROFILE,
                "source_commit": commit,
                "source_tree_sha256": tree,
                "rally_id": rally,
                "candidate_id": candidate,
                "primary_owner": owner,
                "probe": HARNESS._expected_probe(rally, candidate),
                "assignment_revision": HARNESS.LOCAL_ASSIGNMENT_REVISION,
                "probe_assignment_revision": HARNESS.LOCAL_PROBE_REVISION,
                "route_identity": {
                    "role": owner,
                    "provider": "local",
                    "model": {
                        role: f"model-{role[-1]}" for role in HARNESS.LOCAL_ROLES
                    }[owner],
                    "location": "local",
                },
                "lease": {
                    "kind": "LocalStructuredSession",
                    "foreground": True,
                    "inflight": 1,
                },
                "live_recall": {"unaffected": True, "remote_egress": 0},
                "max_inflight": 1,
                "failure_injection": True,
                "outcome": outcome,
            }
        )
    next_index = len(rows)
    while {row["primary_owner"] for row in rows} != set(HARNESS.LOCAL_ROLES):
        rally = f"rally-{next_index}"
        candidate = f"candidate-{next_index}"
        owner = HARNESS._expected_owner(rally, candidate)
        if owner in {row["primary_owner"] for row in rows}:
            next_index += 1
            continue
        rows.append(
            {
                "profile": HARNESS.LOCAL_PROFILE,
                "source_commit": commit,
                "source_tree_sha256": tree,
                "rally_id": rally,
                "candidate_id": candidate,
                "primary_owner": owner,
                "probe": HARNESS._expected_probe(rally, candidate),
                "assignment_revision": HARNESS.LOCAL_ASSIGNMENT_REVISION,
                "probe_assignment_revision": HARNESS.LOCAL_PROBE_REVISION,
                "route_identity": {
                    "role": owner,
                    "provider": "local",
                    "model": f"model-{owner[-1]}",
                    "location": "local",
                },
                "lease": {"kind": "LLMRuntime", "foreground": True, "inflight": 1},
                "live_recall": {"unaffected": True, "remote_egress": 0},
                "max_inflight": 1,
                "failure_injection": True,
                "outcome": {
                    "class": "valid",
                    "reason": "ok",
                    "schema_valid": True,
                    "coverage_valid": True,
                },
            }
        )
        next_index += 1
    for index in range(3000):
        rally = f"quality-rally-{index % 17}"
        candidate = f"quality-candidate-{index}"
        owner = HARNESS._expected_owner(rally, candidate)
        rows.append(
            {
                "profile": HARNESS.LOCAL_PROFILE,
                "source_commit": commit,
                "source_tree_sha256": tree,
                "rally_id": rally,
                "candidate_id": candidate,
                "primary_owner": owner,
                "probe": HARNESS._expected_probe(rally, candidate),
                "assignment_revision": HARNESS.LOCAL_ASSIGNMENT_REVISION,
                "probe_assignment_revision": HARNESS.LOCAL_PROBE_REVISION,
                "route_identity": {
                    "role": owner,
                    "provider": "local",
                    "model": f"model-{owner[-1]}",
                    "location": "local",
                },
                "lease": {"kind": "LLMRuntime", "foreground": True, "inflight": 1},
                "live_recall": {"unaffected": True, "remote_egress": 0},
                "max_inflight": 1,
                "outcome": {
                    "class": "valid",
                    "reason": "ok",
                    "schema_valid": True,
                    "coverage_valid": True,
                },
            }
        )
    return rows


def _ox_rows(source: Path, commit: str) -> list[dict[str, object]]:
    tree = HARNESS._source_tree_digest(source)["tree_sha256"]
    contract = {
        "route": HARNESS.OX_ROUTE,
        "model": HARNESS.OX_MODEL,
        "prompt_sha256": HARNESS.OX_PROMPT_SHA256,
        "schema": HARNESS.OX_SCHEMA,
        "schema_sha256": HARNESS.OX_SCHEMA_SHA256,
        "route_sha256": HARNESS.OX_ROUTE_SHA256,
        "model_sha256": HARNESS.OX_MODEL_SHA256,
        "cohort": HARNESS.OX_COHORT,
        "identity_revision": HARNESS.OX_IDENTITY_REVISION,
        "expires_at": "2099-01-01T00:00:00Z",
    }
    contract["contract_id"] = HARNESS._sha256(contract)
    rows: list[dict[str, object]] = []
    for cap in HARNESS.OX_STAGES:
        rows.append(
            {
                "profile": HARNESS.OX_PROFILE,
                "source_commit": commit,
                "source_tree_sha256": tree,
                "contract": contract,
                "negative_veto": {
                    "authenticated": True,
                    "exact_binding": True,
                    "conflicts": 0,
                },
                "blind_repeat": {
                    "revision": HARNESS.OX_PROBE_REVISION,
                    "complete": True,
                    "stability_passed": True,
                    "pairs": HARNESS.OX_MIN_BLIND_REPEAT_PAIRS,
                },
                "order_swap": {
                    "complete": True,
                    "pairs": HARNESS.OX_MIN_BLIND_REPEAT_PAIRS,
                },
                "rollback": {
                    "verified": True,
                    "active_unchanged": True,
                    "status": "not_rolled_back",
                },
                "control": {
                    "ox_enabled": True,
                    "free_only": True,
                    "no_paid_fallback": True,
                    "kill_switch_supported": True,
                    "kill_switch_tripped": False,
                },
                "stage": {
                    "cap": cap,
                    "valid_receipts": 20,
                    "attempts": 20,
                    "labels": [
                        {
                            "label_id": f"label-{cap}-{index}",
                            "commit_id": f"commit-{cap}-{index}",
                            "work_id": f"work-{cap}-{index}",
                        }
                        for index in range(20)
                    ],
                },
                "transition_receipts": [
                    {
                        "category": "429",
                        "before_cap": 10,
                        "after_cap": 5,
                        "status": "deferred",
                    },
                    {
                        "category": "5xx",
                        "attempts": 3,
                        "bounded": True,
                        "status": "deferred",
                    },
                    {
                        "category": "timeout",
                        "attempts": 3,
                        "bounded": True,
                        "status": "deferred",
                    },
                    {"category": "402", "status": "hard_stop"},
                    {"category": "paid", "status": "hard_stop"},
                    {"category": "model_drift", "status": "hard_stop"},
                ],
                "sensitive": 0,
                "raw": 0,
                "billable": 0,
                "unexpected_route": 0,
                "duplicate_label": 0,
                "duplicate_commit": 0,
                "lease_recovery": {"leased_after": 0},
            }
        )
    return rows


def test_source_contract_passes_but_production_stays_false_without_attestation(
    tmp_path: Path,
) -> None:
    source, commit = _git_source(tmp_path)
    local_dir = tmp_path / "local"
    ox_dir = tmp_path / "ox"
    _write_receipts(
        local_dir,
        [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))],
    )
    _write_receipts(
        ox_dir,
        [
            _receipt(row, index + 100)
            for index, row in enumerate(_ox_rows(source, commit))
        ],
    )
    artifact, _ = HARNESS.run(
        source_root=source,
        source_commit=commit,
        output=tmp_path / "evidence",
        local_receipts=local_dir,
        ox_receipts=ox_dir,
    )
    assert artifact["source_contract"]["passed"] is True
    assert artifact["production_certification"]["passed"] is False
    assert (
        "independent_live_provider_attestation_unavailable"
        in artifact["production_certification"]["reasons"]
    )
    assert artifact["provider_calls"] == 0
    artifact_path = next((tmp_path / "evidence").glob("*.json"))
    assert (
        HARNESS.read_artifact(artifact_path)["artifact_id"] == artifact["artifact_id"]
    )


def test_forged_seal_and_arbitrary_source_hash_fail_closed(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    directory = tmp_path / "receipts"
    row = _local_rows(source, commit)[0]
    row["source_tree_sha256"] = "f" * 64
    _write_receipts(directory, [_receipt(row, 1)])
    result = HARNESS._validate_local(
        HARNESS.load_receipts(directory)[0], HARNESS._source_tree_digest(source)
    )
    assert result["passed"] is False
    assert "source_binding_mismatch" in result["reasons"]
    receipt_path = directory / "receipts-000.json"
    payload = json.loads(receipt_path.read_text())
    payload[0]["seal_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(payload))
    with pytest.raises(HARNESS.R4Error, match="seal"):
        HARNESS.load_receipts(directory)


def test_root_matrix_rejects_symlink_and_overlap(tmp_path: Path) -> None:
    source, _ = _git_source(tmp_path)
    output = tmp_path / "output"
    HARNESS.assert_root_matrix(source, output)
    with pytest.raises(HARNESS.R4Error, match="overlap"):
        HARNESS.assert_root_matrix(source, source / "nested")
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(HARNESS.R4Error, match="symlink"):
        HARNESS.assert_root_matrix(link, output)


def test_run_rejects_output_symlink_before_resolve(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    output = tmp_path / "output"
    output.symlink_to(external, target_is_directory=True)
    with pytest.raises(HARNESS.R4Error, match="symlink"):
        HARNESS.run(source_root=source, source_commit=commit, output=output)


def test_tracked_source_symlink_is_rejected(tmp_path: Path) -> None:
    source, _ = _git_source(tmp_path)
    (source / "target.txt").write_text("target\n")
    (source / "tracked-link").symlink_to("target.txt")
    subprocess.run(["git", "add", "target.txt", "tracked-link"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "symlink"], cwd=source, check=True)
    with pytest.raises(HARNESS.R4Error, match="tracked symlink"):
        HARNESS._source_tree_digest(source)


def test_dirty_source_is_rejected_before_receipt_reads(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    (source / "README.md").write_text("dirty\n")
    with pytest.raises(HARNESS.R4Error, match="dirty"):
        HARNESS.run(
            source_root=source, source_commit=commit, output=tmp_path / "evidence"
        )


def test_local_skew_and_probe_revision_are_fixed_gates(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    rows = _local_rows(source, commit)
    quality = [row for row in rows if row.get("failure_injection") is not True]
    selected: list[dict[str, object]] = []
    for role, _limit in zip(HARNESS.LOCAL_ROLES, (102, 5, 1), strict=True):
        selected.extend(row for row in quality if row["primary_owner"] == role)
        selected = selected[: sum((102, 5, 1)[: HARNESS.LOCAL_ROLES.index(role) + 1])]
    skew_result = HARNESS._validate_local(selected, HARNESS._source_tree_digest(source))
    assert skew_result["passed"] is False
    assert "load_skew_above_p0" in skew_result["reasons"]
    probe_row = dict(rows[-1])
    probe_row["probe_assignment_revision"] = "probe-unknown"
    probe_result = HARNESS._validate_local(
        [*rows[:-1], probe_row], HARNESS._source_tree_digest(source)
    )
    assert probe_result["passed"] is False
    assert "probe_assignment_missing" in probe_result["reasons"]


def test_ox_negative_aliases_and_invalid_backoff_are_vetoes(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    source_snapshot = HARNESS._source_tree_digest(source)
    for alias in (
        "paid",
        "paid_calls",
        "drift",
        "route_model_drift",
        "model_drift",
        "dup",
        "duplicate",
    ):
        rows = _ox_rows(source, commit)
        rows[0][alias] = 1
        result = HARNESS._validate_ox(rows, source_snapshot)
        assert result["passed"] is False
        assert any(alias in reason for reason in result["reasons"])
    rows = _ox_rows(source, commit)
    rows[0]["transition_receipts"][0]["before_cap"] = -1  # type: ignore[index]
    result = HARNESS._validate_ox(rows, source_snapshot)
    assert result["passed"] is False
    assert "429_halving_invalid" in result["reasons"]


def test_production_boolean_is_not_an_attestation(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    result = HARNESS._validate_production_attestations(
        [{"production_certification": True, "producer": "fixture"}],
        HARNESS._source_tree_digest(source),
    )
    assert result["passed"] is False
    assert result["reasons"]


def test_self_reported_attestation_is_rejected(tmp_path: Path) -> None:
    source, _ = _git_source(tmp_path)
    result = HARNESS._validate_production_attestations(
        [{"production_certification": True, "producer": "fixture"}],
        HARNESS._source_tree_digest(source),
    )
    assert result["passed"] is False
    assert result["reasons"] == ["independent_live_provider_attestation_unavailable"]


def test_default_cli_requires_production_certification(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    local_dir = tmp_path / "local"
    ox_dir = tmp_path / "ox"
    _write_receipts(
        local_dir,
        [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))],
    )
    _write_receipts(
        ox_dir,
        [
            _receipt(row, index + 100)
            for index, row in enumerate(_ox_rows(source, commit))
        ],
    )
    common = [
        "--source-root",
        str(source),
        "--source-commit",
        commit,
        "--local-receipts",
        str(local_dir),
        "--ox-receipts",
        str(ox_dir),
    ]
    result = HARNESS.main(
        [
            *common,
            "--output",
            str(tmp_path / "evidence"),
        ]
    )
    assert result == 3
    source_only = HARNESS.main(
        [
            *common,
            "--output",
            str(tmp_path / "source-only-evidence"),
            "--source-contract-only",
        ]
    )
    assert source_only == 0
    child = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "recall_r4_harness.py"),
            *common,
            "--output",
            str(tmp_path / "fresh-subprocess-evidence"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 3, child.stderr
    child_payload = json.loads(child.stdout)
    assert child_payload["source_contract"] is True
    assert child_payload["production_certification"] is False
