#!/usr/bin/env python3
"""Fail-closed formal validator for a real-time Recall R7/P8 rollout.

This program does not run Recall, call a service, or mutate a rollout.  It
only accepts sealed replay and recorded receipts, then writes a small immutable
verdict.  The command's clock is always the system UTC clock; ``validate_bundle``
accepts a clock solely so unit tests can exercise time branches.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any

R7_SCHEMA = "chronovisor.recall-r7.v1"
LOCKED_REPLAY_SCHEMA = "chronovisor.recall-r7-locked-replay.v1"
MAX_INPUT_BYTES = 2 * 1024 * 1024
MIN_DAYS = 7
MIN_PAIRED = 500
MAX_ABSTAIN_DELTA = 0.02
MAX_TIMEOUT_RATE = 0.01
MAX_P95_MS = 4_000
STAGES = (("shadow", 0), ("5", 5), ("25", 25), ("100", 100))
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class R7Error(ValueError):
    """A formal R7 receipt is missing, inconsistent, or below a fixed gate."""


def _load_sibling(name: str) -> Any:
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"chronovisor_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise R7Error(f"{path.name} helper unavailable")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


R0 = _load_sibling("recall_r0_harness.py")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise R7Error(f"{label} is not a sha256 identity")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise R7Error(f"{label} is invalid")
    return value


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise R7Error(f"{label} is not UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise R7Error(f"{label} is not UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise R7Error(f"{label} is not UTC")
    return parsed.astimezone(UTC)


def wilson_interval(successes: object, total: object) -> dict[str, float]:
    """Return a 95% Wilson interval; no point-estimate fallback exists."""

    passed = _integer(successes, "Wilson successes")
    samples = _integer(total, "Wilson denominator", minimum=1)
    if passed > samples:
        raise R7Error("Wilson successes exceed denominator")
    z = NormalDist().inv_cdf(0.975)
    point = passed / samples
    denominator = 1 + z * z / samples
    center = (point + z * z / (2 * samples)) / denominator
    radius = (
        z
        * (point * (1 - point) / samples + z * z / (4 * samples * samples)) ** 0.5
        / denominator
    )
    return {
        "point": point,
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
    }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R7Error(f"{label} is not an object")
    return value


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        before = path.lstat()
        if path.is_symlink() or not path.is_file() or before.st_size > MAX_INPUT_BYTES:
            raise R7Error(f"{label} path is unsafe")
        content = path.read_bytes()
        after = path.lstat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise R7Error(f"{label} changed during read")
        value = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise R7Error(f"{label} cannot be read") from exc
    if not isinstance(value, dict):
        raise R7Error(f"{label} is not an object")
    return value, _sha256_bytes(content)


def _assert_identity(
    identity: object,
    *,
    baseline_id: str,
    candidate_id: str,
    lkg_id: str,
    source_commit: str,
) -> dict[str, str]:
    row = _mapping(identity, "identity")
    expected = {
        "active_id": candidate_id,
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "lkg_id": lkg_id,
        "policy_id": candidate_id,
        "source_commit": source_commit,
    }
    if set(row) != set(expected):
        raise R7Error("identity schema is not closed")
    for key, value in expected.items():
        actual = row.get(key)
        if actual != value:
            raise R7Error(f"identity {key} drift")
    return expected


def _receipt_digests(
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    baseline_id: str,
    candidate_id: str,
    lkg_id: str,
    source_commit: str,
) -> dict[str, str]:
    expected_kinds = {"runtime", "archive", "process", "health", "api", "dom", "policy"}
    if set(receipts) != expected_kinds:
        raise R7Error("identity receipt set is incomplete")
    digests: dict[str, str] = {}
    for name, receipt in receipts.items():
        if receipt.get("kind") != f"recall-r7-{name}-receipt":
            raise R7Error(f"{name} receipt kind mismatch")
        _assert_identity(
            receipt.get("identity"),
            baseline_id=baseline_id,
            candidate_id=candidate_id,
            lkg_id=lkg_id,
            source_commit=source_commit,
        )
        digests[name] = _sha256_bytes(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        )
    return digests


def _validate_locked_replay(
    replay: Mapping[str, Any], *, receipt_digests: Mapping[str, str]
) -> dict[str, Any]:
    if (
        replay.get("schema") != LOCKED_REPLAY_SCHEMA
        or replay.get("namespace") != "recall-distillation"
    ):
        raise R7Error("locked replay schema mismatch")
    # Store's seal check is deliberately duplicated here so this pure validator
    # remains usable in tests without source-root module binding.
    unsigned = {key: value for key, value in replay.items() if key != "seal_sha256"}
    seal = _sha256_bytes(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    )
    if replay.get("seal_sha256") != seal:
        raise R7Error("locked replay seal mismatch")
    if (
        replay.get("placeholder") is not False
        or replay.get("replay_status") != "complete"
    ):
        raise R7Error("locked replay is a placeholder")
    splits = _mapping(replay.get("splits"), "locked replay splits")
    if set(splits) != {"train", "validation", "test", "embargo"} or (
        splits.get("train"),
        splits.get("validation"),
        splits.get("test"),
        splits.get("embargo"),
    ) != (70, 15, 15, True):
        raise R7Error("locked replay 70/15/15 embargo contract failed")
    probes = _mapping(replay.get("probes"), "locked replay probes")
    if probes != {"blind_repeat": True, "order_swap": True, "negative_veto": True}:
        raise R7Error("locked replay blinded probe contract failed")
    boundaries = _mapping(replay.get("boundaries"), "locked replay boundaries")
    boundary_keys = (
        "train_end",
        "validation_start",
        "validation_end",
        "embargo_start",
        "embargo_end",
        "test_start",
        "test_end",
    )
    if set(boundaries) != set(boundary_keys):
        raise R7Error("locked replay boundary schema is not closed")
    moments = [_utc(boundaries[key], f"locked replay {key}") for key in boundary_keys]
    if any(
        left >= right for left, right in zip(moments[:-1], moments[1:], strict=True)
    ):
        raise R7Error("locked replay train/validation/test boundaries overlap")
    feature_bytes_sha256 = _identifier(
        replay.get("feature_bytes_sha256"), "locked replay feature bytes"
    )
    if replay.get("identity_receipts") != dict(receipt_digests):
        raise R7Error("locked replay identity receipt drift")
    return {
        "locked_replay_sha256": seal,
        "synthetic_fixture": replay.get("synthetic_fixture") is True,
        "feature_bytes_sha256": feature_bytes_sha256,
    }


def _validate_metrics(metrics: object, *, feature_bytes_sha256: str) -> dict[str, Any]:
    row = _mapping(metrics, "stage metrics")
    required = {"quality", "coverage", "abstain", "latency", "resource", "integrity"}
    if set(row) != required:
        raise R7Error("stage metrics schema is not closed")
    quality = _mapping(row["quality"], "quality")
    total = _integer(quality.get("total"), "quality denominator", minimum=MIN_PAIRED)
    baseline = wilson_interval(quality.get("baseline_successes"), total)
    candidate = wilson_interval(quality.get("candidate_successes"), total)
    if candidate["lower"] < baseline["upper"]:
        raise R7Error("quality Wilson gate failed")
    coverage = _mapping(row["coverage"], "coverage")
    coverage_total = _integer(
        coverage.get("total"), "coverage denominator", minimum=MIN_PAIRED
    )
    coverage_ci = wilson_interval(coverage.get("successes"), coverage_total)
    if coverage_ci["lower"] < 0.95:
        raise R7Error("coverage Wilson gate failed")
    abstain = _mapping(row["abstain"], "abstain")
    abstain_total = _integer(
        abstain.get("total"), "abstain denominator", minimum=MIN_PAIRED
    )
    baseline_abstain = _integer(abstain.get("baseline"), "baseline abstain")
    candidate_abstain = _integer(abstain.get("candidate"), "candidate abstain")
    if (
        max(baseline_abstain, candidate_abstain) > abstain_total
        or (candidate_abstain - baseline_abstain) / abstain_total > MAX_ABSTAIN_DELTA
    ):
        raise R7Error("abstain delta gate failed")
    latency = _mapping(row["latency"], "latency")
    p95 = _integer(latency.get("p95_ms"), "p95 latency")
    deadline = _integer(latency.get("deadline_ms"), "latency deadline", minimum=1)
    breaches = _integer(latency.get("deadline_breaches"), "deadline breaches")
    timeouts = _integer(latency.get("timeout_count"), "timeout count")
    latency_total = _integer(
        latency.get("total"), "latency denominator", minimum=MIN_PAIRED
    )
    if (
        p95 > min(deadline, MAX_P95_MS)
        or breaches
        or timeouts / latency_total > MAX_TIMEOUT_RATE
    ):
        raise R7Error("latency/deadline/timeout gate failed")
    resource = _mapping(row["resource"], "resource")
    if _integer(resource.get("worker_count"), "worker count", minimum=1) > _integer(
        resource.get("declared_max_workers"), "declared worker limit", minimum=1
    ) or _integer(resource.get("resource_violations"), "resource violations"):
        raise R7Error("worker/resource gate failed")
    integrity = _mapping(row["integrity"], "integrity")
    if integrity != {
        "anchor_retained": True,
        "blind_repeat": True,
        "feature_bytes_sha256": feature_bytes_sha256,
        "negative_vetoes": 0,
        "order_swap": True,
    }:
        raise R7Error("integrity/negative-veto gate failed")
    return {
        "quality": {"baseline": baseline, "candidate": candidate},
        "coverage": coverage_ci,
    }


def _validate_stage(
    receipt: Mapping[str, Any],
    *,
    stage: str,
    percent: int,
    now: datetime,
    feature_bytes_sha256: str,
    receipt_digests: Mapping[str, str],
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if receipt.get("stage") != stage or receipt.get("rollout_percent") != percent:
        raise R7Error("stage matrix mismatch")
    run_id = _identifier(receipt.get("run_id"), f"{stage} run id")
    started = _utc(receipt.get("stage_started_at"), f"{stage} stage_started_at")
    observed = _utc(receipt.get("observed_at"), f"{stage} observed_at")
    if started > observed or observed > now or now - started < timedelta(days=MIN_DAYS):
        raise R7Error(f"{stage} stage lacks real seven-day wall time")
    if _integer(
        receipt.get("same_decision_paired_eligible"),
        f"{stage} paired denominator",
        minimum=MIN_PAIRED,
    ) != len(receipt.get("pairs", ())):
        raise R7Error(f"{stage} paired receipt denominator mismatch")
    pairs = receipt.get("pairs")
    if not isinstance(pairs, list):
        raise R7Error(f"{stage} pairs are missing")
    minimums = _mapping(receipt.get("declared_minimums"), f"{stage} declared minimums")
    if set(minimums) != {"cohorts", "hosts"}:
        raise R7Error(f"{stage} minimum schema is not closed")
    hosts = receipt.get("hosts")
    cohorts = receipt.get("cohorts")
    if (
        not isinstance(hosts, list)
        or not isinstance(cohorts, list)
        or len(set(hosts)) != len(hosts)
        or len(set(cohorts)) != len(cohorts)
    ):
        raise R7Error(f"{stage} host/cohort declaration is invalid")
    if len(hosts) < _integer(
        minimums["hosts"], f"{stage} host minimum", minimum=1
    ) or len(cohorts) < _integer(
        minimums["cohorts"], f"{stage} cohort minimum", minimum=1
    ):
        raise R7Error(f"{stage} host/cohort minimum failed")
    pair_ids: set[str] = set()
    for pair in pairs:
        item = _mapping(pair, f"{stage} pair")
        pair_id = _identifier(item.get("receipt_id"), f"{stage} pair receipt")
        if (
            pair_id in pair_ids
            or item.get("host") not in hosts
            or item.get("cohort") not in cohorts
            or item.get("run_id") != run_id
            or item.get("stage") != stage
        ):
            raise R7Error(f"{stage} duplicate or mixed host/cohort receipt")
        pair_ids.add(pair_id)
    observation = receipt.get("observation_mode")
    if observation not in {"paired", "candidate_only_legacy_incumbent"} or (
        observation != "paired" and percent != 100
    ):
        raise R7Error("candidate-only legacy incumbent is allowed only at 100 percent")
    if receipt.get("identity_receipts") != dict(receipt_digests):
        raise R7Error(f"{stage} identity receipt drift")
    if previous is None:
        if receipt.get("previous_stage_run_id") is not None:
            raise R7Error("shadow stage reset reference is invalid")
    elif (
        receipt.get("stage_reset") is not True
        or receipt.get("previous_stage_run_id") != previous["run_id"]
        or run_id == previous["run_id"]
        or started < previous["observed_at"]
    ):
        raise R7Error(f"{stage} stage/run reset failed")
    return {
        "run_id": run_id,
        "observed_at": observed,
        "metrics": _validate_metrics(
            receipt.get("metrics"), feature_bytes_sha256=feature_bytes_sha256
        ),
        "receipt_sha256": _sha256_bytes(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ),
    }


def _validate_forced_failure(
    receipt: Mapping[str, Any],
    *,
    candidate_id: str,
    lkg_id: str,
    receipt_digests: Mapping[str, str],
) -> str:
    required = {
        "kind",
        "deterministic_failure",
        "rolled_back",
        "learning_halted",
        "rollout_percent",
        "active_id",
        "candidate_cleared",
        "candidate_id",
        "lkg_id",
        "quarantine_id",
        "identity_receipts",
    }
    if (
        set(receipt) != required
        or receipt.get("kind") != "recall-r7-forced-failure-receipt"
    ):
        raise R7Error("forced-failure receipt schema is not closed")
    if (
        receipt.get("deterministic_failure") is not True
        or receipt.get("rolled_back") is not True
        or receipt.get("learning_halted") is not True
        or receipt.get("rollout_percent") != 0
        or receipt.get("active_id") != lkg_id
        or receipt.get("candidate_cleared") is not True
        or receipt.get("candidate_id") != candidate_id
        or receipt.get("lkg_id") != lkg_id
    ):
        raise R7Error("forced-failure rollback is incomplete")
    _identifier(receipt.get("quarantine_id"), "quarantine id")
    if receipt.get("identity_receipts") != dict(receipt_digests):
        raise R7Error("forced-failure identity receipt drift")
    return _sha256_bytes(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    )


def validate_bundle(
    *,
    locked_replay: Mapping[str, Any],
    stages: Sequence[Mapping[str, Any]],
    forced_failure: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    baseline_id: str,
    candidate_id: str,
    lkg_id: str,
    source_commit: str,
    now: datetime,
) -> dict[str, Any]:
    """Validate supplied evidence only; callers choose the clock for unit tests."""

    baseline_id = _identifier(baseline_id, "baseline id")
    candidate_id = _identifier(candidate_id, "candidate id")
    lkg_id = _identifier(lkg_id, "LKG id")
    if (
        len({baseline_id, candidate_id, lkg_id}) != 3
        or not isinstance(source_commit, str)
        or _COMMIT.fullmatch(source_commit) is None
    ):
        raise R7Error("policy/source identity is invalid")
    if now.tzinfo is None or now.utcoffset() != timedelta(0):
        raise R7Error("validator clock is not UTC")
    digests = _receipt_digests(
        receipts,
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        lkg_id=lkg_id,
        source_commit=source_commit,
    )
    replay = _validate_locked_replay(locked_replay, receipt_digests=digests)
    if len(stages) != len(STAGES):
        raise R7Error("stage receipt count is incomplete")
    validated: list[dict[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for input_receipt, (stage, percent) in zip(stages, STAGES, strict=True):
        result = _validate_stage(
            input_receipt,
            stage=stage,
            percent=percent,
            now=now,
            feature_bytes_sha256=replay["feature_bytes_sha256"],
            receipt_digests=digests,
            previous=previous,
        )
        validated.append(result)
        previous = result
    rollback_sha = _validate_forced_failure(
        forced_failure,
        candidate_id=candidate_id,
        lkg_id=lkg_id,
        receipt_digests=digests,
    )
    return {
        "certification": not replay["synthetic_fixture"],
        "synthetic_fixture": replay["synthetic_fixture"],
        "locked_replay_sha256": replay["locked_replay_sha256"],
        "identity_receipts": digests,
        "stages": validated,
        "forced_failure_sha256": rollback_sha,
    }


def _has_symlink_component(path: Path) -> bool:
    candidate = path.expanduser()
    return any(parent.is_symlink() for parent in (candidate, *candidate.parents))


def _assert_paths(
    production: Path, source: Path, output: Path, inputs: Sequence[Path]
) -> None:
    if any(
        _has_symlink_component(path) for path in (production, source, output, *inputs)
    ):
        raise R7Error("root/output/input path contains a symlink")
    production = production.resolve(strict=True)
    source = source.resolve(strict=True)
    output = output.resolve(strict=False)
    protected = (production, source)
    if any(
        output == root or output.is_relative_to(root) or root.is_relative_to(output)
        for root in protected
    ):
        raise R7Error("output overlaps protected root")
    if (
        production == source
        or production.is_relative_to(source)
        or source.is_relative_to(production)
    ):
        raise R7Error("production/source roots overlap")


def _source_identity(source: Path, commit: str) -> dict[str, str]:
    if not _COMMIT.fullmatch(commit):
        raise R7Error("source commit format is invalid")
    try:
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise R7Error("source repository cannot be verified") from exc
    if actual != commit or dirty:
        raise R7Error("source commit drift or dirty checkout")
    return {"source_commit": actual}


def _tree_state(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise R7Error("production clone contains a symlink")
        if path.is_file():
            identity = R0._file_identity(path)
            state = identity.get("file_state")
            if state is None:
                raise R7Error("production clone changed during capture")
            digest.update(
                json.dumps(
                    (
                        path.relative_to(root).as_posix(),
                        state["size_bytes"],
                        state["st_mtime_ns"],
                        identity["sha256"],
                    ),
                    separators=(",", ":"),
                ).encode()
            )
            count += 1
    return {"files": count, "state_sha256": digest.hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="immutable artifact directory"
    )
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--lkg-id", required=True)
    parser.add_argument("--locked-replay", type=Path, required=True)
    parser.add_argument("--stage-receipt", type=Path, action="append", required=True)
    parser.add_argument("--forced-failure-receipt", type=Path, required=True)
    for name in ("runtime", "archive", "process", "health", "api", "dom", "policy"):
        parser.add_argument(f"--{name}-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    clone: Path | None = None
    try:
        receipt_paths = {
            name: getattr(args, f"{name}_receipt")
            for name in (
                "runtime",
                "archive",
                "process",
                "health",
                "api",
                "dom",
                "policy",
            )
        }
        all_inputs = [
            args.locked_replay,
            *args.stage_receipt,
            args.forced_failure_receipt,
            *receipt_paths.values(),
        ]
        _assert_paths(args.production_root, args.source_root, args.output, all_inputs)
        production = args.production_root.expanduser().resolve(strict=True)
        source = args.source_root.expanduser().resolve(strict=True)
        output = args.output.expanduser().resolve(strict=False)
        source_before = _source_identity(source, args.source_commit)
        # R0 owns the Darwin/APFS copy-on-write primitive.  This is a read-only
        # production snapshot; no service/API/browser call is made here.
        clone, temporary = R0._clone(production, None)
        if not temporary:
            raise R7Error("production clone is not temporary")
        production_before = _tree_state(production)
        clone_before = _tree_state(clone)
        locked, _ = _read_json(args.locked_replay, "locked replay")
        stages = [_read_json(path, "stage receipt")[0] for path in args.stage_receipt]
        forced_failure, _ = _read_json(
            args.forced_failure_receipt, "forced-failure receipt"
        )
        receipts = {
            name: _read_json(path, f"{name} receipt")[0]
            for name, path in receipt_paths.items()
        }
        verdict = validate_bundle(
            locked_replay=locked,
            stages=stages,
            forced_failure=forced_failure,
            receipts=receipts,
            baseline_id=args.baseline_id,
            candidate_id=args.candidate_id,
            lkg_id=args.lkg_id,
            source_commit=args.source_commit,
            now=datetime.now(UTC),
        )
        if verdict["synthetic_fixture"]:
            raise R7Error("synthetic fixture cannot produce formal certification")
        production_after = _tree_state(production)
        if production_before != production_after:
            raise R7Error("production drift during validation")
        if _source_identity(source, args.source_commit) != source_before:
            raise R7Error("source drift during validation")
        shutil.rmtree(clone)
        clone_cleanup = not clone.exists()
        clone = None
        _, _, store, _, _ = R0._load(source)
        payload = {
            "captured_at": datetime.now(UTC).isoformat(),
            "certification": True,
            "synthetic_fixture": False,
            "source_before": source_before,
            "source_after": source_before,
            "production_before": production_before,
            "production_after": production_after,
            "clone_before": clone_before,
            "cleanup": {"clone_removed": clone_cleanup},
            "thresholds": {
                "min_days": MIN_DAYS,
                "min_paired": MIN_PAIRED,
                "max_abstain_delta": MAX_ABSTAIN_DELTA,
                "max_p95_ms": MAX_P95_MS,
                "max_timeout_rate": MAX_TIMEOUT_RATE,
            },
            "stage_matrix": [
                {"stage": stage, "rollout_percent": percent}
                for stage, percent in STAGES
            ],
            "evidence": verdict,
        }
        artifact_id, artifact_path, artifact = store.write_immutable(
            output, payload, schema=R7_SCHEMA
        )
        print(
            json.dumps(
                {
                    "schema": artifact["schema"],
                    "artifact_id": artifact_id,
                    "path": str(artifact_path),
                },
                sort_keys=True,
            )
        )
    except (R7Error, OSError, ValueError) as exc:
        print(f"r7 harness failed: {str(exc).split(':', 1)[0]}", file=sys.stderr)
        return 2
    finally:
        if clone is not None:
            shutil.rmtree(clone, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
