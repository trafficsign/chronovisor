#!/usr/bin/env python3.14
"""Fail-closed, receipt-only formal validator for a real Recall R7/P8 rollout.

It never calls a provider/service/browser or changes rollout state. A passing
CLI run means only sealed local evidence passed fixed gates; synthetic or
source-only evidence cannot certify a rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any

R7_SCHEMA = "chronovisor.recall-r7.v2"
LOCKED_REPLAY_SCHEMA = "chronovisor.recall-r7-locked-replay.v2"
POLICY_SCHEMA = "chronovisor.recall-distill-policy.v2"
BASELINE_SCHEMA = "chronovisor.recall-distill-baseline.v1"
STORE_SCHEMA = "chronovisor.recall-distillation.v1"
MAX_INPUT_BYTES = 12 * 1024 * 1024
MIN_DAYS = 7
MIN_PAIRED = 500
MIN_STRATUM_PAIRED = 500
MAX_SCORE_P95_MS = 180
MAX_LIVE_P50_MS = 400
MAX_LIVE_P95_MS = 900
MAX_DEADLINE_MS = 1_200
MAX_TIMEOUT_POINT = 0.01
MAX_TIMEOUT_UPPER = 0.03
MAX_ABSTAIN_DELTA = 0.02
STAGES = (("shadow", 0), ("5", 5), ("25", 25), ("100", 100))
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class R7Error(ValueError):
    """Evidence is incomplete, tampered, or below a non-negotiable gate."""


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


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise R7Error("receipt is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _id(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise R7Error(f"{label} is not a sha256 identity")
    return value


def _int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise R7Error(f"{label} is invalid")
    return value


def _map(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R7Error(f"{label} is not an object")
    return value


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise R7Error(f"{label} is not UTC")
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise R7Error(f"{label} is not UTC") from exc
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise R7Error(f"{label} is not UTC")
    return instant.astimezone(UTC)


def _seal(value: Mapping[str, Any], schema: str, keys: set[str], label: str) -> str:
    if (
        set(value) != keys
        or value.get("schema") != schema
        or value.get("namespace") != "recall-distillation"
    ):
        raise R7Error(f"{label} schema is not closed")
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    seal = _digest(unsigned)
    if value.get("seal_sha256") != seal:
        raise R7Error(f"{label} seal mismatch")
    return seal


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        before = path.lstat()
        if path.is_symlink() or not path.is_file() or before.st_size > MAX_INPUT_BYTES:
            raise R7Error(f"{label} path is unsafe")
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise R7Error(f"{label} cannot be read") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise R7Error(f"{label} changed during read")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise R7Error(f"{label} cannot be decoded") from exc
    if not isinstance(value, dict):
        raise R7Error(f"{label} is not an object")
    return value


def wilson_interval(successes: object, total: object) -> dict[str, float]:
    successes = _int(successes, "Wilson successes")
    total = _int(total, "Wilson denominator", 1)
    if successes > total:
        raise R7Error("Wilson successes exceed denominator")
    z = NormalDist().inv_cdf(0.975)
    point = successes / total
    denominator = 1 + z * z / total
    center = (point + z * z / (2 * total)) / denominator
    radius = (
        z
        * (point * (1 - point) / total + z * z / (4 * total * total)) ** 0.5
        / denominator
    )
    return {
        "point": point,
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
    }


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        raise R7Error("latency observations are missing")
    ordered = sorted(values)
    return ordered[max(0, (len(ordered) * fraction).__ceil__() - 1)]


def _identity_expected(
    *, baseline_id: str, candidate_id: str, lkg_id: str, source_commit: str
) -> dict[str, str]:
    return {
        "active_id": candidate_id,
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "lkg_id": lkg_id,
        "policy_id": candidate_id,
        "source_commit": source_commit,
    }


def _validate_identity_receipts(
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    baseline_id: str,
    candidate_id: str,
    lkg_id: str,
    source_commit: str,
) -> dict[str, str]:
    extras = {
        "runtime": {"runtime_commit"},
        "archive": {"archive_commit", "direct_url"},
        "process": {"process_commit", "pid", "started_at"},
        "health": {"health_commit", "status"},
        "api": {"api_commit", "status"},
        "dom": {"dom_commit", "status"},
        "policy": {"active_pointer", "candidate_pointer", "lkg_pointer"},
    }
    if set(receipts) != set(extras):
        raise R7Error("identity receipt set is incomplete")
    expected = _identity_expected(
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        lkg_id=lkg_id,
        source_commit=source_commit,
    )
    result: dict[str, str] = {}
    for name, extra in extras.items():
        receipt = receipts[name]
        _seal(
            receipt,
            f"chronovisor.recall-r7-{name}-receipt.v1",
            {"schema", "namespace", "seal_sha256", "kind", "identity", "actual"},
            f"{name} receipt",
        )
        if receipt.get("kind") != f"recall-r7-{name}-receipt":
            raise R7Error(f"{name} receipt kind mismatch")
        if receipt.get("identity") != expected:
            raise R7Error(f"{name} identity drift")
        actual = _map(receipt.get("actual"), f"{name} actual payload")
        if set(actual) != extra:
            raise R7Error(f"{name} actual payload schema is not closed")
        commit_key = next((key for key in actual if key.endswith("_commit")), None)
        if commit_key is not None and actual[commit_key] != source_commit:
            raise R7Error(f"{name} actual commit drift")
        if name == "archive" and (
            not isinstance(actual["direct_url"], str) or not actual["direct_url"]
        ):
            raise R7Error("archive direct_url is invalid")
        if name == "process":
            _int(actual["pid"], "process pid", 1)
            _utc(actual["started_at"], "process started_at")
        if name == "health" and actual["status"] != "ok":
            raise R7Error("health payload is unhealthy")
        if name == "api" and actual["status"] != 200:
            raise R7Error("API payload is unhealthy")
        if name == "dom" and actual["status"] != "ready":
            raise R7Error("DOM payload is unhealthy")
        if name == "policy" and actual != {
            "active_pointer": candidate_id,
            "candidate_pointer": candidate_id,
            "lkg_pointer": lkg_id,
        }:
            raise R7Error("policy payload drift")
        result[name] = _digest(receipt)
    return result


def _validate_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    baseline_id: str,
    candidate_id: str,
    lkg_id: str,
) -> dict[str, str]:
    expected = {"baseline": baseline_id, "candidate": candidate_id, "lkg": lkg_id}
    needed = {
        "baseline",
        "candidate",
        "lkg",
        "active_pointer",
        "candidate_pointer",
        "lkg_pointer",
    }
    if set(artifacts) != needed:
        raise R7Error("policy artifact set is incomplete")
    output: dict[str, str] = {}
    for name, policy_id in expected.items():
        artifact = artifacts[name]
        if (
            artifact.get("schema")
            != (BASELINE_SCHEMA if name == "baseline" else POLICY_SCHEMA)
            or artifact.get("namespace") != "recall-distillation"
            or artifact.get("artifact_id") != policy_id
        ):
            raise R7Error(f"{name} policy artifact identity drift")
        unsigned = {
            key: value for key, value in artifact.items() if key != "seal_sha256"
        }
        if artifact.get("seal_sha256") != _digest(unsigned):
            raise R7Error(f"{name} policy artifact seal mismatch")
        if name != "baseline":
            _id(artifact.get("feature_bytes_sha256"), f"{name} feature bytes")
        output[name] = _digest(artifact)
    for name, policy_id in (
        ("active_pointer", candidate_id),
        ("candidate_pointer", candidate_id),
        ("lkg_pointer", lkg_id),
    ):
        pointer = artifacts[name]
        _seal(
            pointer,
            STORE_SCHEMA,
            {"schema", "namespace", "seal_sha256", "kind", "policy_id"},
            name,
        )
        if (
            pointer.get("kind") != f"{name.removesuffix('_pointer')}-policy-pointer"
            or pointer.get("policy_id") != policy_id
        ):
            raise R7Error(f"{name} identity drift")
        output[name] = _digest(pointer)
    return output


def _validate_locked_replay(
    replay: Mapping[str, Any],
    *,
    candidate_feature: str,
    identity_refs: Mapping[str, str],
) -> str:
    seal = _seal(
        replay,
        LOCKED_REPLAY_SCHEMA,
        {
            "schema",
            "namespace",
            "seal_sha256",
            "kind",
            "synthetic_fixture",
            "provenance",
            "splits",
            "rows",
            "identity_refs",
        },
        "locked replay",
    )
    if (
        replay.get("kind") != "locked-replay"
        or replay.get("synthetic_fixture") is not False
        or replay.get("provenance") != "production-immutable-locked-replay"
    ):
        raise R7Error("synthetic/source-only replay cannot certify")
    if replay.get("identity_refs") != dict(identity_refs):
        raise R7Error("locked replay identity receipt drift")
    if replay.get("splits") != {
        "train": 70,
        "validation": 15,
        "test": 15,
        "embargo": True,
    }:
        raise R7Error("locked replay 70/15/15 embargo contract failed")
    rows = replay.get("rows")
    if not isinstance(rows, list) or len(rows) != 100:
        raise R7Error("locked replay denominator is invalid")
    keys = {
        "row_id",
        "split",
        "decision_sha256",
        "session_sha256",
        "query_sha256",
        "candidate_pool_sha256",
        "feature_bytes_sha256",
        "timestamp",
        "read_only",
        "route_probe",
        "ox_blind",
        "order_swap",
        "counterfactual",
        "negative_veto",
    }
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        item = _map(row, "locked replay row")
        if set(item) != keys:
            raise R7Error("locked replay row schema is not closed")
        row_id = _id(item["row_id"], "locked replay row id")
        if row_id in seen or item["split"] not in {"train", "validation", "test"}:
            raise R7Error("locked replay duplicate or split is invalid")
        seen.add(row_id)
        counts[str(item["split"])] += 1
        for key in (
            "decision_sha256",
            "session_sha256",
            "query_sha256",
            "candidate_pool_sha256",
        ):
            _id(item[key], f"locked replay {key}")
        if (
            item["feature_bytes_sha256"] != candidate_feature
            or item["read_only"] is not True
            or not all(
                item[key] is True
                for key in ("route_probe", "ox_blind", "order_swap", "counterfactual")
            )
            or item["negative_veto"] is not False
        ):
            raise R7Error("locked replay probes/features/negative-veto failed")
        _utc(item["timestamp"], "locked replay timestamp")
    if counts != Counter({"train": 70, "validation": 15, "test": 15}):
        raise R7Error("locked replay split boundaries failed")
    return seal


def _poll_history(
    value: object,
    *,
    stage: str,
    run_id: str,
    now: datetime,
    rows_sha: str,
    reused_polls: set[str],
) -> tuple[datetime, datetime, str]:
    if not isinstance(value, list) or len(value) < 2:
        raise R7Error(f"{stage} immutable poll history is incomplete")
    keys = {
        "schema",
        "namespace",
        "seal_sha256",
        "kind",
        "artifact_id",
        "stage",
        "run_id",
        "polled_at",
        "rows_sha256",
    }
    times: list[datetime] = []
    ids: set[str] = set()
    last_seal = ""
    for poll in value:
        item = _map(poll, f"{stage} poll")
        last_seal = _seal(item, "chronovisor.recall-r7-poll.v1", keys, f"{stage} poll")
        poll_id = _id(item["artifact_id"], f"{stage} poll id")
        instant = _utc(item["polled_at"], f"{stage} poll time")
        if (
            poll_id in ids
            or poll_id in reused_polls
            or item.get("kind") != "immutable-stage-poll"
            or item.get("stage") != stage
            or item.get("run_id") != run_id
            or item.get("rows_sha256") != rows_sha
            or instant > now
        ):
            raise R7Error(f"{stage} poll history drift")
        ids.add(poll_id)
        times.append(instant)
    if any(
        left >= right for left, right in zip(times[:-1], times[1:], strict=True)
    ) or times[-1] - times[0] < timedelta(days=MIN_DAYS):
        raise R7Error(f"{stage} poll history lacks real seven-day wall time")
    reused_polls.update(ids)
    return times[0], times[-1], last_seal


def _validate_rows(
    rows: object,
    *,
    stage: str,
    feature: str,
    observation_mode: str,
    first: datetime,
    last: datetime,
    reused: set[str],
    minimums: Mapping[str, Any],
    trusted_roster: Mapping[str, frozenset[str]] | None,
) -> dict[str, Any]:
    if not isinstance(rows, list) or len(rows) < MIN_PAIRED:
        raise R7Error(f"{stage} paired denominator is below {MIN_PAIRED}")
    keys = {
        "receipt_id",
        "decision_sha256",
        "session_sha256",
        "query_sha256",
        "candidate_pool_sha256",
        "feature_bytes_sha256",
        "observed_at",
        "host",
        "cohort",
        "baseline_quality",
        "candidate_quality",
        "baseline_covered",
        "candidate_covered",
        "baseline_abstained",
        "candidate_abstained",
        "candidate_score_ms",
        "live_latency_ms",
        "timed_out",
        "deadline_ms",
        "worker_id",
        "resource_ok",
        "integrity_ok",
        "negative_veto",
    }
    ids: set[str] = set()
    observation_keys: set[tuple[str, str, str, str]] = set()
    cross_stage_observations: set[str] = set()
    observed_times: set[datetime] = set()
    host_counts: Counter[str] = Counter()
    cohort_counts: Counter[str] = Counter()
    baseline_quality = candidate_quality = candidate_coverage = 0
    baseline_abstain = candidate_abstain = timeouts = 0
    score: list[int] = []
    live: list[int] = []
    for row in rows:
        item = _map(row, f"{stage} paired row")
        if set(item) != keys:
            raise R7Error(f"{stage} paired row schema is not closed")
        receipt_id = _id(item["receipt_id"], f"{stage} receipt")
        if receipt_id in ids or receipt_id in reused:
            raise R7Error(f"{stage} cross-stage receipt reuse or duplicate")
        ids.add(receipt_id)
        for key in (
            "decision_sha256",
            "session_sha256",
            "query_sha256",
            "candidate_pool_sha256",
        ):
            _id(item[key], f"{stage} {key}")
        observation_key = (
            item["decision_sha256"],
            item["session_sha256"],
            item["query_sha256"],
            item["candidate_pool_sha256"],
        )
        if observation_key in observation_keys:
            raise R7Error(f"{stage} duplicate paired observation")
        observation_keys.add(observation_key)
        observation_digest = _digest(observation_key)
        if observation_digest in reused:
            raise R7Error(f"{stage} cross-stage paired observation reuse")
        cross_stage_observations.add(observation_digest)
        if (
            item["feature_bytes_sha256"] != feature
            or not isinstance(item["host"], str)
            or not item["host"]
            or not isinstance(item["cohort"], str)
            or not item["cohort"]
            or not isinstance(item["worker_id"], str)
            or not item["worker_id"]
        ):
            raise R7Error(f"{stage} paired identity drift")
        observed = _utc(item["observed_at"], f"{stage} paired timestamp")
        if not first <= observed <= last:
            raise R7Error(f"{stage} paired timestamp is outside immutable polls")
        observed_times.add(observed)
        booleans = (
            "candidate_quality",
            "candidate_covered",
            "candidate_abstained",
            "timed_out",
            "resource_ok",
            "integrity_ok",
            "negative_veto",
        )
        if (
            any(not isinstance(item[key], bool) for key in booleans)
            or item["resource_ok"] is not True
            or item["integrity_ok"] is not True
            or item["negative_veto"] is not False
        ):
            raise R7Error(
                f"{stage} worker/resource/integrity/negative-veto gate failed"
            )
        baseline_values = (
            "baseline_quality",
            "baseline_covered",
            "baseline_abstained",
        )
        if observation_mode == "paired":
            if any(not isinstance(item[key], bool) for key in baseline_values):
                raise R7Error(f"{stage} paired baseline observation is invalid")
        elif any(item[key] is not None for key in baseline_values):
            raise R7Error(f"{stage} legacy candidate-only baseline must be absent")
        deadline = _int(item["deadline_ms"], f"{stage} deadline", 1)
        candidate_score = _int(item["candidate_score_ms"], f"{stage} candidate score")
        live_latency = _int(item["live_latency_ms"], f"{stage} live latency")
        if deadline > MAX_DEADLINE_MS or live_latency > deadline:
            raise R7Error(f"{stage} hard deadline gate failed")
        host_counts[item["host"]] += 1
        cohort_counts[item["cohort"]] += 1
        baseline_quality += int(item["baseline_quality"] is True)
        candidate_quality += int(item["candidate_quality"])
        candidate_coverage += int(item["candidate_covered"])
        baseline_abstain += int(item["baseline_abstained"] is True)
        candidate_abstain += int(item["candidate_abstained"])
        timeouts += int(item["timed_out"])
        score.append(candidate_score)
        live.append(live_latency)
    for name, counts in (("host", host_counts), ("cohort", cohort_counts)):
        minimum = _int(minimums.get(name), f"{stage} {name} stratum minimum", 1)
        expected = trusted_roster.get(name) if trusted_roster is not None else None
        if (
            minimum != MIN_STRATUM_PAIRED
            or not counts
            or any(count < minimum for count in counts.values())
            or (expected is not None and set(counts) != expected)
        ):
            raise R7Error(f"{stage} {name} stratum threshold failed")
    if len(observed_times) < 2:
        raise R7Error(f"{stage} paired rows use one synthetic timestamp")
    total = len(rows)
    baseline_ci = wilson_interval(baseline_quality, total)
    candidate_ci = wilson_interval(candidate_quality, total)
    coverage_ci = wilson_interval(candidate_coverage, total)
    timeout_ci = wilson_interval(timeouts, total)
    if (
        (observation_mode == "paired" and candidate_ci["lower"] < baseline_ci["upper"])
        or coverage_ci["lower"] < 0.95
        or (
            (candidate_abstain - baseline_abstain) / total > MAX_ABSTAIN_DELTA
            if observation_mode == "paired"
            else candidate_abstain / total > MAX_ABSTAIN_DELTA
        )
        or _percentile(score, 0.95) > MAX_SCORE_P95_MS
        or _percentile(live, 0.5) > MAX_LIVE_P50_MS
        or _percentile(live, 0.95) >= MAX_LIVE_P95_MS
        or timeouts / total > MAX_TIMEOUT_POINT
        or timeout_ci["upper"] > MAX_TIMEOUT_UPPER
    ):
        raise R7Error(
            f"{stage} quality/coverage/abstain/latency/timeout fixed gate failed"
        )
    reused.update(ids | cross_stage_observations)
    return {
        "rows_sha256": _digest(rows),
        "metrics": {
            "quality": {"baseline": baseline_ci, "candidate": candidate_ci},
            "coverage": coverage_ci,
            "timeout": timeout_ci,
            "candidate_score_p95_ms": _percentile(score, 0.95),
            "live_p50_ms": _percentile(live, 0.5),
            "live_p95_ms": _percentile(live, 0.95),
        },
    }


def _validate_stage(
    receipt: Mapping[str, Any],
    *,
    expected_stage: str,
    percent: int,
    feature: str,
    locked_sha: str,
    identity_refs: Mapping[str, str],
    now: datetime,
    previous: Mapping[str, Any] | None,
    reused: set[str],
    reused_polls: set[str],
    stage_run_ids: set[str],
    trusted_roster: Mapping[str, frozenset[str]] | None,
) -> dict[str, Any]:
    keys = {
        "schema",
        "namespace",
        "seal_sha256",
        "kind",
        "stage",
        "rollout_percent",
        "run_id",
        "stage_started_at",
        "poll_history",
        "rows",
        "stratum_minimums",
        "observation_mode",
        "identity_refs",
        "locked_replay_sha256",
        "feature_bytes_sha256",
        "previous_run_id",
        "legacy_incumbent_proof",
    }
    seal = _seal(
        receipt,
        "chronovisor.recall-r7-stage-receipt.v1",
        keys,
        f"{expected_stage} stage",
    )
    if (
        receipt.get("kind") != "stage-receipt"
        or receipt.get("stage") != expected_stage
        or receipt.get("rollout_percent") != percent
        or receipt.get("identity_refs") != dict(identity_refs)
        or receipt.get("locked_replay_sha256") != locked_sha
        or receipt.get("feature_bytes_sha256") != feature
    ):
        raise R7Error(f"{expected_stage} stage identity/matrix drift")
    run_id = _id(receipt.get("run_id"), f"{expected_stage} run id")
    if run_id in stage_run_ids:
        raise R7Error(f"{expected_stage} run id is globally reused")
    rows_sha = _digest(receipt.get("rows"))
    first, last, poll_seal = _poll_history(
        receipt.get("poll_history"),
        stage=expected_stage,
        run_id=run_id,
        now=now,
        rows_sha=rows_sha,
        reused_polls=reused_polls,
    )
    if _utc(receipt.get("stage_started_at"), f"{expected_stage} stage start") != first:
        raise R7Error(
            f"{expected_stage} stage start is self-attested, not poll-derived"
        )
    stage_run_ids.add(run_id)
    if previous is None:
        if receipt.get("previous_run_id") is not None:
            raise R7Error("shadow reset reference is invalid")
    elif (
        receipt.get("previous_run_id") != previous["run_id"]
        or run_id == previous["run_id"]
        or first < previous["last_poll"]
    ):
        raise R7Error(f"{expected_stage} run reset failed")
    mode = receipt.get("observation_mode")
    if mode == "candidate_only_legacy_incumbent":
        if percent != 100:
            raise R7Error(
                "candidate-only legacy incumbent is allowed only at 100 percent"
            )
        proof = _map(receipt.get("legacy_incumbent_proof"), "legacy incumbent proof")
        _seal(
            proof,
            "chronovisor.recall-r7-legacy-incumbent-proof.v1",
            {
                "schema",
                "namespace",
                "seal_sha256",
                "kind",
                "incumbent_unavailable",
                "run_id",
                "stage",
            },
            "legacy incumbent proof",
        )
        if (
            proof.get("kind") != "legacy-incumbent-proof"
            or proof.get("incumbent_unavailable") is not True
            or proof.get("run_id") != run_id
            or proof.get("stage") != expected_stage
        ):
            raise R7Error("legacy incumbent proof is invalid")
    elif mode != "paired" or receipt.get("legacy_incumbent_proof") is not None:
        raise R7Error("stage observation mode is invalid")
    minimums = _map(receipt.get("stratum_minimums"), f"{expected_stage} minimums")
    if set(minimums) != {"host", "cohort"}:
        raise R7Error(f"{expected_stage} stratum minimum schema is not closed")
    data = _validate_rows(
        receipt.get("rows"),
        stage=expected_stage,
        feature=feature,
        observation_mode=str(mode),
        first=first,
        last=last,
        reused=reused,
        minimums=minimums,
        trusted_roster=trusted_roster,
    )
    return {
        **data,
        "run_id": run_id,
        "last_poll": last,
        "stage_seal_sha256": seal,
        "last_poll_seal_sha256": poll_seal,
    }


def _trusted_roster(
    collector: Mapping[str, Any] | None,
) -> Mapping[str, frozenset[str]] | None:
    """Only an authoritative collector may define active rollout strata."""
    if collector is None or collector.get("certification") is not True:
        return None
    roster = _map(collector.get("active_host_cohort_roster"), "active roster")
    if set(roster) != {"hosts", "cohorts"}:
        raise R7Error("active roster schema is incomplete")
    result: dict[str, frozenset[str]] = {}
    for output, key in (("host", "hosts"), ("cohort", "cohorts")):
        values = roster[key]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise R7Error(f"active {output} roster is invalid")
        result[output] = frozenset(values)
    return result


def _validate_rollback(
    receipt: Mapping[str, Any],
    *,
    final_stage: Mapping[str, Any],
    lkg_id: str,
    identity_refs: Mapping[str, str],
    now: datetime,
) -> str:
    keys = {
        "schema",
        "namespace",
        "seal_sha256",
        "kind",
        "run_id",
        "stage",
        "failure_at",
        "stage_seal_sha256",
        "last_poll_seal_sha256",
        "identity_refs",
        "deterministic_failure",
        "rolled_back",
        "learning_halted",
        "rollout_percent",
        "rollback_state",
        "quarantine_id",
        "rollback_receipt_id",
        "rollback_receipt_sha256",
    }
    seal = _seal(
        receipt,
        "chronovisor.recall-r7-forced-failure-receipt.v1",
        keys,
        "forced-failure receipt",
    )
    state = _map(receipt.get("rollback_state"), "rollback state")
    expected_state = {
        "active_policy_id": lkg_id,
        "candidate_policy_id": None,
        "lkg_policy_id": lkg_id,
    }
    failure_at = _utc(receipt.get("failure_at"), "rollback timestamp")
    if (
        receipt.get("kind") != "forced-failure-receipt"
        or receipt.get("run_id") != final_stage["run_id"]
        or receipt.get("stage") != "100"
        or not final_stage["last_poll"] <= failure_at <= now
        or receipt.get("stage_seal_sha256") != final_stage["stage_seal_sha256"]
        or receipt.get("last_poll_seal_sha256") != final_stage["last_poll_seal_sha256"]
        or receipt.get("identity_refs") != dict(identity_refs)
        or receipt.get("deterministic_failure") is not True
        or receipt.get("rolled_back") is not True
        or receipt.get("learning_halted") is not True
        or receipt.get("rollout_percent") != 0
        or state != expected_state
    ):
        raise R7Error("forced-failure rollback binding/state is incomplete")
    _id(receipt.get("quarantine_id"), "rollback quarantine id")
    _id(receipt.get("rollback_receipt_id"), "authoritative rollback receipt")
    _id(receipt.get("rollback_receipt_sha256"), "authoritative rollback seal")
    return seal


def validate_bundle(
    *,
    locked_replay: Mapping[str, Any],
    stages: Sequence[Mapping[str, Any]],
    forced_failure: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Any]],
    baseline_id: str,
    candidate_id: str,
    lkg_id: str,
    source_commit: str,
    now: datetime,
    collector_evidence_root: Path | None = None,
    collector_runtime_root: Path | None = None,
    source_tree_sha256: str | None = None,
    source_bytes_sha256: str | None = None,
) -> dict[str, Any]:
    """Pure validator; CLI supplies system UTC, tests may supply a fixed clock."""
    baseline_id = _id(baseline_id, "baseline id")
    candidate_id = _id(candidate_id, "candidate id")
    lkg_id = _id(lkg_id, "LKG id")
    if (
        len({baseline_id, candidate_id, lkg_id}) != 3
        or not isinstance(source_commit, str)
        or _COMMIT.fullmatch(source_commit) is None
        or now.tzinfo is None
        or now.utcoffset() != timedelta(0)
    ):
        raise R7Error("policy/source/clock identity is invalid")
    if source_tree_sha256 is not None:
        _id(source_tree_sha256, "source tree identity")
    if source_bytes_sha256 is not None:
        _id(source_bytes_sha256, "source bytes identity")
    artifact_refs = _validate_artifacts(
        artifacts,
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        lkg_id=lkg_id,
    )
    feature = _id(
        _map(artifacts["candidate"], "candidate artifact").get("feature_bytes_sha256"),
        "candidate feature bytes",
    )
    identity_refs = _validate_identity_receipts(
        receipts,
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        lkg_id=lkg_id,
        source_commit=source_commit,
    )
    identity_refs = {**identity_refs, **artifact_refs}
    locked_sha = _validate_locked_replay(
        locked_replay, candidate_feature=feature, identity_refs=identity_refs
    )
    if len(stages) != len(STAGES):
        raise R7Error("stage receipt count is incomplete")
    collector: Mapping[str, Any] | None = None
    if collector_evidence_root is not None:
        from chronovisor.recall.recall_r7_evidence import validate_collector

        collector = validate_collector(
            collector_evidence_root, root=collector_runtime_root
        )
        if collector.get("certification") is True:
            expected_source = {
                "source_commit": source_commit,
                "source_clean": "true",
                "source_tree_sha256": source_tree_sha256,
                "source_bytes_sha256": source_bytes_sha256,
            }
            if (
                source_tree_sha256 is None
                or source_bytes_sha256 is None
                or collector.get("identity")
                != {
                    "baseline_id": baseline_id,
                    "candidate_id": candidate_id,
                    "lkg_id": lkg_id,
                    "candidate_feature_contract_sha256": feature,
                }
                or collector.get("source") != expected_source
            ):
                raise R7Error("collector identity/source binding mismatch")
    trusted_roster = _trusted_roster(collector)
    reused: set[str] = set()
    reused_polls: set[str] = set()
    stage_run_ids: set[str] = set()
    checked: list[dict[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for receipt, (stage, percent) in zip(stages, STAGES, strict=True):
        result = _validate_stage(
            receipt,
            expected_stage=stage,
            percent=percent,
            feature=feature,
            locked_sha=locked_sha,
            identity_refs=identity_refs,
            now=now,
            previous=previous,
            reused=reused,
            reused_polls=reused_polls,
            stage_run_ids=stage_run_ids,
            trusted_roster=trusted_roster,
        )
        checked.append(result)
        previous = result
    authoritative_reference = (
        forced_failure.get("schema") == "chronovisor.recall-r7-rollback.v1"
        and forced_failure.get("kind") == "r7-authoritative-forced-rollback"
        and isinstance(forced_failure.get("artifact_id"), str)
        and _HEX.fullmatch(str(forced_failure.get("artifact_id"))) is not None
        and isinstance(forced_failure.get("seal_sha256"), str)
        and _HEX.fullmatch(str(forced_failure.get("seal_sha256"))) is not None
    )
    rollback_sha = (
        str(forced_failure["seal_sha256"])
        if authoritative_reference
        else _validate_rollback(
            forced_failure,
            final_stage=checked[-1],
            lkg_id=lkg_id,
            identity_refs=identity_refs,
            now=now,
        )
    )
    authoritative_rollback: Mapping[str, str] | None = None
    rollback_id = (
        forced_failure.get("artifact_id")
        if authoritative_reference
        else forced_failure.get("rollback_receipt_id")
    )
    rollback_receipt_sha = (
        forced_failure.get("seal_sha256")
        if authoritative_reference
        else forced_failure.get("rollback_receipt_sha256")
    )
    if (
        collector is not None
        and collector.get("certification") is True
        and collector_evidence_root is not None
        and collector_runtime_root is not None
        and isinstance(rollback_id, str)
        and _HEX.fullmatch(rollback_id) is not None
        and isinstance(rollback_receipt_sha, str)
        and _HEX.fullmatch(rollback_receipt_sha) is not None
    ):
        try:
            from chronovisor.recall.recall_r7_evidence import validate_rollback

            authoritative_rollback = validate_rollback(
                collector_runtime_root,
                collector_evidence_root / "rollbacks" / f"{rollback_id}.json",
            )
        except (OSError, ValueError):
            authoritative_rollback = None
    certified = (
        authoritative_rollback is not None
        and authoritative_rollback.get("artifact_id") == rollback_id
        and authoritative_rollback.get("receipt_sha256") == rollback_receipt_sha
        and authoritative_rollback.get("run_id") == checked[-1]["run_id"]
        and authoritative_rollback.get("stage") == "100"
    )
    hold_reason = (
        "complete_authoritative_r7_bundle"
        if certified
        else "authoritative_rollback_receipt_unavailable"
        if collector is not None and collector.get("certification") is True
        else str(collector.get("certification_reason"))
        if collector is not None
        else "trusted_active_host_cohort_inventory_unavailable"
    )
    if trusted_roster is None and collector is not None:
        hold_reason = "trusted_active_host_cohort_inventory_unavailable"
    for stage_data in checked:
        if not certified:
            stage_data["certified"] = False
        stage_data["reason"] = hold_reason
    return {
        "certification": certified,
        "certification_reason": (hold_reason),
        "synthetic_fixture": False,
        "locked_replay_sha256": locked_sha,
        "identity_refs": identity_refs,
        "stages": checked,
        "forced_failure_sha256": rollback_sha,
        "collector": collector,
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
    resolved_inputs = tuple(path.expanduser().resolve(strict=False) for path in inputs)
    if (
        production == source
        or production.is_relative_to(source)
        or source.is_relative_to(production)
    ):
        raise R7Error("production/source roots overlap")
    if any(
        output == root or output.is_relative_to(root) or root.is_relative_to(output)
        for root in (production, source)
    ):
        raise R7Error("output overlaps protected root")
    if any(
        input_path == output
        or input_path.is_relative_to(output)
        or output.is_relative_to(input_path)
        for input_path in resolved_inputs
    ):
        raise R7Error("output overlaps an evidence input or its parent")


def _source_identity(source: Path, commit: str) -> dict[str, str]:
    if _COMMIT.fullmatch(commit) is None:
        raise R7Error("source commit format is invalid")
    try:
        from chronovisor.recall.recall_r7_evidence import source_identity

        identity = source_identity(source)
    except ValueError as exc:
        raise R7Error(str(exc)) from exc
    except (ImportError, OSError) as exc:
        raise R7Error("source repository cannot be verified") from exc
    if identity["source_commit"] != commit:
        raise R7Error("source commit drift or dirty checkout")
    return identity


def _tree_state(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise R7Error("clone tree contains a symlink")
        if path.is_file():
            identity = R0._file_identity(path)
            digest.update(
                _canonical(
                    (
                        path.relative_to(root).as_posix(),
                        identity["file_state"],
                        identity["sha256"],
                    )
                )
            )
            files += 1
    return {"files": files, "state_sha256": digest.hexdigest()}


def _clone_state(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative in (Path("raw"), Path("runtime") / "recall-distillation"):
        path = root / relative
        if not path.is_dir() or path.is_symlink():
            raise R7Error("R0 clone subtree is absent or unsafe")
        result[relative.as_posix()] = _tree_state(path)
    recall_log = root / "recall" / "recall-log.jsonl"
    if recall_log.exists():
        if recall_log.is_symlink() or not recall_log.is_file():
            raise R7Error("R0 cloned recall log is unsafe")
        result["recall/recall-log.jsonl"] = R0._file_identity(recall_log)
    else:
        result["recall/recall-log.jsonl"] = None
    return result


def main(argv: list[str] | None = None) -> int:
    # The collector owns real wall-clock evidence.  Keep the original bundle
    # validator receipt-only; this merely exposes its safe record/validate CLI.
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in {"record-poll", "validate"}:
        from chronovisor.recall.recall_r7_evidence import main as evidence_main

        return evidence_main(arguments)
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
    parser.add_argument("--collector-evidence-root", type=Path)
    parser.add_argument("--locked-replay", type=Path, required=True)
    parser.add_argument("--stage-receipt", type=Path, action="append", required=True)
    parser.add_argument("--forced-failure-receipt", type=Path, required=True)
    for name in ("runtime", "archive", "process", "health", "api", "dom", "policy"):
        parser.add_argument(f"--{name}-receipt", type=Path, required=True)
    for name in ("baseline", "candidate", "lkg"):
        parser.add_argument(f"--{name}-artifact", type=Path, required=True)
    for name in ("active", "candidate", "lkg"):
        parser.add_argument(f"--{name}-pointer", type=Path, required=True)
    args = parser.parse_args(arguments)
    clone: Path | None = None
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
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
        artifact_paths = {
            name: getattr(args, f"{name}_artifact")
            for name in ("baseline", "candidate", "lkg")
        }
        pointer_paths = {
            f"{name}_pointer": getattr(args, f"{name}_pointer")
            for name in ("active", "candidate", "lkg")
        }
        inputs = [
            args.locked_replay,
            *args.stage_receipt,
            args.forced_failure_receipt,
            *receipt_paths.values(),
            *artifact_paths.values(),
            *pointer_paths.values(),
        ]
        if args.collector_evidence_root is not None:
            inputs.append(args.collector_evidence_root)
        _assert_paths(args.production_root, args.source_root, args.output, inputs)
        production = args.production_root.resolve(strict=True)
        source = args.source_root.resolve(strict=True)
        output = args.output.resolve(strict=False)
        source_before = _source_identity(source, args.source_commit)
        production_before = _clone_state(production)
        clone, temporary = R0._clone(production, None)
        if not temporary or _clone_state(clone) != production_before:
            raise R7Error("APFS clone coherence failed")
        locked = _read_json(args.locked_replay, "locked replay")
        stages = [_read_json(path, "stage receipt") for path in args.stage_receipt]
        forced = _read_json(args.forced_failure_receipt, "forced-failure receipt")
        receipts = {
            name: _read_json(path, f"{name} receipt")
            for name, path in receipt_paths.items()
        }
        artifacts = {
            name: _read_json(path, f"{name} artifact")
            for name, path in artifact_paths.items()
        } | {name: _read_json(path, name) for name, path in pointer_paths.items()}
        verdict = validate_bundle(
            locked_replay=locked,
            stages=stages,
            forced_failure=forced,
            receipts=receipts,
            artifacts=artifacts,
            baseline_id=args.baseline_id,
            candidate_id=args.candidate_id,
            lkg_id=args.lkg_id,
            source_commit=args.source_commit,
            now=datetime.now(UTC),
            collector_evidence_root=args.collector_evidence_root,
            collector_runtime_root=production,
            source_tree_sha256=source_before["source_tree_sha256"],
            source_bytes_sha256=source_before["source_bytes_sha256"],
        )
        production_after = _clone_state(production)
        if (
            production_before != production_after
            or _source_identity(source, args.source_commit) != source_before
        ):
            raise R7Error("production/source drift during validation")
        shutil.rmtree(clone)
        if clone.exists():
            raise R7Error("clone cleanup failed")
        clone = None
        _, _, store, _, _ = R0._load(source)
        payload = {
            "captured_at": datetime.now(UTC).isoformat(),
            "certification": verdict.get("certification", False),
            "certification_reason": verdict.get(
                "certification_reason",
                "independent_live_input_attestation_unavailable",
            ),
            "synthetic_fixture": False,
            "source_before": source_before,
            "source_after": source_before,
            "production_before": production_before,
            "production_after": production_after,
            "cleanup": {"clone_removed": True},
            "thresholds": {
                "min_days": MIN_DAYS,
                "min_paired": MIN_PAIRED,
                "candidate_score_p95_ms": MAX_SCORE_P95_MS,
                "live_p50_ms": MAX_LIVE_P50_MS,
                "live_p95_exclusive_ms": MAX_LIVE_P95_MS,
                "hard_deadline_ms": MAX_DEADLINE_MS,
                "timeout_point": MAX_TIMEOUT_POINT,
                "timeout_wilson_upper": MAX_TIMEOUT_UPPER,
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
        sys.dont_write_bytecode = previous_bytecode
        if clone is not None:
            shutil.rmtree(clone, ignore_errors=True)
    return 0 if verdict.get("certification") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
