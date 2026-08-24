"""Append-only, read-only evidence collection for a real Recall R7 rollout.

This module deliberately does not advance a rollout or call a teacher.  It is
the narrow boundary between live, independently captured observations and the
receipt validator: a poll is accepted only after the current sealed policy
state and local process/dashboard facts have been re-read by this process.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any
from urllib.parse import urlsplit

from chronovisor.core.canonical_json import (
    canonical_json_sha256_strict,
)
from chronovisor.recall import recall_distillation as distillation
from chronovisor.recall import recall_distillation_rollout as rollout
from chronovisor.recall import recall_distillation_store as store

EVIDENCE_SCHEMA = "chronovisor.recall-r7-evidence.v1"
POLL_SCHEMA = "chronovisor.recall-r7-poll.v2"
LEDGER_SCHEMA = "chronovisor.recall-r7-poll-ledger.v1"
STAGES = ("shadow", "5", "25", "100")
MIN_DAYS = 7
MIN_PAIRED = 500
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
MAX_BYTES = 12 * 1024 * 1024


class EvidenceError(ValueError):
    """A live evidence input cannot be safely certified."""


def _digest(value: object) -> str:
    return canonical_json_sha256_strict(value)


def _id(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise EvidenceError(f"{label} is not sha256")
    return value


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise EvidenceError(f"{label} is not UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError(f"{label} is not UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise EvidenceError(f"{label} is not UTC")
    return parsed.astimezone(UTC)


def _sealed(payload: Mapping[str, Any], schema: str, label: str) -> dict[str, Any]:
    if (
        payload.get("schema") != schema
        or payload.get("namespace") != "recall-distillation"
    ):
        raise EvidenceError(f"{label} schema mismatch")
    unsigned = {key: value for key, value in payload.items() if key != "seal_sha256"}
    if payload.get("seal_sha256") != _digest(unsigned):
        raise EvidenceError(f"{label} seal mismatch")
    return dict(payload)


def _safe_json(path: Path, label: str) -> dict[str, Any]:
    try:
        before = path.lstat()
        if path.is_symlink() or not path.is_file() or before.st_size > MAX_BYTES:
            raise EvidenceError(f"{label} path unsafe")
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"{label} unreadable") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise EvidenceError(f"{label} changed during read")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} invalid JSON") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} is not object")
    return value


def _has_symlink_component(path: Path) -> bool:
    candidate = path.expanduser()
    return any(part.is_symlink() for part in (candidate, *candidate.parents))


def _readonly_chain_snapshot(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one checkpoint-bound chain snapshot without creating its lock file."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock_path, os.O_RDONLY)
    except OSError as exc:
        raise EvidenceError("runtime chain lock is unavailable") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        head = store._read_chain_checkpoint(path)
        if head is None:
            raise EvidenceError("runtime chain checkpoint is absent")
        rows = store._read_chain_locked(path, head)
        previous = ""
        for index, row in enumerate(rows):
            unsigned = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            if row.get("previous_sha256") != previous or row.get(
                "record_sha256"
            ) != _digest(unsigned):
                raise EvidenceError(f"runtime chain mismatch at row {index}")
            previous = str(row["record_sha256"])
        if len(rows) != head["records"] or previous != head["head_sha256"]:
            raise EvidenceError("runtime chain checkpoint mismatch")
        return rows, dict(head)
    except store.DistillationStoreError as exc:
        raise EvidenceError("runtime chain snapshot is invalid") from exc
    finally:
        os.close(descriptor)


def _protected_file_state(root: Path) -> str:
    directory = store.distillation_dir(root)
    paths = [
        directory / store.STATE_FILE,
        *(directory / filename for filename in store.POINTER_FILES.values()),
        directory / "shadow-observation-receipts.jsonl",
        store._chain_checkpoint_path(directory / "shadow-observation-receipts.jsonl"),
    ]
    state: list[tuple[str, int, int, int, int, int] | tuple[str, None]] = []
    for path in paths:
        try:
            stat = path.lstat()
        except FileNotFoundError:
            state.append((path.name, None))
            continue
        if path.is_symlink():
            raise EvidenceError("protected runtime path is symlinked")
        state.append(
            (
                path.name,
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        )
    return _digest(state)


def _ignored_source_code(relative: str) -> bool:
    """Whether an ignored path could alter the protected R7 import surface."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return True
    protected = (len(path.parts) >= 2 and path.parts[:2] == ("src", "chronovisor")) or (
        len(path.parts) >= 2
        and path.parts[0] == "scripts"
        and path.parts[1].startswith("recall_r7")
    )
    if not protected:
        return False
    # Bytecode/type/test caches cannot change the R7 import surface. Every
    # other ignored path under a protected namespace is treated as source.
    return not any(
        component in {"__pycache__", ".pytest_cache", ".mypy_cache"}
        for component in path.parts
    )


def source_identity(source: Path) -> dict[str, str]:
    """Seal clean tracked source, excluding harmless ignored environment caches."""
    try:
        source = source.resolve(strict=True)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        ignored = subprocess.run(
            ["git", "ls-files", "-o", "-i", "--exclude-standard", "-z"],
            cwd=source,
            check=True,
            capture_output=True,
        ).stdout
        indexed = subprocess.run(
            ["git", "ls-files", "-s", "-z"],
            cwd=source,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceError("source identity unavailable") from exc
    if _COMMIT.fullmatch(commit) is None or dirty:
        raise EvidenceError("source commit drift or dirty checkout")
    ignored_paths = tuple(os.fsdecode(path) for path in ignored.split(b"\0") if path)
    if any(_ignored_source_code(path) for path in ignored_paths):
        raise EvidenceError("ignored protected source drift")
    tree: list[tuple[str, str, str]] = []
    for record in indexed.split(b"\0"):
        if not record:
            continue
        header, separator, path = record.partition(b"\t")
        fields = header.split()
        if separator != b"\t" or len(fields) != 3:
            raise EvidenceError("source index is malformed")
        tree.append((os.fsdecode(fields[0]), os.fsdecode(fields[1]), os.fsdecode(path)))
    if not tree:
        raise EvidenceError("source index is empty")
    return {
        "source_commit": commit,
        "source_clean": "true",
        "source_tree_sha256": _digest(tree),
    }


_source_identity = source_identity


def _process_identity(executable: Path, pid: int) -> dict[str, Any]:
    if pid < 1 or _has_symlink_component(executable) or not executable.is_file():
        raise EvidenceError("executable/PID is unsafe")
    # argv and lsof text mappings prove only that a file was mentioned/opened,
    # not that this PID executed it.  The stdlib has no macOS exec-image API.
    raise EvidenceError("process executable identity is not independently provable")


def _direct_url(path: Path) -> dict[str, str]:
    value = _safe_json(path, "runtime direct_url")
    commit = (
        value.get("vcs_info", {}).get("commit_id")
        if isinstance(value.get("vcs_info"), Mapping)
        else None
    )
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise EvidenceError("runtime archive commit unavailable")
    return {"archive_commit": commit, "direct_url_sha256": _digest(value)}


def _fetch(url: str, label: str) -> dict[str, Any]:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise EvidenceError(f"{label} must be loopback")
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read(MAX_BYTES + 1)
            status = response.status
    except (OSError, urllib.error.URLError) as exc:
        raise EvidenceError(f"{label} unavailable") from exc
    if len(body) > MAX_BYTES:
        raise EvidenceError(f"{label} too large")
    try:
        payload = json.loads(body)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} is not JSON") from exc
    if not isinstance(payload, Mapping):
        raise EvidenceError(f"{label} is not object")
    return {
        "url": url,
        "status": status,
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "payload": payload,
    }


def _stage_state(root: Path, stage: str) -> dict[str, str | None]:
    if stage not in STAGES:
        raise EvidenceError("unknown rollout stage")
    try:
        state = store.read_sealed(
            store.distillation_dir(root) / store.STATE_FILE,
            schema=store.DISTILLATION_SCHEMA,
        )
        baseline_id = _id(state.get("baseline_artifact_id"), "state baseline")
        baseline = store.read_sealed(
            store.distillation_dir(root) / "baselines" / f"{baseline_id}.json",
            schema="chronovisor.recall-distill-baseline.v1",
        )
        candidate = store.read_pointer(root, "candidate")
        lkg = store.read_pointer(root, "lkg")
        active = store.read_pointer(root, "active")
        candidate_id, lkg_id, active_id = (
            _id(candidate.get("policy_id"), "candidate policy"),
            _id(lkg.get("policy_id"), "LKG policy"),
            _id(active.get("policy_id"), "active policy"),
        )
        if baseline.get("artifact_id") != baseline_id:
            raise EvidenceError("baseline state/artifact mismatch")
        policies: dict[str, Mapping[str, Any]] = {}
        for policy_id in (candidate_id, lkg_id, active_id):
            policies[policy_id] = store.read_sealed(
                store.distillation_dir(root) / "policies" / f"{policy_id}.json",
                schema=rollout.POLICY_SCHEMA,
            )
    except (OSError, store.DistillationStoreError) as exc:
        raise EvidenceError("sealed rollout state unavailable") from exc
    percent = {"shadow": 0, "5": 5, "25": 25, "100": 100}[stage]
    if stage == "shadow":
        expected_active = lkg_id
    elif stage in {"5", "25"}:
        expected_active = lkg_id
        if state.get("status") != "canary" or state.get("rollout_percent") != percent:
            raise EvidenceError("canary state does not match stage")
    else:
        if state.get("status") != "canary" or state.get("rollout_percent") != 100:
            raise EvidenceError("100 stage is not gate-authorized")
        expected_active = candidate_id
    if active_id != expected_active:
        raise EvidenceError("active policy violates stage semantics")
    candidate_policy = policies[candidate_id]
    if (
        candidate_policy.get("feature_keys") != list(distillation.FAST_FEATURE_KEYS)
        or candidate_policy.get("feature_revision")
        != distillation.TEXT_FEATURE_REVISION
    ):
        raise EvidenceError("candidate policy feature contract mismatch")
    feature = _digest(
        {
            "feature_keys": candidate_policy["feature_keys"],
            "feature_revision": candidate_policy["feature_revision"],
            "weights": candidate_policy.get("weights"),
            "bias": candidate_policy.get("bias"),
            "threshold": candidate_policy.get("threshold"),
            "abstain_margin": candidate_policy.get("abstain_margin"),
        }
    )
    return {
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "lkg_id": lkg_id,
        "active_id": active_id,
        "candidate_feature_contract_sha256": feature,
    }


def _runtime_observations(
    root: Path, stage: str, identities: Mapping[str, str | None]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-derive pairs only from the runtime's checkpoint-backed receipt chain."""
    ledger_path = store.distillation_dir(root) / "shadow-observation-receipts.jsonl"
    receipts, head = _readonly_chain_snapshot(ledger_path)
    state = store.read_sealed(
        store.distillation_dir(root) / store.STATE_FILE,
        schema=store.DISTILLATION_SCHEMA,
    )
    expected_runtime_stage = "shadow" if stage == "shadow" else "canary"
    expected_percent = {"shadow": 0, "5": 5, "25": 25, "100": 100}[stage]
    if (
        expected_runtime_stage == "canary"
        and state.get("rollout_percent") != expected_percent
    ):
        raise EvidenceError("runtime canary percent does not match collector stage")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for receipt in receipts:
        keys = (
            "decision_id",
            "host",
            "session_id_sha256",
            "query_semantic_sha256",
            "policy_id",
            "incumbent_policy_id",
            "served_policy_id",
            "stage",
            "stage_started_at",
            "qualified_run_id",
            "selected_candidate_ids",
            "incumbent_selected_candidate_ids",
            "paired_eligible",
            "candidate_pool_sha256",
            "candidate_feature_snapshot_sha256",
            "runtime_observation_sha256",
            "observed_at",
        )
        binding = {key: receipt.get(key) for key in keys}
        artifact_id = receipt.get("shadow_observation_artifact_id")
        if (
            receipt.get("kind") != "shadow-policy-observation"
            or receipt.get("binding_sha256") != _digest(binding)
            or not isinstance(artifact_id, str)
            or _HEX.fullmatch(artifact_id) is None
        ):
            continue
        try:
            artifact = store.read_sealed(
                store.distillation_dir(root)
                / "shadow-observations"
                / f"{artifact_id}.json",
                schema=distillation.SHADOW_OBSERVATION_SCHEMA,
            )
        except store.DistillationStoreError:
            continue
        if artifact.get("artifact_id") != artifact_id or any(
            artifact.get(key) != value for key, value in binding.items()
        ):
            continue
        if (
            artifact.get("policy_id") != identities["candidate_id"]
            or artifact.get("incumbent_policy_id") != identities["lkg_id"]
            or artifact.get("stage") != expected_runtime_stage
            or artifact.get("stage_started_at") != state.get("stage_started_at")
            or artifact.get("qualified_run_id") != state.get("stage_run_id")
            or artifact.get("paired_eligible") is not True
        ):
            continue
        observation = artifact.get("runtime_observation")
        if not isinstance(observation, Mapping):
            continue
        try:
            latency = float(observation["latency_ms"])
            timed_out = observation["timed_out"]
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= latency <= 60_000 or not isinstance(timed_out, bool):
            continue
        observation_id = _id(artifact_id, "shadow observation")
        if observation_id in seen:
            raise EvidenceError("duplicate runtime shadow observation")
        seen.add(observation_id)
        rows.append(
            {
                "observation_id": observation_id,
                "decision_sha256": _digest(artifact["decision_id"]),
                "session_sha256": artifact["session_id_sha256"],
                "query_sha256": artifact["query_semantic_sha256"],
                "candidate_pool_sha256": artifact["candidate_pool_sha256"],
                "feature_contract_sha256": identities[
                    "candidate_feature_contract_sha256"
                ],
                "host": artifact["host"],
                "cohort": artifact["host"],
                "worker_id": "recall-runtime",
                "candidate_covered": bool(artifact.get("selected_candidate_ids")),
                "baseline_covered": bool(
                    artifact.get("incumbent_selected_candidate_ids")
                ),
                "candidate_score_ms": int(latency),
                "live_latency_ms": int(latency),
                "timed_out": timed_out,
                "observed_at": artifact["observed_at"],
                "run_id": artifact["qualified_run_id"],
            }
        )
    return rows, {
        "records": head["records"],
        "head_sha256": head["head_sha256"],
        "stage_run_id": state.get("stage_run_id"),
    }


def _wilson_lower(successes: int, total: int) -> float:
    if total < 1 or not 0 <= successes <= total:
        raise EvidenceError("invalid Wilson denominator")
    z = NormalDist().inv_cdf(0.975)
    point = successes / total
    denominator = 1 + z * z / total
    return max(
        0.0,
        (
            point
            + z * z / (2 * total)
            - z * (point * (1 - point) / total + z * z / (4 * total * total)) ** 0.5
        )
        / denominator,
    )


def _p95(values: Sequence[int]) -> int:
    if not values:
        raise EvidenceError("latency observations absent")
    ordered = sorted(values)
    return ordered[(len(ordered) * 0.95).__ceil__() - 1]


def _ledger_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise EvidenceError("ledger path unsafe")
    rows: list[dict[str, Any]] = []
    previous = ""
    for index, raw in enumerate(path.read_text().splitlines()):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvidenceError("ledger is corrupt") from exc
        if (
            not isinstance(row, dict)
            or row.get("schema") != LEDGER_SCHEMA
            or row.get("previous_sha256") != previous
        ):
            raise EvidenceError("ledger chain mismatch")
        unsigned = {key: value for key, value in row.items() if key != "entry_sha256"}
        if row.get("entry_sha256") != _digest(unsigned):
            raise EvidenceError("ledger hash mismatch")
        _id(row.get("poll_id"), "ledger poll")
        _utc(row.get("observed_at"), "ledger observed_at")
        if isinstance(row.get("monotonic_ns"), bool) or not isinstance(
            row.get("monotonic_ns"), int
        ):
            raise EvidenceError("ledger monotonic clock invalid")
        if index and (
            row["observed_at"] <= rows[-1]["observed_at"]
            or row["monotonic_ns"] <= rows[-1]["monotonic_ns"]
        ):
            raise EvidenceError("system clock moved backwards")
        previous = row["entry_sha256"]
        rows.append(row)
    return rows


def _ledger_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"count": 0, "head_sha256": ""}
    try:
        value = store.read_sealed(path, schema=store.DISTILLATION_SCHEMA)
    except (OSError, store.DistillationStoreError) as exc:
        raise EvidenceError("ledger state is invalid") from exc
    if (
        value.get("kind") != "r7-poll-ledger-state"
        or isinstance(value.get("count"), bool)
        or not isinstance(value.get("count"), int)
        or value["count"] < 0
        or not isinstance(value.get("head_sha256"), str)
    ):
        raise EvidenceError("ledger state schema mismatch")
    if value["head_sha256"] and _HEX.fullmatch(value["head_sha256"]) is None:
        raise EvidenceError("ledger state head is invalid")
    return {"count": value["count"], "head_sha256": value["head_sha256"]}


def _check_ledger_state(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    state = _ledger_state(root / "poll-ledger-state.json")
    head = rows[-1]["entry_sha256"] if rows else ""
    if state != {"count": len(rows), "head_sha256": head}:
        raise EvidenceError("ledger head/count mismatch")


def _append_ledger(
    path: Path, poll_id: str, stage: str, observed_at: datetime, monotonic_ns: int
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = _ledger_rows(path)
        _check_ledger_state(path.parent, rows)
        if poll_id in {row["poll_id"] for row in rows}:
            raise EvidenceError("duplicate poll id")
        prior = rows[-1]["entry_sha256"] if rows else ""
        entry = {
            "schema": LEDGER_SCHEMA,
            "namespace": "recall-distillation",
            "poll_id": poll_id,
            "stage": stage,
            "observed_at": observed_at.isoformat(),
            "monotonic_ns": monotonic_ns,
            "previous_sha256": prior,
        }
        entry["entry_sha256"] = _digest(entry)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        store.write_sealed_state(
            path.parent / "poll-ledger-state.json",
            {
                "kind": "r7-poll-ledger-state",
                "count": len(rows) + 1,
                "head_sha256": entry["entry_sha256"],
            },
        )
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return entry["entry_sha256"]


def collect_poll(
    *,
    root: Path,
    source_root: Path,
    evidence_root: Path,
    stage: str,
    run_id: str,
    dashboard_url: str,
    dom_capture_path: Path,
    direct_url_path: Path,
    executable: Path,
    pid: int,
) -> dict[str, Any]:
    """Read one independent poll and append it immutably.  Time is OS supplied."""
    _id(run_id, "run id")
    if root.resolve() != store.CHRONOVISOR_ROOT.resolve():
        raise EvidenceError("collector root is not the production runtime")
    if any(
        _has_symlink_component(path)
        for path in (
            root,
            source_root,
            evidence_root,
            dom_capture_path,
            direct_url_path,
            executable,
        )
    ):
        raise EvidenceError("symlinked collector root/input")
    if evidence_root.resolve(strict=False).is_relative_to(
        root.resolve()
    ) or evidence_root.resolve(strict=False).is_relative_to(source_root.resolve()):
        raise EvidenceError("evidence output overlaps protected root")
    if any(
        evidence_root.resolve(strict=False).is_relative_to(path.parent.resolve())
        or path.parent.resolve().is_relative_to(evidence_root.resolve(strict=False))
        for path in (dom_capture_path, direct_url_path)
    ):
        raise EvidenceError("evidence output overlaps input parent")
    now = datetime.now(UTC)
    monotonic_ns = time.monotonic_ns()
    identities = _stage_state(root, stage)
    observations, observation_chain = _runtime_observations(root, stage, identities)
    if observation_chain["stage_run_id"] != run_id:
        raise EvidenceError("poll run does not match runtime stage")
    dom = _safe_json(dom_capture_path, "DOM capture")
    if (
        dom.get("synthetic_fixture") is not False
        or dom.get("kind") != "browser-dom-capture"
        or not isinstance(dom.get("html_sha256"), str)
        or not isinstance(dom.get("producer"), Mapping)
        or not isinstance(dom["producer"].get("name"), str)
        or not isinstance(dom["producer"].get("version"), int)
    ):
        raise EvidenceError("independent DOM evidence unavailable")
    health = _fetch(f"{dashboard_url.rstrip('/')}/api/health", "dashboard health")
    api = _fetch(f"{dashboard_url.rstrip('/')}/api/fast-snapshot", "dashboard API")
    runtime = _direct_url(direct_url_path)
    process = _process_identity(executable, pid)
    source = _source_identity(source_root)
    health_payload = health["payload"]
    health_runtime = (
        health_payload.get("health", {}).get("runtime", {})
        if isinstance(health_payload.get("health"), Mapping)
        else {}
    )
    if (
        runtime["archive_commit"] != source["source_commit"]
        or not isinstance(health_runtime, Mapping)
        or health_runtime.get("commit_id") != source["source_commit"]
        or health_runtime.get("drift") is not False
    ):
        raise EvidenceError("runtime/archive/dashboard commit drift")
    payload = {
        "kind": "r7-live-poll",
        "stage": stage,
        "run_id": run_id,
        "captured_at": now.isoformat(),
        "monotonic_ns": monotonic_ns,
        "identities": identities,
        "source": source,
        "runtime": runtime,
        "process": process,
        "health": {key: value for key, value in health.items() if key != "payload"},
        "api": {key: value for key, value in api.items() if key != "payload"},
        "dom_sha256": _digest(dom),
        "observation_chain": observation_chain,
        "observations_sha256": _digest(observations),
        "observations": observations,
        "producer": {
            "name": "chronovisor-r7-evidence",
            "version": 1,
            "synthetic_fixture": False,
        },
    }
    poll_id, _, artifact = store.write_immutable(
        evidence_root / "polls", payload, schema=POLL_SCHEMA
    )
    # Re-stat every mutable input and source/state after writing.  A changed
    # input is not evidence for this immutable poll.
    if (
        _digest(_safe_json(dom_capture_path, "DOM capture")) != _digest(dom)
        or _direct_url(direct_url_path) != runtime
        or _process_identity(executable, pid) != process
        or _source_identity(source_root) != source
        or _stage_state(root, stage) != identities
        or _runtime_observations(root, stage, identities)[1] != observation_chain
    ):
        raise EvidenceError("TOCTOU input/state drift")
    ledger = _append_ledger(
        evidence_root / "poll-ledger.jsonl", poll_id, stage, now, monotonic_ns
    )
    return {
        "poll_id": poll_id,
        "poll_sha256": artifact["seal_sha256"],
        "ledger_entry_sha256": ledger,
        "captured_at": now.isoformat(),
    }


def _verify_runtime_poll(root: Path, poll: Mapping[str, Any]) -> bool:
    """Require every collector row to remain derivable from protected runtime state."""
    try:
        ledger_path = store.distillation_dir(root) / "shadow-observation-receipts.jsonl"
        chain, source_head = _readonly_chain_snapshot(ledger_path)
    except (OSError, EvidenceError):
        return False
    expected_head = poll.get("observation_chain")
    if not isinstance(expected_head, Mapping) or source_head[
        "records"
    ] < expected_head.get("records", -1):
        return False
    if expected_head.get("head_sha256") not in {
        row.get("record_sha256") for row in chain
    }:
        return False
    receipts = {
        row.get("shadow_observation_artifact_id"): row
        for row in chain
        if row.get("kind") == "shadow-policy-observation"
    }
    for row in poll.get("observations", []):
        if not isinstance(row, Mapping):
            return False
        artifact_id = row.get("observation_id")
        receipt = receipts.get(artifact_id)
        if (
            not isinstance(artifact_id, str)
            or _HEX.fullmatch(artifact_id) is None
            or Path(artifact_id).name != artifact_id
            or not isinstance(receipt, Mapping)
        ):
            return False
        try:
            artifact = store.read_sealed(
                store.distillation_dir(root)
                / "shadow-observations"
                / f"{artifact_id}.json",
                schema=distillation.SHADOW_OBSERVATION_SCHEMA,
            )
        except store.DistillationStoreError:
            return False
        if (
            artifact.get("artifact_id") != artifact_id
            or artifact.get("decision_id") != receipt.get("decision_id")
            or row.get("session_sha256") != artifact.get("session_id_sha256")
            or row.get("query_sha256") != artifact.get("query_semantic_sha256")
            or row.get("candidate_pool_sha256") != artifact.get("candidate_pool_sha256")
        ):
            return False
    return True


def _held_collector(reason: str, polls: int = 0) -> dict[str, Any]:
    return {
        "schema": EVIDENCE_SCHEMA,
        "certification": False,
        "certification_reason": reason,
        "stages": {stage: {"certified": False, "reason": reason} for stage in STAGES},
        "polls": polls,
        "protected_state_unchanged": False,
        "identity": {},
        "source": {},
    }


def _fail_closed_collector(function: Any) -> Any:
    def wrapped(evidence_root: Path, *, root: Path | None = None) -> dict[str, Any]:
        try:
            return function(evidence_root, root=root)
        except (
            EvidenceError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
            store.DistillationStoreError,
        ):
            return _held_collector("collector_bundle_invalid")

    return wrapped


@_fail_closed_collector
def validate_collector(
    evidence_root: Path, *, root: Path | None = None
) -> dict[str, Any]:
    """Recompute the only certifiable facts from immutable collector output."""
    try:
        ledger = _ledger_rows(evidence_root / "poll-ledger.jsonl")
        _check_ledger_state(evidence_root, ledger)
    except EvidenceError:
        return _held_collector("collector_ledger_invalid")
    polls: list[dict[str, Any]] = []
    for entry in ledger:
        path = evidence_root / "polls" / f"{entry['poll_id']}.json"
        try:
            poll = store.read_sealed(path, schema=POLL_SCHEMA)
        except store.DistillationStoreError:
            return _held_collector("collector_poll_invalid", len(polls))
        if (
            poll.get("artifact_id") != entry["poll_id"]
            or poll.get("stage") != entry["stage"]
        ):
            raise EvidenceError("poll ledger/artifact binding mismatch")
        producer = poll.get("producer")
        if (
            not isinstance(producer, Mapping)
            or producer.get("synthetic_fixture") is not False
        ):
            raise EvidenceError("synthetic poll cannot certify")
        chain = poll.get("observation_chain")
        if (
            not isinstance(chain, Mapping)
            or isinstance(chain.get("records"), bool)
            or not isinstance(chain.get("records"), int)
            or not isinstance(chain.get("head_sha256"), str)
            or (chain["head_sha256"] and _HEX.fullmatch(chain["head_sha256"]) is None)
            or not isinstance(poll.get("identities"), Mapping)
            or not isinstance(poll.get("source"), Mapping)
        ):
            return _held_collector("collector_poll_provenance_invalid", len(polls))
        try:
            _utc(poll.get("captured_at"), "poll time")
        except EvidenceError:
            return _held_collector("collector_poll_timestamp_invalid", len(polls))
        polls.append(poll)
    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    if any(entry.get("stage") not in stage_order for entry in ledger) or any(
        stage_order[str(left["stage"])] > stage_order[str(right["stage"])]
        for left, right in zip(ledger, ledger[1:], strict=False)
    ):
        raise EvidenceError("rollout stage sequence is invalid")
    stages: dict[str, dict[str, Any]] = {}
    reused: set[str] = set()
    reused_decisions: set[str] = set()
    stage_runs: set[str] = set()
    identities = {
        _digest(
            {
                key: value
                for key, value in poll["identities"].items()
                if key != "active_id"
            }
        )
        for poll in polls
    }
    sources = {_digest(poll["source"]) for poll in polls}
    if len(identities) > 1 or len(sources) > 1:
        raise EvidenceError("poll identity drift")
    for stage in STAGES:
        stage_polls = [poll for poll in polls if poll.get("stage") == stage]
        if not stage_polls:
            stages[stage] = {"certified": False, "reason": "no_real_polls"}
            continue
        times = [_utc(poll["captured_at"], "poll time") for poll in stage_polls]
        observations = [
            row for poll in stage_polls for row in poll.get("observations", [])
        ]
        run_ids = {poll.get("run_id") for poll in stage_polls}
        if (
            len(run_ids) != 1
            or not all(
                isinstance(run_id, str) and _HEX.fullmatch(run_id) for run_id in run_ids
            )
            or stage_runs.intersection(run_ids)
            or not all(isinstance(row, Mapping) for row in observations)
        ):
            raise EvidenceError("stage/run binding mismatch")
        stage_runs.update(run_ids)
        ids = [row.get("observation_id") for row in observations]
        if (
            any(not isinstance(item, str) for item in ids)
            or len(ids) != len(set(ids))
            or reused.intersection(ids)
        ):
            raise EvidenceError("duplicate cross-stage observations")
        reused.update(ids)
        decision_ids = {
            _digest(
                (
                    row.get("decision_sha256"),
                    row.get("session_sha256"),
                    row.get("query_sha256"),
                    row.get("candidate_pool_sha256"),
                    row.get("feature_bytes_sha256"),
                )
            )
            for row in observations
        }
        if len(decision_ids) != len(observations) or reused_decisions.intersection(
            decision_ids
        ):
            raise EvidenceError("same-decision cross-stage reuse")
        reused_decisions.update(decision_ids)
        host = Counter(str(row["host"]) for row in observations)
        cohort = Counter(str(row["cohort"]) for row in observations)
        total = len(observations)
        quality = sum(bool(row.get("candidate_quality")) for row in observations)
        baseline_quality = sum(
            bool(row.get("baseline_quality")) for row in observations
        )
        coverage = sum(bool(row.get("candidate_covered")) for row in observations)
        anchor = sum(bool(row.get("candidate_anchor_retained")) for row in observations)
        baseline_anchor = sum(
            bool(row.get("baseline_anchor_retained")) for row in observations
        )
        abstain_delta = (
            (
                sum(bool(row.get("candidate_abstained")) for row in observations)
                - sum(bool(row.get("baseline_abstained")) for row in observations)
            )
            / total
            if total
            else 1.0
        )
        timeouts = sum(bool(row.get("timed_out")) for row in observations)
        score = [row.get("candidate_score_ms") for row in observations]
        live = [row.get("live_latency_ms") for row in observations]
        metrics_ok = (
            bool(total)
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (*score, *live)
            )
            and quality >= baseline_quality
            and coverage / total >= 0.95
            and anchor >= baseline_anchor
            and abstain_delta <= 0.02
            and _p95(score) <= 180
            and _p95(live) < 900
            and timeouts / total <= 0.01
            and _wilson_lower(total - timeouts, total) >= 0.97
        )
        stages[stage] = {
            "certified": times[-1] - times[0] >= timedelta(days=MIN_DAYS)
            and total >= MIN_PAIRED
            and metrics_ok,
            "days": (times[-1] - times[0]).total_seconds() / 86_400,
            "paired": total,
            "host_counts": dict(host),
            "cohort_counts": dict(cohort),
            "metrics_ok": metrics_ok,
        }
    production_root = store.CHRONOVISOR_ROOT.resolve()
    source_bound = False
    if (
        root is not None
        and not _has_symlink_component(root)
        and root.resolve() == production_root
    ):
        try:
            before = _protected_file_state(production_root)
            source_bound = all(
                _verify_runtime_poll(production_root, poll) for poll in polls
            )
            source_bound = source_bound and before == _protected_file_state(
                production_root
            )
        except (OSError, EvidenceError):
            source_bound = False
    # Current runtime receipts do not carry all R7/P8 dimensions (independent
    # DOM, route probes, quality/anchor outcomes, and resource attestations).
    # Do not convert a complete-looking poll set into a production certificate.
    certified = False
    hold_reason = (
        "full_authoritative_production_snapshot_unavailable"
        if source_bound
        else "authoritative_runtime_observation_chain_unavailable"
    )
    for stage in stages.values():
        stage["certified"] = False
        stage["reason"] = hold_reason
    return {
        "schema": EVIDENCE_SCHEMA,
        "certification": certified,
        "certification_reason": hold_reason,
        "stages": stages,
        "polls": len(polls),
        "protected_state_unchanged": source_bound,
        "identity": (
            {
                key: value
                for key, value in polls[0].get("identities", {}).items()
                if key != "active_id"
            }
            if polls
            else {}
        ),
        "source": polls[0].get("source", {}) if polls else {},
    }


def validate_rollback(root: Path, receipt_path: Path) -> dict[str, str]:
    """Reject rollback certification until its authoritative binding exists."""
    if (
        _has_symlink_component(root)
        or root.resolve() != store.CHRONOVISOR_ROOT.resolve()
    ):
        raise EvidenceError("rollback root is not the production runtime")
    # R7 has no immutable stage-receipt/poll-to-rollback binding producer yet.
    # Sealed post-state alone cannot authenticate caller-supplied refs.
    raise EvidenceError("rollback authoritative R7 binding is unavailable")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record-poll", help="read and seal one live poll")
    for name in (
        "root",
        "source-root",
        "evidence-root",
        "dom-capture",
        "direct-url",
        "executable",
    ):
        record.add_argument(f"--{name}", type=Path, required=True)
    record.add_argument("--stage", choices=STAGES, required=True)
    record.add_argument("--run-id", required=True)
    record.add_argument("--dashboard-url", required=True)
    record.add_argument("--pid", type=int, required=True)
    validate = commands.add_parser("validate", help="recompute sealed poll evidence")
    validate.add_argument("--evidence-root", type=Path, required=True)
    validate.add_argument("--root", type=Path)
    validate.add_argument("--forced-failure-receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "record-poll":
            result = collect_poll(
                root=args.root,
                source_root=args.source_root,
                evidence_root=args.evidence_root,
                stage=args.stage,
                run_id=args.run_id,
                dashboard_url=args.dashboard_url,
                dom_capture_path=args.dom_capture,
                direct_url_path=args.direct_url,
                executable=args.executable,
                pid=args.pid,
            )
        else:
            result = validate_collector(args.evidence_root, root=args.root)
            if args.forced_failure_receipt is not None and args.root is None:
                raise EvidenceError("rollback validation needs root")
            if args.forced_failure_receipt is not None:
                result["rollback"] = validate_rollback(
                    args.root, args.forced_failure_receipt
                )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (EvidenceError, OSError, store.DistillationStoreError) as exc:
        print(f"r7 evidence failed: {str(exc).split(':', 1)[0]}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
