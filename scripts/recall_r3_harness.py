#!/usr/bin/env python3
"""Run the offline Recall R3 workset durability/performance gate.

The gate exercises only the local SQLite workset. It never calls a teacher or
another provider and it runs all writes in throwaway clone-local directories.
The evidence is sealed with the normal Recall immutable-artifact writer and
contains counts, digests, and timings, never work payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import selectors
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

R3_SCHEMA = "chronovisor.recall-r3.v1"
DEFAULT_SAMPLES = 100
MIN_SAMPLES = 100
UNIT_MIN_SAMPLES = 20
CLAIM_P95_LIMIT_NS = 500_000_000
TEACHER_HANDOFF_LIMIT_NS = 10_000_000_000
RECEIPT_COVERAGE_LIMIT = 99.0
SIX_STAGES = (
    "snapshot",
    "teacher",
    "counterfactual",
    "retry_wait",
    "dataset",
    "evaluation",
)
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class R3Error(ValueError):
    """An R3 gate failed closed."""


def _load_r2() -> Any:
    """Load clone/tree helpers without making ``scripts`` a package."""

    path = Path(__file__).with_name("recall_r2_harness.py")
    spec = importlib.util.spec_from_file_location("chronovisor_r2_harness", path)
    if spec is None or spec.loader is None:
        raise R3Error("R2 helper unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


R2 = _load_r2()


def _load_runtime(source_root: Path) -> tuple[Any, Any]:
    """Bind the workset/store modules to the requested source checkout."""

    source_path = str(source_root / "src")
    if not (source_root / "src" / "chronovisor").is_dir():
        raise R3Error("source root does not contain src/chronovisor")
    if source_path in sys.path:
        sys.path.remove(source_path)
    sys.path.insert(0, source_path)
    try:
        workset = __import__(
            "chronovisor.recall.recall_distillation_workset",
            fromlist=["DistillationWorkset"],
        )
        store = __import__(
            "chronovisor.recall.recall_distillation_store",
            fromlist=["write_immutable"],
        )
    except (ImportError, OSError) as exc:
        raise R3Error("R3 runtime modules are unavailable") from exc
    for module in (workset, store):
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(source_root / "src"):
            raise R3Error("R3 runtime module escaped source root")
    return workset, store


def _has_symlink_component(path: Path) -> bool:
    current = path.expanduser()
    while True:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _path_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.expanduser().resolve(strict=False)
    right_resolved = right.expanduser().resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved.is_relative_to(right_resolved)
        or right_resolved.is_relative_to(left_resolved)
    )


def _assert_root_matrix(
    production: Path, source_root: Path, output: Path, clones: tuple[Path, ...] = ()
) -> None:
    """Reject symlink entry points and every protected-root overlap."""

    paths = {
        "production": production,
        "source_root": source_root,
        "output": output,
        **{f"clone[{index}]": value for index, value in enumerate(clones)},
    }
    for name, path in paths.items():
        if _has_symlink_component(path):
            raise R3Error(f"{name} path contains a symlink")
    entries = tuple(paths.items())
    for index, (left_name, left) in enumerate(entries):
        for right_name, right in entries[index + 1 :]:
            if _path_overlap(left, right):
                raise R3Error(f"{left_name}/{right_name} paths overlap")
    for name, path in paths.items():
        if name == "output" or name.startswith("clone["):
            continue
        if not path.is_dir():
            raise R3Error(f"{name} root is not a directory")


def _git_head(source_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise R3Error("source commit lookup failed") from exc
    head = result.stdout.strip()
    if _COMMIT_RE.fullmatch(head) is None:
        raise R3Error("source HEAD is not a full commit")
    return head


def _source_snapshot(source_root: Path) -> dict[str, Any]:
    try:
        return R2._source_tree_digest(source_root)
    except Exception as exc:
        raise R3Error("source snapshot failed") from exc


def _production_snapshot(production: Path) -> dict[str, Any]:
    try:
        return R2._tree_digest(production, label="production")
    except Exception as exc:
        raise R3Error("production snapshot failed") from exc


def _clone_from_root(source: Path) -> Path:
    """Use the existing forced APFS clone implementation."""

    try:
        return R2._clone_from_root(source)
    except Exception as exc:
        raise R3Error(str(exc)) from exc


def _cleanup_clone(path: Path) -> None:
    try:
        R2._cleanup_clone(path)
    except Exception as exc:
        raise R3Error("clone/temp cleanup failed") from exc


def _p95(values: list[int]) -> int:
    if not values:
        raise R3Error("p95 sample is empty")
    ordered = sorted(values)
    return ordered[max(0, (len(ordered) * 95 + 99) // 100 - 1)]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _item(work_id: str, kind: str, *, priority: int = 0) -> dict[str, Any]:
    digest = _digest({"work_id": work_id, "kind": kind})
    return {
        "work_id": work_id,
        "kind": kind,
        "payload_ref": f"candidate-ledger:r3-{work_id}",
        "payload_digest": digest,
        "priority": priority,
        "temporal_split": {"partition": "train", "cutoff": "2026-08-24"},
        "provenance": {"cohort": "r3-harness-v1", "route": "offline"},
    }


def _progress(cursor: int) -> dict[str, Any]:
    digest = f"{cursor + 1:064x}"[-64:]
    return {
        "cursor": {"completed": cursor},
        "ledger_heads": {"workset": digest},
        "provenance": {"cohort": "r3-harness-v1", "revision": "offline"},
        "progress_kind": "r3-harness-v1",
    }


def _completed(claim: Any) -> dict[str, str]:
    return {
        "status": "completed",
        "completion_ref": f"label-ledger:r3-{claim.work_id}",
        "completion_digest": claim.payload_digest,
    }


def _receipt_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT generation, previous_sha256, operation, payload_json, "
                "receipt_sha256 FROM workset_receipts ORDER BY generation"
            ).fetchall()
    except sqlite3.Error as exc:
        raise R3Error("workset receipt read failed") from exc
    result: list[dict[str, Any]] = []
    for generation, previous, operation, payload_json, receipt in rows:
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError) as exc:
            raise R3Error("workset receipt JSON is invalid") from exc
        result.append(
            {
                "generation": int(generation),
                "previous_sha256": str(previous),
                "operation": str(operation),
                "payload": payload,
                "receipt_sha256": str(receipt),
            }
        )
    return result


def _duplicate_count(path: Path) -> int:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT COALESCE(SUM(n - 1), 0) FROM "
                "(SELECT COUNT(*) AS n FROM work_items GROUP BY work_id)"
            ).fetchone()
            receipt_rows = connection.execute(
                "SELECT COALESCE(SUM(n - 1), 0) FROM "
                "(SELECT COUNT(*) AS n FROM workset_receipts GROUP BY receipt_sha256)"
            ).fetchone()
    except sqlite3.Error as exc:
        raise R3Error("duplicate audit failed") from exc
    return int(rows[0] or 0) + int(receipt_rows[0] or 0)


_TEACHER_CHILD = r'''
import json, sys
request = json.loads(sys.stdin.read())
if not isinstance(request, dict) or set(request) != {"count"}:
    raise SystemExit(2)
print(json.dumps({"accepted": int(request["count"])}, separators=(",", ":")))
'''


_SIGTERM_CHILD = r'''
import dataclasses, json, sys, time
from chronovisor.recall.recall_distillation_workset import DistillationWorkset
path = sys.argv[1]
workset = DistillationWorkset(path)
claims = workset.claim("r3-sigterm", 1, "sigterm-child", 0.5)
if len(claims) != 1:
    raise SystemExit(3)
print(json.dumps({"ready": True, "claim": dataclasses.asdict(claims[0])}, separators=(",", ":")), flush=True)
time.sleep(30)
'''


def _child_env(source_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source_path = str(source_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_path, environment.get("PYTHONPATH", "")) if value
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _teacher_handoff(source_root: Path, sample_count: int) -> dict[str, Any]:
    started = time.perf_counter_ns()
    try:
        result = subprocess.run(
            [sys.executable, "-c", _TEACHER_CHILD],
            input=json.dumps({"count": sample_count}),
            text=True,
            capture_output=True,
            check=True,
            timeout=10.0,
            env=_child_env(source_root),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise R3Error("teacher handoff failed") from exc
    elapsed = time.perf_counter_ns() - started
    try:
        response = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        raise R3Error("teacher handoff response is invalid") from exc
    if response != {"accepted": sample_count}:
        raise R3Error("teacher handoff response coverage is invalid")
    return {"wall_time_ns": elapsed, "accepted": sample_count}


def _read_child_line(process: subprocess.Popen[str], timeout: float) -> str:
    if process.stdout is None:
        raise R3Error("SIGTERM child stdout is unavailable")
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        events = selector.select(timeout)
        if not events:
            raise R3Error("SIGTERM child did not report readiness")
        line = process.stdout.readline()
    finally:
        selector.close()
    if not line:
        raise R3Error("SIGTERM child exited before readiness")
    return line


def _sigterm_reopen(workset_module: Any, source_root: Path, path: Path) -> dict[str, Any]:
    workset = workset_module.DistillationWorkset(path)
    started = time.perf_counter_ns()
    process = subprocess.Popen(
        [sys.executable, "-c", _SIGTERM_CHILD, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(source_root),
    )
    try:
        line = _read_child_line(process, 10.0)
        try:
            ready = json.loads(line)
            claim_value = ready["claim"]
        except (TypeError, ValueError, KeyError) as exc:
            raise R3Error("SIGTERM child claim is invalid") from exc
        if ready.get("ready") is not True or not isinstance(claim_value, Mapping):
            raise R3Error("SIGTERM child did not seal a claim")
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
            raise R3Error("SIGTERM child did not terminate") from None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    try:
        old_claim = workset_module.WorkClaim(**dict(claim_value))
    except (TypeError, ValueError) as exc:
        raise R3Error("SIGTERM child claim shape is invalid") from exc
    expiry = float(old_claim.lease_expires_at)
    while time.time() <= expiry:
        time.sleep(min(0.05, max(0.001, expiry - time.time())))
    reopened = workset_module.DistillationWorkset(path)
    claims = reopened.claim("r3-sigterm", 1, "reopened-owner", 5.0)
    if len(claims) != 1 or claims[0].attempt != old_claim.attempt + 1:
        raise R3Error("expired SIGTERM lease was not reclaimed")
    reclaimed = claims[0]
    try:
        workset.commit([old_claim], [_completed(old_claim)])
    except workset_module.DistillationWorksetError:
        old_owner_rejected = True
    else:
        old_owner_rejected = False
    if not old_owner_rejected:
        raise R3Error("expired old owner was accepted")
    outcome = _completed(reclaimed)
    first = reopened.commit([reclaimed], [outcome])
    second = reopened.commit([reclaimed], [outcome])
    if first != second or first.get("completed") != 1:
        raise R3Error("idempotent commit failed after lease reclaim")
    status = reopened.status("r3-sigterm")
    if status.get("completed") != 1 or status.get("leased") != 0:
        raise R3Error("reopened workset status is invalid")
    return {
        "wall_time_ns": time.perf_counter_ns() - started,
        "child_returncode": process.returncode,
        "old_owner_rejected": old_owner_rejected,
        "reclaimed_attempt": reclaimed.attempt,
        "idempotent_commit": True,
        "duplicates": _duplicate_count(path),
    }


def _run_workset(
    workset_module: Any, source_root: Path, root: Path, samples: int
) -> dict[str, Any]:
    if samples < UNIT_MIN_SAMPLES:
        raise R3Error(f"R3 sample count must be at least {UNIT_MIN_SAMPLES}")
    if _has_symlink_component(root):
        raise R3Error("workset root contains a symlink")
    root.mkdir(parents=True, exist_ok=True)
    stage_path = root / "stages.sqlite3"
    crash_path = root / "sigterm.sqlite3"
    clock_value = [0.0]
    stage_workset = workset_module.DistillationWorkset(
        stage_path, clock=lambda: clock_value[0]
    )
    # Make one teacher item old enough to prove cross-kind fairness over a
    # newer, higher-priority kind.
    stage_workset.advance(
        [_item("r3-old-teacher", "local-teacher:old", priority=0)],
        {"source": 0},
        progress=_progress(0),
    )
    clock_value[0] = 61.0
    stage_items: list[dict[str, Any]] = []
    kind_by_stage = {
        "snapshot": "snapshot",
        "teacher": "local-teacher:a",
        "counterfactual": "counterfactual",
        "dataset": "dataset",
        "evaluation": "evaluation",
    }
    for stage, kind in kind_by_stage.items():
        count = samples - 1 if stage == "teacher" else samples
        items = [_item(f"r3-{stage}-{index}", kind, priority=99) for index in range(count)]
        stage_items.extend(items)
        stage_workset.advance(items, {"source": stage}, progress=_progress(0))

    claim_samples: list[int] = []
    first_started = time.perf_counter_ns()
    first_claims = stage_workset.claim(None, 1, "fairness-worker", 60.0)
    claim_samples.append(time.perf_counter_ns() - first_started)
    if len(first_claims) != 1 or first_claims[0].work_id != "r3-old-teacher":
        raise R3Error("cross-kind fairness selected a newer kind")
    completed_count = 1
    stage_workset.commit(
        first_claims,
        [{"status": "retry", "error_class": "transport_error", "retry_after_seconds": 30}],
        progress=_progress(completed_count),
    )
    retry_status = stage_workset.status(include_timing=True)
    stages = retry_status.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != set(SIX_STAGES):
        raise R3Error("six-stage status projection is incomplete")
    if retry_status.get("retry_wait") != 1 or stages["teacher"].get("retry_wait") != 1:
        raise R3Error("retry_wait observability is missing")

    while True:
        started = time.perf_counter_ns()
        claims = stage_workset.claim(None, 1, "fairness-worker", 60.0)
        claim_samples.append(time.perf_counter_ns() - started)
        if not claims:
            break
        claim = claims[0]
        completed_count += 1
        stage_workset.commit(
            claims,
            [_completed(claim)],
            progress=_progress(completed_count),
        )
    expected_completed = len(stage_items)
    final_status = stage_workset.status(include_timing=True)
    if final_status.get("completed") != expected_completed:
        raise R3Error("stage workset did not complete every ready item")
    claim_p95 = _p95(claim_samples)
    if claim_p95 > CLAIM_P95_LIMIT_NS:
        raise R3Error("claim p95 exceeded 500ms")

    handoff = _teacher_handoff(source_root, samples)
    if int(handoff["wall_time_ns"]) > TEACHER_HANDOFF_LIMIT_NS:
        raise R3Error("teacher handoff exceeded 10 seconds")

    crash_workset = workset_module.DistillationWorkset(crash_path)
    crash_workset.advance([_item("r3-sigterm-item", "r3-sigterm")], {"source": 1})
    crash = _sigterm_reopen(workset_module, source_root, crash_path)
    if crash["duplicates"] != 0:
        raise R3Error("SIGTERM recovery produced duplicate rows")

    audit = stage_workset.audit_transition_receipts()
    if audit.get("status") != "verified":
        raise R3Error("stage receipt chain is not verified")
    durable = final_status.get("last_durable_receipt")
    progress = stage_workset.progress()
    if (
        not isinstance(durable, Mapping)
        or audit.get("generation") != durable.get("generation")
        or audit.get("head_sha256") != durable.get("head_sha256")
        or audit.get("progress") != progress
        or final_status.get("last_durable_progress") != progress
    ):
        raise R3Error("last durable receipt/progress parity failed")
    receipts = _receipt_rows(stage_path)
    if not receipts:
        raise R3Error("no durable receipts were emitted")
    claim_count = len(claim_samples) - 1  # final empty claim is an observation only
    expected_receipts = 6 + claim_count + claim_count
    # Once progress is initialized, Workset seals it into claim/commit receipts
    # too; the formal denominator therefore covers every durable receipt.
    expected_progress_receipts = expected_receipts
    operation_counts = Counter(row["operation"] for row in receipts)
    if operation_counts != Counter(
        {"advance": 6, "claim": claim_count, "commit": claim_count}
    ):
        raise R3Error("durable receipt operation coverage is incomplete")
    receipt_coverage = 100.0 * len(receipts) / expected_receipts
    progress_receipts = sum(
        1
        for row in receipts
        if isinstance(row["payload"], Mapping) and row["payload"].get("version") == 2
    )
    progress_coverage = 100.0 * progress_receipts / expected_progress_receipts
    if receipt_coverage < RECEIPT_COVERAGE_LIMIT or progress_coverage < RECEIPT_COVERAGE_LIMIT:
        raise R3Error("durable receipt/progress coverage is below 99%")
    duplicates = _duplicate_count(stage_path) + _duplicate_count(crash_path)
    if duplicates != 0:
        raise R3Error("workset duplicate count is non-zero")
    return {
        "samples": samples,
        "stages": dict(retry_status["stages"]),
        "final_status": {
            key: value
            for key, value in final_status.items()
            if key in {"ready", "leased", "completed", "quarantined", "retry_wait", "total"}
        },
        "fairness": {"oldest_work_id_sha256": _digest("r3-old-teacher"), "passed": True},
        "claim": {
            "samples": len(claim_samples),
            "p95_ns": claim_p95,
            "threshold_ns": CLAIM_P95_LIMIT_NS,
        },
        "teacher_handoff": {**handoff, "threshold_ns": TEACHER_HANDOFF_LIMIT_NS},
        "sigterm_reopen": crash,
        "durability": {
            "samples": samples,
            "receipt_count": len(receipts),
            "expected_receipt_count": expected_receipts,
            "progress_receipt_count": progress_receipts,
            "expected_progress_receipt_count": expected_progress_receipts,
            "receipt_coverage_pct": receipt_coverage,
            "progress_coverage_pct": progress_coverage,
            "coverage": {
                "denominator": expected_receipts,
                "receipts": len(receipts),
                "percent": receipt_coverage,
            },
            "progress_coverage": {
                "denominator": expected_progress_receipts,
                "receipts": progress_receipts,
                "percent": progress_coverage,
            },
            "last_durable_receipt": dict(durable),
            "last_durable_progress": progress,
            "audit_status": audit["status"],
        },
        "duplicates": duplicates,
        "payload_free": True,
    }


def _assert_payload_free(payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    forbidden = ("private raw", "authorization:", "bearer ", "api_key")
    if any(marker in encoded.lower() for marker in forbidden):
        raise R3Error("evidence contains a payload or credential marker")
    for key, value in payload.items():
        if "payload" in str(key).lower() and key not in {"payload_free"}:
            raise R3Error("evidence contains a payload field")
        if isinstance(value, Mapping):
            _assert_payload_free(value)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, Mapping):
                    _assert_payload_free(child)


def _run_once(
    *,
    production: Path,
    source_root: Path,
    source_commit: str,
    output: Path,
    samples: int,
) -> dict[str, Any]:
    if samples < MIN_SAMPLES:
        raise R3Error(f"formal R3 sample count must be at least {MIN_SAMPLES}")
    if not _COMMIT_RE.fullmatch(source_commit):
        raise R3Error("source commit must be a full lowercase SHA-1")
    _assert_root_matrix(production, source_root, output)
    if output.exists() and not output.is_dir():
        raise R3Error("output root is not a directory")
    if _git_head(source_root) != source_commit:
        raise R3Error("source commit does not match source HEAD")
    source_before = _source_snapshot(source_root)
    production_before = _production_snapshot(production)
    workset_module, store = _load_runtime(source_root)
    clone: Path | None = None
    work_root: Path | None = None
    try:
        clone = _clone_from_root(production)
        _assert_root_matrix(production, source_root, output, (clone,))
        work_root = Path(tempfile.mkdtemp(prefix="r3-harness-", dir=clone / "runtime"))
        result = _run_workset(workset_module, source_root, work_root, samples)
        if _production_snapshot(production) != production_before:
            raise R3Error("production changed during R3 run")
        if _source_snapshot(source_root) != source_before:
            raise R3Error("source changed during R3 run")
        shutil.rmtree(work_root, ignore_errors=True)
        if work_root.exists():
            raise R3Error("R3 workset temp cleanup failed")
        work_root = None
        _cleanup_clone(clone)
        clone = None
        payload = {
            "runtime": {"source_commit": source_commit, "external_provider_calls": 0},
            "production_unchanged": True,
            "source_unchanged": True,
            "clone_temp_cleanup_verified": True,
            "result": result,
            "thresholds": {
                "claim_p95_ns": CLAIM_P95_LIMIT_NS,
                "teacher_handoff_ns": TEACHER_HANDOFF_LIMIT_NS,
                "durable_coverage_pct": RECEIPT_COVERAGE_LIMIT,
            },
        }
        _assert_payload_free(payload)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > 2 * 1024 * 1024:
            raise R3Error("R3 evidence exceeds bounded size")
        artifact_id, artifact_path, artifact = store.write_immutable(
            output, payload, schema=R3_SCHEMA
        )
        return {
            "schema": artifact["schema"],
            "artifact_id": artifact_id,
            "path": str(artifact_path),
            "claim_p95_ns": result["claim"]["p95_ns"],
            "teacher_handoff_ns": result["teacher_handoff"]["wall_time_ns"],
            "duplicates": result["duplicates"],
            "clone_cleanup_verified": True,
        }
    finally:
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)
            if work_root.exists():
                raise R3Error("R3 workset temp cleanup failed")
        if clone is not None:
            _cleanup_clone(clone)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--dashboard-url")
    parser.add_argument("--isolated-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.isolated_root is not None:
            raise R3Error("--isolated-root is unsupported; use forced APFS clones")
        paths = (args.production_root, args.source_root, args.output)
        if any(_has_symlink_component(path.expanduser()) for path in paths):
            raise R3Error("root/output path contains a symlink")
        production = args.production_root.expanduser().resolve(strict=True)
        source_root = args.source_root.expanduser().resolve(strict=True)
        output = args.output.expanduser().resolve(strict=False)
        result = _run_once(
            production=production,
            source_root=source_root,
            source_commit=args.source_commit,
            output=output,
            samples=args.samples,
        )
        print(json.dumps(result, sort_keys=True))
    except (R3Error, OSError, ValueError, sqlite3.Error) as exc:
        print(f"r3 harness failed: {str(exc).split(':', 1)[0]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
