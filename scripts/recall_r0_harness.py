#!/usr/bin/env python3
"""Capture the Recall R0 baseline without mutating the live root."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import functools
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "chronovisor.recall-r0.v1"
STAGES = (
    "_prepare_distillation_chunk",
    "_run_teacher_batch",
    "_run_counterfactual_block",
    "_prepare_distillation_training",
    "_persist_distillation_chunk",
)
LEDGERS = ("rally-manifest.jsonl", "candidate-ledger.jsonl", "label-ledger.jsonl")
POLICY_SCHEMA = "chronovisor.recall-distill-policy.v2"
RUSAGE_FIELDS = (
    "ri_user_time",
    "ri_system_time",
    "ri_pkg_idle_wkups",
    "ri_interrupt_wkups",
    "ri_pageins",
    "ri_wired_size",
    "ri_resident_size",
    "ri_phys_footprint",
    "ri_proc_start_abstime",
    "ri_proc_exit_abstime",
    "ri_child_user_time",
    "ri_child_system_time",
    "ri_child_pkg_idle_wkups",
    "ri_child_interrupt_wkups",
    "ri_child_pageins",
    "ri_child_elapsed_abstime",
    "ri_diskio_bytesread",
    "ri_diskio_byteswritten",
)
STATE_KEYS = (
    "status",
    "worker_status",
    "rollout_percent",
    "promotion_status",
    "hold_reason",
    "error_code",
    "raw_watermark",
    "baseline_artifact_id",
    "manifest_chain_head",
    "run_id",
    "processed",
    "candidate_snapshots",
    "labels_written",
)


class R0Error(ValueError):
    """A R0 contract failed closed."""


def _stat(path: Path) -> dict[str, int] | None:
    if path.is_symlink():
        raise R0Error("unsafe artifact symlink")
    try:
        value = path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise R0Error("artifact stat failed") from exc
    if not path.is_file():
        raise R0Error("artifact is not a regular file")
    return {
        "size_bytes": int(value.st_size),
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_mtime_ns": int(value.st_mtime_ns),
        "st_ctime_ns": int(value.st_ctime_ns),
    }


def _file_identity(path: Path) -> dict[str, Any]:
    state = _stat(path)
    if state is None:
        raise R0Error("artifact is missing")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if _stat(path) != state:
        raise R0Error("artifact changed during hashing")
    return {"sha256": digest, "file_state": state}


def _chain(store: Any, path: Path) -> dict[str, Any]:
    before = _stat(path)
    try:
        result = store.verify_chain(path)
    except Exception as exc:
        raise R0Error("ledger verification failed") from exc
    after = _stat(path)
    count, head = result.get("records"), result.get("head_sha256")
    if before != after or not isinstance(count, int) or count < 0:
        raise R0Error("ledger changed or count invalid")
    if not isinstance(head, str) or (
        head and (len(head) != 64 or set(head) - set("0123456789abcdef"))
    ):
        raise R0Error("ledger head invalid")
    return {
        "records": count,
        "head_sha256": head,
        "bytes": after["size_bytes"] if after else 0,
        "file_state": after,
    }


def _fts(
    store: Any,
    catalog: Any,
    root: Path,
    watermark: str,
    *,
    require_checkpoint_file_state: bool = True,
) -> dict[str, Any]:
    path = catalog.historical_index_path(root)
    state = _stat(path)
    if state is None:
        raise R0Error("historical FTS index missing")
    try:
        checkpoint = store.read_sealed(
            path.with_suffix(path.suffix + ".checkpoint.json"),
            schema=store.DISTILLATION_SCHEMA,
        )
    except Exception as exc:
        raise R0Error("historical FTS checkpoint invalid") from exc
    digest, count = checkpoint.get("content_sha256"), checkpoint.get("atom_count")
    if (
        checkpoint.get("kind") != "historical-index-checkpoint"
        or checkpoint.get("index_name") != path.name
        or checkpoint.get("historical_index_schema")
        != "chronovisor.recall-historical-fts.v1"
        or checkpoint.get("catalog_watermark") != watermark
        or not isinstance(digest, str)
        or len(digest) != 64
        or set(digest) - set("0123456789abcdef")
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or (
            require_checkpoint_file_state
            and checkpoint.get("file_state") != state
        )
    ):
        raise R0Error("historical FTS checkpoint mismatch")
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            connection.execute("PRAGMA query_only=ON")
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            atoms = int(connection.execute("SELECT count(*) FROM atoms").fetchone()[0])
            fts_rows = int(
                connection.execute("SELECT count(*) FROM atoms_fts").fetchone()[0]
            )
    except (OSError, sqlite3.DatabaseError, TypeError, ValueError) as exc:
        raise R0Error("historical FTS read-only query failed") from exc
    if (
        _stat(path) != state
        or metadata.get("schema") != "chronovisor.recall-historical-fts.v1"
        or metadata.get("content_sha256") != digest
        or atoms != count
        or fts_rows != atoms
    ):
        raise R0Error("historical FTS changed or metadata mismatch")
    return {
        "content_sha256": digest,
        "atom_count": atoms,
        "fts_count": fts_rows,
        "file_state": state,
        "checkpoint_seal_sha256": checkpoint.get("seal_sha256", ""),
    }


def _loopback_json(
    dashboard_url: str, path: str
) -> tuple[int, bytes, dict[str, Any] | None]:
    parsed = urllib.parse.urlsplit(dashboard_url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.path not in {"", "/", "/api/fast-snapshot", "/api/health"}
        or parsed.query
        or parsed.fragment
        or path not in {"/api/fast-snapshot", "/api/health"}
    ):
        raise R0Error("dashboard endpoint is not loopback")
    try:
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise R0Error("dashboard port invalid")
    except ValueError as exc:
        raise R0Error("dashboard port invalid") from exc
    endpoint = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path, "", "")
    )

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
            raise R0Error("dashboard redirect rejected")

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), NoRedirect()
    )
    try:
        with opener.open(
            urllib.request.Request(
                endpoint, headers={"Accept": "application/json"}, method="GET"
            ),
            timeout=3,
        ) as response:
            status = int(response.status)
            body = response.read(4 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return 401, b"", None
        raise R0Error("dashboard status invalid") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise R0Error("dashboard unavailable") from exc
    if status != 200 or len(body) > 4 * 1024 * 1024:
        raise R0Error("dashboard response invalid")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError) as exc:
        raise R0Error("dashboard JSON invalid") from exc
    if not isinstance(payload, dict):
        raise R0Error("dashboard response is not an object")
    return status, body, payload


def _production(
    store: Any,
    catalog: Any,
    raw_store: Any,
    root: Path,
    dashboard_url: str,
    *,
    clone_copy: bool = False,
) -> dict[str, Any]:
    directory = store.distillation_dir(root)
    ledgers = {name: _chain(store, directory / name) for name in LEDGERS}
    try:
        watermark = raw_store.committed_raw_watermark(root / "raw")
    except Exception as exc:
        raise R0Error("committed Raw watermark invalid") from exc
    if (
        not isinstance(watermark, str)
        or len(watermark) != 64
        or set(watermark) - set("0123456789abcdef")
    ):
        raise R0Error("committed Raw watermark malformed")
    state_path = directory / store.STATE_FILE
    state_file_state = _stat(state_path)
    try:
        state = store.read_sealed(state_path, schema=store.DISTILLATION_SCHEMA)
    except Exception as exc:
        raise R0Error("distillation state invalid") from exc
    compact_state = {
        key: state.get(key)
        for key in STATE_KEYS
        if isinstance(state.get(key), (str, int, bool)) or state.get(key) is None
    }
    pointers: dict[str, Any] = {}
    for kind, filename in store.POINTER_FILES.items():
        pointer_path = directory / filename
        if _stat(pointer_path) is None:
            pointers[kind] = None
            continue
        try:
            pointer = store.read_sealed(
                pointer_path, schema=store.DISTILLATION_SCHEMA
            )
        except Exception as exc:
            raise R0Error("policy pointer invalid") from exc
        policy_id = pointer.get("policy_id")
        if (
            pointer.get("kind") != f"{kind}-policy-pointer"
            or not isinstance(policy_id, str)
            or len(policy_id) != 64
            or set(policy_id) - set("0123456789abcdef")
        ):
            raise R0Error("policy pointer identity invalid")
        try:
            policy = store.read_sealed(
                directory / "policies" / f"{policy_id}.json",
                schema=POLICY_SCHEMA,
            )
        except Exception as exc:
            raise R0Error("policy artifact invalid") from exc
        if policy.get("artifact_id") != policy_id:
            raise R0Error("policy artifact identity mismatch")
        pointers[kind] = {
            "policy_id": policy_id,
            "pointer_seal_sha256": pointer.get("seal_sha256", ""),
            "policy_seal_sha256": policy.get("seal_sha256", ""),
            "pointer_file_state": _stat(pointer_path),
            "policy_file_state": _stat(
                directory / "policies" / f"{policy_id}.json"
            ),
        }
    status, body, payload = _loopback_json(dashboard_url, "/api/fast-snapshot")
    fast_snapshot = {"status": status, "payload_sha256": None}
    if payload is not None:
        fast_snapshot.update(
            {
                "payload_sha256": hashlib.sha256(body).hexdigest(),
                "top_level_keys": sorted(str(key) for key in payload),
                "events_count": (
                    len(payload["events"])
                    if isinstance(payload.get("events"), list)
                    else None
                ),
                "metrics_count": (
                    len(payload["metrics"])
                    if isinstance(payload.get("metrics"), list)
                    else None
                ),
            }
        )
    health_status, health_body, health_payload = _loopback_json(
        dashboard_url, "/api/health"
    )
    health = (
        health_payload.get("health")
        if isinstance(health_payload, Mapping)
        and isinstance(health_payload.get("health"), Mapping)
        else {}
    )
    runtime = health.get("runtime") if isinstance(health, Mapping) else {}
    runtime = runtime if isinstance(runtime, Mapping) else {}
    recall = (
        health.get("recall_distillation") if isinstance(health, Mapping) else {}
    )
    recall = recall if isinstance(recall, Mapping) else {}
    live_health = {
        "http_status": health_status,
        "payload_sha256": (
            hashlib.sha256(health_body).hexdigest() if health_body else None
        ),
        "status": health.get("status") if isinstance(health, Mapping) else None,
        "runtime": {
            key: runtime.get(key)
            for key in ("commit_id", "expected_commit", "drift", "package_version")
        },
        "recall_distillation": {
            key: recall.get(key)
            for key in ("status", "worker_status", "rollout", "hold_reason", "alert")
        },
    }
    return {
        "ledgers": ledgers,
        "raw_watermark": watermark,
        "fts": _fts(
            store,
            catalog,
            root,
            watermark,
            require_checkpoint_file_state=not clone_copy,
        ),
        "state": {
            "seal_sha256": state.get("seal_sha256", ""),
            "fields": compact_state,
            "file_state": state_file_state,
        },
        "pointers": pointers,
        "fast_snapshot": fast_snapshot,
        "live_health": live_health,
    }


class RusageInfoV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_ubyte * 16),
        *[(name, ctypes.c_uint64) for name in RUSAGE_FIELDS],
    ]


def _proc_pid_rusage_v2(pid: int | None = None) -> dict[str, int | str]:
    if sys.platform != "darwin" or ctypes.sizeof(RusageInfoV2) != 160:
        raise R0Error("proc_pid_rusage unavailable")
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    value = RusageInfoV2()
    if function(int(pid if pid is not None else os.getpid()), 2, ctypes.byref(value)):
        raise R0Error("proc_pid_rusage failed")
    return {
        "rusage_uuid": bytes(value.ri_uuid).hex(),
        "resident_bytes": int(value.ri_resident_size),
        "footprint_bytes": int(value.ri_phys_footprint),
        "disk_read_bytes": int(value.ri_diskio_bytesread),
        "disk_write_bytes": int(value.ri_diskio_byteswritten),
    }


def _measure_stage(
    name: str, call: Callable[[], Any], metrics: dict[str, list[dict[str, Any]]]
) -> Any:
    if name not in STAGES:
        raise R0Error("unknown stage")
    before, started = _proc_pid_rusage_v2(), time.perf_counter_ns()
    try:
        return call()
    finally:
        finished, after = time.perf_counter_ns(), _proc_pid_rusage_v2()
        if before["rusage_uuid"] != after["rusage_uuid"] or finished < started:
            raise R0Error("measurement counter invalid")
        deltas = {}
        for key in ("disk_read_bytes", "disk_write_bytes"):
            delta = int(after[key]) - int(before[key])
            if delta < 0:
                raise R0Error("proc_pid_rusage counter decreased")
            deltas[key] = delta
        metrics.setdefault(name, []).append(
            {
                "wall_time_ns": finished - started,
                "resident_before_bytes": int(before["resident_bytes"]),
                "resident_after_bytes": int(after["resident_bytes"]),
                "footprint_before_bytes": int(before["footprint_bytes"]),
                "footprint_after_bytes": int(after["footprint_bytes"]),
                "footprint_delta_bytes": int(after["footprint_bytes"])
                - int(before["footprint_bytes"]),
                **deltas,
                "rusage_uuid": str(before["rusage_uuid"]),
            }
        )


def _require_complete_stages(metrics: Mapping[str, list[Mapping[str, Any]]]) -> None:
    if any(not metrics.get(name) for name in STAGES):
        raise R0Error("missing stages")


def _assert_local_workers(teachers: Mapping[str, Any], counterfactual: Any) -> None:
    expected = {
        "recall.distill.teacher.a",
        "recall.distill.teacher.b",
        "recall.distill.teacher.c",
    }
    if (
        set(teachers) != expected
        or any(
            getattr(worker, "local", False) is not True
            or getattr(worker, "role", "") != role
            for role, worker in teachers.items()
        )
        or getattr(counterfactual, "local", False) is not True
    ):
        raise R0Error("provider/OX attempt")


def _assert_identity_stable(
    before: Mapping[str, str], after: Mapping[str, str]
) -> None:
    if dict(before) != dict(after):
        raise R0Error("runtime identity drift")


@contextlib.contextmanager
def _env(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _stage_guards(distill: Any) -> Iterator[None]:
    workers, connect, create = (
        distill._default_workers,
        socket.socket.connect,
        socket.create_connection,
    )

    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise R0Error("provider/OX attempt")

    distill._default_workers = reject
    socket.socket.connect, socket.create_connection = reject, reject
    try:
        yield
    finally:
        distill._default_workers = workers
        socket.socket.connect, socket.create_connection = connect, create


class _Teacher:
    local = True

    def __init__(self, role: str) -> None:
        self.role = role
        self.identity = {
            "role": role,
            "provider": "local",
            "model": f"r0-{role[-1]}",
            "location": "local",
        }
        self.digest = hashlib.sha256(role.encode()).hexdigest()

    def evaluate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise R0Error("fake teacher payload invalid")
        labels = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise R0Error("fake teacher candidate invalid")
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id:
                raise R0Error("fake teacher candidate missing id")
            digest = hashlib.sha256(
                f"{self.role}\0{candidate_id}".encode()
            ).digest()
            labels.append(
                {
                    "candidate_id": candidate_id,
                    "verdict": "relevant" if digest[0] % 2 else "uncertain",
                    "reason": "r0-deterministic",
                }
            )
        return {
            "labels": labels,
            "_route_identity": dict(self.identity),
            "_model_digest": self.digest,
        }


class _Counterfactual:
    local = True

    def __init__(self) -> None:
        self.generator_identity = {
            "role": "r0-generator",
            "provider": "local",
            "model": "r0-generator",
            "location": "local",
        }
        self.judge_identity = {
            "role": "r0-judge",
            "provider": "local",
            "model": "r0-judge",
            "location": "local",
        }
        self.generator_digest = hashlib.sha256(b"r0-generator").hexdigest()
        self.judge_digest = hashlib.sha256(b"r0-judge").hexdigest()

    def compare(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "verdict": "helpful" if int(digest[0], 16) % 2 else "uncertain",
            "reason": "r0-deterministic",
            "order_agreement": True,
            "generator_model_digest": self.generator_digest,
            "judge_model_digest": self.judge_digest,
            "generator_route_identity": dict(self.generator_identity),
            "judge_route_identity": dict(self.judge_identity),
            "a0_sha256": hashlib.sha256(f"a0:{digest}".encode()).hexdigest(),
            "a1_sha256": hashlib.sha256(f"a1:{digest}".encode()).hexdigest(),
            "blind_orders": ["a0", "a1"],
        }


def _run_distillation(
    distill: Any, parity: Any, clone: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = Path(tempfile.mkdtemp(prefix=".r0-", dir=clone))
    config_path = directory / "distillation.toml"
    config_path.write_text(
        """[recall.distillation]
enabled = true
chunk_size = 1
max_input_bytes = 12000
max_candidates = 8
teacher_profile = "local-triad-v1"
teacher_max_inflight = 1
teacher_claim_limit = 1
ox_enabled = false
ox_free_only = true
""",
        encoding="utf-8",
    )
    originals = {name: getattr(distill, name) for name in STAGES}
    metrics: dict[str, list[dict[str, Any]]] = {}
    try:
        config = distill.load_distillation_config(config_path)
        if not config.enabled or config.teacher_profile != "local-triad-v1":
            raise R0Error("provider/OX attempt")
        teachers = {role: _Teacher(role) for role in distill.TEACHER_ROLES}
        counterfactual = _Counterfactual()
        _assert_local_workers(teachers, counterfactual)
        for name, original in originals.items():

            @functools.wraps(original)
            def wrapper(
                *args: Any,
                _name: str = name,
                _original: Callable[..., Any] = original,
                **kwargs: Any,
            ) -> Any:
                return _measure_stage(
                    _name, lambda: _original(*args, **kwargs), metrics
                )

            setattr(distill, name, wrapper)
        try:
            with _stage_guards(distill):
                result = distill._run_distillation_chunk_impl(
                    root=clone,
                    raw_dir=clone / "raw",
                    config_path=config_path,
                    teachers=teachers,
                    counterfactual=counterfactual,
                    structural_verifier=lambda *_args: None,
                    dry_run=False,
                    cold_start=False,
                    max_elapsed_seconds=300,
                )
                if not metrics.get("_run_teacher_batch"):
                    distill._run_distillation_chunk_impl(
                        root=clone,
                        raw_dir=clone / "raw",
                        config_path=config_path,
                        teachers=teachers,
                        counterfactual=None,
                        structural_verifier=lambda *_args: None,
                        dry_run=False,
                        cold_start=False,
                        max_elapsed_seconds=300,
                    )
        finally:
            for name, original in originals.items():
                setattr(distill, name, original)
        _require_complete_stages(metrics)
        if (
            result.get("ox_profile_contract_id")
            or result.get("ox_workset")
            or result.get("ox_profile_stopped")
        ):
            raise R0Error("provider/OX attempt")
        summary = {}
        for name in STAGES:
            rows = list(metrics[name])
            wall = [int(row["wall_time_ns"]) for row in rows]
            summary[name] = {
                "invocations": len(rows),
                "wall_time_ns_total": sum(wall),
                "wall_time_ns": parity._aggregate(wall),
                "disk_read_bytes": sum(int(row["disk_read_bytes"]) for row in rows),
                "disk_write_bytes": sum(
                    int(row["disk_write_bytes"]) for row in rows
                ),
                "footprint_max_bytes": max(
                    int(row["footprint_after_bytes"]) for row in rows
                ),
                "resident_max_bytes": max(
                    int(row["resident_after_bytes"]) for row in rows
                ),
                "invocations_detail": [dict(row) for row in rows],
            }
        keys = (
            "status",
            "processed",
            "p5_allowed",
            "teachers_available",
            "counterfactual_available",
            "candidate_snapshots",
            "labels_written",
            "counterfactuals_written",
            "run_id",
            "state_sha256",
        )
        return {key: result.get(key) for key in keys}, summary
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _capture_recall_latency(
    parity: Any, clone: Path, expected_log: Mapping[str, Any]
) -> dict[str, Any]:
    log_file = clone / "recall" / "recall-log.jsonl"
    before = _stat(log_file)
    if before is None or before != expected_log.get("file_state"):
        raise R0Error("normal Recall log missing")
    cases: list[str] = []
    latency: dict[str, list[int]] = {
        key: [] for key in ("total", "teacher", "reranker", "context")
    }
    for row in parity.eligible_rows(log_file):
        evidence = row.get("evidence_features")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        timings = row.get("stage_timings_ms")
        timings = timings if isinstance(timings, Mapping) else evidence.get(
            "stage_timings_ms"
        )
        total = row.get("latency_ms")
        if (
            not isinstance(timings, Mapping)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total < 0
            or any(
                isinstance(timings.get(key), bool)
                or not isinstance(timings.get(key), int)
                or int(timings[key]) < 0
                for key in ("teacher", "reranker", "context")
            )
        ):
            continue
        case_id = parity._case_id(row)
        if case_id in cases:
            continue
        cases.append(case_id)
        latency["total"].append(total)
        for key in ("teacher", "reranker", "context"):
            latency[key].append(int(timings[key]))
        if len(cases) == 100:
            break
    if len(cases) != 100 or any(len(values) != 100 for values in latency.values()):
        raise R0Error("normal Recall latency cohort incomplete")
    observed_log = _file_identity(log_file)
    if observed_log != dict(expected_log):
        raise R0Error("normal Recall log changed during capture")
    return {
        "schema": "chronovisor.recall-observed-latency.v1",
        "source": "recall-log-observed",
        "cohort_size": len(cases),
        "cohort_sha256": parity._sha(cases),
        "log_sha256": observed_log["sha256"],
        "log_file_state": before,
        "latency_ms": {
            key: parity._aggregate(values) for key, values in latency.items()
        },
    }


def _clone(production: Path, isolated: Path | None) -> tuple[Path, bool]:
    production = production.resolve(strict=True)
    if not production.is_dir():
        raise R0Error("production root invalid")
    if isolated is not None:
        requested = isolated.expanduser()
        if requested.is_symlink():
            raise R0Error("isolated root invalid")
        clone = requested.resolve(strict=False)
        if not clone.is_dir():
            raise R0Error("isolated root invalid")
        temporary = False
    else:
        if sys.platform != "darwin":
            raise R0Error("APFS clone requires macOS or --isolated-root")
        clone = Path(
            tempfile.mkdtemp(prefix="chronovisor-r0-", dir=tempfile.gettempdir())
        )
        temporary = True
        try:
            for relative in (
                Path("raw"),
                Path("runtime") / "recall-distillation",
            ):
                source = production / relative
                if not source.exists() or source.is_symlink():
                    raise R0Error("APFS clone source subtree missing")
                destination = clone / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["/bin/cp", "-cR", "--", str(source), str(destination.parent)],
                    check=True,
                    capture_output=True,
                )
            log_source = production / "recall" / "recall-log.jsonl"
            if log_source.exists():
                destination = clone / "recall" / "recall-log.jsonl"
                destination.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["/bin/cp", "-c", "--", str(log_source), str(destination)],
                    check=True,
                    capture_output=True,
                )
        except (OSError, subprocess.CalledProcessError) as exc:
            shutil.rmtree(clone, ignore_errors=True)
            raise R0Error("APFS clone failed") from exc
    if (
        clone == production
        or clone.is_relative_to(production)
        or production.is_relative_to(clone)
    ):
        if temporary:
            shutil.rmtree(clone, ignore_errors=True)
        raise R0Error("isolated root overlaps production")
    if any(path.is_symlink() for path in clone.rglob("*")):
        if temporary:
            shutil.rmtree(clone, ignore_errors=True)
        raise R0Error("isolated root contains symlink")
    return clone, temporary


def _load(source_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    source_path = str(source_root / "src")
    if any(
        name == "chronovisor" or name.startswith("chronovisor.")
        for name in sys.modules
    ):
        raise R0Error("chronovisor was imported before source binding")
    sys.path[:] = [entry for entry in sys.path if entry != source_path]
    sys.path.insert(0, source_path)
    spec = importlib.util.spec_from_file_location(
        "recall_parity", source_root / "scripts" / "recall_parity.py"
    )
    if spec is None or spec.loader is None:
        raise R0Error("parity script unavailable")
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)
    distill = importlib.import_module("chronovisor.recall.recall_distillation")
    store = importlib.import_module(
        "chronovisor.recall.recall_distillation_store"
    )
    catalog = importlib.import_module(
        "chronovisor.recall.recall_distillation_catalog"
    )
    raw_store = importlib.import_module("chronovisor.core.raw_store")
    for module in (distill, store, catalog, raw_store):
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(source_root / "src"):
            raise R0Error("runtime module is outside source root")
    return parity, distill, store, catalog, raw_store


def _clone_identity(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ledgers": {
            name: {
                key: row.get(key)
                for key in ("records", "head_sha256", "bytes")
            }
            for name, row in snapshot["ledgers"].items()
        },
        "raw_watermark": snapshot["raw_watermark"],
        "fts": {
            key: snapshot["fts"].get(key)
            for key in (
                "content_sha256",
                "atom_count",
                "fts_count",
                "checkpoint_seal_sha256",
            )
        },
        "state": {
            key: snapshot["state"].get(key) for key in ("seal_sha256", "fields")
        },
        "pointers": {
            kind: (
                None
                if row is None
                else {
                    key: row.get(key)
                    for key in (
                        "policy_id",
                        "pointer_seal_sha256",
                        "policy_seal_sha256",
                    )
                }
            )
            for kind, row in snapshot["pointers"].items()
        },
    }


def _runtime_comparison(
    identity: Mapping[str, str], live_health: Mapping[str, Any]
) -> dict[str, Any]:
    runtime = live_health.get("runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    commit = runtime.get("commit_id")
    expected = runtime.get("expected_commit")
    if any(
        not isinstance(value, str)
        or len(value) != 40
        or set(value) - set("0123456789abcdef")
        for value in (commit, expected)
    ):
        raise R0Error("live runtime commit identity missing")
    return {
        "checkout_commit": identity["source_commit"],
        "github_runtime_commit": commit,
        "expected_runtime_commit": expected,
        "runtime_drift": runtime.get("drift"),
        "checkout_differs_from_runtime": identity["source_commit"] != commit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dashboard-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolated-root", type=Path)
    args = parser.parse_args(argv)
    clone: Path | None = None
    temporary = False
    previous_dont_write_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        production = args.production_root.expanduser().resolve(strict=True)
        source_root = args.source_root.expanduser().resolve(strict=True)
        clone, temporary = _clone(production, args.isolated_root)
        output = args.output.expanduser().resolve(strict=False)
        if (
            clone == source_root
            or clone.is_relative_to(source_root)
            or source_root.is_relative_to(clone)
            or output in (production, clone, source_root)
            or output.is_relative_to(production)
            or production.is_relative_to(output)
            or output.is_relative_to(clone)
            or clone.is_relative_to(output)
            or output.is_relative_to(source_root)
            or source_root.is_relative_to(output)
        ):
            raise R0Error("isolated/output root overlap")
        with _env(
            {
                "CHRONOVISOR_ROOT": str(clone),
                "CHRONOVISOR_RECALL_DISTILLATION": "true",
                "CHRONOVISOR_READ_ONLY": "0",
            }
        ):
            parity, distill, store, catalog, raw_store = _load(source_root)
            identity_before = parity._runtime_identity(
                source_root, args.source_commit
            )
            production_before = _production(
                store, catalog, raw_store, production, args.dashboard_url
            )
            clone_before = _production(
                store,
                catalog,
                raw_store,
                clone,
                args.dashboard_url,
                clone_copy=True,
            )
            if _clone_identity(clone_before) != _clone_identity(production_before):
                raise R0Error("isolated clone is not point-in-time coherent")
            recall_log_before = _file_identity(
                clone / "recall" / "recall-log.jsonl"
            )
            result, stages = _run_distillation(distill, parity, clone)
            normal_recall = _capture_recall_latency(
                parity, clone, recall_log_before
            )
            identity_after = parity._runtime_identity(
                source_root, args.source_commit
            )
            _assert_identity_stable(identity_before, identity_after)
            production_after = _production(
                store, catalog, raw_store, production, args.dashboard_url
            )
            before_static = dict(production_before)
            after_static = dict(production_after)
            for key in ("fast_snapshot", "live_health"):
                before_static.pop(key, None)
                after_static.pop(key, None)
            if before_static != after_static:
                raise R0Error("production changed during measurement")
            runtime_comparison = _runtime_comparison(
                identity_after, production_after["live_health"]
            )
            payload = {
                "captured_at": datetime.now(UTC).isoformat(),
                "runtime_identity": identity_before,
                "runtime_comparison": runtime_comparison,
                "production": after_static,
                "live_api": {
                    "fast_snapshot": {
                        "before": production_before["fast_snapshot"],
                        "after": production_after["fast_snapshot"],
                    },
                    "health": {
                        "before": production_before["live_health"],
                        "after": production_after["live_health"],
                    },
                },
                "distillation_result": result,
                "stages": stages,
                "normal_recall": normal_recall,
            }
            artifact_id, artifact_path, artifact = store.write_immutable(
                output, payload, schema=SCHEMA
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
    except (R0Error, OSError, ValueError, sqlite3.Error) as exc:
        print(f"r0 harness failed: {str(exc).split(':', 1)[0]}", file=sys.stderr)
        return 2
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if temporary and clone is not None:
            shutil.rmtree(clone, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
