#!/usr/bin/env python3.14
"""Measure the production-scale Recall R2 incremental catalog contract.

The harness is deliberately clone-only.  It never writes the production root,
and the evidence contains counts, digests, and file metadata, not transcript
text.  A real run requires Darwin/APFS; a normal directory is not an accepted
substitute because copy-on-write isolation is part of the measurement contract.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

R2_SCHEMA = "chronovisor.recall-r2.v1"
FROZEN_CLONE_SCHEMA = "chronovisor.recall-r2-frozen-clone.v1"
R2_COMPLETION_KIND = "chronovisor.recall-r2-completion"
R2_COMPLETION_SUFFIX = ".completion.receipt"
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
DEFAULT_CONTEXT_BYTES = 4096
DEFAULT_NOOP_SAMPLES = 20
DEFAULT_DELTA_SAMPLES = 20
_T = TypeVar("_T")
_ACTIVE_CLONE_PREFIX: str | None = None


class R2Error(ValueError):
    """An R2 contract failed closed."""


class _CooperativeSignal(R2Error):
    """A first termination signal that must unwind clone cleanup."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = signum


class _PhaseReceipt:
    """Write bounded phase telemetry outside the protected production root."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.started_ns = time.monotonic_ns()
        self._intervals: list[tuple[int, int, str]] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, payload: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def event(self, event: str, **fields: Any) -> None:
        self._write({"event": event, **fields})

    def _record_interval(self, started_ns: int, finished_ns: int, name: str) -> None:
        self._intervals.append((started_ns, max(started_ns, finished_ns), name))

    def summary(self, finished_ns: int | None = None) -> dict[str, Any]:
        end_ns = finished_ns or time.monotonic_ns()
        ordered = sorted((start, end) for start, end, _ in self._intervals)
        union_ns = 0
        current_start: int | None = None
        current_end: int | None = None
        for start, end in ordered:
            if current_start is None or current_end is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                union_ns += current_end - current_start
                current_start, current_end = start, end
        if current_start is not None and current_end is not None:
            union_ns += current_end - current_start
        elapsed_ns = max(0, end_ns - self.started_ns)
        return {
            "started_ns": self.started_ns,
            "finished_ns": end_ns,
            "elapsed_ns": elapsed_ns,
            "phase_union_ns": min(elapsed_ns, union_ns),
            "unaccounted_ns": max(0, elapsed_ns - union_ns),
            "interval_count": len(self._intervals),
        }

    def close(self) -> dict[str, Any]:
        summary = self.summary()
        self.event("summary", **summary)
        return summary

    @contextlib.contextmanager
    def phase(self, name: str, **fields: Any) -> Iterator[None]:
        started_ns = time.monotonic_ns()
        self._write({"event": "start", "phase": name, "started_ns": started_ns, **fields})
        try:
            yield
        except BaseException as exc:
            finished_ns = time.monotonic_ns()
            self._record_interval(started_ns, finished_ns, name)
            self._write(
                {
                    "event": "error",
                    "phase": name,
                    "started_ns": started_ns,
                    "elapsed_ns": finished_ns - started_ns,
                    "error_type": type(exc).__name__,
                    **fields,
                }
            )
            raise
        else:
            finished_ns = time.monotonic_ns()
            self._record_interval(started_ns, finished_ns, name)
            self._write(
                {
                    "event": "finish",
                    "phase": name,
                    "started_ns": started_ns,
                    "elapsed_ns": finished_ns - started_ns,
                    **fields,
                }
            )

    def measure(self, name: str, call: Callable[[], _T], **fields: Any) -> _T:
        with self.phase(name, **fields):
            return call()


class _SignalGuard:
    """Turn the first SIGTERM/SIGINT into cleanup unwinding, then fail hard."""

    def __init__(self, receipt: _PhaseReceipt, *, grace_seconds: float = 30.0) -> None:
        self.receipt = receipt
        self.grace_seconds = max(0.1, float(grace_seconds))
        self._old: dict[int, Any] = {}
        self._interrupted = False
        self._closed = False
        self._timer: threading.Timer | None = None

    def _force_exit(self, signum: int) -> None:
        if self._closed:
            return
        os.kill(os.getpid(), signum)

    def _handle(self, signum: int, _frame: Any) -> None:
        if self._interrupted:
            os._exit(128 + signum)
        self._interrupted = True
        self.receipt.event("signal", signum=signum, action="cooperative-unwind")
        self._timer = threading.Timer(self.grace_seconds, self._force_exit, args=(signum,))
        self._timer.daemon = True
        self._timer.start()
        raise _CooperativeSignal(signum)

    def __enter__(self) -> _SignalGuard:
        for signum in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
            if signum is not None:
                self._old[int(signum)] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
        for signum, handler in self._old.items():
            signal.signal(signum, handler)


def _load_r0() -> Any:
    """Load the R0 helpers without making ``scripts`` a package."""

    path = Path(__file__).with_name("recall_r0_harness.py")
    spec = importlib.util.spec_from_file_location("chronovisor_r0_harness", path)
    if spec is None or spec.loader is None:
        raise R2Error("R0 helper unavailable")
    module = importlib.util.module_from_spec(spec)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return module


R0 = _load_r0()


def _require_supported_environment(path: Path) -> None:
    """Require the Darwin/APFS primitive used by the isolation contract."""

    if sys.platform != "darwin":
        raise R2Error("unsupported environment: Darwin/APFS is required")
    try:
        filesystem = R0._filesystem_type(path)
    except (OSError, ValueError) as exc:
        raise R2Error("unsupported environment: APFS volume is unavailable") from exc
    if filesystem != "apfs":
        raise R2Error("unsupported environment: production volume is not APFS")


def _tree_digest(
    root: Path, *, label: str, max_file_bytes: int | None = None
) -> dict[str, Any]:
    """Hash one regular-file tree without retaining file content."""

    if root.is_symlink() or not root.is_dir():
        raise R2Error(f"{label} root is unsafe")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise R2Error(f"{label} tree contains a symlink")
        if not path.is_file():
            continue
        try:
            identity = R0._file_identity(path)
        except Exception as exc:
            raise R2Error(f"{label} file changed during capture") from exc
        state = identity["file_state"]
        if state is None:
            raise R2Error(f"{label} file disappeared during capture")
        if max_file_bytes is not None and int(state["size_bytes"]) > max_file_bytes:
            raise R2Error(f"{label} file exceeds bounded hash limit")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "file_state": state,
                "sha256": identity["sha256"],
            }
        )
    content_rows = [
        {
            "path": row["path"],
            "size_bytes": row["file_state"]["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    content_encoded = json.dumps(
        content_rows, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "file_count": len(rows),
        "content_sha256": hashlib.sha256(content_encoded).hexdigest(),
        "metadata_sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": sum(int(row["file_state"]["size_bytes"]) for row in rows),
    }


def _raw_tree_digest(root: Path) -> dict[str, Any]:
    """Hash only Raw metadata/content; never retain the content itself."""

    return _tree_digest(root / "raw", label="Raw")


def _raw_tree_state_digest(root: Path) -> dict[str, Any]:
    """Capture Raw path/type/size/mode state without opening file bodies."""

    raw_root = root / "raw"
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise R2Error("Raw root is unsafe")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(raw_root.rglob("*")):
        if path.is_symlink():
            raise R2Error("Raw tree contains a symlink")
        if not path.is_file():
            continue
        try:
            state = path.lstat()
        except OSError as exc:
            raise R2Error("Raw file disappeared during state capture") from exc
        if not stat.S_ISREG(state.st_mode):
            raise R2Error("Raw tree contains a non-regular file")
        record = {
            "kind": "file",
            "mode": int(state.st_mode & 0o7777),
            "path": path.relative_to(raw_root).as_posix(),
            "size_bytes": int(state.st_size),
        }
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
        file_count += 1
        total_bytes += int(state.st_size)
    return {
        "file_count": file_count,
        "bytes": total_bytes,
        "state_sha256": digest.hexdigest(),
    }


def _assert_raw_state_parity(
    actual: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    if any(
        actual.get(key) != expected.get(key)
        for key in ("file_count", "bytes", "state_sha256")
    ):
        raise R2Error(f"{label} Raw file state differs before mutation")


SOURCE_HASH_MAX_FILE_BYTES = 512 * 1024 * 1024
SOURCE_GENERATED_COMPONENTS = frozenset(
    {".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__"}
)


def _source_file_state(value: os.stat_result) -> dict[str, int]:
    """Keep stable lstat identity fields while excluding access time."""

    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_mode": int(value.st_mode & 0o7777),
        "st_size": int(value.st_size),
        "st_mtime_ns": int(value.st_mtime_ns),
        "st_ctime_ns": int(value.st_ctime_ns),
    }


def _source_repo_digest(root: Path) -> dict[str, Any]:
    """Hash every tracked/untracked source file without following symlinks."""

    try:
        listed = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise R2Error("source file inventory failed") from exc
    candidates = sorted(
        os.fsdecode(value)
        for value in listed.split(b"\0")
        if value
        and not any(
            component in SOURCE_GENERATED_COMPONENTS
            for component in Path(os.fsdecode(value)).parts
        )
    )
    digest = hashlib.sha256()
    file_count = 0
    symlink_count = 0
    total_bytes = 0
    for relative in candidates:
        path = root / relative
        try:
            before = path.lstat()
        except OSError as exc:
            raise R2Error("source file disappeared during capture") from exc
        if stat.S_ISLNK(before.st_mode):
            target = os.fsdecode(os.readlink(path))
            record = {
                "kind": "symlink",
                "mode": int(before.st_mode & 0o7777),
                "path": relative,
                "target": target,
            }
            symlink_count += 1
        elif stat.S_ISREG(before.st_mode):
            size = int(before.st_size)
            if size > SOURCE_HASH_MAX_FILE_BYTES:
                raise R2Error(f"source file exceeds bounded hash limit: {relative}")
            content = hashlib.sha256()
            try:
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        content.update(chunk)
            except OSError as exc:
                raise R2Error("source file read failed") from exc
            try:
                after = path.lstat()
            except OSError as exc:
                raise R2Error("source file disappeared during capture") from exc
            before_state = _source_file_state(before)
            if _source_file_state(after) != before_state:
                raise R2Error("source file changed during capture")
            record = {
                "kind": "file",
                "path": relative,
                "state": before_state,
                "sha256": content.hexdigest(),
            }
            file_count += 1
            total_bytes += size
        else:
            raise R2Error(f"source path is not a regular file or symlink: {relative}")
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return {
        "file_count": file_count,
        "symlink_count": symlink_count,
        "bytes": total_bytes,
        "content_sha256": digest.hexdigest(),
    }


def _source_tree_digest(root: Path) -> dict[str, Any]:
    """Hash the complete source checkout and status to prove clone-only runs."""

    try:
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise R2Error("source status snapshot failed") from exc
    return {
        "repo": _source_repo_digest(root),
        "trees": {
            "src": _tree_digest(
                root / "src", label="source", max_file_bytes=SOURCE_HASH_MAX_FILE_BYTES
            ),
            "scripts": _tree_digest(
                root / "scripts",
                label="source scripts",
                max_file_bytes=SOURCE_HASH_MAX_FILE_BYTES,
            ),
        },
        "git_status_count": len(status.splitlines()),
        "git_status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _assert_raw_digest_parity(
    actual: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    if any(
        actual.get(key) != expected.get(key)
        for key in ("file_count", "bytes", "content_sha256")
    ):
        raise R2Error(f"{label} Raw tree differs before mutation")


def _has_symlink_component(path: Path) -> bool:
    current = Path(os.path.abspath(path.expanduser()))
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


def _cli_path_record(path: Path) -> dict[str, str]:
    """Keep the user spelling while checking its lexical and resolved paths."""

    expanded = path.expanduser()
    lexical = Path(os.path.abspath(expanded))
    if _has_symlink_component(lexical):
        raise R2Error(f"CLI path contains a symlink: {path}")
    return {
        "original": str(path),
        "expanded": str(expanded),
        "resolved": str(expanded.resolve(strict=False)),
    }


def _assert_artifact_id(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise R2Error(f"{label} artifact ID is not 64 lowercase hex")
    return value


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _stable_file_digest(path: Path, *, label: str) -> tuple[os.stat_result, str]:
    """Hash a regular file while rejecting lstat/size/content TOCTOU changes."""

    if _has_symlink_component(path):
        raise R2Error(f"{label} path contains a symlink")
    try:
        before = path.lstat()
    except OSError as exc:
        raise R2Error(f"{label} path stat failed") from exc
    if not stat.S_ISREG(before.st_mode):
        raise R2Error(f"{label} path is not a regular file")
    try:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after = path.lstat()
    except OSError as exc:
        raise R2Error(f"{label} path read failed") from exc
    if _has_symlink_component(path) or not stat.S_ISREG(after.st_mode):
        raise R2Error(f"{label} path changed to a non-regular file")
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise R2Error(f"{label} path changed during readback")
    return after, digest


def _read_stable_sealed(
    store: Any, path: Path, *, schema: str, label: str
) -> tuple[dict[str, Any], str]:
    before_state, before_digest = _stable_file_digest(path, label=label)
    try:
        artifact = store.read_sealed(path, schema=schema)
    except Exception as exc:
        raise R2Error(f"{label} sealed read failed") from exc
    after_state, after_digest = _stable_file_digest(path, label=label)
    if (
        before_digest != after_digest
        or before_state.st_dev != after_state.st_dev
        or before_state.st_ino != after_state.st_ino
        or before_state.st_size != after_state.st_size
    ):
        raise R2Error(f"{label} changed during sealed readback")
    return dict(artifact), after_digest


def _read_content_addressed_artifact(
    store: Any, path: Path, *, schema: str, label: str
) -> dict[str, Any]:
    """Read back an immutable artifact and independently bind ID, path, seal, schema."""

    if path.suffix != ".json" or path.stem != path.name[:-5]:
        raise R2Error(f"{label} artifact filename is malformed")
    expected_id = _assert_artifact_id(path.stem, label=label)
    artifact, first_digest = _read_stable_sealed(
        store, path, schema=schema, label=label
    )
    persisted, second_digest = _read_stable_sealed(
        store, path, schema=schema, label=label
    )
    if first_digest != second_digest:
        raise R2Error(f"{label} artifact changed during readback")
    if persisted != artifact:
        raise R2Error(f"{label} artifact changed during readback")
    actual_id = _assert_artifact_id(artifact.get("artifact_id"), label=label)
    if actual_id != expected_id:
        raise R2Error(f"{label} artifact ID does not match filename")
    unsigned = {
        key: value for key, value in artifact.items() if key not in {"artifact_id", "seal_sha256"}
    }
    if _canonical_json_sha256(unsigned) != actual_id:
        raise R2Error(f"{label} artifact ID is not content addressed")
    seal = artifact.get("seal_sha256")
    _assert_artifact_id(seal, label=f"{label} seal")
    unsigned_with_id = {key: value for key, value in artifact.items() if key != "seal_sha256"}
    if _canonical_json_sha256(unsigned_with_id) != seal:
        raise R2Error(f"{label} artifact seal is invalid")
    if artifact.get("schema") != schema:
        raise R2Error(f"{label} artifact schema mismatch")
    return dict(artifact)


def _completion_bytes(store: Any, payload: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    unsigned = {
        "schema": store.DISTILLATION_SCHEMA,
        "namespace": "recall-distillation",
        **payload,
    }
    artifact = {**unsigned, "seal_sha256": _canonical_json_sha256(unsigned)}
    encoded = (
        json.dumps(
            artifact,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return encoded, artifact


def _write_temp_receipt(path: Path, encoded: bytes) -> tuple[Path, int]:
    started_ns = time.monotonic_ns()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary, time.monotonic_ns() - started_ns
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _publish_completion_receipt(
    store: Any, path: Path, encoded: bytes, *, schema: str
) -> tuple[dict[str, Any], int]:
    """Publish one final receipt via temp+fsync+replace, then stable-read it."""

    started_ns = time.monotonic_ns()
    if path.exists() or path.is_symlink():
        raise R2Error("R2 completion receipt already exists")
    temporary, _write_elapsed_ns = _write_temp_receipt(path, encoded)
    try:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        receipt, _digest = _read_stable_sealed(
            store, path, schema=schema, label="R2 completion receipt"
        )
        return receipt, time.monotonic_ns() - started_ns
    finally:
        temporary.unlink(missing_ok=True)


def _write_completion_receipt(
    store: Any,
    output: Path,
    artifact: Mapping[str, Any],
    artifact_path: Path,
    phase_timing: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a post-artifact completion receipt without a circular artifact hash."""

    artifact_id = _assert_artifact_id(artifact.get("artifact_id"), label="R2 evidence")
    artifact_seal = _assert_artifact_id(
        artifact.get("seal_sha256"), label="R2 evidence seal"
    )
    path = output / f"{artifact_id}{R2_COMPLETION_SUFFIX}"
    _assert_symlink_free_tree(output, label="R2 output")
    if _has_symlink_component(path) or path.exists():
        raise R2Error("R2 completion receipt path is unsafe")
    started_ns = time.monotonic_ns()
    pre_completion_timing = dict(phase_timing)
    pre_completion_timing["scope"] = "formal-run-pre-completion"
    pre_completion_timing["artifact_seal_readback_included"] = True
    # Calibrate the final publication/readback envelope without leaving a
    # second valid completion receipt in the output directory.  The actual
    # receipt is published exactly once below via ``os.replace``.
    probe_encoded = b"chronovisor-r2-completion-publication-probe\n"
    probe_path: Path | None = None
    try:
        probe_path, _probe_write_ns = _write_temp_receipt(path, probe_encoded)
        _stable_file_digest(probe_path, label="R2 completion publication probe")
        probe_finished_ns = time.monotonic_ns()
        probe_elapsed_ns = probe_finished_ns - started_ns
    finally:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)
    formal_wall_ns = int(pre_completion_timing.get("elapsed_ns", 0)) + probe_elapsed_ns
    formal_phase_union_ns = int(pre_completion_timing.get("phase_union_ns", 0))
    final_timing = {
        **pre_completion_timing,
        "completion_publication_readback_ns": probe_elapsed_ns,
        "formal_wall_ns": formal_wall_ns,
        "formal_phase_union_ns": formal_phase_union_ns,
        "formal_unaccounted_ns": max(0, formal_wall_ns - formal_phase_union_ns),
        "formal_wall_scope": "run-start-through-completion-publication-readback",
    }
    payload = {
        "kind": R2_COMPLETION_KIND,
        "r2_schema": R2_SCHEMA,
        "r2_artifact_id": artifact_id,
        "r2_artifact_path": str(artifact_path),
        "r2_artifact_seal_sha256": artifact_seal,
        "phase_timing": final_timing,
        "completion_started_ns": started_ns,
        "completion_publication_readback_ns": probe_elapsed_ns,
    }
    encoded, _artifact = _completion_bytes(store, payload)
    persisted, publication_elapsed_ns = _publish_completion_receipt(
        store,
        path,
        encoded,
        schema=store.DISTILLATION_SCHEMA,
    )
    finished_ns = time.monotonic_ns()
    if (
        persisted.get("kind") != R2_COMPLETION_KIND
        or persisted.get("r2_schema") != R2_SCHEMA
        or persisted.get("r2_artifact_id") != artifact_id
        or persisted.get("r2_artifact_path") != str(artifact_path)
        or persisted.get("r2_artifact_seal_sha256") != artifact_seal
        or persisted.get("completion_started_ns") != started_ns
        or persisted.get("completion_publication_readback_ns") != probe_elapsed_ns
        or persisted.get("phase_timing", {}).get("formal_wall_ns") != formal_wall_ns
    ):
        raise R2Error("R2 completion receipt binding mismatch")
    return {
        "schema": persisted["schema"],
        "path": str(path),
        "artifact_id": artifact_id,
        "artifact_seal_sha256": artifact_seal,
        "completion_elapsed_ns": publication_elapsed_ns,
        "completion_finished_ns": finished_ns,
        "phase_timing": dict(persisted["phase_timing"]),
    }


def _assert_symlink_free_tree(root: Path, *, label: str) -> None:
    """Reject recursive symlink entries, including ignored clone subtrees."""

    if _has_symlink_component(root) or not root.is_dir():
        raise R2Error(f"{label} root is unsafe")
    for base, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            if (Path(base) / name).is_symlink():
                raise R2Error(f"{label} tree contains a symlink")


def _assert_root_matrix(
    production: Path,
    source_root: Path,
    output: Path,
    clones: Iterable[Path] = (),
    other_paths: Mapping[str, Path] | None = None,
) -> None:
    """Reject every protected-root overlap and symlink entry point."""

    paths = {
        "production": production,
        "source_root": source_root,
        "output": output,
        **{f"clone[{index}]": path for index, path in enumerate(clones)},
    }
    if other_paths:
        paths.update(other_paths)
    for name, path in paths.items():
        if _has_symlink_component(path):
            raise R2Error(f"{name} path contains a symlink")
    protected = tuple(paths.items())
    for index, (left_name, left) in enumerate(protected):
        for right_name, right in protected[index + 1 :]:
            if _path_overlap(left, right):
                raise R2Error(f"{left_name}/{right_name} paths overlap")
    external_names = set(other_paths or {})
    for name, path in paths.items():
        if name == "output" or name.startswith("clone[") or name in external_names:
            continue
        if not path.is_dir():
            raise R2Error(f"{name} root is not a directory")


def _canonical_digest(rows: Iterator[tuple[str, tuple[Any, ...]]]) -> str:
    digest = hashlib.sha256()
    for table, row in rows:
        digest.update(table.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _catalog_snapshot(catalog: Any, root: Path) -> dict[str, Any]:
    path = catalog.catalog_path(root)
    state = R0._stat(path)
    if state is None:
        return {"exists": False, "rows": {}, "columns": [], "digest": None}
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            columns = {
                table: [
                    str(row[1])
                    for row in connection.execute(f"PRAGMA table_info({table})")
                ]
                for table in ("raw_units", "events", "rallies", "metadata")
                if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
            }
            counts = {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in columns
            }
            duplicate_keys = {
                "raw_units": "raw_id",
                "events": "raw_id,event_index",
                "rallies": "rally_id",
            }
            duplicates = {
                table: int(
                    connection.execute(
                        "SELECT COALESCE(SUM(n-1),0) FROM "
                        f"(SELECT COUNT(*) AS n FROM {table} GROUP BY {keys})",
                    ).fetchone()[0]
                )
                for table, keys in duplicate_keys.items()
                if table in columns
            }
            order_by = {
                "raw_units": "raw_id",
                "events": "raw_id,event_index",
                "rallies": "rally_id",
                "metadata": "key",
            }
            table_rows: Iterator[tuple[str, tuple[Any, ...]]] = (
                (table, tuple(row))
                for table in columns
                for row in connection.execute(
                    f"SELECT * FROM {table} ORDER BY {order_by[table]}"
                )
            )
            digest = _canonical_digest(table_rows)
    except sqlite3.DatabaseError as exc:
        raise R2Error("historical catalog snapshot failed") from exc
    checkpoint = catalog._catalog_checkpoint_path(root)
    return {
        "exists": True,
        "file_state": state,
        "rows": counts,
        "duplicates": duplicates,
        "columns": columns,
        "digest": digest,
        "checkpoint_file_state": R0._stat(checkpoint),
    }


def _index_snapshot(
    catalog: Any, store: Any, raw_store_module: Any, raw_dir: Path, root: Path
) -> dict[str, Any]:
    path = catalog.historical_index_path(root)
    state = R0._stat(path)
    if state is None:
        return {"exists": False, "rows": 0, "digest": None, "duplicates": 0}
    try:
        watermark = raw_store_module.committed_raw_watermark(raw_dir)
        checkpoint = R0._fts(
            store, catalog, root, watermark, require_checkpoint_file_state=False
        )
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            duplicates = int(
                connection.execute(
                    "SELECT COALESCE(SUM(n-1),0) FROM "
                    "(SELECT COUNT(*) AS n FROM atoms GROUP BY atom_id)"
                ).fetchone()[0]
            )
    except Exception as exc:
        raise R2Error("historical index snapshot failed") from exc
    return {
        "exists": True,
        "file_state": state,
        "rows": int(checkpoint["atom_count"]),
        "digest": str(checkpoint["content_sha256"]),
        "duplicates": duplicates,
        "checkpoint_file_state": R0._stat(catalog._index_checkpoint_path(path)),
    }


def _index_sqlite_snapshot(catalog: Any, root: Path) -> dict[str, Any]:
    """Read FTS identity without consulting Raw or its committed watermark."""

    path = catalog.historical_index_path(root)
    state = R0._stat(path)
    if state is None:
        return {"exists": False, "rows": 0, "digest": None, "duplicates": 0}
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            rows = int(connection.execute("SELECT COUNT(*) FROM atoms").fetchone()[0])
            duplicates = int(
                connection.execute(
                    "SELECT COALESCE(SUM(n-1),0) FROM "
                    "(SELECT COUNT(*) AS n FROM atoms GROUP BY atom_id)"
                ).fetchone()[0]
            )
    except sqlite3.DatabaseError as exc:
        raise R2Error("historical index SQLite snapshot failed") from exc
    return {
        "exists": True,
        "file_state": state,
        "rows": rows,
        "digest": metadata.get("content_sha256"),
        "duplicates": duplicates,
    }


def _bounded_chain(
    store: Any, path: Path, *, require_checkpoint_file_state: bool
) -> dict[str, Any]:
    """Read a ledger checkpoint without opening a potentially huge JSONL body."""

    before = R0._stat(path)
    if before is None:
        raise R2Error(f"production ledger missing: {path.name}")
    try:
        checkpoint = store.read_sealed(
            store._chain_checkpoint_path(path), schema=store.DISTILLATION_SCHEMA
        )
    except Exception as exc:
        raise R2Error(f"production ledger checkpoint invalid: {path.name}") from exc
    after = R0._stat(path)
    records = checkpoint.get("records")
    head = checkpoint.get("head_sha256")
    checkpoint_state = checkpoint.get("file_state")
    if (
        checkpoint.get("kind") != "ledger-chain-checkpoint"
        or checkpoint.get("ledger_name") != path.name
        or not isinstance(records, int)
        or isinstance(records, bool)
        or records < 0
        or not isinstance(head, str)
        or (head and (len(head) != 64 or set(head) - set("0123456789abcdef")))
        or (records == 0) != (head == "")
        or not isinstance(checkpoint_state, Mapping)
        or checkpoint_state.get("size_bytes") != before["size_bytes"]
        or (require_checkpoint_file_state and checkpoint_state != before)
        or after != before
    ):
        raise R2Error(f"production ledger checkpoint stale: {path.name}")
    return {
        "records": int(records),
        "head_sha256": head,
        "bytes": int(before["size_bytes"]),
        "file_state": before,
    }


def _bounded_production(
    store: Any,
    catalog: Any,
    raw_store: Any,
    root: Path,
    dashboard_url: str,
    *,
    clone_copy: bool = False,
) -> dict[str, Any]:
    """Capture production identity without replaying multi-gigabyte ledgers."""

    directory = store.distillation_dir(root)
    ledgers = {
        name: _bounded_chain(
            store,
            directory / name,
            require_checkpoint_file_state=not clone_copy,
        )
        for name in R0.LEDGERS
    }
    try:
        watermark = raw_store.committed_raw_watermark(root / "raw")
    except Exception as exc:
        raise R2Error("committed Raw watermark invalid") from exc
    if (
        not isinstance(watermark, str)
        or len(watermark) != 64
        or set(watermark) - set("0123456789abcdef")
    ):
        raise R2Error("committed Raw watermark malformed")

    state_path = directory / store.STATE_FILE
    state_file_state = R0._stat(state_path)
    try:
        state = store.read_sealed(state_path, schema=store.DISTILLATION_SCHEMA)
    except Exception as exc:
        raise R2Error("distillation state invalid") from exc
    compact_state = {
        key: state.get(key)
        for key in R0.STATE_KEYS
        if isinstance(state.get(key), (str, int, bool)) or state.get(key) is None
    }

    pointers: dict[str, Any] = {}
    for kind, filename in store.POINTER_FILES.items():
        pointer_path = directory / filename
        if R0._stat(pointer_path) is None:
            pointers[kind] = None
            continue
        try:
            pointer = store.read_sealed(pointer_path, schema=store.DISTILLATION_SCHEMA)
        except Exception as exc:
            raise R2Error("policy pointer invalid") from exc
        policy_id = pointer.get("policy_id")
        if (
            pointer.get("kind") != f"{kind}-policy-pointer"
            or not isinstance(policy_id, str)
            or len(policy_id) != 64
            or set(policy_id) - set("0123456789abcdef")
        ):
            raise R2Error("policy pointer identity invalid")
        policy_path = directory / "policies" / f"{policy_id}.json"
        try:
            policy = store.read_sealed(policy_path, schema=R0.POLICY_SCHEMA)
        except Exception as exc:
            raise R2Error("policy artifact invalid") from exc
        if policy.get("artifact_id") != policy_id:
            raise R2Error("policy artifact identity mismatch")
        pointers[kind] = {
            "policy_id": policy_id,
            "pointer_seal_sha256": pointer.get("seal_sha256", ""),
            "policy_seal_sha256": policy.get("seal_sha256", ""),
            "pointer_file_state": R0._stat(pointer_path),
            "policy_file_state": R0._stat(policy_path),
        }

    status, body, payload = R0._loopback_json(dashboard_url, "/api/fast-snapshot")
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
    health_status, health_body, health_payload = R0._loopback_json(
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
    recall = health.get("recall_distillation") if isinstance(health, Mapping) else {}
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
    fts = R0._fts(
        store,
        catalog,
        root,
        watermark,
        require_checkpoint_file_state=not clone_copy,
    )
    if clone_copy:
        fts_path = catalog.historical_index_path(root)
        try:
            checkpoint = store.read_sealed(
                catalog._index_checkpoint_path(fts_path),
                schema=store.DISTILLATION_SCHEMA,
            )
        except Exception as exc:
            raise R2Error("historical FTS checkpoint invalid") from exc
        checkpoint_state = checkpoint.get("file_state")
        if not isinstance(checkpoint_state, Mapping) or checkpoint_state.get(
            "size_bytes"
        ) != (fts.get("file_state") or {}).get("size_bytes"):
            raise R2Error("historical FTS checkpoint size differs")
    return {
        "ledgers": ledgers,
        "raw_watermark": watermark,
        "fts": fts,
        "state": {
            "seal_sha256": state.get("seal_sha256", ""),
            "fields": compact_state,
            "file_state": state_file_state,
        },
        "pointers": pointers,
        "fast_snapshot": fast_snapshot,
        "live_health": live_health,
    }


def _raw_inventory_snapshot(
    catalog: Any, raw_store_module: Any, raw_dir: Path, root: Path
) -> dict[str, Any]:
    """Require catalog Raw IDs/statuses to equal the committed Raw inventory."""

    try:
        raw_store = raw_store_module.RawStore(raw_dir, mode="v2")
        units = tuple(raw_store.iter_segment_units())
        expected_status = {
            str(unit.raw_id): str(catalog._read_unit_events(raw_store, unit)[0])
            for unit in units
        }
        with sqlite3.connect(
            f"file:{catalog.catalog_path(root)}?mode=ro", uri=True
        ) as connection:
            rows = {
                str(row[0]): str(row[1])
                for row in connection.execute("SELECT raw_id,status FROM raw_units")
            }
    except Exception as exc:
        raise R2Error("Raw inventory/status snapshot failed") from exc
    if set(rows) != set(expected_status):
        raise R2Error("catalog Raw inventory set differs from committed Raw")
    if rows != expected_status:
        raise R2Error("catalog Raw inventory status differs from committed Raw")
    status_counts: dict[str, int] = {}
    for status in rows.values():
        status_counts[status] = status_counts.get(status, 0) + 1
    encoded_ids = json.dumps(sorted(rows), separators=(",", ":")).encode()
    return {
        "count": len(rows),
        "ids_sha256": hashlib.sha256(encoded_ids).hexdigest(),
        "status_counts": dict(sorted(status_counts.items())),
    }


def _raw_inventory_id_snapshot(
    catalog: Any,
    raw_store_module: Any,
    raw_dir: Path,
    root: Path,
    *,
    expected_new_raw_id: str | None = None,
    expected_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate Raw/catalog ID parity without reading committed Raw bodies."""

    try:
        raw_ids = {
            str(unit.raw_id)
            for unit in raw_store_module.RawStore(
                raw_dir, mode="v2"
            ).iter_segment_units()
        }
        with sqlite3.connect(
            f"file:{catalog.catalog_path(root)}?mode=ro", uri=True
        ) as connection:
            rows = {
                str(row[0]): str(row[1])
                for row in connection.execute("SELECT raw_id,status FROM raw_units")
            }
    except Exception as exc:
        raise R2Error("Raw inventory ID validation failed") from exc
    if raw_ids != set(rows):
        raise R2Error("Raw inventory ID set differs from catalog")
    if expected_new_raw_id is not None:
        if expected_new_raw_id not in raw_ids:
            raise R2Error("delta Raw ID is absent from catalog inventory")
        if rows.get(expected_new_raw_id) != "indexed":
            raise R2Error("delta Raw ID status is not indexed")
    if expected_inventory is not None:
        expected_ids = set(expected_inventory.get("ids", ()))
        if expected_new_raw_id is None:
            if raw_ids != expected_ids:
                raise R2Error("Raw inventory IDs changed during clone-only phase")
            expected_rows = dict(expected_inventory.get("statuses", {}))
            if rows != expected_rows:
                raise R2Error("Raw inventory statuses changed during clone-only phase")
        else:
            if raw_ids - {expected_new_raw_id} != expected_ids:
                raise R2Error("delta Raw inventory changed an existing ID")
            expected_rows = dict(expected_inventory.get("statuses", {}))
            observed_rows = {key: value for key, value in rows.items() if key != expected_new_raw_id}
            if observed_rows != expected_rows:
                raise R2Error("delta Raw inventory changed an existing status")
    status_counts: dict[str, int] = {}
    for status in rows.values():
        status_counts[status] = status_counts.get(status, 0) + 1
    encoded_ids = json.dumps(sorted(raw_ids), separators=(",", ":")).encode()
    return {
        "count": len(raw_ids),
        "ids_sha256": hashlib.sha256(encoded_ids).hexdigest(),
        "status_counts": dict(sorted(status_counts.items())),
    }


def _raw_inventory_baseline(
    catalog: Any, raw_store_module: Any, raw_dir: Path, root: Path
) -> dict[str, Any]:
    """Capture Raw IDs/statuses once; later clone checks must stay metadata-only."""

    snapshot = _raw_inventory_snapshot(catalog, raw_store_module, raw_dir, root)
    try:
        raw_ids = {
            str(unit.raw_id)
            for unit in raw_store_module.RawStore(raw_dir, mode="v2").iter_segment_units()
        }
        with sqlite3.connect(
            f"file:{catalog.catalog_path(root)}?mode=ro", uri=True
        ) as connection:
            statuses = {
                str(row[0]): str(row[1])
                for row in connection.execute("SELECT raw_id,status FROM raw_units")
            }
    except Exception as exc:
        raise R2Error("Raw inventory baseline capture failed") from exc
    if raw_ids != set(statuses):
        raise R2Error("Raw inventory baseline IDs differ from catalog")
    if len(raw_ids) != int(snapshot["count"]):
        raise R2Error("Raw inventory baseline count differs from snapshot")
    return {"ids": frozenset(raw_ids), "statuses": statuses, **snapshot}


def _catalog_integrity_snapshot(catalog: Any, root: Path) -> dict[str, Any]:
    """Check counts/duplicates without hashing 200k historical rows per clone."""

    path = catalog.catalog_path(root)
    state = R0._stat(path)
    if state is None:
        return {"exists": False, "rows": {}, "duplicates": {}, "columns": [], "digest": None}
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            columns = {
                table: [str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")]
                for table in ("raw_units", "events", "rallies", "metadata")
                if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone()
            }
            counts = {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in columns
            }
            duplicate_keys = {
                "raw_units": "raw_id",
                "events": "raw_id,event_index",
                "rallies": "rally_id",
            }
            duplicates = {
                table: int(
                    connection.execute(
                        "SELECT COALESCE(SUM(n-1),0) FROM "
                        f"(SELECT COUNT(*) AS n FROM {table} GROUP BY {keys})",
                    ).fetchone()[0]
                )
                for table, keys in duplicate_keys.items()
                if table in columns
            }
    except sqlite3.DatabaseError as exc:
        raise R2Error("historical catalog integrity snapshot failed") from exc
    return {
        "exists": True,
        "file_state": state,
        "rows": counts,
        "duplicates": duplicates,
        "columns": columns,
        "digest": None,
    }


def _derived_snapshot(
    catalog: Any,
    store: Any,
    raw_store_module: Any,
    raw_dir: Path,
    root: Path,
    *,
    include_inventory: bool = True,
    inventory_baseline: Mapping[str, Any] | None = None,
    expected_new_raw_id: str | None = None,
) -> dict[str, Any]:
    snapshot = {
        "catalog": _catalog_snapshot(catalog, root),
        "fts": _index_snapshot(catalog, store, raw_store_module, raw_dir, root),
    }
    if include_inventory:
        if inventory_baseline is None:
            snapshot["inventory"] = _raw_inventory_snapshot(
                catalog, raw_store_module, raw_dir, root
            )
        else:
            snapshot["inventory"] = _raw_inventory_id_snapshot(
                catalog,
                raw_store_module,
                raw_dir,
                root,
                expected_new_raw_id=expected_new_raw_id,
                expected_inventory=inventory_baseline,
            )
    _assert_no_duplicates(snapshot["catalog"])
    return snapshot


def _assert_repair_parity(
    actual: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    """Compare content identities while allowing repair-specific file states."""

    for section, keys in (
        ("catalog", ("exists", "rows", "duplicates", "columns", "digest")),
        ("fts", ("exists", "rows", "digest", "duplicates")),
        ("inventory", ("count", "ids_sha256", "status_counts")),
    ):
        actual_section = actual.get(section)
        expected_section = expected.get(section)
        if not isinstance(actual_section, Mapping) or not isinstance(
            expected_section, Mapping
        ):
            raise R2Error(f"{label} {section} snapshot is missing")
        if any(actual_section.get(key) != expected_section.get(key) for key in keys):
            raise R2Error(f"{label} {section} parity differs from clean reference")


def _raw_units(raw_store: Any, raw_dir: Path) -> tuple[Any, ...]:
    try:
        return tuple(raw_store.RawStore(raw_dir, mode="v2").iter_segment_units())
    except Exception as exc:
        raise R2Error("Raw inventory is invalid") from exc


@dataclass
class ReadCounters:
    """Separate logical unit reads from physical segment range overlap."""

    old_ids: frozenset[str]
    logical_reads: int = 0
    logical_bytes: int = 0
    logical_old_reads: int = 0
    logical_old_bytes: int = 0
    logical_new_reads: int = 0
    logical_new_bytes: int = 0
    physical_reads: int = 0
    physical_bytes: int = 0
    physical_old_bytes: int = 0
    full_raw_scans: int = 0
    full_event_scans: int = 0
    full_rally_scans: int = 0
    full_session_scans: int = 0
    full_fts_scans: int = 0
    assistant_scans: int = 0
    fts_delta_cursor_calls: int = 0
    fts_scan_statements: int = 0
    full_fts_rebuilds: int = 0
    logical_old_id_sha256: list[str] = field(default_factory=list)
    range_overlaps: list[dict[str, Any]] = field(default_factory=list)
    old_ranges: dict[Path, tuple[tuple[int, int], ...]] = field(default_factory=dict)

    def record_logical(self, raw_id: str, length: int) -> None:
        self.logical_reads += 1
        self.logical_bytes += length
        if raw_id in self.old_ids:
            self.logical_old_reads += 1
            self.logical_old_bytes += length
            digest = hashlib.sha256(raw_id.encode()).hexdigest()
            if (
                len(self.logical_old_id_sha256) < 8
                and digest not in self.logical_old_id_sha256
            ):
                self.logical_old_id_sha256.append(digest)
        else:
            self.logical_new_reads += 1
            self.logical_new_bytes += length

    def record_range(self, path: Path, offset: int, length: int) -> None:
        self.physical_reads += 1
        self.physical_bytes += length
        overlaps = 0
        for start, end in self.old_ranges.get(path.resolve(strict=False), ()):
            overlaps += max(0, min(offset + length, end) - max(offset, start))
        self.physical_old_bytes += overlaps
        if overlaps and len(self.range_overlaps) < 8:
            self.range_overlaps.append(
                {
                    "path": path.name,
                    "offset": offset,
                    "length": length,
                    "old_overlap_bytes": overlaps,
                }
            )


def _old_range_map(units: Iterator[Any]) -> dict[Path, tuple[tuple[int, int], ...]]:
    grouped: dict[Path, list[tuple[int, int]]] = {}
    for unit in units:
        if unit.path is None:
            continue
        grouped.setdefault(Path(unit.path).resolve(strict=False), []).append(
            (int(unit.offset), int(unit.offset) + int(unit.length))
        )
    return {path: tuple(rows) for path, rows in grouped.items()}


@contextlib.contextmanager
def _instrument(
    catalog: Any,
    distill: Any,
    raw_store_module: Any,
    store: Any,
    counters: ReadCounters,
) -> Iterator[None]:
    """Instrument only read/scan boundaries; restore every monkeypatch."""

    originals: list[tuple[Any, str, Any]] = []

    def patch(target: Any, name: str, replacement: Any) -> None:
        if not hasattr(target, name):
            return
        originals.append((target, name, getattr(target, name)))
        setattr(target, name, replacement)

    original_read = raw_store_module.RawStore.read_bytes

    def read_bytes(self: Any, raw: Any) -> bytes:
        value = original_read(self, raw)
        raw_id = str(getattr(raw, "raw_id", raw))
        counters.record_logical(raw_id, len(value))
        return value

    patch(raw_store_module.RawStore, "read_bytes", read_bytes)
    original_iter = raw_store_module.RawStore.iter_segment_bytes

    def iter_segment_bytes(
        self: Any, raw_ids: Any = None
    ) -> Iterator[tuple[Any, bytes]]:
        if raw_ids is None:
            counters.full_raw_scans += 1
        for unit, value in original_iter(self, raw_ids):
            counters.record_logical(str(unit.raw_id), len(value))
            yield unit, value

    patch(raw_store_module.RawStore, "iter_segment_bytes", iter_segment_bytes)
    original_open_range = raw_store_module.read_open_range

    def open_range_read(path: Path, offset: int, length: int) -> bytes:
        counters.record_range(path, int(offset), int(length))
        return original_open_range(path, offset, length)

    patch(raw_store_module, "read_open_range", open_range_read)
    original_sealed_range = raw_store_module.read_sealed_range

    def sealed_range_read(path: Path, offset: int, length: int) -> bytes:
        logical_end = int(offset) + int(length)
        counters.record_range(path, 0, logical_end)
        return original_sealed_range(path, offset, length)

    patch(raw_store_module, "read_sealed_range", sealed_range_read)

    for name in ("_events", "extract_rallies"):
        if not hasattr(distill, name):
            continue
        original = getattr(distill, name)

        def full_scan(*args: Any, _name=name, _f=original, **kwargs: Any) -> Any:
            if _name == "_events":
                counters.full_event_scans += 1
            else:
                counters.full_rally_scans += 1
            return _f(*args, **kwargs)

        patch(distill, name, full_scan)
    for name, attr in (
        ("_session_events", "full_session_scans"),
        ("_index_atoms", "full_fts_scans"),
        ("_catalog_assistant_atoms", "assistant_scans"),
    ):
        if not hasattr(catalog, name):
            continue
        original = getattr(catalog, name)

        def scan(*args: Any, _attr=attr, _name=name, _f=original, **kwargs: Any) -> Any:
            if (
                _name == "_catalog_assistant_atoms"
                and kwargs.get("after_rowid") is not None
            ):
                counters.fts_delta_cursor_calls += 1
            else:
                setattr(counters, _attr, getattr(counters, _attr) + 1)
            return _f(*args, **kwargs)

        patch(catalog, name, scan)
    if hasattr(store, "create_historical_index"):
        original_create = store.create_historical_index

        def create_index(*args: Any, **kwargs: Any) -> Any:
            counters.full_fts_rebuilds += 1
            return original_create(*args, **kwargs)

        patch(store, "create_historical_index", create_index)

    original_connect = catalog.sqlite3.connect

    class CountingCursor(sqlite3.Cursor):
        def execute(self, statement: str, parameters: Any = ()) -> Any:
            lowered = str(statement).lower()
            if "select" in lowered and (
                "from atoms" in lowered or "from atoms_fts" in lowered
            ):
                counters.fts_scan_statements += 1
            return super().execute(statement, parameters)

    class CountingConnection(sqlite3.Connection):
        def cursor(self, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
            kwargs["factory"] = CountingCursor
            return super().cursor(*args, **kwargs)

        def execute(self, statement: str, parameters: Any = ()) -> Any:
            lowered = str(statement).lower()
            if "select" in lowered and (
                "from atoms" in lowered or "from atoms_fts" in lowered
            ):
                counters.fts_scan_statements += 1
            return super().execute(statement, parameters)

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        kwargs["factory"] = CountingConnection
        return original_connect(*args, **kwargs)

    patch(catalog.sqlite3, "connect", connect)
    try:
        yield
    finally:
        for target, name, original in reversed(originals):
            setattr(target, name, original)


@dataclass
class StageClock:
    name: str
    counters: ReadCounters
    started_ns: int = 0
    finished_ns: int = 0
    before: dict[str, int | str] | None = None
    after: dict[str, int | str] | None = None

    def begin(self) -> None:
        self.before = R0._proc_pid_rusage_v2()
        self.started_ns = time.perf_counter_ns()

    def finish(self) -> dict[str, Any]:
        self.finished_ns = time.perf_counter_ns()
        self.after = R0._proc_pid_rusage_v2()
        if (
            self.before is None
            or self.after["rusage_uuid"] != self.before["rusage_uuid"]
        ):
            raise R2Error("rusage counter changed during stage")
        return {
            "name": self.name,
            "wall_time_ns": self.finished_ns - self.started_ns,
            "disk_read_bytes": int(self.after["disk_read_bytes"])
            - int(self.before["disk_read_bytes"]),
            "disk_write_bytes": int(self.after["disk_write_bytes"])
            - int(self.before["disk_write_bytes"]),
            "resident_before_bytes": int(self.before["resident_bytes"]),
            "resident_after_bytes": int(self.after["resident_bytes"]),
            "footprint_before_bytes": int(self.before["footprint_bytes"]),
            "footprint_after_bytes": int(self.after["footprint_bytes"]),
            "raw": {
                "logical_reads": self.counters.logical_reads,
                "logical_bytes": self.counters.logical_bytes,
                "logical_old_reads": self.counters.logical_old_reads,
                "logical_old_bytes": self.counters.logical_old_bytes,
                "logical_new_reads": self.counters.logical_new_reads,
                "logical_new_bytes": self.counters.logical_new_bytes,
                "physical_reads": self.counters.physical_reads,
                "physical_bytes": self.counters.physical_bytes,
                "physical_old_bytes": self.counters.physical_old_bytes,
                "full_raw_scans": self.counters.full_raw_scans,
                "logical_old_id_sha256": self.counters.logical_old_id_sha256[:8],
                "range_overlaps": self.counters.range_overlaps[:8],
            },
            "scans": {
                "full_event_scans": self.counters.full_event_scans,
                "full_rally_scans": self.counters.full_rally_scans,
                "full_session_scans": self.counters.full_session_scans,
                "full_fts_scans": self.counters.full_fts_scans,
                "assistant_scans": self.counters.assistant_scans,
                "fts_delta_cursor_calls": self.counters.fts_delta_cursor_calls,
                "fts_scan_statements": self.counters.fts_scan_statements,
                "full_fts_rebuilds": self.counters.full_fts_rebuilds,
            },
        }


def _measure(
    name: str,
    call: Callable[[], Any],
    *,
    catalog: Any,
    distill: Any,
    raw_store_module: Any,
    store: Any,
    old_units: tuple[Any, ...],
) -> tuple[Any, dict[str, Any]]:
    counters = ReadCounters(
        old_ids=frozenset(str(unit.raw_id) for unit in old_units),
        old_ranges=_old_range_map(iter(old_units)),
    )
    clock = StageClock(name, counters)
    clock.begin()
    try:
        with _instrument(catalog, distill, raw_store_module, store, counters):
            result = call()
    finally:
        metrics = clock.finish()
    if any(
        int(value) < 0
        for value in (metrics["disk_read_bytes"], metrics["disk_write_bytes"])
    ):
        raise R2Error("rusage counter decreased")
    return result, metrics


def _p95(values: list[int]) -> int:
    if not values:
        raise R2Error("p95 sample is empty")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _latency_summary(values: list[int]) -> str:
    """Return bounded path-neutral timing diagnostics for a failed gate."""

    return (
        f"p95_ns={_p95(values)} min_ns={min(values)} max_ns={max(values)} "
        f"values_ns={','.join(str(value) for value in values[:20])}"
    )


def _append_events(
    raw_segment: Any,
    root: Path,
    *,
    session_key: str,
    after_line: int,
    events: list[Mapping[str, Any]],
    tag: str,
    logical_source_file: Path | None = None,
) -> tuple[str, str]:
    raw_id = (
        f"save-codex-{session_key}-from{after_line}-to{after_line + len(events)}.md"
    )
    source = root / f".r2-{tag}.jsonl"
    receipt_source = logical_source_file or Path(f".r2-{tag}.jsonl")
    payload = b"".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for event in events
    )
    source.write_bytes(payload)
    try:
        receipt = raw_segment.append_capture(
            raw_dir=root / "raw",
            raw_id=raw_id,
            idempotency_key=raw_id.removeprefix("save-").removesuffix(".md"),
            host="codex",
            session_key=session_key,
            session_id=session_key,
            # The temporary source path is clone-local, while the receipt
            # identity must be path-neutral when two clones are compared.
            source_file=receipt_source,
            after_line=after_line,
            until_line=after_line + len(events),
            source_bytes=payload,
            record_count=len(events),
            now=datetime(2026, 8, 24, tzinfo=ZoneInfo("Asia/Tokyo")),
        )
    finally:
        source.unlink(missing_ok=True)
    receipt_sha256 = hashlib.sha256(
        json.dumps(
            receipt.commit.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return raw_id, receipt_sha256


def _message(role: str, text: str, index: int) -> dict[str, Any]:
    timestamp = f"2026-08-24T00:00:{index % 60:02d}Z"
    return {
        "type": "response_item",
        "timestamp": timestamp,
        "payload": {
            "type": "message",
            "role": role,
            "content": [
                {
                    "type": "input_text" if role == "user" else "output_text",
                    "text": text,
                }
            ],
        },
    }


COPYFILE_ALL = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
COPYFILE_NOFOLLOW = (1 << 18) | (1 << 19)
COPYFILE_CLONE_FORCE = 1 << 25


def _copyfile_clone(source: Path, destination: Path, flags: int) -> None:
    if sys.platform != "darwin":
        raise R2Error("copyfile(3) clone requires Darwin")
    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        function = library.copyfile
        function.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        function.restype = ctypes.c_int
        result = function(
            os.fsencode(source), os.fsencode(destination), None, int(flags)
        )
    except (AttributeError, OSError) as exc:
        raise R2Error("copyfile(3) unavailable") from exc
    if result != 0:
        raise R2Error(f"forced APFS clone failed: errno={ctypes.get_errno()}")


def _new_clone_destination(source: Path, *, prefix: str | None = None) -> Path:
    if sys.platform != "darwin":
        raise R2Error("unsupported environment: Darwin/APFS is required")
    if _has_symlink_component(source) or not source.is_dir():
        raise R2Error("clone source root is unsafe")
    try:
        source_resolved = source.resolve(strict=True)
        temp_parent = Path(tempfile.gettempdir()).resolve(strict=True)
    except OSError as exc:
        raise R2Error("clone temp parent is unavailable") from exc
    if temp_parent == source_resolved or temp_parent.is_relative_to(source_resolved):
        raise R2Error("clone temp destination overlaps source")
    destination: Path | None = None
    try:
        destination = Path(
            tempfile.mkdtemp(
                prefix=prefix or _ACTIVE_CLONE_PREFIX or "chronovisor-r2-",
                dir=temp_parent,
            )
        )
        if _path_overlap(source, destination):
            raise R2Error("APFS clone overlaps source")
        return destination
    except BaseException:
        if destination is not None:
            _cleanup_clone(destination)
        raise


def _rebind_clone_checkpoints(catalog: Any, source: Path, destination: Path) -> None:
    """Re-seal valid copied checkpoints against clone-local file identities."""

    source_catalog = catalog._read_catalog_checkpoint(source)
    if source_catalog is None:
        return
    source_catalog_state = source_catalog.get("file_state")
    clone_catalog_path = catalog.catalog_path(destination)
    clone_catalog_state = catalog._index_file_state(clone_catalog_path)
    if not isinstance(source_catalog_state, Mapping) or clone_catalog_state is None:
        raise R2Error("cloned catalog checkpoint source is incomplete")
    for key in ("size_bytes", "st_dev", "st_mtime_ns"):
        if source_catalog_state.get(key) != clone_catalog_state.get(key):
            raise R2Error("cloned catalog bytes differ from checkpoint source")

    source_index_path = catalog.historical_index_path(source)
    source_index = catalog._read_index_checkpoint(source_index_path)
    if source_index is None:
        source_index = catalog._read_index_checkpoint(
            source_index_path,
            content_digest_schema=catalog.LEGACY_HISTORICAL_INDEX_DIGEST_SCHEMA,
        )
    clone_index_path = catalog.historical_index_path(destination)
    if source_index is not None:
        clone_index_state = catalog._index_file_state(clone_index_path)
        source_index_state = source_index.get("file_state")
        if not isinstance(source_index_state, Mapping) or clone_index_state is None:
            raise R2Error("cloned FTS checkpoint source is incomplete")
        for key in ("size_bytes", "st_dev", "st_mtime_ns"):
            if source_index_state.get(key) != clone_index_state.get(key):
                raise R2Error("cloned FTS bytes differ from checkpoint source")
        if (
            source_index.get("catalog_watermark") != source_catalog["catalog_watermark"]
            or source_index.get("catalog_event_rowid") != source_catalog["event_rowid"]
            or source_index.get("catalog_file_state") != source_catalog_state
        ):
            raise R2Error("source catalog and FTS checkpoints differ")
        source_catalog_lineage = catalog._catalog_lineage(source_catalog)
        source_index_lineage = catalog._catalog_lineage(source_index)
        if (
            source_catalog_lineage is not None
            and source_index_lineage is not None
            and source_catalog_lineage != source_index_lineage
        ):
            raise R2Error("source catalog and FTS checkpoint lineages differ")

    clone_lineage = catalog._new_catalog_lineage()
    catalog._write_catalog_checkpoint(
        destination,
        str(source_catalog["catalog_watermark"]),
        int(source_catalog["event_rowid"]),
        catalog_lineage=clone_lineage,
    )
    rebound_catalog = catalog._read_catalog_checkpoint(destination)
    if rebound_catalog is None:
        raise R2Error("cloned catalog checkpoint rebind failed")
    if source_index is None:
        return
    catalog._write_index_checkpoint(
        clone_index_path,
        rebound_catalog,
        str(source_index["content_sha256"]),
        int(source_index["atom_count"]),
        content_digest_schema=str(source_index["content_digest_schema"]),
    )
    if (
        catalog._read_index_checkpoint(
            clone_index_path,
            content_digest_schema=str(source_index["content_digest_schema"]),
        )
        is None
    ):
        raise R2Error("cloned FTS checkpoint rebind failed")


def _populate_clone(
    source: Path, destination: Path, *, catalog: Any | None = None
) -> Path:
    try:
        from chronovisor.core.durable_state import okf_writer_lock
        from chronovisor.core.okf_cutover import discover_okf_startup

        if (
            _has_symlink_component(destination)
            or not destination.is_dir()
            or any(destination.iterdir())
        ):
            raise R2Error("clone destination is unsafe")

        def reject_walk_error(error: OSError) -> None:
            raise R2Error("clone source walk failed") from error

        with okf_writer_lock(source, allow_create=False):
            source_startup = discover_okf_startup(source, source / "runtime")
            if not source_startup.allowed:
                raise R2Error("clone source OKF startup is unsafe")

            required = [source / "raw", source / "runtime" / "recall-distillation"]
            required_files = [
                source / "pages" / "index.md",
                source / "pages" / "log.md",
                source / "system" / "schema.md",
                source / "runtime" / "activity.jsonl",
                source / "runtime" / "okf-writer.lock",
            ]
            migrations = source / "runtime" / "migrations"
            bootstrap_proof = source / "runtime" / "bootstrap-layout.json"
            try:
                migration_mode = migrations.lstat().st_mode
            except FileNotFoundError:
                migration_mode = None
            if migration_mode is not None:
                if not stat.S_ISDIR(migration_mode):
                    raise R2Error("clone source OKF proof is unsafe")
                required.append(migrations)
            elif bootstrap_proof.is_symlink() or not bootstrap_proof.is_file():
                raise R2Error("clone source has no OKF startup proof")
            else:
                required_files.append(bootstrap_proof)

            files: list[Path] = []
            directories: set[Path] = set()
            for subtree in required:
                if _has_symlink_component(subtree) or not subtree.is_dir():
                    raise R2Error("clone source subtree is unsafe")
                for base, dir_names, file_names in os.walk(
                    subtree, followlinks=False, onerror=reject_walk_error
                ):
                    base_path = Path(base)
                    directories.add(base_path)
                    for name in sorted(dir_names):
                        child = base_path / name
                        if child.is_symlink():
                            raise R2Error("clone source contains a symlink")
                    for name in sorted(file_names):
                        child = base_path / name
                        if child.is_symlink() or not child.is_file():
                            raise R2Error("clone source contains an unsafe file")
                        files.append(child)
            for child in required_files:
                if _has_symlink_component(child) or not child.is_file():
                    raise R2Error("clone source OKF layout is unsafe")
                files.append(child)

            for directory in sorted(directories):
                target = destination / directory.relative_to(source)
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
                target.chmod(0o700)
            flags = COPYFILE_ALL | COPYFILE_NOFOLLOW | COPYFILE_CLONE_FORCE
            for path in files:
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                target.parent.chmod(0o700)
                _copyfile_clone(path, target, flags)
            if any(path.is_symlink() for path in destination.rglob("*")):
                raise R2Error("APFS clone contains a symlink")
            clone_startup = discover_okf_startup(destination, destination / "runtime")
            if not clone_startup.allowed:
                raise R2Error("APFS clone OKF startup proof is invalid")
            if catalog is not None:
                _rebind_clone_checkpoints(catalog, source, destination)
        return destination
    except Exception as exc:
        _cleanup_clone(destination)
        if isinstance(exc, R2Error):
            raise
        raise R2Error("APFS clone failed") from exc


def _clone_from_root(source: Path, *, catalog: Any | None = None) -> Path:
    return _populate_clone(source, _new_clone_destination(source), catalog=catalog)


def _cleanup_clone(path: Path) -> None:
    """Remove a throwaway clone and verify that no path remains."""

    if path.is_symlink():
        raise R2Error("clone cleanup found a symlink")
    if path.exists():
        try:
            shutil.rmtree(path)
        except OSError as exc:
            raise R2Error("clone cleanup failed") from exc
    if path.exists() or path.is_symlink():
        raise R2Error("clone cleanup left an artifact")


def _cleanup_owned_temp_clones(
    parent: Path, prefix: str, *, protected: Iterable[Path] = ()
) -> int:
    """Clean only this run's uniquely prefixed temporary clone directories."""

    if _has_symlink_component(parent) or not parent.is_dir():
        raise R2Error("clone temp parent is unsafe")
    protected_resolved = {path.resolve(strict=False) for path in protected}
    cleaned = 0
    for candidate in sorted(parent.iterdir()):
        if not candidate.name.startswith(prefix):
            continue
        if candidate.resolve(strict=False) in protected_resolved:
            continue
        _cleanup_clone(candidate)
        cleaned += 1
    return cleaned


def _cleanup_clone_set(
    paths: list[Path], parent: Path, prefix: str
) -> int:
    """Attempt every owned path, then re-scan the uniquely owned temp prefix."""

    failure: BaseException | None = None
    for path in reversed(paths):
        try:
            _cleanup_clone(path)
        except BaseException as exc:  # preserve the first failure after best-effort cleanup
            failure = failure or exc
    paths.clear()
    try:
        owned_count = _cleanup_owned_temp_clones(parent, prefix)
    except BaseException as exc:
        failure = failure or exc
        owned_count = 0
    if failure is not None:
        raise failure
    return owned_count


def _static_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Drop live dashboard fields before comparing a frozen clone seal."""

    snapshot = dict(value)
    for key in ("fast_snapshot", "live_health"):
        snapshot.pop(key, None)
    return snapshot


def _prepare_frozen_clone(
    *,
    production: Path,
    source_root: Path,
    source_commit: str,
    dashboard_url: str,
    clone_root: Path,
    manifest_directory: Path,
    cli_paths: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Create and seal one APFS clone while production remains live."""

    _assert_root_matrix(
        production,
        source_root,
        manifest_directory,
        (clone_root,),
    )
    _require_supported_environment(production)
    _require_supported_environment(clone_root.parent)
    if clone_root.exists() or clone_root.is_symlink():
        raise R2Error("frozen clone destination already exists")
    if _path_overlap(manifest_directory, production) or _path_overlap(
        manifest_directory, source_root
    ):
        raise R2Error("frozen clone manifest overlaps protected root")
    clone_root.mkdir(parents=True, mode=0o700)
    try:
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        with R0._env(
            {
                "CHRONOVISOR_ROOT": str(clone_root),
                "CHRONOVISOR_RECALL_DISTILLATION": "true",
                "CHRONOVISOR_READ_ONLY": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        ):
            parity, _distill, store, catalog, raw_store = R0._load(source_root)
            identity_before = parity._runtime_identity(source_root, source_commit)
            production_before = _bounded_production(
                store, catalog, raw_store, production, dashboard_url
            )
            raw_before = _raw_tree_digest(production)
            _populate_clone(production, clone_root, catalog=catalog)
            clone_snapshot = _bounded_production(
                store, catalog, raw_store, clone_root, dashboard_url, clone_copy=True
            )
            clone_raw = _raw_tree_digest(clone_root)
            if R0._clone_identity(clone_snapshot) != R0._clone_identity(
                production_before
            ):
                raise R2Error("frozen APFS clone identity differs from production")
            _assert_raw_digest_parity(clone_raw, raw_before, "frozen APFS clone")
            production_after = _bounded_production(
                store, catalog, raw_store, production, dashboard_url
            )
            raw_after = _raw_tree_digest(production)
            if _static_snapshot(production_before) != _static_snapshot(production_after):
                raise R2Error("production changed while freezing APFS clone")
            if raw_before != raw_after:
                raise R2Error("production Raw changed while freezing APFS clone")
            source_tree = _source_tree_digest(source_root)
            payload = {
                "captured_at": datetime.now().astimezone().isoformat(),
                "clone_root": str(clone_root.resolve(strict=True)),
                "source_commit": source_commit,
                "source_tree": source_tree,
                "runtime_identity": identity_before,
                "production": {
                    "static": _static_snapshot(production_before),
                    "raw_tree": raw_before,
                },
                "clone": {
                    "static": _static_snapshot(clone_snapshot),
                    "raw_tree": clone_raw,
                    "filesystem": R0._filesystem_type(clone_root),
                    "clone_backend": "copyfile(3):COPYFILE_CLONE_FORCE",
                },
                "cli_paths": dict(cli_paths or {}),
                "threshold": {
                    "production_unchanged_during_freeze": True,
                    "raw_parity": True,
                },
            }
            artifact_id, artifact_path, artifact = store.write_immutable(
                manifest_directory, payload, schema=FROZEN_CLONE_SCHEMA
            )
            persisted = _read_content_addressed_artifact(
                store,
                artifact_path,
                schema=FROZEN_CLONE_SCHEMA,
                label="frozen clone manifest",
            )
            return {
                "schema": persisted["schema"],
                "artifact_id": persisted["artifact_id"],
                "path": str(artifact_path),
                "clone_root": str(clone_root),
                "cli_paths": dict(cli_paths or {}),
            }
    except Exception:
        _cleanup_clone(clone_root)
        raise
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode


def _verify_frozen_clone(
    *,
    manifest_path: Path,
    clone_root: Path,
    production: Path,
    source_root: Path,
    source_commit: str,
    parity: Any,
    store: Any,
    catalog: Any,
    raw_store: Any,
    dashboard_url: str,
) -> dict[str, Any]:
    """Verify a sealed external clone before allowing any derived mutation."""

    manifest = _read_content_addressed_artifact(
        store,
        manifest_path,
        schema=FROZEN_CLONE_SCHEMA,
        label="frozen clone manifest",
    )
    expected_root = manifest.get("clone_root")
    if not isinstance(expected_root, str) or Path(expected_root).resolve(strict=False) != clone_root:
        raise R2Error("frozen clone manifest root mismatch")
    if manifest.get("source_commit") != source_commit:
        raise R2Error("frozen clone source commit mismatch")
    _assert_symlink_free_tree(clone_root, label="frozen clone")
    _assert_root_matrix(production, source_root, manifest_path, (clone_root,))
    _require_supported_environment(clone_root)
    source_tree = manifest.get("source_tree")
    if not isinstance(source_tree, Mapping) or _source_tree_digest(source_root) != source_tree:
        raise R2Error("frozen clone source tree changed")
    identity = parity._runtime_identity(source_root, source_commit)
    if identity != manifest.get("runtime_identity"):
        raise R2Error("frozen clone runtime identity changed")
    clone_snapshot = _bounded_production(
        store, catalog, raw_store, clone_root, dashboard_url, clone_copy=True
    )
    expected_clone = manifest.get("clone")
    if not isinstance(expected_clone, Mapping):
        raise R2Error("frozen clone manifest payload is incomplete")
    if _static_snapshot(clone_snapshot) != expected_clone.get("static"):
        raise R2Error("frozen clone derived state changed")
    if _raw_tree_digest(clone_root) != expected_clone.get("raw_tree"):
        raise R2Error("frozen clone Raw changed")
    if expected_clone.get("filesystem") != R0._filesystem_type(clone_root):
        raise R2Error("frozen clone filesystem changed")
    return manifest


def _catalog_session_tail(catalog: Any, root: Path) -> tuple[str, int]:
    path = catalog.catalog_path(root)
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT host,session_key,source_until_line FROM raw_units "
                "WHERE host='codex' AND source_until_line=("
                "SELECT MAX(source_until_line) FROM raw_units WHERE host='codex') "
                "ORDER BY host,session_key,raw_id LIMIT 1"
            ).fetchone()
    except sqlite3.DatabaseError as exc:
        raise R2Error("cannot select existing session") from exc
    if row is None or not isinstance(row[1], str) or row[2] is None:
        raise R2Error("catalog has no existing session tail")
    if str(row[0]) != "codex":
        raise R2Error("synthetic tail host is unsupported")
    return str(row[1]), int(row[2])


def _fault_repair(
    catalog: Any,
    distill: Any,
    store: Any,
    raw_store_module: Any,
    root: Path,
    raw_dir: Path,
    context_bytes: int,
    old_units: tuple[Any, ...],
    reference_snapshot: Mapping[str, Any] | None = None,
    inventory_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Exercise checkpoint/DB recovery on this clone, never the live root."""

    catalog_checkpoint = catalog._catalog_checkpoint_path(root)
    if not catalog_checkpoint.exists():
        raise R2Error("catalog checkpoint missing after migration")
    catalog_checkpoint.write_text("{}", encoding="utf-8")
    try:
        catalog.rallies(root)
    except Exception:
        pass
    else:
        raise R2Error("catalog checkpoint tamper did not fail closed")
    repair_old_units = _raw_units(raw_store_module, raw_dir)
    _, repaired = _measure(
        "catalog-checkpoint-repair",
        lambda: catalog.advance(raw_dir, root, context_bytes),
        catalog=catalog,
        distill=distill,
        raw_store_module=raw_store_module,
        store=store,
        old_units=repair_old_units,
    )
    index_path = catalog.historical_index_path(root)
    index_checkpoint = catalog._index_checkpoint_path(index_path)
    index_checkpoint.unlink(missing_ok=True)
    _, index_repaired = _measure(
        "fts-checkpoint-repair",
        lambda: catalog.sync_historical_index(raw_dir, root),
        catalog=catalog,
        distill=distill,
        raw_store_module=raw_store_module,
        store=store,
        old_units=repair_old_units,
    )
    snapshot = _derived_snapshot(
        catalog,
        store,
        raw_store_module,
        raw_dir,
        root,
        inventory_baseline=inventory_baseline,
    )
    if reference_snapshot is not None:
        _assert_repair_parity(snapshot, reference_snapshot, "checkpoint repair")
    return {
        "catalog": repaired,
        "fts": index_repaired,
        "snapshot": snapshot,
        "parity": reference_snapshot is not None,
    }


def _tamper_repair(
    *,
    base: Path,
    catalog: Any,
    distill: Any,
    store: Any,
    raw_store_module: Any,
    context_bytes: int,
    old_units: tuple[Any, ...],
    reference_snapshot: Mapping[str, Any] | None = None,
    inventory_baseline: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Verify that ordinary catalog/FTS SQL tamper cannot enter the warm path."""

    root = _clone_from_root(base, catalog=catalog)
    catalog_path = catalog.catalog_path(root)
    try:
        _assert_raw_state_parity(
            _raw_tree_state_digest(root),
            _raw_tree_state_digest(base),
            "DB tamper clone",
        )
        repair_old_units = _raw_units(raw_store_module, root / "raw")
        with sqlite3.connect(catalog_path) as connection:
            connection.execute(
                "UPDATE events SET role='tampered' WHERE rowid=(SELECT MIN(rowid) FROM events)"
            )
        try:
            catalog.rallies(root)
        except Exception:
            pass
        else:
            raise R2Error("catalog DB tamper did not fail closed")
        _, catalog_metrics = _measure(
            "catalog-db-tamper-repair",
            lambda: catalog.advance(root / "raw", root, context_bytes),
            catalog=catalog,
            distill=distill,
            raw_store_module=raw_store_module,
            store=store,
            old_units=repair_old_units,
        )
        index_path = catalog.historical_index_path(root)
        with sqlite3.connect(index_path) as connection:
            connection.execute(
                "UPDATE atoms SET host='tampered' WHERE rowid=(SELECT MIN(rowid) FROM atoms)"
            )
        try:
            catalog.sync_historical_index(root / "raw", root)
        except Exception:
            fts_tamper_rejected = True
        else:
            raise R2Error("FTS DB tamper did not fail closed")
        _remove_index_allowlist(catalog, root)
        _, index_metrics = _measure(
            "fts-db-tamper-repair",
            lambda: catalog.sync_historical_index(root / "raw", root),
            catalog=catalog,
            distill=distill,
            raw_store_module=raw_store_module,
            store=store,
            old_units=repair_old_units,
        )
        snapshot = _derived_snapshot(
            catalog,
            store,
            raw_store_module,
            root / "raw",
            root,
            inventory_baseline=inventory_baseline,
        )
        if reference_snapshot is not None:
            _assert_repair_parity(snapshot, reference_snapshot, "DB tamper repair")
        return root, {
            "catalog": catalog_metrics,
            "fts": index_metrics,
            "snapshot": snapshot,
            "catalog_tamper_rejected": True,
            "fts_tamper_rejected": fts_tamper_rejected,
            "parity": reference_snapshot is not None,
        }
    except Exception:
        _cleanup_clone(root)
        raise


_POST_COMMIT_CHILD = r"""
from pathlib import Path
import os
import sqlite3
import sys

source_root = Path(sys.argv[1])
root = Path(sys.argv[2])
mode = sys.argv[3]
context_bytes = int(sys.argv[4])
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(source_root / "src"))
from chronovisor.recall import recall_distillation_catalog as catalog

if mode == "catalog":
    original = catalog._connect

    class CrashConnection:
        def __init__(self, connection):
            self._connection = connection
        def __getattr__(self, name):
            return getattr(self._connection, name)
        def commit(self):
            self._connection.commit()
            os._exit(137)

    def connect(value):
        return CrashConnection(original(value))

    catalog._connect = connect
    catalog.advance(root / "raw", root, context_bytes)
else:
    original = catalog.sqlite3.connect

    class CrashConnection(sqlite3.Connection):
        def commit(self):
            super().commit()
            os._exit(137)

    def connect(*args, **kwargs):
        kwargs["factory"] = CrashConnection
        return original(*args, **kwargs)

    catalog.sqlite3.connect = connect
    catalog.sync_historical_index(root / "raw", root)
"""


def _run_post_commit_crash(
    *,
    base: Path,
    source_root: Path,
    raw_segment: Any,
    catalog: Any,
    distill: Any,
    store: Any,
    raw_store_module: Any,
    context_bytes: int,
    old_units: tuple[Any, ...],
    inventory_baseline: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], Path]:
    """Kill a child immediately after durable commit, then recover in parent."""

    root = _clone_from_root(base, catalog=catalog)
    try:
        clean_root = _clone_from_root(base, catalog=catalog)
    except Exception:
        _cleanup_clone(root)
        raise
    try:
        base_raw_state = _raw_tree_state_digest(base)
        _assert_raw_state_parity(
            _raw_tree_state_digest(root), base_raw_state, "post-commit"
        )
        _assert_raw_state_parity(
            _raw_tree_state_digest(clean_root), base_raw_state, "post-commit clean"
        )
        crash_old_units = _raw_units(raw_store_module, root / "raw")
        new_session = "fac9" * 6
        crash_events = [
            _message("user", "r2 crash query", 0),
            _message("assistant", "r2 crash answer", 1),
        ]
        crash_raw_id, crash_receipt_sha = _append_events(
            raw_segment,
            root,
            session_key=new_session,
            after_line=0,
            events=crash_events,
            tag="post-commit",
            logical_source_file=Path("r2-post-commit.jsonl"),
        )
        clean_raw_id, clean_receipt_sha = _append_events(
            raw_segment,
            clean_root,
            session_key=new_session,
            after_line=0,
            events=crash_events,
            tag="post-commit",
            logical_source_file=Path("r2-post-commit.jsonl"),
        )
        if (crash_raw_id, crash_receipt_sha) != (clean_raw_id, clean_receipt_sha):
            raise R2Error("post-commit paired Raw receipt identity differs")
        env = dict(os.environ)
        env["CHRONOVISOR_ROOT"] = str(root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                _POST_COMMIT_CHILD,
                str(source_root),
                str(root),
                "catalog",
                str(context_bytes),
            ],
            env=env,
            capture_output=True,
        )
        if process.returncode != 137:
            raise R2Error(
                "catalog post-commit child did not terminate at fault boundary "
                f"(returncode={process.returncode})"
            )
        catalog_child_returncode = process.returncode
        _, catalog_metrics = _measure(
            "catalog-post-commit-recovery",
            lambda: catalog.advance(root / "raw", root, context_bytes),
            catalog=catalog,
            distill=distill,
            raw_store_module=raw_store_module,
            store=store,
            old_units=crash_old_units,
        )
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                _POST_COMMIT_CHILD,
                str(source_root),
                str(root),
                "fts",
                str(context_bytes),
            ],
            env=env,
            capture_output=True,
        )
        if process.returncode != 137:
            raise R2Error(
                "FTS post-commit child did not terminate at fault boundary "
                f"(returncode={process.returncode})"
            )
        fts_child_returncode = process.returncode
        _, fts_metrics = _measure(
            "fts-post-commit-recovery",
            lambda: catalog.sync_historical_index(root / "raw", root),
            catalog=catalog,
            distill=distill,
            raw_store_module=raw_store_module,
            store=store,
            old_units=crash_old_units,
        )
        catalog.advance(clean_root / "raw", clean_root, context_bytes)
        catalog.sync_historical_index(clean_root / "raw", clean_root)
        clean_snapshot = _derived_snapshot(
            catalog,
            store,
            raw_store_module,
            clean_root / "raw",
            clean_root,
            inventory_baseline=inventory_baseline,
            expected_new_raw_id=crash_raw_id,
        )
        snapshot = _derived_snapshot(
            catalog,
            store,
            raw_store_module,
            root / "raw",
            root,
            inventory_baseline=inventory_baseline,
            expected_new_raw_id=crash_raw_id,
        )
        _assert_repair_parity(snapshot, clean_snapshot, "post-commit repair")
        return (
            root,
            {
                "catalog": catalog_metrics,
                "fts": fts_metrics,
                "raw_id": crash_raw_id,
                "receipt_sha256": crash_receipt_sha,
                "catalog_child_returncode": catalog_child_returncode,
                "fts_child_returncode": fts_child_returncode,
                "snapshot": snapshot,
                "clean_reference": clean_snapshot,
                "parity": True,
            },
            clean_root,
        )
    except Exception:
        _cleanup_clone(root)
        _cleanup_clone(clean_root)
        raise


def _remove_derived_allowlist(catalog: Any, root: Path) -> None:
    """Delete only derived catalog/index files in a throwaway clone."""

    paths = (
        catalog.catalog_path(root),
        catalog._catalog_checkpoint_path(root),
        catalog.historical_index_path(root),
        catalog._index_checkpoint_path(catalog.historical_index_path(root)),
    )
    for path in paths:
        path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm", "-journal"):
            path.with_name(path.name + suffix).unlink(missing_ok=True)


def _remove_index_allowlist(catalog: Any, root: Path) -> None:
    path = catalog.historical_index_path(root)
    for candidate in (path, catalog._index_checkpoint_path(path)):
        candidate.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm", "-journal"):
            candidate.with_name(candidate.name + suffix).unlink(missing_ok=True)


def _old_raw_hash_tamper(
    *,
    base: Path,
    catalog: Any,
    store: Any,
    raw_store_module: Any,
    raw_segment_module: Any,
    context_bytes: int,
    old_units: tuple[Any, ...],
) -> tuple[Path, dict[str, Any]]:
    """Flip one committed text byte and require source resolution to fail closed."""

    root = _clone_from_root(base, catalog=catalog)
    try:
        _assert_raw_state_parity(
            _raw_tree_state_digest(root),
            _raw_tree_state_digest(base),
            "old Raw hash clone",
        )
        if not old_units:
            raise R2Error("old Raw inventory is empty")
        raw_store = raw_store_module.RawStore(root / "raw", mode="v2")
        units_by_id = {
            str(unit.raw_id): unit for unit in raw_store.iter_segment_units()
        }
        selected: tuple[Any, bytes, bytes, int, sqlite3.Row] | None = None
        with sqlite3.connect(catalog.catalog_path(root)) as connection:
            connection.row_factory = sqlite3.Row
            for template in old_units:
                unit = units_by_id.get(str(template.raw_id))
                if unit is None:
                    raise R2Error("old Raw inventory changed before tamper")
                original = raw_store.read_bytes(unit)
                marker = b'"text":"'
                marker_start = original.find(marker)
                if marker_start < 0:
                    continue
                start = marker_start + len(marker)
                end = original.find(b'"', start)
                if end < 0:
                    continue
                replacement_index = None
                for index in range(start, end):
                    value = original[index]
                    if value == ord('"'):
                        break
                    if value == ord("\\"):
                        continue
                    if 48 <= value <= 57 or 65 <= value <= 90 or 97 <= value <= 122:
                        replacement_index = index
                        break
                if replacement_index is None:
                    for index in range(start, end):
                        if original[index] not in {ord('"'), ord("\\")}:
                            replacement_index = index
                            break
                if replacement_index is None:
                    continue
                replacement = (
                    ord("X") if original[replacement_index] != ord("X") else ord("Y")
                )
                tampered = (
                    original[:replacement_index]
                    + bytes((replacement,))
                    + original[replacement_index + 1 :]
                )
                row = connection.execute(
                    "SELECT * FROM events WHERE raw_id=? "
                    "ORDER BY CASE WHEN role='assistant' AND nonempty=1 "
                    "THEN 0 ELSE 1 END, rowid LIMIT 1",
                    (str(unit.raw_id),),
                ).fetchone()
                if row is None:
                    continue
                selected = (unit, original, tampered, replacement_index, row)
                break
        if selected is None:
            raise R2Error("old Raw has no known text byte to tamper")
        unit, original, tampered, replacement_index, row = selected
        expected_raw_sha256 = str(unit.sha256)
        tampered_raw_sha256 = hashlib.sha256(tampered).hexdigest()
        if tampered_raw_sha256 == expected_raw_sha256:
            raise R2Error("old Raw tamper did not change the committed hash")
        # Snapshot derived state before mutating Raw.  These helpers read only
        # SQLite and therefore remain usable after the Raw hash mismatch.
        before = {
            "catalog": _catalog_snapshot(catalog, root),
            "fts": _index_sqlite_snapshot(catalog, root),
        }
        if unit.storage == "segment_open":
            with unit.path.open("r+b") as handle:
                handle.seek(int(unit.offset) + replacement_index)
                handle.write(tampered[replacement_index : replacement_index + 1])
        else:
            units_in_segment = [
                other for other in units_by_id.values() if other.path == unit.path
            ]
            logical_end = max(
                int(other.offset) + int(other.length) for other in units_in_segment
            )
            segment = raw_segment_module.read_sealed_range(unit.path, 0, logical_end)
            segment = (
                segment[: int(unit.offset)]
                + tampered
                + segment[int(unit.offset) + int(unit.length) :]
            )
            compressed = raw_segment_module.zstd.ZstdCompressor(level=9).compress(
                segment
            )
            unit.path.write_bytes(compressed)
        try:
            catalog.texts(root / "raw", refs=[dict(row)])
        except Exception as exc:
            after = {
                "catalog": _catalog_snapshot(catalog, root),
                "fts": _index_sqlite_snapshot(catalog, root),
            }
            if after != before:
                raise R2Error("old Raw hash tamper changed derived state") from exc
            return root, {
                "raw_id_sha256": hashlib.sha256(str(unit.raw_id).encode()).hexdigest(),
                "expected_raw_sha256": expected_raw_sha256,
                "tampered_raw_sha256": tampered_raw_sha256,
                "hash_mismatch": True,
                "storage": str(unit.storage),
                "byte_offset": int(unit.offset) + replacement_index,
                "byte_before": original[replacement_index],
                "byte_after": tampered[replacement_index],
                "rejected": True,
                "error_type": type(exc).__name__,
                "derived_unchanged": True,
            }
        raise R2Error("old Raw byte tamper was accepted")
    except Exception:
        _cleanup_clone(root)
        raise


def _full_rebuild_parity(
    *,
    base: Path,
    catalog: Any,
    distill: Any,
    store: Any,
    raw_store_module: Any,
    context_bytes: int,
    inventory_baseline: Mapping[str, Any] | None = None,
    expected_new_raw_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = _clone_from_root(base, catalog=catalog)
    try:
        _assert_raw_state_parity(
            _raw_tree_state_digest(root),
            _raw_tree_state_digest(base),
            "full rebuild clone",
        )
        _remove_derived_allowlist(catalog, root)
        old_units = _raw_units(raw_store_module, root / "raw")
        (catalog_result, fts_digest), metrics = _measure(
            "full-rebuild-parity",
            lambda: (
                catalog.advance(root / "raw", root, context_bytes),
                catalog.sync_historical_index(root / "raw", root),
            ),
            catalog=catalog,
            distill=distill,
            raw_store_module=raw_store_module,
            store=store,
            old_units=old_units,
        )
        if getattr(catalog_result, "status", None) not in {"bootstrap", "repaired"}:
            raise R2Error("full rebuild did not bootstrap catalog")
        return root, {
            "metrics": metrics,
            **_derived_snapshot(
                catalog,
                store,
                raw_store_module,
                root / "raw",
                root,
                inventory_baseline=inventory_baseline,
                expected_new_raw_id=expected_new_raw_id,
            ),
            "fts_digest": fts_digest,
        }
    except Exception:
        _cleanup_clone(root)
        raise


def _assert_warm(metrics: Mapping[str, Any]) -> None:
    raw = metrics["raw"]
    scans = metrics["scans"]
    if any(
        int(raw[key]) != 0
        for key in (
            "logical_old_reads",
            "logical_old_bytes",
            "logical_new_reads",
            "logical_new_bytes",
            "physical_old_bytes",
            "full_raw_scans",
        )
    ):
        raise R2Error("warm path read existing Raw")
    if any(
        int(scans[key]) != 0
        for key in (
            "full_event_scans",
            "full_rally_scans",
            "full_session_scans",
            "full_fts_scans",
            "assistant_scans",
            "full_fts_rebuilds",
            "fts_scan_statements",
        )
    ):
        raise R2Error("warm path scanned/rebuilt existing derived data")


def _assert_no_duplicates(snapshot: Mapping[str, Any]) -> None:
    duplicates = snapshot.get("duplicates", {})
    if isinstance(duplicates, Mapping) and any(
        int(value) for value in duplicates.values()
    ):
        raise R2Error("derived catalog contains duplicate records")


def _assert_delta(metrics: Mapping[str, Any], new_raw_id: str) -> None:
    raw = metrics["raw"]
    if (
        int(raw["logical_old_reads"])
        or int(raw["logical_old_bytes"])
        or int(raw["physical_old_bytes"])
        or int(raw["full_raw_scans"])
    ):
        overlap = next(iter(raw.get("range_overlaps", ())), {})
        old_hash = next(iter(raw.get("logical_old_id_sha256", ())), "none")
        raise R2Error(
            "delta path read existing Raw "
            f"logical_old_reads={raw['logical_old_reads']} "
            f"logical_old_bytes={raw['logical_old_bytes']} "
            f"physical_old_bytes={raw['physical_old_bytes']} "
            f"full_raw_scans={raw['full_raw_scans']} "
            f"old_id_sha256={old_hash} "
            f"overlap_path={overlap.get('path', 'none')} "
            f"overlap_offset={overlap.get('offset', 'none')} "
            f"overlap_length={overlap.get('length', 'none')} "
            f"overlap_old_bytes={overlap.get('old_overlap_bytes', 'none')}"
        )
    if int(raw["logical_new_reads"]) == 0 or int(raw["logical_new_bytes"]) == 0:
        raise R2Error(f"delta path did not read new Raw: {new_raw_id}")
    scans = metrics["scans"]
    if any(
        int(scans[key]) != 0
        for key in (
            "full_event_scans",
            "full_rally_scans",
            "full_session_scans",
            "full_fts_rebuilds",
            "full_fts_scans",
            "assistant_scans",
            "fts_scan_statements",
        )
    ):
        raise R2Error("delta path scanned/rebuilt existing derived data")


def _validate_delta_outcome(
    outcome: dict[str, Any],
    sample_root: Path,
    *,
    catalog: Any,
    raw_store_module: Any,
    inventory_baseline: Mapping[str, Any],
    new_raw_id: str,
) -> None:
    outcome["catalog"] = _catalog_integrity_snapshot(catalog, sample_root)
    _assert_no_duplicates(outcome["catalog"])
    outcome["fts"] = _index_sqlite_snapshot(catalog, sample_root)
    outcome["inventory"] = _raw_inventory_id_snapshot(
        catalog,
        raw_store_module,
        sample_root / "raw",
        sample_root,
        expected_new_raw_id=new_raw_id,
        expected_inventory=inventory_baseline,
    )


def _run_once(
    *,
    production: Path,
    source_root: Path,
    source_commit: str,
    dashboard_url: str,
    output: Path,
    noop_samples: int,
    delta_samples: int,
    context_bytes: int,
    phase_receipt: Path | None = None,
    frozen_clone_root: Path | None = None,
    frozen_clone_manifest: Path | None = None,
    cli_paths: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    global _ACTIVE_CLONE_PREFIX
    if (frozen_clone_root is None) != (frozen_clone_manifest is None):
        raise R2Error("frozen clone root and manifest must be provided together")
    other_paths = {
        name: path
        for name, path in {
            "phase_receipt": phase_receipt,
            "frozen_clone_manifest": frozen_clone_manifest,
        }.items()
        if path is not None
    }
    _assert_root_matrix(
        production,
        source_root,
        output,
        (frozen_clone_root,) if frozen_clone_root is not None else (),
        other_paths,
    )
    phases = _PhaseReceipt(phase_receipt)
    phases.event("run-start", cli_paths=dict(cli_paths or {}))
    _require_supported_environment(production)
    if noop_samples < 20 or delta_samples < 20:
        raise R2Error("R2 sample counts are below the required minimum")
    source_before = phases.measure(
        "source-preflight", lambda: _source_tree_digest(source_root)
    )
    clones: list[Path] = []
    owned_temp_parent = Path(tempfile.gettempdir()).expanduser().resolve(strict=True)
    owned_temp_prefix = f"chronovisor-r2-{os.getpid()}-{uuid.uuid4().hex[:12]}-"
    previous_clone_prefix = _ACTIVE_CLONE_PREFIX
    _ACTIVE_CLONE_PREFIX = owned_temp_prefix
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    cleanup_recorded = False
    owned_cleanup_count = 0
    try:
        clone = (
            frozen_clone_root.expanduser().resolve(strict=True)
            if frozen_clone_root is not None
            else phases.measure(
                "initial-clone-destination", lambda: _new_clone_destination(production)
            )
        )
        clones.append(clone)
        _assert_root_matrix(production, source_root, output, iter(clones))
        with R0._env(
            {
                "CHRONOVISOR_ROOT": str(clone),
                "CHRONOVISOR_RECALL_DISTILLATION": "true",
                "CHRONOVISOR_READ_ONLY": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        ):
            parity, distill, store, catalog, raw_store = phases.measure(
                "runtime-load", lambda: R0._load(source_root)
            )
            frozen_manifest = None
            if frozen_clone_manifest is None:
                phases.measure(
                    "initial-clone-populate",
                    lambda: _populate_clone(production, clone, catalog=catalog),
                )
            else:
                frozen_manifest = phases.measure(
                    "frozen-clone-verify",
                    lambda: _verify_frozen_clone(
                        manifest_path=frozen_clone_manifest.expanduser().resolve(strict=True),
                        clone_root=clone,
                        production=production,
                        source_root=source_root,
                        source_commit=source_commit,
                        parity=parity,
                        store=store,
                        catalog=catalog,
                        raw_store=raw_store,
                        dashboard_url=dashboard_url,
                    ),
                )
            raw_segment = __import__(
                "chronovisor.core.raw_segment", fromlist=["append_capture"]
            )
            raw_store_module = sys.modules[raw_store.RawStore.__module__]
            identity_before = phases.measure(
                "runtime-identity-before",
                lambda: parity._runtime_identity(source_root, source_commit),
            )
            production_before = phases.measure(
                "production-derived-before",
                lambda: _bounded_production(
                    store, catalog, raw_store, production, dashboard_url
                ),
            )
            clone_before = phases.measure(
                "clone-derived-before",
                lambda: _bounded_production(
                    store, catalog, raw_store, clone, dashboard_url, clone_copy=True
                ),
            )
            if frozen_manifest is None and R0._clone_identity(clone_before) != R0._clone_identity(
                production_before
            ):
                raise R2Error("APFS clone is not point-in-time coherent")
            raw_before = phases.measure(
                "production-raw-before", lambda: _raw_tree_digest(production)
            )
            clone_raw_before = phases.measure(
                "clone-raw-before", lambda: _raw_tree_digest(clone)
            )
            if frozen_manifest is None:
                _assert_raw_digest_parity(clone_raw_before, raw_before, "APFS clone")
            old_catalog = phases.measure(
                "production-catalog-baseline", lambda: _catalog_snapshot(catalog, production)
            )
            if (
                old_catalog.get("columns", {}).get("raw_units")
                != [
                    "raw_id",
                    "raw_sha256",
                    "receipt_sha256",
                    "host",
                    "session_key",
                    "captured_at",
                    "record_count",
                    "status",
                ]
                or old_catalog.get("rows", {}).get("raw_units") != 18_633
            ):
                raise R2Error(
                    "production catalog is not the expected legacy 8-column baseline"
                )
            old_units = phases.measure(
                "raw-unit-baseline", lambda: _raw_units(raw_store, clone / "raw")
            )
            old_range_map = _old_range_map(iter(old_units))
            del old_range_map  # recalculated per stage by _measure

            migration_catalog, migration_metrics = phases.measure(
                "catalog-migration",
                lambda: _measure(
                    "catalog-migration",
                    lambda: catalog.advance(clone / "raw", clone, context_bytes),
                    catalog=catalog,
                    distill=distill,
                    raw_store_module=raw_store_module,
                    store=store,
                    old_units=old_units,
                ),
                units=len(old_units),
            )
            migration_fts, fts_metrics = phases.measure(
                "fts-migration",
                lambda: _measure(
                    "fts-migration",
                    lambda: catalog.sync_historical_index(clone / "raw", clone),
                    catalog=catalog,
                    distill=distill,
                    raw_store_module=raw_store_module,
                    store=store,
                    old_units=old_units,
                ),
            )
            if getattr(migration_catalog, "status", None) not in {
                "bootstrap",
                "repaired",
            }:
                raise R2Error("legacy catalog did not perform one migration")
            if not isinstance(migration_fts, str) or len(migration_fts) != 64:
                raise R2Error("FTS migration digest invalid")
            migration_snapshot = _derived_snapshot(
                catalog,
                store,
                raw_store_module,
                clone / "raw",
                clone,
                include_inventory=False,
            )
            # The bootstrap already read every committed Raw unit.  Retain
            # that one authoritative status/ID check and make all subsequent
            # clone validations metadata-only; repeating the 8+ GB Raw
            # decode on every repair clone was the R2 timeout bottleneck.
            inventory_baseline = phases.measure(
                "raw-inventory-baseline",
                lambda: _raw_inventory_baseline(
                    catalog, raw_store_module, clone / "raw", clone
                ),
                units=len(old_units),
            )
            migration_snapshot["inventory"] = {
                key: value
                for key, value in inventory_baseline.items()
                if key in {"count", "ids_sha256", "status_counts"}
            }

            noop_metrics: list[dict[str, Any]] = []
            prior_catalog_state = migration_snapshot["catalog"].get("file_state")
            prior_index_state = migration_snapshot["fts"].get("file_state")
            for sample in range(noop_samples):
                _, metrics = phases.measure(
                    "warm-no-new",
                    lambda: _measure(
                        "warm-no-new",
                        lambda: (
                            catalog.advance(clone / "raw", clone, context_bytes),
                            catalog.sync_historical_index(clone / "raw", clone),
                        ),
                        catalog=catalog,
                        distill=distill,
                        raw_store_module=raw_store_module,
                        store=store,
                        old_units=old_units,
                    ),
                    sample=sample,
                    samples=noop_samples,
                )
                _assert_warm(metrics)
                if R0._stat(catalog.catalog_path(clone)) != prior_catalog_state:
                    raise R2Error("warm no-op rewrote catalog DB")
                if R0._stat(catalog.historical_index_path(clone)) != prior_index_state:
                    raise R2Error("warm no-op rewrote FTS DB")
                noop_metrics.append(metrics)
            noop_values = [int(row["wall_time_ns"]) for row in noop_metrics]
            noop_p95 = _p95(noop_values)
            if noop_p95 > 1_000_000_000:
                raise R2Error(
                    "no-new-data latency gate failed "
                    f"threshold_ns=1000000000 {_latency_summary(noop_values)}"
                )

            session_key, tail_after = phases.measure(
                "session-tail-selection", lambda: _catalog_session_tail(catalog, clone)
            )
            delta_metrics: list[dict[str, Any]] = []
            delta_outcomes: list[dict[str, Any]] = []
            delta_roots: list[Path] = []
            delta_ids: list[str] = []
            for sample in range(delta_samples):
                sample_root = phases.measure(
                    "delta-clone",
                    lambda: _clone_from_root(clone, catalog=catalog),
                    sample=sample,
                    samples=delta_samples,
                )
                clones.append(sample_root)
                _assert_root_matrix(production, source_root, output, iter(clones))
                _assert_raw_state_parity(
                    _raw_tree_state_digest(sample_root),
                    _raw_tree_state_digest(clone),
                    "delta clone",
                )
                delta_roots.append(sample_root)
                sample_old_units = _raw_units(raw_store, sample_root / "raw")
                new_session = hashlib.sha256(f"r2-delta-{sample}".encode()).hexdigest()[
                    :24
                ]
                events = [_message("user", "r2 synthetic query", 0)] + [
                    _message("assistant", "r2 synthetic answer", index)
                    for index in range(1, 1000)
                ]
                new_id, new_receipt_sha = phases.measure(
                    "delta-event-append",
                    lambda sample_root=sample_root, new_session=new_session, events=events, sample=sample: _append_events(  # type: ignore[misc]
                        raw_segment,
                        sample_root,
                        session_key=new_session,
                        after_line=0,
                        events=events,
                        tag=f"delta-{sample}",
                    ),
                    sample=sample,
                    samples=delta_samples,
                )
                result, metrics = phases.measure(
                    "delta-1000-events",
                    lambda root=sample_root, sample_old_units=sample_old_units: _measure(  # type: ignore[misc]
                        "delta-1000-events",
                        lambda: (
                            catalog.advance(root / "raw", root, context_bytes),
                            catalog.sync_historical_index(root / "raw", root),
                        ),
                        catalog=catalog,
                        distill=distill,
                        raw_store_module=raw_store_module,
                        store=store,
                        old_units=sample_old_units,
                    ),
                    sample=sample,
                    samples=delta_samples,
                )
                _assert_delta(metrics, new_id)
                delta_metrics.append(metrics)
                delta_ids.append(new_id)
                delta_outcomes.append(
                    {
                        "sample": sample,
                        "catalog_status": getattr(result[0], "status", None),
                        "new_raw_id_sha256": hashlib.sha256(
                            new_id.encode()
                        ).hexdigest(),
                        "receipt_sha256": new_receipt_sha,
                    }
                )
            reference_root = sample_root
            baseline_inventory_ids = phases.measure(
                "delta-baseline-validation",
                lambda: _raw_inventory_id_snapshot(
                    catalog,
                    raw_store_module,
                    clone / "raw",
                    clone,
                    expected_inventory=inventory_baseline,
                ),
            )
            for index, (outcome, sample_root) in enumerate(
                zip(delta_outcomes, delta_roots, strict=True)
            ):
                outcome["validation_stage"] = "post-delta-derived-validation"
                phases.measure(
                    "delta-validation",
                    lambda outcome=outcome, sample_root=sample_root, index=index: _validate_delta_outcome(  # type: ignore[misc]
                        outcome,
                        sample_root,
                        catalog=catalog,
                        raw_store_module=raw_store_module,
                        inventory_baseline=inventory_baseline,
                        new_raw_id=delta_ids[index],
                    ),
                    sample=index,
                    samples=delta_samples,
                )
                if outcome["inventory"]["count"] != baseline_inventory_ids["count"] + 1:
                    raise R2Error(
                        "delta Raw inventory count is not one-unit incremental"
                    )
            delta_inventory_validation = {
                "stage": "post-delta-raw-inventory-validation",
                "snapshot": _raw_inventory_id_snapshot(
                    catalog,
                    raw_store_module,
                    reference_root / "raw",
                    reference_root,
                    expected_new_raw_id=delta_ids[-1],
                    expected_inventory=inventory_baseline,
                ),
            }
            delta_outcomes[-1]["inventory"] = delta_inventory_validation["snapshot"]
            # Keep one full reference digest for independent rebuild parity;
            # all other per-sample snapshots remain compact and bounded.
            reference_snapshot = phases.measure(
                "reference-snapshot",
                lambda: _derived_snapshot(
                    catalog,
                    store,
                    raw_store_module,
                    reference_root / "raw",
                    reference_root,
                    inventory_baseline=inventory_baseline,
                    expected_new_raw_id=delta_ids[-1],
                ),
                raw_units=len(old_units) + 1,
            )
            delta_outcomes[-1]["catalog"] = reference_snapshot["catalog"]
            delta_outcomes[-1]["fts"] = reference_snapshot["fts"]
            delta_values = [int(row["wall_time_ns"]) for row in delta_metrics]
            delta_p95 = _p95(delta_values)
            if delta_p95 > 15_000_000_000:
                raise R2Error(
                    "delta latency gate failed "
                    f"threshold_ns=15000000000 {_latency_summary(delta_values)}"
                )

            tail_root = phases.measure(
                "session-tail-clone",
                lambda: _clone_from_root(clone, catalog=catalog),
            )
            clones.append(tail_root)
            tail_clean_root = phases.measure(
                "session-tail-clean-clone",
                lambda: _clone_from_root(clone, catalog=catalog),
            )
            clones.append(tail_clean_root)
            _assert_root_matrix(
                production,
                source_root,
                output,
                iter(clones),
            )
            _assert_raw_state_parity(
                _raw_tree_state_digest(tail_root),
                _raw_tree_state_digest(clone),
                "session-tail clone",
            )
            _assert_raw_state_parity(
                _raw_tree_state_digest(tail_clean_root),
                _raw_tree_state_digest(clone),
                "session-tail clean clone",
            )
            tail_units = _raw_units(raw_store, tail_root / "raw")
            tail_events = [
                _message("assistant", "r2 tail answer", 1),
                {"type": "unknown"},
            ]
            tail_id, tail_receipt_sha = phases.measure(
                "session-tail-event-append",
                lambda: _append_events(
                    raw_segment,
                    tail_root,
                    session_key=session_key,
                    after_line=tail_after,
                    events=tail_events,
                    tag="session-tail",
                    logical_source_file=Path("r2-session-tail.jsonl"),
                ),
            )
            _, tail_metrics = phases.measure(
                "existing-session-tail",
                lambda: _measure(
                    "existing-session-tail",
                    lambda: (
                        catalog.advance(tail_root / "raw", tail_root, context_bytes),
                        catalog.sync_historical_index(tail_root / "raw", tail_root),
                    ),
                    catalog=catalog,
                    distill=distill,
                    raw_store_module=raw_store_module,
                    store=store,
                    old_units=tail_units,
                ),
            )
            _assert_delta(tail_metrics, tail_id)
            tail_clean_id, tail_clean_receipt_sha = phases.measure(
                "session-tail-clean-event-append",
                lambda: _append_events(
                    raw_segment,
                    tail_clean_root,
                    session_key=session_key,
                    after_line=tail_after,
                    events=tail_events,
                    tag="session-tail",
                    logical_source_file=Path("r2-session-tail.jsonl"),
                ),
            )
            if (tail_id, tail_receipt_sha) != (
                tail_clean_id,
                tail_clean_receipt_sha,
            ):
                raise R2Error("session-tail paired Raw receipt identity differs")
            _remove_derived_allowlist(catalog, tail_clean_root)
            phases.measure(
                "session-tail-clean-rebuild",
                lambda: (
                    catalog.advance(tail_clean_root / "raw", tail_clean_root, context_bytes),
                    catalog.sync_historical_index(tail_clean_root / "raw", tail_clean_root),
                ),
                raw_units=len(old_units) + 1,
            )
            tail_snapshot = phases.measure(
                "session-tail-snapshot",
                lambda: _derived_snapshot(
                    catalog,
                    store,
                    raw_store_module,
                    tail_root / "raw",
                    tail_root,
                    inventory_baseline=inventory_baseline,
                    expected_new_raw_id=tail_id,
                ),
            )
            tail_clean_snapshot = phases.measure(
                "session-tail-clean-snapshot",
                lambda: _derived_snapshot(
                    catalog,
                    store,
                    raw_store_module,
                    tail_clean_root / "raw",
                    tail_clean_root,
                    inventory_baseline=inventory_baseline,
                    expected_new_raw_id=tail_clean_id,
                ),
            )
            with sqlite3.connect(
                f"file:{catalog.catalog_path(tail_root)}?mode=ro", uri=True
            ) as connection:
                if (
                    connection.execute(
                        "SELECT 1 FROM raw_units WHERE raw_id=?", (tail_id,)
                    ).fetchone()
                    is None
                ):
                    raise R2Error("session-tail Raw ID is absent from catalog")
            _assert_repair_parity(
                tail_snapshot, tail_clean_snapshot, "session-tail projection"
            )
            fault_root = phases.measure(
                "checkpoint-clone",
                lambda: _clone_from_root(clone, catalog=catalog),
            )
            clones.append(fault_root)
            _assert_root_matrix(production, source_root, output, iter(clones))
            _assert_raw_state_parity(
                _raw_tree_state_digest(fault_root),
                _raw_tree_state_digest(clone),
                "checkpoint clone",
            )
            fault = phases.measure(
                "checkpoint-repair",
                lambda: _fault_repair(
                    catalog,
                    distill,
                    store,
                    raw_store_module,
                    fault_root,
                    fault_root / "raw",
                    context_bytes,
                    old_units,
                    migration_snapshot,
                    inventory_baseline,
                ),
                raw_units=len(old_units),
            )
            tamper_root, tamper = phases.measure(
                "database-tamper-repair",
                lambda: _tamper_repair(
                    base=clone,
                    catalog=catalog,
                    distill=distill,
                    store=store,
                    raw_store_module=raw_store_module,
                    context_bytes=context_bytes,
                    old_units=old_units,
                    reference_snapshot=migration_snapshot,
                    inventory_baseline=inventory_baseline,
                ),
                raw_units=len(old_units),
            )
            clones.append(tamper_root)
            _assert_root_matrix(production, source_root, output, iter(clones))
            old_hash_root, old_hash_tamper = phases.measure(
                "old-raw-hash-tamper",
                lambda: _old_raw_hash_tamper(
                    base=clone,
                    catalog=catalog,
                    store=store,
                    raw_store_module=raw_store_module,
                    raw_segment_module=raw_segment,
                    context_bytes=context_bytes,
                    old_units=old_units,
                ),
                raw_units=len(old_units),
            )
            clones.append(old_hash_root)
            _assert_root_matrix(production, source_root, output, iter(clones))
            crash_root, crash, crash_clean_root = phases.measure(
                "post-commit-crash-recovery",
                lambda: _run_post_commit_crash(
                    base=clone,
                    source_root=source_root,
                    raw_segment=raw_segment,
                    catalog=catalog,
                    distill=distill,
                    store=store,
                    raw_store_module=raw_store_module,
                    context_bytes=context_bytes,
                    old_units=old_units,
                    inventory_baseline=inventory_baseline,
                ),
                raw_units=len(old_units) + 1,
            )
            clones.append(crash_root)
            clones.append(crash_clean_root)
            _assert_root_matrix(
                production,
                source_root,
                output,
                iter(clones),
            )
            rebuild_root, rebuild = phases.measure(
                "full-rebuild-parity",
                lambda: _full_rebuild_parity(
                    base=reference_root,
                    catalog=catalog,
                    distill=distill,
                    store=store,
                    raw_store_module=raw_store_module,
                    context_bytes=context_bytes,
                    inventory_baseline=inventory_baseline,
                    expected_new_raw_id=delta_ids[-1],
                ),
                raw_units=len(old_units) + 1,
            )
            clones.append(rebuild_root)
            _assert_root_matrix(production, source_root, output, iter(clones))
            rebuild_root_2, rebuild_2 = phases.measure(
                "full-rebuild-parity-independent",
                lambda: _full_rebuild_parity(
                    base=reference_root,
                    catalog=catalog,
                    distill=distill,
                    store=store,
                    raw_store_module=raw_store_module,
                    context_bytes=context_bytes,
                    inventory_baseline=inventory_baseline,
                    expected_new_raw_id=delta_ids[-1],
                ),
                raw_units=len(old_units) + 1,
            )
            clones.append(rebuild_root_2)
            _assert_root_matrix(production, source_root, output, iter(clones))
            reference_catalog_digest = str(reference_snapshot["catalog"]["digest"])
            reference_fts_digest = str(reference_snapshot["fts"]["digest"])
            if rebuild["catalog"].get("digest") != rebuild_2["catalog"].get(
                "digest"
            ) or rebuild["fts"].get("digest") != rebuild_2["fts"].get("digest"):
                raise R2Error("independent full rebuild digests differ")
            _assert_repair_parity(rebuild_2, rebuild, "independent full rebuild")
            if rebuild["catalog"].get("digest") != reference_catalog_digest:
                raise R2Error(
                    "full catalog rebuild digest differs from incremental projection"
                )
            if rebuild["fts"].get("digest") != reference_fts_digest:
                raise R2Error(
                    "full FTS rebuild digest differs from incremental projection"
                )

            identity_after = phases.measure(
                "source-runtime-integrity",
                lambda: parity._runtime_identity(source_root, source_commit),
            )
            R0._assert_identity_stable(identity_before, identity_after)
            source_after = phases.measure(
                "source-tree-integrity", lambda: _source_tree_digest(source_root)
            )
            if source_before != source_after:
                raise R2Error("source tree changed during clone-only run")
            production_after = phases.measure(
                "production-derived-integrity",
                lambda: _bounded_production(
                    store, catalog, raw_store, production, dashboard_url
                ),
            )
            raw_after = phases.measure(
                "production-raw-integrity", lambda: _raw_tree_digest(production)
            )
            if raw_before != raw_after:
                raise R2Error("production Raw tree changed")
            production_catalog_after = phases.measure(
                "production-catalog-integrity",
                lambda: _catalog_snapshot(catalog, production),
            )
            if production_catalog_after != old_catalog:
                raise R2Error("production catalog changed")
            static_before = dict(production_before)
            static_after = dict(production_after)
            for key in ("fast_snapshot", "live_health"):
                static_before.pop(key, None)
                static_after.pop(key, None)
            if static_before != static_after:
                raise R2Error("production derived state changed")

            payload = {
                "captured_at": datetime.now().astimezone().isoformat(),
                "runtime_identity": identity_before,
                "frozen_clone": (
                    {
                        "manifest": str(frozen_clone_manifest),
                        "artifact_id": frozen_manifest.get("artifact_id"),
                    }
                    if frozen_manifest is not None
                    else None
                ),
                "runtime_comparison": R0._runtime_comparison(
                    identity_after, production_after["live_health"]
                ),
                "production": {
                    "raw_tree": raw_after,
                    "legacy_catalog": old_catalog,
                    "catalog_after": production_catalog_after,
                    "source_tree_before": source_before,
                    "source_tree_after": source_after,
                    "before_static": static_before,
                    "after_static": static_after,
                },
                "migration": {
                    "catalog": migration_metrics,
                    "fts": fts_metrics,
                    "catalog_result": {
                        "status": migration_catalog.status,
                        "watermark": migration_catalog.watermark,
                        "indexed_count": len(migration_catalog.indexed_raw_ids),
                        "rally_count": len(migration_catalog.rally_ids),
                    },
                    "fts_digest": migration_fts,
                    "snapshot": migration_snapshot,
                },
                "warm_no_new": {
                    "samples": noop_samples,
                    "p95_wall_time_ns": noop_p95,
                    "stages": noop_metrics,
                },
                "delta_1000_events": {
                    "samples": delta_samples,
                    "p95_wall_time_ns": delta_p95,
                    "stages": delta_metrics,
                    "outcomes": delta_outcomes,
                },
                "existing_session_tail": {
                    "stage": tail_metrics,
                    "raw_id": tail_id,
                    "receipt_sha256": tail_receipt_sha,
                    "snapshot": tail_snapshot,
                    "clean_reference": tail_clean_snapshot,
                    "parity": True,
                },
                "fault_recovery": fault,
                "tamper_recovery": tamper,
                "post_commit_crash": crash,
                "old_raw_hash_tamper": old_hash_tamper,
                "full_rebuild_parity": rebuild,
                "full_rebuild_parity_independent": rebuild_2,
                "thresholds": {
                    "warm_no_new_p95_ns": 1_000_000_000,
                    "delta_1000_events_p95_ns": 15_000_000_000,
                    "warm_old_raw_logical_bytes": 0,
                    "warm_old_raw_physical_bytes": 0,
                    "warm_full_fts_scan_statements": 0,
                    "claim_p95_ns": None,
                    "teacher_handoff_ns": None,
                },
                "raw_inventory_validation": delta_inventory_validation,
            }
            cleanup_count = len(clones)

            def cleanup_clones() -> None:
                nonlocal cleanup_recorded, owned_cleanup_count
                owned_cleanup_count = _cleanup_clone_set(
                    clones, owned_temp_parent, owned_temp_prefix
                )
                cleanup_recorded = True

            phases.measure("clone-cleanup", cleanup_clones, count=cleanup_count)
            payload["clone_cleanup"] = {
                "count": cleanup_count,
                "verified": True,
                "owned_temp_count": owned_cleanup_count,
            }
            payload["phase_timing"] = {
                **phases.summary(),
                "scope": "formal-run-pre-completion",
                "artifact_seal_readback_included": False,
            }
            payload["cli_paths"] = dict(cli_paths or {})
            payload["completion_receipt"] = {
                "kind": R2_COMPLETION_KIND,
                "schema": store.DISTILLATION_SCHEMA,
                "path_template": "{artifact_id}" + R2_COMPLETION_SUFFIX,
            }
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode()
            if len(encoded) > MAX_EVIDENCE_BYTES:
                raise R2Error("R2 evidence is unexpectedly large")
            if output.exists():
                _assert_symlink_free_tree(output, label="R2 output")
            artifact_id, artifact_path, artifact = phases.measure(
                "artifact-seal",
                lambda: store.write_immutable(output, payload, schema=R2_SCHEMA),
            )
            persisted = phases.measure(
                "artifact-readback",
                lambda: _read_content_addressed_artifact(
                    store,
                    artifact_path,
                    schema=R2_SCHEMA,
                    label="R2 evidence",
                ),
            )
            completion = _write_completion_receipt(
                store,
                output,
                persisted,
                artifact_path,
                phases.summary(),
            )
            return {
                "schema": persisted["schema"],
                "artifact_id": persisted["artifact_id"],
                "path": str(artifact_path),
                "warm_p95_ns": noop_p95,
                "delta_p95_ns": delta_p95,
                "clone_cleanup_verified": True,
                "phase_timing": phases.summary(),
                "completion_receipt": completion,
            }
    finally:
        try:
            if not cleanup_recorded:
                def cleanup_after_failure() -> None:
                    nonlocal cleanup_recorded, owned_cleanup_count
                    owned_cleanup_count = _cleanup_clone_set(
                        clones, owned_temp_parent, owned_temp_prefix
                    )
                    cleanup_recorded = True

                phases.measure(
                    "clone-cleanup",
                    cleanup_after_failure,
                    count=len(clones),
                    outcome="failure-or-signal",
                )
        finally:
            try:
                phases.close()
            finally:
                _ACTIVE_CLONE_PREFIX = previous_clone_prefix
                sys.dont_write_bytecode = previous_dont_write_bytecode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dashboard-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolated-root", type=Path)
    parser.add_argument("--noop-samples", type=int, default=DEFAULT_NOOP_SAMPLES)
    parser.add_argument("--delta-samples", type=int, default=DEFAULT_DELTA_SAMPLES)
    parser.add_argument("--max-context-bytes", type=int, default=DEFAULT_CONTEXT_BYTES)
    parser.add_argument(
        "--phase-receipt",
        type=Path,
        help="append bounded phase start/finish/error events outside protected roots",
    )
    parser.add_argument(
        "--frozen-clone-root",
        type=Path,
        help="consume one externally sealed APFS clone instead of cloning production",
    )
    parser.add_argument(
        "--frozen-clone-manifest",
        type=Path,
        help="sealed clone manifest (or destination directory with --prepare-frozen-clone)",
    )
    parser.add_argument(
        "--prepare-frozen-clone",
        type=Path,
        help="create and seal an APFS clone while production remains live",
    )
    args = parser.parse_args(argv)
    if args.isolated_root is not None:
        raise R2Error("--isolated-root is intentionally unsupported; use APFS clones")
    try:
        raw_cli_paths = {
            "production_root": args.production_root,
            "source_root": args.source_root,
            "output": args.output,
            "phase_receipt": args.phase_receipt,
            "frozen_clone_root": args.frozen_clone_root,
            "frozen_clone_manifest": args.frozen_clone_manifest,
            "prepare_frozen_clone": args.prepare_frozen_clone,
        }
        cli_paths = {
            name: _cli_path_record(path)
            for name, path in raw_cli_paths.items()
            if path is not None
        }
        expanded = {name: Path(record["expanded"]) for name, record in cli_paths.items()}
        production = expanded["production_root"].resolve(strict=True)
        source = expanded["source_root"].resolve(strict=True)
        output = expanded["output"].resolve(strict=False)
        phase_path = (
            expanded["phase_receipt"].resolve(strict=False)
            if "phase_receipt" in expanded
            else None
        )
        preflight_inputs = {
            name: path
            for name, path in {
                "phase_receipt": phase_path,
                "frozen_clone_manifest": expanded.get("frozen_clone_manifest"),
                "prepare_frozen_clone": expanded.get("prepare_frozen_clone"),
            }.items()
            if path is not None
        }
        _assert_root_matrix(
            production,
            source,
            output,
            (expanded["frozen_clone_root"],)
            if "frozen_clone_root" in expanded
            else (),
            preflight_inputs,
        )
        signal_receipt = _PhaseReceipt(phase_path)
        with _SignalGuard(signal_receipt):
            if args.prepare_frozen_clone is not None:
                if args.frozen_clone_root is not None:
                    raise R2Error("--prepare-frozen-clone cannot consume --frozen-clone-root")
                if args.frozen_clone_manifest is None:
                    raise R2Error("--prepare-frozen-clone requires --frozen-clone-manifest")
                result = _prepare_frozen_clone(
                    production=production,
                    source_root=source,
                    source_commit=args.source_commit,
                    dashboard_url=args.dashboard_url,
                    clone_root=expanded["prepare_frozen_clone"].resolve(strict=False),
                    manifest_directory=expanded["frozen_clone_manifest"].resolve(strict=False),
                    cli_paths=cli_paths,
                )
            else:
                if (args.frozen_clone_root is None) != (args.frozen_clone_manifest is None):
                    raise R2Error("frozen clone root and manifest must be provided together")
                result = _run_once(
                    production=production,
                    source_root=source,
                    source_commit=args.source_commit,
                    dashboard_url=args.dashboard_url,
                    output=output,
                    noop_samples=args.noop_samples,
                    delta_samples=args.delta_samples,
                    context_bytes=args.max_context_bytes,
                    phase_receipt=phase_path,
                    frozen_clone_root=(
                        expanded["frozen_clone_root"].resolve(strict=True)
                        if "frozen_clone_root" in expanded
                        else None
                    ),
                    frozen_clone_manifest=(
                        expanded["frozen_clone_manifest"].resolve(strict=True)
                        if "frozen_clone_manifest" in expanded
                        else None
                    ),
                    cli_paths=cli_paths,
                )
        print(json.dumps(result, sort_keys=True))
    except (R2Error, OSError, ValueError, sqlite3.Error) as exc:
        print(f"r2 harness failed: {str(exc).split(':', 1)[0]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
