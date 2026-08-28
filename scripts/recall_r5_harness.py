#!/usr/bin/env python3
"""Fail-closed, payload-free formal dataset gate for Recall R5.

This is deliberately a verifier, not a backfill runner: it makes no network or
provider calls and never writes the managed root.  Materialization and
preflight are evaluated only from a harness-owned clone.  Missing provenance is
a declined capture, never an implicit pass.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import ctypes
import fcntl
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
import pwd
import shutil
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
import types
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

R5_SCHEMA = "chronovisor.recall-r5.v1"
R5_COMPLETION_SCHEMA = "chronovisor.r5-supervisor-completion.v1"
R5_FLOOR_POLICY_SCHEMA = "chronovisor.recall-r5-floor-policy.v2"
CANONICAL_TRAINING_SCHEMA = "chronovisor.recall-distill-training.v1"
CANONICAL_GATE_SCHEMA = "chronovisor.recall-offline-training-gate.v2"
NAMESPACE = "recall-distillation"
MIN_RALLIES, MIN_DAYS, MIN_WINDOWS = 1000, 30, 3
MIN_LABELS, MIN_PER_CLASS, MIN_PROBES, MIN_COUNTERFACTUALS = 500, 100, 100, 100
MAX_FILES, MAX_FILE_BYTES = 200_000, 2 * 1024 * 1024 * 1024
MAX_SNAPSHOT_SECONDS = 60
MAX_SUPERVISOR_FUTURE_SKEW_SECONDS = 300
SUPERVISOR_SCHEDULER_TOLERANCE_MS = 100
_HEX = __import__("re").compile(r"[0-9a-f]{64}\Z")
_REASON_CODE = __import__("re").compile(r"[a-z][a-z0-9_]{0,95}\Z")
_SAFE_IDENTIFIER = __import__("re").compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_SAFE_METADATA_KEY = __import__("re").compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_SAFE_METADATA_VALUE = __import__("re").compile(r"[A-Za-z0-9][A-Za-z0-9_.:+,@-]{0,255}\Z")
_SENSITIVE_METADATA_MARKERS = (
    "secret", "token", "password", "credential", "api_key", "authorization", "bearer",
)
_UTC_SECONDS = __import__("re").compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_DECLINE_REASON_CODES = {
    "eligible_native_predicate_missing_or_empty", "eligible_native_rallies_below_floor",
    "eligible_native_span_below_floor", "eligible_native_windows_below_floor",
    "invalid_or_uncertain_label_status", "label_identity_duplicate_or_missing",
    "valid_nonprobe_labels_below_floor", "materialized_label_ledger_not_exact_1_to_1",
    "label_class_below_floor", "revision_append_only_binding_missing",
    "probe_or_blind_repeat_below_floor", "sealed_counterfactual_pairs_below_floor_or_duplicate",
    "counterfactual_sealed_exposure_binding_missing", "future_leakage_detected",
    "feature_parity_not_100_percent", "source_gate_not_passed", "leased_work_remaining",
    "workset_label_reconciliation_failed", "workset_receipt_chain_not_independently_verified",
    "baseline_label_head_or_snapshot_binding_missing", "official_materialization_binding_missing",
    "independent_official_evidence_rederivation_failed", "clone_api_capture_failed",
    "canonical_dataset_policy_missing", "dataset_backlog_remaining",
}


def _trusted_bootstrap_modules() -> dict[str, Any]:
    """Pin the interpreter bootstrap objects before untrusted source imports run."""
    # Built-ins have no verifiable on-disk bytes.  Load and pin their genuine
    # interpreter identities before the isolated checkout can influence cache
    # state, rather than trusting a later origin string.
    for name in sys.builtin_module_names:
        with contextlib.suppress(ImportError):
            importlib.import_module(name)
    trusted: dict[str, Any] = {}
    for name, module in tuple(sys.modules.items()):
        spec = getattr(module, "__spec__", None)
        if (
            name in sys.builtin_module_names
            and getattr(spec, "origin", None) == "built-in"
            and getattr(spec, "loader", None) is importlib.machinery.BuiltinImporter
        ) or (
            getattr(spec, "origin", None) == "frozen"
            and getattr(spec, "loader", None) is importlib.machinery.FrozenImporter
        ):
            trusted[name] = module
    return trusted


_TRUSTED_BOOTSTRAP_MODULES = _trusted_bootstrap_modules()


def _trusted_stdlib_modules() -> dict[str, Any]:
    root = Path(sysconfig.get_path("stdlib") or "/").resolve()
    return {
        name: module
        for name, module in sys.modules.items()
        if isinstance(getattr(module, "__name__", None), str)
        and isinstance(getattr(module, "__file__", None), str)
        and Path(str(module.__file__)).resolve().is_relative_to(root)
    }


_TRUSTED_STDLIB_MODULES = _trusted_stdlib_modules()
ACCOUNT_UID = os.getuid()
ACCOUNT_HOME = Path(pwd.getpwuid(ACCOUNT_UID).pw_dir).resolve()
PRODUCTION_ROOT = ACCOUNT_HOME / ".chronovisor"
_TRUSTED_TOOLS = (
    Path("/usr/bin/git"), Path("/bin/cp"), Path("/bin/ps"), Path("/usr/bin/sandbox-exec"),
)
_PRE_SENTINEL_OPEN = os.open
_PRE_SENTINEL_WRITE = os.write
_PRE_SENTINEL_CLOSE = os.close
_PRE_SENTINEL_POPEN = subprocess.Popen
_PRE_SENTINEL_SOCKET = socket.socket


class R5Error(ValueError):
    """An R5 evidence boundary or data contract was violated."""


def _load_sibling(name: str) -> Any:
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"chronovisor_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise R5Error(f"{name} helper unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


R0 = _load_sibling("recall_r0_harness.py")
R2 = _load_sibling("recall_r2_harness.py")
R3 = _load_sibling("recall_r3_harness.py")
R4 = _load_sibling("recall_r4_harness.py")


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise R5Error("non-canonical evidence") from exc


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_id(value: object) -> bool:
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def _kernel_sandbox_attested() -> bool:
    """Ask the Darwin sandbox itself whether the required denies are active."""
    try:
        library = ctypes.CDLL("/usr/lib/libsandbox.1.dylib")
        check = library.sandbox_check
        check.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
        check.restype = ctypes.c_int
        return all(
            check(os.getpid(), operation, 0) != 0
            for operation in (b"network-outbound", b"process-fork", b"file-write*")
        )
    except (AttributeError, OSError):
        return False


def _has_symlink_component(path: Path) -> bool:
    current = path.absolute()
    while current != current.parent:
        if current.is_symlink():
            return True
        current = current.parent
    return current.is_symlink()


def _overlap(left: Path, right: Path) -> bool:
    a, b = left.resolve(strict=False), right.resolve(strict=False)
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def assert_root_matrix(
    production: Path, source: Path, output: Path, clone: Path | None = None
) -> None:
    roots = {"production": production, "source": source, "output": output}
    if clone is not None:
        roots["clone"] = clone
    for name, root in roots.items():
        if _has_symlink_component(root):
            raise R5Error(f"{name} path contains a symlink")
    entries = list(roots.items())
    for index, (name, root) in enumerate(entries):
        for other_name, other in entries[index + 1 :]:
            if _overlap(root, other):
                raise R5Error(f"{name}/{other_name} paths overlap")
    if not production.is_dir() or not source.is_dir():
        raise R5Error("production/source root unavailable")
    if output.exists() and not output.is_dir():
        raise R5Error("output is not a directory")


def _file_state(path: Path, *, digest: bool = True) -> dict[str, Any]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise R5Error(f"unsafe file: {path.name}")
    if before.st_size > MAX_FILE_BYTES:
        raise R5Error(f"file exceeds bounded evidence limit: {path.name}")
    result: dict[str, Any] = {
        "bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "dev": before.st_dev,
        "ino": before.st_ino,
    }
    if digest:
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.lstat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise R5Error(f"file changed during read: {path.name}")
        result["sha256"] = value
    return result


def tree_state(
    root: Path, *, label: str, include: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Content inventory only; no evidence payload is retained in the receipt."""
    if _has_symlink_component(root) or not root.is_dir():
        raise R5Error(f"{label} root unsafe")
    started = time.monotonic()
    records: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise R5Error(f"{label} contains symlink")
        if path.is_file():
            relative = path.relative_to(root)
            if include is not None and not any(
                relative == candidate or relative.is_relative_to(candidate)
                for candidate in include
            ):
                continue
            if len(records) >= MAX_FILES:
                raise R5Error(f"{label} file count unbounded")
            state = _file_state(path)
            records.append((relative.as_posix(), {key: state[key] for key in ("bytes", "sha256")}))
            if time.monotonic() - started > MAX_SNAPSHOT_SECONDS:
                raise R5Error(f"{label} snapshot deadline expired")
    return {
        "file_count": len(records),
        "tree_sha256": _sha(records),
        "files": _sha([(name, state.get("sha256")) for name, state in records]),
    }


_CLONE_INPUTS = (Path("raw"), Path("runtime") / "recall-distillation", Path("config.toml"))


def _clone_input_state(root: Path) -> dict[str, Any]:
    return tree_state(root, label="clone input", include=_CLONE_INPUTS)


def _stable_tree_state(root: Path, *, label: str) -> dict[str, Any]:
    """Hash the fixed R5 input set without buffering production-sized ledgers."""
    if _has_symlink_component(root) or not root.is_dir():
        raise R5Error(f"{label} root unsafe")
    started = time.monotonic()
    records: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise R5Error(f"{label} contains symlink")
        if not path.is_file():
            continue
        if len(records) >= MAX_FILES:
            raise R5Error(f"{label} file count unbounded")
        before = path.lstat()
        if before.st_size > MAX_FILE_BYTES:
            raise R5Error(f"{label} file exceeds bounded evidence limit")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        after = path.lstat()
        if (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise R5Error(f"{label} changed during read")
        records.append(
            (
                path.relative_to(root).as_posix(),
                {"bytes": before.st_size, "sha256": digest.hexdigest()},
            )
        )
        if time.monotonic() - started > MAX_SNAPSHOT_SECONDS:
            raise R5Error(f"{label} stable read deadline expired")
    return {"file_count": len(records), "tree_sha256": _sha(records)}


def _trusted_tool(path: Path) -> dict[str, Any]:
    if path not in _TRUSTED_TOOLS:
        raise R5Error("untrusted executable")
    value = path.stat()
    if (
        not path.is_file()
        or path.is_symlink()
        or value.st_uid != 0
        or value.st_mode & 0o022
    ):
        raise R5Error("trusted executable identity failed")
    return {
        "path": str(path),
        "uid": value.st_uid,
        "mode": value.st_mode & 0o7777,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _trusted_run(argv: list[str], *, cwd: Path) -> bytes:
    executable = Path(argv[0])
    _trusted_tool(executable)
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            check=True,
            capture_output=True,
            timeout=10,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"},
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise R5Error("trusted command failed") from exc


def _git_files(source: Path) -> tuple[str, str, Path, list[tuple[str, str, str]]]:
    root = (
        _trusted_run(["/usr/bin/git", "rev-parse", "--show-toplevel"], cwd=source)
        .decode()
        .strip()
    )
    if Path(root).resolve() != source.resolve():
        raise R5Error("source is not its git root")
    head = (
        _trusted_run(["/usr/bin/git", "rev-parse", "--verify", "HEAD"], cwd=source)
        .decode()
        .strip()
    )
    tree = (
        _trusted_run(["/usr/bin/git", "rev-parse", "--verify", "HEAD^{tree}"], cwd=source)
        .decode()
        .strip()
    )
    git_dir = Path(
        _trusted_run(["/usr/bin/git", "rev-parse", "--absolute-git-dir"], cwd=source)
        .decode()
        .strip()
    )
    if not git_dir.is_absolute() or _has_symlink_component(git_dir) or not git_dir.is_dir():
        raise R5Error("source git directory is unsafe")
    status = _trusted_run(
        ["/usr/bin/git", "status", "--porcelain=v1", "-z"], cwd=source
    )
    if status:
        raise R5Error("source is not clean")
    entries: list[tuple[str, str, str]] = []
    for entry in _trusted_run(["/usr/bin/git", "ls-files", "-s", "-z"], cwd=source).split(b"\0"):
        if not entry:
            continue
        try:
            header, raw_path = entry.split(b"\t", 1)
            mode, blob, stage = header.decode("ascii").split()
            relative = os.fsdecode(raw_path)
        except (UnicodeError, ValueError) as exc:
            raise R5Error("source git index entry is malformed") from exc
        if stage != "0" or len(blob) not in {40, 64} or set(blob) - set("0123456789abcdef"):
            raise R5Error("source git index entry is unsafe")
        _trusted_run(["/usr/bin/git", "cat-file", "-e", f"{blob}^{{blob}}"], cwd=source)
        entries.append((relative, mode, blob))
    if not entries:
        raise R5Error("source index is empty")
    return head, tree, git_dir, sorted(entries)


def source_state(source: Path, expected_commit: str) -> dict[str, Any]:
    if (
        not isinstance(expected_commit, str)
        or len(expected_commit) != 40
        or set(expected_commit) - set("0123456789abcdef")
    ):
        raise R5Error("source commit is invalid")
    if _has_symlink_component(source) or not source.is_dir():
        raise R5Error("source root unsafe")
    started = time.monotonic()
    head, tree, git_dir, entries = _git_files(source)
    if head != expected_commit:
        raise R5Error("source HEAD differs from requested commit")
    records: list[tuple[Any, ...]] = []
    if len(entries) > MAX_FILES:
        raise R5Error("source index file count unbounded")
    for relative, mode, blob in entries:
        path = source / relative
        if not path.is_relative_to(source) or _has_symlink_component(path):
            raise R5Error("source tracked path unsafe")
        actual_blob = _trusted_run(
            ["/usr/bin/git", "hash-object", "--no-filters", "--", relative], cwd=source
        ).decode().strip()
        if actual_blob != blob:
            raise R5Error("source worktree differs from its verified index blob")
        records.append((relative, mode, blob, _file_state(path)))
        if time.monotonic() - started > MAX_SNAPSHOT_SECONDS:
            raise R5Error("source snapshot deadline expired")
    # R4 owns the production source identity contract.  Preserve its digest
    # verbatim so R5 cannot certify a differently-hashed checkout.
    official = R4._assert_source(source, expected_commit)
    return {
        **official,
        "commit": head,
        "tree": tree,
        "clean": True,
        "index_count": len(entries),
        "index_sha256": _sha(entries),
        "git_index": _file_state(git_dir / "index"),
        "tracked_bytes_sha256": _sha(records),
        "tool_identities": [_trusted_tool(Path("/usr/bin/git"))],
    }


def _managed_inventory(root: Path) -> dict[str, Any]:
    """Hash bounded files and retain metadata identity for oversized files."""
    started = time.monotonic()
    records: list[tuple[str, int, int, int, int, str, str | None]] = []
    for base, directories, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        for name in sorted([*directories, *files]):
            path = base_path / name
            state = path.lstat()
            if len(records) >= MAX_FILES:
                raise R5Error("managed inventory file count unbounded")
            kind = (
                "symlink" if stat.S_ISLNK(state.st_mode) else "file" if stat.S_ISREG(state.st_mode)
                else "directory" if stat.S_ISDIR(state.st_mode) else "special"
            )
            digest: str | None = None
            if kind == "file" and state.st_size <= MAX_FILE_BYTES:
                hasher = hashlib.sha256()
                with path.open("rb") as handle:
                    while block := handle.read(1024 * 1024):
                        hasher.update(block)
                after = path.lstat()
                if (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
                ):
                    raise R5Error("managed inventory changed during read")
                digest = hasher.hexdigest()
            records.append((path.relative_to(root).as_posix(), state.st_mode, state.st_size, state.st_mtime_ns, state.st_ino, kind, digest))
            if time.monotonic() - started > MAX_SNAPSHOT_SECONDS:
                raise R5Error("managed inventory deadline expired")
    return {"entries": len(records), "sha256": _sha(records)}


def production_state(production: Path) -> dict[str, Any]:
    paths = {
        "raw": production / "raw",
        "runtime": production / "runtime" / "recall-distillation",
    }
    result: dict[str, Any] = {
        "root": _root_identity(production),
        "managed_inventory": _managed_inventory(production),
    }
    for name, path in paths.items():
        result[name] = (
            _stable_tree_state(path, label=f"production.{name}")
            if path.exists()
            else None
        )
    config = production / "config.toml"
    result["config"] = _file_state(config) if config.exists() else None
    result["protected_inventory"] = {
        "included": ["raw", "runtime/recall-distillation", "config.toml"],
        "excluded": ["all other production paths are metadata-inventoried only"],
        "excluded_sha256": _sha(["all other production paths are metadata-inventoried only"]),
    }
    return result


def _clone(
    production: Path, *, owned_clone: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Create only a harness-owned APFS COW clone of the protected state."""
    clone = (
        Path(tempfile.mkdtemp(prefix="chronovisor-r5-", dir=production.parent))
        if owned_clone is None
        else owned_clone
    )
    try:
        if (
            not clone.is_dir()
            or clone.is_symlink()
            or any(clone.iterdir())
            or clone.parent != production.parent
        ):
            raise R5Error("owned clone path is unsafe")
        if R0._filesystem_type(production.parent) != "apfs":
            raise R5Error("production volume is not APFS")
        copied_files = 0
        for relative in (Path("raw"), Path("runtime") / "recall-distillation"):
            source_root = production / relative
            if not source_root.is_dir() or _has_symlink_component(source_root):
                raise R5Error(f"clone source unavailable: {relative}")
            for base, directories, files in os.walk(source_root, followlinks=False):
                base_path = Path(base)
                if any((base_path / name).is_symlink() for name in directories):
                    raise R5Error("clone source contains symlinked directory")
                target_base = clone / base_path.relative_to(production)
                target_base.mkdir(parents=True, exist_ok=True)
                target_base.chmod(base_path.stat().st_mode & 0o777)
                for name in files:
                    source = base_path / name
                    if source.is_symlink() or not source.is_file():
                        raise R5Error("clone source contains unsafe file")
                    copied_files += 1
                    if copied_files > MAX_FILES:
                        raise R5Error("clone source file count unbounded")
                    R2._copyfile_clone(
                        source, target_base / name,
                        R2.COPYFILE_ALL | R2.COPYFILE_NOFOLLOW | R2.COPYFILE_CLONE_FORCE,
                    )
        source_config = production / "config.toml"
        if not source_config.is_file() or _has_symlink_component(source_config):
            raise R5Error("clone source config unavailable")
        R2._copyfile_clone(
            source_config, clone / "config.toml",
            R2.COPYFILE_ALL | R2.COPYFILE_NOFOLLOW | R2.COPYFILE_CLONE_FORCE,
        )
        if (
            _overlap(clone, production)
            or clone.stat().st_dev != production.stat().st_dev
        ):
            raise R5Error("clone overlap or volume mismatch")
        if _has_symlink_component(clone) or any(
            path.is_symlink() for path in clone.rglob("*")
        ):
            raise R5Error("clone contains symlink")
        original, copied = production_state(production), production_state(clone)
        original_content = {key: value for key, value in original.items() if key not in {"root", "managed_inventory", "protected_inventory"}}
        copied_content = {key: value for key, value in copied.items() if key not in {"root", "managed_inventory", "protected_inventory"}}
        for content in (original_content, copied_content):
            config = content.get("config")
            if isinstance(config, Mapping):
                content["config"] = {key: config.get(key) for key in ("bytes", "sha256")}
        if original_content != copied_content:
            raise R5Error("clone content parity failed")
        for source in [path for path in production.rglob("*") if path.is_file()]:
            if (
                not source.is_relative_to(production / "raw")
                and not source.is_relative_to(production / "runtime" / "recall-distillation")
                and source != production / "config.toml"
            ):
                continue
            target = clone / source.relative_to(production)
            source_stat, target_stat = source.lstat(), target.lstat()
            if source_stat.st_ino == target_stat.st_ino or target_stat.st_nlink != 1:
                raise R5Error("clone regular-file hardlink boundary failed")
            if target_stat.st_uid != ACCOUNT_UID or (target_stat.st_mode & 0o7777) != (
                source_stat.st_mode & 0o7777
            ):
                raise R5Error("clone ownership or mode mismatch")
        root = clone.stat()
        return clone, {
            "owned": True,
            "cow": "copyfile(3):COPYFILE_CLONE_FORCE",
            "dev": root.st_dev,
            "ino": root.st_ino,
            "volume": "apfs",
            "tool": {"backend": "copyfile(3)", "flags": R2.COPYFILE_CLONE_FORCE},
            # The clone has different root metadata by construction.  Its
            # receipt therefore binds to the source production snapshot, not
            # a self-description of the clone.
            "parity": original,
        }
    except Exception:
        _cleanup_clone(clone)
        raise


def _cleanup_clone(clone: Path) -> None:
    if os.path.lexists(clone):
        if clone.is_symlink():
            raise R5Error("clone cleanup path is symlinked")
        shutil.rmtree(clone)
    if os.path.lexists(clone):
        raise R5Error("clone cleanup failed")


def _module_origin_allowed(module: Any, source: Path) -> bool:
    """Only the checkout, stdlib, or current isolated environment may load code."""
    origin = getattr(module, "__file__", None)
    if origin is None:
        namespace_paths = getattr(module, "__path__", None)
        if namespace_paths is not None:
            try:
                namespace_roots = ((source / "src").resolve(), Path(sys.prefix).resolve())
                return all(
                    any(Path(str(path)).resolve().is_relative_to(root) for root in namespace_roots)
                    for path in namespace_paths
                )
            except (OSError, ValueError):
                return False
        spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
        name = getattr(module, "__name__", "")
        loader = getattr(getattr(module, "__spec__", None), "loader", None)
        return (
            spec_origin == "built-in"
            and name in sys.builtin_module_names
            and loader is importlib.machinery.BuiltinImporter
        ) or (
            spec_origin == "frozen"
            and loader is importlib.machinery.FrozenImporter
        )
    try:
        path = Path(str(origin)).resolve()
        roots: Sequence[Path] = [
            source / "src", Path(sysconfig.get_path("stdlib") or "/"), Path(sys.prefix),
        ]
        return any(path.is_relative_to(root.resolve()) for root in roots)
    except (OSError, ValueError):
        return False


def _module_provenance_allowed(
    module: Any, source: Path,
) -> bool:
    """Bind a loaded module to an on-disk loader, not mutable module attributes."""
    spec = getattr(module, "__spec__", None)
    name = getattr(module, "__name__", "")
    if not isinstance(name, str) or sys.modules.get(name) is not module:
        return False
    if name in _TRUSTED_STDLIB_MODULES:
        return _TRUSTED_STDLIB_MODULES[name] is module
    if (
        spec is None
        and isinstance(name, str)
        and name.startswith("sys.")
        and getattr(sys, name.removeprefix("sys."), None) is module
    ):
        return _TRUSTED_BOOTSTRAP_MODULES.get(name) is module
    if (
        isinstance(name, str)
        and (name == "cython_runtime" or name.startswith("_cython_"))
    ):
        return True
    if (
        getattr(spec, "origin", None) == "frozen"
        and getattr(spec, "loader", None) is importlib.machinery.FrozenImporter
    ):
        # The origin/loader fields are mutable Python attributes.  A frozen
        # module is trusted only when its interpreter object was pinned before
        # the isolated source import and remains the canonical cache entry.
        return _TRUSTED_BOOTSTRAP_MODULES.get(name) is module
    if not _module_origin_allowed(module, source):
        return False
    origin = getattr(module, "__file__", None)
    if origin is None:
        namespace_paths = getattr(module, "__path__", None)
        if namespace_paths is not None:
            return _module_origin_allowed(module, source) and module is sys.modules.get(name)
        return (
            getattr(spec, "origin", None) == "built-in"
            and name in sys.builtin_module_names
            and getattr(spec, "loader", None) is importlib.machinery.BuiltinImporter
            and _TRUSTED_BOOTSTRAP_MODULES.get(name) is module
        )
    if spec is None or spec.loader is None or not isinstance(spec.origin, str):
        return False
    try:
        path = Path(str(origin)).resolve(strict=True)
        if Path(spec.origin).resolve(strict=True) != path:
            return False
        located = importlib.util.find_spec(str(getattr(module, "__name__", "")))
        if located is None or not isinstance(located.origin, str):
            return False
        if Path(located.origin).resolve(strict=True) != path:
            return False
        get_data = getattr(spec.loader, "get_data", None)
        return callable(get_data) and get_data(str(path)) == path.read_bytes()
    except (ImportError, OSError, TypeError, ValueError):
        return False


def _is_stdlib_module(module: Any) -> bool:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str):
        return False
    try:
        return Path(origin).resolve(strict=True).is_relative_to(
            Path(sysconfig.get_path("stdlib") or "/").resolve(strict=True)
        )
    except (OSError, ValueError):
        return False


def _module_references(module: Any) -> set[Any]:
    """Follow module objects held by this import, including preloaded fakes."""
    referenced: set[Any] = set()
    for value in vars(module).values():
        if isinstance(value, type(sys)):
            if module is sys and sys.modules.get(getattr(value, "__name__", "")) is not value:
                continue
            referenced.add(value)
        elif callable(value):
            owner_name = getattr(value, "__module__", "")
            owner = sys.modules.get(owner_name)
            if isinstance(owner_name, str) and owner_name.startswith("chronovisor.") and owner is not None:
                referenced.add(owner)
    return referenced


def _canonicalize_runtime_module_references(modules: Iterable[Any], source: Path) -> None:
    """Replace stale module aliases held by fresh source modules with cache canonicals."""
    for owner in modules:
        if not str(getattr(owner, "__name__", "")).startswith("chronovisor."):
            continue
        for attribute, value in tuple(vars(owner).items()):
            if not isinstance(value, type(sys)):
                continue
            name = getattr(value, "__name__", "")
            canonical = sys.modules.get(name)
            if canonical is value:
                continue
            if canonical is None or not _module_provenance_allowed(canonical, source):
                continue
            try:
                setattr(owner, attribute, canonical)
            except (AttributeError, TypeError):
                raise R5Error("runtime module reference cannot be canonicalized") from None


def _reload_stdlib_references(modules: Iterable[Any]) -> None:
    """Discard source-created stdlib lookalikes before canonicalizing aliases."""
    names = {
        str(getattr(value, "__name__", ""))
        for module in modules
        for value in _module_references(module)
        if _is_stdlib_module(value)
    }
    for name in sorted(names):
        trusted = _TRUSTED_STDLIB_MODULES.get(name)
        if trusted is not None:
            for child_name, child in _TRUSTED_STDLIB_MODULES.items():
                if child_name == name or child_name.startswith(f"{name}."):
                    sys.modules[child_name] = child
            continue
        sys.modules.pop(name, None)
        _TRUSTED_STDLIB_MODULES[name] = importlib.import_module(name)


def _source_import_candidates(source: Path) -> set[str]:
    """Parse the checkout's import graph before execution so cache objects cannot win."""
    candidates = {"chronovisor"}
    src = source / "src"
    pending = [
        src / "chronovisor" / "recall" / f"{name}.py"
        for name in ("recall_distillation", "recall_distillation_store", "recall_distillation_workset")
    ]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise R5Error("runtime import graph is unreadable") from exc
        module = ".".join(path.relative_to(src).with_suffix("").parts)
        package = module.rsplit(".", 1)[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                candidates.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                candidates.add(node.module)
            elif isinstance(node, ast.ImportFrom) and node.level:
                parts = package.split(".")
                base = parts[: max(0, len(parts) - node.level + 1)]
                if node.module:
                    candidates.add(".".join([*base, node.module]))
                else:
                    candidates.update(".".join([*base, alias.name]) for alias in node.names)
        for candidate in tuple(candidates):
            if not candidate.startswith("chronovisor"):
                continue
            relative = Path(*candidate.split("."))
            for local in (src / relative.with_suffix(".py"), src / relative / "__init__.py"):
                if local.is_file() and local not in seen:
                    pending.append(local)
    return candidates


def _evict_import_candidates(candidates: set[str]) -> None:
    for name, module in list(sys.modules.items()):
        if any(name == candidate or name.startswith(f"{candidate}.") for candidate in candidates):
            trusted = _TRUSTED_BOOTSTRAP_MODULES.get(name)
            if trusted is module:
                continue
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _chronovisor_module_snapshot() -> dict[str, Any]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "chronovisor" or name.startswith("chronovisor.")
    }


def _restore_chronovisor_modules(snapshot: Mapping[str, Any]) -> None:
    current = _chronovisor_module_snapshot()
    for name in tuple(sys.modules):
        if name == "chronovisor" or name.startswith("chronovisor."):
            sys.modules.pop(name, None)
    sys.modules.update(snapshot)
    for name, module in current.items():
        parent_name, _, attribute = name.rpartition(".")
        parent = snapshot.get(parent_name)
        if name not in snapshot and parent is not None and getattr(parent, attribute, None) is module:
            with contextlib.suppress(AttributeError):
                delattr(parent, attribute)
    for name, module in snapshot.items():
        parent_name, _, attribute = name.rpartition(".")
        parent = snapshot.get(parent_name)
        if parent is not None:
            setattr(parent, attribute, module)


def _load_runtime(source: Path) -> tuple[Any, Any, Any]:
    source_path = str(source / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    names = (
        "chronovisor.recall.recall_distillation",
        "chronovisor.recall.recall_distillation_store",
        "chronovisor.recall.recall_distillation_workset",
    )
    import_candidates = _source_import_candidates(source)
    _evict_import_candidates(import_candidates)
    before = dict(sys.modules)
    original_import = builtins.__import__
    evicted_dynamic: set[str] = set()
    importing: set[str] = set()

    def isolated_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        caller = globals.get("__name__") if isinstance(globals, Mapping) else ""
        source_request = (
            isinstance(caller, str) and caller.startswith("chronovisor.")
        ) or (
            globals is None and any(active.startswith("chronovisor.") for active in importing)
        )
        nested_source_import = any(
            name == active or name.startswith(f"{active}.") or active.startswith(f"{name}.")
            for active in importing
        )
        if (
            level == 0
            and source_request
            and not nested_source_import
            and not name.startswith("chronovisor.")
            and name not in import_candidates
            and name not in evicted_dynamic
        ):
            # Origin labels on an arbitrary cache object are not authority.  Every
            # absolute import starts fresh except a pinned interpreter bootstrap
            # object (not merely a matching origin string).
            # This must remain cache-only: invalidate_caches() from __import__ can
            # recursively import importlib.metadata before the runtime loads.
            trusted = _TRUSTED_BOOTSTRAP_MODULES.get(name)
            if trusted is not None:
                sys.modules[name] = trusted
            else:
                for cached in list(sys.modules):
                    if cached == name or cached.startswith(f"{name}."):
                        sys.modules.pop(cached, None)
            evicted_dynamic.add(name)
        if level != 0:
            return original_import(name, globals, locals, fromlist, level)
        importing.add(name)
        try:
            return original_import(name, globals, locals, fromlist, level)
        finally:
            importing.discard(name)

    builtins.__import__ = isolated_import
    try:
        def load_fresh(name: str) -> Any:
            importing.add(name)
            try:
                return importlib.import_module(name)
            finally:
                importing.discard(name)

        distill, store, workset = (load_fresh(name) for name in names)
    except (ImportError, OSError, ValueError) as exc:
        raise R5Error("runtime transitive dependency escaped allowed origins") from exc
    finally:
        builtins.__import__ = original_import
    roots = {distill, store, workset}
    _reload_stdlib_references(
        module for name, module in sys.modules.items() if name.startswith("chronovisor.")
    )
    _canonicalize_runtime_module_references(
        (module for name, module in sys.modules.items() if name.startswith("chronovisor.")), source,
    )
    closure = set(roots)
    closure.update(module for name, module in sys.modules.items() if name not in before)
    pending = list(closure)
    while pending:
        module = pending.pop()
        if not str(getattr(module, "__name__", "")).startswith("chronovisor."):
            continue
        for dependency in _module_references(module):
            if dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)
    if any(
        not _module_provenance_allowed(module, source)
        for module in closure
    ):
        raise R5Error("runtime transitive dependency escaped allowed origins")
    if any(
        not _module_provenance_allowed(module, source)
        for module in roots
    ):
        raise R5Error("runtime escaped source checkout")
    return distill, store, workset


@contextlib.contextmanager
def _egress_sentinel(*write_roots: Path) -> Any:
    """Count and block transport/process attempts during source execution."""
    attempts = {"provider_calls": 0, "egress_attempts": 0, "process_attempts": 0}
    original_connect, original_run, original_popen = (
        socket.socket.connect,
        subprocess.run,
        subprocess.Popen,
    )
    original_system = os.system
    original_open, original_write, original_close = os.open, os.write, os.close
    original_file_writes = {
        name: getattr(os, name)
        for name in ("sendfile", "writev", "pwrite", "pwritev")
        if hasattr(os, name)
    }
    original_posix_spawn = getattr(os, "posix_spawn", None)
    original_posix_spawnp = getattr(os, "posix_spawnp", None)
    original_exec = {
        name: getattr(os, name)
        for name in (
            "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp",
            "execvpe", "fork", "forkpty", "setsid", "spawnl", "spawnle", "spawnlp",
            "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
        )
        if hasattr(os, name)
    }
    original_create_connection = socket.create_connection
    original_connect_ex = socket.socket.connect_ex
    original_sends = {
        name: getattr(socket.socket, name)
        for name in ("send", "sendall", "sendto", "sendmsg", "sendfile", "send_fds")
        if hasattr(socket.socket, name)
    }
    original_socket_helpers = {
        name: getattr(socket, name)
        for name in ("send_fds",)
        if hasattr(socket, name)
    }

    def blocked_connect(*_args: Any, **_kwargs: Any) -> None:
        attempts["egress_attempts"] += 1
        raise R5Error("network egress blocked")

    def blocked_process(*_args: Any, **_kwargs: Any) -> Any:
        attempts["process_attempts"] += 1
        raise R5Error("process execution blocked")

    roots = tuple(root.resolve() for root in write_roots)
    root_identities = {
        (root.stat().st_dev, root.stat().st_ino): root
        for root in roots
    }
    owned_fds: set[int] = set()

    def root_for_dir_fd(fd: int) -> Path | None:
        try:
            state = os.fstat(fd)
        except OSError:
            return None
        direct = root_identities.get((state.st_dev, state.st_ino))
        if direct is not None:
            return direct
        try:
            if sys.platform == "darwin":
                raw = fcntl.fcntl(
                    fd, getattr(fcntl, "F_GETPATH", 50), b"\0" * 1024
                )
                if not isinstance(raw, bytes):
                    return None
                path = raw.split(b"\0", 1)[0].decode("utf-8", "strict")
            else:
                path = os.readlink(f"/proc/self/fd/{fd}")
            resolved = Path(path).resolve(strict=True)
        except (OSError, UnicodeError, ValueError):
            return None
        return resolved if any(resolved.is_relative_to(root) for root in roots) else None

    def resolved_path(path: Any, *, dir_fd: int | None = None) -> Path | None:
        try:
            candidate = Path(os.fspath(path))
            if not candidate.is_absolute():
                if dir_fd is None:
                    candidate = Path.cwd() / candidate
                else:
                    parent = root_for_dir_fd(dir_fd)
                    if parent is None:
                        return None
                    candidate = parent / candidate
            return candidate.resolve(strict=False)
        except (OSError, TypeError, ValueError):
            return None

    def owned_path(path: Any, *, dir_fd: int | None = None) -> bool:
        resolved = resolved_path(path, dir_fd=dir_fd)
        return resolved is not None and any(resolved.is_relative_to(root) for root in roots)

    def owned_fd(fd: int) -> bool:
        return fd in owned_fds or owned_path(f"/dev/fd/{fd}")

    def guarded_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        dir_fd = kwargs.get("dir_fd")
        if writing and not owned_path(path, dir_fd=dir_fd if isinstance(dir_fd, int) else None):
            attempts["process_attempts"] += 1
            raise R5Error("filesystem mutation outside owned roots blocked")
        fd = original_open(path, flags, *args, **kwargs)
        if writing:
            owned_fds.add(fd)
        return fd

    def guarded_write(fd: int, data: Any) -> int:
        if not owned_fd(fd):
            attempts["process_attempts"] += 1
            raise R5Error("filesystem mutation outside owned roots blocked")
        return original_write(fd, data)

    def guarded_file_write(out_fd: int, *args: Any, **kwargs: Any) -> Any:
        if not owned_fd(out_fd):
            attempts["process_attempts"] += 1
            raise R5Error("filesystem mutation outside owned roots blocked")
        name = kwargs.pop("_r5_name")
        return original_file_writes[name](out_fd, *args, **kwargs)

    def guarded_close(fd: int) -> None:
        owned_fds.discard(fd)
        original_close(fd)

    socket.socket.connect = blocked_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked_connect  # type: ignore[method-assign,assignment]
    socket.create_connection = blocked_connect  # type: ignore[assignment]
    for name in original_sends:
        setattr(socket.socket, name, blocked_connect)
    for name in original_socket_helpers:
        setattr(socket, name, blocked_connect)
    subprocess.run = blocked_process
    subprocess.Popen = blocked_process  # type: ignore[assignment,misc]
    os.system = blocked_process
    os.open, os.write, os.close = guarded_open, guarded_write, guarded_close
    for name in original_file_writes:
        setattr(os, name, lambda out_fd, *args, _name=name, **kwargs: guarded_file_write(
            out_fd, *args, _r5_name=_name, **kwargs
        ))
    if original_posix_spawn is not None:
        os.posix_spawn = blocked_process
    if original_posix_spawnp is not None:
        os.posix_spawnp = blocked_process
    for name in original_exec:
        setattr(os, name, blocked_process)
    old = os.environ.get("PYTHONDONTWRITEBYTECODE")
    old_dont_write_bytecode = sys.dont_write_bytecode
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    try:
        yield attempts
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection
        for name, function in original_sends.items():
            setattr(socket.socket, name, function)
        for name, function in original_socket_helpers.items():
            setattr(socket, name, function)
        subprocess.run, subprocess.Popen = original_run, original_popen  # type: ignore[misc]
        os.system = original_system
        os.open, os.write, os.close = original_open, original_write, original_close
        for name, function in original_file_writes.items():
            setattr(os, name, function)
        if original_posix_spawn is not None:
            os.posix_spawn = original_posix_spawn
        if original_posix_spawnp is not None:
            os.posix_spawnp = original_posix_spawnp
        for name, function in original_exec.items():
            setattr(os, name, function)
        sys.dont_write_bytecode = old_dont_write_bytecode
        if old is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = old


@contextlib.contextmanager
def _provider_sentinel(attempts: dict[str, int], *modules: Any) -> Any:
    """Every runtime adapter generation/evaluation is a counted hard failure."""
    patched: list[tuple[Any, str, Any, bool]] = []
    seen: set[int] = set()
    provider_names = {"provider", "evaluate", "generate", "compare"}
    provider_containers = {"provider", "adapter", "backend", "client", "holder", "runtime"}

    def blocked_provider(*_args: Any, **_kwargs: Any) -> Any:
        attempts["provider_calls"] += 1
        raise R5Error("provider evaluation blocked")

    def provider_like_name(name: object) -> bool:
        return isinstance(name, str) and name.casefold() in provider_names

    def callable_provider_signature(raw: Any, owner: Any) -> bool:
        """Recognize provider entry points even when an innocuous alias holds them."""
        candidate = raw.fget if isinstance(raw, property) else raw
        if isinstance(candidate, (staticmethod, classmethod)):
            candidate = candidate.__func__
        if not callable(candidate):
            return False
        callable_name = getattr(candidate, "__name__", "")
        callable_qualname = getattr(candidate, "__qualname__", "")
        if provider_like_name(callable_name) or any(
            provider_like_name(part) for part in str(callable_qualname).split(".")
        ):
            return True
        owner_type = owner if isinstance(owner, type) else type(owner)
        return any(
            "provider" in base.__name__.casefold() or "adapter" in base.__name__.casefold()
            for base in owner_type.__mro__
        ) and "provider" in str(getattr(candidate, "__module__", "")).casefold()

    def direct_attributes(value: Any) -> Mapping[str, Any]:
        result = dict(attributes(value))
        value_type = value if isinstance(value, type) else type(value)
        for base in value_type.__mro__:
            for name, raw in vars(base).items():
                result.setdefault(name, raw)
        return result

    def patch_provider_callable(target: Any, name: str, raw: Any) -> None:
        """Replace a named or provider-signed boundary, including inherited aliases."""
        if name == "__annotate__":
            return
        try:
            original = getattr(target, name)
        except AttributeError:
            return
        if not (
            provider_like_name(name)
            or callable_provider_signature(raw, target)
            or callable_provider_signature(original, target)
        ):
            return
        try:
            patched.append((target, name, original, name in vars(target)))
            setattr(target, name, blocked_provider)
        except (AttributeError, TypeError):
            return

    def slot_names(value: type[Any]) -> Iterable[str]:
        for base in value.__mro__:
            raw = vars(base).get("__slots__", ())
            names = (raw,) if isinstance(raw, str) else raw
            for name in names:
                if isinstance(name, str):
                    yield name

    def attributes(value: Any) -> Mapping[str, Any]:
        if isinstance(value, (types.ModuleType, type)):
            return cast(Mapping[str, Any], vars(value))
        result = dict(vars(value)) if hasattr(value, "__dict__") else {}
        for name in slot_names(type(value)):
            if name not in result and hasattr(value, name):
                result[name] = getattr(value, name)
        return result

    def provider_candidate(value: Any) -> bool:
        value_type = value if isinstance(value, type) else type(value)
        return (
            "adapter" in value_type.__name__.casefold()
            or "provider" in value_type.__name__.casefold()
            or any(
                provider_like_name(name) or callable_provider_signature(raw, value)
                for name, raw in direct_attributes(value).items()
            )
        )

    def patch_reachable(value: Any, budget: list[int]) -> None:
        if id(value) in seen:
            return
        if budget[0] <= 0:
            raise R5Error("provider boundary graph exceeds bounded inspection")
        seen.add(id(value))
        budget[0] -= 1
        if isinstance(value, (types.ModuleType, type)):
            values = direct_attributes(value)
            for name, raw in values.items():
                patch_provider_callable(value, name, raw)
        elif hasattr(value, "__dict__") or hasattr(type(value), "__slots__"):
            patch_reachable(type(value), budget)
            values = direct_attributes(value)
            for name, raw in values.items():
                patch_provider_callable(value, name, raw)
        else:
            return
        for name, child in values.items():
            if (
                name.casefold() in provider_containers
                or provider_like_name(name)
                or provider_candidate(child)
            ) and (
                isinstance(child, (types.ModuleType, type))
                or hasattr(child, "__dict__")
                or hasattr(type(child), "__slots__")
            ):
                patch_reachable(child, budget)

    def patch_module(module: Any) -> None:
        patch_reachable(module, [512])

    original_import = builtins.__import__
    original_import_module = importlib.import_module
    fixed_modules = set(sys.modules)

    class DenyNewImportFinder:
        def find_spec(self, fullname: str, _path: Any = None, _target: Any = None) -> Any:
            if fullname not in fixed_modules:
                raise R5Error("provider boundary blocks dynamic imports")
            return None

    import_finder = DenyNewImportFinder()

    def reject_new_modules(before: set[str]) -> None:
        if any(name not in fixed_modules for name in set(sys.modules) - before):
            raise R5Error("provider boundary blocks dynamic imports")

    def guarded_import(*args: Any, **kwargs: Any) -> Any:
        before = set(sys.modules)
        module = original_import(*args, **kwargs)
        reject_new_modules(before)
        patch_module(module)
        return module

    def guarded_import_module(*args: Any, **kwargs: Any) -> Any:
        before = set(sys.modules)
        module = original_import_module(*args, **kwargs)
        reject_new_modules(before)
        patch_module(module)
        return module

    for module in modules:
        patch_module(module)
    builtins.__import__ = guarded_import
    importlib.import_module = guarded_import_module
    sys.meta_path.insert(0, import_finder)
    try:
        yield
    finally:
        builtins.__import__ = original_import
        importlib.import_module = original_import_module
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(import_finder)
        for target, name, original, was_direct in reversed(patched):
            if was_direct:
                setattr(target, name, original)
            else:
                delattr(target, name)


def _read_rows(store: Any, root: Path) -> list[dict[str, Any]]:
    ledger = store.distillation_dir(root) / "label-ledger.jsonl"
    try:
        return [dict(row) for row in store.read_chain(ledger)]
    except Exception as exc:
        raise R5Error("label ledger unavailable") from exc


_WORKSET_STATES = ("ready", "leased", "completed", "quarantined")
_WORKSET_OPERATIONS = {"advance", "claim_reclaim", "claim", "release", "commit"}
_WORKSET_RECEIPT_COLUMNS = (
    "generation", "previous_sha256", "operation", "payload_json", "receipt_sha256",
)
_WORKSET_ITEM_COLUMNS = (
    "sequence", "work_id", "kind", "payload_ref", "payload_digest", "temporal_split_json",
    "provenance_json", "priority", "watermark_json", "stage", "state", "attempt_count",
    "last_error_class", "lease_id", "lease_owner", "lease_expires_at", "next_attempt_at",
    "completion_ref", "completion_digest", "created_at", "updated_at",
)
_WORKSET_STATE_COLUMNS = ("key", "value_json")


def _receipt_metadata(value: object, field: str, *, key: str = "", depth: int = 0) -> Any:
    """Independently validate the payload-free metadata grammar in receipt bytes."""
    if depth > 3:
        raise R5Error(f"{field} is too deeply nested")
    if isinstance(value, Mapping):
        if len(value) > 32:
            raise R5Error(f"{field} has too many entries")
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if (
                not isinstance(raw_key, str)
                or _SAFE_METADATA_KEY.fullmatch(raw_key) is None
                or any(marker in raw_key.casefold() for marker in _SENSITIVE_METADATA_MARKERS)
            ):
                raise R5Error(f"{field} has an unsafe metadata key")
            result[raw_key] = _receipt_metadata(raw_value, field, key=raw_key, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            raise R5Error(f"{field} has too many entries")
        return [_receipt_metadata(item, field, key=key, depth=depth + 1) for item in value]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > 1_000_000_000:
            raise R5Error(f"{field} integer is invalid")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > 1_000_000_000_000:
            raise R5Error(f"{field} number is invalid")
        return value
    if (
        isinstance(value, str)
        and len(value.encode("utf-8")) <= 256
        and all(ord(character) >= 0x20 and ord(character) != 0x7F for character in value)
        and not any(marker in value.casefold() for marker in _SENSITIVE_METADATA_MARKERS)
        and (not value or _SAFE_METADATA_VALUE.fullmatch(value) is not None)
    ):
        return value
    raise R5Error(f"{field} contains unsafe metadata")


def _receipt_progress(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "cursor", "ledger_heads", "provenance", "progress_kind",
    }:
        raise R5Error(f"{field} is not a closed progress object")
    heads = value["ledger_heads"]
    if not isinstance(heads, Mapping):
        raise R5Error(f"{field} ledger heads are invalid")
    normalized_heads: dict[str, str] = {}
    for key, digest in heads.items():
        if not isinstance(key, str) or _SAFE_METADATA_KEY.fullmatch(key) is None:
            raise R5Error(f"{field} ledger heads are invalid")
        if digest != "" and not _is_id(digest):
            raise R5Error(f"{field} ledger heads are invalid")
        normalized_heads[key] = digest
    kind = value["progress_kind"]
    if not isinstance(kind, str) or _SAFE_IDENTIFIER.fullmatch(kind) is None:
        raise R5Error(f"{field} kind is invalid")
    return {
        "cursor": _receipt_metadata(value["cursor"], f"{field}.cursor"),
        "ledger_heads": normalized_heads,
        "provenance": _receipt_metadata(value["provenance"], f"{field}.provenance"),
        "progress_kind": kind,
    }


def _rederive_workset_receipts(
    connection: sqlite3.Connection, counts: Mapping[str, int],
) -> tuple[int, str]:
    """Verify the receipt ledger from cloned SQLite bytes, without its runtime API."""
    columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(workset_receipts)"))
    if columns != _WORKSET_RECEIPT_COLUMNS:
        raise R5Error("workset receipt schema is not closed")
    rows = connection.execute(
        "SELECT generation, previous_sha256, operation, payload_json, receipt_sha256 "
        "FROM workset_receipts ORDER BY generation ASC"
    ).fetchall()
    if rows:
        work_columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(work_items)"))
        state_columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(workset_state)"))
        if work_columns != _WORKSET_ITEM_COLUMNS or state_columns != _WORKSET_STATE_COLUMNS:
            raise R5Error("workset runtime schema is not closed")
        for stage, state, completion_ref, completion_digest in connection.execute(
            "SELECT stage, state, completion_ref, completion_digest FROM work_items"
        ):
            if stage not in {"snapshot", "teacher", "counterfactual", "retry_wait", "dataset", "evaluation"} or state not in _WORKSET_STATES:
                raise R5Error("workset row stage or outcome is invalid")
            if state == "completed" and (
                not isinstance(completion_ref, str)
                or not completion_ref
                or not _is_id(completion_digest)
            ):
                raise R5Error("workset completion outcome is invalid")
    final_state: dict[str, Any] = {"watermark": None, "progress": None}
    if rows:
        state_rows = connection.execute("SELECT key, value_json FROM workset_state").fetchall()
        if any(not isinstance(key, str) or not isinstance(raw, str) for key, raw in state_rows):
            raise R5Error("workset state is invalid")
        raw_state = dict(state_rows)
        if set(raw_state) - {"watermark", "progress"}:
            raise R5Error("workset state is not closed")
        for key in ("watermark", "progress"):
            raw = raw_state.get(key)
            if raw is None:
                continue
            try:
                decoded = json.loads(raw)
            except (RecursionError, json.JSONDecodeError) as exc:
                raise R5Error("workset state JSON is invalid") from exc
            final_state[key] = (
                _receipt_metadata(decoded, "workset state watermark")
                if key == "watermark" else _receipt_progress(decoded, "workset state progress")
            )
    previous = ""
    prior_after: Mapping[str, Any] | None = None
    prior_version = 1
    for generation, row in enumerate(rows, start=1):
        row_generation, row_previous, operation, payload_json, receipt = row
        if (
            not isinstance(row_generation, int)
            or isinstance(row_generation, bool)
            or row_generation != generation
            or row_previous != previous
            or not isinstance(operation, str)
            or operation not in _WORKSET_OPERATIONS
            or not isinstance(payload_json, str)
            or not _is_id(receipt)
        ):
            raise R5Error("workset receipt chain is invalid")
        try:
            payload = json.loads(payload_json)
        except (RecursionError, json.JSONDecodeError) as exc:
            raise R5Error("workset receipt JSON is invalid") from exc
        if not isinstance(payload, Mapping) or _canonical(payload).decode() != payload_json:
            raise R5Error("workset receipt JSON is not canonical")
        envelope = {
            "generation": generation,
            "previous_sha256": previous,
            "operation": operation,
            "payload": payload,
        }
        if _sha(envelope) != receipt:
            raise R5Error("workset receipt hash does not bind payload")
        allowed = {"before", "after", "delta", "details"}
        version = payload.get("version", 1)
        if version == 2:
            allowed.add("version")
            if payload.get("bootstrap") is True:
                allowed.add("bootstrap")
        if version not in (1, 2) or version < prior_version:
            raise R5Error("workset receipt revision is invalid")
        if set(payload) != allowed or not isinstance(payload.get("details"), Mapping):
            raise R5Error("workset receipt payload schema is not closed")
        details = payload["details"]
        bootstrap = payload.get("bootstrap") is True

        def snapshot(
            value: object, label: str, *, before_snapshot: bool,
            version: int = version, bootstrap: bool = bootstrap,
        ) -> Mapping[str, Any]:
            expected = {"counts", "watermark"}
            if version == 2:
                expected.add("progress")
            if not isinstance(value, Mapping) or set(value) != expected:
                raise R5Error(f"workset receipt {label} snapshot is invalid")
            raw_counts = value.get("counts")
            if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(_WORKSET_STATES):
                raise R5Error("workset receipt state counts are invalid")
            if any(
                not isinstance(raw_counts[state], int)
                or isinstance(raw_counts[state], bool)
                or raw_counts[state] < 0
                for state in _WORKSET_STATES
            ):
                raise R5Error("workset receipt state counts are invalid")
            result: dict[str, Any] = {
                "counts": dict(raw_counts),
                "watermark": _receipt_metadata(value["watermark"], "workset receipt watermark"),
            }
            if version == 2:
                progress = value["progress"]
                if before_snapshot and bootstrap:
                    if progress is not None:
                        raise R5Error("workset receipt bootstrap progress is invalid")
                    result["progress"] = None
                else:
                    result["progress"] = _receipt_progress(progress, "workset receipt progress")
            return result

        before = snapshot(payload.get("before"), "before", before_snapshot=True)
        after = snapshot(payload.get("after"), "after", before_snapshot=False)
        delta = payload.get("delta")
        if not isinstance(delta, Mapping) or set(delta) != set(_WORKSET_STATES) or any(
            not isinstance(delta[state], int) or isinstance(delta[state], bool)
            for state in _WORKSET_STATES
        ):
            raise R5Error("workset receipt transition is invalid")
        expected_delta = {
            state: after["counts"][state] - before["counts"][state]
            for state in _WORKSET_STATES
        }
        if dict(delta) != expected_delta:
            raise R5Error("workset receipt transition is discontinuous")
        if prior_after is not None and (
            before["counts"] != prior_after["counts"]
            or _canonical(before["watermark"]) != _canonical(prior_after["watermark"])
            or (
                version == 2 and prior_version == 2
                and _canonical(before["progress"]) != _canonical(prior_after["progress"])
            )
            or (
                version == 2 and prior_version == 1
                and before["progress"] is not None
            )
        ):
            raise R5Error("workset receipt chain is discontinuous")
        if operation == "advance":
            expected_details = {"inserted", "rebound", "watermark_changed", "selection_sha256"}
            if version == 2:
                expected_details.add("progress_changed")
            if (
                set(details) != expected_details
                or not all(isinstance(details[key], int) and not isinstance(details[key], bool) and details[key] >= 0 for key in ("inserted", "rebound"))
                or not isinstance(details["watermark_changed"], bool)
                or not _is_id(details["selection_sha256"])
                or (version == 2 and not isinstance(details["progress_changed"], bool))
                or details["watermark_changed"] != (
                    _canonical(before["watermark"]) != _canonical(after["watermark"])
                )
                or (version == 2 and details["progress_changed"] != (
                    _canonical(before["progress"]) != _canonical(after["progress"])
                ))
                or not (details["inserted"] or details["rebound"] or details["watermark_changed"] or (version == 2 and details["progress_changed"]))
                or delta != {"ready": details["inserted"], "leased": 0, "completed": 0, "quarantined": 0}
            ):
                raise R5Error("workset receipt advance is invalid")
        elif operation in {"claim", "claim_reclaim", "release"}:
            if (
                set(details) != {"kind", "count", "selection_sha256"}
                or not isinstance(details["kind"], str)
                or _SAFE_IDENTIFIER.fullmatch(details["kind"]) is None
                or not isinstance(details["count"], int)
                or isinstance(details["count"], bool)
                or details["count"] < 1
                or not _is_id(details["selection_sha256"])
                or delta["completed"] != 0
                or delta["quarantined"] != 0
                or delta["ready"] + delta["leased"] != 0
            ):
                raise R5Error("workset receipt lease transition is invalid")
        else:
            allowed_details = ({"completed", "retry", "quarantined", "selection_sha256"}, {"completed", "retry", "quarantined", "selection_sha256", "retry_wait", "retry_schedule_sha256"})
            if (
                set(details) not in allowed_details
                or not all(isinstance(details[key], int) and not isinstance(details[key], bool) and details[key] >= 0 for key in ("completed", "retry", "quarantined"))
                or sum(details[key] for key in ("completed", "retry", "quarantined")) < 1
                or not _is_id(details["selection_sha256"])
                or delta != {"ready": details["retry"], "leased": -sum(details[key] for key in ("completed", "retry", "quarantined")), "completed": details["completed"], "quarantined": details["quarantined"]}
            ):
                raise R5Error("workset receipt outcome transition is invalid")
        previous, prior_after, prior_version = receipt, after, version
    if prior_after is not None and (
        prior_after["counts"] != dict(counts)
        or _canonical(prior_after["watermark"]) != _canonical(final_state["watermark"])
        or (prior_version == 2 and _canonical(prior_after["progress"]) != _canonical(final_state["progress"]))
    ):
        raise R5Error("workset receipt final counts do not bind rows")
    return len(rows), previous


def _workset_inventory(
    root: Path, runtime: Any | None = None, *, profile: str = "deepseek-v4-flash-single-v1"
) -> dict[str, Any]:
    filename = (
        "local-workset.sqlite3" if profile == "local-triad-v1" else "ox-workset.sqlite3"
    )
    path = root / "runtime" / "recall-distillation" / filename
    if not path.exists():
        return {
            "present": False,
            "counts": {},
            "completed_refs": [],
            "receipt_head": None,
            "receipt_count": 0,
        }
    if path.is_symlink():
        raise R5Error("workset is symlinked")
    def snapshot() -> dict[str, dict[str, Any]]:
        return {
            suffix: _inventory_state(path.with_name(f"{path.name}{suffix}"))
            for suffix in ("", "-wal", "-shm", "-journal")
        }

    before = snapshot()
    with tempfile.TemporaryDirectory(prefix=".r5-workset-read-", dir=path.parent) as temporary:
        copied = Path(temporary) / path.name
        for suffix in ("", "-wal", "-shm", "-journal"):
            source_file = path.with_name(f"{path.name}{suffix}")
            if source_file.exists():
                shutil.copyfile(source_file, copied.with_name(f"{copied.name}{suffix}"))
        with sqlite3.connect(f"file:{copied}?mode=ro", uri=True) as conn:
            conn.execute("PRAGMA query_only=ON")
            counts = dict(
                conn.execute("SELECT state, COUNT(*) FROM work_items GROUP BY state")
            )
            if any(state not in _WORKSET_STATES for state in counts):
                raise R5Error("workset rows have an invalid state")
            normalized_counts = {state: int(counts.get(state, 0)) for state in _WORKSET_STATES}
            completed = [
                {"ref": str(row[0]), "digest": str(row[1])}
                for row in conn.execute(
                    "SELECT completion_ref, completion_digest FROM work_items "
                    "WHERE state='completed' ORDER BY sequence"
                )
            ]
            receipt_count, receipt_head = _rederive_workset_receipts(conn, normalized_counts)
    if snapshot() != before:
        raise R5Error("workset changed during read")
    audit: Mapping[str, Any] = {}
    if runtime is not None:
        audit = runtime.DistillationWorkset(path).audit_transition_receipts()
        if snapshot() != before:
            raise R5Error("workset changed during receipt audit")
        if (
            not isinstance(audit, Mapping)
            or audit.get("receipts") != receipt_count
            or audit.get("generation") != receipt_count
            or audit.get("head_sha256") != receipt_head
            or audit.get("counts") != normalized_counts
        ):
            raise R5Error("workset audit does not bind clone receipts")
    return {
        "present": True,
        "counts": counts,
        "completed_refs": sorted(completed),
        "receipt_head": receipt_head or None,
        "receipt_count": receipt_count,
        "companions": {suffix: before[suffix] for suffix in ("-wal", "-shm", "-journal")},
        "audit": dict(audit),
    }


def _inventory_state(path: Path) -> dict[str, Any]:
    return {"present": True, "state": _file_state(path)} if path.exists() else {"present": False}


def _inventory_matches(root: Path, relative: str, expected: Mapping[str, Any]) -> bool:
    path = root / relative
    if expected.get("present") is False:
        return not path.exists()
    return (
        expected.get("present") is True
        and isinstance(expected.get("state"), Mapping)
        and path.exists()
        and _file_state(path) == expected["state"]
    )


def _evidence_inventory(
    root: Path, *, profile: str = "deepseek-v4-flash-single-v1"
) -> dict[str, dict[str, Any]]:
    """Capture only the immutable evidence that must predate this verifier."""
    base = root / "runtime" / "recall-distillation"
    directories = (
        "training-snapshots", "baselines", "ox-profile-contracts", "split-plans",
        "exposures",
    )
    paths = [
        base / "label-ledger.jsonl",
        base / "label-ledger.jsonl.checkpoint.json",
        base / "candidate-ledger.jsonl",
        base / "candidate-ledger.jsonl.checkpoint.json",
        base / "exposure-receipts.jsonl",
        base / "exposure-receipts.jsonl.checkpoint.json",
        base / "ox-profile-contract.json",
        base / "split-plan.json",
        base / "rally-manifest.jsonl",
        base / "rally-manifest.jsonl.checkpoint.json",
    ]
    workset_name = (
        "local-workset.sqlite3" if profile == "local-triad-v1" else "ox-workset.sqlite3"
    )
    paths.extend(base / f"{workset_name}{suffix}" for suffix in ("", "-wal", "-shm", "-journal"))
    for directory in directories:
        path = base / directory
        if path.exists():
            paths.extend(item for item in path.glob("*.json") if item.is_file())
    return {str(path.relative_to(root)): _inventory_state(path) for path in paths}


def _stable_sealed(
    store: Any,
    root: Path,
    relative: Path,
    *,
    schema: str,
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Read a pre-existing immutable artifact twice, rejecting replacement."""
    path = root / relative
    key = str(relative)
    before = _file_state(path)
    expected = inventory.get(key)
    if not isinstance(expected, Mapping) or not _inventory_matches(root, key, expected):
        raise R5Error("official evidence was not present before verification")
    value = store.read_sealed(path, schema=schema)
    after = _file_state(path)
    if before != after or not isinstance(value, dict):
        raise R5Error("official evidence changed during stable read")
    return value


def _independent_dataset_binding(
    *,
    root: Path,
    distill: Any,
    store: Any,
    workset_runtime: Any | None,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    preflight: Mapping[str, Any],
    workset: Mapping[str, Any],
    inventory: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any] | None,
) -> bool:
    """Re-derive R5 inputs from the official clone, never caller envelopes."""
    try:
        # Chain/checkpoint and Workset files are authority inputs too.  A
        # validator must not accept a rebuilt checkpoint or swapped DB.
        for relative in (
            "runtime/recall-distillation/label-ledger.jsonl",
            "runtime/recall-distillation/label-ledger.jsonl.checkpoint.json",
            "runtime/recall-distillation/candidate-ledger.jsonl",
            "runtime/recall-distillation/candidate-ledger.jsonl.checkpoint.json",
            "runtime/recall-distillation/exposure-receipts.jsonl",
            "runtime/recall-distillation/exposure-receipts.jsonl.checkpoint.json",
            "runtime/recall-distillation/rally-manifest.jsonl",
            "runtime/recall-distillation/rally-manifest.jsonl.checkpoint.json",
        ):
            expected = inventory.get(relative)
            if not isinstance(expected, Mapping) or not _inventory_matches(root, relative, expected):
                return False
        snapshot_id = str(preflight.get("training_snapshot_sha256") or "")
        label_head = str(preflight.get("label_chain_head") or "")
        if not _is_id(snapshot_id) or not _is_id(label_head):
            return False
        snapshot = _stable_sealed(
            store, root, Path("runtime/recall-distillation/training-snapshots") / f"{snapshot_id}.json",
            schema=CANONICAL_TRAINING_SCHEMA, inventory=inventory,
        )
        official_labels = _read_rows(store, root)
        if list(labels) != official_labels or snapshot.get("rows") != list(rows):
            return False
        if snapshot.get("artifact_id") != snapshot_id or snapshot.get("label_chain_head") != label_head:
            return False
        if not _policy_is_canonical(policy) or snapshot.get("schema") != policy["training_schema"]:
            return False
        # The official row validator reconstructs ledger, rally, snapshot,
        # split-plan, feature bytes, blind order, and CF exposure bindings.
        if not rows or not all(
            distill._materialized_row_integrity(row, root=root) for row in rows
        ):
            return False
        config = distill.load_distillation_config(root / "config.toml")
        if policy["profile"] != getattr(config, "teacher_profile", None):
            return False
        if any(
            row.get("profile") != policy["profile"]
            or row.get("cohort") != policy["cohort"]
            or row.get("profile_contract_id") != policy["profile_contract_id"]
            or row.get("split_plan_id") != policy["split_plan_id"]
            for row in rows
        ):
            return False
        rederived_gate = distill._offline_training_gate(rows, config, root=root)
        if dict(gate) != rederived_gate:
            return False
        contract_id = str(distill._current_ox_profile_contract_id(root) or "")
        ox_rows = [row for row in rows if row.get("profile") == distill.OX_SINGLE_PROFILE]
        if ox_rows:
            contract = _stable_sealed(
                store, root,
                Path("runtime/recall-distillation/ox-profile-contracts") / f"{contract_id}.json",
                schema=distill.OX_PROFILE_SCHEMA, inventory=inventory,
            )
            if (
                not _is_id(contract_id)
                or contract.get("artifact_id") != contract_id
                or contract.get("route") != "opencode-go/deepseek-v4-flash"
                or any(row.get("profile_contract_id") != contract_id for row in ox_rows)
            ):
                return False
        baseline = preflight.get("baseline")
        if not isinstance(baseline, Mapping) or not _is_id(baseline.get("artifact_id")):
            return False
        sealed_baseline = _stable_sealed(
            store, root,
            Path("runtime/recall-distillation/baselines") / f"{baseline['artifact_id']}.json",
            schema="chronovisor.recall-distill-baseline.v1", inventory=inventory,
        )
        if (
            sealed_baseline.get("label_chain_head") != label_head
            or sealed_baseline.get("training_snapshot_sha256") != snapshot_id
        ):
            return False
        plans: dict[str, Mapping[str, Any]] = {}
        now = datetime.now(UTC)
        for row in rows:
            try:
                observed = datetime.fromisoformat(
                    str(row.get("as_of") or "").replace("Z", "+00:00")
                )
            except ValueError:
                return False
            if observed.tzinfo is None or observed.astimezone(UTC) > now:
                return False
            split_id = str(row.get("split_plan_id") or "")
            if not _is_id(split_id):
                return False
            if split_id not in plans:
                plans[split_id] = _stable_sealed(
                    store, root,
                    Path("runtime/recall-distillation/split-plans") / f"{split_id}.json",
                    schema=distill.SPLIT_PLAN_SCHEMA, inventory=inventory,
                )
            plan = plans[split_id]
            assignments, age_bands = plan.get("assignments"), plan.get("age_bands")
            if (
                not isinstance(assignments, Mapping)
                or assignments.get(row.get("rally_id")) != row.get("split")
                or not isinstance(age_bands, Mapping)
                or age_bands.get(row.get("rally_id"))
                not in {"old-history", "recent", "locked-test"}
            ):
                return False
            if row.get("source") == "counterfactual-label":
                exposure_id = str(row.get("counterfactual_ref") or "")
                if not _is_id(exposure_id):
                    return False
                _stable_sealed(
                    store, root,
                    Path("runtime/recall-distillation/exposures") / f"{exposure_id}.json",
                    schema="chronovisor.recall-exact-exposure.v1", inventory=inventory,
                )
        if {
            band
            for plan in plans.values()
            for band in (plan.get("age_bands") or {}).values()
            if band in {"old-history", "recent", "locked-test"}
        } != {"old-history", "recent", "locked-test"}:
            return False
        if any(not _inventory_matches(root, relative, expected) for relative, expected in inventory.items()):
            return False
        selected_workset = (
            "local-workset.sqlite3"
            if policy["profile"] == "local-triad-v1"
            else "ox-workset.sqlite3"
        )
        expected = inventory.get(f"runtime/recall-distillation/{selected_workset}")
        if not isinstance(expected, Mapping) or not _inventory_matches(root, f"runtime/recall-distillation/{selected_workset}", expected):
            return False
        rallies = distill.extract_rallies(root / "raw", root=root)
        manifest_rows = store.read_chain(
            store.distillation_dir(root) / "rally-manifest.jsonl"
        )
        manifest_ids = {
            str(item.get("manifest", {}).get("rally_id") or "")
            for item in manifest_rows
            if isinstance(item, Mapping) and isinstance(item.get("manifest"), Mapping)
        }
        candidate_rows = store.read_chain(
            store.distillation_dir(root) / "candidate-ledger.jsonl"
        )
        candidate_ids = {
            str(item.get("snapshot", {}).get("rally_id") or "")
            for item in candidate_rows
            if isinstance(item, Mapping) and isinstance(item.get("snapshot"), Mapping)
        }
        rally_ids = {str(row.get("rally_id") or "") for row in rallies}
        if rally_ids - manifest_ids or rally_ids - candidate_ids:
            return False
        final_workset = _workset_inventory(
            root, workset_runtime, profile=policy["profile"]
        )
        if final_workset != dict(workset):
            return False
        if policy["backlog"] != {
            "ready": int(final_workset.get("counts", {}).get("ready", -1)),
            "leased": int(final_workset.get("counts", {}).get("leased", -1)),
            "manifest": 0,
            "candidate": 0,
        }:
            return False
        return (
            isinstance(workset.get("receipt_count"), int)
            and workset["receipt_count"] > 0
            and _is_id(workset.get("receipt_head"))
        )
    except Exception:
        return False


_R4_DEPENDENCY_KEYS = {
    "artifact_id", "seal_sha256", "artifact_path", "artifact_file_state",
    "authority_artifact_id", "authority_seal_sha256", "authority_relative_path",
    "authority_file_state", "source_root", "source_commit", "source_tree_sha256",
}


def _verify_r4(
    path: Path, source: Mapping[str, Any], source_root: Path,
) -> dict[str, Any]:
    """Bind R5 to the source-rederived R4 authority, not an artifact ID alone."""
    artifact_path = path.expanduser().resolve(strict=True)
    source_root = source_root.expanduser().resolve(strict=True)
    artifact = R4.read_artifact(artifact_path)
    artifact_state = _file_state(artifact_path)
    production = artifact.get("production_certification")
    source_artifact = artifact.get("source")
    source_after = artifact.get("source_after")
    source_final = artifact.get("source_final")
    source_contract = artifact.get("source_contract")
    receipt_files = artifact.get("receipt_files")
    authority = artifact.get("authority_receipt")
    if (
        not isinstance(production, Mapping)
        or production.get("passed") is not True
        or production.get("provider_calls") != 0
        or production.get("collector") != "fixed-production-root-workset-v1"
        or artifact.get("production_root_used") is not True
        or not isinstance(source_artifact, Mapping)
        or source_artifact.get("clean") is not True
        or source_artifact.get("status_count") != 0
        or source_after != source_artifact
        or source_final != source_artifact
        or not isinstance(source_contract, Mapping)
        or source_contract.get("passed") is not True
        or source_artifact.get("commit") != source.get("commit")
        or source_artifact.get("tree_sha256") != source.get("tree_sha256")
        or not isinstance(receipt_files, Mapping)
        or set(receipt_files) != {"local", "ox", "production"}
        or any(
            not isinstance(receipt, Mapping)
            or not isinstance(receipt.get("count"), int)
            or receipt["count"] <= 0
            or not isinstance(receipt.get("files"), list)
            or not receipt["files"]
            for receipt in (receipt_files.get("local"), receipt_files.get("ox"))
        )
        or receipt_files.get("production") != {"files": [], "count": 0}
        or not isinstance(authority, Mapping)
        or authority.get("available") is not True
        or set(authority) != {
            "available", "artifact_id", "seal_sha256", "relative_path", "file_sha256",
            "parent_dev", "parent_ino",
        }
        or not all(_is_id(authority.get(key)) for key in ("artifact_id", "seal_sha256", "file_sha256"))
        or not isinstance(authority.get("relative_path"), str)
        or Path(authority["relative_path"]).name != authority["relative_path"]
    ):
        raise R5Error("R4 artifact is not a matching certified dependency")
    authority_path = artifact_path.parent / authority["relative_path"]
    try:
        validated = R4.validate_source_bound_authority_receipt(
            authority_path,
            artifact_path=artifact_path,
            source_root=source_root,
            source_commit=str(source.get("commit") or ""),
        )
        artifact_after = R4.read_artifact(artifact_path)
        artifact_state_after = _file_state(artifact_path)
        authority_state_after = _file_state(authority_path)
    except Exception as exc:
        raise R5Error("R4 authority receipt cannot be source-bound") from exc
    authority_state = validated.get("file_state") if isinstance(validated, Mapping) else None
    if (
        artifact_after != artifact
        or artifact_state_after != artifact_state
        or not isinstance(authority_state, Mapping)
        or authority_state_after != authority_state
        or authority_state.get("sha256") != authority["file_sha256"]
        or validated.get("artifact_id") != authority["artifact_id"]
        or validated.get("r4_artifact_id") != artifact["artifact_id"]
    ):
        raise R5Error("R4 authority receipt changed during binding")
    return {
        "artifact_id": artifact["artifact_id"],
        "seal_sha256": artifact["seal_sha256"],
        "artifact_path": str(artifact_path),
        "artifact_file_state": artifact_state,
        "authority_artifact_id": authority["artifact_id"],
        "authority_seal_sha256": authority["seal_sha256"],
        "authority_relative_path": authority["relative_path"],
        "authority_file_state": dict(authority_state),
        "source_root": str(source_root),
        "source_commit": source["commit"],
        "source_tree_sha256": source["tree_sha256"],
    }


def _baseline_binding(
    store: Any, root: Path, materialized: Mapping[str, Any]
) -> dict[str, Any]:
    label_head = materialized.get("label_chain_head")
    snapshot_id = materialized.get("artifact_id")
    if not _is_id(label_head) or not _is_id(snapshot_id):
        return {}
    for path in sorted((store.distillation_dir(root) / "baselines").glob("*.json")):
        try:
            baseline = store.read_sealed(
                path, schema="chronovisor.recall-distill-baseline.v1"
            )
        except Exception:
            continue
        if (
            baseline.get("label_chain_head") == label_head
            and baseline.get("training_snapshot_sha256") == snapshot_id
        ):
            return {
                "artifact_id": baseline.get("artifact_id"),
                "label_chain_head": label_head,
                "training_snapshot_sha256": snapshot_id,
            }
    return {}


def _canonical_floor_policy(
    *,
    rows: Sequence[Mapping[str, Any]],
    materialized: Mapping[str, Any],
    config: Any,
    gate: Mapping[str, Any],
    workset: Mapping[str, Any],
    manifest_backlog: int,
    candidate_backlog: int,
) -> dict[str, Any]:
    """Bind formal R5 to one current profile and the producer's canonical v2 row set."""
    profile = str(getattr(config, "teacher_profile", ""))
    cohorts = {str(row.get("cohort") or "") for row in rows}
    profiles = {str(row.get("profile") or "") for row in rows}
    contracts = {str(row.get("profile_contract_id") or "") for row in rows}
    splits = {str(row.get("split_plan_id") or "") for row in rows}
    counts = workset.get("counts") if isinstance(workset.get("counts"), Mapping) else {}
    return {
        "schema": R5_FLOOR_POLICY_SCHEMA,
        "training_schema": str(materialized.get("schema") or ""),
        "gate_schema": str(gate.get("schema") or ""),
        "truth_authority": str(gate.get("truth_authority") or ""),
        "profile": profile,
        "cohort": next(iter(cohorts)) if len(cohorts) == 1 else "",
        "profile_contract_id": next(iter(contracts)) if len(contracts) == 1 else "",
        "split_plan_id": next(iter(splits)) if len(splits) == 1 else "",
        "hard_floors": {
            "rallies": getattr(config, "hard_floor_rallies", 0),
            "days": getattr(config, "hard_floor_days", 0),
            "windows": getattr(config, "hard_floor_windows", 0),
            "labels": getattr(config, "hard_floor_teacher_labels", 0),
            "per_class": getattr(config, "hard_floor_teacher_per_class", 0),
            "probes": getattr(config, "hard_floor_probe_pairs", 0),
            "counterfactuals": getattr(config, "hard_floor_counterfactual_pairs", 0),
        },
        "backlog": {
            "ready": counts.get("ready", -1),
            "leased": counts.get("leased", -1),
            "manifest": manifest_backlog,
            "candidate": candidate_backlog,
        },
        "rows_profile_bound": bool(rows) and profiles == {profile},
    }


def _policy_is_canonical(policy: object) -> bool:
    if not isinstance(policy, Mapping) or set(policy) != {
        "schema", "training_schema", "gate_schema", "truth_authority", "profile", "cohort",
        "profile_contract_id", "split_plan_id", "hard_floors", "backlog", "rows_profile_bound",
    }:
        return False
    floors = policy.get("hard_floors")
    backlog = policy.get("backlog")
    if (
        policy.get("schema") != R5_FLOOR_POLICY_SCHEMA
        or policy.get("training_schema") != CANONICAL_TRAINING_SCHEMA
        or policy.get("gate_schema") != CANONICAL_GATE_SCHEMA
        or policy.get("truth_authority") != "teacher_only_not_verified"
        or policy.get("profile") not in {"local-triad-v1", "deepseek-v4-flash-single-v1"}
        or not isinstance(policy.get("cohort"), str)
        or not isinstance(policy.get("profile_contract_id"), str)
        or not _is_id(policy.get("split_plan_id"))
        or policy.get("rows_profile_bound") is not True
        or not isinstance(floors, Mapping)
        or set(floors) != {"rallies", "days", "windows", "labels", "per_class", "probes", "counterfactuals"}
        or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in floors.values())
        or not isinstance(backlog, Mapping)
        or set(backlog) != {"ready", "leased", "manifest", "candidate"}
        or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in backlog.values())
    ):
        return False
    if policy["profile"] == "local-triad-v1":
        return policy["cohort"] == "local-triad-v1" and policy["profile_contract_id"] == ""
    return _is_id(policy["profile_contract_id"]) and bool(policy["cohort"])


def _reason(reasons: list[str], condition: bool, name: str) -> None:
    if not condition:
        reasons.append(name)


def validate_dataset(
    *,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    rallies: Sequence[Mapping[str, Any]] = (),
    preflight: Mapping[str, Any],
    gate: Mapping[str, Any],
    workset: Mapping[str, Any],
    root: Path | None = None,
    distill: Any | None = None,
    store: Any | None = None,
    workset_runtime: Any | None = None,
    evidence_inventory: Mapping[str, Mapping[str, Any]] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate only concrete rows; booleans supplied by a producer are not evidence."""
    reasons: list[str] = []
    hard: Mapping[str, Any] = (
        dict(preflight["hard_floor"])
        if isinstance(preflight.get("hard_floor"), Mapping)
        else {}
    )
    native = {
        str(row.get("rally_id"))
        for row in rallies
        if isinstance(row.get("rally_id"), str)
        and isinstance(row.get("eligibility"), Mapping)
        and row["eligibility"].get("native_rally") is True
    }
    native_dates = [
        datetime.fromtimestamp(int(row["as_of_us"]) / 1_000_000, UTC).date()
        for row in rallies
        if str(row.get("rally_id") or "") in native
        and isinstance(row.get("as_of_us"), int)
        and not isinstance(row.get("as_of_us"), bool)
    ]
    age_bands = {"0_7": 0, "8_30": 0, "31_plus": 0}
    if native_dates:
        latest = max(native_dates)
        for date in native_dates:
            age = (latest - date).days
            age_bands["0_7" if age <= 7 else "8_30" if age <= 30 else "31_plus"] += 1
    span_days = (max(native_dates) - min(native_dates)).days + 1 if native_dates else 0
    windows = (
        len({(date - min(native_dates)).days // 7 for date in native_dates})
        if native_dates
        else 0
    )
    _reason(reasons, bool(native), "eligible_native_predicate_missing_or_empty")
    _reason(
        reasons,
        len(native) >= MIN_RALLIES,
        "eligible_native_rallies_below_floor",
    )
    _reason(
        reasons,
        span_days >= MIN_DAYS,
        "eligible_native_span_below_floor",
    )
    _reason(
        reasons,
        windows >= MIN_WINDOWS,
        "eligible_native_windows_below_floor",
    )
    invalid_status = [
        row
        for row in labels
        if str(row.get("status") or "") not in {"completed", "valid"}
    ]
    _reason(reasons, not invalid_status, "invalid_or_uncertain_label_status")
    ids = [str(row.get("record_sha256") or "") for row in labels]
    _reason(
        reasons,
        bool(ids) and all(ids) and len(ids) == len(set(ids)),
        "label_identity_duplicate_or_missing",
    )
    valid = [
        row
        for row in rows
        if row.get("source") == "teacher-label"
        and row.get("probe") is not True
        and row.get("verdict") in {"relevant", "irrelevant"}
    ]
    _reason(reasons, len(valid) >= MIN_LABELS, "valid_nonprobe_labels_below_floor")
    row_label_ids = [str(row.get("label_record_sha256") or "") for row in rows]
    _reason(
        reasons,
        bool(row_label_ids)
        and all(row_label_ids)
        and len(row_label_ids) == len(set(row_label_ids))
        and set(row_label_ids) == set(ids),
        "materialized_label_ledger_not_exact_1_to_1",
    )
    verdicts = Counter(str(row.get("verdict")) for row in valid)
    _reason(
        reasons,
        verdicts["relevant"] >= MIN_PER_CLASS
        and verdicts["irrelevant"] >= MIN_PER_CLASS,
        "label_class_below_floor",
    )
    revisions = ("profile", "cohort", "assignment_revision", "label_set_revision")
    _reason(
        reasons,
        all(
            all(isinstance(row.get(key), str) and row.get(key) for key in revisions)
            for row in labels
        ),
        "revision_append_only_binding_missing",
    )
    probes = defaultdict(set)
    for row in rows:
        if (
            row.get("source") == "teacher-label"
            and row.get("probe") is True
            and row.get("verdict") in {"relevant", "irrelevant"}
        ):
            probes[(str(row.get("rally_id")), str(row.get("candidate_id")))].add(
                str(row.get("route"))
            )
    local_pairs = sum(
        routes
        == {
            "recall.distill.teacher.a",
            "recall.distill.teacher.b",
            "recall.distill.teacher.c",
        }
        for routes in probes.values()
    )
    blind_pairs = {
        (str(row.get("rally_id")), str(row.get("candidate_id")))
        for row in rows
        if row.get("ox_blind") is True and row.get("order_swap") is True
    }
    _reason(
        reasons,
        local_pairs >= MIN_PROBES or len(blind_pairs) >= MIN_PROBES,
        "probe_or_blind_repeat_below_floor",
    )
    cf = [
        row
        for row in rows
        if row.get("source") == "counterfactual-label"
        and row.get("verdict") in {"helpful", "harmful"}
    ]
    cf_ids = [str(row.get("counterfactual_ref") or "") for row in cf]
    _reason(
        reasons,
        len(cf) >= MIN_COUNTERFACTUALS and len(cf_ids) == len(set(cf_ids)),
        "sealed_counterfactual_pairs_below_floor_or_duplicate",
    )
    _reason(
        reasons,
        all(
            _is_id(row.get("counterfactual_ref"))
            and _is_id(row.get("exposure_artifact_id"))
            and _is_id(row.get("sealed_exposure_root"))
            and row.get("order_agreement") is True
            for row in cf
        ),
        "counterfactual_sealed_exposure_binding_missing",
    )
    _reason(
        reasons,
        bool(rows)
        and all(
            "future_leakage" in row and row.get("future_leakage") is False
            for row in rows
        ),
        "future_leakage_detected",
    )
    _reason(
        reasons,
        bool(rows) and all(row.get("feature_parity") is True for row in rows),
        "feature_parity_not_100_percent",
    )
    canonical_policy = _policy_is_canonical(policy)
    _reason(reasons, canonical_policy, "canonical_dataset_policy_missing")
    profile = str(policy.get("profile") or "") if canonical_policy else ""
    _reason(
        reasons,
        (
            gate.get("passed") is True
            and gate.get("schema") == CANONICAL_GATE_SCHEMA
            and gate.get("truth_authority") == "teacher_only_not_verified"
            and hard.get("p5_allowed") is True
            and (
                profile == "local-triad-v1"
                or (
                    isinstance(gate.get("identity"), Mapping)
                    and gate["identity"].get("route") == "opencode-go/deepseek-v4-flash"
                    and gate["identity"].get("profile_contract_id")
                    == policy.get("profile_contract_id")
                    and isinstance(gate.get("blind_repeat"), Mapping)
                    and isinstance(gate["blind_repeat"].get("complete_pairs"), int)
                    and gate["blind_repeat"]["complete_pairs"] >= MIN_PROBES
                )
            )
        ),
        "source_gate_not_passed",
    )
    work_counts: Mapping[str, Any] = (
        dict(workset["counts"]) if isinstance(workset.get("counts"), Mapping) else {}
    )
    _reason(
        reasons,
        work_counts.get("leased", 0) == 0 and work_counts.get("ready", 0) == 0,
        "leased_work_remaining",
    )
    _reason(
        reasons,
        canonical_policy
        and all(policy["backlog"][key] == 0 for key in ("ready", "leased", "manifest", "candidate")),
        "dataset_backlog_remaining",
    )
    completed_raw = workset.get("completed_refs", [])
    completed = completed_raw if isinstance(completed_raw, list) else []
    completed_digests = [
        str(item.get("digest") or "") for item in completed if isinstance(item, Mapping)
    ]
    _reason(
        reasons,
        len(completed_digests) == len(set(completed_digests))
        and all(
            isinstance(item, Mapping)
            and item.get("ref") == f"label-ledger:{item.get('digest')}"
            for item in completed
        )
        and set(completed_digests) == set(ids),
        "workset_label_reconciliation_failed",
    )
    audit = workset.get("audit")
    _reason(
        reasons,
        isinstance(audit, Mapping)
        and audit.get("status") in {"verified", "verified-empty"}
        and audit.get("head_sha256") == workset.get("receipt_head"),
        "workset_receipt_chain_not_independently_verified",
    )
    # A formal dataset cannot treat a generated baseline as current without an
    # explicit label-head and snapshot binding.
    baseline: Mapping[str, Any] = (
        dict(preflight["baseline"])
        if isinstance(preflight.get("baseline"), Mapping)
        else {}
    )
    _reason(
        reasons,
        bool(
            baseline.get("label_chain_head") == preflight.get("label_chain_head")
            and baseline.get("training_snapshot_sha256")
        ),
        "baseline_label_head_or_snapshot_binding_missing",
    )
    official: Mapping[str, Any] = (
        dict(preflight["official_r5_evidence"])
        if isinstance(preflight.get("official_r5_evidence"), Mapping)
        else {}
    )
    _reason(
        reasons,
        (
            _is_id(official.get("materialization_artifact_id"))
            and official.get("materialization_artifact_id")
            == preflight.get("training_snapshot_sha256")
            and official.get("label_chain_head") == preflight.get("label_chain_head")
            and official.get("rows_sha256") == _sha(list(rows))
            and official.get("gate_sha256") == _sha(gate)
            and official.get("floor_policy_sha256") == _sha(policy)
        ),
        "official_materialization_binding_missing",
    )
    _reason(
        reasons,
        root is not None
        and distill is not None
        and store is not None
        and evidence_inventory is not None
        and _independent_dataset_binding(
            root=root,
            distill=distill,
            store=store,
            workset_runtime=workset_runtime,
            rows=rows,
            labels=labels,
            gate=gate,
            preflight=preflight,
            workset=workset,
            inventory=evidence_inventory,
            policy=policy,
        ),
        "independent_official_evidence_rederivation_failed",
    )
    return {
        "passed": not reasons,
        "capture_only": bool(reasons),
        "reasons": sorted(set(reasons)),
        "metrics": {
            "rows": len(rows),
            "labels": len(labels),
            "valid_labels": len(valid),
            "classes": dict(verdicts),
            "local_probe_pairs": local_pairs,
            "blind_pairs": len(blind_pairs),
            "counterfactual_pairs": len(cf),
            "age_bands": age_bands,
            "future_leakage": 0
            if not any(row.get("future_leakage") is True for row in rows)
            else 1,
            "feature_parity_percent": 100
            if rows and all(row.get("feature_parity") is True for row in rows)
            else 0,
        },
        "policy": dict(policy) if isinstance(policy, Mapping) else {},
    }


def _sealed_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    _validate_artifact_nested(payload)
    unsigned = {"schema": R5_SCHEMA, "namespace": NAMESPACE, **payload}
    artifact_id = _sha(unsigned)
    return {
        "artifact_id": artifact_id,
        **unsigned,
        "seal_sha256": _sha({"artifact_id": artifact_id, **unsigned}),
    }


def _reject_sensitive_fields(value: object) -> None:
    """Receipts are hashes and counts only: no source data or credentials."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or any(
                marker in key.casefold()
                for marker in ("payload", "secret", "token", "password", "authorization", "content", "body")
            ):
                raise R5Error("artifact contains a forbidden nested field")
            _reject_sensitive_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_sensitive_fields(item)


def _closed_mapping(value: object, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise R5Error(f"artifact {label} schema is not closed")
    return value


_FILE_STATE_KEYS = {"bytes", "mtime_ns", "dev", "ino", "sha256"}
_SOURCE_KEYS = {
    "commit", "clean", "status_sha256", "status_count", "tree_sha256", "file_count",
    "symlink_count", "ox_identity_sha256", "account_uid", "account_home", "tree",
    "index_count", "index_sha256", "git_index", "tracked_bytes_sha256", "tool_identities",
}
_PRODUCTION_KEYS = {
    "root", "managed_inventory", "raw", "runtime", "config", "protected_inventory",
}
_CLONE_KEYS = {"owned", "cow", "dev", "ino", "volume", "tool", "parity", "state"}


def _closed_state(value: object, allowed: tuple[set[str], ...], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) not in allowed:
        raise R5Error(f"artifact {label} schema is not closed")
    return value


def _validate_file_state(value: object, *, label: str) -> None:
    state = _closed_mapping(value, _FILE_STATE_KEYS, label=label)
    if (
        not all(isinstance(state[key], int) and not isinstance(state[key], bool) and state[key] >= 0
                for key in ("bytes", "mtime_ns", "dev", "ino"))
        or not _is_id(state["sha256"])
    ):
        raise R5Error(f"artifact {label} values are invalid")


def _primitive(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _utc_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_SECONDS.fullmatch(value) is None:
        raise R5Error(f"artifact {label} is not UTC")
    try:
        instant = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise R5Error(f"artifact {label} is not UTC") from exc
    if instant.tzinfo is None or instant.utcoffset() != UTC.utcoffset(None):
        raise R5Error(f"artifact {label} is not UTC")
    return instant.astimezone(UTC)


def _validate_tree_summary(value: object, *, label: str, include_files: bool = False) -> None:
    keys = {"file_count", "tree_sha256", "files"} if include_files else {"file_count", "tree_sha256"}
    state = _closed_mapping(value, keys, label=label)
    if (
        not isinstance(state["file_count"], int)
        or isinstance(state["file_count"], bool)
        or state["file_count"] < 0
        or not _is_id(state["tree_sha256"])
        or (include_files and not _is_id(state["files"]))
    ):
        raise R5Error(f"artifact {label} values are invalid")


def _validate_identity_state(value: object, *, label: str) -> None:
    state = _closed_state(value, (set(), {"commit", "tree_sha256"}, _SOURCE_KEYS), label=label)
    if set(state) == {"commit", "tree_sha256"}:
        if (
            not isinstance(state["commit"], str)
            or len(state["commit"]) != 40
            or set(state["commit"]) - set("0123456789abcdef")
            or not _is_id(state["tree_sha256"])
        ):
            raise R5Error(f"artifact {label} identity values are invalid")
        return
    if set(state) == _SOURCE_KEYS:
        _validate_file_state(state["git_index"], label=f"{label}.git_index")
        if not all(_primitive(state[key]) for key in _SOURCE_KEYS - {"git_index", "tool_identities"}):
            raise R5Error(f"artifact {label} values are invalid")
        tools = state["tool_identities"]
        if not isinstance(tools, list) or any(
            not isinstance(tool, Mapping) or set(tool) != {"path", "uid", "mode", "sha256"}
            or not isinstance(tool["path"], str)
            or not isinstance(tool["uid"], int) or isinstance(tool["uid"], bool) or tool["uid"] < 0
            or not isinstance(tool["mode"], int) or isinstance(tool["mode"], bool) or not 0 <= tool["mode"] <= 0o7777
            or not _is_id(tool["sha256"])
            for tool in tools
        ):
            raise R5Error(f"artifact {label}.tool_identities schema is not closed")
        if (
            not isinstance(state["commit"], str)
            or len(state["commit"]) != 40
            or set(state["commit"]) - set("0123456789abcdef")
            or not isinstance(state["tree"], str)
            or len(state["tree"]) != 40
            or set(state["tree"]) - set("0123456789abcdef")
            or not all(_is_id(state[key]) for key in ("status_sha256", "tree_sha256", "ox_identity_sha256", "index_sha256", "tracked_bytes_sha256"))
            or state["clean"] is not True
            or not all(isinstance(state[key], int) and not isinstance(state[key], bool) and state[key] >= 0 for key in ("status_count", "file_count", "symlink_count", "account_uid", "index_count"))
            or not isinstance(state["account_home"], str)
        ):
            raise R5Error(f"artifact {label} identity values are invalid")


def _validate_production_state(value: object, *, label: str) -> None:
    state = _closed_state(value, (set(), {"raw", "runtime", "config"}, _PRODUCTION_KEYS), label=label)
    if set(state) == set():
        return
    if set(state) == {"raw", "runtime", "config"}:
        for key in ("raw", "runtime"):
            if state[key] is not None:
                _validate_tree_summary(state[key], label=f"{label}.{key}")
        if state["config"] is not None:
            _validate_file_state(state["config"], label=f"{label}.config")
        return
    root = _closed_mapping(state["root"], {"dev", "ino", "uid", "gid", "mode", "ctime_ns"}, label=f"{label}.root")
    managed = _closed_mapping(state["managed_inventory"], {"entries", "sha256"}, label=f"{label}.managed_inventory")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in root.values()):
        raise R5Error(f"artifact {label}.root values are invalid")
    if not isinstance(managed["entries"], int) or isinstance(managed["entries"], bool) or managed["entries"] < 0 or not _is_id(managed["sha256"]):
        raise R5Error(f"artifact {label}.managed_inventory values are invalid")
    for key in ("raw", "runtime"):
        if state[key] is not None:
            _validate_tree_summary(state[key], label=f"{label}.{key}")
    if state["config"] is not None:
        _validate_file_state(state["config"], label=f"{label}.config")
    protected = _closed_mapping(
        state["protected_inventory"], {"included", "excluded", "excluded_sha256"}, label=f"{label}.protected_inventory"
    )
    if (
        not all(isinstance(protected[key], list) and all(_primitive(item) for item in protected[key]) for key in ("included", "excluded"))
        or not _is_id(protected["excluded_sha256"])
    ):
        raise R5Error(f"artifact {label}.protected_inventory schema is not closed")


def _validate_clone_state(value: object, *, production: object | None = None) -> None:
    state = _closed_state(value, (set(), {"test_only"}, {"owned", "state"}, _CLONE_KEYS), label="clone")
    if set(state) == {"test_only"}:
        if not isinstance(state["test_only"], bool):
            raise R5Error("artifact clone.test_only is invalid")
        return
    if set(state) == {"owned", "state"}:
        if not isinstance(state["owned"], bool):
            raise R5Error("artifact clone.owned is invalid")
        _validate_tree_summary(state["state"], label="clone.state", include_files=True)
        return
    if set(state) == _CLONE_KEYS:
        if (
            state["owned"] is not True
            or state["cow"] != "copyfile(3):COPYFILE_CLONE_FORCE"
            or state["volume"] != "apfs"
            or not all(
                isinstance(state[key], int) and not isinstance(state[key], bool) and state[key] > 0
                for key in ("dev", "ino")
            )
        ):
            raise R5Error("artifact clone values are invalid")
        tool = _closed_mapping(state["tool"], {"backend", "flags"}, label="clone.tool")
        if tool["backend"] != "copyfile(3)" or tool["flags"] != R2.COPYFILE_CLONE_FORCE:
            raise R5Error("artifact clone.tool values are invalid")
        _validate_production_state(state["parity"], label="clone.parity")
        if production is not None and state["parity"] != production:
            raise R5Error("artifact clone parity does not match production")
        _validate_tree_summary(state["state"], label="clone.state", include_files=True)


def _validate_artifact_nested(payload: Mapping[str, Any]) -> None:
    """Close the receipt-owned nested records while permitting historic declines."""
    _reject_sensitive_fields(payload)
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_id", "schema", "namespace", "seal_sha256"}
    }
    _closed_mapping(
        core,
        {
            "captured_at", "source", "source_after", "production", "production_after", "clone",
            "r4_dependency", "dataset", "phases", "cleanup", "provider_calls", "egress_attempts",
            "process_attempts", "supervised", "test_only",
        },
        label="payload",
    )
    captured_at = _utc_timestamp(core["captured_at"], label="captured_at")
    if captured_at > datetime.now(UTC) + timedelta(seconds=MAX_SUPERVISOR_FUTURE_SKEW_SECONDS):
        raise R5Error("artifact captured_at is in the future")
    if not isinstance(core["test_only"], bool):
        raise R5Error("artifact test_only is invalid")
    _validate_identity_state(core.get("source"), label="source")
    _validate_identity_state(core.get("source_after"), label="source_after")
    _validate_production_state(core.get("production"), label="production")
    _validate_production_state(core.get("production_after"), label="production_after")
    _validate_clone_state(core.get("clone"), production=core.get("production"))
    cleanup = _closed_mapping(core.get("cleanup"), {"clone_removed", "remaining"}, label="cleanup")
    if not isinstance(cleanup["clone_removed"], bool) or not isinstance(cleanup["remaining"], int) or isinstance(cleanup["remaining"], bool) or cleanup["remaining"] < 0:
        raise R5Error("artifact cleanup values are invalid")
    if not all(isinstance(core[key], int) and not isinstance(core[key], bool) and core[key] >= 0 for key in ("provider_calls", "egress_attempts", "process_attempts")):
        raise R5Error("artifact sentinel counters are invalid")
    dependency = core.get("r4_dependency")
    if not isinstance(dependency, Mapping):
        raise R5Error("artifact r4 dependency schema is not closed")
    if set(dependency) not in (set(), {"artifact_id"}, {"artifact_id", "seal_sha256"}, _R4_DEPENDENCY_KEYS):
        _closed_mapping(dependency, {"artifact_id", "seal_sha256"}, label="r4 dependency")
    if set(dependency) == {"artifact_id", "seal_sha256"} and (
        not _is_id(dependency["artifact_id"]) or not _is_id(dependency["seal_sha256"])
    ):
        raise R5Error("artifact r4 dependency values are invalid")
    if set(dependency) == _R4_DEPENDENCY_KEYS:
        if (
            not all(_is_id(dependency[key]) for key in (
                "artifact_id", "seal_sha256", "authority_artifact_id", "authority_seal_sha256",
            ))
            or not isinstance(dependency["artifact_path"], str)
            or not Path(dependency["artifact_path"]).is_absolute()
            or not isinstance(dependency["authority_relative_path"], str)
            or Path(dependency["authority_relative_path"]).name != dependency["authority_relative_path"]
            or not isinstance(dependency["source_root"], str)
            or not Path(dependency["source_root"]).is_absolute()
            or not isinstance(dependency["source_commit"], str)
            or len(dependency["source_commit"]) != 40
            or bool(set(dependency["source_commit"]) - set("0123456789abcdef"))
            or not _is_id(dependency["source_tree_sha256"])
        ):
            raise R5Error("artifact R4 dependency values are invalid")
        _validate_file_state(dependency["artifact_file_state"], label="r4 dependency artifact")
        _validate_file_state(dependency["authority_file_state"], label="r4 dependency authority")
    phases = core.get("phases")
    if not isinstance(phases, list) or any(
        not isinstance(phase, Mapping) or set(phase) != {"name", "elapsed_ms"}
        or not isinstance(phase["name"], str)
        or not isinstance(phase["elapsed_ms"], (int, float))
        or isinstance(phase["elapsed_ms"], bool)
        or phase["elapsed_ms"] < 0
        for phase in phases
    ):
        raise R5Error("artifact phases schema is not closed")
    dataset = core.get("dataset")
    if isinstance(dataset, Mapping) and set(dataset) == {"passed"}:
        if dataset.get("passed") is False:
            return  # sealed decline used only by static attack fixtures
        raise R5Error("formal artifact dataset schema is incomplete")
    if isinstance(dataset, Mapping) and set(dataset) == {"passed", "test_only"}:
        if dataset.get("passed") is not True or dataset.get("test_only") is not True or core["test_only"] is not True or set(dependency) != {"artifact_id", "seal_sha256"}:
            raise R5Error("test-only artifact schema is invalid")
        return  # test-only child receipts cannot pass formal acceptance
    dataset = _closed_state(
        dataset,
        (
            {"passed", "capture_only", "reasons", "metrics"},
            {"passed", "capture_only", "reasons", "metrics", "policy"},
        ),
        label="dataset",
    )
    if (
        not isinstance(dataset["passed"], bool)
        or not isinstance(dataset["capture_only"], bool)
        or not isinstance(dataset["reasons"], list)
        or not all(
            isinstance(reason, str) and _REASON_CODE.fullmatch(reason) and reason in _DECLINE_REASON_CODES
            for reason in dataset["reasons"]
        )
    ):
        raise R5Error("artifact dataset values are invalid")
    if dataset["passed"] is True and (
        dataset["capture_only"] is not False or dataset["reasons"] != []
    ):
        raise R5Error("passed artifact cannot contain decline state")
    metrics = _closed_mapping(
        dataset.get("metrics"),
        {"rows", "labels", "valid_labels", "classes", "local_probe_pairs", "blind_pairs", "counterfactual_pairs", "age_bands", "future_leakage", "feature_parity_percent"},
        label="dataset metrics",
    )
    policy = dataset.get("policy")
    if policy and not _policy_is_canonical(policy):
        raise R5Error("artifact dataset policy is invalid")
    strict_dataset = dataset.get("passed") is True and core.get("test_only") is False
    classes = metrics.get("classes")
    if not isinstance(classes, Mapping) or set(classes) - {"relevant", "irrelevant"} or (strict_dataset and set(classes) != {"relevant", "irrelevant"}) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in classes.values()
    ):
        raise R5Error("artifact dataset classes schema is not closed")
    age_bands = metrics.get("age_bands")
    if not isinstance(age_bands, Mapping) or set(age_bands) - {"0_7", "8_30", "31_plus"} or (strict_dataset and set(age_bands) != {"0_7", "8_30", "31_plus"}):
        raise R5Error("artifact dataset age bands schema is not closed")
    if not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in age_bands.values()):
        raise R5Error("artifact dataset age bands values are invalid")
    scalar_metrics = set(metrics) - {"classes", "age_bands"}
    if not all(
        isinstance(metrics[key], int) and not isinstance(metrics[key], bool) and metrics[key] >= 0
        for key in scalar_metrics
    ):
        raise R5Error("artifact dataset metrics values are invalid")
    formal = strict_dataset
    if formal and not _policy_is_canonical(policy):
        raise R5Error("formal artifact dataset policy is missing")
    if formal and (
        metrics["rows"] < MIN_LABELS
        or metrics["labels"] < MIN_LABELS
        or metrics["valid_labels"] < MIN_LABELS
        or classes.get("relevant", 0) < MIN_PER_CLASS
        or classes.get("irrelevant", 0) < MIN_PER_CLASS
        or (metrics["local_probe_pairs"] < MIN_PROBES and metrics["blind_pairs"] < MIN_PROBES)
        or metrics["counterfactual_pairs"] < MIN_COUNTERFACTUALS
        or metrics["future_leakage"] != 0
        or metrics["feature_parity_percent"] != 100
        or sum(classes.values()) != metrics["valid_labels"]
    ):
        raise R5Error("formal artifact dataset floors are invalid")
    if formal and policy["backlog"] != {"ready": 0, "leased": 0, "manifest": 0, "candidate": 0}:
        raise R5Error("formal artifact dataset backlog is nonzero")
    if formal and (
        set(core["source"]) != _SOURCE_KEYS
        or set(core["source_after"]) != _SOURCE_KEYS
        or set(core["production"]) != _PRODUCTION_KEYS
        or set(core["production_after"]) != _PRODUCTION_KEYS
        or set(core["clone"]) != _CLONE_KEYS
    ):
        raise R5Error("formal artifact requires complete identity records")
    if formal and (
        core["source"] != core["source_after"]
        or core["production"] != core["production_after"]
    ):
        raise R5Error("formal artifact source or Raw/config identity changed")


def _open_owned_directory(
    path: Path, *, expected: Mapping[str, int] | None = None
) -> tuple[int, dict[str, int]]:
    """Pin an owned directory before using an untrusted child name below it."""
    if _has_symlink_component(path):
        raise R5Error("owned directory path contains a symlink")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        value = os.fstat(fd)
        if (
            not stat.S_ISDIR(value.st_mode)
            or value.st_uid != ACCOUNT_UID
            or value.st_mode & 0o022
        ):
            raise R5Error("owned directory is unsafe")
        identity = {"dev": value.st_dev, "ino": value.st_ino, "mode": value.st_mode & 0o7777}
        if expected is not None and any(identity.get(key) != item for key, item in expected.items()):
            raise R5Error("owned directory identity changed")
        return fd, identity
    except Exception:
        os.close(fd)
        raise


def _read_private_json_at(parent_fd: int, name: str, *, label: str) -> tuple[bytes, dict[str, Any]]:
    if not name or name in {".", ".."} or "/" in name:
        raise R5Error(f"{label} name is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != ACCOUNT_UID
            or before.st_mode & 0o077
            or before.st_size > MAX_FILE_BYTES
        ):
            raise R5Error(f"{label} file is unsafe")
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        if (
            len(raw) != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise R5Error(f"{label} changed during read")
        return raw, {
            "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "dev": before.st_dev,
            "ino": before.st_ino,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    finally:
        os.close(fd)


def read_artifact(
    path: Path, *, parent_identity: Mapping[str, int] | None = None
) -> dict[str, Any]:
    parent_fd, identity = _open_owned_directory(path.parent, expected=parent_identity)
    try:
        raw, _state = _read_private_json_at(parent_fd, path.name, label="artifact")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise R5Error("artifact unreadable") from exc
    finally:
        os.close(parent_fd)
    if (
        not isinstance(value, dict)
        or value.get("schema") != R5_SCHEMA
        or value.get("namespace") != NAMESPACE
    ):
        raise R5Error("artifact schema mismatch")
    _validate_artifact_nested(value)
    expected = {
        "artifact_id",
        "schema",
        "namespace",
        "seal_sha256",
        "captured_at",
        "source",
        "source_after",
        "production",
        "production_after",
        "clone",
        "r4_dependency",
        "dataset",
        "phases",
        "cleanup",
        "provider_calls",
        "egress_attempts",
        "process_attempts",
        "supervised",
        "test_only",
    }
    if set(value) != expected:
        raise R5Error("artifact keys are not closed")
    if raw not in {_canonical(value), _canonical(value) + b"\n"}:
        raise R5Error("artifact is not canonical")
    if (
        not isinstance(value.get("dataset"), Mapping)
        or not isinstance(value.get("cleanup"), Mapping)
        or value.get("supervised") is not False
        or not isinstance(value.get("test_only"), bool)
    ):
        raise R5Error("artifact nested schema invalid")
    unsigned = {
        key: item
        for key, item in value.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    artifact_id = _sha(unsigned)
    if value.get("artifact_id") != artifact_id or value.get("seal_sha256") != _sha(
        {"artifact_id": artifact_id, **unsigned}
    ):
        raise R5Error("artifact seal mismatch")
    if parent_identity is not None and identity != dict(parent_identity):
        raise R5Error("artifact parent changed during read")
    return value


def _write_immutable(
    output: Path, artifact: Mapping[str, Any], *, parent_identity: Mapping[str, int] | None = None
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    output_identity = _directory_identity(output)
    if parent_identity is not None and output_identity != dict(parent_identity):
        raise R5Error("immutable artifact parent changed")
    path = output / f"{artifact['artifact_id']}.json"
    raw = _canonical(artifact) + b"\n"
    if path.exists():
        try:
            existing = read_artifact(path, parent_identity=output_identity)
        except R5Error as exc:
            raise R5Error("immutable artifact conflict") from exc
        if _canonical(existing) + b"\n" != raw:
            raise R5Error("immutable artifact conflict")
        return path
    _write_private_json(path, artifact)
    if _directory_identity(output) != output_identity:
        raise R5Error("immutable artifact parent changed")
    read_artifact(path, parent_identity=output_identity)
    return path


def _publish_inner(output: Path, artifact: Mapping[str, Any]) -> tuple[Path, dict[str, int]]:
    """Persist the child receipt before its private workspace is removed."""
    root = output / "r5-inner"
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise R5Error("official inner root is unsafe")
    root.mkdir(mode=0o700, exist_ok=True)
    if _has_symlink_component(root):
        raise R5Error("official inner root contains a symlink")
    root_identity = _directory_identity(root)
    path = _write_immutable(root, artifact, parent_identity=root_identity)
    if _directory_identity(root) != root_identity:
        raise R5Error("official inner root changed during publication")
    if read_artifact(path, parent_identity=root_identity) != dict(artifact):
        raise R5Error("published inner hash mismatch")
    return path, root_identity


def run(
    *,
    production: Path,
    source: Path,
    source_commit: str,
    output: Path,
    r4_artifact: Path,
    test_only: bool = False,
    watchdog_seconds: int = 600,
    owned_clone: Path | None = None,
    kernel_attested: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Collect one R5 capture.  Expected data gaps yield a sealed decline."""
    if not kernel_attested or not _kernel_sandbox_attested():
        raise R5Error("inner capture requires kernel sandbox attestation")
    if watchdog_seconds <= 0:
        raise R5Error("watchdog must be positive")
    production = production.resolve(strict=True)
    if production != PRODUCTION_ROOT and not test_only:
        raise R5Error("only the fixed OS production root may certify R5")
    assert_root_matrix(production, source, output, owned_clone)
    started = time.monotonic()
    source_before = source_state(source, source_commit)
    r4_dependency = _verify_r4(r4_artifact, source_before, source)
    production_before = production_state(production)
    clone: Path | None = None
    clone_info: dict[str, Any] = {}
    sentinels = {"provider_calls": 0, "egress_attempts": 0, "process_attempts": 0}
    distill: Any | None = None
    store: Any | None = None
    workset_runtime: Any | None = None
    evidence_inventory: Mapping[str, Mapping[str, Any]] | None = None
    policy: Mapping[str, Any] | None = None
    labels: list[dict[str, Any]] = []
    workset: Mapping[str, Any] = {}
    phases: list[dict[str, Any]] = []
    cleanup = {"clone_removed": False, "remaining": 0}
    runtime_modules = _chronovisor_module_snapshot()
    try:
        clone, clone_info = (
            _clone(production)
            if owned_clone is None
            else _clone(production, owned_clone=owned_clone)
        )
        clone_info["state"] = _clone_input_state(clone)
        if clone_info["state"] != _clone_input_state(production):
            raise R5Error("clone input state does not match production")
        if time.monotonic() - started > watchdog_seconds:
            raise R5Error("watchdog expired before clone collection")
        phase = time.monotonic()
        with _egress_sentinel(clone) as sentinels:
            distill, store, workset_runtime = _load_runtime(source)
            with _provider_sentinel(sentinels, distill):
                try:
                    # The official producer owns clone-only manifest/candidate
                    # materialization.  Empty workers prohibit every provider call.
                    distill.run_distillation_chunk(
                        root=clone,
                        teachers={},
                        max_elapsed_seconds=60,
                    )
                    config = distill.load_distillation_config(clone / "config.toml")
                    profile = str(config.teacher_profile)
                    labels = _read_rows(store, clone)
                    materialized = distill.materialize_training_rows(clone)
                    rows = (
                        materialized.get("rows", [])
                        if isinstance(materialized, Mapping)
                        else []
                    )
                    rallies = distill.extract_rallies(clone / "raw", root=clone)
                    preflight = distill.preflight(
                        raw_dir=clone / "raw",
                        root=clone,
                        config_path=clone / "config.toml",
                        runtime_commit="unknown",
                        _training_snapshot=materialized,
                    )
                    gate = distill._offline_training_gate(rows, config, root=clone)
                    snapshot_path = (
                        store.distillation_dir(clone) / "training-snapshots"
                        / f"{materialized.get('artifact_id')}.json"
                    )
                    sealed_snapshot = store.read_sealed(
                        snapshot_path, schema=CANONICAL_TRAINING_SCHEMA
                    )
                    if (
                        sealed_snapshot.get("artifact_id") != materialized.get("artifact_id")
                        or sealed_snapshot.get("rows") != rows
                        or sealed_snapshot.get("label_chain_head")
                        != materialized.get("label_chain_head")
                    ):
                        raise R5Error("official materialization artifact mismatch")
                    workset = _workset_inventory(
                        clone, workset_runtime, profile=profile
                    )
                    rally_ids = {
                        str(row.get("rally_id") or "") for row in rallies
                    }
                    manifest_rows = store.read_chain(
                        store.distillation_dir(clone) / "rally-manifest.jsonl"
                    )
                    candidate_rows = store.read_chain(
                        store.distillation_dir(clone) / "candidate-ledger.jsonl"
                    )
                    manifest_backlog = len(rally_ids - {
                        str(item.get("manifest", {}).get("rally_id") or "")
                        for item in manifest_rows
                        if isinstance(item, Mapping) and isinstance(item.get("manifest"), Mapping)
                    })
                    candidate_backlog = len(rally_ids - {
                        str(item.get("snapshot", {}).get("rally_id") or "")
                        for item in candidate_rows
                        if isinstance(item, Mapping) and isinstance(item.get("snapshot"), Mapping)
                    })
                    policy = _canonical_floor_policy(
                        rows=rows,
                        materialized=materialized,
                        config=config,
                        gate=gate,
                        workset=workset,
                        manifest_backlog=manifest_backlog,
                        candidate_backlog=candidate_backlog,
                    )
                    evidence_inventory = _evidence_inventory(clone, profile=profile)
                    preflight = {
                        **preflight,
                        "baseline": _baseline_binding(store, clone, materialized),
                        "training_snapshot_sha256": materialized.get("artifact_id"),
                        "official_r5_evidence": {
                            "materialization_artifact_id": materialized.get("artifact_id"),
                            "label_chain_head": materialized.get("label_chain_head"),
                            "rows_sha256": _sha(rows),
                            "gate_sha256": _sha(gate),
                            "floor_policy_sha256": _sha(policy),
                        },
                    }
                    api_error = None
                except Exception as exc:  # capture errors must not masquerade as a valid dataset
                    rows, rallies, preflight, gate, workset, policy, api_error = (
                        [], [], {}, {"passed": False}, {}, None, type(exc).__name__,
                    )
        dataset = validate_dataset(
            rows=rows,
            labels=labels,
            rallies=rallies,
            preflight=preflight,
            gate=gate,
            workset=workset,
            root=clone,
            distill=distill,
            store=store,
            workset_runtime=workset_runtime,
            evidence_inventory=evidence_inventory,
            policy=policy,
        )
        if api_error:
            dataset["passed"] = False
            dataset["capture_only"] = True
            dataset["reasons"] = sorted(
                set([*dataset["reasons"], "clone_api_capture_failed"])
            )
        phases.append(
            {
                "name": "clone_materialize_preflight_gate",
                "elapsed_ms": round((time.monotonic() - phase) * 1000, 3),
            }
        )
    finally:
        if clone is not None:
            _cleanup_clone(clone)
            cleanup = {
                "clone_removed": not clone.exists(),
                "remaining": int(clone.exists()),
            }
        _restore_chronovisor_modules(runtime_modules)
        production_after = production_state(production)
        source_after = source_state(source, source_commit)
    if source_before != source_after or production_before != production_after:
        raise R5Error("source or production changed during harness")
    payload = {
        "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source_before,
        "source_after": source_after,
        "production": production_before,
        "production_after": production_after,
        "clone": clone_info,
        "r4_dependency": r4_dependency,
        "dataset": dataset,
        "phases": phases,
        "cleanup": cleanup,
        "provider_calls": sentinels["provider_calls"],
        "egress_attempts": sentinels["egress_attempts"],
        "process_attempts": sentinels["process_attempts"],
        "supervised": False,
        "test_only": test_only,
    }
    artifact = _sealed_artifact(payload)
    return artifact, _write_immutable(output, artifact)


def _directory_identity(path: Path) -> dict[str, int]:
    value = path.lstat()
    if (
        stat.S_ISLNK(value.st_mode)
        or not stat.S_ISDIR(value.st_mode)
        or value.st_uid != ACCOUNT_UID
        or value.st_mode & 0o022
    ):
        raise R5Error("owned directory is unsafe")
    return {"dev": value.st_dev, "ino": value.st_ino, "mode": value.st_mode & 0o7777}


def _root_identity(path: Path) -> dict[str, int]:
    """Production root identity includes metadata that a mutable work root cannot pin."""
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise R5Error("production root is unsafe")
    return {
        "dev": value.st_dev, "ino": value.st_ino, "uid": value.st_uid,
        "gid": value.st_gid, "mode": value.st_mode & 0o7777,
        "ctime_ns": value.st_ctime_ns,
    }


def _cleanup_owned_directory(
    path: Path,
    *,
    parent: Path,
    parent_identity: Mapping[str, int],
    identity: Mapping[str, int],
) -> int:
    """Remove only the directory the parent created; never chase a replacement."""
    if path.parent != parent:
        raise R5Error("owned cleanup parent changed")
    if _directory_identity(parent) != dict(parent_identity):
        raise R5Error("owned cleanup root changed")
    try:
        current = _directory_identity(path)
    except FileNotFoundError:
        return 0
    if current != dict(identity):
        raise R5Error("owned cleanup identity changed")
    shutil.rmtree(path)
    if os.path.lexists(path):
        raise R5Error("owned cleanup failed")
    return 0


def _cleanup_created_directory(path: Path, *, parent: Path, parent_identity: Mapping[str, int]) -> None:
    """Best-effort cleanup for a mkdtemp before its strict identity is acquired."""
    if path.parent != parent or _directory_identity(parent) != dict(parent_identity):
        raise R5Error("created cleanup parent changed")
    try:
        value = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode) or value.st_uid != ACCOUNT_UID:
        raise R5Error("created cleanup path is unsafe")
    shutil.rmtree(path)
    if os.path.lexists(path):
        raise R5Error("created cleanup failed")


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    if _has_symlink_component(path) or path.name in {"", ".", ".."}:
        raise R5Error("private manifest path is unsafe")
    parent_identity = _directory_identity(path.parent)
    raw = _canonical(value) + b"\n"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_fd, _ = _open_owned_directory(path.parent, expected=parent_identity)
    try:
        fd = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        state = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(state.st_mode)
            or state.st_uid != ACCOUNT_UID
            or state.st_mode & 0o077
        ):
            raise R5Error("private manifest identity failed")
        read_fd = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        try:
            if os.read(read_fd, len(raw) + 1) != raw or os.read(read_fd, 1):
                raise R5Error("private manifest readback failed")
        finally:
            os.close(read_fd)
    finally:
        os.close(parent_fd)
    verify_fd, _ = _open_owned_directory(path.parent, expected=parent_identity)
    try:
        readback, readback_state = _read_private_json_at(verify_fd, path.name, label="private manifest")
    finally:
        os.close(verify_fd)
    if readback != raw or readback_state["sha256"] != hashlib.sha256(raw).hexdigest():
        raise R5Error("private manifest changed during finalization")


def _trusted_interpreter() -> Path:
    interpreter = Path(sys.executable).resolve(strict=True)
    value = interpreter.lstat()
    if (
        _has_symlink_component(interpreter)
        or not stat.S_ISREG(value.st_mode)
        or value.st_mode & 0o022
        or value.st_uid not in {0, ACCOUNT_UID}
    ):
        raise R5Error("sys.executable identity failed")
    return interpreter


def _trusted_harness_script() -> Path:
    script = Path(__file__).resolve(strict=True)
    value = script.lstat()
    if (
        _has_symlink_component(script)
        or not stat.S_ISREG(value.st_mode)
        or value.st_uid != ACCOUNT_UID
        or value.st_mode & 0o022
    ):
        raise R5Error("harness script identity failed")
    return script


def _formal_sandbox_policy(interpreter: Path, *write_roots: Path) -> str:
    """Kernel policy: no fork/exec/network or writes outside pinned owned roots."""
    allowed_writes = "".join(
        f"(allow file-write* (subpath {json.dumps(str(root.resolve()))}))"
        for root in write_roots
    )
    return (
        "(version 1)(deny network*)(deny process-fork)(deny process-exec)(deny file-write*)"
        f"(allow process-exec (literal {json.dumps(str(interpreter))})){allowed_writes}"
    )


def _register_child_pid() -> None:
    """Test-only descendants prove ownership before they can escape a process group."""
    raw_fd = os.environ.get("R5_CHILD_PID_REGISTRY_FD")
    if raw_fd is None:
        return
    try:
        fd = int(raw_fd)
    except ValueError as exc:
        raise R5Error("child PID registry descriptor is invalid") from exc
    os.write(fd, f"{os.getpid()}\n".encode("ascii"))


def _process_record(pid: int) -> dict[str, Any] | None:
    """Return one trusted, direct process identity lookup; never scan processes."""
    if pid <= 0:
        return None
    argv = ["/bin/ps", "-o", "pid=", "-o", "ppid=", "-o", "uid=", "-o", "lstart=", "-p", str(pid)]
    _trusted_tool(Path(argv[0]))
    try:
        result = subprocess.run(
            argv,
            cwd=Path("/"),
            check=False,
            capture_output=True,
            timeout=10,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise R5Error("trusted process lookup failed") from exc
    if result.returncode:
        return None
    raw = result.stdout
    lines = raw.decode("ascii", "strict").splitlines()
    if len(lines) != 1:
        return None
    fields = lines[0].split(maxsplit=3)
    if len(fields) != 4 or any(not value.isdecimal() for value in fields[:3]):
        raise R5Error("trusted process lookup is malformed")
    found_pid, ppid, uid = (int(value) for value in fields[:3])
    if found_pid != pid or ppid < 0 or uid < 0 or not fields[3]:
        raise R5Error("trusted process lookup is malformed")
    return {"pid": found_pid, "ppid": ppid, "uid": uid, "start": fields[3]}


def _identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("pid", "uid", "start")}


def _authenticated_descendant(
    pid: int, root: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Accept a registration only while its live parent chain reaches root."""
    current = _process_record(pid)
    if current is None or current["uid"] != root["uid"]:
        return None
    target = _identity(current)
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    while True:
        current_pid = int(current["pid"])
        if current_pid in seen or len(seen) >= 64 or current["uid"] != root["uid"]:
            return None
        seen.add(current_pid)
        chain.append(_identity(current))
        if current_pid == root["pid"]:
            if _identity(current) != _identity(root):
                return None
            return {**target, "chain": chain}
        parent_pid = int(current["ppid"])
        if parent_pid <= 0:
            return None
        current = _process_record(parent_pid)
        if current is None:
            return None


class _ChildPidRegistry:
    """Pipe-backed, registration-time authenticated descendants only."""

    def __init__(self, fd: int, root: Mapping[str, Any]) -> None:
        self.fd = fd
        self.root = _identity(root)
        self.records: dict[int, dict[str, Any]] = {}
        self._buffer = b""
        self.closed = False

    def drain(self) -> None:
        while not self.closed:
            try:
                raw = os.read(self.fd, 64 * 1024)
            except BlockingIOError:
                return
            if not raw:
                self.closed = True
                if self._buffer:
                    raise R5Error("child PID registry is malformed")
                return
            self._buffer += raw
            while b"\n" in self._buffer:
                line, self._buffer = self._buffer.split(b"\n", 1)
                if not line or not line.isdigit():
                    raise R5Error("child PID registry is malformed")
                pid = int(line)
                if pid <= 0:
                    raise R5Error("child PID registry is malformed")
                record = _authenticated_descendant(pid, self.root)
                if record is not None:
                    self.records[pid] = record

    def live_pids(self, *, exclude: int | None = None) -> set[int]:
        result: set[int] = set()
        for pid, registered in self.records.items():
            if pid == exclude:
                continue
            current = _process_record(pid)
            if current is not None and _identity(current) == _identity(registered):
                result.add(pid)
        return result

    def wait_closed(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while not self.closed and time.monotonic() < deadline:
            self.drain()
            if not self.closed:
                time.sleep(0.01)
        self.drain()
        return self.closed


def _signal_registered(registry: _ChildPidRegistry, sig: signal.Signals, root_pid: int) -> None:
    for pid in registry.live_pids(exclude=root_pid):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _root_group_exists(process: subprocess.Popen[bytes], pgid: int, registry: _ChildPidRegistry) -> bool:
    """A process group is safe to signal only while its original leader exists."""
    root = _process_record(process.pid)
    if root is None or _identity(root) != registry.root:
        return False
    try:
        return os.getpgid(process.pid) == pgid
    except ProcessLookupError:
        return False


def _terminate_process_group(
    process: subprocess.Popen[bytes], pgid: int, registry: _ChildPidRegistry
) -> int:
    """Terminate the child group plus only registry-proven escaped children."""
    registry.drain()
    if _root_group_exists(process, pgid, registry):
        os.killpg(pgid, signal.SIGTERM)
    _signal_registered(registry, signal.SIGTERM, process.pid)
    registry.wait_closed(1)
    registry.drain()
    if _root_group_exists(process, pgid, registry):
        os.killpg(pgid, signal.SIGKILL)
    _signal_registered(registry, signal.SIGKILL, process.pid)
    registry.wait_closed(1)
    # Reap buffered pipes after wait; this is bounded even for a hostile child.
    try:
        process.communicate(timeout=1)
    except subprocess.TimeoutExpired as exc:
        raise R5Error("child pipes did not close") from exc
    registry.drain()
    return int(
        _root_group_exists(process, pgid, registry)
        or bool(registry.live_pids(exclude=process.pid))
        or not registry.closed
    )


def _terminate_unregistered_child(process: subprocess.Popen[bytes], pgid: int) -> int:
    """Close Popen's pre-registry gap without trusting a process enumeration."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass  # Direct Popen PID termination below remains the fail-closed fallback.
        try:
            os.kill(process.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            if process.poll() is None:
                raise R5Error("unregistered child cannot be terminated") from None
        try:
            process.communicate(timeout=1)
            return 0
        except subprocess.TimeoutExpired:
            continue
    raise R5Error("unregistered child pipes did not close")


def _communicate_with_watchdog(
    process: subprocess.Popen[bytes], watchdog_seconds: int, registry: _ChildPidRegistry,
) -> None:
    """Poll only the explicit registry until the bounded child completes."""
    deadline = time.monotonic() + watchdog_seconds
    while True:
        registry.drain()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, watchdog_seconds)
        try:
            process.communicate(timeout=min(0.1, remaining))
            registry.drain()
            return
        except subprocess.TimeoutExpired:
            continue


def _sealed_completion(payload: Mapping[str, Any]) -> dict[str, Any]:
    _validate_completion_nested(payload)
    unsigned = {"schema": R5_COMPLETION_SCHEMA, "namespace": NAMESPACE, **payload}
    artifact_id = _sha(unsigned)
    return {
        "artifact_id": artifact_id,
        **unsigned,
        "seal_sha256": _sha({"artifact_id": artifact_id, **unsigned}),
    }


def _validate_completion_nested(payload: Mapping[str, Any]) -> None:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_id", "schema", "namespace", "seal_sha256"}
    }
    _closed_mapping(
        core,
        {
            "inner", "supervisor", "source", "source_after", "source_final", "production",
            "production_after", "production_final", "output", "cleanup", "formal_passed",
        },
        label="completion payload",
    )
    for key in ("source", "source_after", "source_final"):
        _validate_identity_state(core.get(key), label=f"completion.{key}")
    for key in ("production", "production_after", "production_final"):
        _validate_production_state(core.get(key), label=f"completion.{key}")
    _closed_mapping(
        core.get("inner"),
        {"artifact_id", "path", "file_sha256", "source_commit", "r4_artifact_id", "parent_dev", "parent_ino"},
        label="completion.inner",
    )
    _closed_mapping(
        core.get("supervisor"),
        {
            "pid", "pgid", "started_at", "ended_at", "elapsed_ms", "deadline_seconds",
            "returncode", "signal", "timeout", "descendants_remaining", "observed_descendant_pids",
        },
        label="completion.supervisor",
    )
    _closed_mapping(core.get("output"), {"dev", "ino"}, label="completion.output")
    _closed_mapping(
        core.get("cleanup"), {"clone_remaining", "temporary_remaining"}, label="completion.cleanup"
    )
    if core.get("formal_passed") is True and (
        any(set(core[key]) != _SOURCE_KEYS for key in ("source", "source_after", "source_final"))
        or any(set(core[key]) != _PRODUCTION_KEYS for key in ("production", "production_after", "production_final"))
    ):
        raise R5Error("formal completion requires complete identity records")


def read_completion(path: Path) -> dict[str, Any]:
    parent_fd, completion_parent = _open_owned_directory(path.parent)
    try:
        raw, _state = _read_private_json_at(parent_fd, path.name, label="completion")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise R5Error("completion unreadable") from exc
    finally:
        os.close(parent_fd)
    expected = {
        "artifact_id",
        "schema",
        "namespace",
        "seal_sha256",
        "inner",
        "supervisor",
        "source",
        "source_after",
        "source_final",
        "production",
        "production_after",
        "production_final",
        "output",
        "cleanup",
        "formal_passed",
    }
    if (
        not isinstance(value, dict)
        or value.get("schema") != R5_COMPLETION_SCHEMA
        or value.get("namespace") != NAMESPACE
        or set(value) != expected
        or not all(isinstance(value.get(key), Mapping) for key in (
            "inner", "supervisor", "source", "source_after", "production",
            "source_final", "production_after", "production_final", "output", "cleanup",
        ))
        or value.get("formal_passed") is not True
    ):
        raise R5Error("completion schema mismatch")
    if raw not in {_canonical(value), _canonical(value) + b"\n"}:
        raise R5Error("completion is not canonical")
    _validate_completion_nested(value)
    inner = value["inner"]
    supervisor = value["supervisor"]
    output = value["output"]
    cleanup = value["cleanup"]
    started_at = _utc_timestamp(supervisor.get("started_at"), label="completion.started_at")
    ended_at = _utc_timestamp(supervisor.get("ended_at"), label="completion.ended_at")
    now = datetime.now(UTC)
    if (
        set(inner)
        != {"artifact_id", "path", "file_sha256", "source_commit", "r4_artifact_id", "parent_dev", "parent_ino"}
        or set(supervisor)
        != {
            "pid", "pgid", "started_at", "ended_at", "elapsed_ms",
            "deadline_seconds", "returncode", "signal", "timeout",
            "descendants_remaining", "observed_descendant_pids",
        }
        or set(output) != {"dev", "ino"}
        or set(cleanup) != {"clone_remaining", "temporary_remaining"}
    ):
        raise R5Error("completion nested keys are not closed")
    if (
        not all(_is_id(inner.get(key)) for key in ("artifact_id", "file_sha256", "r4_artifact_id"))
        or not isinstance(inner.get("path"), str)
        or not Path(inner["path"]).is_absolute()
        or not isinstance(inner.get("source_commit"), str)
        or len(inner["source_commit"]) != 40
        or bool(set(inner["source_commit"]) - set("0123456789abcdef"))
        or not all(isinstance(inner.get(key), int) and inner[key] > 0 for key in ("parent_dev", "parent_ino"))
        or output != {"dev": completion_parent["dev"], "ino": completion_parent["ino"]}
        or not isinstance(supervisor.get("pid"), int)
        or supervisor["pid"] <= 0
        or not isinstance(supervisor.get("pgid"), int)
        or supervisor["pgid"] != supervisor["pid"]
        or not isinstance(supervisor.get("deadline_seconds"), int)
        or supervisor["deadline_seconds"] <= 0
        or not isinstance(supervisor.get("elapsed_ms"), (int, float))
        or isinstance(supervisor["elapsed_ms"], bool)
        or supervisor["elapsed_ms"] < 0
        or not all(isinstance(supervisor.get(key), str) and supervisor[key] for key in ("started_at", "ended_at"))
        or started_at > now + timedelta(seconds=MAX_SUPERVISOR_FUTURE_SKEW_SECONDS)
        or ended_at > now + timedelta(seconds=MAX_SUPERVISOR_FUTURE_SKEW_SECONDS)
        or ended_at < started_at
        or abs((ended_at - started_at).total_seconds() * 1000 - supervisor["elapsed_ms"]) > 1.0
        or supervisor["elapsed_ms"] > supervisor["deadline_seconds"] * 1000 + SUPERVISOR_SCHEDULER_TOLERANCE_MS
        or not isinstance(supervisor.get("returncode"), int)
        or supervisor["returncode"] != 0
        or supervisor.get("signal") is not None
        or supervisor.get("timeout") is not False
        or supervisor.get("descendants_remaining") != 0
        or not isinstance(supervisor.get("observed_descendant_pids"), list)
        or any(
            not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
            for pid in supervisor["observed_descendant_pids"]
        )
        or supervisor["observed_descendant_pids"]
        != sorted(set(supervisor["observed_descendant_pids"]))
        or cleanup != {"clone_remaining": 0, "temporary_remaining": 0}
    ):
        raise R5Error("completion nested values are invalid")
    unsigned = {
        key: item
        for key, item in value.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    artifact_id = _sha(unsigned)
    if value.get("artifact_id") != artifact_id or value.get("seal_sha256") != _sha(
        {"artifact_id": artifact_id, **unsigned}
    ):
        raise R5Error("completion seal mismatch")
    inner_path = Path(str(inner["path"]))
    if inner_path.parent.name != "r5-inner" or inner_path.name != f"{inner['artifact_id']}.json":
        raise R5Error("completion inner path is not official")
    if _has_symlink_component(inner_path):
        raise R5Error("completion inner path contains a symlink")
    try:
        bound_inner = read_artifact(
            inner_path,
            parent_identity={
                "dev": inner["parent_dev"], "ino": inner["parent_ino"],
                "mode": _directory_identity(inner_path.parent)["mode"],
            },
        )
    except (OSError, R5Error) as exc:
        raise R5Error("completion inner artifact is unavailable") from exc
    if (
        bound_inner.get("artifact_id") != inner["artifact_id"]
        or bound_inner.get("test_only") is not False
        or _file_state(inner_path)["sha256"] != inner["file_sha256"]
        or bound_inner.get("r4_dependency", {}).get("artifact_id")
        != inner["r4_artifact_id"]
    ):
        raise R5Error("completion inner binding failed")
    captured_at = _utc_timestamp(bound_inner.get("captured_at"), label="completion.inner.captured_at")
    if not started_at <= captured_at <= ended_at:
        raise R5Error("completion supervisor interval does not bind inner capture")
    return value


def _write_completion(output: Path, completion: Mapping[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    output_identity = _directory_identity(output)
    if completion.get("output") != {"dev": output_identity["dev"], "ino": output_identity["ino"]}:
        raise R5Error("completion output identity mismatch")
    path = output / f"{completion['artifact_id']}.completion.json"
    raw = _canonical(completion) + b"\n"
    if path.exists():
        try:
            existing = read_completion(path)
        except R5Error as exc:
            raise R5Error("immutable completion conflict") from exc
        if _canonical(existing) + b"\n" != raw:
            raise R5Error("immutable completion conflict")
        return path
    _write_private_json(path, completion)
    if _directory_identity(output) != output_identity:
        raise R5Error("completion parent changed")
    read_completion(path)
    return path


def assert_formal_acceptance(
    inner: Mapping[str, Any],
    completion: Mapping[str, Any],
    *,
    source_root: Path,
    r4_artifact: Path,
) -> None:
    """Reject every execution artifact that lacks its matching parent receipt."""
    _validate_artifact_nested(inner)
    if inner.get("supervised") is not False or inner.get("test_only") is not False:
        raise R5Error("inner artifact is not an execution receipt")
    dataset, inner_cleanup, dependency = (
        inner.get("dataset"),
        inner.get("cleanup"),
        inner.get("r4_dependency"),
    )
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("passed") is not True
        or not isinstance(inner_cleanup, Mapping)
        or inner_cleanup != {"clone_removed": True, "remaining": 0}
        or not isinstance(dependency, Mapping)
        or set(dependency) != _R4_DEPENDENCY_KEYS
        or any(inner.get(key) != 0 for key in (
            "provider_calls", "egress_attempts", "process_attempts"
        ))
    ):
        raise R5Error("inner artifact cannot certify R5")
    receipt = completion.get("inner")
    supervisor = completion.get("supervisor")
    cleanup = completion.get("cleanup")
    if not isinstance(receipt, Mapping) or not isinstance(supervisor, Mapping) or not isinstance(cleanup, Mapping):
        raise R5Error("completion nested schema invalid")
    if not all(isinstance(receipt.get(key), int) and receipt[key] > 0 for key in ("parent_dev", "parent_ino")):
        raise R5Error("completion inner parent identity invalid")
    receipt_path = Path(str(receipt.get("path") or ""))
    bound_inner = read_artifact(
        receipt_path,
        parent_identity={
            "dev": receipt["parent_dev"], "ino": receipt["parent_ino"],
            "mode": _directory_identity(receipt_path.parent)["mode"],
        },
    )
    if dict(bound_inner) != dict(inner):
        raise R5Error("completion does not bind the supplied inner artifact")
    inner_source = inner.get("source")
    inner_source_after = inner.get("source_after")
    inner_production = inner.get("production")
    inner_production_after = inner.get("production_after")
    if (
        not isinstance(inner_source, Mapping)
        or not isinstance(inner_source_after, Mapping)
        or not isinstance(inner_production, Mapping)
        or not isinstance(inner_production_after, Mapping)
    ):
        raise R5Error("inner identity schema invalid")
    if (
        receipt.get("artifact_id") != inner.get("artifact_id")
        or not _is_id(receipt.get("file_sha256"))
        or receipt.get("source_commit") != inner_source.get("commit")
        or receipt.get("r4_artifact_id") != dependency.get("artifact_id")
        or not isinstance(receipt.get("path"), str)
        or not Path(receipt["path"]).is_absolute()
        or supervisor.get("returncode") != 0
        or supervisor.get("signal") is not None
        or supervisor.get("timeout") is not False
        or supervisor.get("descendants_remaining") != 0
        or not isinstance(supervisor.get("observed_descendant_pids"), list)
        or any(
            not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
            for pid in supervisor.get("observed_descendant_pids", [])
        )
        or supervisor.get("observed_descendant_pids")
        != sorted(set(supervisor.get("observed_descendant_pids", [])))
        or cleanup != {"clone_remaining": 0, "temporary_remaining": 0}
        or completion.get("formal_passed") is not True
    ):
        raise R5Error("completion does not close formal acceptance")
    if (
        completion.get("source") != inner_source
        or completion.get("source_after") != inner_source_after
        or completion.get("source_final") != inner_source_after
        or completion.get("production") != inner_production
        or completion.get("production_after") != inner_production_after
        or completion.get("production_final") != inner_production_after
        or completion.get("source") != completion.get("source_after")
        or completion.get("source_after") != completion.get("source_final")
        or completion.get("production") != completion.get("production_after")
        or completion.get("production_after") != completion.get("production_final")
    ):
        raise R5Error("completion source or production binding failed")
    if (
        inner_production != production_state(PRODUCTION_ROOT)
        or inner.get("clone", {}).get("state") != _clone_input_state(PRODUCTION_ROOT)
    ):
        raise R5Error("completion clone or production rederivation failed")
    if (
        dependency.get("source_commit") != inner_source.get("commit")
        or dependency.get("source_tree_sha256") != inner_source.get("tree_sha256")
        or dependency.get("source_root") != str(source_root.expanduser().resolve(strict=True))
        or dependency != _verify_r4(r4_artifact, inner_source, source_root)
    ):
        raise R5Error("completion R4 authority binding failed")


def run_supervised(
    *,
    production: Path,
    source: Path,
    source_commit: str,
    output: Path,
    r4_artifact: Path,
    watchdog_seconds: int = 600,
    test_only: bool = False,
    test_child_action: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run the capture in a killable child and issue the only formal R5 receipt."""
    if not 1 <= watchdog_seconds <= 3600:
        raise R5Error("watchdog must be between one second and one hour")
    if test_child_action is not None and not test_only:
        raise R5Error("test child actions require test_only")
    production = production.resolve(strict=True)
    source = source.resolve(strict=True)
    output = output.resolve(strict=False)
    if production != PRODUCTION_ROOT and not test_only:
        raise R5Error("only the fixed OS production root may certify R5")
    assert_root_matrix(production, source, output)
    output.mkdir(parents=True, exist_ok=True)
    output_identity = _directory_identity(output)
    source_before = source_state(source, source_commit)
    production_before = production_state(production)
    r4_dependency = _verify_r4(r4_artifact, source_before, source)
    production_parent_identity = _directory_identity(production.parent)
    clone: Path | None = None
    clone_identity: dict[str, int] | None = None
    workspace: Path | None = None
    workspace_identity: dict[str, int] | None = None
    try:
        clone = Path(
            tempfile.mkdtemp(
                prefix="chronovisor-r5-supervisor-clone-", dir=production.parent
            )
        )
        clone_identity = _directory_identity(clone)
        workspace = Path(tempfile.mkdtemp(prefix=".chronovisor-r5-supervisor-", dir=output))
        workspace_identity = _directory_identity(workspace)
        child_output = workspace / "inner-output"
        child_output.mkdir(mode=0o700)
        _write_private_json(
            workspace / "manifest.json",
            {
                "clone": str(clone),
                "inner_output": str(child_output),
                "source": str(source),
                "test_inner": {
                    "source": source_before,
                    "production": production_before,
                    "r4_dependency": r4_dependency,
                },
            },
        )
    except Exception:
        cleanup_error: Exception | None = None
        if workspace is not None and workspace_identity is not None:
            try:
                _cleanup_owned_directory(
                    workspace, parent=output, parent_identity=output_identity,
                    identity=workspace_identity,
                )
            except Exception as exc:
                cleanup_error = exc
        elif workspace is not None:
            try:
                _cleanup_created_directory(
                    workspace, parent=output, parent_identity=output_identity,
                )
            except Exception as exc:
                cleanup_error = exc
        if clone is not None and clone_identity is not None:
            try:
                _cleanup_owned_directory(
                    clone, parent=production.parent,
                    parent_identity=production_parent_identity, identity=clone_identity,
                )
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        elif clone is not None:
            try:
                _cleanup_created_directory(
                    clone, parent=production.parent,
                    parent_identity=production_parent_identity,
                )
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            raise cleanup_error from None
        raise
    assert (
        clone is not None
        and clone_identity is not None
        and workspace is not None
        and workspace_identity is not None
    )
    started_wall = datetime.now(UTC)
    process: subprocess.Popen[bytes] | None = None
    pgid: int | None = None
    registry_read: int | None = None
    registry_write: int | None = None
    registry: _ChildPidRegistry | None = None
    timeout = False
    descendants_remaining = 0
    failure: Exception | None = None
    inner: dict[str, Any] | None = None
    inner_path: Path | None = None
    inner_hash: str | None = None
    source_after: dict[str, Any] | None = None
    production_after: dict[str, Any] | None = None
    try:
        interpreter, script = _trusted_interpreter(), _trusted_harness_script()
        sandbox_attestation = hashlib.sha256(os.urandom(32)).hexdigest()
        command = [
            str(interpreter), "-I", str(script), "--inner-run",
            "--source-root", str(source), "--source-commit", source_commit,
            "--output", str(child_output), "--r4-artifact", str(r4_artifact),
            "--watchdog-seconds", str(watchdog_seconds), "--owned-clone", str(clone),
            "--kernel-sandbox-attestation", sandbox_attestation,
        ]
        if test_only:
            command.extend(("--test-production-root", str(production)))
        if test_child_action is not None:
            command = [
                str(interpreter), "-I", str(script), "--test-child-action", test_child_action,
                "--test-only-child", "--output", str(child_output),
                "--source-commit", source_commit,
                "--test-manifest", str(workspace / "manifest.json"),
            ]
        sandbox = Path("/usr/bin/sandbox-exec")
        _trusted_tool(sandbox)
        policy = _formal_sandbox_policy(interpreter, clone, workspace, child_output)
        command = [str(sandbox), "-p", policy, *command]
        registry_read, registry_write = os.pipe()
        os.set_blocking(registry_read, False)
        child_env = {
            "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0", "R5_CHILD_PID_REGISTRY_FD": str(registry_write),
            "R5_KERNEL_SANDBOX_ATTESTATION": sandbox_attestation,
        }
        process = subprocess.Popen(
            command,
            cwd=source,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            start_new_session=True,
            pass_fds=(registry_write,),
        )
        os.close(registry_write)
        registry_write = None
        # start_new_session makes pid the intended group identity; retain it so
        # an os.getpgid failure cannot strand the just-started child.
        pgid = process.pid
        if os.getpgid(process.pid) != pgid:
            raise R5Error("child did not receive its own process group")
        root = _process_record(process.pid)
        if root is None or root["uid"] != ACCOUNT_UID:
            raise R5Error("child process identity failed")
        registry = _ChildPidRegistry(registry_read, root)
        try:
            _communicate_with_watchdog(process, watchdog_seconds, registry)
        except subprocess.TimeoutExpired:
            timeout = True
            descendants_remaining = _terminate_process_group(process, pgid, registry)
            raise R5Error("supervisor watchdog expired") from None
        if (
            _root_group_exists(process, pgid, registry)
            or not registry.wait_closed(0.25)
            or registry.live_pids(exclude=process.pid)
        ):
            descendants_remaining = _terminate_process_group(process, pgid, registry)
            raise R5Error("child left descendants")
        if process.returncode != 0:
            raise R5Error(f"inner child failed with return code {process.returncode}")
        paths = sorted(child_output.glob("*.json"))
        if len(paths) != 1:
            raise R5Error("inner child did not produce exactly one artifact")
        inner_path = paths[0]
        inner = read_artifact(inner_path)
        inner_hash = str(_file_state(inner_path)["sha256"])
    except Exception as exc:
        failure = exc
        if process is not None and pgid is not None and registry is not None:
            try:
                descendants_remaining = _terminate_process_group(process, pgid, registry)
            except Exception as cleanup_exc:
                failure = cleanup_exc
        elif process is not None and pgid is not None:
            try:
                descendants_remaining = _terminate_unregistered_child(process, pgid)
            except Exception as cleanup_exc:
                failure = cleanup_exc
    finally:
        if registry_write is not None:
            os.close(registry_write)
        if registry_read is not None:
            os.close(registry_read)
        cleanup = {"clone_remaining": 1, "temporary_remaining": 1}
        try:
            cleanup_errors: list[Exception] = []
            try:
                clone_remaining = _cleanup_owned_directory(
                    clone, parent=production.parent,
                    parent_identity=production_parent_identity, identity=clone_identity,
                )
            except Exception as exc:
                cleanup_errors.append(exc)
            try:
                temporary_remaining = _cleanup_owned_directory(
                    workspace, parent=output, parent_identity=output_identity,
                    identity=workspace_identity,
                )
            except Exception as exc:
                cleanup_errors.append(exc)
            cleanup = {
                "clone_remaining": clone_remaining,
                "temporary_remaining": temporary_remaining,
            }
            if cleanup_errors:
                raise cleanup_errors[0]
        except Exception as cleanup_exc:
            if failure is None:
                failure = cleanup_exc
        # Snapshot errors never skip cleanup: they are intentionally last.
        try:
            source_after = source_state(source, source_commit)
            production_after = production_state(production)
        except Exception as snapshot_exc:
            if failure is None:
                failure = snapshot_exc
            source_after = None
            production_after = None
    if failure is not None:
        raise R5Error(str(failure)) from failure
    source_final = source_state(source, source_commit)
    production_final = production_state(production)
    if (
        inner is None
        or inner_path is None
        or inner_hash is None
        or source_after is None
        or production_after is None
        or source_before != source_after
        or source_after != source_final
        or production_before != production_after
        or production_after != production_final
        or _directory_identity(output) != output_identity
    ):
        raise R5Error("supervisor source, production, or output changed")
    if test_only:
        raise R5Error("test-only supervisor cannot issue a formal completion")
    persistent_inner_path, persistent_inner_parent = _publish_inner(output, inner)
    persistent_inner_hash = str(_file_state(persistent_inner_path)["sha256"])
    ended_wall = datetime.now(UTC)
    elapsed_ms = int(
        (ended_wall.replace(microsecond=0) - started_wall.replace(microsecond=0)).total_seconds() * 1000
    )
    completion = _sealed_completion(
        {
            "inner": {
                "artifact_id": inner["artifact_id"],
                "path": str(persistent_inner_path),
                "file_sha256": persistent_inner_hash,
                "source_commit": source_commit,
                "r4_artifact_id": inner["r4_dependency"]["artifact_id"],
                "parent_dev": persistent_inner_parent["dev"],
                "parent_ino": persistent_inner_parent["ino"],
            },
            "supervisor": {
                "pid": process.pid if process is not None else None,
                "pgid": pgid,
                "started_at": started_wall.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ended_at": ended_wall.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "elapsed_ms": elapsed_ms,
                "deadline_seconds": watchdog_seconds,
                "returncode": process.returncode if process is not None else None,
                "signal": None,
                "timeout": timeout,
                "descendants_remaining": descendants_remaining,
                "observed_descendant_pids": (
                    sorted(pid for pid in registry.records if process is None or pid != process.pid)
                    if registry is not None
                    else []
                ),
            },
            "source": source_before,
            "source_after": source_after,
            "source_final": source_final,
            "production": production_before,
            "production_after": production_after,
            "production_final": production_final,
            "output": {"dev": output_identity["dev"], "ino": output_identity["ino"]},
            "cleanup": cleanup,
            "formal_passed": True,
        }
    )
    assert_formal_acceptance(
        inner, completion, source_root=source, r4_artifact=r4_artifact,
    )
    path = _write_completion(output, completion)
    read_completion(path)
    return completion, path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--owned-clone", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--kernel-sandbox-attestation", help=argparse.SUPPRESS)
    parser.add_argument("--test-child-action", choices=("exit-0", "fast-valid", "raise", "write-outside", "alias-write-outside", "alias-network", "alias-process", "fork-double-setsid-ignore", "fork-setsid-ignore", "fork-sleep-ignore"), help=argparse.SUPPRESS)
    parser.add_argument("--test-only-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--test-production-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--r4-artifact", type=Path)
    parser.add_argument("--watchdog-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if args.test_child_action is not None:
        if not args.test_only_child:
            print("r5 harness failed: test child action requires --test-only-child", file=sys.stderr)
            return 2
        if args.test_child_action == "raise":
            raise R5Error("test child exception")
        if args.test_child_action == "write-outside":
            with _egress_sentinel():
                fd = os.open("/tmp/chronovisor-r5-outside-write", os.O_WRONLY | os.O_CREAT, 0o600)
                try:
                    os.write(fd, b"blocked")
                finally:
                    os.close(fd)
        if args.test_child_action == "alias-write-outside":
            fd = _PRE_SENTINEL_OPEN("/tmp/chronovisor-r5-alias-write", os.O_WRONLY | os.O_CREAT, 0o600)
            try:
                _PRE_SENTINEL_WRITE(fd, b"blocked")
            finally:
                _PRE_SENTINEL_CLOSE(fd)
        if args.test_child_action == "alias-network":
            _PRE_SENTINEL_SOCKET(socket.AF_INET, socket.SOCK_STREAM).connect(("127.0.0.1", 9))
        if args.test_child_action == "alias-process":
            _PRE_SENTINEL_POPEN(["/usr/bin/true"])
        if args.test_child_action == "fast-valid":
            if args.output is None or args.source_commit is None or args.test_manifest is None:
                print("r5 harness failed: fast-valid test child is incomplete", file=sys.stderr)
                return 2
            manifest = json.loads(args.test_manifest.read_bytes())
            test_inner = manifest.get("test_inner")
            if not isinstance(test_inner, Mapping):
                print("r5 harness failed: fast-valid test manifest is invalid", file=sys.stderr)
                return 2
            payload = {
                "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": test_inner["source"],
                "source_after": test_inner["source"],
                "production": test_inner["production"],
                "production_after": test_inner["production"],
                "clone": {"test_only": True},
                "r4_dependency": test_inner["r4_dependency"],
                "dataset": {"passed": True, "test_only": True},
                "phases": [],
                "cleanup": {"clone_removed": True, "remaining": 0},
                "provider_calls": 0,
                "egress_attempts": 0,
                "process_attempts": 0,
                "supervised": False,
                "test_only": True,
            }
            _write_immutable(args.output, _sealed_artifact(payload))
            return 0
        if args.test_child_action == "fork-sleep-ignore":
            _register_child_pid()
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if os.fork() == 0:
                _register_child_pid()
                time.sleep(60)
                os._exit(0)
            time.sleep(60)
        if args.test_child_action == "fork-setsid-ignore":
            _register_child_pid()
            if os.fork() == 0:
                _register_child_pid()
                os.setsid()
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                time.sleep(60)
                os._exit(0)
            time.sleep(60)
        if args.test_child_action == "fork-double-setsid-ignore":
            _register_child_pid()
            if os.fork() == 0:
                _register_child_pid()
                os.setsid()
                if os.fork() == 0:
                    _register_child_pid()
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    time.sleep(60)
                    os._exit(0)
                time.sleep(60)
                os._exit(0)
            time.sleep(60)
        return 0
    if (
        args.source_root is None
        or args.source_commit is None
        or args.output is None
        or args.r4_artifact is None
    ):
        parser.error("--source-root, --source-commit, --output, and --r4-artifact are required")
    try:
        if args.inner_run:
            if (
                not isinstance(args.kernel_sandbox_attestation, str)
                or os.environ.get("R5_KERNEL_SANDBOX_ATTESTATION") != args.kernel_sandbox_attestation
                or not _kernel_sandbox_attested()
            ):
                raise R5Error("inner capture requires kernel sandbox attestation")
            artifact, path = run(
                production=args.test_production_root or PRODUCTION_ROOT,
                source=args.source_root,
                source_commit=args.source_commit,
                output=args.output,
                r4_artifact=args.r4_artifact,
                test_only=args.test_production_root is not None,
                watchdog_seconds=args.watchdog_seconds,
                owned_clone=args.owned_clone,
                kernel_attested=True,
            )
            passed = artifact["dataset"]["passed"]
        else:
            artifact, path = run_supervised(
                production=args.test_production_root or PRODUCTION_ROOT,
                source=args.source_root,
                source_commit=args.source_commit,
                output=args.output,
                r4_artifact=args.r4_artifact,
                test_only=args.test_production_root is not None,
                watchdog_seconds=args.watchdog_seconds,
            )
            passed = artifact["formal_passed"]
    except (R5Error, OSError, ValueError) as exc:
        print(f"r5 harness failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": artifact["artifact_id"],
                "path": str(path),
                "passed": passed,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 3


_assert_root_matrix = assert_root_matrix

if __name__ == "__main__":
    raise SystemExit(main())
