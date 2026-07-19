"""One-shot, reversible migration from ``~/.wiki`` to ``~/.chronovisor``."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chronovisor.store import DEFAULT_ROOT, LEGACY_ROOT

MANIFEST_SCHEMA = "chronovisor.brand-migration.v1"
MANIFEST_RELATIVE_PATH = Path("runtime/brand-migration.json")
INTERNAL_PATH_RENAMES = (
    (Path("runtime/wiki-mutation.lock"), Path("runtime/chronovisor-mutation.lock")),
)


class BrandMigrationError(RuntimeError):
    """The data-root rename cannot be completed without ambiguity."""


def _same_target(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=True) == right.resolve(strict=True)
    except OSError:
        return False


def _inventory(root: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    bytes_by_area: dict[str, int] = {}
    digest = hashlib.sha256()
    for area in ("raw", "pages", "system", "recall", "research", "claims"):
        base = root / area
        count = 0
        size = 0
        if base.exists():
            for path in sorted(base.rglob("*")):
                if not path.is_file():
                    continue
                stat = path.stat()
                relative = path.relative_to(root).as_posix()
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(b"\n")
                count += 1
                size += stat.st_size
        counts[area] = count
        bytes_by_area[area] = size
    stat = root.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "counts": counts,
        "bytes": bytes_by_area,
        "inventory_sha256": digest.hexdigest(),
    }


def inspect(
    *, legacy_root: Path = LEGACY_ROOT, canonical_root: Path = DEFAULT_ROOT
) -> dict[str, Any]:
    legacy_present = legacy_root.exists() or legacy_root.is_symlink()
    canonical_present = canonical_root.exists()
    if canonical_present and legacy_present and _same_target(legacy_root, canonical_root):
        state = "applied"
    elif legacy_present and not canonical_present and not legacy_root.is_symlink():
        state = "ready"
    elif not legacy_present and canonical_present:
        state = "missing_compatibility_link"
    elif not legacy_present and not canonical_present:
        state = "empty"
    else:
        state = "split_brain"
    return {
        "schema": MANIFEST_SCHEMA,
        "state": state,
        "legacy_root": str(legacy_root),
        "canonical_root": str(canonical_root),
        "legacy_is_symlink": legacy_root.is_symlink(),
    }


def _assert_quiescent(root: Path) -> list[str]:
    """Fail when any persistent writer lock is currently held."""

    acquired: list[tuple[int, Path]] = []
    try:
        runtime = root / "runtime"
        for path in sorted(runtime.rglob("*.lock")) if runtime.exists() else []:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(descriptor)
                raise BrandMigrationError(
                    f"writer lock is held; stop services before migration: {path}"
                ) from exc
            acquired.append((descriptor, path))
        return [path.relative_to(root).as_posix() for _, path in acquired]
    finally:
        for descriptor, _path in reversed(acquired):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def preflight(
    *, legacy_root: Path = LEGACY_ROOT, canonical_root: Path = DEFAULT_ROOT
) -> dict[str, Any]:
    """Validate root state, filesystem locality, inventory, and quiescence."""

    status = inspect(legacy_root=legacy_root, canonical_root=canonical_root)
    if status["state"] == "applied":
        return verify(legacy_root=legacy_root, canonical_root=canonical_root)
    if status["state"] != "ready":
        raise BrandMigrationError(f"data-root migration is not ready: {status['state']}")
    if legacy_root.stat().st_dev != canonical_root.parent.stat().st_dev:
        raise BrandMigrationError("data-root migration must remain on one filesystem")
    for old_relative, new_relative in INTERNAL_PATH_RENAMES:
        if (legacy_root / old_relative).exists() and (
            legacy_root / new_relative
        ).exists():
            raise BrandMigrationError(
                f"internal path split-brain: {old_relative} and {new_relative}"
            )
    checked_locks = _assert_quiescent(legacy_root)
    return {
        **status,
        "status": "ready",
        "quiescent": True,
        "checked_locks": checked_locks,
        "inventory": _inventory(legacy_root),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rewrite_root_references(root: Path, old: Path, new: Path) -> list[str]:
    changed: list[str] = []
    for relative in (Path("config.toml"), Path(".serena/project.yml")):
        path = root / relative
        if not path.exists() or not path.is_file():
            continue
        original = path.read_text(encoding="utf-8")
        updated = original.replace(str(old), str(new))
        if updated == original:
            continue
        path.write_text(updated, encoding="utf-8")
        changed.append(relative.as_posix())
    return changed


def _rename_internal_paths(root: Path, *, reverse: bool = False) -> list[str]:
    changed: list[str] = []
    pairs = (
        ((new, old) if reverse else (old, new))
        for old, new in INTERNAL_PATH_RENAMES
    )
    for source_relative, target_relative in pairs:
        source = root / source_relative
        target = root / target_relative
        if target.exists() and source.exists():
            raise BrandMigrationError(
                f"internal path split-brain: {source_relative} and {target_relative}"
            )
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        changed.append(f"{source_relative.as_posix()}->{target_relative.as_posix()}")
    return changed


def apply(
    *, legacy_root: Path = LEGACY_ROOT, canonical_root: Path = DEFAULT_ROOT
) -> dict[str, Any]:
    status = inspect(legacy_root=legacy_root, canonical_root=canonical_root)
    if status["state"] == "applied":
        return verify(legacy_root=legacy_root, canonical_root=canonical_root)
    preflight_status = preflight(
        legacy_root=legacy_root, canonical_root=canonical_root
    )

    before = _inventory(legacy_root)
    os.replace(legacy_root, canonical_root)
    try:
        legacy_root.symlink_to(canonical_root.name, target_is_directory=True)
    except Exception:
        os.replace(canonical_root, legacy_root)
        raise
    internal_renames: list[str] = []
    rewritten: list[str] = []
    try:
        internal_renames = _rename_internal_paths(canonical_root)
        rewritten = _rewrite_root_references(
            canonical_root, legacy_root, canonical_root
        )
        after = _inventory(canonical_root)
        if before != after:
            raise BrandMigrationError(
                "post-rename inventory differs from source inventory"
            )
    except Exception:
        if rewritten:
            _rewrite_root_references(canonical_root, canonical_root, legacy_root)
        if internal_renames:
            _rename_internal_paths(canonical_root, reverse=True)
        legacy_root.unlink(missing_ok=True)
        os.replace(canonical_root, legacy_root)
        raise

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "applied",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "legacy_root": str(legacy_root),
        "canonical_root": str(canonical_root),
        "compatibility_link": os.readlink(legacy_root),
        "preflight": preflight_status,
        "internal_renames": internal_renames,
        "rewritten_files": rewritten,
        "inventory": after,
    }
    _atomic_json(canonical_root / MANIFEST_RELATIVE_PATH, manifest)
    return manifest


def verify(
    *, legacy_root: Path = LEGACY_ROOT, canonical_root: Path = DEFAULT_ROOT
) -> dict[str, Any]:
    status = inspect(legacy_root=legacy_root, canonical_root=canonical_root)
    if status["state"] != "applied":
        raise BrandMigrationError(f"data-root migration is not applied: {status['state']}")
    manifest_path = canonical_root / MANIFEST_RELATIVE_PATH
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    expected = manifest.get("inventory")
    observed = _inventory(canonical_root)
    # The migration manifest itself is outside the inventoried semantic areas.
    if expected and expected != observed:
        raise BrandMigrationError("canonical root inventory no longer matches migration")
    for old_relative, new_relative in INTERNAL_PATH_RENAMES:
        if (canonical_root / old_relative).exists():
            raise BrandMigrationError(f"legacy internal path remains: {old_relative}")
        renamed = manifest.get("internal_renames", [])
        if renamed and (canonical_root / new_relative).exists() is False:
            raise BrandMigrationError(f"renamed internal path is missing: {new_relative}")
    return {
        **status,
        "status": "verified",
        "inventory": observed,
        "manifest": str(manifest_path),
    }


def rollback(
    *, legacy_root: Path = LEGACY_ROOT, canonical_root: Path = DEFAULT_ROOT
) -> dict[str, Any]:
    status = inspect(legacy_root=legacy_root, canonical_root=canonical_root)
    if status["state"] != "applied":
        raise BrandMigrationError(f"cannot roll back state: {status['state']}")
    _rewrite_root_references(canonical_root, canonical_root, legacy_root)
    _rename_internal_paths(canonical_root, reverse=True)
    legacy_root.unlink()
    os.replace(canonical_root, legacy_root)
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "rolled_back",
        "legacy_root": str(legacy_root),
        "inventory": _inventory(legacy_root),
    }
