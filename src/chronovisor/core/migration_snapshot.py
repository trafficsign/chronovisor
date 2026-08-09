"""Isolated, checksummed, time-limited migration restore points."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.hashutil import sha256_file as _sha256

SCHEMA = "chronovisor.migration-restore-point.v1"


def _utc_now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(UTC)
    return value if value.tzinfo else value.replace(tzinfo=UTC)




def _selected_paths(root: Path) -> Iterable[Path]:
    for directory in ("pages", "system", ".index"):
        base = root / directory
        if base.exists():
            yield from sorted(path for path in base.rglob("*") if path.is_file())
    registry = root / "runtime" / "librarian" / "page-registry.json"
    if registry.exists():
        yield registry


def create_restore_point(
    root: Path,
    *,
    reason: str,
    ttl_days: int = 7,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _utc_now(now)
    restore_id = f"{current.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    restore_root = root / "runtime" / "librarian" / "migration-restore-points"
    final_dir = restore_root / restore_id
    restore_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{restore_id}.", dir=restore_root))
    os.chmod(temp_dir, 0o700)
    rows: list[dict[str, Any]] = []
    try:
        payload_dir = temp_dir / "payload"
        for source in _selected_paths(root):
            relative = source.relative_to(root)
            destination = payload_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            os.chmod(destination, 0o600)
            rows.append(
                {
                    "path": str(relative),
                    "size": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )
        manifest = {
            "schema": SCHEMA,
            "restore_id": restore_id,
            "reason": reason,
            "created_at": current.isoformat(timespec="seconds"),
            "expires_at": (current + timedelta(days=max(0, int(ttl_days)))).isoformat(
                timespec="seconds"
            ),
            "file_count": len(rows),
            "total_bytes": sum(int(row["size"]) for row in rows),
            "verification_status": "checksum_verified",
            "restore_drill_at": None,
            "files": rows,
        }
        write_sealed_json(temp_dir / "manifest.json", manifest, backup=False)
        os.replace(temp_dir, final_dir)
        return {**manifest, "path": str(final_dir)}
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def create_incremental_restore_point(
    root: Path,
    *,
    paths: Iterable[Path],
    reason: str,
    ttl_days: int = 7,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Snapshot an explicit bounded mutation set plus authority pointers."""

    selected: dict[str, Path] = {}
    root_resolved = root.resolve()
    for value in paths:
        resolved = value.resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ValueError("incremental restore path must be inside Chronovisor root")
        if resolved.is_file():
            selected[str(resolved.relative_to(root_resolved))] = resolved
    for relative in (
        "runtime/librarian/page-registry.json",
        ".index/semantic/active.json",
        ".index/bm25.sqlite",
    ):
        path = root / relative
        if path.is_file():
            selected[relative] = path
    if not selected:
        raise ValueError("incremental restore point requires at least one file")

    current = _utc_now(now)
    restore_id = (
        f"{current.strftime('%Y%m%dT%H%M%SZ')}-incremental-{uuid.uuid4().hex[:12]}"
    )
    restore_root = root / "runtime" / "librarian" / "migration-restore-points"
    final_dir = restore_root / restore_id
    restore_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{restore_id}.", dir=restore_root))
    os.chmod(temp_dir, 0o700)
    rows: list[dict[str, Any]] = []
    try:
        payload_dir = temp_dir / "payload"
        for relative, source in sorted(selected.items()):
            destination = payload_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            os.chmod(destination, 0o600)
            rows.append(
                {
                    "path": relative,
                    "size": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )
        manifest = {
            "schema": SCHEMA,
            "restore_id": restore_id,
            "kind": "incremental",
            "reason": reason,
            "created_at": current.isoformat(timespec="seconds"),
            "expires_at": (
                current + timedelta(days=max(0, int(ttl_days)))
            ).isoformat(timespec="seconds"),
            "file_count": len(rows),
            "total_bytes": sum(int(row["size"]) for row in rows),
            "verification_status": "checksum_verified",
            "restore_drill_at": None,
            "files": rows,
        }
        write_sealed_json(temp_dir / "manifest.json", manifest, backup=False)
        os.replace(temp_dir, final_dir)
        return {**manifest, "path": str(final_dir)}
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def verify_restore_point(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for row in payload.get("files") or []:
        candidate = path / "payload" / str(row.get("path") or "")
        if not candidate.is_file():
            failures.append(f"missing:{row.get('path')}")
            continue
        try:
            expected_size = int(row["size"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"manifest-size:{row.get('path')}")
            continue
        if candidate.stat().st_size != expected_size:
            failures.append(f"size:{row.get('path')}")
        elif _sha256(candidate) != row.get("sha256"):
            failures.append(f"sha256:{row.get('path')}")
    return {
        "status": "verified" if not failures else "failed",
        "restore_id": payload.get("restore_id"),
        "file_count": len(payload.get("files") or []),
        "failures": failures,
    }


def restore_drill(path: Path, destination: Path) -> dict[str, Any]:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("restore drill destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(path / "payload", destination, dirs_exist_ok=True)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    failures = [
        str(row.get("path"))
        for row in manifest.get("files") or []
        if not (destination / str(row.get("path"))).is_file()
        or _sha256(destination / str(row.get("path"))) != row.get("sha256")
    ]
    status = "verified" if not failures else "failed"
    if status == "verified":
        manifest["verification_status"] = "verified"
        manifest["restore_drill_at"] = _utc_now().isoformat(timespec="seconds")
        write_sealed_json(path / "manifest.json", manifest, backup=False)
    return {
        "status": status,
        "restore_id": manifest.get("restore_id"),
        "destination": str(destination),
        "failures": failures,
    }


def cleanup_expired_restore_points(
    root: Path,
    *,
    now: datetime | None = None,
    force: bool = False,
) -> dict[str, Any]:
    current = _utc_now(now)
    base = root / "runtime" / "librarian" / "migration-restore-points"
    deleted: list[str] = []
    retained: list[str] = []
    if not base.exists():
        return {"deleted": deleted, "retained": retained}
    release_root = root / "runtime" / "librarian"
    if (
        not force
        and (release_root / "phase6-receipt.json").is_file()
        and not (release_root / "phase12-release.json").is_file()
    ):
        return {
            "deleted": deleted,
            "retained": sorted(
                path.name for path in base.iterdir() if path.is_dir()
            ),
            "reason": "migration_release_insurance_active",
        }
    for path in sorted(item for item in base.iterdir() if item.is_dir()):
        try:
            payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            expires = datetime.fromisoformat(str(payload["expires_at"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            retained.append(path.name)
            continue
        if force or expires <= current:
            shutil.rmtree(path)
            deleted.append(path.name)
        else:
            retained.append(path.name)
    return {"deleted": deleted, "retained": retained}
