"""Offline operations for the date-partitioned Raw Archive."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.hashutil import sha256_bytes as _sha256
from chronovisor.core.legacy_archive import (
    migrate_processed_legacy,
    verify_legacy_manifest,
)
from chronovisor.core.raw_segment import (
    CAPTURE_TIMEZONE,
    RawSegmentCorrupt,
    journal_path_for,
    read_commits,
    read_open_range,
    restored_segment_bytes,
    seal_segment,
    verify_manifest,
)
from chronovisor.core.raw_store import RawStore

_PROJECTION_MANIFEST_RE = re.compile(
    r"^semantic-(?P<projection>[0-9a-f]{64})\.manifest\.json$"
)
_PROJECTION_RECEIPT_RE = re.compile(
    r"^semantic-(?P<projection>[0-9a-f]{64})-manifest-.*\.receipt\.json$"
)



def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_relocation_ledger(raw_dir: Path, event: dict[str, Any]) -> dict[str, Any]:
    """Fsync one idempotent relocation proof before source deletion."""

    runtime_dir = raw_dir.parent / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    ledger = runtime_dir / "raw-relocation-ledger.jsonl"
    lock_path = runtime_dir / "raw-relocation-ledger.lock"
    subject = json.dumps(
        event, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    event_id = hashlib.sha256(subject.encode("utf-8")).hexdigest()
    row = {
        "schema": "chronovisor.raw-relocation-ledger.v1",
        "event_id": event_id,
        "recorded_at": datetime.now(CAPTURE_TIMEZONE).isoformat(),
        **event,
    }
    encoded = (
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if ledger.exists():
                for line in ledger.read_text(encoding="utf-8").splitlines():
                    try:
                        existing = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        isinstance(existing, dict)
                        and existing.get("event_id") == event_id
                    ):
                        return existing
            with ledger.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(runtime_dir)
            return row
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _segment_day(path: Path, raw_dir: Path) -> str:
    relative = path.relative_to(raw_dir)
    if len(relative.parts) < 4:
        raise ValueError(f"segment is not date-partitioned: {path}")
    return "/".join(relative.parts[:3])


def archive_status(raw_dir: Path) -> dict[str, Any]:
    raw_dir = raw_dir.expanduser().resolve(strict=False)
    store = RawStore(raw_dir)
    units = tuple(store.iter_units())
    legacy = tuple(unit for unit in units if not unit.is_segment)
    segments = tuple(store.iter_segment_units())
    open_paths = tuple(
        sorted(raw_dir.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.jsonl.open"))
    )
    manifests = tuple(
        path
        for path in sorted(
            raw_dir.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.manifest.json")
        )
        if not path.name.startswith("legacy-part-")
    )
    journals = tuple(
        raw_dir.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.commits.jsonl")
    )
    legacy_manifests = tuple(
        raw_dir.glob(
            "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/legacy-part-*.manifest.json"
        )
    )
    legacy_archives = tuple(
        raw_dir.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/legacy-part-*.tar.zst")
    )
    projection_artifacts = tuple(sorted(raw_dir.glob("semantic-*.json")))
    stored_paths = (
        {unit.path for unit in units if unit.path.exists()}
        | set(manifests)
        | set(journals)
        | set(legacy_manifests)
        | set(legacy_archives)
        | set(projection_artifacts)
    )
    stored_bytes = sum(path.stat().st_size for path in stored_paths)
    physical_files = tuple(
        path for path in raw_dir.rglob("*") if path.is_file() and not path.is_symlink()
    )
    physical_bytes = sum(path.stat().st_size for path in physical_files)
    logical_bytes = sum(unit.length for unit in units)
    open_bytes = sum(path.stat().st_size for path in open_paths)
    oldest_open = min(
        (
            datetime.fromtimestamp(path.stat().st_mtime, tz=CAPTURE_TIMEZONE)
            for path in open_paths
        ),
        default=None,
    )
    source_captures = sum(unit.raw_id.startswith("save-") for unit in units)
    semantic_children = sum(unit.raw_id.startswith("semantic-") for unit in units)
    manual_units = len(units) - source_captures - semantic_children
    state_path = raw_dir.parent / ".orchestrator_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        state = {}
    processed_value = (
        state.get("processed_raw_files") if isinstance(state, dict) else None
    )
    processed = (
        {value for value in processed_value if isinstance(value, str)}
        if isinstance(processed_value, list)
        else set()
    )
    raw_ids = {unit.raw_id for unit in units}
    dead_letter_files = (
        sum(1 for path in (raw_dir / ".dead-letter").rglob("*") if path.is_file())
        if (raw_dir / ".dead-letter").exists()
        else 0
    )
    return {
        "raw_dir": str(raw_dir),
        "layout": store.mode,
        "logical_units": len(units),
        "legacy_units": len(legacy),
        "segment_units": len(segments),
        "open_segments": len(open_paths),
        "sealed_segments": len(manifests),
        "legacy_archives": len(legacy_manifests),
        "projection_artifacts": len(projection_artifacts),
        "logical_bytes": logical_bytes,
        "stored_bytes": stored_bytes,
        "physical_files": len(physical_files),
        "physical_bytes": physical_bytes,
        "compression_ratio": (
            round(stored_bytes / logical_bytes, 6) if logical_bytes else None
        ),
        "unsealed_bytes": open_bytes,
        "oldest_open_at": oldest_open.isoformat() if oldest_open else None,
        "classification": {
            "source_capture": source_captures,
            "semantic_child": semantic_children,
            "manual": manual_units,
            "processed": len(raw_ids & processed),
            "pending_or_deferred": len(raw_ids - processed),
            "dead_letter_files": dead_letter_files,
        },
    }


def _projection_artifact_dirs(raw_dir: Path) -> tuple[Path, ...]:
    candidates = (
        raw_dir,
        raw_dir.parent / "runtime" / "raw-projections" / "artifacts",
    )
    seen: set[Path] = set()
    roots: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    return tuple(roots)


def _root_relative_path(path: Path, root: Path) -> str | None:
    try:
        resolved = path.expanduser().resolve(strict=False)
        relative = resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    return relative.as_posix()


def _read_projection_json(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _projection_parent_raw_id(row: object) -> str | None:
    if not isinstance(row, dict):
        return None
    receipt = row.get("receipt")
    key = receipt.get("idempotency_key") if isinstance(receipt, dict) else None
    if not isinstance(key, str) or not key or Path(key).name != key:
        return None
    return f"save-{key}.md"


def _projection_quarantine_paths(raw_dir: Path, raw_id: str) -> tuple[Path, ...]:
    roots = (
        raw_dir / ".dead-letter",
        raw_dir.parent / "runtime" / "failures" / "quarantined-raw",
    )
    paths = tuple(root / raw_id for root in roots)
    return tuple(path for path in paths if path.is_file() and not path.is_symlink())


def _projection_status_parent(
    row: object,
    *,
    raw_store: RawStore,
    units_by_id: dict[str, Any],
    units_by_digest: dict[str, list[Any]],
    raw_dir: Path,
    processed: set[str],
) -> tuple[dict[str, Any], str]:
    expected_digest = row.get("raw_sha256") if isinstance(row, dict) else None
    expected_digest = expected_digest if isinstance(expected_digest, str) else None
    raw_id = (
        row.get("raw_id")
        if isinstance(row, dict)
        and isinstance(row.get("raw_id"), str)
        and Path(str(row["raw_id"])).name == str(row["raw_id"])
        else _projection_parent_raw_id(row)
    )
    unit = units_by_id.get(raw_id) if raw_id is not None else None
    quarantine_paths = (
        _projection_quarantine_paths(raw_dir, raw_id) if raw_id is not None else ()
    )
    observed_digest: str | None = None
    source_missing = unit is None

    if unit is not None:
        observed_digest = unit.sha256
        if observed_digest is None:
            try:
                observed_digest = _sha256(raw_store.read_bytes(unit))
            except (OSError, RawSegmentCorrupt, ValueError):
                source_missing = True
    elif raw_id is None and expected_digest:
        matches = units_by_digest.get(expected_digest, [])
        if len(matches) == 1:
            unit = matches[0]
            raw_id = unit.raw_id
            try:
                observed_digest = _sha256(raw_store.read_bytes(unit))
                source_missing = False
            except (OSError, RawSegmentCorrupt, ValueError):
                source_missing = True
    if source_missing and quarantine_paths:
        for candidate in quarantine_paths:
            try:
                observed_digest = _sha256(candidate.read_bytes())
                source_missing = False
                break
            except (OSError, ValueError):
                continue

    digest_invalid = (
        expected_digest is None
        or (observed_digest is not None and observed_digest != expected_digest)
    )
    parent = {
        "raw_id": raw_id,
        "raw_sha256": expected_digest,
        "digest": expected_digest,
        "observed_sha256": observed_digest,
        "processed": bool(raw_id and raw_id in processed),
        "quarantine": sorted(
            path
            for path in (
                _root_relative_path(candidate, raw_dir.parent)
                for candidate in quarantine_paths
            )
            if path is not None
        ),
    }
    if digest_invalid:
        return parent, "invalid"
    return parent, "missing" if source_missing else "valid"


def projection_status(raw_dir: Path, *, full: bool = False) -> dict[str, Any]:
    """Inventory semantic projection manifests without changing runtime state."""

    raw_dir = raw_dir.expanduser().resolve(strict=False)
    root = raw_dir.parent
    artifact_dirs = _projection_artifact_dirs(raw_dir)
    from chronovisor.ingest.raw_semantic_projection import (
        projection_bundle_state_for_parent,
        verify_projection_bundle,
    )

    raw_store = RawStore(raw_dir)
    try:
        units = tuple(raw_store.iter_units())
    except (OSError, RawSegmentCorrupt, ValueError):
        units = ()
    units_by_id = {unit.raw_id: unit for unit in units}
    units_by_digest: dict[str, list[Any]] = {}
    for unit in units:
        digest = unit.sha256
        if digest is None:
            try:
                digest = _sha256(raw_store.read_bytes(unit))
            except (OSError, RawSegmentCorrupt, ValueError):
                digest = None
        if isinstance(digest, str):
            units_by_digest.setdefault(digest, []).append(unit)

    state_path = root / ".orchestrator_state.json"
    state = _read_projection_json(state_path) or {}
    processed_value = state.get("processed_raw_files")
    processed = (
        {value for value in processed_value if isinstance(value, str)}
        if isinstance(processed_value, list)
        else set()
    )

    rows: list[dict[str, Any]] = []
    for artifact_dir in artifact_dirs:
        manifest_paths = sorted(artifact_dir.glob("semantic-*.manifest.json"))
        receipts_by_projection: dict[str, list[Path]] = {}
        for receipt_path in artifact_dir.glob("semantic-*-manifest-*.receipt.json"):
            receipt_match = _PROJECTION_RECEIPT_RE.fullmatch(receipt_path.name)
            if receipt_match is not None:
                receipts_by_projection.setdefault(
                    receipt_match.group("projection"), []
                ).append(receipt_path)
        for manifest_path in manifest_paths:
            payload = _read_projection_json(manifest_path)
            match = _PROJECTION_MANIFEST_RE.fullmatch(manifest_path.name)
            projection_id = match.group("projection") if match else None
            parent_rows = (
                payload.get("source", {}).get("parents")
                if isinstance(payload, dict)
                and isinstance(payload.get("source"), dict)
                else None
            )
            parent_rows = parent_rows if isinstance(parent_rows, list) else []
            parents: list[dict[str, Any]] = []
            parent_states: list[str] = []
            for parent_row in parent_rows:
                parent, parent_state = _projection_status_parent(
                    parent_row,
                    raw_store=raw_store,
                    units_by_id=units_by_id,
                    units_by_digest=units_by_digest,
                    raw_dir=raw_dir,
                    processed=processed,
                )
                parents.append(parent)
                parent_states.append(parent_state)

            children: list[dict[str, Any]] = []
            child_names: list[str] = []
            if isinstance(payload, dict) and isinstance(payload.get("children"), list):
                for child_row in payload["children"]:
                    if not isinstance(child_row, dict):
                        continue
                    filename = child_row.get("filename")
                    if not isinstance(filename, str) or Path(filename).name != filename:
                        continue
                    child_names.append(filename)
                    child_path = artifact_dir / filename
                    children.append(
                        {
                            "raw_id": filename,
                            "path": _root_relative_path(child_path, root),
                            "processed": filename in processed,
                        }
                    )

            manifest_digest: str | None = None
            try:
                manifest_digest = _sha256(manifest_path.read_bytes())
            except OSError:
                pass
            receipt_paths: set[str] = set()
            if (
                projection_id is not None
                and isinstance(payload, dict)
                and payload.get("status") == "delegated"
            ):
                if manifest_digest is not None:
                    receipt_paths.add(
                        _root_relative_path(
                            artifact_dir
                            / f"semantic-{projection_id}-manifest-{manifest_digest}.receipt.json",
                            root,
                        )
                        or ""
                    )
                receipt_paths.update(
                    _root_relative_path(path, root) or ""
                    for path in receipts_by_projection.get(projection_id, [])
                )
            if isinstance(payload, dict):
                noop = payload.get("noop_receipt_filename")
                if isinstance(noop, str) and Path(noop).name == noop:
                    receipt_paths.add(
                        _root_relative_path(artifact_dir / noop, root) or ""
                    )
            receipt_paths.discard("")

            expected_receipts = [artifact_dir / child_name for child_name in child_names]
            if (
                projection_id is not None
                and manifest_digest is not None
                and isinstance(payload, dict)
                and payload.get("status") == "delegated"
            ):
                expected_receipts.append(
                    artifact_dir
                    / f"semantic-{projection_id}-manifest-{manifest_digest}.receipt.json"
                )
            if isinstance(payload, dict):
                noop = payload.get("noop_receipt_filename")
                if isinstance(noop, str) and Path(noop).name == noop:
                    expected_receipts.append(artifact_dir / noop)
            missing_artifacts = [
                path
                for path in expected_receipts
                if not path.is_file() or path.is_symlink()
            ]

            verify_ok = False
            if full and payload is not None and match is not None:
                try:
                    verify_projection_bundle(manifest_path)
                    verify_ok = True
                except Exception:
                    verify_ok = False

            helper_states: list[str] = []
            if not verify_ok and parents:
                for parent in parents:
                    raw_id = parent.get("raw_id")
                    if not isinstance(raw_id, str):
                        continue
                    parent_unit = units_by_id.get(raw_id)
                    if parent_unit is None or parent_unit.storage != "legacy_file":
                        continue
                    try:
                        helper_states.append(
                            projection_bundle_state_for_parent(
                                parent_unit.path,
                                projection_dir=artifact_dir,
                            )
                        )
                    except (OSError, TypeError, ValueError):
                        helper_states.append("invalid")

            if payload is None or match is None or not parent_rows or any(
                state == "invalid" for state in parent_states
            ) or any(state == "invalid" for state in helper_states):
                disposition = "invalid"
            elif any(state == "missing" for state in parent_states) or missing_artifacts or any(
                state in {"incomplete", "absent"} for state in helper_states
            ):
                disposition = "missing"
            elif verify_ok or not full:
                disposition = "valid"
            else:
                disposition = "invalid"

            rows.append(
                {
                    "projection_id": projection_id,
                    "parent": parents[0] if len(parents) == 1 else None,
                    "parents": parents,
                    "manifest": _root_relative_path(manifest_path, root),
                    "children": children,
                    "receipts": sorted(receipt_paths),
                    "quarantine": sorted(
                        {
                            path
                            for parent in parents
                            for path in parent.get("quarantine", [])
                        }
                    ),
                    "processed": bool(parents) and all(
                        parent.get("processed") is True for parent in parents
                    ),
                    "state": disposition,
                }
            )

    counts = {
        "total": len(rows),
        "valid": sum(row["state"] == "valid" for row in rows),
        "missing": sum(row["state"] == "missing" for row in rows),
        "invalid": sum(row["state"] == "invalid" for row in rows),
    }
    return {
        "schema": "chronovisor.raw-projection-status.v1",
        "status": "ok",
        "full": full,
        **counts,
        "counts": counts,
        "projections": rows,
    }


def verify_archive(raw_dir: Path, *, full: bool = False) -> dict[str, Any]:
    raw_dir = raw_dir.expanduser().resolve(strict=False)
    verified_open = 0
    verified_sealed = 0
    verified_units = 0
    verified_legacy_archives = 0
    errors: list[dict[str, str]] = []
    open_pattern = "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.jsonl.open"
    for data_path in sorted(raw_dir.glob(open_pattern)):
        try:
            commits = read_commits(journal_path_for(data_path))
            expected_end = 0
            for commit in commits:
                value = read_open_range(data_path, commit.offset, commit.length)
                if _sha256(value) != commit.sha256:
                    raise RawSegmentCorrupt(
                        f"open range digest mismatch: {commit.raw_id}"
                    )
                expected_end += commit.length
                verified_units += 1
            if data_path.stat().st_size != expected_end:
                raise RawSegmentCorrupt("open segment has an uncommitted tail")
            verified_open += 1
        except Exception as exc:
            errors.append(
                {"path": str(data_path), "error": f"{type(exc).__name__}: {exc}"}
            )
    manifest_pattern = "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.manifest.json"
    for manifest_path in sorted(raw_dir.glob(manifest_pattern)):
        if manifest_path.name.startswith("legacy-part-"):
            continue
        try:
            payload = verify_manifest(manifest_path, full=full)
            commits = payload.get("commits")
            verified_units += len(commits) if isinstance(commits, list) else 0
            verified_sealed += 1
        except Exception as exc:
            errors.append(
                {
                    "path": str(manifest_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    legacy_pattern = (
        "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/legacy-part-*.manifest.json"
    )
    for manifest_path in sorted(raw_dir.glob(legacy_pattern)):
        try:
            payload = verify_legacy_manifest(manifest_path, full=full)
            verified_units += len(payload["members"])
            verified_legacy_archives += 1
        except Exception as exc:
            errors.append(
                {
                    "path": str(manifest_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "status": "ok" if not errors else "error",
        "raw_dir": str(raw_dir),
        "full": full,
        "open_segments": verified_open,
        "sealed_segments": verified_sealed,
        "legacy_archives": verified_legacy_archives,
        "logical_units": verified_units,
        "errors": errors,
    }


def seal_eligible(
    raw_dir: Path,
    *,
    before: str | None = None,
    dry_run: bool = True,
    compression_level: int = 9,
    max_segments: int = 0,
) -> dict[str, Any]:
    from chronovisor.core.store import okf_runtime_operation

    with okf_runtime_operation(raw_dir.parent):
        return _seal_eligible_locked(
            raw_dir,
            before=before,
            dry_run=dry_run,
            compression_level=compression_level,
            max_segments=max_segments,
        )


def _seal_eligible_locked(
    raw_dir: Path,
    *,
    before: str | None,
    dry_run: bool,
    compression_level: int,
    max_segments: int,
) -> dict[str, Any]:
    raw_dir = raw_dir.expanduser().resolve(strict=False)
    cutoff = before or datetime.now(CAPTURE_TIMEZONE).strftime("%Y/%m/%d")
    try:
        datetime.strptime(cutoff, "%Y/%m/%d")
    except ValueError as exc:
        raise ValueError("before must use YYYY/MM/DD") from exc
    today = datetime.now(CAPTURE_TIMEZONE).strftime("%Y/%m/%d")
    if cutoff > today:
        raise ValueError("before cannot include the current capture day")
    pattern = "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.jsonl.open"
    eligible = [
        path
        for path in sorted(raw_dir.glob(pattern))
        if _segment_day(path, raw_dir) < cutoff
    ]
    total_eligible = len(eligible)
    if max_segments > 0:
        eligible = eligible[:max_segments]
    results: list[dict[str, Any]] = []
    for path in eligible:
        if dry_run:
            results.append(
                {
                    "path": str(path),
                    "logical_bytes": path.stat().st_size,
                    "action": "would_seal",
                }
            )
            continue
        manifest = seal_segment(
            path,
            compression_level=compression_level,
            remove_open=False,
        )
        manifest_path = path.with_name(
            path.name.removesuffix(".jsonl.open") + ".manifest.json"
        )
        append_relocation_ledger(
            raw_dir,
            {
                "kind": "segment_seal",
                "source": str(path.relative_to(raw_dir)),
                "manifest": str(manifest_path.relative_to(raw_dir)),
                "logical_sha256": manifest["logical_sha256"],
                "compressed_sha256": manifest["compressed_sha256"],
            },
        )
        path.unlink(missing_ok=True)
        journal_path_for(path).unlink(missing_ok=True)
        _fsync_directory(path.parent)
        results.append(
            {
                "path": str(path),
                "action": "sealed",
                "logical_bytes": manifest["logical_bytes"],
                "compressed_bytes": manifest["compressed_bytes"],
            }
        )
    return {
        "status": "dry_run" if dry_run else "ok",
        "before": cutoff,
        "eligible": len(eligible),
        "eligible_total": total_eligible,
        "results": results,
    }


def export_raw(raw_dir: Path, raw_id: str, output: Path) -> dict[str, Any]:
    store = RawStore(raw_dir)
    unit = store.resolve(raw_id)
    if unit is None:
        raise FileNotFoundError(raw_id)
    value = store.read_bytes(unit)
    output = output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "ok",
        "raw_id": raw_id,
        "output": str(output),
        "bytes": len(value),
        "sha256": _sha256(value),
    }


def restore_segment(manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = verify_manifest(manifest_path.expanduser(), full=True)
    segment = manifest_path.expanduser().with_name(str(manifest["segment"]))
    value = restored_segment_bytes(segment)
    output = output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_bytes(value)
    if _sha256(value) != manifest["logical_sha256"]:
        output.unlink(missing_ok=True)
        raise RawSegmentCorrupt("restored output failed manifest digest")
    return {
        "status": "ok",
        "manifest": str(manifest_path),
        "output": str(output),
        "bytes": len(value),
        "sha256": _sha256(value),
    }


def _completed_projection_artifact_ids(
    raw_dir: Path,
    *,
    processed_raw_ids: set[str],
) -> set[str]:
    """Return verified flat projection JSON that can leave the active store.

    A bundle is eligible only when its manifest verifies in full, every child
    Raw is durably processed, and every source parent has its processed ACK.
    Any incomplete, orphaned, held, or partially processed bundle stays flat.
    """

    from chronovisor.ingest.raw_semantic_projection import verify_projection_bundle

    eligible: set[str] = set()
    for manifest_path in sorted(raw_dir.glob("semantic-*.manifest.json")):
        match = _PROJECTION_MANIFEST_RE.fullmatch(manifest_path.name)
        if match is None:
            continue
        projection_id = match.group("projection")
        try:
            manifest = verify_projection_bundle(manifest_path)
        except (OSError, TypeError, ValueError):
            continue
        children = manifest.get("children")
        if not isinstance(children, list):
            continue
        child_names = {
            str(row["filename"])
            for row in children
            if isinstance(row, dict) and isinstance(row.get("filename"), str)
        }
        if len(child_names) != len(children) or not child_names.issubset(
            processed_raw_ids
        ):
            continue
        source = manifest.get("source")
        parents = source.get("parents") if isinstance(source, dict) else None
        if not isinstance(parents, list) or not parents:
            continue
        parent_names: set[str] = set()
        valid_parents = True
        for row in parents:
            receipt = row.get("receipt") if isinstance(row, dict) else None
            idempotency_key = (
                receipt.get("idempotency_key") if isinstance(receipt, dict) else None
            )
            if not isinstance(idempotency_key, str) or not idempotency_key:
                valid_parents = False
                break
            parent_names.add(f"save-{idempotency_key}.md")
        if not valid_parents or not parent_names.issubset(processed_raw_ids):
            continue
        prefix = f"semantic-{projection_id}"
        group = {
            path.name
            for path in raw_dir.glob(f"{prefix}*.json")
            if path.is_file() and not path.is_symlink()
        }
        declared_json = {manifest_path.name}
        for field in ("bundle_receipt_filename", "noop_receipt_filename"):
            filename = manifest.get(field)
            if isinstance(filename, str):
                declared_json.add(filename)
        if not declared_json.issubset(group):
            continue
        eligible.update(group)
    return eligible


def migrate_legacy(
    raw_dir: Path,
    *,
    before: str | None = None,
    dry_run: bool = True,
    remove_source: bool = False,
    max_archive_bytes: int = 128 * 1024 * 1024,
    compression_level: int = 9,
) -> dict[str, Any]:
    from chronovisor.core.store import okf_runtime_operation

    with okf_runtime_operation(raw_dir.parent):
        return _migrate_legacy_locked(
            raw_dir,
            before=before,
            dry_run=dry_run,
            remove_source=remove_source,
            max_archive_bytes=max_archive_bytes,
            compression_level=compression_level,
        )


def _migrate_legacy_locked(
    raw_dir: Path,
    *,
    before: str | None,
    dry_run: bool,
    remove_source: bool,
    max_archive_bytes: int,
    compression_level: int,
) -> dict[str, Any]:
    state_path = (
        raw_dir.expanduser().resolve(strict=False).parent / ".orchestrator_state.json"
    )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        state = {}
    processed = state.get("processed_raw_files") if isinstance(state, dict) else None
    processed_ids = (
        {value for value in processed if isinstance(value, str)}
        if isinstance(processed, list)
        else set()
    )
    processed_ids.update(
        _completed_projection_artifact_ids(
            raw_dir.expanduser().resolve(strict=False),
            processed_raw_ids=processed_ids,
        )
    )

    def record_relocation(manifest_path: Path, manifest: dict[str, Any]) -> None:
        append_relocation_ledger(
            raw_dir.expanduser().resolve(strict=False),
            {
                "kind": "legacy_archive",
                "manifest": str(
                    manifest_path.relative_to(
                        raw_dir.expanduser().resolve(strict=False)
                    )
                ),
                "archive": str(manifest["archive"]),
                "logical_bytes": manifest["logical_bytes"],
                "compressed_sha256": manifest["compressed_sha256"],
                "member_ids_sha256": hashlib.sha256(
                    "\n".join(
                        sorted(str(row["raw_id"]) for row in manifest["members"])
                    ).encode("utf-8")
                ).hexdigest(),
            },
        )

    return migrate_processed_legacy(
        raw_dir,
        processed_raw_ids=processed_ids,
        before=before,
        dry_run=dry_run,
        remove_source=remove_source,
        max_archive_bytes=max_archive_bytes,
        compression_level=compression_level,
        before_source_delete=record_relocation,
    )
