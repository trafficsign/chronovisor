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
