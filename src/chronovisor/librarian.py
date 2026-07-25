"""Low-priority, local-only Librarian shadow migration worker."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from chronovisor import frontmatter
from chronovisor.classification import (
    classification_source_sha256,
    classification_authority_status,
    load_udc_package,
    propose_from_legacy_metadata,
)
from chronovisor.durable_state import write_sealed_json
from chronovisor.librarian_status import (
    STATE_SCHEMA,
    build_librarian_status,
    load_librarian_state,
)
from chronovisor.page_registry import PageRegistry
from chronovisor.store import CHRONOVISOR_ROOT
from chronovisor.uid_link_index import build_uid_link_index

EVENT_SCHEMA = "chronovisor.librarian-event.v1"
WORKER_VERSION = "1"
BASELINE_SCHEMA = "chronovisor.librarian-baseline.v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _scope_generation(root: Path, pages: Mapping[str, Any]) -> str:
    rows = []
    for row in pages.values():
        if not isinstance(row, Mapping) or row.get("status") == "superseded":
            continue
        relative = str(row.get("path") or "")
        stat = (root / relative).stat()
        rows.append(
            (
                relative,
                stat.st_size,
                stat.st_mtime_ns,
                str(row.get("status") or ""),
            )
        )
    rows.sort()
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _append_event(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as handle:
                os.chmod(path, 0o600)
                handle.write(
                    json.dumps(
                        {
                            "schema": EVENT_SCHEMA,
                            "timestamp": _now_iso(),
                            **dict(row),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _page_metadata(root: Path, row: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = root / str(row.get("path") or "")
    text = path.read_text(encoding="utf-8")
    meta, _body = frontmatter.parse(text)
    return meta, text


def capture_baseline(
    *,
    root: Path = CHRONOVISOR_ROOT,
    repo_root: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Capture the migration boundary without mutating Wiki content."""

    registry = PageRegistry(root)
    manifest = registry.ensure_manifest(write=False)
    pages = manifest["registry"]["pages"]
    tag_counts: Counter[str] = Counter()
    sensitivity_counts: Counter[str] = Counter()
    for row in pages.values():
        if not isinstance(row, Mapping):
            continue
        path = root / str(row.get("path") or "")
        if not path.exists():
            continue
        try:
            meta, _text = _page_metadata(root, row)
        except (OSError, UnicodeError):
            continue
        tags = meta.get("tags")
        if isinstance(tags, list):
            tag_counts.update(str(value) for value in tags)
        sensitivity_counts[str(meta.get("sensitivity") or "normal")] += 1
    raw_count = 0
    try:
        from chronovisor.raw_store import RawStore

        raw_count = sum(1 for _unit in RawStore(root / "raw").iter_units())
    except Exception:
        raw_count = sum(1 for _path in (root / "raw").rglob("*.md"))
    aliases_path = root / "runtime" / "page-aliases.json"
    alias_count = 0
    if aliases_path.exists():
        try:
            aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
            alias_rows = aliases.get("aliases") if isinstance(aliases, dict) else {}
            alias_count = len(alias_rows) if isinstance(alias_rows, dict) else 0
        except json.JSONDecodeError:
            alias_count = -1
    commit = ""
    if repo_root is not None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            commit = result.stdout.strip() if result.returncode == 0 else ""
        except OSError:
            commit = ""
    fixture_root = root / "classification" / "fixtures"
    payload = {
        "schema": BASELINE_SCHEMA,
        "captured_at": _now_iso(),
        "repo_commit": commit,
        "root": str(root),
        "pages": len(pages),
        "raw_logical_units": raw_count,
        "legacy_aliases": alias_count,
        "top_tags": dict(tag_counts.most_common(50)),
        "sensitivity": dict(sorted(sensitivity_counts.items())),
        "index_schema": 10,
        "fixtures": {
            "dev_200": (fixture_root / "classification-dev-200.jsonl").exists(),
            "holdout_100": (fixture_root / "classification-holdout-100.jsonl").exists(),
            "locked": (fixture_root / "manifest.json").exists(),
        },
        "wiki_mutated": False,
    }
    if write:
        write_sealed_json(
            root / "runtime" / "librarian" / "baseline.json",
            payload,
            backup=True,
        )
    return payload


def run_shadow(
    *,
    root: Path = CHRONOVISOR_ROOT,
    limit: int = 250,
    full_sweep: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate deterministic classification proposals without page mutation."""

    started_monotonic = time.monotonic()
    registry = PageRegistry(root)
    manifest = registry.ensure_manifest(write=not dry_run)
    registry_state = manifest["registry"]
    pages = {
        uid: row
        for uid, row in registry_state["pages"].items()
        if isinstance(row, Mapping)
        and row.get("status") != "superseded"
        and (root / str(row.get("path") or "")).exists()
    }
    scope_generation = _scope_generation(root, pages)
    package = load_udc_package(root)
    authority = classification_authority_status(root, package=package)
    candidates = []
    for uid, row in sorted(pages.items()):
        path = root / str(row.get("path") or "")
        source_sha256 = classification_source_sha256(
            path.read_text(encoding="utf-8")
        )
        classification = row.get("classification")
        evidence_refs = (
            classification.get("evidence_refs")
            if isinstance(classification, Mapping)
            else None
        )
        current_ref = (
            str(evidence_refs[0])
            if isinstance(evidence_refs, list) and evidence_refs
            else ""
        )
        if (
            not isinstance(classification, Mapping)
            or current_ref != f"page-sha256:{source_sha256}"
        ):
            candidates.append((uid, row))
    selected = candidates if full_sweep or limit < 0 else candidates[: max(0, limit)]
    updates: dict[str, dict[str, Any]] = {}
    notation_counts: Counter[str] = Counter()
    disposition_counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    for uid, row in selected:
        try:
            meta, _text = _page_metadata(root, row)
            tags = meta.get("tags")
            tags = tags if isinstance(tags, list) else []
            record = propose_from_legacy_metadata(
                tags=[str(value) for value in tags],
                page_type=str(meta.get("type") or "knowledge"),
                lifecycle=str(meta.get("status") or "active"),
                sensitivity=str(
                    meta.get("sensitivity") or row.get("sensitivity") or "normal"
                ),
                # Full-file hashes change when UID/classification metadata is
                # backfilled. Bind evidence to classification-relevant bytes.
                evidence_ref=f"page-sha256:{classification_source_sha256(_text)}",
                package=package,
            )
            updates[uid] = {
                "classification": record.to_dict(),
                "classification_status": record.status,
            }
            notation_counts[record.primary.notation] += 1
            disposition_counts[record.status] += 1
        except Exception as exc:
            failures.append({"uid": uid, "error": f"{type(exc).__name__}: {exc}"})

    apply_result = (
        {
            "status": "dry_run",
            "generation": int(registry_state.get("generation") or 0),
            "updated": len(updates),
        }
        if dry_run
        else registry.apply_page_updates(
            updates,
            expected_generation=int(registry_state.get("generation") or 0),
            event="librarian_shadow_batch",
        )
    )
    current_registry = registry_state if dry_run else registry.load()
    current_pages = [
        row
        for row in current_registry["pages"].values()
        if isinstance(row, Mapping)
        and row.get("status") != "superseded"
        and (root / str(row.get("path") or "")).exists()
    ]
    proposed = sum(
        isinstance(row.get("classification"), Mapping) for row in current_pages
    )
    held = sum(row.get("classification_status") == "held" for row in current_pages)
    total = len(current_pages)
    remaining = max(0, total - proposed)
    previous = load_librarian_state(root)
    link_index: dict[str, Any] | None = None
    if full_sweep or not previous.get("progress", {}).get("links"):
        link_index = build_uid_link_index(root, registry=registry, write=not dry_run)
    else:
        link_path = root / "runtime" / "librarian" / "uid-link-index.json"
        if link_path.exists():
            try:
                link_index = json.loads(link_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                link_index = None
    links = link_index or {}
    scanned_scope = (
        scope_generation
        if remaining == 0 and not failures
        else previous.get("last_swept_scope_generation")
    )
    progress = {
        "uid": {
            "numerator": total,
            "denominator": total,
            "scope_generation": scope_generation,
        },
        "classification_shadow": {
            "numerator": proposed,
            "denominator": total,
            "scope_generation": scope_generation,
        },
        "classification_terminal": {
            "numerator": held,
            "denominator": total,
            "scope_generation": scope_generation,
            "note": "proposals are non-terminal until authority activation",
        },
        "links": {
            "numerator": int(links.get("edge_count") or 0),
            "denominator": int(links.get("edge_count") or 0)
            + int(links.get("unresolved_count") or 0),
            "scope_generation": scope_generation,
            "unresolved": int(links.get("unresolved_count") or 0),
        },
        "migration_batch": {
            "numerator": proposed,
            "denominator": total,
            "scope_generation": scope_generation,
        },
        "full_sweep": {
            "numerator": int(scanned_scope == scope_generation),
            "denominator": 1,
            "scope_generation": scope_generation,
            "current": scanned_scope == scope_generation,
        },
    }
    oldest_age = 0
    if remaining and previous.get("last_run"):
        try:
            last = datetime.fromisoformat(str(previous["last_run"]))
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            oldest_age = max(
                0, int((datetime.now(timezone.utc) - last).total_seconds())
            )
        except ValueError:
            oldest_age = 0
    state = {
        "schema": STATE_SCHEMA,
        "enabled": True,
        "mode": "shadow",
        "generation": int(previous.get("generation") or 0) + (0 if dry_run else 1),
        "scope_generation": scope_generation,
        "last_swept_scope_generation": scanned_scope,
        "initial_organization_complete_at": None,
        "authority": authority,
        "progress": progress,
        "queue": {
            "queued": remaining,
            "actionable": remaining,
            "running": 0,
            "held": held,
            "quarantined": len(failures),
            "completed": proposed - held,
            "oldest_age_seconds": oldest_age,
        },
        "debts": {
            "unclassified": remaining,
            "explicit_hold": held,
            "unresolved_link": int(links.get("unresolved_count") or 0),
            "claim_or_fingerprint_failure": 0,
            "worker_failure": len(failures),
            "reasons": failures[:20],
        },
        "quality": {
            "classification_authority_active": bool(authority["active"]),
            "forced_misclassification_gate": "not_evaluated",
            "locked_holdout": "missing",
            "recall_regression": "not_evaluated",
            "broken_redirects": 0,
            "sensitivity_downgrades": 0,
        },
        "resources": {
            "priority": "P3",
            "model_calls": 0,
            "frontier_calls": 0,
            "p0_preemption_contract": "no_model_shadow_worker",
        },
        "eta": None,
        "blocked_reasons": [],
        "last_run": _now_iso(),
        "last_result": {
            "classified": disposition_counts["proposed"],
            "held": disposition_counts["held"],
            "failures": len(failures),
            "notations": dict(sorted(notation_counts.items())),
            "duration_seconds": round(time.monotonic() - started_monotonic, 3),
        },
    }
    if not dry_run:
        state_path = root / "runtime" / "librarian" / "state.json"
        write_sealed_json(state_path, state, backup=True)
        _append_event(
            root / "runtime" / "librarian" / "events.jsonl",
            {
                "event": "shadow_run",
                "status": "ok" if not failures else "partial",
                "created": int(manifest.get("created") or 0),
                "classified": disposition_counts["proposed"],
                "held": disposition_counts["held"],
                "failed": len(failures),
                "scope_generation": scope_generation,
                "registry_generation": apply_result.get("generation"),
                "remaining": remaining,
            },
        )
    return {
        "status": "dry_run" if dry_run else ("ok" if not failures else "partial"),
        "scope_generation": scope_generation,
        "registry": apply_result,
        "observed": total,
        "selected": len(selected),
        "classified": disposition_counts["proposed"],
        "held": disposition_counts["held"],
        "remaining": remaining,
        "failures": failures,
        "state": state,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chronovisor Librarian worker")
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--full-sweep", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--capture-baseline", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = (
        capture_baseline(
            root=args.root,
            repo_root=args.repo_root,
            write=not args.dry_run,
        )
        if args.capture_baseline
        else build_librarian_status(args.root)
        if args.status
        else run_shadow(
            root=args.root,
            limit=args.limit,
            full_sweep=args.full_sweep,
            dry_run=args.dry_run,
        )
    )
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if args.json
        else "\n".join(f"{key}: {value}" for key, value in payload.items())
    )
    return 0 if payload.get("status") not in {"error", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
