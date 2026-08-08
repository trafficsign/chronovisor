"""Full-corpus classification shadowing and CAS metadata migration."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from chronovisor.classification.classification import (
    classification_authority_status,
    classification_frontmatter,
    load_udc_package,
    record_from_dict,
    render_call_number,
    validate_record,
)
from chronovisor.classification.classification_engine import (
    _page_payload,
    record_from_consensus,
    run_consensus_batches,
)
from chronovisor.classification.classification_resolver import (
    production_candidate_index,
)
from chronovisor.core import frontmatter
from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.hashutil import sha256_bytes as _sha256_bytes
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.ingest.page_mutation import chronovisor_mutation_lock
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.librarian.librarian import _append_event, _now_iso
from chronovisor.ops.migration_snapshot import (
    create_incremental_restore_point,
    create_restore_point,
    restore_drill,
)

CLASSIFICATION_INDEX_SCHEMA = "chronovisor.classification-index.v1"
MIGRATION_RECEIPT_SCHEMA = "chronovisor.classification-migration-receipt.v1"


def _active_pages(root: Path, state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(uid): dict(row)
        for uid, row in state.get("pages", {}).items()
        if isinstance(row, Mapping)
        and row.get("status") != "superseded"
        and (root / str(row.get("path") or "")).is_file()
    }


def build_classification_index(
    root: Path = CHRONOVISOR_ROOT,
    *,
    registry_state: Mapping[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    registry_state = registry_state or PageRegistry(root).load()
    package = load_udc_package(root)
    by_notation: dict[str, list[str]] = defaultdict(list)
    by_status: dict[str, list[str]] = defaultdict(list)
    assignments: dict[str, dict[str, Any]] = {}
    for uid, row in _active_pages(root, registry_state).items():
        classification = row.get("classification")
        if not isinstance(classification, Mapping):
            continue
        record = record_from_dict(classification)
        validate_record(record, package=package)
        by_notation[record.primary.notation].append(uid)
        status = str(row.get("classification_status") or record.status)
        by_status[status].append(uid)
        assignments[uid] = {
            "primary_uri": record.primary.concept_uri,
            "primary_notation": record.primary.notation,
            "secondary_notations": [value.notation for value in record.secondary],
            "status": status,
            "confidence": record.confidence,
            "authority_epoch": record.classifier_authority_epoch,
            "authority_digest": record.classifier_authority_digest,
            "page_path": str(row.get("path") or ""),
        }
    payload = {
        "schema": CLASSIFICATION_INDEX_SCHEMA,
        "generated_at": _now_iso(),
        "registry_generation": int(registry_state.get("generation") or 0),
        "package_release": package.release,
        "package_checksum": package.checksum,
        "assignment_count": len(assignments),
        "by_notation": {
            key: sorted(value) for key, value in sorted(by_notation.items())
        },
        "by_status": {key: sorted(value) for key, value in sorted(by_status.items())},
        "assignments": dict(sorted(assignments.items())),
    }
    if write:
        write_sealed_json(
            root / "runtime" / "librarian" / "classification-index.json",
            payload,
            backup=True,
        )
    return payload


def run_full_model_shadow(
    root: Path = CHRONOVISOR_ROOT,
    *,
    limit: int = -1,
    batch_size: int = 20,
) -> dict[str, Any]:
    """Classify every stale page with local 2-of-3 consensus."""

    started = time.monotonic()
    authority = classification_authority_status(root)
    if authority.get("bundle_resolver_status") != "legacy" and not authority.get(
        "mutation_capability"
    ):
        raise RuntimeError(
            "classification bundle is decision-only; parent Phase 5 mutation CAS "
            "has not enabled Page Registry writes"
        )
    registry = PageRegistry(root)
    state = registry.ensure_manifest(write=True)["registry"]
    package = load_udc_package(root)
    if not package.complete:
        raise RuntimeError("full model shadow requires the complete UDC package")
    index = production_candidate_index(
        root,
        package,
        provider_factory=_library_evidence_provider_factory,
    )
    authority_epoch = int(authority.get("authority_epoch") or 1)
    authority_digest = (
        str(authority.get("threshold_version") or "")
        if authority.get("bundle_resolver_status") != "legacy"
        else ""
    )
    rows: list[dict[str, Any]] = []
    for uid, row in sorted(_active_pages(root, state).items()):
        page = _page_payload(root, uid, row)
        classification = row.get("classification")
        evidence = (
            classification.get("evidence_refs")
            if isinstance(classification, Mapping)
            else None
        )
        current_ref = (
            str(evidence[0]) if isinstance(evidence, list) and evidence else ""
        )
        if (
            isinstance(classification, Mapping)
            and classification.get("subject_checksum") == package.checksum
            and current_ref == f"page-sha256:{page['source_sha256']}"
            and int(classification.get("classifier_authority_epoch") or 0)
            == authority_epoch
            and (
                not authority_digest
                or classification.get("classifier_authority_digest") == authority_digest
            )
        ):
            continue
        page["candidates"] = index.candidates(page)
        rows.append(page)
    if limit >= 0:
        rows = rows[:limit]
    decisions = run_consensus_batches(
        rows,
        root=root,
        batch_size=batch_size,
        purpose="explicit",
        timeout_seconds=1_800,
    )
    by_uid = {str(row["uid"]): row for row in decisions}
    updates: dict[str, dict[str, Any]] = {}
    holds = 0
    for page in rows:
        decision = by_uid[str(page["uid"])]
        status = "proposed" if int(decision.get("quorum") or 0) >= 2 else "held"
        record = record_from_consensus(
            page,
            decision,
            package=package,
            authority_epoch=authority_epoch,
            status=status,
            authority_digest=authority_digest or None,
        )
        updates[str(page["uid"])] = {
            "classification": record.to_dict(),
            "classification_status": status,
            "classification_consensus": {
                "sha256": decision["consensus_sha256"],
                "quorum": decision["quorum"],
                "models": [
                    decision["primary_model"],
                    decision["challenger_model"],
                    *(
                        [decision["tie_break_model"]]
                        if decision.get("tie_break_model")
                        else []
                    ),
                ],
            },
        }
        holds += status == "held"
    current = registry.load()
    result = registry.apply_page_updates(
        updates,
        expected_generation=int(current.get("generation") or 0),
        event="librarian_local_consensus_shadow",
    )
    latest = registry.load()
    index_payload = build_classification_index(root, registry_state=latest, write=True)
    remaining = 0
    for uid, row in _active_pages(root, latest).items():
        page = _page_payload(root, uid, row)
        classification = row.get("classification")
        evidence = (
            classification.get("evidence_refs")
            if isinstance(classification, Mapping)
            else None
        )
        current_ref = (
            str(evidence[0]) if isinstance(evidence, list) and evidence else ""
        )
        if (
            not isinstance(classification, Mapping)
            or classification.get("subject_checksum") != package.checksum
            or current_ref != f"page-sha256:{page['source_sha256']}"
            or int(classification.get("classifier_authority_epoch") or 0)
            != authority_epoch
            or (
                authority_digest
                and classification.get("classifier_authority_digest")
                != authority_digest
            )
        ):
            remaining += 1
    receipt = {
        "schema": MIGRATION_RECEIPT_SCHEMA,
        "event": "full_model_shadow",
        "timestamp": _now_iso(),
        "selected": len(rows),
        "proposed": len(rows) - holds,
        "held": holds,
        "remaining": remaining,
        "status": "ok" if remaining == 0 else "catching_up",
        "registry_generation": result["generation"],
        "classification_index_assignments": index_payload["assignment_count"],
        "duration_seconds": round(time.monotonic() - started, 3),
        "model_calls_are_local_only": True,
    }
    _append_event(
        root / "runtime" / "librarian" / "events.jsonl",
        receipt,
    )
    write_sealed_json(
        root / "runtime" / "librarian" / "phase5-receipt.json",
        receipt,
        backup=True,
    )
    return receipt


def _verified_restore(
    root: Path,
    *,
    paths: Sequence[Path] | None,
    reason: str,
) -> dict[str, Any]:
    restore = (
        create_restore_point(root, reason=reason, ttl_days=7)
        if paths is None
        else create_incremental_restore_point(
            root,
            paths=paths,
            reason=reason,
            ttl_days=7,
        )
    )
    with tempfile.TemporaryDirectory(prefix="chronovisor-restore-drill-") as temp:
        drill = restore_drill(Path(restore["path"]), Path(temp) / "restored")
    if drill["status"] != "verified":
        raise RuntimeError(f"restore drill failed: {drill['failures']}")
    return {**restore, "verification_status": "verified"}


def _classification_updates(
    uid: str,
    row: Mapping[str, Any],
    *,
    package: Any,
    authority_epoch: int,
    authority_digest: str | None,
) -> tuple[dict[str, Any], Any]:
    classification = row.get("classification")
    if not isinstance(classification, Mapping):
        raise TypeError(f"{uid} has no classification proposal")
    record = record_from_dict(classification)
    target_status = (
        "held" if str(row.get("classification_status") or "") == "held" else "adopted"
    )
    adopted = replace(
        record,
        classifier_authority_epoch=authority_epoch,
        classifier_authority_digest=authority_digest,
        status=target_status,
    )
    validate_record(adopted, package=package, require_complete_package=True)
    updates = {
        "uid": uid,
        **classification_frontmatter(adopted),
        "call_number": render_call_number(adopted),
    }
    return updates, adopted


def migrate_active_metadata(
    root: Path = CHRONOVISOR_ROOT,
    *,
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict[str, Any]:
    """CAS-backfill UID and adopted classification into every Active page."""

    authority = classification_authority_status(root)
    if not authority["active"]:
        raise RuntimeError(
            f"classification authority is inactive: {authority['reason']}"
        )
    if authority.get("bundle_resolver_status") != "legacy" and not authority.get(
        "mutation_capability"
    ):
        raise RuntimeError(
            "classification bundle is decision-only; parent Phase 5 mutation CAS "
            "has not enabled Page writes"
        )
    authority_epoch = int(authority.get("authority_epoch") or 1)
    authority_digest = (
        str(authority.get("threshold_version") or "")
        if authority.get("bundle_resolver_status") != "legacy"
        else ""
    )
    package = load_udc_package(root)
    registry = PageRegistry(root)
    state = registry.ensure_manifest(write=True)["registry"]
    pages = _active_pages(root, state)
    pending = []
    for uid, row in sorted(pages.items()):
        path = root / str(row["path"])
        text = path.read_text(encoding="utf-8")
        meta, _body = frontmatter.parse(text)
        current_uid = str(meta.get("uid") or "")
        current_checksum = ""
        current_epoch = 0
        current_authority_digest = ""
        raw_classification = meta.get("classification_json")
        if isinstance(raw_classification, str):
            try:
                current_payload = json.loads(raw_classification)
                current_checksum = str(current_payload.get("subject_checksum") or "")
                current_epoch = int(
                    current_payload.get("classifier_authority_epoch") or 0
                )
                current_authority_digest = str(
                    current_payload.get("classifier_authority_digest") or ""
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                current_checksum = ""
        if (
            current_uid == uid
            and current_checksum == package.checksum
            and current_epoch == authority_epoch
            and (not authority_digest or current_authority_digest == authority_digest)
            and meta.get("classification_status") in {"adopted", "held"}
        ):
            continue
        pending.append((uid, row))
    if dry_run:
        return {
            "status": "dry_run",
            "observed": len(pages),
            "pending": len(pending),
        }

    phase_restore = _verified_restore(
        root,
        paths=None,
        reason="phase6-active-uid-classification-backfill",
    )
    batches = []
    migrated = 0
    for batch_number, offset in enumerate(
        range(0, len(pending), max(1, batch_size)), start=1
    ):
        batch = pending[offset : offset + max(1, batch_size)]
        paths = [root / str(row["path"]) for _uid, row in batch]
        incremental = _verified_restore(
            root,
            paths=paths,
            reason=f"phase6-classification-batch-{batch_number}",
        )
        preimages: dict[Path, bytes] = {}
        owned: dict[Path, bytes] = {}
        adopted_records: dict[str, Any] = {}
        try:
            with chronovisor_mutation_lock():
                for uid, row in batch:
                    path = root / str(row["path"])
                    original = path.read_bytes()
                    if _sha256_bytes(original) != str(row.get("content_sha256") or ""):
                        raise RuntimeError(f"CAS preimage changed for {row['path']}")
                    updates, adopted = _classification_updates(
                        uid,
                        row,
                        package=package,
                        authority_epoch=authority_epoch,
                        authority_digest=authority_digest or None,
                    )
                    text = original.decode("utf-8")
                    _meta, body = frontmatter.parse(text)
                    updated = frontmatter.patch(text, updates)
                    _updated_meta, updated_body = frontmatter.parse(updated)
                    if updated_body != body:
                        raise RuntimeError(
                            f"frontmatter patch changed body for {row['path']}"
                        )
                    preimages[path] = original
                    updated_bytes = updated.encode("utf-8")
                    atomic_write(path, updated)
                    if path.read_bytes() != updated_bytes:
                        raise RuntimeError(f"readback mismatch for {row['path']}")
                    owned[path] = updated_bytes
                    adopted_records[uid] = adopted.to_dict()
        except Exception:
            with chronovisor_mutation_lock():
                for path, original in preimages.items():
                    if path.read_bytes() == owned.get(path):
                        atomic_write(path, original.decode("utf-8"))
            raise
        manifest_result = registry.ensure_manifest(write=True)
        refreshed = manifest_result["registry"]
        updates = {
            uid: {
                "classification": adopted_records[uid],
                "classification_status": str(adopted_records[uid]["status"]),
            }
            for uid, _row in batch
        }
        registry.apply_page_updates(
            updates,
            expected_generation=int(refreshed.get("generation") or 0),
            event="classification_metadata_adopted",
        )
        migrated += len(batch)
        batch_receipt = {
            "batch": batch_number,
            "migrated": len(batch),
            "restore_id": incremental["restore_id"],
            "restore_verified": True,
        }
        batches.append(batch_receipt)
        _append_event(
            root / "runtime" / "librarian" / "events.jsonl",
            {
                "event": "classification_migration_batch",
                "status": "ok",
                **batch_receipt,
            },
        )

    final_registry = registry.ensure_manifest(write=True)["registry"]
    classification_index = build_classification_index(
        root,
        registry_state=final_registry,
        write=True,
    )
    from chronovisor.search.search import get_bm25

    get_bm25().build(force=True)
    try:
        from chronovisor.search.semantic_jobs import enqueue_rebuild

        semantic_rebuild_job_id = enqueue_rebuild()
    except Exception as exc:
        semantic_rebuild_job_id = None
        semantic_rebuild_error = f"{type(exc).__name__}: {exc}"
    else:
        semantic_rebuild_error = None
    receipt = {
        "schema": MIGRATION_RECEIPT_SCHEMA,
        "event": "phase6_metadata_migration",
        "timestamp": _now_iso(),
        "status": "ok",
        "observed": len(pages),
        "migrated": migrated,
        "already_current": len(pages) - len(pending),
        "phase_restore_id": phase_restore["restore_id"],
        "phase_restore_verified": True,
        "batches": batches,
        "classification_index_assignments": classification_index["assignment_count"],
        "semantic_rebuild_job_id": semantic_rebuild_job_id,
        "semantic_rebuild_error": semantic_rebuild_error,
    }
    write_sealed_json(
        root / "runtime" / "librarian" / "phase6-receipt.json",
        receipt,
        backup=True,
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("shadow", "index", "migrate", "status"),
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "shadow":
        result = run_full_model_shadow(
            args.root,
            limit=args.limit,
            batch_size=args.batch_size,
        )
    elif args.command == "index":
        result = build_classification_index(args.root)
    elif args.command == "migrate":
        result = migrate_active_metadata(
            args.root,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    else:
        result = classification_authority_status(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _library_evidence_provider_factory(
    *,
    package: Any,
    provider_manifest: Path,
) -> Any:
    from chronovisor.lab.classification_library_evidence import (
        LibraryEvidenceIndex,
        LibraryEvidenceProvider,
    )

    return LibraryEvidenceProvider(
        package=package,
        evidence_index=LibraryEvidenceIndex(provider_manifest),
    )


if __name__ == "__main__":
    raise SystemExit(main())
