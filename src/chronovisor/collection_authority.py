"""Collection-first classification authority and review-only Librarian.

The physical ``pages/<folder>/`` layout is bootstrap provenance.  Stable
collection UUIDs and logical page assignments live in a sealed registry so
rename, merge, split, and page moves do not require destructive file moves.
UDC/CVO anchors are an audited collection-level interoperability crosswalk,
never a per-page prediction.
"""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from chronovisor.classification_anchor import load_anchor_set
from chronovisor.durable_state import (
    DurableStateError,
    canonical_sha256,
    file_lock,
    read_sealed_json,
    write_sealed_json,
)
from chronovisor.page_identity import new_page_uid, normalize_page_uid
from chronovisor.page_registry import PageRegistry
from chronovisor.research_scheduler import (
    research_lane,
    run_cancellable_command,
)
from chronovisor.store import CHRONOVISOR_ROOT
from chronovisor.uid_link_index import build_uid_link_index

COLLECTION_REGISTRY_SCHEMA = "chronovisor.collection-registry.v1"
COLLECTION_RECEIPT_SCHEMA = "chronovisor.collection-lifecycle-receipt.v1"
COLLECTION_QUEUE_SCHEMA = "chronovisor.collection-review-queue.v1"
COLLECTION_EVALUATION_SCHEMA = "chronovisor.collection-authority-evaluation.v1"
COLLECTION_QUALITY_SCHEMA = "chronovisor.collection-quality.v1"
DEFAULT_CHALLENGER_MODEL = "gpt-oss:20b"


class CollectionAuthorityError(RuntimeError):
    """Collection authority state or an attempted lifecycle mutation is unsafe."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _data_path(filename: str) -> Path:
    return Path(__file__).parent / "data" / filename


def default_contract_path() -> Path:
    return _data_path("collection-authority-contract-v1.json")


def default_crosswalk_path() -> Path:
    return _data_path("collection-crosswalk-v1.json")


def default_preregistration_path() -> Path:
    return _data_path("collection-authority-unseen-prereg-v1.json")


def default_gold_path() -> Path:
    return _data_path("collection-authority-unseen-gold-v1.json")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionAuthorityError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CollectionAuthorityError(f"expected JSON object: {path}")
    return value


def _content_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    unsigned = {str(key): value for key, value in payload.items() if key != "checksum"}
    return "sha256:" + canonical_sha256(unsigned)


def load_contract(path: Path | None = None) -> dict[str, Any]:
    source = path or default_contract_path()
    payload = _read_object(source)
    if (
        payload.get("schema") != "chronovisor.collection-authority-contract.v1"
        or payload.get("status") != "adopted"
        or payload.get("decision") != "existing_collection_is_primary_authority"
    ):
        raise CollectionAuthorityError("collection authority contract is not adopted")
    gates = payload.get("quality_gates")
    reviewer = payload.get("anomaly_reviewer")
    if not isinstance(gates, dict) or not isinstance(reviewer, dict):
        raise CollectionAuthorityError("collection authority gates are missing")
    if reviewer.get("assignment_mutation_capability") is not False:
        raise CollectionAuthorityError("anomaly reviewer must be review-only")
    return {**payload, "content_sha256": _content_sha256(source)}


def load_crosswalk(path: Path | None = None) -> dict[str, Any]:
    source = path or default_crosswalk_path()
    payload = _read_object(source)
    if (
        payload.get("schema") != "chronovisor.collection-crosswalk.v1"
        or payload.get("status") != "frozen"
        or payload.get("mapping_audit") != "all-entries-reviewed"
        or payload.get("checksum") != _payload_checksum(payload)
    ):
        raise CollectionAuthorityError("collection crosswalk contract is invalid")
    anchors = load_anchor_set().by_id
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CollectionAuthorityError("collection crosswalk is empty")
    slugs: set[str] = set()
    for row in entries:
        if not isinstance(row, dict):
            raise CollectionAuthorityError("crosswalk entry is malformed")
        slug = str(row.get("slug") or "")
        mappings = row.get("mappings")
        if not slug or slug in slugs or not isinstance(mappings, list) or not mappings:
            raise CollectionAuthorityError("crosswalk identity is invalid")
        slugs.add(slug)
        exact = 0
        for mapping in mappings:
            if (
                not isinstance(mapping, dict)
                or mapping.get("anchor_id") not in anchors
                or mapping.get("relation") not in {"exact", "broad"}
            ):
                raise CollectionAuthorityError(f"invalid crosswalk mapping for {slug}")
            exact += int(mapping["relation"] == "exact")
        if exact != 1:
            raise CollectionAuthorityError(
                f"crosswalk {slug} must have exactly one exact anchor"
            )
    return {
        **payload,
        "content_sha256": _content_sha256(source),
        "by_slug": {str(row["slug"]): dict(row) for row in entries},
    }


def _physical_collection_slug(path: object) -> str | None:
    parts = str(path or "").split("/")
    if len(parts) >= 3 and parts[0] == "pages" and parts[1]:
        return parts[1]
    return None


class CollectionRegistry:
    """Sealed stable collection identities and logical page assignments."""

    def __init__(
        self,
        root: Path,
        *,
        uid_factory: Callable[[], str] = new_page_uid,
    ) -> None:
        self.root = Path(root)
        self.runtime_dir = self.root / "runtime" / "librarian"
        self.path = self.runtime_dir / "collection-registry.json"
        self.lock_path = self.runtime_dir / "collection-registry.lock"
        self.receipt_dir = self.runtime_dir / "collection-receipts"
        self.uid_factory = uid_factory

    @staticmethod
    def empty() -> dict[str, Any]:
        return {
            "schema": COLLECTION_REGISTRY_SCHEMA,
            "generation": 0,
            "updated_at": None,
            "contract_epoch": "collection-authority-v1",
            "collections": {},
            "slug_index": {},
            "assignments": {},
        }

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self.empty()
        try:
            payload = read_sealed_json(self.path, recover_backup=True)
        except DurableStateError as exc:
            raise CollectionAuthorityError(
                f"collection registry is unreadable: {exc}"
            ) from exc
        if payload.get("schema") != COLLECTION_REGISTRY_SCHEMA:
            raise CollectionAuthorityError("collection registry schema mismatch")
        for field in ("collections", "slug_index", "assignments"):
            if not isinstance(payload.get(field), dict):
                raise CollectionAuthorityError(
                    f"collection registry {field} must be an object"
                )
        for uid, row in payload["collections"].items():
            try:
                normalize_page_uid(uid)
            except ValueError as exc:
                raise CollectionAuthorityError(
                    f"invalid collection UID {uid}"
                ) from exc
            if not isinstance(row, dict) or row.get("uid") != uid:
                raise CollectionAuthorityError(f"malformed collection {uid}")
        return payload

    def _new_collection(
        self,
        *,
        slug: str,
        label: str,
        source_path: str | None,
        unclassified: bool = False,
        created_by: str = "folder-bootstrap",
    ) -> dict[str, Any]:
        uid = normalize_page_uid(self.uid_factory())
        return {
            "uid": uid,
            "slug": slug,
            "label": label,
            "status": "active",
            "canonical_uid": None,
            "aliases": [],
            "source_paths": [source_path] if source_path else [],
            "is_unclassified": unclassified,
            "created_by": created_by,
            "created_at": _now(),
            "updated_at": _now(),
        }

    def _write_receipt(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        transaction_id = str(payload.get("transaction_id") or self.uid_factory())
        receipt = {
            "schema": COLLECTION_RECEIPT_SCHEMA,
            "transaction_id": transaction_id,
            "recorded_at": _now(),
            **dict(payload),
        }
        write_sealed_json(
            self.receipt_dir / f"{transaction_id}.json",
            receipt,
            backup=False,
        )
        return receipt

    def sync_from_pages(
        self,
        *,
        expected_generation: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Bootstrap and reconcile logical assignments from original folders."""

        page_registry = PageRegistry(self.root)
        manifest = page_registry.ensure_manifest(write=not dry_run)
        page_state = manifest["registry"]
        page_generation = int(page_state.get("generation") or 0)
        active_pages = {
            str(uid): row
            for uid, row in page_state["pages"].items()
            if isinstance(row, Mapping)
            and row.get("status") == "active"
            and str(row.get("path") or "").startswith("pages/")
            and (self.root / str(row.get("path") or "")).is_file()
        }
        lock = nullcontext() if dry_run else file_lock(self.lock_path)
        with lock:
            before = self.load()
            generation = int(before.get("generation") or 0)
            if expected_generation is not None and generation != expected_generation:
                raise CollectionAuthorityError(
                    f"collection generation changed: {generation} != "
                    f"{expected_generation}"
                )
            state = {
                **before,
                "collections": {
                    str(uid): dict(row)
                    for uid, row in before["collections"].items()
                },
                "slug_index": dict(before["slug_index"]),
                "assignments": {
                    str(uid): dict(row)
                    for uid, row in before["assignments"].items()
                    if uid in active_pages
                },
            }
            created_collections: list[str] = []
            unclassified_uid = next(
                (
                    uid
                    for uid, row in state["collections"].items()
                    if isinstance(row, Mapping) and row.get("is_unclassified")
                ),
                None,
            )
            if unclassified_uid is None:
                row = self._new_collection(
                    slug="_unclassified",
                    label="Unclassified",
                    source_path=None,
                    unclassified=True,
                    created_by="fail-closed-bootstrap",
                )
                unclassified_uid = str(row["uid"])
                state["collections"][unclassified_uid] = row
                state["slug_index"]["_unclassified"] = unclassified_uid
                created_collections.append(unclassified_uid)

            discovered_slugs = sorted(
                {
                    slug
                    for row in active_pages.values()
                    if (slug := _physical_collection_slug(row.get("path")))
                }
            )
            for slug in discovered_slugs:
                uid = state["slug_index"].get(slug)
                if uid and uid in state["collections"]:
                    row = dict(state["collections"][uid])
                    path = f"pages/{slug}"
                    source_paths = list(row.get("source_paths") or [])
                    if path not in source_paths:
                        source_paths.append(path)
                        row["source_paths"] = sorted(source_paths)
                        row["updated_at"] = _now()
                        state["collections"][uid] = row
                    continue
                row = self._new_collection(
                    slug=slug,
                    label=slug.replace("-", " "),
                    source_path=f"pages/{slug}",
                )
                uid = str(row["uid"])
                state["collections"][uid] = row
                state["slug_index"][slug] = uid
                created_collections.append(uid)

            changed_assignments = 0
            assignments: dict[str, dict[str, Any]] = {}
            for page_uid, page in sorted(active_pages.items()):
                previous = state["assignments"].get(page_uid)
                preserve_logical = (
                    isinstance(previous, Mapping)
                    and previous.get("source") in {"manual", "merge", "split"}
                    and previous.get("collection_uid") in state["collections"]
                    and state["collections"][str(previous["collection_uid"])].get(
                        "status"
                    )
                    == "active"
                )
                physical_slug = _physical_collection_slug(page.get("path"))
                derived_uid = state["slug_index"].get(
                    physical_slug or "_unclassified",
                    unclassified_uid,
                )
                collection_uid = (
                    str(previous["collection_uid"])
                    if preserve_logical
                    else str(derived_uid)
                )
                collection = state["collections"][collection_uid]
                status = (
                    "unclassified"
                    if collection.get("is_unclassified")
                    else "assigned"
                )
                assignment = {
                    "page_uid": page_uid,
                    "collection_uid": collection_uid,
                    "status": status,
                    "source": (
                        str(previous.get("source"))
                        if preserve_logical
                        else "original-order-folder"
                    ),
                    "source_path": str(page.get("path") or ""),
                    "updated_at": (
                        previous.get("updated_at")
                        if isinstance(previous, Mapping)
                        and all(
                            previous.get(key) == value
                            for key, value in {
                                "page_uid": page_uid,
                                "collection_uid": collection_uid,
                                "status": status,
                                "source": (
                                    str(previous.get("source"))
                                    if preserve_logical
                                    else "original-order-folder"
                                ),
                                "source_path": str(page.get("path") or ""),
                            }.items()
                        )
                        else _now()
                    ),
                }
                assignments[page_uid] = assignment
                changed_assignments += int(previous != assignment)
            state["assignments"] = assignments
            core_before = {
                key: before.get(key)
                for key in ("collections", "slug_index", "assignments")
            }
            core_after = {
                key: state.get(key)
                for key in ("collections", "slug_index", "assignments")
            }
            changed = core_before != core_after
            if changed:
                state["generation"] = generation + 1
            state["updated_at"] = _now()
            if not dry_run and (changed or not self.path.exists()):
                write_sealed_json(self.path, state, backup=True)

        mirror_updates = {}
        crosswalk = load_crosswalk()
        for page_uid, assignment in state["assignments"].items():
            collection = state["collections"][assignment["collection_uid"]]
            crosswalk_row = crosswalk["by_slug"].get(collection["slug"])
            review_required = bool(
                collection.get("is_unclassified")
                or (crosswalk_row or {}).get("review_required")
            )
            mirror_updates[page_uid] = {
                "collection_uid": assignment["collection_uid"],
                "collection_status": (
                    "review_required"
                    if review_required
                    else assignment["status"]
                ),
                "collection_generation": int(state["generation"]),
            }
        mirror = {
            "status": "dry_run",
            "updated": len(mirror_updates),
            "generation": page_generation,
        }
        if not dry_run:
            mirror = page_registry.apply_page_updates(
                mirror_updates,
                expected_generation=page_generation,
                event="collection_assignment_mirror",
            )
            receipt = self._write_receipt(
                {
                    "operation": "sync",
                    "status": "ok",
                    "generation_before": generation,
                    "generation_after": int(state["generation"]),
                    "page_registry_generation": mirror["generation"],
                    "collections_created": created_collections,
                    "assignments_observed": len(assignments),
                    "assignments_changed": changed_assignments,
                    "page_mutations": 0,
                }
            )
        else:
            receipt = {}
        return {
            "status": "dry_run" if dry_run else "ok",
            "generation": int(state["generation"]),
            "collection_count": sum(
                isinstance(row, Mapping) and row.get("status") == "active"
                for row in state["collections"].values()
            ),
            "assignment_count": len(state["assignments"]),
            "created_collections": created_collections,
            "changed_assignments": changed_assignments,
            "page_registry_mirror": mirror,
            "receipt": receipt,
            "registry": state,
        }

    def apply_lifecycle(
        self,
        operation: str,
        *,
        expected_generation: int,
        collection_uid: str | None = None,
        target_collection_uid: str | None = None,
        page_uids: Sequence[str] = (),
        new_label: str | None = None,
        new_slug: str | None = None,
    ) -> dict[str, Any]:
        """Apply rename, merge, split, or logical page move under one CAS."""

        if operation not in {"rename", "merge", "split", "move"}:
            raise CollectionAuthorityError(f"unsupported lifecycle operation: {operation}")
        source_uid = (
            normalize_page_uid(collection_uid) if collection_uid is not None else None
        )
        target_uid = (
            normalize_page_uid(target_collection_uid)
            if target_collection_uid is not None
            else None
        )
        normalized_pages = [normalize_page_uid(uid) for uid in page_uids]
        with file_lock(self.lock_path):
            state = self.load()
            generation = int(state.get("generation") or 0)
            if generation != expected_generation:
                raise CollectionAuthorityError(
                    f"collection generation changed: {generation} != "
                    f"{expected_generation}"
                )
            collections = {
                str(uid): dict(row)
                for uid, row in state["collections"].items()
            }
            assignments = {
                str(uid): dict(row)
                for uid, row in state["assignments"].items()
            }
            if source_uid and source_uid not in collections:
                raise CollectionAuthorityError("source collection is missing")
            if target_uid and target_uid not in collections:
                raise CollectionAuthorityError("target collection is missing")
            before_digest = canonical_sha256(
                {"collections": collections, "assignments": assignments}
            )
            affected_pages: list[str] = []
            created_uid: str | None = None

            if operation == "rename":
                if source_uid is None or not str(new_label or "").strip():
                    raise CollectionAuthorityError("rename requires collection and label")
                collections[source_uid]["label"] = str(new_label).strip()
                collections[source_uid]["updated_at"] = _now()
            elif operation == "merge":
                if source_uid is None or target_uid is None or source_uid == target_uid:
                    raise CollectionAuthorityError("merge requires distinct endpoints")
                for page_uid, assignment in assignments.items():
                    if assignment.get("collection_uid") == source_uid:
                        assignment["collection_uid"] = target_uid
                        assignment["source"] = "merge"
                        assignment["updated_at"] = _now()
                        affected_pages.append(page_uid)
                collections[source_uid]["status"] = "merged"
                collections[source_uid]["canonical_uid"] = target_uid
                collections[source_uid]["updated_at"] = _now()
            elif operation == "move":
                if target_uid is None or not normalized_pages:
                    raise CollectionAuthorityError("move requires target and pages")
                for page_uid in normalized_pages:
                    if page_uid not in assignments:
                        raise CollectionAuthorityError(
                            f"page assignment is missing: {page_uid}"
                        )
                    assignments[page_uid]["collection_uid"] = target_uid
                    assignments[page_uid]["status"] = "assigned"
                    assignments[page_uid]["source"] = "manual"
                    assignments[page_uid]["updated_at"] = _now()
                    affected_pages.append(page_uid)
            else:
                if (
                    source_uid is None
                    or not normalized_pages
                    or not str(new_slug or "").strip()
                    or str(new_slug) in state["slug_index"]
                ):
                    raise CollectionAuthorityError(
                        "split requires source, pages, and unused slug"
                    )
                row = self._new_collection(
                    slug=str(new_slug).strip(),
                    label=str(new_label or new_slug).strip(),
                    source_path=None,
                    created_by="split",
                )
                created_uid = str(row["uid"])
                collections[created_uid] = row
                state["slug_index"][str(new_slug).strip()] = created_uid
                for page_uid in normalized_pages:
                    assignment = assignments.get(page_uid)
                    if not isinstance(assignment, dict) or assignment.get(
                        "collection_uid"
                    ) != source_uid:
                        raise CollectionAuthorityError(
                            f"split page is outside source collection: {page_uid}"
                        )
                    assignment["collection_uid"] = created_uid
                    assignment["source"] = "split"
                    assignment["updated_at"] = _now()
                    affected_pages.append(page_uid)

            state["collections"] = collections
            state["assignments"] = assignments
            state["generation"] = generation + 1
            state["updated_at"] = _now()
            write_sealed_json(self.path, state, backup=True)
            receipt = self._write_receipt(
                {
                    "operation": operation,
                    "status": "committed",
                    "generation_before": generation,
                    "generation_after": state["generation"],
                    "source_collection_uid": source_uid,
                    "target_collection_uid": target_uid,
                    "created_collection_uid": created_uid,
                    "affected_page_uids": sorted(affected_pages),
                    "before_digest": "sha256:" + before_digest,
                    "after_digest": "sha256:"
                    + canonical_sha256(
                        {
                            "collections": collections,
                            "assignments": assignments,
                        }
                    ),
                    "page_mutations": 0,
                }
            )
        return receipt


def _load_page_index(root: Path) -> dict[str, Any]:
    path = root / ".index" / "pages.json"
    if not path.is_file():
        return {"entries": {}}
    payload = _read_object(path)
    return payload if isinstance(payload.get("entries"), dict) else {"entries": {}}


def _page_id_maps(
    state: Mapping[str, Any],
    page_registry: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    collection_by_uid = {
        str(uid): str(row.get("slug") or "")
        for uid, row in (state.get("collections") or {}).items()
        if isinstance(row, Mapping)
    }
    slug_by_page_id: dict[str, str] = {}
    uid_by_page_id: dict[str, str] = {}
    slug_by_page_uid: dict[str, str] = {}
    assignments = state.get("assignments") or {}
    for page_uid, row in (page_registry.get("pages") or {}).items():
        if not isinstance(row, Mapping) or row.get("status") != "active":
            continue
        assignment = assignments.get(page_uid)
        if not isinstance(assignment, Mapping):
            continue
        slug = collection_by_uid.get(str(assignment.get("collection_uid")), "")
        page_id = str(row.get("page_id") or "")
        if page_id:
            slug_by_page_id[page_id] = slug
            uid_by_page_id[page_id] = str(page_uid)
        slug_by_page_uid[str(page_uid)] = slug
    return slug_by_page_id, uid_by_page_id, slug_by_page_uid


def build_review_candidates(
    root: Path,
    *,
    state: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic anomaly candidates; never change assignments."""

    registry_state = state or CollectionRegistry(root).load()
    crosswalk = load_crosswalk()
    page_registry = PageRegistry(root).load()
    page_index = _load_page_index(root)
    slug_by_page_id, _uid_by_page_id, slug_by_page_uid = _page_id_maps(
        registry_state,
        page_registry,
    )
    entries = page_index.get("entries") or {}
    candidates: dict[str, dict[str, Any]] = {}
    for page_uid, assignment in sorted(
        (registry_state.get("assignments") or {}).items()
    ):
        if not isinstance(assignment, Mapping):
            continue
        slug = slug_by_page_uid.get(str(page_uid), "")
        crosswalk_row = crosswalk["by_slug"].get(slug) or {}
        reasons = []
        if assignment.get("status") == "unclassified":
            reasons.append(("unclassified", None, {}))
        if crosswalk_row.get("review_required"):
            reasons.append(("collection_requires_review", None, {}))
        page_row = (page_registry.get("pages") or {}).get(page_uid) or {}
        page_id = str(page_row.get("page_id") or "")
        outgoing = (
            entries.get(page_id, {}).get("outlinks") or []
            if isinstance(entries.get(page_id), Mapping)
            else []
        )
        linked = Counter(
            slug_by_page_id.get(str(target), "")
            for target in outgoing
            if slug_by_page_id.get(str(target), "")
        )
        resolved = sum(linked.values())
        current_links = linked.get(slug, 0)
        other = [
            (count, target_slug)
            for target_slug, count in linked.items()
            if target_slug != slug
        ]
        if resolved >= 3 and other:
            strongest_count, strongest_slug = max(
                other, key=lambda value: (value[0], value[1])
            )
            if (
                strongest_count / resolved >= 0.75
                and strongest_count >= current_links + 2
            ):
                reasons.append(
                    (
                        "cross_collection_link_affinity",
                        strongest_slug,
                        {
                            "resolved_outlinks": resolved,
                            "current_collection_outlinks": current_links,
                            "proposed_collection_outlinks": strongest_count,
                        },
                    )
                )
        for reason, proposed_slug, evidence in reasons:
            key = hashlib.sha256(
                f"{page_uid}\0{reason}\0{proposed_slug or ''}".encode()
            ).hexdigest()
            candidates[key] = {
                "candidate_id": key,
                "page_uid": str(page_uid),
                "current_collection_slug": slug,
                "proposed_collection_slug": proposed_slug,
                "reason": reason,
                "evidence": evidence,
                "status": "queued",
                "reviewer_role": "local_llm_or_human_review_only",
                "assignment_mutation": False,
                "page_mutation": False,
            }
    priority = {
        "unclassified": 0,
        "collection_requires_review": 1,
        "cross_collection_link_affinity": 2,
    }
    return sorted(
        candidates.values(),
        key=lambda row: (
            priority.get(str(row.get("reason") or ""), 99),
            str(row.get("page_uid") or ""),
            str(row.get("candidate_id") or ""),
        ),
    )


def refresh_review_queue(
    root: Path,
    *,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = load_contract()
    gates = contract["quality_gates"]
    path = root / "runtime" / "librarian" / "collection-review-queue.json"
    try:
        previous = read_sealed_json(path) if path.is_file() else {}
    except DurableStateError:
        previous = {}
    items = {
        str(key): dict(row)
        for key, row in (previous.get("items") or {}).items()
        if isinstance(row, Mapping)
    }
    candidates = build_review_candidates(root, state=state)
    max_open = int(gates["review_queue_open_max"])
    max_new = int(gates["review_queue_new_per_run_max"])
    open_count = sum(
        row.get("status") in {"queued", "review_recommended"}
        for row in items.values()
    )
    added = 0
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in items:
            continue
        if added >= max_new or open_count + added >= max_open:
            break
        items[candidate_id] = {**candidate, "created_at": _now()}
        added += 1
    active_candidate_ids = {str(row["candidate_id"]) for row in candidates}
    for candidate_id, row in items.items():
        if (
            row.get("status") == "queued"
            and candidate_id not in active_candidate_ids
        ):
            row["status"] = "resolved_by_evidence_change"
            row["resolved_at"] = _now()
    queue = {
        "schema": COLLECTION_QUEUE_SCHEMA,
        "updated_at": _now(),
        "budget": {
            "max_open": max_open,
            "max_new_per_run": max_new,
            "review_candidate_rate_warn": gates["review_candidate_rate_warn"],
        },
        "items": items,
        "candidate_count": len(candidates),
        "added": added,
        "open": sum(
            row.get("status") in {"queued", "review_recommended"}
            for row in items.values()
        ),
        "completed": sum(
            row.get("status")
            in {"dismissed", "move_approved", "resolved_by_evidence_change"}
            for row in items.values()
        ),
        "reviewer_calls": int(previous.get("reviewer_calls") or 0),
        "frontier_calls": 0,
        "page_mutations": 0,
        "assignment_mutations": 0,
    }
    write_sealed_json(path, queue, backup=True)
    return queue


def _checkpoint_review_queue(
    path: Path,
    queue: dict[str, Any],
    *,
    base_reviewer_calls: int,
    reviewed_count: int,
) -> None:
    """Persist review progress after every completed local-model call."""

    queue["updated_at"] = _now()
    queue["reviewer_calls"] = base_reviewer_calls + reviewed_count
    queue["frontier_calls"] = 0
    queue["page_mutations"] = 0
    queue["assignment_mutations"] = 0
    queue["open"] = sum(
        row.get("status") in {"queued", "review_recommended"}
        for row in queue["items"].values()
        if isinstance(row, Mapping)
    )
    queue["completed"] = sum(
        row.get("status")
        in {"dismissed", "move_approved", "resolved_by_evidence_change"}
        for row in queue["items"].values()
        if isinstance(row, Mapping)
    )
    write_sealed_json(path, queue, backup=True)


def _label_propagation(
    nodes: set[str],
    adjacency: Mapping[str, set[str]],
) -> list[list[str]]:
    labels = {node: node for node in nodes}
    for _iteration in range(50):
        changes = 0
        for node in sorted(nodes):
            neighbors = adjacency.get(node) or set()
            if not neighbors:
                continue
            counts = Counter(labels[value] for value in neighbors)
            maximum = max(counts.values())
            chosen = min(
                label for label, count in counts.items() if count == maximum
            )
            if labels[node] != chosen:
                labels[node] = chosen
                changes += 1
        if not changes:
            break
    groups: dict[str, list[str]] = defaultdict(list)
    for node, label in labels.items():
        groups[label].append(node)
    return sorted(
        (sorted(values) for values in groups.values()),
        key=lambda values: (-len(values), values[0]),
    )


def collection_quality_snapshot(
    root: Path,
    *,
    state: Mapping[str, Any] | None = None,
    queue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry_state = state or CollectionRegistry(root).load()
    crosswalk = load_crosswalk()
    contract = load_contract()
    page_registry = PageRegistry(root).load()
    page_index = _load_page_index(root)
    slug_by_page_id, _uid_by_page_id, slug_by_page_uid = _page_id_maps(
        registry_state,
        page_registry,
    )
    counts = Counter(slug_by_page_uid.values())
    total = sum(counts.values())
    active_slugs = {
        str(row.get("slug") or "")
        for row in (registry_state.get("collections") or {}).values()
        if isinstance(row, Mapping)
        and row.get("status") == "active"
        and not row.get("is_unclassified")
        and counts.get(str(row.get("slug") or ""), 0)
    }
    audited_slugs = active_slugs & set(crosswalk["by_slug"])
    duplicate_assignments = max(
        0, len(registry_state.get("assignments") or {}) - len(slug_by_page_uid)
    )
    sizes = sorted(value for key, value in counts.items() if key != "_unclassified")
    queue_state = dict(queue or refresh_review_queue(root, state=registry_state))
    candidate_rate = int(queue_state.get("candidate_count") or 0) / max(1, total)
    gates = contract["quality_gates"]
    split_proposals = []
    entries = page_index.get("entries") or {}
    for slug, count in counts.most_common():
        share = count / max(1, total)
        if share <= float(gates["top_collection_share_warn"]):
            continue
        nodes = {
            page_id
            for page_id, page_slug in slug_by_page_id.items()
            if page_slug == slug
        }
        adjacency = {node: set() for node in nodes}
        for node in nodes:
            row = entries.get(node)
            outlinks = row.get("outlinks") if isinstance(row, Mapping) else []
            for target in outlinks or []:
                target = str(target)
                if target in nodes:
                    adjacency[node].add(target)
                    adjacency[target].add(node)
        communities = _label_propagation(nodes, adjacency)
        split_proposals.append(
            {
                "collection_slug": slug,
                "page_count": count,
                "share": round(share, 6),
                "algorithm": "deterministic_label_propagation_v1",
                "decision": "proposal_only",
                "community_count": len(communities),
                "communities": [
                    {
                        "size": len(values),
                        "representatives": values[:5],
                    }
                    for values in communities[:20]
                    if len(values) >= 2
                ],
                "auto_split": False,
            }
        )
    link_path = root / "runtime" / "librarian" / "uid-link-index.json"
    try:
        link_index = _read_object(link_path)
    except CollectionAuthorityError:
        link_index = {}
    top_slug, top_count = counts.most_common(1)[0] if counts else ("", 0)
    metrics = {
        "page_count": total,
        "assignment_count": len(slug_by_page_uid),
        "assignment_coverage": round(len(slug_by_page_uid) / max(1, total), 6),
        "duplicate_page_assignment_count": duplicate_assignments,
        "active_collection_count": len(active_slugs),
        "crosswalk_audited_collection_count": len(audited_slugs),
        "crosswalk_audit_coverage": round(
            len(audited_slugs) / max(1, len(active_slugs)), 6
        ),
        "missing_crosswalk_slugs": sorted(active_slugs - audited_slugs),
        "top_collection_slug": top_slug,
        "top_collection_page_count": top_count,
        "top_collection_share": round(top_count / max(1, total), 6),
        "median_collection_size": median(sizes) if sizes else 0,
        "unclassified_count": counts.get("_unclassified", 0),
        "review_candidate_count": int(queue_state.get("candidate_count") or 0),
        "review_candidate_rate": round(candidate_rate, 6),
        "review_queue_open": int(queue_state.get("open") or 0),
        "unresolved_link_count": int(link_index.get("unresolved_count") or 0),
    }
    hard_failures = []
    if metrics["assignment_coverage"] < float(gates["assignment_coverage_min"]):
        hard_failures.append("assignment_coverage")
    if metrics["duplicate_page_assignment_count"] > int(
        gates["duplicate_page_assignment_max"]
    ):
        hard_failures.append("duplicate_page_assignment")
    if metrics["crosswalk_audit_coverage"] < float(
        gates["crosswalk_audit_coverage_min"]
    ):
        hard_failures.append("crosswalk_audit_coverage")
    if metrics["top_collection_share"] > float(
        gates["top_collection_share_block"]
    ):
        hard_failures.append("top_collection_share")
    if metrics["review_queue_open"] > int(gates["review_queue_open_max"]):
        hard_failures.append("review_queue_budget")
    if metrics["unresolved_link_count"] > int(gates["unresolved_link_max"]):
        hard_failures.append("unresolved_link")
    if metrics["median_collection_size"] < float(
        gates["median_collection_size_min"]
    ):
        hard_failures.append("median_collection_size")
    warnings = []
    if metrics["top_collection_share"] > float(
        gates["top_collection_share_warn"]
    ):
        warnings.append("top_collection_share")
    if metrics["review_candidate_rate"] > float(
        gates["review_candidate_rate_warn"]
    ):
        warnings.append("review_candidate_rate")
    return {
        "schema": COLLECTION_QUALITY_SCHEMA,
        "generated_at": _now(),
        "contract_epoch": contract["epoch"],
        "crosswalk_epoch": crosswalk["epoch"],
        "metrics": metrics,
        "gates": gates,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "status": "passed" if not hard_failures else "blocked",
        "split_proposals": split_proposals,
        "model_calls": 0,
        "frontier_calls": 0,
        "page_mutations": 0,
    }


def evaluate_unseen40(
    root: Path,
    *,
    selection_path: Path | None = None,
) -> dict[str, Any]:
    """Open the frozen 40 only after the new contract and gold are locked."""

    output = (
        root
        / "classification"
        / "collection-authority-v1"
        / "unseen40-evaluation.json"
    )
    if output.is_file():
        return read_sealed_json(output)
    prereg = _read_object(default_preregistration_path())
    gold = _read_object(default_gold_path())
    selected_path = selection_path or (
        root
        / "classification"
        / "cvo-ab-v1-unseen40"
        / "selection.json"
    )
    selection = read_sealed_json(selected_path)
    seal = str(selection.get("seal_sha256") or "")
    if (
        prereg.get("status") != "locked-before-evaluation"
        or gold.get("status") != "sealed-before-evaluation"
        or prereg.get("selection_seal_sha256") != seal
        or gold.get("selection_seal_sha256") != seal
        or int(prereg.get("case_count") or 0) != 40
    ):
        raise CollectionAuthorityError("unseen40 preregistration boundary mismatch")
    selected_uids = [str(row.get("uid") or "") for row in selection.get("cases") or []]
    gold_by_uid = {
        str(row.get("uid") or ""): row for row in gold.get("cases") or []
    }
    if len(selected_uids) != 40 or set(selected_uids) != set(gold_by_uid):
        raise CollectionAuthorityError("unseen40 gold identity mismatch")
    state = CollectionRegistry(root).load()
    collections = state.get("collections") or {}
    queue = refresh_review_queue(root, state=state)
    queued_by_uid = defaultdict(list)
    for row in (queue.get("items") or {}).values():
        if isinstance(row, Mapping) and row.get("status") in {
            "queued",
            "review_recommended",
        }:
            queued_by_uid[str(row.get("page_uid") or "")].append(dict(row))
    cases = []
    major_errors = 0
    assigned_correct = 0
    review_correct = 0
    crosswalk_invalid = 0
    crosswalk = load_crosswalk()
    for uid in selected_uids:
        expected = gold_by_uid[uid]
        assignment = (state.get("assignments") or {}).get(uid) or {}
        collection = collections.get(str(assignment.get("collection_uid") or "")) or {}
        slug = str(collection.get("slug") or "")
        acceptable = {
            str(value) for value in expected.get("acceptable_collection_slugs") or []
        }
        is_assigned_correct = slug in acceptable
        is_review_correct = (
            expected.get("disposition") == "review"
            and bool(queued_by_uid.get(uid))
        )
        passed = is_assigned_correct or is_review_correct
        mapping = crosswalk["by_slug"].get(slug)
        mapping_valid = bool(mapping and mapping.get("mappings"))
        crosswalk_invalid += int(not mapping_valid)
        assigned_correct += int(is_assigned_correct)
        review_correct += int(is_review_correct and not is_assigned_correct)
        major_errors += int(not passed)
        cases.append(
            {
                "uid": uid,
                "assigned_collection_slug": slug,
                "expected_disposition": expected.get("disposition"),
                "acceptable_collection_slugs": sorted(acceptable),
                "assigned_correct": is_assigned_correct,
                "review_correct": is_review_correct,
                "queue_reasons": sorted(
                    {str(row.get("reason") or "") for row in queued_by_uid.get(uid, [])}
                ),
                "crosswalk_valid": mapping_valid,
                "major_error": not passed,
            }
        )
    gates = prereg["evaluation_contract"]
    success = (
        (assigned_correct + review_correct) / 40
        >= float(gates["assignment_or_review_rate_min"])
        and major_errors <= int(gates["major_error_max"])
        and crosswalk_invalid <= int(gates["crosswalk_invalid_max"])
    )
    result = {
        "schema": COLLECTION_EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "epoch": prereg["epoch"],
        "selection_seal_sha256": seal,
        "preregistration_sha256": _content_sha256(
            default_preregistration_path()
        ),
        "gold_sha256": _content_sha256(default_gold_path()),
        "case_count": 40,
        "assigned_correct": assigned_correct,
        "review_correct": review_correct,
        "assignment_or_review_rate": round(
            (assigned_correct + review_correct) / 40, 6
        ),
        "major_error_count": major_errors,
        "crosswalk_invalid_count": crosswalk_invalid,
        "model_calls": 0,
        "frontier_calls": 0,
        "page_mutations": 0,
        "decision": "adopt" if success else "reject",
        "gates": gates,
        "cases": cases,
    }
    write_sealed_json(output, result, backup=True)
    return read_sealed_json(output)


def collection_authority_status(root: Path) -> dict[str, Any]:
    try:
        contract = load_contract()
        crosswalk = load_crosswalk()
        state = CollectionRegistry(root).load()
        quality_path = (
            root / "runtime" / "librarian" / "collection-quality.json"
        )
        quality = (
            read_sealed_json(quality_path)
            if quality_path.is_file()
            else {"status": "missing", "metrics": {}}
        )
        evaluation_path = (
            root
            / "classification"
            / "collection-authority-v1"
            / "unseen40-evaluation.json"
        )
        evaluation = (
            read_sealed_json(evaluation_path)
            if evaluation_path.is_file()
            else {"decision": "missing"}
        )
    except (
        CollectionAuthorityError,
        DurableStateError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return {
            "active": False,
            "mode": "collection-first",
            "reason": f"collection_authority_unavailable:{type(exc).__name__}",
        }
    reasons = []
    if not state.get("assignments"):
        reasons.append("collection_registry_not_synced")
    if quality.get("status") != "passed":
        reasons.append("collection_quality_gate_not_passed")
    if evaluation.get("decision") != "adopt":
        reasons.append("collection_unseen40_not_adopted")
    active = not reasons
    return {
        "active": active,
        "mode": "collection-first",
        "reason": ",".join(reasons) if reasons else "collection_authority_adopted",
        "authority_epoch": contract["epoch"],
        "threshold_version": contract["epoch"],
        "registry_generation": int(state.get("generation") or 0),
        "collection_count": len(crosswalk["entries"]),
        "crosswalk_epoch": crosswalk["epoch"],
        "quality_status": quality.get("status"),
        "unseen40_decision": evaluation.get("decision"),
        "package_complete": True,
        "calibrated": evaluation.get("decision") == "adopt",
    }


def review_collection_queue(
    root: Path = CHRONOVISOR_ROOT,
    *,
    limit: int = 10,
    model: str | None = None,
    role: str = "primary",
    read_timeout_ms: int = 660_000,
) -> dict[str, Any]:
    """Review queued anomalies locally without assignment mutation."""

    from chronovisor import frontmatter, ollama
    from chronovisor.collection_anomaly_worker import (
        PROMPT_SHA256,
        WORKER_SCHEMA,
    )

    contract = load_contract()
    if role not in {"primary", "challenger"}:
        raise CollectionAuthorityError(f"unsupported anomaly review role: {role}")
    selected_model = model or (
        str(contract["anomaly_reviewer"]["default_model"])
        if role == "primary"
        else DEFAULT_CHALLENGER_MODEL
    )
    digest = ollama.model_digests([selected_model]).get(selected_model, "")
    if not digest:
        raise CollectionAuthorityError(
            f"anomaly reviewer model is unavailable: {selected_model}"
        )
    queue_path = (
        root / "runtime" / "librarian" / "collection-review-queue.json"
    )
    queue = (
        read_sealed_json(queue_path)
        if queue_path.is_file()
        else refresh_review_queue(root)
    )
    state = CollectionRegistry(root).load()
    page_state = PageRegistry(root).load()
    collections = [
        {
            "slug": str(row.get("slug") or ""),
            "label": str(row.get("label") or ""),
        }
        for row in state["collections"].values()
        if isinstance(row, Mapping) and row.get("status") == "active"
    ]
    reconciled = []
    if role == "primary":
        for candidate_id, raw_row in sorted((queue.get("items") or {}).items()):
            if (
                not isinstance(raw_row, Mapping)
                or raw_row.get("status") != "queued"
            ):
                continue
            review = raw_row.get("model_review")
            if (
                not isinstance(review, Mapping)
                or review.get("decision") != "no_issue"
                or review.get("model") != selected_model
                or review.get("model_digest") != digest
                or review.get("prompt_sha256") != PROMPT_SHA256
            ):
                continue
            row = dict(raw_row)
            row["status"] = "dismissed"
            row["resolved_at"] = _now()
            row["resolution"] = "model_no_issue_preserve_original_order"
            queue["items"][candidate_id] = row
            reconciled.append(str(candidate_id))
    base_reviewer_calls = int(queue.get("reviewer_calls") or 0)
    _checkpoint_review_queue(
        queue_path,
        queue,
        base_reviewer_calls=base_reviewer_calls,
        reviewed_count=0,
    )
    pending = [
        (str(candidate_id), dict(row))
        for candidate_id, row in sorted((queue.get("items") or {}).items())
        if isinstance(row, Mapping)
        and (
            (
                role == "primary"
                and row.get("status") in {"queued", "review_recommended"}
                and not isinstance(row.get("model_review"), Mapping)
            )
            or (
                role == "challenger"
                and row.get("status") == "review_recommended"
                and isinstance(row.get("model_review"), Mapping)
                and not isinstance(row.get("challenger_review"), Mapping)
            )
        )
    ][: max(0, limit)]
    reviewed = []
    deferred = []
    for candidate_id, candidate in pending:
        page_uid = str(candidate.get("page_uid") or "")
        page = (page_state.get("pages") or {}).get(page_uid)
        if not isinstance(page, Mapping):
            deferred.append(
                {"candidate_id": candidate_id, "reason": "page_missing"}
            )
            continue
        path = root / str(page.get("path") or "")
        try:
            text = path.read_text(encoding="utf-8")
            meta, body = frontmatter.parse(text)
        except (OSError, UnicodeError) as exc:
            deferred.append(
                {
                    "candidate_id": candidate_id,
                    "reason": f"page_unreadable:{type(exc).__name__}",
                }
            )
            continue
        payload = {
            "schema": WORKER_SCHEMA,
            "model": selected_model,
            "model_digest": digest,
            "read_timeout_ms": read_timeout_ms,
            "candidate": candidate,
            "document": {
                "title": str(meta.get("title") or path.stem),
                "summary": str(meta.get("summary") or ""),
                "evidence_excerpt": body.strip()[:2_400],
            },
            "collections": collections,
        }
        with research_lane(
            f"collection-review-{page_uid[:10]}-{uuid.uuid4().hex[:8]}",
            enabled=True,
            mode="on",
            purpose="explicit",
            needs_model=True,
        ) as lease:
            outcome = run_cancellable_command(
                [
                    sys.executable,
                    "-m",
                    "chronovisor.collection_anomaly_worker",
                ],
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                lease,
                timeout_seconds=max(60.0, read_timeout_ms / 1_000 + 30),
            )
        if outcome.status != "completed" or not isinstance(
            outcome.value, Mapping
        ):
            deferred.append(
                {
                    "candidate_id": candidate_id,
                    "reason": outcome.error or outcome.status,
                }
            )
            if outcome.status in {"deferred", "cancelled"}:
                # Resource pressure and P0 preemption apply to the whole local
                # model lane. Stop this batch instead of pointlessly marking
                # every remaining item with the same transient condition.
                break
            continue
        worker = dict(outcome.value)
        if (
            worker.get("schema") != WORKER_SCHEMA
            or worker.get("model") != selected_model
            or worker.get("model_digest") != digest
            or worker.get("prompt_sha256") != PROMPT_SHA256
            or int(worker.get("model_calls") or 0) != 1
            or int(worker.get("page_mutations") or 0) != 0
            or int(worker.get("assignment_mutations") or 0) != 0
            or not isinstance(worker.get("result"), Mapping)
        ):
            deferred.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "worker_contract_mismatch",
                }
            )
            continue
        review = dict(worker["result"])
        row = dict(queue["items"][candidate_id])
        review_field = (
            "model_review" if role == "primary" else "challenger_review"
        )
        row[review_field] = {
            **review,
            "model": selected_model,
            "model_digest": digest,
            "prompt_sha256": PROMPT_SHA256,
            "reviewed_at": _now(),
        }
        if role == "primary":
            if review.get("decision") == "review_recommended":
                row["status"] = "review_recommended"
            elif review.get("decision") == "no_issue":
                # Preserving original order is the fail-safe action. The local
                # reviewer may close the anomaly without gaining assignment
                # authority because no collection or page mutation occurs.
                row["status"] = "dismissed"
                row["resolved_at"] = _now()
                row["resolution"] = "model_no_issue_preserve_original_order"
        else:
            primary = row["model_review"]
            if review.get("decision") == "no_issue":
                row["status"] = "dismissed"
                row["resolved_at"] = _now()
                row["resolution"] = (
                    "challenger_no_issue_preserve_original_order"
                )
                row["challenge_status"] = "rejected_recommendation"
            elif (
                review.get("decision") == "review_recommended"
                and review.get("suggested_collection_slug")
                == primary.get("suggested_collection_slug")
            ):
                row["challenge_status"] = "consensus_recommended"
            else:
                row["challenge_status"] = "disagreement_or_insufficient"
        queue["items"][candidate_id] = row
        reviewed.append(
            {
                "candidate_id": candidate_id,
                "decision": review.get("decision"),
            }
        )
        _checkpoint_review_queue(
            queue_path,
            queue,
            base_reviewer_calls=base_reviewer_calls,
            reviewed_count=len(reviewed),
        )
    _checkpoint_review_queue(
        queue_path,
        queue,
        base_reviewer_calls=base_reviewer_calls,
        reviewed_count=len(reviewed),
    )
    return {
        "status": "ok" if not deferred else "partial",
        "role": role,
        "model": selected_model,
        "model_digest": digest,
        "reconciled": reconciled,
        "reviewed": reviewed,
        "deferred": deferred,
        "reviewer_calls": len(reviewed),
        "frontier_calls": 0,
        "page_mutations": 0,
        "assignment_mutations": 0,
    }


def run_collection_librarian(
    root: Path = CHRONOVISOR_ROOT,
    *,
    evaluate_unseen: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run collection synchronization, deterministic review triage and gates."""

    registry = CollectionRegistry(root)
    sync = registry.sync_from_pages(dry_run=dry_run)
    if dry_run:
        return {
            "status": "dry_run",
            "sync": sync,
            "model_calls": 0,
            "frontier_calls": 0,
            "page_mutations": 0,
        }
    build_uid_link_index(root, write=True)
    state = registry.load()
    queue = refresh_review_queue(root, state=state)
    quality = collection_quality_snapshot(root, state=state, queue=queue)
    write_sealed_json(
        root / "runtime" / "librarian" / "collection-quality.json",
        quality,
        backup=True,
    )
    evaluation = evaluate_unseen40(root) if evaluate_unseen else None
    status = collection_authority_status(root)
    receipt = {
        "schema": "chronovisor.collection-authority-phase4.v1",
        "status": "adopted" if status.get("active") else "shadow",
        "recorded_at": _now(),
        "contract": load_contract(),
        "crosswalk": {
            "epoch": load_crosswalk()["epoch"],
            "entry_count": len(load_crosswalk()["entries"]),
            "checksum": load_crosswalk()["checksum"],
        },
        "sync": {
            key: sync.get(key)
            for key in (
                "generation",
                "collection_count",
                "assignment_count",
                "changed_assignments",
            )
        },
        "quality": quality,
        "unseen40": (
            {
                key: evaluation.get(key)
                for key in (
                    "decision",
                    "case_count",
                    "assigned_correct",
                    "review_correct",
                    "major_error_count",
                )
            }
            if evaluation
            else {"decision": "not_opened"}
        ),
        "authority": status,
        "model_calls": 0,
        "frontier_calls": 0,
        "page_mutations": 0,
    }
    write_sealed_json(
        root / "runtime" / "librarian" / "phase4-collection-authority.json",
        receipt,
        backup=True,
    )
    return {
        "status": "ok" if quality["status"] == "passed" else "blocked",
        "sync": sync,
        "queue": {
            "candidate_count": queue["candidate_count"],
            "open": queue["open"],
            "added": queue["added"],
        },
        "quality": quality,
        "evaluation": evaluation,
        "authority": status,
        "model_calls": 0,
        "frontier_calls": 0,
        "page_mutations": 0,
    }
