"""Read-only audit of existing Chronovisor folders as native collections."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from chronovisor.classification import ClassificationError
from chronovisor.classification_anchor import (
    UNRESOLVED_ANCHOR_ID,
    load_anchor_set,
)
from chronovisor.classification_anchor_set_dev import (
    default_dev_gold_path,
    load_dev70,
    score_anchor_set,
    summarize_metrics,
    validate_set_gold,
)
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.store import CHRONOVISOR_ROOT

SCHEMA = "chronovisor.collection-authority-pilot.v1"
EXPERIMENT = "cvo-collection-authority-v1"

# Semantic crosswalk is diagnostic only. The proposed authority is the native
# collection itself; UDC/CVO anchors become optional collection-level overlays.
DIAGNOSTIC_CROSSWALK = {
    "ai": "cvo:anchor:0002",
    "ai-coding-support-rules": "cvo:anchor:0001",
    "auto-industry": "cvo:anchor:0011",
    "car-spec": "cvo:anchor:0011",
    "career": "cvo:anchor:0008",
    "chronovisor": "cvo:anchor:0003",
    "chronovisor-recall": "cvo:anchor:0003",
    "chronovisor-recall-redesign": "cvo:anchor:0003",
    "culture": "cvo:anchor:0025",
    "jt": "cvo:anchor:0001",
    "lazarus": "cvo:anchor:0001",
    "sports": "cvo:anchor:0027",
    "workplace": "cvo:anchor:0009",
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def output_path(root: Path) -> Path:
    return root / "classification" / EXPERIMENT / "evaluation.json"


def _collection_from_path(value: object) -> str:
    path = str(value or "")
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "pages" and parts[1]:
        return parts[1]
    return ""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ClassificationError(f"expected JSON object: {path}")
    return value


def audit_native_collections(
    registry: Mapping[str, Any],
    page_index: Mapping[str, Any],
) -> dict[str, Any]:
    raw_pages = registry.get("pages")
    raw_entries = page_index.get("entries")
    if not isinstance(raw_pages, Mapping) or not isinstance(
        raw_entries, Mapping
    ):
        raise ClassificationError("collection audit input contract mismatch")
    collection_by_page_id: dict[str, str] = {}
    collection_by_uid: dict[str, str] = {}
    counts: Counter[str] = Counter()
    active_page_count = 0
    unassigned_uids = []
    duplicate_page_ids = []
    for uid, raw in raw_pages.items():
        if not isinstance(raw, Mapping) or raw.get("status") != "active":
            continue
        path = str(raw.get("path") or "")
        if not path.startswith("pages/"):
            continue
        active_page_count += 1
        collection = _collection_from_path(path)
        page_id = str(raw.get("page_id") or "")
        stable_uid = str(raw.get("uid") or uid)
        if not collection or not page_id:
            unassigned_uids.append(stable_uid)
            continue
        if page_id in collection_by_page_id:
            duplicate_page_ids.append(page_id)
            continue
        collection_by_page_id[page_id] = collection
        collection_by_uid[stable_uid] = collection
        counts[collection] += 1
    resolved_edges = 0
    intra_collection_edges = 0
    cross_collection_edges = 0
    unresolved_edges = 0
    for source_page_id, raw in raw_entries.items():
        if not isinstance(raw, Mapping):
            continue
        source_collection = collection_by_page_id.get(str(source_page_id))
        if not source_collection:
            continue
        for target in raw.get("outlinks") or []:
            target_collection = collection_by_page_id.get(str(target))
            if not target_collection:
                unresolved_edges += 1
                continue
            resolved_edges += 1
            if target_collection == source_collection:
                intra_collection_edges += 1
            else:
                cross_collection_edges += 1
    sizes = sorted(counts.values())
    top_collections = [
        {"collection": name, "page_count": count}
        for name, count in counts.most_common(20)
    ]
    return {
        "active_page_count": active_page_count,
        "assigned_collection_count": len(collection_by_uid),
        "assignment_coverage": round(
            len(collection_by_uid) / max(1, active_page_count), 6
        ),
        "unassigned_count": len(unassigned_uids),
        "unassigned_uids": sorted(unassigned_uids),
        "duplicate_page_id_count": len(duplicate_page_ids),
        "duplicate_page_ids": sorted(set(duplicate_page_ids)),
        "collection_count": len(counts),
        "median_collection_size": median(sizes) if sizes else 0,
        "largest_collection_share": round(
            max(sizes, default=0) / max(1, len(collection_by_uid)), 6
        ),
        "top_collections": top_collections,
        "resolved_link_edges": resolved_edges,
        "intra_collection_edges": intra_collection_edges,
        "cross_collection_edges": cross_collection_edges,
        "unresolved_or_external_edges": unresolved_edges,
        "intra_collection_link_rate": round(
            intra_collection_edges / max(1, resolved_edges), 6
        ),
        "collection_by_uid": collection_by_uid,
    }


def evaluate_opened70_crosswalk(
    root: Path,
    collection_by_uid: Mapping[str, str],
) -> dict[str, Any]:
    pages = load_dev70(root)
    anchor_set = load_anchor_set()
    payload = _load_json(default_dev_gold_path())
    gold = validate_set_gold(
        payload,
        anchor_set,
        [str(page.get("uid") or "") for page in pages],
    )
    cases = []
    for page in pages:
        uid = str(page.get("uid") or "")
        collection = str(collection_by_uid.get(uid) or "")
        selected = [
            DIAGNOSTIC_CROSSWALK.get(
                collection,
                UNRESOLVED_ANCHOR_ID,
            )
        ]
        score = score_anchor_set(
            selected,
            gold[uid]["target"],
            gold[uid]["defensible"],
            gold[uid]["acceptable_sets"],
        )
        cases.append(
            {
                "uid": uid,
                "title": str(page.get("title") or ""),
                "collection": collection,
                "selected_anchor_ids": selected,
                **score,
            }
        )
    return {
        "fixture_set": "opened70-development-only",
        "diagnostic_only": True,
        "crosswalk": dict(sorted(DIAGNOSTIC_CROSSWALK.items())),
        "metrics": summarize_metrics(cases),
        "cases": cases,
    }


def run_pilot(root: Path) -> dict[str, Any]:
    destination = output_path(root)
    if destination.is_file():
        return read_sealed_json(destination)
    registry_path = root / "runtime" / "librarian" / "page-registry.json"
    page_index_path = root / ".index" / "pages.json"
    audit = audit_native_collections(
        _load_json(registry_path),
        _load_json(page_index_path),
    )
    collection_by_uid = dict(audit.pop("collection_by_uid"))
    dev = evaluate_opened70_crosswalk(root, collection_by_uid)
    receipt = {
        "schema": SCHEMA,
        "evaluated_at": _now(),
        "method": "existing-folder-as-native-collection-authority",
        "registry_path": str(registry_path),
        "registry_generation": _load_json(registry_path).get("generation"),
        "page_index_path": str(page_index_path),
        "page_index_generation": _load_json(page_index_path).get(
            "generation"
        ),
        "model_calls": 0,
        "page_mutations": 0,
        "full_corpus": audit,
        "opened70_anchor_crosswalk_diagnostic": dev,
        "decision": (
            "design-native-collection-registry"
            if audit["assignment_coverage"] >= 0.99
            and audit["duplicate_page_id_count"] == 0
            else "repair-collection-identity-inputs"
        ),
        "contract_consequence": (
            "Native collection is primary authority. UDC/CVO anchor becomes "
            "an optional collection-level crosswalk, never a mandatory "
            "per-page prediction."
        ),
    }
    write_sealed_json(destination, receipt, backup=True)
    return read_sealed_json(destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        print(
            json.dumps(
                run_pilot(args.root.expanduser()),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (ClassificationError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
