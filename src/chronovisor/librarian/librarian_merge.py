"""Conservative cluster planning and full-corpus Librarian merge migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from chronovisor.core import frontmatter
from chronovisor.core.canonical_document import (
    Namespace,
    ResolvedMarkdownLink,
    rewrite_internal_markdown_links,
)
from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.migration_snapshot import (
    create_incremental_restore_point,
    restore_drill,
)
from chronovisor.core.raw_store import RawStore
from chronovisor.core.store import CHRONOVISOR_ROOT, okf_runtime_operation
from chronovisor.ingest.page_registry import PageRegistry, PageRegistryError
from chronovisor.ingest.uid_link_index import build_uid_link_index
from chronovisor.recall.classification import strongest_sensitivity
from chronovisor.recall.duplicate_review import build_duplicate_review_queue
from chronovisor.recall.librarian import append_librarian_event, librarian_now_iso
from chronovisor.recall.merge_ledger import build_source_inventory
from chronovisor.recall.merge_transaction import apply_merge_plan, prepare_merge_plan

DISPOSITION_SCHEMA = "chronovisor.librarian-dispositions.v1"
PILOT_SCHEMA = "chronovisor.librarian-pilot.v1"
MERGE_SCORE_THRESHOLD = 0.95


def _load_raw_refs(
    root: Path,
    registry: PageRegistry,
    *,
    only_uids: set[str] | None = None,
    registry_state: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    raw_store = RawStore(root / "raw")
    loaded_state = dict(registry_state or registry.load())
    target_uids = set(only_uids or loaded_state["pages"])
    uid_by_key = {
        str(key).removesuffix(".md").casefold(): str(uid)
        for key, uid in (loaded_state.get("keys") or {}).items()
        if str(uid) in target_uids
    }
    refs: dict[str, set[str]] = defaultdict(set)
    for filename in ("claims/claims.jsonl", "claims/claims-index.jsonl"):
        path = root / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_page = str(row.get("source_page") or "")
            source_raw = str(row.get("source_raw") or "")
            if not source_page or not source_raw:
                continue
            resolved_uid = uid_by_key.get(source_page.removesuffix(".md").casefold())
            if resolved_uid is None:
                continue
            raw_name = Path(source_raw.removeprefix("replay:")).name
            try:
                raw_unit = raw_store.resolve(raw_name)
            except (OSError, ValueError):
                raw_unit = None
            if raw_unit is None:
                continue
            refs[resolved_uid].add(f"{raw_unit.raw_id}#sha256={raw_unit.sha256}")
    return {uid: sorted(values) for uid, values in refs.items()}


def _components(records: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if float(record.get("score") or 0.0) < MERGE_SCORE_THRESHOLD:
            continue
        left = str(record.get("left") or "")
        right = str(record.get("right") or "")
        if not left or not right or left == right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[list[str]] = []
    seen: set[str] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(sorted(adjacency[node] - seen, reverse=True))
        if len(component) > 1:
            components.append(sorted(component))
    return components


def discover_clusters(
    root: Path = CHRONOVISOR_ROOT,
    *,
    limit: int = 200,
) -> dict[str, Any]:
    del root
    records = build_duplicate_review_queue(
        title_threshold=0.90,
        embedding_threshold=0.92,
        include_embeddings=True,
        limit=limit,
        strict=False,
    )
    return {
        "records": records,
        "components": _components(records),
    }


def _rewrite_links(
    text: str,
    *,
    registry: PageRegistry,
    registry_state: Mapping[str, Any],
    loser_uids: set[str],
    source_namespace: Namespace,
    source_path: str,
    canonical_namespace: Namespace,
    canonical_path: str,
    output_namespace: Namespace | None = None,
    output_path: str | None = None,
) -> tuple[str, int, set[str]]:
    changed = 0
    anchors: set[str] = set()

    def replace(link: ResolvedMarkdownLink, _label: str) -> ResolvedMarkdownLink:
        nonlocal changed
        target = (
            f"{link.namespace}/"
            f"{PurePosixPath(link.path).with_suffix('').as_posix()}"
        )
        try:
            resolved = registry.resolve_from_state(registry_state, target)
        except PageRegistryError:
            resolved = None
        if resolved is None or str(resolved["uid"]) not in loser_uids:
            return link
        if link.fragment:
            anchors.add(link.fragment)
        changed += 1
        return ResolvedMarkdownLink(
            namespace=canonical_namespace,
            path=canonical_path,
            fragment=link.fragment,
        )

    rewritten, _rendered_changes = rewrite_internal_markdown_links(
        text,
        source_namespace=source_namespace,
        source_path=source_path,
        rewrite=replace,
        output_namespace=output_namespace,
        output_path=output_path,
    )
    return rewritten, changed, anchors


def _canonical_location(root: Path, path: Path) -> tuple[Namespace, str]:
    relative = path.relative_to(root)
    namespace: Namespace = "system" if relative.parts[0] == "system" else "pages"
    return namespace, PurePosixPath(*relative.parts[1:]).as_posix()


def _build_union_output(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    canonical_uid: str,
    registry: PageRegistry,
    registry_state: Mapping[str, Any],
    root: Path,
) -> tuple[str, dict[str, str], set[str]]:
    canonical = next(row for row in source_rows if str(row["uid"]) == canonical_uid)
    canonical_path = root / str(canonical["path"])
    canonical_namespace, canonical_relative = _canonical_location(root, canonical_path)
    canonical_text = canonical_path.read_text(encoding="utf-8")
    canonical_meta, canonical_body = frontmatter.parse(canonical_text)
    source_metadata = [canonical_meta]
    loser_uids = {
        str(row["uid"]) for row in source_rows if str(row["uid"]) != canonical_uid
    }
    all_anchors: set[str] = set()
    sections = [
        canonical_body.rstrip(),
        "",
        f"^chronovisor-source-uid-{canonical_uid}",
    ]
    for row in sorted(source_rows, key=lambda value: str(value["uid"])):
        if str(row["uid"]) == canonical_uid:
            continue
        source_text = (root / str(row["path"])).read_text(encoding="utf-8")
        source_path = root / str(row["path"])
        source_namespace, source_relative = _canonical_location(root, source_path)
        source_meta, source_body = frontmatter.parse(source_text)
        source_metadata.append(source_meta)
        rewritten, _count, anchors = _rewrite_links(
            source_body,
            registry=registry,
            registry_state=registry_state,
            loser_uids=loser_uids,
            source_namespace=source_namespace,
            source_path=source_relative,
            canonical_namespace=canonical_namespace,
            canonical_path=canonical_relative,
            output_namespace=canonical_namespace,
            output_path=canonical_relative,
        )
        all_anchors.update(anchors)
        title = str(source_meta.get("title") or Path(str(row["path"])).stem)
        sections.extend(
            [
                "",
                f"## Consolidated source: {title}",
                f"<!-- chronovisor-source-uid: {row['uid']} -->",
                f"^chronovisor-source-uid-{row['uid']}",
                rewritten.rstrip(),
            ]
        )
    combined_body = "\n".join(sections).rstrip() + "\n"
    output_sensitivity = strongest_sensitivity(
        str(row.get("sensitivity") or "normal") for row in source_rows
    )
    canonical_meta["sensitivity"] = output_sensitivity
    for field in ("recall_questions", "raw_keywords", "tags", "aliases", "entities"):
        merged: list[str] = []
        seen: set[str] = set()
        for metadata in source_metadata:
            values = metadata.get(field)
            if not isinstance(values, list):
                continue
            for value in values:
                text = str(value).strip()
                key = text.casefold()
                if text and key not in seen:
                    seen.add(key)
                    merged.append(text)
        if merged:
            canonical_meta[field] = merged
    output = frontmatter.patch(combined_body, canonical_meta)
    anchor_maps = {
        str(row["uid"]): {anchor: anchor for anchor in sorted(all_anchors)}
        for row in source_rows
        if str(row["uid"]) != canonical_uid
    }
    return output, anchor_maps, all_anchors


def _incoming_rewrites(
    root: Path,
    *,
    registry: PageRegistry,
    registry_state: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    canonical_uid: str,
) -> tuple[dict[str, str], set[str]]:
    source_uids = {str(row["uid"]) for row in source_rows}
    loser_uids = source_uids - {canonical_uid}
    canonical = next(row for row in source_rows if str(row["uid"]) == canonical_uid)
    canonical_path = root / str(canonical["path"])
    canonical_namespace, canonical_relative = _canonical_location(root, canonical_path)
    updates: dict[str, str] = {}
    anchors: set[str] = set()
    for uid, row in registry.stable_pages(registry_state).items():
        if (
            uid in source_uids
        ):
            continue
        path = root / str(row.get("path") or "")
        text = path.read_text(encoding="utf-8")
        source_namespace, source_relative = _canonical_location(root, path)
        rewritten, count, found_anchors = _rewrite_links(
            text,
            registry=registry,
            registry_state=registry_state,
            loser_uids=loser_uids,
            source_namespace=source_namespace,
            source_path=source_relative,
            canonical_namespace=canonical_namespace,
            canonical_path=canonical_relative,
        )
        if count:
            updates[str(uid)] = rewritten
            anchors.update(found_anchors)
    return updates, anchors


def _choose_canonical(
    rows: Sequence[Mapping[str, Any]],
    *,
    link_index: Mapping[str, Any],
) -> str:
    incoming = {
        str(uid): len(values)
        for uid, values in (link_index.get("reverse") or {}).items()
        if isinstance(values, list)
    }
    ranked = sorted(
        rows,
        key=lambda row: (
            -incoming.get(str(row["uid"]), 0),
            -int(row.get("size") or 0),
            str(row["uid"]),
        ),
    )
    return str(ranked[0]["uid"])


def prepare_cluster_plan(
    root: Path,
    *,
    page_keys: Sequence[str],
) -> dict[str, Any]:
    registry = PageRegistry(root)
    registry_state = registry.load()
    stable_pages = registry.stable_pages(registry_state)
    rows_by_uid = {}
    for key in page_keys:
        row = registry.resolve_from_state(registry_state, key)
        if row is None or row.get("uid") not in stable_pages:
            raise KeyError(key)
        rows_by_uid[str(row["uid"])] = stable_pages[str(row["uid"])]
    rows = list(rows_by_uid.values())
    if len(rows) < 2:
        raise ValueError("merge cluster requires at least two stable pages")
    raw_refs = _load_raw_refs(
        root,
        registry,
        only_uids=set(rows_by_uid),
        registry_state=registry_state,
    )
    missing_raw = sorted(uid for uid in rows_by_uid if not raw_refs.get(uid))
    if missing_raw:
        return {
            "status": "kept",
            "reason": "raw_provenance_unrecoverable_keep_both",
            "uids": sorted(rows_by_uid),
            "missing_raw_uids": missing_raw,
        }
    link_index = build_uid_link_index(root, registry=registry, write=True)
    canonical_uid = _choose_canonical(rows, link_index=link_index)
    output, anchor_maps, _anchors = _build_union_output(
        rows,
        canonical_uid=canonical_uid,
        registry=registry,
        registry_state=registry_state,
        root=root,
    )
    affected, incoming_anchors = _incoming_rewrites(
        root,
        registry=registry,
        registry_state=registry_state,
        source_rows=rows,
        canonical_uid=canonical_uid,
    )
    for uid in anchor_maps:
        anchor_maps[uid].update({anchor: anchor for anchor in sorted(incoming_anchors)})
    missing_heading_anchors = []
    for anchor in sorted(incoming_anchors):
        normalized = anchor.strip().casefold().replace(" ", "-")
        if not any(
            line.startswith("#")
            and line.lstrip("#").strip().casefold().replace(" ", "-") == normalized
            for line in output.splitlines()
        ):
            missing_heading_anchors.append(anchor)
    if missing_heading_anchors:
        _meta, body = frontmatter.parse(output)
        body = (
            body.rstrip()
            + "\n\n"
            + "\n\n".join(f"## {anchor}\n" for anchor in missing_heading_anchors)
        )
        output = frontmatter.patch(body, _meta)

    source_texts = {
        uid: (root / str(row["path"])).read_text(encoding="utf-8")
        for uid, row in rows_by_uid.items()
    }
    inventory = build_source_inventory(source_texts)
    mappings = []
    dispositions = []
    for uid, source in inventory.items():
        refs = raw_refs[uid]
        for span in source["spans"]:
            span_text = str(span["text"])
            if span_text in output:
                action = "output"
            elif span["kind"] == "boilerplate":
                action = "boilerplate"
            else:
                action = "ledger"
                dispositions.append(
                    {
                        "source_uid": uid,
                        "span_index": span["index"],
                        "span_sha256": span["sha256"],
                        "text": span_text,
                        "reason": "preserved_exactly_in_merge_ledger",
                        "raw_refs": refs,
                    }
                )
            mappings.append(
                {
                    "source_uid": uid,
                    "span_index": span["index"],
                    "span_sha256": span["sha256"],
                    "action": action,
                    "output_anchor": (
                        f"chronovisor-source-uid-{uid}" if action == "output" else None
                    ),
                    "raw_refs": refs,
                }
            )
        fingerprints = source.get("fingerprints")
        dispositions.append(
            {
                "source_uid": uid,
                "reason": "deterministic_fingerprint_preservation",
                "fingerprints": fingerprints,
                "raw_refs": refs,
            }
        )
    output_sensitivity = strongest_sensitivity(
        str(row.get("sensitivity") or "normal") for row in rows
    )
    return prepare_merge_plan(
        root,
        source_keys=list(page_keys),
        canonical_key=canonical_uid,
        canonical_content=output,
        mappings=mappings,
        ledger_dispositions=dispositions,
        output_sensitivity=output_sensitivity,
        affected_page_updates=affected,
        anchor_maps=anchor_maps,
    )


def _verified_incremental(
    root: Path,
    *,
    paths: Iterable[Path],
    reason: str,
) -> dict[str, Any]:
    restore = create_incremental_restore_point(
        root,
        paths=paths,
        reason=reason,
        ttl_days=7,
    )
    with tempfile.TemporaryDirectory(prefix="chronovisor-merge-restore-") as temp:
        drill = restore_drill(Path(restore["path"]), Path(temp) / "restored")
    if drill["status"] != "verified":
        raise RuntimeError("merge restore drill failed")
    return restore


def _load_dispositions(root: Path) -> dict[str, Any]:
    path = root / "runtime" / "librarian" / "migration-dispositions.json"
    if not path.exists():
        return {
            "schema": DISPOSITION_SCHEMA,
            "generated_at": librarian_now_iso(),
            "scope_generation": "",
            "pages": {},
            "clusters": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def run_merge_migration(
    root: Path = CHRONOVISOR_ROOT,
    *,
    pilot_limit: int | None = None,
) -> dict[str, Any]:
    with okf_runtime_operation(root):
        return _run_merge_migration_locked(root, pilot_limit=pilot_limit)


def _run_merge_migration_locked(
    root: Path, *, pilot_limit: int | None
) -> dict[str, Any]:
    started = time.monotonic()
    registry = PageRegistry(root)
    registry.ensure_manifest(write=True)
    discovered = discover_clusters(root)
    components = list(discovered["components"])
    if pilot_limit is not None:
        components = components[: max(0, pilot_limit)]
    dispositions = _load_dispositions(root)
    cluster_results = []
    shadow_plans = [
        (component, prepare_cluster_plan(root, page_keys=component))
        for component in components
    ]
    if pilot_limit is not None:
        write_sealed_json(
            root / "runtime" / "librarian" / "phase10-shadow.json",
            {
                "schema": PILOT_SCHEMA,
                "status": "verified",
                "mode": "all_selected_clusters_shadowed_before_apply",
                "components": [
                    {
                        "members": component,
                        "status": plan.get("status"),
                        "reason": plan.get("reason"),
                        "transaction_id": plan.get("transaction_id"),
                        "coverage": plan.get("verification_receipt"),
                        "input_count": len(plan.get("inputs") or []),
                        "link_rewrite_count": len(plan.get("link_rewrites") or []),
                        "output_sensitivity": (
                            (plan.get("output") or {}).get("sensitivity")
                        ),
                    }
                    for component, plan in shadow_plans
                ],
            },
            backup=True,
        )
    for index, (component, shadow_plan) in enumerate(shadow_plans, start=1):
        # Re-read every source and the registry generation immediately before
        # mutation. The earlier all-shadow pass is evidence, not stale authority.
        plan = prepare_cluster_plan(root, page_keys=component)
        if plan.get("status") != shadow_plan.get("status"):
            raise RuntimeError("cluster disposition changed after shadow preflight")
        if plan.get("status") in {"held", "kept"}:
            result = dict(plan)
            for uid in plan.get("uids") or []:
                dispositions["pages"][uid] = {
                    "disposition": (
                        "explicit-hold" if plan.get("status") == "held" else "keep-both"
                    ),
                    "reason": plan["reason"],
                }
        else:
            paths = [root / str(row["path"]) for row in plan.get("inputs") or []] + [
                root / str(row["path"]) for row in plan.get("link_rewrites") or []
            ]
            restore = _verified_incremental(
                root,
                paths=paths,
                reason=(
                    f"phase10-pilot-cluster-{index}"
                    if pilot_limit is not None
                    else f"phase11-merge-cluster-{index}"
                ),
            )
            result = apply_merge_plan(
                root,
                plan,
                activate=True,
                preimage_ttl_days=7,
            )
            result["restore_id"] = restore["restore_id"]
            if result["status"] != "committed":
                raise RuntimeError(f"merge transaction failed: {result}")
            output_uid = str(plan["output"]["uid"])
            for row in plan["inputs"]:
                uid = str(row["uid"])
                dispositions["pages"][uid] = {
                    "disposition": ("merged" if uid != output_uid else "canonical"),
                    "canonical_uid": output_uid,
                    "transaction_id": plan["transaction_id"],
                }
        cluster_id = hashlib.sha256(
            "\n".join(sorted(component)).encode("utf-8")
        ).hexdigest()
        dispositions["clusters"][cluster_id] = {
            "members": component,
            "result": {
                key: value for key, value in result.items() if key not in {"receipt"}
            },
        }
        cluster_results.append(result)

    current = registry.ensure_manifest(write=True)["registry"]
    if pilot_limit is None:
        for uid, row in current["pages"].items():
            if (
                isinstance(row, Mapping)
                and row.get("status") == "stable"
                and not row.get("canonical_uid")
                and uid not in dispositions["pages"]
            ):
                classification_status = str(row.get("classification_status") or "")
                dispositions["pages"][uid] = {
                    "disposition": (
                        "explicit-hold"
                        if classification_status == "held"
                        else "keep-both"
                    ),
                    "reason": (
                        "classification_hold"
                        if classification_status == "held"
                        else "no_safe_merge_candidate"
                    ),
                }
        dispositions["scope_generation"] = hashlib.sha256(
            "\n".join(sorted(current["pages"])).encode("utf-8")
        ).hexdigest()
    dispositions["generated_at"] = librarian_now_iso()
    write_sealed_json(
        root / "runtime" / "librarian" / "migration-dispositions.json",
        dispositions,
        backup=True,
    )
    receipt = {
        "schema": PILOT_SCHEMA,
        "status": "ok",
        "mode": "pilot" if pilot_limit is not None else "full",
        "candidate_records": len(discovered["records"]),
        "components_selected": len(components),
        "committed": sum(
            result.get("status") == "committed" for result in cluster_results
        ),
        "held": sum(result.get("status") == "held" for result in cluster_results),
        "kept": sum(result.get("status") == "kept" for result in cluster_results),
        "terminal_pages": len(dispositions["pages"]),
        "duration_seconds": round(time.monotonic() - started, 3),
        "results": cluster_results,
    }
    link_index = build_uid_link_index(root, registry=registry, write=True)
    if link_index["unresolved_count"]:
        raise RuntimeError(
            f"merge migration left {link_index['unresolved_count']} unresolved links"
        )
    semantic_rebuild_job_id = None
    if receipt["committed"]:
        from chronovisor.core.search import get_bm25
        from chronovisor.core.semantic_jobs import enqueue_rebuild

        get_bm25().build(force=True)
        semantic_rebuild_job_id = enqueue_rebuild()
    receipt["uid_link_edges"] = int(link_index["edge_count"])
    receipt["uid_link_unresolved"] = int(link_index["unresolved_count"])
    receipt["bm25_rebuilt"] = bool(receipt["committed"])
    receipt["semantic_rebuild_job_id"] = semantic_rebuild_job_id
    write_sealed_json(
        root
        / "runtime"
        / "librarian"
        / ("phase10-pilot.json" if pilot_limit is not None else "phase11-receipt.json"),
        receipt,
        backup=True,
    )
    append_librarian_event(
        root / "runtime" / "librarian" / "events.jsonl",
        {
            "event": "merge_migration",
            "status": "ok",
            "mode": receipt["mode"],
            "committed": receipt["committed"],
            "held": receipt["held"],
        },
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-librarian-merge`` command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("discover", "pilot", "migrate"),
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args(argv)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        with okf_runtime_operation(args.root):
            return _main_locked(args)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75


def _main_locked(args: argparse.Namespace) -> int:
    if args.command == "discover":
        result = discover_clusters(args.root)
    elif args.command == "pilot":
        result = run_merge_migration(args.root, pilot_limit=args.limit)
    else:
        result = run_merge_migration(args.root, pilot_limit=None)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
