"""Bounded, restore-backed repairs for verified stale wikilink anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from chronovisor.core.canonical_document import (
    Namespace,
    ResolvedMarkdownLink,
    rewrite_internal_markdown_links,
)
from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.link_fix import (
    atomic_write,
)
from chronovisor.core.migration_snapshot import (
    create_incremental_restore_point,
    restore_drill,
)
from chronovisor.core.page_mutation import chronovisor_mutation_lock
from chronovisor.core.store import CHRONOVISOR_ROOT, okf_runtime_operation
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.ingest.uid_link_index import build_uid_link_index

SCHEMA = "chronovisor.link-anchor-repair.v1"

# Every entry maps a known stale anchor to an existing exact Markdown heading.
# This is intentionally closed-world: newly discovered debt must be reviewed and
# added explicitly rather than guessed by fuzzy matching.
DEFAULT_REPAIRS: dict[tuple[str, str], str] = {
    (
        "chronovisor-dashboard-ui-design",
        "Dashboard-v3-グラフ改善",
    ): "Dashboard v3 グラフ改善（2026-05-31）",
    (
        "claude-model-lifecycle",
        "recommended-targets",
    ): "Recommended Destination Models",
    (
        "managed-agents-core-concepts",
        "環境",
    ): "Console URL の構築（非デフォルトワークスペース）",
    (
        "managed-agents-core-concepts",
        "セッショントラッキング",
    ): "セッションライフサイクル",
    ("khi-interview-recap-2026-07-23", "次のステップ"): "次のアクション",
    ("mazda-selection-consistency-evidence", "1"): "核心事実",
    ("mazda-selection-consistency-evidence", "2"): "面接での使い方",
    ("mazda-selection-consistency-evidence", "3"): "関連する矛盾の解決",
    ("mazda-selection-consistency-evidence", "4"): "注意事項",
    (
        "2026-fifa-world-cup-japan-group-dynamics",
        "2-1-主要選手の状況",
    ): "2.1 Key Player Status",
    (
        "2026-fifa-world-cup-japan-group-dynamics",
        "2-2-戦術的帰結",
    ): "2.2 Tactical Consequences",
    (
        "2026-fifa-world-cup-japan-group-dynamics",
        "3-戦略的推奨事項",
    ): "3. Strategic Recommendations",
    (
        "chronovisor-classification-librarian-plan",
        "gold-corpus-creation",
    ): "7. Gold Corpus Creation",
    (
        "chronovisor-classification-librarian-plan",
        "critical-action-items-before-execution",
    ): "10. Critical Action Items Before Execution",
    (
        "decision-router-cloud-provider-neutrality",
        "quorum・veto・schema・proof",
    ): "Immutable Core (変更不可)",
    (
        "decision-router-cloud-provider-neutrality",
        "不変の制約",
    ): "Immutable Core (変更不可)",
    (
        "local-ai-hardware-strategy-2026",
        "section_22",
    ): "22. M4 Max 128GB の「物理的再現不可能性」と供給逼迫の構造的終着点",
    (
        "child-record-segment-structure",
        "Phase Field Details",
    ): "Segment Phase Field Flexibility",
    (
        "child-record-policy-v2-production-instance",
        "eleventh-instance-boundary-hardening",
    ): "Campaign Boundary Hardening Instance (2026-08-07)",
    (
        "child-record-policy-v2-production-instance",
        "twelfth-instance-v2-only-migration",
    ): "Campaign v2-Only Migration Instance (2026-08-07)",
    (
        "child-record-policy-v2-production-instance",
        "sixteenth-instance-parallelization-strategy",
    ): "Campaign Parallelization Strategy Instance (2026-08-07)",
    (
        "child-record-policy-v2-production-instance",
        "fourth-instance-final-review",
    ): "Campaign Final Review Instance (2026-08-06)",
    (
        "child-record-policy-v2-production-instance",
        "fifteenth-instance-p1-analysis",
    ): "Campaign P1 Analysis Instance (2026-08-07)",
    (
        "child-record-policy-v2-production-instance",
        "ninth-instance-frozen-dependency-hardening",
    ): "Campaign Frozen Dependency Reference Hardening Instance (2026-08-06)",
}


def _rewrite(
    text: str,
    repairs: Mapping[tuple[str, str], str],
    *,
    registry: PageRegistry,
    registry_state: Mapping[str, Any],
    source_namespace: Namespace,
    source_path: str,
) -> tuple[str, list[dict[str, str]]]:
    applied: list[dict[str, str]] = []

    def replace(
        link: ResolvedMarkdownLink,
        _label: str,
    ) -> ResolvedMarkdownLink | None:
        if not link.fragment:
            return None
        target = (
            f"{link.namespace}/"
            f"{PurePosixPath(link.path).with_suffix('').as_posix()}"
        )
        resolved = registry.resolve_from_state(registry_state, target)
        if not isinstance(resolved, Mapping):
            return None
        target_uid = str(resolved.get("uid") or "")
        replacement = repairs.get((target_uid, link.fragment))
        if replacement is None:
            return None
        applied.append(
            {
                "target": target_uid,
                "old_anchor": link.fragment,
                "new_anchor": replacement,
            }
        )
        return ResolvedMarkdownLink(
            namespace=link.namespace,
            path=link.path,
            fragment=replacement,
        )

    rewritten, _count = rewrite_internal_markdown_links(
        text,
        source_namespace=source_namespace,
        source_path=source_path,
        rewrite=replace,
    )
    return rewritten, applied


def repair_known_anchors(
    root: Path = CHRONOVISOR_ROOT,
    *,
    repairs: Mapping[tuple[str, str], str] = DEFAULT_REPAIRS,
) -> dict[str, Any]:
    """Apply only the reviewed anchor map with snapshot, CAS and postflight."""

    registry = PageRegistry(root)
    state = registry.ensure_manifest(write=True)["registry"]
    repairs_by_uid: dict[tuple[str, str], str] = {}
    repair_targets: set[tuple[str, str]] = set()
    for (target, anchor), replacement in repairs.items():
        resolved = registry.resolve_from_state(state, target)
        if not isinstance(resolved, Mapping):
            continue
        repairs_by_uid[(str(resolved["uid"]), anchor)] = replacement
        repair_targets.add(
            (
                PurePosixPath(str(resolved["path"])).with_suffix("").as_posix(),
                anchor,
            )
        )
    before_index = build_uid_link_index(root, registry=registry, write=False)
    page_preimages: dict[Path, bytes] = {}
    updates: dict[Path, tuple[str, list[dict[str, str]]]] = {}
    for row in registry.stable_pages(state).values():
        path = root / str(row.get("path") or "")
        original = path.read_bytes()
        relative = path.relative_to(root)
        namespace: Namespace = (
            "system" if relative.parts[0] == "system" else "pages"
        )
        source_path = PurePosixPath(*relative.parts[1:]).as_posix()
        updated, applied = _rewrite(
            original.decode("utf-8"),
            repairs_by_uid,
            registry=registry,
            registry_state=state,
            source_namespace=namespace,
            source_path=source_path,
        )
        if applied and updated.encode("utf-8") != original:
            page_preimages[path] = original
            updates[path] = (updated, applied)
    if not updates:
        after_index = build_uid_link_index(root, registry=registry, write=True)
        return {
            "schema": SCHEMA,
            "status": "already_current",
            "before_unresolved": int(before_index["unresolved_count"]),
            "after_unresolved": int(after_index["unresolved_count"]),
            "pages_changed": 0,
            "links_changed": 0,
        }

    restore = create_incremental_restore_point(
        root,
        paths=updates,
        reason="phase3-reviewed-stale-anchor-repair",
        ttl_days=7,
    )
    with tempfile.TemporaryDirectory(prefix="chronovisor-anchor-drill-") as temporary:
        drill = restore_drill(Path(restore["path"]), Path(temporary) / "restored")
    if drill["status"] != "verified":
        raise RuntimeError("anchor repair restore drill failed")

    registry_path = root / "runtime" / "librarian" / "page-registry.json"
    registry_preimage = registry_path.read_bytes() if registry_path.is_file() else None
    try:
        with chronovisor_mutation_lock(pages_dir=root / "pages"):
            for path, original in page_preimages.items():
                if path.read_bytes() != original:
                    raise RuntimeError(f"stale anchor repair CAS: {path}")
            for path, (updated, _applied) in updates.items():
                atomic_write(path, updated)
            registry.ensure_manifest(write=True)
            after_index = build_uid_link_index(root, registry=registry, write=True)
            remaining_known = [
                row
                for row in after_index["unresolved"]
                if (str(row.get("target")), str(row.get("anchor")))
                in repair_targets
            ]
            if remaining_known:
                raise RuntimeError(
                    f"reviewed anchor repairs remain unresolved: {remaining_known[:3]}"
                )
    except Exception:
        with chronovisor_mutation_lock(pages_dir=root / "pages"):
            for path, original in page_preimages.items():
                atomic_write(path, original.decode("utf-8"))
            if registry_preimage is not None:
                atomic_write(registry_path, registry_preimage.decode("utf-8"))
            build_uid_link_index(root, registry=PageRegistry(root), write=True)
        raise

    details = [
        {"path": str(path.relative_to(root)), **row}
        for path, (_updated, rows) in updates.items()
        for row in rows
    ]
    receipt = {
        "schema": SCHEMA,
        "status": "committed",
        "before_unresolved": int(before_index["unresolved_count"]),
        "after_unresolved": int(after_index["unresolved_count"]),
        "pages_changed": len(updates),
        "links_changed": len(details),
        "restore_id": restore["restore_id"],
        "restore_verified": True,
        "details": details,
        "postimage_sha256": "sha256:"
        + hashlib.sha256(
            json.dumps(details, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
    }
    write_sealed_json(
        root / "runtime" / "librarian" / "phase3-anchor-repair.json",
        receipt,
        backup=True,
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-link-anchor-repair`` command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    args = parser.parse_args(argv)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        with okf_runtime_operation(args.root):
            result = repair_known_anchors(args.root)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["after_unresolved"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
