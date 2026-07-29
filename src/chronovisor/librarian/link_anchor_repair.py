"""Bounded, restore-backed repairs for verified stale wikilink anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.link_fix import (
    WIKI_LINK_RE,
    atomic_write,
    position_in_spans,
    protected_spans,
)
from chronovisor.ops.migration_snapshot import (
    create_incremental_restore_point,
    restore_drill,
)
from chronovisor.ingest.page_mutation import chronovisor_mutation_lock
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.core.store import CHRONOVISOR_ROOT
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
}


def _rewrite(
    text: str,
    repairs: Mapping[tuple[str, str], str],
) -> tuple[str, list[dict[str, str]]]:
    spans = protected_spans(text)
    applied: list[dict[str, str]] = []

    def replace(match: Any) -> str:
        if position_in_spans(match.start(), spans):
            return match.group(0)
        value = str(match.group(1))
        target_and_anchor, separator, label = value.partition("|")
        target, marker, anchor = target_and_anchor.partition("#")
        if not marker:
            return match.group(0)
        replacement = repairs.get((target.strip(), anchor.strip()))
        if replacement is None:
            return match.group(0)
        new_value = f"{target}#{replacement}"
        if separator:
            new_value += f"|{label}"
        applied.append(
            {
                "target": target.strip(),
                "old_anchor": anchor.strip(),
                "new_anchor": replacement,
            }
        )
        return f"[[{new_value}]]"

    return WIKI_LINK_RE.sub(replace, text), applied


def repair_known_anchors(
    root: Path = CHRONOVISOR_ROOT,
    *,
    repairs: Mapping[tuple[str, str], str] = DEFAULT_REPAIRS,
) -> dict[str, Any]:
    """Apply only the reviewed anchor map with snapshot, CAS and postflight."""

    registry = PageRegistry(root)
    state = registry.ensure_manifest(write=True)["registry"]
    before_index = build_uid_link_index(root, registry=registry, write=False)
    page_preimages: dict[Path, bytes] = {}
    updates: dict[Path, tuple[str, list[dict[str, str]]]] = {}
    for row in state["pages"].values():
        if not isinstance(row, Mapping) or row.get("status") == "superseded":
            continue
        path = root / str(row.get("path") or "")
        if not path.is_file():
            continue
        original = path.read_bytes()
        updated, applied = _rewrite(original.decode("utf-8"), repairs)
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
        with chronovisor_mutation_lock():
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
                if (str(row.get("target")), str(row.get("anchor"))) in repairs
            ]
            if remaining_known:
                raise RuntimeError(
                    f"reviewed anchor repairs remain unresolved: {remaining_known[:3]}"
                )
    except Exception:
        with chronovisor_mutation_lock():
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
    result = repair_known_anchors(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["after_unresolved"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
