"""Deterministic structural normalization for legacy Wiki pages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from chronovisor.core.frontmatter import (
    canonicalize,
    normalize_nested,
    propose_nested_resolution,
)
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.store import CHRONOVISOR_ROOT, PAGES_DIR, page_id_from_path
from chronovisor.ingest.page_mutation import chronovisor_mutation_lock


def _sha256_identity(value: str) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _identity_preflight_for_nested_conflict(
    *,
    page_id: str,
    details: dict[str, Any],
) -> dict[str, Any] | None:
    conflicts = details.get("conflicts")
    permalink = conflicts.get("permalink") if isinstance(conflicts, dict) else None
    if not isinstance(permalink, dict):
        return None
    outer = permalink.get("outer")
    inner = permalink.get("inner")
    if (
        not isinstance(outer, str)
        or not outer
        or not isinstance(inner, str)
        or not inner
        or outer == inner
    ):
        return None
    from chronovisor.decision.decision_lane_prompts import (
        build_identity_preflight_receipt,
    )

    return build_identity_preflight_receipt(
        page_id=page_id,
        field="permalink",
        bindings=[
            {
                "source": "outer_frontmatter",
                "identity": outer,
                "evidence_sha256": _sha256_identity(outer),
            },
            {
                "source": "inner_frontmatter",
                "identity": inner,
                "evidence_sha256": _sha256_identity(inner),
            },
        ],
    )


def normalize_pages(
    *,
    root: Path = PAGES_DIR,
    write: bool = False,
    limit: int = 0,
    max_frontier_calls: int = 3,
    reviewer=None,
) -> dict[str, Any]:
    changed: list[str] = []
    conflicts: list[dict[str, Any]] = []
    resolved_conflicts: list[str] = []
    frontier_calls = 0
    scanned = 0
    for path in sorted(root.rglob("*.md")):
        if limit and scanned >= limit:
            break
        scanned += 1
        original = path.read_text(encoding="utf-8")
        updated, result = normalize_nested(original)
        if result.get("reason") == "conflicting_nested_frontmatter":
            conflicts.append({"path": str(path), **result})
            if write:
                from chronovisor.core.runtime_config import runtime_repo_root
                from chronovisor.ops.lint import (
                    build_semantic_mutation_proposal,
                    review_semantic_mutation,
                    semantic_review_effect_lock,
                )

                proposed, details = propose_nested_resolution(original)
                page_id = page_id_from_path(path)
                identity_preflight = _identity_preflight_for_nested_conflict(
                    page_id=page_id,
                    details=details,
                )
                if identity_preflight is not None:
                    details = {**details, "identity_preflight": identity_preflight}
                proposal = build_semantic_mutation_proposal(
                    page_id=page_id,
                    operation="resolve_nested_frontmatter_conflict",
                    expected_text=original,
                    updated_text=proposed,
                    details=details,
                )
                if reviewer is None:
                    from chronovisor.decision.frontier_review import (
                        run_structured_review,
                    )

                    def actual_frontier(prompt, schema):
                        return run_structured_review(
                            prompt,
                            schema,
                            repo_root=runtime_repo_root(),
                            execute_patch=False,
                            command_env="CHRONOVISOR_PAGE_NORMALIZE_FRONTIER_CMD",
                            decision_lane="page_normalize",
                        )
                else:
                    actual_frontier = reviewer

                def budgeted_frontier(
                    prompt,
                    schema,
                    *,
                    frontier=actual_frontier,
                ):
                    nonlocal frontier_calls
                    if frontier_calls >= max_frontier_calls:
                        return {
                            "decision": "needs_retry",
                            "summary": "page normalization review budget deferred",
                            "tests_run": [],
                            "commit": None,
                            "committed": False,
                            "pushed": False,
                            "risk": None,
                            "notes": None,
                            "frontier_failure": {"failure_class": "budget_deferred"},
                        }
                    frontier_calls += 1
                    return frontier(prompt, schema)

                review = review_semantic_mutation(
                    proposal,
                    expected_text=original,
                    updated_text=proposed,
                    reviewer=budgeted_frontier,
                    artifact_dir=CHRONOVISOR_ROOT / "runtime" / "page-normalize",
                    decision_lane="page_normalize",
                    injected_reviewer=reviewer is not None,
                )
                if review.get("decision") == "approved" and review.get("valid") is True:
                    with semantic_review_effect_lock(
                        review,
                        decision_lane="page_normalize",
                        injected_reviewer=reviewer is not None,
                    ) as authorized:
                        if authorized:
                            with chronovisor_mutation_lock():
                                if path.read_text(encoding="utf-8") == original:
                                    atomic_write(path, proposed)
                                    resolved_conflicts.append(str(path))
            continue
        updated = canonicalize(updated)
        if updated == original:
            continue
        changed.append(str(path))
        if write:
            with chronovisor_mutation_lock():
                current = path.read_text(encoding="utf-8")
                normalized, retry = normalize_nested(current)
                if retry.get("reason") == "conflicting_nested_frontmatter":
                    continue
                normalized = canonicalize(normalized)
                if normalized != current:
                    atomic_write(path, normalized)
    return {
        "status": "ok",
        "scanned": scanned,
        "changed": len(changed),
        "paths": changed,
        "conflicts": conflicts,
        "resolved_conflicts": resolved_conflicts,
        "frontier_calls": frontier_calls,
        "write": write,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-page-normalize`` command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-frontier-calls", type=int, default=3)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            normalize_pages(
                write=args.write,
                limit=max(0, args.limit),
                max_frontier_calls=max(0, args.max_frontier_calls),
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
