"""Deterministic structural normalization for legacy Wiki pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_wiki_mcp.frontmatter import normalize_nested, propose_nested_resolution
from llm_wiki_mcp.link_fix import atomic_write
from llm_wiki_mcp.page_mutation import wiki_mutation_lock
from llm_wiki_mcp.wiki import PAGES_DIR, page_id_from_path, WIKI_ROOT


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
            if write and frontier_calls < max_frontier_calls:
                from llm_wiki_mcp.lint import build_semantic_mutation_proposal, review_semantic_mutation
                from llm_wiki_mcp.runtime_config import runtime_repo_root

                proposed, details = propose_nested_resolution(original)
                proposal = build_semantic_mutation_proposal(
                    page_id=page_id_from_path(path),
                    operation="resolve_nested_frontmatter_conflict",
                    expected_text=original,
                    updated_text=proposed,
                    details=details,
                )
                if reviewer is None:
                    from llm_wiki_mcp.frontier_review import run_structured_review

                    frontier = lambda prompt, schema: run_structured_review(
                        prompt, schema, repo_root=runtime_repo_root(), execute_patch=False,
                        command_env="LLM_WIKI_PAGE_NORMALIZE_FRONTIER_CMD",
                        decision_lane="page_normalize",
                    )
                else:
                    frontier = reviewer
                review = review_semantic_mutation(
                    proposal,
                    expected_text=original,
                    reviewer=frontier,
                    artifact_dir=WIKI_ROOT / "runtime" / "page-normalize",
                )
                frontier_calls += 1
                if review.get("decision") == "approved" and review.get("valid") is True:
                    with wiki_mutation_lock(path):
                        if path.read_text(encoding="utf-8") == original:
                            atomic_write(path, proposed)
                            resolved_conflicts.append(str(path))
            continue
        if updated == original:
            continue
        changed.append(str(path))
        if write:
            with wiki_mutation_lock(path):
                current = path.read_text(encoding="utf-8")
                normalized, retry = normalize_nested(current)
                if retry.get("changed"):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-frontier-calls", type=int, default=3)
    args = parser.parse_args(argv)
    print(json.dumps(normalize_pages(
        write=args.write,
        limit=max(0, args.limit),
        max_frontier_calls=max(0, args.max_frontier_calls),
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
