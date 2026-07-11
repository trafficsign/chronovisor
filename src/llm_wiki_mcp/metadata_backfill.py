"""Frontier-reviewed recall metadata backfill for legacy pages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from llm_wiki_mcp.frontmatter import parse, patch
from llm_wiki_mcp.ingest import _ensure_recall_metadata_frontmatter
from llm_wiki_mcp.link_fix import atomic_write
from llm_wiki_mcp.lint import build_semantic_mutation_proposal, review_semantic_mutation
from llm_wiki_mcp.page_mutation import wiki_mutation_lock
from llm_wiki_mcp.runtime_config import runtime_repo_root
from llm_wiki_mcp.wiki import WIKI_ROOT, all_pages, page_id_from_path

REVIEW_DIR = WIKI_ROOT / "runtime" / "metadata-backfill"
PROPOSAL_VERSION = 1


def _reviewer(prompt: str, schema: dict[str, Any]) -> Mapping[str, Any] | str:
    from llm_wiki_mcp.frontier_review import run_structured_review

    return run_structured_review(
        prompt,
        schema,
        repo_root=runtime_repo_root(),
        execute_patch=False,
        command_env="LLM_WIKI_METADATA_BACKFILL_FRONTIER_CMD",
        decision_lane="metadata_backfill",
    )


def _apply(path: Path, expected: str, updated: str) -> str:
    with wiki_mutation_lock():
        if path.read_text(encoding="utf-8") != expected:
            return "cas_conflict"
        atomic_write(path, updated)
        return "applied" if path.read_text(encoding="utf-8") == updated else "verification_failed"


def _stable_local_proposal(original: str, page_id: str) -> str:
    """Reuse one local proposal for an exact preimage across retries/rejections."""
    expected_sha = hashlib.sha256(original.encode("utf-8")).hexdigest()
    key = hashlib.sha256(f"{PROPOSAL_VERSION}:{page_id}:{expected_sha}".encode("utf-8")).hexdigest()
    artifact = REVIEW_DIR / "local-proposals" / f"{key}.json"
    try:
        envelope = json.loads(artifact.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        envelope = None
    if isinstance(envelope, dict):
        proposed = envelope.get("proposed_text")
        if (
            envelope.get("version") == PROPOSAL_VERSION
            and envelope.get("page_id") == page_id
            and envelope.get("expected_sha256") == expected_sha
            and isinstance(proposed, str)
            and envelope.get("proposed_sha256") == hashlib.sha256(proposed.encode("utf-8")).hexdigest()
        ):
            return proposed
    proposed = _ensure_recall_metadata_frontmatter(original, page_id, parse, patch)
    envelope = {
        "version": PROPOSAL_VERSION,
        "page_id": page_id,
        "expected_sha256": expected_sha,
        "proposed_sha256": hashlib.sha256(proposed.encode("utf-8")).hexdigest(),
        "proposed_text": proposed,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(artifact, json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return proposed


def backfill_metadata(
    *,
    limit: int = 3,
    max_frontier_calls: int = 3,
    dry_run: bool = False,
    reviewer=None,
) -> dict[str, Any]:
    candidates = updated_count = rejected = retry = calls = 0
    pages: list[str] = []
    frontier = reviewer or _reviewer

    def budgeted_reviewer(prompt: str, schema: dict[str, Any]) -> Mapping[str, Any] | str:
        nonlocal calls
        if max_frontier_calls > 0 and calls >= max_frontier_calls:
            return {
                "decision": "needs_retry",
                "summary": "metadata backfill frontier budget deferred",
                "tests_run": [],
                "commit": None,
                "committed": False,
                "pushed": False,
                "risk": None,
                "notes": None,
            }
        calls += 1
        return frontier(prompt, schema)

    for path in all_pages():
        try:
            original = path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, _body = parse(original)
        if meta.get("type") == "reference":
            continue
        summary_missing = not isinstance(meta.get("summary"), str) or not str(meta.get("summary") or "").strip()
        questions = meta.get("recall_questions")
        questions_missing = not isinstance(questions, list) or not questions
        if not (summary_missing or questions_missing):
            continue
        page_id = page_id_from_path(path)
        if dry_run:
            candidates += 1
            pages.append(page_id)
            updated_count += 1
            if limit and candidates >= limit:
                break
            continue
        proposed = _stable_local_proposal(original, page_id)
        if proposed == original:
            continue
        candidates += 1
        proposal = build_semantic_mutation_proposal(
            page_id=page_id,
            operation="backfill_recall_metadata",
            expected_text=original,
            updated_text=proposed,
            details={
                "summary_missing": summary_missing,
                "questions_missing": questions_missing,
                "generated_frontmatter": parse(proposed)[0],
            },
        )
        review = review_semantic_mutation(
            proposal,
            expected_text=original,
            reviewer=budgeted_reviewer,
            artifact_dir=REVIEW_DIR,
        )
        if review.get("decision") == "approved" and review.get("valid") is True:
            status = _apply(path, original, proposed)
            if status == "applied":
                updated_count += 1
                pages.append(page_id)
            else:
                retry += 1
        elif review.get("decision") == "rejected" and review.get("valid") is True:
            rejected += 1
        else:
            retry += 1
        if review.get("summary") == "metadata backfill frontier budget deferred":
            break
        if limit and calls >= limit:
            break
    return {
        "status": "ok",
        "candidates": candidates,
        "updated": updated_count,
        "rejected": rejected,
        "retry": retry,
        "frontier_calls": calls,
        "pages": pages,
        "dry_run": dry_run,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-frontier-calls", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(backfill_metadata(
        limit=max(0, args.limit),
        max_frontier_calls=max(0, args.max_frontier_calls),
        dry_run=args.dry_run,
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
