"""Frontier-reviewed recall metadata backfill for legacy pages."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chronovisor.core import frontmatter
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.page_mutation import chronovisor_mutation_lock
from chronovisor.core.runtime_config import runtime_repo_root
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    all_pages,
    okf_runtime_operation,
    page_id_from_path,
)
from chronovisor.ingest.ingest import ensure_recall_metadata_frontmatter
from chronovisor.ingest.lint import (
    build_semantic_mutation_proposal,
    review_semantic_mutation,
    semantic_review_effect_lock,
)

REVIEW_DIR = CHRONOVISOR_ROOT / "runtime" / "metadata-backfill"
PROPOSAL_VERSION = 2


def _reviewer(prompt: str, schema: dict[str, Any]) -> Mapping[str, Any] | str:
    from chronovisor.decision.routine_review import run_structured_review

    return run_structured_review(
        prompt,
        schema,
        repo_root=runtime_repo_root(),
        execute_patch=False,
        command_env="CHRONOVISOR_METADATA_BACKFILL_FRONTIER_CMD",
        decision_lane="metadata_backfill",
    )


def _apply(path: Path, expected: str, updated: str) -> str:
    with chronovisor_mutation_lock():
        if path.read_text(encoding="utf-8") != expected:
            return "cas_conflict"
        atomic_write(path, updated)
        return (
            "applied"
            if path.read_text(encoding="utf-8") == updated
            else "verification_failed"
        )


def _stable_local_proposal(original: str, page_id: str) -> str:
    """Reuse one local proposal for an exact preimage across retries/rejections."""
    expected_sha = hashlib.sha256(original.encode("utf-8")).hexdigest()
    key = hashlib.sha256(
        f"{PROPOSAL_VERSION}:{page_id}:{expected_sha}".encode()
    ).hexdigest()
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
            and envelope.get("proposed_sha256")
            == hashlib.sha256(proposed.encode("utf-8")).hexdigest()
        ):
            return proposed
    proposed = ensure_recall_metadata_frontmatter(
        original,
        page_id,
        frontmatter.parse,
        frontmatter.patch,
    )
    envelope = {
        "version": PROPOSAL_VERSION,
        "page_id": page_id,
        "expected_sha256": expected_sha,
        "proposed_sha256": hashlib.sha256(proposed.encode("utf-8")).hexdigest(),
        "proposed_text": proposed,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        artifact,
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return proposed


def backfill_metadata(
    *,
    limit: int = 3,
    max_frontier_calls: int = 3,
    dry_run: bool = False,
    reviewer=None,
) -> dict[str, Any]:
    if not dry_run:
        with okf_runtime_operation(CHRONOVISOR_ROOT):
            return _backfill_metadata_locked(
                limit=limit,
                max_frontier_calls=max_frontier_calls,
                dry_run=False,
                reviewer=reviewer,
            )
    return _backfill_metadata_locked(
        limit=limit,
        max_frontier_calls=max_frontier_calls,
        dry_run=True,
        reviewer=reviewer,
    )


def _backfill_metadata_locked(
    *, limit: int, max_frontier_calls: int, dry_run: bool, reviewer=None
) -> dict[str, Any]:
    candidates = updated_count = rejected = retry = calls = 0
    pages: list[str] = []
    frontier = reviewer or _reviewer

    def budgeted_reviewer(
        prompt: str, schema: dict[str, Any]
    ) -> Mapping[str, Any] | str:
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
                "frontier_failure": {"failure_class": "budget_deferred"},
            }
        calls += 1
        return frontier(prompt, schema)

    for path in all_pages():
        try:
            original = path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, _body = frontmatter.parse(original)
        if meta.get("type") == "reference":
            continue
        summary_missing = (
            not isinstance(meta.get("summary"), str)
            or not str(meta.get("summary") or "").strip()
        )
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
                "proposal_generator_version": PROPOSAL_VERSION,
                "summary_missing": summary_missing,
                "questions_missing": questions_missing,
                "generated_frontmatter": frontmatter.review_value(
                    frontmatter.parse(proposed)[0]
                ),
            },
        )
        review = review_semantic_mutation(
            proposal,
            expected_text=original,
            updated_text=proposed,
            reviewer=budgeted_reviewer,
            artifact_dir=REVIEW_DIR,
            decision_lane="metadata_backfill",
            injected_reviewer=reviewer is not None,
        )
        if review.get("decision") == "approved" and review.get("valid") is True:
            with semantic_review_effect_lock(
                review,
                decision_lane="metadata_backfill",
                injected_reviewer=reviewer is not None,
            ) as authorized:
                status = _apply(path, original, proposed) if authorized else "retry"
            if status == "applied":
                updated_count += 1
                pages.append(page_id)
            else:
                retry += 1
        elif review.get("decision") == "rejected" and review.get("valid") is True:
            with semantic_review_effect_lock(
                review,
                decision_lane="metadata_backfill",
                injected_reviewer=reviewer is not None,
            ) as authorized:
                if authorized:
                    rejected += 1
                else:
                    retry += 1
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
    """Run the ``chronovisor-metadata-backfill`` command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-frontier-calls", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        payload = backfill_metadata(
            limit=max(0, args.limit),
            max_frontier_calls=max(0, args.max_frontier_calls),
            dry_run=args.dry_run,
        )
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
