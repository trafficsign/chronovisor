"""Entity registry and lightweight alias extraction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chronovisor.core.frontmatter import parse
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.page_mutation import chronovisor_mutation_lock
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    all_pages,
    okf_runtime_operation,
    page_id_from_path,
)
from chronovisor.decision.entity_backfill_contract import (
    DEFAULT_ALIASES as DEFAULT_ALIASES,
)
from chronovisor.decision.entity_backfill_contract import (
    ENTITY_DIR as ENTITY_DIR,
)
from chronovisor.decision.entity_backfill_contract import (
    ENTITY_ID_RE as ENTITY_ID_RE,
)
from chronovisor.decision.entity_backfill_contract import (
    ENTITY_PROPOSAL_VERSION as ENTITY_PROPOSAL_VERSION,
)
from chronovisor.decision.entity_backfill_contract import (
    ENTITY_REGISTRY_FILE as ENTITY_REGISTRY_FILE,
)
from chronovisor.decision.entity_backfill_contract import (
    ENTITY_REVIEW_DIR as ENTITY_REVIEW_DIR,
)
from chronovisor.decision.entity_backfill_contract import (
    REPO_ROOT,
)
from chronovisor.decision.entity_backfill_contract import (
    extract_entities as extract_entities,
)
from chronovisor.decision.entity_backfill_contract import (
    load_registry as load_registry,
)
from chronovisor.decision.entity_backfill_contract import (
    normalize_entity_id as normalize_entity_id,
)
from chronovisor.decision.entity_backfill_contract import (
    patch_entities_frontmatter as patch_entities_frontmatter,
)
from chronovisor.decision.entity_backfill_contract import (
    review_evidence as _review_evidence,
)
from chronovisor.decision.entity_backfill_contract import (
    review_evidence as review_evidence,
)
from chronovisor.decision.entity_backfill_contract import (
    validate_entity_backfill_proposal as validate_entity_backfill_proposal,
)
from chronovisor.ingest.lint import (
    StructuredReviewer,
    build_semantic_mutation_proposal,
    review_semantic_mutation,
    semantic_review_effect_lock,
)


def write_default_registry(path: Path = ENTITY_REGISTRY_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps({"entities": DEFAULT_ALIASES}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return path




def _default_frontier_reviewer(
    prompt: str,
    schema: dict[str, Any],
) -> Mapping[str, Any] | str:
    from chronovisor.decision import routine_review

    return routine_review.run_structured_review(
        prompt,
        schema,
        repo_root=REPO_ROOT,
        execute_patch=False,
        command_env="CHRONOVISOR_ENTITY_BACKFILL_FRONTIER_CMD",
        decision_lane="entity_backfill",
    )




def _apply_entities_cas(path: Path, *, expected_text: str, updated_text: str) -> str:
    """Apply the exact reviewed frontmatter only while its preimage is current."""

    try:
        with chronovisor_mutation_lock():
            if path.read_text(encoding="utf-8") != expected_text:
                return "cas_conflict"
            atomic_write(path, updated_text)
            return (
                "applied"
                if path.read_text(encoding="utf-8") == updated_text
                else "verification_failed"
            )
    except (OSError, UnicodeDecodeError):
        return "write_error"


def _budget_deferred_review() -> dict[str, Any]:
    return {
        "decision": "needs_retry",
        "summary": "entity backfill frontier budget deferred",
        "tests_run": [],
        "commit": None,
        "committed": False,
        "pushed": False,
        "risk": None,
        "notes": None,
        "frontier_failure": {"failure_class": "budget_deferred"},
    }


def backfill_entities(
    *,
    limit: int = 0,
    dry_run: bool = False,
    include_reference: bool = False,
    max_frontier_calls: int = 0,
    reviewer: StructuredReviewer | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    registry = load_registry()
    scanned = 0
    updated = 0
    candidates = 0
    rejected = 0
    retry = 0
    budget_deferred = 0
    cas_conflicts = 0
    invalid_proposals = 0
    frontier_calls = 0
    skipped_reference = 0
    pages: list[str] = []
    pending_pages: list[str] = []
    frontier = reviewer or _default_frontier_reviewer
    reviews_dir = artifact_dir or ENTITY_REVIEW_DIR

    def budgeted_reviewer(
        prompt: str, schema: dict[str, Any]
    ) -> Mapping[str, Any] | str:
        nonlocal frontier_calls
        if max_frontier_calls > 0 and frontier_calls >= max_frontier_calls:
            return _budget_deferred_review()
        frontier_calls += 1
        return frontier(prompt, schema)

    for path in all_pages():
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        meta, _body = parse(text)
        if not include_reference and (
            meta.get("type") == "reference" or path.parent.name == "car-spec"
        ):
            skipped_reference += 1
            continue
        try:
            new_text = patch_entities_frontmatter(text, registry=registry)
        except (TypeError, ValueError):
            continue
        if new_text == text:
            continue
        candidates += 1
        page_id = page_id_from_path(path)
        if dry_run:
            updated += 1
            pages.append(page_id)
        else:
            proposal = build_semantic_mutation_proposal(
                page_id=page_id,
                operation="backfill_entities_frontmatter",
                expected_text=text,
                updated_text=new_text,
                details=_review_evidence(text, new_text, registry=registry),
            )
            if not validate_entity_backfill_proposal(
                proposal,
                expected_text=text,
                updated_text=new_text,
                registry=registry,
            ):
                invalid_proposals += 1
                retry += 1
                pending_pages.append(page_id)
                if limit and candidates >= limit:
                    break
                continue
            try:
                review = review_semantic_mutation(
                    proposal,
                    expected_text=text,
                    updated_text=new_text,
                    reviewer=budgeted_reviewer,
                    artifact_dir=reviews_dir,
                    decision_lane="entity_backfill",
                    injected_reviewer=reviewer is not None,
                )
            except Exception:
                retry += 1
                pending_pages.append(page_id)
                if limit and candidates >= limit:
                    break
                continue
            decision = review.get("decision")
            if decision == "approved" and review.get("valid") is True:
                if not validate_entity_backfill_proposal(
                    proposal,
                    expected_text=text,
                    updated_text=new_text,
                    registry=registry,
                ):
                    invalid_proposals += 1
                    retry += 1
                    pending_pages.append(page_id)
                    continue
                with semantic_review_effect_lock(
                    review,
                    decision_lane="entity_backfill",
                    injected_reviewer=reviewer is not None,
                ) as authorized:
                    apply_status = (
                        _apply_entities_cas(
                            path,
                            expected_text=text,
                            updated_text=new_text,
                        )
                        if authorized
                        else "authority_changed"
                    )
                if apply_status == "applied":
                    updated += 1
                    pages.append(page_id)
                else:
                    retry += 1
                    pending_pages.append(page_id)
                    if apply_status == "cas_conflict":
                        cas_conflicts += 1
            elif decision == "rejected" and review.get("valid") is True:
                with semantic_review_effect_lock(
                    review,
                    decision_lane="entity_backfill",
                    injected_reviewer=reviewer is not None,
                ) as authorized:
                    if authorized:
                        rejected += 1
                    else:
                        retry += 1
                        pending_pages.append(page_id)
            else:
                retry += 1
                pending_pages.append(page_id)
                if review.get("summary") == "entity backfill frontier budget deferred":
                    budget_deferred += 1
        if dry_run and limit and candidates >= limit:
            break
        if not dry_run:
            if review.get("summary") == "entity backfill frontier budget deferred":
                break
            # Cached terminal verdicts consume no call budget and must not
            # pin the scan to the same leading pages forever.
            if limit and frontier_calls >= limit:
                break
    return {
        "status": "ok",
        "dry_run": dry_run,
        "scanned": scanned,
        "skipped_reference": skipped_reference,
        "candidates": candidates,
        "updated": updated,
        "pages": pages,
        "rejected": rejected,
        "retry": retry,
        "budget_deferred": budget_deferred,
        "cas_conflicts": cas_conflicts,
        "invalid_proposals": invalid_proposals,
        "frontier_calls": frontier_calls,
        "pending_pages": pending_pages,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-entities`` command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Maintain Chronovisor entity frontmatter."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init_cmd = sub.add_parser(
        "init", help="Write a default entity registry if missing."
    )
    init_cmd.add_argument("--json", action="store_true")
    backfill = sub.add_parser("backfill", help="Backfill entities on existing pages.")
    backfill.add_argument("--limit", type=int, default=0)
    backfill.add_argument("--max-frontier-calls", type=int, default=0)
    backfill.add_argument("--dry-run", action="store_true")
    backfill.add_argument("--include-reference", action="store_true")
    backfill.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        with okf_runtime_operation(CHRONOVISOR_ROOT):
            return _main_locked(args)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75


def _main_locked(args: argparse.Namespace) -> int:

    if args.command == "init":
        path = write_default_registry()
        payload = {"status": "ok", "path": str(path)}
    else:
        payload = backfill_entities(
            limit=max(0, args.limit),
            dry_run=args.dry_run,
            include_reference=args.include_reference,
            max_frontier_calls=max(0, args.max_frontier_calls),
        )

    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(
            "\t".join(
                f"{key}={value}" for key, value in payload.items() if key != "pages"
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
