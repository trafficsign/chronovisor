"""Entity registry and lightweight alias extraction."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llm_wiki_mcp.frontmatter import parse, patch
from llm_wiki_mcp.lint import (
    StructuredReviewer,
    _review_packet_error,
    build_semantic_mutation_proposal,
    review_semantic_mutation,
    semantic_review_effect_lock,
)
from llm_wiki_mcp.link_fix import atomic_write
from llm_wiki_mcp.page_mutation import wiki_mutation_lock
from llm_wiki_mcp.runtime_config import runtime_repo_root
from llm_wiki_mcp.wiki import WIKI_ROOT, all_pages, page_id_from_path

ENTITY_DIR = WIKI_ROOT / "entities"
ENTITY_REGISTRY_FILE = ENTITY_DIR / "registry.json"
ENTITY_REVIEW_DIR = WIKI_ROOT / "runtime" / "entity-backfill"
REPO_ROOT = runtime_repo_root()
ENTITY_PROPOSAL_VERSION = 2

DEFAULT_ALIASES: dict[str, list[str]] = {
    "llm-wiki": ["LLM Wiki", "llm wiki", "LLMウィキ", "ウィキ"],
    "codex": ["Codex", "コードエクス", "コーデックス"],
    "claude-code": ["Claude Code", "クラウドコード"],
    "ollama": ["Ollama"],
    "qwen": ["Qwen", "クエン"],
    "gemma": ["Gemma", "ジェンマ"],
    "mhi": ["MHI", "三菱重工", "三菱重工業"],
    "khi": ["KHI", "川崎重工", "川重"],
    "mazda": ["Mazda", "マツダ"],
}

ENTITY_ID_RE = re.compile(r"[^a-z0-9]+")


def normalize_entity_id(value: str) -> str:
    normalized = value.strip().casefold()
    normalized = ENTITY_ID_RE.sub("-", normalized).strip("-")
    return normalized[:80]


def load_registry(path: Path = ENTITY_REGISTRY_FILE) -> dict[str, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_ALIASES)
    aliases: dict[str, list[str]] = dict(DEFAULT_ALIASES)
    if isinstance(data, dict):
        raw_entities = data.get("entities", data)
        if isinstance(raw_entities, dict):
            for key, value in raw_entities.items():
                entity_id = normalize_entity_id(str(key))
                if not entity_id:
                    continue
                values: list[str] = []
                if isinstance(value, list):
                    values = [v for v in value if isinstance(v, str) and v.strip()]
                elif isinstance(value, dict):
                    raw_aliases = value.get("aliases")
                    if isinstance(raw_aliases, list):
                        values = [
                            v for v in raw_aliases if isinstance(v, str) and v.strip()
                        ]
                    label = value.get("label")
                    if isinstance(label, str) and label.strip():
                        values.insert(0, label)
                aliases[entity_id] = list(dict.fromkeys([entity_id, *values]))
    return aliases


def write_default_registry(path: Path = ENTITY_REGISTRY_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps({"entities": DEFAULT_ALIASES}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return path


def extract_entities(
    text: str, *, registry: dict[str, list[str]] | None = None
) -> list[str]:
    registry = registry or load_registry()
    haystack = text.casefold()
    found: list[str] = []
    for entity_id, aliases in registry.items():
        for alias in aliases:
            if alias and alias.casefold() in haystack:
                found.append(entity_id)
                break
    return list(dict.fromkeys(found))[:20]


def patch_entities_frontmatter(
    text: str,
    *,
    registry: dict[str, list[str]] | None = None,
) -> str:
    meta, body = parse(text)
    title = meta.get("title")
    current = meta.get("entities")
    existing = current if isinstance(current, list) else []
    extracted = extract_entities(
        "\n".join([title if isinstance(title, str) else "", body]),
        registry=registry,
    )
    merged = list(dict.fromkeys([*existing, *extracted]))
    if not merged or merged == existing:
        return text
    return patch(text, {"entities": merged})


def _default_frontier_reviewer(
    prompt: str,
    schema: dict[str, Any],
) -> Mapping[str, Any] | str:
    from llm_wiki_mcp import frontier_review

    return frontier_review.run_structured_review(
        prompt,
        schema,
        repo_root=REPO_ROOT,
        execute_patch=False,
        command_env="LLM_WIKI_ENTITY_BACKFILL_FRONTIER_CMD",
        decision_lane="entity_backfill",
    )


def _review_evidence(
    text: str,
    new_text: str,
    *,
    registry: Mapping[str, list[str]],
) -> dict[str, Any]:
    meta, body = parse(text)
    new_meta, _new_body = parse(new_text)
    title = meta.get("title") if isinstance(meta.get("title"), str) else ""
    haystack = f"{title}\n{body}".casefold()
    before = meta.get("entities") if isinstance(meta.get("entities"), list) else []
    after = (
        new_meta.get("entities") if isinstance(new_meta.get("entities"), list) else []
    )
    before_ids = [item for item in before if isinstance(item, str)]
    after_ids = [item for item in after if isinstance(item, str)]
    added = [item for item in after_ids if item not in before_ids]
    alias_evidence = []
    for entity_id in added:
        aliases = [
            alias for alias in registry.get(entity_id, []) if isinstance(alias, str)
        ]
        alias_evidence.append(
            {
                "entity_id": entity_id,
                "matched_aliases": [
                    alias for alias in aliases if alias and alias.casefold() in haystack
                ],
                "aliases_considered": aliases,
            }
        )
    registry_payload = json.dumps(
        {key: list(value) for key, value in registry.items()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "proposal_generator_version": ENTITY_PROPOSAL_VERSION,
        "existing_entities": before_ids,
        "proposed_entities": after_ids,
        "added_entities": added,
        "alias_evidence": alias_evidence,
        "registry_sha256": hashlib.sha256(registry_payload.encode("utf-8")).hexdigest(),
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_entity_backfill_proposal(
    proposal: Mapping[str, Any],
    *,
    expected_text: str | None = None,
    updated_text: str | None = None,
    registry: Mapping[str, list[str]] | None = None,
) -> bool:
    """Prove that an entity proposal is an alias-backed frontmatter-only edit."""

    if (
        proposal.get("schema_version") != 1
        or proposal.get("kind") != "lint_safe_fix_proposal"
        or proposal.get("operation") != "backfill_entities_frontmatter"
        or proposal.get("unified_diff_truncated") is not False
    ):
        return False
    hash_fields = (
        "expected_sha256",
        "updated_sha256",
        "unified_diff_sha256",
        "full_unified_diff_sha256",
    )
    if any(
        not isinstance(proposal.get(name), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(proposal.get(name))) is None
        for name in hash_fields
    ):
        return False
    review_packet = proposal.get("review_packet")
    if not isinstance(review_packet, Mapping):
        return False
    mode = review_packet.get("mode")
    unified_diff = proposal.get("unified_diff")
    if mode == "full":
        if not isinstance(unified_diff, str) or not unified_diff:
            return False
        if _sha256_text(unified_diff) != proposal.get(
            "unified_diff_sha256"
        ) or proposal.get("unified_diff_sha256") != proposal.get(
            "full_unified_diff_sha256"
        ):
            return False
    elif mode == "changed_spans":
        coverage = review_packet.get("coverage")
        if (
            unified_diff is not None
            or proposal.get("unified_diff_repacket") is not True
            or not isinstance(coverage, Mapping)
            or coverage.get("all_changed_spans_rendered") is not True
        ):
            return False
    else:
        return False

    details = proposal.get("details")
    if (
        not isinstance(details, Mapping)
        or details.get("proposal_generator_version") != ENTITY_PROPOSAL_VERSION
    ):
        return False
    existing = details.get("existing_entities")
    proposed = details.get("proposed_entities")
    added = details.get("added_entities")
    evidence = details.get("alias_evidence")
    if not all(
        isinstance(value, list) and all(isinstance(item, str) for item in value)
        for value in (existing, proposed, added)
    ):
        return False
    assert (
        isinstance(existing, list)
        and isinstance(proposed, list)
        and isinstance(added, list)
    )
    if (
        not added
        or len(existing) != len(set(existing))
        or any(normalize_entity_id(item) != item for item in existing)
        or len(proposed) != len(set(proposed))
        or any(normalize_entity_id(item) != item for item in proposed)
        or any(item not in proposed for item in existing)
        or added != [item for item in proposed if item not in existing]
        or not isinstance(evidence, list)
        or len(evidence) != len(added)
    ):
        return False
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for row in evidence:
        if not isinstance(row, Mapping) or not isinstance(row.get("entity_id"), str):
            return False
        entity_id = str(row["entity_id"])
        matched = row.get("matched_aliases")
        considered = row.get("aliases_considered")
        if (
            entity_id in evidence_by_id
            or entity_id not in added
            or not isinstance(matched, list)
            or not matched
            or not all(isinstance(alias, str) and alias for alias in matched)
            or not isinstance(considered, list)
            or not all(isinstance(alias, str) and alias for alias in considered)
            or any(alias not in considered for alias in matched)
        ):
            return False
        evidence_by_id[entity_id] = row
    if set(evidence_by_id) != set(added):
        return False
    registry_sha = details.get("registry_sha256")
    if (
        not isinstance(registry_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", registry_sha) is None
    ):
        return False

    if (expected_text is None) != (updated_text is None):
        return False
    if expected_text is not None and updated_text is not None:
        if _sha256_text(expected_text) != proposal.get(
            "expected_sha256"
        ) or _sha256_text(updated_text) != proposal.get("updated_sha256"):
            return False
        exact_diff = "".join(
            difflib.unified_diff(
                expected_text.splitlines(keepends=True),
                updated_text.splitlines(keepends=True),
                fromfile=f"{proposal.get('page_id')}:before",
                tofile=f"{proposal.get('page_id')}:after",
                n=5,
            )
        )
        if (
            not exact_diff
            or _sha256_text(exact_diff) != proposal.get("unified_diff_sha256")
            or proposal.get("unified_diff_sha256")
            != proposal.get("full_unified_diff_sha256")
        ):
            return False
        changed = []
        for line in exact_diff.splitlines():
            if line.startswith(("---", "+++", "@@")):
                continue
            if line.startswith(("-", "+")):
                changed.append(line[1:].strip())
        if not changed or any(not line.startswith("entities:") for line in changed):
            return False
        before_meta, before_body = parse(expected_text)
        after_meta, after_body = parse(updated_text)
        if before_body != after_body:
            return False
        before_without = {
            key: value for key, value in before_meta.items() if key != "entities"
        }
        after_without = {
            key: value for key, value in after_meta.items() if key != "entities"
        }
        if before_without != after_without:
            return False
        expected_details = (
            _review_evidence(expected_text, updated_text, registry=registry)
            if registry is not None
            else None
        )
        observed_details = dict(details)
        observed_details.pop("review_receipt", None)
        if (
            expected_details is None
            or observed_details != expected_details
            or _review_packet_error(
                proposal,
                expected_text=expected_text,
                updated_text=updated_text,
            )
            is not None
        ):
            return False
    return True


def _apply_entities_cas(path: Path, *, expected_text: str, updated_text: str) -> str:
    """Apply the exact reviewed frontmatter only while its preimage is current."""

    try:
        with wiki_mutation_lock():
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
    parser = argparse.ArgumentParser(
        description="Maintain LLM Wiki entity frontmatter."
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
