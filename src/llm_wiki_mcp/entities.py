"""Entity registry and lightweight alias extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llm_wiki_mcp.frontmatter import parse, patch
from llm_wiki_mcp.lint import (
    SAFE_FIX_REVIEW_SCHEMA,
    StructuredReviewer,
    build_semantic_mutation_proposal,
    review_semantic_mutation,
)
from llm_wiki_mcp.link_fix import atomic_write
from llm_wiki_mcp.page_mutation import wiki_mutation_lock
from llm_wiki_mcp.runtime_config import runtime_repo_root
from llm_wiki_mcp.wiki import WIKI_ROOT, all_pages, page_id_from_path

ENTITY_DIR = WIKI_ROOT / "entities"
ENTITY_REGISTRY_FILE = ENTITY_DIR / "registry.json"
ENTITY_REVIEW_DIR = WIKI_ROOT / "runtime" / "entity-backfill"
REPO_ROOT = runtime_repo_root()

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
                            v for v in raw_aliases
                            if isinstance(v, str) and v.strip()
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
            json.dumps({"entities": DEFAULT_ALIASES}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return path


def extract_entities(text: str, *, registry: dict[str, list[str]] | None = None) -> list[str]:
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
    after = new_meta.get("entities") if isinstance(new_meta.get("entities"), list) else []
    before_ids = [item for item in before if isinstance(item, str)]
    after_ids = [item for item in after if isinstance(item, str)]
    added = [item for item in after_ids if item not in before_ids]
    alias_evidence = []
    for entity_id in added:
        aliases = [alias for alias in registry.get(entity_id, []) if isinstance(alias, str)]
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
        "existing_entities": before_ids,
        "proposed_entities": after_ids,
        "added_entities": added,
        "alias_evidence": alias_evidence,
        "registry_sha256": hashlib.sha256(registry_payload.encode("utf-8")).hexdigest(),
    }


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
    frontier_calls = 0
    skipped_reference = 0
    pages: list[str] = []
    pending_pages: list[str] = []
    frontier = reviewer or _default_frontier_reviewer
    reviews_dir = artifact_dir or ENTITY_REVIEW_DIR

    def budgeted_reviewer(prompt: str, schema: dict[str, Any]) -> Mapping[str, Any] | str:
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
        if (
            not include_reference
            and (meta.get("type") == "reference" or path.parent.name == "car-spec")
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
            try:
                review = review_semantic_mutation(
                    proposal,
                    expected_text=text,
                    reviewer=budgeted_reviewer,
                    artifact_dir=reviews_dir,
                )
            except Exception:
                retry += 1
                pending_pages.append(page_id)
                if limit and candidates >= limit:
                    break
                continue
            decision = review.get("decision")
            if decision == "approved" and review.get("valid") is True:
                apply_status = _apply_entities_cas(
                    path,
                    expected_text=text,
                    updated_text=new_text,
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
                rejected += 1
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
        "frontier_calls": frontier_calls,
        "pending_pages": pending_pages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maintain LLM Wiki entity frontmatter.")
    sub = parser.add_subparsers(dest="command", required=True)
    init_cmd = sub.add_parser("init", help="Write a default entity registry if missing.")
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
        print("\t".join(f"{key}={value}" for key, value in payload.items() if key != "pages"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
