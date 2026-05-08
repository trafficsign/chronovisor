#!/usr/bin/env python3
"""Tag backfill dry-run on N existing pages (no disk mutation).

Generates the *would-be* tag patches for the N most-recently-updated
pages that lack a ``tags:`` frontmatter field, writes per-page artefacts
(original / patched / diff / meta) under
``~/.wiki/.tag-backfill-dryrun/``, and produces a single morning-review
summary in ``~/projects/plan/inbox/`` so a human can decide whether
the LLM's tag choices look reasonable before committing to a full
1631-page sweep.

This script does NOT modify ``pages/`` — every patch is rendered to a
sibling file. The goal is "preview, then decide", not "auto-apply".

Usage:
    python3 scripts/tag_backfill_dryrun.py
    python3 scripts/tag_backfill_dryrun.py --count 5
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_wiki_mcp.frontmatter import parse as fm_parse, patch as fm_patch  # noqa: E402
from llm_wiki_mcp.index_store import get_store  # noqa: E402
from llm_wiki_mcp.ollama import generate as _ollama_generate  # noqa: E402
from llm_wiki_mcp.tag_distribution import (  # noqa: E402
    TAG_REPORT_SYSTEM_PROMPT,
    parse_llm_response,
)
from llm_wiki_mcp.tags import SEED_TAGS  # noqa: E402
from llm_wiki_mcp.wiki import PAGES_DIR, find_page  # noqa: E402


DRY_RUN_DIR = Path.home() / ".wiki" / ".tag-backfill-dryrun"
PLAN_INBOX = Path.home() / "projects" / "plan" / "inbox"


def _flatten_master() -> list[str]:
    return [t for axis in SEED_TAGS.values() for t in axis]


def _select_candidates(store, count: int) -> list[dict]:
    """Pick N candidates: pages without tags, sorted by recency."""
    candidates = []
    for meta in store.all_pages_meta(include_system=False):
        page_id = meta["page_id"]
        if store.tags(page_id):
            continue  # already tagged, skip
        candidates.append(meta)
        if len(candidates) >= count:
            break
    return candidates


def _build_prompt(page_id: str, body: str, master: list[str]) -> str:
    master_str = "\n".join(f"  {t}" for t in master)
    return f"""\
TAG MASTER LIST:
{master_str}

PAGE:
  page_id: {page_id}
  body head:
{body[:2000]}

Output the JSON object per the rules.
"""


def _process_one(page_id: str, master: list[str]) -> dict:
    """Run the LLM, build the patched frontmatter, return the artefact set."""
    path = find_page(page_id)
    if path is None:
        return {"page_id": page_id, "error": "page not found"}

    original = path.read_text()
    meta, body = fm_parse(original)
    prompt = _build_prompt(page_id, body, master)

    try:
        raw = _ollama_generate(prompt, system=TAG_REPORT_SYSTEM_PROMPT)
    except Exception as e:
        return {"page_id": page_id, "error": f"llm error: {e}", "original": original}

    parsed = parse_llm_response(raw, set(master))
    if parsed is None:
        return {
            "page_id": page_id,
            "error": "llm response failed schema validation",
            "raw_response": raw,
            "original": original,
        }

    proposed_tags = parsed["assigned_tags"]
    if not proposed_tags:
        return {
            "page_id": page_id,
            "skipped": "no tags assigned by LLM",
            "raw_response": raw,
            "confidence": parsed["confidence"],
            "rejected_assigned_tags": parsed["rejected_assigned_tags"],
            "suggested_missing_categories": parsed["suggested_missing_categories"],
            "original": original,
        }

    patched = fm_patch(original, {"tags": proposed_tags})

    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"{page_id} (original)",
            tofile=f"{page_id} (proposed)",
            n=2,
        )
    )

    return {
        "page_id": page_id,
        "title": meta.get("title", page_id),
        "proposed_tags": proposed_tags,
        "rejected_assigned_tags": parsed["rejected_assigned_tags"],
        "suggested_missing_categories": parsed["suggested_missing_categories"],
        "confidence": parsed["confidence"],
        "raw_response": raw,
        "original": original,
        "patched": patched,
        "diff": "".join(diff_lines),
    }


def _persist(artefact: dict, root: Path) -> Path:
    page_dir = root / artefact["page_id"]
    page_dir.mkdir(parents=True, exist_ok=True)
    if "original" in artefact:
        (page_dir / "original.md").write_text(artefact["original"])
    if "patched" in artefact:
        (page_dir / "patched.md").write_text(artefact["patched"])
    if "diff" in artefact:
        (page_dir / "diff.md").write_text(artefact["diff"])
    meta_dump = {
        k: v
        for k, v in artefact.items()
        if k not in ("original", "patched", "diff")
    }
    (page_dir / "meta.json").write_text(json.dumps(meta_dump, ensure_ascii=False, indent=2))
    return page_dir


def _write_morning_review(artefacts: list[dict], summary_path: Path) -> None:
    today = date.today().isoformat()
    lines: list[str] = []
    lines.append("---")
    lines.append(f"title: Tag Backfill Dry-Run — Morning Review ({today})")
    lines.append(f"updated: {today}")
    lines.append("---")
    lines.append("")
    lines.append("# Tag Backfill Dry-Run — Morning Review")
    lines.append("")
    lines.append(
        "Each entry below is a *would-be* patch, not yet applied. Inspect "
        "the proposed tags + reasoning, then decide:"
    )
    lines.append("")
    lines.append(
        "1. If the picks look right across the board, run the **full backfill** "
        "(all 1631 pages). That sweep will be a separate step."
    )
    lines.append("2. If a few are off, fix the prompt or master list and re-run "
                 "this dry-run before committing to the full sweep.")
    lines.append("3. If many look wrong, escalate before running the full sweep.")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Artefact directory: `{DRY_RUN_DIR}`")
    lines.append("")

    successes = [a for a in artefacts if "proposed_tags" in a]
    skips = [a for a in artefacts if "skipped" in a]
    errors = [a for a in artefacts if "error" in a]

    lines.append("## Summary counts")
    lines.append(f"- proposed: {len(successes)}")
    lines.append(f"- skipped (no LLM-assigned tags): {len(skips)}")
    lines.append(f"- errors: {len(errors)}")
    lines.append("")

    if successes:
        lines.append("## Proposed patches")
        lines.append("")
        for a in successes:
            lines.append(f"### `{a['page_id']}` — {a.get('title', '')}")
            lines.append(
                f"- proposed_tags: `{', '.join(a['proposed_tags'])}`"
            )
            lines.append(f"- confidence: {a['confidence']:.2f}")
            if a.get("rejected_assigned_tags"):
                lines.append(
                    f"- rejected (LLM tried but master list filtered): "
                    f"`{', '.join(a['rejected_assigned_tags'])}`"
                )
            if a.get("suggested_missing_categories"):
                gaps = ", ".join(
                    f"{c['label']}({c['fallback_axis']})"
                    for c in a["suggested_missing_categories"][:3]
                )
                lines.append(f"- suggested_missing_categories: {gaps}")
            lines.append(
                f"- artefacts: `{DRY_RUN_DIR / a['page_id']}/`"
            )
            lines.append("")

    if skips:
        lines.append("## Skipped (no master tag fit)")
        for a in skips:
            lines.append(f"- `{a['page_id']}` — {a.get('skipped', '')}")
        lines.append("")

    if errors:
        lines.append("## Errors")
        for a in errors:
            lines.append(f"- `{a['page_id']}` — {a.get('error', '')}")
        lines.append("")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="Pages to dry-run.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Morning review summary path. Default: ~/projects/plan/inbox/{today}_morning-review-tag-backfill.md",
    )
    args = parser.parse_args()

    store = get_store()
    store.refresh()

    candidates = _select_candidates(store, args.count)
    if not candidates:
        print("no untagged pages found")
        return 0

    DRY_RUN_DIR.mkdir(parents=True, exist_ok=True)
    master = _flatten_master()
    artefacts: list[dict] = []
    for meta in candidates:
        artefact = _process_one(meta["page_id"], master)
        _persist(artefact, DRY_RUN_DIR)
        artefacts.append(artefact)

    summary = args.summary or (
        PLAN_INBOX / f"{date.today().isoformat()}_morning-review-tag-backfill.md"
    )
    _write_morning_review(artefacts, summary)

    print(json.dumps({
        "candidates": len(candidates),
        "artefact_dir": str(DRY_RUN_DIR),
        "summary_path": str(summary),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
