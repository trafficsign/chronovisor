#!/usr/bin/env python3
"""Obsolete local-model tag backfill (semantic writes disabled).

Walks every untagged page in ``~/.chronovisor/pages/``, asks the LLM for tags
under the v2 prompt (TAG_REPORT_SYSTEM_PROMPT), and writes the result
to a per-page progress log. When a page produces a non-empty tag set,
the page's frontmatter is patched in place.

Resumable: every page is logged to ``~/.chronovisor/.tag-backfill-progress.jsonl``
with status ``applied`` / ``skipped`` / ``error``. Re-running the script
skips any page already present in the progress log.

Failure-tolerant: a single LLM error or parse failure does NOT abort the
sweep; it logs ``error`` and moves on.

Use ``chronovisor-sleep`` instead. This module remains only so historical logs and
diagnostic helpers can be read; every mutation entry point fails closed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chronovisor.core.frontmatter import parse as fm_parse
from chronovisor.core.frontmatter import patch as fm_patch
from chronovisor.core.index_store import get_store
from chronovisor.core.ollama import generate as _ollama_generate
from chronovisor.core.store import find_page
from chronovisor.core.tag_rules import SEED_TAGS
from chronovisor.librarian.tag_distribution import (
    TAG_REPORT_SYSTEM_PROMPT,
    parse_llm_response,
)
from chronovisor.raw.legacy_semantic_write import (
    block_legacy_semantic_mutation,
)

PROGRESS_FILE = Path.home() / ".chronovisor" / ".tag-backfill-progress.jsonl"
PLAN_INBOX = Path.home() / "projects" / "plan" / "inbox"

# tag_status frontmatter values for pages where the LLM gave up.
# Future sweeps look at this field to skip already-attempted pages
# without consulting the progress jsonl.
TAG_STATUS_NO_FIT = "no-fit-master"   # LLM returned an empty assigned_tags list
TAG_STATUS_FORMAT_FAIL = "format-fail"  # JSON / schema validation failed


def _flatten_master() -> list[str]:
    return [t for axis in SEED_TAGS.values() for t in axis]


def _load_done(progress_path: Path) -> set[str]:
    """Read the progress log and return the set of already-processed page ids."""
    if not progress_path.exists():
        return set()
    done: set[str] = set()
    for line in progress_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid = rec.get("page_id")
        if isinstance(pid, str):
            done.add(pid)
    return done


def _is_marked_unfit(page_id: str) -> bool:
    """True if a previous run already concluded this page can't be tagged.

    We re-parse the page's frontmatter (rather than trust the IndexStore)
    because ``tag_status`` is a backfill-private field that the IndexStore
    does not expose.
    """
    path = find_page(page_id)
    if path is None:
        return False
    try:
        meta_fm, _ = fm_parse(path.read_text())
    except Exception:
        return False
    return bool(meta_fm.get("tag_status"))


def _select_candidates(store) -> list[str]:
    """All pages without ``tags`` frontmatter, ordered by recency.

    Pages already marked ``tag_status: …`` from a prior run are skipped
    so a re-run does not hammer Ollama on the same untaggable corpus.
    """
    out: list[str] = []
    for meta in store.all_pages_meta(include_system=False):
        page_id = meta["page_id"]
        if store.tags(page_id):
            continue  # already tagged
        if _is_marked_unfit(page_id):
            continue  # previously attempted, marked untaggable
        out.append(page_id)
    return out


def _mark_unfit(path: Path, status_value: str) -> None:
    """Stamp ``tag_status: <value>`` into the page frontmatter.

    Best-effort: any IO or parse error is swallowed so a stamp failure
    cannot derail the sweep.
    """
    block_legacy_semantic_mutation(
        tool="tag_backfill_apply.py",
        replacement="chronovisor-sleep",
    )
    try:
        original = path.read_text()
        patched = fm_patch(original, {"tag_status": status_value})
        path.write_text(patched)
    except Exception:
        pass


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


def _append_progress(progress_path: Path, record: dict) -> None:
    """Append one JSON line and fsync so a crash doesn't lose progress."""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def _process_one(page_id: str, master: list[str]) -> dict:
    """Run the LLM, patch the page in place if tags came back."""
    started = time.time()
    path = find_page(page_id)
    if path is None:
        return {
            "page_id": page_id,
            "status": "error",
            "reason": "page not found",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "elapsed_ms": int((time.time() - started) * 1000),
        }

    original = path.read_text()
    _, body = fm_parse(original)
    prompt = _build_prompt(page_id, body, master)

    try:
        raw = _ollama_generate(prompt, system=TAG_REPORT_SYSTEM_PROMPT)
    except Exception as e:
        return {
            "page_id": page_id,
            "status": "error",
            "reason": f"llm error: {e}",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "elapsed_ms": int((time.time() - started) * 1000),
        }

    parsed = parse_llm_response(raw, set(master))
    if parsed is None:
        _mark_unfit(path, TAG_STATUS_FORMAT_FAIL)
        return {
            "page_id": page_id,
            "status": "skipped",
            "reason": "llm response failed schema validation",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "elapsed_ms": int((time.time() - started) * 1000),
        }

    tags = parsed["assigned_tags"]
    if not tags:
        _mark_unfit(path, TAG_STATUS_NO_FIT)
        return {
            "page_id": page_id,
            "status": "skipped",
            "reason": "no tags assigned by LLM",
            "confidence": parsed["confidence"],
            "ts": datetime.now().isoformat(timespec="seconds"),
            "elapsed_ms": int((time.time() - started) * 1000),
        }

    # On success, clear any stale tag_status from a prior failed attempt.
    patched = fm_patch(original, {"tags": tags}, deletes=["tag_status"])
    block_legacy_semantic_mutation(
        tool="tag_backfill_apply.py",
        replacement="chronovisor-sleep",
    )
    path.write_text(patched)

    return {
        "page_id": page_id,
        "status": "applied",
        "tags": tags,
        "main_topic": parsed["main_topic"],
        "confidence": parsed["confidence"],
        "ts": datetime.now().isoformat(timespec="seconds"),
        "elapsed_ms": int((time.time() - started) * 1000),
    }


def _write_summary(progress_path: Path, summary_path: Path) -> None:
    """Read the full progress log and write a human-readable summary."""
    if not progress_path.exists():
        return

    counts = {"applied": 0, "skipped": 0, "error": 0}
    applied: list[dict] = []
    skipped: list[dict] = []
    errors: list[dict] = []
    total_elapsed = 0
    for line in progress_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = rec.get("status", "")
        if status in counts:
            counts[status] += 1
        total_elapsed += rec.get("elapsed_ms", 0)
        if status == "applied":
            applied.append(rec)
        elif status == "skipped":
            skipped.append(rec)
        elif status == "error":
            errors.append(rec)

    today = date.today().isoformat()
    out: list[str] = []
    out.append("---")
    out.append(f"title: Tag Backfill Apply — Morning Review ({today})")
    out.append(f"updated: {today}")
    out.append("---")
    out.append("")
    out.append("# Tag Backfill Apply — Morning Review")
    out.append("")
    out.append(
        "Full-sweep result. Pages on disk **were modified** "
        "(``tags:`` added to frontmatter)."
    )
    out.append("")
    out.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    out.append(f"Progress log: `{progress_path}`")
    out.append(
        f"Total wall time of LLM calls: {total_elapsed / 1000:.1f}s "
        f"({total_elapsed / 60000:.1f} min)"
    )
    out.append("")
    out.append("## Summary counts")
    out.append(f"- applied: {counts['applied']}")
    out.append(f"- skipped: {counts['skipped']}")
    out.append(f"- errors: {counts['error']}")
    out.append(
        f"- total processed: "
        f"{counts['applied'] + counts['skipped'] + counts['error']}"
    )
    out.append("")

    if errors:
        out.append("## Errors (these will need re-running)")
        for r in errors[:50]:
            out.append(f"- `{r['page_id']}` — {r.get('reason', '')}")
        if len(errors) > 50:
            out.append(f"- … and {len(errors) - 50} more (see progress log)")
        out.append("")

    if skipped:
        out.append("## Skipped (LLM returned no usable tags)")
        for r in skipped[:30]:
            out.append(f"- `{r['page_id']}` — {r.get('reason', '')}")
        if len(skipped) > 30:
            out.append(f"- … and {len(skipped) - 30} more (see progress log)")
        out.append("")

    if applied:
        out.append("## Applied (sample of 30)")
        for r in applied[:30]:
            tags = ", ".join(r.get("tags", []))
            conf = r.get("confidence", 0.0)
            out.append(f"- `{r['page_id']}` — `{tags}` (conf {conf:.2f})")
        out.append("")
        out.append(
            "Full list of all applied pages is in the progress log."
        )
        out.append("")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(out) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many pages (smoke testing).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Final summary path. Default: ~/projects/plan/inbox/{today}_morning-review-tag-backfill-result.md",
    )
    args = parser.parse_args()

    block_legacy_semantic_mutation(
        tool="tag_backfill_apply.py",
        replacement="chronovisor-sleep",
    )

    store = get_store()
    store.refresh()

    candidates = _select_candidates(store)
    done = _load_done(PROGRESS_FILE)
    todo = [p for p in candidates if p not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"candidates_total={len(candidates)} done={len(done)} todo={len(todo)}")
    if not todo:
        print("nothing to do")
        return 0

    master = _flatten_master()
    started = time.time()
    for i, page_id in enumerate(todo, 1):
        record = _process_one(page_id, master)
        _append_progress(PROGRESS_FILE, record)
        if i % 10 == 0 or i == len(todo):
            elapsed = time.time() - started
            print(
                f"[{i}/{len(todo)}] last={page_id} status={record['status']} "
                f"elapsed={elapsed:.0f}s"
            )

    summary = args.summary or (
        PLAN_INBOX / f"{date.today().isoformat()}_morning-review-tag-backfill-result.md"
    )
    _write_summary(PROGRESS_FILE, summary)
    print(json.dumps({
        "summary_path": str(summary),
        "progress_log": str(PROGRESS_FILE),
        "todo_count": len(todo),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
