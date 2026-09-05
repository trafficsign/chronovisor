#!/usr/bin/env python3.14
"""Obsolete tag-backfill retry reader (semantic writes disabled).

Reads ``~/.chronovisor/.tag-backfill-progress.jsonl``, finds entries that failed
the JSON schema gate (typically a one-shot Ollama format hiccup), and
re-runs each one up to ``--retries`` times. On success, the page's
frontmatter is patched in place and a fresh ``applied`` row is appended
to the progress log. The original ``skipped`` row is left untouched —
the log is append-only — but the latest ``applied`` row will dominate
any downstream tooling that walks the file.

Use ``chronovisor-sleep``. The historical progress reader is retained for audit,
but the executable fails closed before a local-model call or page mutation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Reuse the apply pipeline so behaviour stays identical to the main sweep.
from tag_backfill_apply import (
    PROGRESS_FILE,
    _append_progress,
    _flatten_master,
    _process_one,
)

from chronovisor.raw.legacy_semantic_write import (
    block_legacy_semantic_mutation,
)

SCHEMA_FAIL_REASON = "llm response failed schema validation"


def _collect_failures(progress_path: Path) -> list[str]:
    """Return page_ids whose latest record is a schema-validation failure.

    "Latest" matters: a successful retry already appended an ``applied``
    row, so we only want to retry pages where no later success is on
    record yet.
    """
    if not progress_path.exists():
        return []
    latest_status: dict[str, tuple[str, str]] = {}
    for line in progress_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid = rec.get("page_id")
        status = rec.get("status", "")
        reason = rec.get("reason", "")
        if isinstance(pid, str):
            latest_status[pid] = (status, reason)
    return [
        pid
        for pid, (status, reason) in latest_status.items()
        if status == "skipped" and reason == SCHEMA_FAIL_REASON
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max attempts per page (default 3).",
    )
    args = parser.parse_args()

    block_legacy_semantic_mutation(
        tool="tag_backfill_retry.py",
        replacement="chronovisor-sleep",
    )

    failures = _collect_failures(PROGRESS_FILE)
    if not failures:
        print("no schema-fail rows to retry")
        return 0

    print(f"retrying {len(failures)} pages, max {args.retries} attempts each")
    master = _flatten_master()

    summary = {"applied": 0, "still_failing": 0, "skipped_no_tags": 0}
    for i, page_id in enumerate(failures, 1):
        last_record: dict | None = None
        for attempt in range(1, args.retries + 1):
            record = _process_one(page_id, master)
            last_record = record
            status = record.get("status")
            if status == "applied":
                record["retry_attempt"] = attempt
                _append_progress(PROGRESS_FILE, record)
                summary["applied"] += 1
                print(f"[{i}/{len(failures)}] {page_id}: applied (attempt {attempt})")
                break
            if status == "skipped" and record.get("reason") == "no tags assigned by LLM":
                # LLM gave a real, well-formed answer — empty tags. Stop retrying.
                record["retry_attempt"] = attempt
                _append_progress(PROGRESS_FILE, record)
                summary["skipped_no_tags"] += 1
                print(f"[{i}/{len(failures)}] {page_id}: empty tags (attempt {attempt})")
                break
        else:
            # Loop fell through without a break — every attempt failed.
            if last_record is not None:
                last_record["retry_attempt"] = args.retries
                last_record["retry_exhausted"] = True
                _append_progress(PROGRESS_FILE, last_record)
            summary["still_failing"] += 1
            print(f"[{i}/{len(failures)}] {page_id}: still failing after {args.retries}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
