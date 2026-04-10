#!/usr/bin/env python3
"""Batch ingest all pending raw files, 10 at a time.

Runs synchronously (not threaded) so the process stays alive.
Designed for initial migration processing.
"""

import json
import time
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_wiki_mcp.wiki import init_wiki, RAW_DIR, PAGES_DIR
from llm_wiki_mcp.ollama import generate, is_available, INGEST_SYSTEM_PROMPT
from llm_wiki_mcp.ingest import _build_context, _parse_output, _apply_operations, _rebuild_index, _append_log
from llm_wiki_mcp.orchestrator import get_pending_raw_files, mark_raw_processed

BATCH_SIZE = 10


def run_batch(batch: list[Path]) -> dict:
    """Process a single batch of raw files."""
    contents = []
    filenames = []
    for f in batch:
        contents.append(f"--- Source: {f.name} ---\n{f.read_text()}")
        filenames.append(f.name)

    combined = "\n\n".join(contents)

    existing_pages = list(PAGES_DIR.glob("*.md"))
    context = _build_context(existing_pages)

    prompt = f"""{context}

---
Raw session data to ingest:
---
{combined}
---

Extract wiki-worthy knowledge from the above and produce structured pages.
Use today's date: {date.today().isoformat()} for the updated field."""

    output = generate(prompt, system=INGEST_SYSTEM_PROMPT)

    operations = _parse_output(output)
    if operations:
        created, updated = _apply_operations(operations)
        _rebuild_index()
    else:
        created, updated = [], []

    mark_raw_processed(filenames)

    return {
        "files": len(filenames),
        "created": created,
        "updated": updated,
    }


def main():
    init_wiki()

    if not is_available():
        print("ERROR: Ollama is not running")
        sys.exit(1)

    pending = get_pending_raw_files()
    total = len(pending)
    print(f"Total pending: {total} files")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Estimated batches: {(total + BATCH_SIZE - 1) // BATCH_SIZE}")
    print()

    batch_num = 0
    total_created = 0
    total_updated = 0

    while True:
        pending = get_pending_raw_files()
        if not pending:
            break

        batch = pending[:BATCH_SIZE]
        batch_num += 1
        print(f"=== Batch {batch_num} ({len(batch)} files) ===")

        try:
            result = run_batch(batch)
            total_created += len(result["created"])
            total_updated += len(result["updated"])
            print(f"  Created: {result['created']}")
            print(f"  Updated: {result['updated']}")
            _append_log(f"batch_ingest | batch {batch_num}: {len(result['created'])} created, {len(result['updated'])} updated")
        except Exception as e:
            print(f"  ERROR: {e}")
            _append_log(f"batch_ingest | batch {batch_num} FAILED: {e}")
            # Mark as processed anyway to avoid infinite retry
            mark_raw_processed([f.name for f in batch])
            time.sleep(5)

        remaining = len(get_pending_raw_files())
        print(f"  Remaining: {remaining}")
        print()

    print(f"\n=== Complete ===")
    print(f"Batches: {batch_num}")
    print(f"Pages created: {total_created}")
    print(f"Pages updated: {total_updated}")
    print(f"Total pages: {len(list(PAGES_DIR.glob('*.md')))}")


if __name__ == "__main__":
    main()
