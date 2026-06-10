#!/usr/bin/env python3
"""Backfill summary/recall_questions frontmatter for existing wiki pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill LLM Wiki recall_questions.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum pages to update. 0 means no limit.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from llm_wiki_mcp.frontmatter import parse, patch
    from llm_wiki_mcp.ingest import _ensure_recall_metadata_frontmatter
    from llm_wiki_mcp.link_fix import atomic_write
    from llm_wiki_mcp.wiki import all_pages, page_id_from_path

    scanned = 0
    updated = 0
    changed: list[str] = []
    for path in all_pages():
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        page_id = page_id_from_path(path)
        new_text = _ensure_recall_metadata_frontmatter(text, page_id, parse, patch)
        if new_text == text:
            continue
        updated += 1
        changed.append(page_id)
        if not args.dry_run:
            atomic_write(path, new_text)
        if args.limit and updated >= args.limit:
            break

    print(
        json.dumps(
            {
                "status": "ok",
                "dry_run": args.dry_run,
                "scanned": scanned,
                "updated": updated,
                "pages": changed,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
