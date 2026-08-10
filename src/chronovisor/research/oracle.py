"""Cited wiki oracle bundle.

This command intentionally returns grounded evidence instead of free-form
answers. A local or frontier model can consume the bundle, but every surfaced
fact is tied back to page IDs and derived claim IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chronovisor.core.claims import (
    CLAIM_INDEX_FILE,
    rebuild_claim_index,
    search_claims,
)
from chronovisor.core.search import search
from chronovisor.core.store import CHRONOVISOR_ROOT, okf_runtime_operation


def oracle_bundle(
    query: str,
    *,
    top_n: int = 8,
    claim_limit: int = 12,
    ensure_claim_index: bool = True,
) -> dict[str, Any]:
    if ensure_claim_index and not CLAIM_INDEX_FILE.exists():
        rebuild_claim_index(limit=0)
    pages, mode = search(query, top_n=top_n)
    claims = search_claims(query, limit=claim_limit)
    cited_pages = [
        {
            "page_id": page.page_id,
            "title": page.title,
            "folder": page.folder,
            "updated": page.updated,
            "score": page.score,
            "page_type": page.page_type,
            "sensitivity": page.sensitivity,
        }
        for page in pages
    ]
    cited_claims = [
        {
            "claim_id": claim.get("claim_id"),
            "source_page": claim.get("source_page"),
            "predicate": claim.get("predicate"),
            "value": claim.get("value"),
            "valid_from": claim.get("valid_from"),
            "valid_to": claim.get("valid_to"),
            "score": claim.get("score"),
        }
        for claim in claims
    ]
    return {
        "status": "ok",
        "query": query,
        "answer_mode": "cite-only",
        "search_mode": mode,
        "pages": cited_pages,
        "claims": cited_claims,
        "claim_index": str(Path(CLAIM_INDEX_FILE)),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-oracle`` command-line entry point."""
    parser = argparse.ArgumentParser(description="Return a cited wiki oracle evidence bundle.")
    parser.add_argument("query")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--claim-limit", type=int, default=12)
    parser.add_argument("--no-index-build", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.no_index_build:
        return _main_locked(args)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        with okf_runtime_operation(CHRONOVISOR_ROOT):
            return _main_locked(args)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75


def _main_locked(args: argparse.Namespace) -> int:
    data = oracle_bundle(
        args.query,
        top_n=max(1, args.top_n),
        claim_limit=max(1, args.claim_limit),
        ensure_claim_index=not args.no_index_build,
    )
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"search_mode\t{data['search_mode']}")
        for page in data["pages"]:
            print(f"page\t{page['page_id']}\t{page['title']}")
        for claim in data["claims"]:
            print(f"claim\t{claim['claim_id']}\t{claim['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
