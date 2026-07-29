"""Expand search golden sets from page recall questions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.search.index_store import get_store
from chronovisor.recall.recall_runtime_paths import RECALL_DIR
from chronovisor.search.search_eval import assign_split, language_bucket, query_kind

GOLDEN_FILE = RECALL_DIR / "search-golden.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _key(row: dict[str, Any]) -> tuple[str, tuple[str, ...], bool]:
    expected = row.get("expected_pages")
    pages = tuple(str(item) for item in expected) if isinstance(expected, list) else ()
    return (str(row.get("query") or "").strip(), pages, bool(row.get("negative")))


def rows_from_recall_questions(*, limit: int = 0, include_reference: bool = False) -> list[dict[str, Any]]:
    store = get_store()
    store.refresh()
    metas = store.all_pages_meta(include_system=False)
    rows: list[dict[str, Any]] = []
    for meta in metas:
        page_id = str(meta.get("page_id") or "")
        if not page_id:
            continue
        if not include_reference and meta.get("page_type") == "reference":
            continue
        full = store.meta(page_id) or {}
        questions = full.get("recall_questions")
        if not isinstance(questions, list):
            continue
        for question in questions:
            if not isinstance(question, str) or not question.strip():
                continue
            query = question.strip()
            seed = json.dumps([query, page_id], ensure_ascii=False)
            rows.append(
                {
                    "id": "rq-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16],
                    "query": query,
                    "expected_pages": [page_id],
                    "negative": False,
                    "source": "recall_questions",
                    "source_page": page_id,
                    "split": assign_split(query),
                    "language": language_bucket(query),
                    "kind": query_kind(query),
                    "reviewed": True,
                    "reviewer": "chronovisor:recall-questions",
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            if limit and len(rows) >= limit:
                return rows
    return rows


def expand_golden_from_recall_questions(
    *,
    golden_file: Path = GOLDEN_FILE,
    limit: int = 0,
    include_reference: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    existing = _read_jsonl(golden_file)
    existing_keys = {_key(row) for row in existing}
    candidates = rows_from_recall_questions(limit=limit, include_reference=include_reference)
    additions: list[dict[str, Any]] = []
    seen = set(existing_keys)
    for row in candidates:
        key = _key(row)
        if key in seen:
            continue
        seen.add(key)
        additions.append(row)
    if write and additions:
        golden_file.parent.mkdir(parents=True, exist_ok=True)
        with golden_file.open("a", encoding="utf-8") as f:
            for row in additions:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "status": "ok",
        "path": str(golden_file),
        "existing": len(existing),
        "candidates": len(candidates),
        "added": len(additions),
        "write": write,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-golden-expand`` command-line entry point."""
    parser = argparse.ArgumentParser(description="Expand search golden set from recall_questions.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    data = expand_golden_from_recall_questions(
        limit=max(0, args.limit),
        include_reference=args.include_reference,
        write=not args.no_write,
    )
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"added\t{data['added']}")
        print(f"path\t{data['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
