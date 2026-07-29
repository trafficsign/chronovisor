"""Export wiki-derived QA data for future local-model distillation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.frontmatter import parse
from chronovisor.search.index_store import get_store
from chronovisor.core.store import CHRONOVISOR_ROOT, find_page

DISTILL_DIR = CHRONOVISOR_ROOT / "distill"
DISTILL_FILE = DISTILL_DIR / "wiki-qa.jsonl"


def _answer_for_page(page_id: str, summary: str) -> str:
    if summary.strip():
        return summary.strip()
    path = find_page(page_id)
    if path is None:
        return page_id
    try:
        _meta, body = parse(path.read_text(encoding="utf-8"))
    except OSError:
        return page_id
    for line in body.splitlines():
        clean = line.strip(" #-\t")
        if clean:
            return clean[:600]
    return page_id


def export_distill_dataset(
    *,
    output_file: Path = DISTILL_FILE,
    limit: int = 0,
    include_reference: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    store = get_store()
    store.refresh()
    rows: list[dict[str, Any]] = []
    for meta in store.all_pages_meta(include_system=False):
        page_id = str(meta.get("page_id") or "")
        if not page_id:
            continue
        if not include_reference and meta.get("page_type") == "reference":
            continue
        full = store.meta(page_id) or {}
        questions = full.get("recall_questions")
        if not isinstance(questions, list) or not questions:
            continue
        answer = _answer_for_page(page_id, str(full.get("summary") or ""))
        for question in questions:
            if not isinstance(question, str) or not question.strip():
                continue
            rows.append(
                {
                    "instruction": question.strip(),
                    "input": "",
                    "output": answer,
                    "source_page": page_id,
                    "page_type": meta.get("page_type") or "knowledge",
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break

    if write:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        output_file.write_text(payload, encoding="utf-8")
    return {
        "status": "ok",
        "path": str(output_file),
        "rows": len(rows),
        "write": write,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-distill`` command-line entry point."""
    parser = argparse.ArgumentParser(description="Export wiki QA pairs for distillation.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    data = export_distill_dataset(
        limit=max(0, args.limit),
        include_reference=args.include_reference,
        write=not args.no_write,
    )
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"rows\t{data['rows']}")
        print(f"path\t{data['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
