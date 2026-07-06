"""Append-only claim ledger seed for future event-sourced memory."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp.frontmatter import parse
from llm_wiki_mcp.wiki import WIKI_ROOT, find_page

CLAIMS_DIR = WIKI_ROOT / "claims"
CLAIMS_FILE = CLAIMS_DIR / "claims.jsonl"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def claim_from_page(page_id: str, *, source_raw: str = "", op: str = "upsert") -> dict[str, Any] | None:
    path = find_page(page_id)
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta, _body = parse(text)
    title = meta.get("title")
    summary = meta.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = title if isinstance(title, str) else page_id
    entities = meta.get("entities")
    page_type = meta.get("type") if isinstance(meta.get("type"), str) else "knowledge"
    return {
        "claim_id": f"{datetime.now().strftime('%Y%m%dT%H%M%S%f')}-{page_id}",
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "subject": page_id,
        "predicate": "page.summary",
        "value": summary,
        "source_page": page_id,
        "source_raw": source_raw,
        "op": op,
        "page_type": page_type,
        "entities": entities if isinstance(entities, list) else [],
        "valid_from": str(meta.get("updated") or date.today().isoformat()),
        "valid_to": None,
    }


def append_page_claims(page_ids: list[str], *, source_raw: str = "", op: str = "upsert") -> dict[str, Any]:
    written = 0
    skipped: list[str] = []
    for page_id in page_ids:
        claim = claim_from_page(page_id, source_raw=source_raw, op=op)
        if claim is None:
            skipped.append(page_id)
            continue
        _append_jsonl(CLAIMS_FILE, claim)
        written += 1
    return {
        "status": "ok",
        "claims_file": str(CLAIMS_FILE),
        "written": written,
        "skipped": skipped,
    }
