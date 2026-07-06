"""Append-only claim ledger seed for future event-sourced memory."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp.frontmatter import parse
from llm_wiki_mcp.index_store import get_store
from llm_wiki_mcp.wiki import WIKI_ROOT, find_page

CLAIMS_DIR = WIKI_ROOT / "claims"
CLAIMS_FILE = CLAIMS_DIR / "claims.jsonl"
CLAIM_INDEX_FILE = CLAIMS_DIR / "claims-index.jsonl"


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
    if not source_raw.strip():
        return {
            "status": "skipped",
            "reason": "source_raw required for append-only claim ledger",
            "claims_file": str(CLAIMS_FILE),
            "written": 0,
            "skipped": list(page_ids),
        }
    written = 0
    skipped: list[str] = []
    for page_id in page_ids:
        rows = page_claims(page_id, source_raw=source_raw, op=op)
        if not rows:
            skipped.append(page_id)
            continue
        for claim in rows:
            _append_jsonl(CLAIMS_FILE, claim)
            written += 1
    return {
        "status": "ok",
        "claims_file": str(CLAIMS_FILE),
        "written": written,
        "skipped": skipped,
    }


def page_claims(page_id: str, *, source_raw: str = "", op: str = "index") -> list[dict[str, Any]]:
    path = find_page(page_id)
    if path is None:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    meta, body = parse(text)
    title = meta.get("title") if isinstance(meta.get("title"), str) else page_id
    summary = meta.get("summary") if isinstance(meta.get("summary"), str) else ""
    updated = str(meta.get("updated") or date.today().isoformat())
    entities = meta.get("entities") if isinstance(meta.get("entities"), list) else []
    page_type = meta.get("type") if isinstance(meta.get("type"), str) else "knowledge"
    status = meta.get("status") if isinstance(meta.get("status"), str) else "active"
    base = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "source_page": page_id,
        "source_raw": source_raw,
        "op": op,
        "page_type": page_type,
        "entities": entities,
        "valid_from": updated,
        "valid_to": None if status == "active" else updated,
    }
    claims: list[dict[str, Any]] = [
        {
            **base,
            "claim_id": f"{page_id}:title",
            "subject": page_id,
            "predicate": "page.title",
            "value": title,
        }
    ]
    if summary.strip():
        claims.append(
            {
                **base,
                "claim_id": f"{page_id}:summary",
                "subject": page_id,
                "predicate": "page.summary",
                "value": summary.strip(),
            }
        )
    for entity in entities:
        if isinstance(entity, str) and entity.strip():
            claims.append(
                {
                    **base,
                    "claim_id": f"{page_id}:entity:{entity}",
                    "subject": page_id,
                    "predicate": "page.entity",
                    "value": entity.strip(),
                }
            )
    first_line = next((line.strip(" #-\t") for line in body.splitlines() if line.strip(" #-\t")), "")
    if first_line and first_line != summary:
        claims.append(
            {
                **base,
                "claim_id": f"{page_id}:body.lead",
                "subject": page_id,
                "predicate": "body.lead",
                "value": first_line[:280],
            }
        )
    return claims


def rebuild_claim_index(*, limit: int = 0, path: Path = CLAIM_INDEX_FILE, write: bool = True) -> dict[str, Any]:
    store = get_store()
    store.refresh()
    metas = [meta for meta in store.all_pages_meta(include_system=False) if meta.get("page_type") != "reference"]
    if limit:
        metas = metas[:limit]
    rows: list[dict[str, Any]] = []
    for meta in metas:
        page_id = str(meta.get("page_id") or "")
        if page_id:
            rows.extend(page_claims(page_id))
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            for row in rows
        )
        path.write_text(payload, encoding="utf-8")
    return {"status": "ok", "path": str(path), "pages": len(metas), "claims": len(rows), "write": write}


def _claim_tokens(row: dict[str, Any]) -> set[str]:
    text = " ".join(str(row.get(key) or "") for key in ("subject", "predicate", "value", "entities"))
    return {
        token.lower()
        for token in re.findall(r"[a-z0-9_.+-]{2,}|[\u3040-\u30ff\u3400-\u9fff]{2,}", text.lower())
    }


def search_claims(query: str, *, limit: int = 10, path: Path = CLAIM_INDEX_FILE) -> list[dict[str, Any]]:
    query_tokens = {
        token.lower()
        for token in re.findall(r"[a-z0-9_.+-]{2,}|[\u3040-\u30ff\u3400-\u9fff]{2,}", query.lower())
    }
    if not query_tokens:
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    scored: list[tuple[int, dict[str, Any]]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        score = len(query_tokens & _claim_tokens(row))
        if score:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{**row, "score": score} for score, row in scored[:limit]]


def _is_placeholder_claim(row: dict[str, Any]) -> bool:
    source_raw = row.get("source_raw")
    source_page = row.get("source_page")
    value = str(row.get("value") or "").strip().lower()
    if not isinstance(source_raw, str) or not source_raw.strip():
        return True
    if not isinstance(source_page, str) or not source_page.strip():
        return True
    if re.fullmatch(r"p\d*|foo|bar|baz", source_page.strip()):
        return True
    if value in {"", "body", "test", "placeholder"}:
        return True
    if find_page(source_page) is None:
        return True
    return False


def sanitize_claim_ledger(*, path: Path = CLAIMS_FILE, write: bool = True) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"status": "missing", "path": str(path), "kept": 0, "dropped": 0, "write": write}
    kept: list[dict[str, Any]] = []
    dropped = 0
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if not isinstance(row, dict) or _is_placeholder_claim(row):
            dropped += 1
            continue
        kept.append(row)
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in kept)
        path.write_text(payload, encoding="utf-8")
    return {"status": "ok", "path": str(path), "kept": len(kept), "dropped": dropped, "write": write}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or search the LLM Wiki claim index.")
    sub = parser.add_subparsers(dest="command", required=True)
    rebuild = sub.add_parser("rebuild", help="Rebuild derived claims from current pages.")
    rebuild.add_argument("--limit", type=int, default=0)
    rebuild.add_argument("--json", action="store_true")
    sanitize = sub.add_parser("sanitize", help="Drop placeholder or source-less ledger claims.")
    sanitize.add_argument("--no-write", action="store_true")
    sanitize.add_argument("--json", action="store_true")
    query = sub.add_parser("search", help="Search derived claims.")
    query.add_argument("query")
    query.add_argument("--limit", type=int, default=10)
    query.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "rebuild":
        data = rebuild_claim_index(limit=max(0, args.limit))
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print(f"claims\t{data['claims']}")
            print(f"path\t{data['path']}")
        return 0

    if args.command == "sanitize":
        data = sanitize_claim_ledger(write=not args.no_write)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print(f"kept\t{data['kept']}")
            print(f"dropped\t{data['dropped']}")
        return 0

    rows = search_claims(args.query, limit=max(1, args.limit))
    if args.json:
        print(json.dumps({"claims": rows}, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            print(f"{row.get('score')}\t{row.get('claim_id')}\t{row.get('value')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
