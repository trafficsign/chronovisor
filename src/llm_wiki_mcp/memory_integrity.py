"""Write-side memory integrity evaluation.

This is intentionally deterministic: it gives the sleep cycle a cheap signal
for raw captures that have no obvious independent search footprint yet. Claim
presence is recorded as audit evidence, but it does not make the row pass; the
expected terms are extracted from raw body text rather than ingest metadata.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp.frontmatter import parse as parse_frontmatter
from llm_wiki_mcp.raw_replay import raw_date, select_raws
from llm_wiki_mcp.search import search as run_search
from llm_wiki_mcp.wiki import WIKI_ROOT

EVAL_DIR = WIKI_ROOT / "eval"
LATEST_FILE = EVAL_DIR / "memory-integrity-latest.json"
HISTORY_FILE = EVAL_DIR / "memory-integrity-history.jsonl"
SEARCH_PASS_THRESHOLD = 0.03
GENERIC_TERMS = {
    "codex",
    "claude",
    "session",
    "memory",
    "wiki",
    "llm",
    "user",
    "assistant",
    "request",
    "response",
    "content",
    "should",
    "would",
    "about",
    "これ",
    "それ",
    "あれ",
    "する",
    "した",
    "ある",
}


def raw_host(path: Path) -> str:
    name = path.name.lower()
    if "codex" in name:
        return "codex"
    if "claude" in name:
        return "claude"
    if "cowork" in name:
        return "cowork"
    return "unknown"


def _tokens(text: str, *, limit: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_.+-]{3,}|[\u3040-\u30ff\u3400-\u9fff]{2,}", text):
        normalized = token.strip("._+-").lower()
        if len(normalized) < 2 or normalized in GENERIC_TERMS or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def expected_terms_from_raw(path: Path, *, limit: int = 10) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return _tokens(path.stem, limit=limit)
    try:
        meta, body = parse_frontmatter(text)
    except Exception:
        meta, body = {}, text
    seeds: list[str] = []
    # Do not use frontmatter keywords/raw_keywords/entities here: those are
    # produced by the same ingest path we are trying to audit, so they make the
    # metric circular. Body text is a weaker but independent signal.
    seeds.extend(_tokens(body[:6000], limit=limit * 2))
    if not seeds:
        seeds.extend(_tokens(path.stem.replace("-", " "), limit=limit))
    deduped: list[str] = []
    seen: set[str] = set()
    for term in seeds:
        normalized = term.strip().lower()
        if not normalized or normalized in GENERIC_TERMS or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) >= limit:
            break
    return deduped


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def claimed_raw_names() -> set[str]:
    out: set[str] = set()
    for row in _read_jsonl(WIKI_ROOT / "claims" / "claims.jsonl"):
        source = row.get("source_raw")
        if not isinstance(source, str) or not source.strip():
            continue
        name = source.removeprefix("replay:").strip()
        if name:
            out.add(Path(name).name)
    return out


def search_evidence(query: str, *, top_n: int = 5) -> list[dict[str, Any]]:
    if not query:
        return []
    try:
        results, _mode = run_search(query=query, top_n=top_n, semantic=True)
    except Exception:
        return []
    return [
        {
            "page_id": item.page_id,
            "title": item.title,
            "score": round(float(item.score), 4),
        }
        for item in results[:top_n]
    ]


def evaluate_raw(path: Path, *, claimed: set[str] | None = None) -> dict[str, Any]:
    claimed = claimed if claimed is not None else claimed_raw_names()
    terms = expected_terms_from_raw(path)
    query = " ".join(terms[:8])
    evidence = search_evidence(query)
    claim_present = path.name in claimed
    top_score = max((float(item.get("score") or 0.0) for item in evidence), default=0.0)
    search_present = top_score >= SEARCH_PASS_THRESHOLD
    status = "pass" if search_present else "miss"
    return {
        "raw": path.name,
        "path": str(path),
        "date": raw_date(path),
        "host": raw_host(path),
        "bytes": path.stat().st_size,
        "terms": terms,
        "query": query,
        "claim_present": claim_present,
        "search_present": search_present,
        "claim_present_is_audit_only": True,
        "top_score": round(top_score, 4),
        "search_pass_threshold": SEARCH_PASS_THRESHOLD,
        "top_pages": evidence,
        "status": status,
    }


def run_eval(
    *,
    since: str = "",
    limit: int = 100,
    write: bool = True,
) -> dict[str, Any]:
    raws = select_raws(since=since, limit=max(0, limit))
    claimed = claimed_raw_names()
    rows = [evaluate_raw(path, claimed=claimed) for path in raws]
    total = len(rows)
    passed = sum(1 for row in rows if row["status"] == "pass")
    by_host: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_host.setdefault(str(row["host"]), {"total": 0, "passed": 0, "missed": 0})
        bucket["total"] += 1
        if row["status"] == "pass":
            bucket["passed"] += 1
        else:
            bucket["missed"] += 1
    payload = {
        "status": "ok",
        "method": "independent_raw_body_search",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "passed": passed,
        "missed": total - passed,
        "capture_rate": (passed / total) if total else None,
        "by_host": by_host,
        "rows": rows,
    }
    if write:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        LATEST_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: v for k, v in payload.items() if k != "rows"}, ensure_ascii=False) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate raw capture memory integrity.")
    parser.add_argument("--since", default="")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_eval(since=args.since, limit=max(0, args.limit), write=not args.no_write)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"capture_rate\t{payload['capture_rate']}")
        print(f"passed\t{payload['passed']}")
        print(f"missed\t{payload['missed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
