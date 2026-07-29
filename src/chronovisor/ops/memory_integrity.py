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

from chronovisor.core.frontmatter import parse as parse_frontmatter
from chronovisor.raw.raw_replay import raw_date, select_raws
from chronovisor.raw.raw_store import RawStore
from chronovisor.search import search as run_search
from chronovisor.core.store import CHRONOVISOR_ROOT

EVAL_DIR = CHRONOVISOR_ROOT / "eval"
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
SOURCE_RAW_FIELD_RE = re.compile(
    rb'(?:^|[,{])\s*"source_raw"\s*:\s*("(?:[^"\\]|\\.)*")'
)


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


def expected_terms_from_raw(
    path: Path,
    *,
    limit: int = 10,
    store: RawStore | None = None,
) -> list[str]:
    raw_store = store or RawStore(CHRONOVISOR_ROOT / "raw")
    try:
        unit = raw_store.resolve_reference(path) or raw_store.resolve(path.name)
        text = (
            raw_store.read_text(unit)
            if unit is not None
            else path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError):
        return _tokens(path.stem, limit=limit)
    try:
        _meta, body = parse_frontmatter(text)
    except Exception:
        _meta, body = {}, text
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


def claimed_raw_names(*, path: Path | None = None) -> set[str]:
    """Read only raw-source identities from the append-only claim ledger.

    The ledger can grow past hundreds of megabytes. Materializing every full
    JSON record just to read one scalar field caused gigabyte-scale transient
    memory use and multi-minute sleep cycles.
    """

    target = path or CHRONOVISOR_ROOT / "claims" / "claims.jsonl"
    out: set[str] = set()
    try:
        handle = target.open("rb")
    except OSError:
        return out
    with handle:
        for line in handle:
            match = SOURCE_RAW_FIELD_RE.search(line)
            if match is None:
                continue
            try:
                source = json.loads(match.group(1))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
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


def evaluate_raw(
    path: Path,
    *,
    claimed: set[str] | None = None,
    store: RawStore | None = None,
) -> dict[str, Any]:
    claimed = claimed if claimed is not None else claimed_raw_names()
    raw_store = store or RawStore(CHRONOVISOR_ROOT / "raw")
    terms = expected_terms_from_raw(path, store=raw_store)
    query = " ".join(terms[:8])
    evidence = search_evidence(query)
    claim_present = path.name in claimed
    top_score = max((float(item.get("score") or 0.0) for item in evidence), default=0.0)
    search_present = top_score >= SEARCH_PASS_THRESHOLD
    status = "pass" if search_present else "miss"
    unit = raw_store.resolve_reference(path) or raw_store.resolve(path.name)
    logical_bytes = unit.length if unit is not None else path.stat().st_size
    logical_date = (
        unit.captured_at[:10].replace("-", "")
        if unit is not None and unit.captured_at
        else raw_date(path)
    )
    return {
        "raw": path.name,
        "path": str(path),
        "date": logical_date,
        "host": raw_host(path),
        "bytes": logical_bytes,
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
    raw_store = RawStore(CHRONOVISOR_ROOT / "raw")
    raws = select_raws(
        since=since,
        limit=max(0, limit),
        store=raw_store,
    )
    claimed = claimed_raw_names()
    rows = [
        evaluate_raw(path, claimed=claimed, store=raw_store)
        for path in raws
    ]
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
    """Run the ``chronovisor-memory-integrity`` command-line entry point."""
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
