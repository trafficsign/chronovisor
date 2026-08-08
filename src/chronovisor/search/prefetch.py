"""Speculative recall prefetch cache.

The first version is log-derived: after recall succeeds, the sleep cycle can
compile lightweight host/cwd/query-token buckets. Runtime recall checks these
buckets before expensive search expansion.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from chronovisor.core.store import CHRONOVISOR_ROOT

PREFETCH_FILE = CHRONOVISOR_ROOT / "recall" / "prefetch.json"
PREFETCH_DB_FILE = CHRONOVISOR_ROOT / "recall" / "prefetch.sqlite"


def _prefetch_from_db(
    *,
    host: str,
    cwd: str,
    queries: list[str],
    prompt: str,
    path: Path,
    limit: int,
    positive_weight: int,
    exposure_weight: int,
) -> list[str]:
    scores: Counter[str] = Counter()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        bucket = f"{host}|{Path(cwd).name if cwd else ''}"
        query_tokens = sorted(prefetch_tokens(" ".join(queries) + " " + prompt))
        for supervision, weight in (
            ("positive_used", positive_weight),
            ("exposure", exposure_weight),
        ):
            if weight <= 0:
                continue
            for page_id, count in connection.execute(
                "SELECT page_id, count FROM buckets "
                "WHERE supervision = ? AND bucket = ?",
                (supervision, bucket),
            ):
                scores[str(page_id)] += weight * int(count)
            if query_tokens:
                placeholders = ",".join("?" for _ in query_tokens)
                for page_id, count in connection.execute(
                    f"SELECT page_id, count FROM tokens "
                    f"WHERE supervision = ? AND token IN ({placeholders})",
                    (supervision, *query_tokens),
                ):
                    scores[str(page_id)] += weight * int(count)
    finally:
        connection.close()
    return [page_id for page_id, _count in scores.most_common(limit)]


def prefetch_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(
            r"[a-z0-9][a-z0-9_.+-]{2,}|[\u3040-\u30ff\u3400-\u9fff]{2,}", text.lower()
        )
        if token not in {"codex", "claude", "wiki", "llm", "project", "memory"}
    }


def prefetch_page_ids(
    *,
    host: str,
    cwd: str,
    queries: list[str],
    prompt: str = "",
    path: Path = PREFETCH_FILE,
    limit: int = 4,
    positive_weight: int = 4,
    exposure_weight: int = 1,
) -> list[str]:
    if path == PREFETCH_FILE and PREFETCH_DB_FILE.is_file():
        try:
            return _prefetch_from_db(
                host=host,
                cwd=cwd,
                queries=queries,
                prompt=prompt,
                path=PREFETCH_DB_FILE,
                limit=limit,
                positive_weight=positive_weight,
                exposure_weight=exposure_weight,
            )
        except (OSError, sqlite3.DatabaseError):
            pass
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    scores: Counter[str] = Counter()

    def score_feature(feature: Any, *, weight: int) -> None:
        if weight <= 0 or not isinstance(feature, dict):
            return
        buckets = feature.get("buckets")
        tokens = feature.get("tokens")
        if isinstance(buckets, dict):
            key = f"{host}|{Path(cwd).name if cwd else ''}"
            for row in (
                buckets.get(key, []) if isinstance(buckets.get(key), list) else []
            ):
                if isinstance(row, dict) and isinstance(row.get("page_id"), str):
                    scores[row["page_id"]] += weight * int(row.get("count") or 1)
        if isinstance(tokens, dict):
            for token in prefetch_tokens(" ".join(queries) + " " + prompt):
                rows = tokens.get(token)
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if isinstance(row, dict) and isinstance(row.get("page_id"), str):
                        scores[row["page_id"]] += weight * int(row.get("count") or 1)

    features = payload.get("features")
    if isinstance(features, dict):
        score_feature(features.get("positive_used"), weight=positive_weight)
        score_feature(features.get("exposure"), weight=exposure_weight)
    else:
        score_feature(
            {"buckets": payload.get("buckets"), "tokens": payload.get("tokens")},
            weight=exposure_weight,
        )
    return [page_id for page_id, _count in scores.most_common(limit)]


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-prefetch`` command-line entry point."""
    from chronovisor.recall.recall_prefetch import build_prefetch_cache

    parser = argparse.ArgumentParser(
        description="Build speculative recall prefetch cache."
    )
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = build_prefetch_cache(limit=max(1, args.limit), write=not args.no_write)
    public = {
        key: value
        for key, value in payload.items()
        if key not in {"buckets", "tokens", "features"}
    }
    public["bucket_count"] = len(payload.get("buckets", {}))
    public["token_count"] = len(payload.get("tokens", {}))
    public["positive_bucket_count"] = len(
        payload.get("features", {}).get("positive_used", {}).get("buckets", {})
    )
    if args.json:
        print(json.dumps(public, ensure_ascii=False, indent=2))
    else:
        print(f"episodes\t{public['episodes']}")
        print(f"buckets\t{public['bucket_count']}")
        print(f"tokens\t{public['token_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
