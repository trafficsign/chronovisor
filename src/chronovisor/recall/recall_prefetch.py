"""Build and persist the speculative recall prefetch cache."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.recall.recall_log_schema import (
    canonicalize_page_ids,
    join_used_recall_episodes,
    page_ids_from_record,
)
from chronovisor.recall.recall_runtime_paths import RECALL_DIR
from chronovisor.search.prefetch import (
    PREFETCH_DB_FILE,
    PREFETCH_FILE,
    prefetch_tokens,
)

RECALL_LOG_FILE = RECALL_DIR / "recall-log.jsonl"
RECALL_PULL_LOG_FILE = RECALL_DIR / "pull-log.jsonl"


def _write_prefetch_db(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".prefetch-", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        connection = sqlite3.connect(temporary_path)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                CREATE TABLE buckets (
                    supervision TEXT NOT NULL,
                    bucket TEXT NOT NULL,
                    page_id TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (supervision, bucket, page_id)
                ) WITHOUT ROWID;
                CREATE TABLE tokens (
                    supervision TEXT NOT NULL,
                    token TEXT NOT NULL,
                    page_id TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (supervision, token, page_id)
                ) WITHOUT ROWID;
                """
            )
            features = payload.get("features")
            if isinstance(features, dict):
                for supervision in ("positive_used", "exposure"):
                    feature = features.get(supervision)
                    if not isinstance(feature, dict):
                        continue
                    for bucket, rows in (feature.get("buckets") or {}).items():
                        connection.executemany(
                            "INSERT INTO buckets VALUES (?, ?, ?, ?)",
                            (
                                (
                                    supervision,
                                    str(bucket),
                                    str(row["page_id"]),
                                    int(row.get("count") or 1),
                                )
                                for row in rows
                                if isinstance(row, dict) and row.get("page_id")
                            ),
                        )
                    for token, rows in (feature.get("tokens") or {}).items():
                        connection.executemany(
                            "INSERT INTO tokens VALUES (?, ?, ?, ?)",
                            (
                                (
                                    supervision,
                                    str(token),
                                    str(row["page_id"]),
                                    int(row.get("count") or 1),
                                )
                                for row in rows
                                if isinstance(row, dict) and row.get("page_id")
                            ),
                        )
            connection.commit()
        finally:
            connection.close()
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_recent_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=max(1, limit))
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def build_prefetch_cache(
    *,
    log_file: Path = RECALL_LOG_FILE,
    pull_log_file: Path = RECALL_PULL_LOG_FILE,
    output_file: Path = PREFETCH_FILE,
    limit: int = 5000,
    write: bool = True,
) -> dict[str, Any]:
    recall_rows = _read_recent_jsonl(log_file, limit=limit)
    joined = join_used_recall_episodes(
        recall_rows,
        _read_recent_jsonl(pull_log_file, limit=limit),
    )
    from chronovisor.core.alias_store import load_aliases

    aliases = load_aliases()

    def compile_rows(
        rows: list[tuple[dict[str, Any], list[str]]],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], int]:
        buckets: dict[str, Counter[str]] = defaultdict(Counter)
        token_index: dict[str, Counter[str]] = defaultdict(Counter)
        episodes = 0
        for row, pages in rows:
            if not pages:
                continue
            episodes += 1
            host = str(row.get("host") or "")
            cwd = str(row.get("cwd") or "")
            bucket_key = f"{host}|{Path(cwd).name if cwd else ''}"
            for page_id in pages:
                buckets[bucket_key][page_id] += 1
            query_text = " ".join(str(item) for item in row.get("queries", []) or [])
            query_text += " " + str(
                row.get("prompt_preview") or row.get("prompt") or ""
            )
            for token in prefetch_tokens(query_text):
                for page_id in pages:
                    token_index[token][page_id] += 1
        return (
            {
                key: [
                    {"page_id": page_id, "count": count}
                    for page_id, count in counter.most_common(12)
                ]
                for key, counter in buckets.items()
            },
            {
                key: [
                    {"page_id": page_id, "count": count}
                    for page_id, count in counter.most_common(8)
                ]
                for key, counter in token_index.items()
            },
            episodes,
        )

    exposure_rows = [
        (row, canonicalize_page_ids(page_ids_from_record(row), aliases))
        for row in recall_rows
        if page_ids_from_record(row)
    ]
    positive_rows = [
        (
            episode["recall"],
            canonicalize_page_ids(episode["page_ids"], aliases),
        )
        for episode in joined["episodes"]
    ]
    exposure_buckets, exposure_tokens, exposure_episodes = compile_rows(exposure_rows)
    positive_buckets, positive_tokens, positive_episodes = compile_rows(positive_rows)

    payload = {
        "schema_version": 2,
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "episodes": exposure_episodes,
        "positive_episodes": positive_episodes,
        "join": {key: value for key, value in joined.items() if key != "episodes"},
        "features": {
            "positive_used": {
                "supervision": "explicit_used_receipt",
                "buckets": positive_buckets,
                "tokens": positive_tokens,
            },
            "exposure": {
                "supervision": "recalled_not_confirmed_used",
                "buckets": exposure_buckets,
                "tokens": exposure_tokens,
            },
        },
        # Compatibility aliases are explicitly exposure-only. They are not
        # positive labels and may be removed after all consumers read v2.
        "buckets": exposure_buckets,
        "tokens": exposure_tokens,
    }
    if write:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file == PREFETCH_FILE:
            _write_prefetch_db(payload, PREFETCH_DB_FILE)
            persisted = {
                key: value
                for key, value in payload.items()
                if key not in {"features", "buckets", "tokens"}
            }
            persisted.update(
                {
                    "storage": "sqlite",
                    "database": str(PREFETCH_DB_FILE),
                    "bucket_count": len(payload.get("buckets", {})),
                    "token_count": len(payload.get("tokens", {})),
                    "positive_bucket_count": len(
                        payload.get("features", {})
                        .get("positive_used", {})
                        .get("buckets", {})
                    ),
                    "positive_token_count": len(
                        payload.get("features", {})
                        .get("positive_used", {})
                        .get("tokens", {})
                    ),
                }
            )
        else:
            persisted = payload
        output_file.write_text(
            json.dumps(persisted, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload
