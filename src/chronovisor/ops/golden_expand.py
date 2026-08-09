"""Expand search golden sets from page recall questions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.index_store import get_store
from chronovisor.core.recall_runtime_paths import RECALL_DIR
from chronovisor.core.store import find_page
from chronovisor.recall.recall_answer_eval import (
    BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256,
)
from chronovisor.recall.recall_runtime import page_uid_for_id
from chronovisor.search.search_eval import assign_split, language_bucket, query_kind

GOLDEN_FILE = RECALL_DIR / "search-golden.jsonl"
LABEL_QUEUE_FILE = RECALL_DIR / "search-label-queue.jsonl"


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


def _key(row: dict[str, Any]) -> tuple[object, ...]:
    candidate_sha = str(row.get("candidate_sha256") or "")
    if candidate_sha:
        return ("candidate", candidate_sha)
    expected = row.get("expected_pages")
    pages = tuple(str(item) for item in expected) if isinstance(expected, list) else ()
    return (str(row.get("query") or "").strip(), pages, bool(row.get("negative")))


def rows_from_recall_questions(
    *,
    limit: int = 0,
    include_reference: bool = False,
    refresh_index: bool = True,
) -> list[dict[str, Any]]:
    store = get_store()
    if refresh_index:
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
        path = find_page(page_id)
        try:
            content_bytes = path.read_bytes() if path else b""
        except OSError:
            content_bytes = b""
        page_uid = page_uid_for_id(page_id)
        if not content_bytes or not page_uid:
            continue
        content_sha256 = hashlib.sha256(content_bytes).hexdigest()
        for question in questions:
            if not isinstance(question, str) or not question.strip():
                continue
            query = question.strip()
            search_eval_split = assign_split(query)
            candidate_identity = {
                "query": query,
                "expected_pages": [page_id],
                "source": "recall_questions",
                "page_uid": page_uid,
                "content_sha256": content_sha256,
                "content_byte_length": len(content_bytes),
                "projection_policy_sha256": (
                    BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256
                ),
                "search_eval_split": search_eval_split,
            }
            seed = json.dumps([query, page_id], ensure_ascii=False)
            rows.append(
                {
                    "id": "rq-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16],
                    "query": query,
                    "expected_pages": [page_id],
                    "negative": False,
                    "source": "recall_questions",
                    "source_page": page_id,
                    "split": search_eval_split,
                    "split_role": "search_eval_only_not_answer_benchmark",
                    "language": language_bucket(query),
                    "kind": query_kind(query),
                    # Page-authored recall questions are candidates, not gold.
                    # A later adopted machine-consensus receipt is required
                    # before this row can enter an authority manifest.
                    "reviewed": False,
                    "reviewer": None,
                    "queue_status": "pending_review",
                    "preregistered_at": datetime.now(UTC)
                    .isoformat(timespec="microseconds")
                    .replace("+00:00", "Z"),
                    "candidate_sha256": hashlib.sha256(
                        json.dumps(
                            candidate_identity,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                    "page_uid": page_uid,
                    "content_sha256": content_sha256,
                    "content_byte_length": len(content_bytes),
                    "projection_policy_sha256": (
                        BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256
                    ),
                }
            )
            if limit and len(rows) >= limit:
                return rows
    return rows


def expand_golden_from_recall_questions(
    *,
    golden_file: Path = GOLDEN_FILE,
    candidate_file: Path = LABEL_QUEUE_FILE,
    limit: int = 0,
    include_reference: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    existing = _read_jsonl(golden_file)
    candidates = rows_from_recall_questions(
        limit=limit,
        include_reference=include_reference,
        refresh_index=write,
    )

    def additions_for(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Only the queue suppresses duplicate machine-authority candidates.
        seen = {_key(row) for row in current}
        additions: list[dict[str, Any]] = []
        for row in candidates:
            key = _key(row)
            if key in seen:
                continue
            seen.add(key)
            additions.append(row)
        return additions

    candidates_existing = _read_jsonl(candidate_file)
    additions = additions_for(candidates_existing)
    if write:
        # Candidate generation stays outside the lock. Re-read under the shared
        # sidecar so dedupe and append are one cross-process transaction.
        with _search_label_queue_lock(candidate_file):
            candidates_existing = _read_jsonl(candidate_file)
            additions = additions_for(candidates_existing)
            if additions:
                candidate_file.parent.mkdir(parents=True, exist_ok=True)
                with candidate_file.open("a", encoding="utf-8") as f:
                    for row in additions:
                        f.write(
                            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                        )
    return {
        "status": "ok",
        "path": str(candidate_file),
        "existing": len(existing),
        "candidates": len(candidates),
        "added": len(additions),
        "write": write,
    }


def _search_label_queue_lock(path: Path):
    import contextlib
    import fcntl
    import os

    @contextlib.contextmanager
    def locked():
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            os.chmod(lock_path, 0o600)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return locked()


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
