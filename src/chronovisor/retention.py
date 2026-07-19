"""FSRS-inspired retention scores for memory ranking."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

from chronovisor.index_store import get_store
from chronovisor.recall_log_schema import page_ids_from_record
from chronovisor.recall_runtime_paths import RECALL_DIR

RETENTION_FILE = RECALL_DIR / "retention.json"
FEEDBACK_FILE = RECALL_DIR / "feedback.jsonl"
RECALL_LOG_FILE = RECALL_DIR / "recall-log.jsonl"

TYPE_DECAY_DAYS = {
    "episodic": 21.0,
    "semantic": 365.0,
    "procedural": 180.0,
    "state": 14.0,
    "lesson": 365.0,
    "decision": 365.0,
    "knowledge": 120.0,
    "reference": 9999.0,
}


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


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _age_days(updated: object, *, today: date) -> int:
    parsed = _parse_date(updated)
    if parsed is None:
        return 999
    return max(0, (today - parsed).days)


def build_retention_scores(
    *,
    feedback_file: Path = FEEDBACK_FILE,
    recall_log_file: Path = RECALL_LOG_FILE,
    output_file: Path = RETENTION_FILE,
    limit: int = 5000,
    write: bool = True,
    today: date | None = None,
) -> dict[str, Any]:
    today = today or date.today()
    success: Counter[str] = Counter()
    lapse: Counter[str] = Counter()
    exposures: Counter[str] = Counter()

    for row in _read_recent_jsonl(feedback_file, limit=limit):
        kind = str(row.get("kind") or "")
        page_ids = page_ids_from_record(row)
        if not page_ids:
            continue
        if kind == "injection_used":
            success.update(page_ids)
        elif kind in {"injection_ignored", "false-positive", "missed", "missed_candidate"}:
            lapse.update(page_ids)
        exposures.update(page_ids)

    for row in _read_recent_jsonl(recall_log_file, limit=limit):
        page_ids = page_ids_from_record(row)
        exposures.update(page_ids)
        decision = str(row.get("decision") or "")
        if decision in {"inject", "allow"}:
            success.update(page_ids)

    store = get_store()
    store.refresh()
    pages: dict[str, dict[str, Any]] = {}
    archive_candidates: list[str] = []
    for meta in store.all_pages_meta(include_system=False):
        page_id = str(meta.get("page_id") or "")
        if not page_id:
            continue
        full = store.meta(page_id) or {}
        page_type = str(meta.get("page_type") or "knowledge")
        age = _age_days(meta.get("updated"), today=today)
        half_life = TYPE_DECAY_DAYS.get(page_type, TYPE_DECAY_DAYS["knowledge"])
        recency = math.exp(-age / max(1.0, half_life))
        use = success[page_id]
        miss = lapse[page_id]
        seen = exposures[page_id]
        backlinks = len(store.backlinks(page_id))
        outlinks = len(store.outlinks(page_id))
        link_count = backlinks + outlinks
        summary_present = bool(str(full.get("summary") or "").strip())
        questions = full.get("recall_questions")
        question_count = len(questions) if isinstance(questions, list) else 0
        stability = 1.0 + math.log1p(use) - (0.25 * miss)
        retrievability = max(0.0, min(1.0, recency * (1.0 + 0.15 * use) / (1.0 + 0.1 * miss)))
        score = max(0.0, min(1.5, (0.55 * retrievability) + (0.45 * min(1.0, stability / 3.0))))
        cold_start = min(
            0.55,
            (0.08 * math.log1p(link_count))
            + (0.08 if summary_present else 0.0)
            + (0.08 if question_count else 0.0)
            + (0.08 if age <= 90 else 0.0),
        )
        if seen == 0:
            score = max(score, cold_start)
        if page_type == "reference":
            score = 0.0
        archive_ready = (
            score < 0.18
            and age > 365
            and seen == 0
            and link_count == 0
            and not summary_present
            and not question_count
            and page_type in {"episodic", "knowledge"}
            and str(meta.get("status") or "active") == "active"
        )
        if archive_ready:
            archive_candidates.append(page_id)
        pages[page_id] = {
            "page_id": page_id,
            "page_type": page_type,
            "updated": meta.get("updated"),
            "age_days": age,
            "review_count": int(use),
            "lapse_count": int(miss),
            "exposure_count": int(seen),
            "backlinks": backlinks,
            "outlinks": outlinks,
            "cold_start_prior": round(cold_start, 4),
            "stability": round(stability, 4),
            "retrievability": round(retrievability, 4),
            "score": round(score, 4),
        }

    payload = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pages": pages,
        "archive_candidates": archive_candidates[:200],
        "counts": {
            "pages": len(pages),
            "archive_candidates": len(archive_candidates),
            "feedback_rows": len(_read_recent_jsonl(feedback_file, limit=limit)),
        },
    }
    if write:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def retention_score(page_id: str, *, path: Path = RETENTION_FILE) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    pages = payload.get("pages")
    if not isinstance(pages, dict):
        return 0.0
    row = pages.get(page_id)
    if not isinstance(row, dict):
        return 0.0
    try:
        return float(row.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build FSRS-inspired retention scores.")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    data = build_retention_scores(limit=max(1, args.limit), write=not args.no_write)
    public = {key: value for key, value in data.items() if key != "pages"}
    if args.json:
        print(json.dumps(public, ensure_ascii=False, indent=2))
    else:
        print(f"pages\t{public['counts']['pages']}")
        print(f"archive_candidates\t{public['counts']['archive_candidates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
