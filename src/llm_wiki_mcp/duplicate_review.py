"""Build a review queue for likely duplicate knowledge pages."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from llm_wiki_mcp.index_store import get_store
from llm_wiki_mcp.search import _iter_all_embeddings
from llm_wiki_mcp.wiki import WIKI_ROOT


REVIEW_QUEUE = WIKI_ROOT / "review" / "duplicate-candidates.jsonl"


@dataclass(frozen=True)
class DuplicateCandidate:
    left: str
    right: str
    score: float
    method: str
    left_title: str = ""
    right_title: str = ""

    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.left, self.right)))  # type: ignore[return-value]

    def to_record(self) -> dict:
        return {
            "type": "duplicate_candidate",
            "lane": "autonomous",
            "left": self.left,
            "right": self.right,
            "left_title": self.left_title,
            "right_title": self.right_title,
            "score": round(self.score, 4),
            "method": self.method,
            "recommendation": (
                "Autonomy cycle will supersede only exact, high-confidence, reversible pairs; "
                "uncertain pairs are deferred and re-evaluated on the next sleep cycle."
            ),
        }


def _normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    normalized = text.casefold()
    normalized = re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def _knowledge_metas() -> list[dict]:
    store = get_store()
    store.refresh()
    metas: list[dict] = []
    for item in store.all_pages_meta(include_system=False):
        if item.get("page_type") == "reference":
            continue
        if item.get("status") not in (None, "active"):
            continue
        meta = store.meta(str(item.get("page_id", "")))
        if meta is not None:
            metas.append(meta)
    return metas


def title_duplicate_candidates(
    metas: Iterable[dict],
    *,
    threshold: float = 0.90,
) -> list[DuplicateCandidate]:
    records = list(metas)
    out: list[DuplicateCandidate] = []
    for idx, left in enumerate(records):
        left_title = _normalize_text(left.get("title"))
        if not left_title:
            continue
        for right in records[idx + 1 :]:
            right_title = _normalize_text(right.get("title"))
            if not right_title:
                continue
            score = SequenceMatcher(None, left_title, right_title).ratio()
            if score >= threshold:
                out.append(
                    DuplicateCandidate(
                        left=str(left.get("page_id", "")),
                        right=str(right.get("page_id", "")),
                        score=score,
                        method="title",
                        left_title=str(left.get("title", "")),
                        right_title=str(right.get("title", "")),
                    )
                )
    return out


def embedding_duplicate_candidates(
    metas: Iterable[dict],
    *,
    threshold: float = 0.92,
    limit: int = 200,
) -> list[DuplicateCandidate]:
    meta_by_id = {str(meta.get("page_id", "")): meta for meta in metas}
    rows = [
        (pid, vec, norm)
        for pid, vec, _mtime, norm in _iter_all_embeddings()
        if pid in meta_by_id and norm > 0
    ]
    out: list[DuplicateCandidate] = []
    for idx, (left_id, left_vec, left_norm) in enumerate(rows):
        for right_id, right_vec, right_norm in rows[idx + 1 :]:
            dot = math.sumprod(left_vec, right_vec)
            score = dot / (left_norm * right_norm)
            if score >= threshold:
                left = meta_by_id[left_id]
                right = meta_by_id[right_id]
                out.append(
                    DuplicateCandidate(
                        left=left_id,
                        right=right_id,
                        score=score,
                        method="embedding",
                        left_title=str(left.get("title", "")),
                        right_title=str(right.get("title", "")),
                    )
                )
    out.sort(key=lambda candidate: candidate.score, reverse=True)
    return out[:limit]


def build_duplicate_review_queue(
    *,
    title_threshold: float = 0.90,
    embedding_threshold: float = 0.92,
    include_embeddings: bool = True,
    limit: int = 200,
) -> list[dict]:
    metas = _knowledge_metas()
    candidates = title_duplicate_candidates(metas, threshold=title_threshold)
    if include_embeddings:
        try:
            candidates.extend(
                embedding_duplicate_candidates(
                    metas,
                    threshold=embedding_threshold,
                    limit=limit,
                )
            )
        except Exception:
            pass

    by_pair: dict[tuple[str, str], DuplicateCandidate] = {}
    for candidate in candidates:
        if not candidate.left or not candidate.right or candidate.left == candidate.right:
            continue
        key = candidate.key()
        current = by_pair.get(key)
        if current is None or candidate.score > current.score:
            by_pair[key] = candidate
    out = sorted(by_pair.values(), key=lambda candidate: candidate.score, reverse=True)
    return [candidate.to_record() for candidate in out[:limit]]


def write_review_queue(records: list[dict], path: Path = REVIEW_QUEUE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build LLM Wiki duplicate review queue.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--title-threshold", type=float, default=0.90)
    parser.add_argument("--embedding-threshold", type=float, default=0.92)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    records = build_duplicate_review_queue(
        title_threshold=args.title_threshold,
        embedding_threshold=args.embedding_threshold,
        include_embeddings=not args.no_embeddings,
        limit=args.limit,
    )
    payload = {"status": "ok", "count": len(records), "records": records[:20]}
    if args.write:
        payload["path"] = str(write_review_queue(records))
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
