"""Build a review queue for likely duplicate knowledge pages."""

from __future__ import annotations

import argparse
import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.search.index_store import get_store
from chronovisor.search.search import _iter_all_embeddings

REVIEW_QUEUE = CHRONOVISOR_ROOT / "review" / "duplicate-candidates.jsonl"


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


def _temporal_family(page_id: str) -> str | None:
    """Return a stable family for dated snapshots that must coexist."""
    match = re.match(r"^(.+?)-(?:19|20)\d{2}-\d{2}-\d{2}(?:-.+)?$", page_id)
    if not match:
        return None
    family = match.group(1)
    return (
        family
        if family in {"memory-reflection", "job-change-status", "current-state"}
        else None
    )


def _title_grams(title: str, *, width: int = 3) -> set[str]:
    if len(title) <= width:
        return {title}
    return {title[index : index + width] for index in range(len(title) - width + 1)}


def _blocked_title_pairs(
    titles: list[str],
    *,
    max_posting: int = 96,
    signatures_per_title: int = 8,
    max_pairs: int = 100_000,
) -> list[tuple[int, int]]:
    """Return deterministic, bounded candidates for a large title corpus.

    A full title cross-product made every sleep cycle quadratic in the number
    of pages. Near-identical titles share several character trigrams, so rare
    trigrams provide a high-recall blocking key while common trigrams are
    deliberately capped. Embedding candidates remain the independent fallback
    for reordered or otherwise lexically unusual duplicates.
    """

    grams_by_index = [_title_grams(title) for title in titles]
    frequencies = Counter(gram for grams in grams_by_index for gram in grams if gram)
    exact_titles: dict[str, list[int]] = defaultdict(list)
    for index, title in enumerate(titles):
        if title:
            exact_titles[title].append(index)
    pairs: set[tuple[int, int]] = {
        (left_index, right_index)
        for indices in exact_titles.values()
        for left_index, right_index in zip(indices, indices[1:], strict=False)
    }

    postings: dict[str, list[int]] = defaultdict(list)
    for index, grams in enumerate(grams_by_index):
        signatures = sorted(
            (gram for gram in grams if frequencies[gram] <= max_posting),
            key=lambda gram: (frequencies[gram], gram),
        )[:signatures_per_title]
        for gram in signatures:
            postings[gram].append(index)

    for gram in sorted(postings, key=lambda value: (frequencies[value], value)):
        indices = postings[gram]
        for offset, left_index in enumerate(indices):
            for right_index in indices[offset + 1 :]:
                pairs.add((left_index, right_index))
                if len(pairs) >= max_pairs:
                    return sorted(pairs)
    return sorted(pairs)


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
    normalized_titles = [_normalize_text(record.get("title")) for record in records]
    if len(records) <= 500:
        pairs = [
            (left_index, right_index)
            for left_index in range(len(records))
            for right_index in range(left_index + 1, len(records))
        ]
    else:
        pairs = _blocked_title_pairs(normalized_titles)
    out: list[DuplicateCandidate] = []
    for left_index, right_index in pairs:
        left = records[left_index]
        right = records[right_index]
        left_title = normalized_titles[left_index]
        right_title = normalized_titles[right_index]
        if not left_title or not right_title:
            continue
        # SequenceMatcher cannot exceed this length-only upper bound.
        if (2 * min(len(left_title), len(right_title))) / (
            len(left_title) + len(right_title)
        ) < threshold:
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
    if not rows or limit <= 0:
        return []
    dimensions = Counter(len(vector) for _pid, vector, _norm in rows)
    dimension = dimensions.most_common(1)[0][0]
    rows = [row for row in rows if len(row[1]) == dimension]
    matrix = np.asarray([row[1] for row in rows], dtype=np.float64)
    norms = np.asarray([row[2] for row in rows], dtype=np.float64)
    matrix /= norms[:, np.newaxis]

    # Exact cosine search remains deterministic, but BLAS evaluates it in
    # bounded blocks instead of millions of Python-level vector loops.
    heap: list[tuple[float, int, int]] = []
    block_size = 256
    for start in range(0, len(rows), block_size):
        stop = min(start + block_size, len(rows))
        scores = matrix[start:stop] @ matrix.T
        for local_left, right_index in np.argwhere(scores >= threshold):
            left_index = start + int(local_left)
            right_index = int(right_index)
            if right_index <= left_index:
                continue
            score = float(scores[local_left, right_index])
            item = (score, left_index, right_index)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    out: list[DuplicateCandidate] = []
    for score, left_index, right_index in sorted(heap, reverse=True):
        left_id = rows[left_index][0]
        right_id = rows[right_index][0]
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
    return out


def build_duplicate_review_queue(
    *,
    title_threshold: float = 0.90,
    embedding_threshold: float = 0.92,
    include_embeddings: bool = True,
    limit: int = 200,
    strict: bool = False,
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
            if strict:
                raise

    by_pair: dict[tuple[str, str], DuplicateCandidate] = {}
    for candidate in candidates:
        if (
            not candidate.left
            or not candidate.right
            or candidate.left == candidate.right
        ):
            continue
        left_family = _temporal_family(candidate.left)
        if left_family and left_family == _temporal_family(candidate.right):
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
    """Run the ``chronovisor-duplicate-review`` command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Build Chronovisor duplicate review queue."
    )
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
