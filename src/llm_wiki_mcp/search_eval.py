"""Search ranking evaluation for LLM Wiki.

This is intentionally separate from ``recall_eval.py``. Recall eval measures
whether the synchronous gate injects useful context; this module measures the
ranking quality of search candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp.search import (
    DEFAULT_FUSION_WEIGHTS,
    ScoredPage,
    apply_filters,
    fuse_results,
    get_bm25,
    graph_expand_results,
    semantic_search,
    usage_prior_results,
)
from llm_wiki_mcp.reranker import rerank_results
from llm_wiki_mcp.negative_feedback import apply_penalties, penalties_for_query
from llm_wiki_mcp.runtime_config import load_negative_feedback_config, load_reranker_config
from llm_wiki_mcp.wiki import WIKI_ROOT


RECALL_DIR = WIKI_ROOT / "recall"
RECALL_FEEDBACK_FILE = RECALL_DIR / "feedback.jsonl"
RECALL_LOG_FILE = RECALL_DIR / "recall-log.jsonl"
GOLDEN_FILE = RECALL_DIR / "search-golden.jsonl"
LABEL_QUEUE_FILE = RECALL_DIR / "search-label-queue.jsonl"
FAILURE_INDEX_FILE = RECALL_DIR / "search-failures.jsonl"
BASELINE_DIR = WIKI_ROOT / "runtime" / "search-eval"
SELF_TUNE_HISTORY_FILE = BASELINE_DIR / "self-tune-history.jsonl"

DEFAULT_VARIANTS = (
    "bm25",
    "semantic",
    "hybrid-current",
    "hybrid-plain-rrf",
    "hybrid-graph",
)


@dataclass(frozen=True)
class SearchExample:
    query: str
    expected_pages: tuple[str, ...] = ()
    negative_pages: tuple[str, ...] = ()
    stale_pages: tuple[str, ...] = ()
    split: str = "dev"
    language: str = "unknown"
    kind: str = "manual"
    source: str = "manual"
    ref: str = ""
    ts: str = ""
    reviewed: bool = False

    @property
    def positive(self) -> bool:
        return bool(self.expected_pages)

    @property
    def bad_pages(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.negative_pages + self.stale_pages))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(item for item in value if isinstance(item, str) and item))


def _top_page_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("page_id"), str):
            out.append(item["page_id"])
        elif isinstance(item, str):
            out.append(item)
    return tuple(dict.fromkeys(out))


def language_bucket(text: str) -> str:
    has_cjk = any(
        ("\u3040" <= ch <= "\u30ff")
        or ("\u3400" <= ch <= "\u4dbf")
        or ("\u4e00" <= ch <= "\u9fff")
        or ("\uff66" <= ch <= "\uff9f")
        for ch in text
    )
    has_ascii_word = any(("a" <= ch.lower() <= "z") for ch in text)
    if has_cjk and has_ascii_word:
        return "mixed"
    if has_cjk:
        return "ja"
    if has_ascii_word:
        return "en"
    return "unknown"


def query_kind(text: str) -> str:
    compact = text.strip()
    if len(compact) <= 24:
        return "short"
    if "?" in compact or "？" in compact:
        return "question"
    if any(token in compact for token in ("```", "def ", "class ", "import ", "pytest", "uv run")):
        return "code"
    return "statement"


def assign_split(seed: str) -> str:
    bucket = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 2:
        return "locked-test"
    if bucket < 4:
        return "dev"
    return "train"


def build_candidates(
    *,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    log_file: Path = RECALL_LOG_FILE,
    limit: int = 100,
) -> list[SearchExample]:
    logs_by_id = {
        str(row.get("decision_id", "")): row
        for row in read_jsonl(log_file)
        if row.get("decision_id")
    }
    positive_examples: list[SearchExample] = []
    negative_examples: list[SearchExample] = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()

    for feedback in read_jsonl(feedback_file):
        kind = str(feedback.get("kind", ""))
        if kind not in {
            "missed",
            "missed_candidate",
            "injection_used",
            "injection_ignored",
            "false-positive",
        }:
            continue

        ref = str(feedback.get("ref", ""))
        snapshot = feedback.get("snapshot") if isinstance(feedback.get("snapshot"), dict) else {}
        record = logs_by_id.get(ref) or snapshot or {}
        query = str(feedback.get("prompt") or record.get("prompt_preview") or "").strip()
        if not query:
            continue

        raw_expected = _str_tuple(feedback.get("expected_pages"))
        raw_injected = (
            _str_tuple(feedback.get("injected_pages"))
            or _str_tuple(record.get("pages"))
            or _top_page_ids(feedback.get("top_pages"))
        )

        expected: tuple[str, ...] = ()
        negative: tuple[str, ...] = ()
        if kind in {"missed", "missed_candidate"}:
            expected = raw_expected or raw_injected
        elif kind == "injection_used":
            expected = raw_expected or raw_injected
        else:
            negative = raw_injected or raw_expected

        if not expected and not negative:
            continue
        key = (query, expected, negative)
        if key in seen:
            continue
        seen.add(key)

        seed = json.dumps([query, expected, negative], ensure_ascii=False, sort_keys=True)
        example = SearchExample(
            query=query,
            expected_pages=expected,
            negative_pages=negative,
            split=assign_split(seed),
            language=language_bucket(query),
            kind=kind,
            source=str(feedback.get("source") or "feedback"),
            ref=ref,
            ts=str(feedback.get("ts") or record.get("ts") or ""),
            reviewed=False,
        )
        if negative:
            negative_examples.append(example)
        else:
            positive_examples.append(example)

    if limit <= 0:
        return []
    if not negative_examples:
        return positive_examples[:limit]
    negative_quota = min(len(negative_examples), max(1, limit // 5))
    positive_quota = max(0, limit - negative_quota)
    return positive_examples[:positive_quota] + negative_examples[:negative_quota]


def load_examples(path: Path = GOLDEN_FILE) -> list[SearchExample]:
    examples: list[SearchExample] = []
    for row in read_jsonl(path):
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        expected = _str_tuple(row.get("expected_pages"))
        negative = _str_tuple(row.get("negative_pages"))
        stale = _str_tuple(row.get("stale_pages"))
        if not expected and not negative and not stale:
            continue
        examples.append(
            SearchExample(
                query=query,
                expected_pages=expected,
                negative_pages=negative,
                stale_pages=stale,
                split=str(row.get("split") or assign_split(query)),
                language=str(row.get("language") or language_bucket(query)),
                kind=str(row.get("kind") or query_kind(query)),
                source=str(row.get("source") or "manual"),
                ref=str(row.get("ref") or ""),
                ts=str(row.get("ts") or ""),
                reviewed=bool(row.get("reviewed", False)),
            )
        )
    return examples


def examples_to_rows(examples: list[SearchExample]) -> list[dict[str, Any]]:
    return [
        {
            **asdict(example),
            "expected_pages": list(example.expected_pages),
            "negative_pages": list(example.negative_pages),
            "stale_pages": list(example.stale_pages),
        }
        for example in examples
    ]


def _plain_rrf(
    channels: list[tuple[str, list[ScoredPage]]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[ScoredPage]:
    weights = weights or {}
    scores: dict[str, float] = {}
    meta: dict[str, ScoredPage] = {}
    for channel, results in channels:
        weight = max(0.0, float(weights.get(channel, 1.0)))
        if weight == 0:
            continue
        for rank, page in enumerate(results):
            scores[page.page_id] = scores.get(page.page_id, 0.0) + weight / (k + rank)
            meta.setdefault(page.page_id, page)

    fused: list[ScoredPage] = []
    for page_id, score in scores.items():
        page = meta[page_id]
        fused.append(
            ScoredPage(
                page_id=page.page_id,
                title=page.title,
                folder=page.folder,
                updated=page.updated,
                score=score,
                status=page.status,
                superseded_by=page.superseded_by,
            )
        )
    return sorted(fused, key=lambda page: page.score, reverse=True)


def run_variant(query: str, variant: str, *, top_n: int = 20) -> dict[str, Any]:
    fetch_n = max(top_n * 5, 100)
    started = time.perf_counter()

    bm25 = get_bm25()
    bm25.build()
    bm25_results = bm25.query(query, top_n=fetch_n)
    sem_results: list[ScoredPage] = []
    graph_results: list[ScoredPage] = []
    usage_results: list[ScoredPage] = []

    if variant in {
        "semantic",
        "hybrid-current",
        "hybrid-plain-rrf",
        "hybrid-graph",
        "hybrid-rerank",
    }:
        sem_results = semantic_search(query, top_n=fetch_n)
    if variant == "hybrid-graph":
        graph_results = graph_expand_results(
            bm25_results + sem_results,
            decay=0.5,
            limit=fetch_n,
        )
    if variant == "hybrid-usage":
        sem_results = semantic_search(query, top_n=fetch_n)
        candidate_ids = {page.page_id for page in bm25_results + sem_results}
        usage_results = usage_prior_results(candidate_ids, limit=fetch_n)

    reranker_meta: dict[str, Any] = {"status": "not_requested"}

    if variant == "bm25":
        results = bm25_results
    elif variant == "semantic":
        results = sem_results
    elif variant == "hybrid-plain-rrf":
        results = _plain_rrf(
            [("bm25", bm25_results), ("semantic", sem_results)],
            weights={"bm25": 1.0, "semantic": 1.0},
        )
    elif variant == "hybrid-graph":
        results = fuse_results(
            bm25_results,
            sem_results,
            graph_results,
            [],
            weights={**DEFAULT_FUSION_WEIGHTS, "graph": 0.5, "usage_prior": 0.0},
        )
    elif variant == "hybrid-usage":
        results = fuse_results(
            bm25_results,
            sem_results,
            [],
            usage_results,
            weights={**DEFAULT_FUSION_WEIGHTS, "graph": 0.0, "usage_prior": 0.2},
        )
    elif variant == "hybrid-current":
        results = fuse_results(
            bm25_results,
            sem_results,
            [],
            [],
            weights=DEFAULT_FUSION_WEIGHTS,
        )
    elif variant == "hybrid-rerank":
        results = fuse_results(
            bm25_results,
            sem_results,
            [],
            [],
            weights=DEFAULT_FUSION_WEIGHTS,
        )
        rerank_outcome = rerank_results(
            query,
            apply_filters(results),
            config=load_reranker_config(),
        )
        results = rerank_outcome.results
        reranker_meta = rerank_outcome.metadata
    else:
        raise ValueError(f"unknown search eval variant: {variant}")

    # Match production search(): query-conditioned negative feedback demotion.
    negative_meta: dict[str, Any] = {"status": "disabled"}
    negative_config = load_negative_feedback_config()
    if negative_config.enabled:
        penalties = penalties_for_query(query, negative_config)
        if penalties:
            results = apply_penalties(results, penalties)
            negative_meta = {"status": "applied", "pages": sorted(penalties)}
        else:
            negative_meta = {"status": "no_match"}

    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    out = apply_filters(results)[:top_n]
    return {
        "variant": variant,
        "results": out,
        "latency_ms": elapsed_ms,
        "channels": {
            "bm25": [page.page_id for page in bm25_results[:top_n]],
            "semantic": [page.page_id for page in sem_results[:top_n]],
            "graph": [page.page_id for page in graph_results[:top_n]],
            "usage_prior": [page.page_id for page in usage_results[:top_n]],
            "reranker": reranker_meta,
            "negative_feedback": negative_meta,
        },
    }


def _dcg(ranks: list[int], *, k: int) -> float:
    return sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= k)


def _ideal_dcg(relevant_count: int, *, k: int) -> float:
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, min(relevant_count, k) + 1))


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["expected_pages"]]
    negative_labeled = [row for row in rows if row["negative_pages"]]
    stale_labeled = [row for row in rows if row["stale_pages"]]
    latencies = [int(row["latency_ms"]) for row in rows]

    def recall_at(k: int) -> float:
        if not positives:
            return 0.0
        return sum(bool(set(row["expected_pages"]) & set(row["result_pages"][:k])) for row in positives) / len(positives)

    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for row in positives:
        expected = set(row["expected_pages"])
        ranks = [
            idx
            for idx, page_id in enumerate(row["result_pages"], start=1)
            if page_id in expected
        ]
        reciprocal_ranks.append((1.0 / ranks[0]) if ranks and ranks[0] <= 10 else 0.0)
        ideal = _ideal_dcg(len(expected), k=10)
        ndcgs.append((_dcg(ranks, k=10) / ideal) if ideal else 0.0)

    negative_hits = 0
    for row in negative_labeled:
        if set(row["negative_pages"]) & set(row["result_pages"][:20]):
            negative_hits += 1

    stale_hits = 0
    for row in stale_labeled:
        if set(row["stale_pages"]) & set(row["result_pages"][:20]):
            stale_hits += 1

    return {
        "examples": len(rows),
        "positives": len(positives),
        "negative_label_examples": len(negative_labeled),
        "stale_label_examples": len(stale_labeled),
        "recall_at_5": recall_at(5),
        "recall_at_20": recall_at(20),
        "mrr_at_10": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "ndcg_at_10": statistics.mean(ndcgs) if ndcgs else 0.0,
        "negative_hit_rate_at_20": (
            negative_hits / len(negative_labeled)
        ) if negative_labeled else 0.0,
        "stale_hit_rate_at_20": (stale_hits / len(stale_labeled)) if stale_labeled else 0.0,
        "latency_ms": {
            "p50": float(statistics.median(latencies)) if latencies else 0.0,
            "p95": percentile(latencies, 0.95),
            "max": float(max(latencies)) if latencies else 0.0,
        },
    }


def _bucket_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {"all": rows}
    for row in rows:
        for key in (
            f"split:{row['split']}",
            f"language:{row['language']}",
            f"kind:{row['kind']}",
        ):
            buckets.setdefault(key, []).append(row)
    return {key: _metrics(value) for key, value in sorted(buckets.items())}


def evaluate_examples(
    examples: list[SearchExample],
    *,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    top_n: int = 20,
) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    debug_rows: list[dict[str, Any]] = []
    for variant in variants:
        rows: list[dict[str, Any]] = []
        for example in examples:
            result = run_variant(example.query, variant, top_n=top_n)
            result_pages = [page.page_id for page in result["results"]]
            row = {
                "query": example.query,
                "split": example.split,
                "language": example.language,
                "kind": example.kind,
                "source": example.source,
                "reviewed": example.reviewed,
                "expected_pages": list(example.expected_pages),
                "negative_pages": list(example.negative_pages),
                "stale_pages": list(example.stale_pages),
                "bad_pages": list(example.bad_pages),
                "result_pages": result_pages,
                "latency_ms": result["latency_ms"],
            }
            rows.append(row)
            debug_rows.append({**row, "variant": variant, "channels": result["channels"]})
        by_variant[variant] = {
            "metrics": _metrics(rows),
            "by_bucket": _bucket_metrics(rows),
        }
    return {"variants": by_variant, "debug_rows": debug_rows}


def save_baseline(payload: dict[str, Any], *, baseline_dir: Path = BASELINE_DIR) -> Path:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = baseline_dir / f"search-baseline-{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def build_golden(
    *,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    log_file: Path = RECALL_LOG_FILE,
    output_file: Path = GOLDEN_FILE,
    limit: int = 100,
) -> dict[str, Any]:
    examples = build_candidates(feedback_file=feedback_file, log_file=log_file, limit=limit)
    write_jsonl(output_file, examples_to_rows(examples))
    return {
        "status": "ok",
        "output_file": str(output_file),
        "examples": len(examples),
        "reviewed": sum(1 for example in examples if example.reviewed),
        "splits": _count_by(examples, "split"),
        "languages": _count_by(examples, "language"),
        "kinds": _count_by(examples, "kind"),
    }


def build_label_queue(
    *,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    log_file: Path = RECALL_LOG_FILE,
    output_file: Path = LABEL_QUEUE_FILE,
    limit: int = 100,
) -> dict[str, Any]:
    examples = build_candidates(feedback_file=feedback_file, log_file=log_file, limit=limit)
    rows = []
    for row in examples_to_rows(examples):
        rows.append(
            {
                **row,
                "queue_status": "pending_review",
                "promoted_to_golden": False,
                "review_note": "",
            }
        )
    write_jsonl(output_file, rows)
    return {
        "status": "ok",
        "output_file": str(output_file),
        "examples": len(rows),
        "reviewed": 0,
        "note": "Candidates are not added to search-golden.jsonl until human review.",
    }


def _count_by(examples: list[SearchExample], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        value = str(getattr(example, field))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _failure_index_rows(debug_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in debug_rows:
        expected = [page for page in row.get("expected_pages", []) if isinstance(page, str)]
        if not expected:
            continue
        result_pages = [page for page in row.get("result_pages", []) if isinstance(page, str)]
        if set(expected) & set(result_pages[:20]):
            continue
        channels = row.get("channels") if isinstance(row.get("channels"), dict) else {}
        channel_candidates: dict[str, list[str]] = {}
        channel_hit = False
        for name, values in channels.items():
            if not isinstance(values, list):
                continue
            pages = [page for page in values[:20] if isinstance(page, str)]
            channel_candidates[str(name)] = pages
            if set(expected) & set(pages):
                channel_hit = True

        failed_stage = "fusion" if channel_hit else "retrieval"
        rows.append(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "variant": row.get("variant", ""),
                "query": row.get("query", ""),
                "split": row.get("split", ""),
                "language": row.get("language", ""),
                "kind": row.get("kind", ""),
                "expected_pages": expected,
                "result_pages": result_pages[:20],
                "channel_candidates": channel_candidates,
                "failed_stage": failed_stage,
                "reason_code": "fusion_missed" if channel_hit else "retrieval_missed",
                "fix_kind": "fusion" if channel_hit else "data_or_rewrite",
            }
        )
    return rows


def write_failure_index(debug_rows: list[dict[str, Any]], path: Path = FAILURE_INDEX_FILE) -> dict[str, Any]:
    rows = _failure_index_rows(debug_rows)
    write_jsonl(path, rows)
    return {"path": str(path), "failures": len(rows)}


def run_weighted_hybrid(query: str, weights: dict[str, float], *, top_n: int = 20) -> dict[str, Any]:
    fetch_n = max(top_n * 5, 100)
    started = time.perf_counter()
    bm25 = get_bm25()
    bm25.build()
    bm25_results = bm25.query(query, top_n=fetch_n)
    sem_results = semantic_search(query, top_n=fetch_n)
    results = fuse_results(bm25_results, sem_results, [], [], weights=weights)
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    return {"results": apply_filters(results)[:top_n], "latency_ms": elapsed_ms}


def _rows_for_weight_eval(examples: list[SearchExample], weights: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        result = run_weighted_hybrid(example.query, weights, top_n=20)
        rows.append(
            {
                "query": example.query,
                "split": example.split,
                "language": example.language,
                "kind": example.kind,
                "source": example.source,
                "reviewed": example.reviewed,
                "expected_pages": list(example.expected_pages),
                "negative_pages": list(example.negative_pages),
                "stale_pages": list(example.stale_pages),
                "bad_pages": list(example.bad_pages),
                "result_pages": [page.page_id for page in result["results"]],
                "latency_ms": result["latency_ms"],
            }
        )
    return rows


def self_tune(
    *,
    golden_file: Path = GOLDEN_FILE,
    history_file: Path = SELF_TUNE_HISTORY_FILE,
) -> dict[str, Any]:
    examples = load_examples(golden_file)
    dev = [example for example in examples if example.split == "dev"] or examples
    locked = [example for example in examples if example.split == "locked-test"] or examples
    baseline_weights = dict(DEFAULT_FUSION_WEIGHTS)
    baseline_dev = _metrics(_rows_for_weight_eval(dev, baseline_weights))
    baseline_locked = _metrics(_rows_for_weight_eval(locked, baseline_weights))

    candidates = []
    for semantic_weight in (0.4, 0.5, 0.6, 0.7, 0.8):
        weights = {**DEFAULT_FUSION_WEIGHTS, "semantic": semantic_weight}
        dev_metrics = _metrics(_rows_for_weight_eval(dev, weights))
        candidates.append({"weights": weights, "dev": dev_metrics})
    best = max(
        candidates,
        key=lambda item: (
            item["dev"]["mrr_at_10"],
            item["dev"]["recall_at_5"],
            item["dev"]["ndcg_at_10"],
        ),
    )
    locked_metrics = _metrics(_rows_for_weight_eval(locked, best["weights"]))
    locked_ok = (
        locked_metrics["recall_at_5"] >= baseline_locked["recall_at_5"]
        and locked_metrics["mrr_at_10"] >= baseline_locked["mrr_at_10"]
        and locked_metrics["stale_hit_rate_at_20"] <= baseline_locked["stale_hit_rate_at_20"]
        and locked_metrics["negative_hit_rate_at_20"] <= baseline_locked["negative_hit_rate_at_20"]
    )
    dev_improved = best["dev"]["mrr_at_10"] > baseline_dev["mrr_at_10"]
    status = "shadow_pass" if dev_improved and locked_ok else "blocked"
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "applied": False,
        "reason": "" if status == "shadow_pass" else "dev improvement or locked-test guard failed",
        "baseline": {"dev": baseline_dev, "locked-test": baseline_locked},
        "best": {"weights": best["weights"], "dev": best["dev"], "locked-test": locked_metrics},
        "guardrails": {
            "dev_improved": dev_improved,
            "locked_non_degrading": locked_ok,
            "apply_policy": "shadow_only",
        },
    }
    append_jsonl(history_file, record)
    return record


def run_report(
    *,
    golden_file: Path = GOLDEN_FILE,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    top_n: int = 20,
    save: bool = False,
    debug_dump: Path | None = None,
    failure_index: Path | None = None,
) -> dict[str, Any]:
    examples = load_examples(golden_file)
    result = evaluate_examples(examples, variants=variants, top_n=top_n)
    payload = {
        "status": "ok",
        "dataset": {
            "golden_file": str(golden_file),
            "examples": len(examples),
            "reviewed": sum(1 for example in examples if example.reviewed),
            "splits": _count_by(examples, "split"),
            "languages": _count_by(examples, "language"),
            "kinds": _count_by(examples, "kind"),
        },
        "top_n": top_n,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "variants": result["variants"],
    }
    if debug_dump is not None:
        write_jsonl(debug_dump, result["debug_rows"])
        payload["debug_dump"] = str(debug_dump)
    if failure_index is not None:
        payload["failure_index"] = write_failure_index(result["debug_rows"], failure_index)
    if save:
        payload["baseline_file"] = str(save_baseline(payload))
    return payload


def _parse_variants(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_VARIANTS
    variants = tuple(item.strip() for item in raw.split(",") if item.strip())
    return variants or DEFAULT_VARIANTS


def print_report(payload: dict[str, Any]) -> None:
    dataset = payload["dataset"]
    print(f"dataset\t{dataset['examples']} examples\t{dataset['golden_file']}")
    print(f"reviewed\t{dataset['reviewed']}")
    for variant, data in payload["variants"].items():
        metrics = data["metrics"]
        print(
            "\t".join(
                [
                    variant,
                    f"recall@5={metrics['recall_at_5']:.3f}",
                    f"recall@20={metrics['recall_at_20']:.3f}",
                    f"mrr@10={metrics['mrr_at_10']:.3f}",
                    f"ndcg@10={metrics['ndcg_at_10']:.3f}",
                    f"negative@20={metrics['negative_hit_rate_at_20']:.3f}",
                    f"stale@20={metrics['stale_hit_rate_at_20']:.3f}",
                    f"p95={metrics['latency_ms']['p95']:.0f}ms",
                ]
            )
        )
    if payload.get("debug_dump"):
        print(f"debug_dump\t{payload['debug_dump']}")
    if payload.get("baseline_file"):
        print(f"baseline\t{payload['baseline_file']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LLM Wiki search ranking quality.")
    parser.add_argument("--golden-file", default=str(GOLDEN_FILE))
    parser.add_argument("--feedback-file", default=str(RECALL_FEEDBACK_FILE))
    parser.add_argument("--log-file", default=str(RECALL_LOG_FILE))
    parser.add_argument("--output-file", default=str(GOLDEN_FILE))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--variants", help="Comma-separated variants to evaluate.")
    parser.add_argument("--debug-dump", help="Write per-query channel/result rows as JSONL.")
    parser.add_argument("--failure-index", nargs="?", const=str(FAILURE_INDEX_FILE), help="Write failed query index JSONL.")
    parser.add_argument("--build-golden", action="store_true")
    parser.add_argument("--build-label-queue", action="store_true")
    parser.add_argument("--self-tune", action="store_true", help="Run dev-only shadow self-tune with locked-test guard.")
    parser.add_argument("--self-tune-history", default=str(SELF_TUNE_HISTORY_FILE))
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.build_golden:
        payload = build_golden(
            feedback_file=Path(args.feedback_file).expanduser(),
            log_file=Path(args.log_file).expanduser(),
            output_file=Path(args.output_file).expanduser(),
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"golden_file\t{payload['output_file']}")
            print(f"examples\t{payload['examples']}")
            print(f"reviewed\t{payload['reviewed']}")
        return 0
    if args.build_label_queue:
        payload = build_label_queue(
            feedback_file=Path(args.feedback_file).expanduser(),
            log_file=Path(args.log_file).expanduser(),
            output_file=Path(args.output_file).expanduser()
            if args.output_file != str(GOLDEN_FILE)
            else LABEL_QUEUE_FILE,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"label_queue\t{payload['output_file']}")
            print(f"examples\t{payload['examples']}")
            print(payload["note"])
        return 0
    if args.self_tune:
        payload = self_tune(
            golden_file=Path(args.golden_file).expanduser(),
            history_file=Path(args.self_tune_history).expanduser(),
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"self_tune\t{payload['status']}")
            print(f"applied\t{payload['applied']}")
            print(f"history\t{args.self_tune_history}")
        return 0

    payload = run_report(
        golden_file=Path(args.golden_file).expanduser(),
        variants=_parse_variants(args.variants),
        top_n=args.top_n,
        save=args.save_baseline,
        debug_dump=Path(args.debug_dump).expanduser() if args.debug_dump else None,
        failure_index=Path(args.failure_index).expanduser() if args.failure_index else None,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
