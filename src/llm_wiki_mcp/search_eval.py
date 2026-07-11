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
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from llm_wiki_mcp.convergence import is_human_required_result
from llm_wiki_mcp.feedback_ledger import active_feedback_rows
from llm_wiki_mcp.search import (
    ACTIVE_SEARCH_POLICY_FILE,
    DEFAULT_FUSION_WEIGHTS,
    ScoredPage,
    apply_filters,
    fuse_results,
    get_bm25,
    graph_expand_results,
    semantic_search,
    usage_prior_results,
    load_active_fusion_weights,
)
from llm_wiki_mcp.reranker import rerank_results
from llm_wiki_mcp.negative_feedback import apply_penalties, penalties_for_query
from llm_wiki_mcp.pipeline import (
    PipelineConfig,
    PipelineDependencies,
    apply_negative_feedback_stage,
    apply_rerank_stage,
    production_pipeline_config,
    run_search_pipeline,
)
from llm_wiki_mcp.runtime_config import (
    load_negative_feedback_config,
    load_reranker_config,
    runtime_repo_root,
)
from llm_wiki_mcp.wiki import SYSTEM_DIR, WIKI_ROOT, find_page


REPO_ROOT = runtime_repo_root()
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

FRONTIER_LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "confidence",
        "expected_pages",
        "negative_pages",
        "stale_pages",
        "summary",
        "notes",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "uncertain", "needs_retry"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "expected_pages": {"type": "array", "items": {"type": "string"}},
        "negative_pages": {"type": "array", "items": {"type": "string"}},
        "stale_pages": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "notes": {"type": ["string", "null"]},
    },
}

FRONTIER_PENDING_STATUSES = {
    "",
    "pending_review",
    "pending_frontier_review",
    "frontier_retry",
    "frontier_uncertain",
}

FRONTIER_TERMINAL_STATUSES = {
    "frontier_approved",
    "frontier_rejected",
    "frontier_quarantined",
    "human_required",
}
DEFAULT_QUARANTINE_RETRY_SECONDS = 6 * 60 * 60

FrontierLabelReviewer = Callable[[dict[str, Any]], dict[str, Any]]


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
        lines = path.read_text(encoding="utf-8").split("\n")
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


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(path, payload)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(item for item in value if isinstance(item, str) and item))


def _str_list(value: Any) -> list[str]:
    return list(_str_tuple(value))


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

    for feedback in active_feedback_rows(feedback_file):
        kind = str(feedback.get("kind", ""))
        if kind not in {
            "missed",
            "missed_candidate",
            "injection_used",
            "injection_ignored",
            "false-positive",
            "page_ignored",
        }:
            continue

        ref = str(feedback.get("ref", ""))
        snapshot = feedback.get("snapshot") if isinstance(feedback.get("snapshot"), dict) else {}
        record = logs_by_id.get(ref) or snapshot or {}
        query = str(feedback.get("prompt") or record.get("prompt_preview") or "").strip()
        if not query:
            continue

        raw_expected = _str_tuple(feedback.get("expected_pages"))
        raw_negative = _str_tuple(feedback.get("negative_pages"))
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
        elif kind == "page_ignored":
            # Page-scoped feedback must never turn every injected page into a
            # negative example when only one candidate was rejected.
            negative = raw_negative
        else:
            # Prefer the explicit page-scoped field when present while
            # retaining compatibility with legacy prompt-scoped feedback.
            negative = raw_negative or raw_injected or raw_expected

        # A reviewed search label may carry both relevant and irrelevant
        # candidates. Preserve that mixed supervision for ranking evaluation.
        if raw_negative:
            negative = raw_negative

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


def _source_allowed(source: str, source_filter: str) -> bool:
    if source_filter == "all":
        return True
    is_auto = source in {"recall_questions", "auto", "generated"}
    if source_filter == "auto":
        return is_auto
    if source_filter == "manual":
        return not is_auto
    return True


def load_examples(
    path: Path = GOLDEN_FILE,
    *,
    limit: int = 0,
    source_filter: str = "all",
    reviewed_only: bool = True,
) -> list[SearchExample]:
    examples: list[SearchExample] = []
    for row in read_jsonl(path):
        # Active evaluation and self-tune must never consume a locally
        # generated label. Candidate rows live in the label queue until a
        # frontier reviewer promotes them with reviewed=true.
        if reviewed_only and row.get("reviewed") is not True:
            continue
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        source = str(row.get("source") or "manual")
        if not _source_allowed(source, source_filter):
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
                source=source,
                ref=str(row.get("ref") or ""),
                ts=str(row.get("ts") or ""),
                reviewed=bool(row.get("reviewed", False)),
            )
        )
        if limit > 0 and len(examples) >= limit:
            break
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


def _pipeline_dependencies() -> PipelineDependencies:
    return PipelineDependencies(
        get_bm25=get_bm25,
        semantic_search=semantic_search,
        graph_expand_results=graph_expand_results,
        usage_prior_results=usage_prior_results,
        fuse_results=fuse_results,
        apply_filters=apply_filters,
        apply_sort=lambda results, sort_by="relevance": results,
        load_negative_feedback_config=load_negative_feedback_config,
        penalties_for_query=penalties_for_query,
        apply_penalties=apply_penalties,
    )


def _variant_pipeline_config(variant: str, *, top_n: int) -> tuple[PipelineConfig, bool]:
    weights = dict(DEFAULT_FUSION_WEIGHTS)
    if variant == "bm25":
        return (
            PipelineConfig(
                top_n=top_n,
                semantic=False,
                fusion_weights=weights,
                result_strategy="bm25",
                graph_strategy="disabled",
                usage_strategy="disabled",
            ),
            False,
        )
    if variant == "semantic":
        return (
            PipelineConfig(
                top_n=top_n,
                semantic=True,
                fusion_weights=weights,
                result_strategy="semantic",
                graph_strategy="disabled",
                usage_strategy="disabled",
            ),
            False,
        )
    if variant == "hybrid-plain-rrf":
        return (
            PipelineConfig(
                top_n=top_n,
                semantic=True,
                fusion_weights=weights,
                result_strategy="plain_rrf",
                graph_strategy="disabled",
                usage_strategy="disabled",
                plain_rrf_weights={"bm25": 1.0, "semantic": 1.0},
            ),
            False,
        )
    if variant == "hybrid-graph":
        return (
            PipelineConfig(
                top_n=top_n,
                semantic=True,
                fusion_weights={**weights, "graph": 0.5, "usage_prior": 0.0},
                result_strategy="weighted_fusion",
                graph_strategy="fixed",
                graph_decay=0.5,
                usage_strategy="disabled",
            ),
            False,
        )
    if variant == "hybrid-usage":
        return (
            PipelineConfig(
                top_n=top_n,
                semantic=True,
                fusion_weights={**weights, "graph": 0.0, "usage_prior": 0.2},
                result_strategy="weighted_fusion",
                graph_strategy="disabled",
                usage_strategy="always",
                usage_include_graph=False,
            ),
            False,
        )
    if variant == "hybrid-current":
        return (
            production_pipeline_config(top_n=top_n, fusion_weights=weights),
            False,
        )
    if variant == "hybrid-rerank":
        return (
            replace(
                production_pipeline_config(top_n=top_n, fusion_weights=weights),
                apply_negative_feedback=False,
                filter_results=False,
                sort_results=False,
                truncate_results=False,
            ),
            True,
        )
    raise ValueError(f"unknown search eval variant: {variant}")


def run_variant(query: str, variant: str, *, top_n: int = 20) -> dict[str, Any]:
    started = time.perf_counter()
    config, needs_rerank = _variant_pipeline_config(variant, top_n=top_n)
    deps = _pipeline_dependencies()
    pipeline_result = run_search_pipeline(query, config=config, deps=deps)
    results = pipeline_result.results
    reranker_meta: dict[str, Any] = {"status": "not_requested"}
    negative_meta = pipeline_result.negative_feedback
    if needs_rerank:
        rerank_stage = apply_rerank_stage(
            query,
            apply_filters(results),
            reranker_config=load_reranker_config(),
            rerank_results=rerank_results,
        )
        results = rerank_stage.results
        reranker_meta = rerank_stage.metadata
        results, negative_meta = apply_negative_feedback_stage(query, results, deps=deps)

    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    out = apply_filters(results)[:top_n]
    return {
        "variant": variant,
        "results": out,
        "latency_ms": elapsed_ms,
        "channels": {
            "bm25": [page.page_id for page in pipeline_result.bm25_results[:top_n]],
            "semantic": [page.page_id for page in pipeline_result.semantic_results[:top_n]],
            "graph": [page.page_id for page in pipeline_result.graph_results[:top_n]],
            "usage_prior": [page.page_id for page in pipeline_result.usage_results[:top_n]],
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
    output_file: Path = LABEL_QUEUE_FILE,
    limit: int = 100,
) -> dict[str, Any]:
    # Compatibility wrapper: the old command name may remain in scripts, but
    # local candidates can no longer overwrite the authoritative golden set.
    target = LABEL_QUEUE_FILE if output_file == GOLDEN_FILE else output_file
    queued = build_label_queue(
        feedback_file=feedback_file,
        log_file=log_file,
        output_file=target,
        limit=limit,
    )
    return {
        **queued,
        "status": (
            queued.get("status")
            if queued.get("status") != "ok"
            else "queued_for_frontier_review"
        ),
        "legacy_command": "build-golden",
        "authoritative_golden_unchanged": True,
    }


def build_label_queue(
    *,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    log_file: Path = RECALL_LOG_FILE,
    output_file: Path = LABEL_QUEUE_FILE,
    limit: int = 100,
    dry_run: bool = False,
    budget: Any | None = None,
) -> dict[str, Any]:
    examples = build_candidates(feedback_file=feedback_file, log_file=log_file, limit=limit)
    existing_rows = read_jsonl(output_file)
    existing_by_key = {_golden_key(row): row for row in existing_rows}
    rows = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = set()
    for row in examples_to_rows(examples):
        key = _golden_key(row)
        seen.add(key)
        previous = existing_by_key.get(key, {})
        rows.append(
            {
                **row,
                "queue_status": previous.get("queue_status", "pending_frontier_review"),
                "promoted_to_golden": bool(previous.get("promoted_to_golden", False)),
                "reviewer": previous.get("reviewer", ""),
                "review_confidence": previous.get("review_confidence"),
                "review_note": previous.get("review_note", ""),
                **{
                    key_: value
                    for key_, value in previous.items()
                    if key_
                    in {
                        "frontier_attempts",
                        "frontier_review",
                        "last_attempt_at",
                        "next_attempt_at",
                        "reviewed",
                        "reviewed_at",
                    }
                },
            }
        )
    # A refresh must never resurrect or erase a prior decision. Keep rows that
    # fell outside the latest candidate window so terminal state remains an
    # exact-once ledger and retryable work can continue draining.
    rows.extend(row for row in existing_rows if _golden_key(row) not in seen)
    changed = rows != existing_rows
    if not dry_run and changed:
        if budget is not None:
            allowed, reason = budget.consume("mutation")
            if not allowed:
                return {
                    "status": "budget_deferred",
                    "reason": reason,
                    "output_file": str(output_file),
                    "examples": len(existing_rows),
                    "reviewed": sum(
                        1
                        for row in existing_rows
                        if str(row.get("queue_status") or "") in FRONTIER_TERMINAL_STATUSES
                    ),
                    "preserved": len(existing_rows),
                    "dry_run": False,
                }
        write_jsonl(output_file, rows)
    return {
        "status": "ok",
        "output_file": str(output_file),
        "examples": len(rows),
        "reviewed": sum(1 for row in rows if str(row.get("queue_status") or "") in FRONTIER_TERMINAL_STATUSES),
        "preserved": len(existing_rows),
        "dry_run": dry_run,
        "changed": changed,
        "note": "Candidates are not added to search-golden.jsonl until trusted frontier review.",
    }


def _page_for_label(page_id: str) -> Path | None:
    page = find_page(page_id)
    if page is not None:
        return page
    system_page = SYSTEM_DIR / f"{page_id}.md"
    return system_page if system_page.exists() else None


def _page_excerpt(page_id: str, *, limit: int = 1800) -> dict[str, Any]:
    path = _page_for_label(page_id)
    if path is None:
        return {"page_id": page_id, "exists": False, "path": "", "excerpt": ""}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"page_id": page_id, "exists": False, "path": str(path), "error": str(exc), "excerpt": ""}
    return {
        "page_id": page_id,
        "exists": True,
        "path": str(path),
        "excerpt": text[:limit],
    }


def _candidate_label_pages(row: dict[str, Any]) -> list[str]:
    pages: list[str] = []
    for field in ("expected_pages", "negative_pages", "stale_pages"):
        pages.extend(_str_list(row.get(field)))
    return list(dict.fromkeys(pages))


def build_frontier_label_prompt(row: dict[str, Any]) -> str:
    payload = {
        "query": str(row.get("query") or ""),
        "candidate_labels": {
            "expected_pages": _str_list(row.get("expected_pages")),
            "negative_pages": _str_list(row.get("negative_pages")),
            "stale_pages": _str_list(row.get("stale_pages")),
        },
        "metadata": {
            "split": row.get("split"),
            "language": row.get("language"),
            "kind": row.get("kind"),
            "source": row.get("source"),
            "ref": row.get("ref"),
            "ts": row.get("ts"),
        },
        "page_excerpts": [_page_excerpt(page_id) for page_id in _candidate_label_pages(row)],
    }
    schema = FRONTIER_LABEL_SCHEMA
    return f"""\
You are the trusted frontier label reviewer for LLM Wiki search evaluation.

Goal:
- Decide whether the candidate labels are trustworthy for the user's search query.
- Promote only labels that are clearly supported by the query and page excerpts.
- Keep false positives out of search-golden.jsonl.
- Do not edit files, run commands, ask a human, or invent unrelated page ids.

Decision policy:
- approved: the label set is trustworthy. You may move page ids between expected_pages,
  negative_pages, and stale_pages if the candidate type is wrong but the evidence is clear.
- rejected: the candidate labels are clearly wrong and should not be promoted.
- uncertain: evidence is insufficient or ambiguous; leave it for another frontier pass.
- needs_retry: the review cannot be completed because context is malformed or unavailable.

Return JSON only matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Candidate:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def _frontier_label_failure(summary: str, *, output: str = "", failure: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "decision": "needs_retry",
        "confidence": 0.0,
        "expected_pages": [],
        "negative_pages": [],
        "stale_pages": [],
        "summary": summary,
        "notes": None,
        "reviewer": "frontier",
        "frontier_failure": failure,
        "raw_output": output[-4000:] if output else "",
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _normalize_frontier_label_result(raw: dict[str, Any], *, raw_output: str = "") -> dict[str, Any]:
    decision = raw.get("decision")
    if decision not in {"approved", "rejected", "uncertain", "needs_retry"}:
        return _frontier_label_failure("frontier label JSON failed schema validation", output=raw_output)
    confidence = raw.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return _frontier_label_failure(
            "frontier label confidence metadata failed schema validation",
            output=raw_output,
        )
    confidence_value = float(confidence)
    summary = raw.get("summary")
    normalized = {
        "decision": decision,
        "confidence": confidence_value,
        "expected_pages": _str_list(raw.get("expected_pages")),
        "negative_pages": _str_list(raw.get("negative_pages")),
        "stale_pages": _str_list(raw.get("stale_pages")),
        "summary": summary if isinstance(summary, str) and summary.strip() else decision,
        "notes": raw.get("notes") if isinstance(raw.get("notes"), str) else None,
        "reviewer": str(raw.get("reviewer") or "frontier"),
    }
    raw_text = raw_output or str(raw.get("raw_output") or "")
    if raw_text and decision == "needs_retry":
        normalized["raw_output"] = raw_text[-4000:]
    for key in (
        "frontier_failure",
        "access_repair",
        "votes",
    ):
        if key in raw:
            normalized[key] = raw[key]
    if any(key in raw for key in ("frontier_failure", "human_required", "notify_user")):
        needs_human = is_human_required_result(raw)
        normalized["human_required"] = needs_human
        normalized["notify_user"] = needs_human
    return normalized


def _parse_frontier_label_output(output: str) -> dict[str, Any]:
    parsed = _extract_json_object(output)
    if parsed is None:
        return _frontier_label_failure("frontier label output did not contain JSON", output=output)
    return _normalize_frontier_label_result(parsed, raw_output=output)


def run_frontier_label_review(
    row: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    timeout: int | None = None,
) -> dict[str, Any]:
    from llm_wiki_mcp import frontier_review

    prompt = build_frontier_label_prompt(row)
    timeout_seconds = timeout or int(os.environ.get("LLM_WIKI_FRONTIER_TIMEOUT_SECONDS", "3600"))
    raw = frontier_review.run_structured_review(
        prompt,
        FRONTIER_LABEL_SCHEMA,
        repo_root=repo_root,
        timeout=timeout_seconds,
        execute_patch=False,
        command_env="LLM_WIKI_LABEL_REVIEW_CMD",
        decision_lane="search_label",
    )
    return _normalize_frontier_label_result(raw)


def _label_tuple_from_review(review: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(_str_list(review.get("expected_pages"))),
        tuple(_str_list(review.get("negative_pages"))),
        tuple(_str_list(review.get("stale_pages"))),
    )


def _combine_frontier_label_reviews(
    reviews: list[dict[str, Any]],
    *,
    min_confidence: float,
) -> dict[str, Any]:
    # Kept for API compatibility only. Confidence is diagnostic metadata and
    # never participates in consensus or promotion.
    del min_confidence
    if not reviews:
        return _frontier_label_failure("no frontier label reviews were attempted")
    if len(reviews) == 1:
        return reviews[0]

    if any(is_human_required_result(review) for review in reviews):
        first = next(review for review in reviews if is_human_required_result(review))
        return {**first, "summary": f"frontier label review needs human action: {first.get('summary', '')}"}

    retry = [review for review in reviews if review.get("decision") == "needs_retry"]
    if retry:
        return {
            **retry[0],
            "summary": f"frontier label consensus needs retry: {retry[0].get('summary', '')}",
            "votes": reviews,
        }

    approvals = [
        review
        for review in reviews
        if review.get("decision") == "approved"
    ]
    label_sets = {_label_tuple_from_review(review) for review in approvals}
    if len(approvals) == len(reviews) and len(label_sets) == 1:
        agreed = approvals[0]
        return {
            **agreed,
            "reviewer": "frontier_consensus",
            "summary": f"frontier consensus approved: {agreed.get('summary', '')}",
            "votes": reviews,
        }

    if any(review.get("decision") == "rejected" for review in reviews):
        return {
            **next(review for review in reviews if review.get("decision") == "rejected"),
            "reviewer": "frontier_consensus",
            "summary": "frontier consensus rejected or disagreed on labels",
            "votes": reviews,
        }

    return {
        "decision": "uncertain",
        "confidence": 0.0,
        "expected_pages": [],
        "negative_pages": [],
        "stale_pages": [],
        "summary": "frontier reviewers did not agree on one label action",
        "notes": None,
        "reviewer": "frontier_consensus",
        "votes": reviews,
    }


def _golden_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        str(row.get("query") or ""),
        tuple(_str_list(row.get("expected_pages"))),
        tuple(_str_list(row.get("negative_pages"))),
        tuple(_str_list(row.get("stale_pages"))),
    )


def _golden_row_from_review(row: dict[str, Any], review: dict[str, Any], *, reviewed_at: str) -> dict[str, Any]:
    expected = _str_list(review.get("expected_pages"))
    negative = _str_list(review.get("negative_pages"))
    stale = _str_list(review.get("stale_pages"))
    out = {
        "query": str(row.get("query") or ""),
        "expected_pages": expected,
        "negative_pages": negative,
        "stale_pages": stale,
        "split": str(row.get("split") or assign_split(str(row.get("query") or ""))),
        "language": str(row.get("language") or language_bucket(str(row.get("query") or ""))),
        "kind": str(row.get("kind") or query_kind(str(row.get("query") or ""))),
        "source": str(row.get("source") or "frontier_label_review"),
        "ref": str(row.get("ref") or ""),
        "ts": str(row.get("ts") or ""),
        "reviewed": True,
        "reviewer": str(review.get("reviewer") or "frontier"),
        "review_confidence": float(review.get("confidence") or 0.0),
        "reviewed_at": reviewed_at,
        "review_note": str(review.get("summary") or ""),
    }
    return out


def _queue_status_for_review(review: dict[str, Any], *, min_confidence: float) -> str:
    # Deprecated compatibility input; decision + exact label action determine
    # queue state.
    del min_confidence
    if is_human_required_result(review):
        return "human_required"
    decision = review.get("decision")
    if decision == "approved" and any(_label_tuple_from_review(review)):
        return "frontier_approved"
    if decision == "approved":
        return "frontier_uncertain"
    if decision == "rejected":
        return "frontier_rejected"
    if decision == "needs_retry":
        return "frontier_retry"
    return "frontier_uncertain"


def review_label_queue_with_frontier(
    *,
    queue_file: Path = LABEL_QUEUE_FILE,
    golden_file: Path = GOLDEN_FILE,
    limit: int = 100,
    min_confidence: float = 0.8,
    votes: int = 1,
    timeout: int | None = None,
    repo_root: Path = REPO_ROOT,
    reviewer: FrontierLabelReviewer | None = None,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    dry_run: bool = False,
    now: datetime | None = None,
    budget: Any | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(queue_file)
    original_rows = json.loads(json.dumps(rows, ensure_ascii=False, default=str))
    golden_rows = read_jsonl(golden_file)
    golden_keys = {_golden_key(row) for row in golden_rows}
    current_time = now or datetime.now()
    reviewed_at = current_time.isoformat(timespec="seconds")
    attempted = 0
    promoted = 0
    status_counts: dict[str, int] = {}
    updated_rows: list[dict[str, Any]] = []
    max_votes = max(1, votes)
    attempts_cap = max(1, max_attempts)
    budget_exhausted = False
    mutation_reserved = False

    for row in rows:
        review = row.get("frontier_review")
        if (
            row.get("queue_status") == "human_required"
            and isinstance(review, dict)
            and not is_human_required_result(review)
        ):
            attempts = int(row.get("frontier_attempts") or 0)
            row["queue_status"] = (
                "frontier_quarantined" if attempts >= attempts_cap else "frontier_retry"
            )
            row["human_boundary_reclassified_at"] = reviewed_at
            if row["queue_status"] == "frontier_quarantined":
                row["quarantined_at"] = reviewed_at
            row.pop("next_attempt_at", None)

    try:
        quarantine_retry_seconds = max(
            0,
            int(
                os.getenv(
                    "LLM_WIKI_CONVERGENCE_QUARANTINE_RETRY_SECONDS",
                    str(DEFAULT_QUARANTINE_RETRY_SECONDS),
                )
            ),
        )
    except (TypeError, ValueError):
        quarantine_retry_seconds = DEFAULT_QUARANTINE_RETRY_SECONDS
    for row in rows:
        if row.get("queue_status") != "frontier_quarantined":
            continue
        raw_quarantined_at = row.get("quarantined_at") or row.get("last_attempt_at")
        try:
            quarantined_at = (
                datetime.fromisoformat(raw_quarantined_at)
                if isinstance(raw_quarantined_at, str) and raw_quarantined_at
                else None
            )
        except ValueError:
            quarantined_at = None
        compare_now = current_time
        if quarantined_at is not None:
            if quarantined_at.tzinfo is None and compare_now.tzinfo is not None:
                compare_now = compare_now.replace(tzinfo=None)
            elif quarantined_at.tzinfo is not None and compare_now.tzinfo is None:
                quarantined_at = quarantined_at.replace(tzinfo=None)
        if (
            quarantined_at is not None
            and (compare_now - quarantined_at).total_seconds() < quarantine_retry_seconds
        ):
            continue
        row["queue_status"] = "frontier_retry"
        row["frontier_attempts"] = 0
        row["quarantine_reopened_at"] = reviewed_at
        row["quarantine_reopen_count"] = int(row.get("quarantine_reopen_count") or 0) + 1
        row.pop("next_attempt_at", None)

    # Recover both cross-file crash windows. The current golden-first commit can
    # exit before acknowledging the queue; older builds could acknowledge the
    # queue first. In either direction the durable side contains enough trusted
    # evidence to reconcile without paying for another frontier call.
    reviewed_golden = [row for row in golden_rows if row.get("reviewed") is True]
    golden_by_key = {_golden_key(row): row for row in reviewed_golden}
    golden_by_ref = {
        (str(row.get("query") or ""), str(row.get("ref") or "")): row
        for row in reviewed_golden
        if str(row.get("ref") or "")
    }
    recovered_queue = 0
    for row in rows:
        if bool(row.get("promoted_to_golden")) or row.get("queue_status") == "frontier_approved":
            continue
        golden_match = golden_by_key.get(_golden_key(row))
        ref = str(row.get("ref") or "")
        if golden_match is None and ref:
            golden_match = golden_by_ref.get((str(row.get("query") or ""), ref))
        if golden_match is None:
            continue
        review = {
            "decision": "approved",
            "confidence": float(golden_match.get("review_confidence") or 1.0),
            "expected_pages": _str_list(golden_match.get("expected_pages")),
            "negative_pages": _str_list(golden_match.get("negative_pages")),
            "stale_pages": _str_list(golden_match.get("stale_pages")),
            "summary": str(golden_match.get("review_note") or "recovered from trusted golden label"),
            "notes": None,
            "reviewer": str(golden_match.get("reviewer") or "frontier"),
        }
        row.update(
            {
                "queue_status": "frontier_approved",
                "promoted_to_golden": True,
                "reviewed": True,
                "reviewed_at": str(golden_match.get("reviewed_at") or reviewed_at),
                "reviewer": review["reviewer"],
                "review_confidence": review["confidence"],
                "review_note": review["summary"],
                "frontier_review": review,
            }
        )
        recovered_queue += 1

    recovered_golden = 0
    for row in rows:
        if not bool(row.get("promoted_to_golden")) and row.get("queue_status") != "frontier_approved":
            continue
        review = row.get("frontier_review")
        if not isinstance(review, dict):
            continue
        if _queue_status_for_review(review, min_confidence=min_confidence) != "frontier_approved":
            continue
        golden_row = _golden_row_from_review(
            row,
            review,
            reviewed_at=str(row.get("reviewed_at") or reviewed_at),
        )
        key = _golden_key(golden_row)
        if key not in golden_keys:
            golden_rows.append(golden_row)
            golden_keys.add(key)
            recovered_golden += 1
        row["promoted_to_golden"] = True
        row["reviewed"] = True
        row["reviewed_at"] = str(row.get("reviewed_at") or reviewed_at)

    if (recovered_queue or recovered_golden) and not dry_run and budget is not None:
        allowed, reason = budget.consume("mutation")
        if not allowed:
            return {
                "status": "budget_deferred",
                "reason": reason,
                "queue_file": str(queue_file),
                "golden_file": str(golden_file),
                "attempted": 0,
                "promoted": 0,
                "remaining": sum(
                    1
                    for row in original_rows
                    if str(row.get("queue_status") or "") in FRONTIER_PENDING_STATUSES
                    and not bool(row.get("promoted_to_golden"))
                ),
                "dry_run": False,
                "budget_exhausted": True,
                "recovered": 0,
            }
        mutation_reserved = True

    # Golden is the source of truth for a successful promotion. Persist any
    # recovered rows before reviewing new work; queue reconciliation is safe to
    # repeat if the process exits immediately afterwards.
    if recovered_golden and not dry_run:
        write_jsonl(golden_file, golden_rows)

    def retry_due(row: dict[str, Any]) -> bool:
        raw = row.get("next_attempt_at")
        if not isinstance(raw, str) or not raw:
            return True
        try:
            return datetime.fromisoformat(raw) <= current_time
        except ValueError:
            return True

    for row in rows:
        status = str(row.get("queue_status") or "")
        if (
            attempted >= limit
            or bool(row.get("promoted_to_golden"))
            or status not in FRONTIER_PENDING_STATUSES
            or not retry_due(row)
        ):
            updated_rows.append(row)
            continue

        if dry_run:
            updated_rows.append(row)
            attempted += 1
            continue

        if budget is not None:
            frontier_allowed, _frontier_reason = budget.can_consume("frontier", max_votes)
            mutation_allowed, _mutation_reason = (
                budget.can_consume("mutation")
                if not mutation_reserved
                else (True, "ok")
            )
            if not frontier_allowed or not mutation_allowed:
                budget_exhausted = True
                updated_rows.append(row)
                continue
            budget.consume("frontier", max_votes)
            if not mutation_reserved:
                budget.consume("mutation")
                mutation_reserved = True

        reviews: list[dict[str, Any]] = []
        try:
            for _idx in range(max_votes):
                review = (
                    reviewer(row)
                    if reviewer is not None
                    else run_frontier_label_review(row, repo_root=repo_root, timeout=timeout)
                )
                reviews.append(_normalize_frontier_label_result(review) if "decision" in review else review)
        except Exception as exc:
            reviews = [
                _frontier_label_failure(
                    f"frontier label reviewer raised {exc.__class__.__name__}: {exc}"
                )
            ]
        combined = _combine_frontier_label_reviews(reviews, min_confidence=min_confidence)
        next_status = _queue_status_for_review(combined, min_confidence=min_confidence)
        frontier_attempts = int(row.get("frontier_attempts") or 0) + 1
        if next_status in {"frontier_retry", "frontier_uncertain"} and frontier_attempts >= attempts_cap:
            next_status = "frontier_quarantined"
        attempted += 1
        status_counts[next_status] = status_counts.get(next_status, 0) + 1

        updated = {
            **row,
            "queue_status": next_status,
            "reviewer": combined.get("reviewer") or "frontier",
            "review_confidence": float(combined.get("confidence") or 0.0),
            "review_note": combined.get("summary") or "",
            "frontier_review": combined,
            "frontier_attempts": frontier_attempts,
            "last_attempt_at": reviewed_at,
        }

        if next_status in {"frontier_retry", "frontier_uncertain"}:
            delay = max(0, backoff_base_seconds) * (2 ** max(0, frontier_attempts - 1))
            updated["next_attempt_at"] = (current_time + timedelta(seconds=delay)).isoformat(timespec="seconds")
        else:
            updated.pop("next_attempt_at", None)
        if next_status == "frontier_quarantined":
            updated["quarantined_at"] = reviewed_at

        if next_status == "frontier_approved":
            golden_row = _golden_row_from_review(row, combined, reviewed_at=reviewed_at)
            key = _golden_key(golden_row)
            if key not in golden_keys:
                golden_rows.append(golden_row)
                golden_keys.add(key)
                promoted += 1
            updated["promoted_to_golden"] = True
            updated["reviewed"] = True
            updated["reviewed_at"] = reviewed_at
        else:
            updated["promoted_to_golden"] = False
        updated_rows.append(updated)

    # Commit the durable effect before its queue acknowledgement. A crash after
    # the golden replace merely causes an idempotent queue reconciliation; the
    # reverse ordering could permanently lose an approved label.
    if promoted and not dry_run:
        write_jsonl(golden_file, golden_rows)
    if not dry_run and updated_rows != original_rows:
        write_jsonl(queue_file, updated_rows)
    return {
        "status": "ok",
        "queue_file": str(queue_file),
        "golden_file": str(golden_file),
        "attempted": attempted,
        "promoted": promoted,
        "remaining": sum(
            1
            for row in updated_rows
            if str(row.get("queue_status") or "") in FRONTIER_PENDING_STATUSES
            and not bool(row.get("promoted_to_golden"))
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "min_confidence": min_confidence,
        "votes": max_votes,
        "dry_run": dry_run,
        "max_attempts": attempts_cap,
        "budget_exhausted": budget_exhausted,
        "recovered": recovered_queue + recovered_golden,
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
    started = time.perf_counter()
    pipeline_result = run_search_pipeline(
        query,
        config=production_pipeline_config(
            top_n=top_n,
            semantic=True,
            fusion_weights=dict(weights),
        ),
        deps=_pipeline_dependencies(),
    )
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    return {"results": pipeline_result.results, "latency_ms": elapsed_ms}


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
    policy_file: Path = ACTIVE_SEARCH_POLICY_FILE,
    apply: bool = False,
    dry_run: bool = False,
    frontier_mode: str = "off",
    frontier_reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    budget: Any | None = None,
    max_examples: int = 200,
    max_elapsed_seconds: float = 10 * 60,
) -> dict[str, Any]:
    examples = load_examples(golden_file)
    dev = [example for example in examples if example.split == "dev"]
    locked = [example for example in examples if example.split == "locked-test"]
    example_cap = max(2, int(max_examples))
    if len(dev) + len(locked) > example_cap:
        locked_quota = min(len(locked), max(1, example_cap // 5))
        dev_quota = min(len(dev), example_cap - locked_quota)
        if dev_quota < example_cap - locked_quota:
            locked_quota = min(len(locked), example_cap - dev_quota)
        dev = dev[-dev_quota:] if dev_quota else []
        locked = locked[-locked_quota:] if locked_quota else []
    if not dev or not locked:
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "status": "blocked",
            "applied": False,
            "reason": "independent dev and locked-test examples are required",
            "dataset": {"dev": len(dev), "locked-test": len(locked)},
        }
        if not dry_run:
            append_jsonl(history_file, record)
        return record
    deadline = time.monotonic() + max(0.0, float(max_elapsed_seconds))

    def evaluate_bounded(items: list[SearchExample], weights: dict[str, float]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for offset in range(0, len(items), 10):
            if time.monotonic() >= deadline:
                raise TimeoutError("search self-tune runtime budget exhausted")
            rows.extend(_rows_for_weight_eval(items[offset:offset + 10], weights))
        return rows

    baseline_weights = load_active_fusion_weights(policy_file)
    try:
        baseline_dev = _metrics(evaluate_bounded(dev, baseline_weights))
        baseline_locked = _metrics(evaluate_bounded(locked, baseline_weights))

        candidates = []
        for semantic_weight in (0.4, 0.5, 0.6, 0.7, 0.8):
            weights = {**baseline_weights, "semantic": semantic_weight}
            dev_metrics = _metrics(evaluate_bounded(dev, weights))
            candidates.append({"weights": weights, "dev": dev_metrics})
    except TimeoutError as exc:
        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "status": "budget_deferred",
            "applied": False,
            "reason": str(exc),
            "dataset": {"dev": len(dev), "locked-test": len(locked), "max_examples": example_cap},
        }
    best = max(
        candidates,
        key=lambda item: (
            item["dev"]["mrr_at_10"],
            item["dev"]["recall_at_5"],
            item["dev"]["ndcg_at_10"],
        ),
    )
    try:
        locked_metrics = _metrics(evaluate_bounded(locked, best["weights"]))
    except TimeoutError as exc:
        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "status": "budget_deferred",
            "applied": False,
            "reason": str(exc),
            "dataset": {"dev": len(dev), "locked-test": len(locked), "max_examples": example_cap},
        }
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
            "apply_policy": "validated_auto" if apply else "shadow_only",
        },
    }
    if status == "shadow_pass" and apply:
        frontier: dict[str, Any] | None = None
        if frontier_mode == "auto" and not dry_run:
            allowed = True
            if budget is not None:
                allowed, _reason = budget.consume("frontier")
            if not allowed:
                record["status"] = "budget_deferred"
                record["reason"] = "frontier cycle budget exhausted"
            else:
                frontier = (
                    frontier_reviewer(record)
                    if frontier_reviewer is not None
                    else review_search_policy_with_frontier(record)
                )
        if frontier is not None:
            record["frontier_review"] = frontier
            if is_human_required_result(frontier):
                record["status"] = "human_required"
                record["reason"] = str(frontier.get("summary") or "frontier access requires external authority")
            elif frontier.get("decision") != "approved":
                record["status"] = (
                    "frontier_rejected" if frontier.get("decision") in {"rejected", "quarantined"}
                    else "frontier_retry"
                )
                record["reason"] = str(frontier.get("summary") or "frontier did not approve search policy")
        if record["status"] == "shadow_pass":
            old = {}
            try:
                parsed = json.loads(policy_file.read_text(encoding="utf-8"))
                old = parsed if isinstance(parsed, dict) else {}
            except (OSError, json.JSONDecodeError):
                pass
            artifact = {
                "version": 1,
                "created_at": record["ts"],
                "source": "search_eval.self_tune",
                "weights": best["weights"],
                "holdout": locked_metrics,
                "previous": old,
            }
            record["policy"] = artifact
            if dry_run:
                record["status"] = "dry_run"
            else:
                mutation_allowed = True
                mutation_reason = "ok"
                if budget is not None:
                    mutation_allowed, mutation_reason = budget.consume("mutation")
                if not mutation_allowed:
                    record["status"] = "budget_deferred"
                    record["reason"] = mutation_reason
                else:
                    _atomic_write_text(
                        policy_file,
                        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
                    )
                    record["status"] = "applied"
                    record["applied"] = True
    if record.get("status") == "frontier_retry":
        candidate_payload = {
            "weights": (record.get("best") or {}).get("weights", {}),
            "dev": (record.get("best") or {}).get("dev", {}),
            "locked-test": (record.get("best") or {}).get("locked-test", {}),
        }
        candidate_hash = hashlib.sha256(
            json.dumps(candidate_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        prior_history = read_jsonl(history_file)
        prior = prior_history[-1] if prior_history else {}
        attempts = (
            int(prior.get("frontier_attempts") or 0) + 1
            if prior.get("candidate_hash") == candidate_hash
            else 1
        )
        record["candidate_hash"] = candidate_hash
        record["frontier_attempts"] = attempts
        if attempts >= 3:
            record["status"] = "frontier_quarantined"
            record["reason"] = f"{record.get('reason', '')}; frontier retry limit exhausted"
            record["next_attempt_at"] = None
        else:
            record["next_attempt_at"] = (
                datetime.now() + timedelta(minutes=15 * (2 ** max(0, attempts - 1)))
            ).isoformat(timespec="seconds")
    if not dry_run and record.get("status") != "budget_deferred":
        append_jsonl(history_file, record)
    return record


def review_search_policy_with_frontier(
    record: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Ask the frontier model for the final veto on a validated policy."""
    from llm_wiki_mcp import frontier_review

    prompt = f"""\
You are the final autonomous reviewer for an LLM Wiki search ranking policy.
The candidate already passed an independent locked-test non-regression gate.
Approve only when the evidence supports the change and no metric or safety
guard regresses. Do not edit files, commit, push, or ask a human. Return JSON
matching the supplied frontier decision schema.

Candidate evidence:
{json.dumps(record, ensure_ascii=False, indent=2)}
"""
    return frontier_review.run_structured_review(
        prompt,
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=repo_root,
        timeout=timeout or int(os.environ.get("LLM_WIKI_FRONTIER_TIMEOUT_SECONDS", "3600")),
        execute_patch=False,
        decision_lane="search_self_tune",
    )


def run_self_tune_due(
    *,
    golden_file: Path = GOLDEN_FILE,
    history_file: Path = SELF_TUNE_HISTORY_FILE,
    policy_file: Path = ACTIVE_SEARCH_POLICY_FILE,
    min_interval_hours: float = 7 * 24,
    apply: bool = True,
    dry_run: bool = False,
    frontier_mode: str = "auto",
    budget: Any | None = None,
    max_examples: int = 200,
    max_elapsed_seconds: float = 10 * 60,
) -> dict[str, Any]:
    history = read_jsonl(history_file)
    latest = history[-1] if history else {}
    last_ts = str(latest.get("ts") or "")
    due = True
    retry_pending = latest.get("status") == "frontier_retry"
    retry_at = str(latest.get("next_attempt_at") or "")
    if retry_pending and retry_at:
        try:
            due = datetime.now() >= datetime.fromisoformat(retry_at)
        except ValueError:
            due = True
    elif last_ts:
        try:
            due = datetime.now() - datetime.fromisoformat(last_ts) >= timedelta(hours=max(0.0, min_interval_hours))
        except ValueError:
            due = True
    if not due:
        return {"status": "skipped", "reason": "interval_not_due", "last_run_at": last_ts}
    return self_tune(
        golden_file=golden_file,
        history_file=history_file,
        policy_file=policy_file,
        apply=apply,
        dry_run=dry_run,
        frontier_mode=frontier_mode,
        budget=budget,
        max_examples=max_examples,
        max_elapsed_seconds=max_elapsed_seconds,
    )


def run_report(
    *,
    golden_file: Path = GOLDEN_FILE,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    top_n: int = 20,
    limit: int = 0,
    source_filter: str = "all",
    save: bool = False,
    debug_dump: Path | None = None,
    failure_index: Path | None = None,
) -> dict[str, Any]:
    examples = load_examples(golden_file, limit=max(0, limit), source_filter=source_filter)
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
            "sources": _count_by(examples, "source"),
            "source_filter": source_filter,
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


def ci_gate(
    payload: dict[str, Any],
    *,
    variant: str = "hybrid-current",
    min_recall_at_5: float = 0.0,
    min_mrr_at_10: float = 0.0,
    max_negative_hit_rate_at_20: float = 1.0,
) -> dict[str, Any]:
    variants = payload.get("variants") if isinstance(payload.get("variants"), dict) else {}
    selected = variants.get(variant) if isinstance(variants, dict) else None
    if selected is None and variants:
        variant, selected = next(iter(variants.items()))
    if not isinstance(selected, dict):
        return {"status": "failed", "reason": "no variant metrics", "variant": variant}
    metrics = selected.get("metrics")
    if not isinstance(metrics, dict):
        return {"status": "failed", "reason": "missing metrics", "variant": variant}
    def metric(name: str, default: float) -> float:
        try:
            return float(metrics.get(name, default))
        except (TypeError, ValueError):
            return default
    failures = []
    if metric("recall_at_5", 0.0) < min_recall_at_5:
        failures.append("recall_at_5")
    if metric("mrr_at_10", 0.0) < min_mrr_at_10:
        failures.append("mrr_at_10")
    if metric("negative_hit_rate_at_20", 1.0) > max_negative_hit_rate_at_20:
        failures.append("negative_hit_rate_at_20")
    return {
        "status": "passed" if not failures else "failed",
        "variant": variant,
        "failures": failures,
        "thresholds": {
            "min_recall_at_5": min_recall_at_5,
            "min_mrr_at_10": min_mrr_at_10,
            "max_negative_hit_rate_at_20": max_negative_hit_rate_at_20,
        },
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LLM Wiki search ranking quality.")
    parser.add_argument("--golden-file", default=str(GOLDEN_FILE))
    parser.add_argument("--label-queue-file", default=str(LABEL_QUEUE_FILE))
    parser.add_argument("--feedback-file", default=str(RECALL_FEEDBACK_FILE))
    parser.add_argument("--log-file", default=str(RECALL_LOG_FILE))
    parser.add_argument("--output-file", default=str(GOLDEN_FILE))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--source-filter", choices=("all", "manual", "auto"), default="all")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--variants", help="Comma-separated variants to evaluate.")
    parser.add_argument("--debug-dump", help="Write per-query channel/result rows as JSONL.")
    parser.add_argument("--failure-index", nargs="?", const=str(FAILURE_INDEX_FILE), help="Write failed query index JSONL.")
    parser.add_argument("--build-golden", action="store_true")
    parser.add_argument("--build-label-queue", action="store_true")
    parser.add_argument("--frontier-review-labels", action="store_true", help="Use a frontier model to promote trusted label-queue rows into the golden set.")
    parser.add_argument(
        "--frontier-min-confidence",
        type=float,
        default=0.8,
        help="Deprecated no-op; confidence is retained only as review metadata.",
    )
    parser.add_argument("--frontier-votes", type=int, default=1, help="Number of frontier votes required to agree before promotion.")
    parser.add_argument("--frontier-timeout", type=int, default=None)
    parser.add_argument("--self-tune", action="store_true", help="Run dev-only shadow self-tune with locked-test guard.")
    parser.add_argument("--self-tune-history", default=str(SELF_TUNE_HISTORY_FILE))
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--ci", action="store_true", help="Fail non-zero when metrics miss thresholds.")
    parser.add_argument("--ci-variant", default="hybrid-current")
    parser.add_argument("--min-recall-at-5", type=float, default=0.0)
    parser.add_argument("--min-mrr-at-10", type=float, default=0.0)
    parser.add_argument("--max-negative-hit-rate-at-20", type=float, default=1.0)
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
        queue_output = (
            Path(args.label_queue_file).expanduser()
            if args.label_queue_file != str(LABEL_QUEUE_FILE)
            else (
                Path(args.output_file).expanduser()
                if args.output_file != str(GOLDEN_FILE)
                else LABEL_QUEUE_FILE
            )
        )
        payload = build_label_queue(
            feedback_file=Path(args.feedback_file).expanduser(),
            log_file=Path(args.log_file).expanduser(),
            output_file=queue_output,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"label_queue\t{payload['output_file']}")
            print(f"examples\t{payload['examples']}")
            print(payload["note"])
        return 0
    if args.frontier_review_labels:
        payload = review_label_queue_with_frontier(
            queue_file=Path(args.label_queue_file).expanduser(),
            golden_file=Path(args.golden_file).expanduser(),
            limit=args.limit,
            min_confidence=args.frontier_min_confidence,
            votes=args.frontier_votes,
            timeout=args.frontier_timeout,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"frontier_label_review\t{payload['status']}")
            print(f"attempted\t{payload['attempted']}")
            print(f"promoted\t{payload['promoted']}")
            print(f"remaining\t{payload['remaining']}")
            print(f"golden_file\t{payload['golden_file']}")
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
        limit=args.limit,
        source_filter="manual" if args.ci and args.source_filter == "all" else args.source_filter,
        save=args.save_baseline,
        debug_dump=Path(args.debug_dump).expanduser() if args.debug_dump else None,
        failure_index=Path(args.failure_index).expanduser() if args.failure_index else None,
    )
    if args.ci:
        payload["ci_gate"] = ci_gate(
            payload,
            variant=args.ci_variant,
            min_recall_at_5=max(0.0, args.min_recall_at_5),
            min_mrr_at_10=max(0.0, args.min_mrr_at_10),
            max_negative_hit_rate_at_20=max(0.0, args.max_negative_hit_rate_at_20),
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print_report(payload)
        if args.ci:
            print(f"ci_gate\t{payload['ci_gate']['status']}")
    if args.ci and payload["ci_gate"]["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
