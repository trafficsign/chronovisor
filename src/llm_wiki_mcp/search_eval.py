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
import shlex
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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
from llm_wiki_mcp.pipeline import (
    PipelineConfig,
    PipelineDependencies,
    apply_negative_feedback_stage,
    production_pipeline_config,
    run_search_pipeline,
)
from llm_wiki_mcp.runtime_config import load_negative_feedback_config, load_reranker_config
from llm_wiki_mcp.wiki import SYSTEM_DIR, WIKI_ROOT, find_page


REPO_ROOT = Path(__file__).resolve().parents[2]
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
        rerank_outcome = rerank_results(
            query,
            apply_filters(results),
            config=load_reranker_config(),
        )
        results = rerank_outcome.results
        reranker_meta = rerank_outcome.metadata
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
                "queue_status": "pending_frontier_review",
                "promoted_to_golden": False,
                "reviewer": "",
                "review_confidence": None,
                "review_note": "",
            }
        )
    write_jsonl(output_file, rows)
    return {
        "status": "ok",
        "output_file": str(output_file),
        "examples": len(rows),
        "reviewed": 0,
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
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    confidence_value = max(0.0, min(1.0, confidence_value))
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
        "human_required",
        "notify_user",
        "access_repair",
        "votes",
    ):
        if key in raw:
            normalized[key] = raw[key]
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
    command = os.environ.get("LLM_WIKI_LABEL_REVIEW_CMD")
    if command:
        completed = subprocess.run(
            shlex.split(command),
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=frontier_review._frontier_env(),
        )
        output = frontier_review.redact_sensitive_text((completed.stdout or "") + "\n" + (completed.stderr or ""))
        if completed.returncode != 0:
            failure = frontier_review.classify_frontier_failure(output).to_dict()
            return _frontier_label_failure(
                f"frontier label command failed with exit {completed.returncode}",
                output=output,
                failure=failure,
            )
        return _parse_frontier_label_output(output)

    codex = shutil.which("codex")
    if codex is None:
        failure = {
            "failure_class": "frontier_tool_unavailable",
            "rescue_status": "human_required",
            "summary": "codex executable not found",
            "human_required": True,
            "notify_user": True,
        }
        return _frontier_label_failure("codex executable not found", failure=failure)

    preflight = frontier_review.run_frontier_preflight()
    if not preflight.get("ok"):
        failure = preflight.get("failure") if isinstance(preflight.get("failure"), dict) else None
        return _frontier_label_failure("frontier preflight failed", failure=failure)

    with tempfile.TemporaryDirectory() as td:
        schema_path = Path(td) / "frontier-label.schema.json"
        output_path = Path(td) / "frontier-label-output.json"
        strict_schema, _schema_repair = frontier_review._strict_schema_with_repair(FRONTIER_LABEL_SCHEMA)
        schema_path.write_text(json.dumps(strict_schema, indent=2) + "\n", encoding="utf-8")
        invocation = frontier_review._build_codex_exec_invocation(
            codex,
            repo_root=repo_root,
            schema_path=schema_path,
            output_path=output_path,
            execute_patch=False,
            preflight=preflight,
        )
        completed = subprocess.run(
            invocation["cmd"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=frontier_review._frontier_env(),
            cwd=invocation.get("cwd") or None,
        )
        output_text = ""
        if output_path.exists():
            output_text += output_path.read_text(encoding="utf-8", errors="replace")
        output_text += "\n" + (completed.stdout or "") + "\n" + (completed.stderr or "")
        output_text = frontier_review.redact_sensitive_text(output_text)
        if completed.returncode != 0:
            failure = frontier_review.classify_frontier_failure(output_text).to_dict()
            return _frontier_label_failure(
                f"codex frontier label review failed with exit {completed.returncode}",
                output=output_text,
                failure=failure,
            )
        result = _parse_frontier_label_output(output_text)
        result["access_repair"] = {
            "invocation": {
                "source": invocation.get("source"),
                "cmd": invocation.get("cmd"),
                "cwd": invocation.get("cwd"),
                "schema_path": invocation.get("schema_path"),
                "output_path": invocation.get("output_path"),
            },
            "preflight": {
                "codex_home": preflight.get("codex_home"),
                "auth_path": preflight.get("auth_path"),
            },
        }
        return result


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
    if not reviews:
        return _frontier_label_failure("no frontier label reviews were attempted")
    if len(reviews) == 1:
        return reviews[0]

    if any(review.get("human_required") for review in reviews):
        first = next(review for review in reviews if review.get("human_required"))
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
        if review.get("decision") == "approved" and float(review.get("confidence") or 0.0) >= min_confidence
    ]
    label_sets = {_label_tuple_from_review(review) for review in approvals}
    if len(approvals) == len(reviews) and len(label_sets) == 1:
        best = max(approvals, key=lambda review: float(review.get("confidence") or 0.0))
        return {
            **best,
            "reviewer": "frontier_consensus",
            "summary": f"frontier consensus approved: {best.get('summary', '')}",
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
        "summary": "frontier reviewers did not agree on a high-confidence label set",
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
    failure = review.get("frontier_failure") if isinstance(review.get("frontier_failure"), dict) else {}
    if review.get("human_required") or failure.get("human_required"):
        return "human_required"
    decision = review.get("decision")
    confidence = float(review.get("confidence") or 0.0)
    if decision == "approved" and confidence >= min_confidence and any(_label_tuple_from_review(review)):
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
) -> dict[str, Any]:
    rows = read_jsonl(queue_file)
    golden_rows = read_jsonl(golden_file)
    golden_keys = {_golden_key(row) for row in golden_rows}
    reviewed_at = datetime.now().isoformat(timespec="seconds")
    attempted = 0
    promoted = 0
    status_counts: dict[str, int] = {}
    updated_rows: list[dict[str, Any]] = []
    max_votes = max(1, votes)

    for row in rows:
        status = str(row.get("queue_status") or "")
        if attempted >= limit or bool(row.get("promoted_to_golden")) or status not in FRONTIER_PENDING_STATUSES:
            updated_rows.append(row)
            continue

        reviews: list[dict[str, Any]] = []
        for _idx in range(max_votes):
            review = (
                reviewer(row)
                if reviewer is not None
                else run_frontier_label_review(row, repo_root=repo_root, timeout=timeout)
            )
            reviews.append(_normalize_frontier_label_result(review) if "decision" in review else review)
        combined = _combine_frontier_label_reviews(reviews, min_confidence=min_confidence)
        next_status = _queue_status_for_review(combined, min_confidence=min_confidence)
        attempted += 1
        status_counts[next_status] = status_counts.get(next_status, 0) + 1

        updated = {
            **row,
            "queue_status": next_status,
            "reviewer": combined.get("reviewer") or "frontier",
            "review_confidence": float(combined.get("confidence") or 0.0),
            "review_note": combined.get("summary") or "",
            "frontier_review": combined,
        }

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

    write_jsonl(queue_file, updated_rows)
    if promoted:
        write_jsonl(golden_file, golden_rows)
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
        config=PipelineConfig(
            top_n=top_n,
            semantic=True,
            fusion_weights=dict(weights),
            result_strategy="weighted_fusion",
            graph_strategy="disabled",
            usage_strategy="disabled",
            apply_negative_feedback=False,
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
    parser.add_argument("--label-queue-file", default=str(LABEL_QUEUE_FILE))
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
    parser.add_argument("--frontier-review-labels", action="store_true", help="Use a frontier model to promote trusted label-queue rows into the golden set.")
    parser.add_argument("--frontier-min-confidence", type=float, default=0.8)
    parser.add_argument("--frontier-votes", type=int, default=1, help="Number of frontier votes required to agree before promotion.")
    parser.add_argument("--frontier-timeout", type=int, default=None)
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
