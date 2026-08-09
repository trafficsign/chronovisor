#!/usr/bin/env python3
"""Capture and compare search-pipeline characterization snapshots.

The snapshot intentionally excludes latency and timestamps. It records exact
page ordering and floating scores so refactors can prove they preserved search
behavior before and after the change.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from chronovisor.core.search import search
from chronovisor.search.search_eval import (
    DEFAULT_VARIANTS,
    GOLDEN_FILE,
    run_report,
    run_variant,
)

DEFAULT_QUERIES = (
    "Chronovisor 検索 ロードマップ",
    "検索パイプライン reranker negative feedback",
    "Codex 保存 hook trusted_hash",
    "uvx GitHub runtime cache",
    "recall audit missed_candidate",
    "search golden frontier review",
    "Claude Code memory Chronovisor",
    "semantic search bge-m3",
    "BM25 CJK bigram",
    "usage prior injection_used",
    "graph expand backlinks outlinks",
    "search eval hybrid-current",
    "dashboard recall metrics",
    "self tune fusion weights",
    "negative feedback injection_ignored",
    "MCP chronovisor_search tag filter rerank",
    "ollama embeddings sqlite",
    "frontier labels confidence votes",
    "plan inbox Chronovisor",
    "runtime config search reranker",
)

SCORE_TOLERANCE = 1e-12


def _page_payload(page: Any) -> dict[str, Any]:
    return {
        "page_id": page.page_id,
        "title": page.title,
        "folder": page.folder,
        "updated": page.updated,
        "score": float(page.score),
        "status": page.status,
        "superseded_by": page.superseded_by,
    }


def _strip_latency_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "latency_ms"}


def _metrics_payload(golden_file: Path, variants: tuple[str, ...], top_n: int) -> dict[str, Any]:
    report = run_report(golden_file=golden_file, variants=variants, top_n=top_n)
    return {
        "dataset": {
            key: value
            for key, value in report["dataset"].items()
            if key != "golden_file"
        },
        "variants": {
            variant: {
                "metrics": _strip_latency_metrics(data["metrics"]),
                "by_bucket": {
                    bucket: _strip_latency_metrics(metrics)
                    for bucket, metrics in data["by_bucket"].items()
                },
            }
            for variant, data in report["variants"].items()
        },
    }


def build_snapshot(
    *,
    queries: tuple[str, ...],
    variants: tuple[str, ...],
    top_n: int,
    include_metrics: bool,
    golden_file: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "chronovisor-search-pipeline-snapshot-v1",
        "top_n": top_n,
        "queries": [],
    }
    for query in queries:
        production_results, search_mode = search(query, top_n=top_n)
        row: dict[str, Any] = {
            "query": query,
            "production": {
                "search_mode": search_mode,
                "results": [_page_payload(page) for page in production_results],
            },
            "eval": {},
        }
        for variant in variants:
            result = run_variant(query, variant, top_n=top_n)
            row["eval"][variant] = {
                "results": [_page_payload(page) for page in result["results"]],
                "channels": result["channels"],
            }
        payload["queries"].append(row)
    if include_metrics:
        payload["metrics"] = _metrics_payload(golden_file, variants, top_n)
    return payload


def _compare(expected: Any, actual: Any, path: str, diffs: list[str]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            diffs.append(f"{path}.{key}: missing from actual")
        for key in sorted(actual_keys - expected_keys):
            diffs.append(f"{path}.{key}: unexpected in actual")
        for key in sorted(expected_keys & actual_keys):
            _compare(expected[key], actual[key], f"{path}.{key}", diffs)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length {len(expected)} != {len(actual)}")
            return
        for idx, (left, right) in enumerate(zip(expected, actual, strict=False)):
            _compare(left, right, f"{path}[{idx}]", diffs)
        return
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            left = float(expected)
            right = float(actual)
        except (TypeError, ValueError):
            diffs.append(f"{path}: {expected!r} != {actual!r}")
            return
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=SCORE_TOLERANCE):
            diffs.append(f"{path}: {left!r} != {right!r}")
        return
    if expected != actual:
        diffs.append(f"{path}: {expected!r} != {actual!r}")


def compare_snapshot(expected_path: Path, actual: dict[str, Any]) -> list[str]:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    diffs: list[str] = []
    _compare(expected, actual, "$", diffs)
    return diffs


def _parse_variants(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_VARIANTS
    variants = tuple(item.strip() for item in raw.split(",") if item.strip())
    return variants or DEFAULT_VARIANTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", help="Query to include. Repeatable.")
    parser.add_argument("--variants", help="Comma-separated eval variants.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--golden-file", default=str(GOLDEN_FILE))
    parser.add_argument("--include-metrics", action="store_true")
    parser.add_argument("--output", help="Write the snapshot JSON to this path.")
    parser.add_argument("--compare", help="Compare against an existing snapshot JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot = build_snapshot(
        queries=tuple(args.query) if args.query else DEFAULT_QUERIES,
        variants=_parse_variants(args.variants),
        top_n=args.top_n,
        include_metrics=args.include_metrics,
        golden_file=Path(args.golden_file).expanduser(),
    )
    if args.output:
        Path(args.output).expanduser().write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))

    if args.compare:
        diffs = compare_snapshot(Path(args.compare).expanduser(), snapshot)
        if diffs:
            for diff in diffs[:200]:
                print(diff, file=sys.stderr)
            if len(diffs) > 200:
                print(f"... {len(diffs) - 200} more diffs", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
