#!/usr/bin/env python3
"""Run Chronovisor's canonical decision corpus against an OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from chronovisor.decision.decision_router import decision_effective_request
from chronovisor.decision.local_model_eval import replay_semantic_effect
from chronovisor.decision.local_structured import normalize_json_output, validate_json


def post(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("BENCH_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    result["wall_seconds"] = time.perf_counter() - started
    return result


def run_case(
    row: dict[str, Any], base_url: str, model: str, timeout: float, max_tokens: int
) -> dict[str, Any]:
    prompt, system = decision_effective_request(
        prompt=row["prompt"],
        schema=row["schema"],
        system=row.get("system"),
        decision_lane=row.get("decision_lane"),
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "chronovisor_decision",
                "strict": True,
                "schema": row["schema"],
            },
        },
    }
    started = time.perf_counter()
    expected_effect = replay_semantic_effect(
        row["expected"],
        row["schema"],
        prompt=row["prompt"],
        decision_lane=row.get("decision_lane"),
    )
    try:
        response = post(base_url, payload, timeout)
        content = str(response["choices"][0]["message"]["content"] or "")
        normalized, _ = normalize_json_output(content)
        parsed = json.loads(normalized)
        validate_json(parsed, row["schema"])
        actual_effect = replay_semantic_effect(
            parsed,
            row["schema"],
            prompt=row["prompt"],
            decision_lane=row.get("decision_lane"),
        )
        error = None
        usage = response.get("usage", {})
        wall_seconds = response["wall_seconds"]
    except Exception as exc:
        content = ""
        parsed = None
        actual_effect = None
        error = f"{type(exc).__name__}: {exc}"
        usage = {}
        wall_seconds = time.perf_counter() - started
    return {
        "case_id": row.get("contract_id"),
        "lane": row.get("decision_lane"),
        "schema_valid": error is None,
        "effect_match": error is None and actual_effect == expected_effect,
        "exact_json_match": error is None and parsed == row["expected"],
        "expected_effect": expected_effect,
        "actual_effect": actual_effect,
        "observed": content[:4000],
        "error": error,
        "usage": usage,
        "wall_seconds": wall_seconds,
    }


def monitor_rss(pid: int, stop: threading.Event, samples: list[int]) -> None:
    while not stop.wait(0.2):
        try:
            kib = int(
                subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(pid)], text=True
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError, ValueError):
            continue
        samples.append(kib * 1024)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_lane[str(row["lane"])].append(row)

    def rates(selected: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(selected)
        latencies = [float(row["wall_seconds"]) for row in selected]
        return {
            "total": total,
            "schema_valid": sum(bool(row["schema_valid"]) for row in selected),
            "schema_rate": sum(bool(row["schema_valid"]) for row in selected) / total,
            "effect_matches": sum(bool(row["effect_match"]) for row in selected),
            "effect_match_rate": sum(bool(row["effect_match"]) for row in selected)
            / total,
            "exact_json_matches": sum(
                bool(row["exact_json_match"]) for row in selected
            ),
            "median_wall_seconds": statistics.median(latencies),
            "p95_wall_seconds": sorted(latencies)[max(0, int(total * 0.95) - 1)],
        }

    return {
        "overall": rates(rows),
        "by_lane": {lane: rates(selected) for lane, selected in sorted(by_lane.items())},
    }


def select_rows(rows: list[dict[str, Any]], per_lane: int) -> list[dict[str, Any]]:
    if per_lane == 0:
        return rows
    counts: dict[str, int] = defaultdict(int)
    selected = []
    for row in rows:
        lane = str(row.get("decision_lane"))
        if counts[lane] < per_lane:
            selected.append(row)
            counts[lane] += 1
    return selected


def self_check() -> None:
    rows = [
        {
            "lane": "x",
            "schema_valid": True,
            "effect_match": True,
            "exact_json_match": False,
            "wall_seconds": 1.0,
        },
        {
            "lane": "x",
            "schema_valid": False,
            "effect_match": False,
            "exact_json_match": False,
            "wall_seconds": 3.0,
        },
    ]
    assert summarize(rows)["overall"] == {
        "total": 2,
        "schema_valid": 1,
        "schema_rate": 0.5,
        "effect_matches": 1,
        "effect_match_rate": 0.5,
        "exact_json_matches": 0,
        "median_wall_seconds": 2.0,
        "p95_wall_seconds": 1.0,
    }
    assert [row["id"] for row in select_rows([
        {"id": 1, "decision_lane": "a"},
        {"id": 2, "decision_lane": "a"},
        {"id": 3, "decision_lane": "b"},
    ], 1)] == [1, 3]


def main() -> int:
    if "--self-check" in sys.argv:
        self_check()
        print("self-check: ok")
        return 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=660)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-lane", type=int, default=0)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()

    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    selected = select_rows(corpus, args.per_lane)
    selected = selected[
        args.offset : None if args.limit == 0 else args.offset + args.limit
    ]
    stop = threading.Event()
    rss_samples: list[int] = []
    monitor = None
    if args.pid:
        monitor = threading.Thread(
            target=monitor_rss, args=(args.pid, stop, rss_samples), daemon=True
        )
        monitor.start()
    started = time.time()
    try:
        rows = [
            run_case(row, args.base_url, args.model, args.timeout, args.max_tokens)
            for row in selected
        ]
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=2)
    result = {
        "suite": "chronovisor-canonical-lane-contract-v1",
        "model": args.model,
        "corpus": str(args.corpus.resolve()),
        "offset": args.offset,
        "limit": args.limit,
        "per_lane": args.per_lane,
        "started_at": started,
        "elapsed_seconds": time.time() - started,
        "peak_rss_bytes": max(rss_samples, default=0),
        "summary": summarize(rows),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
