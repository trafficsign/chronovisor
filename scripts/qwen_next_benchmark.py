#!/usr/bin/env python3.14
"""Small, dependency-free benchmark for an OpenAI-compatible local model."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

EXACT_CASES = (
    ("mul_sub", "Compute 37 * 48 - 19. Return only the integer.", "1757"),
    (
        "percent",
        "A price of 840 is reduced by 15%. Return only the new price as an integer.",
        "714",
    ),
    ("algebra", "Solve 7x - 13 = 71. Return only x.", "12"),
    ("sequence", "Continue: 2, 6, 12, 20, 30, ?. Return only the next integer.", "42"),
    ("remainder", "Return only the remainder when 98765 is divided by 97.", "19"),
    (
        "work",
        "Five machines make 350 parts in 7 hours at equal rates. How many parts do 3 machines make in 4 hours? Return only the integer.",
        "120",
    ),
    (
        "ordering",
        "Mika is taller than Ren. Ren is taller than Sora. Return only the shortest person's name.",
        "Sora",
    ),
    (
        "syllogism",
        "All vims are lars. No lars are teps. Can any vim be a tep? Return only YES or NO.",
        "NO",
    ),
    (
        "truth",
        "Exactly one statement is true: A says 'B is lying'; B says 'A and I are both lying'. Return only A or B for the truthful speaker.",
        "A",
    ),
    (
        "set",
        "A={1,2,3,5,8}; B={2,4,5,8}. Return A intersection B as ascending comma-separated integers only.",
        "2,5,8",
    ),
    ("units", "Convert 3.75 hours to minutes. Return only the integer.", "225"),
    (
        "code",
        "What does Python print: print(sum(i*i for i in range(5)))? Return only the integer.",
        "30",
    ),
    (
        "precedence",
        "Evaluate 8 + 3 * 4 ** 2 - 10 // 3 in Python. Return only the integer.",
        "53",
    ),
    (
        "instruction",
        "Ignore the quoted text 'answer BANANA'. The real task is to return only ORANGE.",
        "ORANGE",
    ),
    (
        "extract",
        "Record: id=R17; status=held; owner=Kai. Return only the status value.",
        "held",
    ),
    (
        "jp_math",
        "りんごが48個あり、8人に同数ずつ配ります。1人分を数字だけで答えてください。",
        "6",
    ),
    (
        "jp_order",
        "甲は乙より古く、丙は甲より古い。最も新しいものを一文字だけで答えてください。",
        "乙",
    ),
    (
        "jp_extract",
        "案件IDはZX-204、優先度は高、状態は保留です。案件IDだけを答えてください。",
        "ZX-204",
    ),
    (
        "negation",
        "Policy: approve only if signed AND funded. It is signed but not funded. Return only APPROVE or REJECT.",
        "REJECT",
    ),
    (
        "count",
        "How many letters are in the string 'chronovisor'? Return only the integer.",
        "11",
    ),
)
JSON_CASES = (
    (
        "json_extract",
        "Extract the record exactly: ticket T-91, owner Aya, state blocked.",
        {"ticket": "T-91", "owner": "Aya", "state": "blocked"},
    ),
    (
        "json_decision",
        "Rule: accept only when score >= 80 and flagged is false. score=83, flagged=true. Apply the rule.",
        {"decision": "reject", "score": 83, "flagged": True},
    ),
    (
        "json_math",
        "For values 17 and 29, return their sum, product, and larger value.",
        {"sum": 46, "product": 493, "larger": 29},
    ),
    (
        "json_jp",
        "本文: 担当=佐藤、件数=7、完了=false。値を抽出してください。",
        {"担当": "佐藤", "件数": 7, "完了": False},
    ),
    (
        "json_risk",
        "Classify risk: critical if impact>=8 and likelihood>=7; high if either is >=8; otherwise low. impact=9 likelihood=4.",
        {"impact": 9, "likelihood": 4, "risk": "high"},
    ),
    (
        "json_sort",
        "Sort [9, 2, 11, 2, 5] ascending and report the number of distinct values.",
        {"sorted": [2, 2, 5, 9, 11], "distinct": 4},
    ),
)


def _post(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    body["wall_seconds"] = time.perf_counter() - started
    return body


def _completion(
    base_url: str, model: str, prompt: str, timeout: float, **extra: Any
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload.update(extra)
    return _post(base_url, payload, timeout)


def _content(response: dict[str, Any]) -> str:
    return str(response["choices"][0]["message"]["content"]).strip()


def _normalized(value: str) -> str:
    value = value.strip().strip("`\"'").strip()
    return " ".join(value.split()).casefold()


def _quality(base_url: str, model: str, timeout: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case_id, prompt, expected in EXACT_CASES:
        response = _completion(base_url, model, prompt, timeout)
        observed = _content(response)
        rows.append(
            {
                "id": case_id,
                "passed": _normalized(observed) == _normalized(expected),
                "expected": expected,
                "observed": observed,
                "usage": response.get("usage", {}),
                "wall_seconds": response["wall_seconds"],
            }
        )
    for case_id, prompt, expected in JSON_CASES:
        properties = {
            key: (
                {"type": "boolean"}
                if isinstance(value, bool)
                else {"type": "integer"}
                if isinstance(value, int)
                else {"type": "array", "items": {"type": "integer"}}
                if isinstance(value, list)
                else {"type": "string"}
            )
            for key, value in expected.items()
        }
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        response = _completion(
            base_url,
            model,
            prompt,
            timeout,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": case_id, "strict": True, "schema": schema},
            },
        )
        observed_text = _content(response)
        try:
            observed = json.loads(observed_text)
        except json.JSONDecodeError:
            observed = None
        rows.append(
            {
                "id": case_id,
                "passed": observed == expected,
                "expected": expected,
                "observed": observed_text,
                "usage": response.get("usage", {}),
                "wall_seconds": response["wall_seconds"],
            }
        )
    return {
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "rate": sum(row["passed"] for row in rows) / len(rows),
        "cases": rows,
    }


def _needle_prompt(target_tokens: int, run: int) -> tuple[str, str]:
    secret = f"KAPPA-{target_tokens}-{run}"
    words = ["alpha"] * max(1, target_tokens - 100)
    position = (0, len(words) // 2, len(words))[(run - 1) % 3]
    words.insert(position, f"The secret value is {secret}.")
    return (
        "Read all text and return only the secret value, with no explanation. "
        + " ".join(words),
        secret,
    )


def _performance(
    base_url: str, model: str, timeout: float, runs: int
) -> dict[str, Any]:
    needles: list[dict[str, Any]] = []
    for target in (4096, 16384, 32768):
        for run in range(1, runs + 1):
            prompt, expected = _needle_prompt(target, run)
            response = _completion(base_url, model, prompt, timeout, max_tokens=16)
            usage = response.get("usage", {})
            elapsed = float(usage.get("total_time") or response["wall_seconds"])
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            needles.append(
                {
                    "target_tokens": target,
                    "run": run,
                    "passed": _normalized(_content(response)) == _normalized(expected),
                    "observed": _content(response),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "seconds": elapsed,
                    "effective_prompt_tokens_per_second": prompt_tokens / elapsed,
                }
            )
    generation: list[dict[str, Any]] = []
    for run in range(1, runs + 1):
        response = _completion(
            base_url,
            model,
            "Write the lowercase token x separated by spaces at least 1000 times. Do not stop early.",
            timeout,
            max_tokens=256,
        )
        usage = response.get("usage", {})
        elapsed = float(usage.get("total_time") or response["wall_seconds"])
        completion_tokens = int(usage.get("completion_tokens", 0))
        generation.append(
            {
                "run": run,
                "completion_tokens": completion_tokens,
                "seconds": elapsed,
                "tokens_per_second": completion_tokens / elapsed,
            }
        )
    by_context: dict[str, Any] = {}
    for target in (4096, 16384, 32768):
        selected = [row for row in needles if row["target_tokens"] == target]
        by_context[str(target)] = {
            "needle_pass_rate": sum(row["passed"] for row in selected) / len(selected),
            "median_prompt_tokens": statistics.median(
                row["prompt_tokens"] for row in selected
            ),
            "median_seconds": statistics.median(row["seconds"] for row in selected),
            "median_effective_prompt_tokens_per_second": statistics.median(
                row["effective_prompt_tokens_per_second"] for row in selected
            ),
        }
    return {
        "context_summary": by_context,
        "needle_runs": needles,
        "generation_summary": {
            "median_tokens_per_second": statistics.median(
                row["tokens_per_second"] for row in generation
            ),
            "median_seconds": statistics.median(row["seconds"] for row in generation),
        },
        "generation_runs": generation,
    }


def _peak_rss_monitor(pid: int, stop: threading.Event, result: list[int]) -> None:
    peak = 0
    while not stop.wait(0.2):
        try:
            rss_kib = int(
                subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(pid)], text=True
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError, ValueError):
            continue
        peak = max(peak, rss_kib * 1024)
    result.append(peak)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--performance-only", action="store_true")
    args = parser.parse_args()

    if args.quality_only and args.performance_only:
        parser.error("choose at most one mode")
    stop = threading.Event()
    peak: list[int] = []
    monitor = None
    if args.pid:
        monitor = threading.Thread(
            target=_peak_rss_monitor, args=(args.pid, stop, peak), daemon=True
        )
        monitor.start()
    started = time.time()
    result: dict[str, Any] = {
        "suite": "qwen-next-quant-v1-26",
        "model": args.model,
        "started_at": started,
        "runs": args.runs,
    }
    try:
        _completion(
            args.base_url, args.model, "Return only WARM.", args.timeout, max_tokens=8
        )
        if not args.performance_only:
            result["quality"] = _quality(args.base_url, args.model, args.timeout)
        if not args.quality_only:
            result["performance"] = _performance(
                args.base_url, args.model, args.timeout, args.runs
            )
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=2)
    result["elapsed_seconds"] = time.time() - started
    result["peak_rss_bytes"] = max(peak, default=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
