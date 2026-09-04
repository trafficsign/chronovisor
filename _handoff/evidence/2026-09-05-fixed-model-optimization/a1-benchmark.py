#!/usr/bin/env python3
"""Compare the legacy and optimized recall JSONL tail readers.

The benchmark extracts both helpers without importing the application, so it
does not load models, services, or Chronovisor runtime state.  It reports only
timings, counts, file metadata, and isolated peak RSS values.
"""

from __future__ import annotations

import argparse
import ast
import codecs
import hashlib
import json
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
BASELINE = "7b872fa"
SOURCE_PATH = "src/chronovisor/recall/recall_auditor.py"
NEWLINE_RE = re.compile(rb"[\r\n]")


def _load_function(source: str) -> Callable[[Path, int], list[dict[str, Any]]]:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "read_jsonl_tail"
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "codecs": codecs,
        "deque": deque,
        "json": json,
        "re": re,
        "_JSONL_NEWLINE_BYTES": NEWLINE_RE,
        "_JSONL_READ_CHUNK_BYTES": 1 << 15,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<jsonl-tail>", "exec"), namespace)
    return namespace["read_jsonl_tail"]


def _legacy_function() -> Callable[[Path, int], list[dict[str, Any]]]:
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{SOURCE_PATH}"], cwd=REPO, text=True
    )
    return _load_function(source)


def _current_function() -> Callable[[Path, int], list[dict[str, Any]]]:
    return _load_function((REPO / SOURCE_PATH).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _worker(mode: str, path: Path, limit: int) -> None:
    function = _legacy_function() if mode == "legacy" else _current_function()
    rows = function(path, limit)
    print(json.dumps({"records": len(rows), "rss_mib": _rss_mib()}))


def _isolated_rss(mode: str, path: Path, limit: int) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            sys.executable,
            str(Path(__file__)),
            "--worker",
            mode,
            str(path),
            "--limit",
            str(limit),
        ],
        cwd=REPO,
        text=True,
    )
    return json.loads(output)


def _comparison(
    legacy: Callable[[Path, int], list[dict[str, Any]]],
    optimized: Callable[[Path, int], list[dict[str, Any]]],
    path: Path,
    limit: int,
    pairs: int,
) -> dict[str, Any]:
    legacy_ms: list[float] = []
    optimized_ms: list[float] = []
    equivalent = True
    record_count = 0
    for pair in range(pairs):
        first, second = (legacy, optimized) if pair % 2 == 0 else (optimized, legacy)
        started = time.perf_counter_ns()
        first_rows = first(path, limit)
        first_ms = (time.perf_counter_ns() - started) / 1e6
        started = time.perf_counter_ns()
        second_rows = second(path, limit)
        second_ms = (time.perf_counter_ns() - started) / 1e6
        if pair % 2 == 0:
            legacy_rows, optimized_rows = first_rows, second_rows
            legacy_ms.append(first_ms)
            optimized_ms.append(second_ms)
        else:
            legacy_rows, optimized_rows = second_rows, first_rows
            legacy_ms.append(second_ms)
            optimized_ms.append(first_ms)
        equivalent &= legacy_rows == optimized_rows
        record_count = len(legacy_rows)
    legacy_median = statistics.median(legacy_ms)
    optimized_median = statistics.median(optimized_ms)
    return {
        "limit": limit,
        "pairs": pairs,
        "legacy_ms": legacy_ms,
        "optimized_ms": optimized_ms,
        "legacy_median_ms": legacy_median,
        "optimized_median_ms": optimized_median,
        "legacy_p95_ms": _p95(legacy_ms),
        "optimized_p95_ms": _p95(optimized_ms),
        "median_change_percent": (optimized_median / legacy_median - 1) * 100,
        "records": record_count,
        "result_equivalence": equivalent,
    }


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", nargs="?", type=Path)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--worker", choices=("legacy", "optimized"))
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        if args.snapshot is None:
            parser.error("worker requires a snapshot")
        _worker(args.worker, args.snapshot, args.limit)
        return
    if args.snapshot is None:
        parser.error("snapshot is required")
    if args.pairs < 1:
        parser.error("--pairs must be >= 1")
    path = args.snapshot
    before = path.stat()
    legacy = _legacy_function()
    optimized = _current_function()
    comparisons = [
        _comparison(legacy, optimized, path, limit, args.pairs)
        for limit in (500, 5000)
    ]
    after = path.stat()
    snapshot_rss: list[dict[str, Any]] = []
    for limit in (500, 5000):
        modes: dict[str, Any] = {}
        for mode in ("legacy", "optimized"):
            samples = [_isolated_rss(mode, path, limit) for _ in range(3)]
            modes[mode] = {
                "rss_mib_samples": [sample["rss_mib"] for sample in samples],
                "rss_mib_median": statistics.median(
                    sample["rss_mib"] for sample in samples
                ),
                "records": sorted(sample["records"] for sample in samples)[1],
            }
        snapshot_rss.append({"limit": limit, "modes": modes})
    with tempfile.TemporaryDirectory(prefix="chronovisor-a1-cr-") as name:
        root = Path(name)
        cr_rss: list[dict[str, Any]] = []
        for count in (50_000, 500_000):
            cr_path = root / f"cr-{count}.jsonl"
            cr_path.write_bytes(b'{"id":1}\r' * count)
            modes: dict[str, Any] = {}
            for mode in ("legacy", "optimized"):
                samples = [_isolated_rss(mode, cr_path, 3) for _ in range(3)]
                modes[mode] = {
                    "rss_mib_samples": [sample["rss_mib"] for sample in samples],
                    "rss_mib_median": statistics.median(
                        sample["rss_mib"] for sample in samples
                    ),
                    "records": sorted(sample["records"] for sample in samples)[
                        len(samples) // 2
                    ],
                }
            cr_rss.append(
                {
                    "lines": count,
                    "bytes": cr_path.stat().st_size,
                    "modes": modes,
                }
            )
    print(
        json.dumps(
            {
                "baseline_reference": f"{BASELINE}:{SOURCE_PATH}:read_jsonl_tail",
                "python": sys.version.split()[0],
                "snapshot_bytes": before.st_size,
                "snapshot_sha256": _sha256(path),
                "snapshot_identity_same": (
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                )
                == (after.st_ino, after.st_size, after.st_mtime_ns),
                "comparisons": comparisons,
                "snapshot_rss": snapshot_rss,
                "cr_only_rss": cr_rss,
                "models_or_services": "none",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
