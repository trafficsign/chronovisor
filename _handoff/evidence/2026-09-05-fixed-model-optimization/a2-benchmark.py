#!/usr/bin/env python3
"""Measure the A2 claim-readback and read-helper optimizations.

The legacy claim path and ``ConvergenceStore.load/get/list_items`` methods are
compiled from commit ``7b872fa``.  The optimized path uses the current source,
so both comparisons exercise the same state schema and caller behavior while
isolating removed reads and redundant defensive copies without starting a
model or service.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.ingest.convergence import ConvergenceStore, CycleBudget
from chronovisor.ops import lint_repair

NOW = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)


def _legacy_symbols() -> tuple[type[ConvergenceStore], Any]:
    """Compile only the pre-change methods against current module globals."""

    repo = Path(__file__).resolve().parents[3]
    old_convergence_source = subprocess.check_output(
        ["git", "show", "7b872fa:src/chronovisor/ingest/convergence.py"],
        cwd=repo,
        text=True,
    )
    old_tree = ast.parse(old_convergence_source)
    old_methods = {
        node.name: node
        for node in next(
            node
            for node in old_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ConvergenceStore"
        ).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"load", "get", "list_items"}
    }
    namespace = dict(vars(sys.modules[ConvergenceStore.__module__]))
    namespace["__name__"] = ConvergenceStore.__module__
    legacy_methods: dict[str, Any] = {}
    for name in ("load", "get", "list_items"):
        node = old_methods[name]
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(module, f"{repo}/src/chronovisor/ingest/convergence.py", "exec"), namespace)
        legacy_methods[name] = namespace[name]
    legacy_store = type("LegacyConvergenceStore", (ConvergenceStore,), legacy_methods)

    old_lint_source = subprocess.check_output(
        ["git", "show", "7b872fa:src/chronovisor/ops/lint_repair.py"],
        cwd=repo,
        text=True,
    )
    old_lint_tree = ast.parse(old_lint_source)
    old_process = next(
        node
        for node in old_lint_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_process_tag_candidate"
    )
    lint_namespace = dict(vars(lint_repair))
    lint_namespace["__name__"] = lint_repair.__name__
    exec(
        compile(
            ast.Module(body=[old_process], type_ignores=[]),
            f"{repo}/src/chronovisor/ops/lint_repair.py",
            "exec",
        ),
        lint_namespace,
    )
    return legacy_store, lint_namespace["_process_tag_candidate"]


LEGACY_CONVERGENCE_STORE, LEGACY_PROCESS_TAG_CANDIDATE = _legacy_symbols()


def _run(mode: str, *, items: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="chronovisor-a2-") as name:
        root = Path(name)
        page = root / "pages" / "target.md"
        page.parent.mkdir(parents=True)
        page_text = (
            "---\n"
            "title: Target\n"
            "updated: 2026-01-01\n"
            "status: stable\n"
            "type: knowledge\n"
            "---\n\n"
            "# Page\n\nUseful content.\n"
        )
        page.write_text(page_text, encoding="utf-8")
        queue = root / "review" / "lint-repair-queue.jsonl"
        queue.parent.mkdir(parents=True)
        row = {
            "type": "lint_repair_candidate",
            "issue_key": "synthetic-target",
            "lane": "heavy_model_batch",
            "issue_type": "tag_missing",
            "severity": "high",
            "page": "target",
            "detail": "synthetic",
            "auto_fixable": False,
        }
        queue.write_text(json.dumps(row) + "\n", encoding="utf-8")
        store_type = (
            LEGACY_CONVERGENCE_STORE if mode == "legacy" else ConvergenceStore
        )
        store = store_type(root / "runtime" / "convergence" / "state.json")
        source_id, input_data = lint_repair._candidate_identity(row, page_text)
        store.merge_item(
            lane="lint_repair",
            source_id=source_id,
            input_data=input_data,
            resolver_version=lint_repair.REPAIR_RESOLVER_VERSION,
            metadata=row,
            now=NOW,
        )
        state = store.load()
        template = {
            "schema_version": 1,
            "lane": "other",
            "input_hash": "0" * 64,
            "resolver_version": "v1",
            "metadata": {},
            "status": "applied",
            "local_attempts": 0,
            "frontier_attempts": 0,
            "next_attempt_at": None,
            "lease_stage": None,
            "lease_owner": None,
            "lease_expires_at": None,
            "last_error": None,
            "last_failure_class": None,
            "human_required": False,
            "quarantine_reason": None,
            "result": None,
            "created_at": NOW.isoformat(),
            "updated_at": NOW.isoformat(),
        }
        for index in range(items):
            key = f"synthetic-{index:05d}"
            state["items"][key] = {**template, "key": key, "source_id": key}
        store.state_file.write_text(
            json.dumps(state, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        real_load = store._load_unlocked
        load_calls = 0

        def counted_load() -> dict[str, Any]:
            nonlocal load_calls
            load_calls += 1
            return real_load()

        store._load_unlocked = counted_load  # type: ignore[method-assign]
        original_find_page = lint_repair.chronovisor_store.find_page
        original_process = lint_repair._process_tag_candidate
        lint_repair.chronovisor_store.find_page = lambda _page_id: page
        if mode == "legacy":
            lint_repair._process_tag_candidate = LEGACY_PROCESS_TAG_CANDIDATE

        def unavailable_reviewer(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("synthetic reviewer unavailable")

        try:
            gc.collect()
            started = time.perf_counter_ns()
            result = lint_repair.run_lint_repair(
                queue_file=queue,
                store=store,
                budget=CycleBudget(
                    max_local_calls=20,
                    max_frontier_calls=20,
                    max_mutations=20,
                    max_elapsed_seconds=60,
                ),
                local_reviewer=unavailable_reviewer,
                frontier_reviewer=unavailable_reviewer,
                now=NOW,
            )
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        finally:
            lint_repair.chronovisor_store.find_page = original_find_page
            lint_repair._process_tag_candidate = original_process
        return {
            "elapsed_ms": elapsed_ms,
            "load_calls": load_calls,
            "result_status": result["results"][0]["status"],
        }


def _paired_summary(*, pairs: int, items: int) -> list[dict[str, Any]]:
    samples: dict[str, list[dict[str, Any]]] = {"legacy": [], "optimized": []}
    for pair in range(pairs):
        modes = (
            ("legacy", "optimized")
            if pair % 2 == 0
            else ("optimized", "legacy")
        )
        for mode in modes:
            samples[mode].append(_run(mode, items=items))
    return [
        _summarize_samples(mode, samples[mode], pairs=pairs, items=items)
        for mode in ("legacy", "optimized")
    ]


def _summarize_samples(
    mode: str,
    samples: list[dict[str, Any]],
    *,
    pairs: int,
    items: int,
) -> dict[str, Any]:
    durations = [sample["elapsed_ms"] for sample in samples]
    ordered = sorted(durations)
    p95 = ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))]
    return {
        "mode": mode,
        "pairs": pairs,
        "synthetic_state_items": items,
        "elapsed_ms": durations,
        "median_ms": statistics.median(durations),
        "p95_ms": p95,
        "load_calls": sorted({sample["load_calls"] for sample in samples}),
        "result_status": sorted({sample["result_status"] for sample in samples}),
    }


def _build_synthetic_state(
    items: int,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temporary = tempfile.TemporaryDirectory(prefix="chronovisor-a2-read-")
    state_path = Path(temporary.name) / "state.json"
    template = {
        "schema_version": 1,
        "lane": "other",
        "input_hash": "0" * 64,
        "resolver_version": "v1",
        "metadata": {"x": "y"},
        "status": "applied",
        "local_attempts": 0,
        "frontier_attempts": 0,
        "next_attempt_at": None,
        "lease_stage": None,
        "lease_owner": None,
        "lease_expires_at": None,
        "last_error": None,
        "last_failure_class": None,
        "human_required": False,
        "quarantine_reason": None,
        "result": None,
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
    }
    payload = {
        "schema_version": 1,
        "items": {
            f"synthetic-{index:05d}": {
                **template,
                "key": f"synthetic-{index:05d}",
                "source_id": f"synthetic-{index:05d}",
            }
            for index in range(items)
        },
    }
    state_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return temporary, state_path


def _lookup_key(state_path: Path) -> str:
    """Select an existing key before timing, without using a store read."""

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, dict) or not items:
        raise ValueError("benchmark state contains no items")
    key = next(iter(items))
    if not isinstance(key, str):
        raise ValueError("benchmark state item key is not a string")
    return key


def _returned_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_read_helper_equivalence(
    *,
    state_path: Path | None,
    items: int,
) -> dict[str, bool]:
    """Check one legacy/current return digest pair for each read helper."""

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if state_path is None:
        temporary, state_path = _build_synthetic_state(items)
    try:
        key = _lookup_key(state_path)
        legacy = LEGACY_CONVERGENCE_STORE(state_path)
        optimized = ConvergenceStore(state_path)
        get_equal = _returned_digest(legacy.get(key)) == _returned_digest(
            optimized.get(key)
        )
        list_equal = _returned_digest(
            legacy.list_items(statuses={"applied"})
        ) == _returned_digest(optimized.list_items(statuses={"applied"}))
        if not (get_equal and list_equal):
            raise AssertionError("legacy/current read helper digest mismatch")
        return {"get": get_equal, "list": list_equal}
    finally:
        if temporary is not None:
            temporary.cleanup()


def _paired_read_helper_summary(
    operation: str,
    *,
    pairs: int,
    state_path: Path | None,
    items: int,
) -> list[dict[str, Any]]:
    samples: dict[str, list[float]] = {"legacy": [], "optimized": []}
    fixed_lookup_key = _lookup_key(state_path) if state_path is not None else None
    for pair in range(pairs):
        modes = (
            ("legacy", "optimized")
            if pair % 2 == 0
            else ("optimized", "legacy")
        )
        for mode in modes:
            samples[mode].append(
                _read_helper_once(
                    mode,
                    operation,
                    state_path=state_path,
                    items=items,
                    lookup_key=fixed_lookup_key,
                )
            )
    return [
        _summarize_read_helper(
            mode,
            operation,
            samples[mode],
            pairs=pairs,
            state_path=state_path,
            items=items,
        )
        for mode in ("legacy", "optimized")
    ]


def _summarize_read_helper(
    mode: str,
    operation: str,
    samples: list[float],
    *,
    pairs: int,
    state_path: Path | None,
    items: int,
) -> dict[str, Any]:
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))]
    return {
        "mode": mode,
        "operation": operation,
        "pairs": pairs,
        "state_path": str(state_path) if state_path is not None else "synthetic",
        "synthetic_state_items": items if state_path is None else None,
        "elapsed_ms": samples,
        "median_ms": statistics.median(samples),
        "p95_ms": p95,
    }


def _read_helper_once(
    mode: str,
    operation: str,
    *,
    state_path: Path | None,
    items: int,
    lookup_key: str | None = None,
) -> float:
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if state_path is None:
        temporary, state_path = _build_synthetic_state(items)
    if lookup_key is None:
        lookup_key = _lookup_key(state_path)
    store_type = LEGACY_CONVERGENCE_STORE if mode == "legacy" else ConvergenceStore
    store = store_type(state_path)
    started = time.perf_counter_ns()
    if operation == "get":
        store.get(lookup_key)
    else:
        store.list_items(statuses={"applied"})
    elapsed = (time.perf_counter_ns() - started) / 1e6
    if temporary is not None:
        temporary.cleanup()
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--items", type=int, default=2_000)
    parser.add_argument(
        "--mode",
        choices=("both", "legacy", "optimized"),
        default="both",
        help="compare both modes, or run one mode for an isolated RSS sample",
    )
    parser.add_argument(
        "--operation",
        choices=("all", "claim", "get", "list"),
        default="all",
        help="benchmark all operations or one operation",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        help="read an existing state snapshot instead of generating synthetic data",
    )
    args = parser.parse_args()
    if args.pairs < 1 or args.items < 1:
        parser.error("--pairs and --items must be >= 1")
    runs: list[dict[str, Any]] = []
    read_helpers: list[dict[str, Any]] = []
    read_helper_equivalence: dict[str, bool] | None = None
    if args.operation in {"all", "claim"}:
        if args.mode == "both":
            runs = _paired_summary(pairs=args.pairs, items=args.items)
        else:
            samples = [_run(args.mode, items=args.items) for _ in range(args.pairs)]
            runs = [
                _summarize_samples(
                    args.mode,
                    samples,
                    pairs=args.pairs,
                    items=args.items,
                )
            ]
    if args.operation in {"all", "get", "list"}:
        if args.mode == "both":
            read_helper_equivalence = _verify_read_helper_equivalence(
                state_path=args.state_path,
                items=args.items,
            )
        operations = ("get", "list") if args.operation == "all" else (args.operation,)
        if args.mode == "both":
            read_helpers = [
                result
                for operation in operations
                for result in _paired_read_helper_summary(
                    operation,
                    pairs=args.pairs,
                    state_path=args.state_path,
                    items=args.items,
                )
            ]
        else:
            fixed_lookup_key = (
                _lookup_key(args.state_path) if args.state_path is not None else None
            )
            for operation in operations:
                samples = [
                    _read_helper_once(
                        args.mode,
                        operation,
                        state_path=args.state_path,
                        items=args.items,
                        lookup_key=fixed_lookup_key,
                    )
                    for _ in range(args.pairs)
                ]
                read_helpers.append(
                    _summarize_read_helper(
                        args.mode,
                        operation,
                        samples,
                        pairs=args.pairs,
                        state_path=args.state_path,
                        items=args.items,
                    )
                )
    isolated_peak_rss_bytes = None
    if args.mode != "both" and args.pairs:
        isolated_peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "baseline_reference": "7b872fa:src/chronovisor/ops/lint_repair.py",
                "python": sys.version.split()[0],
                "mode": args.mode,
                "operation": args.operation,
                "runs": runs,
                "read_helpers": read_helpers,
                **(
                    {"read_helper_equivalence": read_helper_equivalence}
                    if read_helper_equivalence is not None
                    else {}
                ),
                **(
                    {"isolated_peak_rss_bytes": isolated_peak_rss_bytes}
                    if isolated_peak_rss_bytes is not None
                    else {}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
