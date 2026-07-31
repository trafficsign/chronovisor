"""Fail-closed comparison harness for Recall speed challengers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from chronovisor.core.link_fix import atomic_write
from chronovisor.core.store import CHRONOVISOR_ROOT

DEFAULT_ARTIFACT = (
    CHRONOVISOR_ROOT / "runtime" / "search-eval" / "phase8-challengers.json"
)
EXCLUDED_PRODUCTION = {
    "dsi": "requires independent lab plan and corpus-update proof",
    "memory_lora": "cannot safely update the host model memory",
    "speculative_injection": "current-turn context pollution is irreversible",
}


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return round(float(ordered[index]), 3)


def _page_collection(limit: int = 0) -> tuple[list[str], list[str]]:
    """Return stable full-corpus page IDs/text for an isolated lab index."""

    from chronovisor.core.frontmatter import parse
    from chronovisor.search.index_store import get_store

    store = get_store()
    store.refresh_if_stale()
    metas = sorted(
        store.all_pages_meta(include_system=True),
        key=lambda row: str(row.get("page_id") or ""),
    )
    if limit > 0:
        metas = metas[:limit]
    page_ids: list[str] = []
    collection: list[str] = []
    from chronovisor.core.store import find_page

    for meta in metas:
        page_id = str(meta.get("page_id") or "")
        if not page_id:
            continue
        path = find_page(page_id)
        if path is None:
            continue
        try:
            _frontmatter, body = parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            body = ""
        title = str(meta.get("title") or page_id)
        page_ids.append(page_id)
        collection.append(f"{title}\n\n{body[:4000]}")
    return page_ids, collection


def measure_colbert(
    *,
    checkpoint: str,
    limit: int = 94,
    collection_limit: int = 0,
    root: Path | None = None,
) -> dict[str, Any]:
    """Build a separate real ColBERT index and evaluate manual/locked rows."""

    if importlib.util.find_spec("colbert") is None or importlib.util.find_spec(
        "faiss"
    ) is None:
        return {"status": "unavailable", "reason": "colbert_or_faiss_missing"}
    from colbert import Indexer, Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig
    from transformers import PreTrainedModel

    # colbert-ai 0.2.22 still targets the Transformers 4 loading contract.
    # Transformers 5 asks wrappers for this derived map, although ColBERT's
    # projection model has no tied parameters. Keep the compatibility shim
    # isolated to this optional lab process.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}  # type: ignore[attr-defined]
        original_move_missing = PreTrainedModel._move_missing_keys_from_meta_to_device
        original_adjust_keys = PreTrainedModel._adjust_missing_and_unexpected_keys

        def _move_missing_compat(
            self: Any,
            missing_keys: Any,
            device_map: Any,
            device_mesh: Any,
            hf_quantizer: Any,
        ) -> Any:
            return original_move_missing(
                self,
                set(missing_keys),
                device_map,
                device_mesh,
                hf_quantizer,
            )

        PreTrainedModel._move_missing_keys_from_meta_to_device = (  # type: ignore[method-assign]
            _move_missing_compat
        )

        def _adjust_keys_compat(self: Any, loading_info: Any) -> Any:
            unexpected = getattr(self, "_keys_to_ignore_on_load_unexpected", None)
            if isinstance(unexpected, list):
                self._keys_to_ignore_on_load_unexpected = set(unexpected)
            missing = getattr(self, "_keys_to_ignore_on_load_missing", None)
            if isinstance(missing, list):
                self._keys_to_ignore_on_load_missing = set(missing)
            return original_adjust_keys(self, loading_info)

        PreTrainedModel._adjust_missing_and_unexpected_keys = (  # type: ignore[method-assign]
            _adjust_keys_compat
        )

    from chronovisor.search.search_eval import GOLDEN_FILE, load_examples

    examples = load_examples(GOLDEN_FILE, limit=max(1, limit), source_filter="manual")
    page_ids, collection = _page_collection(collection_limit)
    if not examples or not collection:
        return {"status": "unavailable", "reason": "empty_eval_or_collection"}
    lab_root = root or CHRONOVISOR_ROOT / "runtime" / "search-eval" / "colbert"
    experiment = "manual-94"
    index_name = "chronovisor-pages"
    config = ColBERTConfig(
        root=str(lab_root),
        experiment=experiment,
        gpus=0,
        nranks=1,
        bsize=16,
        query_maxlen=64,
        doc_maxlen=180,
        nbits=2,
        kmeans_niters=4,
        # CPU lab execution should surface exceptions in this process instead
        # of leaving the parent blocked on a crashed spawn worker.
        avoid_fork_if_possible=True,
    )
    build_started = time.perf_counter()
    try:
        with Run().context(
            RunConfig(
                root=str(lab_root),
                experiment=experiment,
                nranks=1,
                avoid_fork_if_possible=True,
            )
        ):
            indexer = Indexer(checkpoint=checkpoint, config=config, verbose=0)
            indexer.index(name=index_name, collection=collection, overwrite=True)
            searcher = Searcher(
                index=index_name,
                checkpoint=checkpoint,
                collection=collection,
                config=config,
                verbose=0,
            )
            latencies: list[float] = []
            hits = 0
            positives = 0
            negative_hits = 0
            negative_cases = 0
            over_4s = 0
            for example in examples:
                started = time.perf_counter()
                pids, _ranks, _scores = searcher.search(example.query, k=20)
                elapsed = (time.perf_counter() - started) * 1_000
                latencies.append(elapsed)
                over_4s += elapsed > 4_000
                results = [page_ids[int(pid)] for pid in pids if int(pid) < len(page_ids)]
                if example.expected_pages:
                    positives += 1
                    hits += bool(set(example.expected_pages) & set(results[:5]))
                if example.negative_pages:
                    negative_cases += 1
                    negative_hits += bool(
                        set(example.negative_pages) & set(results[:20])
                    )
    except Exception as exc:
        return {
            "status": "failed",
            "reason": type(exc).__name__,
            "detail": str(exc)[:500],
            "traceback": traceback.format_exc()[-2_000:],
            "checkpoint": checkpoint,
            "documents": len(collection),
        }
    index_dir = lab_root / experiment / "indexes" / index_name
    resource_bytes = sum(
        path.stat().st_size for path in index_dir.rglob("*") if path.is_file()
    )
    return {
        "status": "measured",
        "backend": "colbert-ai",
        "checkpoint": checkpoint,
        "device": "cpu",
        "mps_supported": False,
        "examples": len(examples),
        "documents": len(collection),
        "recall_at_5": hits / positives if positives else 0.0,
        "negative_hit_at_20": (
            negative_hits / negative_cases if negative_cases else 0.0
        ),
        "p50_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_ms": _percentile(latencies, 0.95),
        "max_ms": round(max(latencies), 3) if latencies else 0.0,
        "over_4s": float(over_4s),
        "resource_bytes": float(resource_bytes),
        "index_build_ms": round((time.perf_counter() - build_started) * 1_000, 3),
        "incremental_supported": False,
        "incremental_reason": "colbert_indexer_requires_rebuild",
    }


def top_k_reproduction(
    teacher: list[list[str]],
    student: list[list[str]],
    *,
    k: int = 50,
) -> float:
    if not teacher or len(teacher) != len(student):
        return 0.0
    return sum(
        len(set(left[:k]) & set(right[:k])) / max(1, len(set(left[:k])))
        for left, right in zip(teacher, student, strict=True)
    ) / len(teacher)


def adoption_gate(
    baseline: dict[str, float],
    challenger: dict[str, float],
) -> dict[str, Any]:
    required = {
        "recall_at_5",
        "negative_hit_at_20",
        "p95_ms",
        "max_ms",
        "over_4s",
        "resource_bytes",
    }
    missing = sorted(required - set(challenger))
    failures: list[str] = []
    if missing:
        failures.append("missing_metrics")
    else:
        if challenger["recall_at_5"] < baseline["recall_at_5"] - 0.01:
            failures.append("recall_regression")
        if (
            challenger["negative_hit_at_20"]
            > baseline["negative_hit_at_20"]
        ):
            failures.append("negative_hit_regression")
        if challenger["p95_ms"] >= baseline["p95_ms"]:
            failures.append("no_p95_win")
        if challenger["max_ms"] >= baseline["max_ms"]:
            failures.append("no_max_win")
        if challenger["over_4s"] > 0:
            failures.append("over_4s")
        if challenger["resource_bytes"] <= 0:
            failures.append("resource_unmeasured")
    return {
        "status": "passed" if not failures else "rejected",
        "failures": failures,
        "missing": missing,
    }


def environment_probe() -> dict[str, Any]:
    ollama = shutil.which("ollama")
    ollama_version = ""
    if ollama:
        try:
            ollama_version = subprocess.run(
                [ollama, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            ollama_version = "probe_failed"
    return {
        "platform": platform.platform(),
        "colbert_dependency": importlib.util.find_spec("colbert") is not None,
        "faiss_dependency": importlib.util.find_spec("faiss") is not None,
        "coremltools_dependency": (
            importlib.util.find_spec("coremltools") is not None
        ),
        "ollama_executable": ollama or "",
        "ollama_version": ollama_version,
        "decoder_prefix_handle_api": False,
    }


def _measure_colbert_isolated(
    *, checkpoint: str, limit: int, collection_limit: int
) -> dict[str, Any]:
    """Contain native backend crashes and turn them into a fail-closed result."""

    worker_output = (
        CHRONOVISOR_ROOT
        / "runtime"
        / "search-eval"
        / f"colbert-worker-{os.getpid()}.json"
    )
    command = [
        sys.executable,
        "-m",
        "chronovisor.lab.recall_challengers",
        "--worker-output",
        str(worker_output),
        "--checkpoint",
        checkpoint,
        "--limit",
        str(limit),
        "--collection-limit",
        str(collection_limit),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=3_600,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "reason": "worker_timeout",
            "detail": str(exc),
            "checkpoint": checkpoint,
            "mps_supported": False,
        }
    if worker_output.exists():
        try:
            return json.loads(worker_output.read_text(encoding="utf-8"))
        finally:
            worker_output.unlink(missing_ok=True)
    output_tail = f"{completed.stdout}\n{completed.stderr}"[-2_000:]
    reason = "worker_exit"
    if "OMP: Error #15" in output_tail:
        reason = "macos_openmp_runtime_conflict"
    return {
        "status": "failed",
        "reason": reason,
        "detail": output_tail,
        "checkpoint": checkpoint,
        "returncode": completed.returncode,
        "mps_supported": False,
        "attempted_examples": limit,
        "attempted_collection_limit": collection_limit,
        "failed_before_query_evaluation": True,
    }


def run_report(
    *,
    baseline: dict[str, float],
    measurements: dict[str, dict[str, Any]] | None = None,
    output_file: Path = DEFAULT_ARTIFACT,
) -> dict[str, Any]:
    probe = environment_probe()
    measured = measurements or {}
    challengers = {
        "colbert": {
            "status": (
                "ready_for_locked_eval"
                if probe["colbert_dependency"] and probe["faiss_dependency"]
                else "unavailable"
            ),
            "reason": (
                ""
                if probe["colbert_dependency"] and probe["faiss_dependency"]
                else "colbert_or_faiss_dependency_missing"
            ),
            "index": "separate_colbert_index",
            "incremental": False,
        },
        "decoder_prefix_kv": {
            "status": "unavailable",
            "reason": "ollama_backend_exposes_no_reusable_prefix_handle",
            "scope": "fixed_prompt_plus_support_span_only",
            "per_page_cache": False,
        },
        "ane_query_distillation": {
            "status": "unavailable",
            "reason": (
                "student_artifact_missing"
                if probe["coremltools_dependency"]
                else "coremltools_dependency_missing"
            ),
            "required_top50_reproduction": 0.99,
        },
    }
    for name, challenger in challengers.items():
        measurement = measured.get(name, {})
        if measurement:
            challenger["measurement"] = measurement
            challenger["status"] = str(measurement.get("status") or "measured")
            challenger["reason"] = str(measurement.get("reason") or "")
        gate_metrics = (
            measurement if measurement.get("status") == "measured" else {}
        )
        challenger["gate"] = adoption_gate(baseline, gate_metrics)
        challenger["adopted"] = challenger["gate"]["status"] == "passed"
    winners = [
        name for name, challenger in challengers.items() if challenger["adopted"]
    ]
    winner = min(
        winners,
        key=lambda name: float(
            challengers[name]["measurement"].get("p95_ms", float("inf"))
        ),
        default=None,
    )
    payload = {
        "schema_version": 1,
        "generated_at_epoch": round(time.time(), 3),
        "baseline": baseline,
        "environment": probe,
        "challengers": challengers,
        "excluded_production": EXCLUDED_PRODUCTION,
        "winner": winner,
        # This lab records a winner but never mutates production by itself.
        "production_changed": False,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        output_file,
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )
    return {**payload, "output_file": str(output_file)}


def _baseline_from_artifact(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    latency = metrics.get("latency_ms", {})
    return {
        "recall_at_5": float(metrics.get("recall_at_5") or 0.0),
        "negative_hit_at_20": float(
            metrics.get("negative_hit_rate_at_20") or 0.0
        ),
        "p95_ms": float(latency.get("p95") or 0.0),
        "max_ms": float(latency.get("max") or 0.0),
        "over_4s": float(1 if float(latency.get("max") or 0.0) > 4_000 else 0),
        # The current BGE service is an external baseline process. The Phase 8
        # gate requires challengers to report a measured non-zero footprint.
        "resource_bytes": 1.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure fail-closed Recall speed challengers."
    )
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        default=(
            CHRONOVISOR_ROOT
            / "runtime"
            / "search-eval"
            / "recall-field-locked-e2e.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--measure-colbert", action="store_true")
    parser.add_argument(
        "--checkpoint", default="colbert-ir/colbertv2.0"
    )
    parser.add_argument("--limit", type=int, default=94)
    parser.add_argument("--collection-limit", type=int, default=0)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_output is not None:
        measurement = measure_colbert(
            checkpoint=args.checkpoint,
            limit=max(1, args.limit),
            collection_limit=max(0, args.collection_limit),
        )
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            args.worker_output,
            json.dumps(measurement, ensure_ascii=False, sort_keys=True) + "\n",
        )
        return 0
    if not args.baseline_artifact.exists():
        parser.error(f"baseline artifact missing: {args.baseline_artifact}")
    measurements: dict[str, dict[str, Any]] = {}
    if args.measure_colbert:
        measurements["colbert"] = _measure_colbert_isolated(
            checkpoint=args.checkpoint,
            limit=max(1, args.limit),
            collection_limit=max(0, args.collection_limit),
        )
    report = run_report(
        baseline=_baseline_from_artifact(args.baseline_artifact),
        measurements=measurements,
        output_file=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
