"""Fail-closed comparison harness for Recall speed challengers."""

from __future__ import annotations

import importlib.util
import json
import platform
import time
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
    return {
        "platform": platform.platform(),
        "colbert_dependency": importlib.util.find_spec("colbert") is not None,
        "coremltools_dependency": (
            importlib.util.find_spec("coremltools") is not None
        ),
        "decoder_prefix_handle_api": False,
    }


def run_report(
    *,
    baseline: dict[str, float],
    output_file: Path = DEFAULT_ARTIFACT,
) -> dict[str, Any]:
    probe = environment_probe()
    challengers = {
        "colbert": {
            "status": (
                "ready_for_locked_eval"
                if probe["colbert_dependency"]
                else "unavailable"
            ),
            "reason": (
                ""
                if probe["colbert_dependency"]
                else "colbert_dependency_missing"
            ),
            "index": "separate_sqlite_token_vector_index",
            "incremental": True,
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
    for challenger in challengers.values():
        challenger["gate"] = adoption_gate(baseline, {})
        challenger["adopted"] = False
    payload = {
        "schema_version": 1,
        "generated_at_epoch": round(time.time(), 3),
        "baseline": baseline,
        "environment": probe,
        "challengers": challengers,
        "excluded_production": EXCLUDED_PRODUCTION,
        "winner": None,
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
