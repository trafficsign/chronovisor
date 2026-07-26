"""Evidence-paired local judgment and Recall/resource safety receipts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor import ollama
from chronovisor.classification import ClassificationError
from chronovisor.classification_engine import run_consensus_batches
from chronovisor.classification_fixture_set import _write_jsonl, sha256_bytes
from chronovisor.durable_state import write_sealed_json
from chronovisor.research_scheduler import foreground_lane

PAIRED_JUDGMENT_SCHEMA = "chronovisor.classification-paired-judgment.v1"
RESOURCE_GATE_SCHEMA = "chronovisor.classification-resource-ready-gate.v1"
ARMS = ("J1", "J2", "J3")


def latin_square_order(uid: str) -> tuple[str, str, str]:
    offset = int(hashlib.sha256(uid.encode("utf-8")).hexdigest(), 16) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def _sibling_counterexamples(
    candidates: Sequence[Mapping[str, Any]],
    selected: set[str],
) -> list[dict[str, str]]:
    examples = []
    for candidate in candidates:
        notation = str(candidate.get("notation") or "")
        if notation in selected:
            continue
        broader = str(candidate.get("broader_notation") or "")
        if any(
            notation.startswith(value) or value.startswith(notation)
            for value in selected
        ):
            examples.append(
                {
                    "notation": notation,
                    "label_en": str(candidate.get("label_en") or ""),
                    "broader_notation": broader,
                    "reason": "nearby hierarchy alternative requiring exclusion",
                }
            )
        if len(examples) >= 3:
            break
    return examples


def evidence_card(
    provider_result: Mapping[str, Any],
    *,
    arm: str,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ClassificationError(f"unknown judgment arm: {arm}")
    candidates = provider_result.get("union") or []
    fixed_notations = {
        str(row.get("notation") or "") for row in candidates if isinstance(row, Mapping)
    }
    source_support = []
    candidate_evidence = []
    if arm in {"J2", "J3"}:
        external_rows = [
            row
            for row in provider_result.get("external_only") or []
            if isinstance(row, Mapping)
        ]
        for index, row in enumerate(external_rows):
            if not isinstance(row, Mapping):
                continue
            supports = [
                support
                for support in row.get("source_support") or []
                if isinstance(support, Mapping)
            ]
            for support in row.get("source_support") or []:
                if isinstance(support, Mapping):
                    source_support.append(dict(support))
                if len(source_support) >= 10:
                    break
            methods: dict[str, int] = {}
            intellectual: dict[str, int] = {}
            for support in supports:
                method = str(support.get("generation_method") or "unknown")
                methods[method] = methods.get(method, 0) + 1
                status = str(support.get("intellectual_assignment") or "unconfirmed")
                intellectual[status] = intellectual.get(status, 0) + 1
            next_score = (
                float(external_rows[index + 1].get("external_score") or 0.0)
                if index + 1 < len(external_rows)
                else 0.0
            )
            candidate_evidence.append(
                {
                    "notation": str(row.get("notation") or ""),
                    "support_count": int(row.get("support_count") or 0),
                    "source_diversity": len(
                        {
                            str(support.get("source") or "")
                            for support in supports
                            if support.get("source")
                        }
                    ),
                    "generation_method_counts": methods,
                    "intellectual_assignment_counts": intellectual,
                    "neighbor_margin": round(
                        float(row.get("external_score") or 0.0) - next_score,
                        6,
                    ),
                    "representative_examples": [dict(value) for value in supports[:3]],
                }
            )
    query_expansion = []
    if arm == "J3":
        query_expansion = [
            dict(row)
            for row in provider_result.get("query_expansion") or []
            if isinstance(row, Mapping) and row.get("vocabulary_role") in {"B2", "C1"}
        ][:10]
    externally_supported = {
        str(row.get("notation") or "")
        for row in provider_result.get("external_only") or []
        if isinstance(row, Mapping)
    }
    return {
        "arm": arm,
        "source_support": source_support,
        "candidate_evidence": candidate_evidence,
        "query_expansion": query_expansion,
        "sibling_counterexamples": (
            _sibling_counterexamples(candidates, externally_supported)
            if arm in {"J2", "J3"}
            else []
        ),
        "specificity_rule": (
            "choose the most specific candidate supported by page text; "
            "library evidence may support but never create a candidate"
        ),
        "binary_veto_axes": {
            "unsupported_notation": False,
            "source_provenance_invalid": False,
            "specificity_contradicted": False,
        },
        "candidate_notations_sha256": sha256_bytes(
            "\n".join(sorted(fixed_notations)).encode("utf-8")
        ),
    }


def paired_rows(
    pages: Sequence[Mapping[str, Any]],
    provider_results: Sequence[Mapping[str, Any]],
    *,
    candidate_limit: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    result_by_uid = {str(row.get("uid") or ""): row for row in provider_results}
    output = {arm: [] for arm in ARMS}
    for page_source in pages:
        page = dict(page_source)
        uid = str(page.get("uid") or "")
        provider = result_by_uid.get(uid)
        if provider is None:
            raise ClassificationError(f"provider result missing for {uid}")
        candidates = [
            dict(row) for row in (provider.get("union") or [])[:candidate_limit]
        ]
        if not candidates:
            raise ClassificationError(f"fixed union candidate set is empty for {uid}")
        candidate_digest = sha256_bytes(
            json.dumps(
                candidates,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for arm in ARMS:
            output[arm].append(
                {
                    **page,
                    "candidates": candidates,
                    "candidate_digest": candidate_digest,
                    "evidence_card": evidence_card(provider, arm=arm),
                    "latin_square_order": list(latin_square_order(uid)),
                }
            )
    for index in range(len(pages)):
        digests = {output[arm][index]["candidate_digest"] for arm in ARMS}
        if len(digests) != 1:
            raise ClassificationError("paired judgment candidate sets differ by arm")
    return output


def run_paired_judgment(
    *,
    root: Path,
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    batch_size: int = 20,
    timeout_seconds: float = 1_800,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_results: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        rows = list(rows_by_arm.get(arm) or [])
        decisions = run_consensus_batches(
            rows,
            root=root,
            batch_size=batch_size,
            purpose="explicit",
            timeout_seconds=timeout_seconds,
            run_namespace=f"library-evidence-{arm.casefold()}",
        )
        arm_results[arm] = decisions
        path = output_dir / f"{arm.casefold()}-decisions.jsonl"
        _write_jsonl(path, decisions)
    manifest = {
        "schema": PAIRED_JUDGMENT_SCHEMA,
        "arms": list(ARMS),
        "arm_counts": {arm: len(rows) for arm, rows in arm_results.items()},
        "same_candidate_set": True,
        "same_model_policy": True,
        "latin_square": True,
        "model_calls_are_local_only": True,
        "outputs": {
            arm: str(output_dir / f"{arm.casefold()}-decisions.jsonl") for arm in ARMS
        },
    }
    write_sealed_json(output_dir / "manifest.json", manifest, backup=True)
    return manifest


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def resource_ready_gate(
    *,
    recall_latencies_ms: Sequence[float],
    recall_misses: int,
    cancel_to_ready_ms: Sequence[float],
    protected_models: Sequence[str],
    resident_models: Sequence[str],
    minimum_overlap_samples: int = 30,
) -> dict[str, Any]:
    protected = set(protected_models)
    resident = set(resident_models)
    gates = {
        "sample_size": len(recall_latencies_ms) >= minimum_overlap_samples,
        "recall_p99": _percentile(recall_latencies_ms, 0.99) <= 4_000,
        "recall_max": max(recall_latencies_ms, default=math.inf) <= 4_000,
        "recall_miss": recall_misses == 0,
        "protected_residency": protected <= resident,
        "cancel_resource_ready": (
            bool(cancel_to_ready_ms) and max(cancel_to_ready_ms) <= 4_000
        ),
    }
    return {
        "schema": RESOURCE_GATE_SCHEMA,
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "sample_count": len(recall_latencies_ms),
        "recall_p95_ms": _percentile(recall_latencies_ms, 0.95),
        "recall_p99_ms": _percentile(recall_latencies_ms, 0.99),
        "recall_max_ms": max(recall_latencies_ms, default=None),
        "recall_misses": recall_misses,
        "cancel_to_ready_max_ms": max(cancel_to_ready_ms, default=None),
        "protected_models": sorted(protected),
        "resident_models": sorted(resident),
    }


def measure_recall_overlap(
    recall_call: Callable[[], bool],
    *,
    samples: int,
) -> tuple[list[float], int]:
    """Measure the real foreground lane; failed calls remain hard misses."""

    latencies = []
    misses = 0
    for _ in range(samples):
        started = time.monotonic()
        with foreground_lane():
            success = bool(recall_call())
        latency = (time.monotonic() - started) * 1_000
        latencies.append(latency)
        if not success or latency > 4_000:
            misses += 1
    return latencies, misses


def current_resident_models() -> list[str]:
    return sorted(
        {
            str(row.get("name") or row.get("model") or "")
            for row in ollama.resident_model_rows()
            if str(row.get("name") or row.get("model") or "")
        }
    )
