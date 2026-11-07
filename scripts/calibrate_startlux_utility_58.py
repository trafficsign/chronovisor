#!/usr/bin/env python3.14
"""Reconstruct and cross-validate the 58 historical utility pairs on StartLux.

The label ledger stores answer hashes, not answer text. This script rebuilds
each pair from the sealed rally/exposure artifacts, regenerates the two answers
with the recorded Qwen generator, verifies both SHA-256 values, and only then
calls the StartLux utility judge in both blind orders. It is read-only apart
from ``--output-dir`` and uses PRIORITY_SEARCH for every service call.

Prepared for a later run; it is intentionally not invoked by the test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from chronovisor.core import reranker_client
from chronovisor.core.decision_mlx import logit_probability
from chronovisor.core.runtime_config import load_reranker_config
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_catalog import CatalogTextCache

PRIORITY_SEARCH = 1
JUDGE_ROLE = "recall.distill.utility_judge"
GENERATOR_ROLE = "recall.distill.answer_generator"
MAX_INPUT_BYTES = 12_000
DEPENDENT = "helpful"
HARMFUL = "harmful"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"dimension":"answer_utility"' not in line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _rallies(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            manifest = value.get("manifest") if isinstance(value, dict) else None
            if not isinstance(manifest, dict):
                continue
            rally_id = str(manifest.get("rally_id") or "")
            if rally_id in wanted:
                found[rally_id] = manifest
                if len(found) == len(wanted):
                    break
    return found


def _rendered(ref: dict[str, Any], texts: CatalogTextCache) -> str:
    rendered = ref.get("rendered_context")
    if isinstance(rendered, str) and rendered:
        return rendered
    refs = ref.get("evidence_refs")
    if isinstance(refs, list):
        parts = [
            texts.get(str(item.get("semantic_sha256") or ""), "")
            for item in refs
            if isinstance(item, dict)
        ]
        return "\n".join(part for part in parts if part)
    return ""


def _payload(
    row: dict[str, Any],
    rally: dict[str, Any],
    artifact: dict[str, Any],
    texts: CatalogTextCache,
) -> dict[str, Any] | None:
    refs = artifact.get("candidate_refs")
    if not isinstance(refs, list):
        return None
    original: dict[str, str] = {}
    target_text = ""
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        candidate_id = str(ref.get("candidate_id") or "")
        rendered = _rendered(ref, texts)
        if not candidate_id or not rendered:
            continue
        original[candidate_id] = rendered
    target = str(row.get("candidate_id") or "")
    if target in original:
        mode = "remove"
        target_text = original[target]
    else:
        mode = "add"
        for pool in artifact.get("candidate_pool_refs") or []:
            if not isinstance(pool, dict) or pool.get("selected") is not False:
                continue
            if str(pool.get("candidate_id") or "") != target:
                continue
            target_text = _rendered(pool, texts)
            if not target_text:
                text_hash = str(
                    pool.get("text_sha256") or pool.get("content_sha256") or ""
                )
                target_text = texts.get(text_hash, "")
            break
    if not target_text:
        return None
    query = texts.get(str(rally.get("query_sha256") or ""), "")
    if not query:
        return None
    context = [
        texts.get(str(ref.get("semantic_sha256") or ""), "")
        for ref in rally.get("context_refs") or []
        if isinstance(ref, dict)
    ]
    context = [value for value in context if value]
    values = list(original.values())
    return {
        "rally_id": str(rally.get("rally_id") or ""),
        "candidate_id": target,
        "query": query,
        "context": context,
        "a0_evidence": [v for k, v in original.items() if k != target]
        if mode == "remove"
        else values,
        "a1_evidence": values if mode == "remove" else [*values, target_text],
        "mode": mode,
    }


def _answer(
    evidence: list[str],
    base: dict[str, Any],
    generator_identity: dict[str, str],
    generator_digest: str,
) -> str:
    result = distill._worker_call(
        "answer",
        GENERATOR_ROLE,
        {**base, "evidence": evidence},
        max_input_bytes=MAX_INPUT_BYTES,
        expected_route=generator_identity,
        expected_digest=generator_digest,
        deadline_ms=60_000,
    )
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer:
        raise distill.DistillationError("answer generator returned no answer")
    return answer


def _judge(
    *,
    query: str,
    context: list[str],
    answer_a: str,
    answer_b: str,
    config: Any,
) -> float:
    response = reranker_client.decision_judge(
        "utility",
        {
            "query": query,
            "context": "\n".join(context),
            "answer_a": answer_a,
            "answer_b": answer_b,
        },
        role=JUDGE_ROLE,
        config=config,
        timeout_ms=10_000,
        priority=PRIORITY_SEARCH,
    )
    logit = response.get("logit")
    if isinstance(logit, bool) or not isinstance(logit, int | float):
        raise RuntimeError("utility judge returned no raw logit")
    return float(logit)


def _choice(probability: float, decisive: float, tie_band: float) -> str:
    if probability >= decisive:
        return "a"
    if probability <= 1.0 - decisive:
        return "b"
    if abs(probability - 0.5) <= tie_band:
        return "tie"
    return "uncertain"


def _decode(choice: str, a0_first: bool) -> str:
    if choice == "tie":
        return "neutral"
    if choice not in {"a", "b"}:
        return "uncertain"
    candidate_arm = "b" if a0_first else "a"
    return "helpful" if choice == candidate_arm else "harmful"


def _verdict(
    first_logit: float,
    second_logit: float,
    a0_first: bool,
    decisive: float,
    tie_band: float,
) -> tuple[str, bool]:
    first = logit_probability(first_logit)
    second = logit_probability(second_logit)
    first_choice = _choice(first, decisive, tie_band)
    second_choice = _choice(second, decisive, tie_band)
    first_verdict = _decode(first_choice, a0_first)
    second_verdict = _decode(second_choice, not a0_first)
    if first_verdict == second_verdict and first_verdict != "uncertain":
        return first_verdict, True
    return "uncertain", False


def _fit(rows: list[dict[str, Any]], folds: int) -> dict[str, Any]:
    def fit_one(group: list[dict[str, Any]]) -> tuple[float, float]:
        best = (-1.0, 0.75, 0.10)
        for decisive in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            for tie_band in (0.05, 0.10, 0.15):
                correct = sum(
                    _verdict(
                        row["first_logit"],
                        row["second_logit"],
                        row["a0_first"],
                        decisive,
                        tie_band,
                    )[0]
                    == row["gold"]
                    for row in group
                )
                score = correct / len(group) if group else 0.0
                if score > best[0]:
                    best = (score, decisive, tie_band)
        return best[1], best[2]

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["fold"]].append(row)
    predictions: list[dict[str, Any]] = []
    folds_out: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [
            row for name, values in groups.items() if name != fold for row in values
        ]
        test = groups.get(fold, [])
        if not train or not test:
            continue
        decisive, tie_band = fit_one(train)
        correct = 0
        directional = 0
        directional_total = 0
        agreements = 0
        uncertain = 0
        for row in test:
            predicted, agreement = _verdict(
                row["first_logit"],
                row["second_logit"],
                row["a0_first"],
                decisive,
                tie_band,
            )
            correct += predicted == row["gold"]
            if row["gold"] in {DEPENDENT, HARMFUL}:
                directional_total += 1
                directional += predicted == row["gold"]
            agreements += agreement
            uncertain += predicted == "uncertain"
            predictions.append({**row, "predicted": predicted})
        folds_out.append(
            {
                "fold": fold,
                "decisive": decisive,
                "tie_band": tie_band,
                "cases": len(test),
                "accuracy": round(correct / len(test), 6),
                "directional_accuracy": round(directional / directional_total, 6)
                if directional_total
                else 0.0,
                "order_agreement": round(agreements / len(test), 6),
                "uncertain_rate": round(uncertain / len(test), 6),
            }
        )
    accuracy = (
        sum(predicted["predicted"] == predicted["gold"] for predicted in predictions)
        / len(predictions)
        if predictions
        else 0.0
    )
    directional_total = sum(
        predicted["gold"] in {DEPENDENT, HARMFUL} for predicted in predictions
    )
    directional = sum(
        predicted["predicted"] == predicted["gold"]
        for predicted in predictions
        if predicted["gold"] in {DEPENDENT, HARMFUL}
    )
    return {
        "folds": folds_out,
        "cases": len(predictions),
        "accuracy": round(accuracy, 6),
        "directional_accuracy": round(directional / directional_total, 6)
        if directional_total
        else 0.0,
        "uncertain_rate": round(
            sum(predicted["predicted"] == "uncertain" for predicted in predictions)
            / len(predictions),
            6,
        )
        if predictions
        else 0.0,
        "predictions": predictions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=CHRONOVISOR_ROOT
        / "runtime"
        / "recall-distillation"
        / "label-ledger.jsonl",
    )
    parser.add_argument(
        "--rallies",
        type=Path,
        default=CHRONOVISOR_ROOT
        / "runtime"
        / "recall-distillation"
        / "rally-manifest.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = args.root.expanduser().resolve()
    texts = CatalogTextCache(root / "raw", root)
    rows = _jsonl(args.labels.expanduser())
    wanted = {str(row.get("rally_id") or "") for row in rows}
    rallies = _rallies(args.rallies.expanduser(), wanted)
    generator_identity, generator_digest, _judge_identity, _judge_digest = (
        distill._local_counterfactual_routes()
    )
    config = load_reranker_config()
    enriched: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda value: str(value.get("rally_id"))):
        rally_id = str(row.get("rally_id") or "")
        rally = rallies.get(rally_id)
        receipt = (rally or {}).get("exposure_receipts")
        exposure_id = ""
        if isinstance(receipt, list) and receipt and isinstance(receipt[0], dict):
            exposure_id = str(receipt[0].get("exposure_artifact_id") or "")
        if not rally or not exposure_id:
            skipped.append({"rally_id": rally_id, "reason": "rally_exposure_missing"})
            continue
        try:
            artifact = store.read_sealed(
                store.distillation_dir(root) / "exposures" / f"{exposure_id}.json",
                schema="chronovisor.recall-exact-exposure.v1",
            )
        except Exception:
            skipped.append({"rally_id": rally_id, "reason": "exposure_unreadable"})
            continue
        payload = _payload(row, rally, artifact, texts)
        if payload is None:
            skipped.append(
                {"rally_id": rally_id, "reason": "payload_unreconstructable"}
            )
            continue
        base = {
            key: payload[key]
            for key in ("rally_id", "candidate_id", "query", "context")
        }
        try:
            a0 = _answer(
                payload["a0_evidence"], base, generator_identity, generator_digest
            )
            a1 = _answer(
                payload["a1_evidence"], base, generator_identity, generator_digest
            )
        except Exception as exc:
            skipped.append(
                {
                    "rally_id": rally_id,
                    "reason": f"answer_unavailable:{type(exc).__name__}",
                }
            )
            continue
        if hashlib.sha256(a0.encode()).hexdigest() != row.get("a0_sha256"):
            skipped.append({"rally_id": rally_id, "reason": "a0_hash_mismatch"})
            continue
        if hashlib.sha256(a1.encode()).hexdigest() != row.get("a1_sha256"):
            skipped.append({"rally_id": rally_id, "reason": "a1_hash_mismatch"})
            continue
        a0_first = (
            int.from_bytes(
                hashlib.sha256(
                    f"blind-order-v1\0{rally_id}\0{row.get('candidate_id') or ''}".encode()
                ).digest()[:8],
                "big",
            )
            % 2
            == 0
        )
        try:
            first_logit = _judge(
                query=payload["query"],
                context=payload["context"],
                answer_a=a0 if a0_first else a1,
                answer_b=a1 if a0_first else a0,
                config=config,
            )
            second_logit = _judge(
                query=payload["query"],
                context=payload["context"],
                answer_a=a1 if a0_first else a0,
                answer_b=a0 if a0_first else a1,
                config=config,
            )
        except Exception as exc:
            skipped.append(
                {
                    "rally_id": rally_id,
                    "reason": f"judge_unavailable:{type(exc).__name__}",
                }
            )
            continue
        gold = str(row.get("verdict") or "uncertain")
        enriched.append(
            {
                "rally_id": rally_id,
                "a0_first": a0_first,
                "first_logit": first_logit,
                "second_logit": second_logit,
                "gold": gold,
                "fold": int(hashlib.sha256(rally_id.encode()).hexdigest()[:8], 16)
                % max(1, args.folds),
            }
        )
    metrics = _fit(enriched, max(1, args.folds))
    payload = {
        "status": "complete" if metrics["cases"] == len(rows) else "held",
        "labels": len(rows),
        "replayed": metrics["cases"],
        "skipped": skipped,
        "metrics": metrics,
        "priority": PRIORITY_SEARCH,
    }
    (output_dir / "utility-cv.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
