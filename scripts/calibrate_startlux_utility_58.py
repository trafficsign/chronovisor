#!/usr/bin/env python3.14
"""Reconstruct and cross-validate the 58 historical utility pairs on StartLux.

The label ledger stores answer hashes, not answer text. Supply archived inputs
with --answers (JSONL: rally_id, candidate_id, query, context, a0, a1). Query and
context hashes must match the rally manifest, and both answer hashes must match
the label ledger before either blind order is judged. No generation, catalog
connection, lock, repair, or production output is performed. Only --output-dir
is writable; every service call uses PRIORITY_SEARCH.

Prepared for a later run; it is intentionally not invoked by the test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from chronovisor.core import reranker_client
from chronovisor.core.decision_mlx import logit_probability
from chronovisor.core.durable_state import atomic_write_bytes
from chronovisor.core.runtime_config import load_reranker_config
from chronovisor.core.store import CHRONOVISOR_ROOT

PRIORITY_SEARCH = 1
JUDGE_ROLE = "recall.distill.utility_judge"
DEPENDENT = "helpful"
HARMFUL = "harmful"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("dimension") == "answer_utility":
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


def _verified_input(
    row: dict[str, Any], rally: dict[str, Any], archived: dict[str, Any]
) -> bool:
    query, context = archived.get("query"), archived.get("context")
    a0, a1 = archived.get("a0"), archived.get("a1")
    refs = rally.get("context_refs")
    if (
        not isinstance(query, str)
        or not query
        or not isinstance(context, list)
        or not all(isinstance(text, str) for text in context)
        or not isinstance(a0, str)
        or not a0
        or not isinstance(a1, str)
        or not a1
        or not isinstance(refs, list)
        or not all(isinstance(ref, dict) for ref in refs)
    ):
        return False
    return (
        hashlib.sha256(query.encode()).hexdigest() == rally.get("query_sha256")
        and [hashlib.sha256(text.encode()).hexdigest() for text in context]
        == [ref.get("semantic_sha256") for ref in refs]
        and hashlib.sha256(a0.encode()).hexdigest() == row.get("a0_sha256")
        and hashlib.sha256(a1.encode()).hexdigest() == row.get("a1_sha256")
    )


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


def _choice(probability: float, decisive: float) -> str:
    if probability >= decisive:
        return "a"
    if probability <= 1.0 - decisive:
        return "b"
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
) -> tuple[str, bool]:
    first = logit_probability(first_logit)
    second = logit_probability(second_logit)
    first_choice = _choice(first, decisive)
    second_choice = _choice(second, decisive)
    if first_choice == second_choice == "b":
        return "neutral", True
    first_verdict = _decode(first_choice, a0_first)
    second_verdict = _decode(second_choice, not a0_first)
    if first_verdict == second_verdict and first_verdict != "uncertain":
        return first_verdict, True
    return "uncertain", False


def _fit(rows: list[dict[str, Any]], folds: int) -> dict[str, Any]:
    def fit_one(group: list[dict[str, Any]]) -> float:
        best = (-1.0, 0.75)
        for decisive in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            correct = sum(
                _verdict(
                    row["first_logit"], row["second_logit"], row["a0_first"], decisive
                )[0]
                == row["gold"]
                for row in group
            )
            score = correct / len(group) if group else 0.0
            if score > best[0]:
                best = (score, decisive)
        return best[1]

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
        decisive = fit_one(train)
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
            )
            correct += predicted == row["gold"]
            if row["gold"] in {DEPENDENT, HARMFUL}:
                directional_total += 1
                directional += predicted == row["gold"]
            agreements += agreement
            uncertain += predicted == "uncertain"
            predictions.append(
                {**row, "predicted": predicted, "order_agreement": agreement}
            )
        folds_out.append(
            {
                "fold": fold,
                "decisive": decisive,
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
    n = directional_total
    p = directional / n if n else 0.0
    z = 1.959963984540054
    lower_bound = (
        (
            (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))
            / (1 + z * z / n)
        )
        if n
        else 0.0
    )
    counts = {
        label: sum(row["predicted"] == label for row in predictions)
        for label in ("helpful", "harmful", "neutral", "uncertain")
    }
    agreement_rate = (
        sum(row["order_agreement"] for row in predictions) / len(predictions)
        if predictions
        else 0.0
    )
    uncertain_rate = counts["uncertain"] / len(predictions) if predictions else 1.0
    gates = {
        "accuracy": accuracy >= 0.80,
        "directional_accuracy": p >= 0.85,
        "directional_lcb": lower_bound >= 0.65,
        "order_agreement": agreement_rate >= 0.90,
        "uncertain_rate": uncertain_rate <= 0.10,
        "helpful_count": counts["helpful"] >= 20,
        "harmful_count": counts["harmful"] >= 20,
    }
    return {
        "gates": gates,
        "passed": all(gates.values()),
        "directional_lcb": lower_bound,
        "order_agreement": agreement_rate,
        "distribution": counts,
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
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.is_relative_to(CHRONOVISOR_ROOT.expanduser().resolve()):
        raise SystemExit("output-dir must be outside the production root")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _jsonl(args.labels.expanduser())
    if (
        len(rows) != 58
        or any(
            not isinstance(row.get("rally_id"), str)
            or not row["rally_id"]
            or not isinstance(row.get("candidate_id"), str)
            or not row["candidate_id"]
            or row.get("verdict") not in {"helpful", "harmful", "neutral", "uncertain"}
            for row in rows
        )
        or len({(row["rally_id"], row["candidate_id"]) for row in rows}) != 58
    ):
        raise SystemExit("expected 58 distinct labeled utility pairs")
    wanted = {str(row.get("rally_id") or "") for row in rows}
    rallies = _rallies(args.rallies.expanduser(), wanted)
    answers = {}
    for line in args.answers.expanduser().read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["rally_id"], row["candidate_id"])
        if key in answers:
            raise SystemExit("duplicate archived answer pair")
        answers[key] = row
    config = load_reranker_config()
    enriched: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda value: str(value.get("rally_id"))):
        rally_id = str(row.get("rally_id") or "")
        rally = rallies.get(rally_id)
        archived = answers.get((rally_id, row.get("candidate_id")), {})
        if not rally or not _verified_input(row, rally, archived):
            skipped.append(
                {"rally_id": rally_id, "reason": "input_missing_or_hash_mismatch"}
            )
            continue
        a0, a1 = archived["a0"], archived["a1"]
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
                query=archived["query"],
                context=archived["context"],
                answer_a=a0 if a0_first else a1,
                answer_b=a1 if a0_first else a0,
                config=config,
            )
            second_logit = _judge(
                query=archived["query"],
                context=archived["context"],
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
        "status": "complete"
        if len(rows) == 58 and metrics["cases"] == 58 and metrics["passed"]
        else "held",
        "labels": len(rows),
        "replayed": metrics["cases"],
        "skipped": skipped,
        "metrics": metrics,
        "priority": PRIORITY_SEARCH,
    }
    atomic_write_bytes(
        output_dir / "utility-cv.json",
        (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        backup=False,
    )
    print(json.dumps(payload))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
