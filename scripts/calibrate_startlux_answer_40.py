#!/usr/bin/env python3.14
"""Score the frozen 40 answer-controls through the resident StartLux judge.

This is a prepared calibration harness. It is read-only apart from
``--output-dir`` and never falls back to another model. The answer-axis judge
uses the service's background priority (PRIORITY_SEARCH). Run only after the
live config routes ``recall.answer.scorer`` to the local decision provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chronovisor.core.decision_mlx import decision_route
from chronovisor.core.runtime_config import load_reranker_config
from chronovisor.recall import recall_answer_eval
from chronovisor.recall.recall_answer_adapters import (
    builtin_answer_adapter_identities,
    builtin_startlux_answer_scorer,
)


def _controls(benchmark: Path) -> list[dict[str, Any]]:
    check = recall_answer_eval.validate_independent_answer_benchmark(benchmark)
    if check.get("passed") is not True:
        raise SystemExit(f"benchmark_unavailable:{check.get('reason') or 'invalid'}")
    controls = recall_answer_eval._machine_calibration_controls(check)
    if len(controls) != recall_answer_eval.MACHINE_SCORER_CALIBRATION_MIN_CASES:
        raise SystemExit(f"control_count_invalid:{len(controls)}")
    return controls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=recall_answer_eval.INDEPENDENT_ANSWER_BENCHMARK,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    decision_route("recall.answer.scorer")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    controls = _controls(args.benchmark.expanduser().resolve())
    _runner, scorer_identity = builtin_answer_adapter_identities(
        rubric_sha256="0" * 64,
        evidence_manifest_sha256=str(
            recall_answer_eval._canonical_sha({"benchmark": str(args.benchmark)})
        ),
    )
    config = load_reranker_config()
    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for control in controls:
        packet = control["source_packet"]
        gold = {
            "evidence": {"source_packet": packet},
            "evidence_sha256": control["evidence_sha256"],
            "rubric_sha256": "0" * 64,
        }
        scoring = {
            "seed": 0,
            "base_state_sha256": "",
            "evidence_manifest_sha256": scorer_identity["evidence_manifest_sha256"],
        }
        result = builtin_startlux_answer_scorer(
            control["source_packet"]["prompt"],
            control["answer"],
            gold,
            scoring,
        )
        scores = result.get("dimensions")
        if not isinstance(scores, dict):
            raise SystemExit("scorer_returned_invalid_dimensions")
        cases.append({"control": control, "scorer_scores": scores})
        rows.append(
            {
                "control_id": control["control_id"],
                "variant": control["variant"],
                "scores": scores,
                "expected": control["expected_scores"],
            }
        )
    metrics, gates = recall_answer_eval._machine_calibration_case_metrics(cases)
    payload = {
        "status": "complete" if all(gates.values()) else "held",
        "cases": len(cases),
        "metrics": metrics,
        "gates": gates,
        "scorer_identity": scorer_identity,
        "service_timeout_ms": config.service.timeout_ms,
        "rows": rows,
    }
    (output_dir / "answer-calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "metrics": metrics, "gates": gates}))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
