"""Evaluate the deterministic CVO intent lane on opened development data."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.classification import ClassificationError
from chronovisor.classification_anchor import load_anchor_set
from chronovisor.classification_anchor_set_dev import (
    score_anchor_set,
    summarize_metrics,
)
from chronovisor.classification_intent_lexicon import (
    classify_complement,
    load_intent_lexicon,
)
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.store import CHRONOVISOR_ROOT

EVALUATION_SCHEMA = "chronovisor.cvo-intent-lexicon-dev.v2"
EXPERIMENT = "cvo-intent-v2-dev80"
SOURCE_EXPERIMENTS = (
    "cvo-anchor-set-v1-unseen40",
    "cvo-intent-v1-unseen40",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def run_dev(root: Path) -> dict[str, Any]:
    destination = root / "classification" / EXPERIMENT
    evaluation_path = destination / "evaluation.json"
    if evaluation_path.is_file():
        return read_sealed_json(evaluation_path)
    sources = []
    fixture_by_uid = {}
    for source_experiment in SOURCE_EXPERIMENTS:
        source_root = root / "classification" / source_experiment
        source = read_sealed_json(source_root / "evaluation.json")
        fixture = read_sealed_json(source_root / "fixture.json")
        fixture_by_uid.update(
            {
                str(page.get("uid") or ""): dict(page)
                for page in fixture.get("cases") or []
                if isinstance(page, Mapping)
            }
        )
        sources.append((source_experiment, source_root, source))
    anchor_set = load_anchor_set()
    lexicon = load_intent_lexicon(anchor_set=anchor_set)
    cases = []
    for source_experiment, _source_root, source in sources:
        for source_case in source.get("cases") or []:
            if not isinstance(source_case, Mapping):
                raise ClassificationError(
                    "CVO intent dev source case is invalid"
                )
            uid = str(source_case.get("uid") or "")
            page = fixture_by_uid[uid]
            core_anchor_id = str(source_case.get("core_anchor_id") or "")
            intent = classify_complement(
                page,
                core_anchor_id=core_anchor_id,
                lexicon=lexicon,
            )
            selected = [core_anchor_id]
            if intent["second_anchor_id"] != "NONE":
                selected.append(str(intent["second_anchor_id"]))
            selected = sorted(dict.fromkeys(selected))
            acceptable_sets = [
                [str(value) for value in acceptable]
                for acceptable in source_case.get("acceptable_anchor_sets") or []
            ]
            acceptable_union = sorted(
                {
                    value
                    for acceptable in acceptable_sets
                    for value in acceptable
                }
            )
            defensible = [
                str(value)
                for value in source_case.get("defensible_anchor_ids") or []
            ]
            score = score_anchor_set(
                selected,
                acceptable_union,
                defensible,
                acceptable_sets,
            )
            cases.append(
                {
                    "uid": uid,
                    "title": str(page.get("title") or ""),
                    "source_experiment": source_experiment,
                    "core_anchor_id": core_anchor_id,
                    "intent": intent,
                    "selected_anchor_ids": selected,
                    "acceptable_anchor_sets": acceptable_sets,
                    "defensible_anchor_ids": defensible,
                    **score,
                }
            )
    metrics = summarize_metrics(cases)
    passed = (
        metrics["case_count"] == 80
        and metrics["exact_sets"] >= 76
        and metrics["semantic_coverage_cases"] >= 78
        and metrics["excess_anchor_rate"] <= 0.10
        and metrics["missing_anchor_rate"] <= 0.10
        and metrics["dual_assignment_rate"] <= 0.40
        and metrics["holds"] <= 2
        and metrics["major_errors"] == 0
    )
    evaluation = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "fixture_status": "opened-development-only",
        "source_evaluation_paths": [
            str(source_root / "evaluation.json")
            for _, source_root, _ in sources
        ],
        "lexicon_epoch": lexicon.epoch,
        "lexicon_checksum": lexicon.checksum,
        "model_calls": 0,
        "page_mutations": 0,
        "metrics": metrics,
        "cases": cases,
        "decision": (
            "continue-cvo-intent-v2"
            if passed
            else "kill-cvo-intent-v2"
        ),
    }
    write_sealed_json(evaluation_path, evaluation, backup=True)
    return read_sealed_json(evaluation_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the CVO intent lexicon on opened data"
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                run_dev(args.root.expanduser()),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (
        ClassificationError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
