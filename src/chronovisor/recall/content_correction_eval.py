"""Deterministic quality evaluation for correction-capture admission."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from chronovisor.recall.content_correction import correction_signal


DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "content-correction-golden.jsonl"
)


def load_cases(path: Path = DEFAULT_CORPUS) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"case must be an object at {path}:{line_number}")
            if not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError(f"case id is required at {path}:{line_number}")
            if not isinstance(row.get("prompt"), str):
                raise ValueError(f"prompt is required at {path}:{line_number}")
            if not isinstance(row.get("expected"), bool):
                raise ValueError(f"expected must be boolean at {path}:{line_number}")
            cases.append(row)
    return cases


def _metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    tp = sum(row["expected"] and row["actual"] for row in selected)
    fp = sum(not row["expected"] and row["actual"] for row in selected)
    tn = sum(not row["expected"] and not row["actual"] for row in selected)
    fn = sum(row["expected"] and not row["actual"] for row in selected)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "cases": len(selected),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "false_negative_rate": round(1.0 - recall, 6),
        "false_positive_rate": round(fp / (fp + tn), 6) if fp + tn else 0.0,
    }


def evaluate_cases(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    for case in cases:
        actual = (
            correction_signal(
                str(case["prompt"]),
                recall_provenance=bool(case.get("recall_provenance", False)),
            )
            is not None
        )
        evaluated.append({**case, "actual": actual, "passed": actual is case["expected"]})

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        by_category[str(row.get("category") or "uncategorized")].append(row)
        by_split[str(row.get("split") or "golden")].append(row)
    failures = [
        {
            "id": row["id"],
            "category": row.get("category", ""),
            "expected": row["expected"],
            "actual": row["actual"],
        }
        for row in evaluated
        if not row["passed"]
    ]
    return {
        "status": "passed" if not failures else "failed",
        "metrics": _metrics(evaluated),
        "by_category": {
            key: _metrics(rows) for key, rows in sorted(by_category.items())
        },
        "by_split": {key: _metrics(rows) for key, rows in sorted(by_split.items())},
        "failures": failures,
    }


def run_eval(path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    return {"corpus": str(path), **evaluate_cases(load_cases(path))}


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-content-correction-eval`` command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic correction-capture precision and recall."
    )
    parser.add_argument("--input", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = run_eval(Path(args.input).expanduser())
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
