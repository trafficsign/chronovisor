"""Validated recall gate calibration.

The calibration artifact is intentionally separate from the user's TOML.  The
TOML enables the feature; this module writes the learned runtime artifact only
after a holdout check passes and records the old artifact for rollback.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp.recall_eval import read_jsonl
from llm_wiki_mcp.recall_runtime import (
    RECALL_FEEDBACK_FILE,
    RECALL_LOG_FILE,
    RecallPolicy,
    RecallRequest,
    append_jsonl,
    run_recall,
)
from llm_wiki_mcp.recall_runtime_paths import RECALL_DIR


CALIBRATION_FILE = RECALL_DIR / "calibration.json"
CALIBRATION_HISTORY_FILE = RECALL_DIR / "calibration-history.jsonl"

FEATURE_KEYS = (
    "top1_score_norm",
    "margin_norm",
    "hit_count_norm",
    "heuristic_score",
    "ambiguity",
    "prompt_len_norm",
    "rewrite_confidence",
)


@dataclass(frozen=True)
class CalibrationPolicy:
    enabled: bool = True
    min_samples: int = 500
    holdout_ratio: float = 0.2
    min_improvement: float = 0.02
    learning_rate: float = 0.15
    epochs: int = 180


def load_calibration(path: Path = CALIBRATION_FILE) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("weights"), dict) else None


def feature_vector(features: dict[str, Any]) -> list[float]:
    out: list[float] = []
    for key in FEATURE_KEYS:
        value = features.get(key, 0.0)
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def sigmoid(value: float) -> float:
    if value < -40:
        return 0.0
    if value > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def predict(weights: list[float], bias: float, xs: list[float]) -> float:
    return sigmoid(sum(w * x for w, x in zip(weights, xs)) + bias)


def train_logistic(rows: list[tuple[list[float], int]], policy: CalibrationPolicy) -> tuple[list[float], float]:
    weights = [0.0 for _ in FEATURE_KEYS]
    bias = 0.0
    if not rows:
        return weights, bias
    for _ in range(policy.epochs):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        for xs, label in rows:
            error = predict(weights, bias, xs) - label
            for i, x in enumerate(xs):
                grad_w[i] += error * x
            grad_b += error
        scale = policy.learning_rate / len(rows)
        for i in range(len(weights)):
            weights[i] -= scale * grad_w[i]
        bias -= scale * grad_b
    return weights, bias


def score_rows(rows: list[tuple[list[float], int]], weights: list[float], bias: float) -> dict[str, float]:
    if not rows:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    tp = fp = tn = fn = 0
    for xs, label in rows:
        pred = predict(weights, bias, xs) >= 0.5
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def _label_for_feedback(record: dict[str, Any]) -> int | None:
    kind = record.get("kind")
    if kind in {"missed", "missed_candidate", "injection_used"}:
        return 1
    if kind in {"false-positive", "injection_ignored"}:
        return 0
    return None


def load_labeled_rows(
    *,
    log_file: Path = RECALL_LOG_FILE,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
) -> list[dict[str, Any]]:
    logs = {
        str(record.get("decision_id", "")): record
        for record in read_jsonl(log_file)
        if record.get("decision_id")
    }
    rows: list[dict[str, Any]] = []
    for feedback in read_jsonl(feedback_file):
        label = _label_for_feedback(feedback)
        if label is None:
            continue
        ref = str(feedback.get("ref", ""))
        snapshot = feedback.get("snapshot") if isinstance(feedback.get("snapshot"), dict) else None
        source = logs.get(ref) or snapshot or {}
        features = source.get("evidence_features") or source.get("features")
        if not isinstance(features, dict):
            prompt = str(feedback.get("prompt") or source.get("prompt_preview") or "")
            if prompt:
                policy = RecallPolicy(
                    log_decisions=False,
                    semantic=False,
                    rewrite_enabled=False,
                    judge_mode="off",
                )
                result = run_recall(
                    RecallRequest(
                        host=str(feedback.get("host") or source.get("host") or "calibration"),
                        event="UserPromptSubmit",
                        prompt=prompt,
                        cwd=str(source.get("cwd") or ""),
                        session_id="",
                    ),
                    policy,
                    perform_search=True,
                )
                features = result.evidence_features
        if not isinstance(features, dict):
            continue
        rows.append(
            {
                "ts": str(feedback.get("ts") or source.get("ts") or ""),
                "label": label,
                "features": features,
                "ref": ref,
            }
        )
    rows.sort(key=lambda row: row["ts"])
    return rows


def split_holdout(
    rows: list[dict[str, Any]],
    *,
    holdout_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    ratio = max(0.05, min(0.8, holdout_ratio))
    holdout_n = max(1, int(round(len(rows) * ratio)))
    split_at = max(0, len(rows) - holdout_n)
    return rows[:split_at], rows[split_at:]


def calibrate(
    *,
    policy: CalibrationPolicy = CalibrationPolicy(),
    log_file: Path = RECALL_LOG_FILE,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    dry_run: bool = False,
) -> dict[str, Any]:
    rows = load_labeled_rows(log_file=log_file, feedback_file=feedback_file)
    if len(rows) < policy.min_samples:
        return {
            "status": "skipped",
            "reason": f"not enough labeled samples ({len(rows)} < {policy.min_samples})",
            "samples": len(rows),
        }
    train_rows_raw, holdout_raw = split_holdout(rows, holdout_ratio=policy.holdout_ratio)
    train_rows = [(feature_vector(row["features"]), int(row["label"])) for row in train_rows_raw]
    holdout_rows = [(feature_vector(row["features"]), int(row["label"])) for row in holdout_raw]
    weights, bias = train_logistic(train_rows, policy)
    baseline_weights = [1.0 if key == "top1_score_norm" else 0.0 for key in FEATURE_KEYS]
    baseline = score_rows(holdout_rows, baseline_weights, 0.0)
    candidate = score_rows(holdout_rows, weights, bias)
    improvement = candidate["accuracy"] - baseline["accuracy"]
    artifact = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "lane": "validated-auto",
        "feature_keys": list(FEATURE_KEYS),
        "weights": {key: weights[i] for i, key in enumerate(FEATURE_KEYS)},
        "bias": bias,
        "thresholds": {
            "search": 0.45,
            "read": 0.72,
        },
        "samples": len(rows),
        "train_samples": len(train_rows),
        "holdout_samples": len(holdout_rows),
        "holdout": {
            "baseline": baseline,
            "candidate": candidate,
            "improvement": improvement,
        },
    }
    if improvement < policy.min_improvement:
        return {
            "status": "skipped",
            "reason": f"holdout improvement below threshold ({improvement:.4f} < {policy.min_improvement:.4f})",
            "candidate": artifact,
        }
    old = load_calibration() or {}
    if not dry_run:
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CALIBRATION_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(CALIBRATION_FILE)
        append_jsonl(
            CALIBRATION_HISTORY_FILE,
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "action": "apply",
                "lane": "validated-auto",
                "old": old,
                "new": artifact,
            },
        )
    return {"status": "dry_run" if dry_run else "applied", "calibration": artifact}


def rollback_last() -> dict[str, Any]:
    history = read_jsonl(CALIBRATION_HISTORY_FILE)
    for record in reversed(history):
        if record.get("action") != "apply":
            continue
        old = record.get("old")
        if not isinstance(old, dict):
            continue
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        if old:
            CALIBRATION_FILE.write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif CALIBRATION_FILE.exists():
            CALIBRATION_FILE.unlink()
        append_jsonl(
            CALIBRATION_HISTORY_FILE,
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "action": "rollback",
                "restored_from": record.get("ts", ""),
                "restored": old,
            },
        )
        return {"status": "rolled_back", "restored_from": record.get("ts", ""), "restored": old}
    return {"status": "skipped", "reason": "no applied calibration history"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate the LLM Wiki recall evidence gate.")
    parser.add_argument("--log-file", default=str(RECALL_LOG_FILE))
    parser.add_argument("--feedback-file", default=str(RECALL_FEEDBACK_FILE))
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rollback:
        payload = rollback_last()
    else:
        payload = calibrate(
            policy=CalibrationPolicy(
                min_samples=max(1, args.min_samples),
                holdout_ratio=args.holdout_ratio,
                min_improvement=max(0.0, args.min_improvement),
            ),
            log_file=Path(args.log_file).expanduser(),
            feedback_file=Path(args.feedback_file).expanduser(),
            dry_run=args.dry_run,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload.get("status") not in {"error"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
