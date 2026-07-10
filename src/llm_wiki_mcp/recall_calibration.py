"""Validated recall gate calibration.

The calibration artifact is intentionally separate from the user's TOML.  The
TOML enables the feature; this module writes the learned runtime artifact only
after a holdout check passes and records the old artifact for rollback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from llm_wiki_mcp.convergence import is_human_required_result
from llm_wiki_mcp.page_mutation import wiki_mutation_lock
from llm_wiki_mcp.recall_eval import read_jsonl
from llm_wiki_mcp.recall_runtime import (
    RECALL_FEEDBACK_FILE,
    RECALL_LOG_FILE,
    RecallPolicy,
    RecallRequest,
    append_jsonl,
    evidence_score,
    load_policy,
    run_recall,
)
from llm_wiki_mcp.recall_runtime_paths import RECALL_DIR


CALIBRATION_FILE = RECALL_DIR / "calibration.json"
CALIBRATION_HISTORY_FILE = RECALL_DIR / "calibration-history.jsonl"
CALIBRATION_RUN_HISTORY_FILE = RECALL_DIR / "calibration-run-history.jsonl"
CALIBRATION_REVIEW_DIR = RECALL_DIR / "calibration-reviews"
CALIBRATION_REVIEW_SCHEMA_VERSION = 1

FEATURE_KEYS = (
    "top1_score_norm",
    "margin_norm",
    "hit_count_norm",
    "heuristic_score",
    "ambiguity",
    "prompt_len_norm",
    "rewrite_confidence",
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _calibration_candidate_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the semantic candidate, excluding timestamps/provenance."""

    return {
        key: artifact.get(key)
        for key in (
            "version",
            "lane",
            "feature_keys",
            "weights",
            "bias",
            "thresholds",
            "samples",
            "train_samples",
            "holdout_samples",
            "holdout",
        )
    }


def _calibration_review_path(
    proposal_hash: str,
    *,
    review_dir: Path = CALIBRATION_REVIEW_DIR,
) -> Path:
    return review_dir / f"candidate-{proposal_hash}.json"


def _load_calibration_review(
    path: Path,
    *,
    proposal_hash: str,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    review = payload.get("review")
    if (
        payload.get("schema_version") != CALIBRATION_REVIEW_SCHEMA_VERSION
        or payload.get("proposal_sha256") != proposal_hash
        or not isinstance(payload.get("proposal"), dict)
        or _canonical_hash(payload.get("proposal")) != proposal_hash
        or not isinstance(review, dict)
        or review.get("decision") not in {"approved", "rejected"}
    ):
        return None
    return payload


def _persist_calibration_review(
    path: Path,
    *,
    proposal: dict[str, Any],
    proposal_hash: str,
    review: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": CALIBRATION_REVIEW_SCHEMA_VERSION,
        "kind": "recall_calibration_frontier_review",
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "proposal_sha256": proposal_hash,
        "proposal": proposal,
        "review": review,
    }
    _atomic_write_json(path, payload)
    verified = _load_calibration_review(path, proposal_hash=proposal_hash)
    if verified is None or verified.get("review") != review:
        raise OSError("calibration frontier review artifact failed read-back verification")
    return verified


@dataclass(frozen=True)
class CalibrationPolicy:
    enabled: bool = True
    min_samples: int = 500
    holdout_ratio: float = 0.2
    min_improvement: float = 0.02
    learning_rate: float = 0.15
    epochs: int = 180
    min_class_samples: int = 25
    search_threshold: float = 0.35
    read_threshold: float = 0.65


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


def score_probabilities(rows: list[tuple[float, int]], *, threshold: float) -> dict[str, float]:
    if not rows:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    tp = fp = tn = fn = 0
    for probability, label in rows:
        pred = probability >= threshold
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
    limit: int = 2000,
    max_recomputed_features: int = 100,
) -> list[dict[str, Any]]:
    logs = {
        str(record.get("decision_id", "")): record
        for record in read_jsonl(log_file)
        if record.get("decision_id")
    }
    rows: list[dict[str, Any]] = []
    feedback_rows = read_jsonl(feedback_file)
    if limit > 0:
        feedback_rows = feedback_rows[-limit:]
    recomputed = 0
    for feedback in feedback_rows:
        label = _label_for_feedback(feedback)
        if label is None:
            continue
        ref = str(feedback.get("ref", ""))
        snapshot = feedback.get("snapshot") if isinstance(feedback.get("snapshot"), dict) else None
        source = logs.get(ref) or snapshot or {}
        features = source.get("evidence_features") or source.get("features")
        if not isinstance(features, dict):
            prompt = str(feedback.get("prompt") or source.get("prompt_preview") or "")
            if prompt and recomputed < max(0, max_recomputed_features):
                recomputed += 1
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
    max_samples: int = 2000,
    max_recomputed_features: int = 100,
    frontier_mode: str = "off",
    frontier_reviewer: Any | None = None,
    budget: Any | None = None,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    if not policy.enabled:
        return {"status": "disabled", "reason": "recall calibration is disabled"}
    rows = load_labeled_rows(
        log_file=log_file,
        feedback_file=feedback_file,
        limit=max_samples,
        max_recomputed_features=max_recomputed_features,
    )
    if len(rows) < policy.min_samples:
        return {
            "status": "skipped",
            "reason": f"not enough labeled samples ({len(rows)} < {policy.min_samples})",
            "samples": len(rows),
        }
    label_counts = {
        0: sum(1 for row in rows if int(row["label"]) == 0),
        1: sum(1 for row in rows if int(row["label"]) == 1),
    }
    if min(label_counts.values()) < max(1, policy.min_class_samples):
        return {
            "status": "skipped",
            "reason": "both positive and negative calibration classes require sufficient support",
            "samples": len(rows),
            "label_counts": label_counts,
        }
    train_rows_raw, holdout_raw = split_holdout(rows, holdout_ratio=policy.holdout_ratio)
    split_label_counts = {
        "train": {
            0: sum(1 for row in train_rows_raw if int(row["label"]) == 0),
            1: sum(1 for row in train_rows_raw if int(row["label"]) == 1),
        },
        "holdout": {
            0: sum(1 for row in holdout_raw if int(row["label"]) == 0),
            1: sum(1 for row in holdout_raw if int(row["label"]) == 1),
        },
    }
    if any(min(counts.values()) < 1 for counts in split_label_counts.values()):
        return {
            "status": "skipped",
            "reason": "temporal train and holdout splits must each contain both calibration classes",
            "samples": len(rows),
            "label_counts": label_counts,
            "split_label_counts": split_label_counts,
        }
    train_rows = [(feature_vector(row["features"]), int(row["label"])) for row in train_rows_raw]
    holdout_rows = [(feature_vector(row["features"]), int(row["label"])) for row in holdout_raw]
    weights, bias = train_logistic(train_rows, policy)
    baseline_policy = RecallPolicy(calibration_enabled=False)
    baseline = score_probabilities(
        [
            (evidence_score(dict(row["features"]), baseline_policy), int(row["label"]))
            for row in holdout_raw
        ],
        threshold=policy.search_threshold,
    )
    candidate = score_probabilities(
        [
            (predict(weights, bias, feature_vector(row["features"])), int(row["label"]))
            for row in holdout_raw
        ],
        threshold=policy.search_threshold,
    )
    improvement = candidate["accuracy"] - baseline["accuracy"]
    artifact = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "lane": "validated-auto",
        "feature_keys": list(FEATURE_KEYS),
        "weights": {key: weights[i] for i, key in enumerate(FEATURE_KEYS)},
        "bias": bias,
        "thresholds": {
            "search": policy.search_threshold,
            "read": policy.read_threshold,
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
    if candidate["precision"] < baseline["precision"] or candidate["recall"] < baseline["recall"]:
        return {
            "status": "skipped",
            "reason": "candidate precision or recall regressed on the temporal holdout",
            "candidate": artifact,
        }
    # Compatibility only: no caller may disable the final semantic reviewer.
    _ = frontier_mode
    old = load_calibration(CALIBRATION_FILE) or {}
    candidate_payload = _calibration_candidate_payload(artifact)
    if old and _calibration_candidate_payload(old) == candidate_payload:
        return {
            "status": "dry_run" if dry_run else "unchanged",
            "calibration": old,
            "reason": "candidate already active",
        }

    old_hash = _canonical_hash(old)
    proposal = {
        "schema_version": CALIBRATION_REVIEW_SCHEMA_VERSION,
        "kind": "recall_calibration",
        "active_calibration_sha256": old_hash,
        "candidate": candidate_payload,
    }
    proposal_hash = _canonical_hash(proposal)
    review_path = _calibration_review_path(
        proposal_hash,
        review_dir=review_dir or CALIBRATION_REVIEW_DIR,
    )
    if dry_run:
        return {
            "status": "dry_run",
            "calibration": artifact,
            "frontier_proposal": proposal,
            "frontier_review_path": str(review_path),
        }

    persisted = _load_calibration_review(
        review_path,
        proposal_hash=proposal_hash,
    )
    review_reused = persisted is not None
    frontier = persisted.get("review") if persisted is not None else None
    if not isinstance(frontier, dict):
        if budget is not None:
            allowed, reason = budget.consume("frontier")
            if not allowed:
                return {
                    "status": "budget_deferred",
                    "reason": reason,
                    "candidate": artifact,
                    "frontier_proposal": proposal,
                }
        frontier = (
            frontier_reviewer(proposal)
            if callable(frontier_reviewer)
            else review_calibration_with_frontier(proposal)
        )
        if not isinstance(frontier, dict):
            frontier = {
                "decision": "needs_retry",
                "summary": "frontier result is not an object",
            }
        if is_human_required_result(frontier):
            return {
                "status": "human_required",
                "reason": str(
                    frontier.get("summary")
                    or "frontier access requires external authority"
                ),
                "candidate": artifact,
                "frontier_review": frontier,
            }
        if frontier.get("decision") in {"approved", "rejected"}:
            try:
                persisted = _persist_calibration_review(
                    review_path,
                    proposal=proposal,
                    proposal_hash=proposal_hash,
                    review=frontier,
                )
                frontier = dict(persisted["review"])
            except OSError as exc:
                return {
                    "status": "frontier_retry",
                    "reason": str(exc),
                    "candidate": artifact,
                    "frontier_review": frontier,
                }

    if frontier.get("decision") != "approved":
        return {
            "status": (
                "frontier_rejected"
                if frontier.get("decision") == "rejected"
                else "frontier_retry"
            ),
            "reason": str(
                frontier.get("summary") or "frontier did not approve calibration"
            ),
            "candidate": artifact,
            "frontier_review": frontier,
            "frontier_review_reused": review_reused,
        }

    if budget is not None:
        mutation_allowed, mutation_reason = budget.consume("mutation")
        if not mutation_allowed:
            return {
                "status": "budget_deferred",
                "reason": mutation_reason,
                "candidate": artifact,
                "frontier_review": frontier,
                "frontier_review_reused": review_reused,
            }

    applied_artifact = {
        **artifact,
        "frontier_provenance": {
            "proposal_sha256": proposal_hash,
            "review_artifact": str(review_path),
            "review_summary": str(frontier.get("summary") or ""),
        },
    }
    try:
        with wiki_mutation_lock():
            current = load_calibration(CALIBRATION_FILE) or {}
            if _canonical_hash(current) != old_hash:
                return {
                    "status": "frontier_retry",
                    "reason": "active calibration changed before apply",
                    "candidate": artifact,
                    "frontier_review": frontier,
                }
            _atomic_write_json(CALIBRATION_FILE, applied_artifact)
            append_jsonl(
                CALIBRATION_HISTORY_FILE,
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "action": "apply",
                    "lane": "validated-auto",
                    "old": old,
                    "new": applied_artifact,
                    "frontier_proposal_sha256": proposal_hash,
                },
            )
    except OSError as exc:
        return {
            "status": "frontier_retry",
            "reason": str(exc),
            "candidate": artifact,
            "frontier_review": frontier,
        }
    return {
        "status": "applied",
        "calibration": applied_artifact,
        "frontier_review": frontier,
        "frontier_review_reused": review_reused,
    }


def review_calibration_with_frontier(
    artifact: dict[str, Any],
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    from llm_wiki_mcp import frontier_review

    prompt = f"""\
You are the final autonomous reviewer for an LLM Wiki recall calibration.
Approve only if the independent holdout evidence supports applying this
artifact without weakening precision or recall safety. Do not edit files,
commit, push, or ask a human. Return JSON matching the supplied frontier
decision schema.

Candidate calibration:
{json.dumps(artifact, ensure_ascii=False, indent=2)}
"""
    repo_root = Path(__file__).resolve().parents[2]
    return frontier_review.run_structured_review(
        prompt,
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=repo_root,
        timeout=timeout or 3600,
        execute_patch=False,
    )


def run_due(
    *,
    policy: CalibrationPolicy | None = None,
    log_file: Path = RECALL_LOG_FILE,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    run_history_file: Path = CALIBRATION_RUN_HISTORY_FILE,
    min_interval_hours: float = 7 * 24,
    max_samples: int = 2000,
    max_recomputed_features: int = 100,
    dry_run: bool = False,
    frontier_mode: str = "auto",
    budget: Any | None = None,
) -> dict[str, Any]:
    if policy is None:
        runtime_policy = load_policy()
        policy = CalibrationPolicy(
            enabled=runtime_policy.calibration_enabled,
            min_samples=runtime_policy.calibration_min_samples,
            holdout_ratio=runtime_policy.calibration_holdout_ratio,
            min_improvement=runtime_policy.calibration_min_improvement,
            search_threshold=runtime_policy.search_threshold,
            read_threshold=runtime_policy.read_threshold,
        )
    history = read_jsonl(run_history_file)
    latest = history[-1] if history else {}
    last_ts = str(latest.get("ts") or "")
    retry_pending = latest.get("status") == "frontier_retry"
    retry_at = str(latest.get("next_retry_at") or "")
    if retry_pending and retry_at:
        try:
            retry_time = datetime.fromisoformat(retry_at)
            retry_now = datetime.now(retry_time.tzinfo) if retry_time.tzinfo is not None else datetime.now()
            if retry_now < retry_time:
                return {"status": "skipped", "reason": "frontier_retry_backoff", "next_retry_at": retry_at}
        except ValueError:
            pass
    if last_ts and not retry_pending:
        try:
            last_time = datetime.fromisoformat(last_ts)
            current_time = datetime.now(last_time.tzinfo) if last_time.tzinfo is not None else datetime.now()
            if current_time - last_time < timedelta(hours=max(0.0, min_interval_hours)):
                return {"status": "skipped", "reason": "interval_not_due", "last_run_at": last_ts}
        except ValueError:
            pass
    result = calibrate(
        policy=policy,
        log_file=log_file,
        feedback_file=feedback_file,
        dry_run=dry_run,
        max_samples=max_samples,
        max_recomputed_features=max_recomputed_features,
        frontier_mode=frontier_mode,
        budget=budget,
    )
    artifact = result.get("calibration") or result.get("candidate") or {}
    candidate_payload = {
        key: artifact.get(key)
        for key in ("weights", "bias", "thresholds", "samples", "holdout")
        if isinstance(artifact, dict)
    }
    candidate_hash = (
        hashlib.sha256(
            json.dumps(candidate_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if candidate_payload
        else ""
    )
    frontier_attempts = 0
    next_retry_at: str | None = None
    if result.get("status") == "frontier_retry":
        frontier_attempts = (
            int(latest.get("frontier_attempts") or 0) + 1
            if latest.get("candidate_hash") == candidate_hash
            else 1
        )
        if frontier_attempts >= 3:
            result = {
                **result,
                "status": "frontier_quarantined",
                "reason": f"{result.get('reason', '')}; frontier retry limit exhausted",
            }
        else:
            next_retry_at = (
                datetime.now() + timedelta(minutes=15 * (2 ** max(0, frontier_attempts - 1)))
            ).isoformat(timespec="seconds")
    if not dry_run and result.get("status") != "budget_deferred":
        append_jsonl(
            run_history_file,
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "status": result.get("status"),
                "samples": (result.get("calibration") or result.get("candidate") or {}).get("samples", 0),
                "reason": result.get("reason", ""),
                "candidate_hash": candidate_hash,
                "frontier_attempts": frontier_attempts,
                "next_retry_at": next_retry_at,
            },
        )
    return result


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
