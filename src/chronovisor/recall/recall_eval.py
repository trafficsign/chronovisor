"""Replay evaluation for Chronovisor recall decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.recall.feedback_ledger import active_feedback_rows
from chronovisor.recall.recall_runtime import (
    RECALL_FEEDBACK_FILE,
    RECALL_LOG_FILE,
    RecallPolicy,
    RecallRequest,
    RecallResult,
    load_policy,
    run_recall,
)
from chronovisor.core.store import CHRONOVISOR_ROOT


BASELINE_DIR = CHRONOVISOR_ROOT / "runtime" / "eval"


@dataclass(frozen=True)
class RecallExample:
    prompt: str
    host: str = "eval"
    cwd: str = ""
    session_id: str = ""
    expected_pages: tuple[str, ...] = ()
    negative_pages: tuple[str, ...] = ()
    injected_pages: tuple[str, ...] = ()
    kind: str = ""
    ref: str = ""
    ts: str = ""

    @property
    def is_positive(self) -> bool:
        # ``page_ignored`` is page-level ranking supervision, not evidence
        # that the recall gate should inject any page for the whole prompt.
        return self.kind != "page_ignored" and bool(self.expected_pages)

    @property
    def is_false_positive(self) -> bool:
        # Lack of an explicit usage receipt is exposure-only supervision. It
        # must never be promoted to a negative label. Only an explicit
        # false-positive receipt contributes to the waste denominator.
        return self.kind == "false-positive"


@dataclass(frozen=True)
class EvalMetrics:
    examples: int
    positives: int
    false_positives: int
    recall_at_1: float
    recall_at_3: float
    mrr: float
    waste_injection_rate: float
    avg_pages: float
    decision_counts: dict[str, int]
    latency_ms: dict[str, float]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def build_dataset(
    *,
    log_file: Path = RECALL_LOG_FILE,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
) -> list[RecallExample]:
    logs_by_id = {
        str(record.get("decision_id", "")): record
        for record in read_jsonl(log_file)
        if record.get("decision_id")
    }
    examples: list[RecallExample] = []
    seen: set[tuple[str, str, tuple[str, ...], tuple[str, ...], str]] = set()
    for feedback in active_feedback_rows(feedback_file):
        kind = str(feedback.get("kind", ""))
        if kind not in {
            "missed",
            "missed_candidate",
            "false-positive",
            "injection_used",
            "injection_ignored",
            "page_ignored",
        }:
            continue
        ref = str(feedback.get("ref", ""))
        snapshot = (
            feedback.get("snapshot")
            if isinstance(feedback.get("snapshot"), dict)
            else None
        )
        record = logs_by_id.get(ref) or snapshot or {}
        prompt = str(feedback.get("prompt") or record.get("prompt_preview") or "")
        if not prompt:
            continue
        expected = _str_tuple(feedback.get("expected_pages"))
        negative = _str_tuple(feedback.get("negative_pages"))
        injected = _str_tuple(record.get("pages")) or _str_tuple(
            feedback.get("injected_pages")
        )
        if kind == "injection_used" and not expected:
            expected = injected
        key = (kind, prompt, expected, negative, ref)
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            RecallExample(
                prompt=prompt,
                host=str(feedback.get("host") or record.get("host") or "eval"),
                cwd=str(record.get("cwd") or ""),
                session_id=str(record.get("session_id") or ""),
                expected_pages=expected,
                negative_pages=negative,
                injected_pages=injected,
                kind=kind,
                ref=ref,
                ts=str(feedback.get("ts") or record.get("ts") or ""),
            )
        )
    return examples


def apply_overrides(policy: RecallPolicy, overrides: list[str] | None) -> RecallPolicy:
    if not overrides:
        return policy
    values = dict(policy.__dict__)
    for item in overrides:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if key not in values:
            continue
        current = values[key]
        if isinstance(current, bool):
            values[key] = raw.lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int):
            try:
                values[key] = int(raw)
            except ValueError:
                pass
        elif isinstance(current, float):
            try:
                values[key] = float(raw)
            except ValueError:
                pass
        else:
            values[key] = raw
    return RecallPolicy(**values)


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def select_examples(
    examples: list[RecallExample], *, limit: int = 0
) -> list[RecallExample]:
    """Select a stable kind-balanced paired corpus instead of log-order rows."""

    if limit <= 0 or len(examples) <= limit:
        return list(examples)
    buckets: dict[str, list[RecallExample]] = {}
    for example in examples:
        buckets.setdefault(example.kind or "unknown", []).append(example)
    for rows in buckets.values():
        rows.sort(
            key=lambda item: hashlib.sha256(
                json.dumps(
                    [
                        item.kind,
                        item.prompt,
                        list(item.expected_pages),
                        list(item.negative_pages),
                        item.ref,
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
    selected: list[RecallExample] = []
    kinds = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for kind in kinds:
            if buckets[kind]:
                selected.append(buckets[kind].pop(0))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def evaluate_examples(
    examples: list[RecallExample],
    *,
    policy: RecallPolicy,
    replay: bool = True,
    k_values: tuple[int, ...] = (1, 3),
    deadline: float | None = None,
) -> dict[str, Any]:
    decision_counts: dict[str, int] = {}
    latencies: list[int] = []
    positives = 0
    false_positives = 0
    hits = {k: 0 for k in k_values}
    waste = 0
    reciprocal_ranks: list[float] = []
    page_counts: list[int] = []
    replay_rows: list[dict[str, Any]] = []

    eval_policy = RecallPolicy(**{**policy.__dict__, "log_decisions": False})
    for example in examples:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("recall evaluation runtime budget exhausted")
        if replay:
            request = RecallRequest(
                host=example.host,
                event="UserPromptSubmit",
                prompt=example.prompt,
                cwd=example.cwd,
                session_id="",
            )
            from chronovisor.hosts.hook_dispatcher import (
                RecallWallClockTimeout,
                recall_outer_deadline_ms,
                recall_wall_clock_deadline,
            )

            started = time.monotonic()
            try:
                with recall_wall_clock_deadline(recall_outer_deadline_ms(eval_policy)):
                    result = run_recall(request, eval_policy, perform_search=True)
            except RecallWallClockTimeout as exc:
                result = RecallResult(
                    status="timeout",
                    decision="none",
                    confidence=0.0,
                    queries=[],
                    reasons=["synchronous recall failed open"],
                    matched_terms={},
                    error=str(exc),
                )
                result.latency_ms = min(
                    eval_policy.total_timeout_ms,
                    round((time.monotonic() - started) * 1000),
                )
            pages = [item.page_id for item in result.context_items]
            decision = result.decision
            latency = result.latency_ms
        else:
            pages = list(example.injected_pages)
            decision = "read" if pages else "none"
            latency = 0

        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        latencies.append(latency)
        page_counts.append(len(pages))
        if example.is_positive:
            positives += 1
            expected = set(example.expected_pages)
            for k in k_values:
                if expected & set(pages[:k]):
                    hits[k] += 1
            rank = next(
                (idx + 1 for idx, page in enumerate(pages) if page in expected), 0
            )
            reciprocal_ranks.append((1.0 / rank) if rank else 0.0)
        if example.is_false_positive:
            false_positives += 1
            if decision != "none" and pages:
                waste += 1
        replay_rows.append(
            {
                "prompt": example.prompt[:180],
                "kind": example.kind,
                "expected_pages": list(example.expected_pages),
                "negative_pages": list(example.negative_pages),
                "pages": pages,
                "decision": decision,
                "latency_ms": latency,
            }
        )

    metrics = EvalMetrics(
        examples=len(examples),
        positives=positives,
        false_positives=false_positives,
        recall_at_1=(hits.get(1, 0) / positives) if positives else 0.0,
        recall_at_3=(hits.get(3, 0) / positives) if positives else 0.0,
        mrr=(sum(reciprocal_ranks) / positives) if positives else 0.0,
        waste_injection_rate=(waste / false_positives) if false_positives else 0.0,
        avg_pages=(sum(page_counts) / len(page_counts)) if page_counts else 0.0,
        decision_counts=decision_counts,
        latency_ms={
            "p50": float(statistics.median(latencies)) if latencies else 0.0,
            "p95": percentile(latencies, 0.95),
            "max": float(max(latencies)) if latencies else 0.0,
        },
    )
    return {"metrics": asdict(metrics), "rows": replay_rows}


def save_baseline(
    payload: dict[str, Any], *, baseline_dir: Path = BASELINE_DIR
) -> Path:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    path = baseline_dir / f"baseline-{stamp}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def run_eval(
    *,
    config_file: Path | None = None,
    log_file: Path = RECALL_LOG_FILE,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    replay: bool = True,
    save: bool = False,
    overrides: list[str] | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    all_examples = build_dataset(log_file=log_file, feedback_file=feedback_file)
    examples = select_examples(all_examples, limit=max(0, limit))
    policy = load_policy(config_file) if config_file else load_policy()
    policy = apply_overrides(policy, overrides)
    result = evaluate_examples(examples, policy=policy, replay=replay)
    payload = {
        "status": "ok",
        "dataset": {
            "examples": len(examples),
            "available_examples": len(all_examples),
            "selection": "stable-kind-balanced"
            if len(examples) < len(all_examples)
            else "all",
            "selection_sha256": hashlib.sha256(
                json.dumps(
                    [[row.kind, row.prompt, row.ref] for row in examples],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "log_file": str(log_file),
            "feedback_file": str(feedback_file),
        },
        "policy": {
            "gate_mode": getattr(policy, "gate_mode", "legacy"),
            "context_style": getattr(policy, "context_style", "legacy"),
            "semantic": policy.semantic,
            "rewrite_enabled": getattr(policy, "rewrite_enabled", False),
            "judge_mode": getattr(policy, "judge_mode", "off"),
        },
        **result,
    }
    if save:
        payload["baseline_file"] = str(save_baseline(payload))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay-evaluate Chronovisor recall decisions."
    )
    parser.add_argument("--config", help="Config TOML path.")
    parser.add_argument("--log-file", default=str(RECALL_LOG_FILE))
    parser.add_argument("--feedback-file", default=str(RECALL_FEEDBACK_FILE))
    parser.add_argument("--replay", action="store_true", default=True)
    parser.add_argument("--no-replay", dest="replay", action="store_false")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stable kind-balanced paired corpus size (0 evaluates every row).",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-recall-eval`` command-line entry point."""
    args = build_parser().parse_args(argv)
    payload = run_eval(
        config_file=Path(args.config).expanduser() if args.config else None,
        log_file=Path(args.log_file).expanduser(),
        feedback_file=Path(args.feedback_file).expanduser(),
        replay=args.replay,
        save=args.save_baseline,
        overrides=args.config_override,
        limit=max(0, args.limit),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        metrics = payload["metrics"]
        print(f"examples\t{metrics['examples']}")
        print(f"recall@3\t{metrics['recall_at_3']:.3f}")
        print(f"waste_injection_rate\t{metrics['waste_injection_rate']:.3f}")
        print(f"latency_p95_ms\t{metrics['latency_ms']['p95']:.1f}")
        if payload.get("baseline_file"):
            print(f"baseline\t{payload['baseline_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
