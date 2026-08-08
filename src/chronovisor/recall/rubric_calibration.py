"""Offline rubric calibration, metrics, promotion, and sync-safe loading."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from chronovisor.core.canonical_json import canonical_json_sha256_stringifying
from chronovisor.core.durable_state import (
    DurableStateError,
    read_sealed_json,
    write_sealed_json,
)
from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.runtime_config import load_decision_router_config
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.decision.graph_decisions import (
    RECALL_RUBRIC_CALIBRATION_SCHEMA,
    RECALL_USEFULNESS_SCHEMA,
    build_recall_rubric_calibration_prompt,
    build_recall_usefulness_prompt,
)
from chronovisor.decision.local_structured import LocalStructuredSession
from chronovisor.knowledge_graph.consensus import router_for_producer
from chronovisor.knowledge_graph.schema import sha256

RUBRIC_ROOT = CHRONOVISOR_ROOT / "runtime" / "recall-rubric"
MIN_RUBRIC_SESSIONS = 5
CANDIDATE_FILE = RUBRIC_ROOT / "candidate.json"
ACTIVE_FILE = RUBRIC_ROOT / "active.json"
LAST_KNOWN_GOOD_FILE = RUBRIC_ROOT / "last-known-good.json"
STATUS_FILE = RUBRIC_ROOT / "status.json"
RUBRIC_SCHEMA_VERSION = 1
RUBRIC_GOLD_STATE_FILE = RUBRIC_ROOT / "gold-builder-state.json"
DEFAULT_RUBRIC = (
    "Pass only answer-bearing evidence that is topically relevant, adds "
    "information not already present, is worth reading now, and is neither "
    "stale nor harmful. Reject weak topical overlap and prefer abstention when "
    "the supplied evidence is insufficient."
)
STRATA = (
    "relevant",
    "adjacent_unneeded",
    "hub_false_positive",
    "self_reference",
    "topic_switch",
    "stale_info",
    "personal_context",
    "read_worthy",
    "multi_hop",
)
RUBRIC_VARIANTS = {
    "current": DEFAULT_RUBRIC,
    "generated": (
        "Approve only if the page contains answer-bearing evidence for the current "
        "question, adds marginal information, is current, and is safe. Reject mere "
        "topic overlap, hubs, self-reference, and adjacent but unnecessary context."
    ),
    "diverse_few_shot": (
        "Judge necessity, not similarity. Relevant and read-worthy evidence passes; "
        "adjacent-unneeded, hub false positives, self-reference, topic switches, stale "
        "facts, and unnecessary personal context fail. Abstain on incomplete evidence."
    ),
    "calibrated": (
        "First enforce valid page ID, digest, freshness, provenance, and certificate. "
        "Then pass only evidence with direct topical relevance, marginal utility, and "
        "enough detail to be worth reading now. Prefer abstention over a false pass."
    ),
}


def _page_excerpt(root: Path, page_id: str) -> str:
    candidates = [
        root / "pages" / f"{page_id}.md",
        root / "system" / f"{page_id}.md",
    ]
    if (root / "pages").exists():
        candidates.extend((root / "pages").rglob(f"{page_id}.md"))
    for path in candidates:
        try:
            return path.read_text(encoding="utf-8")[:2_000]
        except (OSError, UnicodeError):
            continue
    return ""


def _gold_stratum(row: Mapping[str, Any], *, page_id: str, gold: bool) -> str:
    query = str(row.get("query") or "").casefold()
    kind = str(row.get("kind") or "")
    source = str(row.get("source") or "")
    stale = {str(value) for value in row.get("stale_pages") or []}
    if page_id in stale:
        return "stale_info"
    negative_pages = [
        str(value) for value in row.get("negative_pages") or [] if isinstance(value, str)
    ]
    if (
        not gold
        and kind == "injection_ignored"
        and negative_pages
        and page_id == negative_pages[0]
    ):
        return "topic_switch"
    if not gold and source == "auditor_precision":
        return "hub_false_positive"
    if not gold and (
        kind == "injection_ignored" or "関係ない" in query or "topic" in query
    ):
        return "topic_switch"
    system_terms = (
        "chronovisor",
        "クロノバイザー",
        "llmウィキ",
        "システム",
        "自律",
        "保存",
    )
    system_pages = (
        "chronovisor-",
        "wiki-memory-",
        "memory-system-",
        "codex-wiki-",
    )
    if gold and any(term in query for term in system_terms) and page_id.startswith(
        system_pages
    ):
        return "self_reference"
    expected = row.get("expected_pages")
    if gold and isinstance(expected, list) and len(expected) >= 2:
        return "multi_hop"
    if any(term in query for term in ("俺", "自分", "personal", "面接", "career")):
        return "personal_context"
    if any(term in query for term in ("関係", "経由", "つなが", "multi", "between")):
        return "multi_hop"
    if page_id.casefold() in query or "このページ" in query:
        return "self_reference"
    if not gold:
        return "adjacent_unneeded"
    return "read_worthy" if len(query) >= 80 else "relevant"


def _review_session_hashes(root: Path) -> dict[str, str]:
    """Join reviewed refs to privacy-safe session identities."""

    output: dict[str, str] = {}
    try:
        lines = (root / "recall" / "feedback.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError:
        return output
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        ref = str(row.get("ref") or "")
        snapshot = row.get("snapshot")
        turn_ref = row.get("turn_ref")
        session_id = str(
            (
                snapshot.get("session_id")
                if isinstance(snapshot, dict)
                else ""
            )
            or (
                turn_ref.get("session_id")
                if isinstance(turn_ref, dict)
                else ""
            )
            or ""
        )
        if ref and session_id:
            output[ref] = sha256(["rubric-session-v1", session_id])
    return output


def _locked_manual_manifest(root: Path) -> dict[str, Any]:
    """Read and verify the search-eval manifest's legacy custom seal."""

    path = root / "runtime" / "search-eval" / "manual-94-manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    observed = str(payload.get("manifest_sha256") or "")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if observed != canonical_json_sha256_stringifying(unsigned):
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    for entry in entries:
        if not isinstance(entry, dict):
            return {}
        entry_seal = str(entry.get("entry_sha256") or "")
        entry_unsigned = {
            key: value for key, value in entry.items() if key != "entry_sha256"
        }
        if entry_seal != canonical_json_sha256_stringifying(entry_unsigned):
            return {}
    return payload


def _gold_cases(root: Path, golden_file: Path) -> list[dict[str, Any]]:
    manifest = _locked_manual_manifest(root)
    if not manifest:
        return []
    entries = manifest.get("entries")
    allowed = {
        str(row.get("query_sha256") or "")
        for row in entries or []
        if isinstance(row, dict) and row.get("reviewed") is True
    }
    if not allowed:
        return []
    session_hashes = _review_session_hashes(root)
    cases: list[dict[str, Any]] = []
    try:
        lines = golden_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("reviewed") is not True:
            continue
        query = str(row.get("query") or "")
        query_sha = sha256(query)
        if not query or query_sha not in allowed:
            continue
        labeled: list[tuple[str, bool]] = [
            (str(page_id), True)
            for page_id in row.get("expected_pages") or []
            if isinstance(page_id, str)
        ]
        labeled.extend(
            (str(page_id), False)
            for page_id in [
                *list(row.get("negative_pages") or []),
                *list(row.get("stale_pages") or []),
            ]
            if isinstance(page_id, str)
        )
        for page_id, gold in labeled:
            if not _page_excerpt(root, page_id):
                continue
            case_id = "rubric_case_" + sha256([query_sha, page_id, gold])[:24]
            cases.append(
                {
                    "case_id": case_id,
                    "query": query,
                    "query_sha256": query_sha,
                    "page_id": page_id,
                    "page_id_sha256": sha256(page_id),
                    "gold": gold,
                    "stratum": _gold_stratum(row, page_id=page_id, gold=gold),
                    "split": str(row.get("split") or ""),
                    "session_hash": session_hashes.get(str(row.get("ref") or ""), ""),
                    "review_receipt_id": "manual94:" + str(row.get("ref") or query_sha),
                }
            )
    # Build the same diverse, query-deduplicated order that calibration will
    # consume.  Otherwise hundreds of duplicate-page cases can spend the daily
    # model budget before all strata/splits have a single usable example.
    return select_diverse_cases(cases, limit=len(cases))


def _usefulness_prediction(value: Mapping[str, Any] | None) -> bool | str:
    if not isinstance(value, Mapping):
        return "abstain"
    if value.get("decision") in {"abstained", "needs_retry"}:
        return "abstain"
    return bool(
        value.get("decision") == "approved"
        and value.get("topically_relevant") is True
        and value.get("marginally_useful") is True
        and value.get("read_worthy") is True
        and value.get("stale_or_harmful") is False
    )


def _locked_gold_rows(path: Path) -> list[dict[str, Any]]:
    """Load only rows carrying the current privacy and split contract."""

    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    required_predictions = {
        *RUBRIC_VARIANTS,
        "primary",
        "challenger",
        "tie_break",
        "ensemble",
    }
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and value.get("reviewed") is True
            and str(value.get("split") or "") in {"train", "dev", "locked-test"}
            and re.fullmatch(r"[0-9a-f]{64}", str(value.get("session_hash") or ""))
            and required_predictions.issubset(value)
        ):
            rows.append(value)
    return rows


def _judge_variant(
    case: Mapping[str, Any],
    *,
    rubric_name: str,
    model: str,
    root: Path,
) -> tuple[bool | str, float]:
    excerpt = _page_excerpt(root, str(case.get("page_id") or ""))
    evidence = {
        "query": str(case.get("query") or "")[:1_000],
        "page_id": str(case.get("page_id") or ""),
        "page_excerpt": excerpt,
        "objective_checks": {
            "known_page_id": bool(excerpt),
            "content_digest_valid": True,
            "not_stale": str(case.get("stratum") or "") != "stale_info",
            "certificate_present": True,
        },
    }
    result = LocalStructuredSession(
        model=model,
        role=f"recall_rubric:{rubric_name}",
        audit_root=root / "runtime" / "recall-rubric" / "structured-audit",
        num_ctx=8_192,
        num_predict=160,
        keep_alive="20m",
        read_timeout_ms=180_000,
        max_input_chars=12_000,
        max_output_chars=2_000,
        max_responses=2,
        resource_managed=True,
        resource_lease_timeout_ms=25,
    ).run(
        build_recall_usefulness_prompt(evidence, RUBRIC_VARIANTS[rubric_name]),
        RECALL_USEFULNESS_SCHEMA,
        system="Return one local recall-usefulness decision as JSON only.",
    )
    value = result.value if result.ok and isinstance(result.value, dict) else None
    confidence = float(value.get("confidence") or 0.0) if value else 0.0
    return _usefulness_prediction(value), max(0.0, min(1.0, confidence))


def _judge_consensus(case: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    excerpt = _page_excerpt(root, str(case.get("page_id") or ""))
    evidence = {
        "query": str(case.get("query") or "")[:1_000],
        "page_id": str(case.get("page_id") or ""),
        "page_excerpt": excerpt,
    }
    result = router_for_producer("deterministic", "recall_usefulness_judgment").decide(
        build_recall_usefulness_prompt(evidence, RUBRIC_VARIANTS["calibrated"]),
        RECALL_USEFULNESS_SCHEMA,
        decision_lane="recall_usefulness_judgment",
    )
    output: dict[str, Any] = {
        "primary": "abstain",
        "primary_confidence": 0.0,
        "challenger": "abstain",
        "challenger_confidence": 0.0,
        "tie_break": "abstain",
        "tie_break_confidence": 0.0,
        "ensemble": _usefulness_prediction(
            result.value if isinstance(result.value, dict) else None
        ),
        "ensemble_confidence": float(
            result.value.get("confidence") or 0.0
            if isinstance(result.value, dict)
            else 0.0
        ),
        "consensus_receipt_sha256": result.agreement_sha256,
    }
    for vote in result.votes:
        role = vote.role if vote.role in {"primary", "challenger", "tie_break"} else ""
        if not role:
            continue
        value = vote.result.value if isinstance(vote.result.value, dict) else None
        output[role] = _usefulness_prediction(value)
        output[f"{role}_confidence"] = (
            float(value.get("confidence") or 0.0) if value else 0.0
        )
    return output


def build_locked_gold_cycle(
    *,
    root: Path,
    golden_file: Path,
    output_file: Path,
    state_file: Path = RUBRIC_GOLD_STATE_FILE,
    max_steps_per_day: int = 4,
    max_model_seconds_per_day: int = 7_200,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Incrementally label reviewed cases without persisting query or page text."""

    cases = _gold_cases(root, golden_file)
    try:
        state = read_sealed_json(state_file, recover_backup=True)
    except Exception:
        state = {}
    today = datetime.now(UTC).date().isoformat()
    steps_today = (
        int(state.get("steps_today") or 0) if state.get("date") == today else 0
    )
    model_seconds_today = (
        float(state.get("model_seconds_today") or 0.0)
        if state.get("date") == today
        else 0.0
    )
    completed = {
        str(row.get("case_id") or "")
        for row in _locked_gold_rows(output_file)
        if str(row.get("case_id") or "")
    }
    pending_value = state.get("pending")
    pending = dict(pending_value) if isinstance(pending_value, dict) else {}
    case = next(
        (
            row
            for row in cases
            if row["case_id"] == pending.get("case_id")
            and row["case_id"] not in completed
        ),
        None,
    )
    if case is None:
        case = next((row for row in cases if row["case_id"] not in completed), None)
        pending = {
            key: value for key, value in (case or {}).items() if key not in {"query"}
        }
    if case is None:
        return {
            "status": "complete",
            "cases": len(completed),
            "pending": False,
            "external_model_calls": 0,
        }
    if (
        dry_run
        or steps_today >= max(0, max_steps_per_day)
        or model_seconds_today >= max_model_seconds_per_day
    ):
        return {
            "status": "waiting",
            "reason": "dry_run_or_daily_budget",
            "cases": len(completed),
            "pending": True,
            "external_model_calls": 0,
        }
    predictions_value = pending.get("predictions")
    predictions = dict(predictions_value) if isinstance(predictions_value, dict) else {}
    next_variant = next(
        (name for name in RUBRIC_VARIANTS if name not in predictions), None
    )
    started = monotonic()
    if next_variant is not None:
        model = load_decision_router_config().tie_break_model
        prediction, confidence = _judge_variant(
            case, rubric_name=next_variant, model=model, root=root
        )
        predictions[next_variant] = prediction
        predictions[f"{next_variant}_confidence"] = confidence
        step = next_variant
    else:
        predictions.update(_judge_consensus(case, root=root))
        step = "local_consensus"
    elapsed = max(0.0, monotonic() - started)
    pending["predictions"] = predictions
    required = {
        *RUBRIC_VARIANTS,
        "primary",
        "challenger",
        "tie_break",
        "ensemble",
    }
    finished = required.issubset(predictions)
    if finished and not dry_run:
        append_jsonl_durable(
            output_file,
            [
                {
                    "schema_version": 1,
                    "case_id": case["case_id"],
                    "query_sha256": case["query_sha256"],
                    "page_id_sha256": case["page_id_sha256"],
                    "gold": case["gold"],
                    "stratum": case["stratum"],
                    "split": case["split"],
                    "session_hash": case["session_hash"],
                    "reviewed": True,
                    "review_receipt_sha256": sha256(case["review_receipt_id"]),
                    **predictions,
                    "observed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "external_model_calls": 0,
                }
            ],
            sort_keys=True,
        )
        completed.add(str(case["case_id"]))
        pending = {}
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "date": today,
        "status": "ok",
        "step": step,
        "cases": len(completed),
        "available_cases": len(cases),
        "pending": pending,
        "completed_case_ids": sorted(completed),
        "steps_today": steps_today + 1,
        "model_seconds_today": round(model_seconds_today + elapsed, 3),
        "external_model_calls": 0,
    }
    return payload if dry_run else write_sealed_json(state_file, payload)


def build_rubric_artifact(
    *,
    rubric_text: str,
    task: str = "recall_usefulness",
    case_refs: Sequence[str] = (),
    model_sha256: str = "",
    version: int = 1,
) -> dict[str, Any]:
    if not rubric_text.strip() or len(rubric_text) > 8_000:
        raise ValueError("rubric text must be bounded and non-empty")
    core = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_id": "rubric_" + sha256([task, version, rubric_text])[:24],
        "version": version,
        "task": task,
        "objective_checks": [
            "known_page_id",
            "content_digest_valid",
            "not_stale",
            "certificate_present",
        ],
        "subjective_criteria": [
            "topical_relevance",
            "marginal_utility",
            "read_worthy",
            "personal_context_necessity",
        ],
        "rubric_text": rubric_text,
        "case_ref_sha256s": sorted(sha256(value) for value in case_refs),
        "model_sha256": model_sha256,
        "prompt_sha256": sha256([task, rubric_text, sorted(case_refs)]),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return {**core, "artifact_sha256": sha256(core)}


def select_diverse_cases(
    rows: Sequence[Mapping[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    """Round-robin strata while excluding repeated query/session identities."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stratum = str(row.get("stratum") or "")
        if stratum in STRATA:
            buckets[stratum].append(dict(row))
    selected: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    seen_sessions: set[str] = set()
    while len(selected) < max(0, limit):
        changed = False
        for stratum in STRATA:
            eligible = [
                (index, row)
                for index, row in enumerate(buckets[stratum])
                if str(row.get("query_sha256") or "") not in seen_queries
            ]
            if not eligible:
                buckets[stratum].clear()
                continue
            index, row = next(
                (
                    (index, value)
                    for index, value in eligible
                    if str(value.get("session_hash") or "")
                    and str(value.get("session_hash") or "") not in seen_sessions
                ),
                eligible[0],
            )
            buckets[stratum].pop(index)
            query = str(row.get("query_sha256") or "")
            session = str(row.get("session_hash") or "")
            selected.append(row)
            seen_queries.add(query)
            if session:
                seen_sessions.add(session)
            changed = True
            if len(selected) >= limit:
                break
        if not changed:
            break
    return selected


def _binary_metrics(rows: Sequence[Mapping[str, Any]], model: str) -> dict[str, float]:
    tp = fp = fn = correct = decided = 0
    brier_values: list[float] = []
    calibration: list[tuple[float, int]] = []
    abstained = 0
    for row in rows:
        label = row.get("gold")
        prediction = row.get(model)
        confidence = row.get(f"{model}_confidence", 0.5)
        if not isinstance(label, bool):
            continue
        if prediction == "abstain" or prediction is None:
            abstained += 1
            continue
        predicted = bool(prediction)
        decided += 1
        correct += predicted == label
        tp += predicted and label
        fp += predicted and not label
        fn += not predicted and label
        probability = (
            max(0.0, min(1.0, float(confidence)))
            if isinstance(confidence, int | float)
            else 0.5
        )
        probability = probability if predicted else 1.0 - probability
        brier_values.append((probability - float(label)) ** 2)
        calibration.append((probability, int(label)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    ece = 0.0
    if calibration:
        for start in (0.0, 0.2, 0.4, 0.6, 0.8):
            bucket = [row for row in calibration if start <= row[0] <= start + 0.2]
            if bucket:
                ece += (
                    len(bucket)
                    / len(calibration)
                    * abs(
                        sum(item[0] for item in bucket) / len(bucket)
                        - sum(item[1] for item in bucket) / len(bucket)
                    )
                )
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "accuracy": round(correct / decided if decided else 0.0, 6),
        "brier": round(sum(brier_values) / len(brier_values), 6)
        if brier_values
        else 1.0,
        "ece": round(ece, 6),
        "abstention": round(abstained / max(1, len(rows)), 6),
    }


def _correlation(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return round(numerator / denominator, 6) if denominator else 0.0


def evaluate_judges(
    rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str] = ("primary", "challenger", "tie_break", "ensemble"),
) -> dict[str, Any]:
    per_model = {model: _binary_metrics(rows, model) for model in models}
    errors: dict[str, list[int]] = {}
    for model in models:
        errors[model] = [
            int(row.get(model) not in {row.get("gold"), "abstain"})
            for row in rows
            if isinstance(row.get("gold"), bool)
        ]
    pairwise = {
        f"{left}:{right}": _correlation(errors[left], errors[right])
        for index, left in enumerate(models)
        for right in models[index + 1 :]
    }
    unanimous_wrong = sum(
        all(row.get(model) not in {row.get("gold"), "abstain"} for model in models[:3])
        for row in rows
        if isinstance(row.get("gold"), bool)
    )
    best_single = max(
        (per_model[model]["accuracy"] for model in models[:3]), default=0.0
    )
    ensemble = per_model.get("ensemble", {}).get("accuracy", 0.0)
    strata_counts = {
        stratum: sum(row.get("stratum") == stratum for row in rows)
        for stratum in STRATA
    }
    sessions = {
        str(row.get("session_hash") or "")
        for row in rows
        if str(row.get("session_hash") or "")
    }
    split_counts = {
        split: {
            "positive": sum(
                row.get("split") == split and row.get("gold") is True for row in rows
            ),
            "negative": sum(
                row.get("split") == split and row.get("gold") is False for row in rows
            ),
        }
        for split in ("train", "dev", "locked-test")
    }
    return {
        "schema_version": 1,
        "samples": len(rows),
        "models": per_model,
        "pairwise_error_correlation": pairwise,
        "unanimous_wrong_rate": round(unanimous_wrong / max(1, len(rows)), 6),
        "best_single_accuracy": best_single,
        "ensemble_accuracy": ensemble,
        "ensemble_gain": round(ensemble - best_single, 6),
        "strata_counts": strata_counts,
        "session_count": len(sessions),
        "split_counts": split_counts,
    }


def promote_candidate(
    *,
    candidate_file: Path = CANDIDATE_FILE,
    active_file: Path = ACTIVE_FILE,
    last_known_good_file: Path = LAST_KNOWN_GOOD_FILE,
    metrics: Mapping[str, Any],
    gold_count: int,
    status_file: Path = STATUS_FILE,
) -> dict[str, Any]:
    candidate = read_sealed_json(candidate_file)
    ensemble_gain = metrics.get("ensemble_gain")
    model_metrics = metrics.get("models")
    ensemble_metrics = (
        model_metrics.get("ensemble") if isinstance(model_metrics, Mapping) else None
    )
    precision = (
        float(ensemble_metrics.get("precision", 0.0))
        if isinstance(ensemble_metrics, Mapping)
        else 0.0
    )
    ece = (
        float(ensemble_metrics.get("ece", 1.0))
        if isinstance(ensemble_metrics, Mapping)
        else 1.0
    )
    abstention = (
        float(ensemble_metrics.get("abstention", 1.0))
        if isinstance(ensemble_metrics, Mapping)
        else 1.0
    )
    strata_value = metrics.get("strata_counts")
    strata_counts = strata_value if isinstance(strata_value, Mapping) else {}
    split_value = metrics.get("split_counts")
    split_counts = split_value if isinstance(split_value, Mapping) else {}
    gates = {
        "gold_samples": gold_count >= 30,
        "session_diversity": int(metrics.get("session_count") or 0)
        >= MIN_RUBRIC_SESSIONS,
        "holdout_non_regression": precision >= 0.90,
        "calibration": ece <= 0.10,
        "coverage": abstention <= 0.50,
        "ensemble_value": isinstance(ensemble_gain, int | float)
        and float(ensemble_gain) > 0.0,
        "strata_coverage": all(
            isinstance(strata_counts.get(stratum), int)
            and not isinstance(strata_counts.get(stratum), bool)
            and int(strata_counts[stratum]) > 0
            for stratum in STRATA
        ),
        "split_balance": all(
            isinstance(split_counts.get(split), Mapping)
            and int(split_counts[split].get("positive") or 0) > 0
            and int(split_counts[split].get("negative") or 0) > 0
            for split in ("train", "dev", "locked-test")
        ),
    }
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rubric_id": candidate.get("rubric_id"),
        "candidate_sha256": candidate.get("artifact_sha256"),
        "gates": gates,
        "status": "adopted" if all(gates.values()) else "held",
    }
    if all(gates.values()):
        try:
            prior = read_sealed_json(active_file)
            write_sealed_json(last_known_good_file, prior)
        except DurableStateError:
            pass
        write_sealed_json(active_file, candidate)
    write_sealed_json(status_file, result)
    return result


def load_active_rubric(path: Path = ACTIVE_FILE) -> dict[str, Any]:
    """Load only a sealed adopted artifact; no model is called synchronously."""

    try:
        payload = read_sealed_json(path, recover_backup=True)
    except DurableStateError:
        return {
            "rubric_id": "builtin-v1",
            "rubric_text": DEFAULT_RUBRIC,
            "source": "builtin",
        }
    if payload.get("schema_version") != RUBRIC_SCHEMA_VERSION or not isinstance(
        payload.get("rubric_text"), str
    ):
        return {
            "rubric_id": "builtin-v1",
            "rubric_text": DEFAULT_RUBRIC,
            "source": "builtin",
        }
    return {
        "rubric_id": str(payload.get("rubric_id") or ""),
        "rubric_text": str(payload["rubric_text"]),
        "artifact_sha256": str(payload.get("artifact_sha256") or ""),
        "source": "active",
    }


def write_candidate(
    artifact: Mapping[str, Any], path: Path = CANDIDATE_FILE
) -> dict[str, Any]:
    return write_sealed_json(path, dict(artifact))


def evaluate_rubric_variants(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = {name: _binary_metrics(rows, name) for name in RUBRIC_VARIANTS}
    eligible = [
        name
        for name, value in metrics.items()
        if value["precision"] >= metrics["current"]["precision"]
        and value["abstention"] <= 0.50
    ]
    winner = max(
        eligible,
        key=lambda name: (
            metrics[name]["accuracy"],
            -metrics[name]["brier"],
            -metrics[name]["ece"],
        ),
        default="current",
    )
    return {"metrics": metrics, "eligible": eligible, "winner": winner}


def run_calibration_cycle(
    *,
    rows_file: Path = RUBRIC_ROOT / "locked-gold.jsonl",
    candidate_file: Path = CANDIDATE_FILE,
    status_file: Path = STATUS_FILE,
    outcomes_file: Path = RUBRIC_ROOT / "outcomes.jsonl",
    active_file: Path = ACTIVE_FILE,
    last_known_good_file: Path = LAST_KNOWN_GOOD_FILE,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Auto-compare 10/30-case rubric variants and adopt only sealed winners."""

    rows = _locked_gold_rows(rows_file)
    selected = select_diverse_cases(rows, limit=30)
    selected_sessions = {
        str(row.get("session_hash") or "")
        for row in selected
        if str(row.get("session_hash") or "")
    }
    sample_gate = 30 if len(selected) >= 30 else 10 if len(selected) >= 10 else 0
    comparison = (
        evaluate_rubric_variants(selected[:sample_gate])
        if sample_gate
        else {"metrics": {}, "eligible": [], "winner": "current"}
    )
    winner = str(comparison["winner"])
    artifact = build_rubric_artifact(
        rubric_text=RUBRIC_VARIANTS[winner],
        case_refs=[
            str(row.get("case_id") or row.get("query_sha256") or "")
            for row in selected[:sample_gate]
        ],
        model_sha256=sha256("local-three-model-calibration-v1"),
        version=sample_gate or 1,
    )
    judge_metrics = (
        evaluate_judges(selected[:sample_gate])
        if sample_gate
        else {"models": {}, "ensemble_gain": -1.0}
    )
    if not dry_run:
        write_candidate(artifact, candidate_file)
    if sample_gate < 30 or len(selected_sessions) < MIN_RUBRIC_SESSIONS:
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": "collecting",
            "samples": len(selected),
            "sessions": len(selected_sessions),
            "next_gate": 10 if len(selected) < 10 else 30,
            "next_session_gate": MIN_RUBRIC_SESSIONS,
            "winner": winner,
            "comparison": comparison,
            "external_model_calls": 0,
        }
        return payload if dry_run else write_sealed_json(status_file, payload)
    calibration_evidence = {
        "rubric_id": artifact["rubric_id"],
        "candidate_sha256": artifact["artifact_sha256"],
        "case_manifest_sha256": sha256(
            sorted(str(row.get("query_sha256") or "") for row in selected)
        ),
        "comparison": comparison,
        "judge_metrics": judge_metrics,
        "gold_count": len(selected),
        "rollback_target": str(last_known_good_file),
    }
    calibration_epoch = datetime.now(UTC).date().isoformat()
    try:
        prior_status = read_sealed_json(status_file, recover_backup=True)
    except DurableStateError:
        prior_status = {}
    cached_receipt = prior_status.get("consensus")
    cache_hit = bool(
        isinstance(cached_receipt, dict)
        and prior_status.get("candidate_sha256") == artifact["artifact_sha256"]
        and prior_status.get("calibration_epoch") == calibration_epoch
    )
    consensus_result = (
        None
        if dry_run or cache_hit
        else router_for_producer("deterministic", "recall_rubric_calibration").decide(
            build_recall_rubric_calibration_prompt(calibration_evidence),
            RECALL_RUBRIC_CALIBRATION_SCHEMA,
            decision_lane="recall_rubric_calibration",
        )
    )
    consensus_value = (
        consensus_result.value
        if consensus_result is not None and isinstance(consensus_result.value, dict)
        else {}
    )
    consensus_passed = (
        bool(cached_receipt.get("passed"))
        if cache_hit and isinstance(cached_receipt, dict)
        else bool(
            consensus_result is not None
            and consensus_result.ok
            and consensus_value.get("decision") == "approved"
            and consensus_value.get("holdout_non_regression") is True
            and consensus_value.get("calibration_improved") is True
            and consensus_value.get("coverage_preserved") is True
            and consensus_value.get("rollback_safe") is True
        )
    )
    consensus_receipt = (
        dict(cached_receipt)
        if cache_hit and isinstance(cached_receipt, dict)
        else {
            "receipt_id": "rubric_receipt_"
            + sha256(
                [
                    artifact["artifact_sha256"],
                    getattr(consensus_result, "agreement_sha256", ""),
                    consensus_passed,
                ]
            )[:20],
            "passed": consensus_passed,
            "agreement_sha256": getattr(consensus_result, "agreement_sha256", ""),
            "failure_class": getattr(consensus_result, "failure_class", None),
            "vote_manifest_sha256": sha256(
                [
                    vote.signature_sha256
                    for vote in getattr(consensus_result, "votes", ())
                ]
            ),
            "external_model_calls": 0,
        }
    )
    if not dry_run and not cache_hit:
        append_jsonl_durable(
            outcomes_file.parent / "consensus-receipts.jsonl",
            [
                {
                    "schema_version": 1,
                    "rubric_id": artifact["rubric_id"],
                    **consensus_receipt,
                }
            ],
            sort_keys=True,
        )
    if not dry_run and not consensus_passed:
        return write_sealed_json(
            status_file,
            {
                "schema_version": 1,
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "status": "held",
                "candidate_sha256": artifact["artifact_sha256"],
                "calibration_epoch": calibration_epoch,
                "samples": len(selected),
                "sessions": len(selected_sessions),
                "winner": winner,
                "comparison": comparison,
                "judge_metrics": judge_metrics,
                "gates": {"local_consensus": False},
                "consensus": consensus_receipt,
                "external_model_calls": 0,
            },
        )
    result = (
        promote_candidate(
            candidate_file=candidate_file,
            active_file=active_file,
            last_known_good_file=last_known_good_file,
            metrics=judge_metrics,
            gold_count=len(selected),
            status_file=status_file,
        )
        if not dry_run
        else {"status": "dry_run", "gates": {}}
    )
    result_gate_value = result.get("gates")
    result_gates = result_gate_value if isinstance(result_gate_value, Mapping) else {}
    payload = {
        **result,
        "samples": len(selected),
        "sessions": len(selected_sessions),
        "winner": winner,
        "comparison": comparison,
        "judge_metrics": judge_metrics,
        "consensus": consensus_receipt,
        "candidate_sha256": artifact["artifact_sha256"],
        "calibration_epoch": calibration_epoch,
        "gates": {**dict(result_gates), "local_consensus": consensus_passed},
        "external_model_calls": 0,
    }
    if not dry_run:
        write_sealed_json(status_file, payload)
        existing_ids: set[str] = set()
        try:
            existing_ids = {
                str(json.loads(line).get("rubric_id") or "")
                for line in outcomes_file.read_text(encoding="utf-8").splitlines()
            }
        except (OSError, json.JSONDecodeError):
            pass
        outcome_rows = [
            {
                "schema_version": 1,
                "rubric_id": f"{artifact['rubric_id']}:{sha256(str(row.get('case_id') or row.get('query_sha256') or ''))[:12]}",
                "subject_id": artifact["rubric_id"],
                "query_sha256": str(
                    row.get("query_sha256") or sha256(str(row.get("case_id") or ""))
                ),
                "polarity": "positive",
                "quality": "gold",
                "receipt_id": str(
                    row.get("review_receipt_id") or "reviewed_locked_gold"
                ),
                "observed_at": str(row.get("observed_at") or ""),
            }
            for row in selected
            if f"{artifact['rubric_id']}:{sha256(str(row.get('case_id') or row.get('query_sha256') or ''))[:12]}"
            not in existing_ids
        ]
        if outcome_rows:
            append_jsonl_durable(outcomes_file, outcome_rows, sort_keys=True)
    return payload
