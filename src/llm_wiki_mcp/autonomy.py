"""Autonomous operation layer for LLM Wiki.

This module replaces human review queues with reversible machine decisions:
safe items are applied, uncertain items are deferred for the next cycle, and
health regressions quarantine the batch instead of waiting for a person.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import plistlib
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from llm_wiki_mcp.convergence import ConvergenceStore, CycleBudget
from llm_wiki_mcp.frontmatter import parse as parse_frontmatter
from llm_wiki_mcp.frontmatter import patch as patch_frontmatter
from llm_wiki_mcp.wiki import WIKI_ROOT, find_page


AUTONOMY_DIR = WIKI_ROOT / "autonomy"
DECISIONS_FILE = AUTONOMY_DIR / "decisions.jsonl"
LATEST_FILE = AUTONOMY_DIR / "latest.json"
WATCHDOG_FILE = AUTONOMY_DIR / "watchdog-latest.json"
WATCHDOG_HISTORY = AUTONOMY_DIR / "watchdog-history.jsonl"
DIGEST_FILE = AUTONOMY_DIR / "digest-latest.md"
QUARANTINE_FILE = AUTONOMY_DIR / "quarantine.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SLEEP_LABEL = "com.trafficsign.llm-wiki-sleep"
WATCHDOG_LABEL = "com.trafficsign.llm-wiki-watchdog"
LAUNCH_AGENT_DIR = Path.home() / "Library" / "LaunchAgents"
WRAPPER_DIR = WIKI_ROOT / "bin"
DUPLICATE_FRONTIER_LANE = "duplicate_frontier"
DUPLICATE_FRONTIER_RESOLVER_VERSION = "duplicate-frontier-v1"
DUPLICATE_SUPERSEDE_MIN_CONFIDENCE = 0.8
DUPLICATE_FRONTIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "confidence", "summary"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["supersede_left", "supersede_right", "keep_both", "needs_retry"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
    },
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _latest_jsonl(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
    except OSError:
        return {}
    if not lines:
        return {}
    try:
        row = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}
    return row if isinstance(row, dict) else {}


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _norm(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.casefold()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _git(args: list[str], *, cwd: Path = WIKI_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _page_meta(page_id: str) -> dict[str, Any]:
    from llm_wiki_mcp.index_store import get_store

    store = get_store()
    store.refresh()
    return store.meta(page_id) or {}


def _page_quality(page_id: str, meta: dict[str, Any] | None = None) -> float:
    from llm_wiki_mcp.index_store import get_store

    meta = meta or _page_meta(page_id)
    if not meta:
        return -1.0
    score = 0.0
    if str(meta.get("summary") or "").strip():
        score += 3.0
    questions = meta.get("recall_questions")
    if isinstance(questions, list):
        score += min(3.0, len(questions) * 0.75)
    path_value = meta.get("path")
    if isinstance(path_value, str):
        try:
            size = Path(path_value).stat().st_size
            score += min(3.0, size / 3000.0)
        except OSError:
            pass
    try:
        store = get_store()
        score += min(2.0, len(store.backlinks(page_id)) * 0.3)
        score += min(1.0, len(store.outlinks(page_id)) * 0.1)
    except Exception:
        pass
    updated = _parse_dt(str(meta.get("updated") or ""))
    if updated and datetime.now() - updated <= timedelta(days=90):
        score += 0.5
    return score


def _patch_page_status(
    page_id: str,
    updates: dict[str, Any],
    *,
    expected_hash: str | None = None,
) -> dict[str, Any]:
    path = find_page(page_id)
    if path is None:
        return {"status": "skipped", "reason": "page_not_found", "page_id": page_id}
    try:
        original = path.read_bytes()
        text = original.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {"status": "skipped", "reason": f"read_error: {exc}", "page_id": page_id}
    observed_hash = hashlib.sha256(original).hexdigest()
    if expected_hash is not None and observed_hash != expected_hash:
        return {"status": "retry", "reason": "page_changed_before_apply", "page_id": page_id}
    new_text = patch_frontmatter(text, updates)
    if new_text == text:
        return {"status": "unchanged", "page_id": page_id, "path": str(path)}
    written = new_text.encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{observed_hash[:12]}.tmp")
    try:
        tmp.write_bytes(written)
        if path.read_bytes() != original:
            tmp.unlink(missing_ok=True)
            return {"status": "retry", "reason": "page_changed_before_replace", "page_id": page_id}
        os.replace(tmp, path)
        if path.read_bytes() != written:
            return {"status": "retry", "reason": "post_write_verification_failed", "page_id": page_id}
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        return {"status": "retry", "reason": f"write_error: {exc}", "page_id": page_id}
    return {"status": "applied", "page_id": page_id, "path": str(path), "updates": updates}


def decide_duplicate(record: dict[str, Any]) -> dict[str, Any]:
    """Return a reversible autonomous decision for one duplicate candidate."""
    left = str(record.get("left") or "")
    right = str(record.get("right") or "")
    score = float(record.get("score") or 0.0)
    left_title = str(record.get("left_title") or "")
    right_title = str(record.get("right_title") or "")
    decision = {
        "type": "duplicate_decision",
        "ts": _now(),
        "left": left,
        "right": right,
        "score": score,
        "method": record.get("method"),
        "action": "defer",
        "apply": False,
        "reason": "insufficient_confidence",
    }
    if not left or not right or left == right:
        decision["reason"] = "invalid_pair"
        return decision
    if _norm(left_title) != _norm(right_title):
        decision["reason"] = "title_mismatch"
        return decision
    if score < 0.995:
        decision["reason"] = "score_below_auto_supersede_threshold"
        return decision
    left_meta = _page_meta(left)
    right_meta = _page_meta(right)
    if not left_meta or not right_meta:
        decision["reason"] = "missing_page"
        return decision
    if left_meta.get("sensitivity") == "high" or right_meta.get("sensitivity") == "high":
        decision["reason"] = "high_sensitivity_page"
        return decision
    left_q = _page_quality(left, left_meta)
    right_q = _page_quality(right, right_meta)
    if abs(left_q - right_q) < 1.0:
        decision["reason"] = "quality_tie"
        decision["quality"] = {"left": round(left_q, 3), "right": round(right_q, 3)}
        return decision
    winner, loser = (left, right) if left_q > right_q else (right, left)
    decision.update(
        {
            "action": "supersede",
            "apply": True,
            "winner": winner,
            "loser": loser,
            "reason": "exact_title_high_score_quality_gap",
            "quality": {"left": round(left_q, 3), "right": round(right_q, 3)},
        }
    )
    return decision


def resolve_duplicate_candidates(
    records: list[dict[str, Any]],
    *,
    apply: bool = True,
    write: bool = True,
    budget: CycleBudget | None = None,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    applied = 0
    deferred = 0
    for record in records:
        decision = decide_duplicate(record)
        if decision.get("apply") and apply:
            loser_snapshot = _duplicate_page_snapshot(str(decision["loser"]))
            winner_snapshot = _duplicate_page_snapshot(str(decision["winner"]))
            snapshots_ok = (
                loser_snapshot.get("status") == "ok"
                and winner_snapshot.get("status") == "ok"
            )
            if not snapshots_ok:
                result = {"status": "retry", "reason": "page_snapshot_unavailable"}
            else:
                allowed, reason = (
                    budget.consume("mutation") if budget is not None else (True, "ok")
                )
                if not allowed:
                    result = {"status": "deferred", "reason": reason}
                else:
                    result = _soft_supersede_page(
                        loser=str(decision["loser"]),
                        winner=str(decision["winner"]),
                        expected_loser_hash=str(loser_snapshot["content_hash"]),
                        expected_winner_hash=str(winner_snapshot["content_hash"]),
                        decision_at=str(decision["ts"]),
                        autonomy_decision="duplicate_supersede",
                    )
            decision["result"] = result
            if result.get("status") in {"applied", "already_applied"}:
                applied += 1
            else:
                decision["apply"] = False
                decision["action"] = "defer"
                result_reason = str(result.get("reason") or result.get("status"))
                decision["reason"] = (
                    result_reason
                    if result_reason.endswith("_budget_exhausted")
                    else f"apply_failed:{result_reason}"
                )
                deferred += 1
        else:
            deferred += 1
        decisions.append(decision)
        if write:
            _append_jsonl(DECISIONS_FILE, decision)
    payload = {
        "status": "ok",
        "candidates": len(records),
        "applied": applied,
        "deferred": deferred,
        "decisions": decisions[:20],
    }
    if budget is not None:
        payload["budget"] = budget.snapshot()
    return payload


def _canonical_duplicate_record(record: dict[str, Any]) -> dict[str, Any] | None:
    original_left = str(record.get("left") or "").strip()
    original_right = str(record.get("right") or "").strip()
    if not original_left or not original_right or original_left == original_right:
        return None
    left, right = sorted((original_left, original_right))
    title_by_page = {
        original_left: str(record.get("left_title") or ""),
        original_right: str(record.get("right_title") or ""),
    }
    try:
        score = float(record.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "left": left,
        "right": right,
        "left_title": title_by_page.get(left, ""),
        "right_title": title_by_page.get(right, ""),
        "score": score,
        "method": str(record.get("method") or ""),
    }


def _duplicate_page_snapshot(page_id: str, *, excerpt_chars: int = 5000) -> dict[str, Any]:
    path = find_page(page_id)
    if path is None:
        return {
            "page_id": page_id,
            "status": "missing",
            "content_hash": "missing",
            "path": None,
            "meta": {},
            "excerpt": "",
        }
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {
            "page_id": page_id,
            "status": "unreadable",
            "content_hash": "unreadable",
            "path": str(path),
            "meta": {},
            "excerpt": "",
            "error": str(exc),
        }
    meta, body = parse_frontmatter(text)
    excerpt = re.sub(r"\s+", " ", body).strip()[:excerpt_chars]
    return {
        "page_id": page_id,
        "status": "ok",
        "content_hash": hashlib.sha256(raw).hexdigest(),
        "path": str(path),
        "meta": meta,
        "excerpt": excerpt,
    }


def _existing_duplicate_resolution(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any] | None:
    left_meta = left.get("meta") if isinstance(left.get("meta"), dict) else {}
    right_meta = right.get("meta") if isinstance(right.get("meta"), dict) else {}
    if left_meta.get("status") == "deprecated" and left_meta.get("superseded_by") == right["page_id"]:
        return {"decision": "supersede_left", "winner": right["page_id"], "loser": left["page_id"]}
    if right_meta.get("status") == "deprecated" and right_meta.get("superseded_by") == left["page_id"]:
        return {"decision": "supersede_right", "winner": left["page_id"], "loser": right["page_id"]}
    return None


def _rollback_owned_page_write(
    path: Path,
    *,
    expected_written: bytes,
    original: bytes,
) -> bool:
    """Restore ``original`` only while the page still contains our exact write."""
    rollback: Path | None = None
    try:
        if path.read_bytes() != expected_written:
            return False
        rollback = path.with_name(f".{path.name}.{os.getpid()}.rollback.tmp")
        rollback.write_bytes(original)
        os.replace(rollback, path)
        return path.read_bytes() == original
    except OSError:
        return False
    finally:
        if rollback is not None:
            try:
                rollback.unlink()
            except OSError:
                pass


def _soft_supersede_page(
    *,
    loser: str,
    winner: str,
    expected_loser_hash: str,
    expected_winner_hash: str,
    decision_at: str,
    autonomy_decision: str = "duplicate_frontier_supersede",
) -> dict[str, Any]:
    """Soft-supersede one page with content CAS; never delete or merge bodies."""
    loser_path = find_page(loser)
    winner_path = find_page(winner)
    if loser_path is None or winner_path is None:
        return {"status": "retry", "reason": "winner_or_loser_missing"}
    try:
        loser_raw = loser_path.read_bytes()
        winner_raw = winner_path.read_bytes()
        loser_text = loser_raw.decode("utf-8")
        winner_text = winner_raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {"status": "retry", "reason": f"read_error:{exc}"}
    if hashlib.sha256(loser_raw).hexdigest() != expected_loser_hash:
        return {"status": "retry", "reason": "loser_content_changed"}
    if hashlib.sha256(winner_raw).hexdigest() != expected_winner_hash:
        return {"status": "retry", "reason": "winner_content_changed"}

    loser_meta, loser_body = parse_frontmatter(loser_text)
    winner_meta, _winner_body = parse_frontmatter(winner_text)
    if loser_meta.get("status") == "deprecated":
        if loser_meta.get("superseded_by") == winner:
            return {"status": "already_applied", "loser": loser, "winner": winner}
        return {"status": "retry", "reason": "loser_already_superseded_elsewhere"}
    if winner_meta.get("status") in {"deprecated", "archived"}:
        return {"status": "retry", "reason": "winner_is_not_active"}

    updated = patch_frontmatter(
        loser_text,
        {
            "status": "deprecated",
            "superseded_by": winner,
            "autonomy_decision": autonomy_decision,
            "autonomy_decision_at": decision_at,
        },
    )
    _updated_meta, updated_body = parse_frontmatter(updated)
    if updated_body != loser_body:
        return {"status": "retry", "reason": "body_change_refused"}
    updated_raw = updated.encode("utf-8")
    tmp: Path | None = None
    replaced = False
    try:
        if loser_path.read_bytes() != loser_raw or winner_path.read_bytes() != winner_raw:
            return {"status": "retry", "reason": "content_changed_before_apply"}
        tmp = loser_path.with_name(f".{loser_path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(updated_raw)
        os.replace(tmp, loser_path)
        replaced = True
        written_raw = loser_path.read_bytes()
        winner_after = winner_path.read_bytes()
    except OSError as exc:
        rolled_back = (
            _rollback_owned_page_write(
                loser_path,
                expected_written=updated_raw,
                original=loser_raw,
            )
            if replaced
            else False
        )
        return {
            "status": "retry",
            "reason": f"write_error:{exc}",
            "rolled_back": rolled_back,
        }
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass
    try:
        written = written_raw.decode("utf-8")
        written_meta, written_body = parse_frontmatter(written)
    except (UnicodeDecodeError, ValueError) as exc:
        rolled_back = _rollback_owned_page_write(
            loser_path,
            expected_written=updated_raw,
            original=loser_raw,
        )
        return {
            "status": "retry",
            "reason": f"post_write_parse_failed:{exc}",
            "rolled_back": rolled_back,
        }
    verified = (
        written_raw == updated_raw
        and written_body == loser_body
        and written_meta.get("status") == "deprecated"
        and written_meta.get("superseded_by") == winner
        and written_meta.get("autonomy_decision") == autonomy_decision
        and written_meta.get("autonomy_decision_at") == decision_at
        and hashlib.sha256(winner_after).hexdigest() == expected_winner_hash
    )
    if not verified:
        rolled_back = _rollback_owned_page_write(
            loser_path,
            expected_written=updated_raw,
            original=loser_raw,
        )
        return {
            "status": "retry",
            "reason": "post_write_verification_failed",
            "rolled_back": rolled_back,
        }
    return {"status": "applied", "loser": loser, "winner": winner, "path": str(loser_path)}


def _duplicate_frontier_failure_class(review: dict[str, Any]) -> str | None:
    failure = review.get("frontier_failure")
    if isinstance(failure, dict) and isinstance(failure.get("failure_class"), str):
        return failure["failure_class"]
    return None


def _normalize_duplicate_frontier_review(review: object) -> dict[str, Any]:
    def invalid(summary: str) -> dict[str, Any]:
        normalized = {
            "decision": "needs_retry",
            "confidence": 0.0,
            "summary": summary,
            "schema_valid": False,
            "raw_review": review,
        }
        if isinstance(review, dict):
            failure = review.get("frontier_failure")
            if isinstance(failure, dict):
                normalized["frontier_failure"] = failure
            if isinstance(review.get("human_required"), bool):
                normalized["human_required"] = review["human_required"]
        return normalized

    if not isinstance(review, dict):
        return invalid("frontier result is not an object")
    required = set(DUPLICATE_FRONTIER_SCHEMA["required"])
    missing = sorted(required - set(review))
    if missing:
        return invalid(f"frontier result is missing required fields: {', '.join(missing)}")
    decision = review.get("decision")
    summary = review.get("summary")
    confidence = review.get("confidence")
    valid_decisions = set(DUPLICATE_FRONTIER_SCHEMA["properties"]["decision"]["enum"])
    if not isinstance(decision, str) or decision not in valid_decisions:
        return invalid("frontier result has an invalid decision")
    if not isinstance(summary, str):
        return invalid("frontier result is missing a string summary")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return invalid("frontier result confidence is not a finite number in [0, 1]")

    schema_fields = set(DUPLICATE_FRONTIER_SCHEMA["properties"])
    diagnostic_fields = {"frontier_failure", "human_required"}
    allowed_fields = schema_fields | {"reviewer"} | (
        diagnostic_fields if decision == "needs_retry" else set()
    )
    extras = sorted(set(review) - allowed_fields)
    if extras:
        return invalid(f"frontier result has unexpected fields: {', '.join(extras)}")
    return {
        **review,
        "decision": decision,
        "summary": summary,
        "confidence": float(confidence),
        "schema_valid": True,
    }


def _review_deferred_duplicate(
    candidate: dict[str, Any],
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    from llm_wiki_mcp.frontier_review import run_structured_review

    prompt = f"""\
You are the final autonomous duplicate-page judge for LLM Wiki.
The LEFT and RIGHT labels below are canonical and stable. `supersede_left`
means mark LEFT deprecated with `superseded_by: RIGHT`; `supersede_right`
means the reverse. Choose `keep_both` whenever the pages are complementary,
record distinct events, or uncertainty remains. Choose `needs_retry` only when
the evidence is unavailable or malformed. Never request deletion or a body
merge. Do not ask a human. Return JSON matching the supplied schema only.

Candidate:
{json.dumps(candidate, ensure_ascii=False, indent=2)}
"""
    return run_structured_review(
        prompt,
        DUPLICATE_FRONTIER_SCHEMA,
        repo_root=PROJECT_ROOT,
        timeout=timeout,
        execute_patch=False,
        command_env="LLM_WIKI_DUPLICATE_REVIEW_CMD",
    )


def resolve_deferred_duplicates_with_frontier(
    records: list[dict[str, Any]],
    *,
    convergence_store: ConvergenceStore | None = None,
    budget: CycleBudget | None = None,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    timeout: int | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Boundedly converge deterministic duplicate deferrals via a frontier judge."""
    state = convergence_store or ConvergenceStore()
    cycle_budget = budget or CycleBudget(
        max_local_calls=20,
        max_frontier_calls=3,
        max_mutations=3,
        max_elapsed_seconds=900,
    )
    decision_at = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    retired_stale = state.retire_stale(
        lane=DUPLICATE_FRONTIER_LANE,
        reason="duplicate_candidate_expired",
        now=now,
        dry_run=dry_run,
    )
    frontier_remaining = int(cycle_budget.snapshot()["remaining"]["frontier"])
    seen_keys: set[str] = set()
    results: list[dict[str, Any]] = []
    deferred_seen = 0
    frontier_calls = 0
    applied = 0
    kept_both = 0

    for record in records:
        local_decision = decide_duplicate(record)
        deterministic_supersede = (
            local_decision.get("action") == "supersede"
            and bool(local_decision.get("apply"))
        )
        if not deterministic_supersede:
            deferred_seen += 1
        candidate = _canonical_duplicate_record(record)
        if candidate is None:
            results.append({"status": "invalid_pair", "reason": local_decision.get("reason")})
            continue
        left_snapshot = _duplicate_page_snapshot(candidate["left"])
        right_snapshot = _duplicate_page_snapshot(candidate["right"])
        existing = _existing_duplicate_resolution(left_snapshot, right_snapshot)
        if existing is not None:
            results.append({"status": "already_applied", **existing})
            continue
        input_data = {
            "pair": [candidate["left"], candidate["right"]],
            "content_hashes": {
                candidate["left"]: left_snapshot["content_hash"],
                candidate["right"]: right_snapshot["content_hash"],
            },
        }
        merged = state.merge_item(
            lane=DUPLICATE_FRONTIER_LANE,
            source_id=f"{candidate['left']}<->{candidate['right']}",
            input_data=input_data,
            resolver_version=DUPLICATE_FRONTIER_RESOLVER_VERSION,
            metadata={
                "candidate": candidate,
                "local_action": local_decision.get("action"),
                "local_reason": local_decision.get("reason"),
            },
            now=now,
            dry_run=dry_run,
        )
        item = merged["item"]
        key = str(item["key"])
        if key in seen_keys:
            results.append({"status": "duplicate_in_cycle", "key": key})
            continue
        seen_keys.add(key)
        result: dict[str, Any] = {"key": key, "pair": input_data["pair"]}
        if item.get("status") in {"applied", "rejected", "quarantined", "human_required"}:
            results.append({**result, "status": item.get("status"), "cached": True})
            continue
        snapshots_ok = left_snapshot["status"] == "ok" and right_snapshot["status"] == "ok"

        if dry_run:
            if not snapshots_ok:
                results.append({**result, "status": "would_retry", "reason": "page_snapshot_unavailable"})
                continue
            if deterministic_supersede and item.get("status") in {
                "pending_local",
                "local_retry",
                "local_running",
            }:
                if not merged.get("created"):
                    projected = state.claim_attempt(key, "local", now=now, dry_run=True)
                    if not projected["claimed"]:
                        results.append(
                            {
                                **result,
                                "status": projected["item"].get("status"),
                                "reason": projected["reason"],
                            }
                        )
                        continue
                local_allowed, local_reason = cycle_budget.can_consume("local")
                mutation_allowed, mutation_reason = cycle_budget.can_consume("mutation")
                if not local_allowed:
                    results.append({**result, "status": local_reason, "reason": local_reason})
                    continue
                if not mutation_allowed:
                    results.append({**result, "status": mutation_reason, "reason": mutation_reason})
                    continue
                results.append({**result, "status": "would_apply_locally"})
                continue
            if item.get("status") in {"pending_frontier", "frontier_retry", "frontier_running"}:
                projected = state.claim_attempt(key, "frontier", now=now, dry_run=True)
                if not projected["claimed"]:
                    results.append({**result, "status": projected["reason"]})
                    continue
            if frontier_remaining <= 0 or not cycle_budget.can_consume("frontier")[0]:
                results.append({**result, "status": "frontier_budget_exhausted"})
                continue
            frontier_remaining -= 1
            results.append({**result, "status": "would_review"})
            continue

        if deterministic_supersede and item.get("status") in {
            "pending_local",
            "local_retry",
            "local_running",
        }:
            local_allowed, local_reason = cycle_budget.can_consume("local")
            if not local_allowed:
                results.append({**result, "status": local_reason, "reason": local_reason})
                continue
            if snapshots_ok:
                mutation_allowed, mutation_reason = cycle_budget.can_consume("mutation")
                if not mutation_allowed:
                    results.append(
                        {**result, "status": mutation_reason, "reason": mutation_reason}
                    )
                    continue
            claim = state.claim_attempt(key, "local", budget=cycle_budget, now=now)
            if not claim["claimed"]:
                results.append(
                    {
                        **result,
                        "status": claim["item"].get("status"),
                        "reason": claim["reason"],
                    }
                )
                continue
            owner = claim["owner"]
            local_apply: dict[str, Any] | None = None
            if not snapshots_ok:
                transition = state.fail_attempt(
                    key,
                    "local",
                    error="page snapshot unavailable",
                    owner=owner,
                    now=now,
                )
            else:
                mutation_consumed, mutation_reason = cycle_budget.consume("mutation")
                if not mutation_consumed:
                    # A shared budget may be consumed concurrently after the
                    # preflight. Preserve bounded convergence in that rare race.
                    transition = state.fail_attempt(
                        key,
                        "local",
                        error=mutation_reason,
                        owner=owner,
                        now=now,
                    )
                else:
                    snapshots_by_page = {
                        str(left_snapshot["page_id"]): left_snapshot,
                        str(right_snapshot["page_id"]): right_snapshot,
                    }
                    loser_snapshot = snapshots_by_page[str(local_decision["loser"])]
                    winner_snapshot = snapshots_by_page[str(local_decision["winner"])]
                    local_apply = _soft_supersede_page(
                        loser=str(loser_snapshot["page_id"]),
                        winner=str(winner_snapshot["page_id"]),
                        expected_loser_hash=str(loser_snapshot["content_hash"]),
                        expected_winner_hash=str(winner_snapshot["content_hash"]),
                        decision_at=decision_at,
                        autonomy_decision="duplicate_supersede",
                    )
                    if local_apply.get("status") in {"applied", "already_applied"}:
                        transition = state.complete(
                            key,
                            "applied",
                            result={"local": local_decision, "apply": local_apply},
                            owner=owner,
                            now=now,
                        )
                        applied += 1
                    else:
                        transition = state.fail_attempt(
                            key,
                            "local",
                            error=str(local_apply.get("reason") or local_apply.get("status")),
                            owner=owner,
                            now=now,
                        )
            final_status = str(transition["item"]["status"])
            if write:
                _append_jsonl(
                    DECISIONS_FILE,
                    {
                        "type": "duplicate_local_decision",
                        "ts": decision_at,
                        "key": key,
                        "left": candidate["left"],
                        "right": candidate["right"],
                        "decision": local_decision,
                        "result": local_apply,
                        "status": final_status,
                    },
                )
            results.append(
                {
                    **result,
                    "status": final_status,
                    "decision": "deterministic_supersede",
                    "apply": local_apply,
                }
            )
            continue

        if item.get("status") in {"pending_local", "local_retry"}:
            escalated = state.escalate(
                key,
                reason=str(local_decision.get("reason") or "deterministic duplicate defer"),
                now=now,
            )
            item = escalated["item"]
        if item.get("status") not in {"pending_frontier", "frontier_retry", "frontier_running"}:
            results.append({**result, "status": item.get("status") or "not_frontier_pending"})
            continue

        if snapshots_ok:
            claim = state.claim_attempt(key, "frontier", budget=cycle_budget, now=now)
        else:
            allowed, reason = cycle_budget.consume("local")
            if not allowed:
                results.append({**result, "status": reason})
                continue
            claim = state.claim_attempt(key, "frontier", now=now)
        if not claim["claimed"]:
            results.append({**result, "status": claim["item"].get("status"), "reason": claim["reason"]})
            continue
        owner = claim["owner"]
        if not snapshots_ok:
            transition = state.fail_attempt(
                key,
                "frontier",
                error="page snapshot unavailable",
                owner=owner,
                now=now,
            )
            results.append({**result, "status": transition["item"]["status"]})
            continue

        frontier_calls += 1
        review_candidate = {
            **candidate,
            "left_content_hash": left_snapshot["content_hash"],
            "right_content_hash": right_snapshot["content_hash"],
            "left_meta": left_snapshot["meta"],
            "right_meta": right_snapshot["meta"],
            "left_excerpt": left_snapshot["excerpt"],
            "right_excerpt": right_snapshot["excerpt"],
            "local_reason": local_decision.get("reason"),
        }
        try:
            raw_review = (
                reviewer(review_candidate)
                if reviewer is not None
                else _review_deferred_duplicate(review_candidate, timeout=timeout)
            )
        except Exception as exc:
            from llm_wiki_mcp.frontier_review import classify_frontier_failure

            failure = classify_frontier_failure(str(exc)).to_dict()
            raw_review = {
                "decision": "needs_retry",
                "confidence": 0.0,
                "summary": str(exc),
                "frontier_failure": failure,
            }
        review = _normalize_duplicate_frontier_review(raw_review)
        frontier_decision = str(review.get("decision"))
        if (
            frontier_decision in {"supersede_left", "supersede_right"}
            and float(review.get("confidence") or 0.0) < DUPLICATE_SUPERSEDE_MIN_CONFIDENCE
        ):
            original_decision = frontier_decision
            review = {
                **review,
                "decision": "keep_both",
                "original_decision": original_decision,
                "low_confidence_supersede": True,
                "summary": (
                    f"supersede confidence below {DUPLICATE_SUPERSEDE_MIN_CONFIDENCE:.2f}; "
                    "keeping both pages"
                ),
            }
            frontier_decision = "keep_both"
        transition: dict[str, Any]
        apply_result: dict[str, Any] | None = None
        if frontier_decision == "needs_retry":
            transition = state.fail_attempt(
                key,
                "frontier",
                error=str(review.get("summary") or "frontier needs retry"),
                failure_class=_duplicate_frontier_failure_class(review),
                owner=owner,
                now=now,
            )
        elif frontier_decision == "keep_both":
            transition = state.complete(
                key,
                "rejected",
                result={"decision": "keep_both", "frontier": review},
                owner=owner,
                now=now,
            )
            kept_both += 1
        else:
            allowed, reason = cycle_budget.consume("mutation")
            if not allowed:
                transition = state.fail_attempt(
                    key,
                    "frontier",
                    error=reason,
                    owner=owner,
                    now=now,
                )
            else:
                loser_snapshot = left_snapshot if frontier_decision == "supersede_left" else right_snapshot
                winner_snapshot = right_snapshot if frontier_decision == "supersede_left" else left_snapshot
                apply_result = _soft_supersede_page(
                    loser=str(loser_snapshot["page_id"]),
                    winner=str(winner_snapshot["page_id"]),
                    expected_loser_hash=str(loser_snapshot["content_hash"]),
                    expected_winner_hash=str(winner_snapshot["content_hash"]),
                    decision_at=decision_at,
                )
                if apply_result.get("status") in {"applied", "already_applied"}:
                    transition = state.complete(
                        key,
                        "applied",
                        result={"frontier": review, "apply": apply_result},
                        owner=owner,
                        now=now,
                    )
                    applied += 1
                else:
                    transition = state.fail_attempt(
                        key,
                        "frontier",
                        error=str(apply_result.get("reason") or apply_result.get("status")),
                        owner=owner,
                        now=now,
                    )
        final_status = str(transition["item"]["status"])
        audit = {
            "type": "duplicate_frontier_decision",
            "ts": decision_at,
            "key": key,
            "left": candidate["left"],
            "right": candidate["right"],
            "decision": frontier_decision,
            "review": review,
            "result": apply_result,
            "status": final_status,
        }
        if write:
            _append_jsonl(DECISIONS_FILE, audit)
        results.append({**result, "status": final_status, "decision": frontier_decision, "apply": apply_result})

    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "status": "ok",
        "dry_run": dry_run,
        "candidates": len(records),
        "deferred_seen": deferred_seen,
        "unique_items": len(seen_keys),
        "frontier_calls": frontier_calls,
        "applied": applied,
        "kept_both": kept_both,
        "retired": retired_stale.get("retired", []),
        "status_counts": dict(sorted(status_counts.items())),
        "budget": cycle_budget.snapshot(),
        "results": results,
    }


def apply_retention_archives(
    retention_payload: dict[str, Any],
    *,
    apply: bool = True,
    write: bool = True,
    limit: int = 25,
    budget: CycleBudget | None = None,
) -> dict[str, Any]:
    candidates = retention_payload.get("archive_candidates")
    if not isinstance(candidates, list):
        candidates = []
    pages = retention_payload.get("pages")
    pages = pages if isinstance(pages, dict) else {}
    decisions: list[dict[str, Any]] = []
    applied = 0
    actionable_seen = 0
    for page_id in [str(item) for item in candidates if isinstance(item, str)]:
        if not apply and actionable_seen >= max(0, limit):
            break
        if not apply:
            actionable_seen += 1
        row = pages.get(page_id) if isinstance(pages.get(page_id), dict) else {}
        decision = {
            "type": "archive_decision",
            "ts": _now(),
            "page_id": page_id,
            "action": "archive",
            "apply": apply,
            "score": row.get("score"),
            "reason": "retention_archive_candidate",
        }
        if apply:
            snapshot = _duplicate_page_snapshot(page_id)
            if snapshot.get("status") != "ok":
                decision["apply"] = False
                decision["action"] = "defer"
                decision["reason"] = "archive_page_snapshot_unavailable"
                decisions.append(decision)
                if write:
                    _append_jsonl(DECISIONS_FILE, decision)
                continue
            snapshot_meta = snapshot.get("meta") if isinstance(snapshot.get("meta"), dict) else {}
            if snapshot_meta.get("status") == "archived":
                decision["apply"] = False
                decision["action"] = "already_archived"
                decision["reason"] = "retention_archive_already_applied"
                decision["result"] = {"status": "already_applied", "page_id": page_id}
                decisions.append(decision)
                if write:
                    _append_jsonl(DECISIONS_FILE, decision)
                continue
            if actionable_seen >= max(0, limit):
                break
            actionable_seen += 1
            allowed, reason = (
                budget.consume("mutation") if budget is not None else (True, "ok")
            )
            if not allowed:
                decision["apply"] = False
                decision["action"] = "defer"
                decision["reason"] = reason
                decisions.append(decision)
                if write:
                    _append_jsonl(DECISIONS_FILE, decision)
                continue
            result = _patch_page_status(
                page_id,
                {
                    "status": "archived",
                    "autonomy_decision": "retention_archive",
                    "autonomy_decision_at": decision["ts"],
                    "archive_reason": "low_retention_reversible_soft_archive",
                },
                expected_hash=str(snapshot["content_hash"]),
            )
            decision["result"] = result
            if result.get("status") in {"applied", "unchanged"}:
                applied += 1
            else:
                decision["apply"] = False
                decision["action"] = "defer"
                decision["reason"] = f"archive_failed:{result.get('reason', result.get('status'))}"
        decisions.append(decision)
        if write:
            _append_jsonl(DECISIONS_FILE, decision)
    return {
        "status": "ok",
        "candidates": len(candidates),
        "considered": len(decisions),
        "applied": applied,
        "decisions": decisions[:20],
    }


def _capture_rate(health: dict[str, Any]) -> float | None:
    memory = health.get("memory_integrity")
    if not isinstance(memory, dict):
        return None
    value = memory.get("capture_rate")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _queue_value(health: dict[str, Any], key: str) -> int:
    queues = health.get("queues")
    if not isinstance(queues, dict):
        return 0
    try:
        return int(queues.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _watchdog_history_summary(payload: dict[str, Any]) -> dict[str, Any]:
    health = payload.get("health") if isinstance(payload.get("health"), dict) else {}
    latest_sleep = (
        payload.get("latest_sleep")
        if isinstance(payload.get("latest_sleep"), dict)
        else {}
    )
    alerts = payload.get("alerts") if isinstance(payload.get("alerts"), list) else []
    if alerts:
        alert_types = [
            str(alert.get("type") or "unknown")[:200]
            for alert in alerts[:32]
            if isinstance(alert, dict)
        ]
    else:
        existing_types = payload.get("alert_types")
        alert_types = (
            [str(value)[:200] for value in existing_types[:32]]
            if isinstance(existing_types, list)
            else []
        )
    if health:
        capture_rate = _capture_rate(health)
        queues = {
            "duplicate_candidates": _queue_value(health, "duplicate_candidates"),
            "lint_repair": _queue_value(health, "lint_repair"),
        }
    else:
        try:
            capture_rate = float(payload.get("capture_rate"))
        except (TypeError, ValueError):
            capture_rate = None
        existing_queues = payload.get("queues")
        existing_queues = existing_queues if isinstance(existing_queues, dict) else {}
        compact_health = {"queues": existing_queues}
        queues = {
            "duplicate_candidates": _queue_value(compact_health, "duplicate_candidates"),
            "lint_repair": _queue_value(compact_health, "lint_repair"),
        }
    try:
        alert_count = max(0, int(payload.get("alert_count") or 0))
    except (TypeError, ValueError):
        alert_count = 0
    return {
        "ts": str(payload.get("ts"))[:200] if payload.get("ts") is not None else None,
        "status": (
            str(payload.get("status"))[:200]
            if payload.get("status") is not None
            else None
        ),
        "alert_count": len(alerts) if alerts else alert_count,
        "alert_types": alert_types,
        "capture_rate": capture_rate,
        "queues": queues,
        "latest_sleep": _sleep_run_summary(latest_sleep),
    }


def _sleep_run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "status": str(payload.get("status"))[:200] if payload.get("status") is not None else None,
        "started_at": (
            str(payload.get("started_at"))[:200]
            if payload.get("started_at") is not None
            else None
        ),
        "finished_at": (
            str(payload.get("finished_at"))[:200]
            if payload.get("finished_at") is not None
            else None
        ),
        "run_id": str(payload.get("run_id"))[:200] if payload.get("run_id") is not None else None,
    }


def _write_watchdog_history(
    payload: dict[str, Any],
    *,
    path: Path | None = None,
    max_lines: int = 1000,
) -> None:
    target = path or WATCHDOG_HISTORY
    max_lines = max(1, int(max_lines))
    previous: list[dict[str, Any]] = []
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    keep_previous = max_lines - 1
    recent_lines = lines[-keep_previous:] if keep_previous else []
    for line in recent_lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        previous.append(_watchdog_history_summary(row))
    rows = [*previous, _watchdog_history_summary(payload)][-max_lines:]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
            encoding="utf-8",
        )
        os.replace(tmp, target)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def watchdog_snapshot(
    *,
    before_health: dict[str, Any] | None = None,
    write: bool = True,
    notify: bool = False,
    max_sleep_age_hours: float = 30.0,
    min_capture_rate: float = 0.80,
) -> dict[str, Any]:
    from llm_wiki_mcp.health import health_snapshot
    from llm_wiki_mcp.sleep_cycle import HISTORY_FILE

    health = health_snapshot()
    latest_sleep = _latest_jsonl(HISTORY_FILE)
    alerts: list[dict[str, Any]] = []
    capture = _capture_rate(health)
    if capture is not None and capture < min_capture_rate:
        alerts.append({"type": "capture_rate_low", "value": capture, "threshold": min_capture_rate})
    sleep_started = _parse_dt(latest_sleep.get("started_at")) if latest_sleep else None
    if not latest_sleep:
        alerts.append({"type": "sleep_never_ran"})
    elif latest_sleep.get("status") != "ok":
        alerts.append({"type": "sleep_status_not_ok", "status": latest_sleep.get("status")})
    elif sleep_started and datetime.now() - sleep_started > timedelta(hours=max_sleep_age_hours):
        alerts.append({"type": "sleep_stale", "started_at": latest_sleep.get("started_at")})
    if _queue_value(health, "duplicate_candidates") > 500:
        alerts.append({"type": "duplicate_backlog_high", "value": _queue_value(health, "duplicate_candidates")})
    if _queue_value(health, "lint_repair") > 2000:
        alerts.append({"type": "lint_backlog_high", "value": _queue_value(health, "lint_repair")})

    if before_health:
        before_capture = _capture_rate(before_health)
        if before_capture is not None and capture is not None and capture + 0.15 < before_capture:
            alerts.append({"type": "capture_rate_regression", "before": before_capture, "after": capture})

    payload = {
        "status": "alert" if alerts else "ok",
        "ts": _now(),
        "alerts": alerts,
        "health": health,
        "latest_sleep": _sleep_run_summary(latest_sleep),
    }
    if write:
        _write_json(WATCHDOG_FILE, payload)
        _write_watchdog_history(payload)
    if notify and alerts:
        _send_notification("LLM Wiki watchdog", f"{len(alerts)} autonomy alert(s)")
    return payload


def _send_notification(title: str, body: str) -> dict[str, Any]:
    script = f"display notification {json.dumps(body)} with title {json.dumps(title)}"
    try:
        proc = subprocess.run(["osascript", "-e", script], text=True, capture_output=True, timeout=5)
    except Exception as exc:
        return {"sent": False, "error": str(exc)}
    return {"sent": proc.returncode == 0, "stderr": proc.stderr.strip()}


def regression_guard(
    *,
    before_health: dict[str, Any],
    after_watchdog: dict[str, Any],
    wiki_snapshot: dict[str, Any],
    auto_revert: bool,
    write: bool = True,
) -> dict[str, Any]:
    alerts = [
        item for item in after_watchdog.get("alerts", [])
        if isinstance(item, dict) and item.get("type") in {"capture_rate_regression"}
    ]
    payload = {
        "status": "ok" if not alerts else "regression",
        "ts": _now(),
        "alerts": alerts,
        "auto_revert": auto_revert,
        "auto_revert_effective": False,
        "reverted": False,
        "rollback_scope": "mutation_cas_only",
        "global_reset_disabled": True,
    }
    head = str(wiki_snapshot.get("head") or "")
    if alerts:
        quarantine_action = {
            "ts": payload["ts"],
            "reason": "capture_rate_regression",
            "scope": "cycle",
            "snapshot_head": head or None,
            "note": "Global git reset is disabled; each mutation owns its CAS rollback.",
        }
        payload["quarantine"] = quarantine_action
        if write:
            quarantine = _read_json(QUARANTINE_FILE)
            quarantined = quarantine.get("actions")
            if not isinstance(quarantined, list):
                quarantined = []
            quarantined.append(quarantine_action)
            _write_json(QUARANTINE_FILE, {"actions": quarantined[-100:]})
    if write:
        _append_jsonl(DECISIONS_FILE, {"type": "regression_guard", **payload})
    return payload


def build_digest(payload: dict[str, Any], *, write: bool = True) -> dict[str, Any]:
    watchdog = payload.get("watchdog") if isinstance(payload.get("watchdog"), dict) else {}
    duplicate = payload.get("duplicates") if isinstance(payload.get("duplicates"), dict) else {}
    retention = payload.get("archives") if isinstance(payload.get("archives"), dict) else {}
    guard = payload.get("regression_guard") if isinstance(payload.get("regression_guard"), dict) else {}
    lines = [
        "# LLM Wiki Autonomy Digest",
        "",
        f"- Generated: {_now()}",
        f"- Status: {watchdog.get('status', payload.get('status', 'unknown'))}",
        f"- Alerts: {len(watchdog.get('alerts', []) or [])}",
        f"- Duplicate decisions: {duplicate.get('applied', 0)} applied, {duplicate.get('deferred', 0)} deferred",
        f"- Retention archives: {retention.get('applied', 0)} applied",
        f"- Regression guard: {guard.get('status', 'unknown')}",
    ]
    if watchdog.get("alerts"):
        lines.extend(["", "## Alerts"])
        for alert in watchdog["alerts"]:
            lines.append(f"- `{alert.get('type', 'unknown')}` {json.dumps(alert, ensure_ascii=False)}")
    text = "\n".join(lines).rstrip() + "\n"
    if write:
        DIGEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        DIGEST_FILE.write_text(text, encoding="utf-8")
    return {"status": "ok", "path": str(DIGEST_FILE), "text": text}


def run_autonomy_cycle(
    *,
    duplicates: list[dict[str, Any]],
    retention: dict[str, Any],
    before_health: dict[str, Any],
    wiki_snapshot: dict[str, Any],
    dry_run: bool = False,
    auto_revert: bool = True,
    budget: CycleBudget | None = None,
    retention_budget: CycleBudget | None = None,
) -> dict[str, Any]:
    duplicate_result = resolve_duplicate_candidates(
        duplicates,
        apply=not dry_run,
        write=not dry_run,
        budget=budget,
    )
    archive_result = apply_retention_archives(
        retention,
        apply=not dry_run,
        write=not dry_run,
        budget=retention_budget or budget,
    )
    watchdog = watchdog_snapshot(before_health=before_health, write=not dry_run)
    guard = regression_guard(
        before_health=before_health,
        after_watchdog=watchdog,
        wiki_snapshot=wiki_snapshot,
        auto_revert=(not dry_run and auto_revert),
        write=not dry_run,
    )
    payload = {
        "status": "ok" if watchdog.get("status") == "ok" and guard.get("status") == "ok" else "attention",
        "ts": _now(),
        "dry_run": dry_run,
        "duplicates": duplicate_result,
        "archives": archive_result,
        "watchdog": watchdog,
        "regression_guard": guard,
    }
    digest = build_digest(payload, write=not dry_run)
    payload["digest"] = {k: v for k, v in digest.items() if k != "text"}
    if not dry_run:
        _write_json(LATEST_FILE, payload)
    return payload


def _uv_path() -> str:
    return shutil.which("uv") or "/opt/homebrew/bin/uv"


def _plist(label: str, args: list[str], *, stdout: Path, stderr: Path, start_interval: int | None = None, calendar: dict[str, int] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "Label": label,
        "ProgramArguments": args,
        "WorkingDirectory": str(PROJECT_ROOT),
        "StandardOutPath": str(stdout),
        "StandardErrorPath": str(stderr),
        "RunAtLoad": False,
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONUNBUFFERED": "1",
        },
    }
    if start_interval is not None:
        data["StartInterval"] = start_interval
    if calendar is not None:
        data["StartCalendarInterval"] = calendar
    return data


def _write_plist(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    plistlib.dump(data, buf, sort_keys=False)
    path.write_bytes(buf.getvalue())


def _write_wrapper(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = "#!/bin/sh\nexec " + shlex.join(command) + "\n"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def install_launchd(*, dry_run: bool = False, load: bool = False) -> dict[str, Any]:
    logs = WIKI_ROOT / "logs"
    uv = _uv_path()
    sleep_path = LAUNCH_AGENT_DIR / f"{SLEEP_LABEL}.plist"
    watchdog_path = LAUNCH_AGENT_DIR / f"{WATCHDOG_LABEL}.plist"
    sleep_wrapper = WRAPPER_DIR / "llm-wiki-sleep"
    watchdog_wrapper = WRAPPER_DIR / "llm-wiki-watchdog"
    sleep_command = [
        uv,
        "run",
        "--project",
        str(PROJECT_ROOT),
        "llm-wiki",
        "sleep",
        "--raw-limit",
        "200",
        "--eval-limit",
        "150",
        "--duplicate-limit",
        "300",
    ]
    watchdog_command = [
        uv,
        "run",
        "--project",
        str(PROJECT_ROOT),
        "llm-wiki",
        "autonomy",
        "watchdog",
        "--notify",
        "--json",
    ]
    sleep_plist = _plist(
        SLEEP_LABEL,
        [str(sleep_wrapper)],
        stdout=logs / "sleep-cycle.launchd.out.log",
        stderr=logs / "sleep-cycle.launchd.err.log",
        calendar={"Hour": 3, "Minute": 40},
    )
    watchdog_plist = _plist(
        WATCHDOG_LABEL,
        [str(watchdog_wrapper)],
        stdout=Path(os.devnull),
        stderr=logs / "watchdog.launchd.err.log",
        start_interval=900,
    )
    payload: dict[str, Any] = {
        "status": "ok",
        "dry_run": dry_run,
        "load": load,
        "plists": [
            {
                "label": SLEEP_LABEL,
                "path": str(sleep_path),
                "program": sleep_plist["ProgramArguments"],
                "stdout": sleep_plist["StandardOutPath"],
            },
            {
                "label": WATCHDOG_LABEL,
                "path": str(watchdog_path),
                "program": watchdog_plist["ProgramArguments"],
                "stdout": watchdog_plist["StandardOutPath"],
            },
        ],
        "wrappers": [
            {"path": str(sleep_wrapper), "command": sleep_command},
            {"path": str(watchdog_wrapper), "command": watchdog_command},
        ],
    }
    if not dry_run:
        logs.mkdir(parents=True, exist_ok=True)
        _write_wrapper(sleep_wrapper, sleep_command)
        _write_wrapper(watchdog_wrapper, watchdog_command)
        _write_plist(sleep_path, sleep_plist)
        _write_plist(watchdog_path, watchdog_plist)
    if load and not dry_run:
        uid = os.getuid()
        loads = []
        for path in (sleep_path, watchdog_path):
            subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(path)], text=True, capture_output=True)
            proc = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(path)], text=True, capture_output=True)
            loads.append({"path": str(path), "returncode": proc.returncode, "stderr": proc.stderr.strip()})
        payload["launchctl"] = loads
        if any(item["returncode"] != 0 for item in loads):
            payload["status"] = "error"
    return payload


def status() -> dict[str, Any]:
    return {
        "status": "ok",
        "latest": _read_json(LATEST_FILE),
        "watchdog": _read_json(WATCHDOG_FILE),
        "digest_path": str(DIGEST_FILE),
        "launchd": {
            "sleep": str(LAUNCH_AGENT_DIR / f"{SLEEP_LABEL}.plist"),
            "watchdog": str(LAUNCH_AGENT_DIR / f"{WATCHDOG_LABEL}.plist"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or install LLM Wiki autonomous operation.")
    sub = parser.add_subparsers(dest="command", required=True)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--json", action="store_true")
    watchdog_parser = sub.add_parser("watchdog")
    watchdog_parser.add_argument("--notify", action="store_true")
    watchdog_parser.add_argument("--json", action="store_true")
    digest_parser = sub.add_parser("digest")
    digest_parser.add_argument("--json", action="store_true")
    install_parser = sub.add_parser("install-launchd")
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--load", action="store_true")
    install_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "status":
        payload = status()
    elif args.command == "watchdog":
        payload = watchdog_snapshot(notify=args.notify)
    elif args.command == "digest":
        payload = build_digest(_read_json(LATEST_FILE))
    else:
        payload = install_launchd(dry_run=args.dry_run, load=args.load)
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print("\t".join(f"{key}={value}" for key, value in payload.items() if key != "latest"))
    return 0 if payload.get("status") in {"ok", "alert"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
