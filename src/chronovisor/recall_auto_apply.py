"""Apply safe auto-lane recall improvements from missed candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.canonical_json import (
    canonical_json_sha256_stringifying as _canonical_json_sha256,
)

import tomllib

from chronovisor import decision_authority, store as chronovisor_store
from chronovisor.alias_store import add_alias, load_aliases
from chronovisor.convergence import is_human_required_result
from chronovisor.frontmatter import parse as parse_frontmatter
from chronovisor.frontmatter import patch as patch_frontmatter
from chronovisor.link_fix import atomic_write
from chronovisor.jsonl import read_jsonl
from chronovisor.page_mutation import decision_authority_lock, chronovisor_mutation_lock
from chronovisor.recall_hints import (
    add_query_hint,
    load_query_hints,
    normalize_query_text,
)
from chronovisor.recall_runtime import (
    RECALL_CONFIG_FILE,
    RECALL_DIR,
    RECALL_FEEDBACK_FILE,
    append_jsonl,
)
from chronovisor.runtime_config import active_config_file
from chronovisor.semantic_hold import (
    LOCAL_SEMANTIC_NO_QUORUM,
    build_semantic_no_quorum_hold,
    canonical_sha256,
    frontier_failure_class,
    is_local_semantic_no_quorum,
    persisted_semantic_no_quorum_hold,
    semantic_no_quorum_hold_error,
)
from chronovisor.tags import record_new_tag, validate_tag


AUTO_ACTIONS = frozenset({"alias", "query_hint", "page_tag"})
REVIEW_ACTIONS = frozenset({"few_shot", "threshold"})
VALIDATED_AUTO_LANE = "validated-auto"
AUTO_APPLY_LOG_FILE = RECALL_DIR / "auto-apply.jsonl"
AUTO_APPLY_REVIEW_DIR = RECALL_DIR / "auto-apply-reviews"
TERMINAL_SUCCESS_STATUSES = frozenset(
    {"applied", "already_applied", "fallback_applied", "routed_to_recall_lab"}
)
TERMINAL_CONVERGENCE_STATUSES = frozenset(
    {"applied", "rejected", "quarantined", "human_required"}
)
DEFAULT_QUARANTINE_COOLDOWN_SECONDS = 6 * 60 * 60
AUTO_APPLY_REVIEW_SCHEMA_VERSION = 2
AUTO_APPLY_DECISION_LANE = "recall_auto_apply"


@dataclass(frozen=True)
class AutoApplyPolicy:
    enabled: bool = True
    min_count: int = 1
    actions: tuple[str, ...] = ("alias", "query_hint", "page_tag")


def load_auto_apply_policy(path: Path = RECALL_CONFIG_FILE) -> AutoApplyPolicy:
    policy = AutoApplyPolicy()
    path = active_config_file(path)
    if not path.exists():
        return policy
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return policy
    section = data.get("auto_apply")
    if not isinstance(section, dict):
        return policy
    values = dict(policy.__dict__)
    if isinstance(section.get("enabled"), bool):
        values["enabled"] = section["enabled"]
    if isinstance(section.get("min_count"), int):
        values["min_count"] = max(1, section["min_count"])
    if isinstance(section.get("actions"), list) and all(
        isinstance(v, str) for v in section["actions"]
    ):
        values["actions"] = tuple(
            action for action in section["actions"] if action in AUTO_ACTIONS
        )
    return AutoApplyPolicy(**values)


def _feedback_file(path: Path | None = None) -> Path:
    if path is not None:
        return path
    from chronovisor import recall_runtime

    return recall_runtime.RECALL_FEEDBACK_FILE


def _auto_apply_log_file(path: Path | None = None) -> Path:
    return path or AUTO_APPLY_LOG_FILE


def read_applied_keys(path: Path | None = None, limit: int = 0) -> set[str]:
    path = _auto_apply_log_file(path)
    keys: set[str] = set()
    try:
        with path.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=limit) if limit > 0 else f
            for line in lines:
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(parsed, dict)
                    and isinstance(parsed.get("apply_key"), str)
                    and (
                        parsed.get("status") in TERMINAL_SUCCESS_STATUSES
                        or parsed.get("convergence_status") == "applied"
                    )
                ):
                    keys.add(parsed["apply_key"])
    except OSError:
        return set()
    return keys


def read_apply_states(
    path: Path | None = None, limit: int = 0
) -> dict[str, dict[str, Any]]:
    """Return the latest convergence record per apply key."""
    states: dict[str, dict[str, Any]] = {}
    try:
        with _auto_apply_log_file(path).open(encoding="utf-8") as handle:
            lines = deque(handle, maxlen=limit) if limit > 0 else handle
            for line in lines:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                key = record.get("apply_key")
                if isinstance(key, str) and key:
                    states[key] = record
    except OSError:
        return {}
    return states


def read_apply_histories(
    path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(_auto_apply_log_file(path)):
        key = record.get("apply_key") if isinstance(record, dict) else None
        if isinstance(key, str) and key:
            histories[key].append(record)
    return dict(histories)


def _parsed_state_time(value: object, *, now: datetime) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None and now.tzinfo is None:
        return parsed.replace(tzinfo=None)
    if parsed.tzinfo is None and now.tzinfo is not None:
        return parsed.replace(tzinfo=now.tzinfo)
    return parsed


def _quarantine_retry_ready(
    state: dict[str, Any] | None,
    *,
    now: datetime,
    cooldown_seconds: int,
) -> bool:
    if not state or str(state.get("convergence_status") or "") != "quarantined":
        return False
    retry_at = _parsed_state_time(state.get("quarantine_retry_at"), now=now)
    if retry_at is not None:
        return retry_at <= now
    quarantined_at = _parsed_state_time(
        state.get("quarantined_at") or state.get("ts"),
        now=now,
    )
    if quarantined_at is None:
        return True
    return quarantined_at + timedelta(seconds=max(0, cooldown_seconds)) <= now


def _retry_ready(
    state: dict[str, Any] | None,
    *,
    now: datetime,
    quarantine_cooldown_seconds: int = DEFAULT_QUARANTINE_COOLDOWN_SECONDS,
) -> bool:
    if not state:
        return True
    convergence_status = str(state.get("convergence_status") or "")
    if convergence_status == "quarantined":
        return _quarantine_retry_ready(
            state,
            now=now,
            cooldown_seconds=quarantine_cooldown_seconds,
        )
    if convergence_status in TERMINAL_CONVERGENCE_STATUSES:
        return False
    raw = state.get("next_attempt_at")
    if not isinstance(raw, str) or not raw:
        return True
    parsed = _parsed_state_time(raw, now=now)
    return parsed is None or parsed <= now


def record_apply_log(record: dict[str, Any], path: Path | None = None) -> None:
    path = _auto_apply_log_file(path)
    append_jsonl(path, record)


def action_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("action_payload")
    return payload if isinstance(payload, dict) else {}


def expected_pages(record: dict[str, Any]) -> list[str]:
    pages = record.get("expected_pages")
    if not isinstance(pages, list):
        return []
    return [page for page in pages if isinstance(page, str) and page]


def apply_key_for(record: dict[str, Any]) -> str:
    action = str(record.get("action_type", ""))
    normalize_key = str(record.get("normalize_key", ""))
    page = (expected_pages(record) or [""])[0]
    payload = action_payload(record)
    payload_key = (
        payload.get("alias")
        or payload.get("tag")
        or payload.get("query")
        or record.get("missing_signal")
        or ""
    )
    return f"{action}:{normalize_key}:{page}:{payload_key}"


def _resolved_review_dir(log_file: Path | None, review_dir: Path | None) -> Path:
    if review_dir is not None:
        return review_dir
    resolved_log = _auto_apply_log_file(log_file)
    return resolved_log.parent / f"{resolved_log.stem}-reviews"


def _review_artifact_path(
    review_dir: Path, *, apply_key: str, proposal_sha256: str
) -> Path:
    key_sha256 = hashlib.sha256(apply_key.encode("utf-8")).hexdigest()
    return review_dir / f"{key_sha256[:24]}-{proposal_sha256[:24]}.json"


def _target_page_id(record: dict[str, Any], *, effective_action: str) -> str:
    payload = action_payload(record)
    pages = expected_pages(record)
    if effective_action == "alias":
        return str(
            payload.get("target_page")
            or payload.get("page_id")
            or (pages[0] if pages else "")
        ).strip()
    return str(payload.get("page_id") or (pages[0] if pages else "")).strip()


def _bounded_page_evidence(page_id: str, *, max_chars: int = 4_000) -> dict[str, Any]:
    if not page_id:
        return {"page_id": "", "exists": False, "sha256": "", "content": ""}
    path = chronovisor_store.find_page(page_id)
    if path is None:
        return {"page_id": page_id, "exists": False, "sha256": "", "content": ""}
    try:
        raw = path.read_bytes()
        content = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return {"page_id": page_id, "exists": False, "sha256": "", "content": ""}
    if len(content) > max_chars:
        half = max(1, max_chars // 2)
        content = (
            content[:half] + "\n\n[... bounded page evidence ...]\n\n" + content[-half:]
        )
    return {
        "page_id": page_id,
        "exists": True,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content": content,
    }


def _frontier_action_proposal(
    record: dict[str, Any],
    *,
    apply_key: str,
) -> dict[str, Any]:
    preview = apply_record(record, dry_run=True)
    effective_action = str(record.get("action_type") or "")
    if preview.get("status") == "fallback_dry_run":
        effective_action = str(preview.get("fallback_to") or effective_action)
    page_id = _target_page_id(record, effective_action=effective_action)
    return {
        "schema_version": 1,
        "apply_key": apply_key,
        "action_type": str(record.get("action_type") or ""),
        "effective_action": effective_action,
        "normalize_key": str(record.get("normalize_key") or ""),
        "source_ref": str(record.get("ref") or ""),
        "expected_pages": expected_pages(record),
        "action_payload": action_payload(record),
        "missing_signal": str(record.get("missing_signal") or "")[:2_000],
        "prompt": str(record.get("prompt") or "")[:4_000],
        "local_validation": preview,
        "page_evidence": _bounded_page_evidence(page_id),
    }


def _proposal_requires_mutation(proposal: dict[str, Any]) -> bool:
    preview = proposal.get("local_validation")
    if not isinstance(preview, dict):
        return False
    return str(preview.get("status") or "") in {"dry_run", "fallback_dry_run"}


def _has_semantic_no_quorum_marker(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        "semantic_hold" in value
        or value.get("last_failure_class") == LOCAL_SEMANTIC_NO_QUORUM
        or frontier_failure_class(value) == LOCAL_SEMANTIC_NO_QUORUM
    )


def _auto_apply_semantic_epoch(
    *, apply_key: str, proposal_sha256: str
) -> dict[str, Any]:
    return {
        "review_schema_version": AUTO_APPLY_REVIEW_SCHEMA_VERSION,
        "apply_key_sha256": hashlib.sha256(apply_key.encode("utf-8")).hexdigest(),
        "proposal_sha256": proposal_sha256,
    }


def _semantic_recheck_sha256(epoch: dict[str, Any], authority: dict[str, Any]) -> str:
    return canonical_sha256({"epoch": epoch, "authority": authority})


def _write_semantic_hold_artifact(
    path: Path,
    *,
    apply_key: str,
    proposal_sha256: str,
    hold: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        path,
        json.dumps(
            {
                "schema_version": AUTO_APPLY_REVIEW_SCHEMA_VERSION,
                "kind": "recall_auto_apply_semantic_hold",
                "apply_key_sha256": hashlib.sha256(
                    apply_key.encode("utf-8")
                ).hexdigest(),
                "proposal_sha256": proposal_sha256,
                "semantic_hold": hold,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _semantic_hold_artifact_state(
    path: Path,
    *,
    apply_key: str,
    proposal_sha256: str,
    epoch: dict[str, Any],
    authority: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "none", None
    if not isinstance(payload, dict) or (
        payload.get("kind") != "recall_auto_apply_semantic_hold"
        and "semantic_hold" not in payload
    ):
        return "none", None
    if (
        payload.get("apply_key_sha256")
        != hashlib.sha256(apply_key.encode("utf-8")).hexdigest()
        or payload.get("proposal_sha256") != proposal_sha256
    ):
        return "malformed", None
    hold = persisted_semantic_no_quorum_hold(payload, lane=AUTO_APPLY_DECISION_LANE)
    if hold is None:
        return "malformed", None
    error = semantic_no_quorum_hold_error(
        hold,
        AUTO_APPLY_DECISION_LANE,
        epoch=epoch,
        authority=authority,
    )
    if error is None:
        return "same", hold
    if error in {"semantic hold epoch changed", "semantic hold authority changed"}:
        return "changed", hold
    return "malformed", None


def review_auto_apply_with_frontier(
    proposal: dict[str, Any],
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Ask the frontier model for the authoritative action verdict."""
    from chronovisor import frontier_review
    from chronovisor.decision_lane_prompts import build_recall_auto_apply_prompt

    prompt = build_recall_auto_apply_prompt(proposal)
    timeout_seconds = timeout or int(
        os.environ.get("CHRONOVISOR_RECALL_AUTO_APPLY_FRONTIER_TIMEOUT", "1800")
    )
    return frontier_review.run_structured_review(
        prompt,
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=Path(__file__).resolve().parents[2],
        timeout=timeout_seconds,
        execute_patch=False,
        decision_lane="recall_auto_apply",
    )


def _current_review_authority(
    *, reviewer: Any | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve the shared, complete authority epoch for this lane."""

    return decision_authority.current_semantic_authority(
        AUTO_APPLY_DECISION_LANE,
        injected_reviewer=reviewer is not None,
    )


def _review_authority_error(
    *,
    expected_authority: object,
    review: object | None,
    reviewer: Any | None,
    require_verdict_proof: bool,
) -> str | None:
    """Re-resolve and validate one review while the authority lease is held."""

    current_authority, authority_error = _current_review_authority(reviewer=reviewer)
    error = authority_error or decision_authority.compare_semantic_authority(
        expected_authority,
        current_authority,
        lane=AUTO_APPLY_DECISION_LANE,
    )
    if error is not None or not require_verdict_proof:
        return error
    return decision_authority.semantic_verdict_authority_error(
        review,
        expected_authority,
        lane=AUTO_APPLY_DECISION_LANE,
    )


def _load_review_artifact(
    path: Path,
    *,
    apply_key: str,
    proposal_sha256: str,
    proposal: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema_version") != AUTO_APPLY_REVIEW_SCHEMA_VERSION
        or payload.get("kind") != "recall_auto_apply_frontier_verdict"
        or payload.get("apply_key") != apply_key
        or payload.get("proposal_sha256") != proposal_sha256
        or payload.get("proposal") != proposal
        or _canonical_json_sha256(payload.get("proposal")) != proposal_sha256
        or decision_authority.compare_semantic_authority(
            payload.get("authority"),
            authority,
            lane=AUTO_APPLY_DECISION_LANE,
        )
        is not None
    ):
        return None
    review = payload.get("review")
    if not isinstance(review, dict) or review.get("decision") not in {
        "approved",
        "rejected",
    }:
        return None
    if (
        decision_authority.semantic_verdict_authority_error(
            review,
            payload.get("authority"),
            lane=AUTO_APPLY_DECISION_LANE,
        )
        is not None
    ):
        return None
    return payload


def _write_review_artifact(
    path: Path,
    *,
    apply_key: str,
    proposal_sha256: str,
    proposal: dict[str, Any],
    authority: dict[str, Any],
    review: dict[str, Any],
    now: datetime,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = decision_authority.seal_semantic_artifact(
        {
            "schema_version": AUTO_APPLY_REVIEW_SCHEMA_VERSION,
            "kind": "recall_auto_apply_frontier_verdict",
            "apply_key": apply_key,
            "proposal_sha256": proposal_sha256,
            "proposal": proposal,
            "review": review,
            "created_at": now.isoformat(timespec="seconds"),
        },
        authority=authority,
        lane=AUTO_APPLY_DECISION_LANE,
    )
    atomic_write(
        path,
        json.dumps(
            envelope,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
    )


def _frontier_gate(
    record: dict[str, Any],
    *,
    apply_key: str,
    review_dir: Path,
    reviewer: Any | None,
    timeout: int | None,
    budget: Any | None,
    now: datetime,
    prior: dict[str, Any] | None = None,
    prior_history: list[dict[str, Any]] | None = None,
    quarantine_cooldown_seconds: int = DEFAULT_QUARANTINE_COOLDOWN_SECONDS,
) -> dict[str, Any]:
    authority, authority_error = _current_review_authority(reviewer=reviewer)
    if authority is None or authority_error is not None:
        return {
            "status": "needs_retry",
            "review": {
                "decision": "needs_retry",
                "summary": authority_error or "review authority is unavailable",
            },
        }
    try:
        proposal = _frontier_action_proposal(record, apply_key=apply_key)
    except Exception as exc:
        return {
            "status": "proposal_error",
            "authority": authority,
            "result": {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"},
        }
    if not _proposal_requires_mutation(proposal):
        preview = dict(proposal.get("local_validation") or {})
        return {
            "status": "no_mutation",
            "proposal": proposal,
            "result": preview,
            "authority": authority,
        }

    proposal_sha256 = _canonical_json_sha256(proposal)
    semantic_epoch = _auto_apply_semantic_epoch(
        apply_key=apply_key, proposal_sha256=proposal_sha256
    )
    if _has_semantic_no_quorum_marker(prior):
        persisted_hold = persisted_semantic_no_quorum_hold(
            prior, lane=AUTO_APPLY_DECISION_LANE
        )
        if persisted_hold is None:
            return {
                "status": "semantic_hold_malformed",
                "proposal_sha256": proposal_sha256,
                "authority": authority,
            }
        hold_error = semantic_no_quorum_hold_error(
            persisted_hold,
            AUTO_APPLY_DECISION_LANE,
            epoch=semantic_epoch,
            authority=authority,
        )
        if hold_error is None:
            return {
                "status": "semantic_hold_reused",
                "proposal_sha256": proposal_sha256,
                "semantic_hold": persisted_hold,
                "authority": authority,
            }
        if hold_error not in {
            "semantic hold epoch changed",
            "semantic hold authority changed",
        }:
            return {
                "status": "semantic_hold_malformed",
                "proposal_sha256": proposal_sha256,
                "authority": authority,
            }
        for historical_state in reversed(prior_history or []):
            historical_hold = persisted_semantic_no_quorum_hold(
                historical_state,
                lane=AUTO_APPLY_DECISION_LANE,
                epoch=semantic_epoch,
                authority=authority,
            )
            if historical_hold is not None:
                return {
                    "status": "semantic_hold_reused",
                    "proposal_sha256": proposal_sha256,
                    "semantic_hold": historical_hold,
                    "authority": authority,
                }
        recheck_sha256 = _semantic_recheck_sha256(semantic_epoch, authority)
        if (
            isinstance(prior, dict)
            and prior.get("semantic_hold_recheck_sha256") == recheck_sha256
            and not _retry_ready(
                prior,
                now=now,
                quarantine_cooldown_seconds=quarantine_cooldown_seconds,
            )
        ):
            return {
                "status": "retry_deferred",
                "proposal_sha256": proposal_sha256,
                "semantic_hold": persisted_hold,
                "semantic_hold_recheck_sha256": recheck_sha256,
                "authority": authority,
            }
    artifact_path = _review_artifact_path(
        review_dir,
        apply_key=apply_key,
        proposal_sha256=proposal_sha256,
    )
    artifact_hold_state, artifact_hold = _semantic_hold_artifact_state(
        artifact_path,
        apply_key=apply_key,
        proposal_sha256=proposal_sha256,
        epoch=semantic_epoch,
        authority=authority,
    )
    if artifact_hold_state == "same":
        return {
            "status": "semantic_hold_reused",
            "proposal_sha256": proposal_sha256,
            "semantic_hold": artifact_hold,
            "artifact_path": str(artifact_path),
            "authority": authority,
        }
    if artifact_hold_state == "malformed":
        return {
            "status": "semantic_hold_malformed",
            "proposal_sha256": proposal_sha256,
            "artifact_path": str(artifact_path),
            "authority": authority,
        }
    artifact = _load_review_artifact(
        artifact_path,
        apply_key=apply_key,
        proposal_sha256=proposal_sha256,
        proposal=proposal,
        authority=authority,
    )
    if artifact is not None:
        return {
            "status": str(artifact["review"]["decision"]),
            "proposal": proposal,
            "proposal_sha256": proposal_sha256,
            "review": artifact["review"],
            "artifact_path": str(artifact_path),
            "artifact_reused": True,
            "authority": authority,
        }

    if budget is not None:
        allowed, reason = budget.consume("frontier")
        if not allowed:
            return {
                "status": "budget_deferred",
                "reason": reason,
                "proposal": proposal,
                "proposal_sha256": proposal_sha256,
                "artifact_path": str(artifact_path),
            }
    try:
        review = (
            review_auto_apply_with_frontier(proposal, timeout=timeout)
            if reviewer is None
            else reviewer(proposal)
        )
    except Exception as exc:
        return {
            "status": "needs_retry",
            "proposal": proposal,
            "proposal_sha256": proposal_sha256,
            "review": {
                "decision": "needs_retry",
                "summary": f"frontier reviewer failed: {exc.__class__.__name__}: {exc}",
            },
            "artifact_path": str(artifact_path),
            "authority": authority,
        }
    if not isinstance(review, dict):
        review = {
            "decision": "needs_retry",
            "summary": "frontier reviewer returned invalid payload",
        }
    if is_local_semantic_no_quorum(review):
        try:
            with decision_authority_lock():
                authority_error = _review_authority_error(
                    expected_authority=authority,
                    review=None,
                    reviewer=reviewer,
                    require_verdict_proof=False,
                )
                current_proposal = _frontier_action_proposal(
                    record, apply_key=apply_key
                )
                if (
                    authority_error is None
                    and _canonical_json_sha256(current_proposal) != proposal_sha256
                ):
                    authority_error = (
                        "recall action proposal changed before semantic hold"
                    )
                if authority_error is not None:
                    raise ValueError(authority_error)
                hold = build_semantic_no_quorum_hold(
                    AUTO_APPLY_DECISION_LANE,
                    semantic_epoch,
                    authority,
                    review,
                )
                _write_semantic_hold_artifact(
                    artifact_path,
                    apply_key=apply_key,
                    proposal_sha256=proposal_sha256,
                    hold=hold,
                )
        except (OSError, TypeError, ValueError) as exc:
            return {
                "status": "needs_retry",
                "proposal_sha256": proposal_sha256,
                "result": {"status": "error", "error": str(exc)},
                "authority": authority,
            }
        return {
            "status": "semantic_hold",
            "proposal_sha256": proposal_sha256,
            "semantic_hold": hold,
            "artifact_path": str(artifact_path),
            "authority": authority,
        }
    decision = str(review.get("decision") or "needs_retry")
    if decision not in {"approved", "rejected", "quarantined", "needs_retry"}:
        decision = "needs_retry"
        review = {
            **review,
            "decision": decision,
            "summary": "invalid frontier decision",
        }
    if decision in {"approved", "rejected"}:
        try:
            # Re-resolve after the model response and hold the shared lease
            # through artifact persistence. This prevents a just-retired
            # model triplet or lane contract from minting a durable verdict.
            with decision_authority_lock():
                verdict_authority_error = _review_authority_error(
                    expected_authority=authority,
                    review=review,
                    reviewer=reviewer,
                    require_verdict_proof=True,
                )
                if verdict_authority_error is not None:
                    return {
                        "status": "needs_retry",
                        "proposal": proposal,
                        "proposal_sha256": proposal_sha256,
                        "review": review,
                        "result": {
                            "status": "error",
                            "error": verdict_authority_error,
                        },
                        "artifact_path": str(artifact_path),
                        "authority": authority,
                    }
                _write_review_artifact(
                    artifact_path,
                    apply_key=apply_key,
                    proposal_sha256=proposal_sha256,
                    proposal=proposal,
                    authority=authority,
                    review=review,
                    now=now,
                )
                if (
                    _load_review_artifact(
                        artifact_path,
                        apply_key=apply_key,
                        proposal_sha256=proposal_sha256,
                        proposal=proposal,
                        authority=authority,
                    )
                    is None
                ):
                    raise OSError(
                        "local-consensus verdict artifact read-back validation failed"
                    )
        except (OSError, ValueError) as exc:
            return {
                "status": "needs_retry",
                "proposal": proposal,
                "proposal_sha256": proposal_sha256,
                "review": review,
                "result": {
                    "status": "error",
                    "error": f"local-consensus verdict artifact write failed: {exc}",
                },
                "artifact_path": str(artifact_path),
                "authority": authority,
            }
    return {
        "status": decision,
        "proposal": proposal,
        "proposal_sha256": proposal_sha256,
        "review": review,
        "artifact_path": str(artifact_path),
        "artifact_reused": False,
        "authority": authority,
    }


def eligible_records(
    records: list[dict[str, Any]],
    *,
    policy: AutoApplyPolicy,
    applied_keys: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    allowed_actions = set(policy.actions) & AUTO_ACTIONS
    for record in records:
        if record.get("kind") != "missed_candidate":
            continue
        if record.get("source") not in {"auditor", "pull-log"}:
            continue
        if record.get("source") == "pull-log":
            session_id = str(record.get("session_id") or "").strip()
            pull_event = record.get("pull_event")
            pull_session = (
                str(pull_event.get("session_id") or "").strip()
                if isinstance(pull_event, dict)
                else ""
            )
            if not session_id or pull_session != session_id:
                continue
        action = record.get("action_type")
        if action not in allowed_actions:
            continue
        if (
            record.get("lane") != "auto"
            or record.get("auto_apply_eligible") is not True
        ):
            continue
        if not record.get("normalize_key"):
            continue
        key = apply_key_for(record)
        if key in applied_keys:
            continue
        grouped[(str(action), str(record["normalize_key"]))].append(record)

    out: list[dict[str, Any]] = []
    for _group_key, group in grouped.items():
        if len(group) >= policy.min_count:
            out.append(group[-1])
    return out


def _page_ref(page_id: str) -> str:
    path = chronovisor_store.find_page(page_id)
    if path is None:
        raise ValueError(f"page does not exist: {page_id!r}")
    try:
        return str(path.relative_to(chronovisor_store.PAGES_DIR).with_suffix(""))
    except ValueError:
        return path.stem


def apply_query_hint(record: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    payload = action_payload(record)
    pages = expected_pages(record)
    page_id = str(payload.get("page_id") or (pages[0] if pages else "")).strip()
    query = str(
        payload.get("query")
        or record.get("prompt")
        or record.get("missing_signal")
        or ""
    ).strip()
    signal = str(payload.get("signal") or record.get("missing_signal") or "")
    if not page_id:
        return {
            "action": "query_hint",
            "status": "skipped",
            "reason": "query_hint missing page_id",
            "query": query,
        }
    if not query:
        return {
            "action": "query_hint",
            "status": "skipped",
            "reason": "query_hint missing query",
            "page_id": page_id,
        }
    query_key = normalize_query_text(query)
    for existing in load_query_hints():
        existing_key = str(
            existing.get("query_key")
            or normalize_query_text(str(existing.get("query") or ""))
        )
        if str(existing.get("page_id") or "") == page_id and existing_key == query_key:
            return {
                "action": "query_hint",
                "status": "already_applied",
                "page_id": page_id,
                "query": query,
                "hint": existing,
            }
    if dry_run:
        return {
            "action": "query_hint",
            "status": "dry_run",
            "page_id": page_id,
            "query": query,
        }
    hint = add_query_hint(
        page_id=page_id,
        query=query,
        signal=signal,
        source="recall-auto-apply",
        normalize_key=str(record.get("normalize_key", "")),
        increment_existing=False,
        provenance={
            "schema_version": 2,
            "feedback_ref": str(record.get("ref") or record.get("decision_id") or ""),
            "session_id": str(record.get("session_id") or ""),
            "source": str(record.get("source") or ""),
            "frontier_approved": True,
        },
    )
    return {"action": "query_hint", "status": "applied", "hint": hint}


def valid_alias_candidate(value: str) -> bool:
    text = value.strip()
    if text.endswith(".md"):
        text = text[:-3]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+", text))


def normalized_alias_candidate(value: str) -> str:
    text = value.strip()
    if text.endswith(".md"):
        text = text[:-3]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text


def fallback_to_query_hint(
    record: dict[str, Any],
    *,
    dry_run: bool,
    reason: str,
) -> dict[str, Any]:
    try:
        result = apply_query_hint(record, dry_run=dry_run)
    except Exception as exc:
        return {
            "action": str(record.get("action_type", "")),
            "status": "skipped",
            "fallback_to": "query_hint",
            "reason": reason,
            "fallback_error": f"{exc.__class__.__name__}: {exc}",
        }
    if result.get("status") == "skipped":
        return {
            "action": str(record.get("action_type", "")),
            "status": "skipped",
            "fallback_to": "query_hint",
            "reason": reason,
            "result": result,
        }
    if result.get("status") == "already_applied":
        return {
            "action": str(record.get("action_type", "")),
            "status": "already_applied",
            "fallback_to": "query_hint",
            "reason": reason,
            "result": result,
        }
    return {
        "action": str(record.get("action_type", "")),
        "status": "fallback_dry_run" if dry_run else "fallback_applied",
        "fallback_to": "query_hint",
        "reason": reason,
        "result": result,
    }


def apply_alias(record: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    payload = action_payload(record)
    pages = expected_pages(record)
    target = str(
        payload.get("target_page")
        or payload.get("page_id")
        or (pages[0] if pages else "")
    )
    raw_alias = payload.get("alias") or record.get("missing_signal") or ""
    if not isinstance(raw_alias, str):
        return fallback_to_query_hint(
            record,
            dry_run=dry_run,
            reason=f"alias payload is not a string: {type(raw_alias).__name__}",
        )
    alias = raw_alias.strip()
    if not alias:
        raise ValueError("alias action requires alias or missing_signal")
    if not valid_alias_candidate(alias):
        return fallback_to_query_hint(
            record,
            dry_run=dry_run,
            reason=f"invalid alias page_id: {alias!r}",
        )
    try:
        target_ref = _page_ref(target)
    except ValueError as exc:
        return fallback_to_query_hint(
            record,
            dry_run=dry_run,
            reason=str(exc),
        )
    existing_target = load_aliases().get(normalized_alias_candidate(alias))
    if existing_target == target_ref:
        return {
            "action": "alias",
            "status": "already_applied",
            "alias": alias,
            "target": target_ref,
        }
    if existing_target is not None:
        return {
            "action": "alias",
            "status": "skipped",
            "alias": alias,
            "target": target_ref,
            "reason": f"alias already points at {existing_target!r}",
        }
    if dry_run:
        return {
            "action": "alias",
            "status": "dry_run",
            "alias": alias,
            "target": target_ref,
        }
    try:
        add_alias(
            alias,
            target_ref,
            source=f"recall-auto-apply:{record.get('normalize_key', '')}",
        )
    except ValueError as exc:
        if "invalid alias page_id" in str(exc):
            return fallback_to_query_hint(
                record,
                dry_run=dry_run,
                reason=str(exc),
            )
        raise
    return {
        "action": "alias",
        "status": "applied",
        "alias": alias,
        "target": target_ref,
    }


def apply_page_tag(
    record: dict[str, Any],
    *,
    dry_run: bool,
    expected_page_sha256: str | None = None,
    allow_fallback: bool = True,
    authority_validator: Any | None = None,
) -> dict[str, Any]:
    payload = action_payload(record)
    pages = expected_pages(record)
    page_id = str(payload.get("page_id") or (pages[0] if pages else ""))
    path = chronovisor_store.find_page(page_id)
    if path is None:
        if not allow_fallback:
            return {
                "action": "page_tag",
                "status": "retry",
                "reason": "frontier-reviewed page_tag target changed before apply",
                "page_id": page_id,
            }
        return fallback_to_query_hint(
            record,
            dry_run=dry_run,
            reason=f"page_tag target page does not exist: {page_id!r}",
        )
    raw_tag = payload.get("tag") or record.get("missing_signal") or ""
    if not isinstance(raw_tag, str):
        if not allow_fallback:
            return {
                "action": "page_tag",
                "status": "retry",
                "reason": "frontier-reviewed page_tag payload changed before apply",
                "page_id": page_id,
            }
        return fallback_to_query_hint(
            record,
            dry_run=dry_run,
            reason=f"page_tag payload is not a string: {type(raw_tag).__name__}",
        )
    tag = raw_tag.strip()
    valid, reason = validate_tag(tag)
    if not valid:
        if not allow_fallback:
            return {
                "action": "page_tag",
                "status": "retry",
                "reason": "frontier-reviewed page_tag is no longer valid",
                "page_id": page_id,
                "tag": tag,
            }
        return fallback_to_query_hint(
            record,
            dry_run=dry_run,
            reason=f"invalid page tag {tag!r}: {reason}",
        )
    original = path.read_bytes()
    if (
        expected_page_sha256
        and hashlib.sha256(original).hexdigest() != expected_page_sha256
    ):
        return {
            "action": "page_tag",
            "status": "retry",
            "reason": "page changed after frontier page_tag review",
            "page_id": page_id,
            "tag": tag,
        }
    text = original.decode("utf-8")
    meta, _body = parse_frontmatter(text)
    existing = meta.get("tags")
    tags = list(existing) if isinstance(existing, list) else []
    if tag in tags:
        return {
            "action": "page_tag",
            "status": "already_applied",
            "page_id": page_id,
            "tag": tag,
        }
    new_tags = tags + [tag]
    updated = patch_frontmatter(
        text,
        {"tags": new_tags, "updated": date.today().isoformat()},
    )
    if dry_run:
        return {
            "action": "page_tag",
            "status": "dry_run",
            "page_id": page_id,
            "tag": tag,
        }
    try:
        with chronovisor_mutation_lock():
            if authority_validator is not None:
                authority_error = authority_validator()
                if authority_error:
                    return {
                        "action": "page_tag",
                        "status": "retry",
                        "reason": str(authority_error),
                        "page_id": page_id,
                        "tag": tag,
                    }
            # The tag proposal is based on ``original``. Re-check that exact
            # preimage while holding the same lock as content correction and
            # the other autonomous page writers, so a late tag write cannot
            # replace a newer correction.
            if path.read_bytes() != original:
                return {
                    "action": "page_tag",
                    "status": "retry",
                    "reason": "page changed before page_tag apply",
                    "page_id": page_id,
                    "tag": tag,
                }
            atomic_write(path, updated)
            if path.read_text(encoding="utf-8") != updated:
                return {
                    "action": "page_tag",
                    "status": "retry",
                    "reason": "page_tag post-write verification failed",
                    "page_id": page_id,
                    "tag": tag,
                }
    except (OSError, UnicodeDecodeError) as exc:
        return {
            "action": "page_tag",
            "status": "retry",
            "reason": f"page_tag mutation failed: {exc}",
            "page_id": page_id,
            "tag": tag,
        }
    record_new_tag(tag, reason=f"recall auto-apply {record.get('normalize_key', '')}")
    return {"action": "page_tag", "status": "applied", "page_id": page_id, "tag": tag}


def apply_record(record: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    action = record.get("action_type")
    if action == "query_hint":
        return apply_query_hint(record, dry_run=dry_run)
    if action == "alias":
        return apply_alias(record, dry_run=dry_run)
    if action == "page_tag":
        return apply_page_tag(record, dry_run=dry_run)
    raise ValueError(f"unsupported auto action: {action!r}")


def _apply_frontier_approved(
    record: dict[str, Any],
    proposal: dict[str, Any],
    *,
    expected_authority: dict[str, Any],
    expected_review: dict[str, Any],
    reviewer: Any | None,
) -> dict[str, Any]:
    """Apply only the effective action and page bytes the frontier reviewed."""

    def authority_mismatch() -> str | None:
        return _review_authority_error(
            expected_authority=expected_authority,
            review=expected_review,
            reviewer=reviewer,
            require_verdict_proof=True,
        )

    effective_action = str(proposal.get("effective_action") or "")
    evidence = proposal.get("page_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    expected_sha256 = str(evidence.get("sha256") or "")
    approved_sha256 = _canonical_json_sha256(proposal)

    # Keep the authority epoch stable from the final validation through the
    # durable mutation.  Adoption-artifact writers take the same lease.
    with decision_authority_lock():
        authority_error = authority_mismatch()
        if authority_error is not None:
            return {
                "action": effective_action,
                "status": "retry",
                "reason": authority_error,
            }

        if effective_action == "page_tag":
            current = _frontier_action_proposal(
                record, apply_key=str(proposal.get("apply_key") or "")
            )
            if _canonical_json_sha256(current) != approved_sha256:
                return {
                    "action": "page_tag",
                    "status": "retry",
                    "reason": "page_tag proposal changed after frontier review",
                }
            return apply_page_tag(
                record,
                dry_run=False,
                expected_page_sha256=expected_sha256 or None,
                allow_fallback=False,
                authority_validator=authority_mismatch,
            )

        with chronovisor_mutation_lock():
            authority_error = authority_mismatch()
            if authority_error is not None:
                return {
                    "action": effective_action,
                    "status": "retry",
                    "reason": authority_error,
                }
            current = _frontier_action_proposal(
                record, apply_key=str(proposal.get("apply_key") or "")
            )
            if _canonical_json_sha256(current) != approved_sha256:
                return {
                    "action": effective_action,
                    "status": "retry",
                    "reason": "recall action proposal changed after frontier review",
                }
            if effective_action == "query_hint":
                query_result = apply_query_hint(record, dry_run=False)
                if str(record.get("action_type") or "") != "query_hint":
                    if query_result.get("status") == "skipped":
                        return {
                            "action": str(record.get("action_type") or ""),
                            "status": "skipped",
                            "fallback_to": "query_hint",
                            "result": query_result,
                        }
                    return {
                        "action": str(record.get("action_type") or ""),
                        "status": "fallback_applied",
                        "fallback_to": "query_hint",
                    }
                return query_result
            if effective_action == "alias":
                # Target-page existence/hash was rechecked under the shared page
                # mutation lock, so the alias cannot silently turn into a fallback.
                return apply_alias(record, dry_run=False)
    return {
        "action": effective_action,
        "status": "retry",
        "reason": "frontier-approved effective action is unsupported",
    }


def _commit_convergence_entry(
    entry: dict[str, Any],
    *,
    record: dict[str, Any],
    gate: dict[str, Any],
    reviewer: Any | None,
    log_file: Path | None,
    prior: dict[str, Any] | None,
    deterministic_transition: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Durably commit one convergence effect under the current authority epoch.

    A model-approved mutation and its convergence bookkeeping are separate
    durable effects.  If the epoch changes between them, the mutation is left
    for the next pass to recover via the exact ``already_applied`` path rather
    than recording a stale approval.  Operational retry bookkeeping validates
    the epoch but does not pretend to carry a semantic quorum verdict.
    """

    with decision_authority_lock():
        expected_authority = gate.get("authority")
        review = gate.get("review")
        gate_status = str(gate.get("status") or "")
        if deterministic_transition:
            authority_error = None
        elif expected_authority is None:
            authority_error = "decision authority is unavailable for convergence"
        else:
            authority_error = _review_authority_error(
                expected_authority=expected_authority,
                review=review,
                reviewer=reviewer,
                require_verdict_proof=gate_status in {"approved", "rejected"},
            )

        status = str(entry.get("status") or "")
        if authority_error is None and status == "already_applied":
            # Recovery is an observation of an exact installed state, not a
            # replay of the old semantic authorization. Recheck it under the
            # lease immediately before writing only bookkeeping.
            try:
                current = _frontier_action_proposal(
                    record,
                    apply_key=str(entry.get("apply_key") or ""),
                )
                current_result = current.get("local_validation")
                current_status = (
                    str(current_result.get("status") or "")
                    if isinstance(current_result, dict)
                    else ""
                )
            except Exception as exc:
                authority_error = (
                    "already-applied recovery recheck failed: "
                    f"{exc.__class__.__name__}: {exc}"
                )
            else:
                if current_status != "already_applied":
                    authority_error = (
                        "already-applied recovery state changed before convergence"
                    )
                else:
                    entry["recovery_only"] = True
                    entry["recovery_proposal_sha256"] = _canonical_json_sha256(current)
        elif deterministic_transition:
            entry["deterministic_transition"] = True

        if authority_error is None:
            record_apply_log(entry, log_file)
            return entry, True

    fail_closed = {
        **entry,
        "status": "retry",
        "convergence_status": str((prior or {}).get("convergence_status") or "pending"),
        "attempt": int((prior or {}).get("attempt") or 0),
        "next_attempt_at": None,
        "result": {"status": "retry", "reason": authority_error},
        "uncommitted_result": entry.get("result"),
        "authority_transition_blocked": True,
    }
    return fail_closed, False


def apply_feedback_records(
    records: list[dict[str, Any]],
    *,
    policy: AutoApplyPolicy,
    dry_run: bool = False,
    log_file: Path | None = None,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    quarantine_cooldown_seconds: int = DEFAULT_QUARANTINE_COOLDOWN_SECONDS,
    budget: Any | None = None,
    now: datetime | None = None,
    review_dir: Path | None = None,
    frontier_reviewer: Any | None = None,
    frontier_timeout: int | None = None,
) -> dict[str, Any]:
    if not policy.enabled:
        return {"status": "disabled", "actions": []}
    applied_keys = read_applied_keys(log_file)
    states = read_apply_states(log_file)
    state_histories = read_apply_histories(log_file)
    actions: list[dict[str, Any]] = []
    now = now or datetime.now()
    resolved_review_dir = _resolved_review_dir(log_file, review_dir)
    for record in eligible_records(records, policy=policy, applied_keys=applied_keys):
        key = apply_key_for(record)
        prior = states.get(key)
        if not _has_semantic_no_quorum_marker(prior) and not _retry_ready(
            prior,
            now=now,
            quarantine_cooldown_seconds=quarantine_cooldown_seconds,
        ):
            continue
        resumed_from_quarantine = bool(
            prior and str(prior.get("convergence_status") or "") == "quarantined"
        )
        if dry_run:
            try:
                proposal = _frontier_action_proposal(record, apply_key=key)
                gate = {
                    "status": "dry_run",
                    "proposal": proposal,
                    "result": proposal.get("local_validation") or {"status": "dry_run"},
                }
            except Exception as exc:
                gate = {
                    "status": "proposal_error",
                    "result": {
                        "status": "error",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    },
                }
        else:
            gate = _frontier_gate(
                record,
                apply_key=key,
                review_dir=resolved_review_dir,
                reviewer=frontier_reviewer,
                timeout=frontier_timeout,
                budget=budget,
                now=now,
                prior=prior,
                prior_history=state_histories.get(key),
                quarantine_cooldown_seconds=quarantine_cooldown_seconds,
            )
        if gate.get("status") in {
            "semantic_hold_reused",
            "semantic_hold_malformed",
            "retry_deferred",
        }:
            # Exact holds and malformed legacy markers are read-only terminal
            # observations.  Do not append an identical log row forever.
            actions.append(
                {
                    "ts": now.isoformat(timespec="seconds"),
                    "apply_key": key,
                    "status": str(gate.get("status")),
                    "convergence_status": "quarantined",
                    "semantic_hold": gate.get("semantic_hold")
                    or (prior or {}).get("semantic_hold"),
                    "semantic_hold_reused": gate.get("status")
                    == "semantic_hold_reused",
                }
            )
            continue
        if gate.get("status") == "budget_deferred":
            actions.append(
                {
                    "ts": now.isoformat(timespec="seconds"),
                    "apply_key": key,
                    "normalize_key": record.get("normalize_key", ""),
                    "action_type": record.get("action_type", ""),
                    "source_ref": record.get("ref", ""),
                    "dry_run": False,
                    "status": "budget_deferred",
                    "convergence_status": str(
                        (prior or {}).get("convergence_status") or "pending"
                    ),
                    "attempt": int((prior or {}).get("attempt") or 0),
                    "reason": gate.get("reason") or "frontier budget exhausted",
                    "frontier_artifact": gate.get("artifact_path"),
                }
            )
            continue
        gate_status = str(gate.get("status") or "needs_retry")
        attempt = (
            int((prior or {}).get("attempt") or 0)
            if gate_status == "semantic_hold"
            else (
                1
                if resumed_from_quarantine
                else int((prior or {}).get("attempt") or 0) + 1
            )
        )
        review = gate.get("review") if isinstance(gate.get("review"), dict) else None
        if gate_status == "approved" and not dry_run:
            if budget is not None:
                mutation_allowed, mutation_reason = budget.consume("mutation")
                if not mutation_allowed:
                    actions.append(
                        {
                            "ts": now.isoformat(timespec="seconds"),
                            "apply_key": key,
                            "normalize_key": record.get("normalize_key", ""),
                            "action_type": record.get("action_type", ""),
                            "source_ref": record.get("ref", ""),
                            "dry_run": False,
                            "status": "budget_deferred",
                            "convergence_status": str(
                                (prior or {}).get("convergence_status") or "pending"
                            ),
                            "attempt": int((prior or {}).get("attempt") or 0),
                            "reason": mutation_reason,
                            "frontier_artifact": gate.get("artifact_path"),
                            "frontier_review": review,
                        }
                    )
                    continue
            try:
                result = _apply_frontier_approved(
                    record,
                    dict(gate.get("proposal") or {}),
                    expected_authority=dict(gate.get("authority") or {}),
                    expected_review=dict(review or {}),
                    reviewer=frontier_reviewer,
                )
            except Exception as exc:
                result = {
                    "status": "error",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
        elif gate_status in {"dry_run", "no_mutation", "proposal_error"}:
            result = dict(gate.get("result") or {"status": "error"})
        elif review is not None and is_human_required_result(review):
            result = {
                "status": "human_required",
                "reason": review.get("summary") or "frontier authorization required",
            }
        elif gate_status == "rejected":
            result = {
                "status": "frontier_rejected",
                "reason": (review or {}).get("summary")
                or "frontier rejected recall mutation",
            }
        elif gate_status == "semantic_hold":
            result = {
                "status": "semantic_hold",
                "reason": "local semantic models did not reach a safe quorum",
            }
        else:
            result = {
                "status": "frontier_retry",
                "reason": (gate.get("result") or {}).get("error")
                or (review or {}).get("summary")
                or "local-consensus verdict is not ready",
            }
        status = str(result.get("status") or "error")
        if status in TERMINAL_SUCCESS_STATUSES or (dry_run and status == "dry_run"):
            convergence_status = "applied"
        elif status == "frontier_rejected":
            convergence_status = "rejected"
        elif status == "human_required":
            convergence_status = "human_required"
        elif status == "semantic_hold":
            convergence_status = "quarantined"
        else:
            convergence_status = "retry_wait"
        next_attempt_at: str | None = None
        if convergence_status == "retry_wait":
            if attempt >= max(1, max_attempts):
                convergence_status = "quarantined"
            else:
                delay = max(0, backoff_base_seconds) * (2 ** max(0, attempt - 1))
                next_attempt_at = (now + timedelta(seconds=delay)).isoformat(
                    timespec="seconds"
                )
        entry = {
            "ts": now.isoformat(timespec="seconds"),
            "apply_key": key,
            "normalize_key": record.get("normalize_key", ""),
            "action_type": record.get("action_type", ""),
            "source_ref": record.get("ref", ""),
            "dry_run": dry_run,
            "status": status,
            "convergence_status": convergence_status,
            "attempt": attempt,
            "next_attempt_at": next_attempt_at,
            "result": result,
            "frontier_review": review,
            "frontier_artifact": gate.get("artifact_path"),
            "frontier_artifact_reused": bool(gate.get("artifact_reused")),
            "proposal_sha256": gate.get("proposal_sha256"),
            "review_authority": gate.get("authority"),
        }
        if gate_status == "semantic_hold":
            entry["semantic_hold"] = gate.get("semantic_hold")
            entry["last_failure_class"] = LOCAL_SEMANTIC_NO_QUORUM
            entry["frontier_review"] = None
        elif (
            _has_semantic_no_quorum_marker(prior)
            and convergence_status == "retry_wait"
            and isinstance((prior or {}).get("semantic_hold"), dict)
            and isinstance(gate.get("authority"), dict)
            and isinstance(gate.get("proposal_sha256"), str)
        ):
            # Preserve the old A epoch while B has an operational retry.  A
            # rollback to A can restore the hold before another model call.
            entry["semantic_hold"] = (prior or {})["semantic_hold"]
            entry["last_failure_class"] = LOCAL_SEMANTIC_NO_QUORUM
            entry["semantic_hold_recheck_sha256"] = _semantic_recheck_sha256(
                _auto_apply_semantic_epoch(
                    apply_key=key,
                    proposal_sha256=str(gate["proposal_sha256"]),
                ),
                gate["authority"],
            )
        if resumed_from_quarantine:
            entry["resumed_from_quarantine"] = True
            entry["quarantine_resume_count"] = (
                int((prior or {}).get("quarantine_resume_count") or 0) + 1
            )
        if convergence_status == "quarantined" and gate_status != "semantic_hold":
            entry["quarantined_at"] = now.isoformat(timespec="seconds")
            entry["quarantine_retry_at"] = (
                now + timedelta(seconds=max(0, quarantine_cooldown_seconds))
            ).isoformat(timespec="seconds")
        persisted = False
        if not dry_run:
            entry, persisted = _commit_convergence_entry(
                entry,
                record=record,
                gate=gate,
                reviewer=frontier_reviewer,
                log_file=log_file,
                prior=prior,
            )
        actions.append(entry)
        if persisted:
            states[key] = entry
            state_histories.setdefault(key, []).append(entry)
            if entry.get("convergence_status") == "applied":
                applied_keys.add(key)
    if not dry_run:
        errors = [action for action in actions if action.get("status") == "error"]
        if errors:
            try:
                from chronovisor.auto_apply_error_supervisor import (
                    supervise_error_records,
                )

                supervisor = supervise_error_records(errors)
            except Exception as exc:
                supervisor = {
                    "status": "error",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            return {
                "status": (
                    "budget_deferred"
                    if any(
                        action.get("status") == "budget_deferred" for action in actions
                    )
                    else "ok"
                ),
                "actions": actions,
                "auto_apply_self_heal": supervisor,
            }
    return {
        "status": (
            "budget_deferred"
            if any(action.get("status") == "budget_deferred" for action in actions)
            else "ok"
        ),
        "actions": actions,
    }


def apply_review_feedback_records(
    records: list[dict[str, Any]],
    *,
    dry_run: bool = False,
    log_file: Path | None = None,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    quarantine_cooldown_seconds: int = DEFAULT_QUARANTINE_COOLDOWN_SECONDS,
    budget: Any | None = None,
    now: datetime | None = None,
    review_dir: Path | None = None,
    frontier_reviewer: Any | None = None,
    frontier_timeout: int | None = None,
) -> dict[str, Any]:
    """Close auditor review actions without creating a human queue.

    ``few_shot`` is materialized as the already-safe query-hint primitive;
    the same feedback is also picked up by the frontier-reviewed search label
    queue. ``threshold`` is routed into Recall Lab's replay/adoption loop and
    is never applied directly.
    """
    candidates_by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        if (
            record.get("kind") != "missed_candidate"
            or record.get("source") != "auditor"
            or record.get("action_type") not in REVIEW_ACTIONS
            or record.get("lane") != "review"
        ):
            continue
        key = apply_key_for(record)
        candidates_by_key[key] = record

    states = read_apply_states(log_file)
    state_histories = read_apply_histories(log_file)
    now = now or datetime.now()
    resolved_review_dir = _resolved_review_dir(log_file, review_dir)
    actions: list[dict[str, Any]] = []
    for record in candidates_by_key.values():
        key = apply_key_for(record)
        prior = states.get(key)
        if not _has_semantic_no_quorum_marker(prior) and not _retry_ready(
            prior,
            now=now,
            quarantine_cooldown_seconds=quarantine_cooldown_seconds,
        ):
            continue
        resumed_from_quarantine = bool(
            prior and str(prior.get("convergence_status") or "") == "quarantined"
        )
        action = str(record.get("action_type") or "")
        effect_record = record
        if action == "threshold":
            gate: dict[str, Any] = {"status": "no_mutation"}
            result = {
                "action": action,
                "status": "routed_to_recall_lab",
                "reason": "threshold changes are replay-gated by recall_improvement",
            }
        else:
            converted = {
                **record,
                "action_type": "query_hint",
                "action_payload": {
                    **action_payload(record),
                    "query": action_payload(record).get("query")
                    or record.get("prompt")
                    or record.get("missing_signal"),
                },
            }
            effect_record = converted
            if dry_run:
                try:
                    proposal = _frontier_action_proposal(converted, apply_key=key)
                    gate = {
                        "status": "dry_run",
                        "proposal": proposal,
                        "result": proposal.get("local_validation")
                        or {"status": "dry_run"},
                    }
                except Exception as exc:
                    gate = {
                        "status": "proposal_error",
                        "result": {
                            "status": "error",
                            "error": f"{exc.__class__.__name__}: {exc}",
                        },
                    }
            else:
                gate = _frontier_gate(
                    converted,
                    apply_key=key,
                    review_dir=resolved_review_dir,
                    reviewer=frontier_reviewer,
                    timeout=frontier_timeout,
                    budget=budget,
                    now=now,
                    prior=prior,
                    prior_history=state_histories.get(key),
                    quarantine_cooldown_seconds=quarantine_cooldown_seconds,
                )
            if gate.get("status") in {
                "semantic_hold_reused",
                "semantic_hold_malformed",
                "retry_deferred",
            }:
                actions.append(
                    {
                        "ts": now.isoformat(timespec="seconds"),
                        "apply_key": key,
                        "status": str(gate.get("status")),
                        "convergence_status": "quarantined",
                        "semantic_hold": gate.get("semantic_hold")
                        or (prior or {}).get("semantic_hold"),
                    }
                )
                continue
            if gate.get("status") == "budget_deferred":
                actions.append(
                    {
                        "ts": now.isoformat(timespec="seconds"),
                        "apply_key": key,
                        "normalize_key": record.get("normalize_key", ""),
                        "action_type": action,
                        "source_ref": record.get("ref", ""),
                        "dry_run": False,
                        "status": "budget_deferred",
                        "convergence_status": str(
                            (prior or {}).get("convergence_status") or "pending"
                        ),
                        "attempt": int((prior or {}).get("attempt") or 0),
                        "reason": gate.get("reason") or "frontier budget exhausted",
                        "frontier_artifact": gate.get("artifact_path"),
                    }
                )
                continue
            gate_status = str(gate.get("status") or "needs_retry")
            review = (
                gate.get("review") if isinstance(gate.get("review"), dict) else None
            )
            if gate_status == "approved" and not dry_run:
                if budget is not None:
                    mutation_allowed, mutation_reason = budget.consume("mutation")
                    if not mutation_allowed:
                        actions.append(
                            {
                                "ts": now.isoformat(timespec="seconds"),
                                "apply_key": key,
                                "normalize_key": record.get("normalize_key", ""),
                                "action_type": action,
                                "source_ref": record.get("ref", ""),
                                "dry_run": False,
                                "status": "budget_deferred",
                                "convergence_status": str(
                                    (prior or {}).get("convergence_status") or "pending"
                                ),
                                "attempt": int((prior or {}).get("attempt") or 0),
                                "reason": mutation_reason,
                                "frontier_artifact": gate.get("artifact_path"),
                                "frontier_review": review,
                            }
                        )
                        continue
                try:
                    result = _apply_frontier_approved(
                        converted,
                        dict(gate.get("proposal") or {}),
                        expected_authority=dict(gate.get("authority") or {}),
                        expected_review=dict(review or {}),
                        reviewer=frontier_reviewer,
                    )
                except Exception as exc:
                    result = {
                        "status": "error",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
            elif gate_status in {"dry_run", "no_mutation", "proposal_error"}:
                result = dict(gate.get("result") or {"status": "error"})
            elif review is not None and is_human_required_result(review):
                result = {
                    "status": "human_required",
                    "reason": review.get("summary")
                    or "frontier authorization required",
                }
            elif gate_status == "rejected":
                result = {
                    "status": "frontier_rejected",
                    "reason": review.get("summary")
                    if review
                    else "frontier rejected recall mutation",
                }
            elif gate_status == "semantic_hold":
                result = {
                    "status": "semantic_hold",
                    "reason": "local semantic models did not reach a safe quorum",
                }
            else:
                result = {
                    "status": "frontier_retry",
                    "reason": (gate.get("result") or {}).get("error")
                    or (review or {}).get("summary")
                    or "local-consensus verdict is not ready",
                }
        attempt = (
            int((prior or {}).get("attempt") or 0)
            if str(gate.get("status") or "") == "semantic_hold"
            else (
                1
                if resumed_from_quarantine
                else int((prior or {}).get("attempt") or 0) + 1
            )
        )
        status = str(result.get("status") or "error")
        if status in TERMINAL_SUCCESS_STATUSES or status == "dry_run":
            convergence_status = "applied"
        elif status == "frontier_rejected":
            convergence_status = "rejected"
        elif status == "human_required":
            convergence_status = "human_required"
        elif status == "semantic_hold":
            convergence_status = "quarantined"
        else:
            convergence_status = "retry_wait"
        next_attempt_at: str | None = None
        if convergence_status == "retry_wait":
            if attempt >= max(1, max_attempts):
                convergence_status = "quarantined"
            else:
                delay = max(0, backoff_base_seconds) * (2 ** max(0, attempt - 1))
                next_attempt_at = (now + timedelta(seconds=delay)).isoformat(
                    timespec="seconds"
                )
        entry = {
            "ts": now.isoformat(timespec="seconds"),
            "apply_key": key,
            "normalize_key": record.get("normalize_key", ""),
            "action_type": action,
            "source_ref": record.get("ref", ""),
            "dry_run": dry_run,
            "status": status,
            "convergence_status": convergence_status,
            "attempt": attempt,
            "next_attempt_at": next_attempt_at,
            "result": result,
            "frontier_review": (
                gate.get("review") if isinstance(gate.get("review"), dict) else None
            ),
            "frontier_artifact": gate.get("artifact_path"),
            "frontier_artifact_reused": bool(gate.get("artifact_reused")),
            "proposal_sha256": gate.get("proposal_sha256"),
            "review_authority": gate.get("authority"),
        }
        if str(gate.get("status") or "") == "semantic_hold":
            entry["semantic_hold"] = gate.get("semantic_hold")
            entry["last_failure_class"] = LOCAL_SEMANTIC_NO_QUORUM
            entry["frontier_review"] = None
        elif (
            _has_semantic_no_quorum_marker(prior)
            and convergence_status == "retry_wait"
            and isinstance((prior or {}).get("semantic_hold"), dict)
            and isinstance(gate.get("authority"), dict)
            and isinstance(gate.get("proposal_sha256"), str)
        ):
            entry["semantic_hold"] = (prior or {})["semantic_hold"]
            entry["last_failure_class"] = LOCAL_SEMANTIC_NO_QUORUM
            entry["semantic_hold_recheck_sha256"] = _semantic_recheck_sha256(
                _auto_apply_semantic_epoch(
                    apply_key=key,
                    proposal_sha256=str(gate["proposal_sha256"]),
                ),
                gate["authority"],
            )
        if resumed_from_quarantine:
            entry["resumed_from_quarantine"] = True
            entry["quarantine_resume_count"] = (
                int((prior or {}).get("quarantine_resume_count") or 0) + 1
            )
        if (
            convergence_status == "quarantined"
            and str(gate.get("status") or "") != "semantic_hold"
        ):
            entry["quarantined_at"] = now.isoformat(timespec="seconds")
            entry["quarantine_retry_at"] = (
                now + timedelta(seconds=max(0, quarantine_cooldown_seconds))
            ).isoformat(timespec="seconds")
        persisted = False
        if not dry_run:
            entry, persisted = _commit_convergence_entry(
                entry,
                record=effect_record,
                gate=gate,
                reviewer=frontier_reviewer,
                log_file=log_file,
                prior=prior,
                deterministic_transition=action == "threshold",
            )
        actions.append(entry)
        if persisted:
            states[key] = entry
            state_histories.setdefault(key, []).append(entry)
    return {
        "status": (
            "budget_deferred"
            if any(action.get("status") == "budget_deferred" for action in actions)
            else "ok"
        ),
        "actions": actions,
    }


def apply_feedback_file(
    *,
    feedback_file: Path | None = None,
    config_file: Path = RECALL_CONFIG_FILE,
    min_count: int | None = None,
    dry_run: bool = False,
    budget: Any | None = None,
) -> dict[str, Any]:
    policy = load_auto_apply_policy(config_file)
    if min_count is not None:
        policy = AutoApplyPolicy(
            enabled=policy.enabled,
            min_count=max(1, min_count),
            actions=policy.actions,
        )
    records = read_jsonl(_feedback_file(feedback_file))
    auto = apply_feedback_records(
        records, policy=policy, dry_run=dry_run, budget=budget
    )
    review = apply_review_feedback_records(records, dry_run=dry_run, budget=budget)
    actions = [*(auto.get("actions") or []), *(review.get("actions") or [])]
    status = "ok"
    if auto.get("status") == "disabled":
        status = "disabled"
    elif any(action.get("status") == "budget_deferred" for action in actions):
        status = "budget_deferred"
    return {
        "status": status,
        "actions": actions,
        "auto": auto,
        "review": review,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply safe recall missed-candidate improvements."
    )
    parser.add_argument("--feedback-file", default=str(RECALL_FEEDBACK_FILE))
    parser.add_argument("--config", default=str(RECALL_CONFIG_FILE))
    parser.add_argument("--min-count", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = apply_feedback_file(
        feedback_file=Path(args.feedback_file).expanduser(),
        config_file=Path(args.config).expanduser(),
        min_count=args.min_count,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
