"""Deterministic and local-consensus repair decisions for failure packets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from llm_wiki_mcp.decision_authority import (
    compare_semantic_authority,
    current_semantic_authority,
    seal_semantic_artifact,
    semantic_verdict_authority_error,
)
from llm_wiki_mcp.decision_schema_manifest import schema_sha256
from llm_wiki_mcp.page_mutation import decision_authority_lock
from llm_wiki_mcp.semantic_hold import (
    build_semantic_no_quorum_hold,
    canonical_sha256,
    is_local_semantic_no_quorum,
)


ALLOWED_ACTIONS = {
    "resolve_update_target",
    "retry_raw",
    "quarantine_raw",
    "escalate_to_frontier",
    "propose_prompt_fix",
    "propose_test_case",
}


LOCAL_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "action", "confidence", "reason"],
    "properties": {
        "status": {"type": "string", "enum": ["resolved", "escalate", "rejected"]},
        "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "requested_page_id": {"type": ["string", "null"]},
        "target_page_id": {"type": ["string", "null"]},
        "reason": {"type": "string"},
        "notes": {"type": ["string", "null"]},
    },
}


LOCAL_REPAIR_SYSTEM_PROMPT = """\
You are one independent local repair voter for LLM Wiki.
Return JSON only. Choose only one whitelisted action.
Prefer conservative repairs: if the packet has exactly one similar_existing_pages
candidate for apply.update_target_not_found, resolve_update_target is allowed.
If apply.update_target_not_found has no similar_existing_pages and the requested
page id is safe ASCII kebab-case, retry_raw is allowed because ingest can
retype a missing update into a create.
If a code change appears necessary, propose_test_case.  Routine packets never
invoke a frontier model; only the separate trusted system-incident lane may do so.
If ingest.frontier_nonconvergent was caused by frontier call budget exhaustion,
retry_raw is allowed; do not escalate the packet back to frontier.
Apply these status/action pairs exactly:
- exactly one authorized similar_existing_pages candidate for
  apply.update_target_not_found -> resolved + resolve_update_target, echo the
  packet requested_page_id unchanged, and set target_page_id to that sole
  candidate exactly;
- no candidate and a safe ASCII kebab-case requested page id -> resolved +
  retry_raw;
- an unsafe or path-like requested page id -> rejected + quarantine_raw;
- a repeated local contract/schema implementation failure that needs a code
  regression guard -> escalate + propose_test_case;
- a structured-output failure whose validator feedback identifies a narrow
  instruction defect -> escalate + propose_prompt_fix.
Never combine an action with a different status. Do not use retry_raw for an
unsafe page id, and do not quarantine a packet merely because a test or prompt
repair is needed.
"""


_CREATE_SAFE_PAGE_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_MAX_PAGE_ID_LEN = 200
LOCAL_REPAIR_SEMANTIC_HOLD_RESOLVER_VERSION = "local-repair-semantic-hold-v2"

# The repair model must see only immutable failure evidence.  Self-heal updates
# the fields below while leasing, retrying, quarantining, and reporting a
# packet.  Excluding that bookkeeping from both the prompt and the hold epoch
# makes their identities identical, while retaining every current or future
# producer-owned evidence field by default.
_LOCAL_REPAIR_BOOKKEEPING_FIELDS = frozenset(
    {
        "action_error",
        "applied_action_path",
        "approved_action",
        "frontier_attempts",
        "frontier_eligibility_error",
        "frontier_error",
        "frontier_queue_path",
        "frontier_result",
        "frontier_status",
        "human_boundary_reclassified_at",
        "human_notification",
        "human_recheck_count",
        "human_required_at",
        "invalidated_semantic_hold",
        "last_attempt_at",
        "last_failure_class",
        "lease_expires_at",
        "lease_owner",
        "local_decision",
        "local_error",
        "local_failure_reason",
        "local_model",
        "local_repair_attempts",
        "next_attempt_at",
        "operational_local_repair_evidence",
        "pending_frontier_review_path",
        "quarantine_reopen_count",
        "quarantined_at",
        "rejected_action_path",
        "self_heal_attempts",
        "self_heal_queued",
        "semantic_hold",
        "semantic_hold_history",
        "semantic_hold_invalidated_at",
        "status",
        "system_incident_fingerprint",
        "system_incident_packet_path",
        "system_incident_status",
        "terminal_reason",
        "terminal_resume_kind",
        "terminal_resumed_at",
        "transient_read_back_retired_at",
        "updated_at",
        "verified_local_repair",
        "verified_local_repair_applied_at",
    }
)


@dataclass(frozen=True)
class LocalRepairDecision:
    status: str
    action: str
    confidence: float
    reason: str
    requested_page_id: str | None = None
    target_page_id: str | None = None
    notes: str | None = None
    source: str = "qwen"
    authority: dict[str, Any] | None = None
    decision_policy: dict[str, Any] | None = None
    local_consensus: dict[str, Any] | None = None
    semantic_hold: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": (
                3
                if self.semantic_hold is not None
                else 2
                if self.authority is not None
                else 1
            ),
            "status": self.status,
            "action": self.action,
            "confidence": self.confidence,
            "requested_page_id": self.requested_page_id,
            "target_page_id": self.target_page_id,
            "reason": self.reason,
            "notes": self.notes,
            "source": self.source,
            "authority": self.authority,
            "decision_policy": self.decision_policy,
            "local_consensus": self.local_consensus,
        }
        if self.semantic_hold is not None:
            payload["semantic_hold"] = self.semantic_hold
        return payload


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(cleaned, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _deterministic_update_target(
    packet: dict[str, Any],
) -> tuple[str, str] | None:
    """Return the only packet-authorized update target, if it is exact.

    A model may echo packet identities, but it cannot create either identity.
    Keeping this check shared by the deterministic and consensus paths avoids
    accepting a plausible target for the wrong failure or requested page.
    """

    requested = packet.get("requested_page_id")
    candidates = packet.get("similar_existing_pages")
    if (
        packet.get("failure_class") != "apply.update_target_not_found"
        or not isinstance(requested, str)
        or not requested
        or not isinstance(candidates, list)
        or len(candidates) != 1
        or not isinstance(candidates[0], str)
        or not candidates[0]
    ):
        return None
    return requested, candidates[0]


def _validate_decision(
    data: dict[str, Any], packet: dict[str, Any]
) -> LocalRepairDecision | None:
    status = data.get("status")
    action = data.get("action")
    confidence = data.get("confidence")
    reason = data.get("reason")
    if status not in {"resolved", "escalate", "rejected"}:
        return None
    if action not in ALLOWED_ACTIONS:
        return None
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        return None
    if not isinstance(reason, str) or not reason.strip():
        return None

    packet_requested = packet.get("requested_page_id")
    requested = data.get("requested_page_id")
    if requested is None:
        requested = packet_requested
    if requested is not None and not isinstance(requested, str):
        return None

    target = data.get("target_page_id")
    if target is not None and not isinstance(target, str):
        return None

    if action == "resolve_update_target":
        authorized = _deterministic_update_target(packet)
        if authorized is None:
            return None
        authorized_requested, authorized_target = authorized
        if requested != authorized_requested or target != authorized_target:
            return None
        if status != "resolved":
            return None

    notes = data.get("notes")
    if notes is not None and not isinstance(notes, str):
        return None

    return LocalRepairDecision(
        status=status,
        action=action,
        confidence=float(confidence),
        requested_page_id=requested,
        target_page_id=target,
        reason=reason.strip(),
        notes=notes,
    )


def _is_create_safe_page_id(page_id: str | None) -> bool:
    if not isinstance(page_id, str):
        return False
    value = page_id.strip()
    return bool(
        value
        and len(value) <= _MAX_PAGE_ID_LEN
        and _CREATE_SAFE_PAGE_ID.fullmatch(value)
    )


_REVIEW_NONCONVERGENT_CLASSES = frozenset(
    {
        "ingest.frontier_nonconvergent",
        "ingest.local_consensus_nonconvergent",
    }
)
_REVIEW_BUDGET_EXHAUSTION_MARKERS = (
    "frontier call budget exhausted",
    "local review budget exhausted",
    "structured review budget exhausted",
)


def is_review_budget_nonconvergence(packet: dict[str, Any]) -> bool:
    """Recognize new and legacy bounded-review packets without model calls."""

    if packet.get("failure_class") not in _REVIEW_NONCONVERGENT_CLASSES:
        return False
    error_text = str(packet.get("error") or "").casefold()
    return any(marker in error_text for marker in _REVIEW_BUDGET_EXHAUSTION_MARKERS)


def deterministic_repair(packet: dict[str, Any]) -> LocalRepairDecision:
    failure_class = packet.get("failure_class")
    candidates = packet.get("similar_existing_pages")
    if is_review_budget_nonconvergence(packet):
        return LocalRepairDecision(
            status="resolved",
            action="retry_raw",
            confidence=0.88,
            requested_page_id=packet.get("requested_page_id"),
            reason=(
                "ingest already exhausted its bounded local review budget; "
                "restore the raw for local replay instead of escalating the "
                "self-heal packet back to frontier"
            ),
            source="deterministic",
        )
    authorized_target = _deterministic_update_target(packet)
    if authorized_target is not None:
        requested_page_id, target_page_id = authorized_target
        return LocalRepairDecision(
            status="resolved",
            action="resolve_update_target",
            confidence=0.92,
            requested_page_id=requested_page_id,
            target_page_id=target_page_id,
            reason="single existing page candidate for missing update target",
            source="deterministic",
        )
    if (
        failure_class == "apply.update_target_not_found"
        and isinstance(candidates, list)
        and len(candidates) == 0
        and _is_create_safe_page_id(packet.get("requested_page_id"))
    ):
        return LocalRepairDecision(
            status="resolved",
            action="retry_raw",
            confidence=0.9,
            requested_page_id=packet.get("requested_page_id"),
            reason=(
                "missing update target has no existing-page candidates and uses "
                "a create-safe page id; retry raw so ingest can normalize the "
                "missing update into a create"
            ),
            source="deterministic",
        )
    if failure_class == "recall.auto_apply_error":
        return LocalRepairDecision(
            status="escalate",
            action="propose_test_case",
            confidence=0.88,
            requested_page_id=packet.get("requested_page_id"),
            reason=(
                "repeated recall auto-apply errors indicate a system-level "
                "policy or code defect; preserve the packet for a reproducible "
                "local test instead of granting routine mutation authority"
            ),
            notes="frontier repair remains unavailable without trusted incident evidence",
            source="deterministic",
        )
    return LocalRepairDecision(
        status="escalate",
        action="propose_test_case",
        confidence=0.5,
        requested_page_id=packet.get("requested_page_id"),
        reason="local deterministic rules could not safely repair this packet; preserve a test case",
        source="deterministic",
    )


def semantic_request_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Project one packet to the exact immutable evidence shown to voters.

    Unknown fields are evidence by default.  That is intentionally fail-safe:
    a future producer can add evidence without also remembering to extend the
    semantic-hold resolver, while newly added self-heal bookkeeping merely
    causes an unnecessary cache miss until explicitly classified here.
    """

    return {
        str(field): value
        for field, value in packet.items()
        if str(field) not in _LOCAL_REPAIR_BOOKKEEPING_FIELDS
    }


def semantic_hold_epoch(packet: dict[str, Any]) -> dict[str, Any]:
    """Bind a hold to the exact request, system policy, and output schema."""

    evidence = semantic_request_packet(packet)
    prompt = build_prompt(packet)
    return {
        "resolver_version": LOCAL_REPAIR_SEMANTIC_HOLD_RESOLVER_VERSION,
        "failure_id": str(packet.get("failure_id") or ""),
        "failure_class": str(packet.get("failure_class") or ""),
        "evidence_sha256": canonical_sha256(evidence),
        "prompt_sha256": canonical_sha256(prompt),
        "system_sha256": canonical_sha256(LOCAL_REPAIR_SYSTEM_PROMPT),
        "schema_sha256": schema_sha256(LOCAL_REPAIR_SCHEMA),
    }


def build_prompt(packet: dict[str, Any]) -> str:
    return (
        "Diagnose this LLM Wiki failure packet and return one JSON repair decision.\n\n"
        + json.dumps(semantic_request_packet(packet), ensure_ascii=False, indent=2)
    )


def propose_repair(
    packet: dict[str, Any],
    *,
    generator: Callable[..., str] | None = None,
    use_qwen: bool = True,
) -> LocalRepairDecision:
    """Reach a local repair decision without any frontier-model fallback.

    Deterministically provable repairs return immediately.  Every production
    decision that still requires inference uses the independent local quorum,
    whose structured sessions provide targeted JSON repair turns.  The
    ``generator`` argument is retained only as a narrow compatibility/test
    seam and never becomes the production default.
    """

    deterministic = deterministic_repair(packet)
    if deterministic.status == "resolved" and deterministic.action in {
        "resolve_update_target",
        "retry_raw",
        "quarantine_raw",
    }:
        return deterministic

    if use_qwen:
        try:
            if generator is None:
                authority, authority_error = current_semantic_authority("local_repair")
                if authority is None or authority_error is not None:
                    return deterministic
                from llm_wiki_mcp.frontier_review import run_structured_review

                review = run_structured_review(
                    build_prompt(packet),
                    LOCAL_REPAIR_SCHEMA,
                    repo_root=Path(__file__).resolve().parents[2],
                    execute_patch=False,
                    decision_lane="local_repair",
                    system=LOCAL_REPAIR_SYSTEM_PROMPT,
                )
                if not isinstance(review, dict):
                    return deterministic
                policy_value = review.get("decision_policy")
                consensus_value = review.get("local_consensus")
                decision_policy = (
                    dict(policy_value) if isinstance(policy_value, dict) else {}
                )
                local_consensus = (
                    dict(consensus_value) if isinstance(consensus_value, dict) else {}
                )
                source = "local_consensus"
                parsed = (
                    None if isinstance(review.get("frontier_failure"), dict) else review
                )
            else:
                output = generator(
                    build_prompt(packet),
                    system=LOCAL_REPAIR_SYSTEM_PROMPT,
                    format=LOCAL_REPAIR_SCHEMA,
                )
                parsed = _extract_json_object(output)
                source = "legacy_generator"
            if source == "local_consensus" and is_local_semantic_no_quorum(review):
                failure = review.get("frontier_failure")
                reason = str(
                    (failure.get("summary") if isinstance(failure, dict) else None)
                    or review.get("summary")
                    or "local_models_did_not_reach_two_vote_quorum"
                )
                try:
                    with decision_authority_lock():
                        current, current_error = current_semantic_authority(
                            "local_repair"
                        )
                        epoch_error = current_error or compare_semantic_authority(
                            authority,
                            current,
                            lane="local_repair",
                        )
                        if epoch_error is not None:
                            raise ValueError(epoch_error)
                        semantic_hold = build_semantic_no_quorum_hold(
                            "local_repair",
                            semantic_hold_epoch(packet),
                            authority,
                            review,
                        )
                except Exception as exc:
                    return LocalRepairDecision(
                        status="rejected",
                        action="quarantine_raw",
                        confidence=0.0,
                        requested_page_id=packet.get("requested_page_id"),
                        reason=f"local semantic hold deferred: {exc}",
                        source="local_deferred",
                    )
                return LocalRepairDecision(
                    status="rejected",
                    action="quarantine_raw",
                    confidence=0.0,
                    requested_page_id=packet.get("requested_page_id"),
                    reason=reason,
                    source="semantic_hold",
                    authority=dict(authority),
                    decision_policy=dict(decision_policy),
                    local_consensus=dict(local_consensus),
                    semantic_hold=semantic_hold,
                )
            if parsed is not None:
                decision = _validate_decision(parsed, packet)
                if decision is not None:
                    if source != "local_consensus":
                        return replace(decision, source=source)
                    # Canonical authority validation must see the exact
                    # action-bearing payload that the quorum signed, not only
                    # its audit envelope.
                    review = decision.to_dict()
                    review["decision_policy"] = decision_policy
                    review["local_consensus"] = local_consensus
                    if (
                        semantic_verdict_authority_error(
                            review,
                            authority,
                            lane="local_repair",
                        )
                        is not None
                    ):
                        return deterministic
                    current, current_error = current_semantic_authority("local_repair")
                    if current is None or current_error is not None:
                        return deterministic
                    if (
                        compare_semantic_authority(
                            authority,
                            current,
                            lane="local_repair",
                        )
                        is not None
                    ):
                        return deterministic
                    sealed = seal_semantic_artifact(
                        {"schema_version": 2},
                        authority=authority,
                        lane="local_repair",
                    )
                    return replace(
                        decision,
                        source=source,
                        authority=sealed["authority"],
                        decision_policy=decision_policy,
                        local_consensus=local_consensus,
                    )
        except Exception:
            pass
    return deterministic
