"""Local Qwen repair agent for failure packets."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable


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
You are the local repair agent for LLM Wiki.
Return JSON only. Choose only one whitelisted action.
Prefer conservative repairs: if the packet has exactly one similar_existing_pages
candidate for apply.update_target_not_found, resolve_update_target is allowed.
If apply.update_target_not_found has no similar_existing_pages and the requested
page id is safe ASCII kebab-case, retry_raw is allowed because ingest can
retype a missing update into a create.
If a code change is required, escalate_to_frontier.
If ingest.frontier_nonconvergent was caused by frontier call budget exhaustion,
retry_raw is allowed; do not escalate the packet back to frontier.
"""


_CREATE_SAFE_PAGE_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_MAX_PAGE_ID_LEN = 200


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action,
            "confidence": self.confidence,
            "requested_page_id": self.requested_page_id,
            "target_page_id": self.target_page_id,
            "reason": self.reason,
            "notes": self.notes,
            "source": self.source,
        }


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


def _validate_decision(data: dict[str, Any], packet: dict[str, Any]) -> LocalRepairDecision | None:
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

    requested = data.get("requested_page_id")
    if requested is None:
        requested = packet.get("requested_page_id")
    if requested is not None and not isinstance(requested, str):
        return None

    target = data.get("target_page_id")
    if target is not None and not isinstance(target, str):
        return None

    if action == "resolve_update_target":
        candidates = packet.get("similar_existing_pages")
        if not isinstance(candidates, list) or not all(isinstance(c, str) for c in candidates):
            return None
        if target not in candidates:
            return None
        if status != "resolved" or float(confidence) < 0.85:
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


def deterministic_repair(packet: dict[str, Any]) -> LocalRepairDecision:
    failure_class = packet.get("failure_class")
    candidates = packet.get("similar_existing_pages")
    error_text = str(packet.get("error") or "").casefold()
    if (
        failure_class == "ingest.frontier_nonconvergent"
        and "frontier call budget exhausted" in error_text
    ):
        return LocalRepairDecision(
            status="resolved",
            action="retry_raw",
            confidence=0.88,
            requested_page_id=packet.get("requested_page_id"),
            reason=(
                "ingest already exhausted its bounded frontier review budget; "
                "restore the raw for local replay instead of escalating the "
                "self-heal packet back to frontier"
            ),
            source="deterministic",
        )
    if (
        failure_class == "apply.update_target_not_found"
        and isinstance(candidates, list)
        and len(candidates) == 1
        and isinstance(candidates[0], str)
    ):
        return LocalRepairDecision(
            status="resolved",
            action="resolve_update_target",
            confidence=0.92,
            requested_page_id=packet.get("requested_page_id"),
            target_page_id=candidates[0],
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
            action="escalate_to_frontier",
            confidence=0.88,
            requested_page_id=packet.get("requested_page_id"),
            reason=(
                "repeated recall auto-apply errors indicate a system-level "
                "policy or code fix that requires frontier approval"
            ),
            notes="local deterministic path intentionally escalates auto-apply error clusters",
            source="deterministic",
        )
    return LocalRepairDecision(
        status="escalate",
        action="escalate_to_frontier",
        confidence=0.5,
        requested_page_id=packet.get("requested_page_id"),
        reason="local deterministic rules could not safely repair this packet",
        source="deterministic",
    )


def build_prompt(packet: dict[str, Any]) -> str:
    return (
        "Diagnose this LLM Wiki failure packet and return one JSON repair decision.\n\n"
        + json.dumps(packet, ensure_ascii=False, indent=2)
    )


def propose_repair(
    packet: dict[str, Any],
    *,
    generator: Callable[..., str] | None = None,
    use_qwen: bool = True,
) -> LocalRepairDecision:
    """Ask Qwen for a repair decision, falling back to deterministic rules."""

    deterministic = deterministic_repair(packet)
    if (
        deterministic.status == "resolved"
        and deterministic.action
        in {"resolve_update_target", "retry_raw", "quarantine_raw"}
        and deterministic.confidence >= 0.85
    ):
        return deterministic

    if use_qwen:
        try:
            if generator is None:
                from llm_wiki_mcp.ollama import generate, is_available

                if not is_available():
                    raise RuntimeError("ollama unavailable")
                generator = generate
            output = generator(
                build_prompt(packet),
                system=LOCAL_REPAIR_SYSTEM_PROMPT,
                format=LOCAL_REPAIR_SCHEMA,
            )
            parsed = _extract_json_object(output)
            if parsed is not None:
                decision = _validate_decision(parsed, packet)
                if decision is not None:
                    return decision
        except Exception:
            pass
    return deterministic
