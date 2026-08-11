"""Recall context injection and host output rendering."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from chronovisor.core.recall_context import render_recall_payload
from chronovisor.recall import evidence_provider

if TYPE_CHECKING:
    from chronovisor.recall.recall_runtime import (
        ContextItem,
        RecallPolicy,
        RecallResult,
    )


def context_item_annotations(item: ContextItem) -> str:
    parts: list[str] = []
    if item.updated:
        parts.append(f"updated: {item.updated}")
    if item.sensitivity == "high":
        parts.append("sensitivity: high")
    return f" ({', '.join(parts)})" if parts else ""


def _neutralize_context_delimiters(text: str) -> str:
    for marker in (
        "[RECALL_CONTEXT]",
        "[/RECALL_CONTEXT]",
        "[WORKING_MEMORY]",
        "[/WORKING_MEMORY]",
    ):
        text = re.sub(
            re.escape(marker),
            marker.replace("[", "［").replace("]", "］"),
            text,
            flags=re.IGNORECASE,
        )
    return text


def _recall_payload(
    result: RecallResult,
    policy: RecallPolicy,
    *,
    page_summary: Callable[[str], str],
) -> dict[str, Any]:
    if result.evidence_packet is not None:
        evidence_metadata = result.evidence_features.get("evidence_reconstruction")
        if not isinstance(evidence_metadata, dict):
            return {}
        evidence = evidence_provider.publication_payload(
            result.evidence_packet,
            evidence_metadata,
            policy.max_context_chars,
        )
        if not evidence:
            return {}
        trace = {
            "decision_id": _neutralize_context_delimiters(
                _one_line(result.decision_id, 80)
            ),
            "session_id": _neutralize_context_delimiters(
                _one_line(result.session_id, 120)
            ),
            "evidence_trace": evidence["evidence_trace"],
            "evidence_trace_sha256": evidence["evidence_trace_sha256"],
        }
        return {
            "trace": trace,
            "decision": result.decision,
            "confidence": round(result.confidence, 3),
            "authority": "evidence_reconstruction",
            "evidence_packet": evidence["evidence_packet"],
        }
    items: list[dict[str, Any]] = []
    for item in result.context_items:
        evidence = item.snippets[0] if item.snippets else ""
        payload_item: dict[str, Any] = {
            "page_id": item.page_id,
            "title": _neutralize_context_delimiters(_one_line(item.title, 160)),
            "updated": item.updated,
            "sensitivity": item.sensitivity,
        }
        if item.certificate_id:
            payload_item["certificate_id"] = item.certificate_id
            payload_item["evidence_kind"] = item.evidence_kind
            if item.evidence_kind == "rich" and evidence:
                payload_item["evidence"] = _neutralize_context_delimiters(
                    _one_line(evidence, 220)
                )
                if item.source_line > 0:
                    payload_item["source_line"] = item.source_line
        else:
            legacy_evidence = evidence or page_summary(item.page_id)
            payload_item["score"] = item.score
            if legacy_evidence:
                payload_item["evidence"] = _neutralize_context_delimiters(
                    _one_line(legacy_evidence, 220)
                )
        items.append(payload_item)
    return {
        "trace": {
            "decision_id": _neutralize_context_delimiters(
                _one_line(result.decision_id, 80)
            ),
            "session_id": _neutralize_context_delimiters(
                _one_line(result.session_id, 120)
            ),
        },
        "decision": result.decision,
        "confidence": round(result.confidence, 3),
        "context_style": policy.context_style,
        "queries": [
            _neutralize_context_delimiters(_one_line(query, 160))
            for query in result.queries[:3]
        ],
        "reasons": [
            _neutralize_context_delimiters(_one_line(reason, 120))
            for reason in result.reasons[:4]
        ],
        "items": items,
    }


def _render_recall_payload(payload: dict[str, Any], max_chars: int) -> str:
    return render_recall_payload(payload, max_chars)


def format_recall_context(
    result: RecallResult,
    policy: RecallPolicy,
    *,
    page_summary: Callable[[str], str],
) -> str:
    if result.evidence_packet is not None:
        payload = _recall_payload(result, policy, page_summary=page_summary)
        return (
            _render_recall_payload(payload, policy.max_context_chars) if payload else ""
        )
    if result.decision == "none" or not result.context_items:
        return ""
    return _render_recall_payload(
        _recall_payload(result, policy, page_summary=page_summary),
        policy.max_context_chars,
    )


def _retained_context_page_ids(context: str) -> list[str]:
    """Return only page IDs that survived the exact rendered context budget."""

    marker = "payload_json=\n"
    if marker not in context:
        return []
    encoded = context.split(marker, 1)[1].rsplit("\n[/RECALL_CONTEXT]", 1)[0]
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError):
        return []
    items = payload.get("items") if isinstance(payload, dict) else None
    return [
        str(item.get("page_id") or "")
        for item in items
        if isinstance(item, dict) and str(item.get("page_id") or "")
    ] if isinstance(items, list) else []


def merge_context_blocks(*blocks: str, max_chars: int) -> str:
    selected: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        candidate = "\n\n".join([*selected, block])
        if len(candidate) <= max_chars:
            selected.append(block)
    return "\n\n".join(selected)


def _one_line(text: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def result_to_dict(result: RecallResult) -> dict[str, Any]:
    data = asdict(result)
    data["context_items"] = [asdict(item) for item in result.context_items]
    return data


def render_output(result: RecallResult, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(result_to_dict(result), ensure_ascii=False)
    if not result.context:
        return "{}" if output_format in {"codex", "hook-json"} else ""
    if output_format == "claude":
        return result.context
    if output_format in {"codex", "hook-json"}:
        return json.dumps(
            {
                "systemMessage": "Chronovisor recall context injected.",
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": result.context,
                },
            },
            ensure_ascii=False,
        )
    return result.context
