"""Pure lint semantic-mutation proposal and review contracts."""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from chronovisor.core.canonical_json import canonical_json_permissive as _canonical_json
from chronovisor.core.hashutil import sha256_text as _sha256_text
from chronovisor.decision.decision_schema_manifest import SAFE_FIX_REVIEW_SCHEMA

# The packet limit is measured from the exact pretty-printed JSON sent to the
# model, not tokens. 106k characters leaves room for the trusted rubric and
# response schema inside the largest 112k production bucket while allowing
# most pages to be reviewed with complete pre/post bytes.
SAFE_FIX_REVIEW_PACKET_MAX_CHARS = 106_000
SAFE_FIX_REPACKET_CONTEXT_LINES = 4
SAFE_FIX_SEMANTIC_HOLD_RESOLVER_VERSION = "lint-safe-fix-semantic-hold-v1"

StructuredReviewer = Callable[[str, dict[str, Any]], Mapping[str, Any] | str]


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return _sha256_text(_canonical_json(dict(value)))


def _render_review_packet(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _bounded_text(value: str, limit: int = 30_000) -> str:
    if len(value) <= limit:
        return value
    half = max(1, (limit - 35) // 2)
    return value[:half] + "\n[... bounded review payload ...]\n" + value[-half:]


def _review_opcode_manifest(
    expected_lines: list[str],
    updated_lines: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a complete deterministic line-coverage manifest and changed bytes."""

    matcher = difflib.SequenceMatcher(
        None,
        expected_lines,
        updated_lines,
        autojunk=True,
    )
    manifest: list[dict[str, Any]] = []
    changed_spans: list[dict[str, Any]] = []
    for ordinal, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes(), 1):
        expected_segment = "".join(expected_lines[i1:i2])
        updated_segment = "".join(updated_lines[j1:j2])
        entry = {
            "ordinal": ordinal,
            "tag": tag,
            "expected_line_range": [i1, i2],
            "updated_line_range": [j1, j2],
            "expected_chars": len(expected_segment),
            "updated_chars": len(updated_segment),
            "expected_sha256": _sha256_text(expected_segment),
            "updated_sha256": _sha256_text(updated_segment),
        }
        manifest.append(entry)
        if tag == "equal":
            continue
        leading_context = "".join(
            expected_lines[max(0, i1 - SAFE_FIX_REPACKET_CONTEXT_LINES) : i1]
        )
        trailing_context = "".join(
            expected_lines[i2 : i2 + SAFE_FIX_REPACKET_CONTEXT_LINES]
        )
        span = {
            **entry,
            "expected_text": expected_segment,
            "updated_text": updated_segment,
            "leading_context": leading_context,
            "trailing_context": trailing_context,
            "leading_context_sha256": _sha256_text(leading_context),
            "trailing_context_sha256": _sha256_text(trailing_context),
        }
        span["span_sha256"] = _canonical_hash(span)
        changed_spans.append(span)
    return manifest, changed_spans


def _receipt_core_matches(value: Mapping[str, Any]) -> bool:
    receipt_sha256 = value.get("receipt_sha256")
    if not isinstance(receipt_sha256, str):
        return False
    core = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return receipt_sha256 == _canonical_hash(core)


def _valid_target_lookup_receipt(
    value: object,
    *,
    target: str | None = None,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_keys = {
        "schema_version",
        "kind",
        "target",
        "index_snapshot",
        "target_absent",
        "fuzzy_candidates",
        "fuzzy_candidate",
        "no_acceptable_fuzzy_candidate",
        "receipt_sha256",
    }
    snapshot = value.get("index_snapshot")
    candidates = value.get("fuzzy_candidates")
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("kind") != "broken_link_target_lookup_receipt"
        or not isinstance(value.get("target"), str)
        or (target is not None and value.get("target") != target)
        or not isinstance(snapshot, Mapping)
        or set(snapshot) != {"corpus_version", "page_count", "page_ids_sha256"}
        or not isinstance(snapshot.get("corpus_version"), str)
        or not isinstance(snapshot.get("page_count"), int)
        or not isinstance(snapshot.get("page_ids_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", snapshot["page_ids_sha256"]) is None
        or not isinstance(value.get("target_absent"), bool)
        or not isinstance(candidates, list)
        or not all(isinstance(candidate, str) for candidate in candidates)
        or (
            value.get("fuzzy_candidate") is not None
            and not isinstance(value.get("fuzzy_candidate"), str)
        )
        or not isinstance(value.get("no_acceptable_fuzzy_candidate"), bool)
    ):
        return False
    return _receipt_core_matches(value)


def _valid_external_page_evidence(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_keys = {
        "schema_version",
        "kind",
        "page_id",
        "source_chars",
        "source_sha256",
        "text",
        "receipt_sha256",
    }
    text = value.get("text")
    return bool(
        set(value) == expected_keys
        and value.get("schema_version") == 1
        and value.get("kind") == "external_page_review_evidence"
        and isinstance(value.get("page_id"), str)
        and value.get("page_id")
        and isinstance(text, str)
        and value.get("source_chars") == len(text)
        and value.get("source_sha256") == _sha256_text(text)
        and _receipt_core_matches(value)
    )


def _exact_source_span(
    source_text: str,
    *,
    start: int,
    end: int,
) -> dict[str, Any]:
    rendered = source_text[start:end]
    return {
        "char_range": [start, end],
        "text": rendered,
        "text_sha256": _sha256_text(rendered),
        "source_sha256": _sha256_text(source_text),
    }


def _entity_alias_evidence_spans(
    expected_text: str,
    details: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    rows = details.get("alias_evidence")
    if not isinstance(rows, list) or not rows:
        return None, "entity_alias_evidence_missing"
    folded = expected_text.casefold()
    spans: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("entity_id"), str):
            return None, "entity_alias_evidence_invalid"
        aliases = row.get("matched_aliases")
        if not isinstance(aliases, list) or not aliases:
            return None, "entity_alias_match_missing"
        for alias in aliases:
            if not isinstance(alias, str) or not alias:
                return None, "entity_alias_match_invalid"
            offset = folded.find(alias.casefold())
            if offset < 0:
                return None, "entity_alias_match_not_in_preimage"
            start = max(0, offset - 500)
            end = min(len(expected_text), offset + len(alias) + 500)
            spans.append(
                {
                    "entity_id": row["entity_id"],
                    "matched_alias": alias,
                    **_exact_source_span(expected_text, start=start, end=end),
                }
            )
    return {
        "schema_version": 1,
        "kind": "entity_alias_semantic_evidence",
        "source_sha256": _sha256_text(expected_text),
        "spans": spans,
    }, None


def _operation_review_evidence(
    *,
    operation: str | None,
    details: Mapping[str, Any],
    expected_text: str,
    repacket: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    identity = details.get("identity_preflight")
    if identity is not None:
        from chronovisor.decision.decision_lane_prompts import (
            validate_identity_preflight_receipt,
        )

        if not validate_identity_preflight_receipt(identity):
            return None, "identity_preflight_invalid"

    if operation in {"broken_link_plaintext", "broken_link_retarget"}:
        target = details.get("target")
        lookup = details.get("target_lookup_receipt")
        if not isinstance(target, str) or not _valid_target_lookup_receipt(
            lookup,
            target=target,
        ):
            return None, "target_lookup_receipt_invalid"
        assert isinstance(lookup, Mapping)
        if lookup.get("target_absent") is not True:
            return None, "broken_link_target_not_absent"
        replacement = details.get("replacement")
        if operation == "broken_link_plaintext":
            if (
                replacement is not None
                or lookup.get("fuzzy_candidate") is not None
                or lookup.get("no_acceptable_fuzzy_candidate") is not True
            ):
                return None, "plaintext_target_lookup_not_exhaustive"
            return {"target_lookup_receipt": dict(lookup)}, None
        replacement_evidence = details.get("replacement_evidence")
        if (
            not isinstance(replacement, str)
            or lookup.get("fuzzy_candidate") != replacement
            or not _valid_external_page_evidence(replacement_evidence)
            or replacement_evidence.get("page_id") != replacement
        ):
            return None, "retarget_replacement_evidence_invalid"
        assert isinstance(replacement_evidence, Mapping)
        return {
            "target_lookup_receipt": dict(lookup),
            "replacement_evidence": dict(replacement_evidence),
        }, None

    if operation == "resolve_nested_frontmatter_conflict":
        conflicts = details.get("conflicts")
        if isinstance(conflicts, Mapping) and "permalink" in conflicts:
            if identity is None:
                return None, "permalink_identity_preflight_missing"
            return {"identity_preflight": dict(identity)}, None
        return None, None

    if operation == "drop_invalid_tags":
        validation = details.get("tag_validation_receipt")
        if validation is not None:
            if (
                not isinstance(validation, Mapping)
                or validation.get("kind") != "invalid_tag_validation_receipt"
                or validation.get("source_sha256") != _sha256_text(expected_text)
                or not _receipt_core_matches(validation)
            ):
                return None, "tag_validation_receipt_invalid"
            return {"tag_validation_receipt": dict(validation)}, None
        if repacket:
            return None, "tag_validation_receipt_missing"
        return None, None

    if not repacket:
        return None, None
    if operation == "backfill_entities_frontmatter":
        return _entity_alias_evidence_spans(expected_text, details)
    if operation == "backfill_recall_metadata":
        return None, "metadata_source_evidence_not_repacketable"
    if operation == "resolve_nested_frontmatter_conflict":
        return ({"identity_preflight": dict(identity)} if identity else None), None
    return None, "operation_semantic_evidence_not_repacketable"


def _review_receipt_from_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    rendered = _render_review_packet(packet)
    packet_coverage = packet.get("coverage")
    coverage_receipt = (
        {
            key: value
            for key, value in packet_coverage.items()
            if key != "opcode_manifest"
        }
        if isinstance(packet_coverage, Mapping)
        else {}
    )
    full_identity = {
        "page_id": packet.get("page_id"),
        "expected_chars": packet.get("expected_chars"),
        "updated_chars": packet.get("updated_chars"),
        "unified_diff_chars": packet.get("unified_diff_chars"),
        "expected_sha256": packet.get("expected_sha256"),
        "updated_sha256": packet.get("updated_sha256"),
        "full_unified_diff_sha256": packet.get("full_unified_diff_sha256"),
    }
    receipt = {
        "schema_version": 1,
        "kind": "semantic_mutation_review_receipt",
        "mode": packet.get("mode"),
        "full_chars": sum(
            int(packet.get(field) or 0)
            for field in ("expected_chars", "updated_chars", "unified_diff_chars")
        ),
        "full_sha256": _canonical_hash(full_identity),
        "rendered_chars": len(rendered),
        "rendered_sha256": _sha256_text(rendered),
        "complete": packet.get("mode") in {"full", "changed_spans"},
        "truncated": False,
        "repacket": packet.get("mode") in {"changed_spans", "insufficient"},
        "coverage": coverage_receipt,
    }
    if packet.get("mode") == "insufficient":
        receipt["insufficient_evidence_sha256"] = packet.get(
            "insufficient_evidence_sha256"
        )
    return receipt


def build_semantic_review_packet(
    *,
    page_id: str,
    expected_text: str,
    updated_text: str,
    max_chars: int | None = None,
    operation: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build complete model evidence, deterministically repacketed when needed.

    No prefix/suffix truncation is ever sent to a semantic reviewer. If the
    complete page packet is too large, every changed line span is rendered and
    a hash/range manifest covers all equal and changed spans. If that packet is
    still too large, the result is an explicit insufficient-evidence receipt;
    callers must hold it durably instead of starting a model session.
    """

    packet_limit = (
        SAFE_FIX_REVIEW_PACKET_MAX_CHARS if max_chars is None else max(1, max_chars)
    )
    expected_lines = expected_text.splitlines(keepends=True)
    updated_lines = updated_text.splitlines(keepends=True)
    unified_diff = "".join(
        difflib.unified_diff(
            expected_lines,
            updated_lines,
            fromfile=f"{page_id}:before",
            tofile=f"{page_id}:after",
            n=5,
        )
    )
    manifest, changed_spans = _review_opcode_manifest(expected_lines, updated_lines)
    manifest_sha256 = _sha256_text(_canonical_json(manifest))
    changed_spans_sha256 = _sha256_text(_canonical_json(changed_spans))
    full_identity = {
        "page_id": page_id,
        "expected_chars": len(expected_text),
        "updated_chars": len(updated_text),
        "unified_diff_chars": len(unified_diff),
        "expected_sha256": _sha256_text(expected_text),
        "updated_sha256": _sha256_text(updated_text),
        "full_unified_diff_sha256": _sha256_text(unified_diff),
    }
    coverage = {
        "schema_version": 1,
        "expected_lines": len(expected_lines),
        "updated_lines": len(updated_lines),
        "opcode_count": len(manifest),
        "changed_span_count": len(changed_spans),
        "opcode_manifest_sha256": manifest_sha256,
        "changed_spans_sha256": changed_spans_sha256,
    }
    packet_base = {
        "schema_version": 1,
        "kind": "semantic_mutation_review_packet",
        **full_identity,
    }
    operation_evidence, evidence_error = _operation_review_evidence(
        operation=operation,
        details=details or {},
        expected_text=expected_text,
        repacket=False,
    )
    full_packet: dict[str, Any] = {
        **packet_base,
        "mode": "full",
        "coverage": {
            **coverage,
            "rendered_changed_span_count": len(changed_spans),
            "all_changed_spans_rendered": True,
        },
        "preimage": expected_text,
        "postimage": updated_text,
        "unified_diff": unified_diff,
    }
    if operation_evidence is not None:
        full_packet["operation_evidence"] = operation_evidence
    if evidence_error is not None:
        full_packet_chars = len(_render_review_packet(full_packet))
        insufficient_identity = {
            **full_identity,
            "coverage": coverage,
            "attempted_full_packet_chars": full_packet_chars,
            "max_chars": packet_limit,
            "reason": evidence_error,
        }
        packet = {
            **packet_base,
            "mode": "insufficient",
            "coverage": {
                **coverage,
                "rendered_changed_span_count": 0,
                "all_changed_spans_rendered": False,
            },
            "reason": evidence_error,
            "attempted_full_packet_chars": full_packet_chars,
            "max_chars": packet_limit,
            "insufficient_evidence_sha256": _canonical_hash(insufficient_identity),
        }
    elif len(_render_review_packet(full_packet)) <= packet_limit:
        packet = full_packet
        full_packet_chars = len(_render_review_packet(full_packet))
    else:
        full_packet_chars = len(_render_review_packet(full_packet))
        operation_evidence, evidence_error = _operation_review_evidence(
            operation=operation,
            details=details or {},
            expected_text=expected_text,
            repacket=True,
        )
        repacket: dict[str, Any] = {
            **packet_base,
            "mode": "changed_spans",
            "coverage": {
                **coverage,
                "rendered_changed_span_count": len(changed_spans),
                "all_changed_spans_rendered": True,
                "opcode_manifest": manifest,
            },
            "changed_spans": changed_spans,
        }
        if operation_evidence is not None:
            repacket["operation_evidence"] = operation_evidence
        repacket_chars = len(_render_review_packet(repacket))
        if evidence_error is None and repacket_chars <= packet_limit:
            packet = repacket
        else:
            insufficient_identity = {
                **full_identity,
                "coverage": coverage,
                "attempted_repacket_chars": repacket_chars,
                "max_chars": packet_limit,
                "reason": evidence_error
                or "complete_changed_spans_exceed_review_packet_limit",
            }
            packet = {
                **packet_base,
                "mode": "insufficient",
                "coverage": {
                    **coverage,
                    "rendered_changed_span_count": 0,
                    "all_changed_spans_rendered": False,
                },
                "reason": insufficient_identity["reason"],
                "attempted_repacket_chars": repacket_chars,
                "attempted_full_packet_chars": full_packet_chars,
                "max_chars": packet_limit,
                "insufficient_evidence_sha256": _canonical_hash(insufficient_identity),
            }

    return packet, _review_receipt_from_packet(packet)


def _build_safe_fix_proposal(
    *,
    page_id: str,
    operation: str,
    expected_text: str,
    updated_text: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    unified_diff = "".join(
        difflib.unified_diff(
            expected_text.splitlines(keepends=True),
            updated_text.splitlines(keepends=True),
            fromfile=f"{page_id}:before",
            tofile=f"{page_id}:after",
            n=5,
        )
    )
    review_packet, review_receipt = build_semantic_review_packet(
        page_id=page_id,
        expected_text=expected_text,
        updated_text=updated_text,
        operation=operation,
        details=details,
    )
    proposal_details = dict(details)
    existing_receipt = proposal_details.get("review_receipt")
    if existing_receipt is not None and existing_receipt != review_receipt:
        raise ValueError("proposal details contain a conflicting review_receipt")
    proposal_details["review_receipt"] = review_receipt
    # A full diff is retained only when the review packet itself is complete
    # and small. Large proposals are represented by exact changed spans rather
    # than a misleading bounded prefix/suffix.
    rendered_diff = unified_diff if review_packet["mode"] == "full" else None
    return {
        "schema_version": 1,
        "kind": "lint_safe_fix_proposal",
        "page_id": page_id,
        "operation": operation,
        "expected_sha256": _sha256_text(expected_text),
        "updated_sha256": _sha256_text(updated_text),
        "details": proposal_details,
        "review_packet": review_packet,
        "unified_diff": rendered_diff,
        "unified_diff_sha256": _sha256_text(unified_diff),
        "full_unified_diff_sha256": _sha256_text(unified_diff),
        "unified_diff_truncated": False,
        "unified_diff_repacket": review_packet["mode"] != "full",
    }


def _runtime_insufficient_review_packet(
    proposal: Mapping[str, Any],
    *,
    expected_text: str,
    reason: str,
) -> dict[str, Any]:
    identity = {
        "proposal_sha256": _canonical_hash(proposal),
        "proposal_expected_sha256": proposal.get("expected_sha256"),
        "observed_expected_sha256": _sha256_text(expected_text),
        "reason": reason,
    }
    return {
        "schema_version": 1,
        "kind": "semantic_mutation_review_packet",
        "mode": "insufficient",
        "page_id": proposal.get("page_id"),
        "expected_sha256": proposal.get("expected_sha256"),
        "updated_sha256": proposal.get("updated_sha256"),
        "full_unified_diff_sha256": proposal.get("full_unified_diff_sha256"),
        "coverage": {
            "schema_version": 1,
            "rendered_changed_span_count": 0,
            "all_changed_spans_rendered": False,
        },
        "reason": reason,
        "insufficient_evidence_sha256": _canonical_hash(identity),
    }


def _review_packet_error(
    proposal: Mapping[str, Any],
    *,
    expected_text: str,
    updated_text: str | None = None,
) -> str | None:
    packet = proposal.get("review_packet")
    details = proposal.get("details")
    receipt = details.get("review_receipt") if isinstance(details, Mapping) else None
    if not isinstance(packet, Mapping) or not isinstance(receipt, Mapping):
        return "review_packet_or_receipt_missing"
    rendered = _render_review_packet(packet)
    packet_coverage = packet.get("coverage")
    coverage_receipt = (
        {
            key: value
            for key, value in packet_coverage.items()
            if key != "opcode_manifest"
        }
        if isinstance(packet_coverage, Mapping)
        else {}
    )
    if (
        receipt.get("rendered_sha256") != _sha256_text(rendered)
        or receipt.get("rendered_chars") != len(rendered)
        or receipt.get("truncated") is not False
        or receipt.get("mode") != packet.get("mode")
        or receipt.get("coverage") != coverage_receipt
    ):
        return "review_receipt_integrity_mismatch"
    if (
        proposal.get("expected_sha256") != _sha256_text(expected_text)
        or packet.get("expected_sha256") != proposal.get("expected_sha256")
        or packet.get("updated_sha256") != proposal.get("updated_sha256")
        or packet.get("full_unified_diff_sha256")
        != proposal.get("full_unified_diff_sha256")
    ):
        return "review_packet_proposal_hash_mismatch"
    if updated_text is not None:
        if proposal.get("updated_sha256") != _sha256_text(updated_text):
            return "review_packet_postimage_hash_mismatch"
        recompute_details = dict(details) if isinstance(details, Mapping) else {}
        recompute_details.pop("review_receipt", None)
        recomputed_packet, recomputed_receipt = build_semantic_review_packet(
            page_id=str(proposal.get("page_id") or ""),
            operation=str(proposal.get("operation") or ""),
            expected_text=expected_text,
            updated_text=updated_text,
            details=recompute_details,
        )
        if dict(packet) != recomputed_packet or dict(receipt) != recomputed_receipt:
            return "review_packet_trusted_recomputation_mismatch"
    mode = packet.get("mode")
    if mode == "full":
        preimage = packet.get("preimage")
        postimage = packet.get("postimage")
        unified_diff = packet.get("unified_diff")
        if not all(
            isinstance(value, str) for value in (preimage, postimage, unified_diff)
        ):
            return "full_review_packet_bytes_missing"
        assert isinstance(preimage, str)
        assert isinstance(postimage, str)
        assert isinstance(unified_diff, str)
        if (
            _sha256_text(preimage) != proposal.get("expected_sha256")
            or _sha256_text(postimage) != proposal.get("updated_sha256")
            or _sha256_text(unified_diff) != proposal.get("full_unified_diff_sha256")
            or receipt.get("complete") is not True
        ):
            return "full_review_packet_integrity_mismatch"
        return None
    if mode == "changed_spans":
        coverage = packet.get("coverage")
        spans = packet.get("changed_spans")
        if not isinstance(coverage, Mapping) or not isinstance(spans, list):
            return "changed_span_packet_missing"
        manifest = coverage.get("opcode_manifest")
        if not isinstance(manifest, list):
            return "changed_span_coverage_manifest_missing"
        if (
            coverage.get("opcode_manifest_sha256")
            != _sha256_text(_canonical_json(manifest))
            or coverage.get("changed_spans_sha256")
            != _sha256_text(_canonical_json(spans))
            or coverage.get("changed_span_count") != len(spans)
            or coverage.get("rendered_changed_span_count") != len(spans)
            or coverage.get("all_changed_spans_rendered") is not True
            or receipt.get("complete") is not True
        ):
            return "changed_span_coverage_integrity_mismatch"
        changed_manifest = [
            entry
            for entry in manifest
            if isinstance(entry, Mapping) and entry.get("tag") != "equal"
        ]
        if len(changed_manifest) != len(spans):
            return "changed_span_manifest_count_mismatch"
        for manifest_entry, span in zip(changed_manifest, spans, strict=True):
            if not isinstance(span, Mapping):
                return "changed_span_invalid"
            expected_segment = span.get("expected_text")
            updated_segment = span.get("updated_text")
            if not isinstance(expected_segment, str) or not isinstance(
                updated_segment, str
            ):
                return "changed_span_bytes_missing"
            for field in (
                "ordinal",
                "tag",
                "expected_line_range",
                "updated_line_range",
                "expected_chars",
                "updated_chars",
                "expected_sha256",
                "updated_sha256",
            ):
                if span.get(field) != manifest_entry.get(field):
                    return "changed_span_manifest_mismatch"
            span_without_hash = dict(span)
            span_sha256 = span_without_hash.pop("span_sha256", None)
            if (
                _sha256_text(expected_segment) != span.get("expected_sha256")
                or _sha256_text(updated_segment) != span.get("updated_sha256")
                or span_sha256 != _canonical_hash(span_without_hash)
            ):
                return "changed_span_hash_mismatch"
        return None
    if mode == "insufficient":
        if (
            receipt.get("complete") is not False
            or not isinstance(packet.get("reason"), str)
            or not isinstance(packet.get("insufficient_evidence_sha256"), str)
        ):
            return "insufficient_review_packet_invalid"
        return None
    return "review_packet_mode_invalid"


def _review_packet_for_prompt(
    proposal: Mapping[str, Any],
    *,
    expected_text: str,
    updated_text: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    error = _review_packet_error(
        proposal,
        expected_text=expected_text,
        updated_text=updated_text,
    )
    if error is not None:
        return (
            _runtime_insufficient_review_packet(
                proposal,
                expected_text=expected_text,
                reason=error,
            ),
            error,
        )
    packet = proposal.get("review_packet")
    assert isinstance(packet, Mapping)
    return dict(packet), None


def _render_safe_fix_prompt(
    proposal: Mapping[str, Any],
    *,
    review_packet: Mapping[str, Any],
) -> str:
    from chronovisor.decision.decision_lane_prompts import (
        semantic_mutation_decision_rubric,
        semantic_mutation_final_check,
    )

    operation = str(proposal.get("operation") or "")
    rubric = semantic_mutation_decision_rubric(operation)
    final_check = semantic_mutation_final_check(operation)
    rendered_proposal = dict(proposal)
    rendered_proposal.pop("review_packet", None)
    # The packet below is the only review rendering. Avoid duplicating a full
    # diff or exposing an old bounded storage rendering beside it.
    rendered_proposal["unified_diff"] = None
    rendered_proposal["proposal_sha256"] = _canonical_hash(proposal)
    rendered_details = rendered_proposal.get("details")
    if isinstance(rendered_details, Mapping):
        rendered_details = dict(rendered_details)
        replacement_evidence = rendered_details.get("replacement_evidence")
        if isinstance(replacement_evidence, Mapping) and isinstance(
            replacement_evidence.get("text"),
            str,
        ):
            compact_replacement = dict(replacement_evidence)
            compact_replacement["text"] = None
            compact_replacement["text_rendering"] = (
                "exact bytes are in review_packet.operation_evidence"
            )
            rendered_details["replacement_evidence"] = compact_replacement
        rendered_proposal["details"] = rendered_details
    return f"""\
You are the final autonomous reviewer for a Chronovisor semantic page mutation.
A local deterministic checker produced the proposal below, but it has no
authority to change content or metadata. Independently decide whether this
exact mutation is correct. Approve only when every change is justified by the
supplied page context and operation-specific evidence. Do not propose or apply
a different patch.

{rubric}

Exact proposal:
{json.dumps(rendered_proposal, ensure_ascii=False, indent=2, sort_keys=True)}

Complete deterministic review packet (never prefix/suffix truncated):
{_render_review_packet(review_packet)}

{final_check}

Return JSON matching this schema:
{json.dumps(SAFE_FIX_REVIEW_SCHEMA, ensure_ascii=False, indent=2)}
"""



def _build_safe_fix_prompt(
    proposal: Mapping[str, Any],
    *,
    expected_text: str,
    updated_text: str | None = None,
) -> str:
    review_packet, _packet_error = _review_packet_for_prompt(
        proposal,
        expected_text=expected_text,
        updated_text=updated_text,
    )
    return _render_safe_fix_prompt(proposal, review_packet=review_packet)


build_safe_fix_prompt = _build_safe_fix_prompt

# Public cross-package names; ops.lint re-exports its historical private seams.
canonical_hash = _canonical_hash
render_review_packet = _render_review_packet
render_safe_fix_prompt = _render_safe_fix_prompt
review_receipt_from_packet = _review_receipt_from_packet


def build_semantic_mutation_proposal(
    *,
    page_id: str,
    operation: str,
    expected_text: str,
    updated_text: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact durable proposal shared by deterministic page lanes."""

    return _build_safe_fix_proposal(
        page_id=page_id,
        operation=operation,
        expected_text=expected_text,
        updated_text=updated_text,
        details=details,
    )


build_safe_fix_proposal = _build_safe_fix_proposal
