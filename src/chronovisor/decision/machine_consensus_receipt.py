"""Canonical, subject-bound receipts for configured route consensus.

Persisted vote dictionaries are diagnostic only.  Authority comes from joining
the receipt to the exact sealed DecisionArtifactStore object published by the
DecisionRouter invocation for the subject-bound request.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import (
    canonical_sha256,
    sidecar_exclusive_lock,
)
from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.decision.decision_artifact import (
    SINGLE_MODEL_DECISION_ARTIFACT_SCHEMA,
    DecisionArtifactStore,
    default_store_root,
)
from chronovisor.decision.decision_authority import (
    SINGLE_MODEL_AUTHORITY_KIND,
    compare_semantic_authority,
    current_semantic_authority,
    returned_model_evidence_is_safe,
    semantic_authority_shape_error,
)
from chronovisor.decision.decision_lane_contracts import bind_lane_contract_request
from chronovisor.decision.decision_router import (
    DecisionRouter,
    canonical_agreement_signature,
    decision_system_with_policy,
)
from chronovisor.decision.local_structured import (
    structured_generation_policy_sha256,
    structured_request_sha256,
)

MACHINE_CONSENSUS_RECEIPT_VERSION = 2
SINGLE_MODEL_RECEIPT_VERSION = 3
DETERMINISTIC_PRODUCER_KIND = "deterministic_evidence_projection"
_SHA256_ZERO = "0" * 64
_RECEIPT_FIELDS = {
    "schema_version",
    "kind",
    "lane",
    "subject",
    "subject_sha256",
    "producer",
    "authority",
    "authority_sha256",
    "request_sha256",
    "schema_sha256",
    "system_sha256",
    "execution_fingerprint",
    "decision_artifact_seal_sha256",
    "agreement_sha256",
    "created_at",
    "previous_receipt_sha256",
    "receipt_sha256",
}
_SINGLE_RECEIPT_FIELDS = _RECEIPT_FIELDS | {"authority_kind"}
_KIND_TO_SUBJECT_KIND = {
    "gold_entry_review": "gold_entry",
    "scorer_calibration_case_review": "scorer_calibration_case",
    "search_label_candidate_review": "search_label_candidate",
}
GOLD_ENTRY_PRODUCER_POLICY_SHA256 = (
    "6aba691713aca1f85c183c2f9c3ebec5ee00eba790922494eefec7577e89dc25"
)
SCORER_CALIBRATION_PRODUCER_POLICY_SHA256 = (
    "be79e9d35592ddc14c55e01df955ad64071cde73f5c854ee36a31b1a20d4a33c"
)
SEARCH_LABEL_CANDIDATE_PRODUCER_POLICY_SHA256 = canonical_sha256(
    {
        "version": 1,
        "source": "preregistered_recall_question_and_frozen_page_bytes",
        "maximum_page_bytes": 12_000,
        "maximum_total_bytes": 32_000,
        "production_answer_used": False,
    }
)
BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256 = (
    "eeebe9d71e6a385b80d43e52b622b94fa4be510c0c6f1d31b4f13f8a06ff018e"
)
_KIND_PRODUCER_POLICY_SHA256 = {
    "gold_entry_review": GOLD_ENTRY_PRODUCER_POLICY_SHA256,
    "scorer_calibration_case_review": SCORER_CALIBRATION_PRODUCER_POLICY_SHA256,
    "search_label_candidate_review": SEARCH_LABEL_CANDIDATE_PRODUCER_POLICY_SHA256,
}

RouterFactory = Callable[[str], DecisionRouter]
AuthorityProvider = Callable[[str], tuple[dict[str, Any] | None, str | None]]


def _authority_models(authority: Mapping[str, Any]) -> list[object]:
    router = authority.get("router")
    routes = router.get("routes") if isinstance(router, Mapping) else None
    if not isinstance(routes, list):
        return []
    return [route.get("model") for route in routes if isinstance(route, Mapping)]


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], ""
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return [], "machine_consensus_ledger_json_invalid"
        if not isinstance(value, dict):
            return [], "machine_consensus_ledger_row_invalid"
        rows.append(value)
    return rows, ""


def _utc(value: object) -> str:
    if not isinstance(value, str) or not value or not value.endswith("Z"):
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return ""
    return (
        parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha_without_newline(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def search_label_candidate_packet_error(packet: object) -> str:
    """Validate one frozen RQ packet without consulting mutable Wiki pages."""

    if not isinstance(packet, Mapping) or set(packet) != {
        "schema_version",
        "packet_kind",
        "candidate_preregistration_sha256",
        "candidate",
        "page_binding",
        "evidence_chunk",
        "reference_evidence_sha256",
        "projection_policy_sha256",
    }:
        return "search_label_candidate_packet_invalid"
    candidate = packet.get("candidate")
    binding = packet.get("page_binding")
    chunk = packet.get("evidence_chunk")
    if (
        packet.get("schema_version") != 1
        or packet.get("packet_kind") != "preregistered_rq_page_evidence"
        or packet.get("projection_policy_sha256")
        != BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256
        or not isinstance(candidate, Mapping)
        or set(candidate)
        != {
            "query",
            "expected_pages",
            "negative_pages",
            "stale_pages",
            "source",
            "source_page",
            "search_eval_split",
            "split_role",
            "language",
            "kind",
            "preregistered_at",
            "candidate_preregistration_sha256",
            "page_uid",
            "content_sha256",
            "content_byte_length",
            "projection_policy_sha256",
        }
        or not isinstance(binding, Mapping)
        or set(binding)
        != {"page_id", "page_uid", "content_sha256", "content_byte_length"}
        or not isinstance(chunk, Mapping)
        or set(chunk)
        != {
            "page_id",
            "content_sha256",
            "byte_start",
            "byte_end",
            "excerpt",
            "excerpt_sha256",
            "truncated",
        }
    ):
        return "search_label_candidate_packet_invalid"
    query = candidate.get("query")
    expected_pages = candidate.get("expected_pages")
    preregistration_sha = candidate.get("candidate_preregistration_sha256")
    identity = {
        "query": query,
        "expected_pages": expected_pages,
        "source": candidate.get("source"),
        "page_uid": candidate.get("page_uid"),
        "content_sha256": candidate.get("content_sha256"),
        "content_byte_length": candidate.get("content_byte_length"),
        "projection_policy_sha256": candidate.get("projection_policy_sha256"),
        "search_eval_split": candidate.get("search_eval_split"),
    }
    excerpt = chunk.get("excerpt")
    excerpt_bytes = excerpt.encode("utf-8") if isinstance(excerpt, str) else b""
    reference = (
        f"[PAGE {chunk.get('page_id')}]\n{excerpt}" if isinstance(excerpt, str) else ""
    )
    if (
        not isinstance(query, str)
        or not query.strip()
        or not isinstance(expected_pages, list)
        or len(expected_pages) != 1
        or expected_pages != [candidate.get("source_page")]
        or candidate.get("negative_pages") != []
        or candidate.get("stale_pages") != []
        or candidate.get("source") != "recall_questions"
        or candidate.get("split_role") != "search_eval_only_not_answer_benchmark"
        or candidate.get("projection_policy_sha256")
        != BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256
        or not _utc(candidate.get("preregistered_at"))
        or not isinstance(preregistration_sha, str)
        or len(preregistration_sha) != 64
        or _sha_without_newline(identity) != preregistration_sha
        or packet.get("candidate_preregistration_sha256") != preregistration_sha
        or binding.get("page_id") != candidate.get("source_page")
        or binding.get("page_uid") != candidate.get("page_uid")
        or binding.get("content_sha256") != candidate.get("content_sha256")
        or binding.get("content_byte_length") != candidate.get("content_byte_length")
        or not isinstance(binding.get("content_byte_length"), int)
        or isinstance(binding.get("content_byte_length"), bool)
        or int(binding.get("content_byte_length") or 0) <= 0
        or chunk.get("page_id") != binding.get("page_id")
        or chunk.get("content_sha256") != binding.get("content_sha256")
        or chunk.get("byte_start") != 0
        or not isinstance(chunk.get("byte_end"), int)
        or isinstance(chunk.get("byte_end"), bool)
        or chunk.get("byte_end") != len(excerpt_bytes)
        or not 0 < len(excerpt_bytes) <= 12_000
        or chunk.get("excerpt_sha256") != hashlib.sha256(excerpt_bytes).hexdigest()
        or not isinstance(chunk.get("truncated"), bool)
        or chunk.get("truncated")
        is not (len(excerpt_bytes) < int(binding.get("content_byte_length") or 0))
        or packet.get("reference_evidence_sha256")
        != hashlib.sha256(reference.encode("utf-8")).hexdigest()
    ):
        return "search_label_candidate_packet_invalid"
    return ""


def _producer_error(producer: object, models: object, *, kind: str) -> str:
    allowed_policy = _KIND_PRODUCER_POLICY_SHA256.get(kind)
    if (
        not isinstance(producer, Mapping)
        or set(producer) != {"kind", "model", "policy_sha256"}
        or producer.get("kind") != DETERMINISTIC_PRODUCER_KIND
        or producer.get("model") is not None
        or allowed_policy is None
        or producer.get("policy_sha256") != allowed_policy
        or not isinstance(producer.get("policy_sha256"), str)
        or len(str(producer["policy_sha256"])) != 64
        or any(char not in "0123456789abcdef" for char in producer["policy_sha256"])
    ):
        return "machine_consensus_producer_invalid"
    if isinstance(models, list) and producer.get("model") in models:
        return "machine_consensus_self_vote"
    return ""


def _subject_shape_error(
    subject: Mapping[str, Any], *, expected_kind: str, producer_policy_sha256: str
) -> str:
    expected_subject_kind = _KIND_TO_SUBJECT_KIND.get(expected_kind)
    if expected_subject_kind is None:
        return "machine_consensus_receipt_kind_invalid"
    if (
        subject.get("schema_version") != 1
        or subject.get("subject_kind") != expected_subject_kind
        or subject.get("producer_kind") != DETERMINISTIC_PRODUCER_KIND
        or subject.get("producer_model") is not None
        or subject.get("producer_policy_sha256") != producer_policy_sha256
        or subject.get("production_answer_used") is not False
    ):
        return "machine_consensus_subject_shape_invalid"
    if expected_kind == "gold_entry_review":
        required_sha_fields = (
            "split_epoch_id",
            "rubric_sha256",
            "source_packet_sha256",
            "evidence_sha256",
        )
        if (
            not isinstance(subject.get("episode_id"), str)
            or not str(subject.get("episode_id") or "")
            or subject.get("split") not in {"train", "holdout", "locked-test"}
            or any(
                not isinstance(subject.get(field), str)
                or len(str(subject.get(field) or "")) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(subject.get(field) or "")
                )
                for field in required_sha_fields
            )
        ):
            return "machine_consensus_subject_shape_invalid"
    if expected_kind == "search_label_candidate_review":
        packet = subject.get("source_packet")
        if (
            set(subject)
            != {
                "schema_version",
                "subject_kind",
                "candidate_preregistration_sha256",
                "source_packet_sha256",
                "source_packet",
                "evidence_sha256",
                "producer_kind",
                "producer_model",
                "producer_policy_sha256",
                "production_answer_used",
            }
            or not isinstance(subject.get("candidate_preregistration_sha256"), str)
            or len(str(subject.get("candidate_preregistration_sha256") or "")) != 64
            or not isinstance(packet, Mapping)
            or subject.get("source_packet_sha256") != canonical_sha256(packet)
            or packet.get("candidate_preregistration_sha256")
            != subject.get("candidate_preregistration_sha256")
            or subject.get("evidence_sha256") != packet.get("reference_evidence_sha256")
            or search_label_candidate_packet_error(packet)
        ):
            return "machine_consensus_subject_shape_invalid"
    return ""


def _trusted_request_error(
    *,
    subject: Mapping[str, Any],
    prompt: str,
    schema: Mapping[str, Any],
    system: str | None,
    lane: str,
) -> str:
    if lane != "recall_answer_adjudication":
        return "machine_consensus_lane_not_supported"
    from chronovisor.decision.graph_decisions import (
        RECALL_ANSWER_ADJUDICATION_SCHEMA,
        build_recall_answer_adjudication_prompt,
    )

    expected_prompt = build_recall_answer_adjudication_prompt(
        {"subject": dict(subject), "subject_sha256": canonical_sha256(subject)}
    )
    if (
        prompt != expected_prompt
        or schema != RECALL_ANSWER_ADJUDICATION_SCHEMA
        or system is not None
    ):
        return "machine_consensus_untrusted_request"
    return ""


def _vote_manifest_error(
    value: object,
    *,
    decision: object,
    agreement_sha256: object,
    authority: Mapping[str, Any],
    artifact_proof: object,
) -> str:
    router = authority.get("router")
    routes = router.get("routes") if isinstance(router, Mapping) else None
    if authority.get("authority_kind") == SINGLE_MODEL_AUTHORITY_KIND:
        return _single_vote_manifest_error(
            value,
            decision=decision,
            agreement_sha256=agreement_sha256,
            authority=authority,
            artifact_proof=artifact_proof,
        )
    if (
        not isinstance(value, list)
        or not isinstance(routes, list)
        or len(routes) != 3
        or len(value) not in {2, 3}
    ):
        return "machine_consensus_vote_manifest_invalid"
    expected_roles = ("primary", "challenger", "tie_break")
    expected_signature = canonical_agreement_signature(
        decision,
        schema=_decision_schema_for_authority(authority),
    )
    expected_agreement = hashlib.sha256(expected_signature.encode("utf-8")).hexdigest()
    if agreement_sha256 != expected_agreement:
        return "machine_consensus_agreement_invalid"
    agreeing: list[dict[str, Any]] = []
    first_signatures: list[str | None] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            return "machine_consensus_vote_invalid"
        role = expected_roles[index]
        route = routes[index]
        if not isinstance(route, Mapping):
            return "machine_consensus_vote_authority_invalid"
        model = route.get("model")
        expected_identity = canonical_sha256(route)
        valid = raw.get("valid")
        signature = raw.get("signature_sha256")
        returned_model = raw.get("returned_model")
        if (
            set(raw)
            != {
                "role",
                "provider",
                "model",
                "route_provenance",
                "returned_model",
                "model_identity_sha256",
                "valid",
                "signature_sha256",
                "invalid_reason",
            }
            or raw.get("role") != role
            or raw.get("provider") != route.get("provider")
            or raw.get("model") != model
            or raw.get("route_provenance") != route
            or not returned_model_evidence_is_safe(returned_model)
            or raw.get("model_identity_sha256") != expected_identity
            or not isinstance(valid, bool)
        ):
            return "machine_consensus_vote_authority_invalid"
        if valid:
            if (
                (route.get("location") == "remote" and returned_model != model)
                or not isinstance(signature, str)
                or len(signature) != 64
            ):
                return "machine_consensus_vote_invalid"
            if raw.get("invalid_reason") is not None:
                return "machine_consensus_vote_invalid"
            if signature == agreement_sha256:
                agreeing.append(
                    {
                        "role": role,
                        "provider": route.get("provider"),
                        "model": model,
                        "route_provenance": dict(route),
                        "returned_model": returned_model,
                        "signature_sha256": signature,
                    }
                )
        elif (
            signature is not None
            or not isinstance(raw.get("invalid_reason"), str)
            or not raw["invalid_reason"]
        ):
            return "machine_consensus_vote_invalid"
        if index < 2:
            first_signatures.append(signature if valid else None)
    if len(value) == 2:
        if len(agreeing) != 2 or first_signatures[0] != first_signatures[1]:
            return "machine_consensus_tie_break_missing"
    else:
        if (
            first_signatures[0] is not None
            and first_signatures[0] == first_signatures[1]
        ):
            return "machine_consensus_unnecessary_tie_break"
        if not bool(value[2].get("valid")):
            return "machine_consensus_tie_break_invalid"
    if len(agreeing) < 2:
        return "machine_consensus_no_quorum"
    if artifact_proof != agreeing:
        return "machine_consensus_artifact_proof_mismatch"
    return ""


def _single_vote_manifest_error(
    value: object,
    *,
    decision: object,
    agreement_sha256: object,
    authority: Mapping[str, Any],
    artifact_proof: object,
) -> str:
    router = authority.get("router")
    routes = router.get("routes") if isinstance(router, Mapping) else None
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(routes, list)
        or len(routes) != 1
    ):
        return "machine_single_model_vote_manifest_invalid"
    route = routes[0]
    if not isinstance(route, Mapping):
        return "machine_single_model_vote_authority_invalid"
    expected_signature = canonical_agreement_signature(
        decision,
        schema=_decision_schema_for_authority(authority),
    )
    expected_agreement = hashlib.sha256(expected_signature.encode("utf-8")).hexdigest()
    if agreement_sha256 != expected_agreement:
        return "machine_consensus_agreement_invalid"
    raw = value[0]
    if not isinstance(raw, Mapping):
        return "machine_single_model_vote_invalid"
    if set(raw) != {
        "role",
        "provider",
        "model",
        "route_provenance",
        "returned_model",
        "valid",
        "signature_sha256",
        "invalid_reason",
    }:
        return "machine_single_model_vote_invalid"
    if (
        raw.get("role") != "authority"
        or raw.get("provider") != route.get("provider")
        or raw.get("model") != route.get("model")
        or raw.get("route_provenance") != route
        or not returned_model_evidence_is_safe(raw.get("returned_model"))
        or raw.get("valid") is not True
        or raw.get("invalid_reason") is not None
        or raw.get("signature_sha256") != agreement_sha256
    ):
        return "machine_single_model_vote_authority_invalid"
    expected_proof = {
        "role": route.get("role"),
        "provider": route.get("provider"),
        "model": route.get("model"),
        "route_provenance": dict(route),
        "returned_model": raw.get("returned_model"),
        "signature_sha256": raw.get("signature_sha256"),
    }
    if isinstance(artifact_proof, Mapping):
        observed_proof = {
            key: artifact_proof.get(key)
            for key in expected_proof
        }
    elif isinstance(artifact_proof, list) and len(artifact_proof) == 1:
        observed_proof = artifact_proof[0]
    else:
        observed_proof = None
    if observed_proof != expected_proof:
        return "machine_single_model_artifact_proof_mismatch"
    return ""


def _decision_schema_for_authority(authority: Mapping[str, Any]) -> Mapping[str, Any]:
    from chronovisor.decision.decision_schema_manifest import (
        background_decision_schemas,
        production_decision_schemas,
    )

    policy = authority.get("policy")
    name = policy.get("schema_name") if isinstance(policy, Mapping) else None
    schemas = {**production_decision_schemas(), **background_decision_schemas()}
    schema = schemas.get(name) if isinstance(name, str) else None
    if not isinstance(schema, Mapping):
        raise ValueError("machine consensus authority schema unavailable")
    return schema


def _approved_subject_decision_error(
    decision: object,
    *,
    expected_kind: str,
    subject_sha256: str,
) -> str:
    expected_subject_kind = _KIND_TO_SUBJECT_KIND.get(expected_kind)
    if expected_subject_kind is None:
        return "machine_consensus_receipt_kind_invalid"
    if not isinstance(decision, Mapping):
        return "machine_consensus_decision_invalid"
    if (
        decision.get("decision") != "approved"
        or decision.get("subject_kind") != expected_subject_kind
        or decision.get("subject_sha256") != subject_sha256
        or decision.get("evidence_complete") is not True
        or decision.get("reference_independent") is not True
        or decision.get("preregistered_before_evaluation") is not True
        or decision.get("split_safe") is not True
    ):
        return "machine_consensus_subject_not_approved"
    return ""


def _ledger_chain_error(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows, read_error = _read_jsonl(path)
    if read_error:
        return [], read_error
    previous = _SHA256_ZERO
    seen: set[str] = set()
    for row in rows:
        receipt_sha = row.get("receipt_sha256")
        unsigned = {key: value for key, value in row.items() if key != "receipt_sha256"}
        schema_version = row.get("schema_version")
        expected_fields = (
            _SINGLE_RECEIPT_FIELDS
            if schema_version == SINGLE_MODEL_RECEIPT_VERSION
            else _RECEIPT_FIELDS
        )
        if (
            set(row) != expected_fields
            or schema_version not in {
                MACHINE_CONSENSUS_RECEIPT_VERSION,
                SINGLE_MODEL_RECEIPT_VERSION,
            }
            or row.get("previous_receipt_sha256") != previous
            or receipt_sha != canonical_sha256(unsigned)
            or not isinstance(receipt_sha, str)
            or receipt_sha in seen
        ):
            return [], "machine_consensus_ledger_chain_invalid"
        seen.add(receipt_sha)
        previous = receipt_sha
    return rows, ""


def load_machine_consensus_receipt(
    receipt_sha256: object, *, ledger_file: Path
) -> dict[str, Any]:
    """Return one chain-validated receipt, or a fail-closed result."""

    rows, error = _ledger_chain_error(ledger_file)
    if error:
        return {"passed": False, "reason": error}
    matches = [row for row in rows if row.get("receipt_sha256") == receipt_sha256]
    if len(matches) != 1:
        return {"passed": False, "reason": "machine_consensus_receipt_missing"}
    return {"passed": True, "reason": "verified_receipt_chain", "receipt": matches[0]}


def list_machine_consensus_receipts(*, ledger_file: Path) -> dict[str, Any]:
    """Return the complete chain only when every physical JSONL row is valid."""

    rows, error = _ledger_chain_error(ledger_file)
    if error:
        return {"passed": False, "reason": error, "receipts": []}
    return {"passed": True, "reason": "verified_receipt_chain", "receipts": rows}


def validate_machine_consensus_receipt(
    receipt_sha256: object,
    *,
    expected_kind: str,
    expected_subject: Mapping[str, Any],
    expected_producer_policy_sha256: str,
    prompt: str,
    schema: Mapping[str, Any],
    system: str | None,
    lane: str,
    ledger_file: Path,
    chronovisor_root: Path,
    current_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Join one receipt to its canonical DecisionArtifactStore object."""

    loaded = load_machine_consensus_receipt(receipt_sha256, ledger_file=ledger_file)
    if loaded.get("passed") is not True:
        return loaded
    row = loaded["receipt"]
    if not isinstance(row, Mapping) or set(row) not in (
        _RECEIPT_FIELDS,
        _SINGLE_RECEIPT_FIELDS,
    ):
        return {"passed": False, "reason": "machine_consensus_receipt_fields_invalid"}
    single_mode = row.get("schema_version") == SINGLE_MODEL_RECEIPT_VERSION
    if single_mode and row.get("authority_kind") != SINGLE_MODEL_AUTHORITY_KIND:
        return {"passed": False, "reason": "machine_consensus_authority_kind_invalid"}
    authority = row.get("authority")
    if not isinstance(authority, Mapping):
        return {"passed": False, "reason": "machine_consensus_authority_invalid"}
    if single_mode and authority.get("authority_kind") != SINGLE_MODEL_AUTHORITY_KIND:
        return {"passed": False, "reason": "machine_consensus_authority_kind_invalid"}
    shape_error = semantic_authority_shape_error(authority, lane=lane)
    if shape_error is not None:
        return {"passed": False, "reason": shape_error}
    authority_now = current_authority
    if authority_now is None:
        authority_now, authority_error = current_semantic_authority(lane)
        if authority_error is not None or authority_now is None:
            return {
                "passed": False,
                "reason": authority_error or "machine_consensus_authority_unavailable",
            }
    drift_error = compare_semantic_authority(authority, authority_now, lane=lane)
    if drift_error is not None:
        return {"passed": False, "reason": drift_error}
    subject = copy.deepcopy(dict(expected_subject))
    subject_sha = canonical_sha256(subject)
    subject_shape_error = _subject_shape_error(
        subject,
        expected_kind=expected_kind,
        producer_policy_sha256=expected_producer_policy_sha256,
    )
    if subject_shape_error:
        return {"passed": False, "reason": subject_shape_error}
    trusted_request_error = _trusted_request_error(
        subject=subject,
        prompt=prompt,
        schema=schema,
        system=system,
        lane=lane,
    )
    if trusted_request_error:
        return {"passed": False, "reason": trusted_request_error}
    try:
        bound_prompt, bound_system = bind_lane_contract_request(
            lane, prompt, schema, system
        )
        effective_system = decision_system_with_policy(schema, bound_system)
        request_sha = structured_request_sha256(bound_prompt, schema, effective_system)
    except (TypeError, ValueError):
        return {"passed": False, "reason": "machine_consensus_request_invalid"}
    producer_error = _producer_error(
        row.get("producer"), _authority_models(authority), kind=expected_kind
    )
    producer = row.get("producer")
    if (
        row.get("kind") != expected_kind
        or row.get("lane") != lane
        or row.get("subject") != subject
        or row.get("subject_sha256") != subject_sha
        or row.get("authority_sha256") != canonical_sha256(authority)
        or row.get("request_sha256") != request_sha
        or row.get("schema_sha256") != canonical_sha256(schema)
        or row.get("system_sha256") != canonical_sha256(bound_system)
        or not _utc(row.get("created_at"))
        or datetime.fromisoformat(_utc(row.get("created_at")).replace("Z", "+00:00"))
        > datetime.now(UTC) + timedelta(minutes=5)
        or producer_error
        or not isinstance(producer, Mapping)
        or producer.get("policy_sha256") != expected_producer_policy_sha256
        or subject.get("producer_policy_sha256") != expected_producer_policy_sha256
    ):
        return {
            "passed": False,
            "reason": producer_error or "machine_consensus_subject_binding_invalid",
        }
    fingerprint = row.get("execution_fingerprint")
    if not isinstance(fingerprint, str):
        return {"passed": False, "reason": "machine_consensus_artifact_missing"}
    try:
        artifact = DecisionArtifactStore(default_store_root(chronovisor_root)).load(
            fingerprint
        )
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"machine_consensus_artifact_invalid:{type(exc).__name__}",
        }
    if artifact is None:
        return {"passed": False, "reason": "machine_consensus_artifact_missing"}
    identity = artifact.get("execution_identity")
    provenance = artifact.get("provenance")
    router = authority.get("router")
    vote_manifest = (
        provenance.get("vote_manifest") if isinstance(provenance, Mapping) else None
    )
    vote_error = _vote_manifest_error(
        vote_manifest,
        decision=artifact.get("decision"),
        agreement_sha256=artifact.get("agreement_sha256"),
        authority=authority,
        artifact_proof=(
            artifact.get("single_model_proof")
            if single_mode
            else artifact.get("quorum_proof")
        ),
    )
    decision = artifact.get("decision")
    decision_error = _approved_subject_decision_error(
        decision,
        expected_kind=expected_kind,
        subject_sha256=subject_sha,
    )
    if (
        not isinstance(identity, Mapping)
        or not isinstance(provenance, Mapping)
        or identity.get("request_sha256") != request_sha
        or identity.get("lane") != lane
        or identity.get("authority_sha256") != canonical_sha256(authority)
        or identity.get("router_policy_sha256") != canonical_sha256(router)
        or identity.get("generation_policy_sha256")
        != structured_generation_policy_sha256()
        or provenance.get("router_policy") != router
        or provenance.get("vote_manifest_sha256") != canonical_sha256(vote_manifest)
        or artifact.get("seal_sha256") != row.get("decision_artifact_seal_sha256")
        or artifact.get("agreement_sha256") != row.get("agreement_sha256")
        or artifact.get("decision_sha256") != canonical_sha256(decision)
        or (
            single_mode
            and artifact.get("schema") != SINGLE_MODEL_DECISION_ARTIFACT_SCHEMA
        )
        or decision_error
        or vote_error
    ):
        return {
            "passed": False,
            "reason": decision_error
            or vote_error
            or "machine_consensus_artifact_binding_invalid",
        }
    return {
        "passed": True,
        "reason": "verified_machine_consensus",
        "receipt": row,
        "artifact": artifact,
    }


def append_machine_consensus_receipt(
    *,
    kind: str,
    subject: Mapping[str, Any],
    producer_policy_sha256: str,
    prompt: str,
    schema: Mapping[str, Any],
    system: str | None,
    lane: str,
    ledger_file: Path,
    chronovisor_root: Path,
    router_factory: RouterFactory | None = None,
    authority_provider: AuthorityProvider = current_semantic_authority,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Own router invocation and durable publication; never accepts vote dicts."""

    receipt_created_at = _utc(created_at or _now())
    if not receipt_created_at or datetime.fromisoformat(
        receipt_created_at.replace("Z", "+00:00")
    ) > datetime.now(UTC) + timedelta(minutes=5):
        return {"status": "held", "reason": "machine_consensus_created_at_invalid"}
    router = (
        router_factory(lane)
        if router_factory is not None
        else DecisionRouter(
            decision_lane=lane,
            audit_role="recall_machine_consensus",
            require_adopted=True,
        )
    )

    def resolve_authority(
        *, refresh: bool = False
    ) -> tuple[dict[str, Any] | None, str | None]:
        if authority_provider is current_semantic_authority:
            if refresh:
                return current_semantic_authority(lane)
            return current_semantic_authority(
                lane,
                router=router,
            )
        return authority_provider(lane)

    authority, authority_error = resolve_authority()
    if authority_error is not None or authority is None:
        return {
            "status": "waiting",
            "reason": authority_error or "machine_consensus_authority_unavailable",
        }
    shape_error = semantic_authority_shape_error(authority, lane=lane)
    producer = {
        "kind": DETERMINISTIC_PRODUCER_KIND,
        "model": None,
        "policy_sha256": producer_policy_sha256,
    }
    producer_error = _producer_error(producer, _authority_models(authority), kind=kind)
    subject_shape_error = _subject_shape_error(
        subject,
        expected_kind=kind,
        producer_policy_sha256=producer_policy_sha256,
    )
    if shape_error is not None or producer_error or subject_shape_error:
        return {
            "status": "held",
            "reason": producer_error or subject_shape_error or shape_error,
        }
    trusted_request_error = _trusted_request_error(
        subject=subject,
        prompt=prompt,
        schema=schema,
        system=system,
        lane=lane,
    )
    if trusted_request_error:
        return {"status": "held", "reason": trusted_request_error}
    result = router.decide(prompt, schema, system=system, decision_lane=lane)
    if not result.ok:
        waiting_failures = {
            "adoption_artifact_invalid",
            "local_resource_quarantined",
            "decision_artifact_invalid",
        }
        return {
            "status": "waiting" if result.failure_class in waiting_failures else "held",
            "reason": result.failure_class or result.quarantine_reason or "no_quorum",
        }
    subject_payload = copy.deepcopy(dict(subject))
    decision_error = _approved_subject_decision_error(
        result.value,
        expected_kind=kind,
        subject_sha256=canonical_sha256(subject_payload),
    )
    if decision_error:
        return {"status": "held", "reason": decision_error}
    current, current_error = resolve_authority(refresh=True)
    drift_error = current_error or compare_semantic_authority(
        authority, current, lane=lane
    )
    residency = result.residency if isinstance(result.residency, Mapping) else {}
    fingerprint = residency.get("execution_fingerprint")
    artifact_seal = residency.get("decision_artifact_seal_sha256")
    if drift_error is not None or not isinstance(fingerprint, str):
        return {
            "status": "waiting",
            "reason": drift_error or "canonical_decision_artifact_missing",
        }
    bound_prompt, bound_system = bind_lane_contract_request(
        lane, prompt, schema, system
    )
    effective_system = decision_system_with_policy(schema, bound_system)
    single_mode = authority.get("authority_kind") == SINGLE_MODEL_AUTHORITY_KIND
    with sidecar_exclusive_lock(ledger_file):
        rows, chain_error = _ledger_chain_error(ledger_file)
        if chain_error:
            raise ValueError(chain_error)
        existing = [
            row
            for row in rows
            if row.get("kind") == kind
            and row.get("lane") == lane
            and row.get("subject_sha256") == canonical_sha256(subject_payload)
            and row.get("authority_sha256") == canonical_sha256(authority)
            and row.get("execution_fingerprint") == fingerprint
        ]
        if len(existing) > 1:
            raise ValueError("machine_consensus_subject_receipt_duplicate")
        if existing:
            row = existing[0]
        else:
            previous = str(rows[-1]["receipt_sha256"]) if rows else _SHA256_ZERO
            row = {
                "schema_version": (
                    SINGLE_MODEL_RECEIPT_VERSION
                    if single_mode
                    else MACHINE_CONSENSUS_RECEIPT_VERSION
                ),
                "kind": kind,
                "lane": lane,
                "subject": subject_payload,
                "subject_sha256": canonical_sha256(subject_payload),
                "producer": producer,
                "authority": copy.deepcopy(dict(authority)),
                "authority_sha256": canonical_sha256(authority),
                "request_sha256": structured_request_sha256(
                    bound_prompt, schema, effective_system
                ),
                "schema_sha256": canonical_sha256(schema),
                "system_sha256": canonical_sha256(bound_system),
                "execution_fingerprint": fingerprint,
                "decision_artifact_seal_sha256": artifact_seal,
                "agreement_sha256": result.agreement_sha256,
                "created_at": receipt_created_at,
                "previous_receipt_sha256": previous,
            }
            if single_mode:
                row["authority_kind"] = SINGLE_MODEL_AUTHORITY_KIND
            row["receipt_sha256"] = canonical_sha256(row)
            append_jsonl_durable(ledger_file, [row], sort_keys=True)
    checked = validate_machine_consensus_receipt(
        row["receipt_sha256"],
        expected_kind=kind,
        expected_subject=subject_payload,
        expected_producer_policy_sha256=producer_policy_sha256,
        prompt=prompt,
        schema=schema,
        system=system,
        lane=lane,
        ledger_file=ledger_file,
        chronovisor_root=chronovisor_root,
        current_authority=current,
    )
    if checked.get("passed") is not True:
        raise ValueError(
            str(checked.get("reason") or "machine consensus read-back failed")
        )
    try:
        router.audit_store.append(
            {
                "kind": "machine_consensus_receipt",
                "request_sha256": row["request_sha256"],
                "role": router.audit_role,
                "decision_lane": lane,
                "status": "accepted",
                "execution_fingerprint": row["execution_fingerprint"],
                "decision_artifact_seal_sha256": row["decision_artifact_seal_sha256"],
                "agreement_sha256": row["agreement_sha256"],
                "receipt_sha256": row["receipt_sha256"],
                "schema_sha256": row["schema_sha256"],
            }
        )
    except Exception:
        # Receipt publication is authoritative; telemetry remains best-effort.
        pass
    return {
        "status": "accepted",
        "reason": "verified_machine_consensus",
        "receipt": row,
    }


__all__ = [
    "BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256",
    "DETERMINISTIC_PRODUCER_KIND",
    "GOLD_ENTRY_PRODUCER_POLICY_SHA256",
    "MACHINE_CONSENSUS_RECEIPT_VERSION",
    "SINGLE_MODEL_RECEIPT_VERSION",
    "SCORER_CALIBRATION_PRODUCER_POLICY_SHA256",
    "SEARCH_LABEL_CANDIDATE_PRODUCER_POLICY_SHA256",
    "append_machine_consensus_receipt",
    "load_machine_consensus_receipt",
    "list_machine_consensus_receipts",
    "search_label_candidate_packet_error",
    "validate_machine_consensus_receipt",
]
