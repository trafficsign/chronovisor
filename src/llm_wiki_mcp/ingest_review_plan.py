"""Pure, deterministic planning for bounded ingest review shards."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from llm_wiki_mcp.canonical_json import canonical_json_sha256_stringifying_strict
from llm_wiki_mcp.decision_lane_prompts import build_ingest_reconciliation_prompt
from llm_wiki_mcp.decision_router import (
    decision_effective_request,
    decision_request_context,
    decision_request_fingerprint_sha256,
)
from llm_wiki_mcp.ingest_schemas import (
    INGEST_FRONTIER_DECISION_SCHEMA,
    INGEST_REVIEW_SHARD_POLICY_VERSION,
    INGEST_REVIEW_SHARD_SCHEMA_VERSION,
    MAX_INGEST_REVIEW_SHARDS,
)
from llm_wiki_mcp.local_structured import preflight_structured_request


class IngestReviewShardCapacityError(RuntimeError):
    """The exact proposal cannot be reviewed in bounded local contexts."""

    def __init__(self, failure_class: str, reason: str) -> None:
        self.failure_class = failure_class
        self.reason = reason
        super().__init__(reason)


class IngestReviewBudgetExhausted(RuntimeError):
    """A local-consensus call was blocked before inference by its raw budget."""


@dataclass(frozen=True)
class IngestReviewShard:
    original_operation_indices: tuple[int, ...]
    proposal: dict[str, Any]
    prompt: str
    proposal_sha256: str
    effective_request_sha256: str
    effective_input_chars: int
    effective_input_bytes: int
    required_num_ctx: int
    selected_num_ctx: int


@dataclass(frozen=True)
class IngestReviewShardPlan:
    full_proposal_sha256: str
    manifest: dict[str, Any]
    manifest_sha256: str
    shards: tuple[IngestReviewShard, ...]


@dataclass(frozen=True)
class IngestReviewShardPlanState:
    statuses: tuple[str, ...]
    reviews: tuple[dict[str, Any] | None, ...]
    authorities: tuple[dict[str, Any] | None, ...]
    invalid_reason: str | None = None

    @property
    def current_approved_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, status in enumerate(self.statuses)
            if status == "current_approved"
        )

    @property
    def approved_shards(self) -> int:
        return len(self.current_approved_indices)


def measure_ingest_review_request(
    proposal: dict[str, Any],
    *,
    original_operation_indices: tuple[int, ...],
    config: Any,
) -> IngestReviewShard:
    prompt = build_ingest_reconciliation_prompt(proposal)
    effective_prompt, effective_system = decision_effective_request(
        prompt=prompt,
        schema=INGEST_FRONTIER_DECISION_SCHEMA,
        system=None,
        decision_lane="ingest_reconciliation",
    )
    preflight = preflight_structured_request(
        effective_prompt,
        INGEST_FRONTIER_DECISION_SCHEMA,
        system=effective_system,
        max_input_chars=config.max_input_chars,
    )
    required_num_ctx, selected_num_ctx = decision_request_context(
        config,
        prompt,
        INGEST_FRONTIER_DECISION_SCHEMA,
        None,
        "ingest_reconciliation",
    )
    if not preflight.ok:
        raise IngestReviewShardCapacityError(
            preflight.failure_class or "input_invalid",
            preflight.failure_reason or "structured review preflight failed",
        )
    if required_num_ctx > config.num_ctx:
        raise IngestReviewShardCapacityError(
            "context_window_exceeded",
            f"structured ingest review requires context {required_num_ctx}>{config.num_ctx}",
        )
    return IngestReviewShard(
        original_operation_indices=original_operation_indices,
        proposal=proposal,
        prompt=prompt,
        proposal_sha256=canonical_json_sha256_stringifying_strict(proposal),
        effective_request_sha256=decision_request_fingerprint_sha256(
            prompt=prompt,
            schema=INGEST_FRONTIER_DECISION_SCHEMA,
            system=None,
            decision_lane="ingest_reconciliation",
        ),
        effective_input_chars=sum(len(row["content"]) for row in preflight.messages),
        effective_input_bytes=preflight.input_bytes,
        required_num_ctx=required_num_ctx,
        selected_num_ctx=selected_num_ctx,
    )


def validate_ingest_shard_source_rows(
    proposal: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = proposal.get("prepared_operations")
    generated = proposal.get("local_generated_operations")
    triage = proposal.get("triage_plan")
    if not isinstance(prepared, list) or not isinstance(generated, list):
        raise IngestReviewShardCapacityError(
            "input_invalid", "oversized proposal operation arrays are invalid"
        )
    if proposal.get("failed_operation_specs"):
        raise IngestReviewShardCapacityError(
            "input_too_large",
            "oversized proposals with failed_operation_specs cannot be sharded safely",
        )
    if len(prepared) != len(generated) or not prepared:
        raise IngestReviewShardCapacityError(
            "input_too_large",
            "oversized proposal lacks a complete one-to-one operation set",
        )
    by_source: dict[int, dict[str, Any]] = {}
    for row in prepared:
        source_index = row.get("source_operation_index") if isinstance(row, dict) else None
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or source_index in by_source
            or not 0 <= source_index < len(generated)
        ):
            raise IngestReviewShardCapacityError(
                "input_invalid",
                "oversized proposal source-operation provenance is incomplete",
            )
        generated_row = generated[source_index]
        if (
            not isinstance(generated_row, dict)
            or row.get("source_operation_type") != generated_row.get("type")
            or row.get("source_filename") != generated_row.get("filename")
        ):
            raise IngestReviewShardCapacityError(
                "input_invalid",
                "oversized proposal source-operation binding is invalid",
            )
        by_source[source_index] = row
    if set(by_source) != set(range(len(generated))):
        raise IngestReviewShardCapacityError(
            "input_invalid", "oversized proposal operation coverage is incomplete"
        )
    if not isinstance(triage, list) or (triage and len(triage) != len(generated)):
        raise IngestReviewShardCapacityError(
            "input_invalid", "oversized proposal triage coverage is incomplete"
        )
    return [by_source[index] for index in range(len(generated))], generated, triage


def build_ingest_review_shard_proposal(
    proposal: dict[str, Any],
    *,
    group: tuple[int, ...],
    groups: tuple[tuple[int, ...], ...],
    shard_index: int,
    full_proposal_sha256: str,
) -> dict[str, Any]:
    prepared, generated, triage = validate_ingest_shard_source_rows(proposal)
    shard = copy.deepcopy(proposal)
    shard_prepared: list[dict[str, Any]] = []
    shard_generated: list[dict[str, Any]] = []
    shard_triage: list[dict[str, Any]] = []
    for local_index, original_index in enumerate(group):
        prepared_row = copy.deepcopy(prepared[original_index])
        prepared_row["source_operation_index"] = local_index
        shard_prepared.append(prepared_row)
        shard_generated.append(copy.deepcopy(generated[original_index]))
        if triage:
            shard_triage.append(copy.deepcopy(triage[original_index]))
    shard["prepared_operations"] = shard_prepared
    shard["local_generated_operations"] = shard_generated
    shard["triage_plan"] = shard_triage
    audit = shard.get("audit_decision")
    audit = copy.deepcopy(audit) if isinstance(audit, dict) else {}
    audit["review_shard_contract"] = {
        "schema_version": INGEST_REVIEW_SHARD_SCHEMA_VERSION,
        "policy_version": INGEST_REVIEW_SHARD_POLICY_VERSION,
        "full_proposal_sha256": full_proposal_sha256,
        "full_operation_count": len(generated),
        "shard_index": shard_index,
        "original_operation_indices": list(group),
        "complete_nonoverlap_operation_index_shards": [list(row) for row in groups],
    }
    shard["audit_decision"] = audit
    return shard


def build_ingest_review_shard_plan(
    proposal: dict[str, Any],
    *,
    config: Any,
    force_review_unit: bool = False,
) -> IngestReviewShardPlan | None:
    prepared_raw = proposal.get("prepared_operations")
    full_indices = tuple(
        range(len(prepared_raw)) if isinstance(prepared_raw, list) else range(0)
    )
    full_failure_class: str | None = None
    try:
        measure_ingest_review_request(
            proposal, original_operation_indices=full_indices, config=config
        )
        if not force_review_unit:
            return None
    except IngestReviewShardCapacityError as full_error:
        if full_error.failure_class not in {
            "input_too_large",
            "context_window_exceeded",
        }:
            raise
        full_failure_class = full_error.failure_class
    prepared, _generated, _triage = validate_ingest_shard_source_rows(proposal)
    full_indices = tuple(range(len(prepared)))
    full_proposal_sha256 = canonical_json_sha256_stringifying_strict(proposal)
    groups: tuple[tuple[int, ...], ...] = (full_indices,)
    while True:
        measured: list[IngestReviewShard] = []
        split_index: int | None = None
        split_error: IngestReviewShardCapacityError | None = None
        for shard_index, group in enumerate(groups):
            shard_proposal = build_ingest_review_shard_proposal(
                proposal,
                group=group,
                groups=groups,
                shard_index=shard_index,
                full_proposal_sha256=full_proposal_sha256,
            )
            try:
                measured.append(
                    measure_ingest_review_request(
                        shard_proposal,
                        original_operation_indices=group,
                        config=config,
                    )
                )
            except IngestReviewShardCapacityError as error:
                if error.failure_class not in {
                    "input_too_large",
                    "context_window_exceeded",
                }:
                    raise
                split_index, split_error = shard_index, error
                break
        if split_index is None:
            shards = tuple(measured)
            break
        group = groups[split_index]
        if len(group) == 1:
            assert split_error is not None
            raise IngestReviewShardCapacityError(
                split_error.failure_class,
                "one prepared ingest operation exceeds the bounded review capacity: "
                + split_error.reason,
            )
        if len(groups) >= MAX_INGEST_REVIEW_SHARDS:
            raise IngestReviewShardCapacityError(
                split_error.failure_class
                if split_error
                else (full_failure_class or "input_too_large"),
                f"ingest review requires more than {MAX_INGEST_REVIEW_SHARDS} shards",
            )
        midpoint = len(group) // 2
        groups = (
            *groups[:split_index],
            group[:midpoint],
            group[midpoint:],
            *groups[split_index + 1 :],
        )
    flattened = [index for group in groups for index in group]
    if flattened != list(full_indices) or len(flattened) != len(set(flattened)):
        raise IngestReviewShardCapacityError(
            "input_invalid", "ingest review shard partition is not exact"
        )
    manifest = {
        "schema_version": INGEST_REVIEW_SHARD_SCHEMA_VERSION,
        "policy_version": INGEST_REVIEW_SHARD_POLICY_VERSION,
        "kind": "ingest_review_shard_manifest",
        "source_key": proposal.get("source_key"),
        "full_proposal_sha256": full_proposal_sha256,
        "full_operation_count": len(full_indices),
        "review_limits": {
            name: getattr(config, name)
            for name in (
                "num_ctx",
                "min_num_ctx",
                "num_predict",
                "max_input_chars",
                "max_output_chars",
                "max_feedback_chars",
            )
        },
        "shards": [
            {
                "shard_index": index,
                "original_operation_indices": list(shard.original_operation_indices),
                "proposal_sha256": shard.proposal_sha256,
                "effective_request_sha256": shard.effective_request_sha256,
                "effective_input_chars": shard.effective_input_chars,
                "effective_input_bytes": shard.effective_input_bytes,
                "required_num_ctx": shard.required_num_ctx,
                "selected_num_ctx": shard.selected_num_ctx,
            }
            for index, shard in enumerate(shards)
        ],
    }
    return IngestReviewShardPlan(
        full_proposal_sha256=full_proposal_sha256,
        manifest=manifest,
        manifest_sha256=canonical_json_sha256_stringifying_strict(manifest),
        shards=shards,
    )
