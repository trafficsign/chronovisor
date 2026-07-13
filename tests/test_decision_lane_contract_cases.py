from __future__ import annotations

import importlib
import hashlib
import inspect
import json
from collections import Counter

import pytest

from llm_wiki_mcp.decision_lane_contract_cases import (
    CASES_PER_MODEL_BACKED_LANE,
    decision_lane_contract_case_manifest,
    decision_lane_contract_case_manifest_sha256,
    decision_lane_contract_case_specs,
)
from llm_wiki_mcp.decision_lane_contracts import (
    LANE_CONTRACT_CASE_VERSION,
    LANE_CONTRACT_POLICY_VERSION,
    LANE_CONTRACT_SOURCE,
    LANE_PROMPT_POLICY_VERSIONS,
    bind_lane_contract_request,
    lane_contract_manifest,
    lane_contract_manifest_sha256,
    model_backed_lane_names,
)
from llm_wiki_mcp.decision_lane_prompts import (
    INGEST_REPAIR_OPTION_ID_RE,
    INGEST_REPAIR_OPTION_POLICY_VERSION,
    build_ingest_reconciliation_prompt,
    build_raw_replay_reconciliation_prompt,
)
from llm_wiki_mcp.decision_router import (
    decision_context_buckets,
    decision_request_context,
    decision_request_fingerprint_sha256,
)
from llm_wiki_mcp.decision_policy import DECISION_POLICIES
from llm_wiki_mcp.decision_schema_manifest import production_decision_schemas
from llm_wiki_mcp.runtime_config import DecisionRouterConfig


def test_contract_cases_cover_every_model_backed_lane_independently() -> None:
    cases = decision_lane_contract_case_specs()
    expected_lanes = set(model_backed_lane_names())
    counts = Counter(case.lane for case in cases)

    assert len(cases) == 100
    assert set(counts) == expected_lanes
    assert min(counts.values()) == CASES_PER_MODEL_BACKED_LANE
    assert counts["content_correction_classification"] == 6
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(case.case_id.startswith("lane-contract-v20:") for case in cases)


def test_contract_cases_bind_to_the_exact_live_lane_envelope() -> None:
    schemas = production_decision_schemas()
    for case in decision_lane_contract_case_specs():
        prompt, system = bind_lane_contract_request(
            case.lane,
            case.prompt,
            schemas[case.schema_name],
            case.system,
        )
        prompt_policy_version = LANE_PROMPT_POLICY_VERSIONS[case.lane]
        assert f'policy="{prompt_policy_version}"' in prompt
        assert f'lane="{case.lane}"' in prompt
        assert f"LLM_WIKI_LANE_CONTRACT_POLICY={prompt_policy_version}" in system
        assert f"LLM_WIKI_LANE={case.lane}" in system
        assert case.prompt in prompt


def test_one_lane_prompt_version_bump_preserves_other_lane_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schemas = production_decision_schemas()
    cases = decision_lane_contract_case_specs()

    def effective_requests() -> dict[str, str]:
        return {
            case.case_id: decision_request_fingerprint_sha256(
                prompt=case.prompt,
                schema=schemas[case.schema_name],
                system=case.system,
                decision_lane=case.lane,
            )
            for case in cases
        }

    baseline_manifest = lane_contract_manifest()
    baseline_manifest_sha256 = lane_contract_manifest_sha256()
    baseline_requests = effective_requests()
    bumped_lane = "recall_improvement"
    monkeypatch.setitem(
        LANE_PROMPT_POLICY_VERSIONS,
        bumped_lane,
        LANE_PROMPT_POLICY_VERSIONS[bumped_lane] + 1,
    )

    changed_manifest = lane_contract_manifest()
    changed_requests = effective_requests()

    assert lane_contract_manifest_sha256() != baseline_manifest_sha256
    assert {
        lane
        for lane in model_backed_lane_names()
        if changed_manifest[lane]["contract_sha256"]
        != baseline_manifest[lane]["contract_sha256"]
    } == {bumped_lane}
    assert {
        case.lane
        for case in cases
        if changed_requests[case.case_id] != baseline_requests[case.case_id]
    } == {bumped_lane}


def test_shared_schema_lanes_keep_distinct_prompt_contracts() -> None:
    cases = decision_lane_contract_case_specs()
    by_lane = {
        lane: [case for case in cases if case.lane == lane]
        for lane in model_backed_lane_names()
    }
    safe_lanes = {
        "entity_backfill": (
            "backfill_entities_frontmatter",
            '"proposal_generator_version": 2',
        ),
        "lint_safe_semantic_mutation": ("broken_link_", None),
        "metadata_backfill": (
            "backfill_recall_metadata",
            '"proposal_generator_version": 2',
        ),
        "page_normalize": (
            "resolve_nested_frontmatter_conflict",
            '"policy": "outer scalar wins; lists are outer-first stable unions"',
        ),
    }
    assert {DECISION_POLICIES[lane].schema_name for lane in safe_lanes} == {
        "lint_safe_semantic_mutation"
    }
    for lane, (operation_marker, version_marker) in safe_lanes.items():
        combined = "\n".join(case.prompt for case in by_lane[lane])
        assert operation_marker in combined
        if version_marker is not None:
            assert version_marker in combined

    generic_lanes = {
        "recall_auto_apply",
        "recall_calibration",
        "recall_improvement",
        "search_self_tune",
    }
    assert {DECISION_POLICIES[lane].schema_name for lane in generic_lanes} == {
        "generic_decision"
    }
    for lane in generic_lanes:
        own_prompts = {case.prompt for case in by_lane[lane]}
        other_prompts = {
            case.prompt
            for other_lane in generic_lanes - {lane}
            for case in by_lane[other_lane]
        }
        assert own_prompts.isdisjoint(other_prompts)


def test_contract_case_serialization_contains_no_model_self_label() -> None:
    rows = [case.as_dict() for case in decision_lane_contract_case_specs()]
    assert all("source" not in row for row in rows)
    assert all("evidence_provenance" not in row for row in rows)


def test_classification_contract_outcomes_follow_production_decision_semantics() -> (
    None
):
    cases = [
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "content_correction_classification"
    ]

    rejected = [
        case.expected for case in cases if case.expected["decision"] == "rejected"
    ]
    ambiguous = [
        case.expected
        for case in cases
        if case.expected["classification"] == "ambiguous"
    ]

    assert rejected
    assert all(
        expected["classification"] == "none" and expected["ignored_pages"] == []
        for expected in rejected
    )
    assert ambiguous
    assert all(expected["decision"] == "needs_retry" for expected in ambiguous)

    by_classification = {case.expected["classification"]: case.prompt for case in cases}
    assert "never true" in by_classification["page_fact_wrong"]
    assert "used to be correct" in by_classification["outdated"]
    assert "do not change memory yet" in by_classification["ambiguous"]
    assert "Thanks, that answers my question" in by_classification["none"]


def test_content_review_contracts_distinguish_three_nonapproval_evidence_states() -> (
    None
):
    cases = {
        case.ordinal: case
        for case in decision_lane_contract_case_specs()
        if case.lane == "content_correction_review"
    }

    def block(case, name: str):
        start = f"<{name}>\n"
        end = f"\n</{name}>"
        return json.loads(case.prompt.split(start, 1)[1].split(end, 1)[0])

    contradicted = cases[2]
    contradicted_event = block(contradicted, "CORRECTION_EVENT_UNTRUSTED_JSON")
    contradicted_evidence = block(
        contradicted, "CANDIDATE_PAGE_EVIDENCE_UNTRUSTED_JSON"
    )
    contradicted_mutations = block(contradicted, "PREPARED_MUTATIONS_UNTRUSTED_JSON")
    assert contradicted.expected["decision"] == "rejected"
    assert "64GB" in contradicted_event["correction_prompt"]
    assert contradicted_evidence[0]["page_id"] == "hardware-profile"
    assert contradicted_mutations[0]["replacements"][0]["new_text"].endswith("32GB.")

    unavailable = cases[3]
    unavailable_event = block(unavailable, "CORRECTION_EVENT_UNTRUSTED_JSON")
    unavailable_preflight = block(unavailable, "DETERMINISTIC_PREFLIGHT_JSON")
    assert unavailable.expected["decision"] == "needs_retry"
    assert unavailable_event["candidate_pages"] == [
        "hardware-profile",
        "missing-secondary-profile",
    ]
    assert unavailable_preflight["status"] == "needs_retry"
    assert "missing-secondary-profile" in unavailable_preflight["reason"]

    temporal = cases[5]
    temporal_evidence = block(temporal, "CANDIDATE_PAGE_EVIDENCE_UNTRUSTED_JSON")
    temporal_mutations = block(temporal, "PREPARED_MUTATIONS_UNTRUSTED_JSON")
    temporal_triage = block(temporal, "AUTHORITATIVE_TRIAGE_REVIEW_JSON")
    assert temporal.expected["decision"] == "rejected"
    assert temporal_evidence[0]["page_id"] == "hardware-history"
    assert "16GB in 2024" in temporal_evidence[0]["content"]
    assert temporal_mutations[0]["page_id"] == "hardware-history"
    assert temporal_triage["classification"] == "outdated"
    assert len({case.prompt for case in cases.values()}) == len(cases)


def test_lane_contract_case_source_version_tracks_the_resealed_cases() -> None:
    assert LANE_CONTRACT_POLICY_VERSION == 9
    assert INGEST_REPAIR_OPTION_POLICY_VERSION == 2
    assert LANE_CONTRACT_CASE_VERSION == 20
    assert LANE_CONTRACT_SOURCE == "deterministic_lane_contract_v20"
    assert set(LANE_PROMPT_POLICY_VERSIONS) == set(model_backed_lane_names())
    assert LANE_PROMPT_POLICY_VERSIONS["ingest_reconciliation"] == 11
    assert LANE_PROMPT_POLICY_VERSIONS["raw_replay_reconciliation"] == 8
    assert {
        version
        for lane, version in LANE_PROMPT_POLICY_VERSIONS.items()
        if lane not in {"ingest_reconciliation", "raw_replay_reconciliation"}
    } == {7}


def test_raw_replay_prompt_treats_process_missing_as_observed_after_receipt() -> None:
    prompt = build_raw_replay_reconciliation_prompt(
        {
            "claims": [
                {
                    "operation": "create",
                    "page_id": "durable-page",
                    "receipt": "verified",
                }
            ],
            "runtime_status": {"state": "process_missing"},
        }
    )

    assert "process_missing is an observed state, not missing" in prompt
    assert "receipt=verified and a concrete mutation" in prompt


def _safe_mutation_proposal(case) -> dict:
    payload = case.prompt.split("Exact proposal:\n", 1)[1]
    proposal, _end = json.JSONDecoder().raw_decode(payload)
    assert isinstance(proposal, dict)
    packet_payload = case.prompt.split(
        "Complete deterministic review packet (never prefix/suffix truncated):\n",
        1,
    )[1]
    packet, _end = json.JSONDecoder().raw_decode(packet_payload)
    assert isinstance(packet, dict)
    proposal["review_packet"] = packet
    return proposal


def test_entity_contract_model_coverage_contains_only_reachable_terminals() -> None:
    cases = [
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "entity_backfill"
    ]

    assert {case.expected["decision"] for case in cases} == {
        "approved",
        "rejected",
    }
    for case in cases:
        proposal = _safe_mutation_proposal(case)
        assert proposal["unified_diff_truncated"] is False
        assert proposal["details"]["added_entities"]
        assert all(
            evidence["matched_aliases"]
            for evidence in proposal["details"]["alias_evidence"]
        )
    homonym = next(case for case in cases if case.ordinal == 3)
    homonym_proposal = _safe_mutation_proposal(homonym)
    assert homonym.expected["decision"] == "rejected"
    assert homonym_proposal["details"]["added_entities"] == ["apple-inc"]
    assert "Apple pie recipe" in homonym.prompt


def test_entity_contract_builder_fails_if_production_preflight_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import decision_lane_contract_cases, entities

    monkeypatch.setattr(
        entities,
        "validate_entity_backfill_proposal",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(ValueError, match="cannot reach production model"):
        decision_lane_contract_cases._entity_backfill_cases()


def test_entity_preflight_rejects_alias_incomplete_fixture_before_model() -> None:
    from llm_wiki_mcp.entities import validate_entity_backfill_proposal

    case = next(
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "entity_backfill"
    )
    proposal = _safe_mutation_proposal(case)
    proposal["details"]["alias_evidence"][0]["matched_aliases"] = []

    assert validate_entity_backfill_proposal(proposal) is False


def test_shared_page_mutation_contracts_use_reachable_evidence_states() -> None:
    cases = {
        (case.lane, case.ordinal): case
        for case in decision_lane_contract_case_specs()
        if case.lane
        in {
            "lint_safe_semantic_mutation",
            "metadata_backfill",
            "page_normalize",
        }
    }

    proposals = [_safe_mutation_proposal(case) for case in cases.values()]
    assert all(
        proposal["review_packet"]["mode"] in {"full", "changed_spans"}
        and proposal["details"]["review_receipt"]["complete"] is True
        and proposal["details"]["review_receipt"]["truncated"] is False
        for proposal in proposals
    )
    assert all(case.expected["decision"] != "needs_retry" for case in cases.values())

    metadata_quarantine = cases[("metadata_backfill", 6)]
    metadata_proposal = _safe_mutation_proposal(metadata_quarantine)
    assert metadata_quarantine.expected["decision"] == "quarantined"
    assert metadata_proposal["details"]["identity_preflight"]["status"] == (
        "unresolved_conflict"
    )

    lint_plaintext = cases[("lint_safe_semantic_mutation", 5)]
    lint_proposal = _safe_mutation_proposal(lint_plaintext)
    assert lint_plaintext.expected["decision"] == "approved"
    assert lint_proposal["operation"] == "broken_link_plaintext"
    assert lint_proposal["details"]["target_lookup_receipt"]["target_absent"] is True
    assert lint_proposal["review_packet"]["mode"] == "full"

    normalize_quarantine = cases[("page_normalize", 4)]
    normalize_large = cases[("page_normalize", 6)]
    quarantine_proposal = _safe_mutation_proposal(normalize_quarantine)
    large_proposal = _safe_mutation_proposal(normalize_large)
    assert normalize_quarantine.expected["decision"] == "quarantined"
    assert cases[("page_normalize", 3)].expected["decision"] == "approved"
    assert set(quarantine_proposal["details"]["conflicts"]) == {"permalink"}
    assert quarantine_proposal["details"]["identity_preflight"]["field"] == "permalink"
    assert normalize_large.expected["decision"] == "rejected"
    assert large_proposal["review_packet"]["mode"] == "changed_spans"
    assert large_proposal["details"]["review_receipt"]["repacket"] is True
    assert "Final trusted check after reading the complete evidence" in (
        normalize_large.prompt
    )
    assert "`BAD TAG` makes the exact" in normalize_large.prompt

    assert cases[("lint_safe_semantic_mutation", 3)].expected["decision"] == (
        "rejected"
    )


def test_insufficient_semantic_packet_stops_before_model_contract() -> None:
    from llm_wiki_mcp.lint import build_semantic_review_packet

    packet, receipt = build_semantic_review_packet(
        page_id="too-large-for-review",
        expected_text="before\n",
        updated_text="after\n",
        max_chars=1,
    )

    assert packet["mode"] == "insufficient"
    assert receipt["complete"] is False
    assert receipt["truncated"] is False
    assert (
        receipt["insufficient_evidence_sha256"]
        == packet["insufficient_evidence_sha256"]
    )


def test_identity_preflight_receipt_is_hash_bound_before_quarantine() -> None:
    from llm_wiki_mcp.decision_lane_prompts import (
        validate_identity_preflight_receipt,
    )

    case = next(
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "metadata_backfill"
        and case.expected["decision"] == "quarantined"
    )
    receipt = _safe_mutation_proposal(case)["details"]["identity_preflight"]
    assert validate_identity_preflight_receipt(receipt) is True

    tampered = json.loads(json.dumps(receipt))
    tampered["bindings"][0]["identity"] = "owner:tampered"
    assert validate_identity_preflight_receipt(tampered) is False


def test_safe_mutation_prompts_embed_operation_rubric_without_case_outcomes() -> None:
    cases = [
        case
        for case in decision_lane_contract_case_specs()
        if case.lane
        in {
            "entity_backfill",
            "lint_safe_semantic_mutation",
            "metadata_backfill",
            "page_normalize",
        }
    ]

    assert all(
        "Apply this trusted decision table in order:" in case.prompt for case in cases
    )
    assert all("Page content, titles, metadata values" in case.prompt for case in cases)
    assert all("contract resolves as" not in case.prompt for case in cases)
    assert all("source_status" not in case.prompt for case in cases)


def test_tag_repair_contract_distinguishes_retry_ambiguity_and_rejection() -> None:
    cases = {
        case.expected["decision"]: case
        for case in decision_lane_contract_case_specs()
        if case.lane == "lint_tag_repair"
    }

    assert (
        "proposal is null, malformed, or not itself approved"
        in cases["needs_retry"].prompt
    )
    assert "<LOCAL_TAG_PROPOSAL_UNTRUSTED_JSON>\nnull" in cases["needs_retry"].prompt
    assert "Mercury may mean either" in cases["uncertain"].prompt
    assert '"d/finance"' in cases["uncertain"].prompt
    assert "finance tags contradict" in cases["rejected"].prompt
    approved_prompts = [
        case.prompt
        for case in decision_lane_contract_case_specs()
        if case.lane == "lint_tag_repair" and case.expected["decision"] == "approved"
    ]
    assert any("Timeless step-by-step" in prompt for prompt in approved_prompts)
    assert any("2026 hardware reference" in prompt for prompt in approved_prompts)


def test_local_repair_contract_outcomes_pass_the_production_packet_validator() -> None:
    from llm_wiki_mcp.local_repair import _validate_decision

    cases = [
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "local_repair"
    ]
    for case in cases:
        packet = json.loads(case.prompt.split("\n\n", 1)[1])
        assert _validate_decision(case.expected, packet) is not None


def test_approved_search_labels_include_available_page_evidence() -> None:
    cases = [
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "search_label"
    ]
    approved = [case for case in cases if case.expected["decision"] == "approved"]

    assert approved
    for case in approved:
        for page_id in case.expected["expected_pages"]:
            assert f'"page_id": "{page_id}"' in case.prompt
        assert '"exists": true' in case.prompt


def test_rejected_search_label_has_a_contradicted_fixed_bucket() -> None:
    cases = [
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "search_label" and case.expected["decision"] == "rejected"
    ]

    assert len(cases) == 1
    prompt = cases[0].prompt
    assert '"expected_pages": [\n      "gpu-memory-sizing"' in prompt
    assert '"query": "same-session JSON repair validator feedback"' in prompt
    assert "Do not repair a wrong label by moving page ids between buckets." in prompt
    assert "Return all three arrays empty" in prompt


def test_empty_ambiguous_search_label_is_never_approved() -> None:
    case = next(
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "search_label" and case.expected["decision"] == "uncertain"
    )

    assert '"query": "Mercury"' in case.prompt
    assert '"expected_pages": []' in case.prompt
    assert (
        "no candidate labels is uncertain, not approved or needs_retry" in case.prompt
    )


def test_missing_search_label_evidence_requires_retry() -> None:
    case = next(
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "search_label" and case.expected["decision"] == "needs_retry"
    )

    assert '"exists": false' in case.prompt
    assert "Missing candidate evidence is needs_retry" in case.prompt


def test_recall_auto_apply_contracts_use_the_production_proposal_shape() -> None:
    cases = [
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "recall_auto_apply"
    ]
    required = {
        "schema_version",
        "apply_key",
        "action_type",
        "effective_action",
        "normalize_key",
        "source_ref",
        "expected_pages",
        "action_payload",
        "missing_signal",
        "prompt",
        "local_validation",
        "page_evidence",
    }

    assert len(cases) == CASES_PER_MODEL_BACKED_LANE
    for case in cases:
        proposal = json.loads(case.prompt.split("Proposal:\n", 1)[1])
        assert set(proposal) == required
        assert isinstance(proposal["effective_action"], str)
        assert isinstance(proposal["action_payload"], dict)
        assert isinstance(proposal["page_evidence"], dict)
        assert proposal["local_validation"]["status"] in {
            "dry_run",
            "fallback_dry_run",
        }


def test_recall_improvement_contracts_are_post_gate_production_candidates() -> None:
    cases = [
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "recall_improvement"
    ]

    assert len(cases) == CASES_PER_MODEL_BACKED_LANE
    assert {case.expected["decision"] for case in cases} == {"approved", "rejected"}
    for case in cases:
        assert "This is a routine local decision, not a frontier review." in case.prompt
        assert "high risk, four-or-more changed fields" in case.prompt
        payload = json.loads(case.prompt.split("Payload:\n", 1)[1])
        best = payload["best"]
        assert best["status"] == "candidate_pass"
        assert best["blockers"] == []
        assert set(best) == {
            "proposal",
            "status",
            "applied_fields",
            "candidate_policy",
            "dev",
            "holdout",
            "checks",
            "blockers",
        }
        assert best["applied_fields"]
        assert all(
            best["checks"][name] is True
            for name in (
                "dev_improved",
                "holdout_score_ok",
                "holdout_recall_ok",
                "holdout_waste_ok",
                "latency_ok",
            )
        )


def test_recall_improvement_regression_is_stopped_before_model_audit() -> None:
    from llm_wiki_mcp.recall_improvement import _gate_candidate

    accepted, checks = _gate_candidate(
        baseline_dev={"score": 0.70, "metrics": {}},
        baseline_holdout={
            "score": 0.94,
            "metrics": {
                "recall_at_3": 0.94,
                "waste_injection_rate": 0.10,
                "latency_ms": {"p95": 900.0},
            },
        },
        candidate_dev={"score": 0.82, "metrics": {}},
        candidate_holdout={
            "score": 0.72,
            "metrics": {
                "recall_at_3": 0.70,
                "waste_injection_rate": 0.32,
                "latency_ms": {"p95": 2200.0},
            },
        },
        min_improvement=0.05,
    )

    assert accepted is False
    assert checks["dev_improved"] is True
    assert checks["holdout_score_ok"] is False
    assert checks["holdout_recall_ok"] is False
    assert checks["holdout_waste_ok"] is False
    assert checks["latency_ok"] is False


def test_ingest_contracts_use_byte_exact_production_proposals() -> None:
    cases = [
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "ingest_reconciliation"
    ]
    required = {
        "schema_version",
        "kind",
        "source_key",
        "source_raw",
        "raw_content",
        "raw_sha256",
        "raw_keywords",
        "local_disposition",
        "triage_plan",
        "failed_operation_specs",
        "local_generated_operations",
        "prepared_operations",
        "link_reconciliation",
        "audit_decision",
    }
    prepared_required = {
        "op_type",
        "path",
        "page_id",
        "preimage_exists",
        "previous_text",
        "previous_sha256",
        "proposed_text",
        "proposed_sha256",
        "new_tags",
    }

    assert len(cases) > CASES_PER_MODEL_BACKED_LANE
    for case in cases:
        proposal = json.loads(case.prompt.split("Exact proposal:\n", 1)[1])
        assert set(proposal) == required
        assert proposal["kind"] == "ingest_semantic_mutation_proposal"
        assert (
            proposal["raw_sha256"]
            == hashlib.sha256(proposal["raw_content"].encode("utf-8")).hexdigest()
        )
        assert proposal["local_disposition"] in {
            "operations_available",
            "triage_no_operations",
            "all_generation_failed",
            "partial_generation_failed",
        }
        for prepared in proposal["prepared_operations"]:
            assert set(prepared) == prepared_required
            assert (
                prepared["previous_sha256"]
                == hashlib.sha256(prepared["previous_text"].encode("utf-8")).hexdigest()
            )
            assert (
                prepared["proposed_sha256"]
                == hashlib.sha256(prepared["proposed_text"].encode("utf-8")).hexdigest()
            )

    grounded_apply = next(
        case
        for case in cases
        if case.expected["decision"] == "apply_available"
        and case.expected["failed_operations_disposition"] == "none"
    )
    apply_proposal = json.loads(grounded_apply.prompt.split("Exact proposal:\n", 1)[1])
    [prepared] = apply_proposal["prepared_operations"]
    assert apply_proposal["raw_content"] in prepared["proposed_text"]

    repair = next(
        case
        for case in cases
        if case.expected.get("invalid_tags")
        and case.expected.get("replacement_operations")
    )
    repair_proposal = json.loads(repair.prompt.split("Exact proposal:\n", 1)[1])
    [replacement] = repair.expected["replacement_operations"]
    preflight = json.loads(
        repair.prompt.split("<DETERMINISTIC_INGEST_REPAIR_PREFLIGHT_JSON>\n", 1)[
            1
        ].split("\n</DETERMINISTIC_INGEST_REPAIR_PREFLIGHT_JSON>", 1)[0]
    )
    assert repair.expected["decision"] == "retry"
    assert repair.expected["invalid_tags"] == ["d/finance"]
    assert preflight["status"] == "none"
    assert preflight["tag_authority"] == "local_quorum_only"
    assert (
        preflight["repair_option_policy_version"] == INGEST_REPAIR_OPTION_POLICY_VERSION
    )
    assert preflight["deterministic_repair_option_id"] is None
    assert preflight["replacement_operations"] == []
    assert "invalid_tags" not in preflight
    selected_option = next(
        option
        for option in preflight["semantic_tag_options"]
        if option["invalid_tags"] == repair.expected["invalid_tags"]
    )
    assert {
        tuple(option["invalid_tags"]) for option in preflight["semantic_tag_options"]
    } == {("d/configuration",), ("d/finance",)}
    assert all(
        set(option)
        == {
            "repair_option_id",
            "filename",
            "invalid_tags",
            "replacement_operations",
        }
        for option in preflight["semantic_tag_options"]
    )
    assert (
        selected_option["replacement_operations"]
        == repair.expected["replacement_operations"]
    )
    assert INGEST_REPAIR_OPTION_ID_RE.fullmatch(selected_option["repair_option_id"])
    assert replacement["filename"] in {
        operation["filename"]
        for operation in repair_proposal["local_generated_operations"]
    }
    assert "d/finance" not in replacement["content"]
    assert repair_proposal["raw_content"] in replacement["content"]


def test_ingest_prompt_orders_quarantine_retry_and_safe_apply() -> None:
    prompts = [
        case.prompt
        for case in decision_lane_contract_case_specs()
        if case.lane == "ingest_reconciliation"
    ]

    assert prompts
    assert all("Apply this decision table in order:" in prompt for prompt in prompts)
    assert all(
        "choose quarantined with failed_operations_disposition=retry_required" in prompt
        for prompt in prompts
    )
    assert all("Return exactly one repair_option_id" in prompt for prompt in prompts)
    assert all(
        "Do not return\ninvalid_tags or replacement_operations yourself" in prompt
        for prompt in prompts
    )


@pytest.mark.parametrize(
    ("raw", "title", "body"),
    [
        (
            "The microfinance policy uses a local Ollama model.",
            "Microfinance policy",
            "The microfinance policy uses a local Ollama model.",
        ),
        (
            "The runtime uses a local Ollama model.",
            "Finance operations runtime",
            "The runtime uses a local Ollama model.",
        ),
        (
            "The runtime uses a local Ollama model.",
            "Local runtime",
            (
                "The runtime uses a local Ollama model. This page also describes "
                "finance policy."
            ),
        ),
    ],
    ids=["raw-synonym", "title-support", "body-contradiction"],
)
def test_ingest_preflight_never_lets_negative_triage_authorize_tag_deletion(
    raw: str,
    title: str,
    body: str,
) -> None:
    prompt = build_ingest_reconciliation_prompt(
        {
            "raw_content": raw,
            "triage_plan": [
                {
                    "filename": "memory/microfinance-policy.md",
                    "summary": "Record the runtime; the raw contains no finance subject.",
                }
            ],
            "local_generated_operations": [
                {
                    "type": "create",
                    "filename": "memory/microfinance-policy.md",
                    "content": (
                        f"---\ntitle: {title}\n"
                        "tags: [d/configuration, d/finance, t/reference, "
                        "s/evergreen]\n---\n"
                        f"{body}\n"
                    ),
                }
            ],
        }
    )
    preflight = json.loads(
        prompt.split("<DETERMINISTIC_INGEST_REPAIR_PREFLIGHT_JSON>\n", 1)[1].split(
            "\n</DETERMINISTIC_INGEST_REPAIR_PREFLIGHT_JSON>", 1
        )[0]
    )

    assert preflight["tag_authority"] == "local_quorum_only"
    assert "invalid_tags" not in preflight
    assert any(
        option["invalid_tags"] == ["d/finance"]
        for option in preflight["semantic_tag_options"]
    )


def test_ingest_preflight_scopes_a_shared_tag_option_to_one_filename() -> None:
    raw = "A shared taxonomy example."
    operations = [
        {
            "type": "create",
            "filename": "memory/first.md",
            "content": (
                "---\ntitle: First\ntags: [d/alpha, d/shared, t/reference, "
                f"s/evergreen]\n---\n{raw}\n"
            ),
        },
        {
            "type": "create",
            "filename": "memory/second.md",
            "content": (
                "---\ntitle: Second\ntags: [d/beta, d/shared, t/reference, "
                f"s/evergreen]\n---\n{raw}\n"
            ),
        },
    ]
    prompt = build_ingest_reconciliation_prompt(
        {
            "raw_content": raw,
            "triage_plan": [],
            "local_generated_operations": operations,
        }
    )
    preflight = json.loads(
        prompt.split("<DETERMINISTIC_INGEST_REPAIR_PREFLIGHT_JSON>\n", 1)[1].split(
            "\n</DETERMINISTIC_INGEST_REPAIR_PREFLIGHT_JSON>", 1
        )[0]
    )
    shared_options = [
        option
        for option in preflight["semantic_tag_options"]
        if option["invalid_tags"] == ["d/shared"]
    ]

    assert preflight["status"] == "none"
    assert preflight["deterministic_repair_option_id"] is None
    assert {option["filename"] for option in shared_options} == {
        "memory/first.md",
        "memory/second.md",
    }
    assert len({option["repair_option_id"] for option in shared_options}) == 2
    assert all(len(option["replacement_operations"]) == 1 for option in shared_options)
    for option in shared_options:
        [replacement] = option["replacement_operations"]
        assert replacement["filename"] == option["filename"]
        assert "d/shared" not in replacement["content"]


def test_canonical_manifest_seals_all_effective_requests_and_outcomes() -> None:
    manifest = decision_lane_contract_case_manifest()
    assert manifest["total_cases"] == 100
    assert len(manifest["lanes"]) == 19
    assert len(decision_lane_contract_case_manifest_sha256()) == 64
    assert all(
        lane["case_count"] >= CASES_PER_MODEL_BACKED_LANE
        and len(lane["cases"]) == lane["case_count"]
        and len(lane["effective_request_sha256s"]) == lane["case_count"]
        for lane in manifest["lanes"].values()
    )


def test_lane_overlay_excludes_the_infeasible_16k_bucket() -> None:
    config = DecisionRouterConfig()
    schemas = production_decision_schemas()
    buckets = Counter(
        decision_request_context(
            config,
            case.prompt,
            schemas[case.schema_name],
            case.system,
            decision_lane=case.lane,
        )[1]
        for case in decision_lane_contract_case_specs()
    )
    assert buckets == {
        32_768: 97,
        65_536: 1,
        98_304: 1,
        114_688: 1,
    }
    assert decision_context_buckets(config) == (
        32_768,
        65_536,
        98_304,
        114_688,
    )


def test_large_complete_repacket_covers_the_112k_gate_bucket() -> None:
    config = DecisionRouterConfig()
    schema = production_decision_schemas()["lint_safe_semantic_mutation"]
    case = next(
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "page_normalize" and case.ordinal == 6
    )

    required, bucket = decision_request_context(
        config,
        case.prompt,
        schema,
        None,
        decision_lane=case.lane,
    )

    assert len(case.prompt) <= config.max_input_chars
    assert 98_304 < required <= 114_688
    assert bucket == 114_688
    assert (
        max(
            decision_request_context(
                config,
                candidate.prompt,
                production_decision_schemas()[candidate.schema_name],
                candidate.system,
                decision_lane=candidate.lane,
            )[0]
            for candidate in decision_lane_contract_case_specs()
        )
        <= config.num_ctx
    )
    proposal = _safe_mutation_proposal(case)
    assert proposal["review_packet"]["mode"] == "changed_spans"
    assert proposal["review_packet"]["coverage"]["all_changed_spans_rendered"] is True


@pytest.mark.parametrize(
    ("module_name", "function_name", "builder_name"),
    [
        (
            "autonomy",
            "_review_deferred_duplicate",
            "build_autonomy_duplicate_review_prompt",
        ),
        (
            "autonomy",
            "_review_retention_candidate",
            "build_autonomy_retention_review_prompt",
        ),
        ("ingest", "_run_ingest_frontier_review", "build_ingest_reconciliation_prompt"),
        ("orphan_link", "_review_orphan_proposal", "build_orphan_link_review_prompt"),
        (
            "raw_replay",
            "_review_indeterminate_rows",
            "build_raw_replay_reconciliation_prompt",
        ),
        ("read_back_repair", "_review_query_hint", "build_read_back_repair_request"),
        (
            "recall_auto_apply",
            "review_auto_apply_with_frontier",
            "build_recall_auto_apply_prompt",
        ),
        (
            "recall_calibration",
            "review_calibration_with_frontier",
            "build_recall_calibration_prompt",
        ),
        (
            "search_eval",
            "review_search_policy_with_frontier",
            "build_search_self_tune_prompt",
        ),
    ],
)
def test_inline_production_lanes_use_the_shared_canonical_prompt_builder(
    module_name: str,
    function_name: str,
    builder_name: str,
) -> None:
    module = importlib.import_module(f"llm_wiki_mcp.{module_name}")
    production_source = inspect.getsource(getattr(module, function_name))

    assert builder_name in production_source
