from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections import Counter

import pytest

from chronovisor.core.runtime_config import DecisionRouterConfig
from chronovisor.decision.decision_lane_contract_cases import (
    CASES_PER_MODEL_BACKED_LANE,
    QUORUM_VETO_CASES_PER_POLICY_LANE,
    background_decision_lane_contract_cases,
    decision_lane_contract_case_manifest,
    decision_lane_contract_case_manifest_sha256,
    decision_lane_contract_case_specs,
    quorum_veto_lane_contract_cases,
)
from chronovisor.decision.decision_lane_contracts import (
    LANE_CONTRACT_CASE_VERSION,
    LANE_CONTRACT_POLICY_VERSION,
    LANE_CONTRACT_SOURCE,
    LANE_PROMPT_POLICY_VERSIONS,
    bind_lane_contract_request,
    lane_contract,
    lane_contract_manifest,
    lane_contract_manifest_sha256,
    model_backed_lane_names,
)
from chronovisor.decision.decision_lane_prompts import (
    INGEST_PROPOSAL_SCHEMA_VERSION,
    INGEST_REPAIR_HOST_BLOCK,
    INGEST_REPAIR_MODEL_BLOCK,
    INGEST_REPAIR_OPTION_ID_RE,
    INGEST_REPAIR_OPTION_POLICY_VERSION,
    INGEST_REVIEW_MODEL_BLOCK,
    _exact_text_change_projection,
    _frontmatter_identity_fields,
    build_ingest_reconciliation_prompt,
    build_ingest_review_projection,
    build_raw_replay_reconciliation_prompt,
    canonical_json_sha256,
    validate_ingest_review_projection,
)
from chronovisor.decision.decision_policy import DECISION_POLICIES
from chronovisor.decision.decision_router import (
    QUORUM_SAFETY_POLICY_VERSION,
    TIE_BREAK_MUTATING_MAJORITY_LANES,
    _strip_ingest_repair_host_block,
    decision_context_buckets,
    decision_request_context,
    decision_request_fingerprint_sha256,
    decision_system_with_policy,
)
from chronovisor.decision.decision_schema_manifest import (
    background_decision_schemas,
    production_decision_schemas,
)
from chronovisor.decision.local_structured import (
    StructuredRequestPreflight,
    preflight_structured_request,
)


def _json_prompt_block(prompt: str, marker: str) -> dict[str, object]:
    return json.loads(prompt.split(f"<{marker}>\n", 1)[1].split(f"\n</{marker}>", 1)[0])


def _host_repair_preflight(prompt: str) -> dict[str, object]:
    sealed = _json_prompt_block(prompt, INGEST_REPAIR_HOST_BLOCK)
    return dict(sealed["full_preflight"])


def _render_projected_parts(parts: list[dict[str, object]], *, raw: str) -> str:
    rendered: list[str] = []
    for part in parts:
        if part["kind"] == "literal":
            rendered.append(str(part["text"]))
        else:
            assert part["kind"] == "raw_content_ref"
            assert part["sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
            assert part["utf8_bytes"] == len(raw.encode("utf-8"))
            rendered.append(raw)
    return "".join(rendered)


def _assert_exact_bounded_context(
    document: str,
    context: dict[str, object],
    *,
    raw: str,
) -> str:
    rendered = _render_projected_parts(context["parts"], raw=raw)
    start, end = context["utf8_range"]
    assert document.encode("utf-8")[start:end].decode("utf-8") == rendered
    assert len(rendered.encode("utf-8")) <= 256
    return rendered


def _complete_create_ingest_proposal(
    *,
    raw: str,
    operations: list[dict[str, object]],
    triage_plan: list[dict[str, object]],
) -> dict[str, object]:
    prepared = []
    for index, operation in enumerate(operations):
        filename = str(operation["filename"])
        proposed = str(operation["content"])
        prepared.append(
            {
                "op_type": "create",
                "path": filename,
                "page_id": filename.rsplit("/", 1)[-1].removesuffix(".md"),
                "source_operation_index": index,
                "source_operation_type": "create",
                "source_filename": filename,
                "preimage_exists": False,
                "previous_text": None,
                "previous_sha256": None,
                "proposed_text": proposed,
                "proposed_sha256": hashlib.sha256(proposed.encode("utf-8")).hexdigest(),
                "new_tags": [],
            }
        )
    return {
        "schema_version": INGEST_PROPOSAL_SCHEMA_VERSION,
        "kind": "ingest_semantic_mutation_proposal",
        "source_key": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "source_raw": "raw/contracts.md",
        "raw_content": raw,
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "raw_keywords": [],
        "local_disposition": "operations_available",
        "triage_plan": triage_plan,
        "failed_operation_specs": [],
        "local_generated_operations": operations,
        "prepared_operations": prepared,
        "link_reconciliation": {"resolved": 0, "rewritten": 0, "unwrapped": 0},
        "audit_decision": {"required": True},
    }


def _ingest_model_preflight(
    proposal: dict[str, object],
) -> tuple[str, str, StructuredRequestPreflight]:
    prompt = build_ingest_reconciliation_prompt(proposal)
    schema = production_decision_schemas()["ingest_reconciliation"]
    bound_prompt, bound_system = bind_lane_contract_request(
        "ingest_reconciliation", prompt, schema, None
    )
    model_prompt = _strip_ingest_repair_host_block(bound_prompt)
    preflight = preflight_structured_request(
        model_prompt,
        schema,
        system=decision_system_with_policy(schema, bound_system),
        max_input_chars=DecisionRouterConfig().max_input_chars,
    )
    return prompt, model_prompt, preflight


def test_contract_cases_cover_every_model_backed_lane_independently() -> None:
    cases = decision_lane_contract_case_specs()
    expected_lanes = set(model_backed_lane_names())
    counts = Counter(case.lane for case in cases)

    assert len(cases) == 100
    assert set(counts) == expected_lanes
    assert min(counts.values()) == CASES_PER_MODEL_BACKED_LANE
    assert counts["content_correction_classification"] == 6
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(case.case_id.startswith("lane-contract-v27:") for case in cases)


def test_background_graph_contracts_are_separate_from_adopted_fleet() -> None:
    cases = background_decision_lane_contract_cases()
    schemas = background_decision_schemas()

    assert set(cases) == {
        "relation_verification",
        "entity_merge_verification",
        "recall_usefulness_judgment",
        "recall_rubric_calibration",
        "recall_answer_adjudication",
    }
    assert all(len(rows) == 5 for rows in cases.values())
    assert set(cases).isdisjoint(model_backed_lane_names())
    for lane in cases:
        assert DECISION_POLICIES[lane].adoption_scoped is False
        contract = lane_contract(lane)
        assert contract["schema_name"] in schemas
        assert len(contract["contract_sha256"]) == 64


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
        assert (
            f"CHRONOVISOR_LANE_CONTRACT_POLICY={prompt_policy_version}" in system
        )
        assert f"CHRONOVISOR_LANE={case.lane}" in system
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
    assert LANE_CONTRACT_POLICY_VERSION == 10
    assert INGEST_REPAIR_OPTION_POLICY_VERSION == 2
    assert LANE_CONTRACT_CASE_VERSION == 27
    assert LANE_CONTRACT_SOURCE == "deterministic_lane_contract_v27"
    assert set(model_backed_lane_names()).issubset(LANE_PROMPT_POLICY_VERSIONS)
    assert set(LANE_PROMPT_POLICY_VERSIONS) - set(model_backed_lane_names()) == set(
        background_decision_lane_contract_cases()
    )
    assert LANE_PROMPT_POLICY_VERSIONS["ingest_reconciliation"] == 16
    assert LANE_PROMPT_POLICY_VERSIONS["raw_replay_reconciliation"] == 9
    assert LANE_PROMPT_POLICY_VERSIONS["recall_auto_apply"] == 9
    assert {
        version
        for lane, version in LANE_PROMPT_POLICY_VERSIONS.items()
        if lane
        not in {
            "ingest_reconciliation",
                "raw_replay_reconciliation",
                "recall_auto_apply",
                "autonomy_retention",
            }
    } == {8}


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
    from chronovisor.decision import (
        decision_lane_contract_cases,
        entity_backfill_contract,
    )

    monkeypatch.setattr(
        entity_backfill_contract,
        "validate_entity_backfill_proposal",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(ValueError, match="cannot reach production model"):
        decision_lane_contract_cases._entity_backfill_cases()


def test_entity_preflight_rejects_alias_incomplete_fixture_before_model() -> None:
    from chronovisor.decision.entity_backfill_contract import (
        validate_entity_backfill_proposal,
    )

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
    from chronovisor.decision.lint_mutation_contract import (
        build_semantic_review_packet,
    )

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
    from chronovisor.decision.decision_lane_prompts import (
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
    from chronovisor.decision.local_repair import _validate_decision

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
    from chronovisor.decision.recall_improvement_contract import gate_candidate

    accepted, checks = gate_candidate(
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


def test_ingest_contracts_use_complete_hash_bound_review_projections() -> None:
    cases = [
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "ingest_reconciliation"
    ]
    required = {
        "projection_policy_version",
        "schema_version",
        "kind",
        "full_proposal_kind",
        "full_proposal_sha256",
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
        "projection_sha256",
    }
    prepared_required = {
        "index",
        "op_type",
        "path",
        "page_id",
        "source_operation_index",
        "source_operation_type",
        "source_filename",
        "preimage_exists",
        "new_tags",
        "previous_utf8_bytes",
        "proposed_utf8_bytes",
        "previous_sha256",
        "proposed_sha256",
        "preimage_binding_verified",
        "proposed_hash_verified",
        "metadata_verified",
        "page_identity",
        "previous_ends_with_newline",
        "proposed_ends_with_newline",
        "previous_content_sha256",
        "proposed_content_sha256",
        "exact_change_hunks",
        "exact_change_hunks_sha256",
        "coverage_receipt",
        "coverage_status",
    }

    assert len(cases) > CASES_PER_MODEL_BACKED_LANE
    for case in cases:
        proposal = _json_prompt_block(case.prompt, INGEST_REVIEW_MODEL_BLOCK)
        assert set(proposal) == required
        assert proposal["kind"] == "ingest_semantic_mutation_review_projection"
        assert proposal["full_proposal_kind"] == "ingest_semantic_mutation_proposal"
        assert (
            proposal["raw_sha256"]
            == hashlib.sha256(proposal["raw_content"].encode("utf-8")).hexdigest()
        )
        projection_core = dict(proposal)
        projection_sha256 = projection_core.pop("projection_sha256")
        assert (
            projection_sha256
            == hashlib.sha256(
                json.dumps(
                    projection_core,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        )
        assert proposal["local_disposition"] in {
            "operations_available",
            "triage_no_operations",
            "all_generation_failed",
            "partial_generation_failed",
        }
        for prepared in proposal["prepared_operations"]:
            assert set(prepared) == prepared_required
            assert prepared["preimage_binding_verified"] is True
            assert prepared["proposed_hash_verified"] is True
            assert prepared["metadata_verified"] is True
            assert prepared["coverage_status"] == "complete"
            assert prepared["coverage_receipt"]["all_opcodes_contiguous"] is True
            assert (
                prepared["coverage_receipt"]["all_equal_spans_byte_identical"] is True
            )
            assert (
                prepared["exact_change_hunks_sha256"]
                == hashlib.sha256(
                    json.dumps(
                        prepared["exact_change_hunks"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
            )

    grounded_apply = next(
        case
        for case in cases
        if case.expected["decision"] == "apply_available"
        and case.expected["failed_operations_disposition"] == "none"
    )
    apply_proposal = _json_prompt_block(
        grounded_apply.prompt, INGEST_REVIEW_MODEL_BLOCK
    )
    [prepared] = apply_proposal["prepared_operations"]
    rendered_parts = [
        part
        for hunk in prepared["exact_change_hunks"]
        for part in hunk["proposed_changed_parts"]
    ]
    assert any(part["kind"] == "raw_content_ref" for part in rendered_parts)

    repair = next(
        case
        for case in cases
        if case.expected.get("invalid_tags")
        and case.expected.get("replacement_operations")
    )
    repair_proposal = _json_prompt_block(repair.prompt, INGEST_REVIEW_MODEL_BLOCK)
    [replacement] = repair.expected["replacement_operations"]
    preflight = _host_repair_preflight(repair.prompt)
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


def test_ingest_review_projection_recomputes_and_rejects_tampering() -> None:
    raw = "The runtime keeps exact change evidence."
    previous = "---\ntitle: Runtime\n---\nOld body without the fact."
    proposed = previous + "\n" + raw + "\n"
    proposal = {
        "schema_version": INGEST_PROPOSAL_SCHEMA_VERSION,
        "kind": "ingest_semantic_mutation_proposal",
        "source_key": "source",
        "source_raw": "raw.md",
        "raw_content": raw,
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "raw_keywords": ["runtime"],
        "local_disposition": "operations_available",
        "triage_plan": [],
        "failed_operation_specs": [],
        "local_generated_operations": [
            {
                "type": "update",
                "filename": "runtime.md",
                "content": raw,
                "raw_keywords": ["runtime"],
            }
        ],
        "prepared_operations": [
            {
                "op_type": "update",
                "path": "/wiki/pages/runtime.md",
                "page_id": "runtime",
                "source_operation_index": 0,
                "source_operation_type": "update",
                "source_filename": "runtime.md",
                "preimage_exists": True,
                "previous_text": previous,
                "previous_sha256": hashlib.sha256(previous.encode("utf-8")).hexdigest(),
                "proposed_text": proposed,
                "proposed_sha256": hashlib.sha256(proposed.encode("utf-8")).hexdigest(),
                "new_tags": [],
            }
        ],
        "link_reconciliation": {"resolved": 0, "rewritten": 0, "unwrapped": 0},
        "audit_decision": {"required": True},
    }

    projection = build_ingest_review_projection(proposal)

    assert validate_ingest_review_projection(proposal, projection) is True
    [hunk] = projection["prepared_operations"][0]["exact_change_hunks"]
    assert (
        _render_projected_parts(hunk["proposed_changed_parts"], raw=raw) == f"\n{raw}\n"
    )
    assert any(
        part["kind"] == "raw_content_ref" for part in hunk["proposed_changed_parts"]
    )
    assert proposal["prepared_operations"][0]["previous_text"] == previous
    assert proposal["prepared_operations"][0]["proposed_text"] == proposed
    tampered = json.loads(json.dumps(projection))
    tampered["prepared_operations"][0]["exact_change_hunks"][0][
        "proposed_changed_parts"
    ].append({"kind": "literal", "text": "\n+tampered"})
    assert validate_ingest_review_projection(proposal, tampered) is False


def test_ingest_review_projection_keeps_full_page_identity_for_distant_hunk() -> None:
    raw = "The generic state is now current."
    frontmatter = "---\ntitle: Alice account state\ntags: [d/account]\n---\n"
    previous = frontmatter + ("Unchanged history.\n" * 200) + "State: old\n"
    proposed = previous.removesuffix("State: old\n") + "State: current\n"
    proposal = {
        "schema_version": INGEST_PROPOSAL_SCHEMA_VERSION,
        "kind": "ingest_semantic_mutation_proposal",
        "source_key": "identity-source",
        "source_raw": "raw.md",
        "raw_content": raw,
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "raw_keywords": ["Alice"],
        "local_disposition": "operations_available",
        "triage_plan": [],
        "failed_operation_specs": [],
        "local_generated_operations": [
            {"type": "update", "filename": "alice.md", "content": "State: current\n"}
        ],
        "prepared_operations": [
            {
                "op_type": "update",
                "path": "alice.md",
                "page_id": "alice",
                "source_operation_index": 0,
                "source_operation_type": "update",
                "source_filename": "alice.md",
                "preimage_exists": True,
                "previous_text": previous,
                "previous_sha256": hashlib.sha256(previous.encode("utf-8")).hexdigest(),
                "proposed_text": proposed,
                "proposed_sha256": hashlib.sha256(proposed.encode("utf-8")).hexdigest(),
                "new_tags": ["d/account"],
            }
        ],
        "link_reconciliation": {"resolved": 0, "rewritten": 0, "unwrapped": 0},
        "audit_decision": {"required": True},
    }

    projection = build_ingest_review_projection(proposal)
    [prepared] = projection["prepared_operations"]

    assert prepared["page_identity"] == {
        "mode": "shared",
        "previous_frontmatter_utf8_bytes": len(frontmatter.encode("utf-8")),
        "previous_frontmatter_sha256": hashlib.sha256(
            frontmatter.encode("utf-8")
        ).hexdigest(),
        "proposed_frontmatter_utf8_bytes": len(frontmatter.encode("utf-8")),
        "proposed_frontmatter_sha256": hashlib.sha256(
            frontmatter.encode("utf-8")
        ).hexdigest(),
        "identity_fields": "title: Alice account state\n",
        "identity_fields_utf8_bytes": len(b"title: Alice account state\n"),
        "identity_fields_sha256": hashlib.sha256(
            b"title: Alice account state\n"
        ).hexdigest(),
    }
    assert "Alice account state" not in json.dumps(
        prepared["exact_change_hunks"], ensure_ascii=False
    )


def test_frontmatter_identity_fields_require_exact_top_level_keys() -> None:
    frontmatter = (
        "---\n"
        "subtitle: Not a title\n"
        "title: Exact title\n"
        "title_suffix: Not a title\n"
        "canonical: canonical-page\n"
        "canonical_notes: not identity\n"
        "slug: exact-slug\n"
        "slug_history: [old-slug]\n"
        "page_id: exact-page\n"
        "page_id_backup: old-page\n"
        "aliases:\n  - exact-alias\n"
        "---\n"
    )

    assert _frontmatter_identity_fields(frontmatter) == (
        "title: Exact title\n"
        "canonical: canonical-page\n"
        "slug: exact-slug\n"
        "page_id: exact-page\n"
        "aliases:\n  - exact-alias\n"
    )


def test_frontmatter_only_hunks_do_not_duplicate_large_metadata_context() -> None:
    raw = "Grounded body update."
    keywords = ", ".join(f"keyword-{index}" for index in range(300))
    previous = (
        "---\ntitle: Stable identity\nupdated: 2026-07-13\n"
        f"raw_keywords: [{keywords}]\n---\nOld body.\n"
    )
    proposed = previous.replace("updated: 2026-07-13", "updated: 2026-07-14").replace(
        "Old body.", raw
    )
    proposal = {
        "schema_version": INGEST_PROPOSAL_SCHEMA_VERSION,
        "kind": "ingest_semantic_mutation_proposal",
        "source_key": "metadata-context-source",
        "source_raw": "raw.md",
        "raw_content": raw,
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "raw_keywords": [],
        "local_disposition": "operations_available",
        "triage_plan": [],
        "failed_operation_specs": [],
        "local_generated_operations": [
            {"type": "update", "filename": "stable.md", "content": raw + "\n"}
        ],
        "prepared_operations": [
            {
                "op_type": "update",
                "path": "stable.md",
                "page_id": "stable",
                "source_operation_index": 0,
                "source_operation_type": "update",
                "source_filename": "stable.md",
                "preimage_exists": True,
                "previous_text": previous,
                "previous_sha256": hashlib.sha256(previous.encode()).hexdigest(),
                "proposed_text": proposed,
                "proposed_sha256": hashlib.sha256(proposed.encode()).hexdigest(),
                "new_tags": [],
            }
        ],
        "link_reconciliation": {"resolved": 0, "rewritten": 0, "unwrapped": 0},
        "audit_decision": {"required": True},
    }

    projection = build_ingest_review_projection(proposal)
    hunks = projection["prepared_operations"][0]["exact_change_hunks"]
    frontmatter_hunk = next(hunk for hunk in hunks if "change_context" not in hunk)
    body_hunk = next(hunk for hunk in hunks if "change_context" in hunk)

    assert "change_context" not in frontmatter_hunk
    assert (
        _render_projected_parts(frontmatter_hunk["previous_changed_parts"], raw=raw)
        == "updated: 2026-07-13\n"
    )
    assert (
        _render_projected_parts(frontmatter_hunk["proposed_changed_parts"], raw=raw)
        == "updated: 2026-07-14\n"
    )
    assert body_hunk["change_context"]["mode"] == "shared"
    assert "keyword-0" not in json.dumps(hunks, ensure_ascii=False)


def test_large_frontmatter_field_keeps_key_and_local_change_context() -> None:
    raw = "The taxonomy metadata changed."
    keywords = ", ".join(f"keyword-{index}" for index in range(300))
    previous = (
        f"---\ntitle: Stable identity\nraw_keywords: [{keywords}]\n---\nStable body.\n"
    )
    proposed = previous.replace("keyword-150", "topic-150")
    proposal = {
        "schema_version": INGEST_PROPOSAL_SCHEMA_VERSION,
        "kind": "ingest_semantic_mutation_proposal",
        "source_key": "large-field-context-source",
        "source_raw": "raw.md",
        "raw_content": raw,
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "raw_keywords": [],
        "local_disposition": "operations_available",
        "triage_plan": [],
        "failed_operation_specs": [],
        "local_generated_operations": [
            {"type": "update", "filename": "stable.md", "content": raw + "\n"}
        ],
        "prepared_operations": [
            {
                "op_type": "update",
                "path": "stable.md",
                "page_id": "stable",
                "source_operation_index": 0,
                "source_operation_type": "update",
                "source_filename": "stable.md",
                "preimage_exists": True,
                "previous_text": previous,
                "previous_sha256": hashlib.sha256(previous.encode()).hexdigest(),
                "proposed_text": proposed,
                "proposed_sha256": hashlib.sha256(proposed.encode()).hexdigest(),
                "new_tags": [],
            }
        ],
        "link_reconciliation": {"resolved": 0, "rewritten": 0, "unwrapped": 0},
        "audit_decision": {"required": True},
    }

    projection = build_ingest_review_projection(proposal)
    [hunk] = projection["prepared_operations"][0]["exact_change_hunks"]
    field_context = hunk["frontmatter_field_context"]

    assert field_context["previous_fields"] == ["raw_keywords"]
    assert field_context["proposed_fields"] == ["raw_keywords"]
    assert "keyword-149" in field_context["previous_before"]
    assert field_context["previous_before"] == field_context["proposed_before"]
    assert field_context["previous_after"].startswith("-150, keyword-151")
    assert field_context["previous_after"] == field_context["proposed_after"]
    assert "keyword-151" in field_context["previous_after"]
    assert len(json.dumps(hunk, ensure_ascii=False).encode()) < 2_000


def test_deterministic_repair_preserves_exact_raw_whitespace() -> None:
    raw = "  Exact raw with leading spaces.  \n\n"
    proposal = _complete_create_ingest_proposal(
        raw=raw,
        operations=[
            {
                "type": "create",
                "filename": "memory/exact-whitespace.md",
                "content": (
                    "---\ntitle: Exact whitespace\n"
                    "tags: [d/configuration, t/reference]\n---\n"
                    f"{raw}Unsupported suffix.\n"
                ),
            }
        ],
        triage_plan=[],
    )

    prompt = build_ingest_reconciliation_prompt(proposal)
    preflight = _host_repair_preflight(prompt)
    [replacement] = preflight["replacement_operations"]

    assert replacement["content"].endswith("---\n" + raw)
    assert not replacement["content"].endswith(raw + "\n")


@pytest.mark.parametrize("change_kind", ["same_line_replace", "line_insert"])
def test_ingest_change_hunks_include_bounded_semantic_page_context(
    change_kind: str,
) -> None:
    raw = "New grounded fact"
    if change_kind == "same_line_replace":
        previous = (
            "A" * 600
            + " Subject decides old value while existing context remains. "
            + "B" * 600
        )
        proposed = previous.replace("old value", "new value")
        expected_before = "Subject decides "
        expected_after = " value while existing context remains."
    else:
        previous = (
            "P" * 600
            + "\nTarget page heading\n"
            + "Existing tail context\n"
            + "S" * 600
        )
        proposed = previous.replace(
            "Target page heading\n",
            f"Target page heading\n{raw}\n",
        )
        expected_before = "Target page heading\n"
        expected_after = "Existing tail context\n"

    projection = _exact_text_change_projection(
        previous,
        proposed,
        raw_content=raw,
    )
    [hunk] = projection["exact_change_hunks"]

    assert hunk["context_parts_complete"] is True
    context = hunk["change_context"]
    assert context["mode"] == "shared"
    for document, prefix in ((previous, "previous"), (proposed, "proposed")):
        before = _assert_exact_bounded_context(
            document,
            {
                "parts": context["before"]["parts"],
                "utf8_range": context["before"][f"{prefix}_utf8_range"],
            },
            raw=raw,
        )
        after = _assert_exact_bounded_context(
            document,
            {
                "parts": context["after"]["parts"],
                "utf8_range": context["after"][f"{prefix}_utf8_range"],
            },
            raw=raw,
        )
        assert before.endswith(expected_before)
        assert after.startswith(expected_after)


def test_ingest_review_projection_accepts_production_create_absent_preimage() -> None:
    raw = "A production create retains its complete postimage."
    content = f"---\ntitle: Create\n---\n{raw}\n"
    proposal = _complete_create_ingest_proposal(
        raw=raw,
        operations=[
            {
                "type": "create",
                "filename": "memory/create.md",
                "content": content,
            }
        ],
        triage_plan=[],
    )

    projection = build_ingest_review_projection(proposal)
    [prepared] = projection["prepared_operations"]

    assert prepared["coverage_status"] == "complete"
    assert prepared["preimage_exists"] is False
    assert prepared["previous_sha256"] is None
    assert prepared["preimage_binding_verified"] is True
    assert prepared["previous_utf8_bytes"] == 0
    [hunk] = prepared["exact_change_hunks"]
    assert _render_projected_parts(hunk["previous_changed_parts"], raw=raw) == ""
    assert _render_projected_parts(hunk["proposed_changed_parts"], raw=raw) == content
    assert any(
        part["kind"] == "raw_content_ref" for part in hunk["proposed_changed_parts"]
    )
    assert projection["full_proposal_sha256"] == canonical_json_sha256(proposal)


@pytest.mark.parametrize(
    "failure",
    ["create_preimage", "update_hash", "binding", "bool_binding"],
)
def test_ingest_review_projection_fails_before_transport_on_invalid_proof(
    failure: str,
) -> None:
    raw = "A proof mismatch must never reach a model."
    content = f"---\ntitle: Proof\n---\n{raw}\n"
    proposal = _complete_create_ingest_proposal(
        raw=raw,
        operations=[
            {
                "type": "create",
                "filename": "memory/proof.md",
                "content": content,
            }
        ],
        triage_plan=[],
    )
    prepared = proposal["prepared_operations"][0]
    if failure == "create_preimage":
        prepared["previous_text"] = ""
        prepared["previous_sha256"] = hashlib.sha256(b"").hexdigest()
    elif failure == "update_hash":
        prepared["op_type"] = "update"
        prepared["preimage_exists"] = True
        prepared["previous_text"] = "before\n"
        prepared["previous_sha256"] = "0" * 64
    elif failure == "binding":
        prepared["source_operation_index"] = 1
    else:
        prepared["source_operation_index"] = True

    with pytest.raises(ValueError, match="incomplete or invalid proof"):
        build_ingest_reconciliation_prompt(proposal)


@pytest.mark.parametrize(
    "patch",
    [
        {"schema_version": INGEST_PROPOSAL_SCHEMA_VERSION + 1},
        {"kind": "stale_ingest_proposal"},
        {"raw_sha256": "0" * 64},
        {"raw_content": None},
    ],
)
def test_ingest_review_projection_rejects_invalid_proposal_envelope(
    patch: dict[str, object],
) -> None:
    raw = "The proposal envelope must bind exact raw evidence."
    content = f"---\ntitle: Envelope\n---\n{raw}\n"
    proposal = _complete_create_ingest_proposal(
        raw=raw,
        operations=[
            {
                "type": "create",
                "filename": "memory/envelope.md",
                "content": content,
            }
        ],
        triage_plan=[],
    )
    proposal.update(patch)

    with pytest.raises(ValueError, match="envelope is invalid or stale"):
        build_ingest_reconciliation_prompt(proposal)


@pytest.mark.parametrize(
    ("previous", "proposed", "previous_changed", "proposed_changed"),
    [
        ("a\r\n", "a\n", "\r", ""),
        ("a\n", "a", "\n", ""),
        ("a\r", "a\n", "\r", "\n"),
    ],
)
def test_ingest_review_projection_keeps_newline_only_changes_visible(
    previous: str,
    proposed: str,
    previous_changed: str,
    proposed_changed: str,
) -> None:
    raw = "newline preservation"
    proposal = {
        "schema_version": INGEST_PROPOSAL_SCHEMA_VERSION,
        "kind": "ingest_semantic_mutation_proposal",
        "source_key": "newline",
        "source_raw": "raw/newline.md",
        "raw_content": raw,
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "raw_keywords": [],
        "local_disposition": "operations_available",
        "triage_plan": [],
        "failed_operation_specs": [],
        "local_generated_operations": [
            {"type": "update", "filename": "newline.md", "content": proposed}
        ],
        "prepared_operations": [
            {
                "op_type": "update",
                "path": "newline.md",
                "page_id": "newline",
                "source_operation_index": 0,
                "source_operation_type": "update",
                "source_filename": "newline.md",
                "preimage_exists": True,
                "previous_text": previous,
                "previous_sha256": hashlib.sha256(previous.encode("utf-8")).hexdigest(),
                "proposed_text": proposed,
                "proposed_sha256": hashlib.sha256(proposed.encode("utf-8")).hexdigest(),
                "new_tags": [],
            }
        ],
        "link_reconciliation": {"resolved": 0, "rewritten": 0, "unwrapped": 0},
        "audit_decision": {"required": True},
    }

    projection = build_ingest_review_projection(proposal)
    [prepared] = projection["prepared_operations"]
    [hunk] = prepared["exact_change_hunks"]

    assert (
        _render_projected_parts(hunk["previous_changed_parts"], raw=raw)
        == previous_changed
    )
    assert (
        _render_projected_parts(hunk["proposed_changed_parts"], raw=raw)
        == proposed_changed
    )
    assert prepared["coverage_receipt"]["changed_opcode_count"] == 1
    assert prepared["coverage_receipt"]["all_changed_spans_rendered"] is True


def test_large_ingest_review_projection_fits_current_router_input_cap() -> None:
    raw = "\n".join(f"raw fact {index}: durable evidence" for index in range(500))
    prepared = []
    generated = []
    for index, line_count in enumerate((550, 650, 1_150, 3_900, 400)):
        page_id = f"large-page-{index}"
        previous = "".join(
            f"existing {index} line {line}\n" for line in range(line_count)
        )
        addition = f"\n## New evidence {index}\nraw fact {index}: durable evidence\n"
        proposed = previous + addition
        prepared.append(
            {
                "op_type": "update",
                "path": f"/wiki/pages/{page_id}.md",
                "page_id": page_id,
                "source_operation_index": index,
                "source_operation_type": "update",
                "source_filename": f"{page_id}.md",
                "preimage_exists": True,
                "previous_text": previous,
                "previous_sha256": hashlib.sha256(previous.encode("utf-8")).hexdigest(),
                "proposed_text": proposed,
                "proposed_sha256": hashlib.sha256(proposed.encode("utf-8")).hexdigest(),
                "new_tags": [],
            }
        )
        generated.append(
            {
                "type": "update",
                "filename": f"{page_id}.md",
                "content": addition,
                "raw_keywords": ["durable"],
            }
        )
    proposal = {
        "schema_version": INGEST_PROPOSAL_SCHEMA_VERSION,
        "kind": "ingest_semantic_mutation_proposal",
        "source_key": "large-source",
        "source_raw": "large-raw.md",
        "raw_content": raw,
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "raw_keywords": ["durable"],
        "local_disposition": "operations_available",
        "triage_plan": [],
        "failed_operation_specs": [],
        "local_generated_operations": generated,
        "prepared_operations": prepared,
        "link_reconciliation": {"resolved": 0, "rewritten": 0, "unwrapped": 0},
        "audit_decision": {"required": True},
    }
    legacy_bytes = len(
        json.dumps(proposal, ensure_ascii=False, indent=2).encode("utf-8")
    )
    prompt = build_ingest_reconciliation_prompt(proposal)
    schema = production_decision_schemas()["ingest_reconciliation"]
    bound_prompt, bound_system = bind_lane_contract_request(
        "ingest_reconciliation", prompt, schema, None
    )
    effective_system = decision_system_with_policy(schema, bound_system)
    model_prompt = _strip_ingest_repair_host_block(bound_prompt)
    preflight = preflight_structured_request(
        model_prompt,
        schema,
        system=effective_system,
        max_input_chars=DecisionRouterConfig().max_input_chars,
    )

    assert legacy_bytes > 300_000
    assert preflight.ok is True
    assert preflight.input_bytes < DecisionRouterConfig().max_input_chars
    assert all(
        operation["coverage_status"] == "complete"
        for operation in _json_prompt_block(prompt, INGEST_REVIEW_MODEL_BLOCK)[
            "prepared_operations"
        ]
    )


def test_large_tagged_create_with_same_line_suffix_fits_router_input_cap() -> None:
    raw = " ".join(f"durable-{index}" for index in range(2_100))
    content = (
        "---\ntitle: Large repair\n"
        "tags: [d/configuration, d/finance, t/reference, s/evergreen]\n---\n"
        f"{raw} Unsupported generated suffix.\n"
    )
    proposal = _complete_create_ingest_proposal(
        raw=raw,
        operations=[
            {
                "type": "create",
                "filename": "memory/large-repair.md",
                "content": content,
            }
        ],
        triage_plan=[],
    )
    prompt = build_ingest_reconciliation_prompt(proposal)
    schema = production_decision_schemas()["ingest_reconciliation"]
    bound_prompt, bound_system = bind_lane_contract_request(
        "ingest_reconciliation", prompt, schema, None
    )
    model_prompt = _strip_ingest_repair_host_block(bound_prompt)
    preflight = preflight_structured_request(
        model_prompt,
        schema,
        system=decision_system_with_policy(schema, bound_system),
        max_input_chars=DecisionRouterConfig().max_input_chars,
    )
    repair_projection = _json_prompt_block(prompt, INGEST_REPAIR_MODEL_BLOCK)
    full_preflight = _host_repair_preflight(prompt)

    assert len(prompt.encode("utf-8")) > len(model_prompt.encode("utf-8")) * 2
    assert f"<{INGEST_REPAIR_HOST_BLOCK}>" not in model_prompt
    assert preflight.ok is True
    assert preflight.input_bytes < DecisionRouterConfig().max_input_chars
    assert full_preflight["replacement_operations"]
    assert (
        repair_projection["deterministic_repair_option"]["coverage_status"]
        == "complete"
    )
    mutation_ids = [
        mutation["mutation_id"] for mutation in repair_projection["mutations"]
    ]
    assert len(mutation_ids) == len(set(mutation_ids))


def test_eight_tagged_creates_fit_router_input_cap() -> None:
    raw = " ".join(f"fact-{index}" for index in range(500))
    operations = [
        {
            "type": "create",
            "filename": f"memory/multi-create-{index}.md",
            "content": (
                f"---\ntitle: Multi create {index}\n"
                "tags: [d/configuration, d/finance, t/reference, s/evergreen]\n"
                f"---\n{raw}\n"
            ),
        }
        for index in range(8)
    ]
    proposal = _complete_create_ingest_proposal(
        raw=raw,
        operations=operations,
        triage_plan=[],
    )

    prompt, model_prompt, preflight = _ingest_model_preflight(proposal)

    assert preflight.ok is True
    assert preflight.input_bytes < DecisionRouterConfig().max_input_chars
    assert len(prompt.encode("utf-8")) > len(model_prompt.encode("utf-8")) * 2
    review_projection = _json_prompt_block(prompt, INGEST_REVIEW_MODEL_BLOCK)
    assert len(review_projection["prepared_operations"]) == 8
    assert all(
        any(
            part["kind"] == "raw_content_ref"
            for hunk in operation["exact_change_hunks"]
            for part in hunk["proposed_changed_parts"]
        )
        for operation in review_projection["prepared_operations"]
    )


def test_single_near_max_raw_create_fits_router_input_cap() -> None:
    raw = " ".join(f"fact-{index}" for index in range(7_000))
    proposal = _complete_create_ingest_proposal(
        raw=raw,
        operations=[
            {
                "type": "create",
                "filename": "memory/near-max-raw.md",
                "content": (
                    "---\ntitle: Near max raw\n"
                    "tags: [d/configuration, d/finance, t/reference, s/evergreen]\n"
                    f"---\n{raw}\n"
                ),
            }
        ],
        triage_plan=[],
    )

    prompt, model_prompt, preflight = _ingest_model_preflight(proposal)

    assert preflight.ok is True
    assert preflight.input_bytes < DecisionRouterConfig().max_input_chars
    assert preflight.input_bytes > DecisionRouterConfig().max_input_chars * 3 // 4
    assert len(prompt.encode("utf-8")) > len(model_prompt.encode("utf-8")) * 2


def test_multi_operation_incident_compacts_model_projection_under_router_cap() -> None:
    """Six complete changes must fit without weakening the fixed 93KB cap."""

    raw = " ".join(f"rawfact{index:05d}" for index in range(1_400))
    operation_sizes = (5_274, 3_706, 4_363, 5_227, 7_109, 18_701)
    operations: list[dict[str, object]] = []
    for index, target_bytes in enumerate(operation_sizes):
        repeated = (f"fact-{index} " * (target_bytes // 7 + 1))[:target_bytes]
        operations.append(
            {
                "type": "create",
                "filename": f"memory/incident-page-{index}.md",
                "content": (
                    f"---\ntitle: Incident page {index}\n"
                    "tags: [d/configuration, t/reference, s/evergreen]\n"
                    f"---\n{repeated}"
                ),
            }
        )
    proposal = _complete_create_ingest_proposal(
        raw=raw,
        operations=operations,
        triage_plan=[],
    )

    prompt, _model_prompt, preflight = _ingest_model_preflight(proposal)
    config = DecisionRouterConfig()
    schema = production_decision_schemas()["ingest_reconciliation"]
    required_context, selected_context = decision_request_context(
        config,
        prompt,
        schema,
        None,
        "ingest_reconciliation",
    )

    assert preflight.ok is True
    assert config.max_input_chars * 3 // 4 < preflight.input_bytes < 93_000
    assert required_context <= selected_context == 114_688
    for marker in (INGEST_REPAIR_MODEL_BLOCK, INGEST_REVIEW_MODEL_BLOCK):
        encoded = prompt.split(f"<{marker}>\n", 1)[1].split(f"\n</{marker}>", 1)[0]
        assert encoded == json.dumps(
            json.loads(encoded),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )


def test_ingest_prompt_orders_quarantine_retry_and_safe_apply() -> None:
    prompts = [
        case.prompt
        for case in decision_lane_contract_case_specs()
        if case.lane == "ingest_reconciliation"
    ]

    assert prompts
    assert all(
        "Apply this decision table in order; stop at the first matching step:" in prompt
        for prompt in prompts
    )
    assert all(
        "Only if step 1 is false: if readable, internally consistent evidence has a\n"
        "   failed local operation another local attempt could resolve, choose retry\n"
        "   with retry_required." in prompt
        for prompt in prompts
    )
    assert all(
        "A coherent report that incompatible states are both current still matches\n"
        "   step 1 when no provenance resolves which state is authoritative." in prompt
        for prompt in prompts
    )
    assert all(
        prompt.index(f"</{INGEST_REVIEW_MODEL_BLOCK}>")
        < prompt.index(
            "Apply this decision table in order; stop at the first matching step:"
        )
        for prompt in prompts
    )
    assert all(
        "choose quarantined with failed_operations_disposition=retry_required" in prompt
        for prompt in prompts
    )
    assert all("Return exactly one repair_option_id" in prompt for prompt in prompts)
    assert all(
        "Do not return\ninvalid_tags or replacement_operations yourself" in prompt
        for prompt in prompts
    )


def test_ingest_quarantine_precedes_retryable_generation_failure() -> None:
    case = next(
        case
        for case in decision_lane_contract_case_specs()
        if case.lane == "ingest_reconciliation"
        and case.expected["decision"] == "quarantined"
    )
    projection = _json_prompt_block(case.prompt, INGEST_REVIEW_MODEL_BLOCK)

    assert projection["local_disposition"] == "all_generation_failed"
    assert projection["failed_operation_specs"]
    assert "Current setting: enabled." in projection["raw_content"]
    assert "Current setting: disabled." in projection["raw_content"]
    assert "neither has a source" in projection["raw_content"]


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
    operations = [
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
    ]
    prompt = build_ingest_reconciliation_prompt(
        _complete_create_ingest_proposal(
            raw=raw,
            triage_plan=[
                {
                    "filename": "memory/microfinance-policy.md",
                    "summary": "Record the runtime; the raw contains no finance subject.",
                }
            ],
            operations=operations,
        )
    )
    preflight = _json_prompt_block(prompt, INGEST_REPAIR_MODEL_BLOCK)

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
        _complete_create_ingest_proposal(
            raw=raw,
            triage_plan=[],
            operations=operations,
        )
    )
    preflight = _host_repair_preflight(prompt)
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
    assert manifest["total_contract_cases"] == 106
    assert len(manifest["lanes"]) == 19
    assert manifest["quorum_safety_policy_version"] == QUORUM_SAFETY_POLICY_VERSION
    assert manifest["quorum_veto_cases_per_policy_lane"] == (
        QUORUM_VETO_CASES_PER_POLICY_LANE
    )
    assert manifest["quorum_veto_case_count"] == 6
    assert len(decision_lane_contract_case_manifest_sha256()) == 64
    assert all(
        lane["case_count"] >= CASES_PER_MODEL_BACKED_LANE
        and len(lane["cases"]) == lane["case_count"]
        and len(lane["effective_request_sha256s"]) == lane["case_count"]
        for lane in manifest["lanes"].values()
    )


def test_quorum_veto_policy_cases_cover_five_bypasses_and_ingest_veto() -> None:
    cases = quorum_veto_lane_contract_cases()
    by_lane = {case.lane: case for case in cases}

    assert set(by_lane) == {
        *TIE_BREAK_MUTATING_MAJORITY_LANES,
        "ingest_reconciliation",
    }
    assert len(cases) == 6
    assert all(case.case_id.startswith("quorum-veto-v27:") for case in cases)
    assert all(len(case.as_dict()["case_sha256"]) == 64 for case in cases)
    for lane in TIE_BREAK_MUTATING_MAJORITY_LANES:
        case = by_lane[lane]
        assert case.expected_status == "agreed"
        assert case.expected_bypass is True
        assert case.expected_quarantine_reason is None
        assert case.conservative_veto_fired is True
        assert case.majority_effect_class == "mutating"
        assert case.dissent_effect_class == "conservative"
    ingest = by_lane["ingest_reconciliation"]
    assert ingest.expected_status == "quarantined"
    assert ingest.expected_bypass is False
    assert ingest.expected_quarantine_reason == (
        "mutating_local_majority_vetoed_by_conservative_vote"
    )
    assert decision_lane_contract_case_manifest()["quorum_veto_cases"] == [
        case.as_dict() for case in cases
    ]


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
            "chronovisor.ops.autonomy",
            "_review_deferred_duplicate",
            "build_autonomy_duplicate_review_prompt",
        ),
        (
            "chronovisor.ops.autonomy",
            "_review_retention_candidate",
            "build_autonomy_retention_review_prompt",
        ),
        (
            "chronovisor.ingest.ingest",
            "_run_ingest_frontier_review",
            "build_ingest_reconciliation_prompt",
        ),
        (
            "chronovisor.ops.orphan_link",
            "_review_orphan_proposal",
            "build_orphan_link_review_prompt",
        ),
        (
            "chronovisor.ingest.raw_replay",
            "_review_indeterminate_rows",
            "build_raw_replay_reconciliation_prompt",
        ),
        (
            "chronovisor.ingest.read_back_repair",
            "_review_query_hint",
            "build_read_back_repair_request",
        ),
        (
            "chronovisor.recall.recall_auto_apply",
            "review_auto_apply_with_frontier",
            "build_recall_auto_apply_prompt",
        ),
        (
            "chronovisor.recall.recall_calibration",
            "review_calibration_with_frontier",
            "build_recall_calibration_prompt",
        ),
        (
            "chronovisor.search.search_eval",
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
    module = importlib.import_module(module_name)
    production_source = inspect.getsource(getattr(module, function_name))

    assert builder_name in production_source
