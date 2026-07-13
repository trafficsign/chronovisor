from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_wiki_mcp import adoption_corpus, read_back_repair
from llm_wiki_mcp.adoption_corpus import (
    CONTRACT_SOURCE,
    HISTORICAL_SOURCE,
    INDEPENDENT_LABEL_EVIDENCE_KINDS,
    LANE_POLICY_SOURCE_PREFIX,
    LEGACY_UNFILTERED_EXCLUSION,
    LOCAL_CONSENSUS_SELF_LABEL_EXCLUSION,
    MIN_CURRENT_READ_BACK_POLICY_CASES,
    NONPRODUCTION_SCHEMA_EXCLUSION,
    NON_USER_TRANSPORT_EXCLUSION,
    READ_BACK_EVIDENCE_POLICY_MARKER,
    RETIRED_CORRECTION_SIGNAL_EXCLUSION,
    STALE_ENTITY_PROPOSAL_EXCLUSION,
    STALE_METADATA_PROPOSAL_EXCLUSION,
    STALE_READ_BACK_REVIEW_EXCLUSION,
    STALE_SEARCH_LABEL_SEMANTICS_EXCLUSION,
    STALE_UNBOUND_AUTHORITY_EXCLUSION,
    compile_adoption_corpus,
    contract_candidates,
)
from llm_wiki_mcp.decision_router import decision_context_buckets
from llm_wiki_mcp.decision_schema_manifest import (
    production_decision_schemas,
    production_schema_manifest,
)
from llm_wiki_mcp.local_model_eval import (
    ReplayInputError,
    inspect_replays,
    load_replay_corpus,
)
from llm_wiki_mcp.runtime_config import DecisionRouterConfig


def _expected(schema: dict[str, object]) -> dict[str, object]:
    properties = schema["properties"]
    assert isinstance(properties, dict)
    for field in ("decision", "action"):
        child = properties.get(field)
        if not isinstance(child, dict):
            continue
        values = child.get("enum")
        if isinstance(values, list) and values:
            return {field: values[0]}
    raise AssertionError("production schema has no decision-bearing enum")


def _legacy_row(
    schema_name: str,
    prompt: str,
    *,
    source: str | None = None,
    expected: dict[str, object] | None = None,
) -> dict[str, object]:
    schema = json.loads(json.dumps(production_decision_schemas()[schema_name]))
    row: dict[str, object] = {
        "timestamp": "2026-07-12T00:00:00+00:00",
        "role": "semantic_judge",
        "model": "historical:test",
        "effort": "test",
        "prompt": prompt,
        "prompt_truncated": False,
        "schema": schema,
        "expected": expected or _expected(schema),
        "latency_seconds": 0.1,
    }
    if source is not None:
        row["source"] = source
    return row


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _source(path: Path) -> Path:
    return _write_rows(
        path,
        [
            _legacy_row(schema_name, f"legacy-unbound-{schema_name}")
            for schema_name in production_decision_schemas()
        ],
    )


def _old_search_label_row(index: int, *, parseable: bool) -> dict[str, object]:
    payload = (
        json.dumps(
            {
                "query": f"old search query {index}",
                "candidate_labels": {
                    "expected_pages": ["old-page"],
                    "negative_pages": [],
                    "stale_pages": [],
                },
            },
            ensure_ascii=False,
        )
        if parseable
        else "not-json"
    )
    return _legacy_row(
        "search_label",
        "You are the trusted frontier label reviewer for LLM Wiki search "
        f"evaluation.\nCandidate:\n{payload}",
    )


def _independent_bound_row() -> dict[str, object]:
    candidate = next(
        row
        for row in contract_candidates()
        if row.row.get("decision_lane") == "content_correction_review"
    )
    row = json.loads(json.dumps(candidate.row))
    source = "frontier:gpt-5.4-independent-holdout"
    lane = str(row["decision_lane"])
    row.update(
        {
            "source": source,
            "evidence_provenance": {
                "kind": "independent_frontier_label",
                "label_source": source,
                "policy_source": f"{LANE_POLICY_SOURCE_PREFIX}{lane}",
                "policy_artifact_sha256": row["lane_contract_sha256"],
            },
            "model": source,
            "effort": "independent_holdout",
        }
    )
    for key in (
        "contract_id",
        "contract_version",
        "effective_request_sha256",
        "lane_contract_case_manifest_sha256",
    ):
        row.pop(key, None)
    return row


def _correction_prompt(
    correction_prompt: str,
    *,
    matched: str,
) -> str:
    event = {
        "correction_prompt": correction_prompt,
        "signal": {"matched": matched, "confidence": "candidate"},
    }
    return (
        "<CORRECTION_EVENT_UNTRUSTED_JSON>\n"
        f"{json.dumps(event, ensure_ascii=False)}\n"
        "</CORRECTION_EVENT_UNTRUSTED_JSON>"
    )


def _runtime_exclusion(
    prompt: str,
    schema_name: str,
    *,
    system: str | None = None,
) -> str | None:
    return adoption_corpus._historical_runtime_exclusion(
        prompt=prompt,
        system=system,
        schema_digest=production_schema_manifest()[schema_name],
    )


def test_compiler_is_deterministic_contract_only_and_source_read_only(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "replay.jsonl")
    before = source.read_bytes()
    output = tmp_path / "adoption.jsonl"
    canonical_count = len(contract_candidates())

    first = compile_adoption_corpus(source, output, minimum_cases=100)

    assert canonical_count >= 100
    assert first["status"] == "valid"
    assert first["changed"] is True
    assert first["selected_cases"] == canonical_count
    assert first["contract_cases"] == canonical_count
    assert first["historical_cases"] == 0
    assert first["validation"]["full_usable_selection"] is True
    assert first["validation"]["coverage"]["production_schema_coverage_rate"] == 1.0
    assert first["validation"]["coverage"]["minimum_production_schema_cases"] >= 5
    assert set(first["planned_context_bucket_counts"]) == set(
        decision_context_buckets(DecisionRouterConfig())
    )
    assert all(first["planned_context_bucket_counts"].values())
    assert source.read_bytes() == before
    assert output.stat().st_mode & 0o777 == 0o600
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(row.get("source") == CONTRACT_SOURCE for row in rows)
    assert all(row.get("selection_reasons") for row in rows)
    assert all(row.get("effective_request_sha256") for row in rows)
    assert first["selection_seal"]["unique_effective_requests"] == len(rows)
    assert first["selection_seal"]["schema_counts"] == first["schema_counts"]
    assert first["selection_seal"]["role_counts"] == first["role_counts"]
    assert first["selection_seal"]["source_counts"] == first["source_counts"]
    assert (
        first["selection_seal"]["schema_source_counts"] == first["schema_source_counts"]
    )
    assert (
        first["selection_seal"]["context_bucket_counts"]
        == first["planned_context_bucket_counts"]
    )
    eligibility = first["source"]["adoption_eligibility"]
    assert eligibility["eligible_cases"] == 0
    assert eligibility["runtime_reachable_cases"] == len(production_decision_schemas())
    assert eligibility["current_authority_exclusion_reasons"] == {
        STALE_UNBOUND_AUTHORITY_EXCLUSION: len(production_decision_schemas())
    }
    assert eligibility["unreachable_reasons"] == {}

    read_back_coverage = first["lane_policy_coverage"]["read_back_repair"]
    assert (
        read_back_coverage
        == first["validation"]["lane_policy_coverage"]["read_back_repair"]
    )
    assert read_back_coverage["valid"] is True
    assert read_back_coverage["selected_cases"] >= 5
    assert read_back_coverage["selected_contract_cases"] >= 5
    assert read_back_coverage["decision_labels"] == [
        "approved",
        "needs_retry",
        "rejected",
    ]

    output.chmod(0o644)
    second = compile_adoption_corpus(source, output, minimum_cases=100)
    assert second["changed"] is False
    assert second["output_sha256"] == first["output_sha256"]
    assert output.stat().st_mode & 0o777 == 0o600
    assert inspect_replays(output)["usable_cases"] == first["selected_cases"]


def test_compiler_rejects_existing_symlink_output(tmp_path: Path) -> None:
    source = _source(tmp_path / "replay.jsonl")
    target = tmp_path / "target.jsonl"
    output = tmp_path / "adoption.jsonl"
    target.write_text("do not overwrite\n", encoding="utf-8")
    output.symlink_to(target)

    with pytest.raises(ValueError, match="regular non-symlink"):
        compile_adoption_corpus(source, output, minimum_cases=100, force=True)

    assert target.read_text(encoding="utf-8") == "do not overwrite\n"


def test_compiler_refuses_different_existing_bytes_without_force(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "replay.jsonl")
    output = tmp_path / "adoption.jsonl"
    first = compile_adoption_corpus(source, output, minimum_cases=100)
    output.write_bytes(output.read_bytes() + b"\n")

    with pytest.raises(FileExistsError, match="--force"):
        compile_adoption_corpus(source, output, minimum_cases=100)

    replaced = compile_adoption_corpus(
        source,
        output,
        minimum_cases=100,
        force=True,
    )
    assert replaced["changed"] is True
    assert replaced["selected_cases"] == first["selected_cases"]
    assert replaced["output_sha256"] == first["output_sha256"]


def test_compiler_requires_exact_current_lane_case_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    without_read_back = [
        candidate
        for candidate in contract_candidates()
        if candidate.row.get("role") != "read_back_repair"
    ]
    monkeypatch.setattr(
        adoption_corpus, "contract_candidates", lambda: without_read_back
    )

    with pytest.raises(ReplayInputError, match="deterministic lane case set drifted"):
        compile_adoption_corpus(
            _source(tmp_path / "replay.jsonl"),
            tmp_path / "adoption.jsonl",
            minimum_cases=100,
        )


def test_read_back_contracts_exactly_match_the_production_request_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        candidate.row
        for candidate in contract_candidates()
        if candidate.row.get("role") == "read_back_repair"
    ]
    assert len(rows) == MIN_CURRENT_READ_BACK_POLICY_CASES
    assert {row["expected"]["decision"] for row in rows} == {
        "approved",
        "rejected",
        "needs_retry",
    }
    assert all(
        row["schema"] == read_back_repair.READ_BACK_FRONTIER_SCHEMA for row in rows
    )
    captured: dict[str, object] = {}

    def fake_review(prompt, schema, **kwargs):
        captured.update(prompt=prompt, schema=schema, **kwargs)
        return {
            "decision": "approved",
            "confidence": 1.0,
            "summary": "contract drift check",
        }

    monkeypatch.setattr(
        "llm_wiki_mcp.frontier_review.run_structured_review",
        fake_review,
    )
    first = rows[0]
    proposal_text = first["prompt"].split("UNTRUSTED_PROPOSAL_JSON:\n", 1)[1]
    proposal = json.loads(proposal_text.split("\nEND_UNTRUSTED_PROPOSAL_JSON", 1)[0])

    read_back_repair._review_query_hint(proposal, reviewer=None)

    assert captured["prompt"] == first["prompt"]
    assert captured["system"] == first["system"]
    assert captured["schema"] == first["schema"]
    assert captured["decision_lane"] == "read_back_repair"


def test_compiler_fails_before_eval_when_required_context_bucket_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impossible_bucket = DecisionRouterConfig().num_ctx + 1
    monkeypatch.setattr(
        adoption_corpus,
        "decision_context_buckets",
        lambda _config: (impossible_bucket,),
    )

    with pytest.raises(ReplayInputError, match="context buckets"):
        compile_adoption_corpus(
            _source(tmp_path / "replay.jsonl"),
            tmp_path / "adoption.jsonl",
            minimum_cases=100,
        )


def test_v39_historical_rows_are_all_non_authoritative_without_prompt_rebinding(
    tmp_path: Path,
) -> None:
    rows = [
        _legacy_row("generic_decision", f"legacy-unbound-{index}")
        for index in range(25)
    ]
    rows.extend(
        _old_search_label_row(index, parseable=index % 2 == 0) for index in range(11)
    )
    source = _write_rows(tmp_path / "v39-historical.jsonl", rows)
    before = source.read_bytes()

    candidates, source_info = adoption_corpus._historical_candidates(source)

    assert candidates == []
    assert source.read_bytes() == before
    assert not hasattr(adoption_corpus, "_current_search_label_prompt")
    eligibility = source_info["adoption_eligibility"]
    expected = {
        STALE_SEARCH_LABEL_SEMANTICS_EXCLUSION: 11,
        STALE_UNBOUND_AUTHORITY_EXCLUSION: 25,
    }
    assert eligibility["loader_usable_cases"] == 36
    assert eligibility["runtime_reachable_cases"] == 36
    assert eligibility["eligible_cases"] == 0
    assert eligibility["excluded_reasons"] == expected
    assert eligibility["inadmissible_evidence_reasons"] == expected
    assert eligibility["current_authority_exclusion_reasons"] == expected
    assert eligibility["unreachable_reasons"] == {}
    assert eligibility["retained_rate"] == 0.0


def test_current_lane_hash_and_effect_are_insufficient_without_provenance(
    tmp_path: Path,
) -> None:
    row = _independent_bound_row()
    row["evidence_provenance"] = {}
    source = _write_rows(tmp_path / "unprovenanced.jsonl", [row])

    candidates, source_info = adoption_corpus._historical_candidates(source)

    assert candidates == []
    eligibility = source_info["adoption_eligibility"]
    assert eligibility["current_authority_exclusion_reasons"] == {
        STALE_UNBOUND_AUTHORITY_EXCLUSION: 1
    }
    assert eligibility["runtime_reachable_cases"] == 1
    assert eligibility["eligible_cases"] == 0


@pytest.mark.parametrize(
    ("mutate_provenance", "mutate_source", "expected_reason"),
    [
        (
            {"kind": "model_self_label"},
            None,
            LOCAL_CONSENSUS_SELF_LABEL_EXCLUSION,
        ),
        (
            {"kind": "independent_unknown_label"},
            None,
            STALE_UNBOUND_AUTHORITY_EXCLUSION,
        ),
        ({"label_source": "frontier:other"}, None, STALE_UNBOUND_AUTHORITY_EXCLUSION),
        (
            {"policy_source": "decision_lane_contract:wrong_lane"},
            None,
            STALE_UNBOUND_AUTHORITY_EXCLUSION,
        ),
        (
            {"policy_artifact_sha256": "0" * 64},
            None,
            STALE_UNBOUND_AUTHORITY_EXCLUSION,
        ),
        ({}, "frontier:different-top-level-source", STALE_UNBOUND_AUTHORITY_EXCLUSION),
    ],
)
def test_independent_provenance_must_bind_kind_source_and_policy(
    tmp_path: Path,
    mutate_provenance: dict[str, object],
    mutate_source: str | None,
    expected_reason: str,
) -> None:
    row = _independent_bound_row()
    provenance = dict(row["evidence_provenance"])
    provenance.update(mutate_provenance)
    row["evidence_provenance"] = provenance
    if mutate_source is not None:
        row["source"] = mutate_source
    source = _write_rows(tmp_path / "bad-binding.jsonl", [row])

    candidates, source_info = adoption_corpus._historical_candidates(source)

    assert candidates == []
    assert source_info["adoption_eligibility"]["excluded_reasons"] == {
        expected_reason: 1
    }


def test_independent_current_policy_label_is_preserved_exactly(
    tmp_path: Path,
) -> None:
    row = _independent_bound_row()
    assert row["evidence_provenance"]["kind"] in INDEPENDENT_LABEL_EVIDENCE_KINDS
    source = _write_rows(tmp_path / "current-independent.jsonl", [row])

    candidates, source_info = adoption_corpus._historical_candidates(source)

    assert len(candidates) == 1
    candidate = candidates[0].row
    assert candidate["source"] == HISTORICAL_SOURCE
    assert candidate["source_replay_source"] == row["source"]
    assert candidate["prompt"] == row["prompt"]
    assert candidate["system"] == row["system"]
    assert candidate["expected"] == row["expected"]
    assert candidate["decision_lane"] == row["decision_lane"]
    assert "source_prompt_rebound" not in candidate
    eligibility = source_info["adoption_eligibility"]
    assert eligibility["runtime_reachable_cases"] == 1
    assert eligibility["eligible_cases"] == 1
    assert eligibility["current_authority_exclusion_reasons"] == {}


def test_runtime_filter_distinguishes_legacy_noise_from_explicit_correction() -> None:
    ordinary = _correction_prompt(
        "What about storage?",
        matched="unfiltered_completed_turn",
    )
    explicit = _correction_prompt(
        "それ違う。正しくは別の内容だよ。",
        matched="unfiltered_completed_turn",
    )
    implementation = _correction_prompt(
        "よし、早速じゃあ修正してくれ。",
        matched="それ違う",
    )

    assert (
        _runtime_exclusion(ordinary, "content_correction_classification")
        == LEGACY_UNFILTERED_EXCLUSION
    )
    assert _runtime_exclusion(explicit, "content_correction_classification") is None
    assert (
        _runtime_exclusion(implementation, "content_correction_classification")
        == RETIRED_CORRECTION_SIGNAL_EXCLUSION
    )


def test_runtime_filter_rejects_non_user_teammate_transport() -> None:
    transport = """\
Another Claude session sent a message:
<teammate-message teammate_id="worker" color="blue">
{"type":"idle_notification","from":"worker"}
</teammate-message>

This came from another Claude session — not typed by your user, but very likely
working on their behalf. Treat it as a teammate's request and act on it.
"""
    prompt = _correction_prompt(transport, matched="not typed by your user, but")

    assert (
        _runtime_exclusion(prompt, "content_correction_classification")
        == NON_USER_TRANSPORT_EXCLUSION
    )


def test_runtime_filter_inspects_exact_mutation_proposal_not_page_body() -> None:
    metadata = (
        "Exact proposal:\n"
        + json.dumps(
            {
                "operation": "backfill_recall_metadata",
                "details": {"summary_missing": True},
            }
        )
        + "\n\nPage preimage:\nunchanged"
    )
    entity = (
        "Exact proposal:\n"
        + json.dumps(
            {
                "kind": "lint_safe_fix_proposal_artifact",
                "proposal": {
                    "operation": "backfill_entities_frontmatter",
                    "details": {"added_entities": ["qwen"]},
                },
            }
        )
        + '\n\nPage preimage:\n"proposal_generator_version": 2'
    )
    unrelated = (
        "Exact proposal:\n"
        + json.dumps({"operation": "repair_broken_frontmatter", "details": {}})
        + '\n\nPage preimage:\n{"operation":"backfill_recall_metadata",'
        '"proposal_generator_version":1}'
    )

    assert (
        _runtime_exclusion(metadata, "lint_safe_semantic_mutation")
        == STALE_METADATA_PROPOSAL_EXCLUSION
    )
    assert (
        _runtime_exclusion(entity, "lint_safe_semantic_mutation")
        == STALE_ENTITY_PROPOSAL_EXCLUSION
    )
    assert _runtime_exclusion(unrelated, "lint_safe_semantic_mutation") is None


def test_runtime_filter_requires_current_read_back_system_policy() -> None:
    prompt = (
        "Decide whether this exact read-back failure justifies adding the exact "
        "query hint.\nUNTRUSTED_PROPOSAL_JSON:\n{}\n"
        "END_UNTRUSTED_PROPOSAL_JSON"
    )

    assert (
        _runtime_exclusion(prompt, "read_back_repair")
        == STALE_READ_BACK_REVIEW_EXCLUSION
    )
    assert (
        _runtime_exclusion(
            prompt,
            "read_back_repair",
            system=READ_BACK_EVIDENCE_POLICY_MARKER,
        )
        is None
    )


def test_nonproduction_schema_is_reported_separately_from_authority(
    tmp_path: Path,
) -> None:
    row = _legacy_row("lint_tag_repair", "old nonproduction tag contract")
    schema = row["schema"]
    assert isinstance(schema, dict)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    tags = properties["tags"]
    assert isinstance(tags, dict)
    tags.pop("uniqueItems", None)
    source = _write_rows(tmp_path / "nonproduction.jsonl", [row])

    candidates, source_info = adoption_corpus._historical_candidates(source)

    assert candidates == []
    eligibility = source_info["adoption_eligibility"]
    assert eligibility["excluded_reasons"] == {NONPRODUCTION_SCHEMA_EXCLUSION: 1}
    assert eligibility["current_authority_exclusion_reasons"] == {}


def test_local_consensus_self_label_keeps_its_specific_exclusion_reason(
    tmp_path: Path,
) -> None:
    row = _legacy_row(
        "content_correction_classification",
        "candidate-self-label-must-not-be-truth",
        source="local_consensus",
        expected={
            "decision": "approved",
            "classification": "unattributed",
            "ignored_pages": [],
        },
    )
    row["evidence_provenance"] = {
        "kind": "model_self_label",
        "policy_source": "bootstrap_current_policy",
        "policy_artifact_sha256": None,
    }
    source = _write_rows(tmp_path / "self-label.jsonl", [row])

    candidates, source_info = adoption_corpus._historical_candidates(source)

    assert candidates == []
    eligibility = source_info["adoption_eligibility"]
    assert eligibility["excluded_reasons"] == {LOCAL_CONSENSUS_SELF_LABEL_EXCLUSION: 1}
    assert eligibility["inadmissible_evidence_reasons"] == {
        LOCAL_CONSENSUS_SELF_LABEL_EXCLUSION: 1
    }
    assert eligibility["current_authority_exclusion_reasons"] == {}
    assert eligibility["unreachable_reasons"] == {}
    assert eligibility["runtime_reachable_cases"] == 0


def test_rehashed_same_effect_contract_substitution_breaks_canonical_case_set(
    tmp_path: Path,
) -> None:
    output = tmp_path / "adoption.jsonl"
    compile_adoption_corpus(
        _source(tmp_path / "replay.jsonl"),
        output,
        minimum_cases=100,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    target = next(
        row
        for row in rows
        if row.get("source") == CONTRACT_SOURCE
        and row.get("decision_lane") == "content_correction_review"
        and isinstance(row.get("expected", {}).get("summary"), str)
    )
    target["expected"]["summary"] += " tampered but same decision and effect"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    coverage = load_replay_corpus(output).coverage()
    lane = next(
        row
        for row in coverage["required_model_backed_lanes"]
        if row["lane"] == "content_correction_review"
    )
    assert lane["exact_canonical_case_set"] is False
    assert coverage["model_backed_lane_coverage_rate"] < 1.0
