from __future__ import annotations

from chronovisor.classification.classification_evidence_judgment import (
    latin_square_order,
    paired_rows,
    resource_ready_gate,
)


def test_paired_judgment_keeps_candidates_identical_and_hides_evidence_in_j1() -> None:
    pages = [{"uid": "a", "title": "AI"}]
    provider = [
        {
            "uid": "a",
            "union": [
                {"notation": "0", "label_en": "General"},
                {"notation": "004.8", "label_en": "AI", "broader_notation": "004"},
            ],
            "external_only": [
                {
                    "notation": "004.8",
                    "source_support": [{"source": "czech", "record_id": "1"}],
                }
            ],
            "query_expansion": [
                {
                    "source": "ndlsh",
                    "label": "人工知能",
                    "vocabulary_role": "C1",
                    "direct_udc_vote": False,
                }
            ],
        }
    ]

    rows = paired_rows(pages, provider)

    assert rows["J1"][0]["candidates"] == rows["J2"][0]["candidates"]
    assert rows["J2"][0]["candidates"] == rows["J3"][0]["candidates"]
    assert rows["J1"][0]["evidence_card"]["source_support"] == []
    assert rows["J1"][0]["evidence_card"]["query_expansion"] == []
    assert rows["J2"][0]["evidence_card"]["source_support"]
    assert rows["J2"][0]["evidence_card"]["query_expansion"] == []
    assert rows["J3"][0]["evidence_card"]["query_expansion"]
    assert set(latin_square_order("a")) == {"J1", "J2", "J3"}


def test_resource_gate_is_fail_closed() -> None:
    failed = resource_ready_gate(
        recall_latencies_ms=[100.0] * 29,
        recall_misses=0,
        cancel_to_ready_ms=[100.0],
        protected_models=["ornith"],
        resident_models=["ornith"],
    )
    passed = resource_ready_gate(
        recall_latencies_ms=[100.0] * 30,
        recall_misses=0,
        cancel_to_ready_ms=[200.0],
        protected_models=["ornith"],
        resident_models=["ornith", "gemma"],
    )

    assert failed["status"] == "failed"
    assert failed["gates"]["sample_size"] is False
    assert passed["status"] == "passed"
