from __future__ import annotations

from types import SimpleNamespace

from chronovisor.recall import classification_model_worker
from chronovisor.recall.classification_engine import CONSENSUS_SCHEMA


def test_dual_blind_hides_primary_from_challenger_and_can_adjudicate_hold(
    tmp_path,
    monkeypatch,
) -> None:
    runtime_routes = tuple(
        {
            "role": f"classification.{role}",
            "provider": "remote",
            "model": model,
            "location": "remote",
            "model_digest": None,
        }
        for role, model in (
            ("primary", "ornith"),
            ("challenger", "gpt-oss"),
            ("tie_break", "gemma"),
        )
    )
    monkeypatch.setattr(
        classification_model_worker,
        "resolve_consensus_runtime_routes",
        lambda _supplied=None: runtime_routes,
    )
    monkeypatch.setattr(
        classification_model_worker,
        "load_decision_router_config",
        lambda: SimpleNamespace(
            primary_model="ornith",
            challenger_model="gpt-oss",
            tie_break_model="gemma",
            primary_keep_alive="1m",
            challenger_keep_alive="1m",
            tie_break_keep_alive="1m",
        ),
    )
    calls = []

    def stage_call(**kwargs):
        calls.append(kwargs)
        page = kwargs["pages"][0]
        return (
            [
                {
                    "uid": page["uid"],
                    "primary_notation": "004.8",
                    "secondary_notations": [],
                    "confidence": 0.9,
                    "rationale": "insufficiently specific",
                    "expected_status": "held",
                }
            ],
            1,
        )

    monkeypatch.setattr(
        classification_model_worker,
        "_cached_stage_call",
        stage_call,
    )
    result = classification_model_worker.run(
        {
            "schema": CONSENSUS_SCHEMA,
            "root": str(tmp_path),
            "adjudication_mode": "dual-blind",
            "pages": [
                {
                    "uid": "uid-1",
                    "source_sha256": "sha256:page",
                    "title": "Ambiguous note",
                    "candidates": [
                        {
                            "notation": "004.8",
                            "label_en": "Artificial intelligence",
                        }
                    ],
                }
            ],
        }
    )

    challenger = next(call for call in calls if call["stage"] == "challenger")
    assert challenger["prior"] is None
    assert challenger["dual_blind"] is True
    assert result["decisions"][0]["status"] == "held"
    assert result["decisions"][0]["expected_status"] == "held"
    assert result["decisions"][0]["quorum"] == 2
    assert result["runtime_routes"] == list(runtime_routes)
