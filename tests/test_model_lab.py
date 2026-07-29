from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chronovisor.lab import model_lab


def _isolate(monkeypatch, tmp_path: Path) -> None:
    lab = tmp_path / "model-lab"
    monkeypatch.setattr(model_lab, "LAB_DIR", lab)
    monkeypatch.setattr(model_lab, "POLICY_FILE", lab / "active-policy.json")
    monkeypatch.setattr(model_lab, "STATE_FILE", lab / "state.json")
    monkeypatch.setattr(model_lab, "REPLAY_FILE", lab / "replay.jsonl")
    monkeypatch.setattr(model_lab, "HISTORY_FILE", lab / "history.jsonl")
    monkeypatch.setattr(model_lab, "LOCK_FILE", lab / "model-lab.lock")


def _cache(tmp_path: Path) -> Path:
    path = tmp_path / "models_cache.json"
    path.write_text(
        json.dumps(
            {
                "fetched_at": "2026-07-11T00:00:00Z",
                "models": [
                    {
                        "slug": f"gpt-5.6-{tier}",
                        "visibility": "list",
                        "priority": index,
                        "supported_reasoning_levels": [
                            {"effort": effort} for effort in ["low", "medium", "high"]
                        ],
                    }
                    for index, tier in enumerate(["luna", "terra", "sol"], start=1)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_bootstrap_routes_only_code_repair_to_latest_tier(
    monkeypatch, tmp_path: Path
) -> None:
    _isolate(monkeypatch, tmp_path)
    discovery = model_lab.discover_models([_cache(tmp_path)])
    policy = model_lab.bootstrap_policy(write=True, discovery=discovery)

    assert set(policy["roles"]) == {"code_repair"}
    assert policy["roles"]["code_repair"]["model"] == "gpt-5.6-sol"
    assert policy["roles"]["code_repair"]["effort"] == "high"


def test_discovery_promotes_repair_model_without_frontier_replay_then_rolls_back_bad_canary(
    monkeypatch, tmp_path: Path
) -> None:
    _isolate(monkeypatch, tmp_path)
    cache = _cache(tmp_path)
    monkeypatch.setattr(model_lab, "cache_paths", lambda: [cache])
    model_lab.POLICY_FILE.parent.mkdir(parents=True)
    old = model_lab.bootstrap_policy(write=False, discovery={"latest": {}})
    model_lab.POLICY_FILE.write_text(json.dumps(old), encoding="utf-8")

    result = model_lab.run_due()

    assert result["promoted"] == ["code_repair"]
    assert model_lab.load_policy()["roles"]["code_repair"]["model"] == "gpt-5.6-sol"
    for _ in range(5):
        model_lab.record_live_result(
            role="code_repair",
            model="gpt-5.6-sol",
            ok=False,
            failure_class="test_failure",
        )
    assert model_lab.load_policy()["roles"]["code_repair"]["model"] == "gpt-5.5"


def test_unknown_frontier_role_is_hard_rejected(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)

    try:
        model_lab.resolve_role("semantic_judge")
    except ValueError as exc:
        assert "not permitted" in str(exc)
    else:
        raise AssertionError("routine frontier roles must not resolve")


def test_replay_record_marks_prompt_truncation_explicitly(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("CHRONOVISOR_MODEL_LAB_REPLAY", raising=False)

    model_lab.record_replay_case(
        role="code_repair",
        prompt="x" * 50_001,
        schema={"type": "object"},
        result={"decision": "approved"},
        model="gpt-test",
        effort="high",
        latency_seconds=1.25,
    )
    model_lab.record_replay_case(
        role="code_repair",
        prompt="complete",
        schema={"type": "object"},
        result={"decision": "approved"},
        model="gpt-test",
        effort="high",
        latency_seconds=0.5,
    )

    rows = [json.loads(line) for line in model_lab.REPLAY_FILE.read_text().splitlines()]
    assert len(rows[0]["prompt"]) == 50_000
    assert rows[0]["prompt_truncated"] is True
    assert rows[0]["prompt_original_chars"] == 50_001
    assert rows[1]["prompt"] == "complete"
    assert rows[1]["prompt_truncated"] is False
    assert rows[1]["prompt_original_chars"] == 8


def test_local_consensus_replay_over_cap_is_explicitly_non_adoptable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _isolate(monkeypatch, tmp_path)

    written = model_lab.record_local_replay_case(
        role="local_repair",
        prompt="z" * 50_001,
        schema={
            "type": "object",
            "properties": {"action": {"type": "string"}},
        },
        result={"action": "retry_raw", "reason": "bounded"},
        models=["ornith:test", "gpt-oss:test"],
        latency_seconds=2.0,
        policy_source="bootstrap_current_policy",
        policy_artifact_sha256=None,
    )

    row = json.loads(model_lab.REPLAY_FILE.read_text())
    assert written is True
    assert len(row["prompt"]) == 50_000
    assert row["prompt_truncated"] is True
    assert row["prompt_original_chars"] == 50_001
    assert row["expected"] == {"action": "retry_raw"}
    assert row["source"] == "local_consensus"
    assert row["evidence_provenance"] == {
        "kind": "model_self_label",
        "policy_source": "bootstrap_current_policy",
        "policy_artifact_sha256": None,
    }


def test_local_consensus_replay_marks_truncated_system_as_non_adoptable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _isolate(monkeypatch, tmp_path)

    written = model_lab.record_local_replay_case(
        role="content_correction",
        prompt="complete prompt",
        system="s" * 50_001,
        schema={
            "type": "object",
            "properties": {"decision": {"type": "string"}},
        },
        result={"decision": "apply"},
        models=["ornith:test", "gpt-oss:test"],
        latency_seconds=1.0,
    )

    row = json.loads(model_lab.REPLAY_FILE.read_text())
    assert written is True
    assert row["prompt"] == "complete prompt"
    assert len(row["system"]) == 50_000
    assert row["prompt_truncated"] is True
    assert row["prompt_original_chars"] == len("complete prompt")
    assert row["system_original_chars"] == 50_001


def test_host_sidecar_does_not_false_truncate_bounded_model_prompt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _isolate(monkeypatch, tmp_path)
    model_prompt = "bounded model request"
    host_bound_prompt = model_prompt + "\n" + ("h" * 60_000)
    base_system = "lane-bound base system"
    effective_system = base_system + "\n\neffective decision overlay"

    written = model_lab.record_local_replay_case(
        role="content_correction",
        prompt=host_bound_prompt,
        effective_model_prompt=model_prompt,
        system=base_system,
        effective_model_system=effective_system,
        schema={
            "type": "object",
            "properties": {"decision": {"type": "string"}},
        },
        result={"decision": "apply"},
        models=["ornith:test", "gpt-oss:test"],
        latency_seconds=1.0,
        decision_lane="content_correction_classification",
        lane_contract_sha256="a" * 64,
        lane_contract_effect="negative_retrieval_feedback",
        effective_request_sha256="b" * 64,
    )

    row = json.loads(model_lab.REPLAY_FILE.read_text())
    assert written is True
    assert row["prompt"] == host_bound_prompt
    assert row["prompt_truncated"] is False
    assert row["prompt_original_chars"] == len(host_bound_prompt)
    assert row["effective_model_prompt_chars"] == len(model_prompt)
    assert (
        row["effective_model_prompt_sha256"]
        == hashlib.sha256(model_prompt.encode("utf-8")).hexdigest()
    )
    assert row["host_sidecar_present"] is True
    assert row["system"] == base_system
    assert row["effective_model_system"] == effective_system
    assert row["effective_model_system_chars"] == len(effective_system)
    assert (
        row["effective_model_system_sha256"]
        == hashlib.sha256(effective_system.encode("utf-8")).hexdigest()
    )
    assert row["lane_contract_effect"] == "negative_retrieval_feedback"
    assert row["effective_request_sha256"] == "b" * 64
