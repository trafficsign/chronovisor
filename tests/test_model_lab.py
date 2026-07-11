from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp import model_lab


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
    path.write_text(json.dumps({
        "fetched_at": "2026-07-11T00:00:00Z",
        "models": [
            {"slug": f"gpt-5.6-{tier}", "visibility": "list", "priority": index,
             "supported_reasoning_levels": [{"effort": effort} for effort in ["low", "medium", "high"]]}
            for index, tier in enumerate(["luna", "terra", "sol"], start=1)
        ],
    }), encoding="utf-8")
    return path


def test_bootstrap_routes_roles_to_latest_tiers(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    discovery = model_lab.discover_models([_cache(tmp_path)])
    policy = model_lab.bootstrap_policy(write=True, discovery=discovery)

    assert policy["roles"]["raw_writer"] == {
        "model": "gpt-5.6-luna", "effort": "low", "tier": "luna", "source": "codex-model-cache",
    }
    assert policy["roles"]["semantic_judge"]["model"] == "gpt-5.6-terra"
    assert policy["roles"]["mutation_approver"]["effort"] == "low"
    assert policy["roles"]["code_repair"]["effort"] == "high"


def test_replay_gate_promotes_then_rolls_back_bad_canary(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    cache = _cache(tmp_path)
    monkeypatch.setattr(model_lab, "cache_paths", lambda: [cache])
    monkeypatch.setenv("LLM_WIKI_MODEL_LAB_MIN_REPLAYS", "3")
    model_lab.POLICY_FILE.parent.mkdir(parents=True)
    old = model_lab.bootstrap_policy(write=False, discovery={"latest": {}})
    model_lab.POLICY_FILE.write_text(json.dumps(old), encoding="utf-8")
    with model_lab.REPLAY_FILE.open("w", encoding="utf-8") as handle:
        for _ in range(3):
            handle.write(json.dumps({
                "role": "raw_writer", "prompt": "p", "schema": {"type": "object"},
                "expected": {"decision": "approved"},
            }) + "\n")

    result = model_lab.run_due(
        max_evaluations=3,
        reviewer=lambda _case, _candidate: {"decision": "approved"},
    )

    assert result["promoted"] == ["raw_writer"]
    assert model_lab.load_policy()["roles"]["raw_writer"]["model"] == "gpt-5.6-luna"
    for _ in range(5):
        model_lab.record_live_result(role="raw_writer", model="gpt-5.6-luna", ok=False, failure_class="schema_invalid")
    assert model_lab.load_policy()["roles"]["raw_writer"]["model"] == "gpt-5.4-mini"
