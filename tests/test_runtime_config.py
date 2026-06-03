from __future__ import annotations

from pathlib import Path

from llm_wiki_mcp import runtime_config


def test_active_config_prefers_unified_config(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    legacy = tmp_path / "recall.toml"
    config.write_text("[hooks.stop]\naudit = true\n", encoding="utf-8")
    legacy.write_text("enabled = false\n", encoding="utf-8")
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)
    monkeypatch.setattr(runtime_config, "LEGACY_RECALL_CONFIG_FILE", legacy)

    assert runtime_config.active_config_file() == config
    assert runtime_config.config_summary()["mode"] == "unified"


def test_active_config_falls_back_to_legacy_recall(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    legacy = tmp_path / "recall.toml"
    legacy.write_text("enabled = true\n", encoding="utf-8")
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)
    monkeypatch.setattr(runtime_config, "LEGACY_RECALL_CONFIG_FILE", legacy)

    assert runtime_config.active_config_file() == legacy
    assert runtime_config.config_summary()["mode"] == "legacy-recall"


def test_normalize_recall_config_maps_nested_unified_shape() -> None:
    normalized = runtime_config.normalize_recall_config(
        {
            "recall": {
                "enabled": True,
                "semantic": False,
                "judge_mode": "auto",
                "thresholds": {"search": 0.2, "read": 0.8},
                "gate": {"model": "qwen3.5:4b", "timeout_ms": 1000},
                "policy": {"fail_silent_on_judge_unavailable": True},
            }
        }
    )

    assert normalized["enabled"] is True
    assert normalized["thresholds"]["read"] == 0.8
    assert normalized["gate"]["model"] == "qwen3.5:4b"
    assert normalized["recall"] == {"semantic": False, "judge_mode": "auto"}


def test_hook_policy_reads_nested_hooks_section(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[hooks.user_prompt]
recall = false

[hooks.stop]
save = true
audit = false
""",
        encoding="utf-8",
    )

    policy = runtime_config.load_hook_policy(config)

    assert policy.user_prompt_recall is False
    assert policy.stop_save is True
    assert policy.stop_audit is False
