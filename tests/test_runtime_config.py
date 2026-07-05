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
                "gate": {"model": "qwen3.5:4b-mlx", "timeout_ms": 1000},
                "policy": {"fail_silent_on_judge_unavailable": True},
            }
        }
    )

    assert normalized["enabled"] is True
    assert normalized["thresholds"]["read"] == 0.8
    assert normalized["gate"]["model"] == "qwen3.5:4b-mlx"
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
recall_improve = false
""",
        encoding="utf-8",
    )

    policy = runtime_config.load_hook_policy(config)

    assert policy.user_prompt_recall is False
    assert policy.stop_save is True
    assert policy.stop_audit is False
    assert policy.stop_recall_improve is False


def test_embedding_config_reads_model_and_prefixes(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    legacy = tmp_path / "recall.toml"
    config.write_text(
        """
[embedding]
model = "bge-m3"
document_prefix = "検索文書: "
query_prefix = "検索クエリ: "
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)
    monkeypatch.setattr(runtime_config, "LEGACY_RECALL_CONFIG_FILE", legacy)

    cfg = runtime_config.load_embedding_config()

    assert cfg.model == "bge-m3"
    assert cfg.document_prefix == "検索文書: "
    assert cfg.query_prefix == "検索クエリ: "


def test_reranker_config_reads_nested_search_section(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    legacy = tmp_path / "recall.toml"
    config.write_text(
        """
[search.reranker]
enabled = true
model = "BAAI/bge-reranker-v2-m3"
backend = "transformers"
top_n = 20
max_length = 1024
batch_size = 4
device = "mps"
weight = 0.4
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)
    monkeypatch.setattr(runtime_config, "LEGACY_RECALL_CONFIG_FILE", legacy)

    cfg = runtime_config.load_reranker_config()

    assert cfg.enabled is True
    assert cfg.model == "BAAI/bge-reranker-v2-m3"
    assert cfg.backend == "transformers"
    assert cfg.top_n == 20
    assert cfg.max_length == 1024
    assert cfg.batch_size == 4
    assert cfg.device == "mps"
    assert cfg.weight == 0.4
