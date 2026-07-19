from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor import runtime_config


def test_uvx_runtime_command_uses_pushed_github_source(monkeypatch) -> None:
    monkeypatch.delenv("CHRONOVISOR_RUNTIME_SOURCE", raising=False)

    command = runtime_config.uvx_runtime_command(
        "chronovisor-sleep",
        executable="/opt/homebrew/bin/uvx",
        refresh=True,
    )

    assert command == [
        "/opt/homebrew/bin/uvx",
        "--refresh-package",
        "chronovisor",
        "--from",
        "git+ssh://git@github.com/trafficsign/chronovisor",
        "chronovisor-sleep",
    ]


def test_runtime_source_override_is_explicit(monkeypatch) -> None:
    monkeypatch.setenv("CHRONOVISOR_RUNTIME_SOURCE", "git+ssh://example.invalid/fork")

    assert runtime_config.runtime_source() == "git+ssh://example.invalid/fork"


def test_runtime_repo_root_honors_explicit_checkout(tmp_path, monkeypatch) -> None:
    checkout = tmp_path / "chronovisor"
    monkeypatch.setenv("CHRONOVISOR_REPO_ROOT", str(checkout))

    assert runtime_config.runtime_repo_root() == checkout


def test_active_config_uses_canonical_config(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[hooks.stop]\naudit = true\n", encoding="utf-8")
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)

    assert runtime_config.active_config_file() == config
    assert runtime_config.config_summary()["mode"] == "canonical"


def test_hook_policy_reads_nested_hooks_section(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[hooks.user_prompt]
recall = false

[hooks.stop]
save = true
audit = false
content_correction = false
recall_improve = false
""",
        encoding="utf-8",
    )

    policy = runtime_config.load_hook_policy(config)

    assert policy.user_prompt_recall is False
    assert policy.stop_save is True
    assert policy.stop_audit is False
    assert policy.stop_content_correction is False
    assert policy.stop_recall_improve is False


def test_embedding_config_reads_model_and_prefixes(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
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

    cfg = runtime_config.load_embedding_config()

    assert cfg.model == "bge-m3"
    assert cfg.document_prefix == "検索文書: "
    assert cfg.query_prefix == "検索クエリ: "


def test_ingest_config_reads_ollama_generation_knobs(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[ingest]
model = "qwen3.6:35b-a3b-mxfp8"
keep_alive = "10m"
temperature = 0.1
num_ctx = 32768
max_num_ctx = 131072
num_predict = 4096
read_timeout_ms = 120000
memory_reserve_gib = 24
max_related_context_bytes = 12288
semantic_projection_max_child_bytes = 16384
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)

    cfg = runtime_config.load_ingest_config()

    assert cfg.model == "qwen3.6:35b-a3b-mxfp8"
    assert cfg.keep_alive == "10m"
    assert cfg.temperature == 0.1
    assert cfg.num_ctx == 32768
    assert cfg.max_num_ctx == 131072
    assert cfg.num_predict == 4096
    assert cfg.read_timeout_ms == 120000
    assert cfg.memory_reserve_gib == 24
    assert cfg.max_related_context_bytes == 12288
    assert cfg.semantic_projection_max_child_bytes == 16384


def test_ingest_config_defaults_to_dynamic_context_envelope(tmp_path: Path) -> None:
    cfg = runtime_config.load_ingest_config(tmp_path / "missing.toml")

    assert cfg.num_ctx == 32768
    assert cfg.max_num_ctx == 262144
    assert cfg.memory_reserve_gib == 16
    assert cfg.max_related_context_bytes == 8192
    assert cfg.semantic_projection_max_child_bytes == 24000


def test_ingest_projection_child_envelope_rejects_unsafe_override(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[ingest]\nsemantic_projection_max_child_bytes = 48000\n",
        encoding="utf-8",
    )

    cfg = runtime_config.load_ingest_config(config)

    assert cfg.semantic_projection_max_child_bytes == 24000


def test_ingest_audit_config_reads_risk_sampling_knobs(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[ingest.audit]
enabled = true
sample_rate = 0.05
update_sample_rate = 0.12
noop_sample_rate = 0.25
adaptive = true
adaptive_window = 40
adaptive_min_audits = 4
elevated_reject_rate = 0.08
critical_reject_rate = 0.18
elevated_sample_rate = 0.30
critical_sample_rate = 0.60
max_sample_rate = 0.09
max_operations_without_audit = 3
""",
        encoding="utf-8",
    )

    cfg = runtime_config.load_ingest_audit_config(config)

    assert cfg.sample_rate == 0.05
    assert cfg.update_sample_rate == 0.12
    assert cfg.noop_sample_rate == 0.25
    assert cfg.adaptive_window == 40
    assert cfg.adaptive_min_audits == 4
    assert cfg.elevated_sample_rate == 0.30
    assert cfg.critical_sample_rate == 0.60
    assert cfg.max_sample_rate == 0.09
    assert cfg.max_operations_without_audit == 3


def test_decision_router_config_defaults_to_local_three_model_quorum() -> None:
    cfg = runtime_config.load_decision_router_config("/does/not/exist.toml")

    assert cfg.primary_model == "maxwell1500/ornith-35b:Q5_K_M"
    assert cfg.challenger_model == "gpt-oss:20b"
    assert cfg.tie_break_model == "gemma4:26b"
    assert cfg.num_ctx == 114688
    assert cfg.min_num_ctx == 16384
    assert cfg.num_predict == 3072
    assert cfg.max_input_chars == 93000
    assert cfg.max_output_chars == 4000
    assert cfg.max_feedback_chars == 2000
    assert cfg.quorum == 2
    assert cfg.adaptive_residency is True
    assert cfg.residency_policy_version == 2
    assert cfg.memory_reserve_gib == 16
    assert cfg.max_resident_models == 3


def test_decision_router_config_allows_exact_installed_tag_overrides(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[decision_router]
primary_model = "ornith-local:35b-q5"
challenger_model = "gpt-oss:20b"
tie_break_model = "gemma4:26b-mxfp8"
primary_keep_alive = "21m"
challenger_keep_alive = "19m"
tie_break_keep_alive = "90s"
num_ctx = 16384
min_num_ctx = 8192
num_predict = 2048
read_timeout_ms = 180000
max_input_chars = 80000
max_output_chars = 12000
max_feedback_chars = 6000
quorum = 2
adaptive_residency = false
residency_policy_version = 2
memory_reserve_gib = 20
max_resident_models = 2
adoption_artifact = "~/.chronovisor/runtime/model-lab/candidate.json"
""",
        encoding="utf-8",
    )

    cfg = runtime_config.load_decision_router_config(config)

    assert cfg.primary_model == "ornith-local:35b-q5"
    assert cfg.challenger_model == "gpt-oss:20b"
    assert cfg.tie_break_model == "gemma4:26b-mxfp8"
    assert cfg.primary_keep_alive == "21m"
    assert cfg.challenger_keep_alive == "19m"
    assert cfg.tie_break_keep_alive == "90s"
    assert cfg.num_ctx == 16384
    assert cfg.min_num_ctx == 8192
    assert cfg.num_predict == 2048
    assert cfg.read_timeout_ms == 180000
    assert cfg.max_input_chars == 80000
    assert cfg.max_output_chars == 12000
    assert cfg.max_feedback_chars == 6000
    assert cfg.quorum == 2
    assert cfg.adaptive_residency is False
    assert cfg.residency_policy_version == 2
    assert cfg.memory_reserve_gib == 20
    assert cfg.max_resident_models == 2
    assert cfg.adoption_artifact == "~/.chronovisor/runtime/model-lab/candidate.json"


def test_decision_router_config_accepts_tie_model_alias(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[decision_router]
tie_model = "gemma4:26b-local"
tie_keep_alive = "3m"
""",
        encoding="utf-8",
    )

    cfg = runtime_config.load_decision_router_config(config)

    assert cfg.tie_break_model == "gemma4:26b-local"
    assert cfg.tie_break_keep_alive == "3m"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("not = [valid", "malformed TOML"),
        ("[recall]\nenabled = true\n", "requires a non-empty"),
        ("[decision_router]\nnum_ctx = 'large'\n", "num_ctx must be an integer"),
        ("[decision_router]\nprimry_model = 'typo'\n", "unknown"),
        (
            "[decision_router]\nadoption_artifact = 'old.json'\n",
            "adoption_artifact must be empty",
        ),
    ],
)
def test_candidate_decision_router_config_fails_closed(
    tmp_path: Path, body: str, message: str
) -> None:
    config = tmp_path / "candidate.toml"
    config.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        runtime_config.load_candidate_decision_router_config(config)


def test_candidate_decision_router_config_rejects_missing_file() -> None:
    with pytest.raises(ValueError, match="unreadable"):
        runtime_config.load_candidate_decision_router_config("/does/not/exist.toml")


def test_reranker_config_reads_nested_search_section(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.toml"
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

    cfg = runtime_config.load_reranker_config()

    assert cfg.enabled is True
    assert cfg.model == "BAAI/bge-reranker-v2-m3"
    assert cfg.backend == "transformers"
    assert cfg.top_n == 20
    assert cfg.max_length == 1024
    assert cfg.batch_size == 4
    assert cfg.device == "mps"
    assert cfg.weight == 0.4
