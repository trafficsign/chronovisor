from __future__ import annotations

import os
from pathlib import Path

import pytest

from chronovisor.core import runtime_config


def test_toml_loader_uses_one_old_or_new_snapshot_during_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.toml"
    replacement = tmp_path / "config.next.toml"
    old = '[decision_router]\nprimary_keep_alive = "21m"\n'
    new = '[decision_router]\nprimary_keep_alive = "22m"\n'
    config.write_text(old, encoding="utf-8")
    replacement.write_text(new, encoding="utf-8")
    real_read_bytes = Path.read_bytes
    replaced = False

    def read_once(path: Path) -> bytes:
        nonlocal replaced
        snapshot = real_read_bytes(path)
        if path == config and not replaced:
            os.replace(replacement, config)
            replaced = True
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", read_once)

    first = runtime_config.load_decision_router_config(config)
    second = runtime_config.load_decision_router_config(config)

    assert first.primary_keep_alive == "21m"
    assert second.primary_keep_alive == "22m"
    assert config.read_text(encoding="utf-8") == new


def test_uvx_runtime_command_uses_pushed_github_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "python3.14"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    monkeypatch.setenv("CHRONOVISOR_PYTHON", str(python))
    monkeypatch.delenv("CHRONOVISOR_RUNTIME_SOURCE", raising=False)

    command = runtime_config.uvx_runtime_command(
        "chronovisor-sleep",
        executable="/opt/homebrew/bin/uvx",
        refresh=True,
    )

    assert command == [
        "/opt/homebrew/bin/uvx",
        "--python",
        str(python),
        "--refresh-package",
        "chronovisor",
        "--from",
        "git+ssh://git@github.com/trafficsign/chronovisor",
        "chronovisor-sleep",
    ]


def test_runtime_python_rejects_archive_and_failed_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archived = tmp_path / "archive-v0" / "runtime" / "bin" / "python3.14"
    archived.parent.mkdir(parents=True)
    archived.write_text("#!/bin/sh\n", encoding="utf-8")
    archived.chmod(0o755)
    stable_link = tmp_path / "stable" / "python3.14"
    stable_link.parent.mkdir()
    stable_link.symlink_to(archived)
    monkeypatch.setenv("CHRONOVISOR_PYTHON", str(stable_link))

    with pytest.raises(RuntimeError, match="executable not found"):
        runtime_config.resolve_runtime_python()

    failed = tmp_path / "python3.14"
    failed.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    failed.chmod(0o755)
    monkeypatch.setenv("CHRONOVISOR_PYTHON", str(failed))

    with pytest.raises(RuntimeError, match="standard GIL Python 3.14"):
        runtime_config.resolve_runtime_python()


def test_runtime_source_override_is_explicit(monkeypatch) -> None:
    monkeypatch.setenv("CHRONOVISOR_RUNTIME_SOURCE", "git+ssh://example.invalid/fork")

    assert runtime_config.runtime_source() == "git+ssh://example.invalid/fork"


def test_runtime_identity_and_local_endpoints_are_configurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[runtime]
source = "git+ssh://git@github.com/example/chronovisor"
github_repository = "example/chronovisor"
user_agent = "ExampleChronovisor/1.0"
launchd_label_prefix = "org.example.chronovisor-"
ollama_url = "http://127.0.0.1:22434"

[dashboard]
host = "localhost"
port = 9876
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)
    monkeypatch.delenv("CHRONOVISOR_RUNTIME_SOURCE", raising=False)
    monkeypatch.delenv("CHRONOVISOR_GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("CHRONOVISOR_USER_AGENT", raising=False)
    monkeypatch.delenv("CHRONOVISOR_LAUNCHD_LABEL_PREFIX", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)

    assert runtime_config.runtime_source() == (
        "git+ssh://git@github.com/example/chronovisor"
    )
    assert runtime_config.github_repository() == "example/chronovisor"
    assert runtime_config.user_agent() == "ExampleChronovisor/1.0"
    assert runtime_config.launchd_label("dashboard") == (
        "org.example.chronovisor-dashboard"
    )
    assert runtime_config.ollama_url() == "http://127.0.0.1:22434"
    assert runtime_config.dashboard_url() == "http://localhost:9876"
    assert runtime_config.runtime_identity(config_only=True) == {
        "github_repository": "example/chronovisor",
        "user_agent": "ExampleChronovisor/1.0",
        "launchd_label_prefix": "org.example.chronovisor-",
        "ollama_url": "http://127.0.0.1:22434",
        "dashboard": {
            "host": "localhost",
            "port": 9876,
            "url": "http://localhost:9876",
        },
        "dashboard_lan": {
            "host": None,
            "port": 8766,
            "tls_cert_file": None,
            "tls_key_file": None,
            "credentials_file": None,
        },
    }
    assert runtime_config.load_toml_file(config, runtime_defaults=True)["runtime"] == {
        "source": "git+ssh://git@github.com/example/chronovisor",
        "github_repository": "example/chronovisor",
        "user_agent": "ExampleChronovisor/1.0",
        "launchd_label_prefix": "org.example.chronovisor-",
        "ollama_url": "http://127.0.0.1:22434",
    }


def test_non_loopback_runtime_endpoints_fall_back_to_safe_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[runtime]
ollama_url = "https://example.com"

[dashboard]
host = "0.0.0.0"
port = 70000
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("OLLAMA_URL", raising=False)

    assert runtime_config.ollama_url(config) == runtime_config.DEFAULT_OLLAMA_URL
    assert (
        runtime_config.load_dashboard_config(config) == runtime_config.DashboardConfig()
    )


def test_mcp_raw_content_exposure_is_explicit_opt_in(tmp_path: Path) -> None:
    missing = runtime_config.load_mcp_config(tmp_path / "missing.toml")
    assert missing.expose_raw_content is False

    config = tmp_path / "config.toml"
    config.write_text("[mcp]\nexpose_raw_content = true\n", encoding="utf-8")
    assert runtime_config.load_mcp_config(config).expose_raw_content is True

    config.write_text("[mcp]\nexpose_raw_content = \"true\"\n", encoding="utf-8")
    assert runtime_config.load_mcp_config(config).expose_raw_content is False


def test_dashboard_lan_config_is_separate_and_requires_absolute_secret_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[dashboard]
host = "localhost"
port = 9876

[dashboard_lan]
host = "192.168.50.20"
port = 9877
tls_cert_file = "{tmp_path / "dashboard.crt"}"
tls_key_file = "relative.key"
credentials_file = "{tmp_path / "credentials.json"}"
""",
        encoding="utf-8",
    )

    assert runtime_config.load_dashboard_config(
        config
    ) == runtime_config.DashboardConfig(host="localhost", port=9876)
    assert runtime_config.load_dashboard_lan_config(
        config
    ) == runtime_config.DashboardLanConfig(
        host="192.168.50.20",
        port=9877,
        tls_cert_file=tmp_path / "dashboard.crt",
        tls_key_file=None,
        credentials_file=tmp_path / "credentials.json",
    )
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)
    assert runtime_config.runtime_identity(config_only=True)["dashboard_lan"] == {
        "host": "192.168.50.20",
        "port": 9877,
        "tls_cert_file": str(tmp_path / "dashboard.crt"),
        "tls_key_file": None,
        "credentials_file": str(tmp_path / "credentials.json"),
    }


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


def test_search_embedding_config_reads_runtime_knobs(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[search.embedding]
enabled = true
revision = "abc123"
dimensions = 2048
storage_dtype = "float32"
query_prefix = "query: "
document_prefix = "passage: "

[search.embedding.service]
socket = "~/.chronovisor/runtime/test-semantic.sock"
query_device = "mps"
query_replicas = 7
foreground_batch_window_ms = 2
foreground_max_batch = 8
incremental_device = "cpu"
incremental_enabled = true
incremental_max_batch = 9
incremental_pause_during_research = true
incremental_pause_during_ingest_generation = true
incremental_idle_unload_seconds = 600
maintenance_max_batch = 32
offline = true
query_timeout_ms = 300

[search.embedding.rollout]
mode = "canary"
canary_percent = 25
sync_recall = false
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)

    search = runtime_config.load_search_embedding_config()

    assert search.revision == "abc123"
    assert search.dimensions == 2048
    assert search.query_prefix == "query: "
    assert search.document_prefix == "passage: "
    assert search.query_replicas == 1
    assert search.foreground_max_batch == 8
    assert search.incremental_enabled is True
    assert search.incremental_max_batch == 1
    assert search.maintenance_max_batch == 32
    assert search.rollout_mode == "canary"
    assert search.canary_percent == 25
    assert search.sync_recall is False
    assert search.query_timeout_ms == 300


def test_search_embedding_is_disabled_when_config_is_absent(
    tmp_path: Path, monkeypatch
) -> None:
    missing = tmp_path / "missing.toml"
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", missing)

    assert runtime_config.load_search_embedding_config().enabled is False


def test_ingest_config_reads_generation_knobs(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[ingest]
keep_alive = "10m"
temperature = 0.1
num_ctx = 32768
max_num_ctx = 131072
num_predict = 4096
read_timeout_ms = 120000
memory_reserve_gib = 24
max_related_context_bytes = 12288
semantic_projection_max_child_bytes = 16384
processed_projection_reconciler_enabled = false
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)

    cfg = runtime_config.load_ingest_config()

    assert not hasattr(cfg, "model")
    assert cfg.keep_alive == "10m"
    assert cfg.temperature == 0.1
    assert cfg.num_ctx == 32768
    assert cfg.max_num_ctx == 131072
    assert cfg.num_predict == 4096
    assert cfg.read_timeout_ms == 120000
    assert cfg.memory_reserve_gib == 24
    assert cfg.max_related_context_bytes == 12288
    assert cfg.semantic_projection_max_child_bytes == 16384
    assert cfg.processed_projection_reconciler_enabled is False


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


def test_ingest_projection_reconciler_defaults_enabled(tmp_path: Path) -> None:
    assert runtime_config.load_ingest_config(tmp_path / "missing.toml").processed_projection_reconciler_enabled

    config = tmp_path / "config.toml"
    config.write_text(
        "[ingest]\nprocessed_projection_reconciler_enabled = true\n",
        encoding="utf-8",
    )
    assert runtime_config.load_ingest_config(config).processed_projection_reconciler_enabled


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

    assert cfg.primary_model == "qwen3.8:27b-axq4"
    assert cfg.challenger_model == "muse-glimmer:30b-q4k-dynamic"
    assert cfg.tie_break_model == "gemma4:26b-optiq4"
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


def test_decision_router_config_ignores_models_but_candidate_reads_them(
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
adoption_artifact = ""
""",
        encoding="utf-8",
    )

    cfg = runtime_config.load_decision_router_config(config)

    assert cfg.primary_model == runtime_config.DecisionRouterConfig.primary_model
    assert cfg.challenger_model == runtime_config.DecisionRouterConfig.challenger_model
    assert cfg.tie_break_model == runtime_config.DecisionRouterConfig.tie_break_model
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
    assert cfg.adoption_artifact == ""

    candidate = runtime_config.load_candidate_decision_router_config(config)
    assert candidate.primary_model == "ornith-local:35b-q5"
    assert candidate.challenger_model == "gpt-oss:20b"
    assert candidate.tie_break_model == "gemma4:26b-mxfp8"


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

    assert cfg.tie_break_model == runtime_config.DecisionRouterConfig.tie_break_model
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
top_n = 20
weight = 0.4

[search.reranker.service]
enabled = true
socket = "/tmp/chronovisor-reranker.sock"
timeout_ms = 1700
mode = "canary"
canary_percent = 25
queue_size = 12
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)

    cfg = runtime_config.load_reranker_config()

    assert cfg.enabled is True
    assert cfg.top_n == 20
    assert cfg.model == runtime_config.RerankerConfig.model
    assert cfg.backend == runtime_config.RerankerConfig.backend
    assert cfg.max_length == runtime_config.RerankerConfig.max_length
    assert cfg.batch_size == runtime_config.RerankerConfig.batch_size
    assert cfg.device == runtime_config.RerankerConfig.device
    assert cfg.weight == 0.4
    assert cfg.service.enabled is True
    assert cfg.service.socket == "/tmp/chronovisor-reranker.sock"
    assert cfg.service.timeout_ms == 1700
    assert cfg.service.mode == "canary"
    assert cfg.service.canary_percent == 25
    assert cfg.service.queue_size == 12
