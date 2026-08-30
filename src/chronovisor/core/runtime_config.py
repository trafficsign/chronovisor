"""Shared Chronovisor runtime configuration helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Any

from chronovisor.core.store import CHRONOVISOR_ROOT

CONFIG_FILE = CHRONOVISOR_ROOT / "config.toml"

FALSE_VALUES = {"0", "false", "False", "no", "NO", "off", "OFF"}
TRUE_VALUES = {"1", "true", "True", "yes", "YES", "on", "ON"}
DEFAULT_DECISION_PRIMARY_MODEL = "qwen3.8:27b-axq4"
DEFAULT_DECISION_CHALLENGER_MODEL = "muse-glimmer:30b-q4k-dynamic"
DEFAULT_DECISION_TIE_BREAK_MODEL = "gemma4:26b-optiq4"
# The production decision authority is intentionally a single configured
# runtime route.  The legacy triplet remains in this dataclass so old
# evaluation/replay records can still be decoded without making them current.
SINGLE_MODEL_AUTHORITY_KIND = "single_model_v1"
QUORUM_AUTHORITY_KIND = "quorum_v1"
DEFAULT_DECISION_SINGLE_RUNTIME_ROLE = "classification.authority"
# Immutable production route identity for the Jundot Qwen cutover.  The
# model name remains a runtime-role concern; these constants are shared by the
# route/proof validators and structured-session compatibility table.
SINGLE_MODEL_RUNTIME_MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"
SINGLE_MODEL_RUNTIME_REVISION = "2615fc0e976e65c2f3b55daca3a948f1cdc5b9f8"
SINGLE_MODEL_RUNTIME_MODEL_TYPE = "qwen4_exp"
SINGLE_MODEL_RUNTIME_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
DEFAULT_HEAVY_NUM_CTX = 32_768
DEFAULT_HEAVY_KEEP_ALIVE = "20m"
MAX_SEMANTIC_PROJECTION_CHILD_BYTES = 24_000
DEFAULT_GITHUB_REPOSITORY = "trafficsign/chronovisor"
DEFAULT_RUNTIME_SOURCE = f"git+ssh://git@github.com/{DEFAULT_GITHUB_REPOSITORY}"
DEFAULT_USER_AGENT = (
    f"Chronovisor/0.1 (+https://github.com/{DEFAULT_GITHUB_REPOSITORY})"
)
DEFAULT_LAUNCHD_LABEL_PREFIX = "com.trafficsign.chronovisor-"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 8765
DEFAULT_DASHBOARD_LAN_PORT = 8766
RUNTIME_PACKAGE = "chronovisor"
_GITHUB_REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_LAUNCHD_PREFIX_RE = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9-]*\.)+[A-Za-z0-9][A-Za-z0-9-]*-\Z"
)
_LAUNCHD_SERVICE_RE = re.compile(r"[a-z0-9][a-z0-9-]*\Z")


@dataclass(frozen=True)
class RuntimeSettings:
    source: str = DEFAULT_RUNTIME_SOURCE
    github_repository: str = DEFAULT_GITHUB_REPOSITORY
    user_agent: str = DEFAULT_USER_AGENT
    launchd_label_prefix: str = DEFAULT_LAUNCHD_LABEL_PREFIX
    ollama_url: str = DEFAULT_OLLAMA_URL


@dataclass(frozen=True)
class DashboardConfig:
    host: str = DEFAULT_DASHBOARD_HOST
    port: int = DEFAULT_DASHBOARD_PORT


@dataclass(frozen=True)
class DashboardLanConfig:
    host: str | None = None
    port: int = DEFAULT_DASHBOARD_LAN_PORT
    tls_cert_file: Path | None = None
    tls_key_file: Path | None = None
    credentials_file: Path | None = None


@dataclass(frozen=True)
class McpConfig:
    """MCP response exposure policy."""

    expose_raw_content: bool = False


def _clean_text(value: object, default: str) -> str:
    if (
        isinstance(value, str)
        and value
        and value == value.strip()
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return value
    return default


def _loopback_ollama_url(value: object) -> str:
    from chronovisor.core.llm_security import (
        CredentialSecurityError,
        canonical_endpoint,
    )

    candidate = _clean_text(value, DEFAULT_OLLAMA_URL)
    try:
        endpoint = canonical_endpoint(candidate, cloud_secret=False)
    except CredentialSecurityError:
        return DEFAULT_OLLAMA_URL
    if not endpoint.is_loopback or endpoint.url != endpoint.origin:
        return DEFAULT_OLLAMA_URL
    return endpoint.origin


def _runtime_settings_from_data(data: dict[str, Any]) -> RuntimeSettings:
    section = data.get("runtime")
    section = section if isinstance(section, dict) else {}
    repository = _clean_text(
        section.get("github_repository"), DEFAULT_GITHUB_REPOSITORY
    )
    if _GITHUB_REPOSITORY_RE.fullmatch(repository) is None:
        repository = DEFAULT_GITHUB_REPOSITORY
    default_source = f"git+ssh://git@github.com/{repository}"
    source = _clean_text(section.get("source"), default_source)
    default_user_agent = f"Chronovisor/0.1 (+https://github.com/{repository})"
    user_agent = _clean_text(section.get("user_agent"), default_user_agent)
    label_prefix = _clean_text(
        section.get("launchd_label_prefix"), DEFAULT_LAUNCHD_LABEL_PREFIX
    )
    if _LAUNCHD_PREFIX_RE.fullmatch(label_prefix) is None:
        label_prefix = DEFAULT_LAUNCHD_LABEL_PREFIX
    return RuntimeSettings(
        source=source,
        github_repository=repository,
        user_agent=user_agent,
        launchd_label_prefix=label_prefix,
        ollama_url=_loopback_ollama_url(section.get("ollama_url")),
    )


def load_runtime_settings(path: Path | str | None = None) -> RuntimeSettings:
    """Load portable process identity and local endpoint defaults."""

    return _runtime_settings_from_data(load_toml_file(path))


def github_repository(path: Path | str | None = None) -> str:
    configured = os.environ.get("CHRONOVISOR_GITHUB_REPOSITORY", "").strip()
    if _GITHUB_REPOSITORY_RE.fullmatch(configured):
        return configured
    return load_runtime_settings(path).github_repository


def user_agent(path: Path | str | None = None) -> str:
    configured = os.environ.get("CHRONOVISOR_USER_AGENT", "")
    return _clean_text(configured, load_runtime_settings(path).user_agent)


def launchd_label(service: str, path: Path | str | None = None) -> str:
    if _LAUNCHD_SERVICE_RE.fullmatch(service) is None:
        raise ValueError("invalid launchd service name")
    configured = os.environ.get("CHRONOVISOR_LAUNCHD_LABEL_PREFIX", "").strip()
    prefix = (
        configured
        if _LAUNCHD_PREFIX_RE.fullmatch(configured)
        else load_runtime_settings(path).launchd_label_prefix
    )
    return f"{prefix}{service}"


def ollama_url(path: Path | str | None = None) -> str:
    configured = os.environ.get("OLLAMA_URL", "").strip()
    return _loopback_ollama_url(configured or load_runtime_settings(path).ollama_url)


def load_dashboard_config(path: Path | str | None = None) -> DashboardConfig:
    data = load_toml_file(path)
    section = data.get("dashboard")
    section = section if isinstance(section, dict) else {}
    host = _clean_text(section.get("host"), DEFAULT_DASHBOARD_HOST).casefold()
    if host not in {"localhost", DEFAULT_DASHBOARD_HOST}:
        host = DEFAULT_DASHBOARD_HOST
    port = section.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        port = DEFAULT_DASHBOARD_PORT
    return DashboardConfig(host=host, port=port)


def load_mcp_config(path: Path | str | None = None) -> McpConfig:
    """Load the opt-in MCP raw-content exposure policy."""

    data = load_toml_file(path)
    section = data.get("mcp")
    if not isinstance(section, dict):
        return McpConfig()
    return McpConfig(expose_raw_content=section.get("expose_raw_content") is True)


def dashboard_url(path: Path | str | None = None) -> str:
    config = load_dashboard_config(path)
    return f"http://{config.host}:{config.port}"


def load_dashboard_lan_config(path: Path | str | None = None) -> DashboardLanConfig:
    data = load_toml_file(path)
    section = data.get("dashboard_lan")
    section = section if isinstance(section, dict) else {}
    host = _clean_text(section.get("host"), "") or None
    port = section.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        port = DEFAULT_DASHBOARD_LAN_PORT

    def absolute_path(key: str) -> Path | None:
        value = section.get(key)
        candidate = (
            Path(value).expanduser() if isinstance(value, str) and value else None
        )
        return candidate if candidate is not None and candidate.is_absolute() else None

    return DashboardLanConfig(
        host=host,
        port=port,
        tls_cert_file=absolute_path("tls_cert_file"),
        tls_key_file=absolute_path("tls_key_file"),
        credentials_file=absolute_path("credentials_file"),
    )


def runtime_source() -> str:
    """Return the pushed package source used by production entry points."""

    configured = os.environ.get("CHRONOVISOR_RUNTIME_SOURCE", "").strip()
    return configured or load_runtime_settings().source


def runtime_repo_root() -> Path:
    """Return the explicit checkout used only for reviewed code repair/context."""

    configured = os.environ.get("CHRONOVISOR_REPO_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    checkout = Path.home() / "projects" / "personal" / RUNTIME_PACKAGE
    if (checkout / ".git").exists():
        return checkout
    return Path(__file__).resolve().parents[3]


def resolve_runtime_python(executable: str | None = None) -> str:
    """Return an absolute standard-GIL Python 3.14 executable."""

    requested = (
        executable
        or os.environ.get("CHRONOVISOR_PYTHON", "").strip()
        or "python3.14"
    )
    found = shutil.which(requested)
    if found is None:
        raise RuntimeError("standard Python 3.14 executable not found")
    try:
        resolved = Path(found).resolve(strict=True)
    except OSError:
        raise RuntimeError("standard Python 3.14 executable not found") from None
    if not resolved.is_absolute() or "archive-v0" in resolved.parts:
        raise RuntimeError("standard Python 3.14 executable not found")
    probe = subprocess.run(
        [
            str(resolved),
            "-c",
            "import sys; raise SystemExit(not (sys.version_info[:2] == (3, 14) "
            "and getattr(sys, '_is_gil_enabled', lambda: False)()))",
        ],
        text=True,
        capture_output=True,
    )
    if probe.returncode != 0:
        raise RuntimeError("standard GIL Python 3.14 executable required")
    return str(resolved)


def uvx_runtime_command(
    entrypoint: str,
    *,
    executable: str = "uvx",
    python: str | None = None,
    refresh: bool = False,
) -> list[str]:
    """Build a production command that cannot import an unpushed worktree."""

    command = [executable, "--python", resolve_runtime_python(python)]
    if refresh:
        command.extend(["--refresh-package", RUNTIME_PACKAGE])
    command.extend(["--from", runtime_source(), entrypoint])
    return command


def runtime_identity(*, config_only: bool = False) -> dict[str, Any]:
    """Expose the installed Git revision and compare it with pushed main."""
    dashboard = load_dashboard_config()
    portability = {
        "github_repository": github_repository(),
        "user_agent": user_agent(),
        "launchd_label_prefix": launchd_label("dashboard").removesuffix("dashboard"),
        "ollama_url": ollama_url(),
        "dashboard": {
            "host": dashboard.host,
            "port": dashboard.port,
            "url": f"http://{dashboard.host}:{dashboard.port}",
        },
    }
    if config_only:
        dashboard_lan = load_dashboard_lan_config()
        return {
            **portability,
            "dashboard_lan": {
                "host": dashboard_lan.host,
                "port": dashboard_lan.port,
                "tls_cert_file": (
                    str(dashboard_lan.tls_cert_file)
                    if dashboard_lan.tls_cert_file is not None
                    else None
                ),
                "tls_key_file": (
                    str(dashboard_lan.tls_key_file)
                    if dashboard_lan.tls_key_file is not None
                    else None
                ),
                "credentials_file": (
                    str(dashboard_lan.credentials_file)
                    if dashboard_lan.credentials_file is not None
                    else None
                ),
            },
        }
    commit_id = None
    direct_url: dict[str, Any] = {}
    package_version = None
    try:
        dist = metadata.distribution(RUNTIME_PACKAGE)
        package_version = dist.version
        raw = dist.read_text("direct_url.json")
        if raw:
            direct_url = json.loads(raw)
            vcs = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
            if isinstance(vcs, dict):
                commit_id = vcs.get("commit_id")
    except Exception:
        pass
    expected_commit = None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=runtime_repo_root(),
            text=True,
            capture_output=True,
            timeout=5,
        )
        if completed.returncode == 0:
            expected_commit = completed.stdout.strip()
    except Exception:
        pass
    module_path = Path(__file__).resolve()
    return {
        **portability,
        "runtime_source": runtime_source(),
        "commit_id": commit_id,
        "expected_commit": expected_commit,
        "drift": bool(commit_id and expected_commit and commit_id != expected_commit),
        "archive_path": str(module_path.parents[3]),
        "module_path": str(module_path),
        "package_version": package_version,
        "direct_url": direct_url,
    }


@dataclass(frozen=True)
class HookPolicy:
    user_prompt_recall: bool = True
    stop_save: bool = True
    stop_audit: bool = True
    stop_content_correction: bool = True
    stop_recall_improve: bool = True


@dataclass(frozen=True)
class SearchEmbeddingConfig:
    """Search-only semantic retrieval service profile."""

    enabled: bool = False
    revision: str = "a5e0f804b9e90a1ca6784ecbf6e41595774fc834"
    dimensions: int = 2_048
    storage_dtype: str = "float32"
    query_prefix: str = "query: "
    document_prefix: str = "passage: "
    fusion_weight: float = 0.6
    min_top_score: float = 0.20
    min_margin: float = 0.001
    low_confidence_weight: float = 0.25
    socket: str = "~/.chronovisor/runtime/semantic.sock"
    query_device: str = "mps"
    query_replicas: int = 1
    foreground_batch_window_ms: int = 2
    foreground_max_batch: int = 4
    incremental_device: str = "cpu"
    incremental_enabled: bool = False
    incremental_max_batch: int = 1
    incremental_pause_during_research: bool = True
    incremental_pause_during_ingest_generation: bool = True
    incremental_idle_unload_seconds: int = 300
    maintenance_max_batch: int = 32
    offline: bool = True
    rollout_mode: str = "off"
    canary_percent: int = 0
    sync_recall: bool = False
    query_timeout_ms: int = 250
    interactive_timeout_ms: int = 5_000


@dataclass(frozen=True)
class RerankerServiceConfig:
    enabled: bool = False
    socket: str = "~/.chronovisor/runtime/reranker.sock"
    timeout_ms: int = 1_500
    mode: str = "off"
    canary_percent: int = 0
    queue_size: int = 8


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    backend: str = "transformers"
    top_n: int = 10
    max_length: int = 384
    batch_size: int = 10
    device: str = ""
    dtype: str = "float32"
    weight: float = 1.0
    service: RerankerServiceConfig = field(default_factory=RerankerServiceConfig)


@dataclass(frozen=True)
class IngestConfig:
    keep_alive: str = DEFAULT_HEAVY_KEEP_ALIVE
    temperature: float = 0.3
    # Ingest selects the smallest safe 32K/64K/128K/256K bucket from the
    # complete request envelope. A larger compatible resident runner is
    # reused so a backlog grows monotonically instead of reloading per raw.
    num_ctx: int = DEFAULT_HEAVY_NUM_CTX
    max_num_ctx: int = 262_144
    num_predict: int = 8_192
    read_timeout_ms: int = 660_000
    memory_reserve_gib: int = 16
    max_related_context_bytes: int = 8_192
    semantic_projection_max_child_bytes: int = MAX_SEMANTIC_PROJECTION_CHILD_BYTES
    processed_projection_reconciler_enabled: bool = True


@dataclass(frozen=True)
class IngestAuditConfig:
    enabled: bool = True
    sample_rate: float = 0.05
    update_sample_rate: float = 0.08
    noop_sample_rate: float = 0.05
    adaptive: bool = True
    adaptive_window: int = 50
    adaptive_min_audits: int = 5
    elevated_reject_rate: float = 0.10
    critical_reject_rate: float = 0.20
    elevated_sample_rate: float = 0.08
    critical_sample_rate: float = 0.10
    max_sample_rate: float = 0.10
    max_operations_without_audit: int = 4


@dataclass(frozen=True)
class DecisionRouterConfig:
    """Decision runtime limits and the versioned authority contract.

    ``primary_model``/``challenger_model``/``tie_break_model`` are retained as
    compatibility fields for candidate evaluation and audit-only legacy
    quorum artifacts.  Production model selection is resolved from the
    configured runtime role named by :attr:`single_runtime_role`.
    """

    authority_kind: str = SINGLE_MODEL_AUTHORITY_KIND
    single_runtime_role: str = DEFAULT_DECISION_SINGLE_RUNTIME_ROLE
    authority_keep_alive: str = DEFAULT_HEAVY_KEEP_ALIVE

    primary_model: str = DEFAULT_DECISION_PRIMARY_MODEL
    challenger_model: str = DEFAULT_DECISION_CHALLENGER_MODEL
    tie_break_model: str = DEFAULT_DECISION_TIE_BREAK_MODEL
    primary_keep_alive: str = DEFAULT_HEAVY_KEEP_ALIVE
    challenger_keep_alive: str = DEFAULT_HEAVY_KEEP_ALIVE
    tie_break_keep_alive: str = "2m"
    # All three adopted decision models support at least 131K.  This is the
    # ceiling; each request is rounded to the smallest safe bucket so short
    # jobs do not pay the KV-cache cost of the longest corpus case.
    num_ctx: int = 114_688
    min_num_ctx: int = 16_384
    num_predict: int = 3_072
    read_timeout_ms: int = 660_000
    max_input_chars: int = 93_000
    max_output_chars: int = 4_000
    max_feedback_chars: int = 2_000
    quorum: int = 2
    adaptive_residency: bool = True
    residency_policy_version: int = 2
    memory_reserve_gib: int = 16
    max_resident_models: int = 3
    # Evaluation callers may nominate an artifact; production model selection
    # is resolved only from llm.roles.classification.* by DecisionRouter.
    adoption_artifact: str = ""

    @property
    def single_route_identity(self) -> str:
        """Return the canonical runtime role used by single-model authority."""

        return self.single_runtime_role

    @property
    def is_single_model(self) -> bool:
        return self.authority_kind == SINGLE_MODEL_AUTHORITY_KIND


def active_config_file(path: Path | str | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    return CONFIG_FILE


def load_toml_file(
    path: Path | str | None = None,
    *,
    runtime_defaults: bool = False,
) -> dict[str, Any]:
    resolved = active_config_file(path)
    try:
        # Read one immutable byte snapshot.  Checking existence separately or
        # reopening the path while parsing can mix generations when an
        # external operator atomically replaces the runtime configuration.
        snapshot = resolved.read_bytes()
        data = tomllib.loads(snapshot.decode("utf-8"))
        if not isinstance(data, dict):
            return {}
        if not runtime_defaults:
            return data
        normalized = dict(data)
        settings = _runtime_settings_from_data(data)
        normalized["runtime"] = {
            "source": settings.source,
            "github_repository": settings.github_repository,
            "user_agent": _clean_text(
                os.environ.get("CHRONOVISOR_USER_AGENT", ""),
                settings.user_agent,
            ),
            "launchd_label_prefix": settings.launchd_label_prefix,
            "ollama_url": settings.ollama_url,
        }
        return normalized
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}


def env_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    if value in FALSE_VALUES:
        return False
    if value in TRUE_VALUES:
        return True
    return bool(value)


def nested_bool(data: dict[str, Any], path: tuple[str, ...], default: bool) -> bool:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if isinstance(cur, bool) else default


def load_hook_policy(path: Path | str | None = None) -> HookPolicy:
    data = load_toml_file(path)
    return HookPolicy(
        user_prompt_recall=nested_bool(data, ("hooks", "user_prompt", "recall"), True),
        stop_save=nested_bool(data, ("hooks", "stop", "save"), True),
        stop_audit=nested_bool(data, ("hooks", "stop", "audit"), True),
        stop_content_correction=nested_bool(
            data, ("hooks", "stop", "content_correction"), True
        ),
        stop_recall_improve=nested_bool(
            data, ("hooks", "stop", "recall_improve"), True
        ),
    )


def load_search_embedding_config(
    path: Path | str | None = None,
) -> SearchEmbeddingConfig:
    data = load_toml_file(path)
    search = data.get("search")
    section = search.get("embedding") if isinstance(search, dict) else None
    if not isinstance(section, dict):
        return SearchEmbeddingConfig()
    service = section.get("service")
    if not isinstance(service, dict):
        service = {}
    rollout = section.get("rollout")
    if not isinstance(rollout, dict):
        rollout = {}

    def text(
        source: dict[str, Any],
        name: str,
        default: str,
        *,
        choices: set[str] | None = None,
    ) -> str:
        value = source.get(name)
        if not isinstance(value, str) or not value.strip():
            return default
        normalized = value.strip()
        if choices is not None and normalized not in choices:
            return default
        return normalized

    return SearchEmbeddingConfig(
        enabled=(
            section["enabled"]
            if isinstance(section.get("enabled"), bool)
            else SearchEmbeddingConfig.enabled
        ),
        revision=text(section, "revision", SearchEmbeddingConfig.revision),
        dimensions=_bounded_int(
            section.get("dimensions"),
            SearchEmbeddingConfig.dimensions,
            minimum=128,
            maximum=4_096,
        ),
        storage_dtype=text(
            section,
            "storage_dtype",
            SearchEmbeddingConfig.storage_dtype,
            choices={"float32"},
        ),
        query_prefix=(
            section["query_prefix"]
            if isinstance(section.get("query_prefix"), str)
            else SearchEmbeddingConfig.query_prefix
        ),
        document_prefix=(
            section["document_prefix"]
            if isinstance(section.get("document_prefix"), str)
            else SearchEmbeddingConfig.document_prefix
        ),
        fusion_weight=_bounded_float(
            section.get("fusion_weight"),
            SearchEmbeddingConfig.fusion_weight,
            minimum=0.0,
            maximum=2.0,
        ),
        min_top_score=_bounded_float(
            section.get("min_top_score"),
            SearchEmbeddingConfig.min_top_score,
            minimum=0.0,
            maximum=1.0,
        ),
        min_margin=_bounded_float(
            section.get("min_margin"),
            SearchEmbeddingConfig.min_margin,
            minimum=0.0,
            maximum=1.0,
        ),
        low_confidence_weight=_bounded_float(
            section.get("low_confidence_weight"),
            SearchEmbeddingConfig.low_confidence_weight,
            minimum=0.0,
            maximum=1.0,
        ),
        socket=text(service, "socket", SearchEmbeddingConfig.socket),
        query_device=text(
            service,
            "query_device",
            SearchEmbeddingConfig.query_device,
            choices={"mps", "cpu"},
        ),
        query_replicas=_bounded_int(
            service.get("query_replicas"),
            SearchEmbeddingConfig.query_replicas,
            minimum=1,
            maximum=1,
        ),
        foreground_batch_window_ms=_bounded_int(
            service.get("foreground_batch_window_ms"),
            SearchEmbeddingConfig.foreground_batch_window_ms,
            minimum=0,
            maximum=5,
        ),
        foreground_max_batch=_bounded_int(
            service.get("foreground_max_batch"),
            SearchEmbeddingConfig.foreground_max_batch,
            minimum=1,
            maximum=8,
        ),
        incremental_device=text(
            service,
            "incremental_device",
            SearchEmbeddingConfig.incremental_device,
            choices={"cpu"},
        ),
        incremental_enabled=(
            service["incremental_enabled"]
            if isinstance(service.get("incremental_enabled"), bool)
            else SearchEmbeddingConfig.incremental_enabled
        ),
        incremental_max_batch=_bounded_int(
            service.get("incremental_max_batch"),
            SearchEmbeddingConfig.incremental_max_batch,
            minimum=1,
            maximum=1,
        ),
        incremental_pause_during_research=(
            service["incremental_pause_during_research"]
            if isinstance(service.get("incremental_pause_during_research"), bool)
            else SearchEmbeddingConfig.incremental_pause_during_research
        ),
        incremental_pause_during_ingest_generation=(
            service["incremental_pause_during_ingest_generation"]
            if isinstance(
                service.get("incremental_pause_during_ingest_generation"), bool
            )
            else SearchEmbeddingConfig.incremental_pause_during_ingest_generation
        ),
        incremental_idle_unload_seconds=_bounded_int(
            service.get("incremental_idle_unload_seconds"),
            SearchEmbeddingConfig.incremental_idle_unload_seconds,
            minimum=30,
            maximum=3_600,
        ),
        maintenance_max_batch=_bounded_int(
            service.get("maintenance_max_batch"),
            SearchEmbeddingConfig.maintenance_max_batch,
            minimum=1,
            maximum=64,
        ),
        offline=(
            service["offline"]
            if isinstance(service.get("offline"), bool)
            else SearchEmbeddingConfig.offline
        ),
        rollout_mode=text(
            rollout,
            "mode",
            SearchEmbeddingConfig.rollout_mode,
            choices={"off", "shadow", "canary", "on"},
        ),
        canary_percent=_bounded_int(
            rollout.get("canary_percent"),
            SearchEmbeddingConfig.canary_percent,
            minimum=0,
            maximum=100,
        ),
        sync_recall=(
            rollout["sync_recall"]
            if isinstance(rollout.get("sync_recall"), bool)
            else SearchEmbeddingConfig.sync_recall
        ),
        query_timeout_ms=_bounded_int(
            service.get("query_timeout_ms"),
            SearchEmbeddingConfig.query_timeout_ms,
            minimum=25,
            maximum=1_000,
        ),
        interactive_timeout_ms=_bounded_int(
            service.get("interactive_timeout_ms"),
            SearchEmbeddingConfig.interactive_timeout_ms,
            minimum=250,
            maximum=5_000,
        ),
    )


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value if value >= minimum else default
    return default


def _nonnegative_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else default
    return default


def _bounded_float(
    value: Any, default: float, *, minimum: float, maximum: float
) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        number = float(value)
        return number if minimum <= number <= maximum else default
    return default


def _bounded_int(
    value: Any,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and minimum <= value <= maximum:
        return value
    return default


def load_ingest_config(path: Path | str | None = None) -> IngestConfig:
    data = load_toml_file(path)
    section = data.get("ingest")
    if not isinstance(section, dict):
        return IngestConfig()

    keep_alive = section.get("keep_alive")
    max_num_ctx = _positive_int(
        section.get("max_num_ctx"), IngestConfig.max_num_ctx, minimum=2_048
    )
    num_ctx = _positive_int(section.get("num_ctx"), IngestConfig.num_ctx, minimum=2_048)
    if num_ctx > max_num_ctx:
        num_ctx = max_num_ctx

    return IngestConfig(
        keep_alive=keep_alive
        if isinstance(keep_alive, str) and keep_alive.strip()
        else IngestConfig.keep_alive,
        temperature=_bounded_float(
            section.get("temperature"),
            IngestConfig.temperature,
            minimum=0.0,
            maximum=2.0,
        ),
        num_ctx=num_ctx,
        max_num_ctx=max_num_ctx,
        num_predict=_positive_int(
            section.get("num_predict"), IngestConfig.num_predict, minimum=128
        ),
        read_timeout_ms=_positive_int(
            section.get("read_timeout_ms"), IngestConfig.read_timeout_ms, minimum=1_000
        ),
        memory_reserve_gib=_positive_int(
            section.get("memory_reserve_gib"),
            IngestConfig.memory_reserve_gib,
            minimum=1,
        ),
        max_related_context_bytes=_positive_int(
            section.get("max_related_context_bytes"),
            IngestConfig.max_related_context_bytes,
            minimum=1_024,
        ),
        semantic_projection_max_child_bytes=_bounded_int(
            section.get("semantic_projection_max_child_bytes"),
            IngestConfig.semantic_projection_max_child_bytes,
            minimum=2_048,
            maximum=MAX_SEMANTIC_PROJECTION_CHILD_BYTES,
        ),
        processed_projection_reconciler_enabled=(
            section.get("processed_projection_reconciler_enabled") is not False
        ),
    )


def load_ingest_audit_config(path: Path | str | None = None) -> IngestAuditConfig:
    data = load_toml_file(path)
    ingest = data.get("ingest")
    section = ingest.get("audit") if isinstance(ingest, dict) else None
    if not isinstance(section, dict):
        return IngestAuditConfig()

    def rate(name: str, default: float) -> float:
        return _bounded_float(section.get(name), default, minimum=0.0, maximum=1.0)

    return IngestAuditConfig(
        enabled=section.get("enabled") is not False,
        sample_rate=rate("sample_rate", IngestAuditConfig.sample_rate),
        update_sample_rate=rate(
            "update_sample_rate", IngestAuditConfig.update_sample_rate
        ),
        noop_sample_rate=rate("noop_sample_rate", IngestAuditConfig.noop_sample_rate),
        adaptive=section.get("adaptive") is not False,
        adaptive_window=_positive_int(
            section.get("adaptive_window"),
            IngestAuditConfig.adaptive_window,
            minimum=5,
        ),
        adaptive_min_audits=_positive_int(
            section.get("adaptive_min_audits"),
            IngestAuditConfig.adaptive_min_audits,
            minimum=1,
        ),
        elevated_reject_rate=rate(
            "elevated_reject_rate", IngestAuditConfig.elevated_reject_rate
        ),
        critical_reject_rate=rate(
            "critical_reject_rate", IngestAuditConfig.critical_reject_rate
        ),
        elevated_sample_rate=rate(
            "elevated_sample_rate", IngestAuditConfig.elevated_sample_rate
        ),
        critical_sample_rate=rate(
            "critical_sample_rate", IngestAuditConfig.critical_sample_rate
        ),
        max_sample_rate=rate("max_sample_rate", IngestAuditConfig.max_sample_rate),
        max_operations_without_audit=_positive_int(
            section.get("max_operations_without_audit"),
            IngestAuditConfig.max_operations_without_audit,
            minimum=1,
        ),
    )


def load_decision_router_config(
    path: Path | str | None = None,
) -> DecisionRouterConfig:
    """Load production decision limits without accepting model selectors.

    The authority kind is a versioned contract, not a traffic-splitting
    switch.  Unknown values fail closed to the production single-model kind;
    candidate/replay loaders remain responsible for explicitly selecting the
    legacy quorum contract.
    """

    data = load_toml_file(path)
    section = data.get("decision_router")
    if not isinstance(section, dict):
        return DecisionRouterConfig()

    # Production has one immutable authority contract.  A TOML value cannot
    # re-enable the legacy quorum; candidate/replay loading below is the only
    # path that can construct ``quorum_v1`` explicitly.
    authority_kind = SINGLE_MODEL_AUTHORITY_KIND
    single_runtime_role = DecisionRouterConfig.single_runtime_role

    def keep_alive(name: str, default: str, *, alias: str | None = None) -> str:
        value = section.get(name)
        if value is None and alias is not None:
            value = section.get(alias)
        return value.strip() if isinstance(value, str) and value.strip() else default

    return DecisionRouterConfig(
        authority_kind=authority_kind,
        single_runtime_role=single_runtime_role.strip(),
        authority_keep_alive=keep_alive(
            "authority_keep_alive", DecisionRouterConfig.authority_keep_alive
        ),
        primary_keep_alive=keep_alive(
            "primary_keep_alive", DecisionRouterConfig.primary_keep_alive
        ),
        challenger_keep_alive=keep_alive(
            "challenger_keep_alive", DecisionRouterConfig.challenger_keep_alive
        ),
        tie_break_keep_alive=keep_alive(
            "tie_break_keep_alive",
            DecisionRouterConfig.tie_break_keep_alive,
            alias="tie_keep_alive",
        ),
        num_ctx=_positive_int(
            section.get("num_ctx"), DecisionRouterConfig.num_ctx, minimum=2_048
        ),
        min_num_ctx=_positive_int(
            section.get("min_num_ctx"),
            DecisionRouterConfig.min_num_ctx,
            minimum=2_048,
        ),
        num_predict=_positive_int(
            section.get("num_predict"),
            DecisionRouterConfig.num_predict,
            minimum=128,
        ),
        read_timeout_ms=_positive_int(
            section.get("read_timeout_ms"),
            DecisionRouterConfig.read_timeout_ms,
            minimum=1_000,
        ),
        max_input_chars=_positive_int(
            section.get("max_input_chars"),
            DecisionRouterConfig.max_input_chars,
            minimum=4_096,
        ),
        max_output_chars=_positive_int(
            section.get("max_output_chars"),
            DecisionRouterConfig.max_output_chars,
            minimum=256,
        ),
        max_feedback_chars=_positive_int(
            section.get("max_feedback_chars"),
            DecisionRouterConfig.max_feedback_chars,
            minimum=512,
        ),
        quorum=_positive_int(
            section.get("quorum"), DecisionRouterConfig.quorum, minimum=2
        ),
        adaptive_residency=(
            section["adaptive_residency"]
            if isinstance(section.get("adaptive_residency"), bool)
            else DecisionRouterConfig.adaptive_residency
        ),
        residency_policy_version=_positive_int(
            section.get("residency_policy_version"),
            DecisionRouterConfig.residency_policy_version,
            minimum=1,
        ),
        memory_reserve_gib=_positive_int(
            section.get("memory_reserve_gib"),
            DecisionRouterConfig.memory_reserve_gib,
            minimum=4,
        ),
        max_resident_models=min(
            3,
            _positive_int(
                section.get("max_resident_models"),
                3 if authority_kind == QUORUM_AUTHORITY_KIND else 1,
                minimum=1,
            ),
        ),
        adoption_artifact=(
            str(section.get("adoption_artifact") or "").strip()
            if isinstance(section.get("adoption_artifact"), str)
            else ""
        ),
    )


_DECISION_ROUTER_STRING_FIELDS = frozenset(
    {
        "authority_kind",
        "single_runtime_role",
        "authority_keep_alive",
        "primary_model",
        "challenger_model",
        "tie_break_model",
        "tie_model",
        "primary_keep_alive",
        "challenger_keep_alive",
        "tie_break_keep_alive",
        "tie_keep_alive",
    }
)
_DECISION_ROUTER_INTEGER_BOUNDS = {
    "num_ctx": (2_048, None),
    "min_num_ctx": (2_048, None),
    "num_predict": (128, None),
    "read_timeout_ms": (1_000, None),
    "max_input_chars": (4_096, None),
    "max_output_chars": (256, None),
    "max_feedback_chars": (512, None),
    "quorum": (2, 2),
    "residency_policy_version": (1, None),
    "memory_reserve_gib": (4, None),
    "max_resident_models": (1, 3),
}
_DECISION_ROUTER_CANDIDATE_FIELDS = (
    _DECISION_ROUTER_STRING_FIELDS
    | frozenset(_DECISION_ROUTER_INTEGER_BOUNDS)
    | {"adaptive_residency", "adoption_artifact"}
)
_DECISION_ROUTER_REQUIRED_CANDIDATE_FIELDS = frozenset(
    {
        "primary_model",
        "challenger_model",
        "primary_keep_alive",
        "challenger_keep_alive",
        "num_ctx",
        "min_num_ctx",
        "num_predict",
        "read_timeout_ms",
        "max_input_chars",
        "max_output_chars",
        "max_feedback_chars",
        "quorum",
        "adaptive_residency",
        "residency_policy_version",
        "memory_reserve_gib",
        "max_resident_models",
        "adoption_artifact",
    }
)


def load_candidate_decision_router_config(path: Path | str) -> DecisionRouterConfig:
    """Load an explicit replay-gate candidate without permissive fallbacks.

    Production config loading intentionally ignores model selectors.  A replay
    gate is different: silently evaluating defaults after a typo or a missing
    file can spend substantial local compute on the wrong model set.
    """

    resolved = Path(path).expanduser()
    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"candidate config is unreadable: {resolved}: {exc}") from exc
    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"candidate config is malformed TOML: {resolved}: {exc}"
        ) from exc
    section = data.get("decision_router")
    if not isinstance(section, dict) or not section:
        raise ValueError(
            "candidate config requires a non-empty [decision_router] table"
        )
    unknown = sorted(set(section) - _DECISION_ROUTER_CANDIDATE_FIELDS)
    if unknown:
        raise ValueError(
            "candidate config contains unknown [decision_router] fields: "
            + ", ".join(unknown)
        )
    for canonical, alias in (
        ("tie_break_model", "tie_model"),
        ("tie_break_keep_alive", "tie_keep_alive"),
    ):
        if canonical in section and alias in section:
            raise ValueError(
                f"candidate config must not set both {canonical} and {alias}"
            )
    for name in _DECISION_ROUTER_STRING_FIELDS:
        if name in section and (
            not isinstance(section[name], str) or not section[name].strip()
        ):
            raise ValueError(
                f"candidate config field {name} must be a non-empty string"
            )
    if section.get("authority_kind", QUORUM_AUTHORITY_KIND) not in {
        SINGLE_MODEL_AUTHORITY_KIND,
        QUORUM_AUTHORITY_KIND,
    }:
        raise ValueError("candidate config field authority_kind is invalid")
    single_runtime_role = section.get(
        "single_runtime_role", DEFAULT_DECISION_SINGLE_RUNTIME_ROLE
    )
    if not re.fullmatch(
        r"[a-z][a-z0-9_.-]{0,79}", str(single_runtime_role).strip()
    ):
        raise ValueError("candidate config field single_runtime_role is invalid")
    for name, (minimum, maximum) in _DECISION_ROUTER_INTEGER_BOUNDS.items():
        if name not in section:
            continue
        value = section[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"candidate config field {name} must be an integer")
        if value < minimum or (maximum is not None and value > maximum):
            bound = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
            raise ValueError(f"candidate config field {name} must be {bound}")
    if "adaptive_residency" in section and not isinstance(
        section["adaptive_residency"], bool
    ):
        raise ValueError("candidate config field adaptive_residency must be a boolean")
    if not isinstance(section.get("adoption_artifact", ""), str):
        raise ValueError("candidate config field adoption_artifact must be a string")
    if str(section.get("adoption_artifact", "")).strip():
        raise ValueError("candidate config adoption_artifact must be empty")
    missing = sorted(_DECISION_ROUTER_REQUIRED_CANDIDATE_FIELDS - set(section))
    if "tie_break_model" not in section and "tie_model" not in section:
        missing.append("tie_break_model")
    if "tie_break_keep_alive" not in section and "tie_keep_alive" not in section:
        missing.append("tie_break_keep_alive")
    if missing:
        raise ValueError(
            "candidate config is missing required [decision_router] fields: "
            + ", ".join(sorted(missing))
        )

    config = replace(
        load_decision_router_config(resolved),
        authority_kind=str(
            section.get("authority_kind", QUORUM_AUTHORITY_KIND)
        ).strip(),
        single_runtime_role=str(
            section.get("single_runtime_role", DEFAULT_DECISION_SINGLE_RUNTIME_ROLE)
        ).strip(),
        primary_model=str(section["primary_model"]).strip(),
        challenger_model=str(section["challenger_model"]).strip(),
        tie_break_model=str(
            section.get("tie_break_model", section.get("tie_model"))
        ).strip(),
    )
    if config.min_num_ctx > config.num_ctx:
        raise ValueError("candidate config min_num_ctx must not exceed num_ctx")
    models = (config.primary_model, config.challenger_model, config.tie_break_model)
    if len(set(models)) != len(models):
        raise ValueError("candidate config requires three distinct model roles")
    return config


def load_reranker_config(path: Path | str | None = None) -> RerankerConfig:
    data = load_toml_file(path)
    search = data.get("search")
    reranker: Any = data.get("reranker")
    if isinstance(search, dict) and isinstance(search.get("reranker"), dict):
        reranker = search["reranker"]
    if not isinstance(reranker, dict):
        return RerankerConfig()

    raw_service = reranker.get("service")
    service_data: dict[str, Any] = raw_service if isinstance(raw_service, dict) else {}
    service_mode = str(service_data.get("mode") or "off").strip().lower()
    if service_mode not in {"off", "shadow", "canary", "on"}:
        service_mode = "off"
    service_socket = service_data.get("socket")
    return RerankerConfig(
        enabled=reranker.get("enabled") is True,
        top_n=_positive_int(reranker.get("top_n"), RerankerConfig.top_n),
        weight=_nonnegative_float(reranker.get("weight"), RerankerConfig.weight),
        service=RerankerServiceConfig(
            enabled=service_data.get("enabled") is True,
            socket=service_socket
            if isinstance(service_socket, str) and service_socket.strip()
            else RerankerServiceConfig.socket,
            timeout_ms=_positive_int(
                service_data.get("timeout_ms"), RerankerServiceConfig.timeout_ms
            ),
            mode=service_mode,
            canary_percent=_bounded_int(
                service_data.get("canary_percent"),
                RerankerServiceConfig.canary_percent,
                minimum=0,
                maximum=100,
            ),
            queue_size=_bounded_int(
                service_data.get("queue_size"),
                RerankerServiceConfig.queue_size,
                minimum=1,
                maximum=64,
            ),
        ),
    )


@dataclass(frozen=True)
class NegativeFeedbackConfig:
    enabled: bool = False
    kinds: tuple[str, ...] = ("page_ignored", "injection_ignored", "false-positive")
    similarity_threshold: float = 0.35
    penalty: float = 0.85
    max_age_days: int = 180
    max_entries: int = 500


def load_negative_feedback_config(
    path: Path | str | None = None,
) -> NegativeFeedbackConfig:
    data = load_toml_file(path)
    search = data.get("search")
    section: Any = None
    if isinstance(search, dict) and isinstance(search.get("negative_feedback"), dict):
        section = search["negative_feedback"]
    if not isinstance(section, dict):
        return NegativeFeedbackConfig()

    kinds_value = section.get("kinds")
    kinds = NegativeFeedbackConfig.kinds
    if isinstance(kinds_value, list):
        cleaned = tuple(k for k in kinds_value if isinstance(k, str) and k.strip())
        if cleaned:
            kinds = cleaned
    threshold = section.get("similarity_threshold")
    penalty = section.get("penalty")
    return NegativeFeedbackConfig(
        enabled=section.get("enabled") is True,
        kinds=kinds,
        similarity_threshold=(
            float(threshold)
            if isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and 0.0 < float(threshold) <= 1.0
            else NegativeFeedbackConfig.similarity_threshold
        ),
        penalty=(
            float(penalty)
            if isinstance(penalty, (int, float))
            and not isinstance(penalty, bool)
            and 0.0 < float(penalty) <= 1.0
            else NegativeFeedbackConfig.penalty
        ),
        max_age_days=_positive_int(
            section.get("max_age_days"), NegativeFeedbackConfig.max_age_days
        ),
        max_entries=_positive_int(
            section.get("max_entries"), NegativeFeedbackConfig.max_entries
        ),
    )


def normalize_audit_config(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = dict(data)
    audit = data.get("audit")
    if isinstance(audit, dict):
        out["auditor"] = audit
    return out


def config_summary(path: Path | str | None = None) -> dict[str, Any]:
    resolved = active_config_file(path)
    data = load_toml_file(resolved)
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "mode": "canonical" if resolved == CONFIG_FILE else "override",
        "hook_policy": load_hook_policy(resolved).__dict__,
        "sections": sorted(data.keys()),
    }
