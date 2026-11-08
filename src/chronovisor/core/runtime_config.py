"""Shared Chronovisor runtime configuration helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import chronovisor
from chronovisor.core.store import CHRONOVISOR_ROOT

CONFIG_FILE = CHRONOVISOR_ROOT / "config.toml"

FALSE_VALUES = {"0", "false", "False", "no", "NO", "off", "OFF"}
TRUE_VALUES = {"1", "true", "True", "yes", "YES", "on", "ON"}
DEFAULT_GITHUB_REPOSITORY = "trafficsign/chronovisor"
DEFAULT_RUNTIME_SOURCE = f"git+ssh://git@github.com/{DEFAULT_GITHUB_REPOSITORY}"
DEFAULT_USER_AGENT = (
    f"Chronovisor/0.1 (+https://github.com/{DEFAULT_GITHUB_REPOSITORY})"
)
DEFAULT_LAUNCHD_LABEL_PREFIX = "com.trafficsign.chronovisor-"
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


def _clean_text(value: object, default: str) -> str:
    if (
        isinstance(value, str)
        and value
        and value == value.strip()
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return value
    return default


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
        executable or os.environ.get("CHRONOVISOR_PYTHON", "").strip() or "python3.14"
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
        "runtime_source": runtime_source(),
        "commit_id": commit_id,
        "expected_commit": expected_commit,
        "drift": bool(commit_id and expected_commit and commit_id != expected_commit),
        "archive_path": str(Path(chronovisor.__file__).resolve().parents[2]),
        "module_path": str(module_path),
        "package_version": package_version,
        "direct_url": direct_url,
    }


@dataclass(frozen=True)
class HookPolicy:
    stop_save: bool = True


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
    incremental_idle_unload_seconds: int = 300
    maintenance_max_batch: int = 32
    offline: bool = True
    rollout_mode: str = "off"
    canary_percent: int = 0
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
        }
        return normalized
    except OSError, UnicodeDecodeError, tomllib.TOMLDecodeError:
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
    return HookPolicy(
        stop_save=nested_bool(load_toml_file(path), ("hooks", "stop", "save"), True)
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
            choices={"mps", "cpu", "mlx"},
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
            choices={"cpu", "mlx"},
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
