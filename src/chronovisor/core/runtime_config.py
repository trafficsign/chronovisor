"""Shared Chronovisor runtime configuration helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import chronovisor
from chronovisor.core.paths import CHRONOVISOR_ROOT

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


class RuntimeConfigError(RuntimeError):
    """Raised when the present runtime config cannot be read safely."""

    category = "config_error"
    permanent = True

    def __init__(self) -> None:
        super().__init__("config.toml is unreadable or invalid")


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
    except FileNotFoundError:
        return {}
    except OSError:
        raise RuntimeConfigError() from None
    try:
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
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        raise RuntimeConfigError() from None


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
