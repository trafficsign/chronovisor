"""Frontier-model review and autonomous patch execution."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from html import unescape
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from llm_wiki_mcp.convergence import (
    HUMAN_REQUIRED_FAILURE_CLASSES as CONVERGENCE_HUMAN_REQUIRED_FAILURE_CLASSES,
    is_human_required_failure,
)
from llm_wiki_mcp import runtime_status
from llm_wiki_mcp.runtime_config import uvx_runtime_command
from llm_wiki_mcp.wiki import WIKI_ROOT

FRONTIER_ACTIVITY_DIR = WIKI_ROOT / "runtime" / "frontier-reviews" / "active"

FRONTIER_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "summary",
        "tests_run",
        "commit",
        "committed",
        "pushed",
        "risk",
        "notes",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "quarantined", "needs_retry"],
        },
        "summary": {"type": "string"},
        "tests_run": {"type": "array", "items": {"type": "string"}},
        "commit": {"type": ["string", "null"]},
        "committed": {"type": "boolean"},
        "pushed": {"type": "boolean"},
        "risk": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
    },
}


def _bounded_timeout(timeout: int | None) -> int:
    requested = timeout or int(
        os.environ.get("LLM_WIKI_FRONTIER_TIMEOUT_SECONDS", "3600")
    )
    deadline_raw = os.environ.get("LLM_WIKI_CYCLE_DEADLINE_MONOTONIC")
    if not deadline_raw:
        return max(1, requested)
    try:
        remaining = max(1, int(float(deadline_raw) - time.monotonic()))
    except ValueError:
        return max(1, requested)
    return max(1, min(requested, remaining))


OFFICIAL_DOC_DOMAINS = {
    "platform.openai.com",
    "docs.openai.com",
    "openai.com",
    "docs.anthropic.com",
    "support.anthropic.com",
    "anthropic.com",
}

OFFICIAL_GITHUB_PREFIXES = {
    "/openai/",
    "/anthropics/",
    "/anthropic-ai/",
    "/trafficsign/llm-wiki-mcp",
}

OFFICIAL_FRONTIER_REFERENCE_URLS = (
    "https://platform.openai.com/docs",
    "https://docs.openai.com/codex",
    "https://github.com/openai/codex",
    "https://docs.anthropic.com/en/docs/claude-code",
    "https://github.com/anthropics/claude-code",
)

CODEX_REQUIRED_EXEC_OPTIONS = (
    "--cd",
    "--output-schema",
    "--skip-git-repo-check",
    "--ephemeral",
    "--ignore-rules",
    "--output-last-message",
)

CODEX_OPTION_ALIASES = {
    "--cd": ("--cd", "-C"),
    "--output-last-message": ("--output-last-message", "-o"),
}

# Autonomous memory reviews must not inherit a user-wide experimental model
# or reasoning level that the installed Codex CLI cannot execute.
DEFAULT_FRONTIER_MODEL = "gpt-5.5"
DEFAULT_FRONTIER_REASONING_EFFORT = "medium"

HUMAN_REQUIRED_FAILURE_CLASSES = CONVERGENCE_HUMAN_REQUIRED_FAILURE_CLASSES

SECRET_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)(api[_-]?key['\"=\s:]+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(token['\"=\s:]+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(cookie['\"=\s:]+)[^\s\n;]{8,}"),
]


@dataclass(frozen=True)
class FrontierFailure:
    failure_class: str
    rescue_status: str
    summary: str
    human_required: bool = False
    notify_user: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_class": self.failure_class,
            "rescue_status": self.rescue_status,
            "summary": self.summary,
            "human_required": self.human_required,
            "notify_user": self.notify_user,
        }


@dataclass(frozen=True)
class FrontierResult:
    decision: str
    summary: str
    tests_run: list[str]
    committed: bool
    pushed: bool
    commit: str | None = None
    risk: str | None = None
    notes: str | None = None
    raw_output: str | None = None
    frontier_failure: dict[str, Any] | None = None
    rescue_status: str | None = None
    human_required: bool = False
    notify_user: bool = False
    rescue_attempt: dict[str, Any] | None = None
    access_repair: dict[str, Any] | None = None
    execution_started: bool = False
    verified: bool = False
    verification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "summary": self.summary,
            "tests_run": self.tests_run,
            "commit": self.commit,
            "committed": self.committed,
            "pushed": self.pushed,
            "risk": self.risk,
            "notes": self.notes,
            "raw_output": self.raw_output,
            "frontier_failure": self.frontier_failure,
            "rescue_status": self.rescue_status,
            "human_required": self.human_required,
            "notify_user": self.notify_user,
            "rescue_attempt": self.rescue_attempt,
            "access_repair": self.access_repair,
            "execution_started": self.execution_started,
            "verified": self.verified,
            "verification": self.verification,
        }


def redact_sensitive_text(text: str | None) -> str:
    redacted = text or ""
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(_redact_match, redacted)
    return redacted


def _redact_match(match: re.Match[str]) -> str:
    if match.groups():
        return f"{match.group(1)}[REDACTED]"
    return "[REDACTED]"


def is_allowed_official_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    host = parsed.netloc.lower()
    if host == "github.com":
        path = parsed.path.lower()
        return any(
            path == prefix.rstrip("/") or path.startswith(prefix)
            for prefix in OFFICIAL_GITHUB_PREFIXES
        )
    return any(
        host == domain or host.endswith(f".{domain}") for domain in OFFICIAL_DOC_DOMAINS
    )


def official_frontier_reference_urls() -> list[str]:
    urls = list(OFFICIAL_FRONTIER_REFERENCE_URLS)
    extra_urls = os.environ.get("LLM_WIKI_FRONTIER_DOC_URLS")
    if extra_urls:
        urls.extend(part.strip() for part in extra_urls.split(",") if part.strip())
    return [url for url in urls if is_allowed_official_url(url)]


def collect_official_frontier_docs(query: str, *, max_docs: int = 3) -> dict[str, Any]:
    if os.environ.get("LLM_WIKI_FRONTIER_DOC_LOOKUP", "1") in {"0", "false", "False"}:
        return {"attempted": False, "reason": "disabled", "documents": []}

    try:
        import httpx
    except Exception as exc:
        return {
            "attempted": False,
            "reason": f"httpx unavailable: {exc}",
            "documents": [],
        }

    documents: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    tokens = [
        token for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query.lower())[:12]
    ]
    for url in official_frontier_reference_urls()[:max_docs]:
        try:
            response = httpx.get(url, timeout=3.0, follow_redirects=True)
            response.raise_for_status()
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})
            continue
        final_url = str(response.url)
        if not is_allowed_official_url(final_url):
            errors.append(
                {"url": url, "error": f"redirected outside allowlist: {final_url}"}
            )
            continue
        text = _html_to_text(response.text)
        snippet = _best_doc_snippet(text, tokens)
        documents.append(
            {
                "url": final_url,
                "status_code": response.status_code,
                "snippet": redact_sensitive_text(snippet),
            }
        )
    return {
        "attempted": True,
        "allowlist": official_frontier_reference_urls(),
        "documents": documents,
        "errors": errors,
    }


def _html_to_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _best_doc_snippet(text: str, tokens: list[str], *, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    lower = text.lower()
    best_idx = 0
    best_score = -1
    for match in re.finditer(r"[.!?]\s+", text):
        idx = match.end()
        window = lower[idx : idx + limit]
        score = sum(1 for token in tokens if token in window)
        if score > best_score:
            best_idx = idx
            best_score = score
    return text[best_idx : best_idx + limit]


def _frontier_failure(
    failure_class: str,
    rescue_status: str,
    summary: str,
    *,
    human_required: bool | None = None,
) -> FrontierFailure:
    # ``human_required`` remains a compatibility-only argument.  Callers and
    # model payloads cannot widen this boundary: it is derived solely from the
    # deterministic failure-class allowlist shared with convergence state.
    _ = human_required
    needs_human = is_human_required_failure(failure_class)
    canonical_status = "human_required" if needs_human else rescue_status
    if not needs_human and canonical_status == "human_required":
        canonical_status = "frontier_retry"
    return FrontierFailure(
        failure_class=failure_class,
        rescue_status=canonical_status,
        summary=summary,
        human_required=needs_human,
        notify_user=needs_human,
    )


def classify_frontier_failure(text: str | None) -> FrontierFailure:
    clean = redact_sensitive_text(text)
    lower = clean.lower()
    if "invalid_json_schema" in lower or "invalid schema for response_format" in lower:
        return _frontier_failure(
            "schema_invalid",
            "pending_frontier_review",
            "frontier structured output schema is invalid for the current API",
            human_required=False,
        )
    auth_state_markers = (
        "denied",
        "expired",
        "invalid",
        "missing",
        "required",
        "revoked",
    )
    if "oauth" in lower and any(
        marker in lower for marker in ("login", "reauth", *auth_state_markers)
    ):
        return _frontier_failure(
            "oauth_required",
            "human_required",
            "frontier OAuth login appears to require human action",
        )
    if (
        "missing bearer" in lower
        or ("401" in lower and "unauthorized" in lower)
        or ("403" in lower and "forbidden" in lower)
        or (
            any(marker in lower for marker in ("api key", "api_key", "authentication"))
            and any(marker in lower for marker in auth_state_markers)
        )
    ):
        return _frontier_failure(
            "auth_required",
            "human_required",
            "frontier API authentication is missing or invalid",
        )
    if "insufficient_quota" in lower or "billing" in lower or "quota exceeded" in lower:
        return _frontier_failure(
            "quota_or_billing_required",
            "human_required",
            "frontier quota or billing state requires human action",
        )
    permission_markers = ("denied", "permission", "not permitted", "access refused")
    if "keychain" in lower and any(marker in lower for marker in permission_markers):
        return _frontier_failure(
            "keychain_permission_required",
            "human_required",
            "Keychain access for frontier credentials requires human action",
        )
    secret_store_markers = (
        "secret store",
        "secret service",
        "credential store",
        "credential helper",
    )
    if any(marker in lower for marker in secret_store_markers) and any(
        marker in lower for marker in permission_markers
    ):
        return _frontier_failure(
            "secret_store_permission_required",
            "human_required",
            "credential secret-store permission requires human action",
        )
    if (
        "unknown option" in lower
        or "unrecognized option" in lower
        or "unexpected argument" in lower
    ):
        return _frontier_failure(
            "cli_option_invalid",
            "pending_frontier_review",
            "frontier CLI option set is incompatible with the installed version",
            human_required=False,
        )
    if "timed out" in lower or "timeout" in lower or "temporarily unavailable" in lower:
        return _frontier_failure(
            "network_transient",
            "frontier_retry",
            "frontier call failed with a transient network or service error",
            human_required=False,
        )
    return _frontier_failure(
        "unknown_frontier_failure",
        "frontier_retry",
        "frontier call failed for an unknown reason",
        human_required=False,
    )


def _schema_validation_failure(schema: dict[str, Any]) -> FrontierFailure | None:
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        return _frontier_failure(
            "schema_invalid",
            "frontier_preflight_failed",
            "frontier output schema must declare properties and required",
            human_required=False,
        )
    missing = sorted(set(properties) - set(required))
    if missing:
        return _frontier_failure(
            "schema_invalid",
            "frontier_preflight_failed",
            f"frontier output schema required is missing: {', '.join(missing)}",
            human_required=False,
        )
    if schema.get("additionalProperties") is not False:
        return _frontier_failure(
            "schema_invalid",
            "frontier_preflight_failed",
            "frontier output schema must set additionalProperties=false",
            human_required=False,
        )
    return None


def _strict_schema_with_repair(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    strict_schema = json.loads(json.dumps(schema))
    properties = strict_schema.get("properties")
    if not isinstance(properties, dict):
        return strict_schema, None

    required = strict_schema.get("required")
    expected_required = list(properties.keys())
    changes: list[dict[str, Any]] = []
    if required != expected_required:
        strict_schema["required"] = expected_required
        changes.append(
            {
                "field": "required",
                "before": required,
                "after": expected_required,
            }
        )
    if strict_schema.get("additionalProperties") is not False:
        changes.append(
            {
                "field": "additionalProperties",
                "before": strict_schema.get("additionalProperties"),
                "after": False,
            }
        )
        strict_schema["additionalProperties"] = False

    if not changes:
        return strict_schema, None
    return strict_schema, {
        "type": "schema_strictness_autofix",
        "applied": True,
        "changes": changes,
        "summary": "normalized frontier output schema for strict structured output",
    }


def run_frontier_preflight() -> dict[str, Any]:
    strict_schema, schema_repair = _strict_schema_with_repair(FRONTIER_DECISION_SCHEMA)
    schema_failure = _schema_validation_failure(strict_schema)
    if schema_failure:
        return {"ok": False, "failure": schema_failure.to_dict()}
    codex = shutil.which("codex")
    if codex is None:
        failure = _frontier_failure(
            "frontier_tool_unavailable",
            "frontier_retry",
            "codex executable not found",
        )
        return {"ok": False, "failure": failure.to_dict()}
    metadata = _codex_cli_metadata(codex)
    if not metadata.get("ok"):
        return metadata
    auth_path = _codex_home() / "auth.json"
    if not auth_path.exists() and not os.environ.get("OPENAI_API_KEY"):
        failure = _frontier_failure(
            "auth_required",
            "human_required",
            f"Codex auth not found at {auth_path}",
        )
        return {"ok": False, "failure": failure.to_dict()}
    return {
        "ok": True,
        "codex_home": str(_codex_home()),
        "auth_path": str(auth_path),
        "codex": metadata.get("codex"),
        "repairs": [schema_repair] if schema_repair else [],
    }


def _codex_cli_metadata(codex: str) -> dict[str, Any]:
    version = _run_small_command([codex, "--version"])
    if not version["ok"]:
        failure = classify_frontier_failure(version.get("output"))
        return {
            "ok": False,
            "failure": failure.to_dict(),
            "codex": {"version": version},
        }

    help_result = _run_small_command([codex, "exec", "--help"])
    if not help_result["ok"]:
        failure = classify_frontier_failure(help_result.get("output"))
        return {
            "ok": False,
            "failure": failure.to_dict(),
            "codex": {"help": help_result},
        }

    help_text = str(help_result.get("output") or "")
    missing = [
        option for option in CODEX_REQUIRED_EXEC_OPTIONS if option not in help_text
    ]
    return {
        "ok": True,
        "codex": {
            "version": {**version, "output": str(version.get("output") or "")[-500:]},
            "exec_help": {**help_result, "output": help_text[-4000:]},
            "missing_exec_options": missing,
            "adaptive_required": bool(missing),
        },
    }


def _run_small_command(cmd: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=10,
            env=_frontier_env(),
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "command": cmd}
    output = redact_sensitive_text(
        (completed.stdout or "") + "\n" + (completed.stderr or "")
    )
    return {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "command": cmd,
        "output": output[-4000:],
    }


def _option_supported(exec_help: str, option: str) -> bool:
    aliases = CODEX_OPTION_ALIASES.get(option, (option,))
    return any(alias in exec_help for alias in aliases)


def _preferred_option(exec_help: str, option: str) -> str | None:
    aliases = CODEX_OPTION_ALIASES.get(option, (option,))
    for alias in aliases:
        if alias in exec_help:
            return alias
    return None


def _build_codex_exec_invocation(
    codex: str,
    *,
    repo_root: Path,
    schema_path: Path,
    output_path: Path,
    execute_patch: bool,
    preflight: dict[str, Any],
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    codex_meta = (
        preflight.get("codex") if isinstance(preflight.get("codex"), dict) else {}
    )
    exec_help_data = (
        codex_meta.get("exec_help")
        if isinstance(codex_meta.get("exec_help"), dict)
        else {}
    )
    exec_help = str(exec_help_data.get("output") or "")
    cmd = [codex, "exec"]
    cwd: Path | None = None
    repairs: list[dict[str, Any]] = []

    cd_option = _preferred_option(exec_help, "--cd")
    if cd_option:
        cmd.extend([cd_option, str(repo_root)])
    else:
        cwd = repo_root
        repairs.append(
            {
                "type": "cli_option_adapted",
                "option": "--cd",
                "replacement": "subprocess.cwd",
            }
        )

    if _option_supported(exec_help, "--output-schema"):
        cmd.extend(["--output-schema", str(schema_path)])
    else:
        repairs.append(
            {
                "type": "cli_option_adapted",
                "option": "--output-schema",
                "replacement": "prompt_json_contract",
            }
        )

    output_option = _preferred_option(exec_help, "--output-last-message")
    if output_option:
        cmd.extend([output_option, str(output_path)])
    else:
        repairs.append(
            {
                "type": "cli_option_adapted",
                "option": "--output-last-message",
                "replacement": "stdout_capture",
            }
        )

    if _option_supported(exec_help, "--skip-git-repo-check"):
        cmd.append("--skip-git-repo-check")
    else:
        repairs.append(
            {
                "type": "cli_option_adapted",
                "option": "--skip-git-repo-check",
                "replacement": "omitted",
            }
        )

    if _option_supported(exec_help, "--ephemeral"):
        cmd.append("--ephemeral")
    else:
        repairs.append(
            {
                "type": "cli_option_adapted",
                "option": "--ephemeral",
                "replacement": "omitted",
            }
        )

    if _option_supported(exec_help, "--ignore-rules"):
        cmd.append("--ignore-rules")
    else:
        repairs.append(
            {
                "type": "cli_option_adapted",
                "option": "--ignore-rules",
                "replacement": "omitted",
            }
        )

    # A frontier child must never execute the parent host's Stop hooks. Without
    # this, each review recursively schedules save/correction/review children.
    if _option_supported(exec_help, "--disable"):
        cmd.extend(["--disable", "hooks"])
    else:
        repairs.append(
            {
                "type": "cli_option_adapted",
                "option": "--disable hooks",
                "replacement": "LLM_WIKI_INTERNAL_FRONTIER guard",
            }
        )

    if not execute_patch:
        sandbox_option = _preferred_option(exec_help, "--sandbox")
        if sandbox_option:
            cmd.extend([sandbox_option, "read-only"])
        else:
            repairs.append(
                {
                    "type": "cli_option_adapted",
                    "option": "--sandbox",
                    "replacement": "default_sandbox",
                }
            )

    model = (
        model or os.environ.get("LLM_WIKI_FRONTIER_MODEL", DEFAULT_FRONTIER_MODEL)
    ).strip()
    reasoning_effort = (
        reasoning_effort
        or os.environ.get(
            "LLM_WIKI_FRONTIER_REASONING_EFFORT",
            DEFAULT_FRONTIER_REASONING_EFFORT,
        )
    ).strip()
    if model:
        cmd.extend(["--model", model])
    if reasoning_effort:
        cmd.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])

    if execute_patch:
        if _option_supported(exec_help, "--dangerously-bypass-approvals-and-sandbox"):
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            repairs.append(
                {
                    "type": "cli_option_adapted",
                    "option": "--dangerously-bypass-approvals-and-sandbox",
                    "replacement": "omitted",
                }
            )

    return {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else None,
        "output_path": str(output_path) if output_option else None,
        "schema_path": str(schema_path)
        if _option_supported(exec_help, "--output-schema")
        else None,
        "repairs": repairs,
        "source": "codex_exec_help",
    }


def _failure_result(
    *,
    summary: str,
    output: str = "",
    failure: FrontierFailure | None = None,
    rescue_attempt: dict[str, Any] | None = None,
    access_repair: dict[str, Any] | None = None,
) -> FrontierResult:
    failure = failure or classify_frontier_failure(output)
    return FrontierResult(
        decision="needs_retry",
        summary=summary,
        tests_run=[],
        committed=False,
        pushed=False,
        raw_output=redact_sensitive_text(output)[-4000:],
        frontier_failure=failure.to_dict(),
        rescue_status=failure.rescue_status,
        human_required=failure.human_required,
        notify_user=failure.notify_user,
        rescue_attempt=rescue_attempt,
        access_repair=access_repair,
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _structured_subprocess_failure(
    exc: subprocess.TimeoutExpired | OSError,
    *,
    reviewer: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Normalize process-launch failures at the structured-review boundary."""

    if isinstance(exc, subprocess.TimeoutExpired):
        # A timeout is always retryable even when partial stderr happens to
        # contain stale auth-like text from the child process.
        failure = _frontier_failure(
            "network_transient",
            "frontier_retry",
            "frontier call timed out and can be retried automatically",
            human_required=False,
        )
    else:
        detail = redact_sensitive_text(f"{exc.__class__.__name__}: {exc}")
        failure = classify_frontier_failure(detail)
        lower = detail.lower()
        unavailable_errnos = {errno.ENOENT, errno.ENOTDIR, errno.EACCES, errno.ENOEXEC}
        unavailable_markers = (
            "no such file or directory",
            "executable not found",
            "permission denied",
            "exec format error",
        )
        if failure.failure_class == "unknown_frontier_failure" and (
            getattr(exc, "errno", None) in unavailable_errnos
            or any(marker in lower for marker in unavailable_markers)
        ):
            failure = _frontier_failure(
                "frontier_tool_unavailable",
                "frontier_retry",
                f"{reviewer} executable could not be started",
            )
    return _structured_failure_payload(
        schema,
        summary=failure.summary,
        failure=failure,
        reviewer=reviewer,
    )


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _structured_validation_error(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str = "$",
) -> str | None:
    expected = schema.get("type")
    allowed_types = expected if isinstance(expected, list) else [expected]
    allowed_types = [item for item in allowed_types if isinstance(item, str)]
    if allowed_types and not any(
        _schema_type_matches(value, item) for item in allowed_types
    ):
        return f"{path}: expected {'|'.join(allowed_types)}"
    if "enum" in schema and value not in schema.get("enum", []):
        return f"{path}: value is outside enum"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return f"{path}: number must be finite"
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: above maximum"
    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            return f"{path}: too few items"
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return f"{path}: too many items"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = _structured_validation_error(
                    item, item_schema, path=f"{path}[{index}]"
                )
                if error:
                    return error
    if isinstance(value, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        required = schema.get("required")
        required = required if isinstance(required, list) else []
        missing = [name for name in required if name not in value]
        if missing:
            return f"{path}: missing required fields: {', '.join(str(name) for name in missing)}"
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                return f"{path}: unexpected fields: {', '.join(extras)}"
        for name, child_schema in properties.items():
            if name not in value or not isinstance(child_schema, dict):
                continue
            error = _structured_validation_error(
                value[name], child_schema, path=f"{path}.{name}"
            )
            if error:
                return error
    return None


def _failure_default(name: str, schema: dict[str, Any], summary: str) -> Any:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        if "needs_retry" in enum:
            return "needs_retry"
        return enum[0]
    expected = schema.get("type")
    types = expected if isinstance(expected, list) else [expected]
    if "null" in types:
        return None
    if name == "decision" and "string" in types:
        return "needs_retry"
    if name in {"summary", "reason", "notes"} and "string" in types:
        return summary
    if name == "confidence" and "number" in types:
        return 0.0
    if "string" in types:
        return ""
    if "number" in types or "integer" in types:
        return 0
    if "boolean" in types:
        return False
    if "array" in types:
        return []
    if "object" in types:
        return {}
    return None


def _structured_failure_payload(
    schema: dict[str, Any],
    *,
    summary: str,
    failure: FrontierFailure,
    reviewer: str,
    diagnostics: str = "",
) -> dict[str, Any]:
    strict_schema, _repair = _strict_schema_with_repair(schema)
    properties = strict_schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    payload = {
        name: _failure_default(name, field_schema, summary)
        for name, field_schema in properties.items()
        if isinstance(field_schema, dict)
    }
    failure_payload = failure.to_dict()
    if diagnostics:
        redacted = redact_sensitive_text(diagnostics)
        failure_payload["diagnostics_tail"] = redacted[-4000:]
        failure_payload["diagnostics_sha256"] = hashlib.sha256(
            redacted.encode("utf-8")
        ).hexdigest()
    payload["frontier_failure"] = failure_payload
    payload["human_required"] = failure.human_required
    payload["reviewer"] = reviewer
    return payload


def _validated_structured_result(
    parsed: dict[str, Any] | None,
    schema: dict[str, Any],
    *,
    reviewer: str,
) -> dict[str, Any]:
    strict_schema, _repair = _strict_schema_with_repair(schema)
    if parsed is None:
        failure = _frontier_failure(
            "schema_invalid",
            "pending_frontier_review",
            "frontier output did not contain JSON",
        )
        return _structured_failure_payload(
            schema, summary=failure.summary, failure=failure, reviewer=reviewer
        )
    error = _structured_validation_error(parsed, strict_schema)
    if error:
        failure = _frontier_failure(
            "schema_invalid",
            "pending_frontier_review",
            f"frontier output failed schema validation: {error}",
        )
        return _structured_failure_payload(
            schema, summary=failure.summary, failure=failure, reviewer=reviewer
        )
    result = dict(parsed)
    result["reviewer"] = reviewer
    return result


def _parse_result(text: str) -> FrontierResult:
    parsed = _extract_json_object(text)
    if parsed is None:
        return _failure_result(
            summary="frontier output did not contain JSON",
            output=text,
        )
    decision = parsed.get("decision")
    summary = parsed.get("summary")
    tests = parsed.get("tests_run")
    committed = parsed.get("committed")
    pushed = parsed.get("pushed")
    if (
        decision not in {"approved", "rejected", "quarantined", "needs_retry"}
        or not isinstance(summary, str)
        or not isinstance(tests, list)
        or not all(isinstance(t, str) for t in tests)
        or not isinstance(committed, bool)
        or not isinstance(pushed, bool)
    ):
        return _failure_result(
            summary="frontier JSON failed schema validation",
            output=text,
        )
    commit = parsed.get("commit")
    risk = parsed.get("risk")
    notes = parsed.get("notes")
    return FrontierResult(
        decision=decision,
        summary=summary,
        tests_run=tests,
        commit=commit if isinstance(commit, str) else None,
        committed=committed,
        pushed=pushed,
        risk=risk if isinstance(risk, str) else None,
        notes=notes if isinstance(notes, str) else None,
        raw_output=redact_sensitive_text(text)[-4000:],
    )


def _codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured).expanduser()
    default_config = Path.home() / ".config" / "codex"
    if default_config.exists():
        return default_config
    return Path.home() / ".codex"


def _frontier_env(*, codex_home: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home or _codex_home())
    env.setdefault("NO_COLOR", "1")
    env["LLM_WIKI_INTERNAL_FRONTIER"] = "1"
    env["LLM_WIKI_CONTENT_CORRECTION_ENABLED"] = "0"
    env["LLM_WIKI_RECALL_AUDIT_ENABLED"] = "0"
    env["LLM_WIKI_RECALL_IMPROVE_ENABLED"] = "0"
    env["CODEX_WIKI_SAVE_ENABLED"] = "0"
    env["CLAUDE_CODE_WIKI_SAVE_ENABLED"] = "0"
    return env


@contextmanager
def _isolated_codex_environment() -> Iterator[dict[str, str]]:
    """Yield a minimal authenticated Codex home with no MCP or hook config."""

    source_home = _codex_home()
    with tempfile.TemporaryDirectory(prefix="llm-wiki-frontier-codex-") as td:
        isolated_home = Path(td)
        for filename in ("auth.json", "models_cache.json", "version.json"):
            source = source_home / filename
            if source.exists():
                (isolated_home / filename).symlink_to(source)
        (isolated_home / "config.toml").write_text(
            "# Isolated LLM Wiki frontier reviewer: intentionally no MCP servers or hooks.\n"
            'approval_policy = "never"\n'
            'sandbox_mode = "workspace-write"\n',
            encoding="utf-8",
        )
        yield _frontier_env(codex_home=isolated_home)


def _write_frontier_activity(record: dict[str, Any]) -> Path | None:
    try:
        FRONTIER_ACTIVITY_DIR.mkdir(parents=True, exist_ok=True)
        path = FRONTIER_ACTIVITY_DIR / f"{record['review_id']}.json"
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return path
    except Exception:
        return None


@contextmanager
def _frontier_activity(
    *,
    kind: str,
    reviewer: str,
    model: str | None,
    prompt: str,
    repo_root: Path | None,
) -> Iterator[dict[str, Any]]:
    review_id = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    started_at = datetime.now().isoformat(timespec="seconds")
    record: dict[str, Any] = {
        "review_id": review_id,
        "active": True,
        "kind": kind,
        "reviewer": reviewer,
        "model": model,
        "pid": os.getpid(),
        "repo_root": str(repo_root) if repo_root else None,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "started_at": started_at,
        "updated_at": started_at,
    }
    path = _write_frontier_activity(record)
    runtime_status.safe_append_event(
        "info",
        "frontier | review started",
        source="frontier",
        review_id=review_id,
        kind=kind,
        reviewer=reviewer,
        model=model,
    )
    started = time.monotonic()
    try:
        yield record
    except BaseException as exc:
        record["outcome"] = "error"
        record["error"] = exc.__class__.__name__
        raise
    finally:
        elapsed = round(max(0.0, time.monotonic() - started), 3)
        outcome = str(record.get("outcome") or "completed")
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        event_level = (
            "error"
            if outcome == "error"
            else "warn"
            if outcome == "failed"
            else "success"
        )
        runtime_status.safe_append_event(
            event_level,
            f"frontier | review {outcome}",
            source="frontier",
            review_id=review_id,
            kind=kind,
            reviewer=reviewer,
            model=model,
            elapsed_seconds=elapsed,
        )


def _run_review_subprocess(
    cmd: list[str],
    *,
    activity_kind: str,
    reviewer: str,
    model: str | None,
    prompt: str,
    repo_root: Path | None,
    isolate_codex: bool,
    permit: Any,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    with _frontier_activity(
        kind=activity_kind,
        reviewer=reviewer,
        model=model,
        prompt=prompt,
        repo_root=repo_root,
    ) as activity:
        if isolate_codex:
            with _isolated_codex_environment() as env:
                kwargs["env"] = env
                completed = _spawn_guarded_process(cmd, permit=permit, **kwargs)
        else:
            completed = _spawn_guarded_process(cmd, permit=permit, **kwargs)
        activity["outcome"] = "completed" if completed.returncode == 0 else "failed"
        return completed


def _spawn_guarded_process(
    cmd: list[str],
    *,
    permit: Any,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Spawn first, then atomically charge the guard to the real child PID."""

    input_text = kwargs.pop("input", None)
    timeout = kwargs.pop("timeout", None)
    capture_output = bool(kwargs.pop("capture_output", False))
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if input_text is not None:
        kwargs["stdin"] = subprocess.PIPE
    process = subprocess.Popen(cmd, **kwargs)
    try:
        permit.start(pid=process.pid)
    except BaseException:
        process.kill()
        process.communicate()
        raise
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(
        cmd,
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def build_frontier_prompt(
    packet: dict[str, Any],
    local_decision: dict[str, Any] | None,
    *,
    execute_patch: bool,
) -> str:
    mode = "IMPLEMENT" if execute_patch else "REVIEW"
    return f"""\
You are the final frontier reviewer for LLM Wiki self-healing.

Mode: {mode}

Goal:
- Diagnose the failure packet.
- If a system/code fix is required and execute mode is enabled, edit the repo.
- Add or update regression tests.
- Run the relevant tests.
- If the change is correct, commit it in the isolated checkout.
- Never push or add a Git remote. The parent verifier alone may publish a
  candidate after independent reproduction and full-suite checks pass.
- If the gate fails, do not leave half-finished changes; report needs_retry or quarantined.

Hard constraints:
- Return JSON only with this shape:
  {{
    "decision": "approved|rejected|quarantined|needs_retry",
    "summary": "...",
    "tests_run": ["..."],
    "commit": "hash or null",
    "committed": true/false,
    "pushed": true/false,
    "risk": "low|medium|high or null",
    "notes": "..."
  }}
- Do not ask a human for permission.
- Prefer tests + rollback-safe changes over broad rewrites.
- In IMPLEMENT mode, a successful result must report committed=true and
  pushed=false. There is intentionally no reachable remote in this checkout.

Failure packet:
{json.dumps(packet, ensure_ascii=False, indent=2)}

Local repair decision:
{json.dumps(local_decision, ensure_ascii=False, indent=2)}
"""


def _run_codex(
    prompt: str,
    *,
    repo_root: Path,
    timeout: int,
    execute_patch: bool,
    permit: Any,
) -> FrontierResult:
    codex = shutil.which("codex")
    if codex is None:
        failure = _frontier_failure(
            "frontier_tool_unavailable",
            "frontier_retry",
            "codex executable not found",
        )
        return _failure_result(
            summary="codex executable not found",
            failure=failure,
        )
    preflight = run_frontier_preflight()
    if not preflight.get("ok"):
        failure_data = preflight.get("failure")
        failure = (
            _frontier_failure(
                str(failure_data.get("failure_class") or "unknown_frontier_failure"),
                str(failure_data.get("rescue_status") or "frontier_retry"),
                str(failure_data.get("summary") or "frontier preflight failed"),
            )
            if isinstance(failure_data, dict)
            else classify_frontier_failure("")
        )
        return _failure_result(
            summary=str(failure.summary),
            failure=failure,
            access_repair={"preflight": preflight},
        )

    with tempfile.TemporaryDirectory() as td:
        schema_path = Path(td) / "frontier-decision.schema.json"
        output_path = Path(td) / "frontier-output.json"
        strict_schema, schema_repair = _strict_schema_with_repair(
            FRONTIER_DECISION_SCHEMA
        )
        schema_path.write_text(
            json.dumps(strict_schema, indent=2) + "\n",
            encoding="utf-8",
        )
        from llm_wiki_mcp.model_lab import resolve_role

        model, reasoning_effort = resolve_role("code_repair")
        invocation = _build_codex_exec_invocation(
            codex,
            repo_root=repo_root,
            schema_path=schema_path,
            output_path=output_path,
            execute_patch=execute_patch,
            preflight=preflight,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        access_repairs = [
            repair
            for repair in [
                *(preflight.get("repairs") or []),
                schema_repair,
                *invocation.get("repairs", []),
            ]
            if repair
        ]
        access_repair = {
            "applied": bool(access_repairs),
            "repairs": access_repairs,
            "invocation": {
                "source": invocation.get("source"),
                "cmd": invocation.get("cmd"),
                "cwd": invocation.get("cwd"),
                "schema_path": invocation.get("schema_path"),
                "output_path": invocation.get("output_path"),
            },
            "preflight": {
                "codex_home": preflight.get("codex_home"),
                "auth_path": preflight.get("auth_path"),
                "codex": preflight.get("codex"),
            },
        }
        completed = _run_review_subprocess(
            invocation["cmd"],
            activity_kind="code_repair",
            reviewer="codex",
            model=model,
            prompt=prompt,
            repo_root=repo_root,
            isolate_codex=True,
            permit=permit,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=invocation.get("cwd") or None,
        )
        output_text = ""
        if output_path.exists():
            output_text += output_path.read_text(encoding="utf-8", errors="replace")
        output_text += "\n" + (completed.stdout or "") + "\n" + (completed.stderr or "")
        if completed.returncode != 0:
            failure = classify_frontier_failure(output_text)
            return _failure_result(
                summary=f"codex exec failed with exit {completed.returncode}",
                output=output_text,
                failure=failure,
                access_repair=access_repair,
            )
        return replace(_parse_result(output_text), access_repair=access_repair)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def run_structured_review(
    prompt: str,
    schema: dict[str, Any],
    *,
    repo_root: Path,
    audit_root: Path | None = None,
    timeout: int | None = None,
    execute_patch: bool = False,
    command_env: str = "LLM_WIKI_STRUCTURED_REVIEW_CMD",
    model_role: str = "semantic_judge",
    decision_lane: str | None = None,
    model_override: str | None = None,
    reasoning_effort_override: str | None = None,
    record_replay: bool = True,
    system: str | None = None,
) -> dict[str, Any]:
    """Resolve a routine structured decision using local models only.

    This compatibility entry point is intentionally unable to invoke Codex,
    Claude, a custom frontier command, or the code-repair path.  The legacy
    frontier-shaped failure envelope is retained so existing callers can fail
    closed without losing their durable queue semantics.
    """
    del (
        repo_root,
        timeout,
        execute_patch,
        command_env,
        model_override,
        reasoning_effort_override,
    )

    from llm_wiki_mcp.decision_policy import resolve_decision_policy
    from llm_wiki_mcp.decision_router import DecisionRouter
    from llm_wiki_mcp.decision_schema_manifest import (
        production_schema_manifest,
        schema_sha256,
    )

    lane_policy, lane_mode, lane_error = resolve_decision_policy(decision_lane)
    policy_audit = {
        "lane": decision_lane,
        "kind": lane_policy.kind if lane_policy is not None else None,
        "schema_name": lane_policy.schema_name if lane_policy is not None else None,
        "mode": lane_mode,
        "error": lane_error,
    }
    if lane_error is not None or lane_mode == "off":
        reason = lane_error or f"decision_lane_off:{decision_lane}"
        failure = _frontier_failure(
            "local_decision_policy_blocked",
            "local_quarantined",
            reason,
            human_required=False,
        )
        result = _structured_failure_payload(
            schema,
            summary=reason,
            failure=failure,
            reviewer="local_policy",
        )
        result["decision_policy"] = policy_audit
        return result

    if lane_policy is None or lane_policy.kind not in {"consensus", "local_batch"}:
        reason = f"decision_lane_not_structured:{decision_lane}"
        failure = _frontier_failure(
            "local_decision_policy_kind_invalid",
            "local_quarantined",
            reason,
            human_required=False,
        )
        result = _structured_failure_payload(
            schema,
            summary=reason,
            failure=failure,
            reviewer="local_policy",
        )
        result["decision_policy"] = policy_audit
        return result

    expected_digest = production_schema_manifest().get(str(lane_policy.schema_name))
    actual_digest = schema_sha256(schema)
    policy_audit["expected_schema_sha256"] = expected_digest
    policy_audit["actual_schema_sha256"] = actual_digest
    if expected_digest is None or actual_digest != expected_digest:
        reason = f"decision_lane_schema_mismatch:{decision_lane}"
        failure = _frontier_failure(
            "local_decision_schema_mismatch",
            "local_quarantined",
            reason,
            human_required=False,
        )
        result = _structured_failure_payload(
            schema,
            summary=reason,
            failure=failure,
            reviewer="local_policy",
        )
        result["decision_policy"] = policy_audit
        return result

    router = DecisionRouter(
        audit_root=audit_root,
        audit_role=decision_lane or model_role,
        record_replay=record_replay,
        require_adopted=lane_mode == "enabled",
        decision_lane=decision_lane,
    )
    routed = (
        router.decide(prompt, schema)
        if system is None
        else router.decide(prompt, schema, system=system)
    )
    policy_audit["router_policy"] = router.policy.audit_record()
    if lane_mode == "shadow":
        reason = f"decision_lane_shadow:{decision_lane}"
        failure = _frontier_failure(
            "local_decision_shadow_only",
            "local_quarantined",
            reason,
            human_required=False,
        )
        result = _structured_failure_payload(
            schema,
            summary=reason,
            failure=failure,
            reviewer="local_consensus_shadow",
        )
        result["local_consensus"] = routed.audit_record()
        result["decision_policy"] = policy_audit
        return result
    if router.policy.source != "adopted_artifact":
        reason = f"decision_lane_unadopted:{decision_lane}"
        failure = _frontier_failure(
            "local_decision_artifact_required",
            "local_quarantined",
            reason,
            human_required=False,
        )
        result = _structured_failure_payload(
            schema,
            summary=reason,
            failure=failure,
            reviewer="local_policy",
        )
        result["local_consensus"] = routed.audit_record()
        result["decision_policy"] = policy_audit
        return result
    if routed.ok and isinstance(routed.decision, dict):
        result = _validated_structured_result(
            routed.decision,
            schema,
            reviewer="local_consensus",
        )
        result["local_consensus"] = routed.audit_record()
        result["decision_policy"] = policy_audit
        return result

    reason = (
        routed.quarantine_reason or routed.failure_class or "local consensus failed"
    )
    failure = _frontier_failure(
        "local_consensus_failed",
        "local_quarantined",
        reason,
        human_required=False,
    )
    result = _structured_failure_payload(
        schema,
        summary=reason,
        failure=failure,
        reviewer="local_consensus",
    )
    result["local_consensus"] = routed.audit_record()
    result["decision_policy"] = policy_audit
    return result


def _git_probe(
    repo_root: Path,
    args: list[str],
    *,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _capture_repair_baseline(repo_root: Path) -> dict[str, Any]:
    """Capture a clean, pushed main baseline before a repair process starts."""

    result: dict[str, Any] = {
        "ok": False,
        "head": None,
        "origin_main": None,
        "clean": False,
        "branch": None,
        "failure_class": None,
    }
    try:
        fetched = _git_probe(
            repo_root, ["fetch", "--quiet", "origin", "main"], timeout=120
        )
        head = _git_probe(repo_root, ["rev-parse", "HEAD"])
        origin = _git_probe(repo_root, ["rev-parse", "origin/main"])
        branch = _git_probe(repo_root, ["symbolic-ref", "--quiet", "--short", "HEAD"])
        status = _git_probe(
            repo_root,
            ["status", "--porcelain=v1", "--untracked-files=no"],
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["failure_class"] = f"git_probe_{exc.__class__.__name__}"
        return result
    if fetched.returncode != 0:
        result["failure_class"] = "git_baseline_fetch_failed"
        return result
    if (
        head.returncode != 0
        or origin.returncode != 0
        or branch.returncode != 0
        or status.returncode != 0
    ):
        result["failure_class"] = "git_baseline_unavailable"
        return result
    head_sha = head.stdout.strip()
    origin_sha = origin.stdout.strip()
    clean = not status.stdout.strip()
    result.update(
        {
            "head": head_sha,
            "origin_main": origin_sha,
            "clean": clean,
            "branch": branch.stdout.strip(),
        }
    )
    if not clean:
        result["failure_class"] = "repair_worktree_dirty"
        return result
    if not head_sha or head_sha != origin_sha:
        result["failure_class"] = "repair_baseline_not_pushed_main"
        return result
    if branch.stdout.strip() != "main":
        result["failure_class"] = "repair_baseline_not_main_branch"
        return result
    result["ok"] = True
    return result


@contextmanager
def _isolated_repair_checkout(
    repo_root: Path,
    baseline: dict[str, Any],
) -> Iterator[Path]:
    """Yield a disposable checkout with no Git remote.

    A local clone is used instead of a linked worktree because worktrees share
    remote configuration with the production checkout.  Removing ``origin``
    from this clone makes a model-initiated push structurally impossible.
    """

    baseline_head = str(baseline.get("head") or "")
    if not baseline_head:
        raise RuntimeError("repair baseline is missing HEAD")
    with tempfile.TemporaryDirectory(prefix="llm-wiki-frontier-repair-") as td:
        checkout = Path(td) / "candidate"
        cloned = subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-local",
                "--no-checkout",
                str(repo_root),
                str(checkout),
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=300,
        )
        if cloned.returncode != 0:
            raise RuntimeError("failed to create isolated repair checkout")
        checked_out = _git_probe(
            checkout, ["checkout", "--quiet", "--detach", baseline_head]
        )
        remote_removed = _git_probe(checkout, ["remote", "remove", "origin"])
        if checked_out.returncode != 0 or remote_removed.returncode != 0:
            raise RuntimeError("failed to isolate repair checkout")
        remotes = _git_probe(checkout, ["remote"])
        if remotes.returncode != 0 or remotes.stdout.strip():
            raise RuntimeError("isolated repair checkout unexpectedly has a remote")
        yield checkout


def _verification_command(
    command: list[str],
    *,
    repo_root: Path,
    timeout: int,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "returncode": None,
            "error_type": exc.__class__.__name__,
        }
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
    }


PERSISTENT_RUNTIME_LABELS = (
    "com.trafficsign.llm-wiki-dashboard",
    "com.trafficsign.llm-wiki-ingest-drain",
)


def _launchd_pid(label: str) -> int | None:
    """Return the running PID for a user LaunchAgent, without starting it."""

    launchctl = shutil.which("launchctl")
    if launchctl is None:
        return None
    completed = subprocess.run(
        [launchctl, "print", f"gui/{os.getuid()}/{label}"],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if completed.returncode != 0:
        return None
    match = re.search(r"(?m)^\s*pid\s*=\s*(\d+)\s*$", completed.stdout or "")
    return int(match.group(1)) if match else None


def _uv_archive_root(archive_path: str) -> Path | None:
    """Normalize a uv runtime identity path to its immutable archive root.

    ``runtime_identity()`` reports the Python library directory (for example,
    ``archive-v0/<id>/lib/python3.13``), while ``ps`` exposes entry points under
    ``archive-v0/<id>/bin``.  Bind both observations to the exact archive-id
    path instead of relying on a substring that can also match a sibling id.
    """

    if not archive_path:
        return None
    path = Path(archive_path).expanduser()
    indices = [index for index, part in enumerate(path.parts) if part == "archive-v0"]
    if not indices:
        return None
    index = indices[-1]
    if index + 1 >= len(path.parts):
        return None
    archive_id = path.parts[index + 1]
    if archive_id in {"", ".", ".."}:
        return None
    return Path(*path.parts[: index + 2])


def _command_uses_uv_archive(command: str, archive_root: Path) -> bool:
    """Return whether a command executes a bin entry point from one archive."""

    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    bin_root = archive_root / "bin"
    names: set[str] = set()
    for token in tokens:
        candidate = Path(token)
        if candidate.parent == bin_root:
            names.add(candidate.name)
    has_python = any(
        name == "python" or re.fullmatch(r"python3(?:\.\d+)?", name) for name in names
    )
    has_entrypoint = any(name.startswith("llm-wiki-") for name in names)
    return has_python and has_entrypoint


def _pid_tree_uses_archive(pid: int, archive_path: str) -> bool:
    """Confirm a service or one of its descendants executes from an archive."""

    archive_root = _uv_archive_root(archive_path)
    if archive_root is None:
        return False
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if completed.returncode != 0:
        return False
    rows: dict[int, tuple[int, str]] = {}
    for line in (completed.stdout or "").splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            child_pid = int(parts[0])
            parent_pid = int(parts[1])
        except ValueError:
            continue
        rows[child_pid] = (parent_pid, parts[2])
    descendants = {pid}
    changed = True
    while changed:
        changed = False
        for child_pid, (parent_pid, _command) in rows.items():
            if parent_pid in descendants and child_pid not in descendants:
                descendants.add(child_pid)
                changed = True
    return any(
        _command_uses_uv_archive(rows[process_pid][1], archive_root)
        for process_pid in descendants
        if process_pid in rows
    )


def _restart_persistent_runtime_services(archive_path: str) -> dict[str, Any]:
    """Restart only persistent services that were already running.

    Scheduled sleep/convergence/watchdog jobs are intentionally not kicked: a
    repair may be running inside one of them. Their wrappers use ``uvx
    --refresh`` and will naturally select the new archive on their next run.
    """

    launchctl = shutil.which("launchctl")
    if launchctl is None:
        return {"ok": False, "failure": "launchctl_unavailable", "services": []}
    timeout_seconds = max(
        5,
        int(os.environ.get("LLM_WIKI_FRONTIER_RESTART_TIMEOUT_SECONDS", "30")),
    )
    services: list[dict[str, Any]] = []
    failures: list[str] = []
    for label in PERSISTENT_RUNTIME_LABELS:
        old_pid = _launchd_pid(label)
        if old_pid is None:
            services.append({"label": label, "status": "not_running", "ok": True})
            continue
        restarted = subprocess.run(
            [launchctl, "kickstart", "-k", f"gui/{os.getuid()}/{label}"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if restarted.returncode != 0:
            failures.append(f"{label}:kickstart_failed")
            services.append(
                {
                    "label": label,
                    "status": "kickstart_failed",
                    "old_pid": old_pid,
                    "ok": False,
                }
            )
            continue
        deadline = time.monotonic() + timeout_seconds
        new_pid: int | None = None
        archive_loaded = False
        while time.monotonic() < deadline:
            new_pid = _launchd_pid(label)
            if new_pid is not None and new_pid != old_pid:
                archive_loaded = _pid_tree_uses_archive(new_pid, archive_path)
                if archive_loaded:
                    break
            time.sleep(0.25)
        ok = new_pid is not None and new_pid != old_pid and archive_loaded
        if not ok:
            failures.append(f"{label}:archive_verification_failed")
        services.append(
            {
                "label": label,
                "status": "restarted" if ok else "archive_verification_failed",
                "old_pid": old_pid,
                "new_pid": new_pid,
                "archive_loaded": archive_loaded,
                "ok": ok,
            }
        )
    return {"ok": not failures, "failures": failures, "services": services}


def _remote_main_sha(repo_root: Path) -> str | None:
    remote = _git_probe(
        repo_root,
        ["ls-remote", "--exit-code", "origin", "refs/heads/main"],
        timeout=120,
    )
    words = remote.stdout.split() if remote.returncode == 0 else []
    return words[0] if words else None


def _publish_verified_candidate(
    *,
    repo_root: Path,
    candidate_root: Path,
    candidate_head: str,
    baseline_head: str,
) -> dict[str, Any]:
    """Publish one tested descendant with an exact remote compare-and-swap."""

    checks: dict[str, Any] = {
        "remote_before": _remote_main_sha(repo_root),
        "candidate_head": candidate_head,
    }
    if checks["remote_before"] != baseline_head:
        return {"ok": False, "failure": "origin_main_changed_before_publish", **checks}
    tracked_status = _git_probe(
        repo_root,
        ["status", "--porcelain=v1", "--untracked-files=no"],
    )
    changed = _git_probe(
        candidate_root,
        ["diff", "--name-only", baseline_head, candidate_head],
    )
    untracked = _git_probe(
        repo_root,
        ["ls-files", "--others", "--exclude-standard"],
    )
    if tracked_status.returncode != 0 or tracked_status.stdout.strip():
        return {"ok": False, "failure": "production_checkout_changed", **checks}
    if changed.returncode != 0 or untracked.returncode != 0:
        return {"ok": False, "failure": "publish_path_preflight_failed", **checks}
    changed_paths = {path for path in changed.stdout.splitlines() if path}
    untracked_paths = {path for path in untracked.stdout.splitlines() if path}
    collisions = sorted(
        (candidate_path, local_path)
        for candidate_path in changed_paths
        for local_path in untracked_paths
        if candidate_path == local_path
        or candidate_path.startswith(f"{local_path}/")
        or local_path.startswith(f"{candidate_path}/")
    )
    checks["untracked_path_collisions"] = collisions
    if collisions:
        return {"ok": False, "failure": "untracked_path_collision", **checks}
    imported = _git_probe(
        repo_root,
        ["fetch", "--quiet", str(candidate_root), candidate_head],
        timeout=300,
    )
    if imported.returncode != 0:
        return {"ok": False, "failure": "candidate_import_failed", **checks}
    pushed = _git_probe(
        repo_root,
        [
            "push",
            f"--force-with-lease=refs/heads/main:{baseline_head}",
            "origin",
            f"{candidate_head}:refs/heads/main",
        ],
        timeout=300,
    )
    checks["push_returncode"] = pushed.returncode
    if pushed.returncode != 0:
        return {"ok": False, "failure": "guarded_push_failed", **checks}
    advanced = _git_probe(
        repo_root, ["merge", "--ff-only", candidate_head], timeout=120
    )
    checks["fast_forward_returncode"] = advanced.returncode
    checks["remote_after"] = _remote_main_sha(repo_root)
    if advanced.returncode != 0:
        return {"ok": False, "failure": "local_main_fast_forward_failed", **checks}
    if checks["remote_after"] != candidate_head:
        return {"ok": False, "failure": "origin_main_publish_mismatch", **checks}
    return {"ok": True, **checks}


def _verify_repair_result(
    result: FrontierResult,
    *,
    repo_root: Path,
    candidate_root: Path,
    evidence: Any,
    baseline: dict[str, Any],
) -> FrontierResult:
    """Verify an isolated candidate, then publish and verify the live archive."""

    timeout = max(
        60,
        int(os.environ.get("LLM_WIKI_FRONTIER_VERIFY_TIMEOUT_SECONDS", "1800")),
    )
    checks: dict[str, Any] = {
        "baseline_head": baseline.get("head"),
        "model_reported_commit": result.commit,
        "model_reported_tests": len(result.tests_run),
        "model_reported_pushed": result.pushed,
    }
    failures: list[str] = []
    if result.decision != "approved":
        return replace(
            result,
            verified=False,
            verification={"ok": False, "checks": checks, "failures": ["not_approved"]},
        )
    if not result.committed or result.pushed or not result.tests_run:
        failures.append("model_report_missing_required_completion_fields")

    try:
        head = _git_probe(candidate_root, ["rev-parse", "HEAD"])
        status = _git_probe(
            candidate_root,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        )
        ancestry = _git_probe(
            candidate_root,
            ["merge-base", "--is-ancestor", str(baseline.get("head") or ""), "HEAD"],
        )
        remotes = _git_probe(candidate_root, ["remote"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        failures.append(f"git_postcondition_{exc.__class__.__name__}")
        head = status = ancestry = remotes = subprocess.CompletedProcess([], 1, "", "")

    head_sha = head.stdout.strip() if head.returncode == 0 else ""
    checks.update(
        {
            "candidate_head": head_sha or None,
            "candidate_has_no_remote": remotes.returncode == 0
            and not remotes.stdout.strip(),
            "candidate_clean": status.returncode == 0 and not status.stdout.strip(),
            "baseline_is_ancestor": ancestry.returncode == 0,
        }
    )
    if not head_sha or head_sha == baseline.get("head"):
        failures.append("no_new_commit")
    if result.commit and not head_sha.startswith(result.commit):
        failures.append("reported_commit_mismatch")
    if status.returncode != 0 or status.stdout.strip():
        failures.append("repair_candidate_not_clean")
    if ancestry.returncode != 0:
        failures.append("repair_commit_not_descended_from_baseline")
    if remotes.returncode != 0 or remotes.stdout.strip():
        failures.append("repair_candidate_remote_present")
    if _remote_main_sha(repo_root) != baseline.get("head"):
        failures.append("origin_main_changed_during_candidate_verification")

    reproduction = list(getattr(evidence, "reproduction_command", ()) or ())
    if reproduction and not failures:
        reproduced = _verification_command(
            reproduction,
            repo_root=candidate_root,
            timeout=timeout,
        )
        checks["reproduction"] = {
            key: value for key, value in reproduced.items() if key != "stdout"
        }
        if not reproduced.get("ok"):
            failures.append("trusted_reproduction_still_fails")
    else:
        if not reproduction:
            failures.append("trusted_reproduction_command_missing")

    if not failures:
        suite = _verification_command(
            ["uv", "run", "pytest", "-q"],
            repo_root=candidate_root,
            timeout=timeout,
        )
        checks["full_test_suite"] = {
            key: value for key, value in suite.items() if key != "stdout"
        }
        if not suite.get("ok"):
            failures.append("full_test_suite_failed")

    published = False
    if not failures:
        publish = _publish_verified_candidate(
            repo_root=repo_root,
            candidate_root=candidate_root,
            candidate_head=head_sha,
            baseline_head=str(baseline.get("head") or ""),
        )
        checks["publish"] = publish
        published = bool(publish.get("ok"))
        if not published:
            failures.append(str(publish.get("failure") or "candidate_publish_failed"))

    runtime_identity_payload: dict[str, Any] = {}
    if published:
        runtime = _verification_command(
            [
                *uvx_runtime_command("llm-wiki", refresh=True),
                "runtime-identity",
                "--json",
            ],
            repo_root=repo_root,
            timeout=timeout,
        )
        if runtime.get("ok") and isinstance(runtime.get("stdout"), str):
            try:
                parsed = json.loads(runtime["stdout"])
                if isinstance(parsed, dict):
                    runtime_identity_payload = parsed
            except json.JSONDecodeError:
                pass
        checks["runtime_archive"] = {
            "ok": bool(runtime.get("ok")),
            "commit_id": runtime_identity_payload.get("commit_id"),
            "archive_path": runtime_identity_payload.get("archive_path"),
        }
        if runtime_identity_payload.get("commit_id") != head_sha:
            failures.append("refreshed_runtime_commit_mismatch")
        else:
            restart = _restart_persistent_runtime_services(
                str(runtime_identity_payload.get("archive_path") or "")
            )
            checks["runtime_restart"] = restart
            if not restart.get("ok"):
                failures.append("persistent_runtime_restart_failed")

    try:
        final_status = _git_probe(
            repo_root,
            ["status", "--porcelain=v1", "--untracked-files=no"],
        )
    except (OSError, subprocess.TimeoutExpired):
        final_status = subprocess.CompletedProcess([], 1, "", "")
    if final_status.returncode != 0 or final_status.stdout.strip():
        failures.append("verification_left_worktree_dirty")

    verification = {
        "ok": not failures,
        "checks": checks,
        "failures": failures,
    }
    if failures:
        failure = _frontier_failure(
            "frontier_postcondition_failed",
            "local_quarantined",
            "frontier repair self-report did not satisfy independent postconditions",
            human_required=False,
        )
        return replace(
            result,
            decision="quarantined",
            summary=failure.summary,
            frontier_failure=failure.to_dict(),
            rescue_status=failure.rescue_status,
            commit=head_sha or result.commit,
            pushed=published,
            verified=False,
            verification=verification,
        )
    return replace(
        result,
        commit=head_sha,
        pushed=True,
        verified=True,
        verification=verification,
    )


def _frontier_guard_outcome(result: FrontierResult) -> str:
    if result.human_required:
        return "human_required"
    if result.decision == "approved" and result.verified:
        return "succeeded"
    if result.decision == "quarantined":
        return "quarantined"
    return "failed"


def _frontier_guard_details(result: FrontierResult) -> dict[str, Any]:
    failure_class = None
    if isinstance(result.frontier_failure, dict):
        failure_class = result.frontier_failure.get("failure_class")
    return {
        "decision": result.decision,
        "summary": result.summary[:1000],
        "tests_run": list(result.tests_run)[:100],
        "commit": result.commit,
        "committed": result.committed,
        "pushed": result.pushed,
        "verified": result.verified,
        "verification": result.verification,
        "failure_class": failure_class,
    }


def run_frontier_review(
    packet: dict[str, Any],
    local_decision: dict[str, Any] | None,
    *,
    repo_root: Path,
    evidence: Any | None = None,
    guard: Any | None = None,
    execute_patch: bool = True,
    timeout: int | None = None,
) -> FrontierResult:
    """Run exactly one guarded frontier attempt for proven code repair.

    Routine callers cannot use this entry point without a strict
    :class:`RepairIncidentEvidence` capability.  Admission, single-flight,
    cooldown and the daily budget are durable across processes.
    """
    from llm_wiki_mcp.frontier_guard import (
        EvidenceValidationError,
        FrontierGuard,
        FrontierGuardError,
        PermitDenied,
        RepairIncidentEvidence,
    )
    from llm_wiki_mcp.decision_policy import resolve_decision_policy

    repair_policy, repair_mode, repair_policy_error = resolve_decision_policy(
        "system_code_repair"
    )
    if (
        repair_policy_error is not None
        or repair_policy is None
        or repair_policy.kind != "repair_only"
        or repair_mode != "enabled"
    ):
        failure = _frontier_failure(
            "frontier_repair_policy_disabled",
            "repair_deferred",
            repair_policy_error or "system code repair lane is disabled",
            human_required=False,
        )
        return _failure_result(summary=failure.summary, failure=failure)

    if not isinstance(evidence, RepairIncidentEvidence):
        failure = _frontier_failure(
            "frontier_guard_evidence_required",
            "local_quarantined",
            "frontier code repair requires validated system incident evidence",
            human_required=False,
        )
        return _failure_result(summary=failure.summary, failure=failure)

    # Validate again immediately before any baseline work, permit reservation,
    # or child process can occur.  The postcondition always executes this
    # command; admitting evidence without it would spend frontier tokens on a
    # result that is guaranteed to be rejected afterward.
    try:
        evidence.validate()
    except (EvidenceValidationError, TypeError, ValueError) as exc:
        failure = _frontier_failure(
            "frontier_guard_evidence_invalid",
            "local_quarantined",
            f"frontier code repair evidence is invalid: {exc}",
            human_required=False,
        )
        return _failure_result(summary=failure.summary, failure=failure)
    if not evidence.reproduction_command:
        failure = _frontier_failure(
            "frontier_guard_reproduction_command_required",
            "local_quarantined",
            "frontier code repair requires a trusted reproduction command before execution",
            human_required=False,
        )
        return _failure_result(summary=failure.summary, failure=failure)

    baseline = (
        _capture_repair_baseline(repo_root)
        if execute_patch
        else {
            "ok": True,
            "head": None,
            "origin_main": None,
            "clean": True,
            "review_only": True,
        }
    )
    if not baseline.get("ok"):
        failure = _frontier_failure(
            "frontier_repair_preflight_failed",
            "repair_deferred",
            f"frontier code repair requires a clean pushed main baseline: {baseline.get('failure_class')}",
            human_required=False,
        )
        return replace(
            _failure_result(summary=failure.summary, failure=failure),
            verification={"ok": False, "baseline": baseline},
        )

    timeout_seconds = _bounded_timeout(timeout)
    prompt = build_frontier_prompt(packet, local_decision, execute_patch=execute_patch)
    controller = guard or FrontierGuard()
    try:
        with controller.permit(evidence) as permit:
            try:
                if execute_patch:
                    with _isolated_repair_checkout(
                        repo_root, baseline
                    ) as candidate_root:
                        result = _run_codex(
                            prompt,
                            repo_root=candidate_root,
                            timeout=timeout_seconds,
                            execute_patch=True,
                            permit=permit,
                        )
                        execution_started = permit.status == "started"
                        if execution_started and result.decision == "approved":
                            result = _verify_repair_result(
                                result,
                                repo_root=repo_root,
                                candidate_root=candidate_root,
                                evidence=evidence,
                                baseline=baseline,
                            )
                else:
                    result = _run_codex(
                        prompt,
                        repo_root=repo_root,
                        timeout=timeout_seconds,
                        execute_patch=False,
                        permit=permit,
                    )
            except (subprocess.TimeoutExpired, OSError) as exc:
                failure = classify_frontier_failure(str(exc))
                result = _failure_result(
                    summary=f"frontier code repair failed to execute: {type(exc).__name__}",
                    output=str(exc),
                    failure=failure,
                )
            except RuntimeError as exc:
                failure = _frontier_failure(
                    "frontier_repair_isolation_failed",
                    "repair_deferred",
                    str(exc),
                    human_required=False,
                )
                result = _failure_result(summary=failure.summary, failure=failure)
            execution_started = permit.status == "started"
            if execution_started and not execute_patch:
                result = replace(
                    result,
                    verified=True,
                    verification={"ok": True, "mode": "review_only"},
                )
            if execution_started:
                permit.finish(
                    _frontier_guard_outcome(result),
                    details=_frontier_guard_details(result),
                )
            return replace(result, execution_started=execution_started)
    except PermitDenied as exc:
        failure = _frontier_failure(
            "frontier_guard_denied",
            "repair_deferred",
            f"frontier code repair deferred: {exc.reason}",
            human_required=False,
        )
        result = _failure_result(summary=failure.summary, failure=failure)
        return replace(
            result,
            rescue_attempt={
                "guard_reason": exc.reason,
                "retry_at": (
                    exc.retry_at.isoformat() if exc.retry_at is not None else None
                ),
            },
        )
    except FrontierGuardError as exc:
        failure = _frontier_failure(
            "frontier_guard_unavailable",
            "repair_deferred",
            f"frontier guard unavailable: {type(exc).__name__}",
            human_required=False,
        )
        return _failure_result(summary=failure.summary, failure=failure)
