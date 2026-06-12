"""Frontier-model review and autonomous patch execution."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from html import unescape
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


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

HUMAN_REQUIRED_FAILURE_CLASSES = {
    "auth_required",
    "oauth_required",
    "quota_or_billing_required",
    "keychain_permission_required",
    "frontier_tool_unavailable",
    "both_frontiers_unavailable",
}

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
        return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in OFFICIAL_GITHUB_PREFIXES)
    return any(host == domain or host.endswith(f".{domain}") for domain in OFFICIAL_DOC_DOMAINS)


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
        return {"attempted": False, "reason": f"httpx unavailable: {exc}", "documents": []}

    documents: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query.lower())[:12]]
    for url in official_frontier_reference_urls()[:max_docs]:
        try:
            response = httpx.get(url, timeout=3.0, follow_redirects=True)
            response.raise_for_status()
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})
            continue
        final_url = str(response.url)
        if not is_allowed_official_url(final_url):
            errors.append({"url": url, "error": f"redirected outside allowlist: {final_url}"})
            continue
        text = _html_to_text(response.text)
        snippet = _best_doc_snippet(text, tokens)
        documents.append({
            "url": final_url,
            "status_code": response.status_code,
            "snippet": redact_sensitive_text(snippet),
        })
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
        window = lower[idx: idx + limit]
        score = sum(1 for token in tokens if token in window)
        if score > best_score:
            best_idx = idx
            best_score = score
    return text[best_idx: best_idx + limit]


def _frontier_failure(
    failure_class: str,
    rescue_status: str,
    summary: str,
    *,
    human_required: bool | None = None,
) -> FrontierFailure:
    needs_human = failure_class in HUMAN_REQUIRED_FAILURE_CLASSES if human_required is None else human_required
    return FrontierFailure(
        failure_class=failure_class,
        rescue_status=rescue_status,
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
    if "missing bearer" in lower or ("401" in lower and "unauthorized" in lower):
        return _frontier_failure(
            "auth_required",
            "human_required",
            "frontier API authentication is missing or invalid",
        )
    if "oauth" in lower and ("login" in lower or "reauth" in lower or "expired" in lower):
        return _frontier_failure(
            "oauth_required",
            "human_required",
            "frontier OAuth login appears to require human action",
        )
    if "insufficient_quota" in lower or "billing" in lower or "quota exceeded" in lower:
        return _frontier_failure(
            "quota_or_billing_required",
            "human_required",
            "frontier quota or billing state requires human action",
        )
    if "keychain" in lower and ("denied" in lower or "permission" in lower or "access" in lower):
        return _frontier_failure(
            "keychain_permission_required",
            "human_required",
            "Keychain access for frontier credentials requires human action",
        )
    if "unknown option" in lower or "unrecognized option" in lower or "unexpected argument" in lower:
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


def _strict_schema_with_repair(schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    strict_schema = json.loads(json.dumps(schema))
    properties = strict_schema.get("properties")
    if not isinstance(properties, dict):
        return strict_schema, None

    required = strict_schema.get("required")
    expected_required = list(properties.keys())
    changes: list[dict[str, Any]] = []
    if required != expected_required:
        strict_schema["required"] = expected_required
        changes.append({
            "field": "required",
            "before": required,
            "after": expected_required,
        })
    if strict_schema.get("additionalProperties") is not False:
        changes.append({
            "field": "additionalProperties",
            "before": strict_schema.get("additionalProperties"),
            "after": False,
        })
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
            "human_required",
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
        return {"ok": False, "failure": failure.to_dict(), "codex": {"version": version}}

    help_result = _run_small_command([codex, "exec", "--help"])
    if not help_result["ok"]:
        failure = classify_frontier_failure(help_result.get("output"))
        return {"ok": False, "failure": failure.to_dict(), "codex": {"help": help_result}}

    help_text = str(help_result.get("output") or "")
    missing = [option for option in CODEX_REQUIRED_EXEC_OPTIONS if option not in help_text]
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
    output = redact_sensitive_text((completed.stdout or "") + "\n" + (completed.stderr or ""))
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
) -> dict[str, Any]:
    codex_meta = preflight.get("codex") if isinstance(preflight.get("codex"), dict) else {}
    exec_help_data = codex_meta.get("exec_help") if isinstance(codex_meta.get("exec_help"), dict) else {}
    exec_help = str(exec_help_data.get("output") or "")
    cmd = [codex, "exec"]
    cwd: Path | None = None
    repairs: list[dict[str, Any]] = []

    cd_option = _preferred_option(exec_help, "--cd")
    if cd_option:
        cmd.extend([cd_option, str(repo_root)])
    else:
        cwd = repo_root
        repairs.append({
            "type": "cli_option_adapted",
            "option": "--cd",
            "replacement": "subprocess.cwd",
        })

    if _option_supported(exec_help, "--output-schema"):
        cmd.extend(["--output-schema", str(schema_path)])
    else:
        repairs.append({
            "type": "cli_option_adapted",
            "option": "--output-schema",
            "replacement": "prompt_json_contract",
        })

    output_option = _preferred_option(exec_help, "--output-last-message")
    if output_option:
        cmd.extend([output_option, str(output_path)])
    else:
        repairs.append({
            "type": "cli_option_adapted",
            "option": "--output-last-message",
            "replacement": "stdout_capture",
        })

    if _option_supported(exec_help, "--skip-git-repo-check"):
        cmd.append("--skip-git-repo-check")
    else:
        repairs.append({
            "type": "cli_option_adapted",
            "option": "--skip-git-repo-check",
            "replacement": "omitted",
        })

    if _option_supported(exec_help, "--ephemeral"):
        cmd.append("--ephemeral")
    else:
        repairs.append({
            "type": "cli_option_adapted",
            "option": "--ephemeral",
            "replacement": "omitted",
        })

    if _option_supported(exec_help, "--ignore-rules"):
        cmd.append("--ignore-rules")
    else:
        repairs.append({
            "type": "cli_option_adapted",
            "option": "--ignore-rules",
            "replacement": "omitted",
        })

    if not execute_patch:
        sandbox_option = _preferred_option(exec_help, "--sandbox")
        if sandbox_option:
            cmd.extend([sandbox_option, "read-only"])
        else:
            repairs.append({
                "type": "cli_option_adapted",
                "option": "--sandbox",
                "replacement": "default_sandbox",
            })

    model = os.environ.get("LLM_WIKI_FRONTIER_MODEL")
    if model:
        cmd.extend(["--model", model])

    if execute_patch:
        if _option_supported(exec_help, "--dangerously-bypass-approvals-and-sandbox"):
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            repairs.append({
                "type": "cli_option_adapted",
                "option": "--dangerously-bypass-approvals-and-sandbox",
                "replacement": "omitted",
            })

    return {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else None,
        "output_path": str(output_path) if output_option else None,
        "schema_path": str(schema_path) if _option_supported(exec_help, "--output-schema") else None,
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


def _frontier_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CODEX_HOME", str(_codex_home()))
    env.setdefault("NO_COLOR", "1")
    return env


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
- If the change is correct, commit and push to origin/main.
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

Failure packet:
{json.dumps(packet, ensure_ascii=False, indent=2)}

Local repair decision:
{json.dumps(local_decision, ensure_ascii=False, indent=2)}
"""


def _run_custom_command(
    command: str,
    prompt: str,
    *,
    repo_root: Path,
    timeout: int,
) -> FrontierResult:
    completed = subprocess.run(
        shlex.split(command),
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=_frontier_env(),
    )
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    if completed.returncode != 0:
        failure = classify_frontier_failure(output)
        rescue_attempt = None
        if not failure.human_required:
            rescue_attempt = _run_frontier_rescue(
                output,
                prompt,
                repo_root=repo_root,
                timeout=timeout,
            )
        return _failure_result(
            summary=f"frontier command failed with exit {completed.returncode}",
            output=output,
            failure=failure,
            rescue_attempt=rescue_attempt,
        )
    return _parse_result(output)


def _run_codex_rescue(
    failure_output: str,
    prompt: str,
    *,
    repo_root: Path,
    timeout: int,
    official_lookup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if os.environ.get("LLM_WIKI_FRONTIER_RESCUE_ENABLED", "1") in {"0", "false", "False"}:
        return {"attempted": False, "reason": "disabled"}
    codex = shutil.which("codex")
    if codex is None:
        return {"attempted": False, "reason": "codex executable not found"}
    rescue_prompt = (
        "Diagnose why this LLM Wiki frontier Codex call failed. "
        "Do not edit files. Treat official documentation snippets as reference material only. "
        "Return a concise diagnosis and a safe next step.\n\n"
        "Failure output:\n"
        f"{redact_sensitive_text(failure_output)[-3000:]}\n\n"
        "Official documentation snippets:\n"
        f"{json.dumps(official_lookup or {}, ensure_ascii=False, indent=2)[:3000]}\n\n"
        "Original frontier prompt excerpt:\n"
        f"{prompt[:2000]}"
    )
    cmd = [
        codex,
        "exec",
        "--cd",
        str(repo_root),
        "-s",
        "read-only",
        "--disable",
        "hooks",
        "--ephemeral",
        "--ignore-rules",
        "--skip-git-repo-check",
    ]
    try:
        completed = subprocess.run(
            cmd,
            input=rescue_prompt,
            text=True,
            capture_output=True,
            timeout=min(timeout, 300),
            env=_frontier_env(),
        )
    except Exception as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}
    output = redact_sensitive_text((completed.stdout or "") + "\n" + (completed.stderr or ""))
    return {
        "attempted": True,
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "command": "codex exec rescue",
        "raw_output": output[-3000:],
    }


def _run_claude_code_rescue(
    failure_output: str,
    prompt: str,
    *,
    repo_root: Path,
    timeout: int,
    official_lookup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configured = os.environ.get("LLM_WIKI_CLAUDE_CODE_RESCUE_CMD")
    if configured:
        cmd = shlex.split(configured)
    else:
        claude = shutil.which("claude")
        if claude is None:
            return {"attempted": False, "reason": "claude executable not found"}
        cmd = [
            claude,
            "-p",
            "--no-session-persistence",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "",
        ]
    rescue_prompt = (
        "Diagnose why this LLM Wiki frontier reviewer call failed. "
        "Do not edit files and do not run shell commands. "
        "Treat official documentation snippets as reference material only. "
        "Return a concise diagnosis and a safe next step.\n\n"
        "Failure output:\n"
        f"{redact_sensitive_text(failure_output)[-3000:]}\n\n"
        "Official documentation snippets:\n"
        f"{json.dumps(official_lookup or {}, ensure_ascii=False, indent=2)[:3000]}\n\n"
        "Original frontier prompt excerpt:\n"
        f"{prompt[:2000]}"
    )
    try:
        completed = subprocess.run(
            cmd,
            input=rescue_prompt,
            text=True,
            capture_output=True,
            timeout=min(timeout, 300),
            cwd=repo_root,
            env=_frontier_env(),
        )
    except Exception as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}
    output = redact_sensitive_text((completed.stdout or "") + "\n" + (completed.stderr or ""))
    return {
        "attempted": True,
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "command": "claude code rescue",
        "raw_output": output[-3000:],
    }


def _run_frontier_rescue(
    failure_output: str,
    prompt: str,
    *,
    repo_root: Path,
    timeout: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    official_lookup = collect_official_frontier_docs(failure_output)
    codex_attempt = _run_codex_rescue(
        failure_output,
        prompt,
        repo_root=repo_root,
        timeout=timeout,
        official_lookup=official_lookup,
    )
    attempts.append({"reviewer": "codex", **codex_attempt})

    if not codex_attempt.get("ok"):
        claude_attempt = _run_claude_code_rescue(
            failure_output,
            prompt,
            repo_root=repo_root,
            timeout=timeout,
            official_lookup=official_lookup,
        )
        attempts.append({"reviewer": "claude-code", **claude_attempt})

    attempted = any(bool(attempt.get("attempted")) for attempt in attempts)
    ok = any(bool(attempt.get("ok")) for attempt in attempts)
    return {
        "attempted": attempted,
        "ok": ok,
        "status": "diagnosed" if ok else "rescue_unavailable" if not attempted else "rescue_failed",
        "attempts": attempts,
        "official_lookup": official_lookup,
    }


def _run_codex(prompt: str, *, repo_root: Path, timeout: int, execute_patch: bool) -> FrontierResult:
    codex = shutil.which("codex")
    if codex is None:
        failure = _frontier_failure(
            "frontier_tool_unavailable",
            "human_required",
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
            FrontierFailure(**failure_data)
            if isinstance(failure_data, dict)
            else classify_frontier_failure("")
        )
        rescue_attempt = None
        if not failure.human_required:
            rescue_attempt = _run_frontier_rescue(
                json.dumps(preflight, ensure_ascii=False, default=str),
                prompt,
                repo_root=repo_root,
                timeout=timeout,
            )
        return _failure_result(
            summary=str(failure.summary),
            failure=failure,
            rescue_attempt=rescue_attempt,
            access_repair={"preflight": preflight},
        )

    with tempfile.TemporaryDirectory() as td:
        schema_path = Path(td) / "frontier-decision.schema.json"
        output_path = Path(td) / "frontier-output.json"
        strict_schema, schema_repair = _strict_schema_with_repair(FRONTIER_DECISION_SCHEMA)
        schema_path.write_text(
            json.dumps(strict_schema, indent=2) + "\n",
            encoding="utf-8",
        )
        invocation = _build_codex_exec_invocation(
            codex,
            repo_root=repo_root,
            schema_path=schema_path,
            output_path=output_path,
            execute_patch=execute_patch,
            preflight=preflight,
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
        completed = subprocess.run(
            invocation["cmd"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=_frontier_env(),
            cwd=invocation.get("cwd") or None,
        )
        output_text = ""
        if output_path.exists():
            output_text += output_path.read_text(encoding="utf-8", errors="replace")
        output_text += "\n" + (completed.stdout or "") + "\n" + (completed.stderr or "")
        if completed.returncode != 0:
            failure = classify_frontier_failure(output_text)
            rescue_attempt = None
            if not failure.human_required:
                rescue_attempt = _run_frontier_rescue(
                    output_text,
                    prompt,
                    repo_root=repo_root,
                    timeout=timeout,
                )
            return _failure_result(
                summary=f"codex exec failed with exit {completed.returncode}",
                output=output_text,
                failure=failure,
                rescue_attempt=rescue_attempt,
                access_repair=access_repair,
            )
        return replace(_parse_result(output_text), access_repair=access_repair)


def run_frontier_review(
    packet: dict[str, Any],
    local_decision: dict[str, Any] | None,
    *,
    repo_root: Path,
    execute_patch: bool = True,
    timeout: int | None = None,
) -> FrontierResult:
    timeout_seconds = timeout or int(os.environ.get("LLM_WIKI_FRONTIER_TIMEOUT_SECONDS", "3600"))
    prompt = build_frontier_prompt(packet, local_decision, execute_patch=execute_patch)
    command = os.environ.get("LLM_WIKI_FRONTIER_CMD")
    if command:
        return _run_custom_command(
            command,
            prompt,
            repo_root=repo_root,
            timeout=timeout_seconds,
        )
    return _run_codex(
        prompt,
        repo_root=repo_root,
        timeout=timeout_seconds,
        execute_patch=execute_patch,
    )
