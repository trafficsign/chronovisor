"""Frontier-model review and autonomous patch execution."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FRONTIER_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "summary", "tests_run", "committed", "pushed"],
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
        }


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
        return FrontierResult(
            decision="needs_retry",
            summary="frontier output did not contain JSON",
            tests_run=[],
            committed=False,
            pushed=False,
            raw_output=text[-4000:],
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
        return FrontierResult(
            decision="needs_retry",
            summary="frontier JSON failed schema validation",
            tests_run=[],
            committed=False,
            pushed=False,
            raw_output=text[-4000:],
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
        raw_output=text[-4000:],
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


def _run_custom_command(command: str, prompt: str, *, timeout: int) -> FrontierResult:
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
        return FrontierResult(
            decision="needs_retry",
            summary=f"frontier command failed with exit {completed.returncode}",
            tests_run=[],
            committed=False,
            pushed=False,
            raw_output=output[-4000:],
        )
    return _parse_result(output)


def _run_codex(prompt: str, *, repo_root: Path, timeout: int, execute_patch: bool) -> FrontierResult:
    codex = shutil.which("codex")
    if codex is None:
        return FrontierResult(
            decision="needs_retry",
            summary="codex executable not found",
            tests_run=[],
            committed=False,
            pushed=False,
        )

    with tempfile.TemporaryDirectory() as td:
        schema_path = Path(td) / "frontier-decision.schema.json"
        output_path = Path(td) / "frontier-output.json"
        schema_path.write_text(
            json.dumps(FRONTIER_DECISION_SCHEMA, indent=2) + "\n",
            encoding="utf-8",
        )
        cmd = [
            codex,
            "exec",
            "--cd",
            str(repo_root),
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
            "--skip-git-repo-check",
        ]
        model = os.environ.get("LLM_WIKI_FRONTIER_MODEL")
        if model:
            cmd.extend(["--model", model])
        if execute_patch:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        completed = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=_frontier_env(),
        )
        output_text = ""
        if output_path.exists():
            output_text += output_path.read_text(encoding="utf-8", errors="replace")
        output_text += "\n" + (completed.stdout or "") + "\n" + (completed.stderr or "")
        if completed.returncode != 0:
            return FrontierResult(
                decision="needs_retry",
                summary=f"codex exec failed with exit {completed.returncode}",
                tests_run=[],
                committed=False,
                pushed=False,
                raw_output=output_text[-4000:],
            )
        return _parse_result(output_text)


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
        return _run_custom_command(command, prompt, timeout=timeout_seconds)
    return _run_codex(
        prompt,
        repo_root=repo_root,
        timeout=timeout_seconds,
        execute_patch=execute_patch,
    )
