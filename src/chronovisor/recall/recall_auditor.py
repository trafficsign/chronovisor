"""Asynchronous recall auditor for missed-candidate discovery.

This module runs after the normal recall gate. It inspects the user prompt,
assistant response, recall decision, and wiki top-k results, then records
``missed_candidate`` feedback when the configured runtime judges that the gate likely
missed useful memory. It never changes runtime recall decisions.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import time
import tomllib
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core import ollama, runtime_config
from chronovisor.core import store as chronovisor_store
from chronovisor.core.search import search as run_search
from chronovisor.decision.local_structured import ChatTransport, LocalStructuredSession
from chronovisor.recall.recall_runtime import (
    RECALL_CONFIG_FILE,
    RECALL_DIR,
    RECALL_LOG_FILE,
    RECALL_PULL_LOG_FILE,
    append_feedback,
    find_recall_log,
    recall_log_snapshot,
    stable_prompt_hash,
)

DEFAULT_STATE_FILE = RECALL_DIR / "audit-state.json"
DEFAULT_LOCK_FILE = RECALL_DIR / "audit.lock"
DEFAULT_PULL_CONSUMED_FILE = RECALL_DIR / "pull-consumed.jsonl"
HOOK_ENABLE_ENV = "CHRONOVISOR_RECALL_AUDIT_ENABLED"
AUDITOR_RUNTIME_ROLE = "recall.auditor"

AUTO_ACTIONS = frozenset({"alias", "query_hint", "page_tag"})
REVIEW_ACTIONS = frozenset({"few_shot", "threshold"})
ACTION_TYPES = AUTO_ACTIONS | REVIEW_ACTIONS | frozenset({"none"})
REASON_CODES = frozenset(
    {"gate_missed", "query_missed", "ranking_missed", "page_missing", "valid_skip"}
)

AUDITOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "missed",
        "confidence",
        "reason_code",
        "auditor_reason",
        "expected_pages",
        "missing_signal",
        "action_type",
    ],
    "properties": {
        "missed": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_code": {
            "type": "string",
            "enum": sorted(REASON_CODES),
        },
        "auditor_reason": {"type": "string"},
        "expected_pages": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "missing_signal": {"type": "string"},
        "action_type": {
            "type": "string",
            "enum": sorted(ACTION_TYPES),
        },
        "action_payload": {
            "type": "object",
            "additionalProperties": True,
        },
        "injection_usefulness": {
            "type": "string",
            "enum": ["used", "ignored", "unknown"],
        },
        "injection_reason": {"type": "string"},
    },
}


@dataclass(frozen=True)
class AuditPolicy:
    enabled: bool = True
    timeout_ms: int = 120_000
    num_ctx: int = runtime_config.DEFAULT_HEAVY_NUM_CTX
    keep_alive: str = runtime_config.DEFAULT_HEAVY_KEEP_ALIVE
    num_predict: int = 1024
    top_k: int = 5
    semantic: bool = False
    min_confidence: float = 0.70
    max_prompt_chars: int = 4000
    max_response_chars: int = 6000
    recent_log_limit: int = 500


@dataclass(frozen=True)
class TurnContext:
    host: str
    prompt: str
    assistant_response: str
    session_id: str = ""
    cwd: str = ""
    session_file: str = ""
    user_line: int = 0
    assistant_line: int = 0
    turn_id: str = ""

    @property
    def prompt_hash(self) -> str:
        return stable_prompt_hash(self.prompt)

    def turn_ref(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "prompt_hash": self.prompt_hash,
            "session_file": self.session_file,
            "user_line": self.user_line,
            "assistant_line": self.assistant_line,
        }


@dataclass(frozen=True)
class AuditDecision:
    missed: bool
    confidence: float
    reason_code: str
    auditor_reason: str
    expected_pages: list[str]
    missing_signal: str
    action_type: str
    action_payload: dict[str, Any]
    lane: str
    auto_apply_eligible: bool
    normalize_key: str
    injection_usefulness: str = "unknown"
    injection_reason: str = ""


def load_audit_policy(path: Path = RECALL_CONFIG_FILE) -> AuditPolicy:
    policy = AuditPolicy()
    path = runtime_config.active_config_file(path)
    if path.exists():
        try:
            data = runtime_config.normalize_audit_config(tomllib.loads(path.read_text()))
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        policy = _apply_config(policy, data)

    enabled_env = os.environ.get(HOOK_ENABLE_ENV)
    if enabled_env is not None:
        policy = AuditPolicy(**{**policy.__dict__, "enabled": enabled_env not in {"0", "false", "False", "no", "NO"}})
    return policy


def _apply_config(policy: AuditPolicy, data: dict[str, Any]) -> AuditPolicy:
    values = dict(policy.__dict__)

    budgets = data.get("budgets", {})
    if isinstance(budgets, dict) and isinstance(budgets.get("auditor_timeout_ms"), int):
        values["timeout_ms"] = max(1000, budgets["auditor_timeout_ms"])

    auditor = data.get("auditor", {})
    if isinstance(auditor, dict):
        if isinstance(auditor.get("enabled"), bool):
            values["enabled"] = auditor["enabled"]
        if isinstance(auditor.get("timeout_ms"), int):
            values["timeout_ms"] = max(1000, auditor["timeout_ms"])
        if isinstance(auditor.get("num_ctx"), int):
            values["num_ctx"] = max(2048, auditor["num_ctx"])
        if isinstance(auditor.get("keep_alive"), str) and auditor["keep_alive"].strip():
            values["keep_alive"] = auditor["keep_alive"].strip()
        if isinstance(auditor.get("num_predict"), int):
            values["num_predict"] = max(128, auditor["num_predict"])
        if isinstance(auditor.get("top_k"), int):
            values["top_k"] = max(1, min(20, auditor["top_k"]))
        if isinstance(auditor.get("semantic"), bool):
            values["semantic"] = auditor["semantic"]
        if isinstance(auditor.get("min_confidence"), (int, float)):
            values["min_confidence"] = max(0.0, min(1.0, float(auditor["min_confidence"])))
        if isinstance(auditor.get("max_prompt_chars"), int):
            values["max_prompt_chars"] = max(500, auditor["max_prompt_chars"])
        if isinstance(auditor.get("max_response_chars"), int):
            values["max_response_chars"] = max(500, auditor["max_response_chars"])
        if isinstance(auditor.get("recent_log_limit"), int):
            values["recent_log_limit"] = max(50, auditor["recent_log_limit"])

    return AuditPolicy(**values)


def action_lane(action_type: str) -> tuple[str, bool]:
    if action_type in AUTO_ACTIONS:
        return "auto", True
    return "review", False


def normalize_action_type(value: Any) -> str:
    if isinstance(value, str) and value in ACTION_TYPES:
        return value
    return "none"


def normalize_reason_code(value: Any, *, missed: bool) -> str:
    if isinstance(value, str) and value in REASON_CODES:
        return value
    return "gate_missed" if missed else "valid_skip"


def normalize_token(value: str, *, fallback: str = "none", limit: int = 64) -> str:
    text = value.strip().casefold()
    text = text.replace("_", "-")
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff-]+", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-_")
    return (text or fallback)[:limit]


def build_normalize_key(reason_code: str, expected_pages: list[str], missing_signal: str) -> str:
    page = expected_pages[0] if expected_pages else "no-page"
    signal = normalize_token(missing_signal or reason_code, fallback=reason_code)
    return f"{reason_code}:{signal}:{normalize_token(page, fallback='no-page')}"


def parse_auditor_output(output: str, top_pages: list[dict[str, Any]]) -> AuditDecision:
    parsed = extract_json_object(output)
    if not isinstance(parsed, dict):
        raise ValueError("auditor did not return a JSON object")

    missed = bool(parsed.get("missed"))
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason_code = normalize_reason_code(parsed.get("reason_code"), missed=missed)
    action_type = normalize_action_type(parsed.get("action_type"))
    payload = parsed.get("action_payload")
    action_payload = payload if isinstance(payload, dict) else {}
    if not missed:
        reason_code = "valid_skip"
        action_type = "none"
        action_payload = {}

    top_page_ids = [str(page.get("page_id", "")) for page in top_pages if page.get("page_id")]
    raw_expected = parsed.get("expected_pages")
    expected_pages = [
        value
        for value in raw_expected
        if isinstance(value, str) and (not top_page_ids or value in top_page_ids)
    ] if isinstance(raw_expected, list) else []
    if missed and not expected_pages and top_page_ids and reason_code != "page_missing":
        expected_pages = top_page_ids[:1]

    missing_signal = parsed.get("missing_signal")
    if not isinstance(missing_signal, str) or not missing_signal.strip():
        missing_signal = reason_code

    lane, auto_apply_eligible = action_lane(action_type)
    return AuditDecision(
        missed=missed,
        confidence=confidence,
        reason_code=reason_code,
        auditor_reason=str(parsed.get("auditor_reason", "")).strip(),
        expected_pages=expected_pages,
        missing_signal=missing_signal.strip(),
        action_type=action_type,
        action_payload=action_payload,
        lane=lane,
        auto_apply_eligible=auto_apply_eligible,
        normalize_key=build_normalize_key(reason_code, expected_pages, missing_signal),
        injection_usefulness=str(parsed.get("injection_usefulness", "unknown"))
        if parsed.get("injection_usefulness") in {"used", "ignored", "unknown"}
        else "unknown",
        injection_reason=str(parsed.get("injection_reason", "")).strip(),
    )


def extract_json_object(output: str) -> Any:
    text = output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def collect_top_pages(prompt: str, policy: AuditPolicy) -> tuple[list[dict[str, Any]], str]:
    try:
        results, mode = run_search(query=prompt, top_n=policy.top_k, semantic=policy.semantic)
    except Exception as exc:
        return [], f"error:{exc.__class__.__name__}"
    pages = [
        {
            "page_id": page.page_id,
            "title": page.title,
            "updated": page.updated,
            "score": page.score,
            "snippet": trim_text(page.snippet, 500),
        }
        for page in results
    ]
    return pages, mode


def build_auditor_prompt(
    turn: TurnContext,
    recall_snapshot: dict[str, Any] | None,
    top_pages: list[dict[str, Any]],
    policy: AuditPolicy,
) -> str:
    metadata = {
        "turn_ref": turn.turn_ref(),
        "host": turn.host,
        "cwd": turn.cwd,
        "recall_snapshot": recall_snapshot or {},
        "wiki_top_pages": top_pages,
    }
    return (
        "You are the asynchronous recall auditor for Chronovisor.\n"
        "Your job is to detect false negatives after the assistant response is available.\n"
        "The synchronous gate only saw the user prompt; you can also inspect the assistant response.\n\n"
        "Decide whether this turn should be recorded as a missed_candidate. A missed candidate means:\n"
        "- the recall gate decision was none/search or no useful page was injected, and\n"
        "- the user prompt or assistant response shows that past/project wiki memory was actually useful.\n\n"
        "Do not propose runtime threshold changes unless the evidence is systemic. Prefer additive, local actions.\n"
        "Safe direct action types are alias, query_hint, and page_tag. System-wide action types few_shot "
        "and threshold are routed to automated replay, holdout, and frontier gates. Use action_type=none "
        "when no improvement action is clear.\n\n"
        "Use page_tag only when action_payload.tag is an actual tag that satisfies the taxonomy form "
        "d/<kebab>, t/<kebab>, or s/<kebab>. Do not put a prose reason in action_payload.tag; "
        "use query_hint for prose evidence. Use alias only for ASCII page-id aliases, not natural-language "
        "queries or JSON-shaped suggestions; use query_hint for those.\n\n"
        "Also judge precision when recall_snapshot.pages is non-empty: set injection_usefulness to used, ignored, "
        "or unknown, and explain briefly in injection_reason.\n\n"
        "Return JSON only with this schema:\n"
        f"{json.dumps(AUDITOR_SCHEMA, ensure_ascii=False)}\n\n"
        "Metadata:\n"
        f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        "User prompt:\n"
        f"{trim_text(turn.prompt, policy.max_prompt_chars)}\n\n"
        "Assistant response:\n"
        f"{trim_text(turn.assistant_response, policy.max_response_chars)}\n"
    )


def _auditor_runtime_route(
    route: ollama.RuntimeGenerationRoute | None = None,
) -> ollama.RuntimeGenerationRoute:
    route = route or ollama.runtime_generation_routes((AUDITOR_RUNTIME_ROLE,))[0]
    if route.role != AUDITOR_RUNTIME_ROLE:
        raise ollama.RuntimeBridgeError("route_configuration_invalid")
    if not route.structured_output:
        raise ollama.RuntimeBridgeError("capability_unavailable")
    return route


def _auditor_route_identity(
    route: ollama.RuntimeGenerationRoute,
) -> dict[str, str]:
    return {
        "role": route.role,
        "provider": route.provider,
        "model": route.model,
        "location": route.location,
    }


def run_auditor_judge(
    turn: TurnContext,
    recall_snapshot: dict[str, Any] | None,
    top_pages: list[dict[str, Any]],
    policy: AuditPolicy,
    *,
    transport: ChatTransport | None = None,
    audit_root: Path | None = None,
    resolved_route: ollama.RuntimeGenerationRoute | None = None,
) -> str:
    """Run the asynchronous auditor with bounded same-session JSON repair."""

    if transport is None:
        route = _auditor_runtime_route(resolved_route)
        model = route.model
    else:
        route = None
        model = "injected:recall-auditor"
    result = LocalStructuredSession(
        model=model,
        transport=transport,
        role="recall_auditor",
        runtime_role=AUDITOR_RUNTIME_ROLE,
        runtime_location=None if route is None else route.location,
        source_data_class="raw",
        source_sensitivity="high",
        audit_root=audit_root,
        num_ctx=policy.num_ctx,
        num_predict=policy.num_predict,
        keep_alive=policy.keep_alive,
        read_timeout_ms=policy.timeout_ms,
        max_input_chars=65_536,
        max_output_chars=8_000,
        max_feedback_chars=2_000,
    ).run(
        build_auditor_prompt(turn, recall_snapshot, top_pages, policy),
        AUDITOR_SCHEMA,
    )
    if not result.ok:
        reason = result.failure_class or "structured_session_failed"
        detail = result.failure_reason or "recall auditor did not converge"
        raise ValueError(f"recall auditor failed: {reason}: {detail}")
    if not isinstance(result.value, dict):
        raise ValueError("recall auditor output is not an object")
    return json.dumps(result.value, ensure_ascii=False, separators=(",", ":"))


def acquire_audit_lock(path: Path) -> Any | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    except OSError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    handle.flush()
    return handle


def release_audit_lock(handle: Any | None) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=limit)
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def find_matching_recall_log(turn: TurnContext, *, host: str = "", limit: int = 500) -> dict[str, Any] | None:
    target_hash = turn.prompt_hash
    target_preview = turn.prompt[:300]
    candidates: list[tuple[int, dict[str, Any]]] = []
    for record in read_jsonl_tail(RECALL_LOG_FILE, limit):
        score = 0
        if record.get("prompt_hash") == target_hash:
            score += 5
        preview = record.get("prompt_preview")
        if isinstance(preview, str) and (preview == target_preview or turn.prompt.startswith(preview)):
            score += 3
        if turn.session_id and record.get("session_id") == turn.session_id:
            score += 2
        if host and record.get("host") == host:
            score += 1
        if score >= 5:
            candidates.append((score, record))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def turn_id_for(session_id: str, user_line: int, assistant_line: int, prompt: str) -> str:
    base = f"{session_id}:{user_line}:{assistant_line}:{stable_prompt_hash(prompt)}"
    return stable_prompt_hash(base)[:12]


def latest_complete_turn(records: list[Any], *, host: str, session_file: Path, session_id: str, cwd: str) -> TurnContext | None:
    pending_user: Any | None = None
    latest: TurnContext | None = None
    for record in records:
        if getattr(record, "role", "") == "user":
            pending_user = record
        elif getattr(record, "role", "") == "assistant" and pending_user is not None:
            prompt = getattr(pending_user, "text", "")
            response = getattr(record, "text", "")
            user_line = int(getattr(pending_user, "line", 0) or 0)
            assistant_line = int(getattr(record, "line", 0) or 0)
            latest = TurnContext(
                host=host,
                prompt=prompt,
                assistant_response=response,
                session_id=session_id,
                cwd=cwd,
                session_file=str(session_file),
                user_line=user_line,
                assistant_line=assistant_line,
                turn_id=turn_id_for(session_id, user_line, assistant_line, prompt),
            )
            pending_user = None
    return latest


def read_hook_payload(stdin_text: str | None) -> dict[str, Any]:
    if not stdin_text:
        return {}
    try:
        parsed = json.loads(stdin_text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def hook_hints_for_host(host: str, payload: dict[str, Any]) -> dict[str, str]:
    if host == "codex":
        from chronovisor.core.codex_transcript import hook_hints

        return hook_hints(payload)
    if host == "claude-code":
        from chronovisor.core.claude_code_transcript import hook_hints

        return hook_hints(payload)
    hints: dict[str, str] = {}
    for key in ("session_id", "sessionId", "conversation_id", "thread_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            hints["session_id"] = value
            break
    for key in ("transcript_path", "transcriptPath", "session_file", "sessionFile", "path"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            hints["session_file"] = value
            break
    for key in ("cwd", "working_directory"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            hints["cwd"] = value
            break
    return hints


def resolve_session_file(args: argparse.Namespace, hints: dict[str, str]) -> Path:
    if args.session_file:
        return Path(args.session_file).expanduser()
    if hints.get("session_file"):
        return Path(hints["session_file"]).expanduser()

    session_id = args.session_id or hints.get("session_id")
    cwd = args.cwd or hints.get("cwd") or os.environ.get("PWD", "")
    if args.host == "codex":
        from chronovisor.core.codex_transcript import find_session_file

        return find_session_file(
            session_id=session_id,
            cwd=cwd,
            sessions_root=Path(args.sessions_root).expanduser() if args.sessions_root else None,
        )
    if args.host == "claude-code":
        from chronovisor.core.claude_code_transcript import find_session_file

        return find_session_file(session_id=session_id, transcript_path=None)
    raise ValueError("--session-file is required for generic host transcript auditing")


def extract_turn_from_session(args: argparse.Namespace, hints: dict[str, str], state: dict[str, Any]) -> tuple[TurnContext | None, Path, int]:
    session_file = resolve_session_file(args, hints)
    after_line = 0 if args.ignore_state else saved_line_for(state, session_file)
    if args.host == "codex":
        from chronovisor.core.codex_transcript import extract_transcript_slice
    elif args.host == "claude-code":
        from chronovisor.core.claude_code_transcript import extract_transcript_slice
    else:
        raise ValueError("generic host requires --prompt and --assistant-response")

    transcript_slice = extract_transcript_slice(session_file, after_line=after_line)
    turn = latest_complete_turn(
        transcript_slice.records,
        host=args.host,
        session_file=session_file,
        session_id=transcript_slice.session_id or hints.get("session_id", ""),
        cwd=transcript_slice.cwd or hints.get("cwd", ""),
    )
    return turn, session_file, transcript_slice.scanned_until_line


def direct_turn(args: argparse.Namespace) -> TurnContext:
    prompt = args.prompt or ""
    response = args.assistant_response or ""
    return TurnContext(
        host=args.host,
        prompt=prompt,
        assistant_response=response,
        session_id=args.session_id or "",
        cwd=args.cwd or "",
        user_line=0,
        assistant_line=0,
        turn_id=turn_id_for(args.session_id or "", 0, 0, prompt),
    )


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "files": {}}
    try:
        parsed = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"version": 1, "files": {}}
    if not isinstance(parsed, dict):
        return {"version": 1, "files": {}}
    parsed.setdefault("version", 1)
    parsed.setdefault("files", {})
    if not isinstance(parsed["files"], dict):
        parsed["files"] = {}
    return parsed


def saved_line_for(state: dict[str, Any], session_file: Path) -> int:
    entry = state.get("files", {}).get(str(session_file))
    if not isinstance(entry, dict):
        return 0
    value = entry.get("last_audited_line", 0)
    return value if isinstance(value, int) and value > 0 else 0


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def update_state(state: dict[str, Any], *, session_file: Path, scanned_until_line: int, status: str) -> None:
    files = state.setdefault("files", {})
    files[str(session_file)] = {
        "last_audited_line": scanned_until_line,
        "status": status,
        "updated_at": datetime.now(UTC).isoformat(),
    }


def should_skip_recall_snapshot(snapshot: dict[str, Any] | None, *, audit_read: bool) -> str:
    if not snapshot:
        return ""
    decision = snapshot.get("decision")
    if decision == "read" and not audit_read:
        return "recall decision was already read"
    return ""


def feedback_extra(
    *,
    decision: AuditDecision,
    turn: TurnContext,
    top_pages: list[dict[str, Any]],
    search_mode: str,
    route_identity: dict[str, str] | None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "source": "auditor",
        "turn_ref": turn.turn_ref(),
        "auditor_confidence": decision.confidence,
        "auditor_reason": decision.auditor_reason,
        "reason_code": decision.reason_code,
        "missing_signal": decision.missing_signal,
        "normalize_key": decision.normalize_key,
        "action_type": decision.action_type,
        "action_payload": decision.action_payload,
        "lane": decision.lane,
        "auto_apply_eligible": decision.auto_apply_eligible,
        "top_pages": top_pages,
        "search_mode": search_mode,
        "assistant_response_preview": trim_text(turn.assistant_response, 1000),
    }
    if route_identity is not None:
        extra["auditor_model"] = route_identity["model"]
        extra["auditor_route_identity"] = route_identity
    return extra


def pull_event_key(record: dict[str, Any]) -> str:
    identity = {
        key: record.get(key)
        for key in (
            "ts",
            "session_id",
            "decision_id",
            "type",
            "stage",
            "page_id",
            "page_ids",
            "query",
            "direct_pages",
            "expanded_pages",
        )
    }
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _consumed_pull_keys(path: Path = DEFAULT_PULL_CONSUMED_FILE) -> set[str]:
    keys: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = row.get("event_key") if isinstance(row, dict) else None
                if isinstance(key, str) and key:
                    keys.add(key)
    except OSError:
        return set()
    return keys


def _feedback_pull_keys(path: Path | None = None) -> set[str]:
    """Treat feedback as the commit record for pull-event consumption.

    The consumed ledger is an index, not the source of truth. If a process
    exits after appending feedback but before appending the ledger row, this
    scan prevents duplicate feedback and lets the ledger heal on demand.
    """
    if path is None:
        from chronovisor.recall import recall_runtime

        path = recall_runtime.RECALL_FEEDBACK_FILE
    keys: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = row.get("pull_event_key") if isinstance(row, dict) else None
                if isinstance(key, str) and key:
                    keys.add(key)
    except OSError:
        return set()
    return keys


def _normalized_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed.astimezone(UTC)


def _next_recall_time(session_id: str, after: datetime | None) -> datetime | None:
    if not session_id or after is None:
        return None
    next_time: datetime | None = None
    try:
        with RECALL_LOG_FILE.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get("session_id") != session_id:
                    continue
                timestamp = _normalized_time(row.get("ts"))
                if timestamp is not None and timestamp > after and (
                    next_time is None or timestamp < next_time
                ):
                    next_time = timestamp
    except OSError:
        return None
    return next_time


def matching_pull_events(
    turn: TurnContext,
    recall_snapshot: dict[str, Any] | None,
    *,
    limit: int = 200,
    consumed_file: Path = DEFAULT_PULL_CONSUMED_FILE,
    feedback_file: Path | None = None,
) -> list[dict[str, Any]]:
    # Pull events are global telemetry. Exact decision binding is preferred;
    # legacy rows require a stable session identity and the timestamp window.
    recall_decision_id = str((recall_snapshot or {}).get("decision_id") or "")
    if not turn.session_id and not recall_decision_id:
        return []
    recall_ts = str((recall_snapshot or {}).get("ts") or "")
    recall_time = _normalized_time(recall_ts)
    if recall_time is None:
        return []
    try:
        with RECALL_PULL_LOG_FILE.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=limit)
    except OSError:
        return []
    injected = set((recall_snapshot or {}).get("pages", []) or [])
    consumed = _consumed_pull_keys(consumed_file) | _feedback_pull_keys(feedback_file)
    turn_end = _next_recall_time(turn.session_id, recall_time)
    out: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        record_decision_id = str(record.get("decision_id") or "")
        record_session_id = str(record.get("session_id") or "")
        if record_decision_id:
            if not recall_decision_id or record_decision_id != recall_decision_id:
                continue
            if record_session_id and turn.session_id and record_session_id != turn.session_id:
                continue
        elif not turn.session_id or record_session_id != turn.session_id:
            continue
        event_key = pull_event_key(record)
        if event_key in consumed:
            continue
        event_ts = str(record.get("ts") or "")
        event_time = _normalized_time(event_ts)
        if event_time is None:
            continue
        if event_time <= recall_time:
            continue
        if turn_end is not None and event_time is not None and event_time >= turn_end:
            continue
        # returned/read are exploration telemetry, not positive labels. Only
        # an explicit used event may teach the missed-candidate lane.
        if record.get("type") != "used":
            continue
        values = record.get("page_ids")
        pages = (
            [page for page in values if isinstance(page, str)]
            if isinstance(values, list)
            else []
        )
        missed_pages = [page for page in pages if page and page not in injected]
        if missed_pages:
            event = dict(record)
            event["event_key"] = event_key
            event["missed_pages"] = list(dict.fromkeys(missed_pages))
            out.append(event)
    return out[:5]


def record_pull_missed_candidates(
    *,
    turn: TurnContext,
    recall_snapshot: dict[str, Any] | None,
    pull_events: list[dict[str, Any]],
    host: str,
    consumed_file: Path = DEFAULT_PULL_CONSUMED_FILE,
) -> list[dict[str, Any]]:
    if not pull_events:
        return []
    recorded: list[dict[str, Any]] = []
    committed = _feedback_pull_keys()
    indexed = _consumed_pull_keys(consumed_file)
    for event in pull_events:
        pages = event.get("missed_pages")
        if not isinstance(pages, list) or not pages:
            continue
        event_key = str(event.get("event_key") or pull_event_key(event))
        if event_key in committed:
            if event_key not in indexed:
                consumed_file.parent.mkdir(parents=True, exist_ok=True)
                with consumed_file.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "ts": datetime.now().isoformat(timespec="seconds"),
                                "event_key": event_key,
                                "turn_ref": turn.turn_ref(),
                                "recovered_from_feedback": True,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                indexed.add(event_key)
            continue
        record = append_feedback(
            "missed_candidate",
            note="Agent reported wiki context as materially used after recall did not inject it.",
            prompt=turn.prompt,
            host=host,
            expected_pages=[page for page in pages if isinstance(page, str)][:5],
            expected_queries=[event.get("query")] if isinstance(event.get("query"), str) else [],
            ref=recall_snapshot.get("decision_id", "") if recall_snapshot else "",
            extra={
                "source": "pull-log",
                "pull_event_key": event_key,
                "turn_ref": turn.turn_ref(),
                "pull_event": event,
                "reason_code": "gate_missed",
                "missing_signal": event.get("query", "pull-log"),
                "normalize_key": build_normalize_key(
                    "gate_missed",
                    [page for page in pages if isinstance(page, str)],
                    str(event.get("query", "pull-log")),
                ),
                "action_type": "query_hint",
                "action_payload": {
                    "query": event.get("query", ""),
                    "page_id": pages[0],
                },
                "lane": "auto",
                "auto_apply_eligible": True,
            },
        )
        recorded.append(record)
        committed.add(event_key)
        consumed_file.parent.mkdir(parents=True, exist_ok=True)
        with consumed_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event_key": event_key,
                        "turn_ref": turn.turn_ref(),
                        "feedback_ref": record.get("ref", "") if isinstance(record, dict) else "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        indexed.add(event_key)
    return recorded


def run(args: argparse.Namespace, *, stdin_text: str | None = None) -> dict[str, Any]:
    policy = load_audit_policy(Path(args.config).expanduser())
    if args.top_k is not None:
        policy = AuditPolicy(**{**policy.__dict__, "top_k": max(1, min(20, args.top_k))})
    if args.min_confidence is not None:
        policy = AuditPolicy(**{**policy.__dict__, "min_confidence": max(0.0, min(1.0, args.min_confidence))})

    if not policy.enabled and not args.force:
        return {"status": "disabled", "reason": "recall auditor is disabled"}

    state_file = Path(args.state_file).expanduser()
    state = load_state(state_file)
    session_file: Path | None = None
    scanned_until_line = 0

    if args.prompt is not None:
        turn = direct_turn(args)
    else:
        payload = read_hook_payload(stdin_text) if args.hook else {}
        hints = hook_hints_for_host(args.host, payload)
        turn, session_file, scanned_until_line = extract_turn_from_session(args, hints, state)
        if turn is None:
            return {"status": "skipped", "reason": "no complete user/assistant turn"}

    if not turn.prompt or not turn.assistant_response:
        return {"status": "skipped", "reason": "prompt and assistant response are required"}

    recall_record = find_recall_log(args.decision_id) if args.decision_id else None
    if recall_record is None and args.decision_id:
        return {"status": "skipped", "reason": f"recall decision not found: {args.decision_id}"}
    if recall_record is None:
        recall_record = find_matching_recall_log(turn, host=args.host, limit=policy.recent_log_limit)

    recall_snapshot = recall_log_snapshot(recall_record) if recall_record else None
    lock_handle = None
    if not args.auditor_json:
        lock_handle = acquire_audit_lock(Path(args.lock_file).expanduser())
        if lock_handle is None:
            return {
                "status": "skipped",
                "reason": "another recall audit is already running",
                "turn_ref": turn.turn_ref(),
                "ref": recall_snapshot.get("decision_id", "") if recall_snapshot else "",
            }
    skip_reason = should_skip_recall_snapshot(recall_snapshot, audit_read=args.audit_read)
    if skip_reason:
        release_audit_lock(lock_handle)
        if session_file and not args.dry_run and not args.ignore_state:
            update_state(state, session_file=session_file, scanned_until_line=scanned_until_line, status="skipped")
            write_state(state_file, state)
        return {"status": "skipped", "reason": skip_reason, "turn_ref": turn.turn_ref()}

    top_pages, search_mode = collect_top_pages(turn.prompt, policy)
    if args.extract_only:
        release_audit_lock(lock_handle)
        return {
            "status": "extracted",
            "turn_ref": turn.turn_ref(),
            "recall_snapshot": recall_snapshot,
            "top_pages": top_pages,
            "search_mode": search_mode,
        }

    pull_feedback: list[dict[str, Any]] = []
    pull_events = matching_pull_events(turn, recall_snapshot)
    if pull_events and not args.dry_run:
        pull_feedback = record_pull_missed_candidates(
            turn=turn,
            recall_snapshot=recall_snapshot,
            pull_events=pull_events,
            host=args.host,
        )

    started = time.monotonic()
    auditor_route: ollama.RuntimeGenerationRoute | None = None
    if args.auditor_json:
        raw_output = args.auditor_json
    else:
        try:
            auditor_route = _auditor_runtime_route()
            raw_output = run_auditor_judge(
                turn,
                recall_snapshot,
                top_pages,
                policy,
                resolved_route=auditor_route,
            )
        finally:
            release_audit_lock(lock_handle)
            lock_handle = None
    decision = parse_auditor_output(raw_output, top_pages)
    latency_ms = int((time.monotonic() - started) * 1000)
    route_identity = (
        _auditor_route_identity(auditor_route)
        if auditor_route is not None
        else None
    )

    auditor: dict[str, Any] = {
        "missed": decision.missed,
        "confidence": decision.confidence,
        "reason_code": decision.reason_code,
        "action_type": decision.action_type,
        "lane": decision.lane,
        "auto_apply_eligible": decision.auto_apply_eligible,
        "normalize_key": decision.normalize_key,
        "action_payload": decision.action_payload,
        "latency_ms": latency_ms,
        "injection_usefulness": decision.injection_usefulness,
        "injection_reason": decision.injection_reason,
    }
    if route_identity is not None:
        auditor["route_identity"] = route_identity
    base = {
        "turn_ref": turn.turn_ref(),
        "ref": recall_snapshot.get("decision_id", "") if recall_snapshot else "",
        "auditor": auditor,
    }
    if pull_feedback:
        base["pull_feedback"] = pull_feedback

    precision_record: dict[str, Any] | None = None
    injected_pages = (recall_snapshot or {}).get("pages", [])
    if (
        isinstance(injected_pages, list)
        and injected_pages
        and decision.injection_usefulness in {"used", "ignored"}
        and not args.dry_run
    ):
        precision_record = append_feedback(
            "injection_used" if decision.injection_usefulness == "used" else "injection_ignored",
            note=decision.injection_reason,
            prompt=turn.prompt,
            host=args.host,
            expected_pages=[page for page in injected_pages if isinstance(page, str)],
            expected_queries=[],
            ref=recall_snapshot.get("decision_id", "") if recall_snapshot else "",
            extra={
                "source": "auditor_precision",
                "turn_ref": turn.turn_ref(),
                "auditor_confidence": decision.confidence,
                "assistant_response_preview": trim_text(turn.assistant_response, 1000),
                **(
                    {
                        "auditor_model": route_identity["model"],
                        "auditor_route_identity": route_identity,
                    }
                    if route_identity is not None
                    else {}
                ),
            },
        )
        base["precision_feedback"] = precision_record

    if not decision.missed and (precision_record or pull_feedback):
        if session_file and not args.dry_run and not args.ignore_state:
            update_state(state, session_file=session_file, scanned_until_line=scanned_until_line, status="recorded")
            write_state(state_file, state)
        return {**base, "status": "recorded"}
    if not decision.missed:
        status = "skipped"
        reason = "auditor did not classify this as missed"
    elif decision.confidence < policy.min_confidence:
        status = "skipped"
        reason = f"auditor confidence below threshold ({decision.confidence:.2f} < {policy.min_confidence:.2f})"
    else:
        status = "dry_run" if args.dry_run else "recorded"
        reason = ""

    if status in {"skipped", "dry_run"}:
        result = {**base, "status": status}
        if reason:
            result["reason"] = reason
        if session_file and not args.dry_run and not args.ignore_state:
            update_state(state, session_file=session_file, scanned_until_line=scanned_until_line, status=status)
            write_state(state_file, state)
        return result

    extra = feedback_extra(
        decision=decision,
        turn=turn,
        top_pages=top_pages,
        search_mode=search_mode,
        route_identity=route_identity,
    )
    record = append_feedback(
        "missed_candidate",
        note=decision.auditor_reason,
        prompt=turn.prompt,
        host=args.host,
        expected_pages=decision.expected_pages,
        expected_queries=[],
        ref=recall_snapshot.get("decision_id", "") if recall_snapshot else "",
        extra=extra,
    )
    if session_file and not args.ignore_state:
        update_state(state, session_file=session_file, scanned_until_line=scanned_until_line, status="recorded")
        write_state(state_file, state)
    result = {**base, "status": "recorded", "feedback": record}
    if not getattr(args, "no_auto_apply", False):
        try:
            from chronovisor.recall.recall_auto_apply import apply_feedback_file

            result["auto_apply"] = apply_feedback_file(
                config_file=Path(args.config).expanduser(),
                min_count=getattr(args, "auto_apply_min_count", None),
                dry_run=False,
            )
        except Exception as exc:
            result["auto_apply"] = {
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit recall decisions for missed Chronovisor context.")
    parser.add_argument("--host", choices=["claude-code", "codex", "generic"], default="generic")
    parser.add_argument("--hook", action="store_true", help="Read hook JSON from stdin.")
    parser.add_argument("--session-id")
    parser.add_argument("--session-file")
    parser.add_argument("--sessions-root", help="Codex sessions root override.")
    parser.add_argument("--cwd", default="")
    parser.add_argument("--prompt")
    parser.add_argument("--assistant-response")
    parser.add_argument("--decision-id", default="")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_FILE))
    parser.add_argument("--config", default=str(RECALL_CONFIG_FILE))
    parser.add_argument("--ignore-state", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Run even when auditor config is disabled.")
    parser.add_argument("--audit-read", action="store_true", help="Also audit turns where recall already decided read.")
    parser.add_argument("--no-auto-apply", action="store_true", help="Record candidates without applying auto-lane actions.")
    parser.add_argument("--auto-apply-min-count", type=int, help="Override auto-apply same-pattern threshold.")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument(
        "--auditor-json",
        help="Replay a precomputed auditor JSON object instead of calling the runtime.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-recall-audit`` command-line entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        with chronovisor_store.okf_runtime_operation(
            chronovisor_store.CHRONOVISOR_ROOT
        ):
            stdin_text = sys.stdin.read() if args.hook else None
            result = run(args, stdin_text=stdin_text)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75
    except Exception as exc:
        result = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
        print(json.dumps(result, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
