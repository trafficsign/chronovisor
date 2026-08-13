"""Shared recall runtime for Claude Code and Codex hooks.

The runtime is deliberately host-agnostic: host hooks pass prompt/cwd data in,
and this module decides whether Chronovisor should be searched before the agent
answers. Claude Code and Codex only need thin adapters around this CLI.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import sys
import threading
import time
import tomllib
import uuid
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core import ollama, recall_context, runtime_config
from chronovisor.core.index_store import (
    canonical_document_path_for_id,
    get_store,
)
from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.recall_runtime_paths import RECALL_DIR
from chronovisor.core.search import last_search_trace, search_existing_bm25
from chronovisor.core.search import search as run_search
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    init_chronovisor,
    okf_runtime_operation,
    okf_startup_status,
)
from chronovisor.decision.local_structured import LocalStructuredSession
from chronovisor.decision.recall_policy_contract import RecallPolicy as RecallPolicy
from chronovisor.ingest.page_registry import PageRegistry, PageRegistryError
from chronovisor.ingest.state_register import format_state_context, should_inject_state
from chronovisor.recall import evidence_provider as _evidence_provider
from chronovisor.recall import recall_publication as _recall_publication
from chronovisor.recall.recall_prompt import (
    CODEX_INTERNAL_SUGGESTION_RE,
    SYSTEM_ENVELOPE_RE,
    normalize_recall_prompt,
)
from chronovisor.recall.recall_publication import (
    _neutralize_context_delimiters as _neutralize_context_delimiters,
)
from chronovisor.recall.recall_publication import (
    _one_line,
    _retained_context_page_ids,
    merge_context_blocks,
    render_output,
)
from chronovisor.recall.recall_publication import (
    context_item_annotations as context_item_annotations,
)
from chronovisor.recall.recall_publication import result_to_dict as result_to_dict

_render_recall_payload = recall_context.render_recall_payload

RECALL_LOG_FILE = RECALL_DIR / "recall-log.jsonl"
RECALL_FEEDBACK_FILE = RECALL_DIR / "feedback.jsonl"
RECALL_CONFIG_FILE = runtime_config.active_config_file()
RECALL_PULL_LOG_FILE = RECALL_DIR / "pull-log.jsonl"
RECALL_CALIBRATION_FILE = RECALL_DIR / "calibration.json"
TYPED_GRAPH_TRACE_FILE = (
    CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "candidate-trace.jsonl"
)
RECALL_GATE_RUNTIME_ROLE = "recall.gate"
RECALL_QUERY_REWRITER_RUNTIME_ROLE = "recall.query_rewriter"

TRIVIAL_PROMPT_RE = re.compile(
    r"^\s*(はい|いいえ|うん|おう|ok|okay|yes|no|y|n|ありがとう|thanks|thx|了解|りょ)\s*[。.!！?？]*\s*$",
    re.IGNORECASE,
)
MACHINE_TAG_RE = re.compile(r"^\s*<([a-z][a-z0-9-]{2,})\b", re.IGNORECASE)

PAST_REFERENCE_TERMS = [
    "昨日",
    "前回",
    "この前",
    "前に",
    "前も",
    "前の",
    "以前",
    "さっき",
    "こないだ",
    "続き",
    "例の",
    "あの時",
    "last time",
    "previous",
    "yesterday",
]

AMBIGUITY_TERMS = [
    "あれ",
    "それ",
    "これ",
    "その件",
    "この件",
    "その話",
    "あの話",
    "これ系",
    "あの",
]

OWNERSHIP_TERMS = [
    "俺の",
    "俺は",
    "俺が",
    "俺に",
    "自分の",
    "うちの",
    "my ",
    "me ",
]

PROJECT_TERMS = [
    "chronovisor",
    "chronovisor",
    "wiki",
    "ウィキ",
    "codex",
    "コードエクス",
    "クラウドコード",
    "claude code",
    "claude",
    "hook",
    "フック",
    "mcp",
    "uvx",
    "qwen",
    "ollama",
    "context window",
    "model_context_window",
    "spi",
    "面接",
    "川重",
    "川崎重工",
]

DECISION_TERMS = [
    "設計",
    "方針",
    "定義",
    "実装",
    "導入",
    "改善",
    "直す",
    "修正",
    "運用",
    "切り分け",
    "検証",
    "プラン",
    "計画",
    "どうする",
    "どう思う",
    "どうかな",
    "できる",
    "使える",
]

CHITCHAT_TERMS = [
    "暑い",
    "寒い",
    "眠い",
    "腹減った",
    "疲れた",
    "おはよう",
    "こんにちは",
    "こんばんは",
]


class RecallWallClockTimeout(BaseException):
    """Hard-stop a synchronous Recall hook without being swallowed downstream."""


@contextmanager
def recall_wall_clock_deadline(timeout_ms: int) -> Iterator[None]:
    """Apply a process-local hard deadline when running on the main Unix thread."""

    if (
        timeout_ms <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise RecallWallClockTimeout(f"recall exceeded {timeout_ms}ms")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.0, previous_timer[0] - elapsed),
                previous_timer[1],
            )


def recall_outer_deadline_ms(policy: RecallPolicy) -> int:
    """Return the final host boundary; ``run_recall`` owns its inner reserve."""

    return max(100, int(policy.total_timeout_ms))


@dataclass
class RecallRequest:
    host: str
    event: str
    prompt: str
    cwd: str = ""
    recent_context: str = ""
    session_id: str = ""
    decision_id: str = ""


@dataclass
class ContextItem:
    page_id: str
    title: str
    updated: str
    score: float
    uid: str = ""
    snippets: list[str] = field(default_factory=list)
    sensitivity: str = "normal"
    certificate_id: str = ""
    evidence_kind: str = "legacy"
    source_line: int = 0


@dataclass
class RecallResult:
    status: str
    decision: str
    confidence: float
    queries: list[str]
    reasons: list[str]
    matched_terms: dict[str, list[str]]
    decision_id: str = field(default_factory=lambda: new_decision_id())
    session_id: str = ""
    context_items: list[ContextItem] = field(default_factory=list)
    context: str = ""
    state_context: str = ""
    used_judge: bool = False
    judge_confidence: float | None = None
    judge_reason: str = ""
    evidence_features: dict[str, Any] = field(default_factory=dict)
    evidence_packet: Any | None = None
    search_mode: str = ""
    context_style: str = ""
    latency_ms: int = 0
    error: str = ""


class RecallBudgetExhausted(TimeoutError):
    """Raised when the synchronous recall wall-clock budget is exhausted."""


def _recall_runtime_route(role: str) -> ollama.RuntimeGenerationRoute:
    route = ollama.runtime_generation_routes((role,))[0]
    if route.role != role:
        raise ollama.RuntimeBridgeError("route_configuration_invalid")
    if not route.structured_output:
        raise ollama.RuntimeBridgeError("capability_unavailable")
    return route


def load_policy(path: Path = RECALL_CONFIG_FILE) -> RecallPolicy:
    policy = RecallPolicy()
    path = runtime_config.active_config_file(path)
    if path.exists():
        try:
            data = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        _apply_config(policy, data)

    try:
        from chronovisor.recall.recall_policy_store import apply_active_policy

        apply_active_policy(policy)
    except Exception:
        pass

    _apply_search_embedding_boundary(policy, path)

    policy.max_total_context_chars = max(
        policy.max_total_context_chars,
        policy.max_state_context_chars + policy.max_context_chars + 2,
    )

    enabled_env = os.environ.get("CHRONOVISOR_RECALL_ENABLED")
    if enabled_env is not None:
        policy.enabled = enabled_env not in {"0", "false", "False", "no", "NO"}
    return policy


def new_decision_id() -> str:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def stable_prompt_hash(text: str) -> str:
    return hashlib.sha1(_feedback_key(text).encode("utf-8")).hexdigest()[:16]


def _apply_search_embedding_boundary(policy: RecallPolicy, path: Path) -> None:
    """Apply the independent L2 semantic rollout after learned overrides."""

    try:
        from chronovisor.core.runtime_config import load_search_embedding_config

        search_embedding = load_search_embedding_config(path)
    except Exception:
        policy.semantic = False
        return
    if not search_embedding.enabled or not search_embedding.sync_recall:
        policy.semantic = False
        return
    policy.semantic = True
    policy.fusion_semantic = search_embedding.fusion_weight
    policy.fusion_semantic_min_top_score = search_embedding.min_top_score
    policy.fusion_semantic_min_margin = search_embedding.min_margin
    policy.fusion_semantic_low_confidence_weight = (
        search_embedding.low_confidence_weight
    )


def _apply_config(policy: RecallPolicy, data: dict[str, Any]) -> None:
    recall = data.get("recall")
    recall_root = recall if isinstance(recall, dict) else {}

    enabled = recall_root.get("enabled", data.get("enabled"))
    if isinstance(enabled, bool):
        policy.enabled = enabled
    def section(name: str) -> dict[str, Any]:
        nested = recall_root.get(name)
        if isinstance(nested, dict):
            return nested
        legacy = data.get(name)
        return legacy if isinstance(legacy, dict) else {}

    thresholds = section("thresholds")
    if thresholds:
        if isinstance(thresholds.get("search"), int | float):
            policy.search_threshold = float(thresholds["search"])
        if isinstance(thresholds.get("read"), int | float):
            policy.read_threshold = float(thresholds["read"])

    budgets = section("budgets")
    if budgets:
        if isinstance(budgets.get("max_context_tokens"), int):
            policy.max_context_chars = max(400, budgets["max_context_tokens"] * 4)
        if isinstance(budgets.get("max_context_chars"), int):
            policy.max_context_chars = max(400, budgets["max_context_chars"])
        if isinstance(budgets.get("max_state_context_chars"), int):
            policy.max_state_context_chars = max(
                400, budgets["max_state_context_chars"]
            )
        if isinstance(budgets.get("max_total_context_chars"), int):
            policy.max_total_context_chars = max(
                720, budgets["max_total_context_chars"]
            )
        if isinstance(budgets.get("max_pages"), int):
            policy.max_pages = max(1, budgets["max_pages"])
        if isinstance(budgets.get("max_queries"), int):
            policy.max_queries = max(1, budgets["max_queries"])
        if isinstance(budgets.get("judge_timeout_ms"), int):
            policy.judge_timeout_ms = max(200, budgets["judge_timeout_ms"])
        if isinstance(budgets.get("total_timeout_ms"), int):
            policy.total_timeout_ms = max(500, budgets["total_timeout_ms"])
        if isinstance(budgets.get("deterministic_fallback_reserve_ms"), int):
            policy.deterministic_fallback_reserve_ms = max(
                0,
                min(
                    policy.total_timeout_ms - 100,
                    budgets["deterministic_fallback_reserve_ms"],
                ),
            )

    if isinstance(recall_root.get("semantic"), bool):
        policy.semantic = recall_root["semantic"]
    if isinstance(recall_root.get("gate_mode"), str):
        policy.gate_mode = recall_root["gate_mode"]
    if isinstance(recall_root.get("context_style"), str):
        policy.context_style = recall_root["context_style"]
    if isinstance(recall_root.get("max_context_chars"), int):
        policy.max_context_chars = max(400, recall_root["max_context_chars"])
    if isinstance(recall_root.get("judge_mode"), str):
        policy.judge_mode = recall_root["judge_mode"]
    if isinstance(recall_root.get("session_ttl_seconds"), int):
        policy.session_ttl_seconds = max(3600, recall_root["session_ttl_seconds"])

    gate = section("gate")
    if gate:
        if isinstance(gate.get("think"), bool):
            policy.judge_think = gate["think"]
        if isinstance(gate.get("timeout_ms"), int):
            policy.judge_timeout_ms = max(200, gate["timeout_ms"])
        if isinstance(gate.get("judge_timeout_ms"), int):
            policy.judge_timeout_ms = max(200, gate["judge_timeout_ms"])
        if isinstance(gate.get("num_ctx"), int):
            policy.judge_num_ctx = max(512, gate["num_ctx"])
        if isinstance(gate.get("num_predict"), int):
            policy.judge_num_predict = max(16, gate["num_predict"])
        if isinstance(gate.get("include_queries"), bool):
            policy.judge_include_queries = gate["include_queries"]
        if isinstance(gate.get("keep_alive"), str) and gate["keep_alive"]:
            policy.judge_keep_alive = gate["keep_alive"]
        if isinstance(gate.get("warmup_timeout_ms"), int):
            policy.warmup_timeout_ms = max(1000, gate["warmup_timeout_ms"])

    rewrite = section("rewrite")
    if rewrite:
        if isinstance(rewrite.get("enabled"), bool):
            policy.rewrite_enabled = rewrite["enabled"]
        if isinstance(rewrite.get("timeout_ms"), int):
            policy.rewrite_timeout_ms = max(200, rewrite["timeout_ms"])

    processor = section("processor")
    if processor:
        if isinstance(processor.get("enabled"), bool):
            policy.processor_enabled = processor["enabled"]
        if isinstance(processor.get("shadow_enabled"), bool):
            policy.processor_shadow_enabled = processor["shadow_enabled"]
        if isinstance(processor.get("auto_enable"), bool):
            policy.processor_auto_enable = processor["auto_enable"]
        if isinstance(processor.get("max_candidates"), int):
            policy.processor_max_candidates = max(1, processor["max_candidates"])
        if isinstance(processor.get("max_pointer_cards"), int):
            policy.processor_max_pointer_cards = max(
                0, min(6, processor["max_pointer_cards"])
            )
        if isinstance(processor.get("max_rich_evidence"), int):
            policy.processor_max_rich_evidence = max(
                0,
                min(
                    policy.processor_max_pointer_cards,
                    processor["max_rich_evidence"],
                ),
            )
        if isinstance(processor.get("injection_token_budget"), int):
            policy.processor_injection_token_budget = max(
                64, processor["injection_token_budget"]
            )
        if isinstance(processor.get("certificate_required"), bool):
            policy.processor_certificate_required = processor["certificate_required"]
        if isinstance(processor.get("judge_enabled"), bool):
            policy.processor_judge_enabled = processor["judge_enabled"]
        if isinstance(processor.get("judge_timeout_ms"), int):
            policy.processor_judge_timeout_ms = max(200, processor["judge_timeout_ms"])
        if isinstance(processor.get("escalation_timeout_ms"), int):
            policy.processor_escalation_timeout_ms = max(
                300, processor["escalation_timeout_ms"]
            )

    circuit_breaker = section("circuit_breaker")
    if circuit_breaker:
        if isinstance(circuit_breaker.get("failures"), int):
            policy.circuit_breaker_failures = max(1, circuit_breaker["failures"])
        if isinstance(circuit_breaker.get("cooldown_seconds"), int):
            policy.circuit_breaker_cooldown_seconds = max(
                1, circuit_breaker["cooldown_seconds"]
            )

    fusion = section("fusion")
    if fusion:
        for key, attr in (
            ("anchor", "fusion_anchor"),
            ("bm25", "fusion_bm25"),
            ("semantic", "fusion_semantic"),
            ("graph", "fusion_graph"),
            ("context", "fusion_context"),
            ("usage_prior", "fusion_usage_prior"),
            ("bm25_score_bonus", "fusion_bm25_score_bonus"),
            ("bm25_rank_bonus", "fusion_bm25_rank_bonus"),
            ("bm25_rank_decay", "fusion_bm25_rank_decay"),
            ("semantic_min_top_score", "fusion_semantic_min_top_score"),
            ("semantic_min_margin", "fusion_semantic_min_margin"),
            ("semantic_low_confidence_weight", "fusion_semantic_low_confidence_weight"),
            ("usage_prior_decay", "fusion_usage_prior_decay"),
            ("usage_prior_cap", "fusion_usage_prior_cap"),
        ):
            value = fusion.get(key)
            if isinstance(value, int | float):
                setattr(policy, attr, max(0.0, float(value)))

    calibration = section("calibration")
    if calibration:
        if isinstance(calibration.get("enabled"), bool):
            policy.calibration_enabled = calibration["enabled"]
        if isinstance(calibration.get("min_samples"), int):
            policy.calibration_min_samples = max(1, calibration["min_samples"])
        if isinstance(calibration.get("holdout_ratio"), int | float):
            policy.calibration_holdout_ratio = max(
                0.05, min(0.8, float(calibration["holdout_ratio"]))
            )
        if isinstance(calibration.get("min_improvement"), int | float):
            policy.calibration_min_improvement = max(
                0.0, float(calibration["min_improvement"])
            )

    behavior = section("policy")
    if behavior:
        if isinstance(behavior.get("log_decisions"), bool):
            policy.log_decisions = behavior["log_decisions"]
        if isinstance(behavior.get("avoid_heavy_personal_context_in_chitchat"), bool):
            policy.avoid_heavy_personal_context_in_chitchat = behavior[
                "avoid_heavy_personal_context_in_chitchat"
            ]
        if isinstance(behavior.get("use_feedback_suppressions"), bool):
            policy.use_feedback_suppressions = behavior["use_feedback_suppressions"]
        if isinstance(behavior.get("fail_silent_on_judge_unavailable"), bool):
            policy.fail_silent_on_judge_unavailable = behavior[
                "fail_silent_on_judge_unavailable"
            ]


def request_from_hook_payload(
    payload: dict[str, Any], *, host: str, event: str
) -> RecallRequest:
    prompt = ""
    for key in ("user_prompt", "prompt", "message", "input"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            prompt = value
            break

    cwd = ""
    for key in ("cwd", "current_dir", "working_directory"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            cwd = value
            break
    if not cwd:
        cwd = os.environ.get("PWD", "")

    session_id = ""
    for key in ("session_id", "conversation_id", "thread_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            session_id = value
            break

    recent_context = payload.get("recent_context", "")
    if not isinstance(recent_context, str):
        recent_context = ""

    return RecallRequest(
        host=host,
        event=event,
        prompt=prompt,
        cwd=cwd,
        recent_context=recent_context,
        session_id=session_id,
    )


def evaluate_heuristic(
    request: RecallRequest, policy: RecallPolicy
) -> tuple[float, list[str], dict[str, list[str]]]:
    prompt = request.prompt.strip()
    prompt_lower = prompt.lower()
    cwd_lower = request.cwd.lower()

    matched = {
        "past_reference": _matched_terms(prompt_lower, PAST_REFERENCE_TERMS),
        "ambiguity": _matched_terms(prompt_lower, AMBIGUITY_TERMS),
        "ownership": _matched_terms(prompt_lower, OWNERSHIP_TERMS),
        "project": _matched_terms(prompt_lower, PROJECT_TERMS),
        "decision": _matched_terms(prompt_lower, DECISION_TERMS),
        "chitchat": _matched_terms(prompt_lower, CHITCHAT_TERMS),
        "cwd": [],
    }

    score = 0.0
    reasons: list[str] = []

    if TRIVIAL_PROMPT_RE.match(prompt):
        return 0.0, ["trivial prompt"], matched

    if matched["past_reference"]:
        score += 0.42
        reasons.append("past reference term")
    if matched["ownership"]:
        score += 0.30
        reasons.append("user-owned fact/preference")
    if matched["project"]:
        score += 0.24
        reasons.append("known recurring project/topic")
    if matched["decision"]:
        score += 0.20
        reasons.append("design/operation decision")
    if matched["ambiguity"]:
        score += 0.14
        reasons.append("ambiguous reference")

    if any(
        key in cwd_lower for key in ("chronovisor", "codex", "claude", "jttok", "jttk")
    ):
        score += 0.10
        matched["cwd"].append(Path(request.cwd).name or request.cwd)
        reasons.append("cwd matches recurring work")

    if "?" in prompt or "？" in prompt:
        score += 0.04

    if matched["chitchat"] and not (
        matched["past_reference"] or matched["project"] or matched["ownership"]
    ):
        score -= 0.35
        reasons.append("simple chitchat")

    if (
        policy.avoid_heavy_personal_context_in_chitchat
        and matched["chitchat"]
        and not matched["past_reference"]
    ):
        score = min(score, policy.search_threshold - 0.05)
        reasons.append("personal-context guard")

    return max(0.0, min(1.0, score)), reasons, matched


def classify_non_user_prompt(prompt: str, policy: RecallPolicy | None = None) -> str:
    stripped = prompt.lstrip()
    if SYSTEM_ENVELOPE_RE.match(stripped):
        return "system notification prompt"
    if CODEX_INTERNAL_SUGGESTION_RE.match(stripped):
        return "codex internal suggestion prompt"
    if stripped.startswith("[RECALL_CONTEXT]") or stripped.startswith(
        "[/RECALL_CONTEXT]"
    ):
        return "recall context injection"
    if policy is None or policy.use_feedback_suppressions:
        feedback_reason = classify_feedback_suppressed_prompt(stripped)
        if feedback_reason:
            return feedback_reason
    return ""


def strip_non_user_blocks(prompt: str) -> tuple[str, list[str]]:
    return normalize_recall_prompt(prompt)


def classify_feedback_suppressed_prompt(prompt: str) -> str:
    prompt_key = _feedback_key(prompt)
    if not prompt_key:
        return ""
    prompt_tag = MACHINE_TAG_RE.match(prompt)
    for record in recent_false_positive_feedback():
        feedback_prompt = record.get("prompt", "")
        if not isinstance(feedback_prompt, str):
            continue
        feedback_key = _feedback_key(feedback_prompt)
        if feedback_key and prompt_key == feedback_key:
            return "feedback false-positive prompt"
        feedback_tag = MACHINE_TAG_RE.match(feedback_prompt)
        if feedback_tag and prompt_tag:
            feedback_tag_name = feedback_tag.group(1).lower()
            if prompt_tag.group(1).lower() == feedback_tag_name:
                return f"feedback false-positive tag <{feedback_tag_name}>"
    return ""


def recent_false_positive_feedback(limit: int = 100) -> list[dict[str, Any]]:
    try:
        with RECALL_FEEDBACK_FILE.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=limit)
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("kind") == "false-positive":
            records.append(record)
    return records


def _feedback_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _matched_terms(prompt_lower: str, terms: list[str]) -> list[str]:
    matched: list[str] = []
    for term in terms:
        needle = term.lower().strip()
        if not needle:
            continue
        if re.fullmatch(r"[a-z0-9_][a-z0-9_ ]*[a-z0-9_]", needle):
            pattern = rf"(?<![a-z0-9_]){re.escape(needle)}(?![a-z0-9_])"
            if re.search(pattern, prompt_lower):
                matched.append(term)
            continue
        if needle in prompt_lower:
            matched.append(term)
    return matched


def should_run_judge(
    score: float, policy: RecallPolicy, features: dict[str, Any] | None = None
) -> bool:
    if policy.judge_mode == "off":
        return False
    if policy.judge_mode == "always":
        return True
    if policy.gate_mode == "evidence" and features:
        margin = float(features.get("margin_norm", 0.0) or 0.0)
        hit_count = int(features.get("hit_count", 0) or 0)
        evidence = float(features.get("evidence_score", score) or score)
        if hit_count == 0:
            return False
        return (
            policy.search_threshold <= evidence < policy.read_threshold
            and margin < 0.12
        )
    # Auto judge is for the ambiguous search zone. Obvious read decisions should
    # not depend on a synchronous local model being available.
    return (policy.search_threshold - 0.10) <= score < policy.read_threshold


def run_local_judge(
    request: RecallRequest,
    heuristic_score: float,
    policy: RecallPolicy,
    *,
    timeout_ms: int | None = None,
) -> tuple[float | None, list[str], str]:
    system = "You are a fast Chronovisor recall classifier. Return compact JSON only."
    prompt: dict[str, Any] = {
        "user_prompt": request.prompt,
        "cwd": request.cwd,
        "heuristic_score": round(heuristic_score, 3),
        "task": "Return whether past user/project memory should be recalled.",
        "rubric": "0.0-0.34 none, 0.35-0.64 search, 0.65-1.0 read",
        "rule": "Prefer silence for uncertain/simple chitchat. Do not recall heavy personal context.",
        "output": {
            "decision": "none | search | read",
            "confidence": "number 0..1",
            "reason": "very short Japanese reason",
        },
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["none", "search", "read"]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["decision", "confidence", "reason"],
    }
    if policy.judge_include_queries:
        prompt["output"]["queries"] = ["1-3 short search queries if recall is useful"]
        schema["properties"]["queries"] = {"type": "array", "items": {"type": "string"}}

    try:
        route = _recall_runtime_route(RECALL_GATE_RUNTIME_ROLE)
        result = LocalStructuredSession(
            model=route.model,
            role="recall_judge",
            runtime_role=RECALL_GATE_RUNTIME_ROLE,
            runtime_location=route.location,
            source_data_class="raw",
            source_sensitivity="high",
            num_ctx=policy.judge_num_ctx,
            num_predict=policy.judge_num_predict,
            keep_alive=policy.judge_keep_alive,
            read_timeout_ms=max(
                200,
                min(policy.judge_timeout_ms, timeout_ms)
                if isinstance(timeout_ms, int)
                else policy.judge_timeout_ms,
            ),
            max_input_chars=16_384,
            max_output_chars=384,
            max_feedback_chars=512,
            max_responses=1,
            resource_lease_timeout_ms=25,
        ).run(
            json.dumps(prompt, ensure_ascii=False),
            schema,
            system=system,
        )
        if not result.ok:
            return (
                None,
                [],
                f"judge unavailable: {result.failure_class or 'structured_failure'}",
            )
        parsed = result.value
        decision = parsed.get("decision")
        if decision not in {"none", "search", "read"}:
            return None, [], "judge returned no valid decision"
        confidence = parsed.get("confidence")
        if not isinstance(confidence, int | float):
            return None, [], "judge returned no numeric confidence"
        confidence = normalize_judge_confidence(float(confidence), decision, policy)
        queries = (
            [q for q in parsed.get("queries", []) if isinstance(q, str) and q.strip()]
            if policy.judge_include_queries
            else []
        )
        reason = parsed.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        return confidence, queries, reason
    except Exception as exc:
        return None, [], f"judge unavailable: {exc.__class__.__name__}"


def normalize_judge_confidence(
    confidence: float, decision: str, policy: RecallPolicy
) -> float:
    confidence = max(0.0, min(1.0, confidence))
    if decision == "none":
        return min(confidence, policy.search_threshold - 0.01)
    if decision == "search":
        return min(
            max(confidence, policy.search_threshold), policy.read_threshold - 0.01
        )
    return max(confidence, policy.read_threshold)


def build_queries(
    request: RecallRequest,
    matched: dict[str, list[str]],
    judge_queries: list[str],
    policy: RecallPolicy,
    session_state: Any | None = None,
    rewrite_queries: list[str] | None = None,
) -> list[str]:
    # The current prompt is authoritative. Prior turns are episode context for
    # rewriting ambiguous prompts, never independent retrieval entrances.
    candidates: list[str] = [_compact_query(request.prompt)]
    ambiguous = bool(matched.get("ambiguity") or matched.get("past_reference"))
    if ambiguous:
        candidates.extend(rewrite_queries or [])
        candidates.extend(judge_queries)
    if session_state is not None and ambiguous:
        recent_topics = getattr(session_state, "recent_topics", [])[-8:]
        if recent_topics:
            candidates.append(
                " ".join(recent_topics + [_compact_query(request.prompt, limit=80)])
            )

    return _dedupe_queries(candidates, limit=policy.max_queries)


def should_rewrite_query(
    *,
    request: RecallRequest,
    matched: dict[str, list[str]],
    policy: RecallPolicy,
    preliminary_features: dict[str, Any] | None = None,
) -> bool:
    if not policy.rewrite_enabled:
        return False
    if not (matched.get("ambiguity") or matched.get("past_reference")):
        return False
    features = preliminary_features or {}
    hit_count = int(features.get("hit_count", 0) or 0)
    top1 = float(features.get("top1_score_norm", 0.0) or 0.0)
    return hit_count == 0 or top1 < 0.35 or bool(matched.get("ambiguity"))


def run_query_rewriter(
    request: RecallRequest,
    matched: dict[str, list[str]],
    policy: RecallPolicy,
    session_summary: str,
    *,
    timeout_ms: int | None = None,
) -> tuple[list[str], float, str]:
    prompt = {
        "task": "Rewrite an ambiguous user prompt into 1-3 explicit Chronovisor search queries.",
        "user_prompt": request.prompt,
        "cwd": request.cwd,
        "matched_terms": matched,
        "session_state": session_summary,
        "output": {"queries": ["short search query"], "confidence": "number 0..1"},
    }
    schema = {
        "type": "object",
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "confidence": {"type": "number"},
        },
        "required": ["queries", "confidence"],
    }
    try:
        route = _recall_runtime_route(RECALL_QUERY_REWRITER_RUNTIME_ROLE)
        result = LocalStructuredSession(
            model=route.model,
            role="recall_query_rewriter",
            runtime_role=RECALL_QUERY_REWRITER_RUNTIME_ROLE,
            runtime_location=route.location,
            source_data_class="raw",
            source_sensitivity="high",
            num_ctx=policy.judge_num_ctx,
            num_predict=96,
            keep_alive=policy.judge_keep_alive,
            read_timeout_ms=max(
                200,
                min(policy.rewrite_timeout_ms, timeout_ms)
                if isinstance(timeout_ms, int)
                else policy.rewrite_timeout_ms,
            ),
            max_input_chars=16_384,
            max_output_chars=384,
            max_feedback_chars=512,
            max_responses=1,
            resource_lease_timeout_ms=25,
        ).run(json.dumps(prompt, ensure_ascii=False), schema)
        if not result.ok:
            return (
                [],
                0.0,
                f"rewrite fallback: {result.failure_class or 'structured_failure'}",
            )
        parsed = result.value
        raw_queries = parsed.get("queries")
        queries = (
            [q for q in raw_queries if isinstance(q, str) and q.strip()]
            if isinstance(raw_queries, list)
            else []
        )
        confidence = parsed.get("confidence", 0.0)
        confidence_f = (
            max(0.0, min(1.0, float(confidence)))
            if isinstance(confidence, int | float)
            else 0.0
        )
        return (
            _dedupe_queries(queries, limit=policy.max_queries),
            confidence_f,
            "rewrite ok",
        )
    except Exception as exc:
        return [], 0.0, f"rewrite fallback: {exc.__class__.__name__}"


def warm_recall_model(policy: RecallPolicy) -> dict[str, Any]:
    """Warm the fixed gate/rewrite routes so sync recall avoids cold starts."""
    started = time.monotonic()
    warmed: list[str] = []
    errors: dict[str, str] = {}
    for role in (RECALL_GATE_RUNTIME_ROLE, RECALL_QUERY_REWRITER_RUNTIME_ROLE):
        try:
            route = _recall_runtime_route(role)
            result = LocalStructuredSession(
                model=route.model,
                role="recall_warmup",
                runtime_role=role,
                runtime_location=route.location,
                source_data_class="raw",
                source_sensitivity="high",
                num_ctx=policy.judge_num_ctx,
                num_predict=16,
                keep_alive=policy.judge_keep_alive,
                read_timeout_ms=max(1_000, policy.warmup_timeout_ms),
                # The fixed structured system plus this tiny prompt is already
                # about 415 UTF-8 bytes.  Keep the cap small, but large enough
                # for the real preflight instead of only the mocked transport.
                max_input_chars=512,
                max_output_chars=128,
                max_feedback_chars=512,
            ).run(
                "Warm the model and return an empty JSON object.",
                {"type": "object", "maxProperties": 0},
            )
            if not result.ok:
                errors[route.model] = result.failure_class or "structured_failure"
                continue
            if route.model not in warmed:
                warmed.append(route.model)
        except Exception as exc:
            errors[role] = exc.__class__.__name__
    return {
        "ok": not errors,
        "models": warmed,
        "errors": errors,
        "keep_alive": policy.judge_keep_alive,
        "latency_ms": _elapsed_ms(started),
    }


def _compact_query(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] or text[:limit]


def _split_alphanumeric_query(text: str) -> str:
    """Add searchable token boundaries to joined names such as ``AI2040``."""

    return re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)


def _dedupe_queries(candidates: list[str], limit: int) -> list[str]:
    seen = set()
    out = []
    for candidate in candidates:
        q = _split_alphanumeric_query(re.sub(r"\s+", " ", candidate).strip())
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= limit:
            break
    return out


def decision_from_score(score: float, policy: RecallPolicy) -> str:
    if score >= policy.read_threshold:
        return "read"
    if score >= policy.search_threshold:
        return "search"
    return "none"


def _score_norm(score: float) -> float:
    # Reciprocal-rank fusion scores are intentionally tiny. Treating 0.016 as
    # 0.48 made almost any retrieved page look like strong evidence. Preserve
    # the weak absolute signal until a revision-specific calibrator replaces it.
    if score <= 0:
        return 0.0
    if score < 0.2:
        return min(0.5, score / 0.1)
    return score / (score + 3.0)


def calibration_artifact() -> dict[str, Any] | None:
    try:
        data = json.loads(RECALL_CALIBRATION_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("weights"), dict):
        return None
    return data


def calibrated_score(features: dict[str, Any], policy: RecallPolicy) -> float | None:
    if not policy.calibration_enabled:
        return None
    artifact = calibration_artifact()
    if not artifact:
        return None
    weights = artifact.get("weights")
    if not isinstance(weights, dict):
        return None
    bias = artifact.get("bias", 0.0)
    try:
        total = float(bias)
        for key, weight in weights.items():
            total += float(weight) * float(features.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if total < -40:
        return 0.0
    if total > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-total))


def build_evidence_features(
    *,
    request: RecallRequest,
    matched: dict[str, list[str]],
    heuristic_score: float,
    results: list[Any],
    search_mode: str,
    rewrite_confidence: float = 0.0,
) -> dict[str, Any]:
    top1 = float(results[0].score) if results else 0.0
    top2 = float(results[1].score) if len(results) > 1 else 0.0
    top1_norm = _score_norm(top1)
    top2_norm = _score_norm(top2)
    margin_norm = max(0.0, top1_norm - top2_norm)
    prompt_chars = len(request.prompt)
    return {
        "top1_score": top1,
        "top2_score": top2,
        "top1_score_norm": top1_norm,
        "margin_norm": margin_norm,
        "hit_count": len(results),
        "hit_count_norm": min(1.0, len(results) / 5.0),
        "prompt_chars": prompt_chars,
        "prompt_len_norm": min(1.0, prompt_chars / 300.0),
        "ambiguity": 1.0 if matched.get("ambiguity") else 0.0,
        "past_reference": 1.0 if matched.get("past_reference") else 0.0,
        "project": 1.0 if matched.get("project") else 0.0,
        "heuristic_score": heuristic_score,
        "rewrite_confidence": rewrite_confidence,
        "search_mode": search_mode,
    }


def evidence_score(features: dict[str, Any], policy: RecallPolicy) -> float:
    calibrated = calibrated_score(features, policy)
    if calibrated is not None:
        features["calibrated"] = True
        return calibrated
    score = (
        0.35 * float(features.get("top1_score_norm", 0.0) or 0.0)
        + 0.12 * float(features.get("margin_norm", 0.0) or 0.0)
        + 0.45 * float(features.get("heuristic_score", 0.0) or 0.0)
        + 0.08 * float(features.get("rewrite_confidence", 0.0) or 0.0)
    )
    if features.get("ambiguity") and features.get("hit_count", 0):
        score += 0.04
    if not features.get("hit_count"):
        score = min(score, policy.search_threshold - 0.05)
    features["calibrated"] = False
    return max(0.0, min(1.0, score))


def request_context_terms(request: RecallRequest | None) -> set[str]:
    if request is None:
        return set()
    parts = [request.cwd, Path(request.cwd).name if request.cwd else "", request.host]
    text = " ".join(part for part in parts if part).lower()
    terms = set(re.findall(r"[a-z0-9][a-z0-9_.+-]{2,}", text))
    if "chronovisor" in text:
        terms.update({"chronovisor", "wiki", "mcp"})
    return terms


def context_boost(result: Any, request: RecallRequest | None) -> float:
    terms = request_context_terms(request)
    if not terms:
        return 1.0
    haystack = " ".join(
        str(getattr(result, attr, "") or "").lower()
        for attr in ("page_id", "title", "folder")
    )
    hits = sum(1 for term in terms if term in haystack)
    if hits <= 0:
        return 1.0
    return min(1.35, 1.0 + (0.08 * hits))


def is_work_context(request: RecallRequest | None) -> bool:
    if request is None:
        return False
    cwd = (request.cwd or "").lower()
    return "/projects/work/" in cwd or cwd.endswith("/projects/work")


def prompt_allows_sensitive_context(request: RecallRequest | None) -> bool:
    if request is None:
        return True
    prompt = (request.prompt or "").lower()
    allow_terms = (
        "career",
        "interview",
        "転職",
        "面接",
        "退職",
        "職務",
        "キャリア",
        "mental",
        "sensitive",
    )
    return any(term in prompt for term in allow_terms)


def should_filter_sensitive_result(result: Any, request: RecallRequest | None) -> bool:
    if getattr(result, "sensitivity", "normal") != "high":
        return False
    return is_work_context(request) and not prompt_allows_sensitive_context(request)


def search_candidates(
    queries: list[str],
    policy: RecallPolicy,
    *,
    request: RecallRequest | None = None,
    deadline_at: float | None = None,
    stage_timings_ms: dict[str, int] | None = None,
    diagnostic_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[Any], str]:
    if not queries:
        return [], ""
    _require_remaining_budget(deadline_at, "search")
    trace_decision_id = (
        request.decision_id if request is not None and request.decision_id else new_decision_id()
    )

    def trace_rows(
        query: str,
        paths: dict[str, Any],
        *,
        shadow: bool,
        query_plan: str = "",
    ) -> list[dict[str, Any]]:
        if request is None:
            return []
        query_sha = hashlib.sha256(query.encode()).hexdigest()
        session = (
            hashlib.sha256(request.session_id.encode()).hexdigest()[:16]
            if request.session_id
            else ""
        )
        created_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        output = []
        for page_id, value in paths.items():
            if not isinstance(value, dict):
                continue
            path_ids = [
                str(item)
                for item in value.get("relation_ids") or []
                if isinstance(item, str)
                and (item.startswith("rel_") or item.startswith("merge_"))
            ]
            relation_ids = [item for item in path_ids if item.startswith("rel_")]
            entity_merge_ids = [
                item for item in path_ids if item.startswith("merge_")
            ]
            if not path_ids:
                continue
            path_id = str(value.get("path_id") or "")
            if not path_id:
                path_id = "path_" + hashlib.sha256(
                    "|".join([page_id, *path_ids]).encode()
                ).hexdigest()[:24]
            output.append(
                {
                    "schema_version": 1,
                    "trace_id": hashlib.sha256(
                        f"{trace_decision_id}:{path_id}:{page_id}:{shadow}".encode()
                    ).hexdigest()[:24],
                    "decision_id": trace_decision_id,
                    "session_hash": session,
                    "query_sha256": query_sha,
                    "query_plan": str(value.get("query_plan") or query_plan),
                    "page_id": str(page_id),
                    "path_id": path_id,
                    "page_ids": [
                        str(item)
                        for item in value.get("pages") or value.get("path") or []
                        if isinstance(item, str)
                    ],
                    "relation_ids": relation_ids,
                    "entity_merge_ids": entity_merge_ids,
                    "relations": value.get("relations") or [],
                    "community_id": str(value.get("community_id") or ""),
                    "activation": float(value.get("activation") or 0.0),
                    "shadow": shadow,
                    "supervision": "exposure",
                    "candidate_generated": True,
                    "created_at": created_at,
                    "external_model_calls": 0,
                }
            )
        return output

    def search_one(
        query: str,
    ) -> tuple[list[Any], str, list[dict[str, Any]], dict[str, int]]:
        remaining_ms = _require_remaining_budget(deadline_at, "search")
        search_kwargs: dict[str, Any] = {
            "query": query,
            "top_n": max(policy.max_pages * 3, 8),
            "semantic": policy.semantic,
            "fusion_weights": {
                "anchor": policy.fusion_anchor,
                "bm25": policy.fusion_bm25,
                "semantic": policy.fusion_semantic,
                "graph": policy.fusion_graph,
                "context": policy.fusion_context,
                "usage_prior": policy.fusion_usage_prior,
                "bm25_score_bonus": policy.fusion_bm25_score_bonus,
                "bm25_rank_bonus": policy.fusion_bm25_rank_bonus,
                "bm25_rank_decay": policy.fusion_bm25_rank_decay,
                "semantic_min_top_score": policy.fusion_semantic_min_top_score,
                "semantic_min_margin": policy.fusion_semantic_min_margin,
                "semantic_low_confidence_weight": (
                    policy.fusion_semantic_low_confidence_weight
                ),
                "usage_prior_decay": policy.fusion_usage_prior_decay,
                "usage_prior_cap": policy.fusion_usage_prior_cap,
            },
        }
        if remaining_ms is not None:
            search_kwargs["semantic_timeout_ms"] = remaining_ms
        if request is not None and request.session_id:
            search_kwargs["rollout_key"] = request.session_id
        try:
            results, mode = run_search(**search_kwargs)
        except BaseException:
            failed_trace = last_search_trace()
            failed_timings = failed_trace.get("stage_timings_ms")
            if stage_timings_ms is not None and isinstance(failed_timings, dict):
                for name, elapsed_ms in failed_timings.items():
                    if isinstance(name, str) and isinstance(elapsed_ms, int):
                        stage_timings_ms[name] = (
                            stage_timings_ms.get(name, 0) + elapsed_ms
                        )
            raise
        actual_trace = last_search_trace()
        actual_paths = actual_trace.get("paths")
        rows = trace_rows(
            query,
            actual_paths if isinstance(actual_paths, dict) else {},
            shadow=False,
            query_plan=str(actual_trace.get("query_plan") or ""),
        )
        diagnostic_remaining_ms = _remaining_budget_ms(deadline_at)
        if diagnostic_remaining_ms is None or diagnostic_remaining_ms > 0:
            try:
                from chronovisor.core.knowledge_graph_retrieval import (
                    shadow_candidate_paths,
                )

                shadow_paths = shadow_candidate_paths(
                    [str(result.page_id) for result in results[:20]], query=query
                )
            except Exception:
                shadow_paths = {}
        else:
            shadow_paths = {}
        rows.extend(trace_rows(query, shadow_paths, shadow=True))
        timings = actual_trace.get("stage_timings_ms")
        return results, mode, rows, timings if isinstance(timings, dict) else {}

    # Recall rewrites produce up to three independent entrances. Offline
    # callers may run them together so the semantic service can micro-batch;
    # deadline-bound hooks keep them on the main thread for hard interruption.
    if deadline_at is not None:
        # Keep deadline-bound work on the main thread: executor shutdown would
        # otherwise wait for a timed-out worker and defeat the absolute budget.
        try:
            with recall_wall_clock_deadline(
                _require_remaining_budget(deadline_at, "search") or 0
            ):
                searched = [search_one(query) for query in queries]
        except RecallWallClockTimeout as exc:
            raise RecallBudgetExhausted("recall search budget exhausted") from exc
    elif len(queries) == 1:
        searched = [search_one(queries[0])]
    else:
        with ThreadPoolExecutor(
            max_workers=min(len(queries), policy.max_queries),
            thread_name_prefix="recall-search",
        ) as executor:
            searched = list(executor.map(search_one, queries))

    _require_remaining_budget(deadline_at, "search merge")
    merged: dict[str, Any] = {}
    mode = "bm25"
    typed_trace_rows: list[dict[str, Any]] = []
    for query_index, (results, search_mode, trace_values, timings) in enumerate(searched):
        typed_trace_rows.extend(trace_values)
        if stage_timings_ms is not None:
            for name, elapsed_ms in timings.items():
                if isinstance(name, str) and isinstance(elapsed_ms, int):
                    stage_timings_ms[name] = (
                        stage_timings_ms.get(name, 0) + elapsed_ms
                    )
        if search_mode != "bm25":
            mode = search_mode
        query_weight = max(0.50, 1.0 - (0.25 * query_index))
        for result in results:
            if should_filter_sensitive_result(result, request):
                continue
            adjusted = replace(
                result,
                score=float(result.score)
                * query_weight
                * context_boost(result, request),
            )
            existing = merged.get(result.page_id)
            if existing is None or adjusted.score > existing.score:
                merged[result.page_id] = adjusted
    diagnostic_remaining_ms = _remaining_budget_ms(deadline_at)
    if diagnostic_rows is not None:
        diagnostic_rows.extend(typed_trace_rows)
    elif typed_trace_rows and (
        diagnostic_remaining_ms is None or diagnostic_remaining_ms > 0
    ):
        # Candidate-path telemetry is supervision input, never part of the
        # synchronous recall authority path. A read-only filesystem or a
        # damaged diagnostics ledger must therefore fail open.
        try:
            append_jsonl_durable(
                TYPED_GRAPH_TRACE_FILE, typed_trace_rows, sort_keys=True
            )
        except OSError:
            pass
    out = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    return out, mode


def collect_context(
    queries: list[str],
    decision: str,
    policy: RecallPolicy,
    *,
    request: RecallRequest | None = None,
    session_state: Any | None = None,
    pre_results: list[Any] | None = None,
    deadline_at: float | None = None,
) -> list[ContextItem]:
    if decision == "none" or not queries:
        return []

    if pre_results is None:
        _require_remaining_budget(deadline_at, "context startup")
        startup = okf_startup_status(CHRONOVISOR_ROOT)
        if not (startup.allowed and startup.layout == "okf_v0_2"):
            _require_remaining_budget(deadline_at, "context startup")
            init_chronovisor()
        _require_remaining_budget(deadline_at, "context store")
        store = get_store()
        _require_remaining_budget(deadline_at, "context store refresh")
        store.refresh_if_stale()

    items: list[ContextItem] = []
    seen: set[str] = set()
    _require_remaining_budget(deadline_at, "context hints")
    for page_id in query_hint_page_ids(queries, limit=policy.max_pages):
        _require_remaining_budget(deadline_at, "context hint")
        if page_id in seen:
            continue
        hinted = context_item_from_page_id(page_id, queries, decision, score=1.0)
        if hinted is None:
            continue
        if should_filter_sensitive_result(hinted, request):
            continue
        seen.add(page_id)
        items.append(hinted)
        if len(items) >= policy.max_pages:
            return items

    results = pre_results
    if results is None:
        results, _mode = search_candidates(
            queries,
            policy,
            request=request,
            deadline_at=deadline_at,
        )
    for result in results:
        _require_remaining_budget(deadline_at, "context page")
        if result.page_id in seen:
            continue
        if should_skip_session_page(session_state, result.page_id, result.updated):
            seen.add(result.page_id)
            continue
        if should_filter_sensitive_result(result, request):
            seen.add(result.page_id)
            continue
        seen.add(result.page_id)
        snippets = [result.snippet] if result.snippet else []
        if not snippets and policy.context_style == "cards":
            _require_remaining_budget(deadline_at, "context summary")
            summary = page_summary(result.page_id)
            if summary:
                snippets = [summary]
        if decision == "read":
            _require_remaining_budget(deadline_at, "context excerpt")
            snippet = excerpt_page(result.page_id, queries, max_chars=650)
            if snippet:
                snippets = [snippet]
        _require_remaining_budget(deadline_at, "context page metadata")
        items.append(
            ContextItem(
                page_id=result.page_id,
                title=result.title,
                updated=result.updated,
                score=round(result.score, 4),
                uid=page_uid_for_id(result.page_id),
                snippets=snippets,
                sensitivity=getattr(result, "sensitivity", "normal") or "normal",
            )
        )
        if len(items) >= policy.max_pages:
            return items

    # Prefetch is speculative exposure/usage history.  It may fill an empty
    # result set, but it must never displace direct evidence from the current
    # normalized query.
    _require_remaining_budget(deadline_at, "context prefetch")
    for page_id in prefetch_page_ids_for_request(
        request, queries, limit=policy.max_pages
    ):
        _require_remaining_budget(deadline_at, "context prefetch page")
        if page_id in seen:
            continue
        prefetched = context_item_from_page_id(page_id, queries, decision, score=0.95)
        if prefetched is None:
            continue
        if should_filter_sensitive_result(prefetched, request):
            continue
        seen.add(page_id)
        items.append(prefetched)
        if len(items) >= policy.max_pages:
            return items
    return items


def collect_certified_context(
    query: str,
    policy: RecallPolicy,
    *,
    request: RecallRequest,
    session_state: Any | None,
    candidates: list[Any],
    reranker_metadata: dict[str, Any] | None,
    deadline_at: float | None,
) -> tuple[list[ContextItem], dict[str, Any]]:
    """Select first, then suppress unchanged session pages without backfill."""

    from chronovisor.recall.recall_processor import select_certified_candidates

    remaining_ms = _require_remaining_budget(deadline_at, "certified context")
    judge_timeout_ms = (
        max(0, min(1_800, remaining_ms - 100)) if remaining_ms is not None else 1_800
    )
    selections, metadata = select_certified_candidates(
        query,
        candidates,
        reranker_metadata=reranker_metadata,
        max_candidates=policy.processor_max_candidates,
        max_pointer_cards=policy.processor_max_pointer_cards,
        max_rich_evidence=policy.processor_max_rich_evidence,
        injection_token_budget=policy.processor_injection_token_budget,
        certificate_required=policy.processor_certificate_required,
        judge_policy=policy if policy.processor_judge_enabled else None,
        judge_timeout_ms=judge_timeout_ms,
    )
    items: list[ContextItem] = []
    suppressed: list[str] = []
    for selection in selections:
        candidate = selection.candidate
        certificate = selection.certificate
        if should_skip_session_page(
            session_state,
            str(candidate.page_id),
            str(getattr(candidate, "updated", "") or ""),
        ):
            suppressed.append(str(candidate.page_id))
            continue
        if should_filter_sensitive_result(candidate, request):
            suppressed.append(str(candidate.page_id))
            continue
        items.append(
            ContextItem(
                page_id=str(candidate.page_id),
                title=str(getattr(candidate, "title", "") or candidate.page_id),
                updated=str(getattr(candidate, "updated", "") or ""),
                score=round(float(getattr(candidate, "score", 0.0) or 0.0), 4),
                uid=page_uid_for_id(str(candidate.page_id)),
                snippets=[certificate.supporting_span]
                if selection.evidence_kind == "rich"
                else [],
                sensitivity=str(
                    getattr(candidate, "sensitivity", "normal") or "normal"
                ),
                certificate_id=certificate.certificate_id,
                evidence_kind=selection.evidence_kind,
                source_line=certificate.source_line,
            )
        )
    metadata["session_suppressed_page_ids"] = suppressed
    metadata["committed_count"] = len(items)
    metadata["committed_page_ids"] = [item.page_id for item in items]
    return items, metadata


def observe_processor_shadow(
    query: str,
    policy: RecallPolicy,
    *,
    request: RecallRequest,
    session_state: Any | None,
    candidates: list[Any],
    reranker_metadata: dict[str, Any] | None,
    deadline_at: float | None,
) -> dict[str, Any]:
    """Produce certificates and comparisons without changing injection."""

    remaining_ms = _remaining_budget_ms(deadline_at)
    if not policy.processor_shadow_enabled:
        return {"status": "disabled", "shadow_only": True}
    if not candidates:
        return {"status": "skipped", "reason": "no_candidates", "shadow_only": True}
    if remaining_ms is not None and remaining_ms < 100:
        return {
            "status": "skipped",
            "reason": "insufficient_budget",
            "shadow_only": True,
        }
    try:
        shadow_policy = replace(policy, processor_judge_enabled=False)
        _items, metadata = collect_certified_context(
            query,
            shadow_policy,
            request=request,
            session_state=session_state,
            candidates=candidates,
            reranker_metadata=reranker_metadata,
            deadline_at=deadline_at,
        )
        metadata["authority"] = "teacher"
        metadata["shadow_only"] = True
        return metadata
    except Exception as exc:
        return {
            "status": "error",
            "reason": type(exc).__name__,
            "authority": "teacher",
            "shadow_only": True,
        }


def query_hint_page_ids(queries: list[str], *, limit: int) -> list[str]:
    try:
        from chronovisor.ingest.recall_hints import matching_hint_page_ids

        return matching_hint_page_ids(queries, limit=limit)
    except Exception:
        return []


def prefetch_page_ids_for_request(
    request: RecallRequest | None, queries: list[str], *, limit: int
) -> list[str]:
    if request is None:
        return []
    try:
        from chronovisor.core.prefetch import prefetch_page_ids

        return prefetch_page_ids(
            host=request.host,
            cwd=request.cwd,
            queries=queries,
            prompt=request.prompt,
            limit=limit,
        )
    except Exception:
        return []


def context_item_from_page_id(
    page_id: str, queries: list[str], decision: str, *, score: float
) -> ContextItem | None:
    path = find_readable_page(page_id)
    if not path or not path.exists():
        return None
    title = page_id
    updated = ""
    sensitivity = "normal"
    page_uid = ""
    try:
        from chronovisor.core.frontmatter import parse as parse_frontmatter

        meta, _body = parse_frontmatter(path.read_text(encoding="utf-8"))
        if isinstance(meta.get("title"), str):
            title = meta["title"]
        if meta.get("updated") is not None:
            updated = str(meta["updated"])
        if isinstance(meta.get("sensitivity"), str):
            sensitivity = meta["sensitivity"].strip().lower() or "normal"
        if isinstance(meta.get("uid"), str):
            page_uid = meta["uid"]
        elif path.parent.name == "career":
            sensitivity = "high"
    except OSError:
        return None
    except Exception:
        pass
    snippets: list[str] = []
    if decision == "read":
        snippet = excerpt_page(page_id, queries, max_chars=650)
        if snippet:
            snippets = [snippet]
    return ContextItem(
        page_id=page_id,
        title=title,
        updated=updated,
        score=score,
        uid=page_uid or page_uid_for_id(page_id),
        snippets=snippets,
        sensitivity=sensitivity,
    )


def page_uid_for_id(page_id: str) -> str:
    """Resolve the durable UID without changing the legacy page-id API."""

    try:
        row = PageRegistry(CHRONOVISOR_ROOT).resolve(page_id)
    except Exception:
        row = None
    return str(row.get("uid") or "") if isinstance(row, dict) else ""


def should_skip_session_page(
    session_state: Any | None, page_id: str, updated: str
) -> bool:
    if session_state is None:
        return False
    try:
        from chronovisor.recall.recall_session import should_skip_page

        return should_skip_page(session_state, page_id, updated)
    except Exception:
        return False


def page_summary(page_id: str) -> str:
    path = find_readable_page(page_id)
    if not path or not path.exists():
        return ""
    try:
        from chronovisor.core.frontmatter import parse as parse_frontmatter

        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    summary = meta.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    for line in body.splitlines():
        line = line.strip(" #-\t")
        if line:
            return _one_line(line, limit=220)
    return ""


def excerpt_page(page_id: str, queries: list[str], max_chars: int = 650) -> str:
    path = find_readable_page(page_id)
    if not path or not path.exists():
        return ""
    try:
        content = path.read_text()
    except (OSError, UnicodeDecodeError):
        return ""
    body = strip_frontmatter(content)
    terms = excerpt_terms(queries)
    body_lower = body.lower()
    idx = best_excerpt_index(body_lower, terms, max_chars=max_chars)
    if idx < 0:
        return _trim_text(body, max_chars)
    start = max(0, idx - 180)
    end = min(len(body), start + max_chars)
    return _trim_text(body[start:end], max_chars)


def excerpt_terms(queries: list[str]) -> list[str]:
    generic = {
        "llm",
        "wiki",
        "chronovisor",
        "の",
        "と",
        "を",
        "に",
        "で",
        "したい",
        "思い出したい",
    }
    seen: set[str] = set()
    out: list[str] = []
    for query in queries:
        for token in re.findall(
            r"[a-z0-9_+.-]{2,}|[\u3040-\u30ff\u3400-\u9fff]{2,}", query.lower()
        ):
            token = token.strip("。、.?!！？」』")
            if len(token) < 2 or token in generic or token in seen:
                continue
            seen.add(token)
            out.append(token)
    return sorted(out, key=len, reverse=True)[:30]


def best_excerpt_index(body_lower: str, terms: list[str], *, max_chars: int) -> int:
    best_idx = -1
    best_score = 0
    positions: list[int] = []
    for term in terms:
        start = 0
        while True:
            idx = body_lower.find(term, start)
            if idx < 0:
                break
            positions.append(idx)
            start = idx + max(1, len(term))
            if len(positions) >= 200:
                break
        if len(positions) >= 200:
            break
    for idx in positions:
        window = body_lower[max(0, idx - 180) : idx + max_chars]
        score = sum(len(term) for term in terms if term in window)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def find_readable_page(page_id: str, *, root: Path | None = None) -> Path | None:
    """Resolve one root-bound, uniquely identified stable canonical page."""

    wiki_root = root or CHRONOVISOR_ROOT
    registry = PageRegistry(wiki_root)
    if registry.path.exists():
        try:
            return registry.path_for(page_id)
        except PageRegistryError:
            return None
    pages_dir = wiki_root / "pages"
    system_dir = wiki_root / "system"
    return canonical_document_path_for_id(
        page_id,
        pages_dir=pages_dir,
        system_dir=system_dir,
    )


def strip_frontmatter(content: str) -> str:
    from chronovisor.core.frontmatter import parse as parse_frontmatter

    _meta, body = parse_frontmatter(content)
    return body.strip()


def _trim_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _recall_payload(result: RecallResult, policy: RecallPolicy) -> dict[str, Any]:
    return _recall_publication._recall_payload(
        result,
        policy,
        page_summary=page_summary,
    )


def format_recall_context(result: RecallResult, policy: RecallPolicy) -> str:
    return _recall_publication.format_recall_context(
        result,
        policy,
        page_summary=page_summary,
    )


def state_context_for_request(request: RecallRequest, policy: RecallPolicy) -> str:
    if not should_inject_state(request.host):
        return ""
    try:
        return format_state_context(
            host=request.host,
            cwd=request.cwd,
            max_chars=policy.max_state_context_chars,
        )
    except TypeError:
        return format_state_context(host=request.host, cwd=request.cwd)


def _remaining_budget_ms(deadline_at: float | None) -> int | None:
    if deadline_at is None:
        return None
    return max(0, int((deadline_at - time.monotonic()) * 1000))


def _require_remaining_budget(deadline_at: float | None, stage: str) -> int | None:
    remaining_ms = _remaining_budget_ms(deadline_at)
    if remaining_ms is not None and remaining_ms <= 0:
        raise RecallBudgetExhausted(f"recall {stage} budget exhausted")
    return remaining_ms


def _stage_started(
    telemetry: dict[str, Any] | None,
    stage: str,
    deadline_at: float | None,
) -> None:
    if telemetry is not None:
        telemetry["last_stage_started"] = stage
        telemetry["remaining_ms"] = _remaining_budget_ms(deadline_at)
        telemetry.setdefault("_stage_timers", {})[stage] = time.monotonic()


def _stage_completed(
    telemetry: dict[str, Any] | None,
    stage: str,
    deadline_at: float | None,
) -> None:
    if telemetry is not None:
        telemetry["last_stage_completed"] = stage
        telemetry["remaining_ms"] = _remaining_budget_ms(deadline_at)
        started_at = telemetry.get("_stage_timers", {}).pop(stage, None)
        if started_at is not None:
            stage_timings = telemetry.setdefault("stage_timings_ms", {})
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            stage_timings[stage] = stage_timings.get(stage, 0) + elapsed_ms
        if not telemetry.get("_stage_timers"):
            telemetry.pop("_stage_timers", None)


def _stage_interrupted(telemetry: dict[str, Any] | None) -> None:
    """Preserve elapsed time for stages interrupted by timeout or error."""

    if telemetry is None:
        return
    timers = telemetry.pop("_stage_timers", {})
    now = time.monotonic()
    if isinstance(timers, dict):
        timings = telemetry.setdefault("stage_timings_ms", {})
        for stage, started_at in timers.items():
            if isinstance(stage, str) and isinstance(started_at, int | float):
                timings[stage] = timings.get(stage, 0) + max(
                    0, int((now - float(started_at)) * 1000)
                )


def _fail_open_recall_budget(
    reason: str,
    matched: dict[str, list[str]] | None,
    request: RecallRequest,
    policy: RecallPolicy,
    started: float,
    final_deadline_at: float,
    allow_timeout_fallback: bool,
    perform_search: bool,
    telemetry: dict[str, Any] | None = None,
) -> RecallResult:
    """Return deterministic fallback context or an explicit fail-open timeout."""

    _stage_interrupted(telemetry)
    if allow_timeout_fallback and perform_search:
        remaining_ms = _remaining_budget_ms(final_deadline_at)
        if remaining_ms is not None and remaining_ms >= 100:
            if telemetry is not None:
                telemetry["fallback_started"] = True
            fallback = run_deterministic_fallback(
                request,
                policy,
                perform_search=True,
                timeout_ms=remaining_ms,
                reason=reason,
                _started_at=started,
                _final_deadline_at=final_deadline_at,
                _telemetry=telemetry,
            )
            if telemetry is not None:
                telemetry["fallback_completed"] = True
                _merge_telemetry(fallback.evidence_features, telemetry)
            fallback.latency_ms = _elapsed_ms(started)
            return fallback
    result = RecallResult(
        status="timeout",
        decision="none",
        confidence=0.0,
        queries=[],
        reasons=[reason, "recall budget exhausted; fail-open"],
        matched_terms=matched or {},
        latency_ms=_elapsed_ms(started),
        error=reason,
        decision_id=request.decision_id or new_decision_id(),
    )
    _merge_telemetry(result.evidence_features, telemetry)
    if policy.log_decisions:
        append_recall_log(request, result)
    return result


def observe_evidence_reconstruction(
    result: RecallResult,
    *,
    request: RecallRequest,
    policy: RecallPolicy,
    deadline_at: float | None,
) -> dict[str, Any]:
    """Observe projection-only evidence and keep the page teacher on any gap."""

    metadata: dict[str, Any] = {
        "status": "skipped",
        "authority": "teacher",
        "mode": "shadow",
        "canary_percent": 0,
    }
    if not result.context_items or not request.session_id:
        metadata["reason"] = "cold_candidate"
        return metadata
    if threading.current_thread() is not threading.main_thread():
        metadata["reason"] = "non_main_thread"
        return metadata
    if should_filter_sensitive_result(
        replace(result.context_items[0], sensitivity="high"),
        request,
    ):
        metadata["reason"] = "sensitive_context"
        return metadata
    remaining_ms = _remaining_budget_ms(deadline_at)
    evidence_budget_ms = (
        remaining_ms if remaining_ms is not None else policy.total_timeout_ms
    ) - 25
    if evidence_budget_ms <= 0:
        metadata["reason"] = "insufficient_budget"
        return metadata
    try:
        with recall_wall_clock_deadline(evidence_budget_ms):
            from chronovisor.recall.recall_field_schema import session_hash

            observed = _evidence_provider.observe(
                root=CHRONOVISOR_ROOT,
                prompt=request.prompt,
                session_key=session_hash(request.host, request.session_id),
                deadline_ms=evidence_budget_ms,
            )
            if observed is None:
                metadata["reason"] = "provider_unavailable"
                return metadata
            packet, provider_metadata = observed
            metadata.update(provider_metadata)
            if packet is not None:
                result.evidence_packet = packet
                result.evidence_features["evidence_reconstruction"] = metadata
            return metadata
    except RecallWallClockTimeout:
        result.evidence_packet = None
        metadata.update(status="fallback", reason="deadline_exceeded")
        return metadata
    except Exception as exc:
        result.evidence_packet = None
        metadata.update(status="fallback", reason=type(exc).__name__)
        return metadata


def bind_evidence_provider(
    observer: _evidence_provider.Observer,
    payload_builder: _evidence_provider.PayloadBuilder,
) -> None:
    _evidence_provider.bind(observer, payload_builder)


def _finalize_recall_result(
    result: RecallResult,
    *,
    request: RecallRequest,
    active_request: RecallRequest,
    policy: RecallPolicy,
    session_state: Any,
    queries: list[str],
    deadline_at: float | None = None,
    telemetry: dict[str, Any] | None = None,
) -> RecallResult:
    """Attach bounded context, advance session state, and append one decision log."""

    _stage_started(telemetry, "finalize", deadline_at)
    if request.decision_id:
        result.decision_id = request.decision_id
    recall_context = format_recall_context(result, policy)
    if result.evidence_packet is not None and not recall_context:
        result.evidence_packet = None
        evidence = result.evidence_features.get("evidence_reconstruction")
        if isinstance(evidence, dict):
            evidence.update(
                status="fallback", authority="teacher", reason="context_budget"
            )
        recall_context = format_recall_context(result, policy)
    result.state_context = state_context_for_request(active_request, policy)
    result.context = merge_context_blocks(
        result.state_context,
        recall_context,
        max_chars=policy.max_total_context_chars,
    )
    retained_page_ids = (
        []
        if result.evidence_packet is not None
        else _retained_context_page_ids(recall_context)
        if recall_context and recall_context in result.context
        else []
    )
    retained = set(retained_page_ids)
    result.context_items = [
        item for item in result.context_items if item.page_id in retained
    ]
    if result.state_context:
        result.reasons.extend(["core memory injected", "state register injected"])
    if session_state is not None:
        try:
            from chronovisor.recall.recall_session import update_session_after_recall

            update_session_after_recall(
                session_state,
                queries=queries,
                page_ids=[item.page_id for item in result.context_items],
                page_updated={
                    item.page_id: item.updated for item in result.context_items
                },
            )
        except Exception:
            pass
    if result.context_items and active_request.session_id:
        try:
            from chronovisor.recall.recall_field import queue_teacher_commits

            processor_features = result.evidence_features.get("processor")
            if not isinstance(processor_features, dict):
                processor_features = result.evidence_features.get("processor_shadow")
            if not isinstance(processor_features, dict):
                processor_features = {}

            result.evidence_features["field_teacher_queue"] = queue_teacher_commits(
                host=active_request.host,
                session_id=active_request.session_id,
                page_ids=[item.page_id for item in result.context_items],
                certificate_ids={
                    item.page_id: item.certificate_id
                    for item in result.context_items
                    if item.certificate_id
                },
                ranking_components=(processor_features.get("ranking_components", {})),
            )
        except Exception as exc:
            result.evidence_features["field_teacher_queue"] = {
                "status": "error",
                "reason": type(exc).__name__,
            }
    field_metadata = result.evidence_features.get("field_shadow")
    observer = (
        field_metadata.get("candidate_observer")
        if isinstance(field_metadata, dict)
        else None
    )
    if (
        isinstance(field_metadata, dict)
        and isinstance(observer, dict)
        and observer.get("status") in {"fallback", "observed", "active"}
        and active_request.session_id
        and (
            (remaining_ms := _remaining_budget_ms(deadline_at)) is None
            or remaining_ms > 0
        )
    ):
        try:
            from chronovisor.recall.recall_field_candidate import (
                append_candidate_trace,
            )

            append_candidate_trace(
                session_hash=str(field_metadata.get("session_hash") or ""),
                prompt=active_request.prompt,
                observer=observer,
                committed_page_ids=[item.page_id for item in result.context_items],
                latency_ms=result.latency_ms,
            )
        except Exception:
            pass
    _stage_completed(telemetry, "finalize", deadline_at)
    _merge_telemetry(result.evidence_features, telemetry)
    if policy.log_decisions:
        append_recall_log(request, result)
    return result


def _initial_recall_skip(
    request: RecallRequest,
    policy: RecallPolicy,
    *,
    started: float,
) -> RecallResult | None:
    """Return the two side-effect-free preflight dispositions."""

    if not policy.enabled:
        return RecallResult(
            status="disabled",
            decision="none",
            confidence=0.0,
            queries=[],
            reasons=["recall disabled"],
            matched_terms={},
            latency_ms=_elapsed_ms(started),
        )
    if not request.prompt.strip():
        return RecallResult(
            status="skipped",
            decision="none",
            confidence=0.0,
            queries=[],
            reasons=["empty prompt"],
            matched_terms={},
            latency_ms=_elapsed_ms(started),
        )
    return None


@dataclass
class _EvidenceSearchOutcome:
    score: float
    session_state: Any
    pre_results: list[Any]
    search_mode: str
    evidence_features: dict[str, Any]
    rewrite_queries: list[str]
    reranker_metadata: dict[str, Any]
    field_shadow_metadata: dict[str, Any]
    post_authority: dict[str, Any]


def _merge_search_stage_timings(
    evidence_features: dict[str, Any],
    trace: dict[str, Any] | None,
) -> None:
    """Merge anonymous per-query pipeline stage timings into evidence features."""
    if not isinstance(trace, dict):
        return
    search_timings = trace.get("stage_timings_ms")
    if not isinstance(search_timings, dict):
        return
    merged = evidence_features.setdefault("stage_timings_ms", {})
    for name, ms in search_timings.items():
        if isinstance(ms, int):
            merged[name] = merged.get(name, 0) + ms


def _merge_telemetry(
    evidence_features: dict[str, Any], telemetry: dict[str, Any] | None
) -> None:
    if telemetry is None:
        return
    timings = telemetry.get("stage_timings_ms")
    evidence_features.update(
        {
            key: value
            for key, value in telemetry.items()
            if key != "stage_timings_ms" and not key.startswith("_")
        }
    )
    if isinstance(timings, dict):
        previously_merged = telemetry.get("_merged_stage_timings_ms")
        if not isinstance(previously_merged, dict):
            previously_merged = {}
        _merge_search_stage_timings(
            evidence_features,
            {
                "stage_timings_ms": {
                    name: elapsed_ms - int(previously_merged.get(name, 0) or 0)
                    for name, elapsed_ms in timings.items()
                    if isinstance(name, str)
                    and isinstance(elapsed_ms, int)
                    and (
                        name not in previously_merged
                        or elapsed_ms > int(previously_merged.get(name, 0) or 0)
                    )
                }
            },
        )
        telemetry["_merged_stage_timings_ms"] = dict(timings)


def _run_evidence_search(
    *,
    active_request: RecallRequest,
    policy: RecallPolicy,
    matched: dict[str, list[str]],
    heuristic_score: float,
    reasons: list[str],
    deadline_at: float,
    processor_authority: bool,
    _telemetry: dict[str, Any] | None = None,
) -> _EvidenceSearchOutcome:
    """Run Field observation, teacher search, rewrite, rerank, and evidence score."""

    from chronovisor.recall.recall_field_candidate import (
        effective_rollout,
        run_candidate_teacher_pair,
    )
    from chronovisor.recall.recall_field_schema import load_recall_field_config
    from chronovisor.recall.recall_session import (
        cleanup_sessions,
        load_session_state,
        session_summary,
    )

    _require_remaining_budget(deadline_at, "evidence search")
    _stage_started(_telemetry, "cleanup", deadline_at)
    cleanup_sessions(policy.session_ttl_seconds)
    _stage_completed(_telemetry, "cleanup", deadline_at)
    _require_remaining_budget(deadline_at, "session load")
    _stage_started(_telemetry, "session_load", deadline_at)
    session_state = load_session_state(active_request.session_id)
    _stage_completed(_telemetry, "session_load", deadline_at)
    field_config = effective_rollout(load_recall_field_config())
    field_authority_path = field_config.mode == "active"
    field_metadata: dict[str, Any]
    if field_authority_path:
        _stage_started(_telemetry, "field", deadline_at)
        try:
            from chronovisor.recall.recall_field import run_field_turn

            field_metadata = run_field_turn(
                host=active_request.host,
                session_id=active_request.session_id,
                prompt=active_request.prompt,
            )
        except Exception as exc:
            field_metadata = {
                "status": "error",
                "reason": type(exc).__name__,
            }
        _stage_completed(_telemetry, "field", deadline_at)
    else:
        field_metadata = {
            "status": "deferred",
            "mode": field_config.mode,
            "reason": "post_authority",
        }
    field_metadata["recall_compiler"] = {
        "status": "deferred",
        "reason": "post_authority",
        "page_ids": [],
    }

    initial_queries = build_queries(
        active_request,
        matched,
        [],
        policy,
        session_state=session_state,
    )
    search_stage_timings: dict[str, int] = {}
    deferred_diagnostic_rows: list[dict[str, Any]] = []
    _stage_started(_telemetry, "teacher", deadline_at)

    def teacher_search() -> tuple[list[Any], str]:
        try:
            return search_candidates(
                initial_queries,
                policy,
                request=active_request,
                deadline_at=deadline_at,
                stage_timings_ms=search_stage_timings,
                diagnostic_rows=deferred_diagnostic_rows,
            )
        except BaseException:
            if _telemetry is not None:
                partial = _telemetry.setdefault("stage_timings_ms", {})
                for name, elapsed_ms in search_stage_timings.items():
                    partial[name] = partial.get(name, 0) + elapsed_ms
            raise

    if field_authority_path:
        pre_results, search_mode, candidate_metadata = run_candidate_teacher_pair(
            query=active_request.prompt,
            field_turn=field_metadata,
            teacher_search=teacher_search,
            timeout_ms=max(
                25,
                min(650, _remaining_budget_ms(deadline_at) or 650),
            ),
            config=field_config,
            certificate_boundary_enabled=processor_authority,
        )
    else:
        pre_results, search_mode = teacher_search()
        candidate_metadata = {
            "status": "deferred",
            "authority": "teacher",
            "reason": "post_authority",
        }
    teacher_results = list(pre_results)
    teacher_search_mode = search_mode
    _stage_completed(_telemetry, "teacher", deadline_at)
    field_metadata["candidate_observer"] = candidate_metadata
    evidence_features = build_evidence_features(
        request=active_request,
        matched=matched,
        heuristic_score=heuristic_score,
        results=pre_results,
        search_mode=search_mode,
    )
    _merge_search_stage_timings(evidence_features, {"stage_timings_ms": search_stage_timings})
    evidence_features["field_shadow"] = field_metadata
    preliminary_score = evidence_score(evidence_features, policy)
    evidence_features["evidence_score"] = preliminary_score
    rewrite_queries: list[str] = []
    rewrite_confidence = 0.0
    rewrite_metrics: dict[str, Any] = {}
    if should_rewrite_query(
        request=active_request,
        matched=matched,
        policy=policy,
        preliminary_features=evidence_features,
    ):
        _stage_started(_telemetry, "rewrite", deadline_at)
        rewrite_started = time.monotonic()
        rewrite_timeout_ms = _require_remaining_budget(deadline_at, "rewrite")
        rewrite_queries, rewrite_confidence, rewrite_reason = run_query_rewriter(
            active_request,
            matched,
            policy,
            session_summary(session_state),
            timeout_ms=rewrite_timeout_ms,
        )
        _stage_completed(_telemetry, "rewrite", deadline_at)
        rewrite_metrics = {
            "rewrite_attempted": True,
            "rewrite_latency_ms": _elapsed_ms(rewrite_started),
            "rewrite_reason": rewrite_reason,
            "rewrite_status": (
                "ok"
                if rewrite_queries
                else "fallback"
                if rewrite_reason.startswith("rewrite fallback:")
                else "empty"
            ),
        }
        evidence_features.update(rewrite_metrics)
        if rewrite_reason:
            reasons.append(rewrite_reason)
        if rewrite_queries:
            reasons.append("query rewritten")
            queries_for_search = build_queries(
                active_request,
                matched,
                [],
                policy,
                session_state=session_state,
                rewrite_queries=rewrite_queries,
            )
            _stage_started(_telemetry, "teacher", deadline_at)

            def rewritten_teacher_search() -> tuple[list[Any], str]:
                try:
                    return search_candidates(
                        queries_for_search,
                        policy,
                        request=active_request,
                        deadline_at=deadline_at,
                        stage_timings_ms=search_stage_timings,
                        diagnostic_rows=deferred_diagnostic_rows,
                    )
                except BaseException:
                    if _telemetry is not None:
                        partial = _telemetry.setdefault("stage_timings_ms", {})
                        for name, elapsed_ms in search_stage_timings.items():
                            partial[name] = partial.get(name, 0) + elapsed_ms
                    raise

            if field_authority_path:
                pre_results, search_mode, candidate_metadata = run_candidate_teacher_pair(
                    query=" ".join(queries_for_search),
                    field_turn=field_metadata,
                    teacher_search=rewritten_teacher_search,
                    timeout_ms=max(
                        25,
                        min(650, _remaining_budget_ms(deadline_at) or 650),
                    ),
                    config=field_config,
                    certificate_boundary_enabled=processor_authority,
                )
            else:
                pre_results, search_mode = rewritten_teacher_search()
            teacher_results = list(pre_results)
            teacher_search_mode = search_mode
            _stage_completed(_telemetry, "teacher", deadline_at)
            field_metadata["candidate_observer"] = candidate_metadata
            evidence_features = build_evidence_features(
                request=active_request,
                matched=matched,
                heuristic_score=heuristic_score,
                results=pre_results,
                search_mode=search_mode,
                rewrite_confidence=rewrite_confidence,
            )
            _merge_search_stage_timings(
                evidence_features, {"stage_timings_ms": search_stage_timings}
            )
            evidence_features["field_shadow"] = field_metadata
            evidence_features.update(rewrite_metrics)

    reranker_metadata: dict[str, Any] = {}
    from chronovisor.core.runtime_config import load_reranker_config

    reranker_mode = load_reranker_config().service.mode
    remaining_for_reranker = _remaining_budget_ms(deadline_at)
    _stage_started(_telemetry, "reranker", deadline_at)
    if reranker_mode in {"canary", "on"}:
        if remaining_for_reranker is not None and remaining_for_reranker < 100:
            raise RecallBudgetExhausted(
                "recall authoritative reranker budget exhausted"
            )
        from chronovisor.recall.recall_processor import rank_recall_candidates

        pre_results, reranker_metadata = rank_recall_candidates(
            active_request.prompt,
            pre_results,
            timeout_ms=max(
                25,
                min(1_500, (remaining_for_reranker or 1_500) - 50),
            ),
        )
        reranker_status = str(reranker_metadata.get("status") or "")
        if (
            reranker_status in {"unavailable", "error", "failed", "fallback"}
            or reranker_metadata.get("fail_open") is True
            or reranker_metadata.get("degraded") is True
        ):
            raise RecallBudgetExhausted(
                "recall authoritative reranker unavailable"
            )
        evidence_features = build_evidence_features(
            request=active_request,
            matched=matched,
            heuristic_score=heuristic_score,
            results=pre_results,
            search_mode=search_mode,
            rewrite_confidence=rewrite_confidence,
        )
        _merge_search_stage_timings(
            evidence_features, {"stage_timings_ms": search_stage_timings}
        )
        evidence_features.update(rewrite_metrics)
        evidence_features["reranker"] = reranker_metadata
        evidence_features["field_shadow"] = field_metadata
    elif reranker_mode == "shadow":
        reranker_metadata = {
            "status": "deferred",
            "mode": "shadow",
            "reason": "post_authority",
        }
    else:
        reranker_metadata = {"status": "disabled", "mode": reranker_mode}
    _stage_completed(_telemetry, "reranker", deadline_at)
    score = evidence_score(evidence_features, policy)
    evidence_features["evidence_score"] = score
    evidence_features["decision_pre_judge"] = decision_from_score(score, policy)
    return _EvidenceSearchOutcome(
        score=score,
        session_state=session_state,
        pre_results=pre_results,
        search_mode=search_mode,
        evidence_features=evidence_features,
        rewrite_queries=rewrite_queries,
        reranker_metadata=reranker_metadata,
        field_shadow_metadata=field_metadata,
        post_authority={
            "field_deferred": not field_authority_path,
            "field_config": field_config,
            "teacher_results": teacher_results,
            "teacher_search_mode": teacher_search_mode,
            "reranker_shadow_deferred": reranker_mode == "shadow",
            "diagnostic_rows": deferred_diagnostic_rows,
        },
    )


def processor_authority_for_request(
    policy: RecallPolicy,
    request: RecallRequest,
) -> bool:
    """Select autonomous Processor authority at the same session canary."""

    if policy.processor_enabled:
        return True
    if not policy.processor_auto_enable or not request.session_id:
        return False
    try:
        from dataclasses import replace as dataclass_replace

        from chronovisor.recall.recall_field_candidate import selected_for_canary
        from chronovisor.recall.recall_field_schema import (
            load_recall_field_config,
            session_hash,
        )
        from chronovisor.recall.recall_growth import (
            automatic_processor_authority_allowed,
            automatic_rollout,
        )

        if not automatic_processor_authority_allowed(enabled=True):
            return False
        mode, percent = automatic_rollout(enabled=True)
        if mode != "active":
            return False
        config = dataclass_replace(
            load_recall_field_config(),
            mode="active",
            canary_percent=percent,
        )
        return selected_for_canary(
            session_hash(request.host, request.session_id),
            config,
        )
    except Exception:
        return False


def _run_post_authority_shadows(
    *,
    request: RecallRequest,
    policy: RecallPolicy,
    session_state: Any,
    candidates: list[Any],
    evidence_features: dict[str, Any],
    reranker_metadata: dict[str, Any],
    field_metadata: dict[str, Any],
    post_authority: dict[str, Any],
    processor_authority: bool,
    deadline_at: float,
    telemetry: dict[str, Any] | None,
) -> None:
    """Observe non-authoritative lanes after the injected ranking is fixed."""

    remaining_ms = _remaining_budget_ms(deadline_at)
    if remaining_ms is not None and remaining_ms < 100:
        return
    diagnostic_rows = post_authority.get("diagnostic_rows")
    if isinstance(diagnostic_rows, list) and diagnostic_rows:
        try:
            append_jsonl_durable(
                TYPED_GRAPH_TRACE_FILE,
                [row for row in diagnostic_rows if isinstance(row, dict)],
                sort_keys=True,
            )
        except OSError:
            pass
    if post_authority.get("field_deferred"):
        _stage_started(telemetry, "field", deadline_at)
        try:
            from chronovisor.recall.recall_field import run_field_turn
            from chronovisor.recall.recall_field_candidate import (
                run_candidate_teacher_pair,
            )

            observed = run_field_turn(
                host=request.host,
                session_id=request.session_id,
                prompt=request.prompt,
                config=post_authority.get("field_config"),
            )
            _unused, _mode, observer = run_candidate_teacher_pair(
                query=request.prompt,
                field_turn=observed,
                teacher_search=lambda: (
                    list(post_authority.get("teacher_results") or []),
                    str(post_authority.get("teacher_search_mode") or "bm25"),
                ),
                timeout_ms=max(25, min(650, _remaining_budget_ms(deadline_at) or 650)),
                config=post_authority.get("field_config"),
                certificate_boundary_enabled=False,
            )
            observed["candidate_observer"] = observer
            field_metadata.clear()
            field_metadata.update(observed)
        except Exception as exc:
            field_metadata.update(status="error", reason=type(exc).__name__)
        _stage_completed(telemetry, "field", deadline_at)
    _stage_started(telemetry, "compiler", deadline_at)
    try:
        from chronovisor.recall.recall_compiler import compile_query

        field_metadata["recall_compiler"] = compile_query(request.prompt)
    except Exception as exc:
        field_metadata["recall_compiler"] = {
            "status": "error",
            "reason": type(exc).__name__,
            "page_ids": [],
        }
    _stage_completed(telemetry, "compiler", deadline_at)
    if post_authority.get("reranker_shadow_deferred"):
        _stage_started(telemetry, "reranker_shadow", deadline_at)
        try:
            from chronovisor.recall.recall_processor import rank_recall_candidates

            _unused, observed_reranker = rank_recall_candidates(
                request.prompt,
                candidates,
                timeout_ms=max(25, min(1_500, _remaining_budget_ms(deadline_at) or 25)),
            )
            reranker_metadata.clear()
            reranker_metadata.update(observed_reranker)
        except Exception as exc:
            reranker_metadata.update(status="error", reason=type(exc).__name__)
        _stage_completed(telemetry, "reranker_shadow", deadline_at)
    if policy.processor_shadow_enabled and not processor_authority:
        _stage_started(telemetry, "processor_shadow", deadline_at)
        evidence_features["processor_shadow"] = observe_processor_shadow(
            request.prompt,
            policy,
            request=request,
            session_state=session_state,
            candidates=candidates,
            reranker_metadata=reranker_metadata,
            deadline_at=deadline_at,
        )
        _stage_completed(telemetry, "processor_shadow", deadline_at)


def _prepare_recall_request(
    request: RecallRequest,
    policy: RecallPolicy,
    *,
    started: float,
) -> tuple[RecallRequest | None, list[str], RecallResult | None]:
    """Normalize the user prompt and return any deterministic early result."""

    initial_skip = _initial_recall_skip(request, policy, started=started)
    if initial_skip is not None:
        return None, [], initial_skip

    cleaned_prompt, stripped_reasons = strip_non_user_blocks(request.prompt)
    if not cleaned_prompt:
        skip_reason = classify_non_user_prompt(request.prompt, policy)
        result = RecallResult(
            status="skipped",
            decision="none",
            confidence=0.0,
            queries=[],
            reasons=[skip_reason] if skip_reason else stripped_reasons,
            matched_terms={},
            latency_ms=_elapsed_ms(started),
        )
        if policy.log_decisions:
            append_recall_log(request, result)
        return None, stripped_reasons, result

    active_request = (
        replace(request, prompt=cleaned_prompt) if stripped_reasons else request
    )
    skip_reason = classify_non_user_prompt(active_request.prompt, policy)
    if not skip_reason:
        return active_request, stripped_reasons, None

    result = RecallResult(
        status="skipped",
        decision="none",
        confidence=0.0,
        queries=[],
        reasons=stripped_reasons + [skip_reason],
        matched_terms={},
        latency_ms=_elapsed_ms(started),
    )
    if policy.log_decisions:
        append_recall_log(request, result)
    return None, stripped_reasons, result


def _run_recall_impl(
    request: RecallRequest,
    policy: RecallPolicy | None = None,
    *,
    perform_search: bool = True,
    _allow_timeout_fallback: bool = True,
    _started_at: float | None = None,
    _final_deadline_at: float | None = None,
    _telemetry: dict[str, Any] | None = None,
) -> RecallResult:
    started = _started_at if _started_at is not None else time.monotonic()
    if not request.decision_id:
        request = replace(request, decision_id=new_decision_id())
    policy = policy or load_policy()
    policy.max_total_context_chars = max(
        policy.max_total_context_chars,
        policy.max_state_context_chars + policy.max_context_chars + 2,
    )
    final_deadline_at = (
        _final_deadline_at
        if _final_deadline_at is not None
        else started + (policy.total_timeout_ms / 1000.0)
    )
    reserve_ms = (
        max(
            0,
            min(
                policy.deterministic_fallback_reserve_ms, policy.total_timeout_ms - 100
            ),
        )
        if _allow_timeout_fallback and perform_search
        else 0
    )
    deadline_at = final_deadline_at - (reserve_ms / 1000.0)

    _stage_started(_telemetry, "prepare", deadline_at)
    active_request, stripped_reasons, early_result = _prepare_recall_request(
        request,
        policy,
        started=started,
    )
    if early_result is not None:
        _stage_completed(_telemetry, "prepare", deadline_at)
        return early_result
    assert active_request is not None
    _stage_completed(_telemetry, "prepare", deadline_at)

    processor_authority = processor_authority_for_request(
        policy,
        active_request,
    )

    score, reasons, matched = evaluate_heuristic(active_request, policy)
    reasons = stripped_reasons + reasons
    used_judge = False
    judge_confidence: float | None = None
    judge_reason = ""
    judge_queries: list[str] = []
    rewrite_queries: list[str] = []
    search_mode = ""
    evidence_features: dict[str, Any] = {}
    session_state = None
    pre_results: list[Any] = []
    reranker_metadata: dict[str, Any] = {}
    field_shadow_metadata: dict[str, Any] = {}
    post_authority: dict[str, Any] = {}

    if policy.gate_mode == "evidence" and perform_search:
        try:
            _stage_started(_telemetry, "evidence_search", deadline_at)
            evidence_outcome = _run_evidence_search(
                active_request=active_request,
                policy=policy,
                matched=matched,
                heuristic_score=score,
                reasons=reasons,
                deadline_at=deadline_at,
                processor_authority=processor_authority,
                _telemetry=_telemetry,
            )
            score = evidence_outcome.score
            session_state = evidence_outcome.session_state
            pre_results = evidence_outcome.pre_results
            search_mode = evidence_outcome.search_mode
            evidence_features = evidence_outcome.evidence_features
            rewrite_queries = evidence_outcome.rewrite_queries
            reranker_metadata = evidence_outcome.reranker_metadata
            field_shadow_metadata = evidence_outcome.field_shadow_metadata
            post_authority = evidence_outcome.post_authority
            _stage_completed(_telemetry, "evidence_search", deadline_at)
        except RecallBudgetExhausted as exc:
            return _fail_open_recall_budget(
                str(exc),
                matched,
                request,
                policy,
                started,
                final_deadline_at,
                _allow_timeout_fallback,
                perform_search,
                _telemetry,
            )
        except Exception as exc:
            _stage_interrupted(_telemetry)
            reasons.append(f"evidence gate failed: {exc.__class__.__name__}")
            pre_results = []
            search_mode = "error"

    if not processor_authority and should_run_judge(score, policy, evidence_features):
        try:
            _stage_started(_telemetry, "judge", deadline_at)
            judge_timeout_ms = _require_remaining_budget(deadline_at, "judge")
        except RecallBudgetExhausted as exc:
            return _fail_open_recall_budget(
                str(exc),
                matched,
                request,
                policy,
                started,
                final_deadline_at,
                _allow_timeout_fallback,
                perform_search,
                _telemetry,
            )
        judge_score, judge_queries, judge_reason = run_local_judge(
            active_request,
            score,
            policy,
            timeout_ms=judge_timeout_ms,
        )
        _stage_completed(_telemetry, "judge", deadline_at)
        used_judge = judge_score is not None
        if judge_score is not None:
            score = judge_score
            judge_confidence = judge_score
            if judge_reason:
                reasons.append("judge: " + judge_reason)
        elif judge_reason:
            reasons.append(judge_reason)
            resource_busy = "capacity_unavailable" in judge_reason
            if resource_busy:
                reasons.append("judge resource busy; using deterministic evidence")
            if policy.fail_silent_on_judge_unavailable and not resource_busy:
                reasons.append("judge unavailable; fail-silent")
                result = RecallResult(
                    status="skipped",
                    decision="none",
                    confidence=round(score, 3),
                    queries=[],
                    reasons=reasons,
                    matched_terms=matched,
                    used_judge=False,
                    judge_confidence=None,
                    judge_reason=judge_reason,
                    session_id=active_request.session_id,
                    evidence_features=evidence_features,
                    search_mode=search_mode,
                    context_style=policy.context_style,
                    latency_ms=_elapsed_ms(started),
                )
                return _finalize_recall_result(
                    result,
                    request=request,
                    active_request=active_request,
                    policy=policy,
                    session_state=session_state,
                    queries=[],
                    deadline_at=deadline_at,
                    telemetry=_telemetry,
                )

    decision = decision_from_score(score, policy)
    queries = (
        build_queries(
            active_request,
            matched,
            judge_queries,
            policy,
            session_state=session_state,
            rewrite_queries=rewrite_queries,
        )
        if decision != "none" or (policy.gate_mode == "evidence" and perform_search)
        else []
    )

    context_items: list[ContextItem] = []
    error = ""
    if perform_search and decision != "none":
        try:
            _stage_started(_telemetry, "context", deadline_at)
            _require_remaining_budget(deadline_at, "context")
            if processor_authority:
                context_items, processor_metadata = collect_certified_context(
                    active_request.prompt,
                    policy,
                    request=active_request,
                    session_state=session_state,
                    candidates=pre_results,
                    reranker_metadata=reranker_metadata,
                    deadline_at=deadline_at,
                )
                evidence_features["processor"] = processor_metadata
            else:
                context_items = collect_context(
                    queries,
                    decision,
                    policy,
                    request=active_request,
                    session_state=session_state,
                    pre_results=(
                        pre_results if policy.gate_mode == "evidence" else None
                    ),
                    deadline_at=deadline_at,
                )
            if not context_items:
                reasons.append("no matching pages")
            _stage_completed(_telemetry, "context", deadline_at)
        except RecallBudgetExhausted as exc:
            return _fail_open_recall_budget(
                str(exc),
                matched,
                request,
                policy,
                started,
                final_deadline_at,
                _allow_timeout_fallback,
                perform_search,
                _telemetry,
            )
        except Exception as exc:
            _stage_interrupted(_telemetry)
            error = f"{exc.__class__.__name__}: {exc}"
            reasons.append("search failed")

    result = RecallResult(
        status="ok" if not error else "error",
        decision=decision
        if context_items or decision == "none" or not perform_search
        else "none"
        if processor_authority
        else "search",
        confidence=round(score, 3),
        queries=queries,
        reasons=reasons,
        matched_terms=matched,
        session_id=active_request.session_id,
        context_items=context_items,
        used_judge=used_judge,
        judge_confidence=judge_confidence,
        judge_reason=judge_reason,
        evidence_features=evidence_features,
        search_mode=search_mode,
        context_style=policy.context_style,
        latency_ms=_elapsed_ms(started),
        error=error,
        decision_id=active_request.decision_id,
    )
    _stage_started(_telemetry, "evidence_reconstruction", deadline_at)
    evidence_features["evidence_reconstruction"] = observe_evidence_reconstruction(
        result,
        request=active_request,
        policy=policy,
        deadline_at=deadline_at,
    )
    _stage_completed(_telemetry, "evidence_reconstruction", deadline_at)
    if post_authority:
        remaining_ms = _remaining_budget_ms(deadline_at)
        if remaining_ms is None or remaining_ms >= 100:
            try:
                with recall_wall_clock_deadline(
                    max(25, (remaining_ms or policy.total_timeout_ms) - 25)
                ):
                    _run_post_authority_shadows(
                        request=active_request,
                        policy=policy,
                        session_state=session_state,
                        candidates=pre_results,
                        evidence_features=evidence_features,
                        reranker_metadata=reranker_metadata,
                        field_metadata=field_shadow_metadata,
                        post_authority=post_authority,
                        processor_authority=processor_authority,
                        deadline_at=deadline_at,
                        telemetry=_telemetry,
                    )
            except (RecallBudgetExhausted, RecallWallClockTimeout):
                _stage_interrupted(_telemetry)
                pass
    compiler_metadata = field_shadow_metadata.get("recall_compiler")
    if isinstance(compiler_metadata, dict):
        compiler_ids = {
            str(value)
            for value in compiler_metadata.get("page_ids", [])
            if isinstance(value, str)
        }
        teacher_ids = [str(item.page_id) for item in pre_results[:30]]
        committed_ids = [item.page_id for item in context_items]
        evidence_features["shadow_teacher"] = {
            "compiler_status": str(compiler_metadata.get("status") or ""),
            "compiler_page_ids": sorted(compiler_ids),
            "teacher_overlap": len(compiler_ids & set(teacher_ids)),
            "commit_overlap": len(compiler_ids & set(committed_ids)),
            "authority": "teacher",
        }
        diagnostic_remaining_ms = _remaining_budget_ms(deadline_at)
        if diagnostic_remaining_ms is None or diagnostic_remaining_ms > 0:
            try:
                from chronovisor.recall.recall_compiler import append_shadow_trace

                append_shadow_trace(
                    prompt=active_request.prompt,
                    compiler=compiler_metadata,
                    teacher_page_ids=teacher_ids,
                    committed_page_ids=committed_ids,
                )
            except Exception:
                pass
    result.latency_ms = _elapsed_ms(started)
    _merge_telemetry(result.evidence_features, _telemetry)
    result = _finalize_recall_result(
        result,
        request=request,
        active_request=active_request,
        policy=policy,
        session_state=session_state,
        queries=queries,
        deadline_at=final_deadline_at,
        telemetry=_telemetry,
    )
    if _telemetry is not None:
        _telemetry.pop("_stage_timers", None)
        _merge_telemetry(result.evidence_features, _telemetry)
    return result


def run_recall(
    request: RecallRequest,
    policy: RecallPolicy | None = None,
    *,
    perform_search: bool = True,
    _allow_timeout_fallback: bool = True,
    _telemetry: dict[str, Any] | None = None,
) -> RecallResult:
    """Run synchronous recall while preempting low-priority research work."""

    from chronovisor.core.research_scheduler import foreground_lane

    started = time.monotonic()
    policy = policy or load_policy()
    final_deadline_at = started + (policy.total_timeout_ms / 1000.0)
    try:
        _stage_started(_telemetry, "scheduler", final_deadline_at)
        with foreground_lane(preempt_grace_ms=250) as receipt:
            if _telemetry is not None:
                _telemetry["scheduler_wait_ms"] = receipt.resource_wait_ms
            _stage_completed(_telemetry, "scheduler", final_deadline_at)
            try:
                _require_remaining_budget(final_deadline_at, "scheduler")
            except RecallBudgetExhausted as exc:
                return _fail_open_recall_budget(
                    str(exc),
                    {},
                    request,
                    policy,
                    started,
                    final_deadline_at,
                    False,
                    perform_search,
                    _telemetry,
                )
            result = _run_recall_impl(
                request,
                policy,
                perform_search=perform_search,
                _allow_timeout_fallback=_allow_timeout_fallback,
                _started_at=started,
                _final_deadline_at=final_deadline_at,
                _telemetry=_telemetry,
            )
    except BaseException:
        _stage_interrupted(_telemetry)
        raise
    result.evidence_features.setdefault(
        "scheduler",
        {
            "resource_wait_ms": receipt.resource_wait_ms,
            "research_overlap": receipt.research_overlap,
            "research_preempted": receipt.preempted,
        },
    )
    if _telemetry is not None:
        _telemetry.pop("_stage_timers", None)
        _merge_telemetry(result.evidence_features, _telemetry)
    return result


def run_deterministic_fallback(
    request: RecallRequest,
    policy: RecallPolicy,
    *,
    perform_search: bool = True,
    timeout_ms: int | None = None,
    reason: str = "primary recall unavailable",
    _started_at: float | None = None,
    _final_deadline_at: float | None = None,
    _telemetry: dict[str, Any] | None = None,
) -> RecallResult:
    """Return L1 plus the existing BM25 projection without normal-path work."""

    requested_ms = (
        int(timeout_ms)
        if isinstance(timeout_ms, int)
        else policy.deterministic_fallback_reserve_ms
    )
    budget_ms = max(
        1,
        min(600, policy.deterministic_fallback_reserve_ms, requested_ms),
    )
    entered_at = time.monotonic()
    started = _started_at if _started_at is not None else time.monotonic()
    final_deadline_at = min(
        _final_deadline_at
        if _final_deadline_at is not None
        else entered_at + (budget_ms / 1000.0),
        entered_at + (budget_ms / 1000.0),
    )
    cleaned_prompt, stripped_reasons = strip_non_user_blocks(request.prompt)
    active_request = replace(request, prompt=cleaned_prompt or request.prompt)
    score, heuristic_reasons, matched = evaluate_heuristic(active_request, policy)
    decision = decision_from_score(score, policy)
    queries = [active_request.prompt] if active_request.prompt.strip() else []
    context_items: list[ContextItem] = []
    seen_page_ids: set[str] = set()
    is_main_thread = threading.current_thread() is threading.main_thread()
    try:
        remaining_ms = _require_remaining_budget(final_deadline_at, "fallback")
        with recall_wall_clock_deadline(remaining_ms or budget_ms):
            results = (
                search_existing_bm25(
                    active_request.prompt,
                    top_n=max(policy.max_pages * 3, 8),
                )
                if perform_search
                and queries
                and is_main_thread
                else []
            )
            for candidate in results:
                if len(context_items) >= policy.max_pages:
                    break
                if candidate.page_id in seen_page_ids:
                    continue
                seen_page_ids.add(str(candidate.page_id))
                if should_filter_sensitive_result(candidate, active_request):
                    continue
                item = context_item_from_page_id(
                    str(candidate.page_id),
                    queries,
                    decision if decision != "none" else "search",
                    score=round(float(candidate.score), 4),
                )
                if item is not None and not should_filter_sensitive_result(
                    item, active_request
                ):
                    context_items.append(item)
    except (RecallBudgetExhausted, RecallWallClockTimeout):
        context_items = []
    effective_decision = decision if decision != "none" else "search"
    result = RecallResult(
        status="degraded",
        decision=(effective_decision if context_items else "none"),
        confidence=round(score, 3),
        queries=queries,
        reasons=[
            f"deterministic BM25 fallback: {reason}",
            *stripped_reasons,
            *heuristic_reasons,
        ],
        matched_terms=matched,
        session_id=active_request.session_id,
        context_items=context_items,
        evidence_features={"authority": "teacher", "degraded": True},
        search_mode="bm25-fallback",
        context_style=policy.context_style,
        error=reason,
        decision_id=active_request.decision_id or new_decision_id(),
    )
    if not is_main_thread:
        result.latency_ms = _elapsed_ms(started)
        _merge_telemetry(result.evidence_features, _telemetry)
        return result
    try:
        remaining_ms = _require_remaining_budget(final_deadline_at, "fallback render")
        with recall_wall_clock_deadline(remaining_ms or budget_ms):
            result.state_context = state_context_for_request(active_request, policy)
            recall_block = format_recall_context(result, policy)
            result.context = merge_context_blocks(
                result.state_context,
                recall_block,
                max_chars=policy.max_total_context_chars,
            )
    except (RecallBudgetExhausted, RecallWallClockTimeout):
        result.context_items = []
        result.state_context = ""
        result.context = ""
        recall_block = ""
    retained = (
        set(_retained_context_page_ids(recall_block))
        if recall_block and recall_block in result.context
        else set()
    )
    result.context_items = [
        item for item in result.context_items if item.page_id in retained
    ]
    if result.state_context:
        result.reasons.extend(["core memory injected", "state register injected"])
    result.latency_ms = _elapsed_ms(started)
    _merge_telemetry(result.evidence_features, _telemetry)
    return result


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def append_recall_log(request: RecallRequest, result: RecallResult) -> None:
    page_bindings: list[dict[str, str]] = []
    for item in result.context_items:
        path = find_readable_page(item.page_id)
        try:
            content_sha = hashlib.sha256(path.read_bytes()).hexdigest() if path else ""
        except OSError:
            content_sha = ""
        page_bindings.append(
            {
                "page_id": item.page_id,
                "page_uid": item.uid,
                "content_sha256": content_sha,
            }
        )
    context_receipt = {
        "schema_version": 1,
        "renderer_protocol": "recall-result-context-v1",
        "context_style": result.context_style,
        "rendered_context": result.context,
        "rendered_context_sha256": hashlib.sha256(
            result.context.encode("utf-8")
        ).hexdigest(),
        "page_bindings": page_bindings,
    }
    context_receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            context_receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    record = {
        "schema_version": 2,
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision_id": result.decision_id,
        "stage": "injected"
        if result.context_items or result.evidence_packet is not None
        else "decision",
        "host": request.host,
        "event": request.event,
        "cwd": request.cwd,
        "session_id": request.session_id,
        "prompt_hash": stable_prompt_hash(request.prompt),
        "prompt_chars": len(request.prompt),
        "prompt_preview": request.prompt[:300],
        "decision": result.decision,
        "confidence": result.confidence,
        "queries": result.queries,
        "pages": [item.page_id for item in result.context_items],
        "page_uids": [item.uid for item in result.context_items if item.uid],
        "context_items": page_bindings,
        "context_receipt": context_receipt,
        "reasons": result.reasons,
        "used_judge": result.used_judge,
        "judge_confidence": result.judge_confidence,
        "judge_reason": result.judge_reason,
        "evidence_features": result.evidence_features,
        "stage_timings_ms": (
            result.evidence_features.get("stage_timings_ms") or {}
        ),
        "search_mode": result.search_mode,
        "context_style": result.context_style,
        "latency_ms": result.latency_ms,
        "status": result.status,
        "error": result.error,
    }
    append_jsonl(RECALL_LOG_FILE, record)
    try:
        from chronovisor.recall.recall_policy_store import append_live_episode

        live_path = (
            RECALL_LOG_FILE.parent.parent
            / "runtime"
            / "recall-improvement"
            / "live-episodes.jsonl"
        )
        append_live_episode(record, path=live_path)
    except Exception:
        pass


def recall_log_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": record.get("decision_id", ""),
        "stage": record.get("stage", ""),
        "ts": record.get("ts", ""),
        "host": record.get("host", ""),
        "event": record.get("event", ""),
        "cwd": record.get("cwd", ""),
        "session_id": record.get("session_id", ""),
        "prompt_hash": record.get("prompt_hash", ""),
        "prompt_chars": record.get("prompt_chars", 0),
        "prompt_preview": record.get("prompt_preview", ""),
        "decision": record.get("decision", ""),
        "confidence": record.get("confidence", 0.0),
        "score": record.get("confidence", 0.0),
        "queries": record.get("queries", []),
        "pages": record.get("pages", []),
        "page_uids": record.get("page_uids", []),
        "reasons": record.get("reasons", []),
        "used_judge": record.get("used_judge", False),
        "judge_confidence": record.get("judge_confidence"),
        "judge_reason": record.get("judge_reason", ""),
        "evidence_features": record.get("evidence_features", {}),
        "stage_timings_ms": record.get("stage_timings_ms", {}),
        "search_mode": record.get("search_mode", ""),
        "context_style": record.get("context_style", ""),
        "latency_ms": record.get("latency_ms", 0),
        "status": record.get("status", ""),
        "error": record.get("error", ""),
    }


def recent_recall_logs(limit: int = 10) -> list[dict[str, Any]]:
    limit = max(1, limit)
    try:
        with RECALL_LOG_FILE.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=limit)
    except OSError:
        return []
    items: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            snapshot = recall_log_snapshot(record)
            items.append(
                {
                    "decision_id": snapshot["decision_id"],
                    "ts": snapshot["ts"],
                    "host": snapshot["host"],
                    "decision": snapshot["decision"],
                    "confidence": snapshot["confidence"],
                    "prompt_hash": snapshot["prompt_hash"],
                    "prompt_preview": snapshot["prompt_preview"],
                    "queries": snapshot["queries"],
                    "pages": snapshot["pages"],
                }
            )
    return items


def find_recall_log(decision_id: str, limit: int = 5000) -> dict[str, Any] | None:
    if not decision_id:
        return None
    try:
        with RECALL_LOG_FILE.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=limit)
    except OSError:
        return None
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("decision_id") == decision_id:
            return record
    return None


def append_feedback(
    kind: str,
    note: str = "",
    prompt: str = "",
    host: str = "",
    *,
    expected_pages: list[str] | None = None,
    negative_pages: list[str] | None = None,
    expected_queries: list[str] | None = None,
    ref: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = _build_feedback_record(
        kind,
        note,
        prompt,
        host,
        expected_pages=expected_pages,
        negative_pages=negative_pages,
        expected_queries=expected_queries,
        ref=ref,
        extra=extra,
    )
    with _feedback_exclusive_lock(RECALL_FEEDBACK_FILE):
        _append_feedback_rows_lock_held(RECALL_FEEDBACK_FILE, [record])
    return record


def _build_feedback_record(
    kind: str,
    note: str = "",
    prompt: str = "",
    host: str = "",
    *,
    expected_pages: list[str] | None = None,
    negative_pages: list[str] | None = None,
    expected_queries: list[str] | None = None,
    ref: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = (
        recall_log_snapshot(found) if ref and (found := find_recall_log(ref)) else None
    )
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": kind,
        "host": host,
        "prompt": prompt,
        "note": note,
        "expected_pages": expected_pages or [],
        "negative_pages": negative_pages or [],
        "expected_queries": expected_queries or [],
        "ref": ref,
        "snapshot": snapshot,
    }
    if extra:
        reserved = set(record) & set(extra)
        if reserved:
            raise ValueError(
                "feedback extra cannot overwrite reserved fields: "
                + ", ".join(sorted(reserved))
            )
        for key, value in extra.items():
            record[key] = value
    return record


@contextmanager
def _feedback_exclusive_lock(path: Path) -> Iterator[None]:
    """Hold the non-reentrant 0600 sidecar lock for a feedback ledger.

    The shared durable-state helper does not normalize modes on sidecars that
    already exist.  Keep this stricter protocol recall-local until every lane
    can adopt that mode change together.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        # Existing runtime sidecars may predate the fixed private mode.
        os.chmod(lock_path, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _append_feedback_rows_lock_held(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = False,
) -> None:
    """Append feedback while the caller holds ``_feedback_exclusive_lock``."""

    append_jsonl_durable(path, rows, sort_keys=sort_keys)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    append_jsonl_durable(path, [record])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Chronovisor recall gate.")
    parser.add_argument(
        "--host",
        default="generic",
        help="Host adapter name: claude-code, codex, generic.",
    )
    parser.add_argument("--event", default="UserPromptSubmit")
    parser.add_argument(
        "--prompt",
        help="User prompt. If omitted with --hook, read from hook JSON stdin.",
    )
    parser.add_argument("--cwd", default=os.environ.get("PWD", ""))
    parser.add_argument("--session-id", default="")
    parser.add_argument("--config", default=str(RECALL_CONFIG_FILE))
    parser.add_argument(
        "--hook", action="store_true", help="Read hook JSON from stdin."
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Only evaluate the gate; do not search pages.",
    )
    parser.add_argument(
        "--warmup", action="store_true", help="Warm the gate/rewrite model and exit."
    )
    parser.add_argument(
        "--recent",
        nargs="?",
        type=int,
        const=10,
        help="List recent recall decisions and exit. Defaults to 10 when N is omitted.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "plain", "claude", "codex", "hook-json"],
        default="json",
    )
    parser.add_argument(
        "--feedback",
        choices=[
            "missed",
            "missed_candidate",
            "false-positive",
            "page_ignored",
            "useful",
        ],
        help="Record human feedback instead of running recall.",
    )
    parser.add_argument("--note", default="", help="Feedback note.")
    parser.add_argument(
        "--expected-page",
        action="append",
        default=[],
        help="Expected page id for missed recall.",
    )
    parser.add_argument(
        "--negative-page",
        action="append",
        default=[],
        help="Specific irrelevant page id for page-scoped feedback.",
    )
    parser.add_argument(
        "--expected-query",
        action="append",
        default=[],
        help="Expected query for missed recall.",
    )
    parser.add_argument(
        "--ref", default="", help="Decision id to attach as a feedback snapshot."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-recall`` command-line entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.recent is not None:
        print(
            json.dumps(
                {"status": "ok", "items": recent_recall_logs(args.recent)},
                ensure_ascii=False,
            )
        )
        return 0
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        with okf_runtime_operation(CHRONOVISOR_ROOT):
            return _main_locked(args)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75


def _main_locked(args: argparse.Namespace) -> int:

    if not okf_startup_status(CHRONOVISOR_ROOT).allowed:
        print(
            json.dumps({"status": "blocked", "category": "okf_startup_blocked"})
        )
        return 75

    if args.feedback:
        record = append_feedback(
            args.feedback,
            args.note,
            prompt=args.prompt or "",
            host=args.host,
            expected_pages=args.expected_page,
            negative_pages=args.negative_page,
            expected_queries=args.expected_query,
            ref=args.ref,
        )
        print(
            json.dumps({"status": "recorded", "feedback": record}, ensure_ascii=False)
        )
        return 0

    policy = load_policy(Path(args.config).expanduser())
    if args.warmup:
        warmup_result = warm_recall_model(policy)
        print(
            json.dumps(
                {
                    "status": "ok" if warmup_result["ok"] else "error",
                    **warmup_result,
                },
                ensure_ascii=False,
            )
        )
        return 0 if warmup_result["ok"] else 1

    payload: dict[str, Any] = {}
    if args.hook:
        raw = sys.stdin.read()
        if raw.strip():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}

    request = (
        request_from_hook_payload(payload, host=args.host, event=args.event)
        if args.hook
        else RecallRequest(
            host=args.host,
            event=args.event,
            prompt=args.prompt or "",
            cwd=args.cwd,
            session_id=args.session_id,
        )
    )
    if args.prompt:
        request.prompt = args.prompt
    if args.cwd:
        request.cwd = args.cwd
    if args.session_id:
        request.session_id = args.session_id

    recall_result = run_recall(request, policy, perform_search=not args.no_search)
    output = render_output(recall_result, args.format)
    if output:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
