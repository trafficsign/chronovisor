"""Shared recall runtime for Claude Code and Codex hooks.

The runtime is deliberately host-agnostic: host hooks pass prompt/cwd data in,
and this module decides whether LLM Wiki should be searched before the agent
answers. Claude Code and Codex only need thin adapters around this CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import tomllib
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from llm_wiki_mcp.index_store import get_store
from llm_wiki_mcp.runtime_config import (
    active_config_file,
    normalize_recall_config,
)
from llm_wiki_mcp.recall_runtime_paths import RECALL_DIR
from llm_wiki_mcp.search import search as run_search
from llm_wiki_mcp.state_register import format_state_context, should_inject_state
from llm_wiki_mcp.wiki import WIKI_ROOT, SYSTEM_DIR, find_page, init_wiki


RECALL_LOG_FILE = RECALL_DIR / "recall-log.jsonl"
RECALL_FEEDBACK_FILE = RECALL_DIR / "feedback.jsonl"
RECALL_CONFIG_FILE = active_config_file()
RECALL_PULL_LOG_FILE = RECALL_DIR / "pull-log.jsonl"
RECALL_CALIBRATION_FILE = RECALL_DIR / "calibration.json"

TRIVIAL_PROMPT_RE = re.compile(
    r"^\s*(はい|いいえ|うん|おう|ok|okay|yes|no|y|n|ありがとう|thanks|thx|了解|りょ)\s*[。.!！?？]*\s*$",
    re.IGNORECASE,
)
SYSTEM_ENVELOPE_RE = re.compile(
    r"^\s*<(task-notification|system-reminder|system-notification)\b",
    re.IGNORECASE,
)
SYSTEM_BLOCK_RE = re.compile(
    r"(?ims)(^|\n)\s*<(task-notification|system-reminder|system-notification)\b.*?</\2>\s*"
)
SYSTEM_BLOCK_TO_END_RE = re.compile(
    r"(?ims)(^|\n)\s*<(task-notification|system-reminder|system-notification)\b.*\Z"
)
RECALL_CONTEXT_BLOCK_RE = re.compile(r"(?ms)(^|\n)\s*\[RECALL_CONTEXT\].*?\[/RECALL_CONTEXT\]\s*")
RECALL_CONTEXT_TO_END_RE = re.compile(r"(?ms)(^|\n)\s*\[RECALL_CONTEXT\].*\Z")
CODEX_INTERNAL_SUGGESTION_RE = re.compile(
    r"^\s*#\s*Overview\s+Generate\s+0\s+to\s+3\s+hyperpersonalized\s+suggestions\b",
    re.IGNORECASE,
)
CODEX_INTERNAL_SUGGESTION_BLOCK_RE = re.compile(
    r"(?ims)(^|\n)\s*#\s*Overview\s+Generate\s+0\s+to\s+3\s+hyperpersonalized\s+suggestions\b.*\Z"
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
    "llm wiki",
    "llm-wiki",
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


@dataclass
class RecallPolicy:
    enabled: bool = True
    search_threshold: float = 0.35
    read_threshold: float = 0.65
    max_context_chars: int = 600
    max_pages: int = 3
    max_queries: int = 3
    semantic: bool = True
    gate_mode: str = "evidence"  # legacy | evidence
    context_style: str = "cards"  # legacy | cards
    log_decisions: bool = True
    avoid_heavy_personal_context_in_chitchat: bool = True
    use_feedback_suppressions: bool = True
    fail_silent_on_judge_unavailable: bool = True
    judge_mode: str = "auto"  # off | auto | always
    judge_model: str = "qwen3.5:4b-mlx"
    judge_think: bool = False
    judge_timeout_ms: int = 2000
    judge_num_ctx: int = 4096
    judge_num_predict: int = 64
    judge_keep_alive: str = "24h"
    warmup_timeout_ms: int = 15000
    judge_include_queries: bool = False
    rewrite_enabled: bool = True
    rewrite_model: str = "qwen3.5:4b-mlx"
    rewrite_timeout_ms: int = 3000
    fusion_bm25: float = 1.0
    fusion_semantic: float = 0.6
    fusion_graph: float = 0.0
    fusion_usage_prior: float = 0.0
    fusion_bm25_score_bonus: float = 0.005
    fusion_bm25_rank_bonus: float = 0.006
    fusion_bm25_rank_decay: float = 0.006
    fusion_semantic_min_top_score: float = 0.45
    fusion_semantic_min_margin: float = 0.002
    fusion_semantic_low_confidence_weight: float = 0.25
    fusion_usage_prior_decay: float = 0.98
    fusion_usage_prior_cap: float = 3.0
    calibration_enabled: bool = True
    calibration_min_samples: int = 500
    calibration_holdout_ratio: float = 0.2
    calibration_min_improvement: float = 0.02
    session_ttl_seconds: int = 7 * 24 * 60 * 60


@dataclass
class RecallRequest:
    host: str
    event: str
    prompt: str
    cwd: str = ""
    recent_context: str = ""
    session_id: str = ""


@dataclass
class ContextItem:
    page_id: str
    title: str
    updated: str
    score: float
    snippets: list[str] = field(default_factory=list)
    sensitivity: str = "normal"


@dataclass
class RecallResult:
    status: str
    decision: str
    confidence: float
    queries: list[str]
    reasons: list[str]
    matched_terms: dict[str, list[str]]
    decision_id: str = field(default_factory=lambda: new_decision_id())
    context_items: list[ContextItem] = field(default_factory=list)
    context: str = ""
    state_context: str = ""
    used_judge: bool = False
    judge_confidence: float | None = None
    judge_reason: str = ""
    evidence_features: dict[str, Any] = field(default_factory=dict)
    search_mode: str = ""
    context_style: str = ""
    latency_ms: int = 0
    error: str = ""


def load_policy(path: Path = RECALL_CONFIG_FILE) -> RecallPolicy:
    policy = RecallPolicy()
    path = active_config_file(path)
    if path.exists():
        try:
            data = normalize_recall_config(tomllib.loads(path.read_text()))
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        _apply_config(policy, data)

    try:
        from llm_wiki_mcp.recall_policy_store import apply_active_policy

        apply_active_policy(policy)
    except Exception:
        pass

    enabled_env = os.environ.get("LLM_WIKI_RECALL_ENABLED")
    if enabled_env is not None:
        policy.enabled = enabled_env not in {"0", "false", "False", "no", "NO"}
    return policy


def new_decision_id() -> str:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def stable_prompt_hash(text: str) -> str:
    return hashlib.sha1(_feedback_key(text).encode("utf-8")).hexdigest()[:16]


def _apply_config(policy: RecallPolicy, data: dict[str, Any]) -> None:
    if isinstance(data.get("enabled"), bool):
        policy.enabled = data["enabled"]
    if isinstance(data.get("model"), str):
        policy.judge_model = data["model"]

    thresholds = data.get("thresholds", {})
    if isinstance(thresholds, dict):
        if isinstance(thresholds.get("search"), int | float):
            policy.search_threshold = float(thresholds["search"])
        if isinstance(thresholds.get("read"), int | float):
            policy.read_threshold = float(thresholds["read"])

    budgets = data.get("budgets", {})
    if isinstance(budgets, dict):
        if isinstance(budgets.get("max_context_tokens"), int):
            policy.max_context_chars = max(400, budgets["max_context_tokens"] * 4)
        if isinstance(budgets.get("max_context_chars"), int):
            policy.max_context_chars = max(400, budgets["max_context_chars"])
        if isinstance(budgets.get("max_pages"), int):
            policy.max_pages = max(1, budgets["max_pages"])
        if isinstance(budgets.get("max_queries"), int):
            policy.max_queries = max(1, budgets["max_queries"])
        if isinstance(budgets.get("judge_timeout_ms"), int):
            policy.judge_timeout_ms = max(200, budgets["judge_timeout_ms"])

    recall = data.get("recall", {})
    if isinstance(recall, dict):
        if isinstance(recall.get("semantic"), bool):
            policy.semantic = recall["semantic"]
        if isinstance(recall.get("gate_mode"), str):
            policy.gate_mode = recall["gate_mode"]
        if isinstance(recall.get("context_style"), str):
            policy.context_style = recall["context_style"]
        if isinstance(recall.get("max_context_chars"), int):
            policy.max_context_chars = max(400, recall["max_context_chars"])
        if isinstance(recall.get("judge_mode"), str):
            policy.judge_mode = recall["judge_mode"]
        if isinstance(recall.get("session_ttl_seconds"), int):
            policy.session_ttl_seconds = max(3600, recall["session_ttl_seconds"])

    gate = data.get("gate", {})
    if isinstance(gate, dict):
        if isinstance(gate.get("model"), str):
            policy.judge_model = gate["model"]
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

    rewrite = data.get("rewrite", {})
    if isinstance(rewrite, dict):
        if isinstance(rewrite.get("enabled"), bool):
            policy.rewrite_enabled = rewrite["enabled"]
        if isinstance(rewrite.get("model"), str):
            policy.rewrite_model = rewrite["model"]
        if isinstance(rewrite.get("timeout_ms"), int):
            policy.rewrite_timeout_ms = max(200, rewrite["timeout_ms"])

    fusion = data.get("fusion", {})
    if isinstance(fusion, dict):
        for key, attr in (
            ("bm25", "fusion_bm25"),
            ("semantic", "fusion_semantic"),
            ("graph", "fusion_graph"),
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

    calibration = data.get("calibration", {})
    if isinstance(calibration, dict):
        if isinstance(calibration.get("enabled"), bool):
            policy.calibration_enabled = calibration["enabled"]
        if isinstance(calibration.get("min_samples"), int):
            policy.calibration_min_samples = max(1, calibration["min_samples"])
        if isinstance(calibration.get("holdout_ratio"), int | float):
            policy.calibration_holdout_ratio = max(0.05, min(0.8, float(calibration["holdout_ratio"])))
        if isinstance(calibration.get("min_improvement"), int | float):
            policy.calibration_min_improvement = max(0.0, float(calibration["min_improvement"]))

    behavior = data.get("policy", {})
    if isinstance(behavior, dict):
        if isinstance(behavior.get("log_decisions"), bool):
            policy.log_decisions = behavior["log_decisions"]
        if isinstance(behavior.get("avoid_heavy_personal_context_in_chitchat"), bool):
            policy.avoid_heavy_personal_context_in_chitchat = behavior[
                "avoid_heavy_personal_context_in_chitchat"
            ]
        if isinstance(behavior.get("use_feedback_suppressions"), bool):
            policy.use_feedback_suppressions = behavior["use_feedback_suppressions"]
        if isinstance(behavior.get("fail_silent_on_judge_unavailable"), bool):
            policy.fail_silent_on_judge_unavailable = behavior["fail_silent_on_judge_unavailable"]


def request_from_hook_payload(payload: dict[str, Any], *, host: str, event: str) -> RecallRequest:
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


def evaluate_heuristic(request: RecallRequest, policy: RecallPolicy) -> tuple[float, list[str], dict[str, list[str]]]:
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

    if any(key in cwd_lower for key in ("llm-wiki", "codex", "claude", "jttok", "jttk")):
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

    if policy.avoid_heavy_personal_context_in_chitchat:
        if matched["chitchat"] and not matched["past_reference"]:
            score = min(score, policy.search_threshold - 0.05)
            reasons.append("personal-context guard")

    return max(0.0, min(1.0, score)), reasons, matched


def classify_non_user_prompt(prompt: str, policy: RecallPolicy | None = None) -> str:
    stripped = prompt.lstrip()
    if SYSTEM_ENVELOPE_RE.match(stripped):
        return "system notification prompt"
    if CODEX_INTERNAL_SUGGESTION_RE.match(stripped):
        return "codex internal suggestion prompt"
    if stripped.startswith("[RECALL_CONTEXT]") or stripped.startswith("[/RECALL_CONTEXT]"):
        return "recall context injection"
    if policy is None or policy.use_feedback_suppressions:
        feedback_reason = classify_feedback_suppressed_prompt(stripped)
        if feedback_reason:
            return feedback_reason
    return ""


def strip_non_user_blocks(prompt: str) -> tuple[str, list[str]]:
    cleaned = prompt
    reasons: list[str] = []
    cleaned, removed = _strip_block(cleaned, SYSTEM_BLOCK_RE)
    if removed:
        reasons.append("stripped system notification block")
    cleaned, removed_to_end = _strip_block(cleaned, SYSTEM_BLOCK_TO_END_RE)
    if removed_to_end and "stripped system notification block" not in reasons:
        reasons.append("stripped system notification block")

    cleaned, removed = _strip_block(cleaned, RECALL_CONTEXT_BLOCK_RE)
    if removed:
        reasons.append("stripped recall context block")
    cleaned, removed_to_end = _strip_block(cleaned, RECALL_CONTEXT_TO_END_RE)
    if removed_to_end and "stripped recall context block" not in reasons:
        reasons.append("stripped recall context block")

    cleaned, removed = _strip_block(cleaned, CODEX_INTERNAL_SUGGESTION_BLOCK_RE)
    if removed:
        reasons.append("stripped codex internal suggestion block")

    return re.sub(r"\n{3,}", "\n\n", cleaned).strip(), reasons


def _strip_block(text: str, pattern: re.Pattern[str]) -> tuple[str, bool]:
    cleaned, count = pattern.subn("\n", text)
    return cleaned, count > 0


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


def should_run_judge(score: float, policy: RecallPolicy, features: dict[str, Any] | None = None) -> bool:
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
        return policy.search_threshold <= evidence < policy.read_threshold and margin < 0.12
    # Auto judge is for the ambiguous search zone. Obvious read decisions should
    # not depend on a synchronous local model being available.
    return (policy.search_threshold - 0.10) <= score < policy.read_threshold


def run_local_judge(request: RecallRequest, heuristic_score: float, policy: RecallPolicy) -> tuple[float | None, list[str], str]:
    system = "You are a fast LLM Wiki recall classifier. Return compact JSON only."
    prompt = {
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
    schema = {
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
        from llm_wiki_mcp.ollama import OLLAMA_URL

        timeout_seconds = max(0.2, policy.judge_timeout_ms / 1000)
        timeout = httpx.Timeout(
            connect=min(1.0, timeout_seconds),
            read=timeout_seconds,
            write=1.0,
            pool=0.5,
        )
        with httpx.Client(base_url=OLLAMA_URL, timeout=timeout) as client:
            resp = client.post(
                "/api/generate",
                json={
                    "model": policy.judge_model,
                    "system": system,
                    "prompt": json.dumps(prompt, ensure_ascii=False),
                    "stream": False,
                    "think": policy.judge_think,
                    "keep_alive": policy.judge_keep_alive,
                    "format": schema,
                    "options": {
                        "temperature": 0,
                        "num_ctx": policy.judge_num_ctx,
                        "num_predict": policy.judge_num_predict,
                    },
                },
            )
            resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        parsed = json.loads(raw)
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


def normalize_judge_confidence(confidence: float, decision: str, policy: RecallPolicy) -> float:
    confidence = max(0.0, min(1.0, confidence))
    if decision == "none":
        return min(confidence, policy.search_threshold - 0.01)
    if decision == "search":
        return min(max(confidence, policy.search_threshold), policy.read_threshold - 0.01)
    return max(confidence, policy.read_threshold)


def build_queries(
    request: RecallRequest,
    matched: dict[str, list[str]],
    judge_queries: list[str],
    policy: RecallPolicy,
    session_state: Any | None = None,
    rewrite_queries: list[str] | None = None,
) -> list[str]:
    candidates: list[str] = []
    candidates.extend(rewrite_queries or [])
    candidates.extend(judge_queries)
    candidates.append(_compact_query(request.prompt))
    if session_state is not None:
        candidates.extend(getattr(session_state, "recent_queries", [])[-3:])
        recent_topics = getattr(session_state, "recent_topics", [])[-8:]
        if recent_topics and matched.get("ambiguity"):
            candidates.append(" ".join(recent_topics + [_compact_query(request.prompt, limit=80)]))

    topic_terms = []
    for key in ("project", "past_reference", "ownership"):
        topic_terms.extend(matched.get(key, []))
    decision_terms = matched.get("decision", [])
    if topic_terms:
        candidates.append(" ".join(topic_terms + decision_terms))
    elif len(decision_terms) >= 2:
        candidates.append(" ".join(decision_terms))

    if request.cwd:
        cwd_name = Path(request.cwd).name
        if cwd_name and cwd_name not in {"new-chat", "Documents"}:
            candidates.append(f"{cwd_name} {request.prompt}")

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
) -> tuple[list[str], float, str]:
    prompt = {
        "task": "Rewrite an ambiguous user prompt into 1-3 explicit LLM Wiki search queries.",
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
        from llm_wiki_mcp.ollama import OLLAMA_URL

        timeout_seconds = max(0.2, policy.rewrite_timeout_ms / 1000)
        timeout = httpx.Timeout(
            connect=min(1.0, timeout_seconds),
            read=timeout_seconds,
            write=1.0,
            pool=0.5,
        )
        with httpx.Client(base_url=OLLAMA_URL, timeout=timeout) as client:
            resp = client.post(
                "/api/generate",
                json={
                    "model": policy.rewrite_model or policy.judge_model,
                    "prompt": json.dumps(prompt, ensure_ascii=False),
                    "stream": False,
                    "think": False,
                    "keep_alive": policy.judge_keep_alive,
                    "format": schema,
                    "options": {
                        "temperature": 0,
                        "num_ctx": policy.judge_num_ctx,
                        "num_predict": 96,
                    },
                },
            )
            resp.raise_for_status()
        parsed = json.loads(resp.json().get("response", "{}"))
        raw_queries = parsed.get("queries")
        queries = [q for q in raw_queries if isinstance(q, str) and q.strip()] if isinstance(raw_queries, list) else []
        confidence = parsed.get("confidence", 0.0)
        confidence_f = max(0.0, min(1.0, float(confidence))) if isinstance(confidence, int | float) else 0.0
        return _dedupe_queries(queries, limit=policy.max_queries), confidence_f, "rewrite ok"
    except Exception as exc:
        return [], 0.0, f"rewrite fallback: {exc.__class__.__name__}"


def warm_recall_model(policy: RecallPolicy) -> dict[str, Any]:
    """Warm the gate/rewrite Ollama model so sync recall avoids cold starts."""
    from llm_wiki_mcp.ollama import OLLAMA_URL

    started = time.monotonic()
    models = _dedupe_queries(
        [policy.judge_model, policy.rewrite_model or policy.judge_model],
        limit=2,
    )
    timeout_seconds = max(1.0, policy.warmup_timeout_ms / 1000)
    timeout = httpx.Timeout(
        connect=min(3.0, timeout_seconds),
        read=timeout_seconds,
        write=3.0,
        pool=1.0,
    )
    warmed: list[str] = []
    errors: dict[str, str] = {}
    for model in models:
        try:
            with httpx.Client(base_url=OLLAMA_URL, timeout=timeout) as client:
                resp = client.post(
                    "/api/generate",
                    json={
                        "model": model,
                        "prompt": "warmup",
                        "stream": False,
                        "think": False,
                        "keep_alive": policy.judge_keep_alive,
                        "options": {
                            "temperature": 0,
                            "num_ctx": 128,
                            "num_predict": 1,
                        },
                    },
                )
                resp.raise_for_status()
            warmed.append(model)
        except Exception as exc:
            errors[model] = exc.__class__.__name__
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


def _dedupe_queries(candidates: list[str], limit: int) -> list[str]:
    seen = set()
    out = []
    for candidate in candidates:
        q = re.sub(r"\s+", " ", candidate).strip()
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
    # BM25 can be unbounded and RRF is tiny; this stable squashing keeps the
    # evidence gate comparable across search modes.
    if score <= 0:
        return 0.0
    if score < 0.2:
        return min(1.0, score * 30.0)
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
    return 1.0 / (1.0 + pow(2.718281828459045, -total))


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
        0.52 * float(features.get("top1_score_norm", 0.0) or 0.0)
        + 0.18 * float(features.get("margin_norm", 0.0) or 0.0)
        + 0.12 * float(features.get("hit_count_norm", 0.0) or 0.0)
        + 0.14 * float(features.get("heuristic_score", 0.0) or 0.0)
        + 0.08 * float(features.get("rewrite_confidence", 0.0) or 0.0)
    )
    if features.get("ambiguity") and features.get("hit_count", 0):
        score += 0.04
    if features.get("hit_count", 0):
        score = max(score, float(features.get("heuristic_score", 0.0) or 0.0) * 0.85)
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
    if "llm-wiki-mcp" in text:
        terms.update({"llm-wiki", "wiki", "mcp"})
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
) -> tuple[list[Any], str]:
    if not queries:
        return [], ""
    merged: dict[str, Any] = {}
    mode = "bm25"
    for query_index, query in enumerate(queries):
        results, search_mode = run_search(
            query=query,
            top_n=max(policy.max_pages * 3, 8),
            semantic=policy.semantic,
            fusion_weights={
                "bm25": policy.fusion_bm25,
                "semantic": policy.fusion_semantic,
                "graph": policy.fusion_graph,
                "usage_prior": policy.fusion_usage_prior,
                "bm25_score_bonus": policy.fusion_bm25_score_bonus,
                "bm25_rank_bonus": policy.fusion_bm25_rank_bonus,
                "bm25_rank_decay": policy.fusion_bm25_rank_decay,
                "semantic_min_top_score": policy.fusion_semantic_min_top_score,
                "semantic_min_margin": policy.fusion_semantic_min_margin,
                "semantic_low_confidence_weight": policy.fusion_semantic_low_confidence_weight,
                "usage_prior_decay": policy.fusion_usage_prior_decay,
                "usage_prior_cap": policy.fusion_usage_prior_cap,
            },
        )
        if search_mode != "bm25":
            mode = search_mode
        query_weight = max(0.50, 1.0 - (0.25 * query_index))
        for result in results:
            if should_filter_sensitive_result(result, request):
                continue
            adjusted = replace(result, score=float(result.score) * query_weight * context_boost(result, request))
            existing = merged.get(result.page_id)
            if existing is None or adjusted.score > existing.score:
                merged[result.page_id] = adjusted
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
) -> list[ContextItem]:
    if decision == "none" or not queries:
        return []

    init_wiki()
    store = get_store()
    store.refresh()

    items: list[ContextItem] = []
    seen: set[str] = set()
    for page_id in query_hint_page_ids(queries, limit=policy.max_pages):
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

    for page_id in prefetch_page_ids_for_request(request, queries, limit=policy.max_pages):
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

    results = pre_results
    if results is None:
        results, _mode = search_candidates(queries, policy, request=request)
    for result in results:
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
            summary = page_summary(result.page_id)
            if summary:
                snippets = [summary]
        if decision == "read":
            snippet = excerpt_page(result.page_id, queries, max_chars=650)
            if snippet:
                snippets = [snippet]
        items.append(
            ContextItem(
                page_id=result.page_id,
                title=result.title,
                updated=result.updated,
                score=round(result.score, 4),
                snippets=snippets,
                sensitivity=getattr(result, "sensitivity", "normal") or "normal",
            )
        )
        if len(items) >= policy.max_pages:
            return items
    return items


def query_hint_page_ids(queries: list[str], *, limit: int) -> list[str]:
    try:
        from llm_wiki_mcp.recall_hints import matching_hint_page_ids

        return matching_hint_page_ids(queries, limit=limit)
    except Exception:
        return []


def prefetch_page_ids_for_request(request: RecallRequest | None, queries: list[str], *, limit: int) -> list[str]:
    if request is None:
        return []
    try:
        from llm_wiki_mcp.prefetch import prefetch_page_ids

        return prefetch_page_ids(
            host=request.host,
            cwd=request.cwd,
            queries=queries,
            prompt=request.prompt,
            limit=limit,
        )
    except Exception:
        return []


def context_item_from_page_id(page_id: str, queries: list[str], decision: str, *, score: float) -> ContextItem | None:
    path = find_readable_page(page_id)
    if not path or not path.exists():
        return None
    title = page_id
    updated = ""
    sensitivity = "normal"
    try:
        from llm_wiki_mcp.frontmatter import parse as parse_frontmatter

        meta, _body = parse_frontmatter(path.read_text(encoding="utf-8"))
        if isinstance(meta.get("title"), str):
            title = meta["title"]
        if isinstance(meta.get("updated"), str):
            updated = meta["updated"]
        if isinstance(meta.get("sensitivity"), str):
            sensitivity = meta["sensitivity"].strip().lower() or "normal"
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
        snippets=snippets,
        sensitivity=sensitivity,
    )


def should_skip_session_page(session_state: Any | None, page_id: str, updated: str) -> bool:
    if session_state is None:
        return False
    try:
        from llm_wiki_mcp.recall_session import should_skip_page

        return should_skip_page(session_state, page_id, updated)
    except Exception:
        return False


def page_summary(page_id: str) -> str:
    path = find_readable_page(page_id)
    if not path or not path.exists():
        return ""
    try:
        from llm_wiki_mcp.frontmatter import parse as parse_frontmatter

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
        "llm-wiki-mcp",
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
        for token in re.findall(r"[a-z0-9_+.-]{2,}|[\u3040-\u30ff\u3400-\u9fff]{2,}", query.lower()):
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


def find_readable_page(page_id: str) -> Path | None:
    path = find_page(page_id)
    if path is not None:
        return path
    system_path = SYSTEM_DIR / f"{page_id}.md"
    if system_path.exists():
        return system_path
    return None


def strip_frontmatter(content: str) -> str:
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            return content[end + 4 :].strip()
    return content.strip()


def _trim_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def context_item_annotations(item: ContextItem) -> str:
    parts: list[str] = []
    if item.updated:
        parts.append(f"updated: {item.updated}")
    if item.sensitivity == "high":
        parts.append("sensitivity: high")
    return f" ({', '.join(parts)})" if parts else ""


def format_recall_context(result: RecallResult, policy: RecallPolicy) -> str:
    if result.decision == "none" or not result.context_items:
        return ""

    if policy.context_style == "cards" and result.decision != "read":
        lines = [
            "[RECALL_CONTEXT]",
            "LLM Wiki が過去文脈候補を見つけました。必要なら wiki.search/wiki.read で深掘りしてください。",
            "雑談に重い個人事情を勝手に混ぜないでください。",
            f"decision_id={result.decision_id}",
            f"decision={result.decision} confidence={result.confidence:.2f}",
        ]
        if result.reasons:
            lines.append("reasons: " + ", ".join(result.reasons[:4]))
        if result.queries:
            lines.append("queries: " + " | ".join(result.queries[:3]))
        lines.append("cards:")
        for item in result.context_items:
            summary = item.snippets[0] if item.snippets else page_summary(item.page_id)
            suffix = f" — {_one_line(summary, limit=160)}" if summary else ""
            lines.append(f"- {item.page_id}: {item.title}{context_item_annotations(item)}{suffix}")
        lines.append("詳細が必要なページは wiki.read(page_id) で取得。")
        lines.append("[/RECALL_CONTEXT]")
        context = "\n".join(lines)
        if len(context) > policy.max_context_chars:
            return context[: policy.max_context_chars].rstrip() + "\n[/RECALL_CONTEXT]"
        return context

    lines = [
        "[RECALL_CONTEXT]",
        "LLM Wiki が過去文脈候補を見つけました。関連すると判断した場合だけ使ってください。",
        "雑談に重い個人事情を勝手に混ぜないでください。",
        f"decision_id={result.decision_id}",
        f"decision={result.decision} confidence={result.confidence:.2f}",
    ]
    if result.reasons:
        lines.append("reasons: " + ", ".join(result.reasons[:5]))
    if result.queries:
        lines.append("queries: " + " | ".join(result.queries))
    lines.append("pages:")
    for item in result.context_items:
        annotations = context_item_annotations(item)
        score_note = f", score: {item.score}" if annotations else f" (score: {item.score})"
        if annotations:
            annotations = annotations[:-1] + score_note + ")"
        lines.append(f"- {item.page_id}: {item.title}{annotations or score_note}")
        for snippet in item.snippets[:1]:
            lines.append("  evidence: " + _one_line(snippet))
    lines.append("[/RECALL_CONTEXT]")

    context = "\n".join(lines)
    if len(context) > policy.max_context_chars:
        return context[: policy.max_context_chars].rstrip() + "\n[/RECALL_CONTEXT]"
    return context


def merge_context_blocks(*blocks: str, max_chars: int) -> str:
    context = "\n\n".join(block for block in blocks if block.strip())
    if len(context) <= max_chars:
        return context
    return context[:max_chars].rstrip()


def state_context_for_request(request: RecallRequest, policy: RecallPolicy) -> str:
    del policy
    if not should_inject_state(request.host):
        return ""
    return format_state_context(host=request.host, cwd=request.cwd)


def _one_line(text: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def run_recall(
    request: RecallRequest,
    policy: RecallPolicy | None = None,
    *,
    perform_search: bool = True,
) -> RecallResult:
    started = time.monotonic()
    policy = policy or load_policy()
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

    skip_reason = classify_non_user_prompt(request.prompt, policy)
    if skip_reason:
        result = RecallResult(
            status="skipped",
            decision="none",
            confidence=0.0,
            queries=[],
            reasons=[skip_reason],
            matched_terms={},
            latency_ms=_elapsed_ms(started),
        )
        if policy.log_decisions:
            append_recall_log(request, result)
        return result

    active_request = request
    cleaned_prompt, stripped_reasons = strip_non_user_blocks(request.prompt)
    if stripped_reasons:
        if not cleaned_prompt:
            result = RecallResult(
                status="skipped",
                decision="none",
                confidence=0.0,
                queries=[],
                reasons=stripped_reasons,
                matched_terms={},
                latency_ms=_elapsed_ms(started),
            )
            if policy.log_decisions:
                append_recall_log(request, result)
            return result
        active_request = replace(request, prompt=cleaned_prompt)

    score, reasons, matched = evaluate_heuristic(active_request, policy)
    reasons = stripped_reasons + reasons
    used_judge = False
    judge_confidence: float | None = None
    judge_reason = ""
    judge_queries: list[str] = []
    rewrite_queries: list[str] = []
    rewrite_confidence = 0.0
    search_mode = ""
    evidence_features: dict[str, Any] = {}
    session_state = None
    pre_results: list[Any] = []
    rewrite_metrics: dict[str, Any] = {}

    if policy.gate_mode == "evidence" and perform_search:
        try:
            from llm_wiki_mcp.recall_session import (
                cleanup_sessions,
                load_session_state,
                session_summary,
                update_session_after_recall,
            )

            cleanup_sessions(policy.session_ttl_seconds)
            session_state = load_session_state(active_request.session_id)
            initial_queries = build_queries(active_request, matched, [], policy, session_state=session_state)
            pre_results, search_mode = search_candidates(initial_queries, policy, request=active_request)
            evidence_features = build_evidence_features(
                request=active_request,
                matched=matched,
                heuristic_score=score,
                results=pre_results,
                search_mode=search_mode,
            )
            preliminary_score = evidence_score(evidence_features, policy)
            evidence_features["evidence_score"] = preliminary_score
            if should_rewrite_query(
                request=active_request,
                matched=matched,
                policy=policy,
                preliminary_features=evidence_features,
            ):
                rewrite_started = time.monotonic()
                rewrite_queries, rewrite_confidence, rewrite_reason = run_query_rewriter(
                    active_request,
                    matched,
                    policy,
                    session_summary(session_state),
                )
                rewrite_metrics = {
                    "rewrite_attempted": True,
                    "rewrite_latency_ms": _elapsed_ms(rewrite_started),
                    "rewrite_reason": rewrite_reason,
                    "rewrite_status": "ok"
                    if rewrite_queries
                    else "fallback"
                    if rewrite_reason.startswith("rewrite fallback:")
                    else "empty",
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
                    pre_results, search_mode = search_candidates(queries_for_search, policy, request=active_request)
                    evidence_features = build_evidence_features(
                        request=active_request,
                        matched=matched,
                        heuristic_score=score,
                        results=pre_results,
                        search_mode=search_mode,
                        rewrite_confidence=rewrite_confidence,
                    )
                    evidence_features.update(rewrite_metrics)
            score = evidence_score(evidence_features, policy)
            evidence_features["evidence_score"] = score
            evidence_features["decision_pre_judge"] = decision_from_score(score, policy)
        except Exception as exc:
            reasons.append(f"evidence gate failed: {exc.__class__.__name__}")
            pre_results = []
            search_mode = "error"

    if should_run_judge(score, policy, evidence_features):
        judge_score, judge_queries, judge_reason = run_local_judge(active_request, score, policy)
        used_judge = judge_score is not None
        if judge_score is not None:
            score = judge_score
            judge_confidence = judge_score
            if judge_reason:
                reasons.append("judge: " + judge_reason)
        elif judge_reason:
            reasons.append(judge_reason)
            if policy.fail_silent_on_judge_unavailable:
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
                    latency_ms=_elapsed_ms(started),
                )
                if policy.log_decisions:
                    append_recall_log(request, result)
                return result

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
            context_items = collect_context(
                queries,
                decision,
                policy,
                request=active_request,
                session_state=session_state,
                pre_results=pre_results or None,
            )
            if not context_items:
                reasons.append("no matching pages")
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            reasons.append("search failed")

    result = RecallResult(
        status="ok" if not error else "error",
        decision=decision if context_items or decision == "none" or not perform_search else "search",
        confidence=round(score, 3),
        queries=queries,
        reasons=reasons,
        matched_terms=matched,
        context_items=context_items,
        used_judge=used_judge,
        judge_confidence=judge_confidence,
        judge_reason=judge_reason,
        evidence_features=evidence_features,
        search_mode=search_mode,
        context_style=policy.context_style,
        latency_ms=_elapsed_ms(started),
        error=error,
    )
    recall_context = format_recall_context(result, policy)
    result.state_context = state_context_for_request(active_request, policy)
    result.context = merge_context_blocks(
        result.state_context,
        recall_context,
        max_chars=policy.max_context_chars,
    )
    if result.state_context:
        result.reasons.append("state register injected")
    if session_state is not None:
        try:
            from llm_wiki_mcp.recall_session import update_session_after_recall

            update_session_after_recall(
                session_state,
                queries=queries,
                page_ids=[item.page_id for item in result.context_items],
                page_updated={item.page_id: item.updated for item in result.context_items},
            )
        except Exception:
            pass
    if policy.log_decisions:
        append_recall_log(request, result)
    return result


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def append_recall_log(request: RecallRequest, result: RecallResult) -> None:
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "decision_id": result.decision_id,
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
        "reasons": result.reasons,
        "used_judge": result.used_judge,
        "judge_confidence": result.judge_confidence,
        "judge_reason": result.judge_reason,
        "evidence_features": result.evidence_features,
        "search_mode": result.search_mode,
        "context_style": result.context_style,
        "latency_ms": result.latency_ms,
        "status": result.status,
        "error": result.error,
    }
    append_jsonl(RECALL_LOG_FILE, record)
    try:
        from llm_wiki_mcp.recall_policy_store import append_live_episode

        live_path = RECALL_LOG_FILE.parent.parent / "runtime" / "recall-improvement" / "live-episodes.jsonl"
        append_live_episode(record, path=live_path)
    except Exception:
        pass


def recall_log_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": record.get("decision_id", ""),
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
        "reasons": record.get("reasons", []),
        "used_judge": record.get("used_judge", False),
        "judge_confidence": record.get("judge_confidence"),
        "judge_reason": record.get("judge_reason", ""),
        "evidence_features": record.get("evidence_features", {}),
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
    expected_queries: list[str] | None = None,
    ref: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = recall_log_snapshot(found) if ref and (found := find_recall_log(ref)) else None
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "host": host,
        "prompt": prompt,
        "note": note,
        "expected_pages": expected_pages or [],
        "expected_queries": expected_queries or [],
        "ref": ref,
        "snapshot": snapshot,
    }
    if extra:
        for key, value in extra.items():
            if key not in {"ts", "kind"}:
                record[key] = value
    append_jsonl(RECALL_FEEDBACK_FILE, record)
    return record


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def result_to_dict(result: RecallResult) -> dict[str, Any]:
    data = asdict(result)
    data["context_items"] = [asdict(item) for item in result.context_items]
    return data


def render_output(result: RecallResult, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(result_to_dict(result), ensure_ascii=False)
    if not result.context:
        return "{}" if output_format in {"codex", "hook-json"} else ""
    if output_format == "claude":
        return result.context
    if output_format in {"codex", "hook-json"}:
        return json.dumps(
            {
                "systemMessage": "LLM Wiki recall context injected.",
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": result.context,
                },
            },
            ensure_ascii=False,
        )
    return result.context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LLM Wiki recall gate.")
    parser.add_argument("--host", default="generic", help="Host adapter name: claude-code, codex, generic.")
    parser.add_argument("--event", default="UserPromptSubmit")
    parser.add_argument("--prompt", help="User prompt. If omitted with --hook, read from hook JSON stdin.")
    parser.add_argument("--cwd", default=os.environ.get("PWD", ""))
    parser.add_argument("--session-id", default="")
    parser.add_argument("--config", default=str(RECALL_CONFIG_FILE))
    parser.add_argument("--hook", action="store_true", help="Read hook JSON from stdin.")
    parser.add_argument("--no-search", action="store_true", help="Only evaluate the gate; do not search pages.")
    parser.add_argument("--warmup", action="store_true", help="Warm the gate/rewrite model and exit.")
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
        choices=["missed", "missed_candidate", "false-positive", "useful"],
        help="Record human feedback instead of running recall.",
    )
    parser.add_argument("--note", default="", help="Feedback note.")
    parser.add_argument("--expected-page", action="append", default=[], help="Expected page id for missed recall.")
    parser.add_argument("--expected-query", action="append", default=[], help="Expected query for missed recall.")
    parser.add_argument("--ref", default="", help="Decision id to attach as a feedback snapshot.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.recent is not None:
        print(json.dumps({"status": "ok", "items": recent_recall_logs(args.recent)}, ensure_ascii=False))
        return 0

    if args.feedback:
        record = append_feedback(
            args.feedback,
            args.note,
            prompt=args.prompt or "",
            host=args.host,
            expected_pages=args.expected_page,
            expected_queries=args.expected_query,
            ref=args.ref,
        )
        print(json.dumps({"status": "recorded", "feedback": record}, ensure_ascii=False))
        return 0

    policy = load_policy(Path(args.config).expanduser())
    if args.warmup:
        result = warm_recall_model(policy)
        print(json.dumps({"status": "ok" if result["ok"] else "error", **result}, ensure_ascii=False))
        return 0 if result["ok"] else 1

    payload: dict[str, Any] = {}
    if args.hook:
        raw = sys.stdin.read()
        if raw.strip():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}

    request = request_from_hook_payload(payload, host=args.host, event=args.event) if args.hook else RecallRequest(
        host=args.host,
        event=args.event,
        prompt=args.prompt or "",
        cwd=args.cwd,
        session_id=args.session_id,
    )
    if args.prompt:
        request.prompt = args.prompt
    if args.cwd:
        request.cwd = args.cwd
    if args.session_id:
        request.session_id = args.session_id

    result = run_recall(request, policy, perform_search=not args.no_search)
    output = render_output(result, args.format)
    if output:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
