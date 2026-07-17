"""Shared runtime configuration helpers.

The old runtime grew separate knobs around ``recall.toml``.  Keep that file
working, but prefer ``config.toml`` when it exists so hook, recall, audit, and
auto-apply settings can converge on one public shape.
"""

from __future__ import annotations

import os
import json
import subprocess
import tomllib
from importlib import metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_wiki_mcp.wiki import WIKI_ROOT

CONFIG_FILE = WIKI_ROOT / "config.toml"
LEGACY_RECALL_CONFIG_FILE = WIKI_ROOT / "recall.toml"

FALSE_VALUES = {"0", "false", "False", "no", "NO", "off", "OFF"}
TRUE_VALUES = {"1", "true", "True", "yes", "YES", "on", "ON"}
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_INGEST_MODEL = "maxwell1500/ornith-35b:Q5_K_M"
DEFAULT_DECISION_PRIMARY_MODEL = DEFAULT_INGEST_MODEL
DEFAULT_DECISION_CHALLENGER_MODEL = "gpt-oss:20b"
DEFAULT_DECISION_TIE_BREAK_MODEL = "gemma4:26b"
DEFAULT_HEAVY_NUM_CTX = 32_768
DEFAULT_HEAVY_KEEP_ALIVE = "20m"
MAX_SEMANTIC_PROJECTION_CHILD_BYTES = 24_000
DEFAULT_RUNTIME_SOURCE = "git+ssh://git@github.com/trafficsign/llm-wiki-mcp"
RUNTIME_PACKAGE = "llm-wiki-mcp"


def runtime_source() -> str:
    """Return the pushed package source used by production entry points."""

    configured = os.environ.get("LLM_WIKI_RUNTIME_SOURCE", "").strip()
    return configured or DEFAULT_RUNTIME_SOURCE


def runtime_repo_root() -> Path:
    """Return the explicit checkout used only for reviewed code repair/context."""

    configured = os.environ.get("LLM_WIKI_REPO_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    checkout = Path.home() / "projects" / "personal" / RUNTIME_PACKAGE
    if (checkout / ".git").exists():
        return checkout
    return Path(__file__).resolve().parents[2]


def uvx_runtime_command(
    entrypoint: str,
    *,
    executable: str = "uvx",
    refresh: bool = False,
) -> list[str]:
    """Build a production command that cannot import an unpushed worktree."""

    command = [executable]
    if refresh:
        command.extend(["--refresh-package", RUNTIME_PACKAGE])
    command.extend(["--from", runtime_source(), entrypoint])
    return command


def runtime_identity() -> dict[str, Any]:
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
        "archive_path": str(module_path.parents[2]),
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
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    document_prefix: str = ""
    query_prefix: str = ""


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    backend: str = "transformers"
    top_n: int = 10
    max_length: int = 384
    batch_size: int = 10
    device: str = ""
    weight: float = 1.0


@dataclass(frozen=True)
class IngestConfig:
    model: str = DEFAULT_INGEST_MODEL
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
    """Local-consensus model limits and memory-aware residency policy."""

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
    # Empty means the exact TOML/default model triplet is the trusted
    # bootstrap/current policy.  Setting this path only nominates an artifact;
    # DecisionRouter still validates every adoption gate before switching.
    adoption_artifact: str = ""


def active_config_file(path: Path | str | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    if CONFIG_FILE.exists():
        return CONFIG_FILE
    return LEGACY_RECALL_CONFIG_FILE


def load_toml_file(path: Path | str | None = None) -> dict[str, Any]:
    resolved = active_config_file(path)
    try:
        if not resolved.exists():
            return {}
        data = tomllib.loads(resolved.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, tomllib.TOMLDecodeError):
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


def load_embedding_config(path: Path | str | None = None) -> EmbeddingConfig:
    data = load_toml_file(path)
    embedding = data.get("embedding")
    if not isinstance(embedding, dict):
        return EmbeddingConfig()
    model = embedding.get("model")
    document_prefix = embedding.get("document_prefix")
    query_prefix = embedding.get("query_prefix")
    return EmbeddingConfig(
        model=model
        if isinstance(model, str) and model.strip()
        else DEFAULT_EMBEDDING_MODEL,
        document_prefix=document_prefix if isinstance(document_prefix, str) else "",
        query_prefix=query_prefix if isinstance(query_prefix, str) else "",
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

    model = section.get("model")
    keep_alive = section.get("keep_alive")
    max_num_ctx = _positive_int(
        section.get("max_num_ctx"), IngestConfig.max_num_ctx, minimum=2_048
    )
    num_ctx = _positive_int(section.get("num_ctx"), IngestConfig.num_ctx, minimum=2_048)
    if num_ctx > max_num_ctx:
        num_ctx = max_num_ctx

    return IngestConfig(
        model=model if isinstance(model, str) and model.strip() else IngestConfig.model,
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
    """Load the local semantic-decision ensemble configuration.

    Model names remain ordinary non-empty strings so locally installed Ollama
    tags can be selected without a code change.  Safety-critical generation
    settings are bounded here and the router separately rejects duplicate
    model roles or a quorum other than two.
    """

    data = load_toml_file(path)
    section = data.get("decision_router")
    if not isinstance(section, dict):
        return DecisionRouterConfig()

    def model(name: str, default: str, *, alias: str | None = None) -> str:
        value = section.get(name)
        if value is None and alias is not None:
            value = section.get(alias)
        return value.strip() if isinstance(value, str) and value.strip() else default

    def keep_alive(name: str, default: str, *, alias: str | None = None) -> str:
        value = section.get(name)
        if value is None and alias is not None:
            value = section.get(alias)
        return value.strip() if isinstance(value, str) and value.strip() else default

    return DecisionRouterConfig(
        primary_model=model("primary_model", DecisionRouterConfig.primary_model),
        challenger_model=model(
            "challenger_model", DecisionRouterConfig.challenger_model
        ),
        tie_break_model=model(
            "tie_break_model", DecisionRouterConfig.tie_break_model, alias="tie_model"
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
                DecisionRouterConfig.max_resident_models,
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

    Production config loading intentionally remains backwards compatible.  A
    replay gate is different: silently evaluating defaults after a typo or a
    missing file can spend substantial local compute on the wrong model set.
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

    config = load_decision_router_config(resolved)
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

    model = reranker.get("model")
    backend = reranker.get("backend")
    device = reranker.get("device")
    return RerankerConfig(
        enabled=reranker.get("enabled") is True,
        model=model
        if isinstance(model, str) and model.strip()
        else RerankerConfig.model,
        backend=backend
        if isinstance(backend, str) and backend.strip()
        else RerankerConfig.backend,
        top_n=_positive_int(reranker.get("top_n"), RerankerConfig.top_n),
        max_length=_positive_int(reranker.get("max_length"), RerankerConfig.max_length),
        batch_size=_positive_int(reranker.get("batch_size"), RerankerConfig.batch_size),
        device=device if isinstance(device, str) else "",
        weight=_nonnegative_float(reranker.get("weight"), RerankerConfig.weight),
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
        "mode": "unified" if resolved == CONFIG_FILE else "legacy-recall",
        "hook_policy": load_hook_policy(resolved).__dict__,
        "sections": sorted(data.keys()),
    }
