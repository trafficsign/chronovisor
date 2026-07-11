"""Risk-based frontier auditing policy for ingest proposals."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp.runtime_config import IngestAuditConfig, load_ingest_audit_config


_CORRECTION_RE = re.compile(
    r"(?:それ|その記憶|この記憶).{0,12}(?:違う|間違|誤り)|"
    r"(?:訂正|撤回|記憶を消|忘れて)|"
    r"\b(?:that(?:'s| is) wrong|incorrect memory|correct(?:ion)?|retract|forget that)\b",
    re.IGNORECASE | re.DOTALL,
)
_HIGH_RISK_TARGET_RE = re.compile(
    r"(?:^|[-/])(?:system|security|auth|billing|credential|secret|keychain)"
    r"(?:[-/.]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IngestAuditDecision:
    required: bool
    mode: str
    reasons: tuple[str, ...]
    sample_rate: float
    sample_bucket: float
    base_sample_rate: float
    adaptive_sample_rate: float
    audited_examples: int
    caught_issue_rate: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def _load_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "outcomes": []}
    if not isinstance(value, dict) or not isinstance(value.get("outcomes"), list):
        return {"schema_version": 1, "outcomes": []}
    return value


def _adaptive_rate(
    state: dict[str, Any],
    config: IngestAuditConfig,
) -> tuple[float, int, float]:
    rows = [row for row in state.get("outcomes", []) if isinstance(row, dict)]
    rows = rows[-config.adaptive_window :]
    audited = len(rows)
    caught = sum(row.get("caught_issue") is True for row in rows)
    caught_rate = caught / audited if audited else 0.0
    if not config.adaptive or audited < config.adaptive_min_audits:
        return 0.0, audited, caught_rate
    if caught_rate >= config.critical_reject_rate:
        return config.critical_sample_rate, audited, caught_rate
    if caught_rate >= config.elevated_reject_rate:
        return config.elevated_sample_rate, audited, caught_rate
    return 0.0, audited, caught_rate


def decide_ingest_audit(
    *,
    source_key: str,
    raw_content: str,
    operations: list[dict],
    failed_operation_specs: list[dict],
    local_disposition: str,
    state_path: Path,
    config: IngestAuditConfig | None = None,
    force: bool = False,
    explicit_reviewer: bool = False,
) -> IngestAuditDecision:
    cfg = config or load_ingest_audit_config()
    state = _load_state(state_path)
    adaptive_rate, audited, caught_rate = _adaptive_rate(state, cfg)

    reasons: list[str] = []
    if force:
        reasons.append("frontier convergence already engaged")
    if explicit_reviewer:
        reasons.append("explicit frontier reviewer")
    if failed_operation_specs:
        reasons.append("local generation incomplete")
    if len(operations) > cfg.max_operations_without_audit:
        reasons.append("large mutation batch")
    if _CORRECTION_RE.search(raw_content):
        reasons.append("explicit correction or retraction signal")
    filenames = [
        str(operation.get("filename") or "")
        for operation in operations
        if isinstance(operation, dict)
    ]
    if any(_HIGH_RISK_TARGET_RE.search(filename) for filename in filenames):
        reasons.append("operational or sensitive target")

    mandatory = bool(reasons)
    if not cfg.enabled and not mandatory:
        reasons.append("routine frontier sampling disabled")

    has_update = any(operation.get("type") == "update" for operation in operations)
    if local_disposition == "triage_no_operations":
        base_rate = cfg.noop_sample_rate
    elif has_update:
        base_rate = cfg.update_sample_rate
    else:
        base_rate = cfg.sample_rate
    # Adaptive auditing must never turn a temporary quality incident into a
    # subscription-consuming positive feedback loop. Mandatory high-risk
    # proposals remain mandatory; routine sampling is capped independently.
    effective_rate = min(max(base_rate, adaptive_rate), cfg.max_sample_rate)
    try:
        sample_bucket = int(source_key[:12], 16) / float(16**12)
    except ValueError:
        sample_bucket = 1.0
    sampled = cfg.enabled and sample_bucket < effective_rate

    if mandatory:
        mode = "mandatory"
        required = True
    elif sampled:
        mode = "sampled"
        required = True
        reasons.append("deterministic quality sample")
    else:
        mode = "local"
        required = False
        reasons.append("low-risk local authorization")

    return IngestAuditDecision(
        required=required,
        mode=mode,
        reasons=tuple(dict.fromkeys(reasons)),
        sample_rate=effective_rate,
        sample_bucket=sample_bucket,
        base_sample_rate=base_rate,
        adaptive_sample_rate=adaptive_rate,
        audited_examples=audited,
        caught_issue_rate=caught_rate,
    )


def record_frontier_audit_outcome(
    *,
    state_path: Path,
    source_key: str,
    approved: bool,
    mode: str,
    reasons: list[str],
    config: IngestAuditConfig | None = None,
) -> None:
    """Record one unique raw audit; a caught issue remains sticky after repair."""

    cfg = config or load_ingest_audit_config()
    state = _load_state(state_path)
    rows = [row for row in state.get("outcomes", []) if isinstance(row, dict)]
    existing = next((row for row in rows if row.get("source_key") == source_key), None)
    now = datetime.now().isoformat()
    if existing is None:
        existing = {
            "source_key": source_key,
            "first_audited_at": now,
            "caught_issue": not approved,
        }
        rows.append(existing)
    else:
        existing["caught_issue"] = bool(existing.get("caught_issue")) or not approved
    existing.update({
        "last_audited_at": now,
        "approved": approved,
        "mode": mode,
        "reasons": list(reasons),
    })
    state = {
        "schema_version": 1,
        "updated_at": now,
        "outcomes": rows[-cfg.adaptive_window :],
    }
    try:
        from llm_wiki_mcp.link_fix import atomic_write

        state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            state_path,
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    except OSError:
        return
