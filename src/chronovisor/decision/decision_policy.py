"""Lane-scoped rollout policy for local semantic decisions.

Every routine semantic caller must name a lane.  New lanes fail closed until
they are registered, shadow is the default, and mutation-capable ``enabled``
mode still requires a fully validated adoption artifact in DecisionRouter.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from chronovisor.core.runtime_config import load_toml_file


VALID_MODES = frozenset({"off", "shadow", "enabled"})
VALID_KINDS = frozenset(
    {"validated_local", "consensus", "preserve_conflict", "local_batch", "repair_only"}
)


@dataclass(frozen=True)
class DecisionPolicy:
    lane: str
    kind: str
    schema_name: str | None = None
    default_mode: str = "shadow"

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"unsupported decision policy kind: {self.kind}")
        if self.default_mode not in VALID_MODES:
            raise ValueError(f"unsupported decision policy mode: {self.default_mode}")
        if self.kind in {"consensus", "local_batch"} and not self.schema_name:
            raise ValueError(f"structured decision policy requires a schema: {self.lane}")


DECISION_POLICIES: dict[str, DecisionPolicy] = {
    "autonomy_duplicate_resolution": DecisionPolicy(
        "autonomy_duplicate_resolution", "consensus", "duplicate_resolution"
    ),
    "autonomy_retention": DecisionPolicy(
        "autonomy_retention", "consensus", "retention"
    ),
    "content_correction_classification": DecisionPolicy(
        "content_correction_classification",
        "consensus",
        "content_correction_classification",
    ),
    "content_correction_review": DecisionPolicy(
        "content_correction_review", "consensus", "content_correction_review"
    ),
    "entity_backfill": DecisionPolicy(
        "entity_backfill", "consensus", "lint_safe_semantic_mutation"
    ),
    "ingest_reconciliation": DecisionPolicy(
        "ingest_reconciliation", "consensus", "ingest_reconciliation"
    ),
    "lint_safe_semantic_mutation": DecisionPolicy(
        "lint_safe_semantic_mutation", "consensus", "lint_safe_semantic_mutation"
    ),
    "lint_tag_repair": DecisionPolicy("lint_tag_repair", "consensus", "lint_tag_repair"),
    "local_repair": DecisionPolicy("local_repair", "consensus", "local_repair"),
    "metadata_backfill": DecisionPolicy(
        "metadata_backfill", "consensus", "lint_safe_semantic_mutation"
    ),
    "orphan_link": DecisionPolicy("orphan_link", "consensus", "orphan_link"),
    "page_normalize": DecisionPolicy(
        "page_normalize", "consensus", "lint_safe_semantic_mutation"
    ),
    "raw_replay_reconciliation": DecisionPolicy(
        "raw_replay_reconciliation", "local_batch", "raw_replay_reconciliation"
    ),
    "read_back_repair": DecisionPolicy(
        "read_back_repair", "consensus", "read_back_repair"
    ),
    "recall_auto_apply": DecisionPolicy(
        "recall_auto_apply", "consensus", "generic_decision"
    ),
    "recall_calibration": DecisionPolicy(
        "recall_calibration", "local_batch", "generic_decision"
    ),
    "recall_improvement": DecisionPolicy(
        "recall_improvement", "local_batch", "generic_decision"
    ),
    "search_label": DecisionPolicy("search_label", "local_batch", "search_label"),
    "search_self_tune": DecisionPolicy(
        "search_self_tune", "local_batch", "generic_decision"
    ),
    # Non-model lanes document the complete autonomous trust boundary.  They
    # are enabled because their authorization comes from deterministic
    # validation, not from a semantic model vote.
    "raw_capture": DecisionPolicy(
        "raw_capture", "validated_local", default_mode="enabled"
    ),
    "exact_user_correction": DecisionPolicy(
        "exact_user_correction", "validated_local", default_mode="enabled"
    ),
    "derived_index_rebuild": DecisionPolicy(
        "derived_index_rebuild", "validated_local", default_mode="enabled"
    ),
    "claims_conflict": DecisionPolicy(
        "claims_conflict", "preserve_conflict", default_mode="enabled"
    ),
    "system_code_repair": DecisionPolicy(
        "system_code_repair", "repair_only", default_mode="enabled"
    ),
}


def _env_name(lane: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", lane).strip("_").upper()
    return f"CHRONOVISOR_DECISION_POLICY_{normalized}"


def resolve_decision_policy(
    lane: str | None,
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> tuple[DecisionPolicy | None, str, str | None]:
    """Return ``(policy, mode, error)`` with unknown/invalid lanes closed."""

    if not isinstance(lane, str) or not lane.strip():
        return None, "off", "decision_lane_required"
    name = lane.strip()
    policy = DECISION_POLICIES.get(name)
    if policy is None:
        return None, "off", f"decision_lane_unknown:{name}"

    configured: Any = None
    data = load_toml_file(config_path)
    section = data.get("decision_policies") if isinstance(data, dict) else None
    if isinstance(section, dict):
        configured = section.get(name)
        if isinstance(configured, dict):
            configured = configured.get("mode")
    env = os.environ.get(_env_name(name))
    raw_mode = env if env is not None else configured
    mode = policy.default_mode if raw_mode is None else str(raw_mode).strip().lower()
    if mode not in VALID_MODES:
        return policy, "off", f"decision_lane_mode_invalid:{name}"
    return policy, mode, None


def decision_policy_snapshot() -> dict[str, Any]:
    lanes: dict[str, Any] = {}
    for name in sorted(DECISION_POLICIES):
        policy, mode, error = resolve_decision_policy(name)
        assert policy is not None
        lanes[name] = {
            "kind": policy.kind,
            "schema_name": policy.schema_name,
            "mode": mode,
            "error": error,
        }
    return {"lanes": lanes, "counts": {mode: sum(1 for row in lanes.values() if row["mode"] == mode) for mode in sorted(VALID_MODES)}}


__all__ = [
    "DECISION_POLICIES",
    "DecisionPolicy",
    "VALID_MODES",
    "decision_policy_snapshot",
    "resolve_decision_policy",
]
