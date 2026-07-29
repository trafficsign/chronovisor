"""Replay and adoption gate for the local structured-decision ensemble.

The evaluator compares local decisions with the expected labels in a sealed
read-only corpus. It never imports or calls the frontier repair plane, and its
durable artifact contains only hashes, expected labels, validation diagnostics,
and aggregate metrics -- never prompts or literal model responses.
"""

from __future__ import annotations

from chronovisor.timeutil import utc_iso_seconds as _now

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from chronovisor.canonical_json import (
    canonical_json_sha256_strict as _sha256_json,
    canonical_json_strict as _canonical_json,
)

import httpx

from chronovisor import ollama
from chronovisor.decision_lane_contracts import (
    LANE_CONTRACT_CASE_VERSION,
    LANE_CONTRACT_POLICY_VERSION,
    LANE_CONTRACT_SOURCE,
    MIN_CASES_PER_MODEL_BACKED_LANE,
    lane_contract_manifest,
    lane_contract_manifest_sha256,
    model_backed_lane_names,
    validate_declared_lane_contract,
)
from chronovisor.decision_lane_contract_cases import (
    decision_lane_contract_case_manifest,
    decision_lane_contract_case_manifest_sha256,
)
from chronovisor.decision_router import (
    DECISION_REQUEST_FINGERPRINT_VERSION,
    DECISION_SEMANTICS_POLICY_VERSION,
    QUORUM_SAFETY_POLICY_VERSION,
    DecisionRouter,
    DecisionRouterResult,
    ModelObserver,
    decision_context_buckets,
    decision_effective_request,
    decision_request_fingerprint_sha256,
    decision_request_context,
)
from chronovisor.decision_schema_manifest import (
    decision_signature_value,
    default_decision_value,
    production_decision_schemas,
    production_schema_manifest,
    production_signature_manifest,
    schema_sha256,
)
from chronovisor.local_structured import (
    ChatRequest,
    ChatTransport,
    STRUCTURED_GENERATION_POLICY_VERSION,
    structured_generation_policy,
    structured_generation_policy_sha256,
    validate_json,
    validate_schema_definition,
)
from chronovisor.lab.model_lab import REPLAY_FILE
from chronovisor.runtime_config import (
    DecisionRouterConfig,
    load_candidate_decision_router_config,
    load_decision_router_config,
)

ARTIFACT_SCHEMA_VERSION = 12
# Policy 20 seals the fixed structured-generation sampler (including the
# explicit seed) into replay fingerprints and adoption identity. Policy 19
# allowed Ollama's omitted-seed behavior, which produced different valid votes
# for the same temperature-zero request and made a safety veto nondeterministic.
EVALUATOR_POLICY_VERSION = 21
LEGACY_REPLAY_PROMPT_LIMIT = 50_000
STALE_HISTORICAL_REQUEST_IDENTITY_EXCLUSION = "stale_historical_request_identity"
# This is deliberately not configurable from the CLI.  A tiny hand-picked
# slice is useful for smoke testing, but it must never become an adoption
# artifact merely because every answer in that slice happened to agree.
MIN_ADOPTION_USABLE_CASES = 100
MIN_CASES_PER_PRODUCTION_SCHEMA = 5
UNSAFE_HOLD_DECISIONS = frozenset(
    {
        "blocked",
        "invalid",
        "needs_retry",
        "quarantined",
        "reject",
        "rejected",
    }
)
APPLY_DECISIONS = frozenset(
    {
        "accept",
        "accepted",
        "apply",
        "apply_available",
        "approved",
        "confirmed_noop",
        "supersede_left",
        "supersede_right",
    }
)
TRANSIENT_DECISIONS = frozenset({"defer", "retry", "uncertain"})


class ReplayInputError(ValueError):
    """Raised before inference when the selected replay corpus is invalid."""


class ResumeMismatchError(ValueError):
    """Raised when an artifact cannot safely resume the requested run."""


@lru_cache(maxsize=1)
def _production_schema_names_by_digest() -> dict[str, frozenset[str]]:
    names: dict[str, set[str]] = {}
    for name, digest in production_schema_manifest().items():
        names.setdefault(digest, set()).add(name)
    return {digest: frozenset(values) for digest, values in names.items()}


@dataclass(frozen=True)
class AdoptionThresholds:
    """Minimum conditions from the local-consensus rollout plan."""

    first_pass_schema_rate: float = 0.98
    final_schema_rate: float = 1.0
    pair_valid_rate: float = 0.99
    pair_agreement_rate: float = 0.75
    majority_resolution_rate: float = 0.99
    expected_effect_match_rate: float = 0.90
    max_invalid_output_accepted: int = 0
    max_unsafe_decision_flips: int = 0


@dataclass(frozen=True)
class ReplayCase:
    index: int
    case_id: str
    role: str
    source: str | None
    contract_id: str | None
    decision_lane: str | None
    lane_contract_sha256: str | None
    lane_contract_effect: str | None
    lane_contract_case_manifest_sha256: str | None
    evidence_provenance: dict[str, Any] = field(repr=False)
    prompt: str = field(repr=False)
    system: str | None = field(repr=False)
    schema: dict[str, Any] = field(repr=False)
    expected: dict[str, Any] = field(repr=False)
    schema_sha256: str
    expected_signature_sha256: str
    effective_request_sha256: str

    @property
    def expected_decision(self) -> str | None:
        value = self.expected.get("decision")
        return value if isinstance(value, str) else None

    @property
    def expected_coverage_label(self) -> str | None:
        for key in ("decision", "action", "classification", "approved"):
            if key not in self.expected:
                continue
            value = self.expected[key]
            if isinstance(value, (str, bool, int, float)):
                return f"{key}={_canonical_json(value)}"
        return None

    def listing(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "case_id": self.case_id,
            "role": self.role,
            "source": self.source,
            "contract_id": self.contract_id,
            "decision_lane": self.decision_lane,
            "lane_contract_sha256": self.lane_contract_sha256,
            "lane_contract_effect": self.lane_contract_effect,
            "lane_contract_case_manifest_sha256": (
                self.lane_contract_case_manifest_sha256
            ),
            "evidence_provenance": self.evidence_provenance,
            "schema_sha256": self.schema_sha256,
            "expected_signature_sha256": self.expected_signature_sha256,
            "effective_request_sha256": self.effective_request_sha256,
            "expected_decision": self.expected_decision,
        }

    @property
    def self_labeled(self) -> bool:
        return bool(
            self.source == "local_consensus"
            or self.evidence_provenance.get("kind") == "model_self_label"
        )


@dataclass(frozen=True)
class ReplayCorpus:
    path: Path
    source_sha256: str
    total_cases: int
    usable_cases: int
    excluded_cases: int
    excluded_reasons: tuple[tuple[str, int], ...]
    offset: int
    limit: int
    cases: tuple[ReplayCase, ...]
    usable_roles: tuple[str, ...]
    usable_decisions: tuple[str, ...]
    required_schema_manifest: tuple[tuple[str, str], ...]
    usable_schema_counts: tuple[tuple[str, int], ...]
    usable_lane_contract_counts: tuple[tuple[str, int], ...]
    production_lane_contracts_required: bool
    signature_manifest_sha256: str
    effective_request_unique_count: int
    exact_duplicate_request_groups: int
    exact_duplicate_request_rows: int
    exact_duplicate_redundant_rows: int
    conflicting_request_groups: int
    conflicting_request_rows: int

    @property
    def full_usable_selection(self) -> bool:
        return self.offset == 0 and len(self.cases) == self.usable_cases

    def inspection(self, *, include_cases: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "valid",
            "source_path": str(self.path),
            "source_sha256": self.source_sha256,
            "total_cases": self.total_cases,
            "usable_cases": self.usable_cases,
            "excluded_cases": self.excluded_cases,
            "excluded_reasons": dict(self.excluded_reasons),
            "offset": self.offset,
            "limit": self.limit,
            "selected_cases": len(self.cases),
            "full_usable_selection": self.full_usable_selection,
            "coverage": self.coverage(),
            "selected_case_ids_sha256": _sha256_json(
                [case.case_id for case in self.cases]
            ),
            "selected_effective_requests_sha256": _sha256_json(
                [case.effective_request_sha256 for case in self.cases]
            ),
            "effective_request_fingerprints": {
                "version": DECISION_REQUEST_FINGERPRINT_VERSION,
                "unique_requests": self.effective_request_unique_count,
                "exact_duplicate_groups": self.exact_duplicate_request_groups,
                "exact_duplicate_rows": self.exact_duplicate_request_rows,
                "exact_duplicate_redundant_rows": (self.exact_duplicate_redundant_rows),
                "conflicting_groups": self.conflicting_request_groups,
                "conflicting_rows": self.conflicting_request_rows,
            },
        }
        if include_cases:
            payload["cases"] = [case.listing() for case in self.cases]
        return payload

    def coverage(self) -> dict[str, Any]:
        selected_roles = tuple(sorted({case.role for case in self.cases}))
        selected_decisions = tuple(
            sorted(
                {
                    decision
                    for case in self.cases
                    if (decision := case.expected_coverage_label) is not None
                }
            )
        )
        selected_schema_counts = Counter(case.schema_sha256 for case in self.cases)
        usable_schema_counts = dict(self.usable_schema_counts)
        schema_names_by_digest: dict[str, list[str]] = {}
        for name, digest in self.required_schema_manifest:
            schema_names_by_digest.setdefault(digest, []).append(name)
        required_schemas = [
            {
                "names": sorted(names),
                "sha256": digest,
                "usable_cases": usable_schema_counts.get(digest, 0),
                "selected_cases": selected_schema_counts.get(digest, 0),
            }
            for digest, names in sorted(schema_names_by_digest.items())
        ]
        schema_manifest = [
            {"name": name, "sha256": digest}
            for name, digest in self.required_schema_manifest
        ]
        selected_required_counts = [
            int(row["selected_cases"]) for row in required_schemas
        ]
        covered_schemas = sum(count > 0 for count in selected_required_counts)
        schema_denominator = len(required_schemas)
        current_lane_manifest = lane_contract_manifest()
        canonical_case_manifest = decision_lane_contract_case_manifest()
        required_lane_names = model_backed_lane_names()
        selected_lane_counts = Counter(
            case.decision_lane
            for case in self.cases
            if case.source == LANE_CONTRACT_SOURCE and case.decision_lane is not None
        )
        usable_lane_counts = dict(self.usable_lane_contract_counts)
        required_lanes = []
        for lane in required_lane_names:
            selected_lane_cases = [
                case
                for case in self.cases
                if case.source == LANE_CONTRACT_SOURCE and case.decision_lane == lane
            ]
            observed_labels = sorted(
                {
                    label
                    for case in selected_lane_cases
                    if (label := case.expected_coverage_label) is not None
                }
            )
            observed_effects = sorted(
                {
                    case.lane_contract_effect
                    for case in selected_lane_cases
                    if case.lane_contract_effect is not None
                }
            )
            required_labels = list(
                current_lane_manifest[lane]["required_coverage_labels"]
            )
            required_effects = list(current_lane_manifest[lane]["required_effects"])
            canonical_lane_cases = list(canonical_case_manifest["lanes"][lane]["cases"])
            observed_lane_cases = sorted(
                (
                    {
                        "contract_id": case.contract_id,
                        "effective_request_sha256": case.effective_request_sha256,
                        "expected_sha256": _sha256_json(case.expected),
                        "expected_signature_sha256": (case.expected_signature_sha256),
                        "expected_coverage_label": case.expected_coverage_label,
                        "expected_effect": case.lane_contract_effect,
                    }
                    for case in selected_lane_cases
                ),
                key=lambda row: str(row["contract_id"]),
            )
            exact_case_set = observed_lane_cases == canonical_lane_cases
            required_lanes.append(
                {
                    "lane": lane,
                    "contract_sha256": current_lane_manifest[lane]["contract_sha256"],
                    "schema_name": current_lane_manifest[lane]["schema_name"],
                    "schema_sha256": current_lane_manifest[lane]["schema_sha256"],
                    "usable_contract_cases": usable_lane_counts.get(lane, 0),
                    "selected_contract_cases": selected_lane_counts.get(lane, 0),
                    "required_coverage_labels": required_labels,
                    "observed_coverage_labels": observed_labels,
                    "required_effects": required_effects,
                    "observed_effects": observed_effects,
                    "canonical_case_set_sha256": _sha256_json(canonical_lane_cases),
                    "observed_case_set_sha256": _sha256_json(observed_lane_cases),
                    "exact_canonical_case_set": exact_case_set,
                    "valid": bool(
                        selected_lane_counts.get(lane, 0)
                        >= MIN_CASES_PER_MODEL_BACKED_LANE
                        and set(required_labels).issubset(observed_labels)
                        and set(required_effects).issubset(observed_effects)
                        and exact_case_set
                    ),
                }
            )
        selected_lane_required_counts = [
            int(row["selected_contract_cases"]) for row in required_lanes
        ]
        covered_lanes = sum(row["valid"] is True for row in required_lanes)
        return {
            "usable_roles": list(self.usable_roles),
            "selected_roles": list(selected_roles),
            "role_coverage_rate": _set_coverage_rate(selected_roles, self.usable_roles),
            "usable_decisions": list(self.usable_decisions),
            "selected_decisions": list(selected_decisions),
            "decision_coverage_rate": _set_coverage_rate(
                selected_decisions, self.usable_decisions
            ),
            "required_schemas": required_schemas,
            "schema_manifest_sha256": _sha256_json(schema_manifest),
            "signature_manifest_sha256": self.signature_manifest_sha256,
            "lane_contract_policy_version": LANE_CONTRACT_POLICY_VERSION,
            "lane_contract_manifest_sha256": lane_contract_manifest_sha256(),
            "lane_contract_case_manifest_sha256": (
                decision_lane_contract_case_manifest_sha256()
            ),
            "production_lane_contracts_required": (
                self.production_lane_contracts_required
            ),
            "required_model_backed_lanes": required_lanes,
            "model_backed_lane_coverage_rate": _rate(
                covered_lanes,
                len(required_lanes),
            ),
            "minimum_model_backed_lane_cases": (
                min(selected_lane_required_counts)
                if selected_lane_required_counts
                else 0
            ),
            "missing_model_backed_lanes": [
                str(row["lane"])
                for row in required_lanes
                if int(row["selected_contract_cases"]) == 0
            ],
            "underrepresented_model_backed_lanes": [
                str(row["lane"])
                for row in required_lanes
                if int(row["selected_contract_cases"]) < MIN_CASES_PER_MODEL_BACKED_LANE
            ],
            "production_schema_coverage_rate": _rate(
                covered_schemas, schema_denominator
            ),
            "minimum_production_schema_cases": (
                min(selected_required_counts) if selected_required_counts else 0
            ),
            "missing_production_schemas": [
                name
                for row in required_schemas
                if int(row["selected_cases"]) == 0
                for name in row["names"]
            ],
            "underrepresented_production_schemas": [
                name
                for row in required_schemas
                if int(row["selected_cases"]) < MIN_CASES_PER_PRODUCTION_SCHEMA
                for name in row["names"]
            ],
        }


ModelMetadataProvider = Callable[[Sequence[str]], Mapping[str, Any]]


def _set_coverage_rate(selected: Sequence[str], available: Sequence[str]) -> float:
    expected = set(available)
    if not expected:
        return 0.0
    return round(len(set(selected) & expected) / len(expected), 6)




def _decision_signature(
    value: Any,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = (
        decision_signature_value(schema, value)
        if schema is not None
        else default_decision_value(value)
    )
    return dict(selected) if isinstance(selected, Mapping) else {}


def replay_agreement_value(value: Any) -> dict[str, Any]:
    """Use the same bounded expected decision signature for every vote."""

    signature = _decision_signature(value)
    if not signature:
        raise ValueError("replay output has no decision signature")
    return signature


def _case_id(
    index: int,
    prompt: str,
    system: str | None,
    schema: Mapping[str, Any],
    expected: Mapping[str, Any],
    decision_lane: str | None = None,
    lane_contract_sha256: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "index": index,
        "prompt": prompt,
        "schema": schema,
        "expected": expected,
    }
    if system is not None:
        payload["system"] = system
    if decision_lane is not None:
        payload["decision_lane"] = decision_lane
        payload["lane_contract_sha256"] = lane_contract_sha256
    return _sha256_json(payload)


def _validate_expected(
    expected: Mapping[str, Any], schema: Mapping[str, Any], *, line_number: int
) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        raise ReplayInputError(
            f"line {line_number}: replay schema must expose object properties"
        )
    if not expected:
        raise ReplayInputError(f"line {line_number}: expected signature is empty")
    for key, value in expected.items():
        child = properties.get(key)
        if not isinstance(child, Mapping):
            raise ReplayInputError(
                f"line {line_number}: expected field {key!r} is absent from schema"
            )
        issues = validate_json(value, child)
        if issues:
            issue = issues[0]
            raise ReplayInputError(
                f"line {line_number}: expected field {key!r} violates "
                f"{issue.keyword} at {issue.pointer or '/'}"
            )
    if not _decision_signature(expected, schema):
        raise ReplayInputError(
            f"line {line_number}: expected must contain a decision signature"
        )


def load_replay_corpus(
    path: Path | str = REPLAY_FILE,
    *,
    offset: int = 0,
    limit: int = 0,
    required_schema_manifest: Mapping[str, str] | None = None,
    exclude_stale_historical_identity: bool = False,
    allow_empty_after_stale_exclusion: bool = False,
) -> ReplayCorpus:
    """Load and validate replay JSONL without invoking any model."""

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ReplayInputError("offset must be an integer >= 0")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ReplayInputError("limit must be an integer >= 0")
    uses_production_manifest = required_schema_manifest is None
    manifest = dict(
        production_schema_manifest()
        if uses_production_manifest
        else required_schema_manifest
    )
    if not manifest or any(
        not isinstance(name, str)
        or not name
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for name, digest in manifest.items()
    ):
        raise ReplayInputError(
            "required_schema_manifest must contain named SHA-256 digests"
        )
    signature_manifest = (
        production_signature_manifest()
        if uses_production_manifest
        else {
            name: {
                "policy_version": 1,
                "schema_sha256": digest,
                "fields": [],
            }
            for name, digest in manifest.items()
        }
    )
    # The adoption artifact is consumed from services with unrelated working
    # directories.  Bind it to a durable absolute source path so runtime can
    # re-open the exact corpus instead of trusting copied summary metadata.
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ReplayInputError(f"cannot read replay input: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplayInputError(f"replay input is not UTF-8: {exc}") from exc

    rows: list[ReplayCase] = []
    excluded_reasons: dict[str, int] = {}
    total_cases = 0
    for index, line in enumerate(text.splitlines()):
        line_number = index + 1
        if not line.strip():
            raise ReplayInputError(f"line {line_number}: blank JSONL record")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayInputError(
                f"line {line_number}: invalid JSON at column {exc.colno}"
            ) from exc
        if not isinstance(row, dict):
            raise ReplayInputError(f"line {line_number}: record must be an object")
        total_cases += 1
        prompt = row.get("prompt")
        system = row.get("system")
        schema = row.get("schema")
        expected = row.get("expected")
        role = row.get("role")
        source_name = row.get("source")
        contract_id = row.get("contract_id")
        evidence_provenance = row.get("evidence_provenance")
        decision_lane = row.get("decision_lane")
        declared_lane_contract_sha256 = row.get("lane_contract_sha256")
        declared_lane_contract_effect = row.get("lane_contract_effect")
        declared_case_manifest_sha256 = row.get("lane_contract_case_manifest_sha256")
        if not isinstance(prompt, str) or not prompt:
            raise ReplayInputError(
                f"line {line_number}: prompt must be a non-empty string"
            )
        if system is not None and not isinstance(system, str):
            raise ReplayInputError(
                f"line {line_number}: system must be a string or null"
            )
        prompt_truncated = row.get("prompt_truncated")
        if prompt_truncated is not None and not isinstance(prompt_truncated, bool):
            raise ReplayInputError(
                f"line {line_number}: prompt_truncated must be a boolean"
            )
        exclusion_reason: str | None = None
        if prompt_truncated is True:
            exclusion_reason = "explicit_prompt_truncated"
        elif prompt_truncated is None and len(prompt) == LEGACY_REPLAY_PROMPT_LIMIT:
            # Older record_replay_case() silently retained the last 50,000
            # characters.  An exact-length unmarked prompt from that format
            # cannot prove that its leading instructions are intact.
            exclusion_reason = "legacy_exact_50000_without_marker"
        if exclusion_reason is not None:
            excluded_reasons[exclusion_reason] = (
                excluded_reasons.get(exclusion_reason, 0) + 1
            )
            continue
        if not isinstance(schema, dict):
            raise ReplayInputError(f"line {line_number}: schema must be an object")
        if not isinstance(expected, dict):
            raise ReplayInputError(f"line {line_number}: expected must be an object")
        if not isinstance(role, str) or not role:
            raise ReplayInputError(f"line {line_number}: role must be non-empty")
        if source_name is not None and (
            not isinstance(source_name, str) or not source_name.strip()
        ):
            raise ReplayInputError(
                f"line {line_number}: source must be a non-empty string or null"
            )
        normalized_source = (
            source_name.strip() if isinstance(source_name, str) else None
        )
        is_deterministic_contract = normalized_source == LANE_CONTRACT_SOURCE
        may_exclude_stale_identity = (
            exclude_stale_historical_identity and not is_deterministic_contract
        )

        def reject_or_exclude_stale_identity(message: str) -> bool:
            if may_exclude_stale_identity:
                excluded_reasons[STALE_HISTORICAL_REQUEST_IDENTITY_EXCLUSION] = (
                    excluded_reasons.get(
                        STALE_HISTORICAL_REQUEST_IDENTITY_EXCLUSION,
                        0,
                    )
                    + 1
                )
                return True
            raise ReplayInputError(f"line {line_number}: {message}")

        if evidence_provenance is not None and not isinstance(
            evidence_provenance, dict
        ):
            raise ReplayInputError(
                f"line {line_number}: evidence_provenance must be an object or null"
            )
        lane_fields = (
            decision_lane,
            declared_lane_contract_sha256,
            declared_lane_contract_effect,
        )
        if any(value is not None for value in lane_fields):
            if (
                not isinstance(decision_lane, str)
                or not decision_lane
                or not isinstance(declared_lane_contract_sha256, str)
                or not isinstance(declared_lane_contract_effect, str)
                or not declared_lane_contract_effect
            ):
                if reject_or_exclude_stale_identity(
                    "lane contract metadata is incomplete"
                ):
                    continue
            try:
                validate_declared_lane_contract(
                    lane=decision_lane,
                    contract_sha256=declared_lane_contract_sha256,
                    schema=schema,
                )
            except ValueError as exc:
                if reject_or_exclude_stale_identity(
                    f"invalid lane contract metadata: {exc}"
                ):
                    continue
        if is_deterministic_contract:
            if not all(value is not None for value in lane_fields):
                raise ReplayInputError(
                    f"line {line_number}: deterministic lane contract row lacks identity"
                )
            if (
                row.get("contract_version") != LANE_CONTRACT_CASE_VERSION
                or not isinstance(contract_id, str)
                or not contract_id
                or declared_case_manifest_sha256
                != decision_lane_contract_case_manifest_sha256()
            ):
                raise ReplayInputError(
                    f"line {line_number}: stale deterministic lane case identity"
                )
        try:
            validate_schema_definition(schema)
        except Exception as exc:
            raise ReplayInputError(
                f"line {line_number}: invalid schema: {exc}"
            ) from exc
        _validate_expected(expected, schema, line_number=line_number)
        schema_copy = json.loads(_canonical_json(schema))
        expected_copy = json.loads(_canonical_json(expected))
        signature = _decision_signature(expected_copy, schema_copy)
        try:
            effective_model_prompt, effective_model_system = decision_effective_request(
                prompt=prompt,
                schema=schema_copy,
                system=system,
                decision_lane=(
                    decision_lane if isinstance(decision_lane, str) else None
                ),
            )
            effective_request_sha256 = decision_request_fingerprint_sha256(
                prompt=prompt,
                schema=schema_copy,
                system=system,
                decision_lane=(
                    decision_lane if isinstance(decision_lane, str) else None
                ),
            )
        except ValueError as exc:
            if reject_or_exclude_stale_identity(
                f"invalid effective request identity: {exc}"
            ):
                continue
            raise AssertionError("unreachable") from exc
        evidence_mismatches: list[str] = []
        if "effective_model_prompt_chars" in row:
            declared = row["effective_model_prompt_chars"]
            if (
                isinstance(declared, bool)
                or not isinstance(declared, int)
                or declared != len(effective_model_prompt)
            ):
                evidence_mismatches.append("effective_model_prompt_chars")
        if "effective_model_prompt_sha256" in row:
            declared = row["effective_model_prompt_sha256"]
            if (
                declared
                != hashlib.sha256(effective_model_prompt.encode("utf-8")).hexdigest()
            ):
                evidence_mismatches.append("effective_model_prompt_sha256")
        if "effective_model_system" in row:
            if row["effective_model_system"] != effective_model_system:
                evidence_mismatches.append("effective_model_system")
        if "effective_model_system_chars" in row:
            declared = row["effective_model_system_chars"]
            expected_chars = (
                len(effective_model_system)
                if isinstance(effective_model_system, str)
                else 0
            )
            if (
                isinstance(declared, bool)
                or not isinstance(declared, int)
                or declared != expected_chars
            ):
                evidence_mismatches.append("effective_model_system_chars")
        if "effective_model_system_sha256" in row:
            expected_sha256 = (
                hashlib.sha256(effective_model_system.encode("utf-8")).hexdigest()
                if isinstance(effective_model_system, str)
                else None
            )
            if row["effective_model_system_sha256"] != expected_sha256:
                evidence_mismatches.append("effective_model_system_sha256")
        if "host_sidecar_present" in row:
            declared = row["host_sidecar_present"]
            if not isinstance(declared, bool) or declared is not (
                effective_model_prompt != prompt
            ):
                evidence_mismatches.append("host_sidecar_present")
        if evidence_mismatches and reject_or_exclude_stale_identity(
            "effective model request evidence mismatch: "
            + ", ".join(evidence_mismatches)
        ):
            continue
        declared_request_sha256 = row.get("effective_request_sha256")
        if declared_request_sha256 is not None and (
            not isinstance(declared_request_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", declared_request_sha256) is None
            or declared_request_sha256 != effective_request_sha256
        ):
            if reject_or_exclude_stale_identity(
                "effective request fingerprint mismatch"
            ):
                continue
        expected_effect = _semantic_effect(
            expected_copy,
            schema_copy,
            prompt=prompt,
            decision_lane=decision_lane if isinstance(decision_lane, str) else None,
        )
        if (
            isinstance(declared_lane_contract_effect, str)
            and declared_lane_contract_effect != expected_effect
        ):
            if reject_or_exclude_stale_identity(
                "lane contract effect no longer matches evaluator"
            ):
                continue
        rows.append(
            ReplayCase(
                index=index,
                case_id=_case_id(
                    index,
                    prompt,
                    system,
                    schema_copy,
                    expected_copy,
                    decision_lane if isinstance(decision_lane, str) else None,
                    declared_lane_contract_sha256
                    if isinstance(declared_lane_contract_sha256, str)
                    else None,
                ),
                role=role,
                source=normalized_source,
                contract_id=contract_id if isinstance(contract_id, str) else None,
                decision_lane=(
                    decision_lane if isinstance(decision_lane, str) else None
                ),
                lane_contract_sha256=(
                    declared_lane_contract_sha256
                    if isinstance(declared_lane_contract_sha256, str)
                    else None
                ),
                lane_contract_effect=(
                    declared_lane_contract_effect
                    if isinstance(declared_lane_contract_effect, str)
                    else None
                ),
                lane_contract_case_manifest_sha256=(
                    declared_case_manifest_sha256
                    if isinstance(declared_case_manifest_sha256, str)
                    else None
                ),
                evidence_provenance=json.loads(
                    _canonical_json(evidence_provenance or {})
                ),
                prompt=prompt,
                system=system,
                schema=schema_copy,
                expected=expected_copy,
                schema_sha256=_sha256_json(schema_copy),
                expected_signature_sha256=_sha256_json(signature),
                effective_request_sha256=effective_request_sha256,
            )
        )

    request_groups: dict[str, list[ReplayCase]] = {}
    for case in rows:
        request_groups.setdefault(case.effective_request_sha256, []).append(case)
    exact_duplicate_groups = 0
    exact_duplicate_rows = 0
    exact_duplicate_redundant_rows = 0
    conflicting_groups = 0
    conflicting_rows = 0
    for group in request_groups.values():
        if len(group) < 2:
            continue
        expected_identities = {
            (
                case.expected_signature_sha256,
                _semantic_effect(
                    case.expected,
                    case.schema,
                    prompt=case.prompt,
                    decision_lane=case.decision_lane,
                ),
            )
            for case in group
        }
        if len(expected_identities) > 1:
            conflicting_groups += 1
            conflicting_rows += len(group)
        else:
            exact_duplicate_groups += 1
            exact_duplicate_rows += len(group)
            exact_duplicate_redundant_rows += len(group) - 1

    usable_roles = tuple(sorted({case.role for case in rows}))
    usable_decisions = tuple(
        sorted(
            {
                decision
                for case in rows
                if (decision := case.expected_coverage_label) is not None
            }
        )
    )
    usable_schema_counts = Counter(case.schema_sha256 for case in rows)
    usable_lane_contract_counts = Counter(
        case.decision_lane
        for case in rows
        if case.source == LANE_CONTRACT_SOURCE and case.decision_lane is not None
    )
    end = None if limit == 0 else offset + limit
    selected = tuple(rows[offset:end])
    empty_is_migration_only = bool(
        allow_empty_after_stale_exclusion
        and exclude_stale_historical_identity
        and not rows
        and total_cases > 0
        and excluded_reasons.get(STALE_HISTORICAL_REQUEST_IDENTITY_EXCLUSION)
        == total_cases
    )
    if not selected and not empty_is_migration_only:
        raise ReplayInputError(
            "selection is empty "
            f"(total={total_cases}, usable={len(rows)}, "
            f"excluded={total_cases - len(rows)}, offset={offset}, limit={limit})"
        )
    return ReplayCorpus(
        path=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        total_cases=total_cases,
        usable_cases=len(rows),
        excluded_cases=total_cases - len(rows),
        excluded_reasons=tuple(sorted(excluded_reasons.items())),
        offset=offset,
        limit=limit,
        cases=selected,
        usable_roles=usable_roles,
        usable_decisions=usable_decisions,
        required_schema_manifest=tuple(sorted(manifest.items())),
        usable_schema_counts=tuple(sorted(usable_schema_counts.items())),
        usable_lane_contract_counts=tuple(sorted(usable_lane_contract_counts.items())),
        production_lane_contracts_required=uses_production_manifest,
        signature_manifest_sha256=_sha256_json(signature_manifest),
        effective_request_unique_count=len(request_groups),
        exact_duplicate_request_groups=exact_duplicate_groups,
        exact_duplicate_request_rows=exact_duplicate_rows,
        exact_duplicate_redundant_rows=exact_duplicate_redundant_rows,
        conflicting_request_groups=conflicting_groups,
        conflicting_request_rows=conflicting_rows,
    )


def inspect_replays(
    path: Path | str = REPLAY_FILE,
    *,
    offset: int = 0,
    limit: int = 0,
    include_cases: bool = False,
    required_schema_manifest: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect corpus integrity and selection without model/metadata calls."""

    return load_replay_corpus(
        path,
        offset=offset,
        limit=limit,
        required_schema_manifest=required_schema_manifest,
    ).inspection(include_cases=include_cases)


def fetch_local_model_metadata(models: Sequence[str]) -> Mapping[str, Any]:
    """Read Ollama engine/tag metadata without loading or running a model."""

    timeout = httpx.Timeout(connect=3.0, read=10.0, write=3.0, pool=3.0)
    with httpx.Client(base_url=ollama.OLLAMA_URL, timeout=timeout) as client:
        version_response = client.get("/api/version")
        version_response.raise_for_status()
        tags_response = client.get("/api/tags")
        tags_response.raise_for_status()
    version_body = version_response.json()
    tags_body = tags_response.json()
    available = tags_body.get("models") if isinstance(tags_body, dict) else []
    available = available if isinstance(available, list) else []
    records: dict[str, Any] = {}
    for requested in models:
        match = next(
            (
                row
                for row in available
                if isinstance(row, dict)
                and requested
                in {str(row.get("name") or ""), str(row.get("model") or "")}
            ),
            None,
        )
        records[requested] = match or {"name": requested, "status": "missing"}
    return {
        "engine": {
            "name": "ollama",
            "version": version_body.get("version")
            if isinstance(version_body, dict)
            else None,
        },
        "models": records,
    }


def _safe_model_metadata(
    payload: Mapping[str, Any], models: Sequence[str]
) -> dict[str, Any]:
    engine = payload.get("engine")
    safe_engine = {
        "name": str(engine.get("name") or "") if isinstance(engine, Mapping) else "",
        "version": str(engine.get("version") or "")
        if isinstance(engine, Mapping)
        else "",
    }
    source_models = payload.get("models")
    source_models = source_models if isinstance(source_models, Mapping) else {}
    safe_models: dict[str, Any] = {}
    detail_keys = (
        "families",
        "family",
        "format",
        "parameter_size",
        "parent_model",
        "quantization_level",
    )
    for model in models:
        source = source_models.get(model)
        source = source if isinstance(source, Mapping) else {}
        details = source.get("details")
        details = details if isinstance(details, Mapping) else {}
        safe_models[model] = {
            "name": str(source.get("name") or source.get("model") or model),
            "digest": str(source.get("digest") or ""),
            "size": source.get("size") if isinstance(source.get("size"), int) else None,
            "modified_at": str(source.get("modified_at") or ""),
            "status": str(
                source.get("status") or ("available" if source else "missing")
            ),
            "details": {key: details.get(key) for key in detail_keys if key in details},
            "metadata_sha256": _sha256_json(source),
        }
    return {"engine": safe_engine, "models": safe_models}


def validate_model_metadata_identity(
    metadata: Mapping[str, Any],
    models: Sequence[str],
) -> None:
    """Require the engine and quantized model identities used by the gate."""

    engine = metadata.get("engine")
    if (
        not isinstance(engine, Mapping)
        or engine.get("name") != "ollama"
        or not isinstance(engine.get("version"), str)
        or not str(engine["version"]).strip()
    ):
        raise ValueError("model metadata has no exact Ollama engine identity")
    records = metadata.get("models")
    if not isinstance(records, Mapping):
        raise ValueError("model metadata records are missing")
    digests: list[str] = []
    for model in models:
        record = records.get(model)
        details = record.get("details") if isinstance(record, Mapping) else None
        digest = record.get("digest") if isinstance(record, Mapping) else None
        quantization = (
            details.get("quantization_level") if isinstance(details, Mapping) else None
        )
        if (
            not isinstance(record, Mapping)
            or record.get("status") == "missing"
            or not isinstance(digest, str)
            or not digest.strip()
            or not isinstance(quantization, str)
            or not quantization.strip()
        ):
            raise ValueError(
                f"model {model!r} has no exact digest and quantization identity"
            )
        digests.append(digest)
    if len(set(digests)) != len(digests):
        raise ValueError("model metadata digests must be independent")


def _live_transport(request: ChatRequest) -> str | ollama.ChatResponse:
    return ollama.chat(
        [dict(message) for message in request.messages],
        model=request.model,
        format=request.schema,
        num_ctx=request.num_ctx,
        num_predict=request.num_predict,
        keep_alive=request.keep_alive,
        read_timeout_ms=request.read_timeout_ms,
        max_output_chars=request.max_output_chars,
        temperature=request.temperature,
        seed=request.seed,
        think=request.think,
        return_metadata=True,
    )


class _TimingTransport:
    def __init__(self, transport: ChatTransport) -> None:
        self.transport = transport
        self.events: list[dict[str, Any]] = []

    def __call__(self, request: ChatRequest) -> str | ollama.ChatResponse:
        started = time.perf_counter()
        ok = False
        response: str | ollama.ChatResponse | None = None
        try:
            response = self.transport(request)
            ok = True
            return response
        finally:
            prompt_eval_count = None
            eval_count = None
            if isinstance(response, ollama.ChatResponse):
                prompt_eval_count = response.prompt_eval_count
                eval_count = response.eval_count
            self.events.append(
                {
                    "model": request.model,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "ok": ok,
                    "context_accounting_available": bool(
                        isinstance(prompt_eval_count, int)
                        and not isinstance(prompt_eval_count, bool)
                        and prompt_eval_count >= 0
                        and isinstance(eval_count, int)
                        and not isinstance(eval_count, bool)
                        and eval_count >= 0
                    ),
                    "prompt_eval_count": prompt_eval_count,
                    "eval_count": eval_count,
                }
            )

    def mark(self) -> int:
        return len(self.events)

    def since(self, mark: int) -> list[dict[str, Any]]:
        return self.events[mark:]


def _semantic_effect(
    value: Mapping[str, Any] | None,
    schema: Mapping[str, Any],
    *,
    prompt: str | None = None,
    decision_lane: str | None = None,
    enforce_downstream_authorization: bool = False,
) -> str | None:
    """Map schema-specific labels to their actual durable side effect.

    Several schemas use ``approved`` to approve a classification or a no-op,
    not a page mutation. Comparing only that root word produced false unsafe
    flips such as expected ``rejected/none`` versus actual
    ``approved/unattributed`` even though both paths preserve page bytes.
    """

    if not isinstance(value, Mapping):
        return None
    decision = value.get("decision")
    decision = decision if isinstance(decision, str) else None
    if decision_lane is not None and decision in TRANSIENT_DECISIONS:
        return "hold"
    if decision in TRANSIENT_DECISIONS:
        return None

    # Shared schemas do not imply shared durable effects.  Lane-bound replay
    # cases use these directional identities so one lane cannot satisfy another
    # lane's adoption evidence merely by returning the same root word.
    if decision_lane in {
        "entity_backfill",
        "lint_safe_semantic_mutation",
        "metadata_backfill",
        "page_normalize",
    }:
        return {
            "approved": f"page_mutation:{decision_lane}",
            "rejected": "no_page_mutation",
            "quarantined": "hold",
            "needs_retry": "hold",
        }.get(decision)
    if decision_lane in {
        "recall_auto_apply",
        "recall_calibration",
        "recall_improvement",
        "search_self_tune",
    }:
        approved_effect = {
            "recall_auto_apply": "page_mutation:recall_auto_apply",
            "recall_calibration": "policy_mutation:recall_calibration",
            "recall_improvement": "policy_mutation:recall_improvement",
            "search_self_tune": "policy_mutation:search_self_tune",
        }[decision_lane]
        return {
            "approved": approved_effect,
            "rejected": "no_page_mutation",
            "quarantined": "hold",
            "needs_retry": "hold",
        }.get(decision)
    if decision_lane == "read_back_repair":
        return {
            "approved": "query_hint_mutation:read_back_repair",
            "rejected": "no_page_mutation",
            "needs_retry": "hold",
        }.get(decision)
    if decision_lane == "search_label":
        return {
            "approved": "label_artifact_mutation:search_label",
            "rejected": "no_page_mutation",
            "uncertain": "hold",
            "needs_retry": "hold",
        }.get(decision)

    schema_digest = schema_sha256(schema)
    schema_names = _production_schema_names_by_digest().get(
        schema_digest,
        frozenset(),
    )

    if "content_correction_classification" in schema_names:
        if decision in {"needs_retry", "quarantined"}:
            return "hold"
        if decision == "rejected":
            return "no_page_mutation"
        checks = value.get("semantic_checks")
        if (
            enforce_downstream_authorization
            and decision == "approved"
            and (
                not isinstance(checks, Mapping)
                or not checks
                or not all(check is True for check in checks.values())
            )
        ):
            return "hold"
        classification = value.get("classification")
        if classification in {"page_fact_wrong", "outdated"}:
            return "page_mutation_candidate"
        if classification == "wrong_retrieval":
            return "negative_retrieval_feedback"
        if classification in {
            "ambiguous",
            "none",
            "response_misquote",
            "unattributed",
        }:
            return "no_page_mutation"

    if "content_correction_review" in schema_names:
        if decision in {"needs_retry", "quarantined"}:
            return "hold"
        checks = value.get("semantic_checks")
        if (
            enforce_downstream_authorization
            and decision == "approved"
            and (
                not isinstance(checks, Mapping)
                or not checks
                or not all(check is True for check in checks.values())
            )
        ):
            return "hold"
        mutations = value.get("approved_mutations")
        if decision == "approved" and isinstance(mutations, list) and mutations:
            return "page_mutation"
        return "no_page_mutation"

    if "duplicate_resolution" in schema_names:
        return {
            "supersede_left": "page_mutation:supersede_left",
            "supersede_right": "page_mutation:supersede_right",
            "keep_both": "no_page_mutation",
            "needs_retry": "hold",
        }.get(decision)

    if "ingest_reconciliation" in schema_names:
        return {
            "apply_available": "page_mutation",
            "confirmed_noop": "no_page_mutation",
            "retry": "hold",
            "quarantined": "hold",
        }.get(decision)

    if "local_repair" in schema_names:
        status = value.get("status")
        action = value.get("action")
        if not isinstance(status, str) or not isinstance(action, str):
            return None
        if action == "escalate_to_frontier":
            return "frontier_escalation"
        if action in {
            "propose_prompt_fix",
            "propose_test_case",
            "quarantine_raw",
        }:
            return f"hold:{action}"
        if status in {"escalate", "rejected"}:
            return f"hold:{status}:{action}"
        if status == "resolved" and action == "retry_raw":
            return "repair_action:retry_raw"
        if status == "resolved" and action == "resolve_update_target":
            target_page_id = value.get("target_page_id")
            if not isinstance(target_page_id, str) or not target_page_id:
                return None
            target_sha256 = hashlib.sha256(target_page_id.encode("utf-8")).hexdigest()
            return f"repair_action:resolve_update_target:{target_sha256}"
        return None

    if "raw_replay_reconciliation" in schema_names:
        return {
            "accept_processed": "mark_raw_processed",
            "safe_replay": "raw_replay",
            "quarantine": "hold",
            "needs_retry": "hold",
        }.get(decision)

    if "retention" in schema_names:
        return {
            "archive": "archive",
            "keep_active": "no_page_mutation",
            "needs_retry": "hold",
        }.get(decision)

    if schema_names == {"orphan_link", "read_back_repair"}:
        if isinstance(prompt, str) and (
            "orphan-link disposition" in prompt or '"proposal_kind"' in prompt
        ):
            if decision == "rejected":
                return "no_page_mutation"
            if decision == "needs_retry":
                return "hold"
            match = re.search(r'"proposal_kind"\s*:\s*"([^"]+)"', prompt)
            proposal_kind = match.group(1) if match else None
            if decision == "approved" and proposal_kind == "link":
                return "page_mutation"
            if decision == "approved" and proposal_kind == "no_link":
                return "no_page_mutation"
            if decision == "approved" and proposal_kind == "retry":
                return "hold"
            return None
        if decision == "approved":
            return "durable_mutation"
        if decision == "rejected":
            return "no_page_mutation"
        if decision in {"needs_retry", "quarantined"}:
            return "hold"
        return None

    if schema_names & {
        "generic_decision",
        "lint_safe_semantic_mutation",
        "lint_tag_repair",
        "search_label",
    }:
        if decision == "approved":
            return "durable_mutation"
        if decision == "rejected":
            return "no_page_mutation"
        if decision in {"needs_retry", "quarantined"}:
            return "hold"
        return None

    action = value.get("action")
    if isinstance(action, str):
        return f"action:{action}"
    return f"decision:{decision}" if decision is not None else None


@lru_cache(maxsize=1)
def _production_schema_by_digest() -> dict[str, Mapping[str, Any]]:
    return {
        schema_sha256(schema): schema
        for schema in production_decision_schemas().values()
    }


def _bounded_effect_context(prompt: str | None) -> dict[str, Any]:
    """Persist only prompt facts that change the effect lattice."""

    if not isinstance(prompt, str):
        return {}
    match = re.search(r'"proposal_kind"\s*:\s*"([^"]+)"', prompt)
    if not match:
        return {}
    return {
        "proposal_kind": match.group(1),
        "orphan_disposition": "orphan-link disposition" in prompt,
    }


def replay_effect_context(prompt: str | None) -> dict[str, Any]:
    """Return the bounded prompt facts that affect replay mutation semantics."""

    return _bounded_effect_context(prompt)


def _semantic_checks_all_true(value: Any) -> bool | None:
    if not isinstance(value, Mapping):
        return None
    checks = value.get("semantic_checks")
    if not isinstance(checks, Mapping) or not checks:
        return None
    if any(not isinstance(item, bool) for item in checks.values()):
        return None
    return all(checks.values())


def _semantic_effect_from_redacted_signature(
    value: Mapping[str, Any] | None,
    *,
    schema_digest: str,
    effect_context: Mapping[str, Any],
    decision_lane: str | None = None,
    semantic_checks_all_true: bool | None = None,
    enforce_downstream_authorization: bool = False,
) -> str | None:
    """Rebuild an effect without prompts or literal model output."""

    if not isinstance(value, Mapping):
        return None
    reconstructed = dict(value)
    decision = reconstructed.get("decision")
    if isinstance(decision, str) and decision in TRANSIENT_DECISIONS:
        return "hold" if decision_lane is not None else None
    if semantic_checks_all_true is not None:
        reconstructed["semantic_checks"] = {
            "bounded_all_true": semantic_checks_all_true
        }
    schema = _production_schema_by_digest().get(schema_digest)
    if schema is not None:
        prompt = None
        proposal_kind = effect_context.get("proposal_kind")
        if isinstance(proposal_kind, str):
            prefix = (
                "orphan-link disposition "
                if effect_context.get("orphan_disposition") is True
                else ""
            )
            prompt = prefix + _canonical_json({"proposal_kind": proposal_kind})
        return _semantic_effect(
            reconstructed,
            schema,
            prompt=prompt,
            decision_lane=decision_lane,
            enforce_downstream_authorization=enforce_downstream_authorization,
        )
    action = reconstructed.get("action")
    if isinstance(action, str):
        return f"action:{action}"
    return f"decision:{decision}" if isinstance(decision, str) else None


def replay_semantic_effect(
    value: Mapping[str, Any] | None,
    schema: Mapping[str, Any],
    *,
    prompt: str | None = None,
    decision_lane: str | None = None,
) -> str | None:
    """Expose the evaluator's bounded effect identity to corpus preflight."""

    return _semantic_effect(
        value,
        schema,
        prompt=prompt,
        decision_lane=decision_lane,
    )


_MUTATING_EFFECTS = frozenset(
    {
        "archive",
        "mark_raw_processed",
        "negative_retrieval_feedback",
        "page_mutation",
        "page_mutation_candidate",
        "frontier_escalation",
        "raw_replay",
        "durable_mutation",
        *(f"decision:{decision}" for decision in APPLY_DECISIONS),
    }
)


def _is_mutating_effect(effect: str | None) -> bool:
    return bool(
        effect in _MUTATING_EFFECTS
        or (
            isinstance(effect, str)
            and effect.startswith(
                (
                    "page_mutation:",
                    "repair_action:",
                    "policy_mutation:",
                    "query_hint_mutation:",
                    "label_artifact_mutation:",
                )
            )
        )
    )


def _unsafe_effect_transition(
    *,
    expected_decision: Any,
    expected_effect: str | None,
    actual_effect: str | None,
) -> bool:
    held = (
        expected_decision in UNSAFE_HOLD_DECISIONS
        or expected_effect in {"hold", "no_page_mutation"}
        or (isinstance(expected_effect, str) and expected_effect.startswith("hold:"))
    )
    if held and _is_mutating_effect(actual_effect):
        return True
    # Duplicate lifecycle direction is part of the durable effect. Reversing
    # which page is deprecated is not an acceptable "same mutation" outcome.
    directional_prefixes = ("page_mutation:", "repair_action:")
    return bool(
        isinstance(expected_effect, str)
        and expected_effect.startswith(directional_prefixes)
        and isinstance(actual_effect, str)
        and actual_effect.startswith(directional_prefixes)
        and actual_effect != expected_effect
    )


def _unsafe_flip(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any] | None,
    schema: Mapping[str, Any],
    *,
    prompt: str | None = None,
    decision_lane: str | None = None,
) -> bool:
    expected_decision = expected.get("decision")
    expected_effect = _semantic_effect(
        expected,
        schema,
        prompt=prompt,
        decision_lane=decision_lane,
    )
    actual_effect = _semantic_effect(
        actual,
        schema,
        prompt=prompt,
        decision_lane=decision_lane,
        enforce_downstream_authorization=True,
    )
    return _unsafe_effect_transition(
        expected_decision=expected_decision,
        expected_effect=expected_effect,
        actual_effect=actual_effect,
    )


def _case_record(
    case: ReplayCase,
    result: DecisionRouterResult,
    *,
    latency_ms: float,
    transport_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    timing_by_model: dict[str, dict[str, Any]] = {}
    for event in transport_events:
        model = str(event.get("model") or "")
        aggregate = timing_by_model.setdefault(
            model,
            {
                "transport_calls": 0,
                "transport_failures": 0,
                "latency_ms": 0.0,
                "context_accounting": [],
            },
        )
        aggregate["transport_calls"] += 1
        aggregate["transport_failures"] += 0 if event.get("ok") is True else 1
        aggregate["latency_ms"] += float(event.get("latency_ms") or 0.0)
        aggregate["context_accounting"].append(
            {
                "ok": event.get("ok") is True,
                "available": event.get("context_accounting_available") is True,
                "prompt_eval_count": (
                    event.get("prompt_eval_count")
                    if isinstance(event.get("prompt_eval_count"), int)
                    and not isinstance(event.get("prompt_eval_count"), bool)
                    else None
                ),
                "eval_count": (
                    event.get("eval_count")
                    if isinstance(event.get("eval_count"), int)
                    and not isinstance(event.get("eval_count"), bool)
                    else None
                ),
            }
        )

    vote_records: list[dict[str, Any]] = []
    invalid_output_accepted = 0
    for vote in result.votes:
        attempts = vote.result.attempts
        final_attempt_valid = bool(attempts and attempts[-1].valid)
        if vote.result.ok and not final_attempt_valid:
            invalid_output_accepted += 1
        timing = timing_by_model.get(
            vote.model,
            {
                "transport_calls": 0,
                "transport_failures": 0,
                "latency_ms": 0.0,
                "context_accounting": [],
            },
        )
        context_accounting = list(timing["context_accounting"])
        successful_accounting = [
            row for row in context_accounting if row.get("ok") is True
        ]
        context_accounting_complete = bool(successful_accounting) and all(
            row.get("available") is True for row in successful_accounting
        )
        vote_signature_value = (
            _decision_signature(vote.result.value, case.schema)
            if vote.valid and isinstance(vote.result.value, Mapping)
            else None
        )
        if vote.valid and (
            not vote_signature_value
            or _sha256_json(vote_signature_value) != vote.signature_sha256
        ):
            raise ValueError("valid vote signature evidence is inconsistent")
        vote_records.append(
            {
                "role": vote.role,
                "model": vote.model,
                # num_ctx is deliberately the post-call /api/ps observation,
                # never the planner request. requested_num_ctx remains as
                # separate admission evidence.
                "num_ctx": vote.observed_num_ctx,
                "requested_num_ctx": vote.requested_num_ctx,
                "observed_model_bytes": vote.observed_model_bytes,
                "runtime_observation_status": vote.runtime_observation_status,
                "context_accounting_complete": context_accounting_complete,
                "context_accounting": context_accounting,
                "vote_valid": vote.valid,
                "first_pass_schema_valid": vote.result.first_pass_valid,
                "final_schema_valid": vote.result.ok,
                "repaired_final_valid": bool(
                    vote.result.ok and vote.result.repair_turns > 0
                ),
                "repair_turns": vote.result.repair_turns,
                "transport_calls": timing["transport_calls"],
                "transport_failures": timing["transport_failures"],
                "latency_ms": round(float(timing["latency_ms"]), 3),
                "signature_value": vote_signature_value,
                "semantic_checks_all_true": _semantic_checks_all_true(
                    vote.result.value
                ),
                "audit": vote.audit_record(),
            }
        )

    pair = result.votes[:2]
    pair_valid = len(pair) == 2 and all(vote.valid for vote in pair)
    pair_signature_agreed = bool(
        pair_valid
        and pair[0].signature is not None
        and pair[0].signature == pair[1].signature
    )
    # Deterministic safety lattices may still produce a conservative result
    # when model signatures differ.  That is useful operationally, but it is
    # not independent model agreement and must never inflate the adoption
    # quality gate.
    pair_safe_resolution_without_tie = bool(
        pair_valid
        and result.ok
        and len(result.votes) == 2
        and not pair_signature_agreed
    )
    tie_invoked = len(result.votes) == 3
    signature_counts = Counter(
        vote.signature_sha256
        for vote in result.votes
        if vote.valid and vote.signature_sha256
    )
    actual_signature = (
        _decision_signature(result.value, case.schema) if result.ok else {}
    )
    expected_signature = _decision_signature(case.expected, case.schema)
    result_semantic_checks_all_true = _semantic_checks_all_true(result.value)
    actual_decision = actual_signature.get("decision")
    actual_decision = actual_decision if isinstance(actual_decision, str) else None
    expected_decision = case.expected_decision
    # A retry/defer/uncertain expected label says that the label source did not
    # reach a semantic conclusion. It remains useful schema/latency evidence,
    # but treating a later concrete answer as a mismatch makes transient
    # infrastructure failures look like model regressions.
    actual_signature_sha256 = (
        _sha256_json(actual_signature) if actual_signature else None
    )
    majority_signature = next(
        (signature for signature, count in signature_counts.items() if count >= 2),
        None,
    )
    # Every executable agreement must be the exact action supported by two
    # independent vote signatures.  The router enforces this before returning,
    # and the evaluator independently rejects any inconsistent in-memory result
    # before it can become adoption evidence.
    if result.ok and (
        majority_signature is None
        or actual_signature_sha256 != majority_signature
        or result.agreement_sha256 != actual_signature_sha256
    ):
        raise ValueError("agreed router result lacks an exact two-vote action proof")
    signature_majority_resolved = bool(
        result.ok
        and majority_signature is not None
        and actual_signature_sha256 == majority_signature
    )
    # False semantic checks are normal evidence for an exact rejected/hold
    # decision.  They are a policy-only resolution only when no two-vote
    # action signature supports the returned value; an exact non-mutating
    # majority must retain ordinary adoption-quality credit.
    safe_policy_resolution = bool(result.ok and not signature_majority_resolved)
    comparable = bool(
        not safe_policy_resolution
        and actual_decision is not None
        and expected_decision is not None
        and expected_decision not in TRANSIENT_DECISIONS
    )
    expected_effect = _semantic_effect(
        case.expected,
        case.schema,
        prompt=case.prompt,
        decision_lane=case.decision_lane,
    )
    actual_value = result.value if isinstance(result.value, Mapping) else None
    actual_effect = _semantic_effect(
        actual_value,
        case.schema,
        prompt=case.prompt,
        decision_lane=case.decision_lane,
        enforce_downstream_authorization=True,
    )
    if safe_policy_resolution and _is_mutating_effect(actual_effect):
        raise ValueError(
            "non-majority local resolution cannot authorize a mutating effect"
        )
    effect_comparable = bool(
        not safe_policy_resolution
        and expected_effect is not None
        and actual_effect is not None
    )
    vote_context_counts = Counter(
        int(row["num_ctx"])
        for row in vote_records[:2]
        if isinstance(row.get("num_ctx"), int)
        and not isinstance(row.get("num_ctx"), bool)
        and int(row["num_ctx"]) > 0
        and row.get("runtime_observation_status") == "observed"
        and row.get("context_accounting_complete") is True
        and row.get("vote_valid") is True
        and row.get("role") in {"primary", "challenger"}
        and row.get("requested_num_ctx") == result.num_ctx
        and row.get("num_ctx") == row.get("requested_num_ctx")
    )
    # A bucket is quality evidence only after an independent two-model pair
    # actually ran at that size. One reused runner at another size must not
    # manufacture bucket coverage.
    evaluated_context_buckets = sorted(
        bucket for bucket, count in vote_context_counts.items() if count >= 2
    )
    return {
        "index": case.index,
        "case_id": case.case_id,
        "effective_request_sha256": case.effective_request_sha256,
        "role": case.role,
        "source": case.source,
        "contract_id": case.contract_id,
        "decision_lane": case.decision_lane,
        "lane_contract_sha256": case.lane_contract_sha256,
        "lane_contract_effect": case.lane_contract_effect,
        "lane_contract_case_manifest_sha256": (case.lane_contract_case_manifest_sha256),
        "evidence_provenance_sha256": _sha256_json(case.evidence_provenance),
        "schema_sha256": case.schema_sha256,
        "expected_signature": expected_signature,
        "expected_signature_sha256": case.expected_signature_sha256,
        "expected_coverage_label": case.expected_coverage_label,
        "actual_signature": actual_signature if actual_signature else None,
        "actual_signature_sha256": actual_signature_sha256,
        "semantic_checks_all_true": result_semantic_checks_all_true,
        "effect_context": _bounded_effect_context(case.prompt),
        "expected_decision": expected_decision,
        "actual_decision": actual_decision,
        "expected_decision_comparable": comparable,
        "expected_decision_match": comparable and actual_decision == expected_decision,
        "expected_effect": expected_effect,
        "actual_effect": actual_effect,
        "expected_effect_comparable": effect_comparable,
        "expected_effect_match": effect_comparable and expected_effect == actual_effect,
        "expected_signature_match": bool(
            not safe_policy_resolution
            and result.ok
            and _decision_signature(case.expected, case.schema) == actual_signature
        ),
        "unsafe_decision_flip": _unsafe_flip(
            case.expected,
            actual_value,
            case.schema,
            prompt=case.prompt,
            decision_lane=case.decision_lane,
        ),
        "status": result.status,
        "failure_class": result.failure_class,
        "quarantine_reason": result.quarantine_reason,
        "num_ctx": result.num_ctx,
        "evaluated_context_buckets": evaluated_context_buckets,
        "pair_valid": pair_valid,
        "pair_agreed": pair_signature_agreed,
        "pair_signature_agreed": pair_signature_agreed,
        "pair_safe_resolution_without_tie": pair_safe_resolution_without_tie,
        "signature_majority_resolved": signature_majority_resolved,
        "tie_break_invoked": tie_invoked,
        "tie_break_resolved": bool(tie_invoked and signature_majority_resolved),
        "invalid_output_accepted": invalid_output_accepted,
        "latency_ms": round(latency_ms, 3),
        "votes": vote_records,
    }


def adoption_case_derived_evidence(case: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute gate evidence from authoritative redacted case fields.

    The outer artifact digest detects accidental edits, but it is not a trust
    boundary: a modified artifact can be rehashed.  Gate-critical booleans
    therefore remain claims only when they can be reproduced from the stored
    decision/effect/signature identities and the vote audit.
    """

    expected_evidence_fields = {
        "status",
        "expected_coverage_label",
        "expected_signature",
        "expected_signature_sha256",
        "actual_signature",
        "actual_signature_sha256",
        "semantic_checks_all_true",
        "effect_context",
        "expected_decision",
        "actual_decision",
        "expected_decision_comparable",
        "expected_decision_match",
        "expected_effect",
        "actual_effect",
        "expected_effect_comparable",
        "expected_effect_match",
        "expected_signature_match",
        "unsafe_decision_flip",
    }
    missing_expected_evidence_fields = expected_evidence_fields - case.keys()
    if missing_expected_evidence_fields:
        raise ValueError(
            "adoption case is missing expected evidence fields: "
            + ", ".join(sorted(missing_expected_evidence_fields))
        )

    status = case.get("status")
    if status not in {"agreed", "quarantined"}:
        raise ValueError("adoption case status is invalid")
    coverage_label = case.get("expected_coverage_label")
    if not isinstance(coverage_label, str) or not coverage_label:
        raise ValueError("adoption case expected coverage label is invalid")

    expected_signature = case.get("expected_signature")
    actual_signature = case.get("actual_signature")
    expected_signature_sha256 = case.get("expected_signature_sha256")
    actual_signature_sha256 = case.get("actual_signature_sha256")
    if (
        not isinstance(expected_signature, Mapping)
        or not expected_signature
        or not isinstance(expected_signature_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_signature_sha256) is None
        or _sha256_json(expected_signature) != expected_signature_sha256
        or (
            actual_signature is not None
            and (
                not isinstance(actual_signature, Mapping)
                or not actual_signature
                or not isinstance(actual_signature_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", actual_signature_sha256) is None
                or _sha256_json(actual_signature) != actual_signature_sha256
            )
        )
        or (actual_signature is None) is not (actual_signature_sha256 is None)
    ):
        raise ValueError("adoption case signature identity is invalid")
    if (status == "agreed") is not (actual_signature_sha256 is not None):
        raise ValueError("adoption case status disagrees with its actual signature")

    semantic_checks_all_true = case.get("semantic_checks_all_true")
    if semantic_checks_all_true is not None and not isinstance(
        semantic_checks_all_true, bool
    ):
        raise ValueError("adoption case semantic authorization is invalid")
    if status == "quarantined" and semantic_checks_all_true is not None:
        raise ValueError("quarantined adoption case exposes semantic authorization")
    effect_context = case.get("effect_context")
    if not isinstance(effect_context, Mapping) or set(effect_context) - {
        "proposal_kind",
        "orphan_disposition",
    }:
        raise ValueError("adoption case effect context is invalid")
    schema_digest = case.get("schema_sha256")
    if (
        not isinstance(schema_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", schema_digest) is None
    ):
        raise ValueError("adoption case schema identity is invalid")
    decision_lane = case.get("decision_lane")
    declared_lane_sha256 = case.get("lane_contract_sha256")
    declared_lane_effect = case.get("lane_contract_effect")
    lane_fields = (decision_lane, declared_lane_sha256, declared_lane_effect)
    if case.get("source") == LANE_CONTRACT_SOURCE and not all(
        value is not None for value in lane_fields
    ):
        raise ValueError("deterministic adoption case lacks lane contract identity")
    if any(value is not None for value in lane_fields):
        if (
            not isinstance(decision_lane, str)
            or not decision_lane
            or not isinstance(declared_lane_sha256, str)
            or not isinstance(declared_lane_effect, str)
            or not declared_lane_effect
        ):
            raise ValueError("adoption case lane contract identity is incomplete")
        schema = _production_schema_by_digest().get(schema_digest)
        if schema is None:
            raise ValueError("adoption case lane schema is not production-reachable")
        try:
            validate_declared_lane_contract(
                lane=decision_lane,
                contract_sha256=declared_lane_sha256,
                schema=schema,
            )
        except ValueError as exc:
            raise ValueError("adoption case lane contract identity is stale") from exc
    expected_decision = expected_signature.get("decision")
    expected_decision = (
        expected_decision if isinstance(expected_decision, str) else None
    )
    actual_decision = (
        actual_signature.get("decision")
        if isinstance(actual_signature, Mapping)
        and isinstance(actual_signature.get("decision"), str)
        else None
    )
    expected_effect = _semantic_effect_from_redacted_signature(
        expected_signature,
        schema_digest=schema_digest,
        effect_context=effect_context,
        decision_lane=decision_lane if isinstance(decision_lane, str) else None,
    )
    actual_effect = _semantic_effect_from_redacted_signature(
        actual_signature if isinstance(actual_signature, Mapping) else None,
        schema_digest=schema_digest,
        effect_context=effect_context,
        decision_lane=decision_lane if isinstance(decision_lane, str) else None,
        semantic_checks_all_true=semantic_checks_all_true,
        enforce_downstream_authorization=True,
    )
    if any(
        case.get(name) != value
        for name, value in (
            ("expected_decision", expected_decision),
            ("actual_decision", actual_decision),
            ("expected_effect", expected_effect),
            ("actual_effect", actual_effect),
        )
    ):
        raise ValueError(
            "adoption case decision/effect disagrees with signature evidence"
        )
    if (
        isinstance(declared_lane_effect, str)
        and declared_lane_effect != expected_effect
    ):
        raise ValueError("adoption case lane contract effect is inconsistent")
    for name, value in (
        ("expected_decision", expected_decision),
        ("actual_decision", actual_decision),
        ("expected_effect", expected_effect),
        ("actual_effect", actual_effect),
    ):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"adoption case {name} is invalid")
    if status == "quarantined" and any(
        value is not None
        for value in (actual_decision, actual_effect, actual_signature_sha256)
    ):
        raise ValueError("quarantined adoption case exposes an actual result")

    decision_label = (
        f"decision={_canonical_json(expected_decision)}"
        if expected_decision is not None
        else None
    )
    if (decision_label is not None and coverage_label != decision_label) or (
        decision_label is None and coverage_label.startswith("decision=")
    ):
        raise ValueError(
            "adoption case expected decision disagrees with its coverage label"
        )

    votes = case.get("votes")
    if (
        not isinstance(votes, list)
        or len(votes) > 3
        or (status == "agreed" and len(votes) < 2)
    ):
        raise ValueError(
            "agreed adoption cases require two or three vote records; "
            "quarantined cases may contain the zero-to-three votes reached "
            "before fail-closed termination"
        )
    signatures: list[str | None] = []
    vote_authorizations: list[bool | None] = []
    valid_flags: list[bool] = []
    context_counts: Counter[int] = Counter()
    qualifying_pair_contexts: dict[str, int] = {}
    planned_context = case.get("num_ctx")
    if (
        isinstance(planned_context, bool)
        or not isinstance(planned_context, int)
        or planned_context <= 0
    ):
        raise ValueError("adoption case planned context is invalid")
    invalid_output_accepted = 0
    for vote in votes:
        if not isinstance(vote, Mapping):
            raise ValueError("adoption vote must be an object")
        audit = vote.get("audit")
        session = audit.get("session") if isinstance(audit, Mapping) else None
        attempts = session.get("attempts") if isinstance(session, Mapping) else None
        if (
            not isinstance(audit, Mapping)
            or not isinstance(session, Mapping)
            or not isinstance(attempts, list)
            or vote.get("role") != audit.get("role")
            or vote.get("model") != audit.get("model")
        ):
            raise ValueError("adoption vote audit is incomplete")
        audit_valid = audit.get("valid") is True
        session_ok = session.get("ok") is True
        repair_turns = session.get("repair_turns")
        first_pass = session.get("first_pass_valid") is True
        if (
            vote.get("vote_valid") is not audit_valid
            or vote.get("final_schema_valid") is not session_ok
            or vote.get("first_pass_schema_valid") is not first_pass
            or isinstance(repair_turns, bool)
            or not isinstance(repair_turns, int)
            or repair_turns < 0
            or vote.get("repair_turns") != repair_turns
            or vote.get("repaired_final_valid")
            is not bool(session_ok and repair_turns > 0)
            or (
                len(attempts) != repair_turns + 1
                and not (
                    not attempts
                    and not session_ok
                    and not first_pass
                    and repair_turns == 0
                )
            )
        ):
            raise ValueError("adoption vote summary disagrees with session audit")
        for attempt in attempts:
            if not isinstance(attempt, Mapping) or not isinstance(
                attempt.get("valid"), bool
            ):
                raise ValueError("adoption vote attempt audit is malformed")
        signature = audit.get("signature_sha256")
        signature_value = vote.get("signature_value")
        vote_authorization = vote.get("semantic_checks_all_true")
        if vote_authorization is not None and not isinstance(vote_authorization, bool):
            raise ValueError("adoption vote semantic authorization is invalid")
        if audit_valid:
            if (
                not isinstance(signature, str)
                or re.fullmatch(r"[0-9a-f]{64}", signature) is None
                or not isinstance(signature_value, Mapping)
                or not signature_value
                or _sha256_json(signature_value) != signature
            ):
                raise ValueError("valid adoption vote signature evidence is invalid")
            signatures.append(signature)
        else:
            if signature is not None or signature_value is not None:
                raise ValueError("invalid adoption vote exposes signature evidence")
            signatures.append(None)
        vote_authorizations.append(vote_authorization)
        valid_flags.append(audit_valid)
        requested_context = vote.get("requested_num_ctx")
        if (
            isinstance(requested_context, bool)
            or not isinstance(requested_context, int)
            or requested_context <= 0
            or audit.get("requested_num_ctx") != requested_context
        ):
            raise ValueError("adoption vote requested context is invalid")
        runtime_observation = audit.get("runtime_observation")
        observation_status = vote.get("runtime_observation_status")
        observed_model_bytes = vote.get("observed_model_bytes")
        context = vote.get("num_ctx")
        if (
            not isinstance(runtime_observation, Mapping)
            or observation_status
            not in {"observed", "unavailable", "observer_error", "not_requested"}
            or runtime_observation.get("status") != observation_status
            or runtime_observation.get("model_size_bytes") != observed_model_bytes
            or runtime_observation.get("num_ctx") != context
        ):
            raise ValueError("adoption vote runtime observation is inconsistent")
        if observation_status == "observed":
            if (
                isinstance(observed_model_bytes, bool)
                or not isinstance(observed_model_bytes, int)
                or observed_model_bytes <= 0
                or isinstance(context, bool)
                or not isinstance(context, int)
                or context <= 0
            ):
                raise ValueError("observed adoption vote runtime is invalid")
        elif observed_model_bytes is not None or context is not None:
            raise ValueError("unobserved adoption vote exposes runtime values")
        transport_calls = vote.get("transport_calls")
        transport_failures = vote.get("transport_failures")
        if (
            isinstance(transport_calls, bool)
            or not isinstance(transport_calls, int)
            or transport_calls < len(attempts)
            or isinstance(transport_failures, bool)
            or not isinstance(transport_failures, int)
            or not 0 <= transport_failures <= transport_calls
        ):
            raise ValueError("adoption vote transport accounting is invalid")
        context_accounting = vote.get("context_accounting")
        if (
            not isinstance(context_accounting, list)
            or len(context_accounting) != transport_calls
        ):
            raise ValueError("adoption vote context accounting is incomplete")
        successful_accounting = 0
        complete_accounting = True
        for accounting in context_accounting:
            if (
                not isinstance(accounting, Mapping)
                or not isinstance(accounting.get("ok"), bool)
                or not isinstance(accounting.get("available"), bool)
            ):
                raise ValueError("adoption vote context accounting is malformed")
            accounting_prompt = accounting.get("prompt_eval_count")
            accounting_eval = accounting.get("eval_count")
            for value in (accounting_prompt, accounting_eval):
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                ):
                    raise ValueError("adoption vote context count is invalid")
            if accounting.get("ok") is True:
                successful_accounting += 1
                if accounting.get("available") is not True:
                    complete_accounting = False
            if accounting.get("available") is True and (
                accounting.get("ok") is not True
                or not isinstance(accounting_prompt, int)
                or isinstance(accounting_prompt, bool)
                or not isinstance(accounting_eval, int)
                or isinstance(accounting_eval, bool)
            ):
                raise ValueError("available context accounting has no counts")
        expected_context_accounting_complete = bool(successful_accounting) and (
            complete_accounting
        )
        if (
            vote.get("context_accounting_complete")
            is not expected_context_accounting_complete
        ):
            raise ValueError("adoption vote context accounting summary is inconsistent")
        if (
            audit_valid
            and observation_status == "observed"
            and expected_context_accounting_complete
            and requested_context == planned_context
            and context == requested_context
            and vote.get("role") in {"primary", "challenger"}
        ):
            qualifying_pair_contexts[str(vote["role"])] = context
        if session_ok and attempts and attempts[-1].get("valid") is not True:
            invalid_output_accepted += 1

    # Every agreed artifact row must reconstruct the exact action supported by
    # at least two vote signatures. Quarantined rows may still expose a raw vote
    # majority (for example, one vetoed by a conservative dissent), but they have
    # no result action and therefore receive no resolution or mutation credit.
    signature_counts = Counter(signature for signature in signatures if signature)
    majority_signatures = [
        signature for signature, count in signature_counts.items() if count >= 2
    ]
    if len(majority_signatures) > 1:
        raise ValueError("adoption case exposes multiple signature majorities")
    majority_signature = majority_signatures[0] if majority_signatures else None
    if status == "agreed":
        if majority_signature is None:
            raise ValueError("agreed adoption result lacks a vote signature majority")
        if actual_signature_sha256 != majority_signature:
            raise ValueError(
                "agreed adoption result does not match its vote signature majority"
            )
        winning_authorizations = [
            authorization
            for signature, authorization in zip(
                signatures,
                vote_authorizations,
                strict=True,
            )
            if signature == majority_signature
        ]
        if any(value is not None for value in winning_authorizations):
            if any(value is None for value in winning_authorizations):
                raise ValueError(
                    "majority vote semantic authorization evidence is incomplete"
                )
            expected_result_authorization: bool | None = all(
                value is True for value in winning_authorizations
            )
        else:
            expected_result_authorization = None
        if semantic_checks_all_true is not expected_result_authorization:
            raise ValueError(
                "result semantic authorization disagrees with majority votes"
            )
    safe_policy_resolution = bool(status == "agreed" and majority_signature is None)
    if safe_policy_resolution and _is_mutating_effect(actual_effect):
        raise ValueError("non-quorum adoption resolution exposes a mutating effect")

    expected_decision_comparable = bool(
        not safe_policy_resolution
        and actual_decision is not None
        and expected_decision is not None
        and expected_decision not in TRANSIENT_DECISIONS
    )
    expected_decision_match = bool(
        expected_decision_comparable and actual_decision == expected_decision
    )
    expected_effect_comparable = bool(
        not safe_policy_resolution
        and expected_effect is not None
        and actual_effect is not None
    )
    expected_effect_match = bool(
        expected_effect_comparable and expected_effect == actual_effect
    )
    expected_signature_match = bool(
        not safe_policy_resolution
        and actual_signature_sha256 is not None
        and actual_signature_sha256 == expected_signature_sha256
    )
    if expected_signature_match and actual_decision != expected_decision:
        raise ValueError("matching adoption signatures expose different root decisions")
    unsafe_decision_flip = _unsafe_effect_transition(
        expected_decision=expected_decision,
        expected_effect=expected_effect,
        actual_effect=actual_effect,
    )

    pair_valid = len(votes) >= 2 and all(valid_flags[:2])
    if (
        qualifying_pair_contexts.get("primary") == planned_context
        and qualifying_pair_contexts.get("challenger") == planned_context
    ):
        context_counts[planned_context] = 2
    pair_signature_agreed = bool(
        pair_valid and signatures[0] is not None and signatures[0] == signatures[1]
    )
    signature_majority_resolved = bool(
        majority_signature is not None and actual_signature_sha256 == majority_signature
    )
    tie_break_invoked = len(votes) == 3
    pair_safe_resolution_without_tie = bool(
        pair_valid
        and case.get("status") == "agreed"
        and not tie_break_invoked
        and not pair_signature_agreed
    )
    return {
        "expected_signature_sha256": expected_signature_sha256,
        "actual_signature_sha256": actual_signature_sha256,
        "expected_decision": expected_decision,
        "actual_decision": actual_decision,
        "expected_decision_comparable": expected_decision_comparable,
        "expected_decision_match": expected_decision_match,
        "expected_effect": expected_effect,
        "actual_effect": actual_effect,
        "expected_effect_comparable": expected_effect_comparable,
        "expected_effect_match": expected_effect_match,
        "expected_signature_match": expected_signature_match,
        "unsafe_decision_flip": unsafe_decision_flip,
        "pair_valid": pair_valid,
        "pair_agreed": pair_signature_agreed,
        "pair_signature_agreed": pair_signature_agreed,
        "pair_safe_resolution_without_tie": pair_safe_resolution_without_tie,
        "signature_majority_resolved": signature_majority_resolved,
        "tie_break_invoked": tie_break_invoked,
        "tie_break_resolved": bool(tie_break_invoked and signature_majority_resolved),
        "invalid_output_accepted": invalid_output_accepted,
        "evaluated_context_buckets": sorted(
            context for context, count in context_counts.items() if count >= 2
        ),
    }


def validate_adoption_case_derived_evidence(case: Mapping[str, Any]) -> bool:
    try:
        expected = adoption_case_derived_evidence(case)
    except ValueError:
        return False
    for name, value in expected.items():
        observed = case.get(name)
        if isinstance(value, bool):
            if observed is not value:
                return False
        elif isinstance(value, int):
            if isinstance(observed, bool) or observed != value:
                return False
        elif observed != value:
            return False
    return True


def _rate(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    if denominator <= 0:
        return empty
    return round(numerator / denominator, 6)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return round(ordered[low] + (ordered[high] - ordered[low]) * fraction, 3)


def _canonical_lane_exact_signature_metrics(
    cases: Sequence[Mapping[str, Any]],
    derived_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require every canonical case in every model lane to match exactly."""

    if len(cases) != len(derived_rows):
        raise ValueError("canonical lane metrics require one derived row per case")
    manifest = decision_lane_contract_case_manifest()
    manifest_lanes = manifest.get("lanes")
    required_lanes = model_backed_lane_names()
    if not isinstance(manifest_lanes, Mapping) or set(manifest_lanes) != set(
        required_lanes
    ):
        raise ValueError("canonical lane case manifest does not match runtime lanes")

    by_lane: dict[str, dict[str, Any]] = {}
    matched_lanes = 0
    current_manifest_sha256 = decision_lane_contract_case_manifest_sha256()
    for lane in required_lanes:
        lane_manifest = manifest_lanes[lane]
        if not isinstance(lane_manifest, Mapping):
            raise ValueError(f"canonical lane case manifest is malformed: {lane}")
        canonical_cases = lane_manifest.get("cases")
        if not isinstance(canonical_cases, list) or not canonical_cases:
            raise ValueError(f"canonical lane has no cases: {lane}")
        canonical_identities = sorted(
            (
                {
                    "contract_id": row.get("contract_id"),
                    "effective_request_sha256": row.get("effective_request_sha256"),
                    "expected_signature_sha256": row.get("expected_signature_sha256"),
                }
                for row in canonical_cases
                if isinstance(row, Mapping)
            ),
            key=lambda row: str(row["contract_id"]),
        )
        if len(canonical_identities) != len(canonical_cases):
            raise ValueError(f"canonical lane case manifest is malformed: {lane}")
        canonical_by_id = {str(row["contract_id"]): row for row in canonical_identities}

        observed_identities: list[dict[str, Any]] = []
        exact_match_ids: set[str] = set()
        observed_manifest_is_current = True
        for case, derived in zip(cases, derived_rows, strict=True):
            if (
                case.get("source") != LANE_CONTRACT_SOURCE
                or case.get("decision_lane") != lane
            ):
                continue
            identity = {
                "contract_id": case.get("contract_id"),
                "effective_request_sha256": case.get("effective_request_sha256"),
                "expected_signature_sha256": case.get("expected_signature_sha256"),
            }
            observed_identities.append(identity)
            if (
                case.get("lane_contract_case_manifest_sha256")
                != current_manifest_sha256
            ):
                observed_manifest_is_current = False
            contract_id = case.get("contract_id")
            if (
                isinstance(contract_id, str)
                and canonical_by_id.get(contract_id) == identity
                and case.get("actual_signature_sha256")
                == case.get("expected_signature_sha256")
                and derived.get("signature_majority_resolved") is True
            ):
                exact_match_ids.add(contract_id)

        observed_identities.sort(key=lambda row: str(row["contract_id"]))
        exact_case_set = bool(
            observed_manifest_is_current and observed_identities == canonical_identities
        )
        required_cases = len(canonical_identities)
        exact_matches = len(exact_match_ids)
        all_canonical_cases_match = bool(
            exact_case_set and exact_matches == required_cases
        )
        if all_canonical_cases_match:
            matched_lanes += 1
        by_lane[lane] = {
            "required_cases": required_cases,
            "observed_cases": len(observed_identities),
            "exact_signature_matches": exact_matches,
            "exact_signature_match_rate": _rate(exact_matches, required_cases),
            "canonical_case_set_sha256": _sha256_json(canonical_identities),
            "observed_case_set_sha256": _sha256_json(observed_identities),
            "exact_canonical_case_set": exact_case_set,
            "all_canonical_cases_match": all_canonical_cases_match,
        }

    return {
        "canonical_lane_exact_signature_required_lanes": len(required_lanes),
        "canonical_lane_exact_signature_matched_lanes": matched_lanes,
        "canonical_lane_exact_signature_match_rate": _rate(
            matched_lanes,
            len(required_lanes),
        ),
        "canonical_lane_exact_signature_by_lane": by_lane,
    }


def _canonical_lane_exact_signature_gate_passed(
    metrics: Mapping[str, Any],
) -> bool:
    required_lanes = model_backed_lane_names()
    rows = metrics.get("canonical_lane_exact_signature_by_lane")
    return bool(
        metrics.get("canonical_lane_exact_signature_required_lanes")
        == len(required_lanes)
        and metrics.get("canonical_lane_exact_signature_matched_lanes")
        == len(required_lanes)
        and metrics.get("canonical_lane_exact_signature_match_rate") == 1.0
        and isinstance(rows, Mapping)
        and set(rows) == set(required_lanes)
        and all(
            isinstance(rows[lane], Mapping)
            and rows[lane].get("required_cases")
            == decision_lane_contract_case_manifest()["lanes"][lane]["case_count"]
            and rows[lane].get("observed_cases") == rows[lane].get("required_cases")
            and rows[lane].get("exact_signature_matches")
            == rows[lane].get("required_cases")
            and rows[lane].get("exact_signature_match_rate") == 1.0
            and rows[lane].get("exact_canonical_case_set") is True
            and rows[lane].get("all_canonical_cases_match") is True
            for lane in required_lanes
        )
    )


def adoption_metrics(
    cases: Sequence[Mapping[str, Any]],
    *,
    required_context_buckets: Sequence[int] = (),
) -> dict[str, Any]:
    model_rows: dict[str, list[Mapping[str, Any]]] = {}
    vote_rows: list[Mapping[str, Any]] = []
    derived_rows: list[dict[str, Any]] = []
    for case in cases:
        derived_rows.append(adoption_case_derived_evidence(case))
        for vote in case.get("votes", []):
            if not isinstance(vote, Mapping):
                continue
            vote_rows.append(vote)
            model_rows.setdefault(str(vote.get("model") or ""), []).append(vote)

    def model_metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        sessions = len(rows)
        first = sum(row.get("first_pass_schema_valid") is True for row in rows)
        final = sum(row.get("final_schema_valid") is True for row in rows)
        repaired = sum(row.get("repaired_final_valid") is True for row in rows)
        calls = sum(int(row.get("transport_calls") or 0) for row in rows)
        return {
            "sessions": sessions,
            "first_pass_schema_valid": first,
            "first_pass_schema_rate": _rate(first, sessions),
            "final_schema_valid": final,
            "final_schema_rate": _rate(final, sessions),
            "repaired_final_valid": repaired,
            "invalid_final": sessions - final,
            "repair_turns": sum(int(row.get("repair_turns") or 0) for row in rows),
            "transport_calls": calls,
            "transport_failures": sum(
                int(row.get("transport_failures") or 0) for row in rows
            ),
            "latency_ms_total": round(
                sum(float(row.get("latency_ms") or 0.0) for row in rows), 3
            ),
        }

    processed = len(cases)
    pair_valid = sum(row["pair_valid"] is True for row in derived_rows)
    pair_agreed = sum(row["pair_signature_agreed"] is True for row in derived_rows)
    pair_safe_resolved = sum(
        row["pair_safe_resolution_without_tie"] is True for row in derived_rows
    )
    ties = sum(row["tie_break_invoked"] is True for row in derived_rows)
    ties_resolved = sum(row["tie_break_resolved"] is True for row in derived_rows)
    majority_resolved = sum(
        row["signature_majority_resolved"] is True for row in derived_rows
    )
    safe_policy_resolved = sum(
        case.get("status") == "agreed"
        and derived["signature_majority_resolved"] is not True
        for case, derived in zip(cases, derived_rows, strict=True)
    )
    comparable = sum(
        row["expected_decision_comparable"] is True for row in derived_rows
    )
    matched = sum(row["expected_decision_match"] is True for row in derived_rows)
    expected_signature_matches = sum(
        row["expected_signature_match"] is True for row in derived_rows
    )
    effect_comparable = sum(
        row["expected_effect_comparable"] is True for row in derived_rows
    )
    effect_matches = sum(row["expected_effect_match"] is True for row in derived_rows)
    canonical_lane_metrics = _canonical_lane_exact_signature_metrics(
        cases,
        derived_rows,
    )
    latencies = [float(case.get("latency_ms") or 0.0) for case in cases]
    required_buckets = tuple(
        sorted(
            {
                value
                for value in required_context_buckets
                if isinstance(value, int) and not isinstance(value, bool) and value > 0
            }
        )
    )
    observed_bucket_counts: Counter[int] = Counter()
    for derived in derived_rows:
        buckets = derived["evaluated_context_buckets"]
        observed_bucket_counts.update(
            {
                int(bucket)
                for bucket in buckets
                if isinstance(bucket, int)
                and not isinstance(bucket, bool)
                and bucket > 0
            }
        )
    covered_buckets = sum(
        observed_bucket_counts.get(bucket, 0) > 0 for bucket in required_buckets
    )
    overall = model_metric(vote_rows)
    return {
        "processed_cases": processed,
        "model_sessions": overall["sessions"],
        "first_pass_schema_valid": overall["first_pass_schema_valid"],
        "first_pass_schema_rate": overall["first_pass_schema_rate"],
        "final_schema_valid": overall["final_schema_valid"],
        "final_schema_rate": overall["final_schema_rate"],
        "repaired_final_valid": overall["repaired_final_valid"],
        "invalid_final": overall["invalid_final"],
        "invalid_output_accepted": sum(
            int(row["invalid_output_accepted"]) for row in derived_rows
        ),
        "pair_valid_cases": pair_valid,
        "pair_valid_rate": _rate(pair_valid, processed),
        "pair_agreement_cases": pair_agreed,
        "pair_agreement_rate": _rate(pair_agreed, pair_valid),
        "pair_agreement_rate_of_all": _rate(pair_agreed, processed),
        "pair_safe_resolution_without_tie_cases": pair_safe_resolved,
        "pair_safe_resolution_without_tie_rate": _rate(pair_safe_resolved, pair_valid),
        "tie_break_invoked": ties,
        "tie_break_resolved": ties_resolved,
        "tie_break_resolution_rate": _rate(ties_resolved, ties, empty=1.0),
        # The rollout contract is the fraction of all decisions that reach a
        # two-vote majority. Pair agreements are already valid majorities; a
        # separate pair_agreement metric prevents frequent tie-break use from
        # being hidden here.
        "majority_resolution_rate": _rate(
            majority_resolved,
            processed,
            empty=1.0,
        ),
        "safe_policy_resolution_without_signature_majority": safe_policy_resolved,
        "unresolved_quarantine": sum(
            case.get("status") == "quarantined" for case in cases
        ),
        "expected_decision_comparable": comparable,
        "expected_decision_matches": matched,
        "expected_decision_match_rate": _rate(matched, comparable),
        "expected_signature_matches": expected_signature_matches,
        "expected_signature_match_rate": _rate(expected_signature_matches, processed),
        "expected_effect_comparable": effect_comparable,
        "expected_effect_matches": effect_matches,
        "expected_effect_match_rate": _rate(effect_matches, effect_comparable),
        **canonical_lane_metrics,
        "unsafe_decision_flips": sum(
            row["unsafe_decision_flip"] is True for row in derived_rows
        ),
        "context_buckets_required": list(required_buckets),
        "context_bucket_counts": {
            str(bucket): observed_bucket_counts.get(bucket, 0)
            for bucket in required_buckets
        },
        "context_bucket_coverage_rate": _rate(
            covered_buckets,
            len(required_buckets),
            empty=0.0,
        ),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "models": {
            model: model_metric(rows) for model, rows in sorted(model_rows.items())
        },
    }


def adoption_gate(
    metrics: Mapping[str, Any],
    thresholds: AdoptionThresholds,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    coverage = source.get("coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    lane_contracts_required = (
        coverage.get("production_lane_contracts_required") is not False
    )
    checks = {
        "full_usable_corpus": {
            "observed": source.get("full_usable_selection") is True,
            "required": True,
            "passed": source.get("full_usable_selection") is True,
        },
        "minimum_usable_cases": {
            "observed": source.get("usable_cases"),
            "minimum": MIN_ADOPTION_USABLE_CASES,
        },
        "role_coverage": {
            "observed": coverage.get("role_coverage_rate"),
            "minimum": 1.0,
        },
        "decision_label_coverage": {
            "observed": coverage.get("decision_coverage_rate"),
            "minimum": 1.0,
        },
        "production_schema_coverage": {
            "observed": coverage.get("production_schema_coverage_rate"),
            "minimum": 1.0,
        },
        "minimum_cases_per_production_schema": {
            "observed": coverage.get("minimum_production_schema_cases"),
            "minimum": MIN_CASES_PER_PRODUCTION_SCHEMA,
        },
        "model_backed_lane_coverage": {
            "observed": coverage.get("model_backed_lane_coverage_rate"),
            "minimum": 1.0,
            "passed": bool(
                not lane_contracts_required
                or coverage.get("model_backed_lane_coverage_rate") == 1.0
            ),
        },
        "minimum_cases_per_model_backed_lane": {
            "observed": coverage.get("minimum_model_backed_lane_cases"),
            "minimum": MIN_CASES_PER_MODEL_BACKED_LANE,
            "passed": bool(
                not lane_contracts_required
                or (
                    isinstance(coverage.get("minimum_model_backed_lane_cases"), int)
                    and coverage["minimum_model_backed_lane_cases"]
                    >= MIN_CASES_PER_MODEL_BACKED_LANE
                )
            ),
        },
        "canonical_lane_case_set": {
            "observed": coverage.get("lane_contract_case_manifest_sha256"),
            "required": decision_lane_contract_case_manifest_sha256(),
            "passed": bool(
                not lane_contracts_required
                or (
                    coverage.get("lane_contract_case_manifest_sha256")
                    == decision_lane_contract_case_manifest_sha256()
                    and coverage.get("model_backed_lane_coverage_rate") == 1.0
                )
            ),
        },
        "first_pass_schema_success": {
            "observed": metrics.get("first_pass_schema_rate"),
            "minimum": thresholds.first_pass_schema_rate,
        },
        "final_schema_success": {
            "observed": metrics.get("final_schema_rate"),
            "minimum": thresholds.final_schema_rate,
        },
        "pair_valid_vote": {
            "observed": metrics.get("pair_valid_rate"),
            "minimum": thresholds.pair_valid_rate,
        },
        "pair_agreement": {
            "observed": metrics.get("pair_agreement_rate"),
            "minimum": thresholds.pair_agreement_rate,
        },
        "three_model_majority_resolution": {
            "observed": metrics.get("majority_resolution_rate"),
            "minimum": thresholds.majority_resolution_rate,
        },
        "expected_effect_match": {
            "observed": metrics.get("expected_effect_match_rate"),
            "minimum": thresholds.expected_effect_match_rate,
        },
        "canonical_lane_exact_signature_match": {
            "observed": metrics.get("canonical_lane_exact_signature_match_rate"),
            "minimum": 1.0,
            "passed": bool(
                not lane_contracts_required
                or _canonical_lane_exact_signature_gate_passed(metrics)
            ),
        },
        "context_bucket_coverage": {
            "observed": metrics.get("context_bucket_coverage_rate"),
            "minimum": 1.0,
        },
        "invalid_output_accepted": {
            "observed": metrics.get("invalid_output_accepted"),
            "maximum": thresholds.max_invalid_output_accepted,
        },
        "unsafe_decision_flips": {
            "observed": metrics.get("unsafe_decision_flips"),
            "maximum": thresholds.max_unsafe_decision_flips,
        },
    }
    for check in checks.values():
        if "passed" in check:
            continue
        observed = check["observed"]
        if "minimum" in check:
            check["passed"] = bool(
                isinstance(observed, (int, float)) and observed >= check["minimum"]
            )
        else:
            check["passed"] = bool(
                isinstance(observed, (int, float)) and observed <= check["maximum"]
            )
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from chronovisor.page_mutation import decision_authority_lock

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with decision_authority_lock():
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def _read_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeMismatchError(f"cannot read resume artifact: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResumeMismatchError("resume artifact must be a JSON object")
    return payload


def adoption_evidence_sha256(artifact: Mapping[str, Any]) -> str:
    """Bind every persisted adoption claim except the digest itself."""

    return _sha256_json(
        {key: value for key, value in artifact.items() if key != "evidence_sha256"}
    )


def adoption_result_sha256(artifact: Mapping[str, Any]) -> str:
    """Hash the evaluated cases and every claim derived from them."""

    return _sha256_json(
        {
            "evaluator_policy_version": artifact.get("evaluator_policy_version"),
            "decision_semantics_policy_version": artifact.get(
                "decision_semantics_policy_version"
            ),
            "quorum_safety_policy_version": artifact.get(
                "quorum_safety_policy_version"
            ),
            "structured_generation_policy_sha256": artifact.get(
                "structured_generation_policy_sha256"
            ),
            "lane_contract_policy_version": artifact.get(
                "lane_contract_policy_version"
            ),
            "lane_contract_manifest_sha256": artifact.get(
                "lane_contract_manifest_sha256"
            ),
            "lane_contract_case_manifest_sha256": artifact.get(
                "lane_contract_case_manifest_sha256"
            ),
            "status": artifact.get("status"),
            "selected_cases": artifact.get("selected_cases"),
            "processed_cases": artifact.get("processed_cases"),
            "context_buckets": artifact.get("context_buckets"),
            "cases": artifact.get("cases"),
            "metrics": artifact.get("metrics"),
            "adoption_gate": artifact.get("adoption_gate"),
            "adopted": artifact.get("adopted"),
        }
    )


def validate_adoption_evidence(artifact: Mapping[str, Any]) -> bool:
    digest = artifact.get("evidence_sha256")
    return bool(
        isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest)
        and digest == adoption_evidence_sha256(artifact)
    )


def _refresh_artifact(
    artifact: dict[str, Any],
    *,
    status: str,
    thresholds: AdoptionThresholds,
) -> None:
    artifact["status"] = status
    artifact["updated_at"] = _now()
    artifact["processed_cases"] = len(artifact["cases"])
    metrics = adoption_metrics(
        artifact["cases"],
        required_context_buckets=artifact.get("context_buckets", ()),
    )
    artifact["metrics"] = metrics
    artifact["adoption_gate"] = adoption_gate(metrics, thresholds, artifact["source"])
    artifact["adopted"] = bool(
        status == "complete"
        and artifact["processed_cases"] == artifact["selected_cases"]
        and artifact["adoption_gate"]["passed"]
    )
    artifact["evaluation_result_sha256"] = adoption_result_sha256(artifact)
    artifact["evidence_sha256"] = adoption_evidence_sha256(artifact)


def _validate_resume_artifact(
    artifact: Mapping[str, Any],
    *,
    corpus: ReplayCorpus,
    execution_cases: Sequence[ReplayCase],
    planned_context_by_id: Mapping[str, int],
    identity: Mapping[str, Any],
    run_key: str,
    config_payload: Mapping[str, Any],
    safe_metadata: Mapping[str, Any],
    metadata_sha256: str,
    thresholds: AdoptionThresholds,
    context_buckets: Sequence[int],
    source_inspection: Mapping[str, Any],
    models: Sequence[str],
) -> None:
    """Rebuild every resume claim before reusing even a complete artifact."""

    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ResumeMismatchError("artifact schema version does not match")
    if not validate_adoption_evidence(artifact):
        raise ResumeMismatchError("artifact evidence digest does not match")
    if (
        artifact.get("evaluator_policy_version") != EVALUATOR_POLICY_VERSION
        or artifact.get("decision_semantics_policy_version")
        != DECISION_SEMANTICS_POLICY_VERSION
        or artifact.get("quorum_safety_policy_version") != QUORUM_SAFETY_POLICY_VERSION
        or artifact.get("structured_generation_policy")
        != structured_generation_policy()
        or artifact.get("structured_generation_policy_sha256")
        != structured_generation_policy_sha256()
        or artifact.get("evaluation_result_sha256") != adoption_result_sha256(artifact)
    ):
        raise ResumeMismatchError("artifact evaluation result identity does not match")
    if artifact.get("run_key") != run_key or artifact.get("identity") != dict(identity):
        raise ResumeMismatchError("artifact identity does not match this replay run")

    expected_top_level = {
        "source": dict(source_inspection),
        "config": dict(config_payload),
        "config_sha256": identity["config_sha256"],
        "model_metadata": dict(safe_metadata),
        "model_metadata_sha256": metadata_sha256,
        "thresholds": asdict(thresholds),
        "context_buckets": list(context_buckets),
        "selected_cases": len(corpus.cases),
        "decision_semantics_policy_version": DECISION_SEMANTICS_POLICY_VERSION,
        "quorum_safety_policy_version": QUORUM_SAFETY_POLICY_VERSION,
        "structured_generation_policy": structured_generation_policy(),
        "structured_generation_policy_sha256": (structured_generation_policy_sha256()),
        "lane_contract_policy_version": LANE_CONTRACT_POLICY_VERSION,
        "lane_contract_manifest_sha256": lane_contract_manifest_sha256(),
        "lane_contract_case_manifest_sha256": (
            decision_lane_contract_case_manifest_sha256()
        ),
    }
    for name, expected in expected_top_level.items():
        if artifact.get(name) != expected:
            raise ResumeMismatchError(f"artifact {name} does not match this replay run")

    existing_cases = artifact.get("cases")
    if not isinstance(existing_cases, list):
        raise ResumeMismatchError("artifact cases must be an array")
    authoritative = {case.case_id: case for case in corpus.cases}
    if any(not isinstance(row, Mapping) for row in existing_cases):
        raise ResumeMismatchError("artifact contains a non-object case")
    observed_ids = [str(row.get("case_id") or "") for row in existing_cases]
    if any(case_id not in authoritative for case_id in observed_ids) or len(
        set(observed_ids)
    ) != len(observed_ids):
        raise ResumeMismatchError(
            "artifact contains invalid, duplicate, or out-of-selection case ids"
        )
    status = artifact.get("status")
    if status not in {"in_progress", "complete"}:
        raise ResumeMismatchError("artifact status is invalid")
    expected_ids = (
        [case.case_id for case in corpus.cases]
        if status == "complete"
        else [case.case_id for case in execution_cases[: len(existing_cases)]]
    )
    if observed_ids != expected_ids:
        if status == "complete":
            raise ResumeMismatchError(
                "complete artifact does not exactly cover source case order"
            )
        raise ResumeMismatchError(
            "artifact cases are not an exact evaluation-order prefix"
        )
    expected_vote_identity = [
        ("primary", models[0]),
        ("challenger", models[1]),
        ("tie_break", models[2]),
    ]
    for row in existing_cases:
        case_id = str(row["case_id"])
        case = authoritative[case_id]
        expected_case_fields = {
            "index": case.index,
            "case_id": case.case_id,
            "effective_request_sha256": case.effective_request_sha256,
            "role": case.role,
            "source": case.source,
            "decision_lane": case.decision_lane,
            "contract_id": case.contract_id,
            "lane_contract_sha256": case.lane_contract_sha256,
            "lane_contract_effect": case.lane_contract_effect,
            "lane_contract_case_manifest_sha256": (
                case.lane_contract_case_manifest_sha256
            ),
            "evidence_provenance_sha256": _sha256_json(case.evidence_provenance),
            "schema_sha256": case.schema_sha256,
            "expected_signature": _decision_signature(case.expected, case.schema),
            "expected_signature_sha256": case.expected_signature_sha256,
            "expected_coverage_label": case.expected_coverage_label,
            "expected_decision": case.expected_decision,
            "expected_effect": _semantic_effect(
                case.expected,
                case.schema,
                prompt=case.prompt,
                decision_lane=case.decision_lane,
            ),
            "effect_context": _bounded_effect_context(case.prompt),
            "num_ctx": planned_context_by_id[case.case_id],
        }
        if any(row.get(name) != value for name, value in expected_case_fields.items()):
            raise ResumeMismatchError(
                f"artifact case metadata does not match source case {case.case_id}"
            )
        votes = row.get("votes")
        if not isinstance(votes, list):
            raise ResumeMismatchError("artifact case votes must be an array")
        vote_identity = [
            (vote.get("role"), vote.get("model"))
            for vote in votes
            if isinstance(vote, Mapping)
        ]
        if vote_identity != expected_vote_identity[: len(votes)]:
            raise ResumeMismatchError("artifact vote model identity does not match")
        try:
            derived = adoption_case_derived_evidence(row)
        except ValueError as exc:
            raise ResumeMismatchError(
                f"artifact case evidence is inconsistent: {exc}"
            ) from exc
        if any(row.get(name) != value for name, value in derived.items()):
            raise ResumeMismatchError(
                "artifact case flags disagree with authoritative vote evidence"
            )

    processed = artifact.get("processed_cases")
    if (
        isinstance(processed, bool)
        or not isinstance(processed, int)
        or processed != len(existing_cases)
    ):
        raise ResumeMismatchError("artifact processed case count is inconsistent")
    try:
        recomputed_metrics = adoption_metrics(
            existing_cases,
            required_context_buckets=context_buckets,
        )
    except ValueError as exc:
        raise ResumeMismatchError(
            f"artifact metrics cannot be reconstructed: {exc}"
        ) from exc
    if artifact.get("metrics") != recomputed_metrics:
        raise ResumeMismatchError("artifact metrics do not match case evidence")
    recomputed_gate = adoption_gate(
        recomputed_metrics,
        thresholds,
        source_inspection,
    )
    if artifact.get("adoption_gate") != recomputed_gate:
        raise ResumeMismatchError("artifact gate does not match case evidence")
    expected_adopted = bool(
        status == "complete"
        and len(existing_cases) == len(corpus.cases)
        and recomputed_gate["passed"]
    )
    if artifact.get("adopted") is not expected_adopted:
        raise ResumeMismatchError("artifact adoption result is inconsistent")


def evaluate_replays(
    input_path: Path | str,
    output_path: Path | str,
    *,
    offset: int = 0,
    limit: int = 0,
    resume: bool = False,
    config: DecisionRouterConfig | None = None,
    transport: ChatTransport | None = None,
    model_metadata_provider: ModelMetadataProvider | None = None,
    thresholds: AdoptionThresholds | None = None,
    required_schema_manifest: Mapping[str, str] | None = None,
    live_resource_control: bool = False,
    model_observer: ModelObserver | None = None,
) -> dict[str, Any]:
    """Evaluate a replay slice and atomically checkpoint a redacted artifact."""

    corpus = load_replay_corpus(
        input_path,
        offset=offset,
        limit=limit,
        required_schema_manifest=required_schema_manifest,
    )
    if (
        corpus.full_usable_selection
        and corpus.usable_cases >= MIN_ADOPTION_USABLE_CASES
        and any(case.self_labeled for case in corpus.cases)
    ):
        raise ReplayInputError(
            "full adoption corpus contains self-labeled local consensus evidence"
        )
    if (
        corpus.full_usable_selection
        and corpus.usable_cases >= MIN_ADOPTION_USABLE_CASES
        and (
            corpus.exact_duplicate_request_groups > 0
            or corpus.conflicting_request_groups > 0
        )
    ):
        raise ReplayInputError(
            "full adoption corpus contains duplicate effective requests "
            f"(exact_groups={corpus.exact_duplicate_request_groups}, "
            f"conflicting_groups={corpus.conflicting_request_groups})"
        )
    corpus_coverage = corpus.coverage()
    if (
        corpus.full_usable_selection
        and corpus.usable_cases >= MIN_ADOPTION_USABLE_CASES
        and corpus.production_lane_contracts_required
        and (
            corpus_coverage.get("lane_contract_policy_version")
            != LANE_CONTRACT_POLICY_VERSION
            or corpus_coverage.get("lane_contract_manifest_sha256")
            != lane_contract_manifest_sha256()
            or corpus_coverage.get("lane_contract_case_manifest_sha256")
            != decision_lane_contract_case_manifest_sha256()
            or corpus_coverage.get("model_backed_lane_coverage_rate") != 1.0
            or not isinstance(
                corpus_coverage.get("minimum_model_backed_lane_cases"), int
            )
            or corpus_coverage["minimum_model_backed_lane_cases"]
            < MIN_CASES_PER_MODEL_BACKED_LANE
        )
    ):
        raise ReplayInputError(
            "full adoption corpus lacks current model-backed lane contracts"
        )
    output = Path(output_path).expanduser()
    if corpus.path.resolve() == output.resolve():
        raise ValueError("output_path must not overwrite the read-only replay input")
    config = config or load_decision_router_config()
    thresholds = thresholds or AdoptionThresholds()
    config_payload = asdict(config)
    context_buckets = decision_context_buckets(config)
    planned_context_counts: Counter[int] = Counter()
    planned_context_by_id: dict[str, int] = {}
    oversized_case_ids: list[str] = []
    for case in corpus.cases:
        required, bucket = decision_request_context(
            config,
            case.prompt,
            case.schema,
            case.system,
            decision_lane=case.decision_lane,
        )
        planned_context_by_id[case.case_id] = bucket
        planned_context_counts[bucket] += 1
        if required > config.num_ctx:
            oversized_case_ids.append(case.case_id)
    execution_cases = tuple(
        sorted(
            corpus.cases,
            key=lambda case: (planned_context_by_id[case.case_id], case.index),
        )
    )
    execution_order_sha256 = _sha256_json([case.case_id for case in execution_cases])
    if (
        corpus.full_usable_selection
        and corpus.usable_cases >= MIN_ADOPTION_USABLE_CASES
        and (
            oversized_case_ids
            or any(planned_context_counts[bucket] < 1 for bucket in context_buckets)
        )
    ):
        missing = [
            str(bucket)
            for bucket in context_buckets
            if planned_context_counts[bucket] < 1
        ]
        raise ReplayInputError(
            "full adoption corpus lacks executable planned context coverage "
            f"(missing={','.join(missing) or 'none'}, "
            f"oversized={len(oversized_case_ids)})"
        )
    source_inspection = corpus.inspection(include_cases=False)
    source_inspection["context_plan"] = {
        "mode": "exact_context_ascending_v1",
        "bucket_counts": {
            str(bucket): planned_context_counts[bucket] for bucket in context_buckets
        },
        "oversized_cases": len(oversized_case_ids),
        "execution_order_sha256": execution_order_sha256,
    }
    models = (
        config.primary_model,
        config.challenger_model,
        config.tie_break_model,
    )
    provider = model_metadata_provider or fetch_local_model_metadata
    metadata_payload = provider(models)
    if not isinstance(metadata_payload, Mapping):
        raise ValueError("model metadata provider must return a mapping")
    safe_metadata = _safe_model_metadata(metadata_payload, models)
    validate_model_metadata_identity(safe_metadata, models)
    # Bind the identity to the exact redacted metadata stored in the artifact,
    # so runtime can recompute it instead of trusting an unattached digest.
    metadata_sha256 = _sha256_json(safe_metadata)
    identity = {
        "evaluator_policy_version": EVALUATOR_POLICY_VERSION,
        "decision_semantics_policy_version": DECISION_SEMANTICS_POLICY_VERSION,
        "quorum_safety_policy_version": QUORUM_SAFETY_POLICY_VERSION,
        "structured_generation_policy_version": (STRUCTURED_GENERATION_POLICY_VERSION),
        "structured_generation_policy_sha256": (structured_generation_policy_sha256()),
        "lane_contract_policy_version": LANE_CONTRACT_POLICY_VERSION,
        "lane_contract_manifest_sha256": lane_contract_manifest_sha256(),
        "lane_contract_case_manifest_sha256": (
            decision_lane_contract_case_manifest_sha256()
        ),
        "source_path": str(corpus.path),
        "source_sha256": corpus.source_sha256,
        "offset": corpus.offset,
        "limit": corpus.limit,
        "selected_case_ids_sha256": _sha256_json(
            [case.case_id for case in corpus.cases]
        ),
        "selected_effective_requests_sha256": _sha256_json(
            [case.effective_request_sha256 for case in corpus.cases]
        ),
        "config_sha256": _sha256_json(config_payload),
        "model_metadata_sha256": metadata_sha256,
        "thresholds_sha256": _sha256_json(asdict(thresholds)),
        "schema_manifest_sha256": corpus.coverage()["schema_manifest_sha256"],
        "signature_manifest_sha256": corpus.coverage()["signature_manifest_sha256"],
        "context_buckets_sha256": _sha256_json(context_buckets),
        "evaluation_mode": "exact_context_ascending_v1",
        "evaluation_order_sha256": execution_order_sha256,
    }
    run_key = _sha256_json(identity)

    if resume:
        artifact = _read_artifact(output)
        _validate_resume_artifact(
            artifact,
            corpus=corpus,
            execution_cases=execution_cases,
            planned_context_by_id=planned_context_by_id,
            identity=identity,
            run_key=run_key,
            config_payload=config_payload,
            safe_metadata=safe_metadata,
            metadata_sha256=metadata_sha256,
            thresholds=thresholds,
            context_buckets=context_buckets,
            source_inspection=source_inspection,
            models=models,
        )
        if artifact.get("status") == "complete":
            return artifact
    else:
        if output.exists():
            raise FileExistsError(
                f"output artifact already exists; use --resume or a new path: {output}"
            )
        artifact = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "evaluator_policy_version": EVALUATOR_POLICY_VERSION,
            "decision_semantics_policy_version": DECISION_SEMANTICS_POLICY_VERSION,
            "quorum_safety_policy_version": QUORUM_SAFETY_POLICY_VERSION,
            "structured_generation_policy": structured_generation_policy(),
            "structured_generation_policy_sha256": (
                structured_generation_policy_sha256()
            ),
            "lane_contract_policy_version": LANE_CONTRACT_POLICY_VERSION,
            "lane_contract_manifest_sha256": lane_contract_manifest_sha256(),
            "lane_contract_case_manifest_sha256": (
                decision_lane_contract_case_manifest_sha256()
            ),
            "status": "in_progress",
            "started_at": _now(),
            "updated_at": _now(),
            "run_key": run_key,
            "identity": identity,
            "source": source_inspection,
            "selected_cases": len(corpus.cases),
            "processed_cases": 0,
            "config": config_payload,
            "config_sha256": identity["config_sha256"],
            "model_metadata": safe_metadata,
            "model_metadata_sha256": metadata_sha256,
            "thresholds": asdict(thresholds),
            "context_buckets": list(context_buckets),
            "cases": [],
            "metrics": adoption_metrics([], required_context_buckets=context_buckets),
            "adoption_gate": adoption_gate(
                adoption_metrics([], required_context_buckets=context_buckets),
                thresholds,
                source_inspection,
            ),
            "adopted": False,
        }
        artifact["evaluation_result_sha256"] = adoption_result_sha256(artifact)
        artifact["evidence_sha256"] = adoption_evidence_sha256(artifact)
        _atomic_json(output, artifact)

    timing_transport = _TimingTransport(transport or _live_transport)
    router = DecisionRouter(
        config=config,
        transport=timing_transport,
        audit_role="model_eval",
        resolve_adoption=False,
        record_replay=False,
        live_resource_control=live_resource_control,
        model_observer=model_observer,
        reuse_larger_context=False,
    )
    completed_ids = {
        str(row.get("case_id")) for row in artifact["cases"] if isinstance(row, Mapping)
    }
    if live_resource_control:
        # Evaluation uses exact bucket allocations, unlike production's
        # grow-only hysteresis. Reset surviving candidate runners on fresh and
        # resumed runs so a larger allocation cannot contaminate evidence.
        for model in models:
            try:
                resident = router.model_observer(model)
            except Exception:
                resident = None
            if resident is not None and not router.model_unloader(model):
                raise RuntimeError(
                    f"unable to verify evaluator cold-start unload for {model}"
                )

    for case in execution_cases:
        if case.case_id in completed_ids:
            continue
        mark = timing_transport.mark()
        started = time.perf_counter()
        result = router.decide(
            case.prompt,
            case.schema,
            system=case.system,
            decision_lane=case.decision_lane,
        )
        record = _case_record(
            case,
            result,
            latency_ms=(time.perf_counter() - started) * 1000,
            transport_events=timing_transport.since(mark),
        )
        artifact["cases"].append(record)
        _refresh_artifact(artifact, status="in_progress", thresholds=thresholds)
        _atomic_json(output, artifact)

    artifact["cases"].sort(key=lambda row: int(row["index"]))
    _refresh_artifact(artifact, status="complete", thresholds=thresholds)
    _atomic_json(output, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay-gate the local Chronovisor decision ensemble without frontier calls."
    )
    parser.add_argument("--input", type=Path, default=REPLAY_FILE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=0, help="0 evaluates through end of corpus"
    )
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--list", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = (
            load_candidate_decision_router_config(args.config)
            if args.config is not None
            else load_decision_router_config()
        )
        if args.dry_run or args.list:
            payload = inspect_replays(
                args.input,
                offset=args.offset,
                limit=args.limit,
                include_cases=args.list,
            )
            payload["mode"] = "list" if args.list else "dry_run"
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.output is None:
            parser.error("--output is required unless --dry-run or --list is used")
        artifact = evaluate_replays(
            args.input,
            args.output,
            offset=args.offset,
            limit=args.limit,
            resume=args.resume,
            config=config,
            live_resource_control=True,
        )
    except (ReplayInputError, ResumeMismatchError, FileExistsError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    summary = {
        "status": artifact["status"],
        "adopted": artifact["adopted"],
        "output": str(args.output),
        "processed_cases": artifact["processed_cases"],
        "selected_cases": artifact["selected_cases"],
        "source": {
            key: artifact["source"].get(key)
            for key in (
                "total_cases",
                "usable_cases",
                "excluded_cases",
                "excluded_reasons",
                "full_usable_selection",
                "coverage",
            )
        },
        "metrics": artifact["metrics"],
        "adoption_gate": artifact["adoption_gate"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if artifact["adopted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AdoptionThresholds",
    "ReplayCase",
    "ReplayCorpus",
    "ReplayInputError",
    "ResumeMismatchError",
    "STALE_HISTORICAL_REQUEST_IDENTITY_EXCLUSION",
    "evaluate_replays",
    "fetch_local_model_metadata",
    "inspect_replays",
    "load_replay_corpus",
    "main",
    "replay_agreement_value",
    "replay_effect_context",
    "replay_semantic_effect",
    "validate_model_metadata_identity",
]
