"""Strict durable holds for exact-epoch local semantic disagreement.

The local decision router can return three individually valid votes without a
safe quorum.  Re-sampling the same request under the same adopted authority is
not recovery; it is nondeterministic mutation gambling.  This module defines a
small, deterministic, self-hashed envelope that callers can persist and reuse
until the request epoch or adopted authority actually changes.

Only redacted router audit records and canonical digests are stored.  Prompts,
model responses, and decision payloads are deliberately excluded from holds.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from chronovisor.canonical_json import (
    canonical_json_sha256_strict as canonical_sha256,
    canonical_json_strict as _canonical_json,
)
from chronovisor.decision_authority import (
    semantic_authority_shape_error,
    semantic_verdict_authority_provenance_error,
)
from chronovisor.semantic_epoch import (
    STRUCTURED_REVIEW_HOLD_EPOCH_VERSION,
    build_structured_review_epoch,
    is_sha256 as _is_sha256,
    opaque_text_sha256 as _opaque_text_sha256,
    structured_review_epoch_error,
)

LOCAL_SEMANTIC_NO_QUORUM = "local_semantic_no_quorum"
SEMANTIC_NO_QUORUM_HOLD_KIND = "local_semantic_no_quorum_hold"
SCHEMA_VERSION = 1

STRUCTURED_REVIEW_HOLD_CACHE_SCHEMA_VERSION = 1
STRUCTURED_REVIEW_HOLD_CACHE_KIND = "structured_review_semantic_no_quorum_cache"
STRUCTURED_REVIEW_HOLD_RESOLVER_VERSION = 1

_SEMANTIC_REASONS = frozenset(
    {
        "local_models_did_not_reach_two_vote_quorum",
        "mutating_local_majority_vetoed_by_conservative_vote",
    }
)
_HOLD_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "lane",
        "epoch",
        "epoch_sha256",
        "authority",
        "authority_sha256",
        "frontier_failure",
        "local_consensus",
        "decision_policy",
        "review_sha256",
        "hold_sha256",
    }
)


STRUCTURED_REVIEW_HOLD_RESOLVER_SHA256 = canonical_sha256(
    {
        "resolver_version": STRUCTURED_REVIEW_HOLD_RESOLVER_VERSION,
        "failure_class": LOCAL_SEMANTIC_NO_QUORUM,
        "semantic_reasons": sorted(_SEMANTIC_REASONS),
        "quorum": 2,
        "vote_count": 3,
        "requires_adopted_lane_router_provenance": True,
    }
)


def _file_observation(path: Path) -> dict[str, Any]:
    """Return an opaque change detector including symlink and target metadata."""

    observation: dict[str, Any] = {
        "path_sha256": hashlib.sha256(str(path).encode("utf-8")).hexdigest(),
        "exists": False,
    }
    try:
        link_stat = path.lstat()
        target_stat = path.stat()
    except FileNotFoundError:
        return observation
    except OSError as exc:
        observation["error"] = exc.__class__.__name__
        return observation
    observation.update(
        {
            "exists": True,
            "link": {
                "dev": link_stat.st_dev,
                "ino": link_stat.st_ino,
                "mode": link_stat.st_mode,
                "size": link_stat.st_size,
                "mtime_ns": link_stat.st_mtime_ns,
                "ctime_ns": link_stat.st_ctime_ns,
            },
            "target": {
                "dev": target_stat.st_dev,
                "ino": target_stat.st_ino,
                "mode": target_stat.st_mode,
                "size": target_stat.st_size,
                "mtime_ns": target_stat.st_mtime_ns,
                "ctime_ns": target_stat.st_ctime_ns,
            },
        }
    )
    if path.is_file():
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            observation["content_error"] = exc.__class__.__name__
        else:
            observation["content_sha256"] = digest.hexdigest()
    return observation


def structured_review_authority_observation_sha256(
    authority: Mapping[str, Any],
) -> str:
    """Observe mutable authority sources for one in-flight boundary guard.

    The semantic authority seal identifies the adopted content.  This
    additional opaque token watches the filesystem generations that selected
    it and the live Ollama engine/model metadata before and after a cache read
    or model call.  A temporary switch away from an authority and back during
    that operation therefore fails closed even when the final semantic seal
    again equals the starting seal.  It is deliberately not part of the
    durable cache identity: a stable later A generation may reuse A's exact
    semantic no-quorum hold after the in-flight guard succeeds.
    """

    lane = authority.get("lane") if isinstance(authority, Mapping) else None
    if not isinstance(lane, str) or not lane:
        raise ValueError("structured review authority lane is missing")
    authority_error = semantic_authority_shape_error(authority, lane=lane)
    if authority_error is not None:
        raise ValueError(authority_error)
    router = authority.get("router")
    assert isinstance(router, Mapping)
    models = router.get("models")
    assert isinstance(models, list)

    from chronovisor.local_model_eval import (
        _safe_model_metadata,
        fetch_local_model_metadata,
    )
    from chronovisor.runtime_config import (
        CONFIG_FILE,
        LEGACY_RECALL_CONFIG_FILE,
        load_decision_router_config,
    )

    live_metadata = _safe_model_metadata(fetch_local_model_metadata(models), models)
    config = load_decision_router_config()
    adoption_path = (
        Path(config.adoption_artifact).expanduser()
        if config.adoption_artifact.strip()
        else None
    )
    normalized_lane = re.sub(r"[^A-Za-z0-9]+", "_", lane).strip("_").upper()
    lane_env = os.environ.get(f"CHRONOVISOR_DECISION_POLICY_{normalized_lane}")
    payload = {
        "authority_sha256": canonical_sha256(authority),
        "config_file": _file_observation(CONFIG_FILE),
        "legacy_config_file": _file_observation(LEGACY_RECALL_CONFIG_FILE),
        "adoption_artifact": (
            _file_observation(adoption_path) if adoption_path is not None else None
        ),
        "lane_mode_env_sha256": _opaque_text_sha256(lane_env),
        "live_model_metadata_sha256": canonical_sha256(live_metadata),
    }
    return canonical_sha256(payload)


def frontier_failure_class(review: object) -> str | None:
    """Extract a structured review failure class without trusting its shape."""

    if not isinstance(review, Mapping):
        return None
    failure = review.get("frontier_failure")
    if not isinstance(failure, Mapping):
        return None
    value = failure.get("failure_class")
    return value if isinstance(value, str) and value else None


def _router_policy_error(policy: object, *, lane: str) -> str | None:
    if not isinstance(policy, Mapping):
        return "decision policy audit is missing"
    expected_fields = {
        "lane",
        "kind",
        "schema_name",
        "mode",
        "error",
        "expected_schema_sha256",
        "actual_schema_sha256",
        "router_policy",
    }
    if set(policy) != expected_fields:
        return "decision policy audit fields are invalid"
    if (
        policy.get("lane") != lane
        or policy.get("kind") not in {"consensus", "local_batch"}
        or not isinstance(policy.get("schema_name"), str)
        or not policy.get("schema_name")
        or policy.get("mode") != "enabled"
        or policy.get("error") is not None
    ):
        return "decision policy lane provenance is invalid"
    expected_schema = policy.get("expected_schema_sha256")
    actual_schema = policy.get("actual_schema_sha256")
    if not _is_sha256(expected_schema) or actual_schema != expected_schema:
        return "decision policy schema provenance is invalid"
    router = policy.get("router_policy")
    models = router.get("models") if isinstance(router, Mapping) else None
    if (
        not isinstance(router, Mapping)
        or set(router) != {"source", "artifact_sha256", "error", "models"}
        or router.get("source") != "adopted_artifact"
        or not _is_sha256(router.get("artifact_sha256"))
        or router.get("error") is not None
        or not isinstance(models, list)
        or len(models) != 3
        or len(set(models)) != 3
        or not all(isinstance(model, str) and model for model in models)
    ):
        return "decision policy router provenance is invalid"
    return None


def _attempt_error(attempt: object) -> str | None:
    if not isinstance(attempt, Mapping):
        return "local consensus attempt audit is invalid"
    if set(attempt) != {
        "index",
        "valid",
        "output_sha256",
        "output_chars",
        "normalized",
        "error_fingerprint",
        "issues",
    }:
        return "local consensus attempt audit fields are invalid"
    if (
        isinstance(attempt.get("index"), bool)
        or not isinstance(attempt.get("index"), int)
        or attempt["index"] < 0
        or not isinstance(attempt.get("valid"), bool)
        or not _is_sha256(attempt.get("output_sha256"))
        or isinstance(attempt.get("output_chars"), bool)
        or not isinstance(attempt.get("output_chars"), int)
        or attempt["output_chars"] < 0
        or not isinstance(attempt.get("normalized"), bool)
        or not isinstance(attempt.get("issues"), list)
    ):
        return "local consensus attempt audit is invalid"
    if attempt["valid"]:
        if attempt.get("error_fingerprint") is not None or attempt["issues"]:
            return "local consensus successful attempt audit is invalid"
        return None
    if not _is_sha256(attempt.get("error_fingerprint")) or not attempt["issues"]:
        return "local consensus failed attempt audit is invalid"
    allowed_issue_fields = {
        "pointer_sha256",
        "keyword",
        "expected_sha256",
        "received",
        "message_sha256",
        "line",
        "column",
        "byte_offset",
        "snippet_sha256",
    }
    for issue in attempt["issues"]:
        received = issue.get("received") if isinstance(issue, Mapping) else None
        if (
            not isinstance(issue, Mapping)
            or not set(issue).issubset(allowed_issue_fields)
            or not {
                "pointer_sha256",
                "keyword",
                "expected_sha256",
                "received",
                "message_sha256",
            }.issubset(issue)
            or not _is_sha256(issue.get("pointer_sha256"))
            or not isinstance(issue.get("keyword"), str)
            or not _is_sha256(issue.get("expected_sha256"))
            or not isinstance(received, Mapping)
            or not set(received).issubset(
                {"type", "chars", "length", "sha256", "value_sha256"}
            )
            or not isinstance(received.get("type"), str)
            or ("sha256" in received and not _is_sha256(received.get("sha256")))
            or (
                "value_sha256" in received
                and not _is_sha256(received.get("value_sha256"))
            )
            or not _is_sha256(issue.get("message_sha256"))
            or (
                "snippet_sha256" in issue
                and not _is_sha256(issue.get("snippet_sha256"))
            )
        ):
            return "local consensus validation issue audit is invalid"
    return None


def _vote_error(
    vote: object,
    *,
    role: str,
    model: str,
) -> str | None:
    if not isinstance(vote, Mapping):
        return "local consensus vote audit is invalid"
    if set(vote) != {
        "role",
        "model",
        "requested_num_ctx",
        "valid",
        "signature_sha256",
        "invalid_reason",
        "runtime_observation",
        "session",
    }:
        return "local consensus vote audit fields are invalid"
    if (
        vote.get("role") != role
        or vote.get("model") != model
        or vote.get("valid") is not True
        or not _is_sha256(vote.get("signature_sha256"))
        or vote.get("invalid_reason") is not None
        or isinstance(vote.get("requested_num_ctx"), bool)
        or not isinstance(vote.get("requested_num_ctx"), int)
        or vote["requested_num_ctx"] < 1
    ):
        return "local consensus vote identity is invalid"
    runtime = vote.get("runtime_observation")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != {"status", "model_size_bytes", "num_ctx"}
        or not isinstance(runtime.get("status"), str)
    ):
        return "local consensus runtime audit is invalid"
    for field in ("model_size_bytes", "num_ctx"):
        value = runtime.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            return "local consensus runtime measurement is invalid"
    session = vote.get("session")
    if (
        not isinstance(session, Mapping)
        or set(session)
        != {
            "ok",
            "model",
            "failure_class",
            "first_pass_valid",
            "repair_turns",
            "attempts",
        }
        or session.get("ok") is not True
        or session.get("model") != model
        or session.get("failure_class") is not None
        or not isinstance(session.get("first_pass_valid"), bool)
        or isinstance(session.get("repair_turns"), bool)
        or not isinstance(session.get("repair_turns"), int)
        or session["repair_turns"] < 0
        or not isinstance(session.get("attempts"), list)
        or not session["attempts"]
    ):
        return "local consensus session audit is invalid"
    for attempt in session["attempts"]:
        error = _attempt_error(attempt)
        if error is not None:
            return error
    if (
        session["attempts"][-1].get("valid") is not True
        or session["repair_turns"] != len(session["attempts"]) - 1
        or session["first_pass_valid"] is not (len(session["attempts"]) == 1)
    ):
        return "local consensus session attempt sequence is invalid"
    return None


def _local_consensus_error(
    consensus: object, *, policy: Mapping[str, Any]
) -> str | None:
    if not isinstance(consensus, Mapping):
        return "local consensus audit is missing"
    if set(consensus) != {
        "status",
        "ok",
        "quorum_safety_policy_version",
        "agreement_sha256",
        "failure_class",
        "quarantine_reason",
        "num_ctx",
        "residency",
        "votes",
    }:
        return "local consensus audit fields are invalid"
    reason = consensus.get("quarantine_reason")
    if (
        consensus.get("status") != "quarantined"
        or consensus.get("ok") is not False
        or isinstance(consensus.get("quorum_safety_policy_version"), bool)
        or not isinstance(consensus.get("quorum_safety_policy_version"), int)
        or consensus["quorum_safety_policy_version"] < 1
        or consensus.get("agreement_sha256") is not None
        or consensus.get("failure_class") != "local_consensus_failed"
        or reason not in _SEMANTIC_REASONS
        or isinstance(consensus.get("num_ctx"), bool)
        or not isinstance(consensus.get("num_ctx"), int)
        or consensus["num_ctx"] < 1
        or not isinstance(consensus.get("votes"), list)
        or len(consensus["votes"]) != 3
    ):
        return "local consensus no-quorum proof is invalid"

    router = policy.get("router_policy")
    assert isinstance(router, Mapping)
    models = router.get("models")
    assert isinstance(models, list)
    signatures: list[str] = []
    for vote, role, model in zip(
        consensus["votes"],
        ("primary", "challenger", "tie_break"),
        models,
        strict=True,
    ):
        error = _vote_error(vote, role=role, model=model)
        if error is not None:
            return error
        assert isinstance(vote, Mapping)
        signature = vote.get("signature_sha256")
        assert isinstance(signature, str)
        signatures.append(signature)
    counts = sorted(Counter(signatures).values())
    if reason == "local_models_did_not_reach_two_vote_quorum":
        if counts != [1, 1, 1]:
            return "local consensus three-way disagreement proof is invalid"
    elif counts != [1, 2]:
        return "local consensus conservative-veto proof is invalid"
    return None


def _semantic_review_error(review: object, *, lane: str) -> str | None:
    if not isinstance(review, Mapping):
        return "semantic review is missing"
    failure = review.get("frontier_failure")
    if not isinstance(failure, Mapping):
        return "semantic review frontier failure is missing"
    if set(failure) != {
        "failure_class",
        "rescue_status",
        "summary",
        "human_required",
        "notify_user",
    }:
        return "semantic review frontier failure fields are invalid"
    policy = review.get("decision_policy")
    policy_error = _router_policy_error(policy, lane=lane)
    if policy_error is not None:
        return policy_error
    assert isinstance(policy, Mapping)
    consensus = review.get("local_consensus")
    consensus_error = _local_consensus_error(consensus, policy=policy)
    if consensus_error is not None:
        return consensus_error
    assert isinstance(consensus, Mapping)
    reason = consensus.get("quarantine_reason")
    if (
        failure.get("failure_class") != LOCAL_SEMANTIC_NO_QUORUM
        or failure.get("rescue_status") != "local_quarantined"
        or failure.get("summary") != reason
        or failure.get("human_required") is not False
        or failure.get("notify_user") is not False
        or review.get("reviewer") != "local_consensus"
        or review.get("human_required") is not False
    ):
        return "semantic review no-quorum failure envelope is invalid"
    return None


def is_local_semantic_no_quorum(review: object) -> bool:
    """Return true only for a complete, redacted local disagreement proof."""

    if not isinstance(review, Mapping):
        return False
    policy = review.get("decision_policy")
    lane = policy.get("lane") if isinstance(policy, Mapping) else None
    return bool(
        isinstance(lane, str)
        and lane
        and _semantic_review_error(review, lane=lane) is None
    )


def build_semantic_no_quorum_hold(
    lane: str,
    epoch: Mapping[str, Any],
    authority: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one deterministic hold from a validated no-quorum review."""

    if not isinstance(lane, str) or not lane.strip() or lane != lane.strip():
        raise ValueError("semantic hold lane is invalid")
    if not isinstance(epoch, Mapping) or not epoch:
        raise ValueError("semantic hold epoch is missing")
    authority_error = semantic_authority_shape_error(authority, lane=lane)
    if authority_error is not None:
        raise ValueError(authority_error)
    review_error = _semantic_review_error(review, lane=lane)
    if review_error is not None:
        raise ValueError(review_error)
    provenance_error = semantic_verdict_authority_provenance_error(
        review,
        authority,
        lane=lane,
    )
    if provenance_error is not None:
        raise ValueError(provenance_error)
    forbidden_key = _forbidden_plaintext_key(review)
    if forbidden_key is not None:
        raise ValueError(f"semantic hold forbids plaintext field:{forbidden_key}")

    epoch_copy = copy.deepcopy(dict(epoch))
    authority_copy = copy.deepcopy(dict(authority))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": SEMANTIC_NO_QUORUM_HOLD_KIND,
        "lane": lane,
        "epoch": epoch_copy,
        "epoch_sha256": canonical_sha256(epoch_copy),
        "authority": authority_copy,
        "authority_sha256": canonical_sha256(authority_copy),
        "frontier_failure": copy.deepcopy(dict(review["frontier_failure"])),
        "local_consensus": copy.deepcopy(dict(review["local_consensus"])),
        "decision_policy": copy.deepcopy(dict(review["decision_policy"])),
        "review_sha256": canonical_sha256(review),
    }
    payload["hold_sha256"] = canonical_sha256(payload)
    error = semantic_no_quorum_hold_error(
        payload,
        lane,
        epoch=epoch_copy,
        authority=authority_copy,
    )
    if error is not None:
        raise ValueError(error)
    return payload


def semantic_no_quorum_hold_error(
    hold: object,
    lane: str,
    epoch: Mapping[str, Any] | None = None,
    authority: Mapping[str, Any] | None = None,
) -> str | None:
    """Return a validation error, or ``None`` for one exact durable hold."""

    if not isinstance(hold, Mapping):
        return "semantic hold is missing"
    if set(hold) != _HOLD_FIELDS:
        return "semantic hold fields are invalid"
    if (
        hold.get("schema_version") != SCHEMA_VERSION
        or hold.get("kind") != SEMANTIC_NO_QUORUM_HOLD_KIND
        or not isinstance(lane, str)
        or not lane
        or hold.get("lane") != lane
    ):
        return "semantic hold identity is invalid"
    stored_epoch = hold.get("epoch")
    if not isinstance(stored_epoch, Mapping) or not stored_epoch:
        return "semantic hold epoch is invalid"
    try:
        stored_epoch_sha256 = canonical_sha256(stored_epoch)
    except (TypeError, ValueError):
        return "semantic hold epoch is not canonical JSON"
    if hold.get("epoch_sha256") != stored_epoch_sha256:
        return "semantic hold epoch digest is invalid"
    if epoch is not None:
        try:
            if dict(stored_epoch) != dict(
                epoch
            ) or stored_epoch_sha256 != canonical_sha256(epoch):
                return "semantic hold epoch changed"
        except (TypeError, ValueError):
            return "semantic hold current epoch is not canonical JSON"

    stored_authority = hold.get("authority")
    authority_error = semantic_authority_shape_error(stored_authority, lane=lane)
    if authority_error is not None:
        return authority_error
    assert isinstance(stored_authority, Mapping)
    try:
        stored_authority_sha256 = canonical_sha256(stored_authority)
    except (TypeError, ValueError):
        return "semantic hold authority is not canonical JSON"
    if hold.get("authority_sha256") != stored_authority_sha256:
        return "semantic hold authority digest is invalid"
    if authority is not None:
        authority_shape_error = semantic_authority_shape_error(authority, lane=lane)
        if authority_shape_error is not None:
            return authority_shape_error
        try:
            if dict(stored_authority) != dict(
                authority
            ) or stored_authority_sha256 != canonical_sha256(authority):
                return "semantic hold authority changed"
        except (TypeError, ValueError):
            return "semantic hold current authority is not canonical JSON"

    review_stub = {
        "reviewer": "local_consensus",
        "human_required": False,
        "frontier_failure": hold.get("frontier_failure"),
        "local_consensus": hold.get("local_consensus"),
        "decision_policy": hold.get("decision_policy"),
    }
    review_error = _semantic_review_error(review_stub, lane=lane)
    if review_error is not None:
        return review_error
    provenance_error = semantic_verdict_authority_provenance_error(
        review_stub,
        stored_authority,
        lane=lane,
    )
    if provenance_error is not None:
        return provenance_error
    if not _is_sha256(hold.get("review_sha256")):
        return "semantic hold review digest is invalid"
    if not _is_sha256(hold.get("hold_sha256")):
        return "semantic hold self digest is invalid"
    unsigned = dict(hold)
    unsigned.pop("hold_sha256", None)
    try:
        expected_hold_sha256 = canonical_sha256(unsigned)
    except (TypeError, ValueError):
        return "semantic hold is not canonical JSON"
    if hold.get("hold_sha256") != expected_hold_sha256:
        return "semantic hold self digest changed"
    return None


def persisted_semantic_no_quorum_hold(
    value: object,
    lane: str | None = None,
    epoch: Mapping[str, Any] | None = None,
    authority: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Extract and validate a hold from common durable result envelopes."""

    if not isinstance(value, Mapping):
        return None
    candidate: object
    if value.get("kind") == SEMANTIC_NO_QUORUM_HOLD_KIND:
        candidate = value
    elif "semantic_hold" in value:
        candidate = value.get("semantic_hold")
    else:
        result = value.get("result")
        candidate = result.get("semantic_hold") if isinstance(result, Mapping) else None
    if not isinstance(candidate, Mapping):
        return None
    resolved_lane = lane if lane is not None else candidate.get("lane")
    if not isinstance(resolved_lane, str) or not resolved_lane:
        return None
    if (
        semantic_no_quorum_hold_error(
            candidate,
            resolved_lane,
            epoch=epoch,
            authority=authority,
        )
        is not None
    ):
        return None
    return copy.deepcopy(dict(candidate))


def build_structured_review_hold_epoch(
    *,
    lane: str,
    authority: Mapping[str, Any],
    schema_sha256: str,
    prompt: str,
    system: str | None,
    effective_request_sha256: str,
    resolver_sha256: str = STRUCTURED_REVIEW_HOLD_RESOLVER_SHA256,
) -> dict[str, Any]:
    """Build the opaque exact input identity for the boundary cache."""

    return build_structured_review_epoch(
        lane=lane,
        authority=authority,
        schema_sha256=schema_sha256,
        prompt=prompt,
        system=system,
        effective_request_sha256=effective_request_sha256,
        resolver_sha256=resolver_sha256,
    )


def structured_review_hold_epoch_error(
    epoch: object,
    *,
    lane: str,
    authority: Mapping[str, Any],
) -> str | None:
    """Validate an opaque cache epoch without access to request plaintext."""

    return structured_review_epoch_error(epoch, lane=lane, authority=authority)


_CACHE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "cache_key_sha256",
        "hold",
        "result_sha256",
        "result",
    }
)
_FORBIDDEN_PLAINTEXT_KEYS = frozenset(
    {
        "messages",
        "model_response",
        "prompt",
        "raw_output",
        "raw_response",
    }
)
_MAX_CACHE_BYTES = 4 * 1024 * 1024


def _forbidden_plaintext_key(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_PLAINTEXT_KEYS:
                return key
            nested = _forbidden_plaintext_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _forbidden_plaintext_key(item)
            if nested is not None:
                return nested
    return None


class StructuredReviewHoldLease:
    """One process-exclusive cache-key lease held across local model calls."""

    def __init__(
        self,
        *,
        cache_path: Path,
        cache_key_sha256: str,
        lane: str,
        epoch: Mapping[str, Any],
        authority: Mapping[str, Any],
    ) -> None:
        self.cache_path = cache_path
        self.cache_key_sha256 = cache_key_sha256
        self.lane = lane
        self.epoch = copy.deepcopy(dict(epoch))
        self.authority = copy.deepcopy(dict(authority))

    def load(self) -> dict[str, Any] | None:
        """Return a validated prior review; corruption is a closed cache miss."""

        try:
            if self.cache_path.stat().st_size > _MAX_CACHE_BYTES:
                return None
            raw = self.cache_path.read_bytes()
            if len(raw) > _MAX_CACHE_BYTES:
                return None
            record = json.loads(raw)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            RecursionError,
            OverflowError,
        ):
            return None
        if not isinstance(record, Mapping) or set(record) != _CACHE_FIELDS:
            return None
        if (
            record.get("schema_version") != STRUCTURED_REVIEW_HOLD_CACHE_SCHEMA_VERSION
            or record.get("kind") != STRUCTURED_REVIEW_HOLD_CACHE_KIND
            or record.get("cache_key_sha256") != self.cache_key_sha256
        ):
            return None
        result = record.get("result")
        hold = record.get("hold")
        if not isinstance(result, Mapping) or not isinstance(hold, Mapping):
            return None
        try:
            result_sha256 = canonical_sha256(result)
            forbidden_key = _forbidden_plaintext_key(result)
        except (TypeError, ValueError, RecursionError, OverflowError):
            return None
        if (
            record.get("result_sha256") != result_sha256
            or hold.get("review_sha256") != result_sha256
            or forbidden_key is not None
        ):
            return None
        try:
            hold_error = semantic_no_quorum_hold_error(
                hold,
                self.lane,
                epoch=self.epoch,
                authority=self.authority,
            )
        except (TypeError, ValueError, RecursionError, OverflowError):
            return None
        if hold_error is not None:
            return None
        try:
            rebuilt = build_semantic_no_quorum_hold(
                self.lane,
                self.epoch,
                self.authority,
                result,
            )
        except (TypeError, ValueError, RecursionError, OverflowError):
            return None
        if dict(hold) != rebuilt:
            return None
        return copy.deepcopy(dict(result))

    def store(self, result: Mapping[str, Any]) -> dict[str, Any]:
        """Atomically persist only one strict semantic no-quorum result."""

        forbidden_key = _forbidden_plaintext_key(result)
        if forbidden_key is not None:
            raise ValueError(
                f"structured review cache forbids plaintext field:{forbidden_key}"
            )
        hold = build_semantic_no_quorum_hold(
            self.lane,
            self.epoch,
            self.authority,
            result,
        )
        result_copy = copy.deepcopy(dict(result))
        result_sha256 = canonical_sha256(result_copy)
        record = {
            "schema_version": STRUCTURED_REVIEW_HOLD_CACHE_SCHEMA_VERSION,
            "kind": STRUCTURED_REVIEW_HOLD_CACHE_KIND,
            "cache_key_sha256": self.cache_key_sha256,
            "hold": hold,
            "result_sha256": result_sha256,
            "result": result_copy,
        }
        encoded = (_canonical_json(record) + "\n").encode("utf-8")
        if len(encoded) > _MAX_CACHE_BYTES:
            raise ValueError("structured review cache record exceeds byte limit")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.cache_path.name}.",
            suffix=".tmp",
            dir=self.cache_path.parent,
        )
        tmp_path = Path(tmp_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.cache_path)
            directory_fd = os.open(self.cache_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            tmp_path.unlink(missing_ok=True)
        return copy.deepcopy(hold)


class StructuredReviewSemanticHoldCache:
    """Per-request flock + atomic persistence for structured-review holds."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            from chronovisor.store import CHRONOVISOR_ROOT

            root = CHRONOVISOR_ROOT / "runtime" / "semantic-holds" / "structured-review"
        self.root = Path(root)
        self.entries_dir = self.root / "entries"
        self.locks_dir = self.root / "locks"

    @contextmanager
    def locked(
        self,
        *,
        lane: str,
        epoch: Mapping[str, Any],
        authority: Mapping[str, Any],
    ):
        """Serialize one exact epoch while allowing unrelated lanes in parallel."""

        epoch_error = structured_review_hold_epoch_error(
            epoch,
            lane=lane,
            authority=authority,
        )
        if epoch_error is not None:
            raise ValueError(epoch_error)
        cache_key_sha256 = canonical_sha256(
            {
                "schema_version": STRUCTURED_REVIEW_HOLD_CACHE_SCHEMA_VERSION,
                "kind": STRUCTURED_REVIEW_HOLD_CACHE_KIND,
                "lane": lane,
                "epoch": epoch,
                "authority_sha256": canonical_sha256(authority),
            }
        )
        self.entries_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.locks_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.locks_dir / f"{cache_key_sha256}.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            handle = os.fdopen(fd, "a+")
        except Exception:
            os.close(fd)
            raise
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            handle.close()
            raise
        try:
            yield StructuredReviewHoldLease(
                cache_path=self.entries_dir / f"{cache_key_sha256}.json",
                cache_key_sha256=cache_key_sha256,
                lane=lane,
                epoch=epoch,
                authority=authority,
            )
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                handle.close()
            except OSError:
                pass


__all__ = [
    "LOCAL_SEMANTIC_NO_QUORUM",
    "SCHEMA_VERSION",
    "SEMANTIC_NO_QUORUM_HOLD_KIND",
    "STRUCTURED_REVIEW_HOLD_CACHE_KIND",
    "STRUCTURED_REVIEW_HOLD_CACHE_SCHEMA_VERSION",
    "STRUCTURED_REVIEW_HOLD_EPOCH_VERSION",
    "STRUCTURED_REVIEW_HOLD_RESOLVER_SHA256",
    "STRUCTURED_REVIEW_HOLD_RESOLVER_VERSION",
    "StructuredReviewHoldLease",
    "StructuredReviewSemanticHoldCache",
    "build_semantic_no_quorum_hold",
    "build_structured_review_hold_epoch",
    "canonical_sha256",
    "frontier_failure_class",
    "is_local_semantic_no_quorum",
    "persisted_semantic_no_quorum_hold",
    "semantic_no_quorum_hold_error",
    "structured_review_hold_epoch_error",
    "structured_review_authority_observation_sha256",
]
