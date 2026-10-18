"""Fail-closed offline gate for a temporary single-teacher cohort."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from chronovisor.core import canonical_json
from chronovisor.recall.recall_distillation_luna import (
    NATIVE_IDENTITY_REVISION,
    NATIVE_MODEL,
    NATIVE_OUTPUT_SCHEMA_SHA256,
    NATIVE_REQUEST_REVISION,
    NATIVE_TEACHER_PROMPT_SHA256,
)
from chronovisor.recall.recall_distillation_remote_teacher import (
    OX_ALPHA_FIXED_IDENTITY,
    OX_ALPHA_ROUTE_MODEL,
)

DEEPSEEK_SINGLE_PROFILE = "deepseek-v4-flash-single-v1"
NATIVE_LUNA_SINGLE_PROFILE = "native-luna-single-v1"
NATIVE_LUNA_WORKER_KIND = "luna_worker"
NATIVE_LUNA_ROUTE = "native/luna_worker"
NATIVE_LUNA_MODEL = NATIVE_MODEL
NATIVE_LUNA_FIXED_IDENTITY_REVISION = NATIVE_IDENTITY_REVISION
# Keep the existing DeepSeek/OX request contract stable.  Native Luna has its
# own worker prompt/schema revision and must never silently reuse the remote
# teacher's revision.
TEACHER_REQUEST_REVISION = "json-schema-core-label-abstain-16k-600s-v8"
NATIVE_LUNA_REQUEST_REVISION = NATIVE_REQUEST_REVISION


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


_NATIVE_LUNA_ROUTE_IDENTITY = {
    "provider": "openai",
    "model": NATIVE_LUNA_MODEL,
    "location": "native",
}
NATIVE_LUNA_FIXED_IDENTITY = {
    "revision": NATIVE_LUNA_FIXED_IDENTITY_REVISION,
    "route": NATIVE_LUNA_ROUTE,
    "route_identity": dict(_NATIVE_LUNA_ROUTE_IDENTITY),
    "runtime_role": NATIVE_LUNA_WORKER_KIND,
    "native_luna": True,
    "model_digest": hashlib.sha256(NATIVE_LUNA_MODEL.encode("utf-8")).hexdigest(),
    "route_digest": _sha256_json(_NATIVE_LUNA_ROUTE_IDENTITY),
    "prompt_template_sha256": NATIVE_TEACHER_PROMPT_SHA256,
    "schema_revision_sha256": NATIVE_OUTPUT_SCHEMA_SHA256,
    "request_revision": NATIVE_LUNA_REQUEST_REVISION,
}


def teacher_identity_for_profile(profile: str) -> dict[str, Any]:
    """Return the immutable identity contract for one supported teacher profile."""

    if profile == NATIVE_LUNA_SINGLE_PROFILE:
        identity = NATIVE_LUNA_FIXED_IDENTITY
    elif profile == DEEPSEEK_SINGLE_PROFILE:
        identity = {
            **OX_ALPHA_FIXED_IDENTITY,
            "route": OX_ALPHA_ROUTE_MODEL,
            "request_revision": TEACHER_REQUEST_REVISION,
        }
    else:
        raise ValueError("unsupported single-teacher profile")
    # Return a detached value so a caller cannot mutate the gate's closed map.
    return cast(
        dict[str, Any],
        json.loads(json.dumps(identity, ensure_ascii=False, sort_keys=True)),
    )


def validate_teacher_contract(
    profile: str, supplied: Mapping[str, Any] | None
) -> bool:
    """Accept only the exact closed identity contract for ``profile``."""

    if not isinstance(supplied, Mapping):
        return False
    try:
        expected = teacher_identity_for_profile(profile)
    except ValueError:
        return False
    return dict(supplied) == expected


def expected_teacher_request_sha256(
    *,
    profile_contract_id: str,
    payload_digest: str,
    profile: str = DEEPSEEK_SINGLE_PROFILE,
) -> str:
    """Bind a label to its sealed contract and non-sensitive request payload."""

    identity = teacher_identity_for_profile(profile)

    return hashlib.sha256(
        json.dumps(
            {
                "identity_revision": identity["revision"],
                "payload_digest": payload_digest,
                "profile_contract_id": profile_contract_id,
                "request_revision": identity["request_revision"],
                "route_digest": identity["route_digest"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def expected_ox_request_sha256(*, profile_contract_id: str, payload_digest: str) -> str:
    """Backward-compatible DeepSeek request binding."""

    return expected_teacher_request_sha256(
        profile_contract_id=profile_contract_id,
        payload_digest=payload_digest,
    )


def expected_ox_provider_request_sha256(
    *,
    profile_contract_id: str,
    payload_digest: str,
    work_id: str,
    expires_at: str,
) -> str:
    """Hash the non-sensitive envelope that uniquely binds one OX request."""

    return expected_teacher_provider_request_sha256(
        profile_contract_id=profile_contract_id,
        payload_digest=payload_digest,
        work_id=work_id,
        expires_at=expires_at,
    )


def expected_teacher_provider_request_sha256(
    *,
    profile_contract_id: str,
    payload_digest: str,
    work_id: str,
    expires_at: str,
    profile: str = DEEPSEEK_SINGLE_PROFILE,
) -> str:
    """Hash the envelope binding one request to a closed teacher identity."""

    identity = teacher_identity_for_profile(profile)

    return hashlib.sha256(
        json.dumps(
            {
                "contract": profile_contract_id,
                "expires_at": expires_at,
                "identity_revision": identity["revision"],
                "model_digest": identity["model_digest"],
                "payload_digest": payload_digest,
                "prompt_sha256": identity["prompt_template_sha256"],
                "route_digest": identity["route_digest"],
                "schema_sha256": identity["schema_revision_sha256"],
                "work_id": work_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _wilson_lower(successes: int, total: int) -> float:
    if total <= 0 or successes < 0 or successes > total:
        return 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    spread = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return max(0.0, (center - spread) / denominator)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


_DECISIVE_VERDICTS = frozenset({"relevant", "irrelevant"})


def _complete_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("error_class") in (None, "")
        and (
            (
                row.get("status") == "completed"
                and row.get("verdict") in _DECISIVE_VERDICTS
            )
            or (
                row.get("status") == "abstained"
                and row.get("verdict") == "uncertain"
                and row.get("probe") is True
            )
        )
    )


def _provider_receipt_is_certifying(row: Mapping[str, Any]) -> bool:
    """Return whether a completed row carries an adapter-observed receipt.

    ``provider_request_sha256`` is the deterministic request-intent binding and
    is checked against :func:`expected_ox_provider_request_sha256` below.  It
    must not be reused as a response/receipt surrogate: the producer's actual
    adapter receipt is carried in ``provider_receipt_sha256`` and is only
    certifying when it is a canonical lower-case SHA-256 string distinct from
    that request binding.  Rows carrying the removed
    ``provider_response_request_sha256`` field remain legacy/non-certifying,
    even when they also carry the new receipt field.
    """

    if "provider_response_request_sha256" in row:
        return False
    receipt = _text(row.get("provider_receipt_sha256"))
    if re.fullmatch(r"[0-9a-f]{64}", receipt) is None:
        return False
    return receipt != _text(row.get("provider_request_sha256"))


def _one(values: Sequence[str], name: str, reasons: list[str]) -> str:
    distinct = set(values)
    if not values or "" in distinct or len(distinct) != 1:
        reasons.append(f"{name}_identity_not_exactly_one")
        return ""
    return next(iter(distinct))


def _validate_identity_contract(
    completed: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    profile_contract_id: str,
    expected_identity: Mapping[str, Any] | None,
    identities: Mapping[str, str],
    reasons: list[str],
) -> None:
    """Check closed teacher identity and request bindings without changing floors."""

    if identities["profile"] and identities["profile"] != profile:
        reasons.append("profile_identity_mismatch")
    if profile_contract_id and identities["profile_contract_id"] != profile_contract_id:
        reasons.append("profile_contract_mismatch")
    if expected_identity is None:
        return
    expected_route = _text(expected_identity.get("route"))
    if not expected_route:
        route_identity = expected_identity.get("route_identity")
        if isinstance(route_identity, Mapping):
            expected_route = _text(route_identity.get("model"))
    if identities["route"] and expected_route and identities["route"] != expected_route:
        reasons.append("route_mismatch")
    for name, expected in (
        ("model_digest", expected_identity["model_digest"]),
        ("prompt_sha256", expected_identity["prompt_template_sha256"]),
        ("schema_sha256", expected_identity["schema_revision_sha256"]),
        ("route_digest", expected_identity["route_digest"]),
    ):
        if identities[name] != expected:
            reasons.append(f"{name}_contract_mismatch")
    if any(
        expected_teacher_request_sha256(
            profile_contract_id=_text(row.get("profile_contract_id")),
            payload_digest=_text(row.get("payload_digest")),
            profile=profile,
        )
        != _text(row.get("request_sha256"))
        for row in completed
    ):
        reasons.append("request_sha256_contract_mismatch")
    if any(
        expected_teacher_provider_request_sha256(
            profile_contract_id=_text(row.get("profile_contract_id")),
            payload_digest=_text(row.get("payload_digest")),
            work_id=_text(row.get("work_id")),
            expires_at=_text(row.get("expires_at")),
            profile=profile,
        )
        != _text(row.get("provider_request_sha256"))
        for row in completed
    ):
        reasons.append("provider_request_sha256_contract_mismatch")


def _validate_expiry(
    completed: Sequence[Mapping[str, Any]], *, profile: str, reasons: list[str]
) -> None:
    """Keep remote expiry strict while allowing the sealed native contract's empty expiry."""

    if profile == NATIVE_LUNA_SINGLE_PROFILE:
        if any(row.get("expires_at") != "" for row in completed):
            reasons.append("expiry_invalid")
        return

    def future_utc(value: object) -> bool:
        text = _text(value)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text) is None:
            return False
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")) > datetime.now(
                UTC
            )
        except ValueError:
            return False

    if any(not future_utc(row.get("expires_at")) for row in completed):
        reasons.append("expiry_missing")


def evaluate_single_teacher_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    cohort: str,
    profile_contract_id: str = "",
    min_labels: int = 500,
    min_per_class: int = 100,
    min_repeat_pairs: int = 100,
    min_repeat_stability: float = 0.60,
    row_validator: Callable[[Mapping[str, Any]], bool] | None = None,
    teacher_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate sealed-like rows without granting them verified-truth authority.

    A usable row has ``status='completed'`` with a decisive relevance verdict,
    or a probe with ``status='abstained'`` and an uncertain verdict, no error,
    and a canonical adapter-observed ``provider_receipt_sha256``.  It must carry
    immutable ``profile``, ``cohort``, ``route``, ``model_digest``,
    ``prompt_sha256``, ``schema_sha256``, and ``profile_contract_id`` identities.
    The captured provider/model/location route must match exactly.  Completed
    test rows additionally need ``locked_test_read_only=True`` and a non-empty
    ``locked_test_evidence_ref``.  Repeat probes use ``repeat_pair_id``,
    ``fixed_repeat=True``, ``order_swap=True``, and opposite ``blind_order``s.
    """

    reasons: list[str] = []
    valid_rows = [row for row in rows if isinstance(row, Mapping)]
    if len(valid_rows) != len(rows):
        reasons.append("input_row_invalid")
    rows = valid_rows
    if not isinstance(profile, str) or not profile:
        reasons.append("profile_argument_invalid")
    if not isinstance(cohort, str) or not cohort:
        reasons.append("cohort_argument_invalid")
    if min_labels < 1 or min_per_class < 1 or min_repeat_pairs < 1:
        reasons.append("minimum_argument_invalid")
    if not 0.0 <= min_repeat_stability <= 1.0:
        reasons.append("repeat_stability_argument_invalid")
    expected_identity: dict[str, Any] | None = None
    if isinstance(profile, str) and profile:
        try:
            expected_identity = teacher_identity_for_profile(profile)
        except ValueError:
            reasons.append("profile_contract_unknown")
    if teacher_contract is not None and not validate_teacher_contract(
        profile, teacher_contract
    ):
        reasons.append("teacher_contract_mismatch")

    current_rows = [
        row
        for row in rows
        if not profile_contract_id
        or row.get("profile_contract_id") == profile_contract_id
    ]
    excluded_prior_contract_rows = len(rows) - len(current_rows)
    # Counterfactuals are produced locally and are evaluated by the separate
    # directional gate.  They must not impersonate the remote OX teacher's
    # fixed identity here.
    completed = [
        row
        for row in current_rows
        if _complete_row(row) and row.get("source") != "counterfactual-label"
    ]
    certifying_completed = [
        row for row in completed if _provider_receipt_is_certifying(row)
    ]
    if len(certifying_completed) != len(completed):
        reasons.append("provider_receipt_sha256_invalid")
    required_splits = {"train", "validation", "test"}
    # Rows without an adapter-observed receipt remain auditable below, but are
    # explicitly non-certifying and cannot contribute to any quality floor.
    eligible = [
        row for row in certifying_completed if row.get("split") in required_splits
    ]
    excluded = len(rows) - len(eligible)
    labels = [
        row
        for row in eligible
        if row.get("probe") is not True
        and row.get("verdict") in _DECISIVE_VERDICTS
    ]
    probes = [row for row in eligible if row.get("probe") is True]
    if row_validator is not None and any(not row_validator(row) for row in eligible):
        reasons.append("row_authority_mismatch")
    relevant = sum(row.get("verdict") == "relevant" for row in labels)
    irrelevant = sum(row.get("verdict") == "irrelevant" for row in labels)

    if len(labels) < min_labels:
        reasons.append("teacher_labels_below_floor")
    if relevant < min_per_class or irrelevant < min_per_class:
        reasons.append("teacher_class_below_floor")
    if any(
        row.get("negative_veto_conflict") is not None
        and row.get("negative_veto_conflict") is not False
        for row in current_rows
    ):
        reasons.append("negative_veto_conflict")

    identities = {
        name: _one([_text(row.get(name)) for row in completed], name, reasons)
        for name in (
            "profile",
            "cohort",
            "route",
            "model_digest",
            "prompt_sha256",
            "schema_sha256",
            "split_plan_id",
            "profile_contract_id",
            "identity_revision",
            "request_revision",
            "route_digest",
        )
    }
    if identities["cohort"] and identities["cohort"] != cohort:
        reasons.append("cohort_identity_mismatch")
    _validate_identity_contract(
        completed,
        profile=profile,
        profile_contract_id=profile_contract_id,
        expected_identity=expected_identity,
        identities=identities,
        reasons=reasons,
    )
    if any(
        re.fullmatch(r"[0-9a-f]{64}", _text(row.get("payload_digest"))) is None
        or not isinstance(row.get("payload_source"), Mapping)
        or canonical_json.canonical_json_sha256_strict(row["payload_source"])
        != _text(row.get("payload_digest"))
        for row in completed
    ):
        reasons.append("payload_source_digest_mismatch")

    _validate_expiry(completed, profile=profile, reasons=reasons)
    for name in (
        "model_digest",
        "prompt_sha256",
        "schema_sha256",
        "split_plan_id",
        "profile_contract_id",
        "route_digest",
    ):
        if identities[name] and re.fullmatch(r"[0-9a-f]{64}", identities[name]) is None:
            reasons.append(f"{name}_identity_invalid")
    expected_identity_revision = (
        _text(expected_identity.get("revision"))
        if expected_identity is not None
        else ""
    )
    expected_request_revision = (
        _text(expected_identity.get("request_revision"))
        if expected_identity is not None
        else ""
    )
    if expected_identity is not None and identities["identity_revision"] != expected_identity_revision:
        reasons.append("identity_revision_mismatch")
    if expected_identity is not None and identities["request_revision"] != expected_request_revision:
        reasons.append("request_revision_mismatch")
    if profile == NATIVE_LUNA_SINGLE_PROFILE:
        if any(row.get("native_luna") is not True for row in completed):
            reasons.append("native_luna_attestation_missing")
        if any(
            row.get("teacher_role") != NATIVE_LUNA_WORKER_KIND for row in completed
        ):
            reasons.append("native_luna_role_mismatch")
    if any(row.get("route_identity_exact") is not True for row in completed):
        reasons.append("route_identity_mismatch")
    if any(row.get("fixed_split_plan") is not True for row in completed):
        reasons.append("fixed_split_plan_missing")

    if len(eligible) != len(certifying_completed):
        reasons.append("split_assignment_invalid")
    split_rows = [
        row for row in certifying_completed if row.get("split") in required_splits
    ]
    if {str(row.get("split")) for row in split_rows} != required_splits:
        reasons.append("chronological_split_incomplete")
    missing_group = any(not _text(row.get("group_id")) for row in completed)
    missing_time = any(not _text(row.get("as_of")) for row in completed)
    if missing_group:
        reasons.append("group_identity_missing")
    if any(row.get("group_identity_exact") is not True for row in completed):
        reasons.append("group_identity_mismatch")
    if missing_time:
        reasons.append("chronological_timestamp_missing")
    groups: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        group = _text(row.get("group_id"))
        if group:
            groups[group].add(str(row["split"]))
    if any(len(splits) > 1 for splits in groups.values()):
        reasons.append("group_split_leakage")
    times = {
        split: [_text(row.get("as_of")) for row in split_rows if row["split"] == split]
        for split in required_splits
    }
    if (
        all(times.values())
        and not missing_time
        and not (max(times["train"]) < min(times["validation"]) < min(times["test"]))
    ):
        reasons.append("chronological_order_invalid")

    completed_flags = completed
    if any("future_leakage" not in row for row in completed_flags):
        reasons.append("future_leakage_flag_missing")
    if any(
        "future_leakage" in row and row.get("future_leakage") is not False
        for row in completed_flags
    ):
        reasons.append("future_leakage_detected")
    if any("feature_parity" not in row for row in completed_flags):
        reasons.append("feature_parity_flag_missing")
    if any(row.get("feature_parity") is not True for row in completed_flags):
        reasons.append("feature_parity_failed")

    test_rows = [
        row for row in certifying_completed if row.get("split") == "test"
    ]
    evidence_refs = sorted(
        {
            _text(row.get("locked_test_evidence_ref"))
            for row in test_rows
            if _text(row.get("locked_test_evidence_ref"))
        }
    )
    if not test_rows or any(
        row.get("locked_test_read_only") is not True
        or _text(row.get("locked_test_evidence_ref"))
        != f"split-plan:{_text(row.get('split_plan_id'))}"
        for row in test_rows
    ):
        reasons.append("locked_test_read_only_evidence_missing")

    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    incomplete_pairs = 0
    for row in probes:
        pair_id = _text(row.get("repeat_pair_id"))
        if pair_id:
            pairs[pair_id].append(row)
        else:
            incomplete_pairs += 1
    complete_pairs: list[list[Mapping[str, Any]]] = []
    pair_identity_mismatch = False
    for pair in pairs.values():
        orders = {row.get("blind_order") for row in pair}
        targets = {
            (_text(row.get("rally_id")), _text(row.get("candidate_id"))) for row in pair
        }
        identity_exact = len(targets) == 1 and all(next(iter(targets), ()))
        pair_identity_mismatch = pair_identity_mismatch or not identity_exact
        valid = (
            len(pair) == 2
            and all(row.get("fixed_repeat") is True for row in pair)
            and all(row.get("order_swap") is True for row in pair)
            and orders == {"a_first", "b_first"}
            and identity_exact
        )
        if valid:
            complete_pairs.append(pair)
        else:
            incomplete_pairs += 1
    stable = sum(
        pair[0].get("verdict") in _DECISIVE_VERDICTS
        and pair[1].get("verdict") in _DECISIVE_VERDICTS
        and pair[0].get("verdict") == pair[1].get("verdict")
        for pair in complete_pairs
    )
    stability = _wilson_lower(stable, len(complete_pairs))
    if incomplete_pairs:
        reasons.append("blind_repeat_pair_incomplete")
    if pair_identity_mismatch:
        reasons.append("blind_repeat_identity_mismatch")
    if len(complete_pairs) < min_repeat_pairs:
        reasons.append("blind_repeat_pairs_below_floor")
    if stability < min_repeat_stability:
        reasons.append("blind_repeat_stability_below_gate")

    return {
        "schema": "chronovisor.recall-single-teacher-gate.v1",
        "truth_authority": "teacher_only_not_verified",
        "profile": profile,
        "cohort": cohort,
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "labels": {
            "eligible": len(labels),
            "relevant": relevant,
            "irrelevant": irrelevant,
            "excluded": excluded,
        },
        "excluded_prior_contract_rows": excluded_prior_contract_rows,
        "identity": identities,
        "chronological_grouped": {
            "splits": sorted({str(row.get("split")) for row in split_rows}),
            "groups": len(groups),
        },
        "locked_test": {
            "rows": len(test_rows),
            "read_only": not any(
                row.get("locked_test_read_only") is not True for row in test_rows
            ),
            "evidence_refs": evidence_refs,
        },
        "blind_repeat": {
            "complete_pairs": len(complete_pairs),
            "incomplete_pairs": incomplete_pairs,
            "stable": stable,
            "unstable": len(complete_pairs) - stable,
            "wilson_lower": round(stability, 8),
        },
    }
