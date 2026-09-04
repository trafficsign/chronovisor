"""Autonomous consumer for non-trivial Chronovisor lint repair lanes.

``lint.py`` remains the detector and safe deterministic fixer.  This module
drains the remaining queue with explicit boundaries:

* missing/count-invalid tags are proposed by local Ollama, then every semantic
  tag decision is finalized by a bounded frontier structured review;
* a frontier approval/rejection is durably bound to the exact page preimage
  before any mutation, so crash recovery never falls back to local authority;
* stale monitor rows terminate as observed facts; and
* duplicate/orphan review rows terminate in this lane as routed work for their
  specialized consumers.

No page or convergence state is written in ``dry_run`` mode.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from chronovisor.core import store as chronovisor_store
from chronovisor.core.frontmatter import parse as parse_frontmatter
from chronovisor.core.frontmatter import patch as patch_frontmatter
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.page_mutation import (
    chronovisor_mutation_lock,
    decision_authority_lock,
)
from chronovisor.core.runtime_config import load_ingest_config, runtime_repo_root
from chronovisor.core.tag_rules import (
    SEED_TAGS,
    parse_tags,
    validate_axis_counts,
    validate_tag,
)
from chronovisor.decision import routine_review
from chronovisor.decision.decision_authority import (
    compare_semantic_authority,
    current_semantic_authority,
    seal_semantic_artifact,
    semantic_authority_shape_error,
    semantic_verdict_authority_error,
)
from chronovisor.decision.decision_lane_prompts import (
    TAG_REVIEW_CONTRACT_VERSION,
    build_frontier_tag_repair_prompt,
    tag_repair_page_excerpt,
)
from chronovisor.decision.decision_schema_manifest import TAG_REPAIR_SCHEMA
from chronovisor.decision.local_structured import ChatTransport, LocalStructuredSession
from chronovisor.decision.semantic_hold import (
    LOCAL_SEMANTIC_NO_QUORUM,
    canonical_sha256,
    is_local_semantic_no_quorum,
    persisted_semantic_no_quorum_hold,
    semantic_no_quorum_hold_error,
)
from chronovisor.ingest.convergence import (
    FRONTIER_STATUSES,
    TERMINAL_STATUSES,
    ConvergenceStore,
    CycleBudget,
    is_human_required_result,
    stable_item_key,
)

REPAIR_RESOLVER_VERSION = "lint-repair-v1"
TAG_REPAIR_DECISION_LANE = "lint_tag_repair"
TAG_REPAIR_RUNTIME_ROLE = "lint.tag_repair"
REPO_ROOT = runtime_repo_root()

LOCAL_TAG_SYSTEM = """\
You repair Chronovisor page tags. Return JSON only, matching the supplied schema.
Choose 1-3 domain tags (d/), exactly one type tag (t/), and exactly one scope
tag (s/). Tags must be lowercase ASCII kebab-case and contain at most two words
after the prefix. Prefer existing seed tags when they fit. Use uncertain when
the page excerpt is insufficient; never invent facts to justify a tag.
"""

StructuredReviewer = Callable[[str, dict[str, Any]], Mapping[str, Any] | str]


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return [], 0
    rows: list[dict[str, Any]] = []
    invalid = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            invalid += 1
    return rows, invalid


def _extract_json_object(value: Mapping[str, Any] | str) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(value):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(value, index)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_tag_decision(value: Mapping[str, Any] | str) -> dict[str, Any]:
    """Fail-closed normalization for local and frontier tag decisions."""

    parsed = _extract_json_object(value)
    if parsed is None:
        return {
            "decision": "needs_retry",
            "tags": [],
            "reason": "review output did not contain a JSON object",
            "valid": False,
            "validation_errors": ["missing_json"],
        }
    allowed = {
        "decision",
        "tags",
        "reason",
        "summary",
        "reviewer",
        "frontier_failure",
        "human_required",
        "decision_policy",
        "local_consensus",
    }
    extra = sorted(set(parsed) - allowed)
    decision = parsed.get("decision")
    tags_value = parsed.get("tags")
    reason = parsed.get("reason") or parsed.get("summary")
    errors: list[str] = []
    if extra:
        errors.append("unexpected fields: " + ", ".join(extra))
    if decision not in {"approved", "rejected", "uncertain", "needs_retry"}:
        errors.append("invalid decision")
        decision = "needs_retry"
    if not isinstance(tags_value, list) or not all(
        isinstance(tag, str) for tag in tags_value
    ):
        errors.append("tags must be a list of strings")
        tags: list[str] = []
    else:
        tags = list(tags_value)
        if len(tags) != len(set(tags)):
            errors.append("duplicate tags")
    if len(tags) > 5:
        errors.append("too many tags")
    for tag in tags:
        valid, tag_reason = validate_tag(tag)
        if not valid:
            errors.append(f"{tag!r}: {tag_reason}")
    axis_errors = validate_axis_counts(parse_tags(tags)) if tags else ["empty tag set"]
    if decision == "approved" and axis_errors:
        errors.extend(axis_errors)
    if not isinstance(reason, str) or not reason.strip():
        errors.append("reason is required")
        reason = str(reason or "")
    valid_decision = not errors
    out = {
        "decision": decision if valid_decision else "needs_retry",
        "tags": tags if valid_decision else [],
        "reason": reason.strip(),
        "valid": valid_decision,
        "validation_errors": errors,
    }
    for key in ("reviewer", "frontier_failure"):
        if key in parsed:
            out[key] = parsed[key]
    for key in ("decision_policy", "local_consensus"):
        if isinstance(parsed.get(key), Mapping):
            out[key] = dict(parsed[key])
    if "human_required" in parsed or "frontier_failure" in parsed:
        out["human_required"] = is_human_required_result(parsed)
    return out


def valid_tag_set(tags: object) -> bool:
    if (
        not isinstance(tags, list)
        or not tags
        or not all(isinstance(tag, str) for tag in tags)
    ):
        return False
    if len(tags) != len(set(tags)) or len(tags) > 5:
        return False
    return all(validate_tag(tag)[0] for tag in tags) and not validate_axis_counts(
        parse_tags(tags)
    )


def _page_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tag_semantic_epoch(
    item: Mapping[str, Any],
    *,
    page_text: str,
    prompt: str,
) -> dict[str, Any]:
    """Return the exact deterministic input reviewed by the tag panel."""

    return {
        "resolver_version": REPAIR_RESOLVER_VERSION,
        "contract_version": TAG_REVIEW_CONTRACT_VERSION,
        "input_hash": str(item.get("input_hash") or ""),
        "page_sha256": _page_hash(page_text),
        "prompt_sha256": _page_hash(prompt),
    }


def build_tag_repair_prompt(row: Mapping[str, Any], page_text: str) -> str:
    seed_tags = [tag for values in SEED_TAGS.values() for tag in values]
    return f"""\
Repair the tag set for this Chronovisor page.

Issue:
{json.dumps(dict(row), ensure_ascii=False, indent=2, default=str)}

Seed tags (prefer when semantically correct):
{json.dumps(seed_tags, ensure_ascii=False)}

Page excerpt:
{tag_repair_page_excerpt(page_text)}

Return JSON matching this schema:
{json.dumps(TAG_REPAIR_SCHEMA, ensure_ascii=False, indent=2)}
"""


def _gate_tag_review_to_exact_proposal(
    decision: Mapping[str, Any],
    local_proposal: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Prevent a reviewer from replacing the candidate it was asked to gate."""

    normalized = normalize_tag_decision(decision)
    if normalized.get("decision") != "approved" or normalized.get("valid") is not True:
        if normalized.get("decision") != "approved":
            normalized["tags"] = []
        return normalized

    proposal = (
        normalize_tag_decision(_decision_payload(local_proposal))
        if isinstance(local_proposal, Mapping)
        else None
    )
    proposal_tags = proposal.get("tags") if isinstance(proposal, Mapping) else None
    if (
        not isinstance(proposal, Mapping)
        or proposal.get("decision") != "approved"
        or proposal.get("valid") is not True
        or not isinstance(proposal_tags, list)
        or sorted(normalized.get("tags", [])) != sorted(proposal_tags)
    ):
        return {
            "decision": "needs_retry",
            "tags": [],
            "reason": "review did not approve the exact durable local tag proposal",
            "valid": True,
            "validation_errors": [],
        }
    # Preserve the proposal's canonical order so the applied bytes are exactly
    # the candidate that the panel reviewed even if a model echoed set order
    # differently.
    normalized["tags"] = list(proposal_tags)
    return normalized


def _default_local_reviewer(
    prompt: str,
    schema: dict[str, Any],
    *,
    transport: ChatTransport | None = None,
    audit_root: Path | None = None,
) -> Mapping[str, Any] | str:
    """Run the local tag proposal as a bounded repairable JSON session."""

    config = load_ingest_config()
    result = LocalStructuredSession(
        model="injected:lint-tag-repair" if transport is not None else None,
        transport=transport,
        role="lint_tag_repair",
        runtime_role=TAG_REPAIR_RUNTIME_ROLE,
        source_data_class="page",
        source_sensitivity="high",
        audit_root=audit_root,
        num_ctx=config.num_ctx,
        num_predict=min(config.num_predict, 2_048),
        keep_alive=config.keep_alive,
        read_timeout_ms=config.read_timeout_ms,
        max_input_chars=65_536,
        max_output_chars=4_000,
        max_feedback_chars=2_000,
    ).run(prompt, schema, system=LOCAL_TAG_SYSTEM)
    if not result.ok:
        reason = result.failure_class or "structured_session_failed"
        detail = result.failure_reason or "local tag proposal did not converge"
        raise ValueError(f"local tag proposal failed: {reason}: {detail}")
    if not isinstance(result.value, Mapping):
        raise ValueError("local tag proposal is not an object")
    return dict(result.value)


def _default_frontier_reviewer(
    prompt: str, schema: dict[str, Any]
) -> Mapping[str, Any] | str:
    return routine_review.run_structured_review(
        prompt,
        schema,
        repo_root=REPO_ROOT,
        execute_patch=False,
        command_env="CHRONOVISOR_LINT_REPAIR_FRONTIER_CMD",
        decision_lane="lint_tag_repair",
    )


def _failure_class(decision: Mapping[str, Any], fallback: str) -> str:
    failure = decision.get("frontier_failure")
    if isinstance(failure, Mapping) and isinstance(failure.get("failure_class"), str):
        return str(failure["failure_class"])
    return fallback


def _review_artifact_path(store: ConvergenceStore, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return store.state_file.parent / "lint-repair-frontier-reviews" / f"{digest}.json"


def _proposal_artifact_path(store: ConvergenceStore, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return store.state_file.parent / "lint-repair-local-proposals" / f"{digest}.json"


def _write_json_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _load_json_artifact(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _decision_payload(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Persist schema fields and the production local-consensus audit."""

    payload = {
        "decision": decision.get("decision"),
        "tags": list(decision.get("tags") or []),
        "reason": decision.get("reason"),
    }
    for key in ("decision_policy", "local_consensus"):
        if isinstance(decision.get(key), Mapping):
            payload[key] = dict(decision[key])
    return payload


def _tag_postimage(
    page_text: str,
    tags: list[str],
    *,
    updated_date: str,
) -> str:
    return patch_frontmatter(page_text, {"tags": tags, "updated": updated_date})


def _write_local_proposal_artifact(
    store: ConvergenceStore,
    key: str,
    *,
    page_text: str,
    decision: Mapping[str, Any],
) -> None:
    _write_json_artifact(
        _proposal_artifact_path(store, key),
        {
            "schema_version": 1,
            "kind": "lint_tag_local_proposal",
            "key": key,
            "page_sha256": _page_hash(page_text),
            "proposal": _decision_payload(decision),
        },
    )


def _load_local_proposal_artifact(
    store: ConvergenceStore,
    key: str,
    *,
    page_text: str,
) -> dict[str, Any] | None:
    artifact = _load_json_artifact(_proposal_artifact_path(store, key))
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != 1
        or artifact.get("kind") != "lint_tag_local_proposal"
        or artifact.get("key") != key
        or artifact.get("page_sha256") != _page_hash(page_text)
    ):
        return None
    proposal = artifact.get("proposal")
    if not isinstance(proposal, Mapping):
        return None
    normalized = normalize_tag_decision(proposal)
    if normalized.get("decision") != "approved" or normalized.get("valid") is not True:
        return None
    return normalized


def _write_frontier_review_artifact(
    store: ConvergenceStore,
    key: str,
    *,
    page_text: str,
    prompt: str,
    decision: Mapping[str, Any],
    authority: Mapping[str, Any],
    page_id: str,
    updated_date: str,
) -> dict[str, Any]:
    tags = list(decision.get("tags") or [])
    postimage_sha256 = (
        _page_hash(_tag_postimage(page_text, tags, updated_date=updated_date))
        if decision.get("decision") == "approved" and valid_tag_set(tags)
        else None
    )
    envelope = seal_semantic_artifact(
        {
            "schema_version": 2,
            "kind": "lint_tag_frontier_verdict",
            "key": key,
            "page_id": page_id,
            "page_sha256": _page_hash(page_text),
            "postimage_sha256": postimage_sha256,
            "updated_date": updated_date,
            "prompt_sha256": _page_hash(prompt),
            "verdict": _decision_payload(decision),
        },
        authority=authority,
        lane=TAG_REPAIR_DECISION_LANE,
    )
    _write_json_artifact(
        _review_artifact_path(store, key),
        envelope,
    )
    return envelope


def _load_frontier_review_artifact(
    store: ConvergenceStore,
    key: str,
    *,
    page_text: str,
    prompt: str,
    authority: Mapping[str, Any],
    page_id: str,
) -> dict[str, Any] | None:
    artifact = _load_json_artifact(_review_artifact_path(store, key))
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != 2
        or artifact.get("kind") != "lint_tag_frontier_verdict"
        or artifact.get("key") != key
        or artifact.get("page_id") != page_id
        or artifact.get("page_sha256") != _page_hash(page_text)
        or artifact.get("prompt_sha256") != _page_hash(prompt)
        or artifact.get("authority") != authority
    ):
        return None
    verdict = artifact.get("verdict")
    if not isinstance(verdict, Mapping):
        return None
    normalized = normalize_tag_decision(verdict)
    if normalized.get("decision") not in {"approved", "rejected"}:
        return None
    if normalized.get("valid") is not True:
        return None
    updated_date = artifact.get("updated_date")
    postimage_sha256 = artifact.get("postimage_sha256")
    if not isinstance(updated_date, str):
        return None
    if normalized.get("decision") == "approved":
        expected_postimage_sha256 = _page_hash(
            _tag_postimage(
                page_text,
                list(normalized.get("tags") or []),
                updated_date=updated_date,
            )
        )
        if postimage_sha256 != expected_postimage_sha256:
            return None
    elif postimage_sha256 is not None:
        return None
    if (
        semantic_verdict_authority_error(
            normalized,
            authority,
            lane=TAG_REPAIR_DECISION_LANE,
        )
        is not None
    ):
        return None
    normalized["authority"] = dict(authority)
    normalized["updated_date"] = updated_date
    normalized["postimage_sha256"] = postimage_sha256
    normalized["reused"] = True
    return normalized


def _find_exact_applied_recovery(
    store: ConvergenceStore,
    *,
    page_id: str,
    page_text: str,
    tags: list[str],
) -> dict[str, Any] | None:
    """Find a prior approved artifact whose exact postimage is already present.

    This is bookkeeping recovery only: it never reinstalls an old semantic
    verdict after its authority epoch has changed.
    """

    artifact_dir = store.state_file.parent / "lint-repair-frontier-reviews"
    for path in sorted(artifact_dir.glob("*.json")):
        artifact = _load_json_artifact(path)
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("schema_version") != 2
            or artifact.get("kind") != "lint_tag_frontier_verdict"
            or artifact.get("page_id") != page_id
            or artifact.get("postimage_sha256") != _page_hash(page_text)
        ):
            continue
        authority = artifact.get("authority")
        if (
            semantic_authority_shape_error(
                authority,
                lane=TAG_REPAIR_DECISION_LANE,
            )
            is not None
        ):
            continue
        verdict = artifact.get("verdict")
        if not isinstance(verdict, Mapping):
            continue
        normalized = normalize_tag_decision(verdict)
        if (
            normalized.get("decision") != "approved"
            or normalized.get("valid") is not True
            or normalized.get("tags") != tags
            or semantic_verdict_authority_error(
                normalized,
                authority,
                lane=TAG_REPAIR_DECISION_LANE,
            )
            is not None
        ):
            continue
        return {
            "artifact": str(path),
            "review_key": artifact.get("key"),
            "postimage_sha256": artifact.get("postimage_sha256"),
        }
    return None


def _current_review_authority_error(
    decision: Mapping[str, Any],
    expected_authority: object,
    *,
    injected_reviewer: bool,
) -> str | None:
    current_authority, authority_error = current_semantic_authority(
        TAG_REPAIR_DECISION_LANE,
        injected_reviewer=injected_reviewer,
    )
    return (
        authority_error
        or compare_semantic_authority(
            expected_authority,
            current_authority,
            lane=TAG_REPAIR_DECISION_LANE,
        )
        or semantic_verdict_authority_error(
            decision,
            expected_authority,
            lane=TAG_REPAIR_DECISION_LANE,
        )
    )


def apply_tags_cas(
    path: Path,
    *,
    expected_text: str,
    tags: list[str],
    updated_date: str | None = None,
    expected_postimage_sha256: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Patch tags only if the page still matches the reviewed preimage."""

    if not valid_tag_set(tags):
        return {
            "status": "invalid",
            "reason": "candidate tags failed taxonomy validation",
        }
    try:
        current = path.read_text(encoding="utf-8")
    except Exception as exc:
        return {"status": "error", "reason": f"read_error: {exc}"}
    if current != expected_text:
        return {
            "status": "cas_conflict",
            "reason": "page changed after tag review",
            "expected_hash": _page_hash(expected_text),
            "actual_hash": _page_hash(current),
        }
    try:
        patched = _tag_postimage(
            current,
            tags,
            updated_date=updated_date or date.today().isoformat(),
        )
    except Exception as exc:
        return {"status": "error", "reason": f"frontmatter_error: {exc}"}
    if (
        expected_postimage_sha256 is not None
        and _page_hash(patched) != expected_postimage_sha256
    ):
        return {
            "status": "invalid",
            "reason": "reviewed postimage hash does not match candidate patch",
        }
    if patched == current:
        return {"status": "unchanged", "path": str(path), "tags": tags}
    if dry_run:
        return {"status": "would_apply", "path": str(path), "tags": tags}
    try:
        # Serialize all cooperating page writers, then perform the final CAS
        # immediately next to the replace.  A correction that landed after
        # review therefore wins instead of being overwritten by stale tags.
        with chronovisor_mutation_lock():
            if path.read_text(encoding="utf-8") != expected_text:
                return {
                    "status": "cas_conflict",
                    "reason": "page changed before atomic write",
                }
            atomic_write(path, patched)
            written = path.read_text(encoding="utf-8")
    except Exception as exc:
        return {"status": "error", "reason": f"write_error: {exc}"}
    if written != patched:
        return {
            "status": "error",
            "reason": "post_write_verification_failed",
            "expected_hash": _page_hash(patched),
            "actual_hash": _page_hash(written),
        }
    return {
        "status": "applied",
        "path": str(path),
        "tags": tags,
        "before_hash": _page_hash(expected_text),
        "after_hash": _page_hash(patched),
    }


def _candidate_identity(
    row: Mapping[str, Any],
    page_text: str | None,
) -> tuple[str, dict[str, Any]]:
    issue_type = str(row.get("issue_type") or "unknown")
    page = str(row.get("page") or "")
    source_id = f"{issue_type}:{page or 'no-page'}"
    input_data = {
        "issue_type": issue_type,
        "page": page,
        "detail": str(row.get("detail") or ""),
        "severity": str(row.get("severity") or ""),
        "page_hash": _page_hash(page_text) if page_text is not None else "missing",
    }
    return source_id, input_data


def _route_target(issue_type: str) -> str | None:
    if issue_type == "duplicate":
        return "duplicate_review"
    if issue_type == "orphan":
        return "orphan_link"
    return None


def _terminal_result(
    store: ConvergenceStore,
    key: str,
    *,
    action: str,
    result: Mapping[str, Any],
    now: datetime | None,
) -> dict[str, Any]:
    transition = store.complete(
        key, "applied", result={"action": action, **dict(result)}, now=now
    )
    return transition["item"]


def _apply_reviewed_tags(
    *,
    store: ConvergenceStore,
    budget: CycleBudget,
    key: str,
    stage: Literal["local", "frontier"],
    owner: str,
    path: Path,
    expected_text: str,
    decision: Mapping[str, Any],
    authority: Mapping[str, Any],
    injected_reviewer: bool,
    now: datetime | None,
) -> dict[str, Any]:
    allowed, budget_reason = budget.consume("mutation")
    if not allowed:
        failed = store.fail_attempt(
            key,
            stage,
            owner=owner,
            error=budget_reason,
            failure_class="cycle_budget_exhausted",
            now=now,
        )
        return {
            "status": "budget_exhausted",
            "state": failed["item"],
            "reason": budget_reason,
        }
    with decision_authority_lock():
        authority_error = _current_review_authority_error(
            decision,
            authority,
            injected_reviewer=injected_reviewer,
        )
        if authority_error is not None:
            failed = store.fail_attempt(
                key,
                stage,
                owner=owner,
                error=authority_error,
                failure_class="decision_authority_changed",
                now=now,
            )
            return {
                "status": "frontier_retry",
                "state": failed["item"],
                "reason": authority_error,
            }
        applied = apply_tags_cas(
            path,
            expected_text=expected_text,
            tags=list(decision["tags"]),
            updated_date=(
                str(decision["updated_date"])
                if isinstance(decision.get("updated_date"), str)
                else None
            ),
            expected_postimage_sha256=(
                str(decision["postimage_sha256"])
                if isinstance(decision.get("postimage_sha256"), str)
                else None
            ),
        )
        if applied["status"] in {"applied", "unchanged"}:
            completed = store.complete(
                key,
                "applied",
                owner=owner,
                result={
                    "action": "tag_repair",
                    "review_stage": stage,
                    "decision": dict(decision),
                    "apply": applied,
                },
                now=now,
            )
            return {
                "status": applied["status"],
                "state": completed["item"],
                "apply": applied,
            }
        if applied["status"] == "cas_conflict":
            quarantined = store.quarantine(
                key,
                owner=owner,
                reason="tag_repair_cas_conflict",
                now=now,
            )
            return {
                "status": "quarantined",
                "state": quarantined["item"],
                "apply": applied,
            }
        failed = store.fail_attempt(
            key,
            stage,
            owner=owner,
            error=str(applied.get("reason") or applied.get("status")),
            failure_class="tag_apply_error",
            now=now,
        )
        return {"status": "apply_error", "state": failed["item"], "apply": applied}


def _run_frontier_tag_review(
    *,
    row: Mapping[str, Any],
    path: Path,
    page_text: str,
    store: ConvergenceStore,
    budget: CycleBudget,
    key: str,
    reviewer: StructuredReviewer,
    now: datetime | None,
    local_proposal: Mapping[str, Any] | None = None,
    injected_reviewer: bool = False,
) -> dict[str, Any]:
    prompt = build_frontier_tag_repair_prompt(
        row,
        page_text,
        local_proposal=local_proposal,
    )
    with decision_authority_lock():
        authority, authority_error = current_semantic_authority(
            TAG_REPAIR_DECISION_LANE,
            injected_reviewer=injected_reviewer,
        )
        artifact_decision = (
            _load_frontier_review_artifact(
                store,
                key,
                page_text=page_text,
                prompt=prompt,
                authority=authority,
                page_id=str(row.get("page") or ""),
            )
            if authority_error is None and isinstance(authority, Mapping)
            else None
        )
        current_item = store.get(key) or {}
        restored_hold = (
            store.restore_semantic_no_quorum_hold(
                key,
                lane=TAG_REPAIR_DECISION_LANE,
                epoch=_tag_semantic_epoch(
                    current_item,
                    page_text=page_text,
                    prompt=prompt,
                ),
                authority=authority,
                now=now,
            )
            if authority_error is None and isinstance(authority, Mapping)
            else None
        )
    if restored_hold is not None:
        return {
            "status": "quarantined",
            "state": restored_hold["item"],
            "frontier_lane": True,
            "restored_semantic_hold": True,
        }
    # Replaying a durable verdict does not spend another model-call budget.
    # It still acquires the convergence lease before applying or terminalizing.
    claim = store.claim_attempt(
        key,
        "frontier",
        budget=(
            None
            if artifact_decision is not None or authority_error is not None
            else budget
        ),
        now=now,
    )
    if not claim["claimed"]:
        return {
            "status": "deferred",
            "reason": claim["reason"],
            "state": claim.get("item"),
            "frontier_lane": True,
        }
    owner = str(claim["owner"])
    if authority_error is not None or not isinstance(authority, Mapping):
        failed = store.fail_attempt(
            key,
            "frontier",
            owner=owner,
            error=authority_error or "decision authority unavailable",
            failure_class="decision_authority_unavailable",
            now=now,
        )
        return {
            "status": "frontier_error",
            "state": failed["item"],
            "frontier_lane": True,
        }
    if artifact_decision is not None:
        decision = artifact_decision
    else:
        try:
            raw = reviewer(prompt, TAG_REPAIR_SCHEMA)
            decision = _gate_tag_review_to_exact_proposal(raw, local_proposal)
        except Exception as exc:
            failed = store.fail_attempt(
                key,
                "frontier",
                owner=owner,
                error=f"{exc.__class__.__name__}: {exc}",
                failure_class="frontier_call_error",
                now=now,
            )
            return {
                "status": "frontier_error",
                "state": failed["item"],
                "frontier_lane": True,
            }

        if decision["decision"] in {"approved", "rejected"} and decision["valid"]:
            with decision_authority_lock():
                authority_error = _current_review_authority_error(
                    decision,
                    authority,
                    injected_reviewer=injected_reviewer,
                )
                try:
                    if authority_error is not None:
                        raise RuntimeError(authority_error)
                    updated_date = date.today().isoformat()
                    artifact = _write_frontier_review_artifact(
                        store,
                        key,
                        page_text=page_text,
                        prompt=prompt,
                        decision=decision,
                        authority=authority,
                        page_id=str(row.get("page") or ""),
                        updated_date=updated_date,
                    )
                except Exception as exc:
                    failure_class = (
                        "decision_authority_changed"
                        if authority_error is not None
                        else "review_artifact_write_error"
                    )
                    failed = store.fail_attempt(
                        key,
                        "frontier",
                        owner=owner,
                        error=f"{exc.__class__.__name__}: {exc}",
                        failure_class=failure_class,
                        now=now,
                    )
                    return {
                        "status": "frontier_error",
                        "state": failed["item"],
                        "frontier_lane": True,
                    }
            decision = dict(decision)
            decision["authority"] = dict(authority)
            decision["updated_date"] = artifact.get("updated_date")
            decision["postimage_sha256"] = artifact.get("postimage_sha256")

    if decision["decision"] == "approved" and decision["valid"]:
        applied = _apply_reviewed_tags(
            store=store,
            budget=budget,
            key=key,
            stage="frontier",
            owner=owner,
            path=path,
            expected_text=page_text,
            decision=decision,
            authority=authority,
            injected_reviewer=injected_reviewer,
            now=now,
        )
        applied["frontier_lane"] = True
        return applied
    if decision["decision"] == "rejected" and decision["valid"]:
        with decision_authority_lock():
            authority_error = _current_review_authority_error(
                decision,
                authority,
                injected_reviewer=injected_reviewer,
            )
            if authority_error is not None:
                failed = store.fail_attempt(
                    key,
                    "frontier",
                    owner=owner,
                    error=authority_error,
                    failure_class="decision_authority_changed",
                    now=now,
                )
                return {
                    "status": "frontier_retry",
                    "state": failed["item"],
                    "reason": authority_error,
                    "frontier_lane": True,
                }
            completed = store.complete(
                key,
                "rejected",
                owner=owner,
                result={"action": "tag_repair_rejected", "decision": decision},
                now=now,
            )
        return {
            "status": "rejected",
            "state": completed["item"],
            "decision": decision,
            "frontier_lane": True,
        }
    if is_local_semantic_no_quorum(decision):
        try:
            with decision_authority_lock():
                current_authority, current_error = current_semantic_authority(
                    TAG_REPAIR_DECISION_LANE,
                    injected_reviewer=injected_reviewer,
                )
                epoch_error = current_error or compare_semantic_authority(
                    authority,
                    current_authority,
                    lane=TAG_REPAIR_DECISION_LANE,
                )
                if epoch_error is not None or not isinstance(
                    current_authority, Mapping
                ):
                    raise ValueError(epoch_error or "decision authority is unavailable")
                current_item = store.get(key) or {}
                failed = store.hold_semantic_no_quorum(
                    key,
                    lane=TAG_REPAIR_DECISION_LANE,
                    stage="frontier",
                    review=decision,
                    epoch=_tag_semantic_epoch(
                        current_item,
                        page_text=page_text,
                        prompt=prompt,
                    ),
                    authority=current_authority,
                    owner=owner,
                    error=(
                        str(decision.get("reason") or "") or "local semantic no quorum"
                    ),
                    now=now,
                )
        except (TypeError, ValueError) as exc:
            failed = store.fail_attempt(
                key,
                "frontier",
                owner=owner,
                error=f"semantic hold rejected: {exc}",
                failure_class="review_artifact_invalid",
                now=now,
            )
    else:
        failure_class = _failure_class(decision, "frontier_uncertain")
        failed = store.fail_attempt(
            key,
            "frontier",
            owner=owner,
            error=decision.get("reason") or "frontier tag review was uncertain",
            failure_class=failure_class,
            now=now,
        )
    persisted_status = str(failed["item"].get("status") or "")
    status = (
        persisted_status
        if persisted_status in {"human_required", "quarantined"}
        else "frontier_retry"
    )
    return {
        "status": status,
        "state": failed["item"],
        "decision": decision,
        "frontier_lane": True,
    }


def _process_tag_candidate(
    *,
    row: Mapping[str, Any],
    path: Path,
    page_text: str,
    store: ConvergenceStore,
    budget: CycleBudget,
    key: str,
    local_reviewer: StructuredReviewer,
    frontier_reviewer: StructuredReviewer,
    injected_reviewer: bool,
    now: datetime | None,
) -> dict[str, Any]:
    meta, _body = parse_frontmatter(page_text)
    existing_tags = meta.get("tags")
    if valid_tag_set(existing_tags):
        tags = list(existing_tags)
        recovery = _find_exact_applied_recovery(
            store,
            page_id=str(row.get("page") or ""),
            page_text=page_text,
            tags=tags,
        )
        with decision_authority_lock():
            item = _terminal_result(
                store,
                key,
                action=(
                    "exact_already_applied_recovery"
                    if recovery is not None
                    else "already_resolved_observed"
                ),
                result={
                    "tags": tags,
                    "semantic_effect": False,
                    "recovery_only": recovery is not None,
                    **({"review_recovery": recovery} if recovery is not None else {}),
                },
                now=now,
            )
        return {
            "status": (
                "exact_already_applied_recovery"
                if recovery is not None
                else "already_resolved"
            ),
            "state": item,
        }

    # The local claim is the authoritative read for both stages.  A separate
    # ``store.get`` here used to parse the full convergence JSON before the
    # claim parsed it again under its lock.  Asking the local claim first keeps
    # that lock/freshness check in one place; a frontier item is routed using
    # the claim's readback without spending a local attempt.
    claim = store.claim_attempt(key, "local", budget=budget, now=now)
    if not claim["claimed"]:
        next_item = claim.get("item") if isinstance(claim.get("item"), dict) else {}
        if next_item.get("status") in FRONTIER_STATUSES:
            local_proposal = _load_local_proposal_artifact(
                store,
                key,
                page_text=page_text,
            )
            return _run_frontier_tag_review(
                row=row,
                path=path,
                page_text=page_text,
                store=store,
                budget=budget,
                key=key,
                reviewer=frontier_reviewer,
                now=now,
                local_proposal=local_proposal,
                injected_reviewer=injected_reviewer,
            )
        return {"status": "deferred", "reason": claim["reason"], "state": next_item}
    owner = str(claim["owner"])
    prompt = build_tag_repair_prompt(row, page_text)
    try:
        raw = local_reviewer(prompt, TAG_REPAIR_SCHEMA)
        decision = normalize_tag_decision(raw)
    except Exception as exc:
        failed = store.fail_attempt(
            key,
            "local",
            owner=owner,
            error=f"{exc.__class__.__name__}: {exc}",
            failure_class="local_model_error",
            now=now,
        )
        if failed["item"].get("status") in FRONTIER_STATUSES:
            return _run_frontier_tag_review(
                row=row,
                path=path,
                page_text=page_text,
                store=store,
                budget=budget,
                key=key,
                reviewer=frontier_reviewer,
                now=now,
                injected_reviewer=injected_reviewer,
            )
        return {"status": "local_error", "state": failed["item"]}

    if decision["decision"] == "approved" and decision["valid"]:
        try:
            _write_local_proposal_artifact(
                store,
                key,
                page_text=page_text,
                decision=decision,
            )
        except Exception as exc:
            failed = store.fail_attempt(
                key,
                "local",
                owner=owner,
                error=f"{exc.__class__.__name__}: {exc}",
                failure_class="proposal_artifact_write_error",
                now=now,
            )
            return {"status": "local_error", "state": failed["item"]}

    escalated = store.escalate(
        key,
        owner=owner,
        reason=decision.get("reason") or "local tag review was undecidable",
        now=now,
    )
    if escalated["item"].get("status") == "quarantined":
        return {
            "status": "quarantined",
            "state": escalated["item"],
            "decision": decision,
        }
    return _run_frontier_tag_review(
        row=row,
        path=path,
        page_text=page_text,
        store=store,
        budget=budget,
        key=key,
        reviewer=frontier_reviewer,
        now=now,
        local_proposal=(
            decision
            if decision["decision"] == "approved" and decision["valid"]
            else None
        ),
        injected_reviewer=injected_reviewer,
    )


def run_lint_repair(
    *,
    queue_file: Path | None = None,
    store: ConvergenceStore | None = None,
    budget: CycleBudget | None = None,
    max_items: int = 25,
    dry_run: bool = False,
    local_reviewer: StructuredReviewer | None = None,
    frontier_reviewer: StructuredReviewer | None = None,
    eligible_keys: set[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Drain bounded *actionable* work from ``lint-repair-queue.jsonl``.

    Terminal rows remain in the append-only detector queue.  They must not
    consume ``max_items`` forever and starve later candidates.
    """

    path = queue_file or (chronovisor_store.CHRONOVISOR_ROOT / "review" / "lint-repair-queue.jsonl")
    convergence = store or ConvergenceStore()
    cycle_budget = budget or CycleBudget()
    local = local_reviewer or _default_local_reviewer
    frontier = frontier_reviewer or _default_frontier_reviewer
    injected_frontier_reviewer = frontier_reviewer is not None
    rows, invalid_rows = _read_jsonl(path)
    # Clear deterministic observations/routing before spending scarce model
    # budget on semantic tag repair.  The detector emits rows in page order,
    # so a cluster of tag violations near the head used to make hundreds of
    # cheap monitor/review rows wait behind frontier calls.
    rows.sort(
        key=lambda row: (
            0
            if (
                (str(row.get("lane") or "") == "monitor"
                 and str(row.get("issue_type") or "") == "stale")
                or (
                    str(row.get("lane") or "") == "review"
                    and _route_target(str(row.get("issue_type") or ""))
                )
            )
            else 1,
        )
    )
    work_limit = max(0, max_items)
    work_items = 0
    rows_scanned = 0
    results: list[dict[str, Any]] = []
    counts = {
        "processed": 0,
        "applied": 0,
        "rejected": 0,
        "routed": 0,
        "observed": 0,
        "escalated": 0,
        "quarantined": 0,
        "human_required": 0,
        "deferred": 0,
        "terminal_skipped": 0,
        "out_of_scope": 0,
    }

    for row in rows:
        lane = str(row.get("lane") or "")
        issue_type = str(row.get("issue_type") or "unknown")
        page_id = str(row.get("page") or "")
        page_path = chronovisor_store.find_page(page_id) if page_id else None
        try:
            page_text = (
                page_path.read_text(encoding="utf-8") if page_path is not None else None
            )
        except OSError:
            page_text = None
        source_id, input_data = _candidate_identity(row, page_text)
        key = stable_item_key(
            "lint_repair",
            source_id,
            input_data,
            resolver_version=REPAIR_RESOLVER_VERSION,
        )
        if eligible_keys is not None and key not in eligible_keys:
            counts["out_of_scope"] += 1
            continue
        existing = convergence.get(key)
        if (
            existing is not None
            and existing.get("status") == "quarantined"
            and str(existing.get("last_failure_class") or "")
            == LOCAL_SEMANTIC_NO_QUORUM
            and page_text is not None
        ):
            hold = persisted_semantic_no_quorum_hold(
                existing,
                lane=TAG_REPAIR_DECISION_LANE,
            )
            # Incomplete legacy rows have no trustworthy authority epoch and
            # remain fail-closed. A strict common hold can be reopened only
            # after the exact prompt or adopted authority changes.
            if hold is not None:
                local_proposal = _load_local_proposal_artifact(
                    convergence,
                    key,
                    page_text=page_text,
                )
                prompt = build_frontier_tag_repair_prompt(
                    row,
                    page_text,
                    local_proposal=local_proposal,
                )
                with decision_authority_lock():
                    authority, authority_error = current_semantic_authority(
                        TAG_REPAIR_DECISION_LANE,
                        injected_reviewer=injected_frontier_reviewer,
                    )
                    hold_error = authority_error
                    if isinstance(authority, Mapping) and hold_error is None:
                        hold_error = semantic_no_quorum_hold_error(
                            hold,
                            TAG_REPAIR_DECISION_LANE,
                            epoch=_tag_semantic_epoch(
                                existing,
                                page_text=page_text,
                                prompt=prompt,
                            ),
                            authority=authority,
                        )
                    if (
                        hold_error is not None
                        and authority_error is None
                        and isinstance(authority, Mapping)
                    ):
                        if dry_run:
                            rows_scanned += 1
                            results.append(
                                {
                                    "key": key,
                                    "page": page_id,
                                    "issue_type": issue_type,
                                    "status": "would_resume_semantic_hold",
                                    "reason": hold_error,
                                }
                            )
                            continue
                        transition = convergence.resume_quarantined(
                            key,
                            stage="frontier",
                            reason=hold_error,
                            resume_context={
                                "reason": "semantic_hold_epoch_changed",
                                "decision_lane": TAG_REPAIR_DECISION_LANE,
                                "invalidated_semantic_hold": hold,
                                "invalidated_hold_sha256": str(hold["hold_sha256"]),
                                "expected_epoch": _tag_semantic_epoch(
                                    existing,
                                    page_text=page_text,
                                    prompt=prompt,
                                ),
                                "expected_epoch_sha256": canonical_sha256(
                                    _tag_semantic_epoch(
                                        existing,
                                        page_text=page_text,
                                        prompt=prompt,
                                    )
                                ),
                                "expected_authority": dict(authority),
                            },
                            now=now,
                        )
                        existing = transition["item"]
        if existing is not None and existing.get("status") in TERMINAL_STATUSES:
            rows_scanned += 1
            counts["terminal_skipped"] += 1
            results.append(
                {
                    "key": key,
                    "page": page_id,
                    "issue_type": issue_type,
                    "status": "terminal_skipped",
                }
            )
            continue
        if work_items >= work_limit:
            break

        merged = convergence.merge_item(
            lane="lint_repair",
            source_id=source_id,
            input_data=input_data,
            resolver_version=REPAIR_RESOLVER_VERSION,
            metadata=row,
            now=now,
            dry_run=dry_run,
            supersede_eligible_keys=eligible_keys,
        )
        item = merged["item"]
        if item is None:
            rows_scanned += 1
            counts["out_of_scope"] += 1
            results.append(
                {
                    "key": key,
                    "page": page_id,
                    "issue_type": issue_type,
                    "status": "out_of_scope_source_changed",
                    "blocked_by_out_of_scope": merged.get(
                        "blocked_by_out_of_scope", []
                    ),
                }
            )
            continue
        key = str(item["key"])
        rows_scanned += 1

        # A concurrent worker may have terminalized the item between ``get``
        # and ``merge_item``.  Preserve the same non-starving accounting.
        if item.get("status") in TERMINAL_STATUSES:
            counts["terminal_skipped"] += 1
            results.append(
                {
                    "key": key,
                    "page": page_id,
                    "issue_type": issue_type,
                    "status": "terminal_skipped",
                }
            )
            continue

        work_items += 1

        if dry_run:
            if lane == "monitor" and issue_type == "stale":
                action = "would_observe"
            elif lane == "review" and _route_target(issue_type):
                action = "would_route"
            elif lane == "heavy_model_batch" and issue_type in {
                "tag_missing",
                "tag_count_violation",
            }:
                action = "would_review_tags"
            else:
                action = "would_quarantine"
            results.append(
                {
                    "key": key,
                    "page": page_id,
                    "issue_type": issue_type,
                    "status": action,
                }
            )
            continue

        counts["processed"] += 1
        if lane == "monitor" and issue_type == "stale":
            terminal = _terminal_result(
                convergence,
                key,
                action="observed",
                result={"issue_type": issue_type, "page": page_id},
                now=now,
            )
            counts["observed"] += 1
            result = {"status": "observed", "state": terminal}
        elif lane == "review" and (target := _route_target(issue_type)):
            terminal = _terminal_result(
                convergence,
                key,
                action="routed",
                result={
                    "target_lane": target,
                    "issue_type": issue_type,
                    "page": page_id,
                },
                now=now,
            )
            counts["routed"] += 1
            result = {"status": "routed", "target_lane": target, "state": terminal}
        elif lane == "heavy_model_batch" and issue_type in {
            "tag_missing",
            "tag_count_violation",
        }:
            if page_path is None or page_text is None:
                transition = convergence.complete(
                    key,
                    "rejected",
                    result={"action": "page_missing", "page": page_id},
                    now=now,
                )
                result = {
                    "status": "rejected",
                    "reason": "page_missing",
                    "state": transition["item"],
                }
            else:
                result = _process_tag_candidate(
                    row=row,
                    path=page_path,
                    page_text=page_text,
                    store=convergence,
                    budget=cycle_budget,
                    key=key,
                    local_reviewer=local,
                    frontier_reviewer=frontier,
                    injected_reviewer=injected_frontier_reviewer,
                    now=now,
                )
        else:
            transition = convergence.quarantine(
                key,
                reason=f"unsupported_lint_lane:{lane}:{issue_type}",
                now=now,
            )
            result = {"status": "quarantined", "state": transition["item"]}

        status = str(result.get("status") or "")
        state = result.get("state") if isinstance(result.get("state"), Mapping) else {}
        if status == "deferred":
            # A live lease, retry backoff, or exhausted per-cycle call budget
            # performed no repair work.  Keep scanning so such a row cannot
            # monopolize the head of the append-only queue.
            work_items -= 1
            counts["processed"] -= 1
        if status in {
            "applied",
            "unchanged",
            "already_resolved",
            "exact_already_applied_recovery",
        }:
            counts["applied"] += 1
        if status == "rejected":
            counts["rejected"] += 1
        if status in {
            "deferred",
            "local_error",
            "frontier_error",
            "frontier_retry",
            "budget_exhausted",
            "apply_error",
        }:
            counts["deferred"] += 1
        if result.get("frontier_lane"):
            counts["escalated"] += 1
        if state.get("status") == "quarantined":
            counts["quarantined"] += 1
        if state.get("status") == "human_required":
            counts["human_required"] += 1
        results.append(
            {"key": key, "page": page_id, "issue_type": issue_type, **result}
        )

    return {
        "status": "dry_run" if dry_run else "ok",
        "queue_file": str(path),
        "seen": len(rows),
        "invalid_rows": invalid_rows,
        "bounded": work_items,
        "rows_scanned": rows_scanned,
        "remaining_unseen": max(0, len(rows) - rows_scanned),
        **counts,
        "budget": cycle_budget.snapshot(),
        "results": results,
    }


__all__ = [
    "REPAIR_RESOLVER_VERSION",
    "TAG_REPAIR_SCHEMA",
    "apply_tags_cas",
    "build_tag_repair_prompt",
    "normalize_tag_decision",
    "run_lint_repair",
    "valid_tag_set",
]
