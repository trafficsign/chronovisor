"""Autonomous consumer for non-trivial LLM Wiki lint repair lanes.

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

from llm_wiki_mcp import frontier_review, ollama, wiki
from llm_wiki_mcp.convergence import (
    CycleBudget,
    ConvergenceStore,
    FRONTIER_STATUSES,
    TERMINAL_STATUSES,
    is_human_required_result,
    stable_item_key,
)
from llm_wiki_mcp.frontmatter import parse as parse_frontmatter
from llm_wiki_mcp.frontmatter import patch as patch_frontmatter
from llm_wiki_mcp.link_fix import atomic_write
from llm_wiki_mcp.page_mutation import wiki_mutation_lock
from llm_wiki_mcp.runtime_config import runtime_repo_root
from llm_wiki_mcp.tags import SEED_TAGS, parse_tags, validate_axis_counts, validate_tag


REPAIR_RESOLVER_VERSION = "lint-repair-v1"
REPO_ROOT = runtime_repo_root()

TAG_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "tags", "reason"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "uncertain", "needs_retry"],
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 5,
        },
        "reason": {"type": "string"},
    },
}

LOCAL_TAG_SYSTEM = """\
You repair LLM Wiki page tags. Return JSON only, matching the supplied schema.
Choose 1-3 domain tags (d/), exactly one type tag (t/), and exactly one scope
tag (s/). Tags must be lowercase ASCII kebab-case and contain at most two words
after the prefix. Prefer existing seed tags when they fit. Use uncertain when
the page excerpt is insufficient; never invent facts to justify a tag.
"""

StructuredReviewer = Callable[[str, dict[str, Any]], Mapping[str, Any] | str]


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
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
    if not isinstance(tags_value, list) or not all(isinstance(tag, str) for tag in tags_value):
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
    if "human_required" in parsed or "frontier_failure" in parsed:
        out["human_required"] = is_human_required_result(parsed)
    return out


def valid_tag_set(tags: object) -> bool:
    if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) for tag in tags):
        return False
    if len(tags) != len(set(tags)) or len(tags) > 5:
        return False
    return all(validate_tag(tag)[0] for tag in tags) and not validate_axis_counts(parse_tags(tags))


def _page_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _page_excerpt(text: str, *, limit: int = 6000) -> str:
    meta, body = parse_frontmatter(text)
    header = {
        key: meta.get(key)
        for key in ("title", "summary", "updated", "page_type", "sensitivity", "tags")
        if key in meta
    }
    return json.dumps(header, ensure_ascii=False, indent=2) + "\n\n" + body[:limit]


def build_tag_repair_prompt(row: Mapping[str, Any], page_text: str) -> str:
    seed_tags = [tag for values in SEED_TAGS.values() for tag in values]
    return f"""\
Repair the tag set for this LLM Wiki page.

Issue:
{json.dumps(dict(row), ensure_ascii=False, indent=2, default=str)}

Seed tags (prefer when semantically correct):
{json.dumps(seed_tags, ensure_ascii=False)}

Page excerpt:
{_page_excerpt(page_text)}

Return JSON matching this schema:
{json.dumps(TAG_REPAIR_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_frontier_tag_repair_prompt(
    row: Mapping[str, Any],
    page_text: str,
    *,
    local_proposal: Mapping[str, Any] | None = None,
) -> str:
    """Build the authoritative review prompt with the local output as a proposal only."""

    seed_tags = [tag for values in SEED_TAGS.values() for tag in values]
    proposed = dict(local_proposal) if isinstance(local_proposal, Mapping) else None
    return f"""\
You are the final frontier reviewer for an LLM Wiki tag mutation. The local
review below is an untrusted proposal only. Independently verify it against the
page excerpt. You may approve different tags, reject the mutation, or request a
retry. No page mutation is allowed unless your exact verdict is durably saved.

Issue:
{json.dumps(dict(row), ensure_ascii=False, indent=2, default=str)}

Local proposal (may be null, malformed, or wrong):
{json.dumps(proposed, ensure_ascii=False, indent=2, default=str)}

Seed tags (prefer when semantically correct):
{json.dumps(seed_tags, ensure_ascii=False)}

Page excerpt:
{_page_excerpt(page_text)}

Return JSON matching this schema:
{json.dumps(TAG_REPAIR_SCHEMA, ensure_ascii=False, indent=2)}
"""


def _default_local_reviewer(prompt: str, schema: dict[str, Any]) -> Mapping[str, Any] | str:
    return ollama.generate(prompt, system=LOCAL_TAG_SYSTEM, format=schema)


def _default_frontier_reviewer(prompt: str, schema: dict[str, Any]) -> Mapping[str, Any] | str:
    return frontier_review.run_structured_review(
        prompt,
        schema,
        repo_root=REPO_ROOT,
        execute_patch=False,
        command_env="LLM_WIKI_LINT_REPAIR_FRONTIER_CMD",
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
    """Persist only schema fields; derived validation fields are recomputed on read."""

    return {
        "decision": decision.get("decision"),
        "tags": list(decision.get("tags") or []),
        "reason": decision.get("reason"),
    }


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
) -> None:
    _write_json_artifact(
        _review_artifact_path(store, key),
        {
            "schema_version": 1,
            "kind": "lint_tag_frontier_verdict",
            "key": key,
            "page_sha256": _page_hash(page_text),
            "prompt_sha256": _page_hash(prompt),
            "verdict": _decision_payload(decision),
        },
    )


def _load_frontier_review_artifact(
    store: ConvergenceStore,
    key: str,
    *,
    page_text: str,
    prompt: str,
) -> dict[str, Any] | None:
    artifact = _load_json_artifact(_review_artifact_path(store, key))
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != 1
        or artifact.get("kind") != "lint_tag_frontier_verdict"
        or artifact.get("key") != key
        or artifact.get("page_sha256") != _page_hash(page_text)
        or artifact.get("prompt_sha256") != _page_hash(prompt)
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
    return normalized


def apply_tags_cas(
    path: Path,
    *,
    expected_text: str,
    tags: list[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Patch tags only if the page still matches the reviewed preimage."""

    if not valid_tag_set(tags):
        return {"status": "invalid", "reason": "candidate tags failed taxonomy validation"}
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
        patched = patch_frontmatter(
            current,
            {"tags": tags, "updated": date.today().isoformat()},
        )
    except Exception as exc:
        return {"status": "error", "reason": f"frontmatter_error: {exc}"}
    if patched == current:
        return {"status": "unchanged", "path": str(path), "tags": tags}
    if dry_run:
        return {"status": "would_apply", "path": str(path), "tags": tags}
    try:
        # Serialize all cooperating page writers, then perform the final CAS
        # immediately next to the replace.  A correction that landed after
        # review therefore wins instead of being overwritten by stale tags.
        with wiki_mutation_lock():
            if path.read_text(encoding="utf-8") != expected_text:
                return {"status": "cas_conflict", "reason": "page changed before atomic write"}
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
    transition = store.complete(key, "applied", result={"action": action, **dict(result)}, now=now)
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
        return {"status": "budget_exhausted", "state": failed["item"], "reason": budget_reason}
    applied = apply_tags_cas(path, expected_text=expected_text, tags=list(decision["tags"]))
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
        return {"status": applied["status"], "state": completed["item"], "apply": applied}
    if applied["status"] == "cas_conflict":
        quarantined = store.quarantine(
            key,
            owner=owner,
            reason="tag_repair_cas_conflict",
            now=now,
        )
        return {"status": "quarantined", "state": quarantined["item"], "apply": applied}
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
) -> dict[str, Any]:
    prompt = build_frontier_tag_repair_prompt(
        row,
        page_text,
        local_proposal=local_proposal,
    )
    artifact_decision = _load_frontier_review_artifact(
        store,
        key,
        page_text=page_text,
        prompt=prompt,
    )
    # Replaying a durable verdict does not spend another model-call budget.
    # It still acquires the convergence lease before applying or terminalizing.
    claim = store.claim_attempt(
        key,
        "frontier",
        budget=None if artifact_decision is not None else budget,
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
    if artifact_decision is not None:
        decision = artifact_decision
    else:
        try:
            raw = reviewer(prompt, TAG_REPAIR_SCHEMA)
            decision = normalize_tag_decision(raw)
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
            try:
                _write_frontier_review_artifact(
                    store,
                    key,
                    page_text=page_text,
                    prompt=prompt,
                    decision=decision,
                )
            except Exception as exc:
                failed = store.fail_attempt(
                    key,
                    "frontier",
                    owner=owner,
                    error=f"{exc.__class__.__name__}: {exc}",
                    failure_class="review_artifact_write_error",
                    now=now,
                )
                return {
                    "status": "frontier_error",
                    "state": failed["item"],
                    "frontier_lane": True,
                }

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
            now=now,
        )
        applied["frontier_lane"] = True
        return applied
    if decision["decision"] == "rejected" and decision["valid"]:
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
    now: datetime | None,
) -> dict[str, Any]:
    meta, _body = parse_frontmatter(page_text)
    existing_tags = meta.get("tags")
    if valid_tag_set(existing_tags):
        item = _terminal_result(
            store,
            key,
            action="already_resolved",
            result={"tags": existing_tags},
            now=now,
        )
        return {"status": "already_resolved", "state": item}

    current = store.get(key) or {}
    status = str(current.get("status") or "")
    if status in FRONTIER_STATUSES:
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
        )

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
        return {"status": "quarantined", "state": escalated["item"], "decision": decision}
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
    now: datetime | None = None,
) -> dict[str, Any]:
    """Drain bounded *actionable* work from ``lint-repair-queue.jsonl``.

    Terminal rows remain in the append-only detector queue.  They must not
    consume ``max_items`` forever and starve later candidates.
    """

    path = queue_file or (wiki.WIKI_ROOT / "review" / "lint-repair-queue.jsonl")
    convergence = store or ConvergenceStore()
    cycle_budget = budget or CycleBudget()
    local = local_reviewer or _default_local_reviewer
    frontier = frontier_reviewer or _default_frontier_reviewer
    rows, invalid_rows = _read_jsonl(path)
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
    }

    for row in rows:
        lane = str(row.get("lane") or "")
        issue_type = str(row.get("issue_type") or "unknown")
        page_id = str(row.get("page") or "")
        page_path = wiki.find_page(page_id) if page_id else None
        try:
            page_text = page_path.read_text(encoding="utf-8") if page_path is not None else None
        except OSError:
            page_text = None
        source_id, input_data = _candidate_identity(row, page_text)
        key = stable_item_key(
            "lint_repair",
            source_id,
            input_data,
            resolver_version=REPAIR_RESOLVER_VERSION,
        )
        existing = convergence.get(key)
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
        )
        item = merged["item"]
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
                {"key": key, "page": page_id, "issue_type": issue_type, "status": action}
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
                result={"target_lane": target, "issue_type": issue_type, "page": page_id},
                now=now,
            )
            counts["routed"] += 1
            result = {"status": "routed", "target_lane": target, "state": terminal}
        elif lane == "heavy_model_batch" and issue_type in {"tag_missing", "tag_count_violation"}:
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
        if status in {"applied", "unchanged", "already_resolved"}:
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
        results.append({"key": key, "page": page_id, "issue_type": issue_type, **result})

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
