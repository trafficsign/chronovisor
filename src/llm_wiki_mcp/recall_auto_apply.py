"""Apply safe auto-lane recall improvements from missed candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import tomllib

from llm_wiki_mcp import wiki
from llm_wiki_mcp.alias_store import add_alias
from llm_wiki_mcp.frontmatter import parse as parse_frontmatter
from llm_wiki_mcp.frontmatter import patch as patch_frontmatter
from llm_wiki_mcp.recall_hints import add_query_hint, load_query_hints, normalize_query_text
from llm_wiki_mcp.recall_runtime import RECALL_CONFIG_FILE, RECALL_DIR, RECALL_FEEDBACK_FILE, append_jsonl
from llm_wiki_mcp.runtime_config import active_config_file
from llm_wiki_mcp.tags import record_new_tag, validate_tag


AUTO_ACTIONS = frozenset({"alias", "query_hint", "page_tag"})
REVIEW_ACTIONS = frozenset({"few_shot", "threshold"})
VALIDATED_AUTO_LANE = "validated-auto"
AUTO_APPLY_LOG_FILE = RECALL_DIR / "auto-apply.jsonl"
TERMINAL_SUCCESS_STATUSES = frozenset(
    {"applied", "already_applied", "fallback_applied", "routed_to_recall_lab"}
)
TERMINAL_CONVERGENCE_STATUSES = frozenset({"applied", "rejected", "quarantined", "human_required"})


@dataclass(frozen=True)
class AutoApplyPolicy:
    enabled: bool = True
    min_count: int = 1
    actions: tuple[str, ...] = ("alias", "query_hint", "page_tag")


def load_auto_apply_policy(path: Path = RECALL_CONFIG_FILE) -> AutoApplyPolicy:
    policy = AutoApplyPolicy()
    path = active_config_file(path)
    if not path.exists():
        return policy
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return policy
    section = data.get("auto_apply")
    if not isinstance(section, dict):
        return policy
    values = dict(policy.__dict__)
    if isinstance(section.get("enabled"), bool):
        values["enabled"] = section["enabled"]
    if isinstance(section.get("min_count"), int):
        values["min_count"] = max(1, section["min_count"])
    if isinstance(section.get("actions"), list) and all(isinstance(v, str) for v in section["actions"]):
        values["actions"] = tuple(action for action in section["actions"] if action in AUTO_ACTIONS)
    return AutoApplyPolicy(**values)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _feedback_file(path: Path | None = None) -> Path:
    if path is not None:
        return path
    from llm_wiki_mcp import recall_runtime

    return recall_runtime.RECALL_FEEDBACK_FILE


def _auto_apply_log_file(path: Path | None = None) -> Path:
    return path or AUTO_APPLY_LOG_FILE


def read_applied_keys(path: Path | None = None, limit: int = 0) -> set[str]:
    path = _auto_apply_log_file(path)
    keys: set[str] = set()
    try:
        with path.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=limit) if limit > 0 else f
            for line in lines:
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(parsed, dict)
                    and isinstance(parsed.get("apply_key"), str)
                    and (
                        parsed.get("status") in TERMINAL_SUCCESS_STATUSES
                        or parsed.get("convergence_status") == "applied"
                    )
                ):
                    keys.add(parsed["apply_key"])
    except OSError:
        return set()
    return keys


def read_apply_states(path: Path | None = None, limit: int = 0) -> dict[str, dict[str, Any]]:
    """Return the latest convergence record per apply key."""
    states: dict[str, dict[str, Any]] = {}
    try:
        with _auto_apply_log_file(path).open(encoding="utf-8") as handle:
            lines = deque(handle, maxlen=limit) if limit > 0 else handle
            for line in lines:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                key = record.get("apply_key")
                if isinstance(key, str) and key:
                    states[key] = record
    except OSError:
        return {}
    return states


def _retry_ready(state: dict[str, Any] | None, *, now: datetime) -> bool:
    if not state:
        return True
    convergence_status = str(state.get("convergence_status") or "")
    if convergence_status in TERMINAL_CONVERGENCE_STATUSES:
        return False
    raw = state.get("next_attempt_at")
    if not isinstance(raw, str) or not raw:
        return True
    try:
        return datetime.fromisoformat(raw) <= now
    except ValueError:
        return True


def record_apply_log(record: dict[str, Any], path: Path | None = None) -> None:
    path = _auto_apply_log_file(path)
    append_jsonl(path, record)


def action_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("action_payload")
    return payload if isinstance(payload, dict) else {}


def expected_pages(record: dict[str, Any]) -> list[str]:
    pages = record.get("expected_pages")
    if not isinstance(pages, list):
        return []
    return [page for page in pages if isinstance(page, str) and page]


def apply_key_for(record: dict[str, Any]) -> str:
    action = str(record.get("action_type", ""))
    normalize_key = str(record.get("normalize_key", ""))
    page = (expected_pages(record) or [""])[0]
    payload = action_payload(record)
    payload_key = payload.get("alias") or payload.get("tag") or payload.get("query") or record.get("missing_signal") or ""
    return f"{action}:{normalize_key}:{page}:{payload_key}"


def eligible_records(
    records: list[dict[str, Any]],
    *,
    policy: AutoApplyPolicy,
    applied_keys: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    allowed_actions = set(policy.actions) & AUTO_ACTIONS
    for record in records:
        if record.get("kind") != "missed_candidate":
            continue
        if record.get("source") not in {"auditor", "pull-log"}:
            continue
        action = record.get("action_type")
        if action not in allowed_actions:
            continue
        if record.get("lane") != "auto" or record.get("auto_apply_eligible") is not True:
            continue
        if not record.get("normalize_key"):
            continue
        key = apply_key_for(record)
        if key in applied_keys:
            continue
        grouped[(str(action), str(record["normalize_key"]))].append(record)

    out: list[dict[str, Any]] = []
    for _group_key, group in grouped.items():
        if len(group) >= policy.min_count:
            out.append(group[-1])
    return out


def _page_ref(page_id: str) -> str:
    path = wiki.find_page(page_id)
    if path is None:
        raise ValueError(f"page does not exist: {page_id!r}")
    try:
        return str(path.relative_to(wiki.PAGES_DIR).with_suffix(""))
    except ValueError:
        return path.stem


def apply_query_hint(record: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    payload = action_payload(record)
    pages = expected_pages(record)
    page_id = str(payload.get("page_id") or (pages[0] if pages else "")).strip()
    query = str(payload.get("query") or record.get("prompt") or record.get("missing_signal") or "").strip()
    signal = str(payload.get("signal") or record.get("missing_signal") or "")
    if not page_id:
        return {
            "action": "query_hint",
            "status": "skipped",
            "reason": "query_hint missing page_id",
            "query": query,
        }
    if not query:
        return {
            "action": "query_hint",
            "status": "skipped",
            "reason": "query_hint missing query",
            "page_id": page_id,
        }
    query_key = normalize_query_text(query)
    for existing in load_query_hints():
        existing_key = str(
            existing.get("query_key")
            or normalize_query_text(str(existing.get("query") or ""))
        )
        if str(existing.get("page_id") or "") == page_id and existing_key == query_key:
            return {
                "action": "query_hint",
                "status": "already_applied",
                "page_id": page_id,
                "query": query,
                "hint": existing,
            }
    if dry_run:
        return {"action": "query_hint", "status": "dry_run", "page_id": page_id, "query": query}
    hint = add_query_hint(
        page_id=page_id,
        query=query,
        signal=signal,
        source="recall-auto-apply",
        normalize_key=str(record.get("normalize_key", "")),
        increment_existing=False,
    )
    return {"action": "query_hint", "status": "applied", "hint": hint}


def valid_alias_candidate(value: str) -> bool:
    text = value.strip()
    if text.endswith(".md"):
        text = text[:-3]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+", text))


def fallback_to_query_hint(
    record: dict[str, Any],
    *,
    dry_run: bool,
    reason: str,
) -> dict[str, Any]:
    try:
        result = apply_query_hint(record, dry_run=dry_run)
    except Exception as exc:
        return {
            "action": str(record.get("action_type", "")),
            "status": "skipped",
            "fallback_to": "query_hint",
            "reason": reason,
            "fallback_error": f"{exc.__class__.__name__}: {exc}",
        }
    if result.get("status") == "skipped":
        return {
            "action": str(record.get("action_type", "")),
            "status": "skipped",
            "fallback_to": "query_hint",
            "reason": reason,
            "result": result,
        }
    return {
        "action": str(record.get("action_type", "")),
        "status": "fallback_dry_run" if dry_run else "fallback_applied",
        "fallback_to": "query_hint",
        "reason": reason,
        "result": result,
    }


def apply_alias(record: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    payload = action_payload(record)
    pages = expected_pages(record)
    target = str(payload.get("target_page") or payload.get("page_id") or (pages[0] if pages else ""))
    raw_alias = payload.get("alias") or record.get("missing_signal") or ""
    if not isinstance(raw_alias, str):
        return fallback_to_query_hint(
            record,
            dry_run=dry_run,
            reason=f"alias payload is not a string: {type(raw_alias).__name__}",
        )
    alias = raw_alias.strip()
    if not alias:
        raise ValueError("alias action requires alias or missing_signal")
    if not valid_alias_candidate(alias):
        return fallback_to_query_hint(
            record,
            dry_run=dry_run,
            reason=f"invalid alias page_id: {alias!r}",
        )
    try:
        target_ref = _page_ref(target)
    except ValueError as exc:
        return fallback_to_query_hint(
            record,
            dry_run=dry_run,
            reason=str(exc),
        )
    if dry_run:
        return {"action": "alias", "status": "dry_run", "alias": alias, "target": target_ref}
    try:
        add_alias(alias, target_ref, source=f"recall-auto-apply:{record.get('normalize_key', '')}")
    except ValueError as exc:
        if "invalid alias page_id" in str(exc):
            return fallback_to_query_hint(
                record,
                dry_run=dry_run,
                reason=str(exc),
            )
        raise
    return {"action": "alias", "status": "applied", "alias": alias, "target": target_ref}


def apply_page_tag(record: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    payload = action_payload(record)
    pages = expected_pages(record)
    page_id = str(payload.get("page_id") or (pages[0] if pages else ""))
    path = wiki.find_page(page_id)
    if path is None:
        return fallback_to_query_hint(
            record,
            dry_run=dry_run,
            reason=f"page_tag target page does not exist: {page_id!r}",
        )
    raw_tag = payload.get("tag") or record.get("missing_signal") or ""
    if not isinstance(raw_tag, str):
        return fallback_to_query_hint(
            record,
            dry_run=dry_run,
            reason=f"page_tag payload is not a string: {type(raw_tag).__name__}",
        )
    tag = raw_tag.strip()
    valid, reason = validate_tag(tag)
    if not valid:
        return fallback_to_query_hint(
            record,
            dry_run=dry_run,
            reason=f"invalid page tag {tag!r}: {reason}",
        )
    text = path.read_text(encoding="utf-8")
    meta, _body = parse_frontmatter(text)
    existing = meta.get("tags")
    tags = list(existing) if isinstance(existing, list) else []
    if tag in tags:
        return {"action": "page_tag", "status": "already_applied", "page_id": page_id, "tag": tag}
    new_tags = tags + [tag]
    if dry_run:
        return {"action": "page_tag", "status": "dry_run", "page_id": page_id, "tag": tag}
    path.write_text(
        patch_frontmatter(text, {"tags": new_tags, "updated": date.today().isoformat()}),
        encoding="utf-8",
    )
    record_new_tag(tag, reason=f"recall auto-apply {record.get('normalize_key', '')}")
    return {"action": "page_tag", "status": "applied", "page_id": page_id, "tag": tag}


def apply_record(record: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    action = record.get("action_type")
    if action == "query_hint":
        return apply_query_hint(record, dry_run=dry_run)
    if action == "alias":
        return apply_alias(record, dry_run=dry_run)
    if action == "page_tag":
        return apply_page_tag(record, dry_run=dry_run)
    raise ValueError(f"unsupported auto action: {action!r}")


def apply_feedback_records(
    records: list[dict[str, Any]],
    *,
    policy: AutoApplyPolicy,
    dry_run: bool = False,
    log_file: Path | None = None,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    budget: Any | None = None,
) -> dict[str, Any]:
    if not policy.enabled:
        return {"status": "disabled", "actions": []}
    applied_keys = read_applied_keys(log_file)
    states = read_apply_states(log_file)
    actions: list[dict[str, Any]] = []
    now = datetime.now()
    for record in eligible_records(records, policy=policy, applied_keys=applied_keys):
        key = apply_key_for(record)
        prior = states.get(key)
        if not _retry_ready(prior, now=now):
            continue
        if budget is not None and not dry_run:
            allowed, reason = budget.consume("mutation")
            if not allowed:
                actions.append(
                    {
                        "ts": now.isoformat(timespec="seconds"),
                        "apply_key": key,
                        "normalize_key": record.get("normalize_key", ""),
                        "action_type": record.get("action_type", ""),
                        "source_ref": record.get("ref", ""),
                        "dry_run": False,
                        "status": "budget_deferred",
                        "convergence_status": str(
                            (prior or {}).get("convergence_status") or "pending"
                        ),
                        "attempt": int((prior or {}).get("attempt") or 0),
                        "reason": reason,
                    }
                )
                continue
        attempt = int((prior or {}).get("attempt") or 0) + 1
        try:
            result = apply_record(record, dry_run=dry_run)
            status = result.get("status", "applied")
        except Exception as exc:
            result = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
            status = "error"
        convergence_status = "applied" if status in TERMINAL_SUCCESS_STATUSES else "retry_wait"
        next_attempt_at: str | None = None
        if convergence_status == "retry_wait":
            if attempt >= max(1, max_attempts):
                convergence_status = "quarantined"
            else:
                delay = max(0, backoff_base_seconds) * (2 ** max(0, attempt - 1))
                next_attempt_at = (now + timedelta(seconds=delay)).isoformat(timespec="seconds")
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "apply_key": key,
            "normalize_key": record.get("normalize_key", ""),
            "action_type": record.get("action_type", ""),
            "source_ref": record.get("ref", ""),
            "dry_run": dry_run,
            "status": status,
            "convergence_status": convergence_status,
            "attempt": attempt,
            "next_attempt_at": next_attempt_at,
            "result": result,
        }
        actions.append(entry)
        if not dry_run:
            record_apply_log(entry, log_file)
            states[key] = entry
            if convergence_status == "applied":
                applied_keys.add(key)
    if not dry_run:
        errors = [action for action in actions if action.get("status") == "error"]
        if errors:
            try:
                from llm_wiki_mcp.auto_apply_error_supervisor import supervise_error_records

                supervisor = supervise_error_records(errors)
            except Exception as exc:
                supervisor = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
            return {
                "status": (
                    "budget_deferred"
                    if any(action.get("status") == "budget_deferred" for action in actions)
                    else "ok"
                ),
                "actions": actions,
                "auto_apply_self_heal": supervisor,
            }
    return {
        "status": (
            "budget_deferred"
            if any(action.get("status") == "budget_deferred" for action in actions)
            else "ok"
        ),
        "actions": actions,
    }


def apply_review_feedback_records(
    records: list[dict[str, Any]],
    *,
    dry_run: bool = False,
    log_file: Path | None = None,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    budget: Any | None = None,
) -> dict[str, Any]:
    """Close auditor review actions without creating a human queue.

    ``few_shot`` is materialized as the already-safe query-hint primitive;
    the same feedback is also picked up by the frontier-reviewed search label
    queue. ``threshold`` is routed into Recall Lab's replay/adoption loop and
    is never applied directly.
    """
    candidates_by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        if (
            record.get("kind") != "missed_candidate"
            or record.get("source") != "auditor"
            or record.get("action_type") not in REVIEW_ACTIONS
            or record.get("lane") != "review"
        ):
            continue
        key = apply_key_for(record)
        candidates_by_key[key] = record

    states = read_apply_states(log_file)
    now = datetime.now()
    actions: list[dict[str, Any]] = []
    for record in candidates_by_key.values():
        key = apply_key_for(record)
        prior = states.get(key)
        if not _retry_ready(prior, now=now):
            continue
        if budget is not None and not dry_run:
            allowed, reason = budget.consume("mutation")
            if not allowed:
                actions.append(
                    {
                        "ts": now.isoformat(timespec="seconds"),
                        "apply_key": key,
                        "normalize_key": record.get("normalize_key", ""),
                        "action_type": record.get("action_type", ""),
                        "source_ref": record.get("ref", ""),
                        "dry_run": False,
                        "status": "budget_deferred",
                        "convergence_status": str(
                            (prior or {}).get("convergence_status") or "pending"
                        ),
                        "attempt": int((prior or {}).get("attempt") or 0),
                        "reason": reason,
                    }
                )
                continue
        attempt = int((prior or {}).get("attempt") or 0) + 1
        action = str(record.get("action_type") or "")
        if action == "threshold":
            result = {
                "action": action,
                "status": "routed_to_recall_lab",
                "reason": "threshold changes are replay-gated by recall_improvement",
            }
        else:
            converted = {
                **record,
                "action_type": "query_hint",
                "action_payload": {
                    **action_payload(record),
                    "query": action_payload(record).get("query")
                    or record.get("prompt")
                    or record.get("missing_signal"),
                },
            }
            try:
                result = apply_query_hint(converted, dry_run=dry_run)
            except Exception as exc:
                result = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
        status = str(result.get("status") or "error")
        convergence_status = "applied" if status in TERMINAL_SUCCESS_STATUSES or status == "dry_run" else "retry_wait"
        next_attempt_at: str | None = None
        if convergence_status == "retry_wait":
            if attempt >= max(1, max_attempts):
                convergence_status = "quarantined"
            else:
                delay = max(0, backoff_base_seconds) * (2 ** max(0, attempt - 1))
                next_attempt_at = (now + timedelta(seconds=delay)).isoformat(timespec="seconds")
        entry = {
            "ts": now.isoformat(timespec="seconds"),
            "apply_key": key,
            "normalize_key": record.get("normalize_key", ""),
            "action_type": action,
            "source_ref": record.get("ref", ""),
            "dry_run": dry_run,
            "status": status,
            "convergence_status": convergence_status,
            "attempt": attempt,
            "next_attempt_at": next_attempt_at,
            "result": result,
        }
        actions.append(entry)
        if not dry_run:
            record_apply_log(entry, log_file)
            states[key] = entry
    return {
        "status": (
            "budget_deferred"
            if any(action.get("status") == "budget_deferred" for action in actions)
            else "ok"
        ),
        "actions": actions,
    }


def apply_feedback_file(
    *,
    feedback_file: Path | None = None,
    config_file: Path = RECALL_CONFIG_FILE,
    min_count: int | None = None,
    dry_run: bool = False,
    budget: Any | None = None,
) -> dict[str, Any]:
    policy = load_auto_apply_policy(config_file)
    if min_count is not None:
        policy = AutoApplyPolicy(
            enabled=policy.enabled,
            min_count=max(1, min_count),
            actions=policy.actions,
        )
    records = read_jsonl(_feedback_file(feedback_file))
    auto = apply_feedback_records(records, policy=policy, dry_run=dry_run, budget=budget)
    review = apply_review_feedback_records(records, dry_run=dry_run, budget=budget)
    actions = [*(auto.get("actions") or []), *(review.get("actions") or [])]
    status = "ok"
    if auto.get("status") == "disabled":
        status = "disabled"
    elif any(action.get("status") == "budget_deferred" for action in actions):
        status = "budget_deferred"
    return {
        "status": status,
        "actions": actions,
        "auto": auto,
        "review": review,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply safe recall missed-candidate improvements.")
    parser.add_argument("--feedback-file", default=str(RECALL_FEEDBACK_FILE))
    parser.add_argument("--config", default=str(RECALL_CONFIG_FILE))
    parser.add_argument("--min-count", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = apply_feedback_file(
        feedback_file=Path(args.feedback_file).expanduser(),
        config_file=Path(args.config).expanduser(),
        min_count=args.min_count,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
