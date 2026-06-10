"""Apply safe auto-lane recall improvements from missed candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import tomllib

from llm_wiki_mcp import wiki
from llm_wiki_mcp.alias_store import add_alias
from llm_wiki_mcp.frontmatter import parse as parse_frontmatter
from llm_wiki_mcp.frontmatter import patch as patch_frontmatter
from llm_wiki_mcp.recall_hints import QUERY_HINTS_FILE, add_query_hint
from llm_wiki_mcp.recall_runtime import RECALL_CONFIG_FILE, RECALL_DIR, RECALL_FEEDBACK_FILE, append_jsonl
from llm_wiki_mcp.runtime_config import active_config_file
from llm_wiki_mcp.tags import record_new_tag, validate_tag


AUTO_ACTIONS = frozenset({"alias", "query_hint", "page_tag"})
VALIDATED_AUTO_LANE = "validated-auto"
AUTO_APPLY_LOG_FILE = RECALL_DIR / "auto-apply.jsonl"


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


def read_applied_keys(path: Path | None = None, limit: int = 5000) -> set[str]:
    path = _auto_apply_log_file(path)
    try:
        with path.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=limit)
    except OSError:
        return set()
    keys: set[str] = set()
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("apply_key"), str):
            keys.add(parsed["apply_key"])
    return keys


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
        if record.get("source") != "auditor":
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
    page_id = str(payload.get("page_id") or (pages[0] if pages else ""))
    query = str(payload.get("query") or record.get("prompt") or record.get("missing_signal") or "")
    signal = str(payload.get("signal") or record.get("missing_signal") or "")
    if dry_run:
        return {"action": "query_hint", "status": "dry_run", "page_id": page_id, "query": query}
    hint = add_query_hint(
        page_id=page_id,
        query=query,
        signal=signal,
        source="recall-auto-apply",
        normalize_key=str(record.get("normalize_key", "")),
    )
    return {"action": "query_hint", "status": "applied", "hint": hint}


def apply_alias(record: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    payload = action_payload(record)
    pages = expected_pages(record)
    target = str(payload.get("target_page") or payload.get("page_id") or (pages[0] if pages else ""))
    alias = str(payload.get("alias") or record.get("missing_signal") or "")
    if not alias:
        raise ValueError("alias action requires alias or missing_signal")
    target_ref = _page_ref(target)
    if dry_run:
        return {"action": "alias", "status": "dry_run", "alias": alias, "target": target_ref}
    add_alias(alias, target_ref, source=f"recall-auto-apply:{record.get('normalize_key', '')}")
    return {"action": "alias", "status": "applied", "alias": alias, "target": target_ref}


def apply_page_tag(record: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    payload = action_payload(record)
    pages = expected_pages(record)
    page_id = str(payload.get("page_id") or (pages[0] if pages else ""))
    tag = str(payload.get("tag") or record.get("missing_signal") or "")
    valid, reason = validate_tag(tag)
    if not valid:
        raise ValueError(f"invalid page tag {tag!r}: {reason}")
    path = wiki.find_page(page_id)
    if path is None:
        raise ValueError(f"page does not exist: {page_id!r}")
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
) -> dict[str, Any]:
    if not policy.enabled:
        return {"status": "disabled", "actions": []}
    applied_keys = read_applied_keys(log_file)
    actions: list[dict[str, Any]] = []
    for record in eligible_records(records, policy=policy, applied_keys=applied_keys):
        key = apply_key_for(record)
        try:
            result = apply_record(record, dry_run=dry_run)
            status = result.get("status", "applied")
        except Exception as exc:
            result = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
            status = "error"
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "apply_key": key,
            "normalize_key": record.get("normalize_key", ""),
            "action_type": record.get("action_type", ""),
            "source_ref": record.get("ref", ""),
            "dry_run": dry_run,
            "status": status,
            "result": result,
        }
        actions.append(entry)
        if not dry_run:
            record_apply_log(entry, log_file)
            applied_keys.add(key)
    return {"status": "ok", "actions": actions}


def apply_feedback_file(
    *,
    feedback_file: Path | None = None,
    config_file: Path = RECALL_CONFIG_FILE,
    min_count: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    policy = load_auto_apply_policy(config_file)
    if min_count is not None:
        policy = AutoApplyPolicy(
            enabled=policy.enabled,
            min_count=max(1, min_count),
            actions=policy.actions,
        )
    return apply_feedback_records(read_jsonl(_feedback_file(feedback_file)), policy=policy, dry_run=dry_run)


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
