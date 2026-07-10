"""Lint engine - detect and fix wiki quality issues."""

import difflib
import fcntl
import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping
from collections import Counter
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from llm_wiki_mcp.wiki import SYSTEM_DIR, WIKI_ROOT, all_pages, find_page
from llm_wiki_mcp.index_store import get_store
from llm_wiki_mcp.link_fix import (
    WIKI_LINK_RE,
    atomic_write,
    extract_targets,
    find_fuzzy_match,
    normalize_link_target,
    position_in_spans,
    protected_spans,
)
from llm_wiki_mcp.page_mutation import wiki_mutation_lock
from llm_wiki_mcp.runtime_config import runtime_repo_root
from llm_wiki_mcp.tags import (
    parse_tags,
    validate_axis_counts,
    validate_tag,
)


STALE_DAYS = 90  # Pages not updated in this many days are flagged
REPO_ROOT = runtime_repo_root()

StructuredReviewer = Callable[[str, dict[str, Any]], Mapping[str, Any] | str]

SAFE_FIX_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "summary",
        "tests_run",
        "commit",
        "committed",
        "pushed",
        "risk",
        "notes",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "quarantined", "needs_retry"],
        },
        "summary": {"type": "string"},
        "tests_run": {"type": "array", "items": {"type": "string"}},
        "commit": {"type": ["string", "null"]},
        "committed": {"type": "boolean"},
        "pushed": {"type": "boolean"},
        "risk": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
    },
}

# `wiki_apply` runs `check()` and then re-runs it inside `apply_safe_fixes`,
# so the same issue list is computed twice for an unchanged corpus. Cache
# by corpus version to short-circuit the second call. Cleared whenever the
# fingerprint changes.
_CHECK_CACHE_LOCK = threading.Lock()
_CHECK_CACHE_VERSION: str | None = None
_CHECK_CACHE_RESULT: list[dict] | None = None


def _collect_all_page_ids() -> set[str]:
    """pages/ + system/ の全 page_id を返す。broken_link 判定の母集合。

    Backed by the IndexStore. Refresh is the caller's responsibility.
    """
    return get_store().all_page_ids(include_system=True)


def check() -> list[dict]:
    """Run all lint checks and return a list of issues."""
    global _CHECK_CACHE_VERSION, _CHECK_CACHE_RESULT

    store = get_store()
    store.refresh()
    # Cache key mixes the corpus fingerprint with `date.today()` because
    # the stale-page check both classifies and labels by today's date —
    # without the date component a long-lived server crossing midnight
    # would keep returning yesterday's classifications.
    version = f"{store.corpus_version()}:{date.today().isoformat()}"

    with _CHECK_CACHE_LOCK:
        if _CHECK_CACHE_VERSION == version and _CHECK_CACHE_RESULT is not None:
            # Defensive copy so callers that mutate the list (e.g. filter
            # auto-fixable issues) don't poison the cache for the next call.
            return [dict(i) for i in _CHECK_CACHE_RESULT]

    issues = []

    # System pages are part of the broken_link universe but are not
    # themselves linted (they're treated as a fixed reference set).
    all_page_ids = store.all_page_ids(include_system=True)
    pages_meta = store.all_pages_meta(include_system=False)
    lintable_pages_meta = [m for m in pages_meta if m.get("page_type") != "reference"]
    page_count = len(lintable_pages_meta)

    for meta in lintable_pages_meta:
        page_id = meta["page_id"]

        # 1. Broken links — outlinks come pre-normalized + code-fence-stripped
        #    from the index (same `extract_targets(strip=True)` semantics).
        seen_broken: set[str] = set()
        for target in store.outlinks(page_id):
            if target in all_page_ids or target in seen_broken:
                continue
            seen_broken.add(target)
            issues.append({
                "type": "broken_link",
                "severity": "high",
                "page": page_id,
                "detail": f"Link [[{target}]] points to non-existent page",
                "auto_fixable": True,
            })

        # 2. Stale pages
        updated_str = meta["updated"]
        if updated_str and updated_str != "unknown":
            try:
                updated_date = date.fromisoformat(updated_str)
                if (date.today() - updated_date).days > STALE_DAYS:
                    issues.append({
                        "type": "stale",
                        "severity": "low",
                        "page": page_id,
                        "detail": f"Last updated {updated_str} ({(date.today() - updated_date).days} days ago)",
                        "auto_fixable": False,
                    })
            except ValueError:
                pass

        # 3. Orphan pages (no backlinks)
        if not store.backlinks(page_id) and page_count > 1:
            issues.append({
                "type": "orphan",
                "severity": "medium",
                "page": page_id,
                "detail": "No other pages link to this page",
                "auto_fixable": False,
            })

        # 4. Tag taxonomy enforcement (plan-4)
        page_tags = store.tags(page_id)
        if not page_tags:
            issues.append({
                "type": "tag_missing",
                "severity": "high",
                "page": page_id,
                "detail": (
                    "No tags. Required: 1-3 d/ tags, exactly 1 t/ tag, "
                    "exactly 1 s/ tag"
                ),
                "auto_fixable": False,
            })
        else:
            invalid: list[tuple[str, str]] = []
            for tag in page_tags:
                ok, reason = validate_tag(tag)
                if not ok:
                    invalid.append((tag, reason))
            if invalid:
                preview = ", ".join(f"{t!r} ({r})" for t, r in invalid[:3])
                issues.append({
                    "type": "tag_invalid",
                    "severity": "medium",
                    "page": page_id,
                    "detail": (
                        f"{len(invalid)} invalid tag(s): {preview}"
                        + ("..." if len(invalid) > 3 else "")
                    ),
                    "auto_fixable": True,
                })

            parsed = parse_tags(page_tags)
            count_violations = validate_axis_counts(parsed)
            if count_violations:
                issues.append({
                    "type": "tag_count_violation",
                    "severity": "medium",
                    "page": page_id,
                    "detail": "; ".join(count_violations),
                    "auto_fixable": False,
                })

    # 4. Duplicate detection (pages with very similar titles)
    titles: dict[str, str] = {}
    for meta in lintable_pages_meta:
        title = meta["title"].lower().strip()
        page_id = meta["page_id"]
        if title in titles:
            issues.append({
                "type": "duplicate",
                "severity": "medium",
                "page": page_id,
                "detail": f"Possible duplicate of '{titles[title]}' (same title)",
                "auto_fixable": False,
            })
        else:
            titles[title] = page_id

    # 5. Contradiction detection placeholder
    # Full contradiction detection requires LLM; we check for simple cases
    # like same entity with conflicting facts across linked pages
    # TODO: LLM-based contradiction detection in future iteration

    with _CHECK_CACHE_LOCK:
        _CHECK_CACHE_VERSION = version
        _CHECK_CACHE_RESULT = [dict(i) for i in issues]

    return issues


def summarize_issues(issues: list[dict]) -> dict:
    """Return a compact, MCP-friendly summary for potentially huge lint output."""
    by_type = Counter(str(issue.get("type", "unknown")) for issue in issues)
    by_severity = Counter(str(issue.get("severity", "unknown")) for issue in issues)
    auto_fixable = sum(1 for issue in issues if issue.get("auto_fixable"))
    lanes = {
        "safe_auto_fix": auto_fixable,
        "heavy_model_batch": by_type.get("tag_missing", 0) + by_type.get("tag_count_violation", 0),
        "review": by_type.get("duplicate", 0) + by_type.get("orphan", 0),
        "monitor": by_type.get("stale", 0),
    }
    top_pages = Counter(str(issue.get("page", "")) for issue in issues if issue.get("page"))
    return {
        "total": len(issues),
        "by_type": dict(sorted(by_type.items())),
        "by_severity": dict(sorted(by_severity.items())),
        "lanes": lanes,
        "top_pages": [
            {"page": page, "issues": count}
            for page, count in top_pages.most_common(10)
        ],
    }


def issue_lane(issue: dict) -> str:
    issue_type = issue.get("type")
    if issue.get("auto_fixable"):
        return "safe_auto_fix"
    if issue_type in {"tag_missing", "tag_count_violation"}:
        return "heavy_model_batch"
    if issue_type in {"duplicate", "orphan"}:
        return "review"
    return "monitor"


def repair_queue_records(issues: list[dict]) -> list[dict]:
    records: list[dict] = []
    for issue in issues:
        lane = issue_lane(issue)
        identity = {
            "issue_type": issue.get("type"),
            "page": issue.get("page"),
            "detail": issue.get("detail"),
        }
        issue_key = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        records.append({
            "type": "lint_repair_candidate",
            "issue_key": issue_key,
            "lane": lane,
            "issue_type": issue.get("type"),
            "severity": issue.get("severity"),
            "page": issue.get("page"),
            "detail": issue.get("detail"),
            "auto_fixable": bool(issue.get("auto_fixable")),
        })
    return records


def write_repair_queue(
    issues: list[dict],
    path: Path | None = None,
) -> Path:
    queue_path = path or WIKI_ROOT / "review" / "lint-repair-queue.jsonl"
    records = repair_queue_records(issues)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = queue_path.with_name(f".{queue_path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(queue_path)
    return queue_path


def _broken_link_target(issue: dict) -> str | None:
    """issue["detail"] から target page_id を取り出す。"""
    m = re.search(r"\[\[([^\]]+)\]\]", issue.get("detail", ""))
    if not m:
        return None
    return m.group(1).strip()


def _replace_link_in_content(content: str, target: str, replacement: str | None) -> tuple[str, int]:
    """``[[target]]`` / ``[[target|label]]`` / ``[[target#sec]]`` を置換。

    Args:
        content: 対象ファイルの本文
        target: normalize 済みの target page_id
        replacement: fuzzy match で見つかった置換先 page_id。None か同値なら plaintext 化。

    Returns:
        (new_content, count) — count は置換された箇所数
    """
    skip_ranges = protected_spans(content)
    changed = 0

    def _repl(m: re.Match) -> str:
        nonlocal changed
        if position_in_spans(m.start(), skip_ranges):
            return m.group(0)

        inside = m.group(1)
        if normalize_link_target(inside) != target:
            return m.group(0)

        if replacement and replacement != target:
            changed += 1
            return f"[[{replacement}{_retarget_tail(inside)}]]"
        # plaintext fallback
        changed += 1
        return _display_text_for_unwrap(inside, target)

    new_content = WIKI_LINK_RE.sub(_repl, content)
    return new_content, changed


def _retarget_tail(link_inside: str) -> str:
    """Return the anchor/alias suffix after the normalized target."""
    target_part, sep, alias = link_inside.partition("|")
    if "#" in target_part:
        _target, anchor_body = target_part.split("#", 1)
        anchor = "#" + anchor_body
    else:
        anchor = ""
    return anchor + (sep + alias if sep else "")


def _display_text_for_unwrap(link_inside: str, target: str) -> str:
    """Plaintext replacement for an unresolvable wiki link."""
    _target_part, sep, alias = link_inside.partition("|")
    if sep:
        return alias
    return target


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return _sha256_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _bounded_text(value: str, limit: int = 30_000) -> str:
    if len(value) <= limit:
        return value
    half = max(1, (limit - 35) // 2)
    return value[:half] + "\n[... bounded review payload ...]\n" + value[-half:]


def _safe_fix_artifact_dir(path: Path | None = None) -> Path:
    if path is not None:
        return path
    # Resolve this dynamically: isolated tests and embedded runtimes patch the
    # index-store Wiki root after this module has already been imported.
    from llm_wiki_mcp import index_store

    return Path(index_store.WIKI_ROOT) / "runtime" / "lint-safe-fixes"


def _proposal_artifact_path(artifact_dir: Path, proposal_hash: str) -> Path:
    return artifact_dir / "proposals" / f"{proposal_hash}.json"


def _verdict_artifact_path(artifact_dir: Path, proposal_hash: str) -> Path:
    return artifact_dir / "frontier-verdicts" / f"{proposal_hash}.json"


@contextmanager
def _safe_fix_review_lock(artifact_dir: Path, proposal_hash: str) -> Iterator[None]:
    """Serialize one exact proposal across local processes.

    The lock spans the frontier call and verdict persistence. This avoids two
    workers racing an approval and rejection for the same exact page preimage.
    """

    lock_path = artifact_dir / "locks" / f"{proposal_hash}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _extract_review_object(value: Mapping[str, Any] | str) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
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


def _normalize_safe_fix_review(value: Mapping[str, Any] | str) -> dict[str, Any]:
    parsed = _extract_review_object(value)
    if parsed is None:
        return {
            "decision": "needs_retry",
            "summary": "frontier output did not contain a JSON object",
            "valid": False,
        }
    decision = parsed.get("decision")
    summary = parsed.get("summary")
    tests_run = parsed.get("tests_run")
    commit = parsed.get("commit")
    risk = parsed.get("risk")
    notes = parsed.get("notes")
    valid = (
        decision in {"approved", "rejected", "quarantined", "needs_retry"}
        and isinstance(summary, str)
        and isinstance(tests_run, list)
        and all(isinstance(item, str) for item in tests_run)
        and (commit is None or isinstance(commit, str))
        and isinstance(parsed.get("committed"), bool)
        and isinstance(parsed.get("pushed"), bool)
        and (risk is None or isinstance(risk, str))
        and (notes is None or isinstance(notes, str))
    )
    if not valid:
        return {
            "decision": "needs_retry",
            "summary": "frontier output failed the safe-fix decision schema",
            "valid": False,
        }
    normalized = {
        key: parsed.get(key)
        for key in SAFE_FIX_REVIEW_SCHEMA["required"]
    }
    normalized["valid"] = True
    if isinstance(parsed.get("frontier_failure"), Mapping):
        normalized["frontier_failure"] = dict(parsed["frontier_failure"])
    return normalized


def _build_safe_fix_proposal(
    *,
    page_id: str,
    operation: str,
    expected_text: str,
    updated_text: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    unified_diff = "".join(
        difflib.unified_diff(
            expected_text.splitlines(keepends=True),
            updated_text.splitlines(keepends=True),
            fromfile=f"{page_id}:before",
            tofile=f"{page_id}:after",
            n=5,
        )
    )
    bounded_diff = _bounded_text(unified_diff)
    return {
        "schema_version": 1,
        "kind": "lint_safe_fix_proposal",
        "page_id": page_id,
        "operation": operation,
        "expected_sha256": _sha256_text(expected_text),
        "updated_sha256": _sha256_text(updated_text),
        "details": dict(details),
        "unified_diff": bounded_diff,
        "unified_diff_sha256": _sha256_text(bounded_diff),
        "full_unified_diff_sha256": _sha256_text(unified_diff),
        "unified_diff_truncated": bounded_diff != unified_diff,
    }


def _write_or_validate_proposal(
    artifact_dir: Path,
    proposal: Mapping[str, Any],
) -> str:
    proposal_hash = _canonical_hash(proposal)
    path = _proposal_artifact_path(artifact_dir, proposal_hash)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        existing = None
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read durable safe-fix proposal: {exc}") from exc
    envelope = {
        "schema_version": 1,
        "kind": "lint_safe_fix_proposal_artifact",
        "proposal_sha256": proposal_hash,
        "proposal": dict(proposal),
    }
    if existing is not None:
        if existing != envelope:
            raise RuntimeError("durable safe-fix proposal failed integrity validation")
        return proposal_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return proposal_hash


def _build_safe_fix_prompt(proposal: Mapping[str, Any], *, expected_text: str) -> str:
    return f"""\
You are the final autonomous reviewer for an LLM Wiki semantic page mutation.
A local deterministic checker produced the proposal below, but it has no
authority to change content or metadata. Independently decide whether this
exact mutation is correct. Approve only when every change is justified by the
supplied page context and operation-specific evidence. Do not propose or apply
a different patch. Use needs_retry when the evidence is insufficient.

Exact proposal:
{json.dumps(dict(proposal), ensure_ascii=False, indent=2, sort_keys=True)}

Page preimage (bounded; the exact full hash is in the proposal):
{_bounded_text(expected_text, 24_000)}

Return JSON matching this schema:
{json.dumps(SAFE_FIX_REVIEW_SCHEMA, ensure_ascii=False, indent=2)}
"""


def _default_safe_fix_reviewer(
    prompt: str,
    schema: dict[str, Any],
) -> Mapping[str, Any] | str:
    from llm_wiki_mcp import frontier_review

    return frontier_review.run_structured_review(
        prompt,
        schema,
        repo_root=REPO_ROOT,
        execute_patch=False,
        command_env="LLM_WIKI_LINT_SAFE_FIX_FRONTIER_CMD",
    )


def _load_safe_fix_verdict(
    artifact_dir: Path,
    proposal_hash: str,
    *,
    prompt_hash: str,
) -> dict[str, Any] | None:
    path = _verdict_artifact_path(artifact_dir, proposal_hash)
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(envelope, dict) or any(
        (
            envelope.get("schema_version") != 1,
            envelope.get("kind") != "lint_safe_fix_frontier_verdict",
            envelope.get("proposal_sha256") != proposal_hash,
            envelope.get("prompt_sha256") != prompt_hash,
        )
    ):
        return None
    verdict = envelope.get("verdict")
    if not isinstance(verdict, Mapping):
        return None
    normalized = _normalize_safe_fix_review(verdict)
    if normalized.get("valid") is not True:
        return None
    if normalized.get("decision") not in {"approved", "rejected"}:
        return None
    normalized["reused"] = True
    return normalized


def _write_safe_fix_verdict(
    artifact_dir: Path,
    proposal_hash: str,
    *,
    prompt_hash: str,
    verdict: Mapping[str, Any],
) -> None:
    path = _verdict_artifact_path(artifact_dir, proposal_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": 1,
        "kind": "lint_safe_fix_frontier_verdict",
        "proposal_sha256": proposal_hash,
        "prompt_sha256": prompt_hash,
        "verdict": {
            key: verdict.get(key)
            for key in SAFE_FIX_REVIEW_SCHEMA["required"]
        },
    }
    atomic_write(path, json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _review_safe_fix(
    proposal: Mapping[str, Any],
    *,
    expected_text: str,
    reviewer: StructuredReviewer,
    artifact_dir: Path,
) -> dict[str, Any]:
    proposal_hash = _canonical_hash(proposal)
    prompt = _build_safe_fix_prompt(proposal, expected_text=expected_text)
    prompt_hash = _sha256_text(prompt)
    with _safe_fix_review_lock(artifact_dir, proposal_hash):
        # The exact proposal must be durable before calling the frontier model.
        proposal_hash = _write_or_validate_proposal(artifact_dir, proposal)
        reused = _load_safe_fix_verdict(
            artifact_dir,
            proposal_hash,
            prompt_hash=prompt_hash,
        )
        if reused is not None:
            return reused
        try:
            verdict = _normalize_safe_fix_review(reviewer(prompt, SAFE_FIX_REVIEW_SCHEMA))
        except Exception as exc:
            return {
                "decision": "needs_retry",
                "summary": f"frontier safe-fix review failed: {exc.__class__.__name__}: {exc}",
                "valid": False,
            }
        if verdict.get("valid") is True and verdict.get("decision") in {"approved", "rejected"}:
            try:
                _write_safe_fix_verdict(
                    artifact_dir,
                    proposal_hash,
                    prompt_hash=prompt_hash,
                    verdict=verdict,
                )
            except Exception as exc:
                return {
                    "decision": "needs_retry",
                    "summary": f"durable frontier verdict write failed: {exc}",
                    "valid": False,
                }
        return verdict


def build_semantic_mutation_proposal(
    *,
    page_id: str,
    operation: str,
    expected_text: str,
    updated_text: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact durable proposal shared by deterministic page lanes."""

    return _build_safe_fix_proposal(
        page_id=page_id,
        operation=operation,
        expected_text=expected_text,
        updated_text=updated_text,
        details=details,
    )


def review_semantic_mutation(
    proposal: Mapping[str, Any],
    *,
    expected_text: str,
    reviewer: StructuredReviewer,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Persist and frontier-review one exact semantic mutation proposal."""

    return _review_safe_fix(
        proposal,
        expected_text=expected_text,
        reviewer=reviewer,
        artifact_dir=artifact_dir,
    )


def _atomic_write_if_unchanged(
    path: Path,
    expected: str,
    updated: str,
    *,
    evidence_guards: tuple[tuple[Path, str], ...] = (),
) -> bool:
    """Write a safe fix only while the reviewed page preimage is current."""

    try:
        with wiki_mutation_lock():
            if path.read_text(encoding="utf-8") != expected:
                return False
            if any(
                guard_path.read_text(encoding="utf-8") != guard_text
                for guard_path, guard_text in evidence_guards
            ):
                return False
            atomic_write(path, updated)
            return path.read_text(encoding="utf-8") == updated
    except (OSError, UnicodeDecodeError):
        return False


def apply_safe_fixes(
    issues: list[dict],
    dry_run: bool = False,
    fuzzy: bool = True,
    *,
    reviewer: StructuredReviewer | None = None,
    artifact_dir: Path | None = None,
) -> list[str]:
    """Propose lint fixes and apply only durable frontier approvals.

    Args:
        issues: Issue list from check()
        dry_run: True なら書き込まず actions のプレビューだけ返す
        fuzzy: True なら broken_link を fuzzy match で救い、fallback で plaintext 化する。
               False なら broken_link は放置 (既存の挙動より安全側)。
        reviewer: Structured frontier reviewer override (primarily tests).
        artifact_dir: Durable proposal/verdict directory override.
    """
    store = get_store()
    actions: list[str] = []
    mutated = False
    all_page_ids = store.all_page_ids(include_system=True)
    frontier_reviewer = reviewer or _default_safe_fix_reviewer
    durable_dir = _safe_fix_artifact_dir(artifact_dir)

    for issue in issues:
        if not issue.get("auto_fixable"):
            continue

        if issue["type"] == "broken_link":
            if not fuzzy:
                continue

            page_id = issue["page"]
            path = find_page(page_id)
            if not path:
                continue

            target = _broken_link_target(issue)
            if not target:
                continue

            # system/ 配下に実在するなら書き換え不要 (lint false positive のガード)
            if (SYSTEM_DIR / f"{target}.md").exists():
                continue

            replacement = find_fuzzy_match(target, all_page_ids)
            replacement_guards: tuple[tuple[Path, str], ...] = ()
            replacement_evidence: dict[str, Any] | None = None
            if replacement and replacement != target:
                replacement_path = find_page(replacement)
                if replacement_path is None:
                    system_candidate = SYSTEM_DIR / f"{replacement}.md"
                    replacement_path = system_candidate if system_candidate.is_file() else None
                if replacement_path is None:
                    continue
                try:
                    replacement_text = replacement_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                replacement_guards = ((replacement_path, replacement_text),)
                replacement_evidence = {
                    "page_id": replacement,
                    "sha256": _sha256_text(replacement_text),
                    "excerpt": _bounded_text(replacement_text, 8_000),
                }

            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            new_content, count = _replace_link_in_content(content, target, replacement)
            if count == 0 or new_content == content:
                continue

            if replacement and replacement != target:
                label = f"[{page_id}] [[{target}]] → [[{replacement}]] ({count}x)"
            else:
                label = f"[{page_id}] [[{target}]] → plaintext ({count}x)"

            if dry_run:
                actions.append(f"[dry-run] {label}")
            else:
                proposal = _build_safe_fix_proposal(
                    page_id=page_id,
                    operation=(
                        "broken_link_retarget"
                        if replacement and replacement != target
                        else "broken_link_plaintext"
                    ),
                    expected_text=content,
                    updated_text=new_content,
                    details={
                        "target": target,
                        "replacement": replacement,
                        "occurrences": count,
                        "replacement_evidence": replacement_evidence,
                    },
                )
                try:
                    review = _review_safe_fix(
                        proposal,
                        expected_text=content,
                        reviewer=frontier_reviewer,
                        artifact_dir=durable_dir,
                    )
                except Exception as exc:
                    actions.append(f"[frontier-retry] {label}: durable proposal error: {exc}")
                    continue
                if review.get("decision") == "approved" and review.get("valid") is True:
                    if _atomic_write_if_unchanged(
                        path,
                        content,
                        new_content,
                        evidence_guards=replacement_guards,
                    ):
                        mutated = True
                        actions.append(label)
                elif review.get("decision") == "rejected" and review.get("valid") is True:
                    actions.append(f"[frontier-rejected] {label}")
                else:
                    actions.append(f"[frontier-retry] {label}")

        elif issue["type"] == "tag_invalid":
            # Dropping an invalid tag is still a semantic metadata mutation:
            # the deterministic result is only a proposal until the frontier
            # reviewer authorizes this exact page preimage and output hash.
            page_id = issue["page"]
            path = find_page(page_id)
            if not path:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            from llm_wiki_mcp.frontmatter import (
                parse as _frontmatter_parse,
                patch as _frontmatter_patch,
            )
            meta, _ = _frontmatter_parse(content)
            tags_raw = meta.get("tags")
            if not isinstance(tags_raw, list):
                continue

            kept: list[str] = []
            dropped: list[str] = []
            for t in tags_raw:
                if isinstance(t, str) and validate_tag(t)[0]:
                    kept.append(t)
                else:
                    dropped.append(repr(t))
            if not dropped:
                continue
            new_content = _frontmatter_patch(content, {"tags": kept})
            label = (
                f"[{page_id}] dropped {len(dropped)} invalid tag(s): "
                f"{', '.join(dropped[:3])}"
                + ("..." if len(dropped) > 3 else "")
            )
            if dry_run:
                actions.append(f"[dry-run] {label}")
            else:
                proposal = _build_safe_fix_proposal(
                    page_id=page_id,
                    operation="drop_invalid_tags",
                    expected_text=content,
                    updated_text=new_content,
                    details={"kept_tags": kept, "dropped_tags": dropped},
                )
                try:
                    review = _review_safe_fix(
                        proposal,
                        expected_text=content,
                        reviewer=frontier_reviewer,
                        artifact_dir=durable_dir,
                    )
                except Exception as exc:
                    actions.append(f"[frontier-retry] {label}: durable proposal error: {exc}")
                    continue
                if review.get("decision") == "approved" and review.get("valid") is True:
                    if _atomic_write_if_unchanged(path, content, new_content):
                        mutated = True
                        actions.append(label)
                elif review.get("decision") == "rejected" and review.get("valid") is True:
                    actions.append(f"[frontier-rejected] {label}")
                else:
                    actions.append(f"[frontier-retry] {label}")

    # If we mutated pages, the index is now stale — refresh once at the end
    # so subsequent reads see consistent backlinks/outlinks.
    if mutated and not dry_run:
        store.refresh()

    return actions
