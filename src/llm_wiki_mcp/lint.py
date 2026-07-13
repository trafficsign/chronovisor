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

from llm_wiki_mcp.decision_authority import (
    compare_semantic_authority,
    current_semantic_authority,
    seal_semantic_artifact,
    semantic_verdict_authority_error,
)
from llm_wiki_mcp.wiki import SYSTEM_DIR, WIKI_ROOT, find_page
from llm_wiki_mcp.index_store import get_store
from llm_wiki_mcp.link_fix import (
    WIKI_LINK_RE,
    atomic_write,
    find_fuzzy_match,
    normalize_link_target,
    position_in_spans,
    protected_spans,
)
from llm_wiki_mcp.page_mutation import decision_authority_lock, wiki_mutation_lock
from llm_wiki_mcp.runtime_config import runtime_repo_root
from llm_wiki_mcp.tags import (
    parse_tags,
    validate_axis_counts,
    validate_tag,
)


STALE_DAYS = 90  # Pages not updated in this many days are flagged
REPO_ROOT = runtime_repo_root()

# The packet limit is measured from the exact pretty-printed JSON sent to the
# model, not tokens. 106k characters leaves room for the trusted rubric and
# response schema inside the largest 112k production bucket while allowing
# most pages to be reviewed with complete pre/post bytes.
SAFE_FIX_REVIEW_PACKET_MAX_CHARS = 106_000
SAFE_FIX_REPACKET_CONTEXT_LINES = 4

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
            issues.append(
                {
                    "type": "broken_link",
                    "severity": "high",
                    "page": page_id,
                    "detail": f"Link [[{target}]] points to non-existent page",
                    "auto_fixable": True,
                }
            )

        # 2. Stale pages
        updated_str = meta["updated"]
        if updated_str and updated_str != "unknown":
            try:
                updated_date = date.fromisoformat(updated_str)
                if (date.today() - updated_date).days > STALE_DAYS:
                    issues.append(
                        {
                            "type": "stale",
                            "severity": "low",
                            "page": page_id,
                            "detail": f"Last updated {updated_str} ({(date.today() - updated_date).days} days ago)",
                            "auto_fixable": False,
                        }
                    )
            except ValueError:
                pass

        # 3. Orphan pages (no backlinks)
        if not store.backlinks(page_id) and page_count > 1:
            issues.append(
                {
                    "type": "orphan",
                    "severity": "medium",
                    "page": page_id,
                    "detail": "No other pages link to this page",
                    "auto_fixable": False,
                }
            )

        # 4. Tag taxonomy enforcement (plan-4)
        page_tags = store.tags(page_id)
        if not page_tags:
            issues.append(
                {
                    "type": "tag_missing",
                    "severity": "high",
                    "page": page_id,
                    "detail": (
                        "No tags. Required: 1-3 d/ tags, exactly 1 t/ tag, "
                        "exactly 1 s/ tag"
                    ),
                    "auto_fixable": False,
                }
            )
        else:
            invalid: list[tuple[str, str]] = []
            for tag in page_tags:
                ok, reason = validate_tag(tag)
                if not ok:
                    invalid.append((tag, reason))
            if invalid:
                preview = ", ".join(f"{t!r} ({r})" for t, r in invalid[:3])
                issues.append(
                    {
                        "type": "tag_invalid",
                        "severity": "medium",
                        "page": page_id,
                        "detail": (
                            f"{len(invalid)} invalid tag(s): {preview}"
                            + ("..." if len(invalid) > 3 else "")
                        ),
                        "auto_fixable": True,
                    }
                )

            parsed = parse_tags(page_tags)
            count_violations = validate_axis_counts(parsed)
            if count_violations:
                issues.append(
                    {
                        "type": "tag_count_violation",
                        "severity": "medium",
                        "page": page_id,
                        "detail": "; ".join(count_violations),
                        "auto_fixable": False,
                    }
                )

    # 4. Duplicate detection (pages with very similar titles)
    titles: dict[str, str] = {}
    for meta in lintable_pages_meta:
        title = meta["title"].lower().strip()
        page_id = meta["page_id"]
        if title in titles:
            issues.append(
                {
                    "type": "duplicate",
                    "severity": "medium",
                    "page": page_id,
                    "detail": f"Possible duplicate of '{titles[title]}' (same title)",
                    "auto_fixable": False,
                }
            )
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
        "heavy_model_batch": by_type.get("tag_missing", 0)
        + by_type.get("tag_count_violation", 0),
        "review": by_type.get("duplicate", 0) + by_type.get("orphan", 0),
        "monitor": by_type.get("stale", 0),
    }
    top_pages = Counter(
        str(issue.get("page", "")) for issue in issues if issue.get("page")
    )
    return {
        "total": len(issues),
        "by_type": dict(sorted(by_type.items())),
        "by_severity": dict(sorted(by_severity.items())),
        "lanes": lanes,
        "top_pages": [
            {"page": page, "issues": count} for page, count in top_pages.most_common(10)
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
        records.append(
            {
                "type": "lint_repair_candidate",
                "issue_key": issue_key,
                "lane": lane,
                "issue_type": issue.get("type"),
                "severity": issue.get("severity"),
                "page": issue.get("page"),
                "detail": issue.get("detail"),
                "auto_fixable": bool(issue.get("auto_fixable")),
            }
        )
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


def _replace_link_in_content(
    content: str, target: str, replacement: str | None
) -> tuple[str, int]:
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
    return _sha256_text(_canonical_json(dict(value)))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _render_review_packet(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _bounded_text(value: str, limit: int = 30_000) -> str:
    if len(value) <= limit:
        return value
    half = max(1, (limit - 35) // 2)
    return value[:half] + "\n[... bounded review payload ...]\n" + value[-half:]


def _review_opcode_manifest(
    expected_lines: list[str],
    updated_lines: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a complete deterministic line-coverage manifest and changed bytes."""

    matcher = difflib.SequenceMatcher(
        None,
        expected_lines,
        updated_lines,
        autojunk=True,
    )
    manifest: list[dict[str, Any]] = []
    changed_spans: list[dict[str, Any]] = []
    for ordinal, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes(), 1):
        expected_segment = "".join(expected_lines[i1:i2])
        updated_segment = "".join(updated_lines[j1:j2])
        entry = {
            "ordinal": ordinal,
            "tag": tag,
            "expected_line_range": [i1, i2],
            "updated_line_range": [j1, j2],
            "expected_chars": len(expected_segment),
            "updated_chars": len(updated_segment),
            "expected_sha256": _sha256_text(expected_segment),
            "updated_sha256": _sha256_text(updated_segment),
        }
        manifest.append(entry)
        if tag == "equal":
            continue
        leading_context = "".join(
            expected_lines[max(0, i1 - SAFE_FIX_REPACKET_CONTEXT_LINES) : i1]
        )
        trailing_context = "".join(
            expected_lines[i2 : i2 + SAFE_FIX_REPACKET_CONTEXT_LINES]
        )
        span = {
            **entry,
            "expected_text": expected_segment,
            "updated_text": updated_segment,
            "leading_context": leading_context,
            "trailing_context": trailing_context,
            "leading_context_sha256": _sha256_text(leading_context),
            "trailing_context_sha256": _sha256_text(trailing_context),
        }
        span["span_sha256"] = _canonical_hash(span)
        changed_spans.append(span)
    return manifest, changed_spans


def _receipt_core_matches(value: Mapping[str, Any]) -> bool:
    receipt_sha256 = value.get("receipt_sha256")
    if not isinstance(receipt_sha256, str):
        return False
    core = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return receipt_sha256 == _canonical_hash(core)


def _valid_target_lookup_receipt(
    value: object,
    *,
    target: str | None = None,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_keys = {
        "schema_version",
        "kind",
        "target",
        "index_snapshot",
        "target_absent",
        "fuzzy_candidates",
        "fuzzy_candidate",
        "no_acceptable_fuzzy_candidate",
        "receipt_sha256",
    }
    snapshot = value.get("index_snapshot")
    candidates = value.get("fuzzy_candidates")
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("kind") != "broken_link_target_lookup_receipt"
        or not isinstance(value.get("target"), str)
        or (target is not None and value.get("target") != target)
        or not isinstance(snapshot, Mapping)
        or set(snapshot) != {"corpus_version", "page_count", "page_ids_sha256"}
        or not isinstance(snapshot.get("corpus_version"), str)
        or not isinstance(snapshot.get("page_count"), int)
        or not isinstance(snapshot.get("page_ids_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", snapshot["page_ids_sha256"]) is None
        or not isinstance(value.get("target_absent"), bool)
        or not isinstance(candidates, list)
        or not all(isinstance(candidate, str) for candidate in candidates)
        or (
            value.get("fuzzy_candidate") is not None
            and not isinstance(value.get("fuzzy_candidate"), str)
        )
        or not isinstance(value.get("no_acceptable_fuzzy_candidate"), bool)
    ):
        return False
    return _receipt_core_matches(value)


def _valid_external_page_evidence(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_keys = {
        "schema_version",
        "kind",
        "page_id",
        "source_chars",
        "source_sha256",
        "text",
        "receipt_sha256",
    }
    text = value.get("text")
    return bool(
        set(value) == expected_keys
        and value.get("schema_version") == 1
        and value.get("kind") == "external_page_review_evidence"
        and isinstance(value.get("page_id"), str)
        and value.get("page_id")
        and isinstance(text, str)
        and value.get("source_chars") == len(text)
        and value.get("source_sha256") == _sha256_text(text)
        and _receipt_core_matches(value)
    )


def _exact_source_span(
    source_text: str,
    *,
    start: int,
    end: int,
) -> dict[str, Any]:
    rendered = source_text[start:end]
    return {
        "char_range": [start, end],
        "text": rendered,
        "text_sha256": _sha256_text(rendered),
        "source_sha256": _sha256_text(source_text),
    }


def _entity_alias_evidence_spans(
    expected_text: str,
    details: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    rows = details.get("alias_evidence")
    if not isinstance(rows, list) or not rows:
        return None, "entity_alias_evidence_missing"
    folded = expected_text.casefold()
    spans: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("entity_id"), str):
            return None, "entity_alias_evidence_invalid"
        aliases = row.get("matched_aliases")
        if not isinstance(aliases, list) or not aliases:
            return None, "entity_alias_match_missing"
        for alias in aliases:
            if not isinstance(alias, str) or not alias:
                return None, "entity_alias_match_invalid"
            offset = folded.find(alias.casefold())
            if offset < 0:
                return None, "entity_alias_match_not_in_preimage"
            start = max(0, offset - 500)
            end = min(len(expected_text), offset + len(alias) + 500)
            spans.append(
                {
                    "entity_id": row["entity_id"],
                    "matched_alias": alias,
                    **_exact_source_span(expected_text, start=start, end=end),
                }
            )
    return {
        "schema_version": 1,
        "kind": "entity_alias_semantic_evidence",
        "source_sha256": _sha256_text(expected_text),
        "spans": spans,
    }, None


def _operation_review_evidence(
    *,
    operation: str | None,
    details: Mapping[str, Any],
    expected_text: str,
    repacket: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    identity = details.get("identity_preflight")
    if identity is not None:
        from llm_wiki_mcp.decision_lane_prompts import (
            validate_identity_preflight_receipt,
        )

        if not validate_identity_preflight_receipt(identity):
            return None, "identity_preflight_invalid"

    if operation in {"broken_link_plaintext", "broken_link_retarget"}:
        target = details.get("target")
        lookup = details.get("target_lookup_receipt")
        if not isinstance(target, str) or not _valid_target_lookup_receipt(
            lookup,
            target=target,
        ):
            return None, "target_lookup_receipt_invalid"
        assert isinstance(lookup, Mapping)
        if lookup.get("target_absent") is not True:
            return None, "broken_link_target_not_absent"
        replacement = details.get("replacement")
        if operation == "broken_link_plaintext":
            if (
                replacement is not None
                or lookup.get("fuzzy_candidate") is not None
                or lookup.get("no_acceptable_fuzzy_candidate") is not True
            ):
                return None, "plaintext_target_lookup_not_exhaustive"
            return {"target_lookup_receipt": dict(lookup)}, None
        replacement_evidence = details.get("replacement_evidence")
        if (
            not isinstance(replacement, str)
            or lookup.get("fuzzy_candidate") != replacement
            or not _valid_external_page_evidence(replacement_evidence)
            or replacement_evidence.get("page_id") != replacement
        ):
            return None, "retarget_replacement_evidence_invalid"
        assert isinstance(replacement_evidence, Mapping)
        return {
            "target_lookup_receipt": dict(lookup),
            "replacement_evidence": dict(replacement_evidence),
        }, None

    if operation == "resolve_nested_frontmatter_conflict":
        conflicts = details.get("conflicts")
        if isinstance(conflicts, Mapping) and "permalink" in conflicts:
            if identity is None:
                return None, "permalink_identity_preflight_missing"
            return {"identity_preflight": dict(identity)}, None
        return None, None

    if operation == "drop_invalid_tags":
        validation = details.get("tag_validation_receipt")
        if validation is not None:
            if (
                not isinstance(validation, Mapping)
                or validation.get("kind") != "invalid_tag_validation_receipt"
                or validation.get("source_sha256") != _sha256_text(expected_text)
                or not _receipt_core_matches(validation)
            ):
                return None, "tag_validation_receipt_invalid"
            return {"tag_validation_receipt": dict(validation)}, None
        if repacket:
            return None, "tag_validation_receipt_missing"
        return None, None

    if not repacket:
        return None, None
    if operation == "backfill_entities_frontmatter":
        return _entity_alias_evidence_spans(expected_text, details)
    if operation == "backfill_recall_metadata":
        return None, "metadata_source_evidence_not_repacketable"
    if operation == "resolve_nested_frontmatter_conflict":
        return ({"identity_preflight": dict(identity)} if identity else None), None
    return None, "operation_semantic_evidence_not_repacketable"


def _review_receipt_from_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    rendered = _render_review_packet(packet)
    packet_coverage = packet.get("coverage")
    coverage_receipt = (
        {
            key: value
            for key, value in packet_coverage.items()
            if key != "opcode_manifest"
        }
        if isinstance(packet_coverage, Mapping)
        else {}
    )
    full_identity = {
        "page_id": packet.get("page_id"),
        "expected_chars": packet.get("expected_chars"),
        "updated_chars": packet.get("updated_chars"),
        "unified_diff_chars": packet.get("unified_diff_chars"),
        "expected_sha256": packet.get("expected_sha256"),
        "updated_sha256": packet.get("updated_sha256"),
        "full_unified_diff_sha256": packet.get("full_unified_diff_sha256"),
    }
    receipt = {
        "schema_version": 1,
        "kind": "semantic_mutation_review_receipt",
        "mode": packet.get("mode"),
        "full_chars": sum(
            int(packet.get(field) or 0)
            for field in ("expected_chars", "updated_chars", "unified_diff_chars")
        ),
        "full_sha256": _canonical_hash(full_identity),
        "rendered_chars": len(rendered),
        "rendered_sha256": _sha256_text(rendered),
        "complete": packet.get("mode") in {"full", "changed_spans"},
        "truncated": False,
        "repacket": packet.get("mode") in {"changed_spans", "insufficient"},
        "coverage": coverage_receipt,
    }
    if packet.get("mode") == "insufficient":
        receipt["insufficient_evidence_sha256"] = packet.get(
            "insufficient_evidence_sha256"
        )
    return receipt


def build_semantic_review_packet(
    *,
    page_id: str,
    expected_text: str,
    updated_text: str,
    max_chars: int | None = None,
    operation: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build complete model evidence, deterministically repacketed when needed.

    No prefix/suffix truncation is ever sent to a semantic reviewer. If the
    complete page packet is too large, every changed line span is rendered and
    a hash/range manifest covers all equal and changed spans. If that packet is
    still too large, the result is an explicit insufficient-evidence receipt;
    callers must hold it durably instead of starting a model session.
    """

    packet_limit = (
        SAFE_FIX_REVIEW_PACKET_MAX_CHARS if max_chars is None else max(1, max_chars)
    )
    expected_lines = expected_text.splitlines(keepends=True)
    updated_lines = updated_text.splitlines(keepends=True)
    unified_diff = "".join(
        difflib.unified_diff(
            expected_lines,
            updated_lines,
            fromfile=f"{page_id}:before",
            tofile=f"{page_id}:after",
            n=5,
        )
    )
    manifest, changed_spans = _review_opcode_manifest(expected_lines, updated_lines)
    manifest_sha256 = _sha256_text(_canonical_json(manifest))
    changed_spans_sha256 = _sha256_text(_canonical_json(changed_spans))
    full_identity = {
        "page_id": page_id,
        "expected_chars": len(expected_text),
        "updated_chars": len(updated_text),
        "unified_diff_chars": len(unified_diff),
        "expected_sha256": _sha256_text(expected_text),
        "updated_sha256": _sha256_text(updated_text),
        "full_unified_diff_sha256": _sha256_text(unified_diff),
    }
    coverage = {
        "schema_version": 1,
        "expected_lines": len(expected_lines),
        "updated_lines": len(updated_lines),
        "opcode_count": len(manifest),
        "changed_span_count": len(changed_spans),
        "opcode_manifest_sha256": manifest_sha256,
        "changed_spans_sha256": changed_spans_sha256,
    }
    packet_base = {
        "schema_version": 1,
        "kind": "semantic_mutation_review_packet",
        **full_identity,
    }
    operation_evidence, evidence_error = _operation_review_evidence(
        operation=operation,
        details=details or {},
        expected_text=expected_text,
        repacket=False,
    )
    full_packet: dict[str, Any] = {
        **packet_base,
        "mode": "full",
        "coverage": {
            **coverage,
            "rendered_changed_span_count": len(changed_spans),
            "all_changed_spans_rendered": True,
        },
        "preimage": expected_text,
        "postimage": updated_text,
        "unified_diff": unified_diff,
    }
    if operation_evidence is not None:
        full_packet["operation_evidence"] = operation_evidence
    if evidence_error is not None:
        full_packet_chars = len(_render_review_packet(full_packet))
        insufficient_identity = {
            **full_identity,
            "coverage": coverage,
            "attempted_full_packet_chars": full_packet_chars,
            "max_chars": packet_limit,
            "reason": evidence_error,
        }
        packet = {
            **packet_base,
            "mode": "insufficient",
            "coverage": {
                **coverage,
                "rendered_changed_span_count": 0,
                "all_changed_spans_rendered": False,
            },
            "reason": evidence_error,
            "attempted_full_packet_chars": full_packet_chars,
            "max_chars": packet_limit,
            "insufficient_evidence_sha256": _canonical_hash(insufficient_identity),
        }
    elif len(_render_review_packet(full_packet)) <= packet_limit:
        packet = full_packet
        full_packet_chars = len(_render_review_packet(full_packet))
    else:
        full_packet_chars = len(_render_review_packet(full_packet))
        operation_evidence, evidence_error = _operation_review_evidence(
            operation=operation,
            details=details or {},
            expected_text=expected_text,
            repacket=True,
        )
        repacket: dict[str, Any] = {
            **packet_base,
            "mode": "changed_spans",
            "coverage": {
                **coverage,
                "rendered_changed_span_count": len(changed_spans),
                "all_changed_spans_rendered": True,
                "opcode_manifest": manifest,
            },
            "changed_spans": changed_spans,
        }
        if operation_evidence is not None:
            repacket["operation_evidence"] = operation_evidence
        repacket_chars = len(_render_review_packet(repacket))
        if evidence_error is None and repacket_chars <= packet_limit:
            packet = repacket
        else:
            insufficient_identity = {
                **full_identity,
                "coverage": coverage,
                "attempted_repacket_chars": repacket_chars,
                "max_chars": packet_limit,
                "reason": evidence_error
                or "complete_changed_spans_exceed_review_packet_limit",
            }
            packet = {
                **packet_base,
                "mode": "insufficient",
                "coverage": {
                    **coverage,
                    "rendered_changed_span_count": 0,
                    "all_changed_spans_rendered": False,
                },
                "reason": insufficient_identity["reason"],
                "attempted_repacket_chars": repacket_chars,
                "attempted_full_packet_chars": full_packet_chars,
                "max_chars": packet_limit,
                "insufficient_evidence_sha256": _canonical_hash(insufficient_identity),
            }

    return packet, _review_receipt_from_packet(packet)


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
    normalized = {key: parsed.get(key) for key in SAFE_FIX_REVIEW_SCHEMA["required"]}
    normalized["valid"] = True
    if isinstance(parsed.get("frontier_failure"), Mapping):
        normalized["frontier_failure"] = dict(parsed["frontier_failure"])
    if isinstance(parsed.get("decision_policy"), Mapping):
        normalized["decision_policy"] = dict(parsed["decision_policy"])
    if isinstance(parsed.get("local_consensus"), Mapping):
        normalized["local_consensus"] = dict(parsed["local_consensus"])
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
    review_packet, review_receipt = build_semantic_review_packet(
        page_id=page_id,
        expected_text=expected_text,
        updated_text=updated_text,
        operation=operation,
        details=details,
    )
    proposal_details = dict(details)
    existing_receipt = proposal_details.get("review_receipt")
    if existing_receipt is not None and existing_receipt != review_receipt:
        raise ValueError("proposal details contain a conflicting review_receipt")
    proposal_details["review_receipt"] = review_receipt
    # A full diff is retained only when the review packet itself is complete
    # and small. Large proposals are represented by exact changed spans rather
    # than a misleading bounded prefix/suffix.
    rendered_diff = unified_diff if review_packet["mode"] == "full" else None
    return {
        "schema_version": 1,
        "kind": "lint_safe_fix_proposal",
        "page_id": page_id,
        "operation": operation,
        "expected_sha256": _sha256_text(expected_text),
        "updated_sha256": _sha256_text(updated_text),
        "details": proposal_details,
        "review_packet": review_packet,
        "unified_diff": rendered_diff,
        "unified_diff_sha256": _sha256_text(unified_diff),
        "full_unified_diff_sha256": _sha256_text(unified_diff),
        "unified_diff_truncated": False,
        "unified_diff_repacket": review_packet["mode"] != "full",
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
    atomic_write(
        path, json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return proposal_hash


def _runtime_insufficient_review_packet(
    proposal: Mapping[str, Any],
    *,
    expected_text: str,
    reason: str,
) -> dict[str, Any]:
    identity = {
        "proposal_sha256": _canonical_hash(proposal),
        "proposal_expected_sha256": proposal.get("expected_sha256"),
        "observed_expected_sha256": _sha256_text(expected_text),
        "reason": reason,
    }
    return {
        "schema_version": 1,
        "kind": "semantic_mutation_review_packet",
        "mode": "insufficient",
        "page_id": proposal.get("page_id"),
        "expected_sha256": proposal.get("expected_sha256"),
        "updated_sha256": proposal.get("updated_sha256"),
        "full_unified_diff_sha256": proposal.get("full_unified_diff_sha256"),
        "coverage": {
            "schema_version": 1,
            "rendered_changed_span_count": 0,
            "all_changed_spans_rendered": False,
        },
        "reason": reason,
        "insufficient_evidence_sha256": _canonical_hash(identity),
    }


def _review_packet_error(
    proposal: Mapping[str, Any],
    *,
    expected_text: str,
    updated_text: str | None = None,
) -> str | None:
    packet = proposal.get("review_packet")
    details = proposal.get("details")
    receipt = details.get("review_receipt") if isinstance(details, Mapping) else None
    if not isinstance(packet, Mapping) or not isinstance(receipt, Mapping):
        return "review_packet_or_receipt_missing"
    rendered = _render_review_packet(packet)
    packet_coverage = packet.get("coverage")
    coverage_receipt = (
        {
            key: value
            for key, value in packet_coverage.items()
            if key != "opcode_manifest"
        }
        if isinstance(packet_coverage, Mapping)
        else {}
    )
    if (
        receipt.get("rendered_sha256") != _sha256_text(rendered)
        or receipt.get("rendered_chars") != len(rendered)
        or receipt.get("truncated") is not False
        or receipt.get("mode") != packet.get("mode")
        or receipt.get("coverage") != coverage_receipt
    ):
        return "review_receipt_integrity_mismatch"
    if (
        proposal.get("expected_sha256") != _sha256_text(expected_text)
        or packet.get("expected_sha256") != proposal.get("expected_sha256")
        or packet.get("updated_sha256") != proposal.get("updated_sha256")
        or packet.get("full_unified_diff_sha256")
        != proposal.get("full_unified_diff_sha256")
    ):
        return "review_packet_proposal_hash_mismatch"
    if updated_text is not None:
        if proposal.get("updated_sha256") != _sha256_text(updated_text):
            return "review_packet_postimage_hash_mismatch"
        recompute_details = dict(details) if isinstance(details, Mapping) else {}
        recompute_details.pop("review_receipt", None)
        operation = str(proposal.get("operation") or "")
        if operation in {"broken_link_plaintext", "broken_link_retarget"}:
            target = recompute_details.get("target")
            lookup = recompute_details.get("target_lookup_receipt")
            if not isinstance(target, str) or not isinstance(lookup, Mapping):
                return "target_lookup_receipt_missing"
            try:
                store = get_store()
                store.refresh()
                current_lookup = _build_target_lookup_receipt(
                    store=store,
                    target=target,
                    page_ids=store.all_page_ids(include_system=True),
                )
            except Exception:
                return "target_lookup_index_unavailable"
            if dict(lookup) != current_lookup:
                return "target_lookup_receipt_stale"
            if operation == "broken_link_retarget":
                replacement_evidence = recompute_details.get("replacement_evidence")
                replacement = recompute_details.get("replacement")
                if not isinstance(replacement_evidence, Mapping) or not isinstance(
                    replacement,
                    str,
                ):
                    return "replacement_evidence_missing"
                replacement_path = find_page(replacement)
                if replacement_path is None:
                    system_candidate = SYSTEM_DIR / f"{replacement}.md"
                    replacement_path = (
                        system_candidate if system_candidate.is_file() else None
                    )
                try:
                    current_replacement = (
                        replacement_path.read_text(encoding="utf-8")
                        if replacement_path is not None
                        else None
                    )
                except (OSError, UnicodeDecodeError):
                    current_replacement = None
                if current_replacement != replacement_evidence.get("text"):
                    return "replacement_evidence_stale"
        recomputed_packet, recomputed_receipt = build_semantic_review_packet(
            page_id=str(proposal.get("page_id") or ""),
            operation=str(proposal.get("operation") or ""),
            expected_text=expected_text,
            updated_text=updated_text,
            details=recompute_details,
        )
        if dict(packet) != recomputed_packet or dict(receipt) != recomputed_receipt:
            return "review_packet_trusted_recomputation_mismatch"
    mode = packet.get("mode")
    if mode == "full":
        preimage = packet.get("preimage")
        postimage = packet.get("postimage")
        unified_diff = packet.get("unified_diff")
        if not all(
            isinstance(value, str) for value in (preimage, postimage, unified_diff)
        ):
            return "full_review_packet_bytes_missing"
        assert isinstance(preimage, str)
        assert isinstance(postimage, str)
        assert isinstance(unified_diff, str)
        if (
            _sha256_text(preimage) != proposal.get("expected_sha256")
            or _sha256_text(postimage) != proposal.get("updated_sha256")
            or _sha256_text(unified_diff) != proposal.get("full_unified_diff_sha256")
            or receipt.get("complete") is not True
        ):
            return "full_review_packet_integrity_mismatch"
        return None
    if mode == "changed_spans":
        coverage = packet.get("coverage")
        spans = packet.get("changed_spans")
        if not isinstance(coverage, Mapping) or not isinstance(spans, list):
            return "changed_span_packet_missing"
        manifest = coverage.get("opcode_manifest")
        if not isinstance(manifest, list):
            return "changed_span_coverage_manifest_missing"
        if (
            coverage.get("opcode_manifest_sha256")
            != _sha256_text(_canonical_json(manifest))
            or coverage.get("changed_spans_sha256")
            != _sha256_text(_canonical_json(spans))
            or coverage.get("changed_span_count") != len(spans)
            or coverage.get("rendered_changed_span_count") != len(spans)
            or coverage.get("all_changed_spans_rendered") is not True
            or receipt.get("complete") is not True
        ):
            return "changed_span_coverage_integrity_mismatch"
        changed_manifest = [
            entry
            for entry in manifest
            if isinstance(entry, Mapping) and entry.get("tag") != "equal"
        ]
        if len(changed_manifest) != len(spans):
            return "changed_span_manifest_count_mismatch"
        for manifest_entry, span in zip(changed_manifest, spans, strict=True):
            if not isinstance(span, Mapping):
                return "changed_span_invalid"
            expected_segment = span.get("expected_text")
            updated_segment = span.get("updated_text")
            if not isinstance(expected_segment, str) or not isinstance(
                updated_segment, str
            ):
                return "changed_span_bytes_missing"
            for field in (
                "ordinal",
                "tag",
                "expected_line_range",
                "updated_line_range",
                "expected_chars",
                "updated_chars",
                "expected_sha256",
                "updated_sha256",
            ):
                if span.get(field) != manifest_entry.get(field):
                    return "changed_span_manifest_mismatch"
            span_without_hash = dict(span)
            span_sha256 = span_without_hash.pop("span_sha256", None)
            if (
                _sha256_text(expected_segment) != span.get("expected_sha256")
                or _sha256_text(updated_segment) != span.get("updated_sha256")
                or span_sha256 != _canonical_hash(span_without_hash)
            ):
                return "changed_span_hash_mismatch"
        return None
    if mode == "insufficient":
        if (
            receipt.get("complete") is not False
            or not isinstance(packet.get("reason"), str)
            or not isinstance(packet.get("insufficient_evidence_sha256"), str)
        ):
            return "insufficient_review_packet_invalid"
        return None
    return "review_packet_mode_invalid"


def _review_packet_for_prompt(
    proposal: Mapping[str, Any],
    *,
    expected_text: str,
    updated_text: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    error = _review_packet_error(
        proposal,
        expected_text=expected_text,
        updated_text=updated_text,
    )
    if error is not None:
        return (
            _runtime_insufficient_review_packet(
                proposal,
                expected_text=expected_text,
                reason=error,
            ),
            error,
        )
    packet = proposal.get("review_packet")
    assert isinstance(packet, Mapping)
    return dict(packet), None


def _build_safe_fix_prompt(
    proposal: Mapping[str, Any],
    *,
    expected_text: str,
    updated_text: str | None = None,
) -> str:
    from llm_wiki_mcp.decision_lane_prompts import (
        semantic_mutation_decision_rubric,
        semantic_mutation_final_check,
    )

    operation = str(proposal.get("operation") or "")
    rubric = semantic_mutation_decision_rubric(operation)
    final_check = semantic_mutation_final_check(operation)
    review_packet, _packet_error = _review_packet_for_prompt(
        proposal,
        expected_text=expected_text,
        updated_text=updated_text,
    )
    rendered_proposal = dict(proposal)
    rendered_proposal.pop("review_packet", None)
    # The packet below is the only review rendering. Avoid duplicating a full
    # diff or exposing an old bounded storage rendering beside it.
    rendered_proposal["unified_diff"] = None
    rendered_proposal["proposal_sha256"] = _canonical_hash(proposal)
    rendered_details = rendered_proposal.get("details")
    if isinstance(rendered_details, Mapping):
        rendered_details = dict(rendered_details)
        replacement_evidence = rendered_details.get("replacement_evidence")
        if isinstance(replacement_evidence, Mapping) and isinstance(
            replacement_evidence.get("text"),
            str,
        ):
            compact_replacement = dict(replacement_evidence)
            compact_replacement["text"] = None
            compact_replacement["text_rendering"] = (
                "exact bytes are in review_packet.operation_evidence"
            )
            rendered_details["replacement_evidence"] = compact_replacement
        rendered_proposal["details"] = rendered_details
    return f"""\
You are the final autonomous reviewer for an LLM Wiki semantic page mutation.
A local deterministic checker produced the proposal below, but it has no
authority to change content or metadata. Independently decide whether this
exact mutation is correct. Approve only when every change is justified by the
supplied page context and operation-specific evidence. Do not propose or apply
a different patch.

{rubric}

Exact proposal:
{json.dumps(rendered_proposal, ensure_ascii=False, indent=2, sort_keys=True)}

Complete deterministic review packet (never prefix/suffix truncated):
{_render_review_packet(review_packet)}

{final_check}

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
        decision_lane="lint_safe_semantic_mutation",
    )


def _load_safe_fix_verdict(
    artifact_dir: Path,
    proposal_hash: str,
    *,
    prompt_hash: str,
    evidence_hash: str,
    review_packet: Mapping[str, Any],
    authority: Mapping[str, Any],
    decision_lane: str,
) -> dict[str, Any] | None:
    path = _verdict_artifact_path(artifact_dir, proposal_hash)
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(envelope, dict) or any(
        (
            envelope.get("schema_version") != 3,
            envelope.get("kind") != "lint_safe_fix_frontier_verdict",
            envelope.get("proposal_sha256") != proposal_hash,
            envelope.get("evidence_sha256") != evidence_hash,
            envelope.get("authority") != authority,
            envelope.get("authority_sha256") != _canonical_hash(authority),
        )
    ):
        return None
    verdict = envelope.get("verdict")
    if not isinstance(verdict, Mapping):
        return None
    normalized = _normalize_safe_fix_review(verdict)
    if normalized.get("valid") is not True:
        return None
    decision = normalized.get("decision")
    if decision not in {"approved", "rejected", "quarantined", "needs_retry"}:
        return None
    source = envelope.get("verdict_source")
    if source == "deterministic_preflight":
        if (
            decision != "needs_retry"
            or review_packet.get("mode") != "insufficient"
            or envelope.get("packet_insufficient_evidence_sha256")
            != review_packet.get("insufficient_evidence_sha256")
        ):
            return None
    elif source == "semantic_reviewer":
        if (
            semantic_verdict_authority_error(
                normalized,
                authority,
                lane=decision_lane,
            )
            is not None
        ):
            return None
    else:
        return None
    if decision in {"approved", "rejected"}:
        if envelope.get("prompt_sha256") != prompt_hash:
            return None
    else:
        hold_sha256 = _canonical_hash(
            {
                "proposal_sha256": proposal_hash,
                "evidence_sha256": evidence_hash,
                "authority_sha256": _canonical_hash(authority),
            }
        )
        if envelope.get("hold_sha256") != hold_sha256:
            return None
        if (
            decision == "needs_retry"
            and envelope.get("insufficient_evidence_sha256") != hold_sha256
        ):
            return None
    normalized["authority"] = dict(authority)
    normalized["evidence_sha256"] = evidence_hash
    normalized["hold_sha256"] = envelope.get("hold_sha256")
    normalized["reused"] = True
    return normalized


def _write_safe_fix_verdict(
    artifact_dir: Path,
    proposal_hash: str,
    *,
    prompt_hash: str,
    evidence_hash: str,
    review_packet: Mapping[str, Any],
    verdict: Mapping[str, Any],
    authority: Mapping[str, Any],
    decision_lane: str,
    verdict_source: str,
) -> None:
    path = _verdict_artifact_path(artifact_dir, proposal_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = verdict.get("decision")
    authority_hash = _canonical_hash(authority)
    hold_hash = (
        _canonical_hash(
            {
                "proposal_sha256": proposal_hash,
                "evidence_sha256": evidence_hash,
                "authority_sha256": authority_hash,
            }
        )
        if decision in {"quarantined", "needs_retry"}
        else None
    )
    envelope = seal_semantic_artifact(
        {
            "schema_version": 3,
            "kind": "lint_safe_fix_frontier_verdict",
            "proposal_sha256": proposal_hash,
            "prompt_sha256": prompt_hash,
            "evidence_sha256": evidence_hash,
            "authority_sha256": authority_hash,
            "verdict_source": verdict_source,
            "hold_sha256": hold_hash,
            "insufficient_evidence_sha256": (
                hold_hash if decision == "needs_retry" else None
            ),
            "packet_insufficient_evidence_sha256": review_packet.get(
                "insufficient_evidence_sha256"
            ),
            "verdict": {
                **{key: verdict.get(key) for key in SAFE_FIX_REVIEW_SCHEMA["required"]},
                **(
                    {"decision_policy": dict(verdict["decision_policy"])}
                    if isinstance(verdict.get("decision_policy"), Mapping)
                    else {}
                ),
                **(
                    {"local_consensus": dict(verdict["local_consensus"])}
                    if isinstance(verdict.get("local_consensus"), Mapping)
                    else {}
                ),
            },
        },
        authority=authority,
        lane=decision_lane,
    )
    atomic_write(
        path, json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _deterministic_safe_fix_hold(review_packet: Mapping[str, Any]) -> dict[str, Any]:
    reason = str(review_packet.get("reason") or "semantic review evidence incomplete")
    return {
        "decision": "needs_retry",
        "summary": f"deterministic review preflight held proposal: {reason}",
        "tests_run": ["validated exact proposal and review-packet coverage"],
        "commit": None,
        "committed": False,
        "pushed": False,
        "risk": "model review was not started with incomplete evidence",
        "notes": None,
        "valid": True,
    }


def _review_safe_fix(
    proposal: Mapping[str, Any],
    *,
    expected_text: str,
    updated_text: str,
    reviewer: StructuredReviewer,
    artifact_dir: Path,
    decision_lane: str,
    injected_reviewer: bool,
) -> dict[str, Any]:
    proposal_hash = _canonical_hash(proposal)
    review_packet, _packet_error = _review_packet_for_prompt(
        proposal,
        expected_text=expected_text,
        updated_text=updated_text,
    )
    evidence_hash = _canonical_hash(review_packet)
    prompt = _build_safe_fix_prompt(
        proposal,
        expected_text=expected_text,
        updated_text=updated_text,
    )
    prompt_hash = _sha256_text(prompt)
    with _safe_fix_review_lock(artifact_dir, proposal_hash):
        # The exact proposal must be durable before calling the frontier model.
        proposal_hash = _write_or_validate_proposal(artifact_dir, proposal)
        with decision_authority_lock():
            authority, authority_error = current_semantic_authority(
                decision_lane,
                injected_reviewer=injected_reviewer,
            )
            if authority_error is not None or authority is None:
                return {
                    "decision": "needs_retry",
                    "summary": authority_error or "decision authority unavailable",
                    "valid": False,
                }
            reused = _load_safe_fix_verdict(
                artifact_dir,
                proposal_hash,
                prompt_hash=prompt_hash,
                evidence_hash=evidence_hash,
                review_packet=review_packet,
                authority=authority,
                decision_lane=decision_lane,
            )
            if reused is None and review_packet.get("mode") == "insufficient":
                verdict = _deterministic_safe_fix_hold(review_packet)
                try:
                    _write_safe_fix_verdict(
                        artifact_dir,
                        proposal_hash,
                        prompt_hash=prompt_hash,
                        evidence_hash=evidence_hash,
                        review_packet=review_packet,
                        verdict=verdict,
                        authority=authority,
                        decision_lane=decision_lane,
                        verdict_source="deterministic_preflight",
                    )
                except Exception as exc:
                    return {
                        "decision": "needs_retry",
                        "summary": f"durable deterministic hold write failed: {exc}",
                        "valid": False,
                    }
                verdict["authority"] = dict(authority)
                verdict["evidence_sha256"] = evidence_hash
                return verdict
        if reused is not None:
            return reused
        try:
            verdict = _normalize_safe_fix_review(
                reviewer(prompt, SAFE_FIX_REVIEW_SCHEMA)
            )
        except Exception as exc:
            return {
                "decision": "needs_retry",
                "summary": f"frontier safe-fix review failed: {exc.__class__.__name__}: {exc}",
                "valid": False,
            }
        if verdict.get("valid") is True and verdict.get("decision") in {
            "approved",
            "rejected",
            "quarantined",
            "needs_retry",
        }:
            if verdict.get("decision") == "needs_retry" and isinstance(
                verdict.get("frontier_failure"), Mapping
            ):
                # Transport/resource/budget deferrals are not semantic holds.
                # They may run again after the transient condition changes.
                return verdict
            with decision_authority_lock():
                current_authority, authority_error = current_semantic_authority(
                    decision_lane,
                    injected_reviewer=injected_reviewer,
                )
                authority_error = authority_error or compare_semantic_authority(
                    authority,
                    current_authority,
                    lane=decision_lane,
                )
                authority_error = authority_error or semantic_verdict_authority_error(
                    verdict,
                    authority,
                    lane=decision_lane,
                )
                if authority_error is not None:
                    return {
                        "decision": "needs_retry",
                        "summary": authority_error,
                        "valid": False,
                    }
                try:
                    _write_safe_fix_verdict(
                        artifact_dir,
                        proposal_hash,
                        prompt_hash=prompt_hash,
                        evidence_hash=evidence_hash,
                        review_packet=review_packet,
                        verdict=verdict,
                        authority=authority,
                        decision_lane=decision_lane,
                        verdict_source="semantic_reviewer",
                    )
                except Exception as exc:
                    return {
                        "decision": "needs_retry",
                        "summary": f"durable local-consensus verdict write failed: {exc}",
                        "valid": False,
                    }
            verdict["authority"] = dict(authority)
            verdict["evidence_sha256"] = evidence_hash
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
    updated_text: str,
    reviewer: StructuredReviewer,
    artifact_dir: Path,
    decision_lane: str,
    injected_reviewer: bool = False,
) -> dict[str, Any]:
    """Persist and frontier-review one exact semantic mutation proposal."""

    return _review_safe_fix(
        proposal,
        expected_text=expected_text,
        updated_text=updated_text,
        reviewer=reviewer,
        artifact_dir=artifact_dir,
        decision_lane=decision_lane,
        injected_reviewer=injected_reviewer,
    )


@contextmanager
def semantic_review_effect_lock(
    review: Mapping[str, Any],
    *,
    decision_lane: str,
    injected_reviewer: bool = False,
) -> Iterator[bool]:
    """Keep one reviewed authority epoch stable across its durable effect."""

    expected_authority = review.get("authority")
    with decision_authority_lock():
        current_authority, authority_error = current_semantic_authority(
            decision_lane,
            injected_reviewer=injected_reviewer,
        )
        authority_error = authority_error or compare_semantic_authority(
            expected_authority,
            current_authority,
            lane=decision_lane,
        )
        authority_error = authority_error or semantic_verdict_authority_error(
            review,
            expected_authority,
            lane=decision_lane,
        )
        yield authority_error is None


def _build_target_lookup_receipt(
    *,
    store: Any,
    target: str,
    page_ids: set[str],
) -> dict[str, Any]:
    sorted_ids = sorted(page_ids)
    fuzzy_candidates = difflib.get_close_matches(
        target,
        sorted_ids,
        n=5,
        cutoff=0.6,
    )
    fuzzy_candidate = find_fuzzy_match(target, page_ids)
    core = {
        "schema_version": 1,
        "kind": "broken_link_target_lookup_receipt",
        "target": target,
        "index_snapshot": {
            "corpus_version": store.corpus_version(),
            "page_count": len(sorted_ids),
            "page_ids_sha256": _sha256_text(_canonical_json(sorted_ids)),
        },
        "target_absent": target not in page_ids,
        "fuzzy_candidates": fuzzy_candidates,
        "fuzzy_candidate": fuzzy_candidate,
        "no_acceptable_fuzzy_candidate": fuzzy_candidate is None,
    }
    return {**core, "receipt_sha256": _canonical_hash(core)}


def _target_lookup_receipt_is_current(
    *,
    store: Any,
    target: str,
    expected_receipt: Mapping[str, Any],
) -> bool:
    try:
        store.refresh()
        current_ids = store.all_page_ids(include_system=True)
        current = _build_target_lookup_receipt(
            store=store,
            target=target,
            page_ids=current_ids,
        )
    except Exception:
        return False
    return current == expected_receipt


def _build_external_page_evidence(*, page_id: str, text: str) -> dict[str, Any]:
    core = {
        "schema_version": 1,
        "kind": "external_page_review_evidence",
        "page_id": page_id,
        "source_chars": len(text),
        "source_sha256": _sha256_text(text),
        "text": text,
    }
    return {**core, "receipt_sha256": _canonical_hash(core)}


def _build_tag_validation_receipt(
    *,
    expected_text: str,
    updated_text: str,
    kept: list[str],
    dropped_values: list[Any],
) -> dict[str, Any]:
    invalid: list[dict[str, str]] = []
    for value in dropped_values:
        if isinstance(value, str):
            valid, reason = validate_tag(value)
            if valid:
                raise ValueError(
                    "tag validation receipt cannot mark a valid tag invalid"
                )
        else:
            reason = "tag is not a string"
        invalid.append({"value_repr": repr(value), "reason": reason})
    core = {
        "schema_version": 1,
        "kind": "invalid_tag_validation_receipt",
        "source_sha256": _sha256_text(expected_text),
        "updated_sha256": _sha256_text(updated_text),
        "kept_tags": list(kept),
        "invalid_tags": invalid,
    }
    return {**core, "receipt_sha256": _canonical_hash(core)}


def _atomic_write_if_unchanged(
    path: Path,
    expected: str,
    updated: str,
    *,
    evidence_guards: tuple[tuple[Path, str], ...] = (),
    pre_write_validator: Callable[[], bool] | None = None,
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
            if pre_write_validator is not None and not pre_write_validator():
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
    store.refresh()
    actions: list[str] = []
    mutated = False
    all_page_ids = store.all_page_ids(include_system=True)
    frontier_reviewer = reviewer or _default_safe_fix_reviewer
    injected_reviewer = reviewer is not None
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

            target_lookup_receipt = _build_target_lookup_receipt(
                store=store,
                target=target,
                page_ids=all_page_ids,
            )
            # A stale lint issue must never remove a link that now resolves.
            if target_lookup_receipt["target_absent"] is not True:
                continue

            replacement = target_lookup_receipt["fuzzy_candidate"]
            replacement_guards: tuple[tuple[Path, str], ...] = ()
            replacement_evidence: dict[str, Any] | None = None
            if replacement and replacement != target:
                replacement_path = find_page(replacement)
                if replacement_path is None:
                    system_candidate = SYSTEM_DIR / f"{replacement}.md"
                    replacement_path = (
                        system_candidate if system_candidate.is_file() else None
                    )
                if replacement_path is None:
                    continue
                try:
                    replacement_text = replacement_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                replacement_guards = ((replacement_path, replacement_text),)
                replacement_evidence = _build_external_page_evidence(
                    page_id=replacement,
                    text=replacement_text,
                )

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
                        "target_lookup_receipt": target_lookup_receipt,
                    },
                )
                try:
                    review = _review_safe_fix(
                        proposal,
                        expected_text=content,
                        updated_text=new_content,
                        reviewer=frontier_reviewer,
                        artifact_dir=durable_dir,
                        decision_lane="lint_safe_semantic_mutation",
                        injected_reviewer=injected_reviewer,
                    )
                except Exception as exc:
                    actions.append(
                        f"[frontier-retry] {label}: durable proposal error: {exc}"
                    )
                    continue
                if review.get("decision") == "approved" and review.get("valid") is True:
                    with semantic_review_effect_lock(
                        review,
                        decision_lane="lint_safe_semantic_mutation",
                        injected_reviewer=injected_reviewer,
                    ) as authorized:
                        if authorized and _atomic_write_if_unchanged(
                            path,
                            content,
                            new_content,
                            evidence_guards=replacement_guards,
                            pre_write_validator=(
                                lambda: _target_lookup_receipt_is_current(
                                    store=store,
                                    target=target,
                                    expected_receipt=target_lookup_receipt,
                                )
                            ),
                        ):
                            mutated = True
                            actions.append(label)
                        elif not authorized:
                            actions.append(f"[frontier-retry] {label}")
                elif (
                    review.get("decision") == "rejected" and review.get("valid") is True
                ):
                    with semantic_review_effect_lock(
                        review,
                        decision_lane="lint_safe_semantic_mutation",
                        injected_reviewer=injected_reviewer,
                    ) as authorized:
                        actions.append(
                            f"[frontier-rejected] {label}"
                            if authorized
                            else f"[frontier-retry] {label}"
                        )
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
            dropped_values: list[Any] = []
            for t in tags_raw:
                if isinstance(t, str) and validate_tag(t)[0]:
                    kept.append(t)
                else:
                    dropped.append(repr(t))
                    dropped_values.append(t)
            if not dropped:
                continue
            new_content = _frontmatter_patch(content, {"tags": kept})
            label = (
                f"[{page_id}] dropped {len(dropped)} invalid tag(s): "
                f"{', '.join(dropped[:3])}" + ("..." if len(dropped) > 3 else "")
            )
            if dry_run:
                actions.append(f"[dry-run] {label}")
            else:
                proposal = _build_safe_fix_proposal(
                    page_id=page_id,
                    operation="drop_invalid_tags",
                    expected_text=content,
                    updated_text=new_content,
                    details={
                        "kept_tags": kept,
                        "dropped_tags": dropped,
                        "tag_validation_receipt": _build_tag_validation_receipt(
                            expected_text=content,
                            updated_text=new_content,
                            kept=kept,
                            dropped_values=dropped_values,
                        ),
                    },
                )
                try:
                    review = _review_safe_fix(
                        proposal,
                        expected_text=content,
                        updated_text=new_content,
                        reviewer=frontier_reviewer,
                        artifact_dir=durable_dir,
                        decision_lane="lint_safe_semantic_mutation",
                        injected_reviewer=injected_reviewer,
                    )
                except Exception as exc:
                    actions.append(
                        f"[frontier-retry] {label}: durable proposal error: {exc}"
                    )
                    continue
                if review.get("decision") == "approved" and review.get("valid") is True:
                    with semantic_review_effect_lock(
                        review,
                        decision_lane="lint_safe_semantic_mutation",
                        injected_reviewer=injected_reviewer,
                    ) as authorized:
                        if authorized and _atomic_write_if_unchanged(
                            path, content, new_content
                        ):
                            mutated = True
                            actions.append(label)
                        elif not authorized:
                            actions.append(f"[frontier-retry] {label}")
                elif (
                    review.get("decision") == "rejected" and review.get("valid") is True
                ):
                    with semantic_review_effect_lock(
                        review,
                        decision_lane="lint_safe_semantic_mutation",
                        injected_reviewer=injected_reviewer,
                    ) as authorized:
                        actions.append(
                            f"[frontier-rejected] {label}"
                            if authorized
                            else f"[frontier-retry] {label}"
                        )
                else:
                    actions.append(f"[frontier-retry] {label}")

    # If we mutated pages, the index is now stale — refresh once at the end
    # so subsequent reads see consistent backlinks/outlinks.
    if mutated and not dry_run:
        store.refresh()

    return actions
