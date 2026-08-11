"""Lint engine - detect and fix wiki quality issues."""

import difflib
import fcntl
import hashlib
import json
import threading
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

from chronovisor.core import frontmatter
from chronovisor.core.canonical_document import (
    Namespace,
    ResolvedMarkdownLink,
    rewrite_internal_markdown_links,
)
from chronovisor.core.canonical_json import canonical_json_permissive as _canonical_json
from chronovisor.core.hashutil import sha256_text as _sha256_text
from chronovisor.core.index_store import (
    PAGE_RESERVED_FILENAMES,
    SYSTEM_RESERVED_FILENAMES,
    canonical_document_path,
    get_store,
)
from chronovisor.core.link_fix import (
    atomic_write,
    find_fuzzy_match,
)
from chronovisor.core.page_mutation import (
    chronovisor_mutation_lock,
    decision_authority_lock,
)
from chronovisor.core.runtime_config import runtime_repo_root
from chronovisor.core.store import CHRONOVISOR_ROOT, find_page
from chronovisor.core.tag_rules import parse_tags, validate_axis_counts, validate_tag
from chronovisor.decision.decision_authority import (
    compare_semantic_authority,
    current_semantic_authority,
    seal_semantic_artifact,
    semantic_verdict_authority_error,
)
from chronovisor.decision.decision_schema_manifest import SAFE_FIX_REVIEW_SCHEMA
from chronovisor.decision.lint_mutation_contract import (
    SAFE_FIX_REPACKET_CONTEXT_LINES as SAFE_FIX_REPACKET_CONTEXT_LINES,
)
from chronovisor.decision.lint_mutation_contract import (
    SAFE_FIX_REVIEW_PACKET_MAX_CHARS as SAFE_FIX_REVIEW_PACKET_MAX_CHARS,
)
from chronovisor.decision.lint_mutation_contract import (
    SAFE_FIX_SEMANTIC_HOLD_RESOLVER_VERSION,
    StructuredReviewer,
    build_semantic_review_packet,
)
from chronovisor.decision.lint_mutation_contract import (
    build_safe_fix_prompt as build_safe_fix_prompt,
)
from chronovisor.decision.lint_mutation_contract import (
    build_safe_fix_proposal as _build_safe_fix_proposal,
)
from chronovisor.decision.lint_mutation_contract import (
    build_semantic_mutation_proposal as build_semantic_mutation_proposal,
)
from chronovisor.decision.lint_mutation_contract import (
    canonical_hash as _canonical_hash,
)
from chronovisor.decision.lint_mutation_contract import (
    render_review_packet as _render_review_packet,
)
from chronovisor.decision.lint_mutation_contract import (
    render_safe_fix_prompt as _render_safe_fix_prompt,
)
from chronovisor.decision.semantic_hold import (
    build_semantic_no_quorum_hold,
    is_local_semantic_no_quorum,
    persisted_semantic_no_quorum_hold,
)
from chronovisor.decision.semantic_hold import (
    canonical_sha256 as semantic_hold_sha256,
)

STALE_DAYS = 90  # Pages not updated in this many days are flagged
REPO_ROOT = runtime_repo_root()

# `chronovisor_apply` runs `check()` and then re-runs it inside `apply_safe_fixes`,
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
    return get_store().all_canonical_page_keys(include_system=True)


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
        if version == _CHECK_CACHE_VERSION and _CHECK_CACHE_RESULT is not None:
            # Defensive copy so callers that mutate the list (e.g. filter
            # auto-fixable issues) don't poison the cache for the next call.
            return [dict(i) for i in _CHECK_CACHE_RESULT]

    issues = []

    # System pages are part of the broken_link universe but are not
    # themselves linted (they're treated as a fixed reference set).
    all_page_ids = store.all_canonical_page_keys(include_system=True)
    pages_meta = store.all_pages_meta(include_system=False)
    lintable_pages_meta = [m for m in pages_meta if m.get("page_type") != "reference"]
    page_count = len(lintable_pages_meta)

    for meta in lintable_pages_meta:
        page_id = meta["page_id"]

        # 1. Broken canonical Markdown links.
        seen_broken: set[str] = set()
        for target in store.canonical_outlinks(page_id):
            if target in all_page_ids or target in seen_broken:
                continue
            seen_broken.add(target)
            issues.append(
                {
                    "type": "broken_link",
                    "severity": "high",
                    "page": page_id,
                    "target": target,
                    "detail": f"Link {target} points to non-existent page",
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
    queue_path = path or CHRONOVISOR_ROOT / "review" / "lint-repair-queue.jsonl"
    records = repair_queue_records(issues)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = queue_path.with_name(f".{queue_path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(queue_path)
    return queue_path


def _broken_link_target(issue: dict) -> str | None:
    """Return the exact canonical target recorded by the lint pass."""

    target = issue.get("target")
    return target.strip() if isinstance(target, str) and target.strip() else None


def _replace_link_in_content(
    content: str,
    target: str,
    replacement: str | None,
    *,
    source_namespace: Namespace,
    source_path: str,
) -> tuple[str, int]:
    """Retarget or unwrap one exact broken canonical Markdown link.

    Args:
        content: 対象ファイルの本文
        target: normalize 済みの target page_id
        replacement: fuzzy match で見つかった置換先 page_id。None か同値なら plaintext 化。

    Returns:
        (new_content, count) — count は置換された箇所数
    """
    def replace(
        link: ResolvedMarkdownLink,
        label: str,
    ) -> ResolvedMarkdownLink | str | None:
        canonical_target = (
            f"{link.namespace}/"
            f"{PurePosixPath(link.path).with_suffix('').as_posix()}"
        )
        if canonical_target != target:
            return None
        if replacement and replacement != target:
            replacement_namespace, _, replacement_path = replacement.partition("/")
            if replacement_namespace not in {"pages", "system"} or not replacement_path:
                return None
            namespace: Namespace = (
                "system" if replacement_namespace == "system" else "pages"
            )
            return ResolvedMarkdownLink(
                namespace=namespace,
                path=f"{replacement_path}.md",
                fragment=link.fragment,
            )
        visible = label.replace(r"\[", "[").replace(r"\]", "]")
        return visible or PurePosixPath(link.path).stem

    return rewrite_internal_markdown_links(
        content,
        source_namespace=source_namespace,
        source_path=source_path,
        rewrite=replace,
    )






def _safe_fix_artifact_dir(path: Path | None = None) -> Path:
    if path is not None:
        return path
    # Resolve this dynamically: isolated tests and embedded runtimes patch the
    # index-store Wiki root after this module has already been imported.
    from chronovisor.core import index_store

    return Path(index_store.CHRONOVISOR_ROOT) / "runtime" / "lint-safe-fixes"


def _proposal_artifact_path(artifact_dir: Path, proposal_hash: str) -> Path:
    return artifact_dir / "proposals" / f"{proposal_hash}.json"


def _verdict_artifact_path(artifact_dir: Path, proposal_hash: str) -> Path:
    return artifact_dir / "frontier-verdicts" / f"{proposal_hash}.json"


def _semantic_hold_artifact_path(
    artifact_dir: Path,
    *,
    proposal_hash: str,
    epoch: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> Path:
    """Address a semantic split by its full immutable decision epoch.

    Successful verdicts retain the legacy single-current artifact.  Semantic
    splits are multi-versioned so an A -> B -> A authority rollback can reuse
    the old A hold before consuming a review budget or starting a model.
    """

    identity = {
        "proposal_sha256": proposal_hash,
        "epoch_sha256": semantic_hold_sha256(epoch),
        "authority_sha256": semantic_hold_sha256(authority),
    }
    return artifact_dir / "semantic-holds" / f"{semantic_hold_sha256(identity)}.json"


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
    if isinstance(parsed.get("reviewer"), str):
        normalized["reviewer"] = parsed["reviewer"]
    if isinstance(parsed.get("human_required"), bool):
        normalized["human_required"] = parsed["human_required"]
    return normalized




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
                    page_ids=store.all_canonical_page_keys(include_system=True),
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
                replacement_path = CHRONOVISOR_ROOT / f"{replacement}.md"
                replacement_path = (
                    replacement_path if replacement_path.is_file() else None
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
    review_packet, _packet_error = _review_packet_for_prompt(
        proposal,
        expected_text=expected_text,
        updated_text=updated_text,
    )
    return _render_safe_fix_prompt(proposal, review_packet=review_packet)


def _default_safe_fix_reviewer(
    prompt: str,
    schema: dict[str, Any],
) -> Mapping[str, Any] | str:
    from chronovisor.decision import routine_review

    return routine_review.run_structured_review(
        prompt,
        schema,
        repo_root=REPO_ROOT,
        execute_patch=False,
        command_env="CHRONOVISOR_LINT_SAFE_FIX_FRONTIER_CMD",
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


def _safe_fix_semantic_epoch(
    *,
    proposal_hash: str,
    prompt_hash: str,
    evidence_hash: str,
) -> dict[str, Any]:
    return {
        "resolver_version": SAFE_FIX_SEMANTIC_HOLD_RESOLVER_VERSION,
        "proposal_sha256": proposal_hash,
        "prompt_sha256": prompt_hash,
        "evidence_sha256": evidence_hash,
    }


def _load_safe_fix_semantic_hold(
    artifact_dir: Path,
    proposal_hash: str,
    *,
    epoch: Mapping[str, Any],
    authority: Mapping[str, Any],
    decision_lane: str,
) -> dict[str, Any] | None:
    path = _semantic_hold_artifact_path(
        artifact_dir,
        proposal_hash=proposal_hash,
        epoch=epoch,
        authority=authority,
    )
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(envelope, Mapping)
        or set(envelope)
        != {
            "schema_version",
            "kind",
            "proposal_sha256",
            "semantic_hold",
        }
        or envelope.get("schema_version") != 1
        or envelope.get("kind") != "lint_safe_fix_semantic_no_quorum_hold"
        or envelope.get("proposal_sha256") != proposal_hash
    ):
        return None
    hold = persisted_semantic_no_quorum_hold(
        envelope,
        decision_lane,
        epoch=epoch,
        authority=authority,
    )
    if hold is None:
        return None
    return {
        "decision": "needs_retry",
        "summary": "exact local semantic disagreement remains held",
        "tests_run": [],
        "commit": None,
        "committed": False,
        "pushed": False,
        "risk": "no mutation was authorized without a two-vote quorum",
        "notes": None,
        "valid": True,
        "frontier_failure": dict(hold["frontier_failure"]),
        "decision_policy": dict(hold["decision_policy"]),
        "local_consensus": dict(hold["local_consensus"]),
        "semantic_hold": hold,
        "authority": dict(authority),
        "evidence_sha256": epoch["evidence_sha256"],
        "hold_sha256": hold["hold_sha256"],
        "reused": True,
    }


def _write_safe_fix_semantic_hold(
    artifact_dir: Path,
    proposal_hash: str,
    *,
    epoch: Mapping[str, Any],
    authority: Mapping[str, Any],
    decision_lane: str,
    review: Mapping[str, Any],
) -> dict[str, Any]:
    hold = build_semantic_no_quorum_hold(
        decision_lane,
        epoch,
        authority,
        review,
    )
    path = _semantic_hold_artifact_path(
        artifact_dir,
        proposal_hash=proposal_hash,
        epoch=epoch,
        authority=authority,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": 1,
        "kind": "lint_safe_fix_semantic_no_quorum_hold",
        "proposal_sha256": proposal_hash,
        "semantic_hold": hold,
    }
    atomic_write(
        path,
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    persisted = persisted_semantic_no_quorum_hold(
        json.loads(path.read_text(encoding="utf-8")),
        decision_lane,
        epoch=epoch,
        authority=authority,
    )
    if persisted is None:
        raise RuntimeError("durable semantic hold failed read-back validation")
    return persisted


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
    semantic_epoch = _safe_fix_semantic_epoch(
        proposal_hash=proposal_hash,
        prompt_hash=prompt_hash,
        evidence_hash=evidence_hash,
    )
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
            reused_semantic_hold = _load_safe_fix_semantic_hold(
                artifact_dir,
                proposal_hash,
                epoch=semantic_epoch,
                authority=authority,
                decision_lane=decision_lane,
            )
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
        if reused_semantic_hold is not None:
            return reused_semantic_hold
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
            semantic_no_quorum = is_local_semantic_no_quorum(verdict)
            if (
                verdict.get("decision") == "needs_retry"
                and isinstance(verdict.get("frontier_failure"), Mapping)
                and not semantic_no_quorum
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
                if not semantic_no_quorum:
                    authority_error = (
                        authority_error
                        or semantic_verdict_authority_error(
                            verdict,
                            authority,
                            lane=decision_lane,
                        )
                    )
                if authority_error is not None:
                    return {
                        "decision": "needs_retry",
                        "summary": authority_error,
                        "valid": False,
                    }
                try:
                    if semantic_no_quorum:
                        semantic_hold = _write_safe_fix_semantic_hold(
                            artifact_dir,
                            proposal_hash,
                            epoch=semantic_epoch,
                            authority=authority,
                            decision_lane=decision_lane,
                            review=verdict,
                        )
                    else:
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
            if semantic_no_quorum:
                verdict["semantic_hold"] = semantic_hold
                verdict["hold_sha256"] = semantic_hold["hold_sha256"]
        return verdict




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
    target_namespace = target.partition("/")[0]
    replacement_ids = {
        page_id
        for page_id in page_ids
        if page_id.partition("/")[0] == target_namespace
    }
    fuzzy_candidates = difflib.get_close_matches(
        target,
        sorted(replacement_ids),
        n=5,
        cutoff=0.6,
    )
    fuzzy_candidate = find_fuzzy_match(target, replacement_ids)
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
        current_ids = store.all_canonical_page_keys(include_system=True)
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
    invalid: list[dict[str, Any]] = []
    for value in dropped_values:
        if isinstance(value, str):
            valid, reason = validate_tag(value)
            if valid:
                raise ValueError(
                    "tag validation receipt cannot mark a valid tag invalid"
                )
        else:
            reason = "tag is not a string"
        invalid.append(
            {"value_repr": frontmatter.review_value(value), "reason": reason}
        )
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
        with chronovisor_mutation_lock():
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


def _stable_issue_path(page_id: str) -> Path | None:
    """Resolve one lint mutation target through the canonical stable boundary."""

    path = find_page(page_id)
    if path is None:
        return None
    try:
        relative = path.relative_to(CHRONOVISOR_ROOT)
    except ValueError:
        return None
    if not relative.parts:
        return None
    if relative.parts[0] == "pages":
        namespace: Namespace = "pages"
        root = CHRONOVISOR_ROOT / "pages"
        reserved = PAGE_RESERVED_FILENAMES
    elif relative.parts[0] == "system":
        namespace = "system"
        root = CHRONOVISOR_ROOT / "system"
        reserved = SYSTEM_RESERVED_FILENAMES
    else:
        return None
    return canonical_document_path(
        path,
        root,
        namespace=namespace,
        reserved_filenames=reserved,
        require_stable=True,
    )


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
    all_page_ids = store.all_canonical_page_keys(include_system=True)
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
            path = _stable_issue_path(page_id)
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
                replacement_path = CHRONOVISOR_ROOT / f"{replacement}.md"
                replacement_path = (
                    replacement_path if replacement_path.is_file() else None
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

            relative = path.relative_to(CHRONOVISOR_ROOT)
            source_namespace: Namespace = (
                "system" if relative.parts[0] == "system" else "pages"
            )
            source_path = PurePosixPath(*relative.parts[1:]).as_posix()
            new_content, count = _replace_link_in_content(
                content,
                target,
                replacement,
                source_namespace=source_namespace,
                source_path=source_path,
            )
            if count == 0 or new_content == content:
                continue

            if replacement and replacement != target:
                label = f"[{page_id}] {target} → {replacement} ({count}x)"
            else:
                label = f"[{page_id}] {target} → plaintext ({count}x)"

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
                                lambda current_target=target,
                                expected_receipt=target_lookup_receipt: _target_lookup_receipt_is_current(
                                    store=store,
                                    target=current_target,
                                    expected_receipt=expected_receipt,
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
            path = _stable_issue_path(page_id)
            if not path:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            meta, _ = frontmatter.parse(content)
            tags_raw = meta.get("tags")
            if not isinstance(tags_raw, list):
                continue

            kept: list[str] = []
            dropped: list[Any] = []
            dropped_values: list[Any] = []
            for t in tags_raw:
                if isinstance(t, str) and validate_tag(t)[0]:
                    kept.append(t)
                else:
                    dropped.append(frontmatter.review_value(t))
                    dropped_values.append(t)
            if not dropped:
                continue
            new_content = frontmatter.patch(content, {"tags": kept})
            label = (
                f"[{page_id}] dropped {len(dropped)} invalid tag(s): "
                f"{', '.join(str(value) for value in dropped[:3])}"
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
