from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llm_wiki_mcp import lint_repair
from llm_wiki_mcp.convergence import CycleBudget, ConvergenceStore, RetryPolicy
from llm_wiki_mcp.frontmatter import parse as parse_frontmatter


NOW = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
VALID_TAGS = ["d/tools-config", "t/howto", "s/evergreen"]


def _store(tmp_path: Path, *, policy: RetryPolicy | None = None) -> ConvergenceStore:
    return ConvergenceStore(
        tmp_path / "runtime" / "convergence" / "state.json",
        policy=policy,
    )


def _budget() -> CycleBudget:
    return CycleBudget(
        max_local_calls=20,
        max_frontier_calls=20,
        max_mutations=20,
        max_elapsed_seconds=60,
    )


def _page(path: Path, *, tags: list[str] | None = None, body: str = "# Page\n\nUseful content.\n") -> str:
    tag_line = "" if tags is None else f"tags: [{', '.join(tags)}]\n"
    text = (
        "---\n"
        "title: Test Page\n"
        "updated: 2026-01-01\n"
        f"{tag_line}"
        "---\n\n"
        f"{body}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _row(
    page: str,
    *,
    lane: str = "heavy_model_batch",
    issue_type: str = "tag_missing",
) -> dict[str, object]:
    return {
        "type": "lint_repair_candidate",
        "issue_key": f"key-{page}-{issue_type}",
        "lane": lane,
        "issue_type": issue_type,
        "severity": "high",
        "page": page,
        "detail": f"test {issue_type}",
        "auto_fixable": False,
    }


def _queue(path: Path, rows: list[dict[str, object]]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return payload.encode()


def _find_only(page_id: str, expected_id: str, path: Path) -> Path | None:
    return path if page_id == expected_id else None


def _never(*_args, **_kwargs):
    raise AssertionError("reviewer must not be called")


def test_normalize_tag_decision_is_fail_closed() -> None:
    approved = lint_repair.normalize_tag_decision(
        {"decision": "approved", "tags": VALID_TAGS, "reason": "matches page"}
    )
    invalid_axes = lint_repair.normalize_tag_decision(
        {"decision": "approved", "tags": ["d/tools-config"], "reason": "incomplete"}
    )
    malformed_rejection = lint_repair.normalize_tag_decision(
        {"decision": "rejected", "tags": [], "reason": "no", "unexpected": True}
    )
    duplicate_tags = lint_repair.normalize_tag_decision(
        {
            "decision": "approved",
            "tags": ["d/tools-config", "d/tools-config", "t/howto", "s/evergreen"],
            "reason": "duplicate domain tag",
        }
    )

    assert approved["decision"] == "approved"
    assert approved["valid"] is True
    assert invalid_axes["decision"] == "needs_retry"
    assert invalid_axes["valid"] is False
    assert any("t/ has 0" in error for error in invalid_axes["validation_errors"])
    assert malformed_rejection["decision"] == "needs_retry"
    assert malformed_rejection["valid"] is False
    assert duplicate_tags["decision"] == "needs_retry"
    assert "duplicate tags" in duplicate_tags["validation_errors"]


def test_local_approved_tags_apply_and_finish_terminally(tmp_path: Path, monkeypatch) -> None:
    page_id = "test-page"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(
        lint_repair.wiki,
        "find_page",
        lambda candidate: _find_only(candidate, page_id, page_path),
    )
    store = _store(tmp_path)
    budget = _budget()

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=budget,
        local_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "page is a configuration how-to",
        },
        frontier_reviewer=_never,
        now=NOW,
    )

    meta, body = parse_frontmatter(page_path.read_text(encoding="utf-8"))
    item = store.list_items()[0]
    assert result["applied"] == 1
    assert result["escalated"] == 0
    assert result["rejected"] == 0
    assert result["quarantined"] == 0
    assert result["budget"]["used"] == {
        "local": 1,
        "frontier": 0,
        "mutation": 1,
        "raw_bytes": 0,
    }
    assert meta["tags"] == VALID_TAGS
    assert body == "\n# Page\n\nUseful content.\n"
    assert item["status"] == "applied"
    assert item["result"]["review_stage"] == "local"


def test_invalid_local_proposal_escalates_to_frontier_and_applies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "frontier-page"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id, issue_type="tag_count_violation")])
    monkeypatch.setattr(lint_repair.wiki, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)
    budget = _budget()

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=budget,
        local_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": ["d/tools-config"],
            "reason": "malformed local proposal",
        },
        frontier_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "frontier supplied all required axes",
        },
        now=NOW,
    )

    meta, _body = parse_frontmatter(page_path.read_text(encoding="utf-8"))
    item = store.list_items()[0]
    assert result["applied"] == 1
    assert result["escalated"] == 1
    assert result["budget"]["used"] == {
        "local": 1,
        "frontier": 1,
        "mutation": 1,
        "raw_bytes": 0,
    }
    assert meta["tags"] == VALID_TAGS
    assert item["status"] == "applied"
    assert item["local_attempts"] == 1
    assert item["frontier_attempts"] == 1
    assert item["result"]["review_stage"] == "frontier"


def test_frontier_rejection_is_terminal_and_does_not_mutate_page(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "rejected-page"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.wiki, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=lambda _prompt, _schema: {
            "decision": "uncertain",
            "tags": [],
            "reason": "excerpt is ambiguous",
        },
        frontier_reviewer=lambda _prompt, _schema: {
            "decision": "rejected",
            "tags": [],
            "reason": "no defensible semantic tags",
        },
        now=NOW,
    )

    item = store.list_items()[0]
    assert result["rejected"] == 1
    assert result["applied"] == 0
    assert result["budget"]["used"]["mutation"] == 0
    assert page_path.read_text(encoding="utf-8") == original
    assert item["status"] == "rejected"
    assert item["result"]["action"] == "tag_repair_rejected"


def test_frontier_auth_failure_is_the_only_kind_that_requires_a_human(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "auth-page"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.wiki, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=lambda _prompt, _schema: {
            "decision": "uncertain",
            "tags": [],
            "reason": "local model cannot decide",
        },
        frontier_reviewer=lambda _prompt, _schema: {
            "decision": "needs_retry",
            "summary": "sign-in required",
            "frontier_failure": {"failure_class": "auth_required"},
            "human_required": True,
        },
        now=NOW,
    )

    item = store.list_items()[0]
    assert item["status"] == "human_required"
    assert item["human_required"] is True
    assert item["last_failure_class"] == "auth_required"
    assert result["human_required"] == 1
    assert result["results"][0]["status"] == "human_required"
    assert result["quarantined"] == 0
    assert page_path.read_text(encoding="utf-8") == original


def test_stale_is_observed_and_duplicate_orphan_are_terminally_routed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(
        queue_path,
        [
            _row("stale-page", lane="monitor", issue_type="stale"),
            _row("duplicate-page", lane="review", issue_type="duplicate"),
            _row("orphan-page", lane="review", issue_type="orphan"),
        ],
    )
    monkeypatch.setattr(lint_repair.wiki, "find_page", lambda _page_id: None)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    items = {item["source_id"].split(":", 1)[0]: item for item in store.list_items()}
    assert result["processed"] == 3
    assert result["observed"] == 1
    assert result["routed"] == 2
    assert result["quarantined"] == 0
    assert result["budget"]["used"] == {
        "local": 0,
        "frontier": 0,
        "mutation": 0,
        "raw_bytes": 0,
    }
    assert items["stale"]["status"] == "applied"
    assert items["stale"]["result"]["action"] == "observed"
    assert items["duplicate"]["status"] == "applied"
    assert items["duplicate"]["result"]["target_lane"] == "duplicate_review"
    assert items["orphan"]["status"] == "applied"
    assert items["orphan"]["result"]["target_lane"] == "orphan_link"


def test_dry_run_is_byte_for_byte_read_only_and_calls_no_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "dry-run-page"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    page_before = _page(page_path).encode()
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    queue_before = _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.wiki, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)
    budget = _budget()

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=budget,
        dry_run=True,
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    assert result["status"] == "dry_run"
    assert result["results"][0]["status"] == "would_review_tags"
    assert result["budget"]["used"] == {
        "local": 0,
        "frontier": 0,
        "mutation": 0,
        "raw_bytes": 0,
    }
    assert page_path.read_bytes() == page_before
    assert queue_path.read_bytes() == queue_before
    assert not store.state_file.exists()
    assert not store.events_file.exists()
    assert not store.lock_file.exists()


def test_max_items_bounds_queue_work(tmp_path: Path, monkeypatch) -> None:
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(
        queue_path,
        [
            _row("first", lane="monitor", issue_type="stale"),
            _row("second", lane="monitor", issue_type="stale"),
        ],
    )
    monkeypatch.setattr(lint_repair.wiki, "find_page", lambda _page_id: None)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        max_items=1,
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    assert result["bounded"] == 1
    assert result["remaining_unseen"] == 1
    assert result["processed"] == 1
    assert len(store.list_items()) == 1

    second = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        max_items=1,
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    assert second["bounded"] == 1
    assert second["rows_scanned"] == 2
    assert second["remaining_unseen"] == 0
    assert second["terminal_skipped"] == 1
    assert second["processed"] == 1
    assert second["observed"] == 1
    assert len(store.list_items()) == 2


def test_cas_conflict_quarantines_without_overwriting_concurrent_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "cas-page"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.wiki, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)
    concurrent = "---\ntitle: Concurrent\nupdated: 2026-07-10\n---\n\n# Changed elsewhere\n"

    def local_review(_prompt, _schema):
        page_path.write_text(concurrent, encoding="utf-8")
        return {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "valid proposal for the old preimage",
        }

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=local_review,
        frontier_reviewer=_never,
        now=NOW,
    )

    item = store.list_items()[0]
    assert result["quarantined"] == 1
    assert result["applied"] == 0
    assert item["status"] == "quarantined"
    assert item["quarantine_reason"] == "tag_repair_cas_conflict"
    assert page_path.read_text(encoding="utf-8") == concurrent


def test_backoff_row_does_not_starve_later_actionable_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_path = tmp_path / "pages" / "retrying.md"
    _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(
        queue_path,
        [
            _row("retrying"),
            _row("later", lane="monitor", issue_type="stale"),
        ],
    )
    monkeypatch.setattr(
        lint_repair.wiki,
        "find_page",
        lambda page_id: page_path if page_id == "retrying" else None,
    )
    store = _store(
        tmp_path,
        policy=RetryPolicy(
            max_local_attempts=2,
            local_base_delay_seconds=3600,
        ),
    )

    first = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        max_items=1,
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )
    second = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        max_items=1,
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    assert first["results"][0]["status"] == "local_error"
    assert second["bounded"] == 1
    assert second["rows_scanned"] == 2
    assert second["processed"] == 1
    assert second["deferred"] == 1
    assert second["observed"] == 1
    assert [result["status"] for result in second["results"]] == ["deferred", "observed"]


def test_existing_valid_tags_finish_without_calling_a_model(tmp_path: Path, monkeypatch) -> None:
    page_id = "already-valid"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path, tags=VALID_TAGS)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.wiki, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    item = store.list_items()[0]
    assert result["applied"] == 1
    assert result["budget"]["used"]["local"] == 0
    assert result["budget"]["used"]["mutation"] == 0
    assert item["status"] == "applied"
    assert item["result"]["action"] == "already_resolved"
    assert page_path.read_text(encoding="utf-8") == original


def test_missing_page_is_rejected_once(tmp_path: Path, monkeypatch) -> None:
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row("missing-page")])
    monkeypatch.setattr(lint_repair.wiki, "find_page", lambda _page_id: None)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    assert result["processed"] == 1
    assert result["rejected"] == 1
    assert store.list_items()[0]["status"] == "rejected"
