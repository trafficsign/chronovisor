from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from llm_wiki_mcp import recall_auto_apply, recall_hints, wiki
from llm_wiki_mcp.convergence import CycleBudget
from llm_wiki_mcp.frontmatter import parse as parse_frontmatter
from llm_wiki_mcp.recall_runtime import RecallPolicy, collect_context


def _page(root: Path, page_id: str, body: str = "Recall hook body") -> Path:
    path = root / "pages" / "ai" / f"{page_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntitle: {page_id}\nupdated: 2026-06-02\ntags: [d/tools-config, t/analysis, s/2026]\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def _candidate(action_type: str, *, page_id: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "kind": "missed_candidate",
        "source": "auditor",
        "host": "codex",
        "prompt": "昨日の recall hook の続き",
        "expected_pages": [page_id],
        "ref": "decision-1",
        "reason_code": "gate_missed",
        "missing_signal": "recall hook",
        "normalize_key": f"gate_missed:recall-hook:{page_id}",
        "action_type": action_type,
        "action_payload": payload or {},
        "lane": "auto",
        "auto_apply_eligible": True,
    }


def test_query_hint_auto_apply_feeds_runtime_context(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "claude-code-recall-hook-implementation")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(wiki, "WIKI_ROOT", pages_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)

    result = recall_auto_apply.apply_feedback_records(
        [_candidate("query_hint", page_id="claude-code-recall-hook-implementation")],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    assert result["status"] == "ok"
    assert result["actions"][0]["status"] == "applied"
    assert hints_file.exists()

    context = collect_context(
        ["昨日の recall hook の続き"],
        "read",
        RecallPolicy(max_pages=1, semantic=False),
    )

    assert [item.page_id for item in context] == ["claude-code-recall-hook-implementation"]


def test_query_hint_ignores_generic_context_tokens() -> None:
    hint = {
        "page_id": "wrong-page",
        "query": "old broad prompt",
        "query_key": "old broad prompt",
        "tokens": ["assistant", "codex", "context", "project", "wiki"],
    }

    assert not recall_hints.hint_matches_query(hint, "codex context window 切り分け")


def test_auto_apply_min_count_groups_by_normalize_key(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "claude-code-recall-hook-implementation")
    monkeypatch.setattr(wiki, "WIKI_ROOT", pages_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", tmp_path / "query-hints.json")

    one = [_candidate("query_hint", page_id="claude-code-recall-hook-implementation")]
    assert recall_auto_apply.apply_feedback_records(
        one,
        policy=recall_auto_apply.AutoApplyPolicy(min_count=2),
        log_file=tmp_path / "auto-apply.jsonl",
    )["actions"] == []

    two = one + [_candidate("query_hint", page_id="claude-code-recall-hook-implementation")]
    assert recall_auto_apply.apply_feedback_records(
        two,
        policy=recall_auto_apply.AutoApplyPolicy(min_count=2),
        log_file=tmp_path / "auto-apply.jsonl",
    )["actions"][0]["status"] == "applied"


def test_page_tag_auto_apply_patches_frontmatter(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    page = _page(pages_root, "llm-wiki-recall-audit-architecture")
    monkeypatch.setattr(wiki, "WIKI_ROOT", pages_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages_root / "pages")

    record = _candidate(
        "page_tag",
        page_id="llm-wiki-recall-audit-architecture",
        payload={"tag": "d/theory"},
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )
    meta, _body = parse_frontmatter(page.read_text(encoding="utf-8"))

    assert result["actions"][0]["status"] == "applied"
    assert "d/theory" in meta["tags"]


def test_invalid_page_tag_falls_back_to_query_hint(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "llm-wiki-recall-audit-architecture")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(wiki, "WIKI_ROOT", pages_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)

    record = _candidate(
        "page_tag",
        page_id="llm-wiki-recall-audit-architecture",
        payload={"tag": "Assistant wrote a prose reason instead of a taxonomy tag."},
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    action = result["actions"][0]
    assert action["status"] == "fallback_applied"
    assert action["result"]["fallback_to"] == "query_hint"
    assert hints_file.exists()


def test_page_tag_without_target_is_skipped_not_error(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    monkeypatch.setattr(wiki, "WIKI_ROOT", pages_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", tmp_path / "query-hints.json")

    record = _candidate(
        "page_tag",
        page_id="",
        payload={"tag": "Assistant wrote a prose reason instead of a taxonomy tag."},
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    action = result["actions"][0]
    assert action["status"] == "skipped"
    assert action["result"]["fallback_to"] == "query_hint"


def test_query_hint_without_target_is_skipped_not_error(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    hints_file = tmp_path / "query-hints.json"
    log_file = tmp_path / "auto-apply.jsonl"
    monkeypatch.setattr(wiki, "WIKI_ROOT", pages_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)

    record = _candidate("query_hint", page_id="", payload={"query": "specific missing context"})
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=log_file,
    )

    action = result["actions"][0]
    assert action["status"] == "skipped"
    assert action["result"]["reason"] == "query_hint missing page_id"
    assert "auto_apply_self_heal" not in result
    assert not hints_file.exists()
    assert recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=log_file,
    )["actions"] == []


def test_alias_auto_apply_uses_existing_alias_store(tmp_path, monkeypatch) -> None:
    from llm_wiki_mcp import alias_store

    pages_root = tmp_path / "wiki"
    _page(pages_root, "canonical-recall-page")
    monkeypatch.setattr(wiki, "WIKI_ROOT", pages_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages_root / "pages")

    record = _candidate(
        "alias",
        page_id="canonical-recall-page",
        payload={"alias": "made-up-recall-page"},
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    assert result["actions"][0]["status"] == "applied"
    assert alias_store.load_aliases()["made-up-recall-page"] == "ai/canonical-recall-page"


def test_invalid_alias_falls_back_to_query_hint(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "canonical-recall-page")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(wiki, "WIKI_ROOT", pages_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)

    record = _candidate(
        "alias",
        page_id="canonical-recall-page",
        payload={"alias": "自然言語の説明は alias page_id ではない"},
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    assert result["actions"][0]["status"] == "fallback_applied"
    assert hints_file.exists()


def test_invalid_alias_target_falls_back_to_expected_page_hint(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "canonical-recall-page")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(wiki, "WIKI_ROOT", pages_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)

    record = _candidate(
        "alias",
        page_id="canonical-recall-page",
        payload={"alias": "missing-target", "target_page": "does-not-exist"},
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    assert result["actions"][0]["status"] == "fallback_applied"
    assert hints_file.exists()


def test_error_apply_keys_are_retriable(tmp_path) -> None:
    log_file = tmp_path / "auto-apply.jsonl"
    log_file.write_text(
        "\n".join(
            [
                json.dumps({"apply_key": "failed", "status": "error"}),
                json.dumps({"apply_key": "ok", "status": "applied"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert recall_auto_apply.read_applied_keys(log_file) == {"ok"}


def test_full_apply_history_does_not_resurrect_old_terminal_keys(tmp_path) -> None:
    log_file = tmp_path / "auto-apply.jsonl"
    rows = [{"apply_key": "old", "status": "applied", "convergence_status": "applied"}]
    rows.extend(
        {
            "apply_key": f"retry-{index}",
            "status": "error",
            "convergence_status": "retry_wait",
            "attempt": 1,
        }
        for index in range(5_100)
    )
    log_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert "old" in recall_auto_apply.read_applied_keys(log_file)
    assert recall_auto_apply.read_apply_states(log_file)["old"]["convergence_status"] == "applied"


def test_pull_log_candidate_is_consumed_by_validated_auto_lane(tmp_path, monkeypatch) -> None:
    record = _candidate("query_hint", page_id="target")
    record["source"] = "pull-log"
    monkeypatch.setattr(
        recall_auto_apply,
        "apply_record",
        lambda _record, dry_run=False: {"status": "applied"},
    )

    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    assert result["actions"][0]["status"] == "applied"


def test_apply_feedback_budget_defers_without_burning_attempt(tmp_path, monkeypatch) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "auto-apply.jsonl"
    feedback_file.write_text(json.dumps(_candidate("query_hint", page_id="target")) + "\n")
    monkeypatch.setattr(recall_auto_apply, "AUTO_APPLY_LOG_FILE", log_file)
    calls = []

    def apply(_record, dry_run=False):
        calls.append(dry_run)
        return {"status": "applied"}

    monkeypatch.setattr(recall_auto_apply, "apply_record", apply)
    deferred = recall_auto_apply.apply_feedback_file(
        feedback_file=feedback_file,
        config_file=tmp_path / "missing.toml",
        budget=CycleBudget(max_mutations=0),
    )

    assert deferred["status"] == "budget_deferred"
    assert deferred["actions"][0]["attempt"] == 0
    assert calls == []
    assert not log_file.exists()

    applied = recall_auto_apply.apply_feedback_file(
        feedback_file=feedback_file,
        config_file=tmp_path / "missing.toml",
        budget=CycleBudget(max_mutations=1),
    )
    assert applied["actions"][0]["attempt"] == 1
    assert calls == [False]


def test_existing_query_hint_is_terminal_without_incrementing_evidence_count(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "target")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(wiki, "WIKI_ROOT", pages_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    recall_hints.add_query_hint(page_id="target", query="exact query", path=hints_file)
    record = _candidate(
        "query_hint",
        page_id="target",
        payload={"page_id": "target", "query": "exact query"},
    )

    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    assert result["actions"][0]["status"] == "already_applied"
    assert recall_hints.load_query_hints(hints_file)[0]["count"] == 1


def test_skipped_auto_apply_is_bounded_not_permanently_consumed(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "auto-apply.jsonl"
    record = _candidate("query_hint", page_id="missing", payload={"query": "q", "page_id": "missing"})
    monkeypatch.setattr(recall_auto_apply, "apply_record", lambda _record, dry_run=False: {"status": "skipped"})
    policy = recall_auto_apply.AutoApplyPolicy(min_count=1)

    first = recall_auto_apply.apply_feedback_records(
        [record], policy=policy, log_file=log_file, max_attempts=2, backoff_base_seconds=0
    )
    second = recall_auto_apply.apply_feedback_records(
        [record], policy=policy, log_file=log_file, max_attempts=2, backoff_base_seconds=0
    )

    assert first["actions"][0]["convergence_status"] == "retry_wait"
    assert second["actions"][0]["convergence_status"] == "quarantined"
    assert recall_auto_apply.read_applied_keys(log_file) == set()


def test_threshold_review_action_routes_to_recall_lab(tmp_path) -> None:
    log_file = tmp_path / "auto-apply.jsonl"
    record = {
        "kind": "missed_candidate",
        "source": "auditor",
        "lane": "review",
        "action_type": "threshold",
        "normalize_key": "threshold:systemic",
        "missing_signal": "systemic false positives",
        "ref": "d1",
    }

    result = recall_auto_apply.apply_review_feedback_records([record], log_file=log_file)

    assert result["actions"][0]["status"] == "routed_to_recall_lab"
    assert result["actions"][0]["convergence_status"] == "applied"


def test_query_hint_accepts_system_pages(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    system_dir = pages_root / "system"
    system_dir.mkdir(parents=True)
    (system_dir / "lessons-learned.md").write_text(
        "---\ntitle: Lessons Learned\n---\n\nbody\n",
        encoding="utf-8",
    )
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(wiki, "WIKI_ROOT", pages_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(wiki, "SYSTEM_DIR", system_dir)
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)

    hint = recall_hints.add_query_hint(page_id="lessons-learned", query="反省ルール", path=hints_file)

    assert hint["page_id"] == "lessons-learned"
    assert hints_file.exists()


def test_auditor_recording_invokes_auto_apply(tmp_path, monkeypatch, capsys) -> None:
    from llm_wiki_mcp import recall_auditor, recall_runtime

    pages_root = tmp_path / "wiki"
    _page(pages_root, "claude-code-recall-hook-implementation")
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "recall-log.jsonl"
    hints_file = tmp_path / "query-hints.json"
    prompt = "昨日の recall hook の続き"
    decision_id = "decision-1"
    log_file.write_text(
        json.dumps(
            {
                "decision_id": decision_id,
                "host": "codex",
                "session_id": "s1",
                "prompt_hash": recall_runtime.stable_prompt_hash(prompt),
                "prompt_preview": prompt,
                "decision": "none",
                "confidence": 0.2,
                "queries": [],
                "pages": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(wiki, "WIKI_ROOT", pages_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", feedback_file)
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", log_file)
    monkeypatch.setattr(recall_auditor, "RECALL_LOG_FILE", log_file)
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    monkeypatch.setattr(recall_auto_apply, "AUTO_APPLY_LOG_FILE", tmp_path / "auto-apply.jsonl")
    monkeypatch.setattr(recall_auditor, "collect_top_pages", lambda _prompt, _policy: ([], "bm25"))
    auditor_json = json.dumps(
        {
            "missed": True,
            "confidence": 0.9,
            "reason_code": "gate_missed",
            "auditor_reason": "missed context",
            "expected_pages": ["claude-code-recall-hook-implementation"],
            "missing_signal": "recall hook",
            "action_type": "query_hint",
            "action_payload": {"query": prompt},
        },
        ensure_ascii=False,
    )

    assert recall_auditor.main(
        [
            "--host",
            "codex",
            "--session-id",
            "s1",
            "--prompt",
            prompt,
            "--assistant-response",
            "続きです。",
            "--decision-id",
            decision_id,
            "--state-file",
            str(tmp_path / "state.json"),
            "--auditor-json",
            auditor_json,
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "recorded"
    assert output["auto_apply"]["actions"][0]["status"] == "applied"
    assert hints_file.exists()
