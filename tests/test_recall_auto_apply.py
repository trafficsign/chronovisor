from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from llm_wiki_mcp import recall_auto_apply, recall_hints, wiki
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
