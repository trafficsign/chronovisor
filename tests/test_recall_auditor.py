from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from llm_wiki_mcp import recall_auditor
from llm_wiki_mcp.recall_runtime import stable_prompt_hash


def test_threshold_action_is_review_only_even_if_auditor_is_confident() -> None:
    decision = recall_auditor.parse_auditor_output(
        json.dumps(
            {
                "missed": True,
                "confidence": 0.98,
                "reason_code": "gate_missed",
                "auditor_reason": "Gate ignored a clear past-reference.",
                "expected_pages": ["llm-wiki-recall-configuration"],
                "missing_signal": "past_reference",
                "action_type": "threshold",
            }
        ),
        [{"page_id": "llm-wiki-recall-configuration"}],
    )

    assert decision.action_type == "threshold"
    assert decision.lane == "review"
    assert decision.auto_apply_eligible is False


def test_auto_lane_is_limited_to_additive_actions() -> None:
    for action_type in ("alias", "query_hint", "page_tag"):
        decision = recall_auditor.parse_auditor_output(
            json.dumps(
                {
                    "missed": True,
                    "confidence": 0.9,
                    "reason_code": "query_missed",
                    "auditor_reason": "Search wording missed the page.",
                    "expected_pages": ["claude-code-recall-hook-implementation"],
                    "missing_signal": "recall_hook",
                    "action_type": action_type,
                }
            ),
            [{"page_id": "claude-code-recall-hook-implementation"}],
        )

        assert decision.lane == "auto"
        assert decision.auto_apply_eligible is True


def test_normalize_key_is_stable_for_same_structural_pattern() -> None:
    key1 = recall_auditor.build_normalize_key(
        "gate_missed",
        ["claude-code-recall-hook-implementation"],
        "Past Reference",
    )
    key2 = recall_auditor.build_normalize_key(
        "gate_missed",
        ["claude-code-recall-hook-implementation"],
        "past   reference",
    )

    assert key1 == key2
    assert key1 == "gate_missed:past-reference:claude-code-recall-hook-implementation"


def test_latest_complete_turn_has_stable_turn_ref() -> None:
    turn = recall_auditor.latest_complete_turn(
        [
            SimpleNamespace(role="user", line=10, text="昨日の recall hook の続き"),
            SimpleNamespace(role="assistant", line=11, text="まず recent log を見ます。"),
        ],
        host="codex",
        session_file=Path("/tmp/session.jsonl"),
        session_id="s1",
        cwd="/repo",
    )

    assert turn is not None
    assert turn.turn_ref()["session_id"] == "s1"
    assert turn.turn_ref()["user_line"] == 10
    assert turn.turn_ref()["assistant_line"] == 11
    assert turn.turn_ref()["prompt_hash"] == stable_prompt_hash("昨日の recall hook の続き")
    assert turn.turn_id


def test_matching_recall_log_prefers_prompt_hash_and_session(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    prompt = "昨日の recall hook の続き"
    log_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "decision_id": "wrong",
                        "host": "codex",
                        "session_id": "s2",
                        "prompt_hash": stable_prompt_hash(prompt),
                        "prompt_preview": prompt,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "decision_id": "right",
                        "host": "codex",
                        "session_id": "s1",
                        "prompt_hash": stable_prompt_hash(prompt),
                        "prompt_preview": prompt,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(recall_auditor, "RECALL_LOG_FILE", log_file)
    turn = recall_auditor.TurnContext(
        host="codex",
        prompt=prompt,
        assistant_response="続きです。",
        session_id="s1",
    )

    found = recall_auditor.find_matching_recall_log(turn, host="codex")

    assert found is not None
    assert found["decision_id"] == "right"


def test_cli_records_missed_candidate_with_snapshot(tmp_path, monkeypatch, capsys) -> None:
    from llm_wiki_mcp import recall_runtime

    prompt = "昨日の recall hook の続き"
    decision_id = "20260602T210000-auditme"
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "recall-log.jsonl"
    state_file = tmp_path / "audit-state.json"
    log_file.write_text(
        json.dumps(
            {
                "ts": "2026-06-02T21:00:00",
                "decision_id": decision_id,
                "host": "codex",
                "event": "UserPromptSubmit",
                "cwd": "/repo",
                "session_id": "s1",
                "prompt_hash": stable_prompt_hash(prompt),
                "prompt_chars": len(prompt),
                "prompt_preview": prompt,
                "decision": "none",
                "confidence": 0.2,
                "queries": [],
                "pages": [],
                "reasons": ["judge: skip"],
                "used_judge": True,
                "judge_confidence": 0.2,
                "judge_reason": "skip",
                "latency_ms": 100,
                "status": "ok",
                "error": "",
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", feedback_file)
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", log_file)
    monkeypatch.setattr(recall_auditor, "RECALL_LOG_FILE", log_file)
    monkeypatch.setattr(
        recall_auditor,
        "collect_top_pages",
        lambda _prompt, _policy: (
            [
                {
                    "page_id": "claude-code-recall-hook-implementation",
                    "title": "Claude Code Recall Hook Implementation",
                    "updated": "2026-06-02",
                    "score": 1.0,
                    "snippet": "recall hook",
                }
            ],
            "bm25",
        ),
    )
    auditor_json = json.dumps(
        {
            "missed": True,
            "confidence": 0.91,
            "reason_code": "gate_missed",
            "auditor_reason": "Assistant had to continue recall hook work without injected context.",
            "expected_pages": ["claude-code-recall-hook-implementation"],
            "missing_signal": "past_reference",
            "action_type": "query_hint",
        }
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
            "続きの実装方針を説明しました。",
            "--decision-id",
            decision_id,
            "--state-file",
            str(state_file),
            "--auditor-json",
            auditor_json,
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    record = json.loads(feedback_file.read_text().splitlines()[-1])

    assert output["status"] == "recorded"
    assert record["kind"] == "missed_candidate"
    assert record["source"] == "auditor"
    assert record["ref"] == decision_id
    assert record["snapshot"]["decision_id"] == decision_id
    assert record["expected_pages"] == ["claude-code-recall-hook-implementation"]
    assert record["reason_code"] == "gate_missed"
    assert record["normalize_key"] == (
        "gate_missed:past-reference:claude-code-recall-hook-implementation"
    )
    assert record["action_type"] == "query_hint"
    assert record["lane"] == "auto"
    assert record["auto_apply_eligible"] is True
    assert record["turn_ref"]["session_id"] == "s1"


def test_run_skips_read_decisions_by_default(tmp_path, monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    prompt = "昨日の recall hook の続き"
    decision_id = "20260602T210000-readok"
    log_file = tmp_path / "recall-log.jsonl"
    log_file.write_text(
        json.dumps(
            {
                "decision_id": decision_id,
                "host": "codex",
                "session_id": "s1",
                "prompt_hash": stable_prompt_hash(prompt),
                "prompt_preview": prompt,
                "decision": "read",
                "confidence": 0.9,
                "queries": ["recall hook"],
                "pages": ["claude-code-recall-hook-implementation"],
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", log_file)

    result = recall_auditor.run(
        Namespace(
            host="codex",
            hook=False,
            session_id="s1",
            session_file=None,
            sessions_root=None,
            cwd="",
            prompt=prompt,
            assistant_response="文脈を使って返答しました。",
            decision_id=decision_id,
            state_file=str(tmp_path / "state.json"),
            config=str(tmp_path / "missing.toml"),
            ignore_state=True,
            dry_run=False,
            extract_only=False,
            force=False,
            audit_read=False,
            top_k=None,
            min_confidence=None,
            auditor_json=json.dumps(
                {
                    "missed": True,
                    "confidence": 0.9,
                    "reason_code": "gate_missed",
                    "auditor_reason": "would be missed",
                    "expected_pages": [],
                    "missing_signal": "past_reference",
                    "action_type": "query_hint",
                }
            ),
        )
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "recall decision was already read"
