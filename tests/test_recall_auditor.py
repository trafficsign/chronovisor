from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor import recall_auditor
from chronovisor.recall_runtime import stable_prompt_hash


def _valid_auditor_payload() -> dict[str, object]:
    return {
        "missed": False,
        "confidence": 0.9,
        "reason_code": "valid_skip",
        "auditor_reason": "No prior memory was needed.",
        "expected_pages": [],
        "missing_signal": "valid_skip",
        "action_type": "none",
    }


def test_auditor_judge_repairs_schema_error_in_same_session(tmp_path: Path) -> None:
    requests = []
    invalid = _valid_auditor_payload()
    invalid["confidence"] = "high"
    responses = iter([json.dumps(invalid), json.dumps(_valid_auditor_payload())])

    def transport(request):
        requests.append(request)
        return next(responses)

    output = recall_auditor.run_auditor_judge(
        recall_auditor.TurnContext(
            host="codex",
            prompt="new standalone question",
            assistant_response="standalone answer",
        ),
        None,
        [],
        recall_auditor.AuditPolicy(),
        transport=transport,
        audit_root=tmp_path / "audit",
    )

    assert json.loads(output)["reason_code"] == "valid_skip"
    assert len(requests) == 2
    assert requests[1].messages[-2]["role"] == "assistant"
    assert "Validator errors" in requests[1].messages[-1]["content"]


def test_auditor_judge_rejects_oversized_input_before_transport(tmp_path: Path) -> None:
    calls = 0

    def transport(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("transport must not start")

    with pytest.raises(ValueError, match="input_too_large|context_window_exceeded"):
        recall_auditor.run_auditor_judge(
            recall_auditor.TurnContext(
                host="codex",
                prompt="x" * 80_000,
                assistant_response="answer",
            ),
            None,
            [],
            recall_auditor.AuditPolicy(max_prompt_chars=100_000),
            transport=transport,
            audit_root=tmp_path / "audit",
        )

    assert calls == 0


def test_threshold_action_is_review_only_even_if_auditor_is_confident() -> None:
    decision = recall_auditor.parse_auditor_output(
        json.dumps(
            {
                "missed": True,
                "confidence": 0.98,
                "reason_code": "gate_missed",
                "auditor_reason": "Gate ignored a clear past-reference.",
                "expected_pages": ["chronovisor-recall-configuration"],
                "missing_signal": "past_reference",
                "action_type": "threshold",
            }
        ),
        [{"page_id": "chronovisor-recall-configuration"}],
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
    from chronovisor import recall_runtime

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
            "--no-auto-apply",
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
    from chronovisor import recall_runtime

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


def test_auditor_single_flight_skips_when_lock_is_held(tmp_path, monkeypatch) -> None:
    lock_file = tmp_path / "audit.lock"
    lock_handle = recall_auditor.acquire_audit_lock(lock_file)
    assert lock_handle is not None
    monkeypatch.setattr(recall_auditor, "collect_top_pages", lambda _prompt, _policy: ([], "bm25"))

    def unexpected_judge(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("heavy auditor should not run when lock is held")

    monkeypatch.setattr(recall_auditor, "run_auditor_judge", unexpected_judge)
    try:
        result = recall_auditor.run(
            Namespace(
                host="codex",
                hook=False,
                session_id="s1",
                session_file=None,
                sessions_root=None,
                cwd="",
                prompt="昨日の recall hook の続き",
                assistant_response="続きです。",
                decision_id="",
                state_file=str(tmp_path / "state.json"),
                lock_file=str(lock_file),
                config=str(tmp_path / "missing.toml"),
                ignore_state=True,
                dry_run=False,
                extract_only=False,
                force=False,
                audit_read=False,
                top_k=None,
                min_confidence=None,
                auditor_json=None,
            )
        )
    finally:
        recall_auditor.release_audit_lock(lock_handle)

    assert result["status"] == "skipped"
    assert result["reason"] == "another recall audit is already running"


def test_pull_events_are_after_decision_and_exact_once(tmp_path, monkeypatch) -> None:
    pull_log = tmp_path / "pull-log.jsonl"
    consumed = tmp_path / "consumed.jsonl"
    rows = [
        {"ts": "2026-07-10T10:00:00", "session_id": "s1", "type": "read", "page_id": "old"},
        {"ts": "2026-07-10T10:02:00", "session_id": "s1", "decision_id": "d1", "type": "used", "page_ids": ["new"]},
    ]
    pull_log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(recall_auditor, "RECALL_PULL_LOG_FILE", pull_log)
    turn = recall_auditor.TurnContext(host="codex", prompt="p", assistant_response="a", session_id="s1")
    snapshot = {"ts": "2026-07-10T10:01:00", "pages": [], "decision_id": "d1"}

    first = recall_auditor.matching_pull_events(turn, snapshot, consumed_file=consumed)
    assert [event["missed_pages"] for event in first] == [["new"]]
    consumed.write_text(json.dumps({"event_key": first[0]["event_key"]}) + "\n", encoding="utf-8")
    assert recall_auditor.matching_pull_events(turn, snapshot, consumed_file=consumed) == []


def test_record_pull_candidate_marks_event_consumed_after_feedback(tmp_path, monkeypatch) -> None:
    consumed = tmp_path / "consumed.jsonl"
    monkeypatch.setattr(recall_auditor, "append_feedback", lambda *args, **kwargs: {"ref": kwargs.get("ref", "")})
    turn = recall_auditor.TurnContext(host="codex", prompt="p", assistant_response="a", session_id="s1")
    event = {
        "ts": "2026-07-10T10:02:00",
        "session_id": "s1",
        "decision_id": "d1",
        "type": "used",
        "page_ids": ["new"],
        "missed_pages": ["new"],
    }
    event["event_key"] = recall_auditor.pull_event_key(event)

    records = recall_auditor.record_pull_missed_candidates(
        turn=turn,
        recall_snapshot={"decision_id": "d1"},
        pull_events=[event],
        host="codex",
        consumed_file=consumed,
    )

    assert len(records) == 1
    assert json.loads(consumed.read_text())["event_key"] == event["event_key"]


def test_pull_event_matching_is_bounded_to_exact_session_turn(tmp_path, monkeypatch) -> None:
    pull_log = tmp_path / "pull-log.jsonl"
    recall_log = tmp_path / "recall-log.jsonl"
    feedback_file = tmp_path / "feedback.jsonl"
    consumed = tmp_path / "consumed.jsonl"
    pull_rows = [
        {
            "ts": "2026-07-10T12:02:00+02:00",
            "session_id": "s1",
            "decision_id": "d1",
            "type": "used",
            "page_ids": ["inside-page"],
        },
        {"ts": "2026-07-10T10:02:10Z", "session_id": "", "decision_id": "other", "type": "used", "page_ids": ["blank"]},
        {"ts": "2026-07-10T10:02:20Z", "session_id": "s2", "decision_id": "d1", "type": "used", "page_ids": ["other"]},
        {"ts": "2026-07-10T10:04:00Z", "session_id": "s1", "decision_id": "d2", "type": "used", "page_ids": ["next-turn"]},
        {"ts": "malformed", "session_id": "s1", "decision_id": "d1", "type": "used", "page_ids": ["malformed"]},
    ]
    pull_log.write_text("".join(json.dumps(row) + "\n" for row in pull_rows), encoding="utf-8")
    recall_log.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"ts": "2026-07-10T10:01:00Z", "session_id": "s1", "decision_id": "d1"},
                {"ts": "2026-07-10T10:03:00Z", "session_id": "s1", "decision_id": "d2"},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(recall_auditor, "RECALL_PULL_LOG_FILE", pull_log)
    monkeypatch.setattr(recall_auditor, "RECALL_LOG_FILE", recall_log)
    turn = recall_auditor.TurnContext(
        host="codex", prompt="p", assistant_response="a", session_id="s1"
    )

    events = recall_auditor.matching_pull_events(
        turn,
        {"ts": "2026-07-10T10:01:00Z", "pages": [], "decision_id": "d1"},
        consumed_file=consumed,
        feedback_file=feedback_file,
    )

    assert [event["missed_pages"] for event in events] == [["inside-page"]]
    no_identity = recall_auditor.TurnContext(
        host="codex", prompt="p", assistant_response="a", session_id=""
    )
    assert recall_auditor.matching_pull_events(no_identity, {"ts": "2026-07-10T10:01:00Z"}) == []


def test_feedback_commit_suppresses_pull_duplicate_and_heals_consumed_index(
    tmp_path,
    monkeypatch,
) -> None:
    from chronovisor import recall_runtime

    pull_log = tmp_path / "pull-log.jsonl"
    feedback_file = tmp_path / "feedback.jsonl"
    consumed = tmp_path / "consumed.jsonl"
    event = {
        "ts": "2026-07-10T10:02:00Z",
        "session_id": "s1",
        "decision_id": "d1",
        "type": "used",
        "page_ids": ["target"],
        "missed_pages": ["target"],
    }
    event["event_key"] = recall_auditor.pull_event_key(event)
    pull_log.write_text(json.dumps(event) + "\n", encoding="utf-8")
    feedback_file.write_text(
        json.dumps({"kind": "missed_candidate", "pull_event_key": event["event_key"]}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(recall_auditor, "RECALL_PULL_LOG_FILE", pull_log)
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", feedback_file)
    monkeypatch.setattr(
        recall_auditor,
        "append_feedback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not append twice")),
    )
    turn = recall_auditor.TurnContext(
        host="codex", prompt="p", assistant_response="a", session_id="s1"
    )

    assert (
        recall_auditor.record_pull_missed_candidates(
            turn=turn,
            recall_snapshot={"decision_id": "d1"},
            pull_events=[event],
            host="codex",
            consumed_file=consumed,
        )
        == []
    )
    recovered = json.loads(consumed.read_text(encoding="utf-8"))
    assert recovered["event_key"] == event["event_key"]
    assert recovered["recovered_from_feedback"] is True
