from __future__ import annotations

import json

from chronovisor.core.feedback_ledger import active_feedback_rows, feedback_row_sha256


def test_retraction_requires_exact_key_and_row_digest(tmp_path) -> None:
    path = tmp_path / "feedback.jsonl"
    retracted = {
        "kind": "page_ignored",
        "content_correction_key": "shared-key",
        "prompt": "old ordinary turn",
        "negative_pages": ["old-noise"],
    }
    preserved = {
        **retracted,
        "prompt": "later valid correction",
        "negative_pages": ["valid-noise"],
    }
    tombstone = {
        "kind": "page_ignored_retracted",
        "content_correction_key": "shared-key",
        "target_kind": "page_ignored",
        "target_feedback_sha256": feedback_row_sha256(retracted),
    }
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in (retracted, preserved, tombstone)
        )
        + '{"kind":"page_ignored_retracted"',
        encoding="utf-8",
    )

    assert active_feedback_rows(path) == [preserved]


def test_validation_reads_log_once_but_keeps_exact_one_match(tmp_path, monkeypatch) -> None:
    import hashlib

    from chronovisor.core import feedback_ledger as ledger

    page = tmp_path / "noise.md"
    page.write_text("noise")
    monkeypatch.setattr(ledger, "find_page", lambda _: page)
    prompt = "exact query"
    evidence = {"field_shadow": {"topic_epoch": 0, "session_hash": "session"}}
    snapshot = {"decision_id": "d1", "decision": "search", "host": "codex", "session_id": "s1", "prompt_hash": ledger._prompt_hash(prompt), "evidence_features": evidence}
    row = {"kind": "page_ignored", "frontier_reviewed": True, "label_quality": "strong", "content_correction_key": "producer", "ref": "d1", "host": "codex", "prompt": prompt, "snapshot": snapshot, "source_turn_ref": {"session_id": "s1", "prompt_hash": snapshot["prompt_hash"], "user_line": 1, "assistant_line": 2}, "negative_pages": ["noise"], "negative_page_hashes": {"noise": hashlib.sha256(page.read_bytes()).hexdigest()}}
    feedback = tmp_path / "feedback.jsonl"
    log = tmp_path / "recall.jsonl"
    feedback.write_text((json.dumps(row) + "\n") * 2)
    record = json.dumps({**snapshot, "pages": ["noise"]}) + "\n"
    log.write_text(record)
    original = ledger.read_jsonl_rows
    reads = []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(ledger, "read_jsonl_rows", read)
    assert ledger.trusted_negative_feedback_rows(feedback, recall_log_file=log) == [row, row]
    assert reads.count(log) == 1
    log.write_text(record * 2)
    assert ledger.trusted_negative_feedback_rows(feedback, recall_log_file=log) == []
    assert reads.count(log) == 2
    log.write_text("")
    assert ledger.trusted_negative_feedback_rows(feedback, recall_log_file=log) == []
    assert reads.count(log) == 3
    feedback.write_text('{"kind":"injection_ignored"}\n')
    assert ledger.trusted_negative_feedback_rows(feedback, recall_log_file=log) == []
    assert reads.count(log) == 3
