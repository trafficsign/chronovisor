from __future__ import annotations

import json
from pathlib import Path

from chronovisor.ops import golden_expand


class FakeStore:
    def __init__(self) -> None:
        self.refresh_calls = 0

    def refresh(self) -> None:
        self.refresh_calls += 1

    def all_pages_meta(self, include_system: bool = False):
        return [{"page_id": "target", "page_type": "knowledge"}]

    def meta(self, page_id: str):
        return {"recall_questions": ["Where is the target?"]}


def test_expand_golden_from_recall_questions_appends_new_rows(tmp_path: Path, monkeypatch) -> None:
    golden = tmp_path / "search-golden.jsonl"
    page = tmp_path / "target.md"
    page.write_text("target content", encoding="utf-8")
    monkeypatch.setattr(golden_expand, "get_store", lambda: FakeStore())
    monkeypatch.setattr(golden_expand, "find_page", lambda _page_id: page)
    monkeypatch.setattr(golden_expand, "page_uid_for_id", lambda _page_id: "uid-target")

    payload = golden_expand.expand_golden_from_recall_questions(
        golden_file=tmp_path / "existing-golden.jsonl", candidate_file=golden
    )

    assert payload["added"] == 1
    row = json.loads(golden.read_text(encoding="utf-8"))
    assert row["query"] == "Where is the target?"
    assert row["expected_pages"] == ["target"]
    assert row["reviewed"] is False
    assert row["queue_status"] == "pending_review"
    assert len(row["candidate_sha256"]) == 64
    assert row["page_uid"] == "uid-target"
    assert len(row["content_sha256"]) == 64


def test_legacy_reviewed_rq_is_re_preregistered_and_dry_run_never_refreshes(
    tmp_path: Path, monkeypatch
) -> None:
    golden = tmp_path / "search-golden.jsonl"
    candidate = tmp_path / "search-label-queue.jsonl"
    golden.write_text(
        json.dumps(
            {
                "query": "Where is the target?",
                "expected_pages": ["target"],
                "source": "recall_questions",
                "reviewed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    page = tmp_path / "target.md"
    page.write_text("target content", encoding="utf-8")
    store = FakeStore()
    monkeypatch.setattr(golden_expand, "get_store", lambda: store)
    monkeypatch.setattr(golden_expand, "find_page", lambda _page_id: page)
    monkeypatch.setattr(golden_expand, "page_uid_for_id", lambda _page_id: "uid-target")

    dry = golden_expand.expand_golden_from_recall_questions(
        golden_file=golden,
        candidate_file=candidate,
        write=False,
    )

    assert dry["added"] == 1
    assert store.refresh_calls == 0
    assert not candidate.exists()

    written = golden_expand.expand_golden_from_recall_questions(
        golden_file=golden,
        candidate_file=candidate,
        write=True,
    )
    assert written["added"] == 1
    assert store.refresh_calls == 1
    row = json.loads(candidate.read_text(encoding="utf-8"))
    assert row["reviewed"] is False
    assert row["candidate_sha256"]
