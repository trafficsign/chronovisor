from __future__ import annotations

import json

from chronovisor.recall.feedback_ledger import active_feedback_rows, feedback_row_sha256


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
