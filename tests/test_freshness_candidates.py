from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chronovisor.recall.freshness_candidates import enqueue_from_operations


@dataclass
class Operation:
    page_id: str
    new_body: str
    path: Path


def test_freshness_candidates_separate_user_report_from_verification(tmp_path: Path) -> None:
    queue = tmp_path / "freshness.jsonl"
    operation = Operation(
        "page",
        "現在の価格は100円です。\nこれは時間に依存しない説明です。",
        tmp_path / "page.md",
    )
    first = enqueue_from_operations([operation], path=queue)
    second = enqueue_from_operations([operation], path=queue)
    assert first["enqueued"] == 1
    assert second["enqueued"] == 0
    text = queue.read_text(encoding="utf-8")
    assert '"reported_by_user": true' in text
    assert '"externally_verified": false' in text
