from __future__ import annotations

from datetime import datetime, timedelta, timezone

from llm_wiki_mcp import recall_breaker


def test_breaker_opens_after_threshold_and_closes_after_cooldown(tmp_path) -> None:
    path = tmp_path / "breaker.json"
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)

    first = recall_breaker.record_failure(
        "timeout",
        threshold=2,
        cooldown_seconds=60,
        path=path,
        now=now,
    )
    second = recall_breaker.record_failure(
        "timeout again",
        threshold=2,
        cooldown_seconds=60,
        path=path,
        now=now,
    )

    assert first["status"] == "closed"
    assert second["status"] == "open"
    assert recall_breaker.is_open(path, now=now + timedelta(seconds=59)) is True
    assert recall_breaker.is_open(path, now=now + timedelta(seconds=61)) is False


def test_breaker_success_resets_failure_state(tmp_path) -> None:
    path = tmp_path / "breaker.json"
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    recall_breaker.record_failure(
        "timeout",
        threshold=1,
        cooldown_seconds=60,
        path=path,
        now=now,
    )

    state = recall_breaker.record_success(path, now=now + timedelta(seconds=1))

    assert state["status"] == "closed"
    assert state["failures"] == 0
