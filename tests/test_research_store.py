from __future__ import annotations

from datetime import UTC, datetime, timedelta

from chronovisor.research.research_store import (
    ResearchStore,
    compact_event_context,
    reduce_events,
)


def test_evidence_cas_round_trip_and_checksum(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research")
    artifact = store.put_artifact(
        "evidence text",
        source_type="wiki",
        source_uri="wiki:page",
        citation="wiki:page",
    )

    assert store.read_artifact(artifact.artifact_id) == b"evidence text"
    assert artifact.artifact_id.startswith("sha256:")


def test_checkpoint_gc_protects_active_and_unreceipted_sessions(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research")
    store.checkpoints = tmp_path / "checkpoints"
    active = store.checkpoint("active", {"x": "a" * 100}, active=True, durable_receipt=False)
    pending = store.checkpoint("pending", {"x": "b" * 100}, active=False, durable_receipt=False)
    eligible = store.checkpoint("done", {"x": "c" * 100}, active=False, durable_receipt=True)
    old = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    import json

    payload = json.loads(eligible.read_text())
    payload["updated_at"] = old
    eligible.write_text(json.dumps(payload), encoding="utf-8")

    result = store.gc_checkpoints(ttl_seconds=60, max_total_bytes=1)

    assert active.exists()
    assert pending.exists()
    assert not eligible.exists()
    assert result["converged"] is False


def test_microcompaction_never_splits_action_observation_pair() -> None:
    paired = compact_event_context(
        [
            {"kind": "action", "epoch": 0, "iteration": 1, "value": "a"},
            {"kind": "observation", "epoch": 0, "iteration": 1, "value": "b"},
        ],
        max_chars=1,
    )
    orphan = compact_event_context(
        [{"kind": "action", "epoch": 0, "iteration": 1}], max_chars=1
    )

    assert [row["kind"] for row in paired["events"]] == ["action", "observation"]
    assert orphan["status"] == "full_history"


def test_event_reducer_exposes_orphans_and_terminal_state() -> None:
    state = reduce_events(
        [
            {"kind": "action", "epoch": 0, "iteration": 1, "action": {"type": "chronovisor_search"}},
            {"kind": "stop", "epoch": 1, "stop_reason": "interrupted"},
        ]
    )

    assert state["epoch"] == 1
    assert state["orphan_actions"][0]["iteration"] == 1
    assert state["stop_reason"] == "interrupted"
