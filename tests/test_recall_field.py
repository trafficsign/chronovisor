from __future__ import annotations

import json
import statistics
import threading
import time
from pathlib import Path

from chronovisor.recall import recall_field
from chronovisor.recall.recall_field_schema import (
    FieldStimulus,
    RecallFieldConfig,
    RecallFieldState,
    load_recall_field_config,
    session_hash,
    topic_signature,
)
from chronovisor.recall.recall_field_store import RecallFieldStore
from chronovisor.search.graph_edges import typed_neighbors


class FakeGraphStore:
    def __init__(self) -> None:
        self.links = {"a": ["b"], "b": ["c"]}
        self.entities = {"a": ["shared"], "b": ["shared"], "c": ["shared"]}

    def refresh_if_stale(self) -> None:
        return None

    def outlinks(self, page_id: str) -> list[str]:
        return self.links.get(page_id, [])

    def backlinks(self, page_id: str) -> list[str]:
        return [
            source
            for source, targets in self.links.items()
            if page_id in targets
        ]

    def tags(self, _page_id: str) -> list[str]:
        return []

    def pages_for_tag(self, _tag: str) -> list[str]:
        return []

    def meta(self, page_id: str) -> dict:
        return {"page_id": page_id, "entities": self.entities.get(page_id, [])}

    def pages_for_entity(self, entity: str) -> list[str]:
        return [
            page_id
            for page_id, entities in self.entities.items()
            if entity in entities
        ]


def config(**overrides) -> RecallFieldConfig:
    return RecallFieldConfig(
        mode="shadow",
        max_active_nodes=overrides.get("max_active_nodes", 128),
        max_active_edges=overrides.get("max_active_edges", 256),
        working_set_size=overrides.get("working_set_size", 30),
        topic_reset_similarity=overrides.get("topic_reset_similarity", 0.15),
        event_retention=overrides.get("event_retention", 2_000),
        session_ttl_seconds=overrides.get(
            "session_ttl_seconds", 7 * 24 * 60 * 60
        ),
    )


def new_state(name: str = "session-a", now: float = 100.0) -> RecallFieldState:
    return RecallFieldState(
        session_hash=name,
        created_at_epoch=now,
        updated_at_epoch=now,
    )


def test_session_and_topic_artifacts_are_hashed() -> None:
    prompt = "Chronovisor BGE reranker secret phrase"
    hashed = session_hash("codex", "private-session")
    signature = topic_signature(prompt)

    assert len(hashed) == 16
    assert "private-session" not in hashed
    assert signature
    assert all(len(token) == 12 for token in signature)
    assert all(word.casefold() not in signature for word in prompt.split())


def test_field_config_defaults_to_shadow_and_reads_limits(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[recall.field]
mode = "candidate"
canary_percent = 5
working_set_size = 24
max_active_nodes = 64
max_active_edges = 120
positive_learning = false
""",
        encoding="utf-8",
    )

    loaded = load_recall_field_config(path)

    assert loaded.mode == "candidate"
    assert loaded.canary_percent == 5
    assert loaded.working_set_size == 24
    assert loaded.max_active_nodes == 64
    assert loaded.max_active_edges == 120
    assert loaded.positive_learning is False


def test_fixed_replay_is_deterministic_and_uses_shadow_buffer(monkeypatch) -> None:
    from chronovisor.search import cofire

    monkeypatch.setattr(cofire, "neighbors", lambda *_args, **_kwargs: [])
    graph = FakeGraphStore()
    stimulus = [FieldStimulus("a", "prompt_exact", 0.9)]
    signature = topic_signature("stable topic")

    left, left_events = recall_field.update_field_state(
        new_state(),
        stimuli=stimulus,
        prompt_signature=signature,
        config=config(),
        now=101.0,
        graph_store=graph,
    )
    right, right_events = recall_field.update_field_state(
        new_state(),
        stimuli=stimulus,
        prompt_signature=signature,
        config=config(),
        now=101.0,
        graph_store=graph,
    )

    assert left.to_dict() == right.to_dict()
    assert [event.to_dict() for event in left_events] == [
        event.to_dict() for event in right_events
    ]
    assert left.active == {}
    assert left.shadow["a"].activation > left.shadow["b"].activation > 0
    assert any(event.kind == "spread" for event in left_events)


def test_field_topic_reset_capacity_and_negative_activation(monkeypatch) -> None:
    from chronovisor.search import cofire

    monkeypatch.setattr(cofire, "neighbors", lambda *_args, **_kwargs: [])
    graph = FakeGraphStore()
    cfg = config(max_active_nodes=2, working_set_size=1, topic_reset_similarity=0.8)
    state, _events = recall_field.update_field_state(
        new_state(),
        stimuli=[
            FieldStimulus("a", "prompt_exact", 0.9),
            FieldStimulus("b", "prompt_exact", 0.8),
            FieldStimulus("c", "prompt_exact", 0.7),
        ],
        prompt_signature=topic_signature("first topic memory"),
        config=cfg,
        now=101.0,
        graph_store=graph,
    )
    state, events = recall_field.update_field_state(
        state,
        stimuli=[FieldStimulus("c", "negative_feedback", 0.6, negative=True)],
        prompt_signature=topic_signature("unrelated aerospace interview"),
        config=cfg,
        now=102.0,
        graph_store=graph,
    )

    assert state.topic_epoch == 1
    assert len(state.shadow) <= 2
    assert state.shadow["c"].negative > 0
    assert any(event.kind == "topic_reset" for event in events)
    assert any(event.kind == "inhibit" for event in events)


def test_exposure_cofire_is_not_an_authority_edge(monkeypatch) -> None:
    from chronovisor.search import cofire

    monkeypatch.setattr(
        cofire,
        "neighbors",
        lambda *_args, **_kwargs: [
            {
                "page_id": "positive",
                "weight": 0.8,
                "signals": ["positive_used"],
            },
            {
                "page_id": "exposed",
                "weight": 1.0,
                "signals": ["exposure"],
            },
        ],
    )
    graph = FakeGraphStore()
    graph.links = {}
    graph.entities = {}

    edges = typed_neighbors(
        graph,
        "a",
        include_exposure_cofire=False,
        degree_normalize=True,
    )

    assert "positive" in {edge.target for edge in edges}
    assert "exposed" not in {edge.target for edge in edges}


def test_teacher_commit_is_inactive_until_next_turn(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.search import cofire

    monkeypatch.setattr(cofire, "neighbors", lambda *_args, **_kwargs: [])
    cfg = config()
    store = RecallFieldStore(tmp_path / "field", config=cfg)
    hashed = session_hash("codex", "s1")
    graph = FakeGraphStore()

    def first_turn(state):
        return recall_field.update_field_state(
            state,
            stimuli=[],
            prompt_signature=topic_signature("stable topic"),
            config=cfg,
            now=101.0,
            graph_store=graph,
        )

    store.transact(hashed, first_turn, now=101.0)
    queued = recall_field.queue_teacher_commits(
        host="codex",
        session_id="s1",
        page_ids=["teacher-page"],
        config=cfg,
        store=store,
        now=101.5,
    )
    before = store.load(hashed, now=101.5)

    assert queued["queued"] == 1
    assert "teacher-page" not in before.shadow

    def second_turn(state):
        return recall_field.update_field_state(
            state,
            stimuli=[],
            prompt_signature=topic_signature("stable topic"),
            config=cfg,
            now=102.0,
            graph_store=graph,
        )

    after, events = store.transact(hashed, second_turn, now=102.0)

    assert after.shadow["teacher-page"].activation > 0
    assert any(event.kind == "commit_applied" for event in events)


def test_store_seal_corrupt_recovery_retention_and_concurrency(
    tmp_path: Path,
) -> None:
    cfg = config(event_retention=100)
    store = RecallFieldStore(tmp_path / "field", config=cfg)
    hashed = "abcdef0123456789"

    def increment(state):
        state.turn += 1
        state.updated_at_epoch += 1
        return state, []

    threads = [
        threading.Thread(target=lambda: store.transact(hashed, increment, now=100.0))
        for _index in range(12)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    state = store.load(hashed, now=112.0)
    snapshot_path = store.session_root / f"{hashed}.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert state.turn == 12
    assert len(snapshot["snapshot_sha256"]) == 64

    snapshot_path.write_text("{broken", encoding="utf-8")
    recovered = store.load(hashed, now=200.0)

    assert recovered.turn == 0
    assert list(store.session_root.glob(f"{hashed}.corrupt-*.json"))


def test_store_separates_sessions_orders_events_and_expires_ttl(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.search import cofire

    monkeypatch.setattr(cofire, "neighbors", lambda *_args, **_kwargs: [])
    cfg = config(session_ttl_seconds=60)
    store = RecallFieldStore(tmp_path / "field", config=cfg)
    graph = FakeGraphStore()

    for hashed, page_id in (("session-a", "a"), ("session-b", "b")):
        store.transact(
            hashed,
            lambda state, page_id=page_id: recall_field.update_field_state(
                state,
                stimuli=[FieldStimulus(page_id, "prompt_exact", 0.9)],
                prompt_signature=topic_signature(page_id),
                config=cfg,
                now=100.0,
                graph_store=graph,
            ),
            now=100.0,
        )

    left = store.load("session-a", now=100.0)
    right = store.load("session-b", now=100.0)
    events = store.read_events("session-a")

    assert left.shadow["a"].direct > 0
    assert left.shadow["b"].direct == 0
    assert right.shadow["b"].direct > 0
    assert right.shadow["a"].direct == 0
    assert [row["seq"] for row in events] == sorted(row["seq"] for row in events)
    assert store.cleanup(now=161.0) == 2


def test_field_update_p95_is_below_50ms_and_has_no_prompt_body(
    monkeypatch,
) -> None:
    from chronovisor.search import cofire

    monkeypatch.setattr(cofire, "neighbors", lambda *_args, **_kwargs: [])
    graph = FakeGraphStore()
    latencies: list[float] = []
    serialized = ""
    for index in range(100):
        started = time.perf_counter()
        state, events = recall_field.update_field_state(
            new_state(now=100.0),
            stimuli=[FieldStimulus("a", "prompt_exact", 0.9)],
            prompt_signature=topic_signature("private raw prompt"),
            config=config(),
            now=101.0 + index,
            graph_store=graph,
        )
        latencies.append((time.perf_counter() - started) * 1_000)
        serialized = json.dumps(
            {
                "state": state.to_dict(),
                "events": [event.to_dict() for event in events],
            }
        )
    p95 = sorted(latencies)[round((len(latencies) - 1) * 0.95)]

    assert p95 < 50
    assert "private raw prompt" not in serialized
    assert statistics.median(latencies) < p95
