from __future__ import annotations

import hashlib
import json
import statistics
import threading
import time
from pathlib import Path

import pytest

from chronovisor.core.graph_edges import typed_neighbors
from chronovisor.recall import recall_field
from chronovisor.recall.recall_field_schema import (
    ActivationNode,
    FieldStimulus,
    RecallFieldConfig,
    RecallFieldState,
    load_recall_field_config,
    session_hash,
    topic_signature,
    topic_transition,
)
from chronovisor.recall.recall_field_store import RecallFieldStore


class FakeGraphStore:
    def __init__(self) -> None:
        self.links = {"a": ["b"], "b": ["c"]}
        self.entities = {"a": ["shared"], "b": ["shared"], "c": ["shared"]}

    def refresh_if_stale(self) -> None:
        return None

    def outlinks(self, page_id: str) -> list[str]:
        return self.links.get(page_id, [])

    def backlinks(self, page_id: str) -> list[str]:
        return [source for source, targets in self.links.items() if page_id in targets]

    def tags(self, _page_id: str) -> list[str]:
        return []

    def pages_for_tag(self, _tag: str) -> list[str]:
        return []

    def meta(self, page_id: str) -> dict:
        return {"page_id": page_id, "entities": self.entities.get(page_id, [])}

    def pages_for_entity(self, entity: str) -> list[str]:
        return [
            page_id for page_id, entities in self.entities.items() if entity in entities
        ]


def config(**overrides) -> RecallFieldConfig:
    return RecallFieldConfig(
        mode="shadow",
        max_active_nodes=overrides.get("max_active_nodes", 128),
        max_active_edges=overrides.get("max_active_edges", 256),
        working_set_size=overrides.get("working_set_size", 30),
        topic_reset_similarity=overrides.get("topic_reset_similarity", 0.15),
        event_retention=overrides.get("event_retention", 2_000),
        session_ttl_seconds=overrides.get("session_ttl_seconds", 7 * 24 * 60 * 60),
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
[recall.field.growth]
enabled = true
auto_promote = true
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
    assert loaded.auto_growth is True
    assert loaded.auto_promote is True


def test_effective_config_enables_positive_edges_before_authority(monkeypatch) -> None:
    from chronovisor.recall import recall_growth, recall_learning

    monkeypatch.setattr(recall_learning, "load_last_known_good", lambda _path: {})
    monkeypatch.setattr(
        recall_growth,
        "automatic_learning_allowed",
        lambda **_kwargs: True,
    )
    candidate = RecallFieldConfig(
        mode="candidate",
        auto_growth=True,
        positive_learning=False,
    )

    effective = recall_field._effective_config(candidate)

    assert effective.mode == "candidate"
    assert effective.positive_learning is True


def test_fixed_replay_is_deterministic_and_uses_shadow_buffer(monkeypatch) -> None:
    from chronovisor.core import cofire

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


def test_candidate_mode_uses_active_buffer_for_fallback_and_topic_reset() -> None:
    graph = FakeGraphStore()
    config = RecallFieldConfig(mode="candidate", max_hops=0)
    state = RecallFieldState(
        session_hash="0123456789abcdef",
        topic_signature=topic_signature("stable field topic"),
        active={"page-a": ActivationNode(activation=0.8)},
        updated_at_epoch=100.0,
    )

    continued, _events = recall_field.update_field_state(
        state,
        stimuli=[],
        prompt_signature=topic_signature("stable field topic"),
        config=config,
        now=101.0,
        graph_store=graph,
        prompt_text="stable field topic",
    )

    assert continued.full_search_fallback is False
    assert "page-a" in continued.active

    reset, _events = recall_field.update_field_state(
        continued,
        stimuli=[],
        prompt_signature=topic_signature("unrelated completely new subject"),
        config=config,
        now=102.0,
        graph_store=graph,
        prompt_text="話を変える unrelated completely new subject",
    )

    assert reset.full_search_fallback is True
    assert reset.active == {}


def test_field_topic_reset_capacity_and_negative_activation(monkeypatch) -> None:
    from chronovisor.core import cofire

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


def test_topic_shift_distinguishes_stable_pronoun_and_abrupt_switch() -> None:
    previous = topic_signature("Chronovisor の Recall Field を実装する")

    stable, stable_similarity = topic_transition(
        previous,
        topic_signature("Chronovisor Recall Field の実装を続ける"),
        prompt="Chronovisor Recall Field の実装を続ける",
        reset_similarity=0.15,
    )
    continuation, continuation_similarity = topic_transition(
        previous,
        topic_signature("それの続きはどうなった"),
        prompt="それの続きはどうなった",
        reset_similarity=0.15,
    )
    abrupt, _abrupt_similarity = topic_transition(
        previous,
        topic_signature("ところで旅行の航空券を比較したい"),
        prompt="ところで旅行の航空券を比較したい",
        reset_similarity=0.15,
    )

    assert stable == "stable"
    assert stable_similarity >= 0.15
    assert continuation == "continuation"
    assert continuation_similarity < 0.15
    assert abrupt == "reset"


def test_exposure_cofire_is_not_an_authority_edge(monkeypatch) -> None:
    from chronovisor.core import cofire

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
    from chronovisor.core import cofire

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


def test_reviewed_negative_is_exact_idempotent_and_exactly_retractable(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.core import feedback_ledger

    page = tmp_path / "page.md"
    page.write_text("bound page bytes", encoding="utf-8")
    page_sha = hashlib.sha256(page.read_bytes()).hexdigest()
    monkeypatch.setattr(
        feedback_ledger,
        "find_page",
        lambda page_id: page if page_id == "bound-page" else None,
    )
    monkeypatch.setattr(feedback_ledger, "RECALL_DIR", tmp_path)
    hashed = session_hash("codex", "negative-session")
    field_evidence = {"field_shadow": {"topic_epoch": 0, "session_hash": hashed}}
    (tmp_path / "recall-log.jsonl").write_text(
        json.dumps(
            {
                "decision_id": "decision-1",
                "session_id": "negative-session",
                "host": "codex",
                "prompt_hash": "9c7e05e5868afcf3",
                "decision": "read",
                "pages": ["bound-page"],
                "evidence_features": field_evidence,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = config()
    store = RecallFieldStore(tmp_path / "field", config=cfg)

    def seed(state):
        state.host = "codex"
        state.topic_prompt_hash = "9c7e05e5868afcf3"
        state.shadow["bound-page"] = ActivationNode(activation=0.9, direct=0.9)
        state.pending_teacher_commits = [
            {"page_id": "bound-page", "available_turn": 1, "topic_epoch": 0}
        ]
        return state, []

    store.transact(hashed, seed, now=100.0)
    original = {
        "kind": "page_ignored",
        "host": "codex",
        "frontier_reviewed": True,
        "label_quality": "strong",
        "content_correction_key": "correction-1",
        "ref": "decision-1",
        "prompt": "prompt",
        "snapshot": {
            "decision_id": "decision-1",
            "session_id": "negative-session",
            "host": "codex",
            "prompt_hash": "9c7e05e5868afcf3",
            "decision": "read",
            "evidence_features": field_evidence,
        },
        "source_turn_ref": {
            "session_id": "negative-session",
            "prompt_hash": "9c7e05e5868afcf3",
            "user_line": 1,
            "assistant_line": 2,
        },
        "negative_pages": ["bound-page"],
        "negative_page_hashes": {"bound-page": page_sha},
    }

    applied = recall_field.apply_reviewed_negative_feedback(
        original,
        config=cfg,
        store=store,
        application_file=tmp_path / "applications.jsonl",
        now=101.0,
    )
    duplicate = recall_field.apply_reviewed_negative_feedback(
        original,
        config=cfg,
        store=store,
        application_file=tmp_path / "applications.jsonl",
        now=102.0,
    )
    after_apply = store.load(hashed, now=102.0)

    assert applied["status"] == "applied"
    assert duplicate["status"] == "duplicate"
    assert after_apply.shadow["bound-page"].activation == pytest.approx(0.9)
    assert recall_field.effective_activation(
        after_apply,
        page_id="bound-page",
        node=after_apply.shadow["bound-page"],
        buffer_name="shadow",
    ) == pytest.approx(0.15)
    assert after_apply.shadow["bound-page"].negative == 0.0
    assert after_apply.pending_teacher_commits == []
    assert len(after_apply.negative_contributions) == 1
    suppressed_queue = recall_field.queue_teacher_commits(
        host="codex",
        session_id="negative-session",
        page_ids=["bound-page"],
        config=cfg,
        store=store,
        now=102.5,
    )
    assert suppressed_queue["queued"] == 0

    malformed = {
        "kind": "page_ignored_retracted",
        "target_kind": "page_ignored",
        "content_correction_key": "correction-1",
        "target_feedback_sha256": "0" * 64,
    }
    held = recall_field.retract_reviewed_negative_feedback(
        malformed,
        original,
        config=cfg,
        store=store,
        application_file=tmp_path / "applications.jsonl",
        now=103.0,
    )
    assert held == {
        "status": "held",
        "reason": "retraction_binding_invalid",
        "restored": 0,
    }
    assert store.load(hashed, now=103.0).shadow[
        "bound-page"
    ].activation == pytest.approx(0.9)

    exact = {
        **malformed,
        "target_feedback_sha256": feedback_ledger.feedback_row_sha256(original),
    }
    retracted = recall_field.retract_reviewed_negative_feedback(
        exact,
        original,
        config=cfg,
        store=store,
        application_file=tmp_path / "applications.jsonl",
        now=104.0,
    )
    duplicate_retraction = recall_field.retract_reviewed_negative_feedback(
        exact,
        original,
        config=cfg,
        store=store,
        application_file=tmp_path / "applications.jsonl",
        now=105.0,
    )
    restored = store.load(hashed, now=105.0)

    assert retracted["status"] == "retracted"
    assert duplicate_retraction["status"] == "duplicate_or_missing"
    assert restored.shadow["bound-page"].activation == pytest.approx(0.9)
    assert restored.shadow["bound-page"].negative == 0.0
    assert restored.negative_contributions == {}


def test_reviewed_negative_rejects_stale_page_binding(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.core import feedback_ledger

    page = tmp_path / "page.md"
    page.write_text("current", encoding="utf-8")
    monkeypatch.setattr(feedback_ledger, "find_page", lambda _page_id: page)
    monkeypatch.setattr(feedback_ledger, "RECALL_DIR", tmp_path)
    (tmp_path / "recall-log.jsonl").write_text(
        json.dumps(
            {
                "decision_id": "decision-stale",
                "session_id": "session-stale",
                "host": "codex",
                "prompt_hash": "9c7e05e5868afcf3",
                "decision": "read",
                "pages": ["page"],
                "evidence_features": {
                    "field_shadow": {"topic_epoch": 0, "session_hash": "field-session"}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = recall_field.apply_reviewed_negative_feedback(
        {
            "kind": "page_ignored",
            "host": "codex",
            "frontier_reviewed": True,
            "label_quality": "strong",
            "content_correction_key": "correction-stale",
            "ref": "decision-stale",
            "prompt": "prompt",
            "snapshot": {
                "decision_id": "decision-stale",
                "session_id": "session-stale",
                "host": "codex",
                "prompt_hash": "9c7e05e5868afcf3",
                "decision": "read",
                "evidence_features": {
                    "field_shadow": {
                        "topic_epoch": 0,
                        "session_hash": "field-session",
                    }
                },
            },
            "source_turn_ref": {
                "session_id": "session-stale",
                "prompt_hash": "9c7e05e5868afcf3",
                "user_line": 1,
                "assistant_line": 2,
            },
            "negative_pages": ["page"],
            "negative_page_hashes": {"page": "0" * 64},
        },
        config=config(),
        store=RecallFieldStore(tmp_path / "field", config=config()),
        application_file=tmp_path / "applications.jsonl",
    )

    assert result == {"status": "held", "reason": "page_hash_mismatch", "applied": 0}


def test_v1_snapshot_migrates_destructive_negative_to_composition(
    tmp_path: Path,
) -> None:
    store = RecallFieldStore(tmp_path / "field", config=config())
    hashed = "0123456789abcdef"
    payload = {
        "schema_version": 1,
        "session_hash": hashed,
        "host": "codex",
        "topic_epoch": 0,
        "turn": 1,
        "seq": 1,
        "created_at_epoch": 1.0,
        "updated_at_epoch": 2.0,
        "topic_signature": [],
        "active": {},
        "shadow": {
            "page": {
                "activation": 0.15,
                "direct": 0.9,
                "spread": 0.0,
                "negative": 0.75,
                "inhibition": 0.0,
                "anti_index": 0.0,
                "hub_penalty": 0.0,
                "last_turn": 1,
                "last_seq": 1,
            }
        },
        "pending_teacher_commits": [],
        "negative_contributions": {
            "producer": {
                "producer_key": "producer",
                "feedback_sha256": "a" * 64,
                "page_hashes": {"page": "b" * 64},
                "deltas": {"page": 0.75},
                "buffer": "shadow",
                "status": "active",
            }
        },
        "full_search_fallback": False,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    snapshot = {**payload, "snapshot_sha256": hashlib.sha256(encoded).hexdigest()}
    path = store.legacy_session_root / f"{hashed}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    state = store.load(hashed, now=3.0)

    assert state.shadow["page"].activation == pytest.approx(0.9)
    assert recall_field.effective_activation(
        state,
        page_id="page",
        node=state.shadow["page"],
        buffer_name="shadow",
    ) == pytest.approx(0.15)
    assert json.loads(path.read_text())["schema_version"] == 1
    assert json.loads(store._state_path(hashed).read_text())["schema_version"] == 2


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
    from chronovisor.core import cofire

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


def test_mcp_activity_targets_latest_session_for_client_host(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from chronovisor.core import cofire

    monkeypatch.setattr(cofire, "neighbors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(recall_field, "get_store", FakeGraphStore)
    cfg = config()
    store = RecallFieldStore(tmp_path / "field", config=cfg)
    codex_session = session_hash("codex", "codex-session")
    claude_session = session_hash("claude-code", "claude-session")

    def seed(host: str):
        def mutate(state):
            state.host = host
            state.updated_at_epoch = 100.0
            return state, []

        return mutate

    store.transact(codex_session, seed("codex"), now=100.0)
    store.transact(claude_session, seed("claude-code"), now=101.0)

    result = recall_field.record_mcp_activity(
        host="claude-code",
        page_ids=["a"],
        activity_kind="read",
        config=cfg,
        store=store,
        now=102.0,
    )

    assert result["status"] == "ok"
    assert result["session_hash"] == claude_session
    assert result["host"] == "claude-code"
    assert result["stimulus_count"] == 1
    assert store.load(claude_session, now=102.0).shadow["a"].activation > 0
    assert "a" not in store.load(codex_session, now=102.0).shadow
    assert store.latest_session_hash(host="codex", now=102.0) == codex_session


def test_mcp_record_stimulates_current_working_set_without_inventing_pages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from chronovisor.core import cofire

    monkeypatch.setattr(cofire, "neighbors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(recall_field, "get_store", FakeGraphStore)
    monkeypatch.setattr(recall_field, "prompt_stimuli", lambda *_args, **_kwargs: [])
    cfg = config()
    store = RecallFieldStore(tmp_path / "field", config=cfg)
    hashed = session_hash("codex", "save-session")

    def seed(state):
        state.host = "codex"
        state.shadow["a"] = ActivationNode(activation=0.7, direct=0.7)
        state.updated_at_epoch = 100.0
        return state, []

    store.transact(hashed, seed, now=100.0)

    result = recall_field.record_mcp_content_activity(
        host="codex",
        session_id="save-session",
        content="saved payload without an exact page name",
        config=cfg,
        store=store,
        now=101.0,
    )

    assert result["status"] == "ok"
    assert result["page_ids"] == ["a"]
    assert store.load(hashed, now=101.0).shadow["a"].direct > 0.7
    assert any(row["reason_code"] == "mcp_record" for row in store.read_events(hashed))


def test_field_update_p95_is_below_50ms_and_has_no_prompt_body(
    monkeypatch,
) -> None:
    from chronovisor.core import cofire

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
