from __future__ import annotations

import hashlib
import json
import stat
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from chronovisor.core.legacy_archive import write_legacy_archive
from chronovisor.core.raw_segment import append_capture
from chronovisor.core.store import RuntimeContext, init_chronovisor
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store


def _config(root: Path, **overrides: object) -> Path:
    values = {
        "enabled": True,
        "chunk_size": 10,
        "max_input_bytes": 4096,
        "max_candidates": 20,
        "hard_floor_rallies": 100,
        "hard_floor_days": 30,
        "hard_floor_windows": 3,
        "hard_floor_teacher_labels": 100,
        "hard_floor_teacher_per_class": 10,
        "hard_floor_probe_pairs": 10,
        "hard_floor_counterfactual_pairs": 10,
        "canary_min_days": 7,
        **overrides,
    }
    lines = ["[recall.distillation]"]
    for key, value in values.items():
        text = str(value).lower() if isinstance(value, bool) else str(value)
        lines.append(f"{key} = {text}")
    lines.append("rollout_stages = [5, 25, 100]")
    path = root / "config.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _message(role: str, text: str, timestamp: str) -> dict[str, object]:
    content_type = "input_text" if role == "user" else "output_text"
    return {
        "type": "response_item",
        "timestamp": timestamp,
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": content_type, "text": text}],
        },
    }


def _raw(root: Path) -> Path:
    init_chronovisor(RuntimeContext(root))
    raw_dir = root / "raw"
    events = [
        _message("user", "alpha memory", "2026-08-01T00:00:00Z"),
        _message("assistant", "alpha evidence", "2026-08-01T00:00:01Z"),
        {
            "type": "response_item",
            "timestamp": "2026-08-01T00:00:02Z",
            "payload": {"type": "function_call", "name": "search"},
        },
        _message("assistant", "alpha detail", "2026-08-01T00:00:03Z"),
        _message("user", "alpha question", "2026-08-02T00:00:00Z"),
        _message("assistant", "alpha future", "2026-08-02T00:00:01Z"),
        _message("user", "unanswered", "2026-08-03T00:00:00Z"),
    ]
    payload = b"".join(
        json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events
    )
    source = root / "session.jsonl"
    source.write_bytes(payload)
    append_capture(
        raw_dir=raw_dir,
        raw_id="save-codex-test.md",
        idempotency_key="codex-test",
        host="codex",
        session_key="a" * 24,
        session_id="session-one",
        source_file=source,
        after_line=0,
        until_line=len(events),
        source_bytes=payload,
        record_count=len(events),
        now=datetime(2026, 8, 3, tzinfo=ZoneInfo("Asia/Tokyo")),
    )
    return raw_dir


def _baseline_identity(root: Path) -> str:
    identity, _, _ = store.write_immutable(
        store.distillation_dir(root) / "baselines",
        {"kind": "test-incumbent"},
        schema=distill.BASELINE_SCHEMA,
    )
    return identity


def test_config_is_off_by_default_and_environment_is_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.toml"
    assert not distill.distillation_enabled(missing)
    defaults = distill.load_distillation_config(missing)
    assert (defaults.chunk_size, defaults.max_input_bytes) == (25, 12_000)
    configured = _config(tmp_path)
    assert distill.distillation_enabled(configured)
    monkeypatch.setenv("CHRONOVISOR_RECALL_DISTILLATION", "false")
    assert not distill.distillation_enabled(configured)
    monkeypatch.setenv("CHRONOVISOR_RECALL_DISTILLATION", "true")
    assert distill.distillation_enabled(missing)


def test_migrate_distillation_config_dry_run_apply_and_idempotence(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    original = b'[runtime]\nsource = "keep"\n'
    config.write_bytes(original)

    dry_run = distill.migrate_distillation_config(config)
    assert dry_run["status"] == "dry_run"
    assert set(dry_run["additions"]) == {
        "recall.distillation",
        *distill._DISTILLATION_ROLES,
    }
    assert config.read_bytes() == original
    assert not config.with_name("config.toml.bak").exists()

    applied = distill.migrate_distillation_config(config, apply=True)
    parsed = tomllib.loads(config.read_text(encoding="utf-8"))
    assert applied == dry_run | {"status": "applied"}
    assert parsed["recall"]["distillation"] == distill._DISTILLATION_CONFIG
    assert parsed["llm"]["roles"] == distill._DISTILLATION_ROLES
    assert config.with_name("config.toml.bak").read_bytes() == original
    assert b"enabled = false\nchunk_size" in config.read_bytes()
    assert b"canary_min_days = 7\n\n[llm.roles" in config.read_bytes()
    assert distill.migrate_distillation_config(config) == {
        "status": "noop",
        "additions": [],
    }
    operator_enabled = config.read_text(encoding="utf-8").replace(
        "enabled = false", "enabled = true", 1
    )
    config.write_text(operator_enabled, encoding="utf-8")
    assert distill.migrate_distillation_config(config) == {
        "status": "noop",
        "additions": [],
    }


@pytest.mark.parametrize(
    "contents",
    (
        b"[recall.distillation]\nenabled = false\n",
        b'[llm.roles."recall.distill.teacher.a"]\nmodel = "wrong"\n',
    ),
)
def test_migrate_distillation_config_rejects_partial_or_conflicting_sections(
    tmp_path: Path, contents: bytes
) -> None:
    config = tmp_path / "config.toml"
    config.write_bytes(contents)

    with pytest.raises(distill.DistillationError):
        distill.migrate_distillation_config(config, apply=True)

    assert config.read_bytes() == contents
    assert not config.with_name("config.toml.bak").exists()


def test_local_worker_metadata_and_transient_failure_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama, research_scheduler

    route = SimpleNamespace(
        role="recall.distill.teacher.a",
        provider="ollama",
        model="local-model",
        location="local",
        structured_output=True,
    )
    identity = {
        "role": route.role,
        "provider": route.provider,
        "model": route.model,
        "location": route.location,
    }
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (route,))

    lane_kwargs: list[dict[str, object]] = []

    @contextmanager
    def lane(*_args: object, **kwargs: object):
        lane_kwargs.append(kwargs)
        yield object()

    monkeypatch.setattr(research_scheduler, "research_lane", lane)
    failure = {"value": ""}

    def command(
        _argv: object, encoded: str, _lease: object, **_kwargs: object
    ) -> SimpleNamespace:
        request = json.loads(encoded)
        if failure["value"]:
            value = {
                "schema": "chronovisor.recall-distillation-worker.v1",
                "ok": False,
                "operation": request["operation"],
                "role": request["role"],
                "request_id": request["request_id"],
                "route_identity": identity,
                "model_digest": "d" * 64,
                "result": {},
                "failure_class": failure["value"],
            }
        else:
            value = {
                "schema": "chronovisor.recall-distillation-worker.v1",
                "ok": True,
                "operation": request["operation"],
                "role": request["role"],
                "request_id": request["request_id"],
                "route_identity": identity,
                "model_digest": "d" * 64,
                "result": {"labels": []},
                "failure_class": "",
            }
        return SimpleNamespace(status="completed", value=value)

    monkeypatch.setattr(research_scheduler, "run_cancellable_command", command)
    result = distill._worker_call(
        "teacher",
        route.role,
        {"candidates": []},
        max_input_bytes=12_000,
        expected_route=identity,
        expected_digest="d" * 64,
    )
    assert result["_route_identity"] == identity
    assert lane_kwargs[-1]["mode"] == "sleep"
    assert lane_kwargs[-1]["purpose"] == "sleep"
    failure["value"] = "backend_error"
    with pytest.raises(distill.DistillationDeferred):
        distill._worker_call(
            "teacher",
            route.role,
            {"candidates": []},
            max_input_bytes=12_000,
            expected_route=identity,
            expected_digest="d" * 64,
        )
    failure["value"] = "output_invalid"
    with pytest.raises(distill.DistillationError, match="output"):
        distill._worker_call(
            "teacher",
            route.role,
            {"candidates": []},
            max_input_bytes=12_000,
            expected_route=identity,
            expected_digest="d" * 64,
        )


def test_default_workers_keep_cold_teacher_and_counterfactual_budgets_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    roles = (
        *distill.TEACHER_ROLES,
        "recall.distill.answer_generator",
        "recall.distill.utility_judge",
    )
    routes = tuple(
        SimpleNamespace(
            role=role,
            provider="ollama",
            model=f"model-{index}",
            location="local",
            structured_output=True,
        )
        for index, role in enumerate(roles)
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda models: {model: f"{index + 1:064x}" for index, model in enumerate(models)},
    )

    teachers, counterfactual = distill._default_workers(
        distill.DistillationConfig(),
        teacher_deadline_ms=120_000,
        counterfactual_deadline_ms=45_000,
    )

    assert {worker.deadline_ms for worker in teachers.values()} == {120_000}
    assert counterfactual is not None
    assert counterfactual.deadline_ms == 45_000


def test_rally_v1_folds_assistant_and_tool_refs_without_copying_text(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    rallies = distill.extract_rallies(raw_dir, root=tmp_path)
    assert len(rallies) == 3
    assert len(rallies[0]["actual_answer_refs"]) == 2
    assert len(rallies[0]["tool_refs"]) == 1
    assert rallies[1]["context_refs"][-1]["role"] == "assistant"
    assert rallies[2]["eligibility"]["reason"] == "missing_answer"
    assert "alpha" not in json.dumps(rallies)
    assert "session-one" not in json.dumps(rallies)
    assert "a" * 24 not in json.dumps(rallies)
    assert distill.extract_rallies(raw_dir, root=tmp_path) == rallies


def test_public_raw_watermark_preserves_receipt_inventory_bytes(tmp_path: Path) -> None:
    from chronovisor.core.raw_store import (
        RawStore,
        committed_raw_watermark,
    )
    from chronovisor.research.evidence_reconstruction import (
        committed_raw_watermark as evidence_watermark,
    )

    raw_dir = _raw(tmp_path)
    rows = []
    for unit in RawStore(raw_dir, mode="v2").iter_segment_units():
        assert unit.commit is not None
        rows.append(
            {
                "raw_id": unit.raw_id,
                "byte_range": [0, unit.length],
                "byte_coordinate_space": "logical_raw",
                "raw_sha256": unit.sha256,
                "receipt_sha256": distill.canonical_json.canonical_json_sha256_strict(
                    unit.commit.to_dict()
                ),
                "captured_at": unit.captured_at,
                "host": unit.commit.host,
                "session_key": unit.commit.session_key,
                "source_line_range": [
                    unit.commit.after_line,
                    unit.commit.until_line,
                ],
            }
        )
    expected = distill.canonical_json.canonical_json_sha256_strict(rows)
    assert committed_raw_watermark(raw_dir) == expected
    assert evidence_watermark(raw_dir) == expected


def test_rally_extraction_skips_archived_legacy_but_rejects_malformed_native(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    source_root = tmp_path / "legacy-source"
    source_root.mkdir()
    semantic_child = source_root / "semantic-child.md"
    semantic_child.write_text("archived semantic child\n", encoding="utf-8")
    day_dir = raw_dir / "2026" / "08" / "11"
    manifest = write_legacy_archive(
        [semantic_child],
        archive_path=day_dir / "legacy-part-001.tar.zst",
        captured_date="2026/08/11",
    )
    archive_path = day_dir / str(manifest["archive"])
    legacy = b"---\nraw_keywords: [historical]\n---\nLegacy transcript envelope.\n"
    append_capture(
        raw_dir=raw_dir,
        raw_id="save-codex-legacy-envelope.md",
        idempotency_key="codex-legacy-envelope",
        host="codex",
        session_key="b" * 24,
        session_id=None,
        source_file=archive_path,
        after_line=10,
        until_line=11,
        source_bytes=legacy,
        record_count=1,
        now=datetime(2026, 8, 11, tzinfo=ZoneInfo("Asia/Tokyo")),
    )
    rallies = distill.extract_rallies(raw_dir, root=tmp_path)
    assert len(rallies) == 3
    assert "Legacy transcript" not in json.dumps(rallies)

    bad_root = tmp_path / "bad"
    bad_raw = bad_root / "raw"
    init_chronovisor(RuntimeContext(bad_root))
    malformed = b"---\nnot native JSON\n"
    source = bad_root / "native.jsonl"
    source.write_bytes(malformed)
    append_capture(
        raw_dir=bad_raw,
        raw_id="save-codex-malformed.md",
        idempotency_key="codex-malformed",
        host="codex",
        session_key="c" * 24,
        session_id=None,
        source_file=source,
        after_line=0,
        until_line=1,
        source_bytes=malformed,
        record_count=1,
        now=datetime(2026, 8, 11, tzinfo=ZoneInfo("Asia/Tokyo")),
    )
    with pytest.raises(distill.DistillationError, match="invalid JSON"):
        distill.extract_rallies(bad_raw, root=bad_root)


def test_historical_fts_is_assistant_only_and_strictly_point_in_time(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    index = tmp_path / "runtime" / "recall-distillation" / "historical.sqlite"
    first_digest = distill.build_historical_index(raw_dir, index)
    first_inode = index.stat().st_ino
    second_digest = distill.build_historical_index(raw_dir, index)
    assert first_digest == second_digest
    assert index.stat().st_ino == first_inode
    assert stat.S_IMODE(index.stat().st_mode) == 0o600
    rally = distill.extract_rallies(raw_dir, root=tmp_path)[1]
    snapshot = distill.candidate_snapshot(
        index,
        rally,
        "alpha",
        limit=20,
        candidate_texts={
            hashlib.sha256(text.encode()).hexdigest(): text
            for text in ("alpha evidence", "alpha detail", "alpha future")
        },
    )
    digests = {row["text_sha256"] for row in snapshot["candidates"]}
    assert hashlib.sha256(b"alpha evidence").hexdigest() in digests
    assert hashlib.sha256(b"alpha detail").hexdigest() in digests
    assert hashlib.sha256(b"alpha future").hexdigest() not in digests
    assert all("text" not in row for row in snapshot["candidates"])
    assert all(
        row["feature_revision"] == distill.TEXT_FEATURE_REVISION
        and set(row["features"]) == set(distill.FAST_FEATURE_KEYS)
        and len(row["candidate_feature_text_sha256"]) == 64
        for row in snapshot["candidates"]
    )
    assert len(snapshot["query_feature_text_sha256"]) == 64


def test_historical_cutoff_rejects_later_time_even_with_earlier_source_index(
    tmp_path: Path,
) -> None:
    path = tmp_path / "historical.sqlite"
    atoms = [
        {
            "atom_id": "future",
            "host": "codex",
            "session_cluster_id": "s",
            "source_index": 1,
            "timestamp_us": 200,
            "text_sha256": "a" * 64,
            "ref": {"raw_id": "future"},
            "text": "future marker",
        },
        {
            "atom_id": "same-time-prior",
            "host": "codex",
            "session_cluster_id": "s",
            "source_index": 2,
            "timestamp_us": 100,
            "text_sha256": "b" * 64,
            "ref": {"raw_id": "prior"},
            "text": "prior marker",
        },
    ]
    store.create_historical_index(path, atoms)
    assert not store.search_historical_index(
        path,
        query="future",
        as_of_us=100,
        host="codex",
        session_cluster_id="s",
        source_index=3,
        limit=10,
    )
    assert (
        store.search_historical_index(
            path,
            query="prior",
            as_of_us=100,
            host="codex",
            session_cluster_id="s",
            source_index=3,
            limit=10,
        )[0]["candidate_id"]
        == "same-time-prior"
    )


def test_historical_index_finds_whitespace_free_japanese(tmp_path: Path) -> None:
    path = tmp_path / "historical.sqlite"
    store.create_historical_index(
        path,
        [
            {
                "atom_id": "jp",
                "host": "codex",
                "session_cluster_id": "s",
                "source_index": 1,
                "timestamp_us": 1,
                "text_sha256": "c" * 64,
                "ref": {"raw_id": "jp"},
                "text": "クロノバイザーの検索精度を改善する",
            }
        ],
    )
    rows = store.search_historical_index(
        path,
        query="検索精度",
        as_of_us=2,
        host="codex",
        session_cluster_id="other",
        source_index=1,
        limit=10,
    )
    assert rows[0]["candidate_id"] == "jp"


def test_context_is_a_fixed_event_suffix_and_full_prefix_is_only_a_digest(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    rally = distill.extract_rallies(raw_dir, root=tmp_path, max_context_bytes=5)[1]
    assert rally["context_refs"] == []
    assert rally["full_context"]["event_count"] > 0
    assert len(rally["full_context"]["refs_sha256"]) == 64


def test_assignment_label_authority_and_live_feature_boundary() -> None:
    assignment = distill.teacher_assignment("rally", "candidate")
    assert assignment == distill.teacher_assignment("rally", "candidate")
    assert assignment["owner"] in distill.TEACHER_ROLES
    assert (
        distill.adjudicate_label("helpful", closed_predicate="exact_claim_supported")[
            "authority"
        ]
        == "teacher-only"
    )
    assert (
        distill.adjudicate_label("helpful", closed_predicate="exact_claim_supported")[
            "authority"
        ]
        == "teacher-only"
    )
    assert (
        distill.adjudicate_label("helpful", closed_predicate="exact_test_outcome")[
            "authority"
        ]
        == "teacher-only"
    )
    with pytest.raises(distill.DistillationError, match="not whitelisted"):
        distill.build_fast_features({"answer_delta": 1})
    features = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=0.4
    )
    policy = distill.train_tiny_policy(
        [{"features": features, "verdict": "helpful", "authority": "verified"}]
    )
    assert 0 <= distill.score_fast_features(features, policy) <= 1
    assert distill.policy_decision(0.8, policy, runner_up_score=0.2) == {
        "decision": "read",
        "max_cards": 3,
    }
    teacher_policy = distill.train_tiny_policy(
        [
            {
                "rally_id": "r",
                "candidate_id": "c",
                "dimension": "relevance",
                "features": features,
                "verdict": "relevant",
                "authority": "teacher-only",
            }
        ]
    )
    assert teacher_policy["training_rows"] == 1
    assert teacher_policy["weights"] != distill.train_tiny_policy([])["weights"]


def test_model_lane_scheduler_is_fair_and_reserves_counterfactual_turn() -> None:
    pending = {role: [{}] for role in distill.TEACHER_ROLES}
    labels: list[dict[str, str]] = []
    visited = []
    for _ in range(3):
        route = distill._ordered_teacher_routes(pending, labels)[0]
        visited.append(route)
        labels.append({"route": route})
    assert visited == list(distill.TEACHER_ROLES)
    assert not distill._is_counterfactual_turn(2, 0, available=True)
    assert distill._is_counterfactual_turn(3, 0, available=True)
    assert not distill._is_counterfactual_turn(3, 1, available=True)
    assert not distill._is_counterfactual_turn(6, 1, available=False)


def test_exposure_receipt_is_exact_prospective_and_hash_chained(tmp_path: Path) -> None:
    digest = "d" * 64
    receipt = distill.record_exposure(
        decision_id="decision",
        host="codex",
        session_id="session-one",
        prompt_hash="prompt",
        policy_id="policy",
        candidate_ids=["a", "b"],
        candidate_snapshot_sha256=digest,
        observed_at="2026-08-14T00:00:00Z",
        root=tmp_path,
    )
    assert "session-one" not in json.dumps(receipt)
    path = store.distillation_dir(tmp_path) / "exposure-receipts.jsonl"
    assert store.verify_chain(path)["records"] == 1
    path.write_text(path.read_text().replace('"policy"', '"tampered"'))
    with pytest.raises(store.DistillationStoreError, match="chain mismatch"):
        store.verify_chain(path)


def test_exposure_receipt_retry_is_atomic_and_conflicts_fail(tmp_path: Path) -> None:
    def write() -> dict[str, object]:
        return distill.record_exposure(
            decision_id="one-decision",
            host="codex",
            session_id="session",
            prompt_hash="prompt",
            policy_id="policy",
            candidate_ids=["candidate"],
            candidate_snapshot_sha256="d" * 64,
            observed_at="2026-08-14T00:00:00Z",
            root=tmp_path,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda _index: write(), range(8)))
    assert len({row["record_sha256"] for row in rows}) == 1
    later = distill.record_exposure(
        decision_id="one-decision",
        host="codex",
        session_id="session",
        prompt_hash="prompt",
        policy_id="policy",
        candidate_ids=["candidate"],
        candidate_snapshot_sha256="d" * 64,
        observed_at="2026-08-14T00:00:01Z",
        root=tmp_path,
    )
    assert later["record_sha256"] == rows[0]["record_sha256"]
    path = store.distillation_dir(tmp_path) / "exposure-receipts.jsonl"
    assert store.verify_chain(path)["records"] == 1
    with pytest.raises(store.DistillationStoreError, match="identity conflict"):
        distill.record_exposure(
            decision_id="one-decision",
            host="codex",
            session_id="session",
            prompt_hash="different",
            policy_id="policy",
            candidate_ids=[],
            candidate_snapshot_sha256="d" * 64,
            observed_at="2026-08-14T00:00:00Z",
            root=tmp_path,
        )


def test_nonblocking_page_and_exact_receipts_write_nothing_when_busy(
    tmp_path: Path,
) -> None:
    identity = _baseline_identity(tmp_path)
    runtime_dir = store.distillation_dir(tmp_path)
    ledger = runtime_dir / "exposure-receipts.jsonl"
    lock = store.acquire_nonblocking_lock(ledger.with_suffix(".jsonl.lock"))
    assert lock is not None
    try:
        page = distill.record_exposure(
            decision_id="busy-page",
            host="codex",
            session_id="session",
            prompt_hash="prompt",
            policy_id=identity,
            candidate_ids=[],
            candidate_snapshot_sha256="a" * 64,
            observed_at="2026-08-14T00:00:00Z",
            nonblocking=True,
            root=tmp_path,
        )
        exact = distill.record_exact_exposure(
            decision_id="busy-exact",
            host="codex",
            session_id="session",
            query_semantic_sha256="b" * 64,
            policy_id=identity,
            candidate_refs=[],
            render_sha256="c" * 64,
            candidate_snapshot_sha256="d" * 64,
            observed_at="2026-08-14T00:00:00Z",
            nonblocking=True,
            root=tmp_path,
        )
    finally:
        store.release_lock(lock)
    assert page == exact == {
        "status": "deferred",
        "reason": "receipt_ledger_busy",
    }
    assert not ledger.exists()
    assert not (runtime_dir / "exposures").exists()
    artifact_lock = store.acquire_nonblocking_lock(
        runtime_dir / "exposures" / ".immutable.lock"
    )
    assert artifact_lock is not None
    try:
        artifact_busy = distill.record_exact_exposure(
            decision_id="busy-artifact",
            host="codex",
            session_id="session",
            query_semantic_sha256="b" * 64,
            policy_id=identity,
            candidate_refs=[],
            render_sha256="c" * 64,
            candidate_snapshot_sha256="d" * 64,
            observed_at="2026-08-14T00:00:00Z",
            nonblocking=True,
            root=tmp_path,
        )
    finally:
        store.release_lock(artifact_lock)
    assert artifact_busy == {
        "status": "deferred",
        "reason": "receipt_ledger_busy",
    }
    assert not ledger.exists()
    assert list((runtime_dir / "exposures").glob("*.json")) == []


def test_exposure_join_requires_one_receipt_bound_inside_the_rally(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    rally = distill.extract_rallies(raw_dir, root=tmp_path)[1]
    identity = _baseline_identity(tmp_path)
    assert not rally["eligibility"]["answer_utility"]
    evidence_ref = distill.extract_rallies(raw_dir, root=tmp_path)[0][
        "actual_answer_refs"
    ][0]
    distill.record_exact_exposure(
        decision_id="inside",
        host="codex",
        session_id="session-one",
        query_semantic_sha256=rally["query_sha256"],
        policy_id=identity,
        candidate_refs=[
            {
                "candidate_id": "candidate",
                "content_sha256": evidence_ref["semantic_sha256"],
                "evidence_refs": [evidence_ref],
            }
        ],
        render_sha256="f" * 64,
        candidate_snapshot_sha256="e" * 64,
        observed_at="2026-08-02T00:00:00.500000Z",
        root=tmp_path,
    )
    joined = distill.extract_rallies(raw_dir, root=tmp_path)[1]
    assert joined["eligibility"]["answer_utility"]
    distill.record_exact_exposure(
        decision_id="duplicate",
        host="codex",
        session_id="session-one",
        query_semantic_sha256=rally["query_sha256"],
        policy_id=identity,
        candidate_refs=[
            {
                "candidate_id": "candidate",
                "content_sha256": evidence_ref["semantic_sha256"],
                "evidence_refs": [evidence_ref],
            }
        ],
        render_sha256="f" * 64,
        candidate_snapshot_sha256="e" * 64,
        observed_at="2026-08-02T00:00:00.600000Z",
        root=tmp_path,
    )
    ambiguous = distill.extract_rallies(raw_dir, root=tmp_path)[1]
    assert ambiguous["eligibility"]["reason"] == "ambiguous_exact_exposure"


def test_exact_page_exposure_seals_canonical_live_features(tmp_path: Path) -> None:
    identity = _baseline_identity(tmp_path)
    features = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=0.25
    )
    rendered = "exact rendered card"
    receipt = distill.record_exact_exposure(
        decision_id="page-decision",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="a" * 64,
        policy_id=identity,
        candidate_refs=[
            {
                "candidate_id": "page-v1",
                "page_id": "page",
                "page_content_sha256": "9" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            }
        ],
        candidate_feature_snapshot=[
            {"candidate_id": "page-v1", "features": features},
            {"candidate_id": "unselected", "features": distill.build_fast_features()},
        ],
        candidate_pool_refs=[
            {
                "candidate_id": "page-v1",
                "selected": True,
                "page_id": "page",
                "page_content_sha256": "9" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            },
            {
                "candidate_id": "unselected",
                "selected": False,
                "page_id": "other",
                "page_content_sha256": "8" * 64,
                "rendered_context": "other context",
                "rendered_context_sha256": hashlib.sha256(b"other context").hexdigest(),
            },
        ],
        render_sha256="b" * 64,
        candidate_snapshot_sha256="c" * 64,
        observed_at="2026-08-14T00:00:00Z",
        root=tmp_path,
    )
    artifact_path = (
        store.distillation_dir(tmp_path)
        / "exposures"
        / f"{receipt['exposure_artifact_id']}.json"
    )
    artifact = store.read_sealed(
        artifact_path, schema="chronovisor.recall-exact-exposure.v1"
    )
    assert artifact["candidate_feature_snapshot"][0]["features"] == features
    assert stat.S_IMODE(artifact_path.stat().st_mode) == 0o600
    assert "private-session" not in json.dumps(receipt)
    empty = distill.record_exact_exposure(
        decision_id="empty-et",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="e" * 64,
        policy_id=identity,
        candidate_refs=[],
        candidate_feature_snapshot=[
            {"candidate_id": "unselected", "features": distill.build_fast_features()}
        ],
        candidate_pool_refs=[
            {
                "candidate_id": "unselected",
                "selected": False,
                "page_id": "other",
                "page_content_sha256": "8" * 64,
                "rendered_context": "other context",
                "rendered_context_sha256": hashlib.sha256(b"other context").hexdigest(),
            }
        ],
        render_sha256="f" * 64,
        candidate_snapshot_sha256="0" * 64,
        observed_at="2026-08-14T00:00:01Z",
        root=tmp_path,
    )
    assert empty["candidate_ids"] == []
    with pytest.raises(distill.DistillationError, match="source version"):
        distill.record_exact_exposure(
            decision_id="version-mismatch",
            host="codex",
            session_id="private-session",
            query_semantic_sha256="7" * 64,
            policy_id=identity,
            candidate_refs=[
                {
                    "candidate_id": "page-v1",
                    "page_id": "page",
                    "page_content_sha256": "9" * 64,
                    "rendered_context": rendered,
                    "rendered_context_sha256": hashlib.sha256(
                        rendered.encode()
                    ).hexdigest(),
                }
            ],
            candidate_feature_snapshot=[
                {"candidate_id": "page-v1", "features": features}
            ],
            candidate_pool_refs=[
                {
                    "candidate_id": "page-v1",
                    "selected": True,
                    "page_id": "page",
                    "page_content_sha256": "7" * 64,
                    "rendered_context": rendered,
                    "rendered_context_sha256": hashlib.sha256(
                        rendered.encode()
                    ).hexdigest(),
                }
            ],
            render_sha256="6" * 64,
            candidate_snapshot_sha256="5" * 64,
            observed_at="2026-08-14T00:00:02Z",
            root=tmp_path,
        )
    with pytest.raises(distill.DistillationError, match="canonical"):
        distill.record_exact_exposure(
            decision_id="bad",
            host="codex",
            session_id="private-session",
            query_semantic_sha256="a" * 64,
            policy_id=identity,
            candidate_refs=[],
            candidate_feature_snapshot=[
                {
                    "candidate_id": "bad",
                    "features": {"query_chargram_coverage": 2},
                }
            ],
            render_sha256="b" * 64,
            candidate_snapshot_sha256="c" * 64,
            observed_at="2026-08-14T00:00:00Z",
            root=tmp_path,
        )


def test_structural_verifier_accepts_exact_anchor_not_near_match() -> None:
    commit = "a" * 40
    rally = {"query_ref": {"structural": {"commit": [commit], "path": []}}}
    exact = {"ref": {"structural": {"commit": [commit], "path": []}}}
    near = {"ref": {"structural": {"commit": [commit[:-1] + "b"], "path": []}}}
    assert (
        distill._default_structural_verifier(rally, exact, {}) == "exact_commit_overlap"
    )
    assert distill._default_structural_verifier(rally, near, {}) is None
    label = distill._teacher_label(
        {"verdict": "irrelevant"},
        verified_predicate="exact_commit_overlap",
    )
    assert label["authority"] == "teacher-only"


def test_grouped_rolling_split_never_separates_a_session() -> None:
    rows = [
        {
            "rally_id": f"r{index}",
            "session_cluster_id": f"s{index // 2}",
            "as_of": f"2026-08-{index + 1:02}T00:00:00Z",
        }
        for index in range(12)
    ]
    split = distill.grouped_rolling_split(rows)
    for index in range(0, 12, 2):
        assert split[f"r{index}"] == split[f"r{index + 1}"]
    assert {"train", "validation", "test"}.issubset(set(split.values()))


def test_sealed_policy_pointer_and_nested_rollout_selection(tmp_path: Path) -> None:
    _config(tmp_path)
    policy = distill.train_tiny_policy([])
    candidate = distill.publish_policy(
        policy, lineage={"ledger_head": "x"}, root=tmp_path
    )
    candidate_id = candidate["artifact_id"]
    store.write_pointer(tmp_path, "active", candidate_id)
    store.write_pointer(tmp_path, "lkg", candidate_id)
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": "canary",
            "worker_status": "idle",
            "rollout_percent": 25,
            "stage_started_at": "2026-08-01T00:00:00Z",
        },
    )
    assert distill.load_active_policy(tmp_path)["artifact_id"] == candidate_id
    assert (
        distill.load_policy_for_session("session", tmp_path)["artifact_id"]
        == candidate_id
    )
    health = store.snapshot(tmp_path)
    assert health["rollout"] == 25
    assert health["active_policy_id"] == candidate_id[:12]


def test_bootstrap_is_automatic_and_never_replaces_legacy_serving(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    active_id = store.read_pointer(tmp_path, "active")["policy_id"]
    assert store.read_pointer(tmp_path, "lkg")["policy_id"] == active_id
    bootstrap = store.read_sealed(
        store.distillation_dir(tmp_path) / "policies" / f"{active_id}.json",
        schema=distill.POLICY_SCHEMA,
    )
    assert bootstrap["serve_mode"] == "legacy"
    assert distill.load_active_policy(tmp_path) == {}

    candidate = distill.publish_policy(
        distill.train_tiny_policy([]), lineage={"ledger_head": "x"}, root=tmp_path
    )
    candidate_id = candidate["artifact_id"]
    state_path = store.distillation_dir(tmp_path) / store.STATE_FILE
    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "status": "shadow",
            "rollout_percent": 0,
            "learning_halted": False,
        },
    )
    assert distill.load_policy_for_session("shadow", tmp_path) == {}

    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "status": "canary",
            "rollout_percent": 5,
            "learning_halted": False,
        },
    )
    selected = []
    legacy = []
    for index in range(200):
        value = distill.load_policy_for_session(f"session-{index}", tmp_path)
        (selected if value else legacy).append(value)
    assert any(value.get("artifact_id") == candidate_id for value in selected)
    assert legacy

    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "status": "rolled_back",
            "rollout_percent": 0,
            "learning_halted": True,
            "lkg_policy_id": active_id,
        },
    )
    assert distill.load_policy_for_session("rollback", tmp_path) == {}
    assert distill.load_active_policy(tmp_path) == {}


def test_reserved_store_fields_are_rejected_without_corrupting_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    store.append_chain(path, {"kind": "safe"})
    with pytest.raises(store.DistillationStoreError, match="reserved"):
        store.append_chain(path, {"previous_sha256": "forged"})
    assert store.verify_chain(path)["records"] == 1
    with pytest.raises(store.DistillationStoreError, match="reserved"):
        store.write_sealed_state(tmp_path / "state.json", {"schema": "forged"})


def test_preflight_and_chunk_are_deterministic_capture_only_below_floor(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    first = distill.preflight(
        raw_dir=raw_dir, root=tmp_path, config_path=config, runtime_commit="abcdef0"
    )
    second = distill.preflight(
        raw_dir=raw_dir, root=tmp_path, config_path=config, runtime_commit="abcdef0"
    )
    assert first == second
    assert not first["hard_floor"]["p5_allowed"]
    result = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    assert result["status"] == "capture_only"
    assert result["processed"] == 3
    retry = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    assert retry["processed"] == 0
    assert (
        store.verify_chain(store.distillation_dir(tmp_path) / "rally-manifest.jsonl")[
            "records"
        ]
        == 3
    )


def test_matching_p5_baseline_rejects_current_hold_and_runtime_drift(
    tmp_path: Path,
) -> None:
    _, _, baseline = store.write_immutable(
        store.distillation_dir(tmp_path) / "baselines",
        {
            "kind": "privacy-safe-baseline",
            "raw_watermark": "a" * 64,
            "config_sha256": "b" * 64,
            "runtime_commit": "abcdef0",
            "metrics": {"archive_commit": "abcdef0", "drift": False},
            "frozen_contract": {"feature_revision": distill.TEXT_FEATURE_REVISION},
            "offline_training_gate": {"passed": True},
            "hard_floor": {"p5_allowed": True, "reasons": []},
        },
        schema=distill.BASELINE_SCHEMA,
    )
    assert distill._matching_p5_baseline(tmp_path, baseline) == baseline
    held = {**baseline, "hard_floor": {"p5_allowed": False, "reasons": ["drift"]}}
    assert distill._matching_p5_baseline(tmp_path, held) is None
    changed_commit = {**baseline, "runtime_commit": "1234567"}
    assert distill._matching_p5_baseline(tmp_path, changed_commit) is None
    changed_metrics = {**baseline, "metrics": {"archive_commit": "abcdef0", "drift": True}}
    assert distill._matching_p5_baseline(tmp_path, changed_metrics) is None
    changed_contract = {**baseline, "frozen_contract": {"feature_revision": "other"}}
    assert distill._matching_p5_baseline(tmp_path, changed_contract) is None


def test_chunk_hard_timeout_preserves_state_counters_and_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall.recall_runtime import RecallWallClockTimeout

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    runtime_dir = store.distillation_dir(tmp_path)
    state_path = runtime_dir / store.STATE_FILE
    ledger_path = runtime_dir / "rally-manifest.jsonl"
    before_state = state_path.read_bytes()
    before_ledger = ledger_path.read_bytes()
    before_head = store.verify_chain(ledger_path)
    monkeypatch.setattr(
        distill,
        "_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RecallWallClockTimeout("forced pre-write timeout")
        ),
    )
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        max_elapsed_seconds=60,
    )
    assert result == {
        "status": "deferred",
        "processed": 0,
        "reason": "wall_clock_timeout",
        "atomic_progress_may_be_present": True,
    }
    assert state_path.read_bytes() == before_state
    assert ledger_path.read_bytes() == before_ledger
    assert store.verify_chain(ledger_path) == before_head
    lock = store.acquire_nonblocking_lock(runtime_dir / "distillation-worker.lock")
    assert lock is not None
    store.release_lock(lock)


def test_timeout_after_atomic_batch_resumes_without_duplicate_or_cursor_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall.recall_runtime import RecallWallClockTimeout

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    runtime_dir = store.distillation_dir(tmp_path)
    state_path = runtime_dir / store.STATE_FILE
    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "status": "capture_only",
            "rollout_percent": 0,
            "raw_watermark": "0" * 64,
            "cold_start_lane_turn": 7,
            "teacher_model_calls": 5,
            "counterfactual_model_calls": 2,
        },
    )
    before_state = state_path.read_bytes()
    build_index = distill.build_historical_index
    monkeypatch.setattr(
        distill,
        "build_historical_index",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RecallWallClockTimeout("forced after manifest batch")
        ),
    )
    timed_out = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        cold_start=True,
        max_elapsed_seconds=60,
    )
    assert timed_out == {
        "status": "deferred",
        "processed": 0,
        "reason": "wall_clock_timeout",
        "atomic_progress_may_be_present": True,
    }
    assert state_path.read_bytes() == before_state
    manifest = runtime_dir / "rally-manifest.jsonl"
    committed = store.verify_chain(manifest)
    assert committed["records"] == 3

    monkeypatch.setattr(distill, "build_historical_index", build_index)
    resumed = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        cold_start=True,
        max_elapsed_seconds=300,
    )
    assert resumed["processed"] == 0
    assert store.verify_chain(manifest) == committed
    state = store.read_sealed(state_path)
    assert state["teacher_model_calls"] == 5
    assert state["counterfactual_model_calls"] == 2
    assert state["cold_start_lane_turn"] == 8
    assert state["raw_watermark"] == distill.committed_raw_watermark(raw_dir)


def test_final_state_is_last_commit_and_binds_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall.recall_runtime import RecallWallClockTimeout

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    runtime_dir = store.distillation_dir(tmp_path)
    state_path = runtime_dir / store.STATE_FILE
    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "status": "capture_only",
            "rollout_percent": 0,
            "raw_watermark": "0" * 64,
            "cold_start_lane_turn": 4,
            "teacher_model_calls": 2,
            "counterfactual_model_calls": 1,
        },
    )
    before_state = state_path.read_bytes()
    write_immutable = store.write_immutable

    def timeout_before_run_commit(
        directory: Path, *args: object, **kwargs: object
    ) -> object:
        if directory.name == "runs":
            raise RecallWallClockTimeout("former run boundary")
        return write_immutable(directory, *args, **kwargs)

    monkeypatch.setattr(store, "write_immutable", timeout_before_run_commit)
    deferred = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        max_elapsed_seconds=60,
    )
    assert deferred["status"] == "deferred"
    assert deferred["atomic_progress_may_be_present"] is True
    assert state_path.read_bytes() == before_state

    monkeypatch.setattr(store, "write_immutable", write_immutable)
    setitimer = distill.signal.setitimer
    deadline_cancelled = {"value": False}

    def track_timer(which: int, seconds: float, *args: object) -> object:
        if which == distill.signal.ITIMER_REAL and seconds == 0:
            deadline_cancelled["value"] = True
        return setitimer(which, seconds, *args)

    write_state = store.write_sealed_state

    def require_cancelled(path: Path, payload: object) -> dict[str, object]:
        artifact = write_state(path, payload)  # type: ignore[arg-type]
        if path == state_path and not deadline_cancelled["value"]:
            raise RecallWallClockTimeout("injected immediately after state")
        return artifact

    monkeypatch.setattr(distill.signal, "setitimer", track_timer)
    monkeypatch.setattr(store, "write_sealed_state", require_cancelled)
    completed = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        max_elapsed_seconds=60,
    )
    assert completed["status"] != "deferred"
    state = store.read_sealed(state_path)
    assert state["run_id"] == completed["run_id"]


def test_chunk_parses_committed_raw_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    original_events = distill._events
    calls = 0

    def counted_events(path: Path) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        return original_events(path)

    monkeypatch.setattr(distill, "_events", counted_events)
    result = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    assert result["processed"] == 3
    assert calls == 1


def test_preflight_automatically_aggregates_safe_runtime_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    commit = "a" * 40
    monkeypatch.setattr(
        distill.runtime_config,
        "runtime_identity",
        lambda: {
            "commit_id": commit,
            "expected_commit": commit,
            "drift": False,
        },
    )
    log_path = tmp_path / "recall" / "recall-log.jsonl"
    log_path.parent.mkdir(parents=True)
    rows = [
        {
            "decision_id": "one",
            "stage": "injected",
            "decision": "read",
            "latency_ms": 100,
            "status": "ok",
            "prompt_preview": "must never enter baseline",
        },
        {
            "decision_id": "two",
            "stage": "decision",
            "decision": "none",
            "latency_ms": 300,
            "status": "timeout",
            "session_id": "private-session",
        },
    ]
    log_path.write_text(
        json.dumps(rows[0]) + "\n{malformed\n" + json.dumps(rows[1]) + "\n",
        encoding="utf-8",
    )
    baseline = distill.preflight(raw_dir=raw_dir, root=tmp_path, config_path=config)
    metrics = baseline["metrics"]
    assert metrics["archive_commit"] == commit
    assert metrics["expected_commit"] == commit
    assert metrics["drift"] is False
    assert metrics["coverage_rate"] == 0.5
    assert metrics["abstain_rate"] == 0.5
    assert metrics["latency_p50_ms"] == 100
    assert metrics["latency_p95_ms"] == 300
    assert metrics["timeout_rate"] == 0.5
    assert metrics["wrong_domain_rate"] is None
    assert metrics["exact_outcome_links"] == 0
    assert not baseline["hard_floor"]["p5_allowed"]
    assert "teacher_labels_below_floor" in baseline["hard_floor"]["reasons"]
    assert "counterfactual_pairs_below_floor" in baseline["hard_floor"]["reasons"]
    serialized = json.dumps(baseline)
    assert "must never enter baseline" not in serialized
    assert "private-session" not in serialized

    overridden = distill.preflight(
        raw_dir=raw_dir,
        root=tmp_path,
        config_path=config,
        aggregate_metrics={"coverage_rate": 0.75},
    )
    assert overridden["metrics"]["coverage_rate"] == 0.75
    assert overridden["metrics"]["latency_p95_ms"] == 300


def test_p1_to_p4_teacher_backfill_runs_while_p5_is_held(tmp_path: Path) -> None:
    class FakeTeacher:
        local = True

        def __init__(self, role: str) -> None:
            self.role = role

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "relevant",
                        "rationale": "bounded fake",
                    }
                    for candidate in payload["candidates"]
                ]
            }

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    teachers = {role: FakeTeacher(role) for role in distill.TEACHER_ROLES}
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers=teachers,
    )
    assert result["status"] == "capture_only"
    assert result["candidate_snapshots"] == 3
    assert result["labels_written"] > 0
    labels = store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl")
    assert all(row["authority"] in {"teacher-only", "verified"} for row in labels)
    assert all("reason" not in row and "rationale" not in row for row in labels)


def test_counterfactual_turn_without_real_work_falls_back_to_teacher(
    tmp_path: Path,
) -> None:
    class Teacher:
        local = True

        def __init__(self, role: str) -> None:
            self.role = role

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": item["candidate_id"],
                        "verdict": "uncertain",
                        "rationale": "bounded",
                    }
                    for item in payload["candidates"]
                ]
            }

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": "capture_only",
            "rollout_percent": 0,
            "teacher_model_calls": 3,
            "counterfactual_model_calls": 0,
        },
    )
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={role: Teacher(role) for role in distill.TEACHER_ROLES},
    )
    assert result["labels_written"] > 0


def test_teacher_routes_make_progress_across_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Teacher:
        local = True

        def __init__(self, role: str) -> None:
            self.role = role

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": item["candidate_id"],
                        "verdict": "uncertain",
                        "rationale": "bounded",
                    }
                    for item in payload["candidates"]
                ]
            }

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)

    def assigned(rally_id: str, candidate_id: str) -> dict[str, object]:
        del rally_id, candidate_id
        return {
            "revision": distill.ASSIGNMENT_REVISION,
            "owner": distill.TEACHER_ROLES[0],
            "probe_revision": distill.PROBE_REVISION,
            "probe": True,
            "routes": list(distill.TEACHER_ROLES),
        }

    monkeypatch.setattr(distill, "teacher_assignment", assigned)
    teachers = {role: Teacher(role) for role in distill.TEACHER_ROLES}
    for _ in range(6):
        distill.run_distillation_chunk(
            root=tmp_path,
            raw_dir=raw_dir,
            config_path=config,
            teachers=teachers,
        )
    labels = store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl")
    counts = {
        role: sum(row.get("route") == role for row in labels)
        for role in distill.TEACHER_ROLES
    }
    assert all(count > 0 for count in counts.values())
    assert max(counts.values()) - min(counts.values()) <= 16


def test_transient_teacher_defer_writes_no_label_and_retries(tmp_path: Path) -> None:
    class DeferredTeacher:
        local = True

        def __init__(self, role: str) -> None:
            self.role = role

        def evaluate(self, _payload: object) -> dict[str, object]:
            raise distill.DistillationDeferred("foreground")

    class WorkingTeacher(DeferredTeacher):
        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "uncertain",
                        "rationale": "retry",
                    }
                    for candidate in payload["candidates"]
                ]
            }

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    deferred = {role: DeferredTeacher(role) for role in distill.TEACHER_ROLES}
    first = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers=deferred
    )
    assert first["status"] == "deferred"
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    assert store.read_chain(label_path) == []
    working = {role: WorkingTeacher(role) for role in distill.TEACHER_ROLES}
    second = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers=working
    )
    assert second["labels_written"] > 0


def test_timeout_teacher_is_deferred_without_advancing_label_cursor(
    tmp_path: Path,
) -> None:
    class TimeoutTeacher:
        local = True

        def __init__(self, role: str) -> None:
            self.role = role

        def evaluate(self, _payload: object) -> dict[str, object]:
            raise TimeoutError("temporary backend timeout")

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    teachers = {role: TimeoutTeacher(role) for role in distill.TEACHER_ROLES}
    result = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers=teachers
    )
    assert result["status"] == "deferred"
    assert (
        store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl") == []
    )


def test_chunk_preserves_rollout_traffic_state(tmp_path: Path) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    state_path = store.distillation_dir(tmp_path) / store.STATE_FILE
    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "status": "canary",
            "rollout_percent": 25,
            "stage_started_at": "2026-08-01T00:00:00Z",
        },
    )
    distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    state = store.read_sealed(state_path)
    assert (state["status"], state["rollout_percent"]) == ("canary", 25)
    assert state["worker_status"] == "capture_only"


def test_automatic_rollout_holds_and_ignores_caller_supplied_metrics(
    tmp_path: Path,
) -> None:
    _config(tmp_path)
    policy = distill.train_tiny_policy([])

    def write_policy(name: str) -> tuple[str, dict[str, object]]:
        return_value = store.write_immutable(
            store.distillation_dir(tmp_path) / "policies",
            {
                "kind": "tiny-logistic-policy",
                **policy,
                "lineage": {
                    "locked_replay_id": hashlib.sha256(name.encode()).hexdigest()
                },
            },
            schema=distill.POLICY_SCHEMA,
        )
        return return_value[0], return_value[2]

    incumbent_id, _ = write_policy("incumbent")
    candidate_id, candidate = write_policy("candidate")
    store.write_pointer(tmp_path, "active", incumbent_id)
    store.write_pointer(tmp_path, "lkg", incumbent_id)
    store.write_pointer(tmp_path, "candidate", candidate_id)
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": "replay",
            "rollout_percent": 0,
            "stage_started_at": "2026-08-01T00:00:00Z",
        },
    )
    _, _, baseline = store.write_immutable(
        store.distillation_dir(tmp_path) / "baselines",
        {
            "kind": "test-baseline",
            "raw_watermark": "a" * 64,
            "hard_floor": {"p5_allowed": True, "reasons": []},
            "offline_training_gate": {"passed": True, "revision": "test-v2"},
        },
        schema=distill.BASELINE_SCHEMA,
    )
    evaluation_dir = store.distillation_dir(tmp_path) / "evaluations"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / f"{'f' * 64}.json").write_text("{}\n")
    first = distill._automatic_rollout_evaluation(
        tmp_path, baseline, {"status": "candidate", "policy_id": candidate_id}
    )
    assert first["status"] == "shadow"
    gate = {
        "denominator": 500,
        "min_denominator": 500,
        "min_days": 7,
        "ci_lower": 1.0,
        "min_ci_lower": 0.9,
    }
    metrics = {
        name: dict(gate)
        for name in (
            "coverage_abstain",
            "latency_timeout",
            "cohort_delta",
            "feature_parity",
        )
    }
    run_id = "b" * 64
    store.write_immutable(
        store.distillation_dir(tmp_path) / "evaluations",
        {
            "kind": "runtime-measured-metrics",
            "run_id": run_id,
            "policy_id": candidate_id,
            "baseline_id": baseline["artifact_id"],
            "raw_watermark": baseline["raw_watermark"],
            "incumbent_policy_id": incumbent_id,
            "split_sha256": candidate["lineage"]["locked_replay_id"],
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "feature_parity_sha256": "c" * 64,
            "offline_gate_sha256": distill.canonical_json.canonical_json_sha256_strict(
                baseline["offline_training_gate"]
            ),
            "observation_mode": "paired",
            "replay_metrics": metrics,
            "shadow_metrics": metrics,
            "canary_metrics": metrics,
        },
        schema="chronovisor.recall-distill-rollout-evaluation.v2",
    )
    held = distill._automatic_rollout_evaluation(
        tmp_path, baseline, {"status": "candidate", "policy_id": candidate_id}
    )
    assert held["status"] == "shadow"
    state = store.read_sealed(store.distillation_dir(tmp_path) / store.STATE_FILE)
    assert state["status"] == "shadow"


def test_qualified_shadow_policy_records_private_operational_receipt(
    tmp_path: Path,
) -> None:
    from chronovisor.recall import recall_distillation_rollout as rollout

    _config(tmp_path)
    _, _, baseline = store.write_immutable(
        store.distillation_dir(tmp_path) / "baselines",
        {
            "kind": "test-baseline",
            "raw_watermark": "a" * 64,
            "hard_floor": {"p5_allowed": True, "reasons": []},
            "offline_training_gate": {"passed": True, "revision": "test-v2"},
        },
        schema=distill.BASELINE_SCHEMA,
    )
    bootstrap = distill._ensure_bootstrap_policy(tmp_path, baseline)
    locked_replay_id = "b" * 64
    candidate = distill.publish_policy(
        distill.train_tiny_policy([]),
        lineage={
            "baseline_artifact_id": baseline["artifact_id"],
            "locked_replay_id": locked_replay_id,
        },
        root=tmp_path,
    )
    gate = {
        "denominator": 500,
        "min_denominator": 500,
        "min_days": 7,
        "ci_lower": 1.0,
        "min_ci_lower": 0.9,
    }
    metrics = {
        name: dict(gate)
        for name in (
            "coverage_abstain",
            "latency_timeout",
            "cohort_delta",
            "feature_parity",
        )
    }
    run_id = "c" * 64
    evaluation_id, _, _ = store.write_immutable(
        store.distillation_dir(tmp_path) / "evaluations",
        {
            "kind": "test-runtime-metrics",
            "run_id": run_id,
            "policy_id": candidate["artifact_id"],
            "baseline_id": baseline["artifact_id"],
            "raw_watermark": baseline["raw_watermark"],
            "incumbent_policy_id": bootstrap["artifact_id"],
            "split_sha256": locked_replay_id,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "feature_parity_sha256": "d" * 64,
            "offline_gate_sha256": distill.canonical_json.canonical_json_sha256_strict(
                baseline["offline_training_gate"]
            ),
            "observation_mode": "paired",
            "replay_metrics": metrics,
            "shadow_metrics": metrics,
            "canary_metrics": metrics,
        },
        schema=rollout.EVALUATION_SCHEMA,
    )
    assert (
        rollout.evaluate_and_advance(
            tmp_path,
            "2026-08-14T00:00:00Z",
            {"run_id": run_id, "evaluation_artifact_id": evaluation_id},
        )["status"]
        == "shadow"
    )
    assert (
        distill.load_shadow_policy(tmp_path)["artifact_id"] == candidate["artifact_id"]
    )

    rendered = "private bounded snippet"
    features = distill.build_fast_features(query_chargram_coverage=1)
    shadow_ledger = (
        store.distillation_dir(tmp_path) / "shadow-observation-receipts.jsonl"
    )
    lock = store.acquire_nonblocking_lock(
        shadow_ledger.with_suffix(".jsonl.lock")
    )
    assert lock is not None
    try:
        deferred = distill.record_shadow_observation(
            decision_id="shadow-busy",
            host="codex",
            session_id="private-session",
            query_semantic_sha256="e" * 64,
            policy_id=candidate["artifact_id"],
            incumbent_policy_id=bootstrap["artifact_id"],
            served_policy_id=bootstrap["artifact_id"],
            selected_candidate_ids=["page-v1"],
            incumbent_selected_candidate_ids=["page-v1"],
            paired_eligible=True,
            candidate_feature_snapshot=[
                {"candidate_id": "page-v1", "features": features}
            ],
            candidate_pool_refs=[
                {
                    "candidate_id": "page-v1",
                    "selected": True,
                    "page_id": "page",
                    "page_content_sha256": "f" * 64,
                    "rendered_context": rendered,
                    "rendered_context_sha256": hashlib.sha256(
                        rendered.encode()
                    ).hexdigest(),
                }
            ],
            observed_at="2026-08-14T00:00:01Z",
            decision_latency_ms=42,
            timed_out=False,
            nonblocking=True,
            root=tmp_path,
        )
    finally:
        store.release_lock(lock)
    assert deferred == {"status": "deferred", "reason": "receipt_ledger_busy"}
    assert not shadow_ledger.exists()
    assert not (store.distillation_dir(tmp_path) / "shadow-observations").exists()
    receipt = distill.record_shadow_observation(
        decision_id="shadow-decision",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="e" * 64,
        policy_id=candidate["artifact_id"],
        incumbent_policy_id=bootstrap["artifact_id"],
        served_policy_id=bootstrap["artifact_id"],
        selected_candidate_ids=["page-v1"],
        incumbent_selected_candidate_ids=["page-v1"],
        paired_eligible=True,
        candidate_feature_snapshot=[{"candidate_id": "page-v1", "features": features}],
        candidate_pool_refs=[
            {
                "candidate_id": "page-v1",
                "selected": True,
                "page_id": "page",
                "page_content_sha256": "f" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            }
        ],
        observed_at="2026-08-14T00:00:01Z",
        decision_latency_ms=42,
        timed_out=False,
        root=tmp_path,
    )
    retry = distill.record_shadow_observation(
        decision_id="shadow-decision",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="e" * 64,
        policy_id=candidate["artifact_id"],
        incumbent_policy_id=bootstrap["artifact_id"],
        served_policy_id=bootstrap["artifact_id"],
        selected_candidate_ids=["page-v1"],
        incumbent_selected_candidate_ids=["page-v1"],
        paired_eligible=True,
        candidate_feature_snapshot=[{"candidate_id": "page-v1", "features": features}],
        candidate_pool_refs=[
            {
                "candidate_id": "page-v1",
                "selected": True,
                "page_id": "page",
                "page_content_sha256": "f" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            }
        ],
        observed_at="2026-08-14T00:00:02Z",
        decision_latency_ms=42,
        timed_out=False,
        root=tmp_path,
    )
    assert retry["record_sha256"] == receipt["record_sha256"]
    distill.record_shadow_observation(
        decision_id="shadow-unpaired",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="e" * 64,
        policy_id=candidate["artifact_id"],
        incumbent_policy_id=bootstrap["artifact_id"],
        served_policy_id=bootstrap["artifact_id"],
        selected_candidate_ids=["page-v1"],
        incumbent_selected_candidate_ids=[],
        paired_eligible=False,
        candidate_feature_snapshot=[{"candidate_id": "page-v1", "features": features}],
        candidate_pool_refs=[
            {
                "candidate_id": "page-v1",
                "selected": True,
                "page_id": "page",
                "page_content_sha256": "f" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            }
        ],
        observed_at="2026-08-14T00:00:03Z",
        decision_latency_ms=41,
        timed_out=False,
        root=tmp_path,
    )
    artifact_path = (
        store.distillation_dir(tmp_path)
        / "shadow-observations"
        / f"{receipt['shadow_observation_artifact_id']}.json"
    )
    artifact = store.read_sealed(
        artifact_path, schema=distill.SHADOW_OBSERVATION_SCHEMA
    )
    assert "rendered_context" not in artifact["candidate_pool_refs"][0]
    operational = distill._operational_rollout_metrics(
        tmp_path, candidate["artifact_id"], bootstrap["artifact_id"]
    )
    assert operational["coverage_abstain"]["denominator"] == 1
    assert operational["latency_timeout"]["denominator"] == 1
    artifact_path.write_text("{}\n", encoding="utf-8")
    tampered = distill._operational_rollout_metrics(
        tmp_path, candidate["artifact_id"], bootstrap["artifact_id"]
    )
    assert tampered["coverage_abstain"]["denominator"] == 0


def test_unpaired_exact_receipts_do_not_qualify_rollout_metrics(
    tmp_path: Path,
) -> None:
    candidate_id = _baseline_identity(tmp_path)
    _, _, incumbent = store.write_immutable(
        store.distillation_dir(tmp_path) / "baselines",
        {"kind": "second-incumbent"},
        schema=distill.BASELINE_SCHEMA,
    )
    rendered = "bounded card"
    features = distill.build_fast_features(query_chargram_coverage=1)
    for index, policy_id in enumerate((candidate_id, incumbent["artifact_id"])):
        distill.record_exact_exposure(
            decision_id=f"metric-{index}",
            host="codex",
            session_id=f"session-{index}",
            query_semantic_sha256=hashlib.sha256(f"query-{index}".encode()).hexdigest(),
            policy_id=policy_id,
            candidate_refs=[
                {
                    "candidate_id": f"page-{index}",
                    "page_id": f"page-{index}",
                    "page_content_sha256": "9" * 64,
                    "rendered_context": rendered,
                    "rendered_context_sha256": hashlib.sha256(
                        rendered.encode()
                    ).hexdigest(),
                }
            ],
            candidate_feature_snapshot=[
                {"candidate_id": f"page-{index}", "features": features}
            ],
            candidate_pool_refs=[
                {
                    "candidate_id": f"page-{index}",
                    "selected": True,
                    "page_id": f"page-{index}",
                    "page_content_sha256": "9" * 64,
                    "rendered_context": rendered,
                    "rendered_context_sha256": hashlib.sha256(
                        rendered.encode()
                    ).hexdigest(),
                }
            ],
            render_sha256="8" * 64,
            candidate_snapshot_sha256="7" * 64,
            observed_at=f"2026-08-14T00:00:0{index}Z",
            decision_latency_ms=100 + index,
            timed_out=False,
            root=tmp_path,
        )
    metrics = distill._operational_rollout_metrics(
        tmp_path, candidate_id, incumbent["artifact_id"]
    )
    assert metrics["coverage_abstain"]["denominator"] == 0
    assert metrics["latency_timeout"]["denominator"] == 0
    assert metrics["feature_parity"]["denominator"] == 0
    automatic = distill._automatic_baseline_metrics(tmp_path)
    assert automatic["coverage_rate"] == 1.0
    assert automatic["latency_p95_ms"] == 101


def test_page_fallback_counts_only_coverage_and_runtime_denominators(
    tmp_path: Path,
) -> None:
    candidate_id = _baseline_identity(tmp_path)
    incumbent_id, _, _ = store.write_immutable(
        store.distillation_dir(tmp_path) / "baselines",
        {"kind": "page-fallback-incumbent"},
        schema=distill.BASELINE_SCHEMA,
    )
    for index, policy_id in enumerate((candidate_id, incumbent_id)):
        receipt = distill.record_exposure(
            decision_id=f"capture-error-{index}",
            host="codex",
            session_id=f"private-{index}",
            prompt_hash=f"prompt-{index}",
            policy_id=policy_id,
            candidate_ids=[f"selected-{index}"],
            candidate_snapshot_sha256=hashlib.sha256(
                f"snapshot-{index}".encode()
            ).hexdigest(),
            observed_at=f"2026-08-14T00:00:0{index}Z",
            decision_latency_ms=125 + index,
            timed_out=False,
            error_code="exact_capture_error",
            root=tmp_path,
        )
        assert receipt["runtime_observation"]["error_code"] == "exact_capture_error"
    metrics = distill._operational_rollout_metrics(tmp_path, candidate_id, incumbent_id)
    assert metrics["coverage_abstain"]["denominator"] == 0
    assert metrics["latency_timeout"]["denominator"] == 0
    assert metrics["feature_parity"]["denominator"] == 0
    automatic = distill._automatic_baseline_metrics(tmp_path)
    assert automatic["coverage_rate"] == 1.0
    assert automatic["latency_p95_ms"] == 126
    assert "candidate_recall" not in automatic


def test_page_operational_receipt_is_deduped_when_exact_decision_exists(
    tmp_path: Path,
) -> None:
    policy_id = _baseline_identity(tmp_path)
    distill.record_exact_exposure(
        decision_id="same-decision",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="a" * 64,
        policy_id=policy_id,
        candidate_refs=[],
        render_sha256="b" * 64,
        candidate_snapshot_sha256="c" * 64,
        observed_at="2026-08-14T00:00:00Z",
        decision_latency_ms=50,
        timed_out=False,
        root=tmp_path,
    )
    observation = {
        "decision": "none",
        "selected_count": 0,
        "latency_ms": 50.0,
        "timed_out": False,
        "error_code": "exact_capture_error",
    }
    binding = {
        "decision_id": "same-decision",
        "host": "codex",
        "session_id_sha256": hashlib.sha256(b"private-session").hexdigest(),
        "prompt_hash": "prompt",
        "policy_id": policy_id,
        "candidate_ids": [],
        "candidate_snapshot_sha256": "c" * 64,
        "runtime_observation_sha256": distill.canonical_json.canonical_json_sha256_strict(
            observation
        ),
        "observed_at": "2026-08-14T00:00:00Z",
    }
    store.append_chain(
        store.distillation_dir(tmp_path) / "exposure-receipts.jsonl",
        {
            "kind": "prospective-page-exposure",
            **binding,
            "runtime_observation": observation,
            "binding_sha256": distill.canonical_json.canonical_json_sha256_strict(
                binding
            ),
            "idempotency_sha256": "d" * 64,
        },
    )
    metrics = distill._operational_rollout_metrics(tmp_path, policy_id, "f" * 64)
    assert metrics["latency_timeout"]["denominator"] == 0


def test_counterfactual_uses_exact_arms_and_copies_live_features(
    tmp_path: Path,
) -> None:
    class Counterfactual:
        local = True

        def __init__(self, outcome_receipt_id: str, *, fail_transient: bool) -> None:
            self.outcome_receipt_id = outcome_receipt_id
            self.fail_transient = fail_transient

        def compare(self, payload: object) -> dict[str, object]:
            if self.fail_transient:
                raise OSError("temporary local worker failure")
            assert isinstance(payload, dict)
            assert payload["mode"] == "remove"
            assert payload["a0_evidence"] != payload["a1_evidence"]
            return {
                "verdict": "helpful",
                "reason": "matched",
                "a0_sha256": "1" * 64,
                "a1_sha256": "2" * 64,
                "blind_orders": ["a0_first", "a1_first"],
                "order_agreement": True,
                "generator_route_identity": {"role": "generator"},
                "generator_model_digest": "3" * 64,
                "judge_route_identity": {"role": "judge"},
                "judge_model_digest": "4" * 64,
                "closed_outcome_receipt_id": self.outcome_receipt_id,
            }

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    rally = distill.extract_rallies(raw_dir, root=tmp_path)[1]
    rendered = "page context"
    features = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=1
    )
    identity = _baseline_identity(tmp_path)
    exposure = distill.record_exact_exposure(
        decision_id="counterfactual",
        host="codex",
        session_id="session-one",
        query_semantic_sha256=rally["query_sha256"],
        policy_id=identity,
        candidate_refs=[
            {
                "candidate_id": "page-v1",
                "page_id": "page",
                "page_content_sha256": "9" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            }
        ],
        candidate_feature_snapshot=[{"candidate_id": "page-v1", "features": features}],
        render_sha256="5" * 64,
        candidate_snapshot_sha256="6" * 64,
        observed_at="2026-08-02T00:00:00.500000Z",
        root=tmp_path,
    )
    outcome = distill.record_closed_outcome(
        outcome_id="test-outcome",
        exposure_artifact_id=exposure["exposure_artifact_id"],
        candidate_id="page-v1",
        candidate_version_sha256="9" * 64,
        kind="test",
        status="passed",
        evidence_sha256="7" * 64,
        observed_at="2026-08-02T00:00:00.750000Z",
        root=tmp_path,
    )
    with pytest.raises(distill.DistillationError, match="not exposed"):
        distill.record_closed_outcome(
            outcome_id="pool-only",
            exposure_artifact_id=exposure["exposure_artifact_id"],
            candidate_id="unselected",
            candidate_version_sha256="8" * 64,
            kind="test",
            status="passed",
            evidence_sha256="7" * 64,
            observed_at="2026-08-02T00:00:00.800000Z",
            root=tmp_path,
        )
    failed = distill.record_closed_outcome(
        outcome_id="failed-outcome",
        exposure_artifact_id=exposure["exposure_artifact_id"],
        candidate_id="page-v1",
        candidate_version_sha256="9" * 64,
        kind="test",
        status="failed",
        evidence_sha256="6" * 64,
        observed_at="2026-08-02T00:00:00.900000Z",
        root=tmp_path,
    )
    failed_artifact = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "outcomes"
        / f"{failed['outcome_artifact_id']}.json",
        schema=distill.OUTCOME_SCHEMA,
    )
    assert failed_artifact["authority"] == "capture-only"
    counterfactual = Counterfactual("f" * 64, fail_transient=True)
    deferred = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        counterfactual=counterfactual,
        structural_verifier=lambda *_args: "exact_test_outcome",
    )
    assert deferred["status"] == "deferred"
    assert (
        store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl") == []
    )
    counterfactual.fail_transient = False
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        counterfactual=counterfactual,
        structural_verifier=lambda *_args: "exact_test_outcome",
    )
    assert result["counterfactuals_written"] == 1
    row = store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl")[0]
    assert row["authority"] == "teacher-only"
    assert row["features"] == features
    assert distill.train_tiny_policy([row])["training_rows"] == 1
    training = distill.materialize_training_rows(tmp_path)
    assert training["rows"][0]["features"] == features
    assert (
        distill._resolve_closed_outcome(
            tmp_path,
            outcome["record_sha256"],
            exposure_artifact_id=exposure["exposure_artifact_id"],
            candidate_id="page-v1",
        )
        is None
    )
    outcome_path = (
        store.distillation_dir(tmp_path)
        / "outcomes"
        / f"{outcome['outcome_artifact_id']}.json"
    )
    outcome_path.write_text("{}\n")
    assert (
        distill._resolve_closed_outcome(
            tmp_path,
            outcome["record_sha256"],
            exposure_artifact_id=exposure["exposure_artifact_id"],
            candidate_id="page-v1",
        )
        is None
    )


def test_snapshot_distinguishes_missing_from_tampering(tmp_path: Path) -> None:
    assert store.snapshot(tmp_path)["error_code"] == "missing_state"
    path = store.distillation_dir(tmp_path) / store.STATE_FILE
    path.parent.mkdir(parents=True)
    path.write_text('{"schema":"broken"}\n')
    snapshot = store.snapshot(tmp_path)
    assert snapshot["status"] == "tampered"
    assert snapshot["error_code"] == "invalid_state"


def test_text_feature_v2_is_identical_for_historical_and_live_jp_en() -> None:
    english = distill.build_text_features(
        "Recall rollout status", "Recall rollout status and latency"
    )
    full_width = distill.build_text_features(
        "Ｒｅｃａｌｌ rollout status", "recall ROLLOUT status and latency"
    )
    japanese = distill.build_text_features(
        "クロノバイザーの検索精度", "検索精度を改善するクロノバイザー"
    )
    unrelated = distill.build_text_features("クロノバイザー", "転職と給与の相談")
    assert english == full_width
    assert japanese["query_chargram_coverage"] > 0
    assert japanese["query_chargram_coverage"] > unrelated["query_chargram_coverage"]
    assert set(english) == set(distill.FAST_FEATURE_KEYS)


def test_cwd_and_three_model_votes_never_become_verified() -> None:
    structural = distill._structural_tokens({"cwd": "/private/project/src/app.py"})
    assert structural["path"] == []
    labels = [
        distill._teacher_label(
            {"verdict": "relevant"}, verified_predicate="exact_path_overlap"
        )
        for _route in distill.TEACHER_ROLES
    ]
    assert {row["authority"] for row in labels} == {"teacher-only"}


def test_historical_teacher_row_materializes_without_live_exposure(
    tmp_path: Path,
) -> None:
    rally_id = "historical-rally"
    candidate_id = "historical-candidate"
    features = distill.build_text_features("検索精度", "検索精度を改善")
    store.append_chain(
        store.distillation_dir(tmp_path) / "rally-manifest.jsonl",
        {
            "kind": "rally-manifest",
            "manifest": {
                "rally_id": rally_id,
                "session_cluster_id": "session-cluster",
                "as_of": "2026-01-01T00:00:00Z",
            },
        },
    )
    store.append_chain(
        store.distillation_dir(tmp_path) / "candidate-ledger.jsonl",
        {
            "kind": "candidate-snapshot",
            "rally_id": rally_id,
            "snapshot": {
                "feature_revision": distill.TEXT_FEATURE_REVISION,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "feature_revision": distill.TEXT_FEATURE_REVISION,
                        "features": features,
                    }
                ],
            },
        },
    )
    store.append_chain(
        store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        {
            "kind": "teacher-label",
            "rally_id": rally_id,
            "candidate_id": candidate_id,
            "route": distill.TEACHER_ROLES[0],
            "model_digest": "a" * 64,
            "assignment": {"probe": False},
            "dimension": "relevance",
            "verdict": "relevant",
            "authority": "teacher-only",
        },
    )
    manifest = store.read_chain(
        store.distillation_dir(tmp_path) / "rally-manifest.jsonl"
    )[0]["manifest"]
    split_plan = distill._ensure_split_plan(
        tmp_path,
        [manifest],
        raw_watermark="b" * 64,
        model_cohort_sha256="c" * 64,
    )
    artifact = distill.materialize_training_rows(tmp_path)
    assert artifact["rows"][0]["features"] == features
    assert artifact["rows"][0]["split_plan_id"] == split_plan["artifact_id"]
    assert (
        distill.train_tiny_policy([{**artifact["rows"][0], "split": "train"}])[
            "training_rows"
        ]
        == 1
    )


def test_probe_and_locked_test_rows_cannot_change_policy_bytes() -> None:
    positive = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=1
    )
    negative = distill.build_fast_features()
    base = [
        {
            "rally_id": "train-positive",
            "candidate_id": "positive",
            "dimension": "relevance",
            "verdict": "relevant",
            "authority": "teacher-only",
            "features": positive,
            "split": "train",
        },
        {
            "rally_id": "train-negative",
            "candidate_id": "negative",
            "dimension": "relevance",
            "verdict": "irrelevant",
            "authority": "teacher-only",
            "features": negative,
            "split": "train",
        },
    ]
    locked = distill.train_tiny_policy(base)
    changed_holdouts = [
        *base,
        {
            **base[0],
            "rally_id": "probe",
            "candidate_id": "probe",
            "probe": True,
            "verdict": "irrelevant",
        },
        {
            **base[1],
            "rally_id": "locked-test",
            "candidate_id": "locked-test",
            "split": "test",
            "verdict": "relevant",
        },
    ]
    assert distill.train_tiny_policy(changed_holdouts) == locked


def test_offline_gate_uses_route_stability_and_agreed_counterfactuals(
    tmp_path: Path,
) -> None:
    positive = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=1
    )
    negative = distill.build_fast_features()
    digests = {role: f"{index + 1}" * 64 for index, role in enumerate(distill.TEACHER_ROLES)}
    rows: list[dict[str, object]] = []
    for index in range(680):
        route = distill.TEACHER_ROLES[index % 3]
        verdict = "relevant" if index % 2 == 0 else "irrelevant"
        rows.append(
            {
                "rally_id": f"owner-{index}",
                "candidate_id": f"owner-candidate-{index}",
                "session_cluster_id": f"owner-session-{index}",
                "as_of": f"{index:06}",
                "dimension": "relevance",
                "verdict": verdict,
                "authority": "teacher-only",
                "features": positive if verdict == "relevant" else negative,
                "route": route,
                "model_digest": digests[route],
                "probe": False,
                "source": "teacher-label",
            }
        )
    for index in range(61):
        verdict = "relevant" if index < 31 else "irrelevant"
        for route in distill.TEACHER_ROLES:
            rows.append(
                {
                    "rally_id": f"probe-{index}",
                    "candidate_id": f"probe-candidate-{index}",
                    "session_cluster_id": f"probe-session-{index}",
                    "as_of": f"{680 + index:06}",
                    "dimension": "relevance",
                    "verdict": verdict,
                    "authority": "teacher-only",
                    "features": positive if verdict == "relevant" else negative,
                    "route": route,
                    "model_digest": digests[route],
                    "probe": True,
                    "source": "teacher-label",
                }
            )
    for index in range(60):
        verdict = "helpful" if index < 30 else "harmful"
        rows.append(
            {
                "rally_id": f"cf-{index}",
                "candidate_id": f"cf-candidate-{index}",
                "session_cluster_id": f"cf-session-{index}",
                "as_of": f"{741 + index:06}",
                "dimension": "answer_utility",
                "verdict": verdict,
                "authority": "teacher-only",
                "features": positive if verdict == "helpful" else negative,
                "route": "counterfactual",
                "model_digest": "f" * 64,
                "generator_model_digest": "4" * 64,
                "judge_model_digest": "5" * 64,
                "probe": False,
                "source": "counterfactual-label",
                "order_agreement": True,
            }
        )
    config = distill.DistillationConfig(
        hard_floor_teacher_labels=1,
        hard_floor_teacher_per_class=1,
        hard_floor_probe_pairs=1,
        hard_floor_counterfactual_pairs=1,
    )

    def bind_split(values: list[dict[str, object]]) -> list[dict[str, object]]:
        active, cohort = distill._active_training_cohort(values)
        plan = distill._ensure_split_plan(
            tmp_path,
            active,
            raw_watermark="0" * 64,
            model_cohort_sha256=cohort["cohort_sha256"],
        )
        return [
            {
                **row,
                **(
                    {
                        "split": plan["assignments"][str(row["rally_id"])],
                        "split_plan_id": plan["artifact_id"],
                    }
                    if str(row["rally_id"]) in plan["assignments"]
                    else {}
                ),
            }
            for row in values
        ]

    missing = distill._offline_training_gate(rows, config, root=tmp_path)
    assert "fixed_split_plan_missing" in missing["reasons"]
    rows = bind_split(rows)
    gate = distill._offline_training_gate(rows, config, root=tmp_path)
    assert all(value["passed"] for value in gate["route_folds"].values()), gate[
        "route_folds"
    ]
    assert gate["reasons"] == []
    assert gate["passed"] is True
    assert gate["truth_authority"] == "teacher_only_not_verified"
    unstable = [dict(row) for row in rows]
    for row in unstable:
        if row.get("probe") is True and row.get("route") == distill.TEACHER_ROLES[0]:
            row["verdict"] = (
                "irrelevant" if row["verdict"] == "relevant" else "relevant"
            )
    rejected = distill._offline_training_gate(unstable, config, root=tmp_path)
    assert rejected["passed"] is False
    assert "probe_route_stability_below_gate" in rejected["reasons"]

    mixed = [{**rows[0], "split_plan_id": "f" * 64}, *rows[1:]]
    assert "fixed_split_plan_missing" in distill._offline_training_gate(
        mixed, config, root=tmp_path
    )["reasons"]
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / "split-plan.json",
        {"kind": "split-plan-pointer", "split_plan_id": "e" * 64},
    )
    assert "fixed_split_plan_missing" in distill._offline_training_gate(
        rows, config, root=tmp_path
    )["reasons"]

    updated = bind_split([*rows, {**rows[0], "model_digest": "a" * 64}])
    partial_update = distill._offline_training_gate(updated, config, root=tmp_path)
    assert partial_update["passed"] is False
    assert "probe_pairs_below_floor" in partial_update["reasons"]
    next_digests = {
        role: f"{index + 6}" * 64
        for index, role in enumerate(distill.TEACHER_ROLES)
    }
    complete_update = bind_split([
        *updated,
        *[
            {**row, "model_digest": next_digests[str(row["route"])]}
            for row in rows
            if row.get("source") == "teacher-label"
        ],
    ])
    recovered = distill._offline_training_gate(
        complete_update, config, root=tmp_path
    )
    assert recovered["passed"] is True
    assert set(recovered["model_cohort"]["teacher_model_digests"].values()) == set(
        next_digests.values()
    )


def test_authenticated_correction_is_negative_veto_only(tmp_path: Path) -> None:
    identity = _baseline_identity(tmp_path)
    preimage = b"stale page bytes"
    postimage = b"corrected page bytes"
    rendered = "stale page"
    exposure = distill.record_exact_exposure(
        decision_id="veto-decision",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="a" * 64,
        policy_id=identity,
        candidate_refs=[
            {
                "candidate_id": "page-v1",
                "page_id": "page",
                "page_content_sha256": hashlib.sha256(preimage).hexdigest(),
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            }
        ],
        render_sha256="b" * 64,
        candidate_snapshot_sha256="c" * 64,
        observed_at="2026-08-14T00:00:00Z",
        root=tmp_path,
    )
    receipt = distill.record_authenticated_exact_correction_veto(
        decision_id="veto-decision",
        correction_id="correction-one",
        candidate_id="page-v1",
        page_id="page",
        preimage_bytes=preimage,
        postimage_bytes=postimage,
        readback_bytes=postimage,
        cas_status="applied",
        observed_at="2026-08-14T00:01:00Z",
        root=tmp_path,
    )
    assert receipt["policy_id"] == identity
    assert distill._authenticated_negative_vetoes(tmp_path, identity) == 1
    with pytest.raises(distill.DistillationError, match="readback"):
        distill.record_authenticated_exact_correction_veto(
            decision_id="veto-decision",
            correction_id="forged-correction",
            candidate_id="page-v1",
            page_id="page",
            preimage_bytes=preimage,
            postimage_bytes=postimage,
            readback_bytes=b"different bytes",
            cas_status="applied",
            observed_at="2026-08-14T00:02:00Z",
            root=tmp_path,
        )
    assert exposure["exposure_artifact_id"]


def test_v1_policy_artifact_fails_closed_to_legacy(tmp_path: Path) -> None:
    _config(tmp_path)
    policy_id, _, _ = store.write_immutable(
        store.distillation_dir(tmp_path) / "policies",
        {
            "kind": "tiny-logistic-policy",
            **distill.train_tiny_policy([]),
        },
        schema="chronovisor.recall-distill-policy.v1",
    )
    store.write_pointer(tmp_path, "active", policy_id)
    store.write_pointer(tmp_path, "lkg", policy_id)
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {"kind": "worker-state", "status": "active", "rollout_percent": 100},
    )
    assert distill.load_active_policy(tmp_path) == {}
    assert distill.load_policy_for_session("private-session", tmp_path) == {}


def test_counterfactual_blinding_rejects_same_generator_and_judge_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    answers = iter(("without", "with"))

    def fake_worker_call(
        operation: str, _role: str, payload: object, **_kwargs: object
    ) -> dict[str, object]:
        assert isinstance(payload, dict)
        assert "candidate_arm" not in payload
        calls.append((operation, payload))
        if operation == "answer":
            return {
                "answer": next(answers),
                "_model_digest": "a" * 64,
                "_route_identity": {},
            }
        return {
            "blind_choice": "b" if len(calls) == 3 else "a",
            "_model_digest": "a" * 64,
            "_route_identity": {},
        }

    monkeypatch.setattr(distill, "_worker_call", fake_worker_call)
    worker = distill._WorkerCounterfactual(
        12_000,
        {
            "recall.distill.answer_generator": {},
            "recall.distill.utility_judge": {},
        },
        {
            "recall.distill.answer_generator": "a" * 64,
            "recall.distill.utility_judge": "a" * 64,
        },
    )
    result = worker.compare(
        {
            "rally_id": "rally",
            "candidate_id": "candidate",
            "query": "query",
            "context": [],
            "a0_evidence": [],
            "a1_evidence": ["candidate"],
            "actual_answer": "answer",
        }
    )
    assert result["verdict"] == "uncertain"
    assert result["order_agreement"] is False


def test_cold_start_api_uses_fixed_split_and_nonblocking_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    original_events = distill._events
    monkeypatch.setattr(
        distill,
        "_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Raw scan")),
    )
    assert distill.cold_start_due(tmp_path) is True
    monkeypatch.setattr(distill, "_events", original_events)
    first = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        cold_start=True,
        max_elapsed_seconds=300,
    )
    assert first["cold_start_pending"] is True
    assert first["manifest_backlog"] == 0
    assert first["candidate_backlog"] == 0
    plan = distill._read_split_plan(tmp_path)
    assert plan["artifact_id"] == first["split_plan_id"]
    assert plan["raw_watermark"] == distill.committed_raw_watermark(raw_dir)
    assert plan["feature_revision"] == distill.TEXT_FEATURE_REVISION
    second = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        cold_start=True,
        max_elapsed_seconds=300,
    )
    assert second["split_plan_id"] == first["split_plan_id"]
    normal = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
    )
    assert normal["cold_start_pending"] is True
    assert distill.cold_start_due(tmp_path) is True
    rallies = distill.extract_rallies(raw_dir, root=tmp_path)
    updated_plan = distill._ensure_split_plan(
        tmp_path,
        rallies,
        raw_watermark=distill.committed_raw_watermark(raw_dir),
        model_cohort_sha256="f" * 64,
    )
    assert updated_plan["artifact_id"] != first["split_plan_id"]
    assert updated_plan["assignments"] == plan["assignments"]
    lock = store.acquire_nonblocking_lock(
        store.distillation_dir(tmp_path) / "distillation-worker.lock"
    )
    assert lock is not None
    try:
        busy = distill.run_distillation_chunk(
            root=tmp_path,
            raw_dir=raw_dir,
            config_path=config,
            teachers={},
            cold_start=True,
        )
    finally:
        store.release_lock(lock)
    assert busy == {"status": "deferred", "processed": 0, "reason": "worker_busy"}


def test_chain_batch_keeps_standard_hash_chain(tmp_path: Path) -> None:
    path = store.distillation_dir(tmp_path) / "batch-ledger.jsonl"
    rows = store.append_chain_batch(path, ({"index": index} for index in range(500)))
    assert len(rows) == 500
    assert store.verify_chain(path)["records"] == 500
    assert store.read_chain(path)[-1]["index"] == 499


def test_chain_batch_replace_failure_preserves_old_head_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store.distillation_dir(tmp_path) / "batch-ledger.jsonl"
    store.append_chain(path, {"index": 0})
    before = path.read_bytes()
    head = store.verify_chain(path)
    replace = store.os.replace

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        store.append_chain_batch(path, ({"index": 1}, {"index": 2}))
    assert path.read_bytes() == before
    assert store.verify_chain(path) == head
    monkeypatch.setattr(store.os, "replace", replace)
    store.append_chain_batch(path, ({"index": 1}, {"index": 2}))
    assert [row["index"] for row in store.read_chain(path)] == [0, 1, 2]


def test_cold_start_does_not_begin_counterfactual_without_time_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Counterfactual:
        local = True

        def compare(self, _payload: object) -> dict[str, object]:
            raise AssertionError("counterfactual must not start")

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    monkeypatch.setattr(distill.time, "monotonic", lambda: 0.0)
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        counterfactual=Counterfactual(),
        cold_start=True,
        max_elapsed_seconds=60,
    )
    assert result["status"] == "deferred"
    assert result["counterfactuals_written"] == 0
