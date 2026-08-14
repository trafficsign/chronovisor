from __future__ import annotations

import hashlib
import json
import stat
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
        "hard_floor_verified_labels": 100,
        "hard_floor_per_class": 10,
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

    @contextmanager
    def lane(*_args: object, **_kwargs: object):
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
    snapshot = distill.candidate_snapshot(index, rally, "alpha", limit=20)
    digests = {row["text_sha256"] for row in snapshot["candidates"]}
    assert hashlib.sha256(b"alpha evidence").hexdigest() in digests
    assert hashlib.sha256(b"alpha detail").hexdigest() in digests
    assert hashlib.sha256(b"alpha future").hexdigest() not in digests
    assert all("text" not in row for row in snapshot["candidates"])


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
    features = distill.build_fast_features(exact_anchor=1, margin_norm=0.4)
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
    features = distill.build_fast_features(exact_anchor=1, margin_norm=0.25)
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
                {"candidate_id": "bad", "features": {"exact_anchor": 2}}
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
    assert "wrong_domain_rate_unavailable" in baseline["hard_floor"]["reasons"]
    assert "exact_outcomes_absent" in baseline["hard_floor"]["reasons"]
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
        },
        schema=distill.BASELINE_SCHEMA,
    )
    evaluation_dir = store.distillation_dir(tmp_path) / "evaluations"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / f"{'f' * 64}.json").write_text("{}\n")
    first = distill._automatic_rollout_evaluation(
        tmp_path, baseline, {"status": "candidate", "policy_id": candidate_id}
    )
    assert first["status"] in {"held", "replay"}
    gate = {
        "denominator": 100,
        "min_denominator": 100,
        "min_days": 7,
        "ci_lower": 1.0,
        "min_ci_lower": 0.9,
    }
    metrics = {
        name: dict(gate)
        for name in (
            "candidate_recall",
            "wrong_domain",
            "anchor_rescue",
            "coverage_abstain",
            "latency_timeout",
            "answer_utility",
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
            "feature_revision": "recall-distill-fast-v1",
            "feature_parity_sha256": "c" * 64,
            "replay_metrics": metrics,
            "shadow_metrics": metrics,
            "canary_metrics": metrics,
        },
        schema="chronovisor.recall-distill-rollout-evaluation.v1",
    )
    held = distill._automatic_rollout_evaluation(
        tmp_path, baseline, {"status": "candidate", "policy_id": candidate_id}
    )
    assert held["status"] in {"held", "replay"}
    state = store.read_sealed(store.distillation_dir(tmp_path) / store.STATE_FILE)
    assert state["status"] == "replay"


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
        "denominator": 100,
        "min_denominator": 100,
        "min_days": 7,
        "ci_lower": 1.0,
        "min_ci_lower": 0.9,
    }
    metrics = {
        name: dict(gate)
        for name in (
            "candidate_recall",
            "wrong_domain",
            "anchor_rescue",
            "coverage_abstain",
            "latency_timeout",
            "answer_utility",
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
            "feature_revision": "recall-distill-fast-v1",
            "feature_parity_sha256": "d" * 64,
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
    features = distill.build_fast_features(exact_anchor=1)
    receipt = distill.record_shadow_observation(
        decision_id="shadow-decision",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="e" * 64,
        policy_id=candidate["artifact_id"],
        selected_candidate_ids=["page-v1"],
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
        selected_candidate_ids=["page-v1"],
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
    assert operational["candidate_recall"]["denominator"] == 1
    assert operational["latency_timeout"]["denominator"] == 1
    artifact_path.write_text("{}\n", encoding="utf-8")
    tampered = distill._operational_rollout_metrics(
        tmp_path, candidate["artifact_id"], bootstrap["artifact_id"]
    )
    assert tampered["candidate_recall"]["denominator"] == 0


def test_operational_metrics_are_derived_from_exact_runtime_receipts(
    tmp_path: Path,
) -> None:
    candidate_id = _baseline_identity(tmp_path)
    _, _, incumbent = store.write_immutable(
        store.distillation_dir(tmp_path) / "baselines",
        {"kind": "second-incumbent"},
        schema=distill.BASELINE_SCHEMA,
    )
    rendered = "bounded card"
    features = distill.build_fast_features(exact_anchor=1)
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
    assert metrics["candidate_recall"]["denominator"] == 1
    assert metrics["coverage_abstain"]["denominator"] == 1
    assert metrics["latency_timeout"]["ci_lower"] == 1.0
    assert metrics["wrong_domain"]["denominator"] == 0
    automatic = distill._automatic_baseline_metrics(tmp_path)
    assert automatic["candidate_recall"] == 1.0
    assert automatic["strong_anchor_rescue_rate"] == 1.0
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
    assert metrics["coverage_abstain"]["denominator"] == 1
    assert metrics["latency_timeout"]["denominator"] == 1
    assert metrics["candidate_recall"]["denominator"] == 0
    assert metrics["anchor_rescue"]["denominator"] == 0
    assert metrics["feature_parity"]["denominator"] == 0
    assert metrics["answer_utility"]["denominator"] == 0
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
    assert metrics["latency_timeout"]["denominator"] == 1


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
    features = distill.build_fast_features(exact_anchor=1, same_session=1)
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
