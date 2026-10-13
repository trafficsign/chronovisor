from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from chronovisor.recall.recall_distillation_luna import (
    LunaBulkBundle,
    native_output_sha256,
)
from scripts import recall_luna_daily as daily


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_native_cohort_identity_is_stable_sha256() -> None:
    first = daily._native_model_cohort_sha256()
    second = daily._native_model_cohort_sha256()

    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_build_rows_keeps_embargo_and_adds_blind_repeat_pairs(monkeypatch) -> None:
    monkeypatch.setattr(
        daily,
        "_payload",
        lambda rally, candidate, texts: {
            "query": "query",
            "context": [],
            "candidate": {"text": "candidate"},
        },
    )
    rallies = [
        {
            "rally_id": "rally-train",
            "as_of_us": 1,
            "session_cluster_id": "group-train",
            "query_sha256": "1" * 64,
        },
        {
            "rally_id": "rally-test",
            "as_of_us": 2,
            "session_cluster_id": "group-test",
            "query_sha256": "2" * 64,
        },
        {
            "rally_id": "rally-embargo",
            "as_of_us": 3,
            "session_cluster_id": "group-embargo",
            "query_sha256": "3" * 64,
        },
    ]
    snapshots = {
        rally["rally_id"]: {
            "snapshot_sha256": "a" * 64,
            "candidates": [
                {
                    "candidate_id": f"candidate-{rally['rally_id']}-0",
                    "rank": 0,
                    "text_sha256": "b" * 64,
                },
                {
                    "candidate_id": f"candidate-{rally['rally_id']}-1",
                    "rank": 1,
                    "text_sha256": "c" * 64,
                },
            ],
        }
        for rally in rallies
    }
    assignments = {
        "rally-train": "train",
        "rally-test": "test",
        "rally-embargo": "embargo",
    }

    rows, skipped = daily._build_rows(rallies, snapshots, assignments, {})

    assert not skipped
    assert {row["split"] for row in rows} == {"train", "test", "embargo"}
    repeats = [row for row in rows if row.get("repeat_pair_id")]
    assert len(repeats) == 4
    assert {row["blind_order"] for row in repeats} == {"a_first", "b_first"}
    pair = [row for row in repeats if row["candidate_id"] == repeats[0]["candidate_id"]]
    assert len({item["input"]["candidates"][0]["candidate_id"] for item in pair}) == 2
    assert pair[0]["input"]["candidates"] != pair[1]["input"]["candidates"]
    assert {row["input"]["target_candidate_id"] for row in pair} == {
        repeats[0]["candidate_id"]
    }
    assert all(daily._row_has_native_source_identity(row) for row in rows)


def test_blind_payload_fits_context_after_complete_candidate_display(monkeypatch) -> None:
    target = {"candidate_id": "target"}
    alternate = {"candidate_id": "alternate"}
    target_payload = {"query": "query", "context": ["old" * 3_000, "recent"], "candidate": "t" * 1_000}
    monkeypatch.setattr(daily, "_payload", lambda *_: {"candidate": "a" * 1_000})
    pair = [daily._blind_payload({}, target, alternate, target_payload, {}, blind_order=order)
            for order in ("a_first", "b_first")]
    assert all(value is not None for value in pair)
    for value in pair:
        assert value["context"] == ["recent"]
        assert value["query"] == "query"
        assert value["candidate"] == "t" * 1_000
        assert {item["evidence"] for item in value["candidates"]} == {"t" * 1_000, "a" * 1_000}
        assert len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()) <= daily.MAX_INPUT_BYTES
    assert pair[0]["candidates"][0]["candidate_id"] != pair[1]["candidates"][0]["candidate_id"]


def test_probe_cap_counts_complete_rallies_after_payload_validation(monkeypatch) -> None:
    monkeypatch.setattr(daily, "PROBE_CAP", 1)
    monkeypatch.setattr(daily, "_payload", lambda rally, *_: {
        "query": "query", "context": [],
        "candidate": "x" * (6_000 if rally["rally_id"] == "oversized" else 10),
    })
    rallies = [{"rally_id": name, "as_of_us": index, "query_sha256": "a" * 64}
               for index, name in enumerate(("oversized", "eligible"), 1)]
    snapshots = {row["rally_id"]: {
        "snapshot_sha256": "b" * 64,
        "candidates": [{"candidate_id": f"candidate-{index}", "text_sha256": "c" * 64}
                       for index in range(2)],
    } for row in rallies}
    rows, skipped = daily._build_rows(rallies, snapshots, {row["rally_id"]: "test" for row in rallies}, {})
    probes = [row for row in rows if row.get("repeat_pair_id")]
    assert len(probes) == 4
    assert {row["rally_id"] for row in probes} == {"eligible"}
    assert skipped["probe_payload_over_limit_or_text_missing"] == 4


def test_prepare_reports_authoritative_capture_writer_and_is_resumable(tmp_path: Path) -> None:
    progress = tmp_path / "capture.json"
    progress.write_text(
        json.dumps(
            {
                "status": "running",
                "captured": 10,
                "total_rallies": 20,
                "remaining": 10,
                "candidate_index_count": 10,
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "daily-status.json"

    result = daily.prepare(
        root=tmp_path / "root",
        bundle_path=tmp_path / "bundle",
        receipt_path=receipt,
        capture_progress_path=progress,
    )

    assert result["status"] == "in_progress"
    assert result["reason"] == "authoritative_candidate_writer_active"
    assert result["resumable"] is True
    assert _read(receipt)["status"] == "in_progress"


def test_run_advances_catalog_and_starts_capture_before_prepare(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        daily,
        "_advance_catalog",
        lambda _: {"status": "indexed", "rallies": 1},
    )
    monkeypatch.setattr(
        daily,
        "_start_candidate_capture",
        lambda **_kwargs: {"started": True, "status": "starting"},
    )
    receipt = tmp_path / "daily.json"
    result = daily.run_daily(
        root=tmp_path / "root",
        bundle_path=tmp_path / "bundle",
        receipt_path=receipt,
        capture_progress_path=tmp_path / "capture.json",
    )
    assert result["status"] == "in_progress"
    assert result["reason"] == "authoritative_candidate_writer_starting"
    assert result["prepare"]["status"] == "in_progress"
    assert _read(receipt)["capture"]["started"] is True


def test_capture_resumes_previous_day_inventory_before_starting_another_writer(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    bulk = root / "runtime" / "recall-luna-bulk"
    previous = bulk / "2026-10-13" / "inventory" / "candidate-capture-progress.json"
    requested = bulk / "2026-10-14" / "inventory" / previous.name
    daily._write_private_json(previous, {"status": "running", "updated_at_unix": 1000})
    monkeypatch.setattr(daily.time, "time", lambda: 1001)
    monkeypatch.setattr(daily, "_capture_lock_busy", lambda _: False)
    monkeypatch.setattr(
        daily.subprocess, "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("second writer")),
    )

    path, progress = daily._active_capture_progress(root, requested)
    assert path == previous
    assert progress["status"] == "running"
    result = daily._start_candidate_capture(root=root, progress_path=requested)
    assert result == {
        "started": False, "status": "running", "reason": "capture_already_running"
    }


def test_capture_restart_replaces_completed_receipt_with_new_starting_pid(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    progress = tmp_path / "inventory" / "candidate-capture-progress.json"
    daily._write_private_json(progress, {"status": "complete", "pid": 111})
    monkeypatch.setattr(daily.subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(pid=222))

    result = daily._start_candidate_capture(root=root, progress_path=progress)

    assert result["started"] is True
    assert _read(progress)["status"] == "starting"
    assert _read(progress)["pid"] == 222


def test_repo_owned_capture_without_private_helper_reaches_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    """A fresh day can capture and prepare without an inventory helper file."""

    root = tmp_path / "root"
    root.mkdir()
    private_dir = tmp_path / "next-day" / "inventory"
    progress = private_dir / "candidate-capture-progress.json"
    rally = {
        "rally_id": "rally-capture",
        "as_of": "1970-01-01T00:00:00.000001Z",
        "as_of_us": 1,
        "host": "test-host",
        "session_cluster_id": "group-capture",
        "source_index": 1,
        "query_sha256": "1" * 64,
        "context_refs": [],
    }
    snapshot = {
        "schema": "chronovisor.recall-candidate-snapshot.v1",
        "rally_id": rally["rally_id"],
        "as_of": rally["as_of"],
        "retriever_revision": "historical-fts-v1",
        "feature_revision": "text-features-v1",
        "query_feature_text_sha256": "2" * 64,
        "candidates": [
            {
                "candidate_id": "candidate-capture",
                "rank": 0,
                "text_sha256": "3" * 64,
                "features": {"lexical_overlap": 1.0},
            }
        ],
        "snapshot_sha256": "4" * 64,
    }
    checkpoint = {
        "catalog_watermark": "5" * 64,
        "catalog_event_rowid": 1,
        "catalog_lineage": "6" * 64,
    }
    index_checkpoint = {
        "content_sha256": "7" * 64,
        "catalog_watermark": checkpoint["catalog_watermark"],
        "catalog_event_rowid": checkpoint["catalog_event_rowid"],
    }
    captured = {"done": False}

    monkeypatch.setattr(daily, "_advance_catalog", lambda _: {"status": "noop"})
    monkeypatch.setattr(daily.catalog, "sync_historical_index", lambda *_: "7" * 64)
    monkeypatch.setattr(
        daily.catalog,
        "sync_candidate_index",
        lambda *_args: {"status": "advanced", "record_count": int(captured["done"])},
    )
    monkeypatch.setattr(daily.catalog, "rallies", lambda _: [rally])
    monkeypatch.setattr(
        daily.catalog,
        "candidate_rally_ids",
        lambda _: {rally["rally_id"]} if captured["done"] else set(),
    )
    monkeypatch.setattr(daily, "_capture_checkpoint_identity", lambda _: checkpoint)
    monkeypatch.setattr(daily, "_capture_index_identity", lambda _: index_checkpoint)
    monkeypatch.setattr(daily, "_capture_one", lambda **_kwargs: snapshot)
    monkeypatch.setattr(daily.distill, "_source_age_boundary", lambda _: None)
    monkeypatch.setattr(daily.distill, "_source_age_bands", lambda _: {})

    original_append = daily.store.append_chain_batch

    def append_and_mark(path: Path, payloads: object) -> list[dict[str, object]]:
        result = original_append(path, payloads)  # type: ignore[arg-type]
        captured["done"] = True
        return result

    monkeypatch.setattr(daily.store, "append_chain_batch", append_and_mark)
    assert daily._capture(
        root=root,
        private_dir=private_dir,
        progress_path=progress,
        chunk_size=1,
        read_workers=1,
    ) == 0
    assert _read(progress)["status"] == "complete"

    plan = {
        "artifact_id": "8" * 64,
        "model_cohort_sha256": "9" * 64,
        "raw_watermark": checkpoint["catalog_watermark"],
        "assignments": {rally["rally_id"]: "embargo"},
    }
    monkeypatch.setattr(daily, "_native_model_cohort_sha256", lambda *_: "9" * 64)
    monkeypatch.setattr(daily, "_read_current_split_plan", lambda _: plan)
    monkeypatch.setattr(
        daily,
        "_catalog_identity",
        lambda _: {
            "sha256": "a" * 64,
            "bytes": 1,
            "watermark": checkpoint["catalog_watermark"],
            "event_rowid": 1,
            "lineage": checkpoint["catalog_lineage"],
        },
    )
    monkeypatch.setattr(
        daily,
        "_candidate_identity",
        lambda _: {
            "sha256": "b" * 64,
            "bytes": 1,
            "record_count": 1,
            "head_sha256": "c" * 64,
            "mtime_ns": 1,
            "ctime_ns": 1,
        },
    )
    monkeypatch.setattr(
        daily.catalog,
        "read_candidate_snapshots",
        lambda *_args: {rally["rally_id"]: snapshot},
    )
    monkeypatch.setattr(daily, "_payload", lambda *_args: {"query": "q", "candidate": "c", "context": []})
    class FakeTextCache(dict[str, str]):
        def prefetch(self, _hashes: object) -> None:
            return None

    monkeypatch.setattr(daily.catalog, "CatalogTextCache", lambda *_args: FakeTextCache())

    prepared = daily.prepare(
        root=root,
        bundle_path=tmp_path / "bundle",
        receipt_path=tmp_path / "daily.json",
        capture_progress_path=progress,
    )
    assert prepared["status"] == "pending"
    assert Path(str(prepared["bundle_path"])).is_dir()


def test_prepare_waits_for_native_split_plan_without_reassigning_test(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    monkeypatch.setattr(
        daily,
        "_catalog_identity",
        lambda _: {"sha256": "a" * 64, "bytes": 1, "watermark": "b" * 64, "event_rowid": 1, "lineage": "c" * 64},
    )
    monkeypatch.setattr(
        daily,
        "_candidate_identity",
        lambda _: {"sha256": "d" * 64, "bytes": 1, "record_count": 1, "head_sha256": "e" * 64, "mtime_ns": 1, "ctime_ns": 1},
    )
    monkeypatch.setattr(
        daily.catalog,
        "rallies",
        lambda _: [{"rally_id": "rally-1", "as_of_us": 1}],
    )
    monkeypatch.setattr(daily, "_read_current_split_plan", lambda _: None)
    receipt = tmp_path / "daily-status.json"

    result = daily.prepare(
        root=root,
        bundle_path=tmp_path / "bundle",
        receipt_path=receipt,
        capture_progress_path=tmp_path / "missing-progress.json",
    )

    assert result["status"] == "in_progress"
    assert result["reason"] == "split_plan_pending"
    assert result["resumable"] is True
    assert _read(receipt)["reason"] == "split_plan_pending"


def test_native_cohort_growth_preserves_plan_and_embargoes_only_new_groups(monkeypatch) -> None:
    native_cohort = daily._native_model_cohort_sha256()
    monkeypatch.setattr(daily, "_native_model_cohort_sha256", lambda *_: native_cohort)
    old_plan = {
        "artifact_id": "a" * 64,
        "model_cohort_sha256": native_cohort,
        "raw_watermark": "b" * 64,
        "assignments": {"old-rally": "test"},
    }
    grown_plan = {
        **old_plan,
        "artifact_id": "c" * 64,
        "assignments": {"old-rally": "test", "new-rally": "embargo"},
    }
    monkeypatch.setattr(daily, "_read_current_split_plan", lambda _: old_plan)
    monkeypatch.setattr(
        daily.catalog,
        "_read_catalog_checkpoint",
        lambda _: {"catalog_watermark": "d" * 64},
    )
    calls: list[dict[str, object]] = []

    def ensure(_, rallies, **kwargs):
        calls.append({"rallies": list(rallies), **kwargs})
        return grown_plan

    monkeypatch.setattr(daily.distill, "_ensure_split_plan", ensure)

    plan, result = daily._resolve_split_plan(
        Path("/tmp/chronovisor-daily-test"),
        [
            {"rally_id": "old-rally", "as_of_us": 1},
            {"rally_id": "new-rally", "as_of_us": 2},
        ],
        explicit_path=None,
        allow_write=False,
    )

    assert plan == grown_plan
    assert result["status"] == "ready"
    assert result["split_plan"]["growth_receipt"] is True
    assert calls[0]["model_cohort_sha256"] == native_cohort


def test_prepare_resumes_existing_bundle_without_source_reexport(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(
        daily,
        "_bundle_metadata",
        lambda _: {
            "status": "running",
            "record_count": 10,
            "completed": 4,
            "remaining": 6,
            "class_ratio": {"relevant": 2, "irrelevant": 2, "labeled": 4},
        },
    )
    receipt = tmp_path / "daily-status.json"

    result = daily.prepare(
        root=tmp_path / "root",
        bundle_path=bundle,
        receipt_path=receipt,
        capture_progress_path=tmp_path / "missing-progress.json",
    )

    assert result["status"] == "pending"
    assert result["reason"] == "existing_bundle_resumable"
    assert result["remaining"] == 6


def test_status_surfaces_active_candidate_writer(tmp_path: Path, monkeypatch) -> None:
    progress = tmp_path / "capture.json"
    progress.write_text(
        json.dumps(
            {
                "status": "running",
                "captured": 2,
                "total_rallies": 5,
                "remaining": 3,
                "candidate_index_count": 2,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        daily,
        "_catalog_identity",
        lambda _: {"sha256": "a" * 64, "bytes": 1, "watermark": "b" * 64, "event_rowid": 1, "lineage": "c" * 64},
    )
    monkeypatch.setattr(
        daily,
        "_candidate_identity",
        lambda _: {"sha256": "d" * 64, "bytes": 1, "record_count": 2, "head_sha256": "e" * 64, "mtime_ns": 1, "ctime_ns": 1},
    )
    monkeypatch.setattr(
        daily.catalog,
        "rallies",
        lambda _: [{"rally_id": "rally-1", "as_of_us": 1}],
    )
    monkeypatch.setattr(daily.catalog, "candidate_rally_ids", lambda _: {"rally-1"})

    result = daily.status(
        root=tmp_path / "root",
        bundle_path=tmp_path / "bundle",
        capture_progress_path=progress,
    )

    assert result["status"] == "in_progress"
    assert result["reason"] == "authoritative_candidate_writer_active"


def test_import_waits_when_native_luna_is_not_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        daily.distill,
        "load_distillation_config",
        lambda _: SimpleNamespace(
            teacher_profile="deepseek-v4-flash-single-v1",
            native_luna_enabled=False,
        ),
    )
    receipt = tmp_path / "daily-status.json"

    result = daily.import_completed(
        root=tmp_path / "root",
        bundle_path=tmp_path / "bundle",
        receipt_path=receipt,
    )

    assert result["status"] == "in_progress"
    assert result["reason"] == "native_luna_not_enabled"
    assert _read(receipt)["native_luna_enabled"] is False


def test_zero_label_import_does_not_publish_last_label_at(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        daily.distill,
        "load_distillation_config",
        lambda _: SimpleNamespace(
            teacher_profile=daily.DEFAULT_TEACHER_PROFILE,
            native_luna_enabled=True,
        ),
    )
    monkeypatch.setattr(
        daily.native,
        "ensure_contract",
        lambda *_args: {"artifact_id": "a" * 64},
    )
    monkeypatch.setattr(
        daily.native,
        "import_bundle",
        lambda *_args: {
            "labels": 0,
            "labels_written": 0,
            "labels_appended": 0,
            "remaining": 0,
            "last_label_at": "should-not-be-published",
        },
    )
    result = daily.import_completed(
        root=tmp_path / "root",
        bundle_path=bundle,
        receipt_path=tmp_path / "receipt.json",
    )
    assert result["status"] == "imported"
    assert result["labels_written"] == 0
    assert "last_label_at" not in result


def test_native_bundle_pointer_is_added_to_enabled_config(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[recall.distillation]\n"
        'teacher_profile = "native-luna-single-v1"\n'
        "native_luna_enabled = true\n"
        "\n[llm]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        daily.distill,
        "load_distillation_config",
        lambda _: SimpleNamespace(
            teacher_profile=daily.DEFAULT_TEACHER_PROFILE,
            native_luna_enabled=True,
        ),
    )
    result = daily._update_native_bundle_pointer(tmp_path, tmp_path / "bundle-new")
    assert result["updated"] is True
    assert 'native_luna_bundle = "' in config.read_text(encoding="utf-8")


def _finish_synthetic_bundle(path: Path) -> int:
    bundle = LunaBulkBundle.open(path)
    submitted = 0
    for shard_spec in bundle.manifest["shards"]:
        shard = int(shard_spec["shard"])
        request = bundle.next_batch(shard)
        records = []
        for source in request["records"]:
            label = {
                "task_id": source["task_id"],
                "input_sha256": source["input_sha256"],
                "teacher_profile": daily.DEFAULT_TEACHER_PROFILE,
                "native_luna": True,
                "verdict": "relevant",
                "confidence": 1.0,
                "reason_code": "synthetic",
                "rationale": None,
            }
            label["native_receipt"] = {
                "task_id": source["task_id"],
                "output_sha256": native_output_sha256(label),
                "run_id": f"run-{shard}",
                "assignment_id": f"assignment-{shard}",
            }
            records.append(label)
        if records:
            bundle.submit(shard, cursor=request["cursor"], records=records)
            submitted += len(records)
    bundle.finalize()
    return submitted


def test_new_rally_gets_new_bundle_and_import_count_increases(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    requested_bundle = tmp_path / "bundle"
    receipt = tmp_path / "daily.json"
    query_sha = "1" * 64
    old_candidate_sha = "2" * 64
    new_query_sha = "3" * 64
    new_candidate_sha = "4" * 64
    state: dict[str, object] = {
        "rallies": [
            {
                "rally_id": "rally-old",
                "as_of_us": 1,
                "as_of": "1970-01-01T00:00:00.000001Z",
                "session_cluster_id": "group-old",
                "query_sha256": query_sha,
                "context_refs": [],
            }
        ],
        "snapshots": {
            "rally-old": {
                "snapshot_sha256": "5" * 64,
                "candidates": [
                    {
                        "candidate_id": "candidate-old",
                        "rank": 0,
                        "text_sha256": old_candidate_sha,
                    }
                ],
            }
        },
        "candidate_sha": "6" * 64,
    }
    plan = {
        "artifact_id": "7" * 64,
        "model_cohort_sha256": "8" * 64,
        "raw_watermark": "9" * 64,
        "assignments": {"rally-old": "train"},
    }
    monkeypatch.setattr(daily, "_native_model_cohort_sha256", lambda *_: "8" * 64)
    monkeypatch.setattr(daily, "_read_current_split_plan", lambda _: dict(plan))
    monkeypatch.setattr(daily.catalog, "rallies", lambda _: list(state["rallies"]))
    monkeypatch.setattr(
        daily.catalog,
        "candidate_rally_ids",
        lambda _: set(state["snapshots"]),
    )
    monkeypatch.setattr(
        daily.catalog,
        "read_candidate_snapshots",
        lambda *_args: dict(state["snapshots"]),
    )

    class FakeTextCache(dict[str, str]):
        def prefetch(self, _hashes: object) -> None:
            return None

    monkeypatch.setattr(
        daily.catalog,
        "CatalogTextCache",
        lambda *_args: FakeTextCache(
            {
                query_sha: "query-old",
                old_candidate_sha: "candidate-old",
                new_query_sha: "query-new",
                new_candidate_sha: "candidate-new",
            }
        ),
    )
    monkeypatch.setattr(
        daily,
        "_catalog_identity",
        lambda _: {
            "sha256": "a" * 64,
            "bytes": 1,
            "watermark": "b" * 64,
            "event_rowid": 1,
            "lineage": "c" * 64,
        },
    )
    monkeypatch.setattr(
        daily,
        "_candidate_identity",
        lambda _: {
            "sha256": str(state["candidate_sha"]),
            "bytes": 1,
            "record_count": len(state["snapshots"]),
            "head_sha256": "d" * 64,
            "mtime_ns": 1,
            "ctime_ns": 1,
        },
    )
    labeled_pairs: set[tuple[str, str]] = set()
    monkeypatch.setattr(daily, "_existing_labeled_pairs", lambda _: set(labeled_pairs))

    first = daily.prepare(
        root=root,
        bundle_path=requested_bundle,
        receipt_path=receipt,
        capture_progress_path=tmp_path / "missing-progress.json",
    )
    assert first["status"] == "pending"
    first_bundle = Path(str(first["bundle_path"]))
    assert _finish_synthetic_bundle(first_bundle) == 1

    monkeypatch.setattr(
        daily.distill,
        "load_distillation_config",
        lambda _: SimpleNamespace(
            teacher_profile=daily.DEFAULT_TEACHER_PROFILE,
            native_luna_enabled=True,
        ),
    )
    monkeypatch.setattr(
        daily.native,
        "ensure_contract",
        lambda *_args: {"artifact_id": "e" * 64},
    )
    imported_counts: list[int] = []

    def fake_import(_root: Path, bundle_path: Path) -> dict[str, object]:
        count = int(LunaBulkBundle.open(bundle_path).status()["record_count"])
        imported_counts.append(count)
        return {
            "labels": count,
            "labels_written": count,
            "labels_appended": count,
            "relevant": count,
            "irrelevant": 0,
            "uncertain": 0,
            "remaining": 0,
            "source_snapshot_sha256": "f" * 64,
            "label_chain_head_sha256": "a" * 64,
        }

    monkeypatch.setattr(daily.native, "import_bundle", fake_import)
    imported = daily.import_completed(
        root=root,
        bundle_path=requested_bundle,
        receipt_path=receipt,
    )
    assert imported["status"] == "imported"
    assert daily._bundle_metadata(first_bundle)["production_ledger_written"] is True
    labeled_pairs.add(("rally-old", "candidate-old"))

    state["rallies"] = [
        *state["rallies"],
        {
            "rally_id": "rally-new",
            "as_of_us": 2,
            "as_of": "1970-01-01T00:00:00.000002Z",
            "session_cluster_id": "group-new",
            "query_sha256": new_query_sha,
            "context_refs": [],
        },
    ]
    state["snapshots"] = {
        **state["snapshots"],
        "rally-new": {
            "snapshot_sha256": "0" * 64,
            "candidates": [
                {
                    "candidate_id": "candidate-new",
                    "rank": 0,
                    "text_sha256": new_candidate_sha,
                }
            ],
        },
    }
    state["candidate_sha"] = "1" * 64
    plan["assignments"] = {"rally-old": "train", "rally-new": "embargo"}

    second = daily.prepare(
        root=root,
        bundle_path=requested_bundle,
        receipt_path=receipt,
        capture_progress_path=tmp_path / "missing-progress.json",
    )
    assert second["status"] == "pending"
    second_bundle = Path(str(second["bundle_path"]))
    assert second_bundle != first_bundle
    assert LunaBulkBundle.open(second_bundle).status()["record_count"] == 1
    assert _finish_synthetic_bundle(second_bundle) == 1

    imported_again = daily.import_completed(
        root=root,
        bundle_path=requested_bundle,
        receipt_path=receipt,
    )
    assert imported_again["status"] == "imported"
    assert imported_again["labels_written"] == 1
    assert imported_counts == [1, 1]
