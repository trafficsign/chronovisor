from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from chronovisor.core import okf_cutover, semantic_index
from chronovisor.core import store as chronovisor_store
from chronovisor.core.activity_log import activity_record
from chronovisor.core.canonical_json import canonical_json_line_bytes_strict
from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.index_store import IndexStore, canonical_document_paths
from chronovisor.core.llm_runtime import RouteLocation
from chronovisor.core.okf_cutover import (
    FINAL_RECEIPT_SCHEMA,
    FINALIZE_FAULT_POINTS,
    RECUTOVER_FAULT_POINTS,
    ROLLBACK_DRILL_FAULT_POINTS,
    cleanup_okf_cutover,
    discover_okf_startup,
    execute_okf_cutover,
    finalize_okf_rebuild,
    okf_rebuild_session,
    recover_okf_cutover,
    recutover_okf_rebuild,
    rollback_okf_rebuild,
)
from chronovisor.core.okf_workspace import (
    RESTART_REFUSAL_FILENAME,
    prepare_okf_workspace,
)
from chronovisor.core.page_identity import new_page_uid
from chronovisor.hosts import cli
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.ops import okf_rebuild

FIXTURE = Path(__file__).parent / "fixtures" / "okf_workspace" / "source"
PROFILE = {
    "role": "search.semantic.foreground",
    "provider": "injected-local",
    "model": "fixture-embedding",
    "location": "local",
    "revision": "fixture-v1",
    "dimensions": 4,
    "query_prefix": "query: ",
    "document_prefix": "passage: ",
    "batch_size": 8,
}


class InjectedCrash(RuntimeError):
    pass


def _encoder_calls(calls: list[int]):
    def encode(
        documents: Sequence[semantic_index.SemanticDocument],
        batch_size: int,
    ) -> np.ndarray:
        calls.append(batch_size)
        return np.asarray(
            [
                [float(index + 1), 1.0, 2.0, 3.0]
                for index, _document in enumerate(documents)
            ],
            dtype=np.float32,
        )

    return encode


def _replace_fixture_uids(root: Path) -> None:
    for index, (relative, legacy) in enumerate(
        (
            ("pages/deep/target.md", "uid-target"),
            ("pages/notes/source.md", "uid-source"),
        )
    ):
        path = root / relative
        uid = new_page_uid(
            timestamp_ms=1_700_000_000_000 + index,
            random_bits=index + 1,
        )
        path.write_text(
            path.read_text(encoding="utf-8").replace(legacy, uid),
            encoding="utf-8",
        )


def _setup_committed(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = (tmp_path / "source").resolve()
    shutil.copytree(FIXTURE, root)
    _replace_fixture_uids(root)
    (root / "pages" / "draft.md").write_text(
        "---\n"
        f"uid: {new_page_uid(timestamp_ms=1_700_000_000_010, random_bits=10)}\n"
        "type: Concept\nstatus: draft\ntitle: Draft\n---\nDraft canary.\n",
        encoding="utf-8",
    )
    (root / "pages" / "old.md").write_text(
        "---\n"
        f"uid: {new_page_uid(timestamp_ms=1_700_000_000_011, random_bits=11)}\n"
        "type: Concept\nstatus: archived\ntitle: Old\n---\nDeprecated canary.\n",
        encoding="utf-8",
    )
    runtime = root / "runtime"
    runtime.mkdir()
    (runtime / "activity.jsonl").write_bytes(
        canonical_json_line_bytes_strict(
            activity_record(
                "pre-migration activity",
                source="test",
                timestamp="2026-08-11T00:00:00+09:00",
                event_id="activity-" + "1" * 64,
            )
        )
    )
    workspace = prepare_okf_workspace(root, runtime, "run-001")
    assert (
        execute_okf_cutover(root, runtime, "run-001", is_quiescent=lambda: True)
        == "committed-needs-rebuild"
    )
    return root, runtime, workspace


def _expected_stable_paths(root: Path) -> list[Path]:
    paths = canonical_document_paths(
        root / "pages",
        system_dir=root / "system",
        require_stable=True,
        strict=False,
    )
    return sorted(paths)


def _tree(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _seal(root: Path) -> dict[str, Any]:
    return okf_rebuild.rebuild_okf_derived(
        root,
        "run-001",
        is_quiescent=lambda: True,
        semantic_encoder=_encoder_calls([]),
        semantic_profile=PROFILE,
    )


def test_offline_rebuild_seals_exact_stable_corpus_without_authority(
    tmp_path: Path,
) -> None:
    root, runtime, workspace = _setup_committed(tmp_path)
    (root / "knowledge-graph").mkdir()
    (root / "knowledge-graph" / "legacy-canary").write_text("old graph")
    write_sealed_json(
        runtime / "typed-graph" / "status.json",
        {
            "mode": "active",
            "authority_mature": True,
            "authority": {"enabled": True},
        },
        backup=False,
    )
    calls: list[int] = []

    result = okf_rebuild.rebuild_okf_derived(
        root,
        "run-001",
        is_quiescent=lambda: True,
        semantic_encoder=_encoder_calls(calls),
        semantic_profile=PROFILE,
    )

    expected = _expected_stable_paths(root)
    expected_ids = {path.stem for path in expected}
    assert result["status"] == "sealed-rebuild"
    assert result["stable_page_count"] == len(expected)
    assert calls
    assert not {"draft", "old"}.intersection(expected_ids)
    assert "schema" in expected_ids
    assert discover_okf_startup(root, runtime).category == "rollback_drill_required"
    assert json.loads((workspace / "journal.json").read_bytes())["state"] == (
        "sealed-rebuild"
    )
    assert json.loads(
        (workspace / RESTART_REFUSAL_FILENAME).read_bytes()
    )["state"] == "sealed-rebuild"

    proof_raw = (workspace / "rebuild-proof.json").read_bytes()
    proof = json.loads(proof_raw)
    assert "Target body" not in proof_raw.decode("utf-8")
    assert proof["corpus"]["stable_page_count"] == len(expected)
    components = proof["components"]
    assert components["registry"]["stable_count"] == len(expected)
    assert components["uid_links"]["unresolved_count"] == 0
    assert components["index_store"]["page_count"] == len(expected)
    assert components["lexical"]["page_count"] == len(expected)
    assert components["semantic"]["page_count"] == len(expected)
    assert components["knowledge_graph"]["page_count"] == len(expected)
    assert components["knowledge_graph"]["external_model_calls"] == 0
    assert components["cortex"]["node_count"] == len(expected)
    assert components["cortex"]["authority_enabled"] is False
    assert components["cortex"]["runtime_state_present"] is False
    assert components["invalidation"]["target_count"] == 0

    registry = PageRegistry(root)
    stable_rows = registry.stable_pages()
    assert {str(row["path"]) for row in stable_rows.values()} == {
        str(path.relative_to(root)) for path in expected
    }
    metadata = json.loads((root / ".index" / "pages.json").read_bytes())
    assert {
        page_id
        for page_id, entry in metadata["entries"].items()
        if entry["status"] == "stable"
    } == expected_ids
    connection = sqlite3.connect(
        f"file:{root / '.index' / 'lexical.sqlite'}?mode=ro",
        uri=True,
    )
    try:
        assert {
            str(row[0]) for row in connection.execute("SELECT page_id FROM pages")
        } == expected_ids
    finally:
        connection.close()
    assert (workspace / "derived-rebuild" / "previous-knowledge-graph" / "legacy-canary").is_file()
    assert not (runtime / "typed-graph").exists()
    assert (workspace / "derived-rebuild" / "previous-typed-graph").is_dir()


def test_sealed_rebuild_resume_only_converges_sentinel(
    tmp_path: Path,
) -> None:
    root, runtime, workspace = _setup_committed(tmp_path)
    first = okf_rebuild.rebuild_okf_derived(
        root,
        "run-001",
        is_quiescent=lambda: True,
        semantic_encoder=_encoder_calls([]),
        semantic_profile=PROFILE,
    )
    proof_before = (workspace / "rebuild-proof.json").read_bytes()
    sentinel = json.loads((workspace / RESTART_REFUSAL_FILENAME).read_bytes())
    sentinel["state"] = "rebuild-in-progress"
    (workspace / RESTART_REFUSAL_FILENAME).write_bytes(
        canonical_json_line_bytes_strict(sentinel)
    )

    second = okf_rebuild.rebuild_okf_derived(
        root,
        "run-001",
        is_quiescent=lambda: True,
        semantic_encoder=lambda *_args: (_ for _ in ()).throw(
            AssertionError("sealed resume must not encode")
        ),
        semantic_profile=PROFILE,
    )

    assert second == first
    assert (workspace / "rebuild-proof.json").read_bytes() == proof_before
    assert json.loads(
        (workspace / RESTART_REFUSAL_FILENAME).read_bytes()
    )["state"] == "sealed-rebuild"
    assert discover_okf_startup(root, runtime).category == "rollback_drill_required"

    with okf_rebuild_session(
        root,
        runtime,
        "run-001",
        is_quiescent=lambda: True,
    ) as sealed_session:
        with pytest.raises(RuntimeError, match="already sealed"):
            sealed_session.publish_proof({})


def test_rebuild_session_expires_and_keeps_startup_blocked(tmp_path: Path) -> None:
    root, runtime, workspace = _setup_committed(tmp_path)

    with okf_rebuild_session(
        root,
        runtime,
        "run-001",
        is_quiescent=lambda: True,
    ) as session:
        leaked = session
        assert json.loads((workspace / "journal.json").read_bytes())["state"] == (
            "rebuild-in-progress"
        )

    with pytest.raises(RuntimeError, match="no longer active"):
        leaked.publish_proof({})
    assert discover_okf_startup(root, runtime).category == "rebuild_in_progress"


def test_index_store_rebuild_gate_is_root_and_lifetime_bound(
    tmp_path: Path,
) -> None:
    root, runtime, workspace = _setup_committed(tmp_path)
    staged_index = workspace / "derived-rebuild" / "index"
    staged_index.mkdir(parents=True)

    with okf_rebuild_session(
        root,
        runtime,
        "run-001",
        is_quiescent=lambda: True,
    ) as session:
        store = IndexStore(root, _index_dir=staged_index)
        store._refresh_under_external_gate(session.gate)
        wrong_root = tmp_path / "wrong-root"
        wrong_root.mkdir()
        with pytest.raises(RuntimeError, match="root does not match"):
            IndexStore(wrong_root)._refresh_under_external_gate(session.gate)
        leaked_gate = session.gate

    with pytest.raises(RuntimeError, match="not active"):
        IndexStore(root, _index_dir=staged_index)._refresh_under_external_gate(
            leaked_gate
        )


def test_rebuild_failure_resumes_without_reencoding_semantic_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _runtime, workspace = _setup_committed(tmp_path)
    calls: list[int] = []
    real_graph = okf_rebuild._rebuild_knowledge_graph

    def crash_graph(*_args, **_kwargs):
        raise RuntimeError("injected graph crash")

    monkeypatch.setattr(okf_rebuild, "_rebuild_knowledge_graph", crash_graph)
    with pytest.raises(RuntimeError, match="injected graph crash"):
        okf_rebuild.rebuild_okf_derived(
            root,
            "run-001",
            is_quiescent=lambda: True,
            semantic_encoder=_encoder_calls(calls),
            semantic_profile=PROFILE,
        )
    assert json.loads((workspace / "journal.json").read_bytes())["state"] == (
        "rebuild-in-progress"
    )
    assert len(calls) == 1
    assert (workspace / "derived-rebuild" / "previous-index").is_dir()
    assert not (workspace / "derived-rebuild" / "index").exists()

    monkeypatch.setattr(okf_rebuild, "_rebuild_knowledge_graph", real_graph)
    result = okf_rebuild.rebuild_okf_derived(
        root,
        "run-001",
        is_quiescent=lambda: True,
        semantic_encoder=_encoder_calls(calls),
        semantic_profile=PROFILE,
    )
    assert result["status"] == "sealed-rebuild"
    assert len(calls) == 1


def test_semantic_failure_keeps_live_index_generation(tmp_path: Path) -> None:
    root, _runtime, workspace = _setup_committed(tmp_path)
    live = root / ".index"
    live.mkdir()
    canary = live / "live-canary"
    canary.write_text("old-live", encoding="utf-8")

    def fail(*_args):
        raise RuntimeError("encode failed")

    with pytest.raises(RuntimeError, match="encode failed"):
        okf_rebuild.rebuild_okf_derived(
            root,
            "run-001",
            is_quiescent=lambda: True,
            semantic_encoder=fail,
            semantic_profile=PROFILE,
        )

    assert canary.read_text(encoding="utf-8") == "old-live"
    assert not (workspace / "derived-rebuild" / "previous-index").exists()


@pytest.mark.parametrize(
    "unsafe_path",
    ("live-index", "live-semantic", "workspace-derived"),
)
def test_rebuild_rejects_index_symlinks_without_outside_writes(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    root, _runtime, workspace = _setup_committed(tmp_path)
    outside = tmp_path / "outside-index"
    outside.mkdir()
    canary = outside / "canary"
    canary.write_text("outside", encoding="utf-8")
    if unsafe_path == "live-index":
        (root / ".index").symlink_to(outside, target_is_directory=True)
        error = "derived index directory is unsafe"
    elif unsafe_path == "live-semantic":
        (root / ".index").mkdir()
        (root / ".index" / "semantic").symlink_to(
            outside,
            target_is_directory=True,
        )
        error = "derived semantic directory is unsafe"
    else:
        (workspace / "derived-rebuild").symlink_to(
            outside,
            target_is_directory=True,
        )
        error = "derived rebuild workspace is unsafe"

    calls: list[int] = []
    with pytest.raises((OSError, ValueError), match=error):
        okf_rebuild.rebuild_okf_derived(
            root,
            "run-001",
            is_quiescent=lambda: True,
            semantic_encoder=_encoder_calls(calls),
            semantic_profile=PROFILE,
        )

    assert calls == []
    assert _tree(outside) == {"canary": b"outside"}


def test_semantic_activation_cannot_publish_through_swapped_live_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _runtime, _workspace = _setup_committed(tmp_path)
    live_semantic = root / ".index" / "semantic"
    live_semantic.mkdir(parents=True)
    (live_semantic / "old-canary").write_text("old", encoding="utf-8")
    moved_semantic = root / "moved-semantic"
    outside = tmp_path / "outside-semantic"
    outside.mkdir()
    canary = outside / "canary"
    canary.write_text("outside", encoding="utf-8")
    real_activate = semantic_index.activate_generation

    def swap_then_activate(
        generation_id: str,
        *,
        expected_current: str | None,
        root: Path,
    ):
        live_semantic.rename(moved_semantic)
        live_semantic.symlink_to(outside, target_is_directory=True)
        return real_activate(
            generation_id,
            expected_current=expected_current,
            root=root,
        )

    monkeypatch.setattr(semantic_index, "activate_generation", swap_then_activate)
    with pytest.raises(ValueError, match="derived semantic directory is unsafe"):
        okf_rebuild.rebuild_okf_derived(
            root,
            "run-001",
            is_quiescent=lambda: True,
            semantic_encoder=_encoder_calls([]),
            semantic_profile=PROFILE,
        )

    assert _tree(outside) == {"canary": b"outside"}
    assert (moved_semantic / "old-canary").read_text(encoding="utf-8") == "old"


def test_activity_suffix_drift_prevents_rebuild_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, runtime, workspace = _setup_committed(tmp_path)
    real_graph = okf_rebuild._rebuild_knowledge_graph

    def drift_then_return(*args, **kwargs):
        result = real_graph(*args, **kwargs)
        with (runtime / "activity.jsonl").open("ab") as handle:
            handle.write(
                canonical_json_line_bytes_strict(
                    activity_record(
                        "drift during rebuild",
                        source="test",
                        timestamp="2026-08-11T01:00:00+09:00",
                        event_id="activity-" + "7" * 64,
                    )
                )
            )
        return result

    monkeypatch.setattr(okf_rebuild, "_rebuild_knowledge_graph", drift_then_return)
    with pytest.raises(ValueError, match="suffix changed"):
        okf_rebuild.rebuild_okf_derived(
            root,
            "run-001",
            is_quiescent=lambda: True,
            semantic_encoder=_encoder_calls([]),
            semantic_profile=PROFILE,
        )

    assert json.loads((workspace / "journal.json").read_bytes())["state"] == (
        "rebuild-in-progress"
    )
    assert discover_okf_startup(root, runtime).category == "rebuild_in_progress"


def test_fresh_knowledge_graph_has_same_logical_relation_identity(
    tmp_path: Path,
) -> None:
    hashes = []
    for name in ("first", "second"):
        root, _runtime, workspace = _setup_committed(tmp_path / name)
        okf_rebuild.rebuild_okf_derived(
            root,
            "run-001",
            is_quiescent=lambda: True,
            semantic_encoder=_encoder_calls([]),
            semantic_profile=PROFILE,
        )
        proof = json.loads((workspace / "rebuild-proof.json").read_bytes())
        hashes.append(
            proof["components"]["knowledge_graph"]["relation_set_sha256"]
        )

    assert hashes[0] == hashes[1]


def test_remote_semantic_route_is_rejected_before_embedding_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    class Runtime:
        def resolve_embedding(self, _role: str):
            return SimpleNamespace(
                role="search.semantic.foreground",
                provider="remote-provider",
                model="remote-model",
                location=RouteLocation.REMOTE,
            )

        def embed(self, *_args, **_kwargs):
            nonlocal called
            called = True
            raise AssertionError("remote embed must not be called")

    monkeypatch.setattr(
        okf_rebuild,
        "load_search_embedding_config",
        lambda: SimpleNamespace(enabled=True),
    )
    monkeypatch.setattr(okf_rebuild, "load_default_llm_runtime", Runtime)

    with pytest.raises(RuntimeError, match="local embedding route"):
        okf_rebuild._production_semantic_encoder()
    assert called is False


def test_semantic_encode_failure_preserves_active_pointer(tmp_path: Path) -> None:
    root, runtime, _workspace = _setup_committed(tmp_path)
    semantic_root = root / ".index" / "semantic"
    semantic_root.mkdir(parents=True)
    active = semantic_root / "active.json"
    active.write_text('{"generation_id":"old"}\n', encoding="utf-8")
    before = active.read_bytes()

    with okf_rebuild_session(
        root,
        runtime,
        "run-001",
        is_quiescent=lambda: True,
    ) as session:
        paths, _rows = okf_rebuild._stable_sources(root)

        def fail(*_args):
            raise RuntimeError("encode failed")

        with pytest.raises(RuntimeError, match="encode failed"):
            okf_rebuild._rebuild_semantic(
                root,
                session.gate,
                paths,
                index_dir=root / ".index",
                derived_generation="okf-" + "1" * 24,
                encoder=fail,
                profile=PROFILE,
            )

    assert active.read_bytes() == before


def test_semantic_activation_cas_conflict_does_not_publish_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, runtime, _workspace = _setup_committed(tmp_path)
    semantic_root = root / ".index" / "semantic"
    semantic_root.mkdir(parents=True)
    active = semantic_root / "active.json"
    active.write_text('{"generation_id":"old"}\n', encoding="utf-8")
    real_activate = semantic_index.activate_generation

    def race(generation_id: str, *, expected_current: str | None, root: Path):
        active.write_text('{"generation_id":"racer"}\n', encoding="utf-8")
        return real_activate(
            generation_id,
            expected_current=expected_current,
            root=root,
        )

    monkeypatch.setattr(semantic_index, "activate_generation", race)
    with okf_rebuild_session(
        root,
        runtime,
        "run-001",
        is_quiescent=lambda: True,
    ) as session:
        paths, _rows = okf_rebuild._stable_sources(root)
        with pytest.raises(semantic_index.SemanticIndexError, match="CAS failed"):
            okf_rebuild._rebuild_semantic(
                root,
                session.gate,
                paths,
                index_dir=root / ".index",
                derived_generation="okf-" + "2" * 24,
                encoder=_encoder_calls([]),
                profile=PROFILE,
            )

    assert json.loads(active.read_bytes())["generation_id"] == "racer"


def test_symlink_root_is_rejected_without_following_it(tmp_path: Path) -> None:
    root, _runtime, _workspace = _setup_committed(tmp_path)
    link = tmp_path / "linked-root"
    link.symlink_to(root, target_is_directory=True)

    with pytest.raises((OSError, ValueError)):
        okf_rebuild.rebuild_okf_derived(
            link,
            "run-001",
            is_quiescent=lambda: True,
            semantic_encoder=_encoder_calls([]),
            semantic_profile=PROFILE,
        )


@pytest.mark.parametrize("operation_name", ("_publish_directory", "_retire_directory"))
@pytest.mark.parametrize("swap_parent", ("source", "target"))
def test_directory_moves_stay_inside_pinned_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
    swap_parent: str,
) -> None:
    source_parent = tmp_path / "source-parent"
    target_parent = tmp_path / "target-parent"
    source = source_parent / "projection"
    target = target_parent / "projection"
    source.mkdir(parents=True)
    target_parent.mkdir()
    (source / "inside").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    canary = outside / "canary"
    canary.write_text("outside", encoding="utf-8")
    moved_parent = tmp_path / f"moved-{swap_parent}-parent"
    real_rename = os.rename
    swapped = False

    def swap_then_rename(src, dst, *args, **kwargs):
        nonlocal swapped
        if not swapped and kwargs.get("src_dir_fd") is not None:
            swapped = True
            parent = source_parent if swap_parent == "source" else target_parent
            real_rename(parent, moved_parent)
            parent.symlink_to(outside, target_is_directory=True)
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", swap_then_rename)
    getattr(okf_rebuild, operation_name)(source, target)

    assert canary.read_text(encoding="utf-8") == "outside"
    assert not (outside / "projection").exists()
    actual_target_parent = (
        moved_parent if swap_parent == "target" else target_parent
    )
    assert (actual_target_parent / "projection" / "inside").read_text(
        encoding="utf-8"
    ) == "inside"


def test_rollback_recutover_and_final_receipt_preserve_activity_suffix(
    tmp_path: Path,
) -> None:
    root, runtime, workspace = _setup_committed(tmp_path)
    old_pages = _tree(workspace / "rollback-backup" / "pages")
    new_pages = _tree(root / "pages")
    suffix_row = canonical_json_line_bytes_strict(
        activity_record(
            "post-cutover suffix",
            source="test",
            timestamp="2026-08-11T02:00:00+09:00",
            event_id="activity-" + "8" * 64,
        )
    )
    with (runtime / "activity.jsonl").open("ab") as handle:
        handle.write(suffix_row)
    new_activity = (runtime / "activity.jsonl").read_bytes()
    _seal(root)

    assert rollback_okf_rebuild(
        root, runtime, "run-001", is_quiescent=lambda: True
    ) == "rollback-drill-complete"
    assert _tree(root / "pages") == old_pages
    assert _tree(workspace / "staging" / "pages") == new_pages
    assert (workspace / "staging" / "activity.jsonl").read_bytes() == new_activity
    assert discover_okf_startup(root, runtime).category == "recutover_required"

    assert recutover_okf_rebuild(
        root, runtime, "run-001", is_quiescent=lambda: True
    ) == "finalized-v2"
    assert _tree(root / "pages") == new_pages
    assert (runtime / "activity.jsonl").read_bytes() == new_activity
    live_decision = discover_okf_startup(root, runtime)
    assert live_decision.allowed is True
    assert live_decision.layout == "okf_v0_2"
    assert live_decision.state == "finalized-v2"
    manifest = json.loads((workspace / "dry-run-manifest.json").read_bytes())
    assert all("uid" not in row for row in manifest["system_documents"])
    expected_cohorts = []
    for scope, cohort_field in (
        ("pages", "status_cohorts"),
        ("system", "system_status_cohorts"),
    ):
        for cohort in manifest[cohort_field]:
            identity_set_sha256 = (
                hashlib.sha256(
                    canonical_json_line_bytes_strict(sorted(cohort["uids"]))
                ).hexdigest()
                if scope == "pages"
                else cohort["identity_set_sha256"]
            )
            expected_cohorts.append(
                {
                    "scope": scope,
                    "input_status": cohort["input_status"],
                    "output_status": cohort["output_status"],
                    "count": cohort["count"],
                    "identity_set_sha256": identity_set_sha256,
                }
            )

    assert finalize_okf_rebuild(
        root, runtime, "run-001", is_quiescent=lambda: True
    ) == "finalized-v2"
    assert {path.name for path in workspace.iterdir()} == {"receipt.json"}
    receipt = json.loads((workspace / "receipt.json").read_bytes())
    assert "Target body" not in (workspace / "receipt.json").read_text()
    assert receipt["schema"] == FINAL_RECEIPT_SCHEMA
    assert receipt["state"] == "finalized-v2"
    assert receipt["rollback_recutover"] == {
        "rollback": "complete",
        "recutover": "complete",
    }
    assert receipt["status_mapping_cohorts"] == expected_cohorts
    assert receipt["activity_suffix"]["length"] == len(suffix_row)
    assert not any(
        (root / name).exists() for name in ("index.md", "log.md", "schema.md")
    )
    decision = discover_okf_startup(root, runtime)
    assert decision.allowed is True
    assert decision.layout == "okf_v0_2"
    assert decision.state == "finalized-v2"
    with (runtime / "activity.jsonl").open("ab") as handle:
        handle.write(
            canonical_json_line_bytes_strict(
                activity_record(
                    "valid mutable suffix",
                    source="test",
                    timestamp="2026-08-11T02:01:00+09:00",
                    event_id="activity-" + "a" * 64,
                )
            )
        )
    assert discover_okf_startup(root, runtime).allowed is True


@pytest.mark.parametrize("fault_point", ROLLBACK_DRILL_FAULT_POINTS)
def test_every_rollback_drill_boundary_resumes_to_all_old(
    tmp_path: Path,
    fault_point: str,
) -> None:
    root, runtime, workspace = _setup_committed(tmp_path)
    old_pages = _tree(workspace / "rollback-backup" / "pages")
    _seal(root)

    def crash(point: str) -> None:
        if point == fault_point:
            raise InjectedCrash(point)

    with pytest.raises(InjectedCrash, match=fault_point):
        rollback_okf_rebuild(
            root,
            runtime,
            "run-001",
            is_quiescent=lambda: True,
            fault_inject=crash,
        )
    state = recover_okf_cutover(
        root, runtime, "run-001", is_quiescent=lambda: True
    )
    if state == "sealed-rebuild":
        state = rollback_okf_rebuild(
            root, runtime, "run-001", is_quiescent=lambda: True
        )
    assert state == "rollback-drill-complete"
    assert _tree(root / "pages") == old_pages
    assert discover_okf_startup(root, runtime).category == "recutover_required"


@pytest.mark.parametrize("fault_point", RECUTOVER_FAULT_POINTS)
def test_every_recutover_boundary_resumes_to_all_new(
    tmp_path: Path,
    fault_point: str,
) -> None:
    root, runtime, workspace = _setup_committed(tmp_path)
    new_pages = _tree(root / "pages")
    _seal(root)
    rollback_okf_rebuild(root, runtime, "run-001", is_quiescent=lambda: True)

    def crash(point: str) -> None:
        if point == fault_point:
            raise InjectedCrash(point)

    with pytest.raises(InjectedCrash, match=fault_point):
        recutover_okf_rebuild(
            root,
            runtime,
            "run-001",
            is_quiescent=lambda: True,
            fault_inject=crash,
        )
    state = recover_okf_cutover(
        root, runtime, "run-001", is_quiescent=lambda: True
    )
    if state == "rollback-drill-complete":
        state = recutover_okf_rebuild(
            root, runtime, "run-001", is_quiescent=lambda: True
        )
    assert state == "finalized-v2"
    assert _tree(root / "pages") == new_pages
    assert not (workspace / RESTART_REFUSAL_FILENAME).exists()


@pytest.mark.parametrize("fault_point", FINALIZE_FAULT_POINTS)
def test_every_final_cleanup_boundary_resumes_to_receipt_only(
    tmp_path: Path,
    fault_point: str,
) -> None:
    root, runtime, workspace = _setup_committed(tmp_path)
    _seal(root)
    rollback_okf_rebuild(root, runtime, "run-001", is_quiescent=lambda: True)
    recutover_okf_rebuild(root, runtime, "run-001", is_quiescent=lambda: True)

    def crash(point: str) -> None:
        if point == fault_point:
            raise InjectedCrash(point)

    with pytest.raises(InjectedCrash, match=fault_point):
        finalize_okf_rebuild(
            root,
            runtime,
            "run-001",
            is_quiescent=lambda: True,
            fault_inject=crash,
        )
    assert discover_okf_startup(root, runtime).allowed is (
        fault_point
        in {"before-final-receipt-write", "after-final-remove-journal"}
    )
    assert finalize_okf_rebuild(
        root, runtime, "run-001", is_quiescent=lambda: True
    ) == "finalized-v2"
    assert {path.name for path in workspace.iterdir()} == {"receipt.json"}
    assert discover_okf_startup(root, runtime).allowed is True


def test_recutover_rejects_tampered_staged_activity_suffix(tmp_path: Path) -> None:
    root, runtime, workspace = _setup_committed(tmp_path)
    _seal(root)
    rollback_okf_rebuild(root, runtime, "run-001", is_quiescent=lambda: True)
    with (workspace / "staging" / "activity.jsonl").open("ab") as handle:
        handle.write(
            canonical_json_line_bytes_strict(
                activity_record(
                    "tampered suffix",
                    source="test",
                    timestamp="2026-08-11T03:00:00+09:00",
                    event_id="activity-" + "9" * 64,
                )
            )
        )

    with pytest.raises(ValueError, match="suffix identity changed"):
        recutover_okf_rebuild(
            root, runtime, "run-001", is_quiescent=lambda: True
        )
    assert discover_okf_startup(root, runtime).category == "recutover_required"


@pytest.mark.parametrize("asset", ("pages-log", "system-schema", "activity-prefix"))
def test_final_receipt_startup_rejects_tampered_immutable_asset(
    tmp_path: Path,
    asset: str,
) -> None:
    root, runtime, _workspace = _setup_committed(tmp_path)
    _seal(root)
    rollback_okf_rebuild(root, runtime, "run-001", is_quiescent=lambda: True)
    recutover_okf_rebuild(root, runtime, "run-001", is_quiescent=lambda: True)
    finalize_okf_rebuild(root, runtime, "run-001", is_quiescent=lambda: True)
    if asset == "pages-log":
        path = root / "pages" / "log.md"
        path.write_bytes(path.read_bytes() + b"migration canary\n")
    elif asset == "system-schema":
        path = root / "system" / "schema.md"
        path.write_bytes(path.read_bytes() + b"migration canary\n")
    else:
        path = runtime / "activity.jsonl"
        rows = path.read_bytes().splitlines()
        payload = json.loads(rows[0])
        payload["source"] = "tampered-source"
        rows[0] = canonical_json_line_bytes_strict(payload).rstrip(b"\n")
        path.write_bytes(b"\n".join(rows) + b"\n")

    decision = discover_okf_startup(root, runtime)
    assert decision.allowed is False
    assert decision.category == "migration_receipt_invalid"


@pytest.mark.parametrize("tamper", ("extra", "outcome", "cohort-hash"))
def test_final_receipt_is_canonical_private_and_exact(
    tmp_path: Path,
    tamper: str,
) -> None:
    root, runtime, workspace = _setup_committed(tmp_path)
    _seal(root)
    rollback_okf_rebuild(root, runtime, "run-001", is_quiescent=lambda: True)
    recutover_okf_rebuild(root, runtime, "run-001", is_quiescent=lambda: True)
    finalize_okf_rebuild(root, runtime, "run-001", is_quiescent=lambda: True)
    receipt_path = workspace / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())

    assert receipt_path.read_bytes() == canonical_json_line_bytes_strict(receipt)
    assert receipt_path.stat().st_mode & 0o777 == 0o600

    unsigned = {key: value for key, value in receipt.items() if key != "seal_sha256"}
    if tamper == "extra":
        unsigned["unexpected"] = True
    elif tamper == "outcome":
        unsigned["rollback_recutover"]["rollback"] = "skipped"
    else:
        unsigned["status_mapping_cohorts"][0]["identity_set_sha256"] = "invalid"
    write_sealed_json(receipt_path, unsigned, backup=False, min_free_bytes=0)

    decision = discover_okf_startup(root, runtime)
    assert decision.allowed is False
    assert decision.category == "migration_receipt_invalid"


def test_okf_lifecycle_cli_delegates_to_offline_coordinators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "root"
    calls: list[str] = []
    monkeypatch.setattr(chronovisor_store, "CHRONOVISOR_ROOT", root)
    monkeypatch.setattr(
        chronovisor_store,
        "okf_startup_status",
        lambda _root: okf_cutover.OKFStartupDecision(
            False,
            "blocked",
            "committed-needs-rebuild",
            "offline",
            "run-001",
        ),
    )

    def state_call(name: str, state: str):
        def call(*_args, **_kwargs):
            calls.append(name)
            return state

        return call

    monkeypatch.setattr(
        okf_cutover,
        "execute_okf_cutover",
        state_call("execute", "committed-needs-rebuild"),
    )
    monkeypatch.setattr(
        okf_cutover,
        "recover_okf_cutover",
        state_call("recover", "sealed-rebuild"),
    )
    monkeypatch.setattr(
        okf_cutover,
        "rollback_okf_rebuild",
        state_call("rollback", "rollback-drill-complete"),
    )
    monkeypatch.setattr(
        okf_cutover,
        "recutover_okf_rebuild",
        state_call("recutover", "finalized-v2"),
    )
    monkeypatch.setattr(
        okf_cutover,
        "finalize_okf_rebuild",
        state_call("finalize-cleanup", "finalized-v2"),
    )
    monkeypatch.setattr(
        okf_rebuild,
        "rebuild_okf_derived",
        lambda *_args, **_kwargs: calls.append("rebuild-seal")
        or {"status": "sealed-rebuild"},
    )

    for command in (
        "execute",
        "recover",
        "rollback",
        "recutover",
        "rebuild-seal",
        "finalize-cleanup",
    ):
        assert cli.main(
            [
                "okf",
                command,
                "--run-id",
                "run-001",
                "--json",
            ]
        ) == 0
        assert json.loads(capsys.readouterr().out)["ok"] is True
    assert calls == [
        "execute",
        "recover",
        "rollback",
        "recutover",
        "rebuild-seal",
        "finalize-cleanup",
    ]


def test_okf_mutating_cli_rejects_custom_root_before_coordinator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = False

    def must_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("coordinator must not run for a custom root")

    monkeypatch.setattr(okf_cutover, "execute_okf_cutover", must_not_run)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "okf",
                "execute",
                "--run-id",
                "run-001",
                "--root",
                str(tmp_path / "root"),
                "--json",
            ]
        )
    assert exc_info.value.code == 2
    assert "invalid arguments" in capsys.readouterr().err
    assert called is False


def test_okf_mutating_cli_refuses_while_shared_writer_lease_is_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = (tmp_path / "source").resolve()
    shutil.copytree(FIXTURE, root)
    runtime = root / "runtime"
    runtime.mkdir()
    workspace = prepare_okf_workspace(root, runtime, "run-001")
    before = _tree(workspace)
    monkeypatch.setattr(chronovisor_store, "CHRONOVISOR_ROOT", root)
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from chronovisor.core.durable_state import okf_writer_lock\n"
        "with okf_writer_lock(Path(sys.argv[1])):\n"
        " print('ready', flush=True)\n"
        " sys.stdin.readline()\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        assert cli.main(
            ["okf", "execute", "--run-id", "run-001", "--json"]
        ) == 75
        assert json.loads(capsys.readouterr().out)["category"] == "execute_failed"
        assert _tree(workspace) == before
    finally:
        if process.stdin is not None:
            process.stdin.write("\n")
            process.stdin.flush()
        process.wait(timeout=10)


def test_cleanup_tree_rejects_directory_swapped_to_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "private").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    canary = outside / "canary"
    canary.write_text("outside", encoding="utf-8")
    moved = tmp_path / "moved"
    real_remove = okf_cutover._remove_tree_at
    swapped = False

    def swap_then_remove(parent_fd: int, name: str) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            tree.rename(moved)
            tree.symlink_to(outside, target_is_directory=True)
        real_remove(parent_fd, name)

    monkeypatch.setattr(okf_cutover, "_remove_tree_at", swap_then_remove)

    with pytest.raises(ValueError, match="safe directory"):
        okf_cutover._remove_tree_exact(tree)
    assert canary.read_text(encoding="utf-8") == "outside"
    assert (moved / "private").read_text(encoding="utf-8") == "inside"


def test_cleanup_tree_keeps_ancestor_swap_inside_pinned_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migrations = tmp_path / "runtime" / "migrations"
    tree = migrations / "run-001" / "derived-rebuild"
    tree.mkdir(parents=True)
    (tree / "private").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside_tree = outside / "run-001" / "derived-rebuild"
    outside_tree.mkdir(parents=True)
    canary = outside_tree / "canary"
    canary.write_text("outside", encoding="utf-8")
    moved = tmp_path / "migrations-moved"
    real_remove = okf_cutover._remove_tree_at
    swapped = False

    def swap_ancestor_then_remove(parent_fd: int, name: str) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            migrations.rename(moved)
            migrations.symlink_to(outside, target_is_directory=True)
        real_remove(parent_fd, name)

    monkeypatch.setattr(okf_cutover, "_remove_tree_at", swap_ancestor_then_remove)

    okf_cutover._remove_tree_exact(tree)
    assert canary.read_text(encoding="utf-8") == "outside"
    assert not (moved / "run-001" / "derived-rebuild").exists()


def test_legacy_cleanup_rejects_v2_only_derived_artifacts(tmp_path: Path) -> None:
    root = (tmp_path / "source").resolve()
    shutil.copytree(FIXTURE, root)
    runtime = root / "runtime"
    runtime.mkdir()
    workspace = prepare_okf_workspace(root, runtime, "run-001")
    assert recover_okf_cutover(
        root, runtime, "run-001", is_quiescent=lambda: True
    ) == "rollback-complete"
    derived = workspace / "derived-rebuild"
    derived.mkdir()
    (derived / "canary").write_text("private", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown artifact"):
        cleanup_okf_cutover(
            root, runtime, "run-001", is_quiescent=lambda: True
        )
    assert (derived / "canary").read_text(encoding="utf-8") == "private"
    assert not (workspace / "receipt.json").exists()
