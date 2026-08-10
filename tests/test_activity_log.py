from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from chronovisor.core import activity_log, store
from chronovisor.core.canonical_json import canonical_json_line_bytes_strict
from chronovisor.core.okf_cutover import OKFStartupBlocked, discover_okf_startup


def _append_worker(root_text: str, worker: int, count: int) -> None:
    root = Path(root_text)
    for index in range(count):
        activity_log.append_activity(
            f"worker {worker} event {index}",
            source="test-worker",
            root=root,
        )


def _init_worker(root_text: str) -> None:
    root = Path(root_text)
    store.init_chronovisor(store.RuntimeContext(root))


def _fresh_root(tmp_path: Path) -> Path:
    root = tmp_path / "wiki"
    store.init_chronovisor(store.RuntimeContext(root))
    return root


def test_activity_is_structured_durable_and_never_changes_portable_log(
    tmp_path: Path,
) -> None:
    root = _fresh_root(tmp_path)
    portable_log = root / "pages" / "log.md"
    before = portable_log.read_bytes()

    written = activity_log.append_activity(
        "ingest | created alpha.md",
        source="ingest",
        level="success",
        root=root,
        timestamp="2026-08-11T10:30:00+09:00",
    )

    assert written["schema"] == activity_log.ACTIVITY_SCHEMA
    assert written["event_id"].startswith("activity-")
    assert activity_log.read_activity(root / "runtime" / "activity.jsonl") == [
        written
    ]
    assert portable_log.read_bytes() == before


def test_activity_rejects_unsafe_message_and_symlink_leaf(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    with pytest.raises(ValueError, match="control"):
        activity_log.append_activity("bad\rmessage", source="test", root=root)

    target = root / "runtime" / "activity.jsonl"
    outside = tmp_path / "outside.jsonl"
    outside.write_text("")
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe"):
        activity_log.append_activity("must not escape", source="test", root=root)
    assert outside.read_bytes() == b""
    assert activity_log.read_activity(target) == []


def test_activity_delta_reader_rejects_symlink_parent(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    target = outside / "activity.jsonl"
    target.write_bytes(
        canonical_json_line_bytes_strict(
            activity_log.activity_record("secret", source="test")
        )
    )
    (root / "runtime").symlink_to(outside, target_is_directory=True)

    assert activity_log.read_activity_delta(
        root / "runtime" / "activity.jsonl", offset=0
    ) == ([], 0)


def test_activity_append_is_multiprocess_serialized(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_append_worker, args=(str(root), worker, 20))
        for worker in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    rows = activity_log.read_activity(
        root / "runtime" / "activity.jsonl", limit=500
    )
    assert len(rows) == 80
    assert len({row["event_id"] for row in rows}) == 80


def test_fresh_bootstrap_is_multiprocess_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_init_worker, args=(str(root),)) for _index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    assert discover_okf_startup(root, root / "runtime").allowed
    assert json.loads((root / "runtime" / "bootstrap-layout.json").read_bytes())[
        "state"
    ] == "ready"


def test_activity_tail_reader_does_not_call_full_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fresh_root(tmp_path)
    target = root / "runtime" / "activity.jsonl"
    for index in range(12):
        activity_log.append_activity(f"event {index}", source="test", root=root)

    monkeypatch.setattr(
        activity_log,
        "iter_activity",
        lambda _path: (_ for _ in ()).throw(AssertionError("full scan")),
    )

    assert [row["message"] for row in activity_log.read_activity(target, limit=3)] == [
        "event 9",
        "event 10",
        "event 11",
    ]


def test_activity_append_rejects_torn_existing_row(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    target = root / "runtime" / "activity.jsonl"
    target.write_bytes(b'{"torn":true}')

    with pytest.raises(ValueError, match="torn final row"):
        activity_log.append_activity("must not mask corruption", source="test", root=root)

    assert target.read_bytes() == b'{"torn":true}'


def test_activity_hot_gate_does_not_parse_unrelated_corpus(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    for index in range(40):
        (root / "pages" / f"invalid-{index}.md").write_text("not canonical")

    activity_log.append_activity("bounded gate", source="test", root=root)

    assert activity_log.read_activity(root / "runtime" / "activity.jsonl")[-1][
        "message"
    ] == "bounded gate"


@pytest.mark.parametrize("published", range(5))
def test_fresh_bootstrap_partial_publish_is_init_only_resumable(
    tmp_path: Path,
    published: int,
) -> None:
    from chronovisor.core.durable_state import okf_writer_lock
    from chronovisor.core.live_layout import write_live_layout_proof
    from chronovisor.core.reserved_documents import render_pages_index, render_pages_log

    root = tmp_path / "wiki"
    with okf_writer_lock(root):
        pass
    for directory in (root / "raw", root / "pages", root / "system"):
        directory.mkdir()
    write_live_layout_proof(root, state="in-progress")
    expected = (
        (root / "pages" / "index.md", render_pages_index(())),
        (root / "pages" / "log.md", render_pages_log()),
        (root / "system" / "schema.md", store.SCHEMA_CONTENT.encode()),
        (root / "runtime" / "activity.jsonl", b""),
    )
    for path, raw in expected[:published]:
        path.write_bytes(raw)

    decision = discover_okf_startup(root, root / "runtime")
    assert not decision.allowed
    assert decision.category == "bootstrap_in_progress"
    with pytest.raises(OKFStartupBlocked):
        activity_log.append_activity("too early", source="test", root=root)

    store.init_chronovisor(store.RuntimeContext(root))
    assert discover_okf_startup(root, root / "runtime").allowed
    assert json.loads((root / "runtime" / "bootstrap-layout.json").read_bytes())[
        "state"
    ] == "ready"


def test_lock_created_before_bootstrap_proof_is_init_only_resumable(
    tmp_path: Path,
) -> None:
    from chronovisor.core.durable_state import okf_writer_lock
    from chronovisor.core.live_layout import bootstrap_layout_lock

    root = tmp_path / "wiki"
    with okf_writer_lock(root):
        pass
    with bootstrap_layout_lock(root):
        pass

    decision = discover_okf_startup(root, root / "runtime")
    assert not decision.allowed
    assert decision.category == "bootstrap_in_progress"
    store.init_chronovisor(store.RuntimeContext(root))
    assert discover_okf_startup(root, root / "runtime").allowed


def test_writer_lock_only_bootstrap_is_init_only_resumable(tmp_path: Path) -> None:
    from chronovisor.core.durable_state import okf_writer_lock

    root = tmp_path / "wiki"
    with okf_writer_lock(root):
        pass

    assert discover_okf_startup(root, root / "runtime").allowed
    with pytest.raises(OKFStartupBlocked) as exc_info:
        with store.okf_runtime_operation(root):
            pytest.fail("ordinary operation entered an unsealed bootstrap")
    assert exc_info.value.decision.category == "bootstrap_in_progress"
    assert set((root / "runtime").iterdir()) == {root / "runtime" / "okf-writer.lock"}

    store.init_chronovisor(store.RuntimeContext(root))
    assert discover_okf_startup(root, root / "runtime").allowed


def test_absent_bootstrap_rejects_ordinary_operation_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wiki"

    with pytest.raises(OKFStartupBlocked) as exc_info:
        with store.okf_runtime_operation(root):
            pytest.fail("ordinary operation entered an unsealed bootstrap")
    assert exc_info.value.decision.category == "bootstrap_in_progress"
    assert not root.exists()

    store.init_chronovisor(store.RuntimeContext(root))
    assert discover_okf_startup(root, root / "runtime").allowed


@pytest.mark.parametrize("phase", ["index", "log", "schema", "activity", "ready"])
def test_fresh_bootstrap_parent_swap_never_writes_outside_or_seals_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    from chronovisor.core import live_layout

    root = tmp_path / "wiki"
    outside = tmp_path / f"outside-{phase}"
    outside.mkdir()
    swapped: dict[str, Path] = {}

    def swap(parent: str) -> None:
        if swapped:
            return
        pinned = root / f"{parent}-pinned"
        (root / parent).rename(pinned)
        (root / parent).symlink_to(outside, target_is_directory=True)
        swapped[parent] = pinned

    original_file_write = store.atomic_write_bytes_at

    def file_write(directory_fd: int, name: str, raw: bytes) -> None:
        targets = {
            "index": ("index.md", "pages"),
            "log": ("log.md", "pages"),
            "schema": ("schema.md", "system"),
            "activity": ("activity.jsonl", "runtime"),
        }
        target = targets.get(phase)
        if target is not None and name == target[0]:
            swap(target[1])
        original_file_write(directory_fd, name, raw)

    original_proof_write = live_layout.atomic_write_bytes_at

    def proof_write(directory_fd: int, name: str, raw: bytes) -> None:
        if phase == "ready" and b'"state":"ready"' in raw:
            swap("pages")
        original_proof_write(directory_fd, name, raw)

    monkeypatch.setattr(store, "atomic_write_bytes_at", file_write)
    monkeypatch.setattr(live_layout, "atomic_write_bytes_at", proof_write)

    with pytest.raises(ValueError, match="bootstrap directory changed"):
        store.init_chronovisor(store.RuntimeContext(root))

    assert list(outside.iterdir()) == []
    pinned_runtime = swapped.get("runtime", root / "runtime")
    proof = json.loads((pinned_runtime / "bootstrap-layout.json").read_bytes())
    assert proof["state"] == "in-progress"
