from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_questions_candidate_test_module",
    ROOT / "scripts" / "recall_questions_candidate.py",
)
assert SPEC is not None and SPEC.loader is not None
CANDIDATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CANDIDATE
SPEC.loader.exec_module(CANDIDATE)


def _write_page(path: Path, *, questions: str = "['既存の質問?']") -> None:
    path.write_text(
        "---\n"
        "title: MLX Sushi\n"
        "updated: 2026-10-09\n"
        "status: stable\n"
        "type: knowledge\n"
        f"recall_questions: {questions}\n"
        "---\n"
        "MLX Sushi 2.6bpw の設定。\n",
        encoding="utf-8",
    )


def _local_binding() -> dict[str, object]:
    binding = {"model": "test-local", "location": "local"}
    return {
        "binding": binding,
        "sha256": CANDIDATE._sha256(CANDIDATE._canonical_json(binding)),
    }


def test_candidate_bundle_is_frozen_private_and_never_patches_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    page = source / "sushi.md"
    _write_page(page, questions="['既存の質問?', 'もう一つ?']")
    (source / "draft.md").write_text(
        "---\ntitle: Draft\nstatus: draft\ntype: knowledge\n---\nDraft\n",
        encoding="utf-8",
    )
    (source / "invalid.md").write_text("not canonical\n", encoding="utf-8")
    before = page.read_bytes()
    output = tmp_path / "candidate"
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(CANDIDATE, "_resolve_local_binding", _local_binding)

    def generate(*_args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "summary": "候補要約",
            "recall_questions": ["MLX Sushi の設定をどうした?"] ,
        }

    monkeypatch.setattr(CANDIDATE.ingest, "_generate_recall_metadata", generate)

    manifest = CANDIDATE.generate_candidates(source, output, 1)

    assert manifest["status"] == "complete"
    assert manifest["generated"] == 1
    assert manifest["fallback"] == 0
    assert manifest["source_enumerated"] == 3
    assert manifest["source_excluded"] == 2
    assert manifest["promotion_status"] == "blocked_until_p1"
    assert calls == [{"japanese_questions": True, "require_local_route": True}]
    assert page.read_bytes() == before
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "candidates.jsonl").stat().st_mode) == 0o600
    assert (output / ".gitignore").read_text(encoding="utf-8") == "*\n!.gitignore\n"
    snapshot = [json.loads(line) for line in (output / "source.jsonl").read_text(encoding="utf-8").splitlines()]
    candidates = [json.loads(line) for line in (output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert snapshot[0]["current_question_count"] == 2
    assert snapshot[0]["current_question_count_status"] == "known"
    assert candidates[0]["generation_status"] == "generated"
    assert candidates[0]["generation_success"] is True
    assert candidates[0]["comparison_arm"] == "B_pending_unsupported_count"
    assert candidates[0]["b_status"] == "pending_unsupported_count"


def test_fallback_is_recorded_but_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_page(source / "fallback.md")
    output = tmp_path / "candidate"
    monkeypatch.setattr(CANDIDATE, "_resolve_local_binding", _local_binding)
    monkeypatch.setattr(
        CANDIDATE.ingest,
        "_generate_recall_metadata",
        lambda title, body, page_id, **_kwargs: CANDIDATE.ingest._fallback_recall_metadata(
            title, body, page_id
        ),
    )

    manifest = CANDIDATE.generate_candidates(source, output, 1)

    assert manifest["fallback"] == 1
    row = json.loads((output / "candidates.jsonl").read_text(encoding="utf-8"))
    assert row["generation_status"] == "fallback"
    assert row["generation_success"] is False


def test_b_status_records_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_page(source / "mismatch.md", questions="['一?', '二?', '三?']")
    output = tmp_path / "candidate"
    monkeypatch.setattr(CANDIDATE, "_resolve_local_binding", _local_binding)
    calls: list[dict[str, object]] = []

    def generate(*_args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"summary": "候補", "recall_questions": ["一?", "二?"]}

    monkeypatch.setattr(CANDIDATE.ingest, "_generate_recall_metadata", generate)

    CANDIDATE.generate_candidates(source, output, 1)

    row = json.loads((output / "candidates.jsonl").read_text(encoding="utf-8"))
    assert calls == [{"japanese_questions": True, "require_local_route": True, "question_count": 3}]
    assert row["comparison_arm"] == "B_same_question_count"
    assert row["requested_question_count"] == 3
    assert row["b_status"] == "mismatch"


def test_missing_questions_are_c_arm_and_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_page(source / "missing.md", questions="null")
    output = tmp_path / "candidate"
    monkeypatch.setattr(CANDIDATE, "_resolve_local_binding", _local_binding)
    monkeypatch.setattr(
        CANDIDATE.ingest,
        "_generate_recall_metadata",
        lambda *_args, **_kwargs: {
            "summary": "候補",
            "recall_questions": ["質問1?", "質問2?", "質問3?"],
        },
    )

    CANDIDATE.generate_candidates(source, output, 1)

    row = json.loads((output / "candidates.jsonl").read_text(encoding="utf-8"))
    assert row["comparison_arm"] == "C_missing_completion"
    assert row["b_status"] == "pending_missing_completion"


def test_route_change_fails_closed_and_preserves_resume_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_page(source / "route-change.md")
    output = tmp_path / "candidate"
    bindings = iter((_local_binding(), {"binding": {"model": "changed", "location": "remote"}, "sha256": "b" * 64}))
    monkeypatch.setattr(CANDIDATE, "_resolve_local_binding", lambda: next(bindings))

    with pytest.raises(RuntimeError, match="route/config changed"):
        CANDIDATE.generate_candidates(source, output, 1)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "running"


def test_candidate_requires_positive_limit(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        CANDIDATE.main(
            ["--source", str(tmp_path), "--output", str(tmp_path / "out"), "--limit", "0"]
        )


def test_candidate_refuses_remote_route_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_page(source / "remote.md")
    output = tmp_path / "candidate"

    def remote() -> dict[str, object]:
        raise RuntimeError("candidate generation refuses a non-local ingest route")

    monkeypatch.setattr(CANDIDATE, "_resolve_local_binding", remote)

    with pytest.raises(RuntimeError, match="non-local"):
        CANDIDATE.generate_candidates(source, output, 1)
    assert not output.exists()


def test_candidate_refuses_source_output_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_page(source / "overlap.md")
    monkeypatch.setattr(CANDIDATE, "_resolve_local_binding", _local_binding)

    with pytest.raises(ValueError, match="must not overlap"):
        CANDIDATE.generate_candidates(source, source / "candidate", 1)


def _write_pages(source: Path, count: int) -> None:
    for index in range(1, count + 1):
        _write_page(source / f"page-{index}.md")


def _install_generator(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    monkeypatch.setattr(CANDIDATE, "_resolve_local_binding", _local_binding)

    def generate(title: str, body: str, page_id: str, **_kwargs: object) -> dict[str, object]:
        del title, body
        calls.append(page_id)
        return {
            "summary": f"candidate {page_id}",
            "recall_questions": [f"{page_id} の質問1?"],
        }

    monkeypatch.setattr(CANDIDATE.ingest, "_generate_recall_metadata", generate)


def test_resume_processes_only_unfinished_frozen_pages_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_pages(source, 3)
    output = tmp_path / "candidate"
    calls: list[str] = []
    _install_generator(monkeypatch, calls)

    first = CANDIDATE.generate_candidates(source, output, 3, max_pages=1)
    snapshot_before = (output / "source.jsonl").read_bytes()
    (source / "page-2.md").write_text("changed after freeze\n", encoding="utf-8")
    second = CANDIDATE.generate_candidates(
        source, output, 3, max_pages=1, resume=True
    )

    assert first["status"] == "partial"
    assert second["processed_count"] == 2
    assert calls == ["page-1", "page-2"]
    assert (output / "source.jsonl").read_bytes() == snapshot_before
    rows = [
        json.loads(line)
        for line in (output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["page_id"] for row in rows] == ["page-1", "page-2"]
    final = CANDIDATE.generate_candidates(
        source, output, 3, max_pages=2, resume=True
    )
    assert final["status"] == "complete"
    assert calls == ["page-1", "page-2", "page-3"]


def test_resume_finalizes_a_durable_row_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_pages(source, 2)
    output = tmp_path / "candidate"
    calls: list[str] = []
    _install_generator(monkeypatch, calls)
    original_write = CANDIDATE._write_json

    def interrupt_after_append(path: Path, value: object) -> None:
        if (
            path.name == "manifest.json"
            and isinstance(value, dict)
            and value.get("status") == "partial"
        ):
            raise KeyboardInterrupt
        original_write(path, value)

    monkeypatch.setattr(CANDIDATE, "_write_json", interrupt_after_append)
    with pytest.raises(KeyboardInterrupt):
        CANDIDATE.generate_candidates(source, output, 2, max_pages=1)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "pending_commit"
    assert len((output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    monkeypatch.setattr(CANDIDATE, "_write_json", original_write)
    resumed = CANDIDATE.generate_candidates(
        source, output, 2, max_pages=1, resume=True
    )
    assert resumed["status"] == "complete"
    assert calls == ["page-1", "page-2"]
    rows = [
        json.loads(line)
        for line in (output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["page_id"] for row in rows] == ["page-1", "page-2"]


def test_resume_refuses_a_concurrent_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_pages(source, 2)
    output = tmp_path / "candidate"
    calls: list[str] = []
    _install_generator(monkeypatch, calls)
    CANDIDATE.generate_candidates(source, output, 2, max_pages=1)

    descriptor = os.open(output / ".lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="locked by another writer"):
            CANDIDATE.generate_candidates(
                source, output, 2, max_pages=1, resume=True
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert calls == ["page-1"]


def test_manifest_replacement_is_atomic_across_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_pages(source, 2)
    output = tmp_path / "candidate"
    calls: list[str] = []
    _install_generator(monkeypatch, calls)
    CANDIDATE.generate_candidates(source, output, 2, max_pages=1)
    manifest_path = output / "manifest.json"
    before = manifest_path.read_bytes()
    original_replace = os.replace

    def interrupt_before_replace(_source: object, _destination: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(CANDIDATE.os, "replace", interrupt_before_replace)
    with pytest.raises(KeyboardInterrupt):
        CANDIDATE._write_json(manifest_path, {"marker": "before"})
    assert manifest_path.read_bytes() == before
    assert not list(output.glob(".manifest.json.*.tmp"))

    def interrupt_after_replace(source_path: object, destination: object) -> None:
        original_replace(source_path, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(CANDIDATE.os, "replace", interrupt_after_replace)
    with pytest.raises(KeyboardInterrupt):
        CANDIDATE._write_json(manifest_path, {"marker": "after"})
    assert manifest_path.read_bytes() == b'{"marker":"after"}\n'
    assert not list(output.glob(".manifest.json.*.tmp"))


def test_resume_rejects_snapshot_or_candidate_modification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_pages(source, 2)
    output = tmp_path / "candidate"
    calls: list[str] = []
    _install_generator(monkeypatch, calls)
    CANDIDATE.generate_candidates(source, output, 2, max_pages=1)

    snapshot = output / "source.jsonl"
    snapshot.write_bytes(snapshot.read_bytes().replace(b"MLX Sushi", b"tampered"))
    with pytest.raises(ValueError, match="source snapshot hash"):
        CANDIDATE.generate_candidates(source, output, 2, max_pages=1, resume=True)

    # Restore the exact snapshot, then tamper with the durable candidate prefix.
    snapshot.write_bytes(snapshot.read_bytes().replace(b"tampered", b"MLX Sushi"))
    candidates = output / "candidates.jsonl"
    candidate_row = json.loads(candidates.read_text(encoding="utf-8"))
    candidate_row["page_id"] = "wrong"
    candidates.write_text(json.dumps(candidate_row) + "\n", encoding="utf-8")
    candidates.chmod(0o600)
    with pytest.raises(ValueError, match="candidate artifact hash"):
        CANDIDATE.generate_candidates(source, output, 2, max_pages=1, resume=True)
    assert calls == ["page-1"]


def test_resume_rejects_binding_change_and_partial_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_pages(source, 2)
    output = tmp_path / "candidate"
    calls: list[str] = []
    _install_generator(monkeypatch, calls)
    CANDIDATE.generate_candidates(source, output, 2, max_pages=1)

    monkeypatch.setattr(
        CANDIDATE,
        "_resolve_local_binding",
        lambda: {
            "binding": {"model": "changed", "location": "local"},
            "sha256": CANDIDATE._sha256(
                CANDIDATE._canonical_json(
                    {"model": "changed", "location": "local"}
                )
            ),
        },
    )
    with pytest.raises(RuntimeError, match="does not match frozen manifest"):
        CANDIDATE.generate_candidates(source, output, 2, max_pages=1, resume=True)

    # Restore the route and append an incomplete line. A partial tail is never
    # treated as a successful durable row.
    monkeypatch.setattr(CANDIDATE, "_resolve_local_binding", _local_binding)
    candidates = output / "candidates.jsonl"
    with candidates.open("ab") as handle:
        handle.write(b"{\"page_id\":\"partial\"")
    with pytest.raises(ValueError, match="incomplete tail"):
        CANDIDATE.generate_candidates(source, output, 2, max_pages=1, resume=True)
    assert calls == ["page-1"]


def test_max_pages_bounds_each_run_and_artifacts_start_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pages"
    source.mkdir()
    _write_pages(source, 3)
    output = tmp_path / "candidate"
    calls: list[str] = []
    _install_generator(monkeypatch, calls)

    first = CANDIDATE.generate_candidates(source, output, 3, max_pages=1)
    assert first["processed_count"] == 1
    assert calls == ["page-1"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for name in (".gitignore", "source.jsonl", "candidates.jsonl", "manifest.json"):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o600

    second = CANDIDATE.generate_candidates(
        source, output, 3, max_pages=2, resume=True
    )
    assert second["processed_count"] == 3
    assert second["status"] == "complete"
    assert calls == ["page-1", "page-2", "page-3"]
