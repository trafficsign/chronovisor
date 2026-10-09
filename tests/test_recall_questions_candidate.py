from __future__ import annotations

import importlib.util
import json
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
    return {"binding": {"model": "test-local", "location": "local"}, "sha256": "a" * 64}


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


def test_route_change_fails_closed_and_records_failed_manifest(
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
    assert manifest["status"] == "failed"


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
