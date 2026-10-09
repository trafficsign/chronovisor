from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from chronovisor.core.semantic_index import extract_page_documents

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_p2b_arm_bundle_test_module", ROOT / "scripts" / "recall_p2b_arm_bundle.py"
)
assert SPEC is not None and SPEC.loader is not None
BUNDLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUNDLE
SPEC.loader.exec_module(BUNDLE)


def _source_row(index: int, status: str, count: int | None, questions: list[str]) -> dict[str, object]:
    title = f"Page {index}"
    content = (
        "---\n"
        f"title: {title}\n"
        "status: stable\n"
        "type: knowledge\n"
        f"recall_questions: {json.dumps(questions, ensure_ascii=False)}\n"
        "---\n"
        f"Body for page {index}.\n"
    )
    return {
        "body": f"Body for page {index}.\n",
        "content": content,
        "current_question_count": count,
        "current_question_count_status": status,
        "page_id": f"page-{index}",
        "relative_path": f"page-{index}.md",
        "source_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "title": title,
    }


def _fixtures(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]], list[dict[str, object]]]:
    source_rows = [
        _source_row(0, "known", 5, ["current five"] * 5),
        _source_row(1, "known", 2, ["current two"] * 2),
        _source_row(2, "pending", None, []),
        _source_row(3, "known", 3, ["current three"] * 3),
    ]
    candidate_rows: list[dict[str, object]] = []
    for index, source in enumerate(source_rows):
        requested_count = source["current_question_count"] if source["current_question_count"] in {3, 4, 5} else 5
        questions = [f"luna {index} {n}" for n in range(requested_count)]
        candidate_rows.append(
            {
                "b_status": "test",
                "comparison_arm": "Luna_B_same_question_count",
                "current_question_count": source["current_question_count"],
                "current_question_count_status": source["current_question_count_status"],
                "generation_status": "generated",
                "generation_success": True,
                "generator": {"configured_model": "test"},
                "input_sha256": hashlib.sha256(f"input-{index}".encode()).hexdigest(),
                "model_changed_from_baseline": True,
                "page_id": source["page_id"],
                "recall_questions": questions,
                "relative_path": source["relative_path"],
                "requested_question_count": len(questions),
                "source_index": index,
                "source_sha256": source["source_sha256"],
                "summary": "test",
            }
        )
    source_path = tmp_path / "source.jsonl"
    candidate_path = tmp_path / "candidates.jsonl"
    source_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows), encoding="utf-8")
    candidate_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidate_rows), encoding="utf-8")
    return source_path, candidate_path, source_rows, candidate_rows


def test_arm_boundaries_and_d_e_projections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path, candidate_path, source_rows, candidate_rows = _fixtures(tmp_path)
    monkeypatch.setattr(BUNDLE, "EXPECTED_PAGE_COUNT", 4)
    source_rows, candidate_rows = BUNDLE._validate_inputs(source_path, candidate_path)

    a = list(BUNDLE.iter_arm_projection(source_rows, candidate_rows, "A"))
    b = list(BUNDLE.iter_arm_projection(source_rows, candidate_rows, "B"))
    c = list(BUNDLE.iter_arm_projection(source_rows, candidate_rows, "C"))
    d = list(BUNDLE.iter_arm_projection(source_rows, candidate_rows, "D"))
    e = list(BUNDLE.iter_arm_projection(source_rows, candidate_rows, "E"))

    assert [row["question_origin"] for row in b] == [
        "luna_count_match",
        "current_unsupported_preserved",
        "missing_empty",
        "luna_count_match",
    ]
    assert [row["question_origin"] for row in c] == [
        "luna_count_match",
        "luna_unsupported_replacement",
        "luna_missing_completion",
        "luna_count_match",
    ]
    assert a[0]["questions_sha256"] != b[0]["questions_sha256"]
    assert b[1]["questions_sha256"] == a[1]["questions_sha256"]
    assert b[2]["question_count"] == 0
    assert c[1]["question_count"] == 5
    assert d[0]["semantic"]["page_contains_question_text"] is False
    assert d[0]["semantic"]["question_documents"]
    assert "Q:" not in d[0]["semantic"]["page_text"]
    assert "Q:" in c[0]["semantic"]["page_text"]
    assert e[0]["bm25_tokens"] != c[0]["bm25_tokens"]
    assert len(e[0]["bm25_tokens"]) > len(c[0]["bm25_tokens"])

    without = BUNDLE.student_candidate_text(" Title ", " snippet ", ["Q"], include_questions=False)
    with_questions = BUNDLE.student_candidate_text(" Title ", " snippet ", ["Q"], include_questions=True)
    assert without == "Title\nsnippet"
    assert with_questions.startswith(without + "\nQ: Q")

    page_path = tmp_path / "page-0.md"
    page_path.write_text(source_rows[0]["content"], encoding="utf-8")
    extracted = extract_page_documents(page_path)
    page_document = next(document for document in extracted if document.kind == "page")
    question_documents = [document for document in extracted if document.kind == "question"]
    assert page_document.text == a[0]["semantic"]["page_text"]
    assert [document.text for document in question_documents] == [
        document["text"] for document in a[0]["semantic"]["question_documents"]
    ]
    chunk_documents = [document for document in extracted if document.kind == "chunk"]
    assert [document.text for document in chunk_documents] == [
        document["text"] for document in a[0]["semantic"]["chunk_documents"]
    ]


def test_prepare_verify_and_artifact_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path, candidate_path, _source_rows, _candidate_rows = _fixtures(tmp_path)
    monkeypatch.setattr(BUNDLE, "EXPECTED_PAGE_COUNT", 4)
    monkeypatch.setattr(BUNDLE, "EXPECTED_B_REPLACEMENTS", 2)
    monkeypatch.setattr(BUNDLE, "EXPECTED_UNSUPPORTED", 1)
    monkeypatch.setattr(BUNDLE, "EXPECTED_MISSING", 1)
    output = tmp_path / "bundle"
    manifest = BUNDLE.prepare_bundle(source_path, candidate_path, output)
    assert manifest["page_count"] == 4
    assert manifest["formal_comparison"]["status"] == "not_measured"
    assert BUNDLE.verify_bundle(output)["status"] == "verified"

    candidate_path.write_text(candidate_path.read_text(encoding="utf-8") + "", encoding="utf-8")
    BUNDLE.verify_bundle(output)
    candidate_path.write_text(candidate_path.read_text(encoding="utf-8").replace('"summary": "test"', '"summary": "changed"', 1), encoding="utf-8")
    with pytest.raises(BUNDLE.BundleError, match="artifact hash mismatch"):
        BUNDLE.verify_bundle(output)


def test_b_count_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path, candidate_path, _source_rows, _candidate_rows = _fixtures(tmp_path)
    monkeypatch.setattr(BUNDLE, "EXPECTED_PAGE_COUNT", 4)
    rows = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["requested_question_count"] = 4
    rows[0]["recall_questions"] = rows[0]["recall_questions"][:4]
    candidate_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(BUNDLE.BundleError, match="same-count candidate"):
        BUNDLE._validate_inputs(source_path, candidate_path)


def test_manifest_metadata_tampering_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path, candidate_path, _source_rows, _candidate_rows = _fixtures(tmp_path)
    monkeypatch.setattr(BUNDLE, "EXPECTED_PAGE_COUNT", 4)
    monkeypatch.setattr(BUNDLE, "EXPECTED_B_REPLACEMENTS", 2)
    monkeypatch.setattr(BUNDLE, "EXPECTED_UNSUPPORTED", 1)
    monkeypatch.setattr(BUNDLE, "EXPECTED_MISSING", 1)
    output = tmp_path / "bundle"
    BUNDLE.prepare_bundle(source_path, candidate_path, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["arm_summary"]["A"]["questions"] += 1
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(BUNDLE.BundleError, match="arm summary changed"):
        BUNDLE.verify_bundle(output)


def test_source_body_mutation_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path, candidate_path, _source_rows, _candidate_rows = _fixtures(tmp_path)
    monkeypatch.setattr(BUNDLE, "EXPECTED_PAGE_COUNT", 4)
    source_path.write_text(source_path.read_text(encoding="utf-8").replace("Body for page 0.", "mutated"), encoding="utf-8")
    with pytest.raises(BUNDLE.BundleError, match="source content hash mismatch"):
        BUNDLE._validate_inputs(source_path, candidate_path)
