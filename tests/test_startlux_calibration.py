from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.mark.parametrize(
    "count,tamper,expected",
    [(0, False, 1), (4, False, 1), (6, False, 1), (5, True, 1), (5, False, 0)],
)
def test_certificate_requires_five_hash_verified_rows(
    tmp_path, monkeypatch, count, tamper, expected
):
    module = script("smoke_startlux_certificate_5")
    source = tmp_path / "input"
    source.mkdir()
    monkeypatch.setattr(module, "CHRONOVISOR_ROOT", source)
    gold = write_rows(
        source / "gold.jsonl",
        [
            {
                "case_id": str(i),
                "query_sha256": sha("q"),
                "page_id_sha256": sha("page"),
                "gold": True,
            }
            for i in range(count)
        ],
    )
    cases = write_rows(
        source / "cases.jsonl",
        [
            {
                "case_id": str(i),
                "query": "wrong" if tamper else "q",
                "page_id": "page",
                "evidence_span": "e",
            }
            for i in range(count)
        ],
    )
    monkeypatch.setattr(
        module,
        "load_active_rubric",
        lambda *, recover_backup: (
            {"rubric_text": "r", "rubric_id": "id"}
            if recover_backup is False
            else pytest.fail("repair enabled")
        ),
    )
    monkeypatch.setattr(module, "load_reranker_config", lambda: None)
    calls = []

    def judge(*_args, **kwargs):
        calls.append(kwargs)
        return {"verdicts": [{"decision": "pass", "confidence": 0.9}]}

    monkeypatch.setattr(module.reranker_client, "decision_judge", judge)
    out = tmp_path / "out"
    out.mkdir()
    (out / "certificate-smoke.json").symlink_to(gold)
    before = {p: p.read_bytes() for p in source.iterdir()}
    assert (
        module.main(
            [
                "--gold",
                str(gold),
                "--cases",
                str(cases),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == expected
    )
    assert len(calls) == (5 if expected == 0 else 0)
    assert all(call["priority"] == 1 for call in calls)
    assert before == {p: p.read_bytes() for p in source.iterdir()}


def test_calibration_refuses_output_inside_production_root(tmp_path, monkeypatch):
    module = script("smoke_startlux_certificate_5")
    monkeypatch.setattr(module, "CHRONOVISOR_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="outside the production root"):
        module.main(["--output-dir", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()


def test_answer_calibration_never_falls_back_to_legacy_generation(
    tmp_path, monkeypatch
):
    module = script("calibrate_startlux_answer_40")

    def legacy_route(_role):
        raise ValueError("not a decision route")

    monkeypatch.setattr(module, "decision_route", legacy_route)
    monkeypatch.setattr(
        module, "_controls", lambda *_args: pytest.fail("calibration started")
    )
    with pytest.raises(ValueError, match="not a decision route"):
        module.main(["--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_certificate_smoke_uses_body_before_truncating(tmp_path, monkeypatch):
    module = script("smoke_startlux_certificate_5")
    gold = write_rows(
        tmp_path / "gold.jsonl",
        [
            {
                "case_id": str(i),
                "query_sha256": sha("q"),
                "page_id_sha256": sha("p"),
                "gold": True,
            }
            for i in range(5)
        ],
    )
    cases = write_rows(
        tmp_path / "cases.jsonl",
        [
            {
                "case_id": str(i),
                "query": "q",
                "page_id": "p",
                "evidence_span": "---\ntitle: "
                + "metadata " * 100
                + "\n---\nRelevant body.",
            }
            for i in range(5)
        ],
    )
    monkeypatch.setattr(
        module, "load_active_rubric", lambda **_k: {"rubric_text": "rubric"}
    )
    monkeypatch.setattr(module, "load_reranker_config", lambda: None)
    seen = []

    def judge(_kind, fields, **_k):
        seen.append(fields["evidence"][0]["span"])
        return {"verdicts": [{"decision": "pass", "confidence": 0.9}]}

    monkeypatch.setattr(module.reranker_client, "decision_judge", judge)
    assert (
        module.main(
            [
                "--gold",
                str(gold),
                "--cases",
                str(cases),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert seen == ["Relevant body."] * 5
