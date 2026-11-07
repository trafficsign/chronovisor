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


def test_utility_pair_rule_and_acceptance_gates():
    module = script("calibrate_startlux_utility_58")
    assert module._verdict(-8, -8, True, 0.75) == ("neutral", True)
    assert module._verdict(8, 8, True, 0.75) == ("uncertain", False)
    assert module._verdict(0, 0, True, 0.75) == ("uncertain", False)
    rows = [
        {
            "fold": i % 5,
            "first_logit": -8 if i < 29 else 8,
            "second_logit": 8 if i < 29 else -8,
            "a0_first": True,
            "gold": "helpful" if i < 29 else "harmful",
        }
        for i in range(58)
    ]
    good = module._fit(rows, 5)
    assert good["passed"] and good["directional_lcb"] >= 0.65
    assert good["distribution"] == {
        "helpful": 29,
        "harmful": 29,
        "neutral": 0,
        "uncertain": 0,
    }
    bad = module._fit([{**row, "first_logit": 8, "second_logit": 8} for row in rows], 5)
    assert not bad["passed"]
    assert not bad["gates"]["accuracy"] and not bad["gates"]["order_agreement"]
    assert not bad["gates"]["directional_lcb"] and not bad["gates"]["uncertain_rate"]
    assert not module._fit(rows[:20], 5)["gates"]["harmful_count"]


@pytest.mark.parametrize("bad_logits,expected", [(False, 0), (True, 1)])
def test_utility_exit_checks_gates_and_only_reads_archived_answers(
    tmp_path, monkeypatch, bad_logits, expected
):
    module = script("calibrate_startlux_utility_58")
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(module, "CHRONOVISOR_ROOT", source)
    rows = [
        {
            "dimension": "answer_utility",
            "rally_id": str(i),
            "candidate_id": "c",
            "a0_sha256": sha("a0"),
            "a1_sha256": sha("a1"),
            "verdict": "helpful" if i < 29 else "harmful",
        }
        for i in range(58)
    ]
    labels = write_rows(source / "labels.jsonl", rows)
    answers = write_rows(
        source / "answers.jsonl",
        [
            {
                "rally_id": str(i),
                "candidate_id": "c",
                "a0": "a0",
                "a1": "a1",
                "query": str(i),
                "context": [],
            }
            for i in range(58)
        ],
    )
    monkeypatch.setattr(
        module,
        "_rallies",
        lambda *_args: {
            str(i): {
                "rally_id": str(i),
                "query_sha256": sha(str(i)),
                "context_refs": [],
            }
            for i in range(58)
        },
    )
    monkeypatch.setattr(module, "load_reranker_config", lambda: None)
    calls = []

    def judge(_kind, payload, **kwargs):
        calls.append(kwargs)
        preferred = "a1" if int(payload["query"]) < 29 else "a0"
        return {"logit": 8 if bad_logits or payload["answer_a"] == preferred else -8}

    monkeypatch.setattr(module.reranker_client, "decision_judge", judge)
    from chronovisor.recall import recall_distillation

    monkeypatch.setattr(
        recall_distillation,
        "_worker_call",
        lambda *_a, **_k: pytest.fail("model generation"),
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "utility-cv.json").symlink_to(labels)
    before = {p: p.read_bytes() for p in source.iterdir()}
    assert (
        module.main(
            [
                "--labels",
                str(labels),
                "--answers",
                str(answers),
                "--rallies",
                str(source / "rallies"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == expected
    )
    assert len(calls) == 116 and all(call["priority"] == 1 for call in calls)
    assert before == {p: p.read_bytes() for p in source.iterdir()}


@pytest.mark.parametrize(
    "name", ["smoke_startlux_certificate_5", "calibrate_startlux_utility_58"]
)
def test_calibration_refuses_output_inside_production_root(tmp_path, monkeypatch, name):
    module = script(name)
    monkeypatch.setattr(module, "CHRONOVISOR_ROOT", tmp_path)
    args = ["--output-dir", str(tmp_path / "output")]
    if name.startswith("calibrate"):
        args += ["--answers", str(tmp_path / "answers")]
    with pytest.raises(SystemExit, match="outside the production root"):
        module.main(args)
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


@pytest.mark.parametrize("field", ["query", "context", "a0", "a1"])
def test_archived_utility_input_is_hash_verified(field):
    module = script("calibrate_startlux_utility_58")
    row = {"a0_sha256": sha("without"), "a1_sha256": sha("with")}
    rally = {
        "query_sha256": sha("question"),
        "context_refs": [{"semantic_sha256": sha("preceding")}],
    }
    archived = {
        "query": "question",
        "context": ["preceding"],
        "a0": "without",
        "a1": "with",
    }
    assert module._verified_input(row, rally, archived)
    archived[field] = ["tampered"] if field == "context" else "tampered"
    assert not module._verified_input(row, rally, archived)
