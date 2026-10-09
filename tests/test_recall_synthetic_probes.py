from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_synthetic_probes_test_module",
    ROOT / "scripts" / "recall_synthetic_probes.py",
)
assert SPEC is not None and SPEC.loader is not None
PROBES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBES
SPEC.loader.exec_module(PROBES)


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fixture = tmp_path / "synthetic-fixture"
    fixture.mkdir()
    first = fixture / "alpha.md"
    second = fixture / "beta.md"
    first.write_text("Sushi の現在の設定は 2.6bpw です。\n", encoding="utf-8")
    second.write_text("Recall の保持期間は 30 日です。\n", encoding="utf-8")
    facts: list[dict[str, object]] = []
    for fact_id, path, subject, relation, value, family, validity in (
        (
            "fact-alpha",
            first,
            "Sushi",
            "設定",
            "2.6bpw",
            "runtime",
            {"valid_from": "2026-01-01T00:00:00Z", "valid_to": "2026-12-31T00:00:00Z"},
        ),
        (
            "fact-beta",
            second,
            "Recall",
            "保持期間",
            "30日",
            "retention",
            {"valid_from": "2026-02-01T00:00:00Z", "valid_to": None},
        ),
    ):
        raw = path.read_bytes()
        facts.append(
            {
                "fact_id": fact_id,
                "subject": subject,
                "relation": relation,
                "value": value,
                "validity": validity,
                "source": {
                    "relative_path": path.relative_to(fixture).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "byte_start": 0,
                    "byte_end": len(raw),
                },
                "family": family,
                "question_candidates": [f"{subject}の{relation}は何ですか？"],
            }
        )
    facts_path = tmp_path / "facts.jsonl"
    facts_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in facts),
        encoding="utf-8",
    )
    return facts_path, fixture, facts, tmp_path / "bundle"


def _comparison(
    question: dict[str, object],
    fact: dict[str, object],
    *,
    status: str = "verified",
    teacher_only: bool = False,
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "probe_id": question["question_id"],
        "expected_fact_ids": [fact["fact_id"]],
        "actual_fact_ids": [],
        "expected_fact_bindings": [
            {"fact_id": fact["fact_id"], "fact_sha256": fact["fact_sha256"]}
        ],
        "comparison_status": status,
        "source_truth_status": "verified",
        "verification_receipt_sha256": "a" * 64,
        "teacher_only": teacher_only,
    }
    return {**unsigned, "comparison_sha256": PROBES._sha256_object(unsigned)}


def test_prepare_freezes_source_facts_and_four_controls(tmp_path: Path) -> None:
    facts_path, fixture, _raw_facts, output = _fixture(tmp_path)
    before = (fixture / "alpha.md").read_bytes(), (fixture / "beta.md").read_bytes()

    result = PROBES.prepare_bundle(facts_path, fixture, output)

    assert result["status"] == "prepared"
    assert result["fact_count"] == 2
    assert result["question_count"] == 2
    assert result["control_count"] == 8
    assert result["repair_queue_count"] == 0
    assert result["diagnostic_miss_count"] == 0
    assert (fixture / "alpha.md").read_bytes() == before[0]
    assert (fixture / "beta.md").read_bytes() == before[1]
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "receipt.json").stat().st_mode) == 0o600

    controls = [
        json.loads(line)
        for line in (output / "controls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["control_kind"] for row in controls} == set(PROBES.CONTROL_KINDS)
    assert all(row["fixture_scope"] == "synthetic_fixture_only" for row in controls)
    assert all(
        row["promotion_status"] == "blocked_synthetic_fixture" for row in controls
    )
    assert sum(row["ready"] is True for row in controls) == 6
    topic_switches = [row for row in controls if row["control_kind"] == "topic_switch"]
    assert all(row["expected_fact_ids"] != [] for row in topic_switches)
    assert all(
        row["distractor_fact_ids"] == [row["target_fact_id"]] for row in topic_switches
    )
    assert all(
        any(
            alternate["relation"] in row["question"]
            for alternate in _raw_facts
            if alternate["fact_id"] in row["expected_fact_ids"]
        )
        for row in topic_switches
    )
    assert PROBES.verify_bundle(output)["status"] == "verified"


def test_source_only_facts_and_questions_are_unverified_by_default(
    tmp_path: Path,
) -> None:
    facts_path, fixture, _raw_facts, output = _fixture(tmp_path)
    PROBES.prepare_bundle(facts_path, fixture, output)
    facts = [
        json.loads(line)
        for line in (output / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    questions = [
        json.loads(line)
        for line in (output / "questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert {fact["semantic_status"] for fact in facts} == {"unverified"}
    assert {fact["model_agreement"] for fact in facts} == {"unmeasured"}
    assert {question["verification"]["status"] for question in questions} == {
        "unverified"
    }
    assert {question["verification"]["model_agreement"] for question in questions} == {
        "unmeasured"
    }
    assert (
        json.loads((output / "manifest.json").read_text(encoding="utf-8"))[
            "formal_acceptance"
        ]
        is False
    )


def test_verification_requires_exact_fact_and_question_hash_bindings(
    tmp_path: Path,
) -> None:
    facts_path, fixture, _raw_facts, output = _fixture(tmp_path)
    question_id = (
        "fact-alpha:q:0-" + PROBES._sha256_text("Sushiの設定は何ですか？")[:12]
    )
    verification_path = tmp_path / "verification.jsonl"
    verification_path.write_text(
        json.dumps(
            {
                "question_id": question_id,
                "fact_id": "fact-alpha",
                "fact_sha256": "b" * 64,
                "question_sha256": PROBES._sha256_text("Sushiの設定は何ですか？"),
                "status": "verified",
                "answerability": "answerable",
                "semantic_status": "verified",
                "model_agreement": "unmeasured",
                "verifier_id": "synthetic-independent-review",
                "evidence_sha256": "c" * 64,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PROBES.ProbeError, match="fact hash does not bind"):
        PROBES.prepare_bundle(
            facts_path,
            fixture,
            output,
            verifications_path=verification_path,
        )


def test_external_verified_claim_stays_unverified_and_ineligible(
    tmp_path: Path,
) -> None:
    facts_path, fixture, _raw_facts, output = _fixture(tmp_path)
    question_sha = PROBES._sha256_text("Sushiの設定は何ですか？")
    verification_path = tmp_path / "verification-claim.jsonl"
    verification_path.write_text(
        json.dumps(
            {
                "question_id": f"fact-alpha:q:0-{question_sha[:12]}",
                "fact_id": "fact-alpha",
                "fact_sha256": PROBES._sha256_object(
                    {
                        "subject": "Sushi",
                        "relation": "設定",
                        "value": "2.6bpw",
                        "validity": {
                            "valid_from": "2026-01-01T00:00:00Z",
                            "valid_to": "2026-12-31T00:00:00Z",
                        },
                        "source": {
                            "relative_path": "alpha.md",
                            "sha256": hashlib.sha256(
                                (fixture / "alpha.md").read_bytes()
                            ).hexdigest(),
                            "byte_start": 0,
                            "byte_end": len((fixture / "alpha.md").read_bytes()),
                            "excerpt_sha256": hashlib.sha256(
                                (fixture / "alpha.md").read_bytes()
                            ).hexdigest(),
                        },
                        "family": "runtime",
                    }
                ),
                "question_sha256": question_sha,
                "status": "verified",
                "answerability": "answerable",
                "semantic_status": "verified",
                "model_agreement": "verified",
                "verifier_id": "claimed-reviewer",
                "evidence_sha256": "d" * 64,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    PROBES.prepare_bundle(
        facts_path, fixture, output, verifications_path=verification_path
    )
    question = json.loads(
        (output / "questions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    verification = question["verification"]
    assert verification["status"] == "unverified"
    assert verification["claimed_status"] == "verified"
    assert verification["verification_authenticated"] is False
    assert verification["promotion_eligible"] is False


def test_verified_actual_miss_is_the_only_repair_queue_input(tmp_path: Path) -> None:
    facts_path, fixture, _raw_facts, output = _fixture(tmp_path)
    # Build the question/fact hashes through the same preparation contract.
    probe_output = tmp_path / "probe-for-comparison"
    PROBES.prepare_bundle(facts_path, fixture, probe_output)
    question = json.loads(
        (probe_output / "questions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    fact = json.loads(
        (probe_output / "facts.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    verified = _comparison(question, fact)
    unverified = _comparison(question, fact, status="unverified")
    teacher_only = _comparison(question, fact, teacher_only=True)
    comparisons_path = tmp_path / "comparisons.jsonl"
    comparisons_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in (verified, unverified, teacher_only)
        ),
        encoding="utf-8",
    )

    output = tmp_path / "bundle-with-repairs"
    PROBES.prepare_bundle(
        facts_path, fixture, output, comparisons_path=comparisons_path
    )
    queue = [
        json.loads(line)
        for line in (output / "repair_queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    held = [
        json.loads(line)
        for line in (output / "held_comparisons.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert queue == []
    assert len(held) == 2
    assert {row["reason"] for row in held} == {
        "comparison_not_verified",
        "teacher_only_artifact_is_separate",
    }
    diagnostics = [
        json.loads(line)
        for line in (output / "diagnostic_misses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(diagnostics) == 3
    assert all(
        item["diagnostic_schema"] == PROBES.DIAGNOSTIC_SCHEMA for item in diagnostics
    )
    assert all(item["promotion_eligible"] is False for item in diagnostics)
    assert PROBES.verify_bundle(output)["repair_queue_count"] == 0


def test_source_hash_tamper_and_fixture_output_overlap_fail_closed(
    tmp_path: Path,
) -> None:
    facts_path, fixture, _raw_facts, output = _fixture(tmp_path)
    (fixture / "alpha.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(PROBES.ProbeError, match="does not match fixture bytes"):
        PROBES.prepare_bundle(facts_path, fixture, output)

    facts_path, fixture, _raw_facts, _output = _fixture(tmp_path / "second")
    with pytest.raises(PROBES.ProbeError, match="outside the synthetic fixture corpus"):
        PROBES.prepare_bundle(facts_path, fixture, fixture / "nested-output")


def test_cli_prepare_and_verify_use_private_receipt_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    facts_path, fixture, _raw_facts, output = _fixture(tmp_path)
    assert (
        PROBES.main(
            [
                "--facts",
                str(facts_path),
                "--fixture-root",
                str(fixture),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert PROBES.main(["--verify", "--output", str(output)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert (
        json.loads(lines[0])["promotion_status"] == "blocked_until_verified_actual_data"
    )
    assert json.loads(lines[1])["status"] == "verified"
