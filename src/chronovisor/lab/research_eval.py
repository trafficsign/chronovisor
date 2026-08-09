"""Locked deterministic holdout for agentic retrieval and evidence adoption."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from chronovisor.research.evidence_bundle import simple_assess_claims
from chronovisor.search.research_types import EvidenceArtifact

_PACKAGED_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "research_holdout.jsonl"
)
_SOURCE_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "research_holdout.jsonl"
)
DEFAULT_FIXTURE = _PACKAGED_FIXTURE if _PACKAGED_FIXTURE.is_file() else _SOURCE_FIXTURE


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {number}") from exc
        if not isinstance(row, dict) or row.get("split") != "locked-test":
            raise ValueError(f"line {number} is not a locked-test case")
        rows.append(row)
    if not rows:
        raise ValueError("research holdout is empty")
    if len({str(row.get("case_id") or "") for row in rows}) != len(rows):
        raise ValueError("research holdout case IDs must be unique and non-empty")
    return rows


def run_eval(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    raw = path.read_bytes()
    rows = _rows(path)
    baseline_hits = 0
    agentic_hits = 0
    correct_claims = 0
    unknown_expected = 0
    unknown_preserved = 0
    actions = 0
    wasted = 0
    cases: list[dict[str, Any]] = []
    for row in rows:
        expected_page = str(row.get("expected_page") or "")
        baseline = [str(item) for item in row.get("baseline_direct", [])]
        collected = [str(item) for item in row.get("agentic_collected", [])]
        baseline_hit = expected_page in baseline
        agentic_hit = expected_page in collected
        baseline_hits += int(baseline_hit)
        agentic_hits += int(agentic_hit)

        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        artifacts = []
        if evidence:
            preview = str(evidence.get("preview") or "")
            digest = hashlib.sha256(preview.encode("utf-8")).hexdigest()
            artifacts.append(
                EvidenceArtifact(
                    artifact_id=f"sha256:{digest}",
                    source_type=str(evidence.get("source_type") or "chronovisor_read"),
                    source_uri=str(evidence.get("source_uri") or "wiki:fixture"),
                    retrieved_at="2026-07-18T00:00:00+00:00",
                    sha256=digest,
                    byte_length=len(preview.encode("utf-8")),
                    preview=preview,
                    trust=str(evidence.get("trust") or "local"),
                    title=str(evidence.get("title") or "Fixture"),
                    citation=str(evidence.get("citation") or "wiki:fixture"),
                    durable=True,
                )
            )
        assessment = simple_assess_claims(
            [(str(row.get("claim") or ""), row.get("user_reported") is True)],
            artifacts,
        )[0]
        expected_status = str(row.get("expected_status") or "unknown")
        correct_claims += int(assessment.status.value == expected_status)
        if expected_status == "unknown":
            unknown_expected += 1
            unknown_preserved += int(assessment.status.value == "unknown")
        case_actions = [str(item) for item in row.get("actions", [])]
        necessary = {str(item) for item in row.get("necessary_actions", [])}
        actions += len(case_actions)
        wasted += sum(item not in necessary for item in case_actions)
        cases.append(
            {
                "case_id": row["case_id"],
                "baseline_hit": baseline_hit,
                "agentic_hit": agentic_hit,
                "expected_status": expected_status,
                "actual_status": assessment.status.value,
            }
        )
    count = len(rows)
    baseline_rescue = baseline_hits / count
    agentic_rescue = agentic_hits / count
    evidence_precision = correct_claims / count
    unknown_retention = (
        unknown_preserved / unknown_expected if unknown_expected else 1.0
    )
    waste_rate = wasted / actions if actions else 0.0
    pass_gate = (
        agentic_rescue >= baseline_rescue + 0.03
        and evidence_precision >= 0.95
        and unknown_retention == 1.0
        and waste_rate <= 0.02
    )
    return {
        "schema_version": 1,
        "status": "pass" if pass_gate else "fail",
        "fixture": path.name,
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "cases": count,
        "baseline_rescue_rate": baseline_rescue,
        "agentic_rescue_rate": agentic_rescue,
        "absolute_rescue_improvement": agentic_rescue - baseline_rescue,
        "source_backed_claim_precision": evidence_precision,
        "unknown_retention": unknown_retention,
        "waste_rate": waste_rate,
        "details": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the locked research holdout")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_eval(args.fixture)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            f"Research holdout: {result['status']} "
            f"rescue={result['agentic_rescue_rate']:.3f} "
            f"precision={result['source_backed_claim_precision']:.3f}"
        )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
