#!/usr/bin/env python3.14
"""Replay the five locked-gold rubric rows through the StartLux primary judge.

The gold artifact stores hashes and labels, not prompt text. Supply either a
``--cases`` JSONL with ``case_id``/``query``/``page_id``/``evidence_span``/
``gold`` fields, or a source ledger whose rows expose a query and page text.
The script is read-only apart from ``--output-dir``, calls the service at
PRIORITY_SEARCH, and never falls back to Qwen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from chronovisor.core import reranker_client
from chronovisor.core.runtime_config import load_reranker_config
from chronovisor.recall.recall_processor import PRIMARY_JUDGE_RUNTIME_ROLE
from chronovisor.recall.rubric_calibration import load_active_rubric

PRIORITY_SEARCH = 1
GOLD = Path("~/.chronovisor/runtime/recall-rubric/locked-gold.jsonl")
SOURCE_LEDGER = Path("~/.chronovisor/recall/search-golden.jsonl")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return []
    return rows


def _source_rows(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        query = row.get("query") or row.get("prompt") or row.get("question")
        if not isinstance(query, str) or not query:
            continue
        page_id = row.get("page_id")
        if not isinstance(page_id, str):
            candidates = row.get("candidates")
            if (
                isinstance(candidates, list)
                and candidates
                and isinstance(candidates[0], dict)
            ):
                page_id = candidates[0].get("page_id")
        evidence = (
            row.get("evidence_span")
            or row.get("excerpt")
            or row.get("text")
            or row.get("snippet")
            or ""
        )
        if not isinstance(page_id, str) or not isinstance(evidence, str):
            continue
        index[_sha(query)] = {
            "query": query,
            "page_id": page_id,
            "evidence_span": evidence,
        }
    return index


def _case_index(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("case_id") or ""): row for row in _jsonl(path) if row.get("case_id")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--source-ledger", type=Path, default=SOURCE_LEDGER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-ms", type=int, default=900)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_rows = _jsonl(args.gold.expanduser())
    cases = _case_index(args.cases.expanduser()) if args.cases else {}
    sources = {} if args.cases else _source_rows(args.source_ledger.expanduser())
    rubric = load_active_rubric()
    rubric_text = str(rubric.get("rubric_text") or "")
    rubric_id = str(rubric.get("rubric_id") or "")
    config = load_reranker_config()
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for row in gold_rows:
        case_id = str(row.get("case_id") or "")
        source = cases.get(case_id)
        if source is None:
            source = sources.get(str(row.get("query_sha256") or ""))
        if (
            source is None
            or not isinstance(source.get("query"), str)
            or not isinstance(source.get("page_id"), str)
            or not isinstance(source.get("evidence_span"), str)
        ):
            skipped.append({"case_id": case_id, "reason": "source_text_unavailable"})
            continue
        response = reranker_client.decision_judge(
            "certificate",
            {
                "query": source["query"],
                "rubric": rubric_text,
                "rubric_id": rubric_id,
                "evidence": [
                    {
                        "page_id": source["page_id"],
                        "span": source["evidence_span"],
                    }
                ],
            },
            role=PRIMARY_JUDGE_RUNTIME_ROLE,
            config=config,
            timeout_ms=max(200, args.timeout_ms),
            priority=PRIORITY_SEARCH,
        )
        verdicts = response.get("verdicts")
        verdict = (
            verdicts[0]
            if isinstance(verdicts, list) and verdicts and isinstance(verdicts[0], dict)
            else {}
        )
        expected = "pass" if row.get("gold") is True else "reject"
        actual = str(verdict.get("decision") or "missing")
        results.append(
            {
                "case_id": case_id,
                "expected": expected,
                "actual": actual,
                "confidence": verdict.get("confidence"),
                "matched": actual == expected,
                "false_pass": expected == "reject" and actual == "pass",
            }
        )
    matched = sum(result["matched"] is True for result in results)
    false_pass = sum(result["false_pass"] is True for result in results)
    payload = {
        "status": (
            "complete"
            if len(results) == len(gold_rows)
            and matched == len(gold_rows)
            and false_pass == 0
            else "held"
        ),
        "gold_rows": len(gold_rows),
        "replayed": len(results),
        "skipped": skipped,
        "matched": matched,
        "false_pass": false_pass,
        "accuracy": round(matched / len(results), 6) if results else 0.0,
        "priority": PRIORITY_SEARCH,
        "results": results,
    }
    (output_dir / "certificate-smoke.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
