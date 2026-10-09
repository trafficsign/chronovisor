#!/usr/bin/env python3.14
"""Prepare an isolated, source-bound Recall P3 synthetic probe contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from chronovisor.core.canonical_json import (
    canonical_json_bytes_strict,
    canonical_json_line_bytes_strict,
    canonical_json_sha256_strict,
)

SCHEMA = "chronovisor.recall-p3-synthetic-probes.v1"
REPAIR_SCHEMA = "chronovisor.recall-p3-verified-miss-repair.v1"
DIAGNOSTIC_SCHEMA = "chronovisor.recall-p3-fixture-diagnostic-miss.v1"
TEACHER_ONLY_SCHEMA = "chronovisor.recall-p3-teacher-only.v1"
CONTROL_KINDS = ("nonexistent_fact", "stale_fact", "distractor", "topic_switch")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class ProbeError(ValueError):
    """A preparation input or artifact violated the P3 contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_object(value: object) -> str:
    return canonical_json_sha256_strict(value)


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ProbeError(f"{label} must be a lowercase SHA-256 digest")
    return str(value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProbeError(f"{label} must be a non-empty string")
    return value.strip()


def _time(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProbeError(f"{label} must be an ISO-8601 timestamp or null")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProbeError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ProbeError(f"{label} must include a timezone")
    return parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validity(value: object, label: str) -> dict[str, str | None]:
    if not isinstance(value, Mapping) or not set(value).issubset(
        {"valid_from", "valid_to"}
    ):
        raise ProbeError(f"{label} must contain only valid_from/valid_to")
    start = _time(value.get("valid_from"), f"{label}.valid_from")
    end = _time(value.get("valid_to"), f"{label}.valid_to")
    if not start and not end:
        raise ProbeError(f"{label} needs a bounded endpoint")
    if start and end and start >= end:
        raise ProbeError(f"{label} has an invalid interval")
    return {"valid_from": start, "valid_to": end}


def _relative(value: object, label: str) -> str:
    relative = _text(value, label).replace("\\", "/")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or ".." in pure.parts
        or any(part in {"", "."} for part in pure.parts)
        or pure.as_posix() != relative
    ):
        raise ProbeError(f"{label} must be a canonical relative path")
    return relative


def _fixture_path(root: Path, relative: str) -> Path:
    base = root.resolve(strict=True)
    cursor = base
    for part in PurePosixPath(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ProbeError(f"source fixture is symlinked: {relative}")
    try:
        path = cursor.resolve(strict=True)
        path.relative_to(base)
    except (OSError, ValueError) as exc:
        raise ProbeError(f"source fixture escaped fixture root: {relative}") from exc
    if not path.is_file():
        raise ProbeError(f"source fixture is not a regular file: {relative}")
    return path


def _source(value: object, root: Path, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeError(f"{label} must be an object")
    required = {"relative_path", "sha256", "byte_start", "byte_end"}
    if not required.issubset(value) or not set(value).issubset(
        required | {"excerpt_sha256"}
    ):
        raise ProbeError(f"{label} has invalid keys")
    relative = _relative(value.get("relative_path"), f"{label}.relative_path")
    declared = _require_sha(value.get("sha256"), f"{label}.sha256")
    start, end = value.get("byte_start"), value.get("byte_end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ProbeError(f"{label} byte span is invalid")
    raw = _fixture_path(root, relative).read_bytes()
    if _sha256_bytes(raw) != declared:
        raise ProbeError(f"{label}.sha256 does not match fixture bytes")
    if end > len(raw):
        raise ProbeError(f"{label} byte span exceeds fixture bytes")
    excerpt = _sha256_bytes(raw[start:end])
    if value.get("excerpt_sha256", excerpt) != excerpt:
        raise ProbeError(f"{label}.excerpt_sha256 does not match fixture bytes")
    return {
        "relative_path": relative,
        "sha256": declared,
        "byte_start": start,
        "byte_end": end,
        "excerpt_sha256": excerpt,
    }


def _rows(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ProbeError(f"{label} must be a regular file")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProbeError(f"cannot read {label}") from exc
    if not text.strip():
        raise ProbeError(f"{label} is empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        values: list[Any] = []
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                raise ProbeError(f"{label} has a blank line at {line_no}") from None
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ProbeError(f"invalid JSON in {label}:{line_no}") from exc
    else:
        if isinstance(parsed, list):
            values = parsed
        elif isinstance(parsed, Mapping):
            values = next(
                (
                    parsed[key]
                    for key in ("facts", "verifications", "comparisons")
                    if isinstance(parsed.get(key), list)
                ),
                [parsed],
            )
        else:
            raise ProbeError(f"{label} must contain JSON objects")
    if not values or not all(isinstance(item, dict) for item in values):
        raise ProbeError(f"{label} must contain non-empty JSON objects")
    return [dict(item) for item in values]


def _private(path: Path, *, directory: bool = False) -> None:
    if path.is_symlink() or not path.exists():
        raise ProbeError(f"private artifact is missing or symlinked: {path}")
    if stat.S_IMODE(path.stat().st_mode) != (0o700 if directory else 0o600):
        raise ProbeError(f"private artifact has unsafe permissions: {path}")


def _write_private(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ProbeError(f"artifact already exists: {path}")
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o600)
    except OSError as exc:
        raise ProbeError(f"cannot write private artifact: {path}") from exc


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_line_bytes_strict(row) for row in rows)


def _fact(row: Mapping[str, Any], root: Path, index: int) -> dict[str, Any]:
    required = {"subject", "relation", "value", "validity", "source", "family"}
    if not required.issubset(row):
        raise ProbeError(f"fact {index} is missing required fields")
    subject, relation = (
        _text(row["subject"], f"fact {index}.subject"),
        _text(row["relation"], f"fact {index}.relation"),
    )
    value, family = (
        _text(row["value"], f"fact {index}.value"),
        _text(row["family"], f"fact {index}.family"),
    )
    unsigned = {
        "subject": subject,
        "relation": relation,
        "value": value,
        "validity": _validity(row["validity"], f"fact {index}.validity"),
        "source": _source(row["source"], root, f"fact {index}.source"),
        "family": family,
    }
    digest = _sha256_object(unsigned)
    fact_id = row.get("fact_id", f"fact-{digest[:20]}")
    fact_id = _text(fact_id, f"fact {index}.fact_id")
    if _ID.fullmatch(fact_id) is None:
        raise ProbeError(f"fact {index}.fact_id has invalid characters")
    candidates = row.get("question_candidates")
    if candidates is None:
        candidates = [f"{subject}の{relation}は何ですか？"]
    if (
        not isinstance(candidates, list)
        or not candidates
        or not all(isinstance(item, str) and item.strip() for item in candidates)
    ):
        raise ProbeError(f"fact {index}.question_candidates must be non-empty strings")
    return {
        **unsigned,
        "fact_id": fact_id,
        "fact_sha256": digest,
        "question_candidates": [item.strip() for item in candidates],
        "semantic_status": "unverified",
        "model_agreement": "unmeasured",
        "promotion_status": "blocked_unverified",
    }


def _binding(fact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fact_id": fact["fact_id"],
        "fact_sha256": fact["fact_sha256"],
        "family": fact["family"],
        "source": dict(fact["source"]),
    }


def _default_verification(question: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "verification_kind": "independent_answerability",
        "question_id": question["question_id"],
        "fact_id": question["fact_id"],
        "fact_sha256": question["fact_sha256"],
        "question_sha256": question["question_sha256"],
        "status": "unverified",
        "answerability": "unverified",
        "semantic_status": "unverified",
        "model_agreement": "unmeasured",
        "verifier_id": None,
        "evidence_sha256": None,
        "claimed_status": "unverified",
        "claimed_answerability": "unverified",
        "claimed_semantic_status": "unverified",
        "claimed_model_agreement": "unmeasured",
        "verification_authenticated": False,
        "promotion_eligible": False,
    }
    return {**unsigned, "verification_sha256": _sha256_object(unsigned)}


def _merge_verifications(
    questions: list[dict[str, Any]], rows: Sequence[Mapping[str, Any]] | None
) -> None:
    if rows is None:
        return
    by_id = {str(row["question_id"]): row for row in questions}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        qid = _text(row.get("question_id"), f"verification {index}.question_id")
        if qid in seen or qid not in by_id:
            raise ProbeError(
                f"verification {index} references unknown/duplicate question"
            )
        seen.add(qid)
        question = by_id[qid]
        for key, expected in (
            ("fact_id", question["fact_id"]),
            ("fact_sha256", question["fact_sha256"]),
            ("question_sha256", question["question_sha256"]),
        ):
            if row.get(key) != expected:
                label = "fact hash" if key == "fact_sha256" else key
                raise ProbeError(f"verification {index} {label} does not bind")
        claimed_status = row.get("status", "unverified")
        if claimed_status not in {"unverified", "verified", "rejected"}:
            raise ProbeError(f"verification {index}.status is invalid")
        claimed_answerability = row.get("answerability", "unverified")
        if claimed_answerability not in {
            "unverified",
            "answerable",
            "not_answerable",
            "unknown",
        }:
            raise ProbeError(f"verification {index}.answerability is invalid")
        claimed_semantic = row.get("semantic_status", "unverified")
        if claimed_semantic not in {"unverified", "verified", "rejected"}:
            raise ProbeError(f"verification {index}.semantic_status is invalid")
        claimed_agreement = row.get("model_agreement", "unmeasured")
        if claimed_agreement not in {
            "unmeasured",
            "verified",
            "disagreed",
            "not_applicable",
        }:
            raise ProbeError(f"verification {index}.model_agreement is invalid")
        verifier = row.get("verifier_id")
        if verifier is not None:
            verifier = _text(verifier, f"verification {index}.verifier_id")
        evidence = row.get("evidence_sha256")
        if evidence is not None:
            evidence = _require_sha(evidence, f"verification {index}.evidence_sha256")
        claim = {
            "verification_kind": "independent_answerability",
            "question_id": qid,
            "fact_id": question["fact_id"],
            "fact_sha256": question["fact_sha256"],
            "question_sha256": question["question_sha256"],
            "claimed_status": claimed_status,
            "claimed_answerability": claimed_answerability,
            "claimed_semantic_status": claimed_semantic,
            "claimed_model_agreement": claimed_agreement,
            "verifier_id": verifier,
            "evidence_sha256": evidence,
        }
        if row.get("verification_sha256") is not None and row[
            "verification_sha256"
        ] != _sha256_object(claim):
            raise ProbeError(f"verification {index} digest does not match claim")
        held = {
            **claim,
            "status": "unverified",
            "answerability": "unverified",
            "semantic_status": "unverified",
            "model_agreement": "unmeasured",
            "verification_authenticated": False,
            "promotion_eligible": False,
        }
        question["verification"] = {**held, "verification_sha256": _sha256_object(held)}


def _questions(facts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fact in facts:
        for index, text in enumerate(fact["question_candidates"]):
            qhash = _sha256_text(text)
            question = {
                "question_id": f"{fact['fact_id']}:q:{index}-{qhash[:12]}",
                "fact_id": fact["fact_id"],
                "fact_sha256": fact["fact_sha256"],
                "question": text,
                "question_sha256": qhash,
                "language": "ja",
                "candidate_status": "candidate_only",
            }
            question["verification"] = _default_verification(question)
            rows.append(question)
    return rows


def _probe_time(fact: Mapping[str, Any]) -> str | None:
    validity = fact["validity"]
    endpoint = validity.get("valid_to") or validity.get("valid_from")
    if endpoint is None:
        return None
    value = datetime.fromisoformat(endpoint.replace("Z", "+00:00"))
    value += timedelta(seconds=1 if validity.get("valid_to") else -1)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _alternate(
    fact: Mapping[str, Any],
    facts: Sequence[Mapping[str, Any]],
    *,
    different_family: bool = False,
    confusing: bool = False,
) -> Mapping[str, Any] | None:
    candidates = [item for item in facts if item["fact_id"] != fact["fact_id"]]
    if different_family:
        candidates = [item for item in candidates if item["family"] != fact["family"]]
    elif confusing:
        candidates = [
            item
            for item in candidates
            if item["subject"] == fact["subject"]
            or item["relation"] == fact["relation"]
            or item["family"] == fact["family"]
        ]
    return candidates[0] if candidates else None


def _control(
    fact: Mapping[str, Any], facts: Sequence[Mapping[str, Any]], kind: str
) -> dict[str, Any]:
    target = str(fact["fact_id"])
    alternate = _alternate(
        fact,
        facts,
        different_family=kind == "topic_switch",
        confusing=kind == "distractor",
    )
    if kind == "nonexistent_fact":
        question, expected, candidates, bindings = (
            f"{fact['subject']}の存在しない{fact['relation']}は何ですか？",
            [],
            [],
            [],
        )
        probe_at, ready, semantics = None, True, "fact_is_absent_from_fixture"
        family = f"{fact['family']}:nonexistent"
    elif kind == "stale_fact":
        probe_at = _probe_time(fact)
        question = f"{fact['subject']}の{fact['relation']}（{probe_at or '期限外'}時点）は何ですか？"
        expected, candidates, bindings = [], [target], [_binding(fact)]
        ready, semantics, family = (
            probe_at is not None,
            "fact_is_outside_validity_interval",
            str(fact["family"]),
        )
    elif kind == "distractor":
        ready = alternate is not None
        question = f"{fact['subject']}の{fact['relation']}はどれですか？"
        expected = [target] if ready else []
        candidates = [target, str(alternate["fact_id"])] if ready else []
        bindings = [_binding(fact), _binding(alternate)] if ready else []
        probe_at, semantics, family = (
            None,
            (
                "same_or_related_family_contains_distractor"
                if ready
                else "missing_distractor_fixture"
            ),
            str(fact["family"]),
        )
    elif kind == "topic_switch":
        ready = alternate is not None
        switch_id = str(alternate["fact_id"]) if ready else None
        switch = alternate or {"subject": "別資料", "relation": "内容"}
        question = f"{fact['subject']}の{fact['relation']}を確認した後、別話題の{switch['subject']}の{switch['relation']}は何ですか？"
        expected = [switch_id] if ready and switch_id else []
        candidates = [target, switch_id] if ready and switch_id else []
        bindings = [_binding(fact), _binding(alternate)] if ready else []
        probe_at, semantics, family = (
            None,
            (
                "adjacent_context_switches_topic_family"
                if ready
                else "missing_topic_family_fixture"
            ),
            f"{fact['family']}:topic-switch",
        )
    else:
        raise ProbeError(f"unsupported control kind: {kind}")
    unsigned = {
        "control_id": f"{target}:control:{kind}",
        "control_kind": kind,
        "question": question,
        "question_sha256": _sha256_text(question),
        "target_fact_id": None if kind == "nonexistent_fact" else target,
        "target_fact_sha256": None
        if kind == "nonexistent_fact"
        else fact["fact_sha256"],
        "expected_fact_ids": expected,
        "candidate_fact_ids": candidates,
        "source_bindings": bindings,
        "distractor_fact_ids": (
            [target]
            if kind == "topic_switch" and ready
            else [str(alternate["fact_id"])]
            if kind == "distractor" and alternate is not None
            else []
        ),
        "probe_at": probe_at,
        "family": family,
        "semantics": semantics,
        "ready": ready,
        "control_status": "ready" if ready else "not_ready_missing_fixture_relation",
        "fixture_scope": "synthetic_fixture_only",
        "source_truth_status": "unverified",
        "promotion_status": "blocked_synthetic_fixture",
    }
    return {**unsigned, "control_sha256": _sha256_object(unsigned)}


def _diagnostics(
    comparisons: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_fact = {str(row["fact_id"]): row for row in facts}
    by_probe = {str(row["question_id"]): row for row in questions} | {
        str(row["control_id"]): row for row in controls
    }
    held: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, row in enumerate(comparisons):
        probe_id = _text(row.get("probe_id"), f"comparison {index}.probe_id")
        probe = by_probe.get(probe_id)
        if probe is None:
            raise ProbeError(f"comparison {index} references an unknown probe")
        expected_value, actual_value = (
            row.get("expected_fact_ids"),
            row.get("actual_fact_ids"),
        )
        if not isinstance(expected_value, list) or not all(
            isinstance(item, str) for item in expected_value
        ):
            raise ProbeError(f"comparison {index}.expected_fact_ids is invalid")
        if not isinstance(actual_value, list) or not all(
            isinstance(item, str) for item in actual_value
        ):
            raise ProbeError(f"comparison {index}.actual_fact_ids is invalid")
        expected, actual = (
            list(dict.fromkeys(expected_value)),
            list(dict.fromkeys(actual_value)),
        )
        probe_expected = (
            [str(probe["fact_id"])]
            if "question_id" in probe
            else [str(item) for item in probe.get("expected_fact_ids", [])]
        )
        if expected != probe_expected:
            raise ProbeError(f"comparison {index} expected facts do not bind to probe")
        bindings = row.get("expected_fact_bindings")
        if not isinstance(bindings, list):
            raise ProbeError(f"comparison {index}.expected_fact_bindings is required")
        binding_map: dict[str, str] = {}
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ProbeError(f"comparison {index} expected binding is invalid")
            binding_map[
                _text(binding.get("fact_id"), "comparison expected fact_id")
            ] = _require_sha(
                binding.get("fact_sha256"), "comparison expected fact_sha256"
            )
        if (
            set(binding_map) != set(expected)
            or any(
                binding_map[fact_id] != by_fact[fact_id]["fact_sha256"]
                for fact_id in expected
                if fact_id in by_fact
            )
            or any(fact_id not in by_fact for fact_id in expected)
        ):
            raise ProbeError(f"comparison {index} expected bindings do not match facts")
        comparison_sha = _require_sha(
            row.get("comparison_sha256"), f"comparison {index}.comparison_sha256"
        )
        comparison_unsigned = {
            str(key): value for key, value in row.items() if key != "comparison_sha256"
        }
        if comparison_sha != _sha256_object(comparison_unsigned):
            raise ProbeError(f"comparison {index} digest does not match payload")
        missing = [fact_id for fact_id in expected if fact_id not in actual]
        status, source_truth, receipt = (
            row.get("comparison_status"),
            row.get("source_truth_status"),
            row.get("verification_receipt_sha256"),
        )
        reason = None
        if status != "verified":
            reason = "comparison_not_verified"
        elif source_truth != "verified":
            reason = "source_truth_not_verified"
        elif not isinstance(receipt, str) or _HEX.fullmatch(receipt) is None:
            reason = "verification_receipt_missing"
        elif row.get("teacher_only") is True:
            reason = "teacher_only_artifact_is_separate"
        elif not missing:
            reason = "verified_comparison_has_no_miss"
        if missing:
            unsigned = {
                "diagnostic_schema": DIAGNOSTIC_SCHEMA,
                "diagnostic_id": f"fixture-miss-{comparison_sha[:20]}",
                "probe_id": probe_id,
                "expected_fact_ids": expected,
                "missing_fact_ids": missing,
                "actual_fact_ids": actual,
                "expected_fact_bindings": bindings,
                "comparison_sha256": comparison_sha,
                "claimed_comparison_status": status,
                "claimed_source_truth_status": source_truth,
                "claimed_verification_receipt_sha256": receipt,
                "teacher_only": row.get("teacher_only") is True,
                "verified": False,
                "promotion_eligible": False,
                "reason": "fixture_diagnostic_only",
            }
            diagnostics.append(
                {**unsigned, "diagnostic_sha256": _sha256_object(unsigned)}
            )
        if reason:
            held.append(
                {
                    "probe_id": probe_id,
                    "comparison_sha256": comparison_sha,
                    "status": "held",
                    "reason": reason,
                    "repair_queue_written": False,
                }
            )
    return [], held, diagnostics


def _fixture_digest(facts: Sequence[Mapping[str, Any]]) -> str:
    sources = [
        {
            "relative_path": fact["source"]["relative_path"],
            "sha256": fact["source"]["sha256"],
            "byte_start": fact["source"]["byte_start"],
            "byte_end": fact["source"]["byte_end"],
        }
        for fact in facts
    ]
    return _sha256_object(sorted(sources, key=lambda row: row["relative_path"]))


def _disjoint(root: Path, output: Path) -> None:
    base, target = root.resolve(strict=True), output.resolve(strict=False)
    try:
        target.relative_to(base)
    except ValueError:
        pass
    else:
        raise ProbeError("output must be outside the synthetic fixture corpus")
    try:
        base.relative_to(target)
    except ValueError:
        pass
    else:
        raise ProbeError("output parent must not contain synthetic fixture corpus")


def prepare_bundle(
    facts_path: Path,
    fixture_root: Path,
    output: Path,
    *,
    verifications_path: Path | None = None,
    comparisons_path: Path | None = None,
) -> dict[str, Any]:
    root, destination = fixture_root.expanduser(), output.expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ProbeError("fixture root must be a real directory")
    if (
        destination.exists()
        or destination.is_symlink()
        or not destination.parent.is_dir()
    ):
        raise ProbeError("output must be a new path with an existing parent")
    _disjoint(root, destination)
    destination.mkdir(mode=0o700)
    os.chmod(destination, 0o700)

    facts_input = _rows(facts_path.expanduser(), "facts input")
    facts = [_fact(row, root, index) for index, row in enumerate(facts_input)]
    if len({row["fact_id"] for row in facts}) != len(facts):
        raise ProbeError("fact ids must be unique")
    if len({row["fact_sha256"] for row in facts}) != len(facts):
        raise ProbeError("fact hashes must be unique")
    questions = _questions(facts)
    verification_rows = (
        _rows(verifications_path.expanduser(), "verification input")
        if verifications_path is not None
        else None
    )
    _merge_verifications(questions, verification_rows)
    controls = [_control(fact, facts, kind) for fact in facts for kind in CONTROL_KINDS]
    comparisons = (
        _rows(comparisons_path.expanduser(), "comparison input")
        if comparisons_path is not None
        else []
    )
    repair_queue, held, diagnostics = _diagnostics(
        comparisons, facts, questions, controls
    )
    artifacts = {
        "facts.jsonl": facts,
        "questions.jsonl": questions,
        "controls.jsonl": controls,
        "repair_queue.jsonl": repair_queue,
        "diagnostic_misses.jsonl": diagnostics,
        "held_comparisons.jsonl": held,
    }
    hashes: dict[str, str] = {}
    for name, rows in artifacts.items():
        raw = _jsonl(rows)
        _write_private(destination / name, raw)
        hashes[name] = _sha256_bytes(raw)
    counts = Counter(str(row["control_kind"]) for row in controls)
    ready = sum(row["ready"] is True for row in controls)
    unsigned_manifest = {
        "schema": SCHEMA,
        "status": "prepared",
        "source_kind": "synthetic_fixture",
        "fixture_root": str(root.resolve()),
        "fixture_sources_sha256": _fixture_digest(facts),
        "facts_path_sha256": _sha256_bytes(facts_path.expanduser().read_bytes()),
        "fact_count": len(facts),
        "question_count": len(questions),
        "control_count": len(controls),
        "control_counts": dict(sorted(counts.items())),
        "ready_control_count": ready,
        "not_ready_control_count": len(controls) - ready,
        "repair_queue_count": 0,
        "diagnostic_miss_count": len(diagnostics),
        "held_comparison_count": len(held),
        "artifact_hashes": hashes,
        "model_calls": 0,
        "production_corpus_written": False,
        "teacher_only_schema": TEACHER_ONLY_SCHEMA,
        "repair_queue_schema": REPAIR_SCHEMA,
        "diagnostic_schema": DIAGNOSTIC_SCHEMA,
        "formal_acceptance": False,
        "promotion_status": "blocked_until_verified_actual_data",
    }
    manifest = {
        **unsigned_manifest,
        "manifest_sha256": _sha256_object(unsigned_manifest),
    }
    manifest_raw = canonical_json_bytes_strict(manifest) + b"\n"
    _write_private(destination / "manifest.json", manifest_raw)
    unsigned_receipt = {
        "schema": SCHEMA,
        "receipt_status": "verified_preparation_only",
        "manifest_sha256": _sha256_bytes(manifest_raw),
        "artifact_hashes": hashes,
        "permissions": {"directory": "0700", "files": "0600"},
        "production_corpus_written": False,
        "model_calls": 0,
        "formal_acceptance": False,
        "promotion_status": "blocked_until_verified_actual_data",
    }
    receipt = {**unsigned_receipt, "receipt_sha256": _sha256_object(unsigned_receipt)}
    _write_private(
        destination / "receipt.json", canonical_json_bytes_strict(receipt) + b"\n"
    )
    _write_private(destination / ".gitignore", b"*\n!.gitignore\n")
    return {
        "status": "prepared",
        "output": str(destination.resolve()),
        "fact_count": len(facts),
        "question_count": len(questions),
        "control_count": len(controls),
        "repair_queue_count": 0,
        "diagnostic_miss_count": len(diagnostics),
        "promotion_status": manifest["promotion_status"],
    }


def _private_json(path: Path) -> dict[str, Any]:
    _private(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise ProbeError(f"JSON artifact must be an object: {path.name}")
    return value


def _private_jsonl(path: Path) -> list[dict[str, Any]]:
    _private(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ProbeError(f"cannot read artifact: {path.name}") from exc
    if not raw:
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProbeError(f"invalid artifact JSON at {path.name}:{line_no}") from exc
        if not isinstance(value, dict):
            raise ProbeError(f"artifact row is not an object: {path.name}:{line_no}")
        rows.append(value)
    return rows


def verify_bundle(output: Path) -> dict[str, Any]:
    destination = output.expanduser()
    _private(destination, directory=True)
    manifest = _private_json(destination / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "prepared":
        raise ProbeError("unsupported P3 preparation manifest")
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != _sha256_object(unsigned_manifest):
        raise ProbeError("manifest digest mismatch")
    if (
        manifest.get("production_corpus_written") is not False
        or manifest.get("model_calls") != 0
        or manifest.get("repair_queue_count") != 0
        or manifest.get("formal_acceptance") is not False
    ):
        raise ProbeError("P3 preparation is not held as diagnostic-only")
    names = (
        "facts.jsonl",
        "questions.jsonl",
        "controls.jsonl",
        "repair_queue.jsonl",
        "diagnostic_misses.jsonl",
        "held_comparisons.jsonl",
    )
    artifacts: dict[str, list[dict[str, Any]]] = {}
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, Mapping):
        raise ProbeError("manifest artifact hashes are missing")
    for name in names:
        path, raw = (
            destination / name,
            (destination / name).read_bytes()
            if (destination / name).is_file()
            else b"",
        )
        _private(path)
        if hashes.get(name) != _sha256_bytes(raw):
            raise ProbeError(f"artifact hash mismatch: {name}")
        artifacts[name] = _private_jsonl(path)
    if artifacts["repair_queue.jsonl"]:
        raise ProbeError("formal repair queue must remain empty")
    facts, questions, controls = (
        artifacts["facts.jsonl"],
        artifacts["questions.jsonl"],
        artifacts["controls.jsonl"],
    )
    if (
        manifest.get("fact_count") != len(facts)
        or manifest.get("question_count") != len(questions)
        or manifest.get("control_count") != len(controls)
    ):
        raise ProbeError("manifest row count mismatch")
    facts_by_id = {str(row.get("fact_id")): row for row in facts}
    if len(facts_by_id) != len(facts):
        raise ProbeError("fact ids are not unique")
    root = Path(str(manifest.get("fixture_root", "")))
    if root.is_symlink() or not root.is_dir():
        raise ProbeError("fixture root is unavailable")
    for index, fact in enumerate(facts):
        unsigned = {
            key: fact.get(key)
            for key in ("subject", "relation", "value", "validity", "source", "family")
        }
        if fact.get("fact_sha256") != _sha256_object(unsigned):
            raise ProbeError(f"fact hash mismatch at row {index}")
        _source(fact.get("source"), root, f"artifact fact {index}.source")
    if manifest.get("fixture_sources_sha256") != _fixture_digest(facts):
        raise ProbeError("fixture source digest mismatch")
    for index, question in enumerate(questions):
        fact = facts_by_id.get(str(question.get("fact_id")))
        if fact is None or question.get("fact_sha256") != fact.get("fact_sha256"):
            raise ProbeError(f"question fact binding mismatch at row {index}")
        if question.get("question_sha256") != _sha256_text(
            str(question.get("question") or "")
        ):
            raise ProbeError(f"question hash mismatch at row {index}")
        verification = question.get("verification")
        if (
            not isinstance(verification, Mapping)
            or verification.get("status") != "unverified"
            or verification.get("verification_authenticated") is not False
            or verification.get("promotion_eligible") is not False
        ):
            raise ProbeError(
                f"question verification is not held unverified at row {index}"
            )
        unsigned = {
            key: value
            for key, value in verification.items()
            if key != "verification_sha256"
        }
        if verification.get("verification_sha256") != _sha256_object(unsigned):
            raise ProbeError(f"question verification digest mismatch at row {index}")
    count = Counter(str(row.get("control_kind")) for row in controls)
    if dict(sorted(count.items())) != manifest.get("control_counts"):
        raise ProbeError("control counts changed")
    ready = sum(row.get("ready") is True for row in controls)
    if (
        manifest.get("ready_control_count") != ready
        or manifest.get("not_ready_control_count") != len(controls) - ready
    ):
        raise ProbeError("control readiness counts changed")
    for index, control in enumerate(controls):
        unsigned = {
            key: value for key, value in control.items() if key != "control_sha256"
        }
        if control.get("control_sha256") != _sha256_object(unsigned):
            raise ProbeError(f"control digest mismatch at row {index}")
        if (
            control.get("fixture_scope") != "synthetic_fixture_only"
            or control.get("promotion_status") != "blocked_synthetic_fixture"
        ):
            raise ProbeError(f"control promotion status changed at row {index}")
    diagnostics = artifacts["diagnostic_misses.jsonl"]
    if manifest.get("diagnostic_miss_count") != len(diagnostics):
        raise ProbeError("diagnostic miss count mismatch")
    for index, diagnostic in enumerate(diagnostics):
        unsigned = {
            key: value
            for key, value in diagnostic.items()
            if key != "diagnostic_sha256"
        }
        if (
            diagnostic.get("diagnostic_schema") != DIAGNOSTIC_SCHEMA
            or diagnostic.get("verified") is not False
            or diagnostic.get("promotion_eligible") is not False
        ):
            raise ProbeError(f"diagnostic promotion status changed at row {index}")
        if diagnostic.get("diagnostic_sha256") != _sha256_object(unsigned):
            raise ProbeError(f"diagnostic digest mismatch at row {index}")
    receipt = _private_json(destination / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256_object(unsigned):
        raise ProbeError("receipt digest mismatch")
    if receipt.get("manifest_sha256") != _sha256_bytes(
        (destination / "manifest.json").read_bytes()
    ):
        raise ProbeError("receipt manifest binding mismatch")
    return {
        "status": "verified",
        "fact_count": len(facts),
        "question_count": len(questions),
        "control_count": len(controls),
        "repair_queue_count": 0,
        "diagnostic_miss_count": len(diagnostics),
        "formal_acceptance": False,
        "promotion_status": manifest.get("promotion_status"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifications", type=Path)
    parser.add_argument("--comparisons", type=Path)
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verify:
            summary = verify_bundle(args.output)
        else:
            if args.facts is None or args.fixture_root is None:
                raise ProbeError(
                    "--facts and --fixture-root are required when preparing"
                )
            summary = prepare_bundle(
                args.facts,
                args.fixture_root,
                args.output,
                verifications_path=args.verifications,
                comparisons_path=args.comparisons,
            )
    except ProbeError as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
