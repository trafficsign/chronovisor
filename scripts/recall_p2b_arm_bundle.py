"""Prepare private, source-bound projections for the Recall P2b arms.

The bundle stores hashes and arm metadata only.  Source pages and generated
questions stay in their already sealed JSONL artifacts; callers can use the
iterators below when a full projection is needed for an isolated experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_document import CanonicalDocumentError, parse_document
from chronovisor.core.search import _markdown_chunks, _recall_questions
from chronovisor.core.search_types import tokenize

SCHEMA_VERSION = 2
ARM_NAMES = ("A", "B", "C", "D", "E")
SUPPORTED_COUNTS = frozenset({3, 4, 5})
UNSUPPORTED_COUNTS = frozenset({1, 2, 6})
EXPECTED_PAGE_COUNT = 9_795
EXPECTED_B_REPLACEMENTS = 7_227
EXPECTED_UNSUPPORTED = 94
EXPECTED_MISSING = 2_474


class BundleError(ValueError):
    """Raised when a sealed source or projection violates its contract."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_hash(value: Any) -> str:
    return _sha256_bytes(_json_bytes(value))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BundleError(f"cannot read JSONL artifact: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise BundleError(f"blank JSONL line at {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BundleError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise BundleError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"JSON artifact is not an object: {path}")
    return value


def _private_file(path: Path, *, directory: bool = False) -> None:
    expected = 0o700 if directory else 0o600
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        raise BundleError(f"private artifact is unavailable: {path}") from exc
    if path.is_symlink() or mode != expected:
        raise BundleError(f"private artifact permissions are unsafe: {path}")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise BundleError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise BundleError(f"{label} is not a SHA-256 digest") from exc
    return value


def _questions(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(question, str) and question.strip() for question in value
    ):
        raise BundleError(f"{label} must be a list of non-empty strings")
    return [question for question in value]


def _generation_input_bindings(
    candidate_path: Path, source_path: Path
) -> dict[int, str] | None:
    """Bind candidate input hashes to the sealed generation shard sources."""

    bundle = candidate_path.parent
    manifest_path = bundle / "manifest.json"
    shards = bundle / "shards"
    if not manifest_path.is_file() or not shards.is_dir():
        return None
    _private_file(manifest_path)
    manifest = _read_json(manifest_path)
    if manifest.get("source_snapshot_sha256") != _sha256_file(source_path):
        raise BundleError("generation input source snapshot hash mismatch")
    specs = manifest.get("shards")
    if not isinstance(specs, list) or not specs:
        raise BundleError("generation manifest has no shards")
    bindings: dict[int, str] = {}
    for spec in specs:
        if not isinstance(spec, dict) or type(spec.get("shard")) is not int:
            raise BundleError("generation shard specification is invalid")
        shard_path = shards / f"{spec['shard']:02d}" / "source.jsonl"
        _private_file(shard_path)
        raw = shard_path.read_bytes()
        if _sha256_bytes(raw) != spec.get("source_sha256"):
            raise BundleError(f"generation shard source hash mismatch: {shard_path}")
        for line in raw.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BundleError(f"invalid generation shard row: {shard_path}") from exc
            if not isinstance(row, dict) or type(row.get("source_index")) is not int:
                raise BundleError(f"generation shard row identity is invalid: {shard_path}")
            bindings[row["source_index"]] = _require_sha256(
                row.get("input_sha256"), "generation input_sha256"
            )
    if len(bindings) != manifest.get("selected_pages") or set(bindings) != set(range(int(manifest.get("selected_pages", -1)))):
        raise BundleError("generation input binding has a row gap")
    return bindings


def _source_metadata(source: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    content = source.get("content")
    body = source.get("body")
    if not isinstance(content, str) or not isinstance(body, str):
        raise BundleError("source row requires string content and body")
    declared_hash = _require_sha256(source.get("source_sha256"), "source_sha256")
    actual_hash = _sha256_bytes(content.encode("utf-8"))
    if actual_hash != declared_hash:
        raise BundleError(f"source content hash mismatch: {source.get('page_id')!r}")
    try:
        document = parse_document(content.encode("utf-8"))
        parsed_body = document.body.decode("utf-8")
    except (CanonicalDocumentError, UnicodeDecodeError) as exc:
        raise BundleError(f"source content is not canonical: {source.get('page_id')!r}") from exc
    if parsed_body != body:
        raise BundleError(f"source body changed: {source.get('page_id')!r}")
    if document.metadata.get("status") != "stable":
        raise BundleError(f"source page is not stable: {source.get('page_id')!r}")
    title = source.get("title")
    parsed_title = document.metadata.get("title")
    if not isinstance(title, str) or not isinstance(parsed_title, str):
        raise BundleError(f"source title missing: {source.get('page_id')!r}")
    if title != parsed_title:
        raise BundleError(f"source title changed: {source.get('page_id')!r}")
    return dict(document.metadata), body, declared_hash


def _current_questions(source: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    status = source.get("current_question_count_status")
    count = source.get("current_question_count")
    if status == "pending":
        if count is not None:
            raise BundleError(f"pending source has a count: {source.get('page_id')!r}")
        return []
    if status != "known" or type(count) is not int or count < 0:
        raise BundleError(f"invalid current question count: {source.get('page_id')!r}")
    questions = list(_recall_questions(metadata))
    if len(questions) != count:
        raise BundleError(
            f"source question count mismatch for {source.get('page_id')!r}: "
            f"declared={count} actual={len(questions)}"
        )
    return questions


def _validate_inputs(
    source_path: Path, candidate_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = _read_jsonl(source_path)
    candidate_rows = _read_jsonl(candidate_path)
    if len(source_rows) != len(candidate_rows):
        raise BundleError("source and candidate row counts differ")
    input_bindings = _generation_input_bindings(candidate_path, source_path)
    seen_pages: set[str] = set()
    seen_candidates: set[str] = set()
    for index, (source, candidate) in enumerate(zip(source_rows, candidate_rows, strict=True)):
        page_id = source.get("page_id")
        if not isinstance(page_id, str) or not page_id or page_id in seen_pages:
            raise BundleError(f"invalid or duplicate source page at row {index}")
        seen_pages.add(page_id)
        if candidate.get("page_id") != page_id or candidate.get("relative_path") != source.get("relative_path"):
            raise BundleError(f"source/candidate identity mismatch at row {index}")
        if candidate.get("source_index") != index:
            raise BundleError(f"candidate source index mismatch at row {index}")
        if candidate.get("source_sha256") != source.get("source_sha256"):
            raise BundleError(f"candidate source hash mismatch at row {index}")
        if page_id in seen_candidates:
            raise BundleError(f"duplicate candidate page at row {index}")
        seen_candidates.add(page_id)
        _source_metadata(source)
        source_status = source.get("current_question_count_status")
        source_count = source.get("current_question_count")
        if source_status == "known" and source_count not in UNSUPPORTED_COUNTS | SUPPORTED_COUNTS:
            raise BundleError(f"unexpected known question count at row {index}")
        if candidate.get("generation_status") != "generated" or candidate.get("generation_success") is not True:
            raise BundleError(f"candidate is not generated at row {index}")
        candidate_questions = _questions(candidate.get("recall_questions"), label="candidate recall_questions")
        requested = candidate.get("requested_question_count")
        if type(requested) is not int or requested != len(candidate_questions):
            raise BundleError(f"candidate question count mismatch at row {index}")
        _require_sha256(candidate.get("input_sha256"), "candidate input_sha256")
        if input_bindings is not None and candidate.get("input_sha256") != input_bindings.get(index):
            raise BundleError(f"candidate generation input changed at row {index}")
        if source_status == "known" and source_count in SUPPORTED_COUNTS and requested != source_count:
            raise BundleError(f"B same-count candidate question count mismatch at row {index}")
    if len(source_rows) != EXPECTED_PAGE_COUNT:
        raise BundleError(f"expected {EXPECTED_PAGE_COUNT} source pages, got {len(source_rows)}")
    return source_rows, candidate_rows


def _question_policy(arm: str, source: Mapping[str, Any], candidate: Mapping[str, Any], current: Sequence[str]) -> tuple[list[str], str]:
    if arm == "A":
        return list(current), "current"
    status = source.get("current_question_count_status")
    count = source.get("current_question_count")
    candidate_questions = _questions(candidate.get("recall_questions"), label="candidate recall_questions")
    if status == "known" and count in SUPPORTED_COUNTS:
        return candidate_questions, "luna_count_match"
    if status == "known" and count in UNSUPPORTED_COUNTS:
        if arm in {"C", "D", "E"}:
            return candidate_questions, "luna_unsupported_replacement"
        return list(current), "current_unsupported_preserved"
    if status == "pending" and arm in {"C", "D", "E"}:
        return candidate_questions, "luna_missing_completion"
    if status == "pending":
        return [], "missing_empty"
    raise BundleError(f"cannot select questions for arm {arm}: {source.get('page_id')!r}")


def semantic_projection(
    source: Mapping[str, Any], metadata: Mapping[str, Any], body: str, questions: Sequence[str], *, separate_questions: bool
) -> dict[str, Any]:
    """Mirror ``extract_page_documents`` while keeping D's page/questions separate."""

    page_id = str(source["page_id"])
    title = str(metadata.get("title") or page_id).strip() or page_id
    description_value = metadata.get("description")
    description = description_value.strip() if isinstance(description_value, str) else ""
    recall_text = "\n".join(f"Q: {question}" for question in questions)
    page_parts = (title, description, body[:2000]) if separate_questions else (title, description, recall_text, body[:2000])
    page_text = "\n\n".join(part for part in page_parts if part)
    chunks = list(_markdown_chunks(body, title, metadata))
    return {
        "page_text": page_text,
        "question_documents": [
            {"ordinal": ordinal, "text": question} for ordinal, question in enumerate(questions)
        ],
        "chunk_documents": [{"ordinal": ordinal, "text": chunk} for ordinal, chunk in enumerate(chunks)],
        "page_contains_question_text": bool(recall_text and not separate_questions),
    }


def bm25_tokens(metadata: Mapping[str, Any], body: str, questions: Sequence[str], *, include_questions: bool) -> list[str]:
    title_value = metadata.get("title")
    title = title_value.strip() if isinstance(title_value, str) else ""
    tokens = tokenize(title) * 3 + tokenize(body)
    if include_questions:
        for question in questions:
            tokens.extend(tokenize(question))
    return tokens


def student_candidate_text(title: str, snippet: str, questions: Sequence[str] = (), *, include_questions: bool = False) -> str:
    """Build the student text contract; production bytes are title + snippet only."""

    base = f"{title.strip()}\n{snippet.strip()}"
    if not include_questions or not questions:
        return base
    return base + "\n" + "\n".join(f"Q: {question}" for question in questions)


def iter_arm_projection(
    source_rows: Sequence[Mapping[str, Any]], candidate_rows: Sequence[Mapping[str, Any]], arm: str
) -> Iterator[dict[str, Any]]:
    if arm not in ARM_NAMES:
        raise BundleError(f"unknown arm: {arm}")
    if len(source_rows) != len(candidate_rows):
        raise BundleError("source and candidate row counts differ")
    for source, candidate in zip(source_rows, candidate_rows, strict=True):
        metadata, body, source_hash = _source_metadata(source)
        current = _current_questions(source, metadata)
        questions, question_origin = _question_policy(arm, source, candidate, current)
        separate_questions = arm == "D"
        semantic = semantic_projection(source, metadata, body, questions, separate_questions=separate_questions)
        lexical = bm25_tokens(metadata, body, questions, include_questions=arm == "E")
        question_hash = _value_hash(questions)
        arm_row = {
            "page_id": source["page_id"],
            "relative_path": source["relative_path"],
            "source_sha256": source_hash,
            "body_sha256": _sha256_bytes(body.encode("utf-8")),
            "question_origin": question_origin,
            "question_count": len(questions),
            "questions_sha256": question_hash,
            "semantic": semantic,
            "bm25_tokens": lexical,
            "student_contract": {
                "without_questions": "title+snippet",
                "with_questions": "title+snippet+Q-lines",
            },
        }
        yield arm_row


def _arm_digest(row: Mapping[str, Any]) -> str:
    semantic = row["semantic"]
    descriptor = {
        "page_id": row["page_id"],
        "relative_path": row["relative_path"],
        "source_sha256": row["source_sha256"],
        "body_sha256": row["body_sha256"],
        "question_origin": row["question_origin"],
        "question_count": row["question_count"],
        "questions_sha256": row["questions_sha256"],
        "semantic_page_sha256": _sha256_bytes(semantic["page_text"].encode("utf-8")),
        "semantic_question_sha256": _value_hash([doc["text"] for doc in semantic["question_documents"]]),
        "semantic_chunk_sha256": _value_hash([doc["text"] for doc in semantic["chunk_documents"]]),
        "page_contains_question_text": semantic["page_contains_question_text"],
        "bm25_token_count": len(row["bm25_tokens"]),
        "bm25_tokens_sha256": _value_hash(row["bm25_tokens"]),
        "student_contract": row["student_contract"],
    }
    return _value_hash(descriptor)


def _arm_descriptor(row: Mapping[str, Any], digest: str | None = None) -> dict[str, Any]:
    semantic = row["semantic"]
    return {
        "question_origin": row["question_origin"],
        "question_count": row["question_count"],
        "questions_sha256": row["questions_sha256"],
        "body_sha256": row["body_sha256"],
        "semantic_page_sha256": _sha256_bytes(semantic["page_text"].encode("utf-8")),
        "semantic_question_sha256": _value_hash(
            [doc["text"] for doc in semantic["question_documents"]]
        ),
        "semantic_kinds": ["page", "question", "chunk"],
        "page_contains_question_text": semantic["page_contains_question_text"],
        "bm25_token_count": len(row["bm25_tokens"]),
        "bm25_tokens_sha256": _value_hash(row["bm25_tokens"]),
        "content_sha256": digest if digest is not None else _arm_digest(row),
    }


def _project_descriptors(
    source_rows: Sequence[Mapping[str, Any]], candidate_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    projections: list[dict[str, Any]] = []
    arm_counters: dict[str, Counter[str]] = {arm: Counter() for arm in ARM_NAMES}
    arm_hashes: dict[str, list[str]] = {arm: [] for arm in ARM_NAMES}
    iterators = {arm: iter_arm_projection(source_rows, candidate_rows, arm) for arm in ARM_NAMES}
    for index, (source, *arm_rows) in enumerate(
        zip(source_rows, *(iterators[arm] for arm in ARM_NAMES), strict=True)
    ):
        arm_descriptors: dict[str, Any] = {}
        for arm, row in zip(ARM_NAMES, arm_rows, strict=True):
            digest = _arm_digest(row)
            arm_hashes[arm].append(digest)
            arm_counters[arm]["pages"] += 1
            arm_counters[arm]["questions"] += int(row["question_count"])
            arm_counters[arm][f"origin:{row['question_origin']}"] += 1
            arm_descriptors[arm] = _arm_descriptor(row, digest)
        projections.append(
            {
                "source_index": index,
                "page_id": source["page_id"],
                "source_sha256": source["source_sha256"],
                "arms": arm_descriptors,
            }
        )
    arm_summary: dict[str, Any] = {}
    for arm in ARM_NAMES:
        counts = arm_counters[arm]
        arm_summary[arm] = {
            "pages": counts["pages"],
            "questions": counts["questions"],
            "origin_counts": {
                key.removeprefix("origin:"): value
                for key, value in sorted(counts.items())
                if key.startswith("origin:")
            },
            "content_sha256": _value_hash(arm_hashes[arm]),
        }
    b_origins = arm_summary["B"]["origin_counts"]
    c_origins = arm_summary["C"]["origin_counts"]
    bound_counts = {
        "B_same_count_replacements": b_origins.get("luna_count_match", 0),
        "B_unsupported_preserved": b_origins.get("current_unsupported_preserved", 0),
        "B_missing_empty": b_origins.get("missing_empty", 0),
        "C_same_count_replacements": c_origins.get("luna_count_match", 0),
        "C_unsupported_luna_replacements": c_origins.get("luna_unsupported_replacement", 0),
        "C_missing_completion_replacements": c_origins.get("luna_missing_completion", 0),
    }
    return projections, arm_summary, bound_counts


def _private_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _holdout_binding(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "not_available",
            "independent_provenance": "unverified",
            "entries": 0,
            "sha256": "",
            "path": "",
        }
    value = _read_json(path)
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise BundleError("holdout manifest has no entries list")
    # The ledger seals query/entry identities only; it does not establish independent relevance labels.
    return {
        "status": "bound_but_not_ground_truth",
        "independent_provenance": "unverified",
        "entries": len(entries),
        "sha256": _sha256_file(path),
        "schema_version": value.get("schema_version"),
        "path": str(path.resolve()),
    }


def _artifact_binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _candidate_binding(path: Path) -> dict[str, Any]:
    binding = _artifact_binding(path)
    companion = path.parent / "manifest.json"
    if companion.is_file():
        binding["bundle_manifest"] = _artifact_binding(companion)
    return binding


def prepare_bundle(source_path: Path, candidate_path: Path, output: Path, *, holdout_manifest: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise BundleError(f"output already exists: {output}")
    source_rows, candidate_rows = _validate_inputs(source_path, candidate_path)
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    projections, arm_summary, bound_counts = _project_descriptors(source_rows, candidate_rows)
    projection_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in projections).encode("utf-8")
    _private_write(output / "projections.jsonl", projection_bytes)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "source_binding": _artifact_binding(source_path),
        "candidate_binding": _candidate_binding(candidate_path),
        "page_count": len(source_rows),
        "arm_summary": arm_summary,
        "bound_counts": bound_counts,
        "bound_expectations": {
            "B_same_count_replacements": EXPECTED_B_REPLACEMENTS,
            "B_unsupported_preserved": EXPECTED_UNSUPPORTED,
            "B_missing_empty": EXPECTED_MISSING,
            "C_same_count_replacements": EXPECTED_B_REPLACEMENTS,
            "C_unsupported_luna_replacements": EXPECTED_UNSUPPORTED,
            "C_missing_completion_replacements": EXPECTED_MISSING,
        },
        "student_contract": {
            "status": "projection_available",
            "without_questions": "title+snippet exact bytes",
            "with_questions": "title+snippet exact bytes plus Q-lines",
            "body_or_description_added": False,
        },
        "formal_comparison": {
            "status": "not_measured",
            "cases": 0,
            "reason": "independent_source_span_ground_truth_unavailable",
            "independent_provenance": "unverified",
            "holdout": _holdout_binding(holdout_manifest),
        },
        "projection_artifact": {"path": str((output / "projections.jsonl").resolve()), "sha256": _sha256_bytes(projection_bytes), "bytes": len(projection_bytes)},
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _private_write(output / "manifest.json", manifest_bytes)
    _private_write(output / ".gitignore", b"*\n!.gitignore\n")
    return manifest


def _verify_artifact_binding(binding: Mapping[str, Any], label: str) -> None:
    path = Path(str(binding.get("path", "")))
    if not path.is_file() or _sha256_file(path) != binding.get("sha256"):
        raise BundleError(f"{label} artifact hash mismatch")
    if type(binding.get("bytes")) is not int or path.stat().st_size != binding["bytes"]:
        raise BundleError(f"{label} artifact size mismatch")
    companion = binding.get("bundle_manifest")
    if companion is not None:
        if not isinstance(companion, dict):
            raise BundleError(f"{label} companion manifest binding is invalid")
        _verify_artifact_binding(companion, f"{label} companion manifest")


def _verify_formal_comparison(value: Any) -> None:
    if not isinstance(value, dict):
        raise BundleError("formal comparison metadata is missing")
    if set(value) != {"status", "cases", "reason", "independent_provenance", "holdout"}:
        raise BundleError("formal comparison metadata changed")
    if value.get("status") != "not_measured" or value.get("cases") != 0:
        raise BundleError("formal comparison must remain not_measured with zero cases")
    if value.get("reason") != "independent_source_span_ground_truth_unavailable":
        raise BundleError("formal comparison reason changed")
    if value.get("independent_provenance") != "unverified":
        raise BundleError("formal comparison provenance changed")
    holdout = value.get("holdout")
    if not isinstance(holdout, dict):
        raise BundleError("formal comparison holdout binding is missing")
    if set(holdout) - {"status", "independent_provenance", "entries", "sha256", "path", "schema_version"}:
        raise BundleError("holdout binding metadata changed")
    holdout_sha = holdout.get("sha256")
    holdout_path = holdout.get("path")
    if holdout_sha:
        if not isinstance(holdout_path, str) or not holdout_path:
            raise BundleError("holdout binding path is missing")
        path = Path(holdout_path)
        if not path.is_file() or _sha256_file(path) != holdout_sha:
            raise BundleError("holdout manifest hash mismatch")
    elif holdout.get("status") != "not_available" or holdout_path != "":
        raise BundleError("holdout binding is incomplete")
    if holdout.get("independent_provenance") != "unverified":
        raise BundleError("holdout provenance changed")


def verify_bundle(output: Path) -> dict[str, Any]:
    _private_file(output, directory=True)
    _private_file(output / "manifest.json")
    _private_file(output / "projections.jsonl")
    manifest = _read_json(output / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "prepared":
        raise BundleError("unsupported or unprepared P2b manifest")
    source_binding = manifest.get("source_binding")
    candidate_binding = manifest.get("candidate_binding")
    if not isinstance(source_binding, dict) or not isinstance(candidate_binding, dict):
        raise BundleError("manifest is missing artifact bindings")
    source_path = Path(str(source_binding.get("path")))
    candidate_path = Path(str(candidate_binding.get("path")))
    _verify_artifact_binding(source_binding, "source")
    _verify_artifact_binding(candidate_binding, "candidate")
    source_rows, candidate_rows = _validate_inputs(source_path, candidate_path)
    if manifest.get("page_count") != len(source_rows):
        raise BundleError("manifest page count mismatch")
    projection_path = output / "projections.jsonl"
    projection_artifact = manifest.get("projection_artifact")
    if not isinstance(projection_artifact, dict):
        raise BundleError("projection artifact binding is missing")
    if projection_artifact.get("path") != str(projection_path.resolve()):
        raise BundleError("projection artifact path changed")
    if _sha256_file(projection_path) != projection_artifact.get("sha256") or projection_path.stat().st_size != projection_artifact.get("bytes"):
        raise BundleError("projection artifact hash mismatch")
    saved = _read_jsonl(projection_path)
    if len(saved) != len(source_rows):
        raise BundleError("projection row count mismatch")
    expected_projections, expected_summary, expected_bound_counts = _project_descriptors(source_rows, candidate_rows)
    expected_projection_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in expected_projections).encode("utf-8")
    if saved != expected_projections:
        raise BundleError("projection descriptor or metadata mismatch")
    if projection_artifact.get("sha256") != _sha256_bytes(expected_projection_bytes):
        raise BundleError("projection artifact content changed")
    if manifest.get("arm_summary") != expected_summary:
        raise BundleError("arm summary changed")
    if manifest.get("bound_counts") != expected_bound_counts:
        raise BundleError("bound counts changed")
    expected_bound_expectations = {
        "B_same_count_replacements": EXPECTED_B_REPLACEMENTS,
        "B_unsupported_preserved": EXPECTED_UNSUPPORTED,
        "B_missing_empty": EXPECTED_MISSING,
        "C_same_count_replacements": EXPECTED_B_REPLACEMENTS,
        "C_unsupported_luna_replacements": EXPECTED_UNSUPPORTED,
        "C_missing_completion_replacements": EXPECTED_MISSING,
    }
    if manifest.get("bound_expectations") != expected_bound_expectations:
        raise BundleError("bound count expectations changed")
    expected_student = {
        "status": "projection_available",
        "without_questions": "title+snippet exact bytes",
        "with_questions": "title+snippet exact bytes plus Q-lines",
        "body_or_description_added": False,
    }
    if manifest.get("student_contract") != expected_student:
        raise BundleError("student contract changed")
    _verify_formal_comparison(manifest.get("formal_comparison"))
    return {"status": "verified", "page_count": len(source_rows), "arms": list(ARM_NAMES), "formal_comparison": manifest.get("formal_comparison")}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path)
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.verify:
            summary = verify_bundle(args.output)
        else:
            summary = prepare_bundle(args.source_snapshot, args.candidate_bundle, args.output, holdout_manifest=args.holdout_manifest)
    except BundleError as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps({"status": summary.get("status"), "page_count": summary.get("page_count"), "arms": summary.get("arms") or list(summary.get("arm_summary", {})), "formal_comparison": summary.get("formal_comparison", {})}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
