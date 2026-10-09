#!/usr/bin/env python3.14
"""Generate a bounded, local-only recall-question candidate bundle.

The bundle is a new, private generation directory. It never patches pages or
touches a search index. Resume is explicit and reuses only the frozen snapshot
and a validated candidate prefix.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chronovisor.core import ollama as ollama_runtime
from chronovisor.core.frontmatter import parse
from chronovisor.core.index_store import canonical_document_paths
from chronovisor.core.jsonl_write import atomic_replace_bytes
from chronovisor.core.runtime_config import load_ingest_config
from chronovisor.ingest import ingest

SCHEMA = "chronovisor-recall-questions-candidate-v1"
_ROLE = ollama_runtime.INGEST_GENERATION_RUNTIME_ROLE
_SOURCE_ROW_KEYS = frozenset(
    {
        "page_id",
        "relative_path",
        "title",
        "body",
        "content",
        "source_sha256",
        "current_question_count",
        "current_question_count_status",
    }
)
_RESULT_KEYS = frozenset(
    {
        "page_id",
        "relative_path",
        "source_sha256",
        "current_question_count",
        "current_question_count_status",
        "comparison_arm",
        "requested_question_count",
        "generation_status",
        "generation_success",
        "error",
        "summary",
        "recall_questions",
        "b_status",
    }
)
def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_local_binding() -> dict[str, Any]:
    """Return the exact local route/config used for this candidate run."""

    routes = ollama_runtime.runtime_generation_routes((_ROLE,))
    if len(routes) != 1:
        raise RuntimeError("candidate generation requires one configured ingest route")
    route = routes[0]
    if route.location != "local":
        raise RuntimeError("candidate generation refuses a non-local ingest route")
    binding = {
        "role": route.role,
        "provider": route.provider,
        "model": route.model,
        "location": route.location,
        "protocol": route.protocol,
        "endpoint_sha256": route.endpoint_sha256,
        "revision": route.revision,
        "ingest_config": asdict(load_ingest_config()),
    }
    return {"binding": binding, "sha256": _sha256(_canonical_json(binding))}


def _prepare_output(source_root: Path, output_dir: Path) -> tuple[Path, Path]:
    source = source_root.expanduser()
    if source.is_symlink():
        raise ValueError("source root must not be a symlink")
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise ValueError("source root is unavailable") from exc
    if not source.is_dir():
        raise ValueError("source root must be a directory")

    output = output_dir.expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError("output directory already exists; use --resume for a frozen bundle")
    if not output.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    output_resolved = output.resolve(strict=False)
    if _within(output_resolved, source) or _within(source, output_resolved):
        raise ValueError("source and output directories must not overlap")
    output.mkdir(mode=0o700)
    if output.is_symlink() or output.resolve(strict=True) != output_resolved:
        raise ValueError("output directory escaped its requested path")
    return source, output


def _initialize_output(output: Path) -> None:
    _write_private(output / ".gitignore", b"*\n!.gitignore\n")


def _snapshot_rows(
    source: Path, limit: int
) -> tuple[list[dict[str, Any]], int, int]:
    candidates = tuple(sorted(source.rglob("*.md")))
    paths = canonical_document_paths(source, require_stable=True, strict=False)
    accepted_paths = set(paths)
    excluded_count = sum(path not in accepted_paths for path in candidates)
    selected = paths[:limit]
    rows: list[dict[str, Any]] = []
    seen_page_ids: set[str] = set()
    for path in selected:
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise RuntimeError(f"source page escaped its namespace: {path.name}")
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise RuntimeError(f"source page escaped its namespace: {path.name}")
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise RuntimeError(f"source page changed during snapshot: {path.name}")
        text = raw.decode("utf-8")
        metadata, body = parse(text)
        relative_path = path.relative_to(source).as_posix()
        page_id = relative_path.removesuffix(".md")
        if page_id in seen_page_ids:
            raise ValueError(f"duplicate page id in source snapshot: {page_id}")
        seen_page_ids.add(page_id)
        title_value = metadata.get("title")
        title = title_value if isinstance(title_value, str) and title_value.strip() else path.stem
        questions = metadata.get("recall_questions")
        if isinstance(questions, list) and all(isinstance(value, str) for value in questions):
            question_count = len(questions)
            question_count_status = "known"
        else:
            question_count = None
            question_count_status = "pending"
        rows.append(
            {
                "page_id": page_id,
                "relative_path": relative_path,
                "title": title,
                "body": body,
                "content": text,
                "source_sha256": _sha256(raw),
                "current_question_count": question_count,
                "current_question_count_status": question_count_status,
            }
        )
    if not rows:
        raise ValueError("source snapshot contains no canonical pages")
    return rows, len(candidates), excluded_count


def _write_private(path: Path, value: bytes) -> None:
    atomic_replace_bytes(path, value, mode=0o600)


def _write_json(path: Path, value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must be an object")
    atomic_replace_bytes(path, _canonical_json(value) + b"\n", mode=0o600)


@contextmanager
def _bundle_writer_lock(output_dir: Path):
    """Serialize writers and fail closed when another run owns the bundle."""

    output = output_dir.expanduser()
    if not output.is_dir() or output.is_symlink():
        raise ValueError("candidate bundle directory is unavailable")
    lock_path = output / ".lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise ValueError("candidate bundle lock is unavailable") from exc
    try:
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("candidate bundle is locked by another writer") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _private_file(path: Path, *, directory: bool = False) -> None:
    mode = path.stat().st_mode
    expected = 0o700 if directory else 0o600
    if stat.S_IMODE(mode) != expected or path.is_symlink():
        raise ValueError(f"private artifact has unsafe permissions: {path.name}")


def _read_json(path: Path) -> dict[str, Any]:
    _private_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    _private_file(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read JSONL artifact: {path.name}") from exc
    if not raw:
        return [], raw
    if not raw.endswith(b"\n"):
        raise ValueError(f"JSONL artifact has an incomplete tail: {path.name}")
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line:
            raise ValueError(f"JSONL artifact has a blank row: {path.name}")
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"JSONL artifact has an invalid row: {path.name}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL artifact row must be an object: {path.name}")
        rows.append(value)
    return rows, raw


def _require_sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _comparison_plan(source_row: dict[str, Any]) -> tuple[str, int | None]:
    current_count = source_row["current_question_count"]
    if (
        source_row["current_question_count_status"] == "known"
        and isinstance(current_count, int)
        and current_count in {3, 4, 5}
    ):
        return "B_same_question_count", current_count
    if source_row["current_question_count_status"] == "known" and current_count == 0:
        return "C_missing_completion", None
    if source_row["current_question_count_status"] == "known":
        return "B_pending_unsupported_count", None
    return "C_missing_completion", None


def _validate_source_rows(rows: list[dict[str, Any]], selected_pages: int) -> None:
    if len(rows) != selected_pages or not rows:
        raise ValueError("source snapshot row count does not match manifest")
    seen: set[str] = set()
    for row in rows:
        if set(row) != _SOURCE_ROW_KEYS:
            raise ValueError("source snapshot has unknown or missing fields")
        page_id = row["page_id"]
        relative_path = row["relative_path"]
        title = row["title"]
        body = row["body"]
        content = row["content"]
        if (
            not isinstance(page_id, str)
            or not page_id
            or not isinstance(relative_path, str)
            or not relative_path.endswith(".md")
            or not isinstance(title, str)
            or not isinstance(body, str)
            or not isinstance(content, str)
            or page_id != relative_path.removesuffix(".md")
        ):
            raise ValueError("source snapshot identity is invalid")
        if page_id in seen:
            raise ValueError("source snapshot contains duplicate pages")
        seen.add(page_id)
        if _sha256(content.encode("utf-8")) != _require_sha(
            row["source_sha256"], "source_sha256"
        ):
            raise ValueError("source snapshot content hash mismatch")
        parsed_metadata, parsed_body = parse(content)
        parsed_title = parsed_metadata.get("title")
        expected_title = parsed_title if isinstance(parsed_title, str) and parsed_title.strip() else Path(relative_path).stem
        if parsed_body != body or expected_title != title:
            raise ValueError("source snapshot parsed fields do not match content")
        count = row["current_question_count"]
        count_status = row["current_question_count_status"]
        if count_status not in {"known", "pending"}:
            raise ValueError("source snapshot question count status is invalid")
        if count_status == "known" and (
            not isinstance(count, int) or isinstance(count, bool) or count < 0
        ):
            raise ValueError("source snapshot question count is invalid")
        if count_status == "pending" and count is not None:
            raise ValueError("pending source question count must be null")


def _validate_result_rows(
    source_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]
) -> dict[str, int]:
    if len(result_rows) > len(source_rows):
        raise ValueError("candidate rows exceed frozen source snapshot")
    counts = {"generated": 0, "fallback": 0, "failed": 0}
    seen: set[str] = set()
    for index, row in enumerate(result_rows):
        if set(row) - _RESULT_KEYS:
            raise ValueError("candidate row has unknown fields")
        source = source_rows[index]
        for key in ("page_id", "relative_path", "source_sha256"):
            if row.get(key) != source[key]:
                raise ValueError("candidate rows are not an unchanged source prefix")
        if row.get("current_question_count") != source["current_question_count"]:
            raise ValueError("candidate row changed the frozen question count")
        if row.get("current_question_count_status") != source[
            "current_question_count_status"
        ]:
            raise ValueError("candidate row changed the frozen count status")
        page_id = row["page_id"]
        if page_id in seen:
            raise ValueError("candidate rows contain duplicate pages")
        seen.add(page_id)
        expected_arm, expected_count = _comparison_plan(source)
        if row.get("comparison_arm") != expected_arm or row.get(
            "requested_question_count"
        ) != expected_count:
            raise ValueError("candidate comparison arm changed")
        status = row.get("generation_status")
        success = row.get("generation_success")
        if status not in {"generated", "fallback", "failed"}:
            raise ValueError("candidate generation status is invalid")
        if success is not (status == "generated"):
            raise ValueError("candidate generation success flag is invalid")
        if status == "failed":
            if (
                not isinstance(row.get("error"), str)
                or not row["error"]
                or "summary" in row
                or "recall_questions" in row
            ):
                raise ValueError("failed candidate row contains generated metadata")
            counts["failed"] += 1
        else:
            if not isinstance(row.get("summary"), str) or not row["summary"].strip():
                raise ValueError("candidate summary is missing")
            questions = row.get("recall_questions")
            if not isinstance(questions, list) or not questions or not all(
                isinstance(question, str) and question.strip() for question in questions
            ):
                raise ValueError("candidate questions are invalid")
            counts[status] += 1
        b_status = row.get("b_status")
        if expected_count is None:
            expected_b_status = (
                "pending_unsupported_count"
                if expected_arm == "B_pending_unsupported_count"
                else "pending_missing_completion"
            )
        elif status != "generated":
            expected_b_status = "pending_generation"
        elif isinstance(row["recall_questions"], list) and len(
            row["recall_questions"]
        ) == expected_count:
            expected_b_status = "match"
        else:
            expected_b_status = "mismatch"
        if b_status != expected_b_status:
            raise ValueError("candidate B status is invalid")
    return counts


def _manifest_without_pending(manifest: dict[str, Any]) -> dict[str, Any]:
    clean = dict(manifest)
    for key in (
        "pending_previous_status",
        "pending_previous_count",
        "pending_previous_sha256",
        "pending_count",
        "pending_candidates_sha256",
    ):
        clean.pop(key, None)
    return clean


def _validate_manifest_shape(manifest: dict[str, Any], limit: int) -> None:
    required = {
        "schema",
        "status",
        "source_root",
        "source_snapshot",
        "source_snapshot_sha256",
        "limit",
        "source_enumerated",
        "source_excluded",
        "selected_pages",
        "route_binding",
        "japanese_questions",
        "promotion_status",
        "candidates",
        "processed_count",
        "candidates_sha256",
        "generated",
        "fallback",
        "failed",
    }
    if not required <= set(manifest):
        raise ValueError("manifest is missing required fields")
    if manifest["schema"] != SCHEMA or manifest["limit"] != limit:
        raise ValueError("manifest schema or limit does not match this run")
    if manifest["status"] not in {"running", "partial", "complete", "pending_commit"}:
        raise ValueError("manifest status is invalid")
    if manifest["source_snapshot"] != "source.jsonl" or manifest["candidates"] != "candidates.jsonl":
        raise ValueError("manifest artifact names are invalid")
    if manifest["japanese_questions"] is not True or manifest["promotion_status"] != "blocked_until_p1":
        raise ValueError("manifest promotion contract is invalid")
    for key in (
        "source_snapshot_sha256",
        "candidates_sha256",
    ):
        _require_sha(manifest[key], key)
    integer_fields = (
        "limit",
        "source_enumerated",
        "source_excluded",
        "selected_pages",
        "processed_count",
        "generated",
        "fallback",
        "failed",
    )
    for key in integer_fields:
        value = manifest[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"manifest field is invalid: {key}")
    if manifest["limit"] <= 0 or manifest["selected_pages"] <= 0:
        raise ValueError("manifest page bounds are invalid")
    if manifest["selected_pages"] > manifest["limit"]:
        raise ValueError("manifest selected pages exceed limit")
    if manifest["processed_count"] > manifest["selected_pages"]:
        raise ValueError("manifest processed pages exceed selected pages")
    route_binding = manifest["route_binding"]
    if not isinstance(route_binding, dict) or set(route_binding) != {"binding", "sha256"}:
        raise ValueError("manifest route binding is invalid")
    _require_sha(route_binding["sha256"], "route_binding.sha256")
    if _sha256(_canonical_json(route_binding["binding"])) != route_binding["sha256"]:
        raise ValueError("manifest route binding hash mismatch")
    if manifest["status"] == "pending_commit":
        pending_fields = {
            "pending_previous_status",
            "pending_previous_count",
            "pending_previous_sha256",
            "pending_count",
            "pending_candidates_sha256",
        }
        if not pending_fields <= set(manifest):
            raise ValueError("pending manifest is incomplete")
        if manifest["pending_previous_status"] not in {"running", "partial"}:
            raise ValueError("pending manifest previous status is invalid")
        if manifest["pending_previous_count"] != manifest["processed_count"]:
            raise ValueError("pending manifest previous count is invalid")
        if manifest["pending_count"] != manifest["processed_count"] + 1:
            raise ValueError("pending manifest count is invalid")
        _require_sha(manifest["pending_previous_sha256"], "pending_previous_sha256")
        _require_sha(manifest["pending_candidates_sha256"], "pending_candidates_sha256")
    elif any(
        key in manifest
        for key in (
            "pending_previous_status",
            "pending_previous_count",
            "pending_previous_sha256",
            "pending_count",
            "pending_candidates_sha256",
        )
    ):
        raise ValueError("non-pending manifest has pending fields")


def _manifest_counts(manifest: dict[str, Any], counts: dict[str, int]) -> None:
    if any(manifest[key] != counts[key] for key in ("generated", "fallback", "failed")):
        raise ValueError("manifest counts do not match candidate rows")


def _generate_result(source_row: dict[str, Any]) -> dict[str, Any]:
    comparison_arm, requested_count = _comparison_plan(source_row)
    result: dict[str, Any] = {
        "page_id": source_row["page_id"],
        "relative_path": source_row["relative_path"],
        "source_sha256": source_row["source_sha256"],
        "current_question_count": source_row["current_question_count"],
        "current_question_count_status": source_row["current_question_count_status"],
        "comparison_arm": comparison_arm,
        "requested_question_count": requested_count,
    }
    try:
        generation_kwargs: dict[str, Any] = {
            "japanese_questions": True,
            "require_local_route": True,
        }
        if requested_count is not None:
            generation_kwargs["question_count"] = requested_count
        metadata = ingest._generate_recall_metadata(
            source_row["title"],
            source_row["body"],
            source_row["page_id"],
            **generation_kwargs,
        )
        fallback_metadata = ingest._fallback_recall_metadata(
            source_row["title"], source_row["body"], source_row["page_id"]
        )
        result["summary"] = metadata.get("summary")
        result["recall_questions"] = metadata.get("recall_questions")
        if metadata == fallback_metadata:
            result["generation_status"] = "fallback"
            result["generation_success"] = False
        else:
            result["generation_status"] = "generated"
            result["generation_success"] = True
        generated_questions = result["recall_questions"]
        if requested_count is None:
            result["b_status"] = (
                "pending_unsupported_count"
                if comparison_arm == "B_pending_unsupported_count"
                else "pending_missing_completion"
            )
        elif result["generation_status"] != "generated":
            result["b_status"] = "pending_generation"
        elif (
            isinstance(generated_questions, list)
            and len(generated_questions) == requested_count
        ):
            result["b_status"] = "match"
        else:
            result["b_status"] = "mismatch"
    except Exception as exc:
        result["generation_status"] = "failed"
        result["generation_success"] = False
        result["error"] = type(exc).__name__
        result["b_status"] = (
            "pending_generation"
            if requested_count is not None
            else (
                "pending_unsupported_count"
                if comparison_arm == "B_pending_unsupported_count"
                else "pending_missing_completion"
            )
        )
    return result


def _commit_result(
    manifest: dict[str, Any],
    manifest_path: Path,
    candidates_path: Path,
    source_row: dict[str, Any],
    result: dict[str, Any],
    *,
    next_hash: str,
) -> dict[str, Any]:
    row_bytes = _canonical_json(result) + b"\n"
    row_counts = _validate_result_rows([source_row], [result])
    counts = {
        key: manifest[key] + row_counts[key]
        for key in ("generated", "fallback", "failed")
    }
    _require_sha(next_hash, "next_candidates_sha256")
    pending = dict(manifest)
    pending.update(
        {
            "status": "pending_commit",
            "pending_previous_status": manifest["status"],
            "pending_previous_count": manifest["processed_count"],
            "pending_previous_sha256": manifest["candidates_sha256"],
            "pending_count": manifest["processed_count"] + 1,
            "pending_candidates_sha256": next_hash,
        }
    )
    _write_json(manifest_path, pending)
    with candidates_path.open("ab") as handle:
        handle.write(row_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    final = _manifest_without_pending(pending)
    final.update(
        {
            "status": (
                "complete"
                if manifest["processed_count"] + 1 == manifest["selected_pages"]
                else "partial"
            ),
            "processed_count": manifest["processed_count"] + 1,
            "candidates_sha256": next_hash,
            **counts,
        }
    )
    _write_json(manifest_path, final)
    return final


def _run_batch(
    manifest: dict[str, Any],
    manifest_path: Path,
    candidates_path: Path,
    source_rows: list[dict[str, Any]],
    max_pages: int,
) -> dict[str, Any]:
    existing_rows, existing_bytes = _read_jsonl(candidates_path)
    candidate_hash = _sha256(existing_bytes)
    if candidate_hash != manifest["candidates_sha256"]:
        raise ValueError("candidate prefix hash does not match manifest")
    _manifest_counts(manifest, _validate_result_rows(source_rows, existing_rows))
    if len(existing_rows) != manifest["processed_count"]:
        raise ValueError("candidate row count does not match manifest")
    candidate_digest = hashlib.sha256(existing_bytes)
    start = len(existing_rows)
    for source_row in source_rows[start : start + max_pages]:
        binding = _resolve_local_binding()
        if binding["sha256"] != manifest["route_binding"]["sha256"]:
            raise RuntimeError("ingest route/config changed during candidate generation")
        result = _generate_result(source_row)
        row_bytes = _canonical_json(result) + b"\n"
        next_digest = candidate_digest.copy()
        next_digest.update(row_bytes)
        next_hash = next_digest.hexdigest()
        manifest = _commit_result(
            manifest,
            manifest_path,
            candidates_path,
            source_row,
            result,
            next_hash=next_hash,
        )
        candidate_digest = next_digest
        candidate_hash = next_hash
    return manifest


def _load_resume_bundle(
    source_root: Path,
    output_dir: Path,
    limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    output = output_dir.expanduser()
    if not output.is_dir() or output.is_symlink():
        raise ValueError("resume output directory is unavailable")
    _private_file(output, directory=True)
    manifest_path = output / "manifest.json"
    snapshot_path = output / "source.jsonl"
    candidates_path = output / "candidates.jsonl"
    gitignore_path = output / ".gitignore"
    if not gitignore_path.is_file() or gitignore_path.read_text(encoding="utf-8") != "*\n!.gitignore\n":
        raise ValueError("resume output gitignore is missing or modified")
    _private_file(gitignore_path)
    manifest = _read_json(manifest_path)
    _validate_manifest_shape(manifest, limit)
    requested_source_path = source_root.expanduser()
    if requested_source_path.is_symlink():
        raise ValueError("resume source root must not be a symlink")
    requested_source = requested_source_path.resolve(strict=False)
    if manifest["source_root"] != str(requested_source):
        raise ValueError("resume source path does not match frozen manifest")
    manifest_source = Path(manifest["source_root"])
    output_resolved = output.resolve(strict=True)
    if _within(output_resolved, manifest_source) or _within(manifest_source, output_resolved):
        raise ValueError("source and output directories must not overlap")
    snapshot_rows, snapshot_bytes = _read_jsonl(snapshot_path)
    if _sha256(snapshot_bytes) != manifest["source_snapshot_sha256"]:
        raise ValueError("source snapshot hash does not match manifest")
    _validate_source_rows(snapshot_rows, manifest["selected_pages"])
    candidate_rows, candidate_bytes = _read_jsonl(candidates_path)
    candidate_hash = _sha256(candidate_bytes)
    if manifest["status"] != "pending_commit" and candidate_hash != manifest[
        "candidates_sha256"
    ]:
        raise ValueError("candidate artifact hash does not match manifest")
    if manifest["status"] == "pending_commit" and candidate_hash not in {
        manifest["pending_previous_sha256"],
        manifest["pending_candidates_sha256"],
    }:
        raise ValueError("pending candidate commit is neither durable nor empty")
    candidate_counts = _validate_result_rows(snapshot_rows, candidate_rows)
    if manifest["status"] == "pending_commit":
        previous_hash = manifest["pending_previous_sha256"]
        pending_hash = manifest["pending_candidates_sha256"]
        previous_count = manifest["pending_previous_count"]
        pending_count = manifest["pending_count"]
        if candidate_hash == pending_hash and len(candidate_rows) == pending_count:
            manifest = _manifest_without_pending(manifest)
            manifest.update(
                {
                    "status": (
                        "complete"
                        if pending_count == manifest["selected_pages"]
                        else "partial"
                    ),
                    "processed_count": pending_count,
                    "candidates_sha256": candidate_hash,
                    **candidate_counts,
                }
            )
            _write_json(manifest_path, manifest)
        elif candidate_hash == previous_hash and len(candidate_rows) == previous_count:
            previous_status = manifest["pending_previous_status"]
            manifest = _manifest_without_pending(manifest)
            manifest["status"] = previous_status
            manifest["candidates_sha256"] = candidate_hash
            _write_json(manifest_path, manifest)
        else:
            raise ValueError("pending candidate commit is neither durable nor empty")
    else:
        if len(candidate_rows) != manifest["processed_count"]:
            raise ValueError("candidate row count does not match manifest")
        _manifest_counts(manifest, candidate_counts)
    if manifest["status"] == "complete" and manifest["processed_count"] != manifest["selected_pages"]:
        raise ValueError("complete manifest has unprocessed pages")
    if manifest["status"] != "complete" and manifest["processed_count"] >= manifest["selected_pages"]:
        raise ValueError("partial manifest has no remaining pages")
    current_binding = _resolve_local_binding()
    if current_binding["sha256"] != manifest["route_binding"]["sha256"]:
        raise RuntimeError("current ingest route/config does not match frozen manifest")
    return manifest, snapshot_rows, manifest_path, candidates_path


def _fresh_bundle(
    source: Path,
    output: Path,
    limit: int,
    initial_binding: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    rows, enumerated_count, excluded_count = _snapshot_rows(source, limit)
    snapshot_bytes = b"".join(_canonical_json(row) + b"\n" for row in rows)
    snapshot_path = output / "source.jsonl"
    _write_private(snapshot_path, snapshot_bytes)
    candidates_path = output / "candidates.jsonl"
    candidates_path.touch(mode=0o600, exist_ok=False)
    candidates_path.chmod(0o600)
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "source_root": str(source),
        "source_snapshot": snapshot_path.name,
        "source_snapshot_sha256": _sha256(snapshot_bytes),
        "limit": limit,
        "source_enumerated": enumerated_count,
        "source_excluded": excluded_count,
        "selected_pages": len(rows),
        "route_binding": initial_binding,
        "japanese_questions": True,
        "promotion_status": "blocked_until_p1",
        "candidates": candidates_path.name,
        "processed_count": 0,
        "candidates_sha256": _sha256(b""),
        "generated": 0,
        "fallback": 0,
        "failed": 0,
    }
    _write_json(manifest_path, manifest)
    return manifest, rows, manifest_path, candidates_path


def generate_candidates(
    source_root: Path,
    output_dir: Path,
    limit: int,
    *,
    max_pages: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate one bounded batch or resume one explicit frozen bundle."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive")
    effective_max_pages = min(max_pages or limit, limit)
    if resume:
        with _bundle_writer_lock(output_dir):
            manifest, rows, manifest_path, candidates_path = _load_resume_bundle(
                source_root, output_dir, limit
            )
            if manifest["status"] == "complete":
                return manifest
            return _run_batch(
                manifest,
                manifest_path,
                candidates_path,
                rows,
                effective_max_pages,
            )
    else:
        # Bind the route before copying page content, then check the same
        # binding immediately before every model call.
        initial_binding = _resolve_local_binding()
        source, output = _prepare_output(source_root, output_dir)
        with _bundle_writer_lock(output):
            _initialize_output(output)
            manifest, rows, manifest_path, candidates_path = _fresh_bundle(
                source, output, limit, initial_binding
            )
            return _run_batch(
                manifest,
                manifest_path,
                candidates_path,
                rows,
                effective_max_pages,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Frozen page source directory")
    parser.add_argument("--output", type=Path, required=True, help="New private generation directory")
    parser.add_argument("--limit", type=int, required=True, help="Positive page limit")
    parser.add_argument("--max-pages", type=int, help="Positive per-run page bound")
    parser.add_argument("--resume", action="store_true", help="Resume this exact frozen bundle")
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_pages is not None and args.max_pages <= 0:
        parser.error("--max-pages must be positive")
    manifest = generate_candidates(
        args.source,
        args.output,
        args.limit,
        max_pages=args.max_pages,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "status",
                    "processed_count",
                    "selected_pages",
                    "generated",
                    "fallback",
                    "failed",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
