#!/usr/bin/env python3.14
"""Generate a bounded, local-only recall-question candidate bundle.

The bundle is a new, private generation directory.  It never patches pages or
touches a search index, and an existing output directory is rejected so this
first version has no resume semantics to get wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chronovisor.core import ollama as ollama_runtime
from chronovisor.core.frontmatter import parse
from chronovisor.core.index_store import canonical_document_paths
from chronovisor.core.runtime_config import load_ingest_config
from chronovisor.ingest import ingest

SCHEMA = "chronovisor-recall-questions-candidate-v1"
_ROLE = ollama_runtime.INGEST_GENERATION_RUNTIME_ROLE


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
        raise ValueError("output directory must be new; resume is not supported")
    if not output.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    output_resolved = output.resolve(strict=False)
    if _within(output_resolved, source) or _within(source, output_resolved):
        raise ValueError("source and output directories must not overlap")
    output.mkdir(mode=0o700)
    if output.is_symlink() or output.resolve(strict=True) != output_resolved:
        raise ValueError("output directory escaped its requested path")
    gitignore = output / ".gitignore"
    gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")
    gitignore.chmod(0o600)
    return source, output


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
    path.write_bytes(value)
    path.chmod(0o600)


def _write_json(path: Path, value: object) -> None:
    _write_private(path, _canonical_json(value) + b"\n")


def generate_candidates(source_root: Path, output_dir: Path, limit: int) -> dict[str, Any]:
    """Write one fresh candidate bundle and return its final manifest."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    # Bind the route before copying page content, then check the same binding
    # immediately before every model call so a route switch cannot mix cohorts.
    initial_binding = _resolve_local_binding()
    source, output = _prepare_output(source_root, output_dir)
    rows, enumerated_count, excluded_count = _snapshot_rows(source, limit)
    snapshot_bytes = b"".join(
        _canonical_json(row) + b"\n" for row in rows
    )
    snapshot_path = output / "source.jsonl"
    _write_private(snapshot_path, snapshot_bytes)
    snapshot_sha256 = _sha256(snapshot_bytes)

    candidates_path = output / "candidates.jsonl"
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "source_root": str(source),
        "source_snapshot": snapshot_path.name,
        "source_snapshot_sha256": snapshot_sha256,
        "limit": limit,
        "source_enumerated": enumerated_count,
        "source_excluded": excluded_count,
        "selected_pages": len(rows),
        "route_binding": initial_binding,
        "japanese_questions": True,
        "promotion_status": "blocked_until_p1",
        "candidates": candidates_path.name,
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)

    generated = fallback = failed = 0
    try:
        with candidates_path.open("wb") as handle:
            for source_row in rows:
                binding = _resolve_local_binding()
                if binding["sha256"] != initial_binding["sha256"]:
                    raise RuntimeError("ingest route/config changed during candidate generation")
                current_count = source_row["current_question_count"]
                if (
                    source_row["current_question_count_status"] == "known"
                    and isinstance(current_count, int)
                    and current_count in {3, 4, 5}
                ):
                    comparison_arm = "B_same_question_count"
                    requested_count: int | None = current_count
                elif (
                    source_row["current_question_count_status"] == "known"
                    and current_count == 0
                ):
                    comparison_arm = "C_missing_completion"
                    requested_count = None
                elif source_row["current_question_count_status"] == "known":
                    comparison_arm = "B_pending_unsupported_count"
                    requested_count = None
                else:
                    comparison_arm = "C_missing_completion"
                    requested_count = None
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
                        fallback += 1
                    else:
                        result["generation_status"] = "generated"
                        result["generation_success"] = True
                        generated += 1
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
                    failed += 1
                handle.write(_canonical_json(result) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    except Exception as exc:
        manifest.update({"status": "failed", "failure": type(exc).__name__})
        _write_json(manifest_path, manifest)
        raise
    candidates_path.chmod(0o600)
    candidates_sha256 = _sha256(candidates_path.read_bytes())
    manifest.update(
        {
            "status": "complete",
            "candidates_sha256": candidates_sha256,
            "generated": generated,
            "fallback": fallback,
            "failed": failed,
        }
    )
    _write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Frozen page source directory")
    parser.add_argument("--output", type=Path, required=True, help="New private generation directory")
    parser.add_argument("--limit", type=int, required=True, help="Positive page limit")
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("--limit must be positive")
    manifest = generate_candidates(args.source, args.output, args.limit)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
