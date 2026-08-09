"""Production-owned library evidence indexes and gold-free providers."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from chronovisor.classification.classification import (
    ClassificationError,
    UDCPackage,
    load_udc_package,
)
from chronovisor.classification.classification_embedding_worker import (
    SCHEMA as EMBEDDING_WORKER_SCHEMA,
)
from chronovisor.classification.classification_engine import (
    DEFAULT_CANDIDATE_LIMIT,
    CandidateIndex,
)
from chronovisor.classification.classification_fixture_set import (
    INFERENCE_DTO_SCHEMA,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)
from chronovisor.classification.classification_library_sources import (
    EXTERNAL_PACKAGE_SCHEMA,
)
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.research_scheduler import (
    research_lane,
    run_cancellable_command,
    sync_pending,
)
from chronovisor.core.runtime_config import load_embedding_config
from chronovisor.core.timeutil import utc_iso_milliseconds as _now

INDEX_SCHEMA = "chronovisor.library-evidence-index.v1"
EVIDENCE_SCHEMA = "chronovisor.classification-library-evidence.v1"
PROVIDER_RESULT_SCHEMA = "chronovisor.classification-provider-result.v1"
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*|[\u3040-\u30ff\u3400-\u9fff]{2,}")
COMPOSITE_UDC_RE = re.compile(r"[:/+=\"'`()\\[\\]]")
MAX_WORKING_SET_BYTES = 3 * 1024**3
MAX_BUILD_PEAK_BYTES = 6 * 1024**3
MIN_FREE_BYTES = 8 * 1024**3
DENSE_BUILD_SCHEMA = "chronovisor.library-evidence-dense-build.v1"
EXTERNAL_TEST_CASE_SCHEMA = "chronovisor.external-library-test-case.v1"
DENSE_MODEL_LICENSE = "MIT (as declared by the installed Ollama bge-m3 artifact)"
DENSE_TRAINING_CORPUS_LICENSE = (
    "model-card-described heterogeneous corpus; constituent licenses not "
    "fully enumerated by the installed Ollama artifact"
)




def _tokens(value: str) -> list[str]:
    return sorted(
        {
            match.group(0).casefold()
            for match in TOKEN_RE.finditer(value)
            if len(match.group(0)) >= 2
        }
    )


def split_for_group(group_id: str, *, test_percent: int = 20) -> str:
    if not 1 <= test_percent <= 50:
        raise ClassificationError("external test percentage is unsafe")
    value = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest(), 16) % 100
    return "test" if value < test_percent else "train"


def resource_preflight(
    target: Path,
    *,
    projected_peak_bytes: int,
    projected_working_bytes: int,
) -> dict[str, Any]:
    usage = shutil.disk_usage(
        target.parent if target.parent.exists() else target.anchor
    )
    required = max(MIN_FREE_BYTES, projected_peak_bytes * 2)
    gates = {
        "free_space": usage.free >= required,
        "projected_peak": projected_peak_bytes <= MAX_BUILD_PEAK_BYTES,
        "projected_working": projected_working_bytes <= MAX_WORKING_SET_BYTES,
    }
    return {
        "status": "ready" if all(gates.values()) else "blocked",
        "gates": gates,
        "free_bytes": usage.free,
        "required_free_bytes": required,
        "projected_peak_bytes": projected_peak_bytes,
        "projected_working_bytes": projected_working_bytes,
    }


def embed_texts_cancellable(
    texts: Sequence[str],
    *,
    purpose: str = "explicit",
    timeout_seconds: float = 600,
) -> tuple[str, list[list[float]]]:
    """Embed in an isolated research worker and retry foreground preemption.

    Cancellation is a scheduler event, not a failed classification attempt.
    The batch remains unchanged and is retried only after the foreground
    marker has cleared.
    """

    payload = {
        "schema": EMBEDDING_WORKER_SCHEMA,
        "model": load_embedding_config().model,
        "texts": list(texts),
        "read_timeout_ms": round(timeout_seconds * 1_000),
    }
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ClassificationError(
                "classification embedding remained preempted past its deadline"
            )
        run_id = f"classification-embed-{uuid.uuid4().hex}"
        with research_lane(
            run_id,
            enabled=True,
            mode="on" if purpose == "explicit" else "shadow",
            purpose=purpose,
            needs_model=True,
        ) as lease:
            result = run_cancellable_command(
                [
                    sys.executable,
                    "-m",
                    "chronovisor.classification.classification_embedding_worker",
                ],
                json.dumps(payload, ensure_ascii=False),
                lease,
                timeout_seconds=remaining,
            )
        if result.status in {"cancelled", "deferred"}:
            while sync_pending():
                if time.monotonic() >= deadline:
                    raise ClassificationError(
                        "classification embedding foreground wait exceeded deadline"
                    )
                time.sleep(0.05)
            time.sleep(0.05)
            continue
        if result.status != "completed" or not isinstance(result.value, Mapping):
            raise ClassificationError(
                result.error or "classification embedding worker failed"
            )
        vectors = result.value.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ClassificationError("classification embedding result is invalid")
        normalized = [
            [float(value) for value in vector]
            for vector in vectors
            if isinstance(vector, list)
        ]
        if len(normalized) != len(texts):
            raise ClassificationError("classification embedding vector is invalid")
        return str(result.value.get("model") or payload["model"]), normalized


def _safe_udc_notation(value: str, package: UDCPackage) -> str | None:
    notation = value.strip()
    if not notation or COMPOSITE_UDC_RE.search(notation):
        return None
    return notation if package.by_notation(notation) is not None else None


def _subject_text(row: Mapping[str, Any]) -> str:
    labels = []
    for subject in row.get("subject_headings") or []:
        if not isinstance(subject, Mapping):
            continue
        labels.append(str(subject.get("pref_label") or ""))
        labels.extend(str(value) for value in subject.get("alt_labels") or [])
    return " ".join(value for value in labels if value)


def _support_rows(
    row: Mapping[str, Any],
    *,
    package_name: str,
    package_sha256: str,
    package: UDCPackage,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    supports: list[dict[str, Any]] = []
    vocabulary: list[dict[str, Any]] = []
    rejects: list[str] = []
    group_id = str(
        row.get("source_group_id")
        or row.get("source_record_id")
        or row.get("record_sha256")
        or ""
    )
    if not group_id:
        return [], [], ["missing_source_group"]
    split = split_for_group(group_id)
    title = str(row.get("title") or "")
    subjects = _subject_text(row)
    source_name = str(row.get("rights_ref") or package_name)
    assignments = row.get("source_assignments") or []
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            continue
        role = str(assignment.get("role") or "")
        source_field = str(assignment.get("source_field") or "")
        if role == "bibliographic_assignment":
            if source_field != "080":
                rejects.append("bibliographic_non_080")
                continue
            arm_texts = {"B1a": title, "B1b": f"{title} {subjects}".strip()}
        elif role == "authority_representative_classification":
            if source_field != "089$a":
                rejects.append("authority_non_089a")
                continue
            arm_texts = {"B3": f"{title} {subjects}".strip()}
        else:
            rejects.append("unsupported_assignment_role")
            continue
        notation = _safe_udc_notation(
            str(assignment.get("notation_or_uri") or ""), package
        )
        if not notation:
            rejects.append("unresolvable_or_composite_udc")
            continue
        for arm, text in arm_texts.items():
            if not text:
                rejects.append("empty_surrogate")
                continue
            supports.append(
                {
                    "package_name": package_name,
                    "package_sha256": package_sha256,
                    "source_name": source_name,
                    "source_record_id": str(row.get("source_record_id") or ""),
                    "record_sha256": str(row.get("record_sha256") or ""),
                    "group_id": group_id,
                    "split": split,
                    "arm": arm,
                    "text": text,
                    "notation": notation,
                    "source_field": source_field,
                    "generation_method": str(
                        assignment.get("generation_method") or "unknown"
                    ),
                    "intellectual_assignment": str(
                        assignment.get("intellectual_assignment") or "unconfirmed"
                    ),
                }
            )
    vocabulary_eligible = source_name in {
        "ndlsh-authority",
        "czech-topical-authorities",
        "ndl-created-bibliography",
    }
    for subject in (row.get("subject_headings") or []) if vocabulary_eligible else []:
        if not isinstance(subject, Mapping):
            continue
        label = str(subject.get("pref_label") or "")
        alt = [str(value) for value in subject.get("alt_labels") or []]
        if label:
            vocabulary.append(
                {
                    "package_name": package_name,
                    "package_sha256": package_sha256,
                    "source_name": source_name,
                    "source_record_id": str(row.get("source_record_id") or ""),
                    "record_sha256": str(row.get("record_sha256") or ""),
                    "group_id": group_id,
                    "split": split,
                    "label": label,
                    "alt_labels": alt,
                    "relations": list(row.get("diagnostic_relations") or []),
                    "direct_udc_vote": False,
                    "vocabulary_role": (
                        "C1"
                        if source_name == "ndlsh-authority"
                        else "C2"
                        if source_name == "ndl-created-bibliography"
                        else "B2"
                    ),
                }
            )
    return supports, vocabulary, rejects


def external_test_cases(
    *,
    package_manifest_paths: Sequence[Path],
    package: UDCPackage,
    arms: Sequence[str],
) -> list[dict[str, Any]]:
    """Build gold-isolated cases from source groups excluded before indexing.

    Only observed UDC assignments from the deterministic ``test`` split become
    evaluator labels.  The corresponding provider DTO is produced later through
    ``inference_rows`` so no assignment or expected notation crosses the
    provider boundary.
    """

    allowed_support_arms = {"B1a"} if tuple(arms) == ("B1a",) else {"B1b"}
    if "B3" in arms:
        allowed_support_arms.add("B3")
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for manifest_path in package_manifest_paths:
        manifest = read_sealed_json(manifest_path)
        if manifest.get("schema") != EXTERNAL_PACKAGE_SCHEMA:
            raise ClassificationError("unsupported external package manifest")
        records_path = Path(str(manifest.get("records_path") or ""))
        if not records_path.is_file() or sha256_file(records_path) != manifest.get(
            "package_sha256"
        ):
            raise ClassificationError("external package checksum mismatch")
        package_name = str(manifest.get("source_name") or "")
        package_sha256 = str(manifest.get("package_sha256") or "")
        for source in read_jsonl(records_path):
            supports, _vocabulary, _rejects = _support_rows(
                source,
                package_name=package_name,
                package_sha256=package_sha256,
                package=package,
            )
            for support in supports:
                if (
                    support.get("split") != "test"
                    or support.get("arm") not in allowed_support_arms
                ):
                    continue
                key = (
                    package_name,
                    str(support["group_id"]),
                    str(support["arm"]),
                )
                entry = grouped.setdefault(
                    key,
                    {
                        "source": source,
                        "support": support,
                        "notations": set(),
                    },
                )
                entry["notations"].add(str(support["notation"]))

    rows = []
    for (package_name, group_id, arm), entry in sorted(grouped.items()):
        source = entry["source"]
        support = entry["support"]
        allowed = sorted(entry["notations"])
        uid = (
            "external-test-"
            + hashlib.sha256(f"{package_name}:{group_id}:{arm}".encode()).hexdigest()[
                :24
            ]
        )
        rows.append(
            {
                "schema": EXTERNAL_TEST_CASE_SCHEMA,
                "uid": uid,
                "source_sha256": str(source.get("record_sha256") or ""),
                "title": str(support.get("text") or ""),
                "summary": "",
                "excerpt": "",
                "tags": [],
                "raw_keywords": [],
                "page_type": "reference",
                "lifecycle": "active",
                "language": str(source.get("language") or ""),
                "sensitivity": "normal",
                "fixture_group_id": f"external:{package_name}:{group_id}",
                "external_source": package_name,
                "external_arm": arm,
                "external_major_class": str(
                    source.get("major_class")
                    or (allowed[0].split(".", 1)[0] if allowed else "")
                ),
                "external_year_bucket": str(source.get("year_bucket") or "unknown"),
                "external_assignment_count": len(allowed),
                "gold_primary_notation": allowed[0],
                "gold_allowed_primary_notations": allowed,
                "gold_expected_status": "proposed",
            }
        )
    return rows


def build_source_index(
    *,
    package_manifest_paths: Sequence[Path],
    output_path: Path,
    root: Path,
    dense_limit: int = 25_000,
) -> dict[str, Any]:
    package = load_udc_package(root)
    projected_source = sum(
        Path(str(read_sealed_json(path).get("records_path") or "")).stat().st_size
        for path in package_manifest_paths
    )
    preflight = resource_preflight(
        output_path,
        projected_peak_bytes=min(
            MAX_BUILD_PEAK_BYTES + 1,
            max(32 * 1024**2, projected_source * 5),
        ),
        projected_working_bytes=min(
            MAX_WORKING_SET_BYTES + 1,
            max(16 * 1024**2, projected_source * 2),
        ),
    )
    if preflight["status"] != "ready":
        raise ClassificationError(f"library index resource gate failed: {preflight}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    connection = sqlite3.connect(output_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE support (
          row_id INTEGER PRIMARY KEY,
          package_name TEXT NOT NULL,
          package_sha256 TEXT NOT NULL,
          source_name TEXT NOT NULL,
          source_record_id TEXT NOT NULL,
          record_sha256 TEXT NOT NULL,
          group_id TEXT NOT NULL,
          split TEXT NOT NULL,
          arm TEXT NOT NULL,
          text TEXT NOT NULL,
          notation TEXT NOT NULL,
          source_field TEXT NOT NULL,
          generation_method TEXT NOT NULL,
          intellectual_assignment TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE VIRTUAL TABLE support_fts USING fts5(text, content='support', content_rowid='row_id')"
    )
    connection.execute(
        """
        CREATE TABLE vocabulary (
          row_id INTEGER PRIMARY KEY,
          package_name TEXT NOT NULL,
          package_sha256 TEXT NOT NULL,
          source_name TEXT NOT NULL,
          source_record_id TEXT NOT NULL,
          record_sha256 TEXT NOT NULL,
          group_id TEXT NOT NULL,
          split TEXT NOT NULL,
          label TEXT NOT NULL,
          alt_labels_json TEXT NOT NULL,
          relations_json TEXT NOT NULL,
          direct_udc_vote INTEGER NOT NULL,
          vocabulary_role TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE VIRTUAL TABLE vocabulary_fts USING fts5(label, alt_labels, content='')"
    )
    reject_counts: dict[str, int] = defaultdict(int)
    package_digests = []
    support_count = 0
    vocabulary_count = 0
    try:
        for manifest_path in package_manifest_paths:
            manifest = read_sealed_json(manifest_path)
            if manifest.get("schema") != EXTERNAL_PACKAGE_SCHEMA:
                raise ClassificationError("unsupported external package manifest")
            records_path = Path(str(manifest.get("records_path") or ""))
            if not records_path.is_file():
                raise ClassificationError("external package records are missing")
            if sha256_file(records_path) != manifest.get("package_sha256"):
                raise ClassificationError("external package checksum mismatch")
            package_name = str(manifest["source_name"])
            package_sha256 = str(manifest["package_sha256"])
            package_digests.append(package_sha256)
            for row in read_jsonl(records_path):
                supports, vocabulary, rejects = _support_rows(
                    row,
                    package_name=package_name,
                    package_sha256=package_sha256,
                    package=package,
                )
                for reason in rejects:
                    reject_counts[reason] += 1
                for support in supports:
                    if support["split"] != "train":
                        continue
                    cursor = connection.execute(
                        """
                        INSERT INTO support (
                          package_name, package_sha256, source_name,
                          source_record_id, record_sha256, group_id, split, arm,
                          text, notation, source_field, generation_method,
                          intellectual_assignment
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            support["package_name"],
                            support["package_sha256"],
                            support["source_name"],
                            support["source_record_id"],
                            support["record_sha256"],
                            support["group_id"],
                            support["split"],
                            support["arm"],
                            support["text"],
                            support["notation"],
                            support["source_field"],
                            support["generation_method"],
                            support["intellectual_assignment"],
                        ),
                    )
                    connection.execute(
                        "INSERT INTO support_fts(rowid, text) VALUES (?, ?)",
                        (cursor.lastrowid, support["text"]),
                    )
                    support_count += 1
                for entry in vocabulary:
                    if entry["split"] != "train":
                        continue
                    cursor = connection.execute(
                        """
                        INSERT INTO vocabulary (
                          package_name, package_sha256, source_name,
                          source_record_id, record_sha256, group_id, split,
                          label, alt_labels_json, relations_json,
                          direct_udc_vote, vocabulary_role
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            entry["package_name"],
                            entry["package_sha256"],
                            entry["source_name"],
                            entry["source_record_id"],
                            entry["record_sha256"],
                            entry["group_id"],
                            entry["split"],
                            entry["label"],
                            json.dumps(entry["alt_labels"], ensure_ascii=False),
                            json.dumps(entry["relations"], ensure_ascii=False),
                            0,
                            entry["vocabulary_role"],
                        ),
                    )
                    connection.execute(
                        "INSERT INTO vocabulary_fts(rowid, label, alt_labels) VALUES (?, ?, ?)",
                        (
                            cursor.lastrowid,
                            entry["label"],
                            " ".join(entry["alt_labels"]),
                        ),
                    )
                    vocabulary_count += 1
        connection.commit()
    finally:
        connection.close()
    working_bytes = output_path.stat().st_size
    if working_bytes > MAX_WORKING_SET_BYTES:
        output_path.unlink(missing_ok=True)
        raise ClassificationError("library evidence working set exceeded 3 GiB")
    manifest = {
        "schema": INDEX_SCHEMA,
        "created_at": _now(),
        "index_path": str(output_path),
        "index_sha256": sha256_file(output_path),
        "package_manifest_paths": [str(path) for path in package_manifest_paths],
        "package_sha256": sorted(package_digests),
        "split_policy": "sha256(source-group)%100; test<20; train-only-index",
        "support_count": support_count,
        "vocabulary_count": vocabulary_count,
        "dense_limit": dense_limit,
        "dense_index_built": False,
        "rejected_counts_by_reason": dict(sorted(reject_counts.items())),
        "resource_preflight": preflight,
        "working_set_bytes": working_bytes,
        "working_set_gate": working_bytes <= MAX_WORKING_SET_BYTES,
    }
    write_sealed_json(output_path.with_suffix(".manifest.json"), manifest, backup=True)
    return manifest


def build_dense_index(
    manifest_path: Path,
    *,
    batch_size: int = 64,
    purpose: str = "explicit",
) -> dict[str, Any]:
    """Build or resume the bounded train-only bge-m3 vector matrix."""

    manifest = read_sealed_json(manifest_path)
    if manifest.get("schema") != INDEX_SCHEMA:
        raise ClassificationError("unsupported library evidence index")
    index_path = Path(str(manifest.get("index_path") or ""))
    if not index_path.is_file() or sha256_file(index_path) != manifest.get(
        "index_sha256"
    ):
        raise ClassificationError("library evidence index checksum mismatch")
    if manifest.get("dense_index_built") is True:
        vectors_path = Path(str(manifest.get("dense_vectors_path") or ""))
        row_ids_path = Path(str(manifest.get("dense_row_ids_path") or ""))
        if (
            vectors_path.is_file()
            and row_ids_path.is_file()
            and sha256_file(vectors_path) == manifest.get("dense_vectors_sha256")
            and sha256_file(row_ids_path) == manifest.get("dense_row_ids_sha256")
        ):
            return manifest
        raise ClassificationError("dense index manifest is corrupt")

    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        selected = connection.execute(
            """
            SELECT row_id, text
            FROM support
            WHERE split='train'
            ORDER BY group_id, source_record_id, arm, notation, row_id
            LIMIT ?
            """,
            (max(0, int(manifest.get("dense_limit") or 0)),),
        ).fetchall()
    finally:
        connection.close()
    row_ids = np.asarray([int(row[0]) for row in selected], dtype=np.int64)
    texts = [str(row[1]) for row in selected]
    selection_sha256 = sha256_bytes(
        json.dumps(
            [
                {"row_id": int(row_id), "text_sha256": sha256_bytes(text.encode())}
                for row_id, text in zip(row_ids.tolist(), texts, strict=True)
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    vectors_path = index_path.with_suffix(".dense-vectors.npy")
    row_ids_path = index_path.with_suffix(".dense-row-ids.npy")
    progress_path = index_path.with_suffix(".dense-progress.json")
    progress = (
        read_sealed_json(progress_path)
        if progress_path.exists()
        else {
            "schema": DENSE_BUILD_SCHEMA,
            "status": "building",
            "selection_sha256": selection_sha256,
            "target_count": len(texts),
            "completed_count": 0,
            "dimensions": 0,
            "model": "",
            "cancelled_batches_requeued": 0,
            "attempts_consumed_by_preemption": 0,
        }
    )
    if (
        progress.get("schema") != DENSE_BUILD_SCHEMA
        or progress.get("selection_sha256") != selection_sha256
        or int(progress.get("target_count") or 0) != len(texts)
    ):
        raise ClassificationError("dense build checkpoint does not match source rows")
    completed = int(progress.get("completed_count") or 0)
    dimensions = int(progress.get("dimensions") or 0)
    if not 0 <= completed <= len(texts):
        raise ClassificationError("dense build checkpoint count is invalid")

    if not texts:
        np.save(row_ids_path, row_ids, allow_pickle=False)
        manifest.update(
            {
                "dense_index_built": True,
                "dense_model": load_embedding_config().model,
                "dense_model_license": DENSE_MODEL_LICENSE,
                "dense_training_corpus_license": DENSE_TRAINING_CORPUS_LICENSE,
                "dense_count": 0,
                "dense_dimensions": 0,
                "dense_selection_sha256": selection_sha256,
                "dense_vectors_path": "",
                "dense_vectors_sha256": "",
                "dense_row_ids_path": str(row_ids_path),
                "dense_row_ids_sha256": sha256_file(row_ids_path),
            }
        )
        projected_peak = int(
            (manifest.get("resource_preflight") or {}).get("projected_peak_bytes") or 0
        )
        manifest["build_peak_bound_bytes"] = max(
            int(manifest.get("working_set_bytes") or 0),
            projected_peak,
        )
        manifest["build_peak_gate"] = (
            manifest["build_peak_bound_bytes"] <= MAX_BUILD_PEAK_BYTES
        )
        write_sealed_json(manifest_path, manifest, backup=True)
        return manifest

    matrix: np.memmap | None = None
    if completed:
        if not vectors_path.is_file() or not row_ids_path.is_file():
            raise ClassificationError("dense build checkpoint files are missing")
        stored_row_ids = np.load(row_ids_path, allow_pickle=False)
        if not np.array_equal(stored_row_ids, row_ids):
            raise ClassificationError("dense build row IDs changed")
        matrix = np.lib.format.open_memmap(vectors_path, mode="r+")
        if matrix.shape != (len(texts), dimensions):
            raise ClassificationError("dense build vector shape changed")
    while completed < len(texts):
        end = min(len(texts), completed + max(1, batch_size))
        model, vectors = embed_texts_cancellable(
            texts[completed:end],
            purpose=purpose,
        )
        batch = np.asarray(vectors, dtype=np.float32)
        if batch.ndim != 2 or batch.shape[0] != end - completed:
            raise ClassificationError("dense embedding batch has invalid shape")
        if matrix is None:
            dimensions = int(batch.shape[1])
            if dimensions < 1:
                raise ClassificationError("dense embedding dimensions are empty")
            matrix = np.lib.format.open_memmap(
                vectors_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(texts), dimensions),
            )
            np.save(row_ids_path, row_ids, allow_pickle=False)
        if int(batch.shape[1]) != dimensions:
            raise ClassificationError("dense embedding dimensions changed")
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        batch = np.divide(
            batch,
            norms,
            out=np.zeros_like(batch),
            where=norms != 0,
        )
        matrix[completed:end] = batch
        matrix.flush()
        completed = end
        progress.update(
            {
                "status": "building" if completed < len(texts) else "complete",
                "completed_count": completed,
                "dimensions": dimensions,
                "model": model,
                "vectors_path": str(vectors_path),
                "row_ids_path": str(row_ids_path),
            }
        )
        write_sealed_json(progress_path, progress, backup=True)
    del matrix

    manifest.update(
        {
            "dense_index_built": True,
            "dense_model": str(progress.get("model") or ""),
            "dense_model_license": DENSE_MODEL_LICENSE,
            "dense_training_corpus_license": DENSE_TRAINING_CORPUS_LICENSE,
            "dense_count": len(texts),
            "dense_dimensions": dimensions,
            "dense_selection_sha256": selection_sha256,
            "dense_vectors_path": str(vectors_path),
            "dense_vectors_sha256": sha256_file(vectors_path),
            "dense_row_ids_path": str(row_ids_path),
            "dense_row_ids_sha256": sha256_file(row_ids_path),
            "dense_build_progress_path": str(progress_path),
            "dense_preemption_attempts_consumed": 0,
        }
    )
    working_bytes = (
        index_path.stat().st_size
        + vectors_path.stat().st_size
        + row_ids_path.stat().st_size
    )
    manifest["working_set_bytes"] = working_bytes
    manifest["working_set_gate"] = working_bytes <= MAX_WORKING_SET_BYTES
    projected_peak = int(
        (manifest.get("resource_preflight") or {}).get("projected_peak_bytes") or 0
    )
    manifest["build_peak_bound_bytes"] = max(working_bytes, projected_peak)
    manifest["build_peak_gate"] = (
        manifest["build_peak_bound_bytes"] <= MAX_BUILD_PEAK_BYTES
    )
    if not manifest["working_set_gate"]:
        raise ClassificationError("library evidence working set exceeded 3 GiB")
    if not manifest["build_peak_gate"]:
        raise ClassificationError("library evidence build peak exceeded 6 GiB")
    write_sealed_json(manifest_path, manifest, backup=True)
    return manifest


class LibraryEvidenceIndex:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.manifest = read_sealed_json(manifest_path)
        if self.manifest.get("schema") != INDEX_SCHEMA:
            raise ClassificationError("unsupported library evidence index")
        self.path = Path(str(self.manifest.get("index_path") or ""))
        if not self.path.is_file() or sha256_file(self.path) != self.manifest.get(
            "index_sha256"
        ):
            raise ClassificationError("library evidence index checksum mismatch")
        self._query_vector_cache: dict[str, np.ndarray] = {}
        self._dense_vectors: np.memmap | None = None
        self._dense_row_ids: np.ndarray | None = None
        if self.manifest.get("dense_index_built") is True and int(
            self.manifest.get("dense_count") or 0
        ):
            vectors_path = Path(str(self.manifest.get("dense_vectors_path") or ""))
            row_ids_path = Path(str(self.manifest.get("dense_row_ids_path") or ""))
            if (
                not vectors_path.is_file()
                or not row_ids_path.is_file()
                or sha256_file(vectors_path)
                != self.manifest.get("dense_vectors_sha256")
                or sha256_file(row_ids_path)
                != self.manifest.get("dense_row_ids_sha256")
            ):
                raise ClassificationError("dense evidence index checksum mismatch")
            self._dense_vectors = np.load(
                vectors_path, mmap_mode="r", allow_pickle=False
            )
            self._dense_row_ids = np.load(row_ids_path, allow_pickle=False)

    def prefetch_dense_queries(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        purpose: str = "explicit",
    ) -> dict[str, Any]:
        """Embed unique query texts in bounded batches before retrieval."""

        if self._dense_vectors is None:
            return {
                "model": str(self.manifest.get("dense_model") or ""),
                "requested": len(texts),
                "embedded": 0,
                "cached": 0,
            }
        unique = []
        seen = set()
        for text in texts:
            key = sha256_bytes(str(text).encode("utf-8"))
            if key in seen or key in self._query_vector_cache:
                continue
            seen.add(key)
            unique.append(str(text))
        embedded = 0
        for start in range(0, len(unique), max(1, batch_size)):
            batch_texts = unique[start : start + max(1, batch_size)]
            model, vectors = embed_texts_cancellable(
                batch_texts,
                purpose=purpose,
            )
            if model != self.manifest.get("dense_model"):
                raise ClassificationError("dense query embedding model changed")
            matrix = np.asarray(vectors, dtype=np.float32)
            if (
                matrix.ndim != 2
                or matrix.shape[0] != len(batch_texts)
                or matrix.shape[1] != self._dense_vectors.shape[1]
            ):
                raise ClassificationError("dense query dimensions changed")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = np.divide(
                matrix,
                norms,
                out=np.zeros_like(matrix),
                where=norms != 0,
            )
            for text, vector in zip(batch_texts, matrix, strict=True):
                self._query_vector_cache[sha256_bytes(text.encode("utf-8"))] = vector
            embedded += len(batch_texts)
        return {
            "model": str(self.manifest.get("dense_model") or ""),
            "requested": len(texts),
            "embedded": embedded,
            "cached": len(texts) - embedded,
        }

    def query_support(
        self,
        text: str,
        *,
        arms: Sequence[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        tokens = _tokens(text)
        if not tokens:
            return []
        query = " OR ".join(f'"{token}"' for token in tokens[:32])
        placeholders = ",".join("?" for _ in arms)
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                f"""
                SELECT support.*, bm25(support_fts) AS rank
                FROM support_fts JOIN support ON support.row_id=support_fts.rowid
                WHERE support_fts MATCH ? AND support.arm IN ({placeholders})
                ORDER BY rank, support.notation, support.source_record_id
                LIMIT ?
                """,
                (query, *arms, limit),
            ).fetchall()
        finally:
            connection.close()
        return [dict(row) for row in rows]

    def query_support_dense(
        self,
        text: str,
        *,
        arms: Sequence[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        if self._dense_vectors is None or self._dense_row_ids is None:
            return []
        cache_key = sha256_bytes(text.encode("utf-8"))
        query = self._query_vector_cache.get(cache_key)
        if query is None:
            model, vectors = embed_texts_cancellable([text])
            if model != self.manifest.get("dense_model"):
                raise ClassificationError("dense query embedding model changed")
            query = np.asarray(vectors[0], dtype=np.float32)
            if query.ndim != 1 or query.shape[0] != self._dense_vectors.shape[1]:
                raise ClassificationError("dense query dimensions changed")
            norm = float(np.linalg.norm(query))
            if norm == 0:
                return []
            query = query / norm
            self._query_vector_cache[cache_key] = query
        scores = np.asarray(self._dense_vectors @ query)
        if not len(scores):
            return []
        candidate_count = min(len(scores), max(limit * 16, 512))
        indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
        ranked = sorted(
            ((float(scores[index]), int(index)) for index in indices),
            key=lambda pair: (-pair[0], int(self._dense_row_ids[pair[1]])),
        )
        row_score = {int(self._dense_row_ids[index]): score for score, index in ranked}
        placeholders = ",".join("?" for _ in arms)
        id_placeholders = ",".join("?" for _ in row_score)
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                f"""
                SELECT *
                FROM support
                WHERE row_id IN ({id_placeholders})
                  AND arm IN ({placeholders})
                """,
                (*row_score, *arms),
            ).fetchall()
        finally:
            connection.close()
        result = [dict(row) for row in rows]
        for row in result:
            row["dense_score"] = row_score[int(row["row_id"])]
        result.sort(
            key=lambda row: (
                -float(row["dense_score"]),
                str(row["notation"]),
                str(row["source_record_id"]),
            )
        )
        return result[:limit]

    def expand_query(
        self,
        text: str,
        *,
        roles: Sequence[str],
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        tokens = _tokens(text)
        if not tokens or not roles:
            return []
        query = " OR ".join(f'"{token}"' for token in tokens[:32])
        placeholders = ",".join("?" for _ in roles)
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                f"""
                SELECT vocabulary.*, bm25(vocabulary_fts) AS rank
                FROM vocabulary_fts
                JOIN vocabulary ON vocabulary.row_id=vocabulary_fts.rowid
                WHERE vocabulary_fts MATCH ?
                  AND vocabulary.vocabulary_role IN ({placeholders})
                ORDER BY rank, vocabulary.source_record_id
                LIMIT ?
                """,
                (query, *roles, limit),
            ).fetchall()
        finally:
            connection.close()
        return [dict(row) for row in rows]


class _SemanticCandidateIndex(Protocol):
    def candidates(
        self,
        page: Mapping[str, Any],
        *,
        semantic_limit: int,
        total_limit: int,
    ) -> list[dict[str, Any]]: ...


class LibraryEvidenceProvider:
    """A0-preserving provider. External evidence can append, never evict."""

    def __init__(
        self,
        *,
        package: UDCPackage,
        evidence_index: LibraryEvidenceIndex | None,
        semantic_index: _SemanticCandidateIndex | None = None,
    ) -> None:
        self.package = package
        self.official = CandidateIndex(package)
        self.evidence_index = evidence_index
        self.semantic_index = semantic_index

    @staticmethod
    def _page_text(page: Mapping[str, Any]) -> str:
        return " ".join(
            (
                str(page.get("title") or ""),
                str(page.get("summary") or ""),
                " ".join(str(value) for value in page.get("tags") or []),
                " ".join(str(value) for value in page.get("raw_keywords") or []),
                str(page.get("excerpt") or "")[:1_200],
            )
        )

    def candidates(
        self,
        page: Mapping[str, Any],
        *,
        arms: Sequence[str],
        limit: int = 20,
    ) -> dict[str, Any]:
        if page.get("schema") != INFERENCE_DTO_SCHEMA:
            raise ClassificationError("provider requires the gold-free inference DTO")
        if any(key.startswith(("gold_", "adjudication_")) for key in page):
            raise ClassificationError("provider DTO contains gold data")
        baseline = self.official.candidates(page, limit=DEFAULT_CANDIDATE_LIMIT)
        baseline_tail = self.official.candidates(page, limit=max(20, limit))
        semantic = (
            self.semantic_index.candidates(
                {**dict(page), "candidates": baseline_tail},
                semantic_limit=max(20, limit),
                total_limit=max(20, limit),
            )
            if self.semantic_index is not None
            else []
        )
        page_text = self._page_text(page)
        vocabulary_roles = []
        if "B2" in arms:
            vocabulary_roles.append("B2")
        if "C1" in arms:
            vocabulary_roles.append("C1")
        diagnostic_roles = [*vocabulary_roles]
        if "C2" in arms:
            diagnostic_roles.append("C2")
        expansion = (
            self.evidence_index.expand_query(
                page_text,
                roles=diagnostic_roles,
            )
            if self.evidence_index
            else []
        )
        retrieval_expansion = [
            row for row in expansion if row.get("vocabulary_role") in {"B2", "C1"}
        ]
        expanded_text = " ".join(
            (
                page_text,
                " ".join(
                    " ".join(
                        (
                            str(row.get("label") or ""),
                            " ".join(
                                str(value)
                                for value in json.loads(
                                    str(row.get("alt_labels_json") or "[]")
                                )
                            ),
                        )
                    )
                    for row in retrieval_expansion
                ),
            )
        ).strip()
        support_arms = [arm for arm in arms if arm in {"B1a", "B1b", "B3"}]
        evidence_rows: list[dict[str, Any]] = []
        if self.evidence_index and support_arms:
            lexical = self.evidence_index.query_support(
                expanded_text, arms=support_arms, limit=max(64, limit * 4)
            )
            dense = self.evidence_index.query_support_dense(
                expanded_text, arms=support_arms, limit=max(64, limit * 4)
            )
            rows_by_id = {int(row["row_id"]): dict(row) for row in (*lexical, *dense)}
            for row in lexical:
                rows_by_id[int(row["row_id"])]["lexical_rank"] = float(
                    row.get("rank") or 0.0
                )
            for row in dense:
                rows_by_id[int(row["row_id"])]["dense_score"] = float(
                    row.get("dense_score") or 0.0
                )
            evidence_rows = sorted(
                rows_by_id.values(),
                key=lambda row: (
                    -float(row.get("dense_score") or 0.0),
                    abs(float(row.get("lexical_rank") or 0.0)),
                    int(row["row_id"]),
                ),
            )
        by_notation: dict[str, dict[str, Any]] = {}
        for row in evidence_rows:
            notation = str(row["notation"])
            entry = by_notation.setdefault(
                notation,
                {
                    "notation": notation,
                    "support_count": 0,
                    "source_support": [],
                    "external_score": 0.0,
                },
            )
            entry["support_count"] += 1
            lexical_score = 1.0 / max(
                1.0,
                abs(
                    float(
                        row.get("lexical_rank")
                        if row.get("lexical_rank") is not None
                        else row.get("rank") or 0.0
                    )
                )
                + 1.0,
            )
            dense_score = max(0.0, float(row.get("dense_score") or 0.0))
            entry["external_score"] += max(lexical_score, dense_score)
            if len(entry["source_support"]) < 5:
                entry["source_support"].append(
                    {
                        "source": row["source_name"],
                        "record_id": row["source_record_id"],
                        "record_sha256": row["record_sha256"],
                        "package_sha256": row["package_sha256"],
                        "source_field": row["source_field"],
                        "generation_method": row["generation_method"],
                        "intellectual_assignment": row["intellectual_assignment"],
                    }
                )
        external = sorted(
            by_notation.values(),
            key=lambda row: (
                -float(row["external_score"]),
                -int(row["support_count"]),
                len(str(row["notation"])),
                str(row["notation"]),
            ),
        )
        baseline_notations = {str(row["notation"]) for row in baseline}
        union = list(baseline)
        for row in external:
            if row["notation"] in baseline_notations:
                continue
            concept = self.package.by_notation(str(row["notation"]))
            if concept is None:
                continue
            union.append(
                {
                    "notation": row["notation"],
                    "concept_uri": str(concept.get("uri") or ""),
                    "label_en": str(
                        concept.get("label_en") or concept.get("label") or ""
                    ),
                    "label_ja": str(concept.get("label_ja") or ""),
                    "broader_notation": "",
                    "retrieval_score": round(float(row["external_score"]), 6),
                    "source_support": row["source_support"],
                }
            )
            if len(union) >= limit:
                break
        result = {
            "schema": PROVIDER_RESULT_SCHEMA,
            "uid": str(page.get("uid") or ""),
            "source_sha256": str(page.get("source_sha256") or ""),
            "arms": list(arms),
            "official_baseline": baseline,
            "official_baseline_tail": baseline_tail,
            "official_semantic": semantic,
            "external_only": external[:limit],
            "union": union[:limit],
            "query_expansion": [
                {
                    "source": row["source_name"],
                    "label": row["label"],
                    "alt_labels": json.loads(row["alt_labels_json"]),
                    "relations": json.loads(row["relations_json"]),
                    "vocabulary_role": row["vocabulary_role"],
                    "direct_udc_vote": False,
                }
                for row in expansion
            ],
            "baseline_candidates_evicted": False,
            "shadow_only": True,
        }
        result["evidence_bundle_sha256"] = sha256_bytes(
            json.dumps(
                result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        return result
