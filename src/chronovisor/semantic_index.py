"""Immutable semantic index generations with a durable incremental delta.

The base generation is never edited after ``COMPLETE`` is published.  Small
page updates are stored in a generation-scoped delta SQLite database and
shadow the corresponding base rows at query time.  A later full rebuild
compacts those updates into a new immutable generation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from chronovisor.store import CHRONOVISOR_ROOT, page_id_from_path

SEMANTIC_ROOT = CHRONOVISOR_ROOT / ".index" / "semantic"
GENERATIONS_DIR = SEMANTIC_ROOT / "generations"
DELTAS_DIR = SEMANTIC_ROOT / "deltas"
ACTIVE_FILE = SEMANTIC_ROOT / "active.json"
ACTIVATION_LOCK = SEMANTIC_ROOT / "activation.lock"
EXTRACTOR_SCHEMA_VERSION = 1
INDEX_SCHEMA_VERSION = 2
SUPPORTED_INDEX_SCHEMA_VERSIONS = {1, INDEX_SCHEMA_VERSION}
DEFAULT_COARSE_DIMENSIONS = 512
ANN_FILENAME = "coarse.usearch"
LEGACY_EMBEDDINGS_DB = CHRONOVISOR_ROOT / ".index" / "embeddings.sqlite"
LEGACY_ARCHIVE_DIR = CHRONOVISOR_ROOT / ".index" / "archive"


class SemanticIndexError(RuntimeError):
    """Raised when an index artifact violates a hard invariant."""


@dataclass(frozen=True)
class SemanticDocument:
    doc_id: str
    page_id: str
    kind: str
    ordinal: int
    text: str
    source_path: str
    source_sha256: str
    source_mtime_ns: int


@dataclass(frozen=True)
class GenerationManifest:
    schema_version: int
    generation_id: str
    created_at: str
    model: str
    revision: str
    dimensions: int
    dtype: str
    query_prefix: str
    document_prefix: str
    normalization: str
    extractor_schema_version: int
    repo_commit: str
    corpus_fingerprint: str
    page_count: int
    document_count: int
    kind_counts: dict[str, int]
    metadata_sha256: str
    vectors_sha256: str
    ann_kind: str = ""
    ann_dimensions: int = 0
    ann_sha256: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _secure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _secure_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def _repo_commit() -> str:
    try:
        from chronovisor.runtime_config import runtime_identity

        identity = runtime_identity()
        return str(identity.get("commit_id") or identity.get("expected_commit") or "")
    except Exception:
        return ""


def extract_page_documents(path: Path) -> list[SemanticDocument]:
    """Project one canonical page into page/question/chunk documents."""

    from chronovisor.search import (
        _FRONTMATTER_RE,
        _markdown_chunks,
        _recall_questions_from_content,
    )

    try:
        content = path.read_text(encoding="utf-8")
        mtime_ns = path.stat().st_mtime_ns
    except (OSError, UnicodeDecodeError):
        return []
    page_id = page_id_from_path(path)
    source_sha256 = _sha256_bytes(content.encode("utf-8"))
    title_match = re.search(r"title:\s*(.+)", content)
    title = title_match.group(1).strip() if title_match else page_id
    questions = _recall_questions_from_content(content)
    recall_text = "\n".join(f"Q: {question}" for question in questions)
    page_text = f"{title}\n\n{recall_text}\n\n{_FRONTMATTER_RE.sub('', content)[:2000]}"
    common = {
        "page_id": page_id,
        "source_path": str(path),
        "source_sha256": source_sha256,
        "source_mtime_ns": mtime_ns,
    }
    documents = [
        SemanticDocument(
            doc_id=page_id,
            kind="page",
            ordinal=-1,
            text=page_text,
            **common,
        )
    ]
    documents.extend(
        SemanticDocument(
            doc_id=f"{page_id}#q{index}",
            kind="question",
            ordinal=index,
            text=question,
            **common,
        )
        for index, question in enumerate(questions)
    )
    documents.extend(
        SemanticDocument(
            doc_id=f"{page_id}#c{index}",
            kind="chunk",
            ordinal=index,
            text=chunk,
            **common,
        )
        for index, chunk in enumerate(_markdown_chunks(content, title))
    )
    return documents


def extract_all_documents(
    paths: Iterable[Path] | None = None,
) -> list[SemanticDocument]:
    if paths is None:
        from chronovisor.search import searchable_pages

        paths = searchable_pages()
    documents: list[SemanticDocument] = []
    seen_ids: set[str] = set()
    for path in sorted(paths):
        for document in extract_page_documents(path):
            if document.doc_id in seen_ids:
                raise SemanticIndexError(
                    f"duplicate semantic doc_id: {document.doc_id}"
                )
            seen_ids.add(document.doc_id)
            documents.append(document)
    return documents


def _normalized_float32(vectors: np.ndarray, *, dimensions: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimensions:
        raise SemanticIndexError(
            f"embedding dimension mismatch: shape={matrix.shape}, expected={dimensions}"
        )
    if not np.isfinite(matrix).all():
        raise SemanticIndexError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise SemanticIndexError("embedding matrix contains zero-norm vectors")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _corpus_fingerprint(documents: Sequence[SemanticDocument]) -> str:
    pages: dict[str, str] = {}
    for document in documents:
        existing = pages.setdefault(document.page_id, document.source_sha256)
        if existing != document.source_sha256:
            raise SemanticIndexError(f"page {document.page_id} has mixed source hashes")
    encoded = "\n".join(f"{page_id}:{pages[page_id]}" for page_id in sorted(pages))
    return _sha256_bytes(encoded.encode("utf-8"))


def _write_metadata(path: Path, documents: Sequence[SemanticDocument]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE documents (
                vector_row INTEGER PRIMARY KEY,
                doc_id TEXT NOT NULL UNIQUE,
                page_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                source_mtime_ns INTEGER NOT NULL
            );
            CREATE INDEX documents_page_id_idx ON documents(page_id);
            CREATE INDEX documents_kind_idx ON documents(kind);
            """
        )
        connection.executemany(
            """
            INSERT INTO documents
            (vector_row, doc_id, page_id, kind, ordinal, text, source_path,
             source_sha256, source_mtime_ns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    row,
                    document.doc_id,
                    document.page_id,
                    document.kind,
                    document.ordinal,
                    document.text,
                    document.source_path,
                    document.source_sha256,
                    document.source_mtime_ns,
                )
                for row, document in enumerate(documents)
            ),
        )
        connection.commit()
    finally:
        connection.close()
    os.chmod(path, 0o600)


def _build_ann_index(path: Path, vectors: np.ndarray) -> tuple[str, int, str]:
    """Build a persistent Matryoshka-prefix HNSW candidate index.

    The full 2048-dimensional vectors remain authoritative.  HNSW only
    proposes row ids; every returned row is scored again with the full vector
    before it can reach fusion.
    """

    if len(vectors) < 64:
        return "", 0, ""
    try:
        from usearch.index import Index
    except ImportError:
        return "", 0, ""
    dimensions = min(DEFAULT_COARSE_DIMENSIONS, int(vectors.shape[1]))
    coarse = np.ascontiguousarray(vectors[:, :dimensions], dtype=np.float32)
    norms = np.linalg.norm(coarse, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise SemanticIndexError("coarse ANN vectors contain zero norms")
    coarse /= norms
    index = Index(
        ndim=dimensions,
        metric="cos",
        dtype="f16",
        connectivity=16,
        expansion_add=128,
        expansion_search=96,
    )
    index.add(
        np.arange(len(coarse), dtype=np.uint64),
        coarse,
        threads=0,
    )
    index.save(path)
    os.chmod(path, 0o600)
    return "usearch_hnsw_f16", dimensions, _sha256_file(path)


def build_generation(
    documents: Sequence[SemanticDocument],
    *,
    encode_documents: Callable[[list[str], int], np.ndarray],
    model: str,
    revision: str,
    dimensions: int,
    query_prefix: str,
    document_prefix: str,
    batch_size: int,
    root: Path = SEMANTIC_ROOT,
    repo_commit: str | None = None,
) -> GenerationManifest:
    """Build and seal a complete immutable generation."""

    if not documents:
        raise SemanticIndexError("cannot build an empty semantic generation")
    doc_ids = [document.doc_id for document in documents]
    if len(set(doc_ids)) != len(doc_ids):
        raise SemanticIndexError("semantic generation contains duplicate doc ids")
    page_ids = {document.page_id for document in documents}
    texts = [document.text for document in documents]
    vectors = _normalized_float32(
        encode_documents(texts, batch_size), dimensions=dimensions
    )
    if len(vectors) != len(documents):
        raise SemanticIndexError(
            f"embedding count mismatch: {len(vectors)} != {len(documents)}"
        )

    profile_seed = json.dumps(
        {
            "model": model,
            "revision": revision,
            "dimensions": dimensions,
            "dtype": "float32",
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
            "normalization": "l2",
            "extractor": EXTRACTOR_SCHEMA_VERSION,
            "corpus": _corpus_fingerprint(documents),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generation_id = (
        f"{timestamp}-{_sha256_bytes(profile_seed)[:12]}-{uuid.uuid4().hex[:6]}"
    )
    generations = root / "generations"
    _secure_directory(root)
    _secure_directory(generations)
    staging = generations / f".staging-{generation_id}"
    final = generations / generation_id
    if staging.exists():
        shutil.rmtree(staging)
    _secure_directory(staging)
    try:
        metadata_path = staging / "metadata.sqlite"
        vectors_path = staging / "vectors.npy"
        _write_metadata(metadata_path, documents)
        with vectors_path.open("wb") as handle:
            os.chmod(vectors_path, 0o600)
            np.save(handle, vectors, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        ann_kind, ann_dimensions, ann_sha256 = _build_ann_index(
            staging / ANN_FILENAME,
            vectors,
        )
        kind_counts: dict[str, int] = {}
        for document in documents:
            kind_counts[document.kind] = kind_counts.get(document.kind, 0) + 1
        manifest = GenerationManifest(
            schema_version=INDEX_SCHEMA_VERSION,
            generation_id=generation_id,
            created_at=_utc_now(),
            model=model,
            revision=revision,
            dimensions=dimensions,
            dtype="float32",
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            normalization="l2",
            extractor_schema_version=EXTRACTOR_SCHEMA_VERSION,
            repo_commit=repo_commit if repo_commit is not None else _repo_commit(),
            corpus_fingerprint=_corpus_fingerprint(documents),
            page_count=len(page_ids),
            document_count=len(documents),
            kind_counts=kind_counts,
            metadata_sha256=_sha256_file(metadata_path),
            vectors_sha256=_sha256_file(vectors_path),
            ann_kind=ann_kind,
            ann_dimensions=ann_dimensions,
            ann_sha256=ann_sha256,
        )
        _atomic_json(staging / "manifest.json", asdict(manifest))
        complete_payload = {
            "generation_id": generation_id,
            "manifest_sha256": _sha256_file(staging / "manifest.json"),
            "completed_at": _utc_now(),
        }
        _atomic_json(staging / "COMPLETE", complete_payload)
        _fsync_directory(staging)
        os.replace(staging, final)
        _fsync_directory(generations)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def upgrade_generation_with_ann(
    generation_id: str,
    *,
    root: Path = SEMANTIC_ROOT,
    repo_commit: str | None = None,
) -> GenerationManifest:
    """Clone an immutable flat generation and seal it with a coarse HNSW."""

    source = validate_generation(generation_id, root=root)
    if source.ann_kind:
        return source
    source_dir = generation_dir(generation_id, root=root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    upgraded_id = f"{timestamp}-ann-{uuid.uuid4().hex[:8]}"
    generations = root / "generations"
    staging = generations / f".staging-{upgraded_id}"
    final = generations / upgraded_id
    _secure_directory(generations)
    _secure_directory(staging)
    try:
        metadata_path = staging / "metadata.sqlite"
        vectors_path = staging / "vectors.npy"
        shutil.copy2(source_dir / "metadata.sqlite", metadata_path)
        shutil.copy2(source_dir / "vectors.npy", vectors_path)
        os.chmod(metadata_path, 0o600)
        os.chmod(vectors_path, 0o600)
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
        ann_kind, ann_dimensions, ann_sha256 = _build_ann_index(
            staging / ANN_FILENAME,
            vectors,
        )
        if not ann_kind:
            raise SemanticIndexError("usearch is required for ANN generation upgrade")
        manifest = replace(
            source,
            schema_version=INDEX_SCHEMA_VERSION,
            generation_id=upgraded_id,
            created_at=_utc_now(),
            repo_commit=repo_commit if repo_commit is not None else _repo_commit(),
            metadata_sha256=_sha256_file(metadata_path),
            vectors_sha256=_sha256_file(vectors_path),
            ann_kind=ann_kind,
            ann_dimensions=ann_dimensions,
            ann_sha256=ann_sha256,
        )
        _atomic_json(staging / "manifest.json", asdict(manifest))
        _atomic_json(
            staging / "COMPLETE",
            {
                "generation_id": upgraded_id,
                "manifest_sha256": _sha256_file(staging / "manifest.json"),
                "completed_at": _utc_now(),
                "upgraded_from": generation_id,
            },
        )
        _fsync_directory(staging)
        os.replace(staging, final)
        _fsync_directory(generations)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def generation_dir(generation_id: str, *, root: Path = SEMANTIC_ROOT) -> Path:
    if not generation_id or "/" in generation_id or generation_id.startswith("."):
        raise SemanticIndexError(f"invalid semantic generation id: {generation_id!r}")
    return root / "generations" / generation_id


def read_manifest(
    generation_id: str, *, root: Path = SEMANTIC_ROOT
) -> GenerationManifest:
    directory = generation_dir(generation_id, root=root)
    try:
        raw = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticIndexError(
            f"unreadable generation manifest: {generation_id}"
        ) from exc
    try:
        manifest = GenerationManifest(**raw)
    except TypeError as exc:
        raise SemanticIndexError(
            f"invalid generation manifest: {generation_id}"
        ) from exc
    if manifest.schema_version not in SUPPORTED_INDEX_SCHEMA_VERSIONS:
        raise SemanticIndexError(
            f"unsupported generation schema: {manifest.schema_version}"
        )
    if manifest.generation_id != generation_id:
        raise SemanticIndexError("generation manifest id mismatch")
    return manifest


def validate_generation(
    generation_id: str,
    *,
    root: Path = SEMANTIC_ROOT,
    verify_checksums: bool = True,
) -> GenerationManifest:
    directory = generation_dir(generation_id, root=root)
    if not (directory / "COMPLETE").is_file():
        raise SemanticIndexError(f"generation is incomplete: {generation_id}")
    manifest = read_manifest(generation_id, root=root)
    metadata = directory / "metadata.sqlite"
    vectors = directory / "vectors.npy"
    ann = directory / ANN_FILENAME
    if verify_checksums:
        if _sha256_file(metadata) != manifest.metadata_sha256:
            raise SemanticIndexError("semantic metadata checksum mismatch")
        if _sha256_file(vectors) != manifest.vectors_sha256:
            raise SemanticIndexError("semantic vectors checksum mismatch")
        if manifest.ann_sha256:
            if not ann.is_file() or _sha256_file(ann) != manifest.ann_sha256:
                raise SemanticIndexError("semantic ANN checksum mismatch")
    if bool(manifest.ann_kind) != bool(manifest.ann_sha256):
        raise SemanticIndexError("semantic ANN manifest is incomplete")
    if manifest.ann_kind and not (1 <= manifest.ann_dimensions <= manifest.dimensions):
        raise SemanticIndexError("semantic ANN dimensions are invalid")
    matrix = np.load(vectors, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (manifest.document_count, manifest.dimensions):
        raise SemanticIndexError(f"semantic vectors shape mismatch: {matrix.shape}")
    if matrix.dtype != np.float32:
        raise SemanticIndexError(f"semantic vectors dtype mismatch: {matrix.dtype}")
    connection = sqlite3.connect(f"file:{metadata}?mode=ro", uri=True)
    try:
        row = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT doc_id), COUNT(DISTINCT page_id),
                   MIN(vector_row), MAX(vector_row)
            FROM documents
            """
        ).fetchone()
    finally:
        connection.close()
    count, unique_docs, unique_pages, minimum, maximum = row
    if (
        count != manifest.document_count
        or unique_docs != count
        or unique_pages != manifest.page_count
        or minimum != 0
        or maximum != count - 1
    ):
        raise SemanticIndexError("semantic metadata coverage invariant failed")
    return manifest


def read_active(*, root: Path = SEMANTIC_ROOT) -> dict[str, Any]:
    try:
        payload = json.loads((root / "active.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def activate_generation(
    generation_id: str,
    *,
    expected_current: str | None = None,
    root: Path = SEMANTIC_ROOT,
) -> dict[str, Any]:
    manifest = validate_generation(generation_id, root=root)
    _secure_directory(root)
    lock_path = root / "activation.lock"
    with lock_path.open("a+") as handle:
        os.chmod(lock_path, 0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        current = read_active(root=root)
        current_id = str(current.get("generation_id") or "")
        if expected_current is not None and current_id != expected_current:
            raise SemanticIndexError(
                f"active generation CAS failed: {current_id!r} != {expected_current!r}"
            )
        payload = {
            "schema_version": 1,
            "generation_id": generation_id,
            "previous_generation_id": current_id,
            "manifest_sha256": _sha256_file(
                generation_dir(generation_id, root=root) / "manifest.json"
            ),
            "activated_at": _utc_now(),
            "model": manifest.model,
            "revision": manifest.revision,
        }
        _atomic_json(root / "active.json", payload)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return payload


def rollback_generation(*, root: Path = SEMANTIC_ROOT) -> dict[str, Any]:
    active = read_active(root=root)
    current = str(active.get("generation_id") or "")
    previous = str(active.get("previous_generation_id") or "")
    if not current or not previous:
        raise SemanticIndexError("no previous semantic generation is available")
    return activate_generation(previous, expected_current=current, root=root)


def _metadata_rows(path: Path) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return connection.execute(
            """
            SELECT vector_row, doc_id, page_id, kind, ordinal, source_sha256,
                   source_mtime_ns
            FROM documents ORDER BY vector_row
            """
        ).fetchall()
    finally:
        connection.close()


@dataclass
class LoadedGeneration:
    manifest: GenerationManifest
    vectors: np.ndarray
    doc_ids: list[str]
    page_ids: list[str]
    kinds: list[str]
    source_hashes: list[str]
    overridden_pages: set[str]
    delta_vectors: np.ndarray
    delta_page_ids: list[str]
    delta_kinds: list[str]
    page_rows: dict[str, list[int]]
    ann_index: Any | None = None

    def _normalized_query(self, query_vector: Sequence[float]) -> np.ndarray:
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.shape != (self.manifest.dimensions,) or not np.isfinite(vector).all():
            raise SemanticIndexError("query vector shape/value mismatch")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise SemanticIndexError("query vector has zero/invalid norm")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    def _candidate_rows(self, vector: np.ndarray, *, top_n: int) -> np.ndarray:
        if self.ann_index is None or not self.manifest.ann_dimensions:
            return np.arange(len(self.vectors), dtype=np.int64)
        dimensions = self.manifest.ann_dimensions
        coarse = np.ascontiguousarray(vector[:dimensions], dtype=np.float32)
        norm = float(np.linalg.norm(coarse))
        if not math.isfinite(norm) or norm <= 1e-12:
            return np.arange(len(self.vectors), dtype=np.int64)
        coarse /= norm
        candidate_count = min(
            len(self.vectors),
            max(512, max(1, top_n) * 16),
        )
        matches = self.ann_index.search(coarse, count=candidate_count)
        return np.asarray(matches.keys, dtype=np.int64)

    def _score_base_rows(
        self,
        vector: np.ndarray,
        rows: Sequence[int],
        *,
        page_filter: set[str] | None = None,
    ) -> dict[str, float]:
        selected = np.asarray(rows, dtype=np.int64)
        if not len(selected):
            return {}
        scores = np.asarray(self.vectors[selected] @ vector, dtype=np.float32)
        by_page: dict[str, float] = {}
        for offset, raw_row in enumerate(selected):
            row = int(raw_row)
            page_id = self.page_ids[row]
            if page_id in self.overridden_pages:
                continue
            if page_filter is not None and page_id not in page_filter:
                continue
            score = float(scores[offset])
            if self.kinds and self.kinds[row] == "chunk":
                score *= 0.92
            if score > by_page.get(page_id, float("-inf")):
                by_page[page_id] = score
        return by_page

    def _score_delta(
        self,
        vector: np.ndarray,
        *,
        page_filter: set[str] | None = None,
    ) -> dict[str, float]:
        if not len(self.delta_vectors):
            return {}
        scores = np.asarray(self.delta_vectors @ vector, dtype=np.float32)
        by_page: dict[str, float] = {}
        for row, raw_score in enumerate(scores):
            page_id = self.delta_page_ids[row]
            if page_filter is not None and page_id not in page_filter:
                continue
            score = float(raw_score)
            if self.delta_kinds and self.delta_kinds[row] == "chunk":
                score *= 0.92
            if score > by_page.get(page_id, float("-inf")):
                by_page[page_id] = score
        return by_page

    def search(
        self, query_vector: Sequence[float], *, top_n: int
    ) -> list[tuple[str, float]]:
        vector = self._normalized_query(query_vector)
        by_page = self._score_base_rows(
            vector,
            self._candidate_rows(vector, top_n=top_n),
        )
        for page_id, score in self._score_delta(vector).items():
            if score > by_page.get(page_id, float("-inf")):
                by_page[page_id] = score
        return sorted(by_page.items(), key=lambda item: item[1], reverse=True)[:top_n]

    def score_pages(
        self,
        query_vector: Sequence[float],
        page_ids: Sequence[str],
    ) -> list[tuple[str, float]]:
        """Exactly score a bounded page set with authoritative full vectors."""

        targets = {str(page_id) for page_id in page_ids if page_id}
        if not targets:
            return []
        vector = self._normalized_query(query_vector)
        rows = [row for page_id in targets for row in self.page_rows.get(page_id, ())]
        by_page = self._score_base_rows(vector, rows, page_filter=targets)
        for page_id, score in self._score_delta(
            vector,
            page_filter=targets,
        ).items():
            if score > by_page.get(page_id, float("-inf")):
                by_page[page_id] = score
        return sorted(by_page.items(), key=lambda item: item[1], reverse=True)


def _delta_db(generation_id: str, *, root: Path) -> Path:
    path = root / "deltas" / f"{generation_id}.sqlite"
    _secure_directory(root)
    _secure_directory(path.parent)
    return path


def _connect_delta(generation_id: str, *, root: Path) -> sqlite3.Connection:
    path = _delta_db(generation_id, root=root)
    connection = sqlite3.connect(path)
    os.chmod(path, 0o600)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS page_state (
            page_id TEXT PRIMARY KEY,
            source_sha256 TEXT NOT NULL,
            source_mtime_ns INTEGER NOT NULL,
            deleted INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            page_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            vector BLOB NOT NULL,
            dim INTEGER NOT NULL,
            source_sha256 TEXT NOT NULL,
            source_mtime_ns INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS delta_documents_page_idx ON documents(page_id);
        """
    )
    return connection


def write_page_delta(
    generation_id: str,
    page_id: str,
    documents: Sequence[SemanticDocument],
    vectors: np.ndarray | None,
    *,
    dimensions: int,
    root: Path = SEMANTIC_ROOT,
) -> None:
    if documents:
        if any(document.page_id != page_id for document in documents):
            raise SemanticIndexError("delta documents contain a different page id")
        if vectors is None or len(vectors) != len(documents):
            raise SemanticIndexError("delta embedding count mismatch")
        matrix = _normalized_float32(vectors, dimensions=dimensions)
        source_sha256 = documents[0].source_sha256
        source_mtime_ns = documents[0].source_mtime_ns
        deleted = 0
    else:
        matrix = np.empty((0, dimensions), dtype=np.float32)
        source_sha256 = ""
        source_mtime_ns = 0
        deleted = 1
    connection = _connect_delta(generation_id, root=root)
    try:
        with connection:
            connection.execute("DELETE FROM documents WHERE page_id = ?", (page_id,))
            connection.executemany(
                """
                INSERT INTO documents
                (doc_id, page_id, kind, ordinal, vector, dim, source_sha256,
                 source_mtime_ns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        document.doc_id,
                        page_id,
                        document.kind,
                        document.ordinal,
                        matrix[index].tobytes(order="C"),
                        dimensions,
                        document.source_sha256,
                        document.source_mtime_ns,
                    )
                    for index, document in enumerate(documents)
                ),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO page_state
                (page_id, source_sha256, source_mtime_ns, deleted, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    page_id,
                    source_sha256,
                    source_mtime_ns,
                    deleted,
                    _utc_now(),
                ),
            )
    finally:
        connection.close()


def _load_delta(
    generation_id: str, *, dimensions: int, root: Path
) -> tuple[set[str], np.ndarray, list[str], list[str]]:
    path = root / "deltas" / f"{generation_id}.sqlite"
    if not path.exists():
        return set(), np.empty((0, dimensions), dtype=np.float32), [], []
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        overridden = {
            str(row[0]) for row in connection.execute("SELECT page_id FROM page_state")
        }
        rows = connection.execute(
            "SELECT page_id, kind, vector, dim FROM documents ORDER BY doc_id"
        ).fetchall()
    finally:
        connection.close()
    page_ids: list[str] = []
    kinds: list[str] = []
    vectors: list[np.ndarray] = []
    for page_id, kind, blob, dim in rows:
        if int(dim) != dimensions:
            raise SemanticIndexError("delta vector dimension mismatch")
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.shape != (dimensions,) or not np.isfinite(vector).all():
            raise SemanticIndexError("invalid delta vector")
        page_ids.append(str(page_id))
        kinds.append(str(kind))
        vectors.append(vector)
    matrix = (
        np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
        if vectors
        else np.empty((0, dimensions), dtype=np.float32)
    )
    return overridden, matrix, page_ids, kinds


def load_generation(
    generation_id: str,
    *,
    root: Path = SEMANTIC_ROOT,
    verify_checksums: bool = True,
) -> LoadedGeneration:
    manifest = validate_generation(
        generation_id, root=root, verify_checksums=verify_checksums
    )
    directory = generation_dir(generation_id, root=root)
    rows = _metadata_rows(directory / "metadata.sqlite")
    vectors = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
    ann_index: Any | None = None
    if manifest.ann_kind:
        try:
            from usearch.index import Index
        except ImportError as exc:
            raise SemanticIndexError(
                "active semantic generation requires the usearch runtime"
            ) from exc
        try:
            ann_index = Index.restore(directory / ANN_FILENAME, view=True)
        except Exception as exc:
            raise SemanticIndexError("semantic ANN index could not be loaded") from exc
    overridden, delta_vectors, delta_page_ids, delta_kinds = _load_delta(
        generation_id, dimensions=manifest.dimensions, root=root
    )
    page_rows: dict[str, list[int]] = {}
    for row in rows:
        page_rows.setdefault(str(row[2]), []).append(int(row[0]))
    return LoadedGeneration(
        manifest=manifest,
        vectors=vectors,
        doc_ids=[str(row[1]) for row in rows],
        page_ids=[str(row[2]) for row in rows],
        kinds=[str(row[3]) for row in rows],
        source_hashes=[str(row[5]) for row in rows],
        overridden_pages=overridden,
        delta_vectors=delta_vectors,
        delta_page_ids=delta_page_ids,
        delta_kinds=delta_kinds,
        page_rows=page_rows,
        ann_index=ann_index,
    )


def load_active_generation(
    *, root: Path = SEMANTIC_ROOT, verify_checksums: bool = True
) -> LoadedGeneration:
    active = read_active(root=root)
    generation_id = str(active.get("generation_id") or "")
    if not generation_id:
        raise SemanticIndexError("no active semantic generation")
    manifest_sha = str(active.get("manifest_sha256") or "")
    path = generation_dir(generation_id, root=root) / "manifest.json"
    if manifest_sha != _sha256_file(path):
        raise SemanticIndexError("active semantic manifest pointer mismatch")
    return load_generation(generation_id, root=root, verify_checksums=verify_checksums)


def semantic_index_status(*, root: Path = SEMANTIC_ROOT) -> dict[str, Any]:
    active = read_active(root=root)
    generation_id = str(active.get("generation_id") or "")
    if not generation_id:
        return {
            "status": "missing",
            "root": str(root),
            "generation_id": "",
            "coverage": 0.0,
        }
    try:
        manifest = validate_generation(generation_id, root=root, verify_checksums=False)
        delta_path = root / "deltas" / f"{generation_id}.sqlite"
        delta_pages = 0
        indexed_mtimes: dict[str, int] = {}
        metadata = generation_dir(generation_id, root=root) / "metadata.sqlite"
        connection = sqlite3.connect(f"file:{metadata}?mode=ro", uri=True)
        try:
            indexed_mtimes = {
                str(row[0]): int(row[1])
                for row in connection.execute(
                    "SELECT page_id, MAX(source_mtime_ns) "
                    "FROM documents GROUP BY page_id"
                )
            }
        finally:
            connection.close()
        if delta_path.exists():
            connection = sqlite3.connect(f"file:{delta_path}?mode=ro", uri=True)
            try:
                states = connection.execute(
                    "SELECT page_id, source_mtime_ns, deleted FROM page_state"
                ).fetchall()
                delta_pages = len(states)
                for page_id, source_mtime_ns, deleted in states:
                    if deleted:
                        indexed_mtimes.pop(str(page_id), None)
                    else:
                        indexed_mtimes[str(page_id)] = int(source_mtime_ns)
            finally:
                connection.close()
        from chronovisor.search import searchable_pages

        current_mtimes: dict[str, int] = {}
        for path in searchable_pages():
            try:
                current_mtimes[page_id_from_path(path)] = path.stat().st_mtime_ns
            except OSError:
                continue
        current_ids = set(current_mtimes)
        indexed_ids = set(indexed_mtimes)
        missing_ids = current_ids - indexed_ids
        stale_ids = {
            page_id
            for page_id in current_ids & indexed_ids
            if current_mtimes[page_id] != indexed_mtimes[page_id]
        }
        deleted_ids = indexed_ids - current_ids
        covered = len(current_ids - missing_ids - stale_ids)
        coverage = covered / len(current_ids) if current_ids else 1.0
        return {
            "status": "ok",
            "root": str(root),
            "generation_id": generation_id,
            "previous_generation_id": active.get("previous_generation_id", ""),
            "model": manifest.model,
            "revision": manifest.revision,
            "dimensions": manifest.dimensions,
            "ann_kind": manifest.ann_kind or "flat",
            "ann_dimensions": manifest.ann_dimensions,
            "page_count": manifest.page_count,
            "document_count": manifest.document_count,
            "kind_counts": manifest.kind_counts,
            "delta_pages": delta_pages,
            "current_page_count": len(current_ids),
            "covered_page_count": covered,
            "missing_page_count": len(missing_ids),
            "stale_page_count": len(stale_ids),
            "deleted_page_count": len(deleted_ids),
            "missing_page_ids": sorted(missing_ids),
            "stale_page_ids": sorted(stale_ids),
            "deleted_page_ids": sorted(deleted_ids),
            "coverage": coverage,
            "corpus_fingerprint": manifest.corpus_fingerprint,
            "activated_at": active.get("activated_at"),
        }
    except Exception as exc:
        return {
            "status": "invalid",
            "root": str(root),
            "generation_id": generation_id,
            "error": f"{type(exc).__name__}: {exc}",
            "coverage": 0.0,
        }


def prune_generations(
    *, keep: int = 2, min_age_days: int = 7, root: Path = SEMANTIC_ROOT
) -> list[str]:
    """Remove inactive complete generations only after both retention gates."""

    keep = max(2, keep)
    active = read_active(root=root)
    protected = {
        str(active.get("generation_id") or ""),
        str(active.get("previous_generation_id") or ""),
    }
    generations = root / "generations"
    candidates = sorted(
        (
            path
            for path in generations.glob("*")
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    protected.update(path.name for path in candidates[:keep])
    removed: list[str] = []
    cutoff = time.time() - max(0, min_age_days) * 24 * 60 * 60
    for path in candidates:
        if (
            path.name in protected
            or not (path / "COMPLETE").exists()
            or path.stat().st_mtime > cutoff
        ):
            continue
        shutil.rmtree(path)
        (root / "deltas" / f"{path.name}.sqlite").unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def archive_legacy_search_index(
    *,
    source: Path = LEGACY_EMBEDDINGS_DB,
    archive_dir: Path = LEGACY_ARCHIVE_DIR,
    retain_days: int = 14,
    root: Path = SEMANTIC_ROOT,
) -> dict[str, Any]:
    """Compress and remove the mutable BGE search DB after a complete cutover."""

    status = semantic_index_status(root=root)
    if (
        status.get("status") != "ok"
        or float(status.get("coverage") or 0.0) < 0.999
        or int(status.get("missing_page_count") or 0)
        or int(status.get("stale_page_count") or 0)
    ):
        raise SemanticIndexError("legacy cleanup requires a complete fresh generation")
    if not source.exists():
        return {"status": "skipped", "reason": "legacy index is already absent"}

    import zstandard

    _secure_directory(archive_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = archive_dir / f"embeddings-bge-{stamp}.sqlite.zst"
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    source_sha256 = _sha256_file(source)
    compressor = zstandard.ZstdCompressor(level=10)
    try:
        with source.open("rb") as source_handle, temporary.open("wb") as output:
            os.chmod(temporary, 0o600)
            compressor.copy_stream(source_handle, output)
            output.flush()
            os.fsync(output.fileno())
        decompressed_sha = hashlib.sha256()
        with temporary.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    decompressed_sha.update(chunk)
        if decompressed_sha.hexdigest() != source_sha256:
            raise SemanticIndexError("legacy archive round-trip checksum mismatch")
        os.replace(temporary, archive)
        os.chmod(archive, 0o600)
        _fsync_directory(archive_dir)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "schema_version": 1,
        "kind": "retired_bge_search_index",
        "source": str(source),
        "source_sha256": source_sha256,
        "archive": str(archive),
        "archive_sha256": _sha256_file(archive),
        "archived_at": _utc_now(),
        "expires_after_days": max(1, retain_days),
        "expires_at": (
            datetime.now(timezone.utc) + timedelta(days=max(1, retain_days))
        ).isoformat(timespec="milliseconds"),
        "active_generation_id": status["generation_id"],
        "active_model": status["model"],
        "active_revision": status["revision"],
    }
    _atomic_json(archive.with_suffix(archive.suffix + ".manifest.json"), manifest)
    source.unlink()
    _fsync_directory(source.parent)
    return {"status": "archived", **manifest}


def prune_expired_legacy_archives(
    *, archive_dir: Path = LEGACY_ARCHIVE_DIR
) -> list[str]:
    removed: list[str] = []
    now = datetime.now(timezone.utc)
    archive_root = archive_dir.resolve()
    for manifest_path in archive_dir.glob("*.sqlite.zst.manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expires = datetime.fromisoformat(str(manifest["expires_at"]))
            if expires.tzinfo is None:
                continue
            archive = Path(str(manifest["archive"])).resolve()
            if archive.parent != archive_root or not archive.name.endswith(
                ".sqlite.zst"
            ):
                continue
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
        if expires > now:
            continue
        archive.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        removed.append(str(archive))
    if removed:
        _fsync_directory(archive_dir)
    return removed
