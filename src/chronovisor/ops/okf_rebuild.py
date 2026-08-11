"""One offline coordinator for the post-cutover derived rebuild."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from chronovisor.core import (
    canonical_document,
    canonical_json,
    durable_state,
    index_store,
    knowledge_graph_config,
    knowledge_graph_store,
    lexical_index,
    llm_config,
    llm_runtime,
    okf_cutover,
    page_identity,
    reserved_documents,
    runtime_config,
    semantic_index,
)
from chronovisor.ingest import page_registry, uid_link_index
from chronovisor.knowledge_graph import builder
from chronovisor.ops.cortex import build_cortex_graph

validate_canonical_document = canonical_document.validate_canonical_document
canonical_json_line_bytes_strict = canonical_json.canonical_json_line_bytes_strict
canonical_json_sha256_strict = canonical_json.canonical_json_sha256_strict
open_directory_nofollow = durable_state.open_directory_nofollow
open_regular_nofollow = durable_state.open_regular_nofollow
read_sealed_json = durable_state.read_sealed_json
write_sealed_json = durable_state.write_sealed_json
KnowledgeGraphConfig = knowledge_graph_config.KnowledgeGraphConfig
KnowledgeGraphStore = knowledge_graph_store.KnowledgeGraphStore
LexicalIndex = lexical_index.LexicalIndex
load_default_llm_runtime = llm_config.load_default_llm_runtime
EmbeddingPurpose = llm_runtime.EmbeddingPurpose
EmbeddingRequest = llm_runtime.EmbeddingRequest
LLMRuntime = llm_runtime.LLMRuntime
RouteLocation = llm_runtime.RouteLocation
SourceDataClass = llm_runtime.SourceDataClass
SourceDataClassification = llm_runtime.SourceDataClassification
SourceSensitivity = llm_runtime.SourceSensitivity
OKFRebuildGate = okf_cutover.OKFRebuildGate
okf_rebuild_session = okf_cutover.okf_rebuild_session
rebuild_pages_index = reserved_documents.rebuild_pages_index
load_search_embedding_config = runtime_config.load_search_embedding_config
PageRegistry = page_registry.PageRegistry
build_uid_link_index = uid_link_index.build_uid_link_index
DETERMINISTIC_EXTRACTOR_IDENTITY = builder.DETERMINISTIC_EXTRACTOR_IDENTITY
GRAPH_BUILDER_POLICY_VERSION = builder.GRAPH_BUILDER_POLICY_VERSION
run_builder_cycle = builder.run_builder_cycle
normalize_page_uid = page_identity.normalize_page_uid

_FOREGROUND_ROLE = "search.semantic.foreground"
_SEMANTIC_SOURCE = SourceDataClassification(
    SourceDataClass.SYSTEM,
    SourceSensitivity.HIGH,
)
_DERIVED_GENERATION_SCHEMA = 1

SemanticEncoder = Callable[
    [Sequence[semantic_index.SemanticDocument], int], np.ndarray
]


def _sha256(path: Path) -> str:
    with open_regular_nofollow(path) as handle:
        digest = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_object(path: Path) -> dict[str, Any]:
    with open_regular_nofollow(path) as handle:
        raw = handle.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("migration object exceeds the offline rebuild bound")
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json_line_bytes_strict(value) != raw:
        raise ValueError("migration object is not canonical JSON")
    return value


def _stable_sources(root: Path) -> tuple[list[Path], list[dict[str, str]]]:
    paths = index_store.canonical_document_paths(
        root / "pages",
        system_dir=root / "system",
        require_stable=False,
        strict=True,
    )
    rows: list[dict[str, str]] = []
    stable_paths: list[Path] = []
    page_ids: set[str] = set()
    for path in paths:
        namespace: canonical_document.Namespace = (
            "system" if path.is_relative_to(root / "system") else "pages"
        )
        namespace_root = root / namespace
        relative = path.relative_to(namespace_root).as_posix()
        raw = path.read_bytes()
        document = validate_canonical_document(
            raw,
            namespace=namespace,
            path=relative,
        )
        if document.metadata["status"] != "stable":
            continue
        page_id = path.stem
        if page_id in page_ids:
            raise ValueError("stable canonical corpus has duplicate page IDs")
        page_ids.add(page_id)
        stable_paths.append(path)
        rows.append(
            {
                "path": f"{namespace}/{relative}",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    rows.sort(key=lambda row: row["path"])
    return stable_paths, rows


def _manifest_invalidation(gate: OKFRebuildGate) -> dict[str, int | str]:
    manifest = _canonical_object(gate.workspace / "dry-run-manifest.json")
    documents = manifest.get("documents")
    system_documents = manifest.get("system_documents")
    if not isinstance(documents, list) or not isinstance(system_documents, list):
        raise ValueError("migration manifest has no document identity inventory")
    identities: list[dict[str, str]] = []
    for row in [*documents, *system_documents]:
        if not isinstance(row, Mapping):
            raise ValueError("migration manifest document identity is invalid")
        source = row.get("source_sha256")
        output = row.get("output_sha256")
        if (
            not isinstance(source, str)
            or not isinstance(output, str)
            or re.fullmatch(r"[0-9a-f]{64}", source) is None
            or re.fullmatch(r"[0-9a-f]{64}", output) is None
        ):
            raise ValueError("migration manifest document hash is invalid")
        identities.append({"source": source, "output": output})
    return {
        "changed_page_count": sum(
            row["source"] != row["output"] for row in identities
        ),
        "source_set_sha256": canonical_json_sha256_strict(identities),
        "target_count": 0,
    }


def _rebuild_registry_and_links(
    root: Path,
    gate: OKFRebuildGate,
    source_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registry = PageRegistry(root)
    manifest = registry.ensure_manifest(write=True)
    registry_state = manifest["registry"]
    stable = registry.stable_pages(registry_state)
    expected = {row["path"]: row["sha256"] for row in source_rows}
    observed = {
        str(row.get("path") or ""): str(row.get("content_sha256") or "")
        for row in stable.values()
    }
    if observed != expected:
        raise ValueError("page registry does not cover the exact stable corpus")
    pages = registry_state.get("pages")
    if not isinstance(pages, Mapping):
        raise ValueError("page registry has no identity inventory")
    current_by_path: dict[str, tuple[str, Mapping[str, object]]] = {}
    for uid, row in pages.items():
        if not isinstance(uid, str) or not isinstance(row, Mapping):
            raise ValueError("page registry identity row is invalid")
        if row.get("canonical_uid") is not None:
            continue
        path = row.get("path")
        if not isinstance(path, str) or not path or path in current_by_path:
            raise ValueError("page registry path identity is invalid")
        current_by_path[path] = (uid, row)
    migration = _canonical_object(gate.workspace / "dry-run-manifest.json")
    for scope, field in (("pages", "documents"), ("system", "system_documents")):
        rows = migration.get(field)
        if not isinstance(rows, list):
            raise ValueError("migration identity inventory is invalid")
        for item in rows:
            if not isinstance(item, Mapping):
                raise ValueError("migration identity row is invalid")
            relative = item.get("relative_path")
            output_sha256 = item.get("output_sha256")
            if (
                not isinstance(relative, str)
                or not relative
                or not isinstance(output_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", output_sha256) is None
            ):
                raise ValueError("migration identity row is invalid")
            current = current_by_path.get(f"{scope}/{relative}")
            if current is None or current[1].get("content_sha256") != output_sha256:
                raise ValueError("page registry does not cover migration output")
            manifest_uid = item.get("uid")
            if manifest_uid is not None:
                try:
                    expected_uid = normalize_page_uid(manifest_uid)
                except ValueError as exc:
                    raise ValueError("migration page UID is invalid") from exc
                if current[0] != expected_uid:
                    raise ValueError("page registry changed a migration page UID")
            else:
                identity_sha256 = item.get("identity_sha256")
                if (
                    not isinstance(identity_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", identity_sha256) is None
                ):
                    raise ValueError("UID-less migration identity is invalid")
    uids = sorted(stable)
    if not all(uid and len(uid) <= 128 for uid in uids):
        raise ValueError("page registry has an invalid stable UID")
    link_index = build_uid_link_index(root, registry=registry, write=True)
    if int(link_index["unresolved_count"]) != 0:
        raise ValueError("UID link index has unresolved canonical links")
    corpus = {
        "stable_page_count": len(source_rows),
        "stable_path_set_sha256": canonical_json_sha256_strict(sorted(expected)),
        "stable_source_set_sha256": canonical_json_sha256_strict(
            [dict(row) for row in source_rows]
        ),
        "stable_uid_set_sha256": canonical_json_sha256_strict(uids),
    }
    return (
        {
            "generation": int(manifest["generation"]),
            "stable_count": len(stable),
            "sha256": _sha256(registry.path),
        },
        {
            "edge_count": int(link_index["edge_count"]),
            "unresolved_count": int(link_index["unresolved_count"]),
            "sha256": _sha256(
                root / "runtime" / "librarian" / "uid-link-index.json"
            ),
        },
        corpus,
    )


def _unlink_projection(path: Path) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise ValueError("derived projection path is unsafe")
    path.unlink()


def _derived_workspace(gate: OKFRebuildGate) -> Path:
    """Create and validate the one rebuild staging directory without symlinks."""

    derived = gate.workspace.absolute() / "derived-rebuild"
    with open_directory_nofollow(gate.workspace) as workspace_fd:
        try:
            mode = os.stat(
                derived.name,
                dir_fd=workspace_fd,
                follow_symlinks=False,
            ).st_mode
        except FileNotFoundError:
            os.mkdir(derived.name, mode=0o700, dir_fd=workspace_fd)
            os.fsync(workspace_fd)
            mode = os.stat(
                derived.name,
                dir_fd=workspace_fd,
                follow_symlinks=False,
            ).st_mode
        if not stat.S_ISDIR(mode):
            raise ValueError("derived rebuild workspace is unsafe")
    with open_directory_nofollow(derived):
        pass
    return derived


def _rebuild_text_indexes(
    root: Path,
    gate: OKFRebuildGate,
    stable_paths: list[Path],
    *,
    index_dir: Path,
    reset: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    portable = rebuild_pages_index(root / "pages")
    portable_links = sum(
        line.startswith("- [") for line in portable.decode("utf-8").splitlines()
    )
    if reset:
        for name in (
            "pages.json",
            "backlinks.json",
            "lexical.sqlite",
            "lexical.sqlite-wal",
            "lexical.sqlite-shm",
        ):
            _unlink_projection(index_dir / name)
    metadata = index_store.IndexStore(root, _index_dir=index_dir)
    metadata._refresh_under_external_gate(gate)
    pages_file = index_dir / "pages.json"
    backlinks_file = index_dir / "backlinks.json"
    lexical_file = index_dir / "lexical.sqlite"
    lexical = LexicalIndex(
        path=lexical_file,
        pages=lambda: list(stable_paths),
        refresh_interval_seconds=0,
    )
    lexical.build(force=reset)
    lexical.close()
    connection = sqlite3.connect(f"file:{lexical_file}?mode=ro", uri=True)
    try:
        lexical_ids = {
            str(row[0]) for row in connection.execute("SELECT page_id FROM pages")
        }
    finally:
        connection.close()
    expected_ids = {path.stem for path in stable_paths}
    expected_portable = sum(
        path.is_relative_to(root / "pages") for path in stable_paths
    )
    if portable_links != expected_portable:
        raise ValueError("portable index does not cover the exact page corpus")
    if metadata.all_page_ids(include_system=True) != expected_ids:
        raise ValueError("IndexStore does not cover the exact stable corpus")
    if lexical_ids != expected_ids:
        raise ValueError("lexical index does not cover the exact stable corpus")
    return (
        {
            "page_count": expected_portable,
            "link_count": portable_links,
            "sha256": hashlib.sha256(portable).hexdigest(),
        },
        {
            "page_count": len(metadata.all_page_ids(include_system=True)),
            "pages_sha256": _sha256(pages_file),
            "backlinks_sha256": _sha256(backlinks_file),
        },
        {"page_count": len(lexical_ids), "sha256": _sha256(lexical_file)},
    )


def _production_semantic_encoder() -> tuple[
    SemanticEncoder,
    dict[str, str | int],
    LLMRuntime,
]:
    config = load_search_embedding_config()
    if not config.enabled:
        raise RuntimeError("configured semantic embedding is disabled")
    runtime = load_default_llm_runtime()
    route = runtime.resolve_embedding(_FOREGROUND_ROLE)
    if route.location is not RouteLocation.LOCAL:
        raise RuntimeError("offline semantic rebuild requires a local embedding route")

    def encode(
        documents: Sequence[semantic_index.SemanticDocument], batch_size: int
    ) -> np.ndarray:
        vectors: list[tuple[float, ...]] = []
        for offset in range(0, len(documents), max(1, batch_size)):
            rows = documents[offset : offset + max(1, batch_size)]
            result = runtime.embed(
                _FOREGROUND_ROLE,
                EmbeddingRequest(
                    tuple(document.text for document in rows),
                    _SEMANTIC_SOURCE,
                    purpose=EmbeddingPurpose.DOCUMENT,
                ),
            )
            vectors.extend(result.vectors)
        return np.asarray(vectors, dtype=np.float32)

    return (
        encode,
        {
            "role": route.role,
            "provider": route.provider,
            "model": route.model,
            "location": route.location.value,
            "revision": config.revision,
            "dimensions": config.dimensions,
            "query_prefix": config.query_prefix,
            "document_prefix": config.document_prefix,
            "batch_size": config.maintenance_max_batch,
        },
        runtime,
    )


def _semantic_metadata_identity(
    generation_id: str,
    *,
    root: Path,
) -> dict[str, set[str]]:
    metadata = semantic_index.generation_dir(generation_id, root=root) / "metadata.sqlite"
    connection = sqlite3.connect(f"file:{metadata}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT page_id, source_sha256 FROM documents"
        ).fetchall()
    finally:
        connection.close()
    result: dict[str, set[str]] = {}
    for page_id, source_sha256 in rows:
        result.setdefault(str(page_id), set()).add(str(source_sha256))
    return result


def _rebuild_semantic(
    root: Path,
    gate: OKFRebuildGate,
    stable_paths: list[Path],
    *,
    index_dir: Path,
    derived_generation: str,
    encoder: SemanticEncoder | None,
    profile: Mapping[str, str | int] | None,
    allow_build: bool = True,
) -> dict[str, Any]:
    runtime: LLMRuntime | None = None
    if encoder is None or profile is None:
        if encoder is not None or profile is not None:
            raise ValueError("semantic encoder and profile must be injected together")
        encoder, resolved_profile, runtime = _production_semantic_encoder()
        profile = resolved_profile
    expected_fields = {
        "role",
        "provider",
        "model",
        "location",
        "revision",
        "dimensions",
        "query_prefix",
        "document_prefix",
        "batch_size",
    }
    if set(profile) != expected_fields or profile.get("location") != "local":
        raise ValueError("offline semantic profile is invalid")
    documents = semantic_index.extract_all_documents(stable_paths)
    if not documents:
        raise ValueError("offline semantic rebuild requires a non-empty stable corpus")
    expected_pages: dict[str, str] = {}
    for document in documents:
        prior = expected_pages.setdefault(document.page_id, document.source_sha256)
        if prior != document.source_sha256:
            raise ValueError("semantic corpus has mixed source identities")
    semantic_root = index_dir / "semantic"
    profile_sha256 = canonical_json_sha256_strict(dict(profile))
    marker = (
        f"okf-{gate.manifest_sha256[:12]}-"
        f"{derived_generation.removeprefix('okf-')[:12]}-{profile_sha256[:12]}"
    )
    old_active_id = str(
        semantic_index.read_active(root=semantic_root).get("generation_id") or ""
    )
    generation_id = ""
    generations = semantic_root / "generations"
    if generations.is_dir():
        for path in sorted(generations.iterdir()):
            if not path.is_dir():
                continue
            try:
                candidate = semantic_index.validate_generation(path.name, root=semantic_root)
            except Exception:
                continue
            if candidate.repo_commit == marker:
                generation_id = candidate.generation_id
                break
    try:
        if not generation_id:
            if not allow_build:
                raise ValueError("published semantic generation identity changed")
            manifest = semantic_index.build_generation(
                documents,
                encode_documents=encoder,
                role=str(profile["role"]),
                provider=str(profile["provider"]),
                model=str(profile["model"]),
                location=str(profile["location"]),
                revision=str(profile["revision"]),
                dimensions=int(profile["dimensions"]),
                query_prefix=str(profile["query_prefix"]),
                document_prefix=str(profile["document_prefix"]),
                batch_size=int(profile["batch_size"]),
                root=semantic_root,
                repo_commit=marker,
            )
            generation_id = manifest.generation_id
        manifest = semantic_index.validate_generation(
            generation_id,
            root=semantic_root,
            expected_route={
                "role": str(profile["role"]),
                "provider": str(profile["provider"]),
                "model": str(profile["model"]),
                "location": str(profile["location"]),
            },
        )
        if (
            manifest.page_count != len(expected_pages)
            or manifest.document_count != len(documents)
            or _semantic_metadata_identity(generation_id, root=semantic_root)
            != {page_id: {digest} for page_id, digest in expected_pages.items()}
        ):
            raise ValueError("semantic generation does not cover the exact corpus")
        active = semantic_index.read_active(root=semantic_root)
        active_id = str(active.get("generation_id") or "")
        if active_id != generation_id:
            semantic_index.activate_generation(
                generation_id,
                expected_current=old_active_id,
                root=semantic_root,
            )
        manifest_path = (
            semantic_index.generation_dir(generation_id, root=semantic_root)
            / "manifest.json"
        )
        return {
            "page_count": manifest.page_count,
            "document_count": manifest.document_count,
            "corpus_sha256": manifest.corpus_fingerprint,
            "generation_sha256": canonical_json_sha256_strict(asdict(manifest)),
            "manifest_sha256": _sha256(manifest_path),
        }
    finally:
        if runtime is not None:
            runtime.release_embedding(_FOREGROUND_ROLE)


def _rebuild_index_projection(
    root: Path,
    gate: OKFRebuildGate,
    stable_paths: list[Path],
    *,
    derived_generation: str,
    encoder: SemanticEncoder | None,
    profile: Mapping[str, str | int] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    derived = _derived_workspace(gate)
    live = root / ".index"
    staged = derived / "index"
    previous = derived / "previous-index"

    def directory_kind(path: Path) -> str:
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            return "absent"
        if not stat.S_ISDIR(mode):
            raise ValueError("derived index directory is unsafe")
        semantic = path / "semantic"
        try:
            semantic_mode = os.lstat(semantic).st_mode
        except FileNotFoundError:
            return "directory"
        if not stat.S_ISDIR(semantic_mode):
            raise ValueError("derived semantic directory is unsafe")
        return "directory"

    live_kind = directory_kind(live)
    staged_kind = directory_kind(staged)
    previous_kind = directory_kind(previous)
    reuse_published = live_kind == previous_kind == "directory"
    if reuse_published:
        if staged_kind != "absent":
            raise ValueError("derived index exists in mixed generations")
        index_dir = live
    else:
        if previous_kind == "directory" and (
            live_kind != "absent" or staged_kind != "directory"
        ):
            raise ValueError("derived index publication is incomplete")
        if staged_kind == "absent":
            with open_directory_nofollow(derived) as derived_fd:
                os.mkdir(staged.name, mode=0o700, dir_fd=derived_fd)
                os.fsync(derived_fd)
        index_dir = staged

    # ponytail: the exclusive offline lease is the ceiling; if rebuilds become
    # concurrent in-process, move the index and semantic writers to dir_fd APIs.
    portable, metadata, lexical = _rebuild_text_indexes(
        root,
        gate,
        stable_paths,
        index_dir=index_dir,
        reset=not reuse_published,
    )
    semantic = _rebuild_semantic(
        root,
        gate,
        stable_paths,
        index_dir=index_dir,
        derived_generation=derived_generation,
        encoder=encoder,
        profile=profile,
        allow_build=not reuse_published,
    )
    if not reuse_published:
        if directory_kind(live) != live_kind:
            raise ValueError("live derived index changed during rebuild")
        if live_kind == previous_kind == "absent":
            with open_directory_nofollow(derived) as derived_fd:
                os.mkdir(previous.name, mode=0o700, dir_fd=derived_fd)
                os.fsync(derived_fd)
        _retire_directory(live, previous)
        _publish_directory(staged, live)
    return portable, metadata, lexical, semantic


def _kg_matches(
    store: KnowledgeGraphStore,
    expected_digests: Mapping[str, str],
    derived_generation: str,
) -> bool:
    try:
        state = read_sealed_json(store.builder_state_file, recover_backup=False)
    except Exception:
        return False
    return (
        state.get("policy_version") == GRAPH_BUILDER_POLICY_VERSION
        and state.get("external_model_calls") == 0
        and state.get("page_digests") == expected_digests
        and state.get("extractor_route_identity")
        == DETERMINISTIC_EXTRACTOR_IDENTITY
        and state.get("okf_derived_generation") == derived_generation
    )


def _publish_directory(source: Path, target: Path) -> None:
    """Move one exact regular directory while the offline root lease is held."""

    source = source.absolute()
    target = target.absolute()

    def publish(source_fd: int, target_fd: int) -> None:
        try:
            source_mode = os.stat(
                source.name, dir_fd=source_fd, follow_symlinks=False
            ).st_mode
        except FileNotFoundError:
            raise ValueError(
                f"derived publication source is missing: {source}"
            ) from None
        if not stat.S_ISDIR(source_mode):
            raise ValueError("derived publication source is unsafe")
        try:
            os.stat(target.name, dir_fd=target_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("derived publication target already exists")
        os.rename(
            source.name,
            target.name,
            src_dir_fd=source_fd,
            dst_dir_fd=target_fd,
        )
        os.fsync(source_fd)
        if target_fd != source_fd:
            os.fsync(target_fd)

    with open_directory_nofollow(source.parent) as source_fd:
        if source.parent == target.parent:
            publish(source_fd, source_fd)
        else:
            with open_directory_nofollow(target.parent) as target_fd:
                publish(source_fd, target_fd)


def _retire_directory(source: Path, target: Path) -> None:
    """Retire one optional derived directory exactly once for crash resume."""

    source = source.absolute()
    target = target.absolute()

    def retire(source_fd: int, target_fd: int) -> None:
        try:
            source_mode = os.stat(
                source.name, dir_fd=source_fd, follow_symlinks=False
            ).st_mode
        except FileNotFoundError:
            source_mode = 0
        try:
            target_mode = os.stat(
                target.name, dir_fd=target_fd, follow_symlinks=False
            ).st_mode
        except FileNotFoundError:
            target_mode = 0
        if source_mode and not stat.S_ISDIR(source_mode):
            raise ValueError("derived state source is unsafe")
        if target_mode and not stat.S_ISDIR(target_mode):
            raise ValueError("derived state retirement target is unsafe")
        if source_mode and target_mode:
            raise ValueError("derived state exists in two generations")
        if source_mode:
            os.rename(
                source.name,
                target.name,
                src_dir_fd=source_fd,
                dst_dir_fd=target_fd,
            )
            os.fsync(source_fd)
            if target_fd != source_fd:
                os.fsync(target_fd)

    with open_directory_nofollow(source.parent) as source_fd:
        if source.parent == target.parent:
            retire(source_fd, source_fd)
        else:
            with open_directory_nofollow(target.parent) as target_fd:
                retire(source_fd, target_fd)


def _logical_relation_identity(relation: Any) -> dict[str, Any]:
    """Exclude ledger clocks, event IDs, and runtime usage from rebuild identity."""

    value = relation.to_dict()
    fields = (
        "relation_id",
        "source_page_id",
        "target_page_id",
        "predicate",
        "direction",
        "status",
        "evidence",
        "model_sha256",
        "rubric_sha256",
        "producer_role",
        "confidence",
        "valid_from",
        "valid_to",
        "reason_code",
    )
    return {field: value[field] for field in fields}


def _rebuild_knowledge_graph(
    root: Path,
    gate: OKFRebuildGate,
    stable_paths: list[Path],
    *,
    derived_generation: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    digests = {
        path.stem: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in stable_paths
    }
    live = root / "knowledge-graph"
    derived = _derived_workspace(gate)
    staged = derived / "knowledge-graph"
    previous = derived / "previous-knowledge-graph"
    live_store = KnowledgeGraphStore(live)
    if not _kg_matches(live_store, digests, derived_generation):
        staged_store = KnowledgeGraphStore(staged)
        config = KnowledgeGraphConfig(
            enabled=True,
            mode="shadow",
            local_extraction_enabled=False,
            max_changed_pages_per_cycle=max(1, len(stable_paths)),
            max_queue_size=max(1, len(stable_paths)),
            max_community_summaries_per_cycle=0,
        )
        result = run_builder_cycle(root=root, config=config, store=staged_store)
        if (
            result.get("status") != "ok"
            or result.get("remaining_pages") != 0
            or result.get("external_model_calls") != 0
        ):
            raise RuntimeError("deterministic knowledge graph rebuild is incomplete")
        staged_store.materialize_snapshot()
        builder_state = read_sealed_json(
            staged_store.builder_state_file,
            recover_backup=False,
        )
        write_sealed_json(
            staged_store.builder_state_file,
            {
                **builder_state,
                "okf_derived_generation": derived_generation,
            },
            backup=False,
        )
        if not _kg_matches(staged_store, digests, derived_generation):
            raise RuntimeError("deterministic knowledge graph identity is incomplete")
        _retire_directory(live, previous)
        _publish_directory(staged, live)
        live_store = KnowledgeGraphStore(live)
    relations = live_store.relations()
    if any(
        relation.producer_role != "deterministic"
        or relation.status != "proposed"
        for relation in relations
    ):
        raise ValueError("fresh knowledge graph contains non-deterministic authority")
    typed_graph = root / "runtime" / "typed-graph"
    previous_typed_graph = derived / "previous-typed-graph"
    _retire_directory(typed_graph, previous_typed_graph)
    cortex = build_cortex_graph(
        root,
        generated=f"offline-rebuild:{derived_generation}",
        use_cache=False,
    )
    typed = cortex.get("typedGraph")
    typed = typed if isinstance(typed, Mapping) else {}
    status = typed.get("status")
    status = status if isinstance(status, Mapping) else {}
    authority = status.get("authority")
    authority_enabled = bool(
        status.get("authority_mature") is True
        or (isinstance(authority, Mapping) and authority.get("enabled") is True)
    )
    if authority_enabled:
        raise ValueError("fresh typed graph unexpectedly has mutation authority")
    runtime_state_present = typed_graph.exists()
    if runtime_state_present:
        raise ValueError("fresh typed graph retained runtime authority state")
    if len(cortex.get("nodes") or []) != len(stable_paths):
        raise ValueError("Cortex graph does not cover the exact stable corpus")
    return (
        {
            "page_count": len(digests),
            "relation_count": len(relations),
            "relation_set_sha256": canonical_json_sha256_strict(
                [_logical_relation_identity(relation) for relation in relations]
            ),
            "snapshot_sha256": _sha256(live_store.snapshot_file),
            "builder_sha256": _sha256(live_store.builder_state_file),
            "external_model_calls": 0,
        },
        {
            "node_count": len(cortex.get("nodes") or []),
            "link_count": len(cortex.get("links") or []),
            "typed_relation_count": len(typed.get("relations") or []),
            "sha256": canonical_json_sha256_strict(cortex),
            "authority_enabled": False,
            "runtime_state_present": False,
        },
    )


def rebuild_okf_derived(
    root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
    semantic_encoder: SemanticEncoder | None = None,
    semantic_profile: Mapping[str, str | int] | None = None,
) -> dict[str, Any]:
    """Rebuild and seal every page-derived projection without opening startup."""

    root = root.expanduser().absolute()
    with okf_rebuild_session(
        root,
        root / "runtime",
        run_id,
        is_quiescent=is_quiescent,
    ) as session:
        if session.gate.derived_generation is not None:
            return {
                "status": "sealed-rebuild",
                "run_id": run_id,
                "derived_generation": session.gate.derived_generation,
                "rebuild_proof_sha256": session.gate.rebuild_proof_sha256,
                "stable_page_count": session.gate.stable_page_count,
            }
        stable_paths, source_rows = _stable_sources(root)
        registry, uid_links, corpus = _rebuild_registry_and_links(
            root,
            session.gate,
            source_rows,
        )
        derived_generation = "okf-" + canonical_json_sha256_strict(
            {
                "schema": _DERIVED_GENERATION_SCHEMA,
                "manifest": session.gate.manifest_sha256,
                "corpus": corpus,
            }
        )[:24]
        portable, metadata, lexical, semantic = _rebuild_index_projection(
            root,
            session.gate,
            stable_paths,
            derived_generation=derived_generation,
            encoder=semantic_encoder,
            profile=semantic_profile,
        )
        knowledge_graph, cortex = _rebuild_knowledge_graph(
            root,
            session.gate,
            stable_paths,
            derived_generation=derived_generation,
        )
        invalidation = _manifest_invalidation(session.gate)
        components = {
            "registry": registry,
            "uid_links": uid_links,
            "portable_index": portable,
            "index_store": metadata,
            "lexical": lexical,
            "semantic": semantic,
            "knowledge_graph": knowledge_graph,
            "cortex": cortex,
            "invalidation": invalidation,
        }
        proof_sha256 = session.publish_proof(
            {
                "derived_generation": derived_generation,
                "corpus": corpus,
                "components": components,
            }
        )
        session.seal(
            derived_generation=derived_generation,
            rebuild_proof_sha256=proof_sha256,
        )
        return {
            "status": "sealed-rebuild",
            "run_id": run_id,
            "derived_generation": derived_generation,
            "rebuild_proof_sha256": proof_sha256,
            "stable_page_count": corpus["stable_page_count"],
        }
