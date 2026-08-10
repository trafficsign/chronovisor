"""Resolve generated ingest operations into exact page preimages/postimages."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from chronovisor.core.canonical_document import (
    CanonicalDocumentError,
    Namespace,
    parse_document,
    patch_document_metadata,
    validate_canonical_document,
)
from chronovisor.core.index_store import (
    PAGE_RESERVED_FILENAMES,
    SYSTEM_RESERVED_FILENAMES,
)


def _runtime() -> ModuleType:
    from chronovisor.ingest import ingest

    return ingest


def _runtime_call(name: str) -> Any:
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(_runtime(), name)(*args, **kwargs)

    return call


_ensure_page_metadata_frontmatter = _runtime_call("_ensure_page_metadata_frontmatter")
_find_page_resilient = _runtime_call("_find_page_resilient")
_normalize_for_collision = _runtime_call("_normalize_for_collision")
_process_tags_in_body = _runtime_call("_process_tags_in_body")
_read_exact_utf8 = _runtime_call("_read_exact_utf8")
_reconcile_links = _runtime_call("_reconcile_links")
_reserved_system_page_collision_keys = _runtime_call(
    "_reserved_system_page_collision_keys"
)
_safe_log = _runtime_call("_safe_log")
_safe_resolve_page_path = _runtime_call("_safe_resolve_page_path")
_strip_all_frontmatter = _runtime_call("_strip_all_frontmatter")

from chronovisor.ingest.ingest import (  # noqa: E402
    IngestApplyError,
    PreparedIngestOperation,
)


def _canonical_parse(text: str) -> tuple[dict[str, Any], str]:
    document = parse_document(text.encode("utf-8"))
    return document.metadata, document.body.decode("utf-8")


def _canonical_patch(text: str, updates: dict[str, Any]) -> str:
    return patch_document_metadata(text.encode("utf-8"), updates).decode("utf-8")


def _page_relative_path(path: Path) -> str:
    return (
        path.resolve(strict=False)
        .relative_to(_runtime().PAGES_DIR.resolve(strict=False))
        .as_posix()
    )


def _prepare_context(
    *, read_only: bool
) -> tuple[set[tuple[Namespace, str]], set[str], list[str]]:
    if read_only:
        # ``IndexStore.refresh`` may persist derived cache files. A dry run must
        # leave runtime/index artifacts untouched, so scan the corpus directly.
        from chronovisor.core import store as _wiki

        page_paths = [
            path
            for path in _runtime().PAGES_DIR.rglob("*.md")
            if path.name not in PAGE_RESERVED_FILENAMES
        ]
        system_paths = [
            path
            for path in _wiki.SYSTEM_DIR.rglob("*.md")
            if path.name not in SYSTEM_RESERVED_FILENAMES
        ]
        allowed_targets: set[tuple[Namespace, str]] = set()
        for raw_namespace, root, paths in (
            ("pages", _runtime().PAGES_DIR, page_paths),
            ("system", _wiki.SYSTEM_DIR, system_paths),
        ):
            namespace = cast(Namespace, raw_namespace)
            for path in paths:
                relative_path = (
                    path.resolve(strict=False)
                    .relative_to(root.resolve(strict=False))
                    .as_posix()
                )
                document = validate_canonical_document(
                    path.read_bytes(),
                    namespace=namespace,
                    path=relative_path,
                )
                if document.metadata["status"] == "stable":
                    allowed_targets.add((namespace, relative_path))
        reserved_system_ids = {
            _normalize_for_collision(path.stem) for path in system_paths
        }
        tag_values: set[str] = set()
        for path in page_paths:
            meta, _body = _canonical_parse(path.read_text(encoding="utf-8"))
            tags = meta.get("tags")
            if isinstance(tags, list):
                tag_values.update(tag for tag in tags if isinstance(tag, str))
        return allowed_targets, reserved_system_ids, sorted(tag_values)

    from chronovisor.core.index_store import get_store

    store = get_store()
    store.refresh()
    allowed_targets = set()
    for page_id in store.all_page_ids(include_system=True):
        indexed_meta = store.meta(page_id)
        if not isinstance(indexed_meta, dict):
            continue
        indexed_namespace = indexed_meta.get("namespace")
        relative_path = indexed_meta.get("relative_path")
        if indexed_namespace in {"pages", "system"} and isinstance(relative_path, str):
            allowed_targets.add((cast(Namespace, indexed_namespace), relative_path))
    reserved_system_ids = {
        _normalize_for_collision(page_id)
        for page_id in (
            store.all_page_ids(include_system=True)
            - store.all_page_ids(include_system=False)
        )
    }
    return (
        allowed_targets,
        reserved_system_ids,
        store.all_tags(include_system=False),
    )


def _prepare_operation(
    source_operation_index: int,
    op: dict[str, Any],
    *,
    allowed_targets: set[tuple[Namespace, str]],
    existing_tags_snapshot: list[str],
    read_only: bool,
    frontmatter_parse: Any,
    frontmatter_patch: Any,
) -> tuple[PreparedIngestOperation, dict[str, int]]:
    """Resolve one generated operation into an exact mutation proposal."""

    source_operation_type = op["type"]
    source_filename = op["filename"]
    op_type = source_operation_type
    full_path = _safe_resolve_page_path(source_filename)
    page_id = full_path.stem
    body = op["content"]

    op_raw_keywords = op.get("raw_keywords")
    propagate_raw_keywords = (
        isinstance(op_raw_keywords, list)
        and all(isinstance(value, str) for value in op_raw_keywords)
        and bool(op_raw_keywords)
    )
    raw_keywords = cast(list[str], op_raw_keywords) if propagate_raw_keywords else []

    if op_type == "create":
        existing = _find_page_resilient(page_id, emit_logs=not read_only)
        if existing is not None:
            if not read_only:
                _safe_log(
                    f"ingest | create op for existing page_id {page_id!r} "
                    f"converted to update (existing: {existing}, target: {full_path})"
                )
            op_type = "update"
            full_path = existing
            page_id = existing.stem
            body = _strip_all_frontmatter(body).strip()
            if not body:
                raise IngestApplyError(
                    f"create collision for page_id {page_id!r} produced no update body"
                )

    body, stats = _reconcile_links(
        body,
        allowed_targets,
        source_path=_page_relative_path(full_path),
    )

    if op_type == "create":
        body = _process_tags_in_body(
            body,
            existing_tags_snapshot,
            frontmatter_parse,
            frontmatter_patch,
            record_changes=False,
        )
        if propagate_raw_keywords:
            body = frontmatter_patch(body, {"raw_keywords": raw_keywords})
        body = frontmatter_patch(body, {"updated": date.today().isoformat()})
        try:
            validate_canonical_document(
                body.encode("utf-8"),
                namespace="pages",
                path=_page_relative_path(full_path),
                require_stable=True,
                allowed_targets=allowed_targets,
            )
        except CanonicalDocumentError as exc:
            raise IngestApplyError(
                f"create proposal is not canonical: {page_id}: {exc}"
            ) from exc
        created_meta, _created_body = frontmatter_parse(body)
        created_tags = created_meta.get("tags")
        new_tags = tuple(
            tag
            for tag in (created_tags if isinstance(created_tags, list) else [])
            if isinstance(tag, str) and tag not in set(existing_tags_snapshot)
        )
        prepared = PreparedIngestOperation(
            op_type="create",
            path=full_path,
            page_id=page_id,
            new_body=body.rstrip() + "\n",
            previous_text=None,
            new_tags=new_tags,
            source_operation_index=source_operation_index,
            source_operation_type=source_operation_type,
            source_filename=source_filename,
        )
        return prepared, stats

    existing_path = (
        full_path
        if full_path.exists()
        else _find_page_resilient(page_id, emit_logs=not read_only)
    )
    if existing_path is None or not existing_path.exists():
        raise IngestApplyError(f"update target not found for page_id {page_id!r}")
    page_id = existing_path.stem
    previous = _read_exact_utf8(existing_path)
    try:
        validate_canonical_document(
            previous.encode("utf-8"),
            namespace="pages",
            path=_page_relative_path(existing_path),
            require_stable=True,
        )
    except CanonicalDocumentError as exc:
        raise IngestApplyError(
            f"update target is not stable canonical: {page_id}: {exc}"
        ) from exc
    compact_preimage_sha256 = op.get("_compact_update_preimage_sha256")
    if compact_preimage_sha256 is not None:
        if (
            not isinstance(compact_preimage_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", compact_preimage_sha256) is None
        ):
            raise IngestApplyError(
                f"compact update preimage binding is malformed for {page_id}"
            )
        observed_preimage_sha256 = hashlib.sha256(previous.encode("utf-8")).hexdigest()
        if observed_preimage_sha256 != compact_preimage_sha256:
            raise IngestApplyError(
                f"compact update preimage changed before prepare: {page_id}"
            )
    previous_text_for_rollback = previous
    if propagate_raw_keywords:
        existing_meta, _existing_body = frontmatter_parse(previous)
        existing_kw_raw = existing_meta.get("raw_keywords")
        existing_kw = (
            existing_kw_raw
            if isinstance(existing_kw_raw, list)
            and all(isinstance(value, str) for value in existing_kw_raw)
            else []
        )
        union_kw = list(dict.fromkeys(existing_kw + raw_keywords))
        previous = frontmatter_patch(previous, {"raw_keywords": union_kw})

    stamped = frontmatter_patch(
        previous,
        {"updated": date.today().isoformat()},
    )
    separator = (
        "" if stamped.endswith("\n\n") else ("\n" if stamped.endswith("\n") else "\n\n")
    )
    proposed = stamped + separator + body.rstrip() + "\n"
    try:
        validate_canonical_document(
            proposed.encode("utf-8"),
            namespace="pages",
            path=_page_relative_path(existing_path),
            require_stable=True,
            allowed_targets=allowed_targets,
        )
    except CanonicalDocumentError as exc:
        raise IngestApplyError(
            f"update proposal is not canonical: {page_id}: {exc}"
        ) from exc
    return (
        PreparedIngestOperation(
            op_type="update",
            path=existing_path,
            page_id=page_id,
            new_body=proposed,
            previous_text=previous_text_for_rollback,
            source_operation_index=source_operation_index,
            source_operation_type=source_operation_type,
            source_filename=source_filename,
        ),
        stats,
    )


def prepare_operations(
    operations: list[dict[str, Any]],
    *,
    read_only: bool = False,
) -> tuple[list[PreparedIngestOperation], dict[str, int]]:
    """Resolve local proposals into exact page preimages and postimages.
    This stage is read-only with respect to Wiki pages.  Ollama triage and
    generation are proposals only; the returned byte-exact plan is what the
    frontier model reviews before :func:`_apply_prepared_operations` may run.
    Fail-closed: any unrecoverable problem raises :class:`IngestApplyError`.
    The caller marks the job FAILED without invoking ``on_complete``.

    Phase 4 propagation: any op that carries a non-empty ``raw_keywords``
    list (from the source raw's frontmatter, riding on metadata since
    Phase 3) gets that list patched onto the page frontmatter inside the
    prepare phase — never inside the write phase, so a partial-write
    rollback restores either the pre-batch text or nothing at all, never
    a half-patched frontmatter.

    Plan-4 tag processing: ``create`` op bodies whose generated frontmatter
    already includes a ``tags:`` list (per ``GENERATE_SYSTEM_PROMPT``) get
    each tag form-validated, dedup'd against the existing corpus's tag
    pool (cosine similarity >= 0.80 → reuse), and audited via
    ``tag-changelog.md``. ``update`` ops never touch ``tags`` because
    ``UPDATE_SYSTEM_PROMPT`` forbids the LLM from emitting frontmatter.
    """
    try:
        allowed_targets, reserved_system_ids, existing_tags_snapshot = _prepare_context(
            read_only=read_only
        )
    except Exception as e:
        raise IngestApplyError(f"index_store unavailable: {e}") from e

    # Keep the final mutation boundary fail-closed even when an installed core
    # system page is temporarily absent from the index snapshot.
    reserved_system_ids.update(_reserved_system_page_collision_keys())

    # ---- Prepare phase -----------------------------------------------------
    # Resolve every filename, validate every op, build the final write plan.
    # Nothing here touches disk except for read-only stat/read calls.

    planned: list[PreparedIngestOperation] = []
    seen_norm_ids: set[str] = set()
    seen_paths: set[Path] = set()

    for op in operations:
        op_type = op.get("type")
        if op_type not in ("create", "update"):
            raise IngestApplyError(f"unknown op type: {op_type!r}")

        full_path = _safe_resolve_page_path(op["filename"])
        page_id = full_path.stem

        # Defense in depth: reviewed/replayed artifacts can reach prepare
        # without passing through today's triage validator.  Never let a
        # create operation write directly under pages/, even on those paths.
        relative_parts = full_path.relative_to(_runtime().PAGES_DIR.resolve()).parts
        if op_type == "create" and len(relative_parts) != 2:
            raise IngestApplyError(
                "create target must use exactly one top-level folder "
                f"(expected folder/page-id.md): {op['filename']!r}"
            )

        # Detect intra-batch dups using the same case/Unicode-insensitive key
        # we use against the existing corpus, so two ops whose ids differ
        # only in case or NFC/NFD form are caught before any write.
        norm_key = _normalize_for_collision(page_id)
        if norm_key in reserved_system_ids:
            raise IngestApplyError(
                f"reserved system page_id cannot be mutated by ingest: {page_id!r}"
            )
        if norm_key in seen_norm_ids:
            raise IngestApplyError(
                f"duplicate page_id within batch (case/Unicode-insensitive): "
                f"{page_id!r}"
            )
        if full_path in seen_paths:
            raise IngestApplyError(f"duplicate target path within batch: {full_path}")
        seen_norm_ids.add(norm_key)
        seen_paths.add(full_path)

        target_path = full_path
        if op_type == "create":
            existing = _find_page_resilient(page_id, emit_logs=False)
            if existing is not None:
                target_path = existing
        allowed_targets.add(("pages", _page_relative_path(target_path)))

    totals = {"resolved": 0, "rewritten": 0, "unwrapped": 0}

    for source_operation_index, op in enumerate(operations):
        prepared, stats = _prepare_operation(
            source_operation_index,
            op,
            allowed_targets=allowed_targets,
            existing_tags_snapshot=existing_tags_snapshot,
            read_only=read_only,
            frontmatter_parse=_canonical_parse,
            frontmatter_patch=_canonical_patch,
        )
        planned.append(prepared)
        for key in totals:
            totals[key] += stats[key]

    # Apply every currently active correction tombstone to the exact proposal
    # *before* frontier review, including creates under a brand-new slug.
    # The lock-time pass below then acts only as a staleness detector.
    constrained_plans: list[PreparedIngestOperation] = []
    from chronovisor.core.page_mutation import (
        PageMutationError,
        enforce_correction_constraints,
    )

    for entry in planned:
        try:
            constrained_body, enforced = enforce_correction_constraints(
                entry.page_id,
                entry.previous_text or "",
                entry.new_body,
            )
            # Recall metadata is derived only after active correction
            # tombstones have canonicalized the page.  If a stale claim was
            # rewritten, replace summary/questions deterministically from the
            # corrected body so an LLM paraphrase cannot resurrect it.  Dry
            # runs also stay byte-read-only by avoiding model audit artifacts.
            constrained_body = _ensure_page_metadata_frontmatter(
                constrained_body,
                entry.page_id,
                _canonical_parse,
                _canonical_patch,
                allow_local_model=not read_only and not enforced,
                force_deterministic_rebuild=bool(enforced),
            )
            constrained_body, metadata_enforced = enforce_correction_constraints(
                entry.page_id,
                entry.previous_text or "",
                constrained_body,
            )
            validate_canonical_document(
                constrained_body.encode("utf-8"),
                namespace="pages",
                path=_page_relative_path(entry.path),
                require_stable=True,
                allowed_targets=allowed_targets,
            )
        except (CanonicalDocumentError, PageMutationError) as exc:
            raise IngestApplyError(
                f"content correction constraint failed for {entry.page_id}: {exc}"
            ) from exc
        all_enforced = [*enforced, *metadata_enforced]
        if all_enforced and not read_only:
            _safe_log(
                f"ingest | enforced {len(all_enforced)} global content correction(s) "
                f"for {entry.page_id}"
            )
        constrained_plans.append(
            PreparedIngestOperation(
                op_type=entry.op_type,
                path=entry.path,
                page_id=entry.page_id,
                new_body=constrained_body,
                previous_text=entry.previous_text,
                new_tags=entry.new_tags,
                source_operation_index=entry.source_operation_index,
                source_operation_type=entry.source_operation_type,
                source_filename=entry.source_filename,
            )
        )

    return constrained_plans, totals
