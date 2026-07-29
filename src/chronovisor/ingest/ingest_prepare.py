"""Resolve generated ingest operations into exact page preimages/postimages."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Any


def _runtime():
    from chronovisor import ingest

    return ingest


def _runtime_call(name: str):
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(_runtime(), name)(*args, **kwargs)

    return call


_ensure_page_metadata_frontmatter = _runtime_call("_ensure_page_metadata_frontmatter")
_find_page_resilient = _runtime_call("_find_page_resilient")
_normalize_for_collision = _runtime_call("_normalize_for_collision")
_process_tags_in_body = _runtime_call("_process_tags_in_body")
_read_exact_utf8 = _runtime_call("_read_exact_utf8")
_reconcile_links = _runtime_call("_reconcile_links")
_reserved_system_page_collision_keys = _runtime_call("_reserved_system_page_collision_keys")
_safe_log = _runtime_call("_safe_log")
_safe_resolve_page_path = _runtime_call("_safe_resolve_page_path")
_strip_all_frontmatter = _runtime_call("_strip_all_frontmatter")

from chronovisor.ingest.ingest import (  # noqa: E402
    IngestApplyError,
    PreparedIngestOperation,
)


def prepare_operations(
    operations: list[dict],
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
    from chronovisor.core.frontmatter import (
        parse as _frontmatter_parse,
        patch as _frontmatter_patch,
    )

    # Build the universe of valid link targets: every existing page plus every
    # page about to be created in this batch (so siblings can cross-reference).
    # Fail closed — stale or missing index would silently unwrap every link.
    try:
        if read_only:
            # ``IndexStore.refresh`` may persist derived cache files.  A dry
            # run must leave even runtime/index artifacts untouched, so scan
            # and parse the small corpus directly instead.
            from chronovisor import store as _wiki

            page_paths = list(_runtime().PAGES_DIR.rglob("*.md"))
            system_paths = list(_wiki.SYSTEM_DIR.rglob("*.md"))
            allowed_ids = {path.stem for path in [*page_paths, *system_paths]}
            reserved_system_ids = {
                _normalize_for_collision(path.stem) for path in system_paths
            }
            tag_values: set[str] = set()
            for path in page_paths:
                meta, _body = _frontmatter_parse(path.read_text(encoding="utf-8"))
                tags = meta.get("tags")
                if isinstance(tags, list):
                    tag_values.update(tag for tag in tags if isinstance(tag, str))
            existing_tags_snapshot = sorted(tag_values)
        else:
            from chronovisor.search.index_store import get_store

            store = get_store()
            store.refresh()
            allowed_ids = store.all_page_ids(include_system=True)
            reserved_system_ids = {
                _normalize_for_collision(page_id)
                for page_id in (
                    store.all_page_ids(include_system=True)
                    - store.all_page_ids(include_system=False)
                )
            }
            # Snapshot the tag pool once for the whole batch so dedupe doesn't
            # re-walk the index on every op. Same-batch siblings can't see
            # each other's newly-coined tags here, but that's fine: dedup is
            # only meaningful against the *committed* corpus, and within-batch
            # divergence will be reconciled the next time chronovisor_check runs.
            existing_tags_snapshot = store.all_tags(include_system=False)
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

        allowed_ids.add(page_id)

    totals = {"resolved": 0, "rewritten": 0, "unwrapped": 0}

    for source_operation_index, op in enumerate(operations):
        source_operation_type = op["type"]
        source_filename = op["filename"]
        op_type = source_operation_type
        full_path = _safe_resolve_page_path(op["filename"])
        page_id = full_path.stem

        body, stats = _reconcile_links(op["content"], allowed_ids)
        for k in totals:
            totals[k] += stats[k]

        # Phase 4: lift the raw_keywords side channel off the op. Empty
        # lists are treated as "no propagation" — writing ``raw_keywords:
        # []`` to a page would create a zero-information diff against the
        # existing frontmatter. The propagate flag distinguishes "list[str]
        # with content" from anything else.
        op_raw_keywords = op.get("raw_keywords")
        propagate_raw_keywords = (
            isinstance(op_raw_keywords, list)
            and all(isinstance(v, str) for v in op_raw_keywords)
            and len(op_raw_keywords) > 0
        )

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

        if op_type == "create":
            # Tag processing happens BEFORE raw_keywords patch so the
            # final frontmatter goes through one consistent serialization
            # path. Soft-fail: a missing or malformed ``tags`` list just
            # passes the body through unchanged — chronovisor_check's autonomous
            # lint/repair lane will surface and resolve absent tags.
            body = _process_tags_in_body(
                body,
                existing_tags_snapshot,
                _frontmatter_parse,
                _frontmatter_patch,
                record_changes=False,
            )
            if propagate_raw_keywords:
                # generate output already carries a frontmatter block
                # (enforced by ``_extract_page_body`` for create), so
                # ``patch`` will splice raw_keywords into it without
                # synthesizing a new block.
                body = _frontmatter_patch(body, {"raw_keywords": op_raw_keywords})
            # The model is not a clock. Even when the prompt supplies today's
            # date, enforce it deterministically so a plausible-looking guess
            # can never become page metadata.
            body = _frontmatter_patch(
                body,
                {"updated": date.today().isoformat()},
            )
            created_meta, _created_body = _frontmatter_parse(body)
            created_tags = created_meta.get("tags")
            new_tags = tuple(
                tag
                for tag in (created_tags if isinstance(created_tags, list) else [])
                if isinstance(tag, str) and tag not in set(existing_tags_snapshot)
            )
            planned.append(
                PreparedIngestOperation(
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
            )

        else:  # update
            existing_path = (
                full_path
                if full_path.exists()
                else _find_page_resilient(page_id, emit_logs=not read_only)
            )
            if existing_path is None or not existing_path.exists():
                raise IngestApplyError(
                    f"update target not found for page_id {page_id!r}"
                )
            page_id = existing_path.stem
            previous = _read_exact_utf8(existing_path)
            compact_preimage_sha256 = op.get("_compact_update_preimage_sha256")
            if compact_preimage_sha256 is not None:
                if (
                    not isinstance(compact_preimage_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", compact_preimage_sha256) is None
                ):
                    raise IngestApplyError(
                        f"compact update preimage binding is malformed for {page_id}"
                    )
                observed_preimage_sha256 = hashlib.sha256(
                    previous.encode("utf-8")
                ).hexdigest()
                if observed_preimage_sha256 != compact_preimage_sha256:
                    raise IngestApplyError(
                        f"compact update preimage changed before prepare: {page_id}"
                    )
            # Preserve the on-disk text for rollback BEFORE we mutate
            # ``previous`` with a frontmatter patch — the rollback path
            # restores the file as it was before this batch ran, not as
            # it was after the patch.
            previous_text_for_rollback = previous

            # raw_keywords union with the existing page's value, preserving
            # insertion order so the diff stays deterministic. If the
            # existing field is missing or malformed (legacy data, manual
            # edit), treat it as empty rather than raising — the apply
            # phase shouldn't reject otherwise-valid updates because of
            # frontmatter rot somewhere upstream.
            if propagate_raw_keywords:
                existing_meta, _existing_body = _frontmatter_parse(previous)
                existing_kw_raw = existing_meta.get("raw_keywords")
                if isinstance(existing_kw_raw, list) and all(
                    isinstance(v, str) for v in existing_kw_raw
                ):
                    existing_kw = existing_kw_raw
                else:
                    existing_kw = []
                union_kw = list(dict.fromkeys(existing_kw + op_raw_keywords))
                previous = _frontmatter_patch(previous, {"raw_keywords": union_kw})

            today = date.today().isoformat()
            stamped = re.sub(
                r"updated:\s*.+",
                f"updated: {today}",
                previous,
                count=1,
            )
            new_body = stamped.rstrip() + "\n\n" + body + "\n"
            planned.append(
                PreparedIngestOperation(
                    op_type="update",
                    path=existing_path,
                    page_id=page_id,
                    new_body=new_body,
                    previous_text=previous_text_for_rollback,
                    source_operation_index=source_operation_index,
                    source_operation_type=source_operation_type,
                    source_filename=source_filename,
                )
            )

    # Apply every currently active correction tombstone to the exact proposal
    # *before* frontier review, including creates under a brand-new slug.
    # The lock-time pass below then acts only as a staleness detector.
    constrained_plans: list[PreparedIngestOperation] = []
    from chronovisor.ingest.page_mutation import (
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
                _frontmatter_parse,
                _frontmatter_patch,
                allow_local_model=not read_only and not enforced,
                force_deterministic_rebuild=bool(enforced),
            )
            constrained_body, metadata_enforced = enforce_correction_constraints(
                entry.page_id,
                entry.previous_text or "",
                constrained_body,
            )
        except PageMutationError as exc:
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
