"""CAS-protected application of frontier-approved ingest postimages."""

from __future__ import annotations

from typing import Any


def _runtime():
    from llm_wiki_mcp import ingest

    return ingest


def _runtime_call(name: str):
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(_runtime(), name)(*args, **kwargs)

    return call


_normalize_for_collision = _runtime_call("_normalize_for_collision")
_read_optional_exact_utf8 = _runtime_call("_read_optional_exact_utf8")
_reserved_system_page_collision_keys = _runtime_call("_reserved_system_page_collision_keys")
_safe_log = _runtime_call("_safe_log")

from llm_wiki_mcp.ingest import (  # noqa: E402
    IngestApplyError,
    PreparedIngestOperation,
)


def apply_prepared_operations(
    planned: list[PreparedIngestOperation],
    *,
    link_totals: dict[str, int] | None = None,
    recovery_only: bool = False,
) -> tuple[list[str], list[str]]:
    """Apply an already frontier-approved exact plan with lock-time CAS.

    Current bytes must be either the reviewed preimage or the reviewed
    postimage.  Accepting the latter makes a durable approved proposal
    recoverable after a power loss between a page replace and job completion.
    Any third state is a race and fails closed for autonomous retry.
    """

    from llm_wiki_mcp.link_fix import atomic_write

    written: list[PreparedIngestOperation] = []
    created: list[str] = []
    updated: list[str] = []
    from llm_wiki_mcp.page_mutation import (
        PageMutationError,
        enforce_correction_constraints,
        wiki_mutation_lock,
    )

    # The same lock is used by the autonomous correction lane. This prevents
    # Stop-hook ingest and correction from both passing their read checks and
    # then replacing the same page with different snapshots.
    with wiki_mutation_lock():
        try:
            for entry in planned:
                if (
                    _normalize_for_collision(entry.page_id)
                    in _reserved_system_page_collision_keys()
                ):
                    raise IngestApplyError(
                        "reserved system page_id cannot be mutated by ingest: "
                        f"{entry.page_id!r}"
                    )
                # Re-evaluate global correction tombstones while holding the
                # same mutation lock as the correction lane. Preparation may
                # predate a correction on another page, and a stale replay may
                # choose an entirely new slug, so path-local CAS alone is not
                # sufficient here.
                try:
                    constrained_body, enforced = enforce_correction_constraints(
                        entry.page_id,
                        entry.previous_text or "",
                        entry.new_body,
                    )
                except PageMutationError as exc:
                    raise IngestApplyError(
                        f"content correction constraint failed for {entry.page_id}: {exc}"
                    ) from exc
                # The frontier approved ``entry.new_body`` exactly.  A newly
                # activated correction constraint is valid evidence that the
                # proposal became stale, but it cannot silently rewrite the
                # approved postimage.  Retry preparation + review instead.
                if constrained_body != entry.new_body:
                    raise IngestApplyError(
                        f"content correction constraints changed before ingest apply: "
                        f"{entry.page_id}"
                    )
                current = _read_optional_exact_utf8(entry.path)
                if current == entry.new_body:
                    # Power-loss recovery: this exact reviewed postimage was
                    # already installed, so finish the batch idempotently.
                    (created if entry.op_type == "create" else updated).append(
                        entry.page_id
                    )
                    continue
                if recovery_only:
                    raise IngestApplyError(
                        "reviewed postimage no longer present during recovery: "
                        f"{entry.page_id}"
                    )
                if entry.op_type == "create":
                    entry.path.parent.mkdir(parents=True, exist_ok=True)
                    if current is not None:
                        raise IngestApplyError(
                            f"page appeared before ingest create: {entry.page_id}"
                        )
                    atomic_write(entry.path, entry.new_body)
                    # Append BEFORE logging so a log failure could never drop
                    # an entry from the rollback set. _safe_log additionally
                    # ensures a logging exception (which atomic_write success
                    # already proves is irrelevant to data) never triggers
                    # rollback of a write that succeeded.
                    written.append(entry)
                    created.append(entry.page_id)
                    _safe_log(f"ingest | created {entry.page_id}")
                else:
                    # The prepare phase captured this exact preimage. Refuse
                    # to overwrite a correction or other cooperating writer
                    # that committed while the model was preparing the batch.
                    if current != (entry.previous_text or ""):
                        raise IngestApplyError(
                            f"page changed before ingest apply: {entry.page_id}"
                        )
                    atomic_write(entry.path, entry.new_body)
                    written.append(entry)
                    updated.append(entry.page_id)
                    _safe_log(f"ingest | updated {entry.page_id}")
        except Exception as write_err:
            # Best-effort rollback. Each revert is gated by a CAS check: only
            # restore if the file still contains exactly what we wrote. If
            # another writer has modified it since, leave their change intact.
            rollback_errors: list[str] = []
            for entry in reversed(written):
                try:
                    if entry.op_type == "create":
                        if _read_optional_exact_utf8(entry.path) == entry.new_body:
                            entry.path.unlink()
                        elif entry.path.exists():
                            rollback_errors.append(
                                f"{entry.page_id}: skipped (modified by another writer)"
                            )
                    else:
                        if _read_optional_exact_utf8(entry.path) == entry.new_body:
                            atomic_write(entry.path, entry.previous_text or "")
                        elif entry.path.exists():
                            rollback_errors.append(
                                f"{entry.page_id}: skipped (modified by another writer)"
                            )
                except Exception as rb_err:
                    rollback_errors.append(f"{entry.page_id}: {rb_err}")
            if rollback_errors:
                partial_summary = "; ".join(rollback_errors)
                _safe_log(
                    "ingest | rollback partial (other writers or IO failures): "
                    + partial_summary
                )
                raise IngestApplyError(
                    f"apply write failed: {write_err}; partial rollback: "
                    f"{partial_summary}"
                ) from write_err
            _safe_log(
                f"ingest | rolled back {len(written)} writes after error: {write_err}"
            )
            raise IngestApplyError(f"apply write failed: {write_err}") from write_err

    # Tag changelog entries are derived audit data.  They are emitted only
    # after the exact semantic page batch has frontier approval and commits.
    if created:
        from llm_wiki_mcp.tags import record_new_tag

        created_ids = set(created)
        for entry in planned:
            if entry.op_type != "create" or entry.page_id not in created_ids:
                continue
            for tag in entry.new_tags:
                record_new_tag(tag, reason="ingest auto-gen")

    totals = link_totals or {"resolved": 0, "rewritten": 0, "unwrapped": 0}
    if any(totals.values()):
        _safe_log(
            f"ingest | link reconcile: resolved={totals['resolved']} "
            f"rewritten={totals['rewritten']} unwrapped={totals['unwrapped']}"
        )

    return created, updated

