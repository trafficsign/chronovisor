from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from chronovisor.core.save_transaction import (
    attach_save_transaction_marker,
    make_save_transaction,
)
from chronovisor.ingest.raw_semantic_projection import (
    PROJECTION_BUNDLE_RECEIPT_SCHEMA,
    PROJECTION_CHILD_SCHEMA,
    ProjectionConflictError,
    RawSemanticProjectionError,
    project_parent_raw,
    project_reassembled_raws,
    projection_bundle_state_for_parent,
    verify_projection_bundle,
)


def _transcript_raw(
    tmp_path: Path,
    rows: list[dict],
    *,
    host: str = "codex",
    filename: str = "parent.md",
    after_line: int = 0,
    until_line: int = 10,
) -> Path:
    session_file = tmp_path / f"{host}-session.jsonl"
    transaction = make_save_transaction(
        host=host,
        session_file=session_file,
        session_id="session-1",
        after_line=after_line,
        until_line=until_line,
    )
    label = "Claude Code" if host == "claude-code" else "Codex"
    content = "\n".join(
        [
            f"# {label} Session Transcript Delta",
            "",
            f"- Source: {label}",
            "- Capture mode: deterministic-lossless",
            "",
            "## Transcript Delta",
            "",
            "```json",
            json.dumps(rows, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    path = tmp_path / filename
    path.write_text(
        attach_save_transaction_marker(transaction, content),
        encoding="utf-8",
    )
    return path


def _child_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == PROJECTION_CHILD_SCHEMA
    return payload["records"]


def _write_canonical_json(path: Path, payload: dict) -> bytes:
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def test_atomic_projection_publication_rejects_symlinked_parent_and_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.ingest import raw_semantic_projection as projection_mod

    outside = tmp_path / "outside"
    outside.mkdir()
    symlinked_output = tmp_path / "projection"
    symlinked_output.symlink_to(outside, target_is_directory=True)
    payload = b"canonical artifact\n"

    with pytest.raises(ProjectionConflictError, match="directory is unsafe"):
        projection_mod._atomic_create_or_verify(
            symlinked_output / "semantic-safe.json", payload
        )
    assert not (outside / "semantic-safe.json").exists()

    output = tmp_path / "safe-projection"
    target = output / "semantic-safe.json"
    real_link = projection_mod.os.link

    def replace_after_link(*args, **kwargs):
        result = real_link(*args, **kwargs)
        target.unlink()
        target.symlink_to(outside / "external.json")
        return result

    monkeypatch.setattr(projection_mod.os, "link", replace_after_link)

    with pytest.raises(ProjectionConflictError, match="artifact is unsafe"):
        projection_mod._atomic_create_or_verify(target, payload)
    assert not (outside / "external.json").exists()


def test_atomic_projection_publication_rejects_parent_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.ingest import raw_semantic_projection as projection_mod

    output = tmp_path / "projection"
    output.mkdir()
    target = output / "semantic-safe.json"
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = tmp_path / "projection-moved"
    real_link = projection_mod.os.link

    def replace_parent_after_link(*args, **kwargs):
        result = real_link(*args, **kwargs)
        output.rename(moved)
        output.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(projection_mod.os, "link", replace_parent_after_link)

    with pytest.raises(ProjectionConflictError, match="directory changed"):
        projection_mod._atomic_create_or_verify(target, b"canonical artifact\n")
    assert not (outside / target.name).exists()


def _forge_bundle_receipt_for_manifest(manifest_path: Path, manifest: dict) -> None:
    old_receipt = next(manifest_path.parent.glob("*-manifest-*.receipt.json"))
    receipt = json.loads(old_receipt.read_text(encoding="utf-8"))
    assert receipt["schema"] == PROJECTION_BUNDLE_RECEIPT_SCHEMA
    manifest_bytes = _write_canonical_json(manifest_path, manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    receipt["manifest_sha256"] = manifest_sha256
    new_receipt = manifest_path.parent / (
        f"semantic-{manifest['projection_id']}-manifest-{manifest_sha256}.receipt.json"
    )
    old_receipt.unlink()
    _write_canonical_json(new_receipt, receipt)


def _reconstructed_texts(paths: tuple[Path, ...]) -> dict[int, str]:
    grouped: dict[int, list[tuple[int, str]]] = {}
    for path in paths:
        for row in _child_records(path):
            grouped.setdefault(row["source_record_index"], []).append(
                (int(row["segment_index"]), row["text"])
            )
    return {
        source_index: "".join(text for _, text in sorted(segments))
        for source_index, segments in grouped.items()
    }


def test_projects_only_nonblank_user_assistant_text_byte_exact(tmp_path: Path) -> None:
    user = "  日本語🙂\nexact user bytes  "
    assistant = 'assistant\n```json\n{"a": 1}\n```'
    path = _transcript_raw(
        tmp_path,
        [
            {"line": 1, "role": "user", "text": user, "timestamp": None},
            {
                "line": 2,
                "role": "tool",
                "text": "tool output must not reach semantic input",
                "event": {"payload": "x" * 20_000},
            },
            {"line": 3, "role": "assistant", "text": " \t\n"},
            {"line": 4, "role": "assistant", "text": assistant},
        ],
    )

    result = project_parent_raw(
        path,
        output_dir=tmp_path / "projection",
        max_child_bytes=16_000,
    )

    assert result.kind == "children"
    assert result.record_count == 4
    assert result.selected_record_count == 2
    assert result.role_counts == {"assistant": 2, "tool": 1, "user": 1}
    assert result.child_count == 1
    assert result.children[0].index == 1
    assert result.children[0].count == 1
    reconstructed = _reconstructed_texts(result.child_paths)
    assert reconstructed == {0: user, 3: assistant}
    assert reconstructed[0].encode("utf-8") == user.encode("utf-8")
    assert reconstructed[3].encode("utf-8") == assistant.encode("utf-8")
    assert "tool output" not in result.child_paths[0].read_text(encoding="utf-8")
    manifest = verify_projection_bundle(result.manifest_path)  # type: ignore[arg-type]
    assert manifest["source_sha256"] == result.parent_sha256
    assert manifest["projection_sha256"] == result.projection_sha256
    assert (
        projection_bundle_state_for_parent(
            path,
            projection_dir=result.manifest_path.parent,  # type: ignore[union-attr]
        )
        == "completed"
    )


def test_malformed_delegated_children_are_invalid_not_incomplete(
    tmp_path: Path,
) -> None:
    path = _transcript_raw(
        tmp_path,
        [{"line": 1, "role": "user", "text": "malformed child row"}],
    )
    output = tmp_path / "projection"
    projected = project_parent_raw(path, output_dir=output, max_child_bytes=2_000)
    assert projected.manifest_path is not None
    manifest = json.loads(projected.manifest_path.read_text(encoding="utf-8"))
    manifest["children"] = {"filename": "not-a-list"}
    _write_canonical_json(projected.manifest_path, manifest)

    assert projection_bundle_state_for_parent(path, projection_dir=output) == "invalid"


def test_tampered_transcript_heading_cannot_fall_through_to_full_raw_ingest(
    tmp_path: Path,
) -> None:
    path = _transcript_raw(
        tmp_path,
        [{"line": 1, "role": "user", "text": "private tool trace boundary"}],
        filename="save-codex-0123456789abcdef01234567-from0-to10.md",
    )
    tampered = path.read_text(encoding="utf-8").replace(
        "# Codex Session Transcript Delta",
        "# Codex Session Transcript Deltx",
        1,
    )
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(
        RawSemanticProjectionError,
        match="no canonical transcript envelope",
    ):
        project_parent_raw(
            path,
            output_dir=tmp_path / "projection",
            max_child_bytes=2_000,
        )


def test_deterministic_save_filename_without_receipt_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "save-claude-code-0123456789abcdef01234567-from4-to8.md"
    path.write_text("ordinary-looking raw without receipt", encoding="utf-8")

    with pytest.raises(
        RawSemanticProjectionError,
        match="no canonical transcript envelope",
    ):
        project_parent_raw(
            path,
            output_dir=tmp_path / "projection",
            max_child_bytes=2_000,
        )


def test_verified_archived_legacy_markdown_can_passthrough(tmp_path: Path) -> None:
    path = tmp_path / "save-codex-0123456789abcdef01234567-from4-to8.md"
    raw_bytes = b"---\nraw_keywords: [historical]\n---\nLegacy transcript envelope.\n"

    result = project_parent_raw(
        path,
        raw_bytes=raw_bytes,
        output_dir=tmp_path / "projection",
        max_child_bytes=2_000,
        allow_verified_legacy_markdown=True,
    )

    assert result.kind == "passthrough"
    assert result.manifest_path is None
    assert result.child_paths == ()


def test_verified_archived_legacy_transcript_envelope_can_passthrough(
    tmp_path: Path,
) -> None:
    path = tmp_path / "save-claude-code-0123456789abcdef01234567-from4-to8.md"
    raw_bytes = b"""---
raw_keywords: [historical]
---
<!-- llm-wiki-save-transaction: legacy -->
# Claude Code Session Transcript Delta

## Transcript Delta

```json
[{"line":1,"role":"user","text":"historical"}]
```
"""

    with pytest.raises(
        RawSemanticProjectionError,
        match="save transaction receipt",
    ):
        project_parent_raw(
            path,
            raw_bytes=raw_bytes,
            output_dir=tmp_path / "unverified",
            max_child_bytes=2_000,
        )

    result = project_parent_raw(
        path,
        raw_bytes=raw_bytes,
        output_dir=tmp_path / "verified",
        max_child_bytes=2_000,
        allow_verified_legacy_markdown=True,
    )

    assert result.kind == "passthrough"
    assert result.parent_paths == (path,)
    assert result.parent_sha256 == hashlib.sha256(raw_bytes).hexdigest()
    assert result.manifest_path is None
    assert result.child_paths == ()
    assert not (tmp_path / "verified").exists()


def test_deterministic_save_filename_must_match_verified_receipt(
    tmp_path: Path,
) -> None:
    original = _transcript_raw(
        tmp_path,
        [{"line": 1, "role": "user", "text": "exact source"}],
    )
    mismatched = tmp_path / "save-codex-0123456789abcdef01234567-from0-to10.md"
    original.rename(mismatched)

    with pytest.raises(
        RawSemanticProjectionError,
        match="filename does not match",
    ):
        project_parent_raw(
            mismatched,
            output_dir=tmp_path / "projection",
            max_child_bytes=2_000,
        )


def test_tool_only_raw_writes_verified_deterministic_noop(tmp_path: Path) -> None:
    path = _transcript_raw(
        tmp_path,
        [
            {"line": 1, "role": "tool", "text": "result"},
            {"line": 2, "role": "event", "text": ""},
            {"line": 3, "role": "assistant", "text": " \n"},
        ],
    )

    first = project_parent_raw(
        path,
        output_dir=tmp_path / "projection",
        max_child_bytes=2_000,
    )
    second = project_parent_raw(
        path,
        output_dir=tmp_path / "projection",
        max_child_bytes=2_000,
    )

    assert first.kind == second.kind == "noop"
    assert first.child_paths == ()
    assert first.projection_paths == (first.noop_receipt_path,)
    assert first.noop_receipt_path is not None
    assert first.noop_receipt_path.read_bytes() == second.noop_receipt_path.read_bytes()  # type: ignore[union-attr]
    manifest = verify_projection_bundle(first.manifest_path)  # type: ignore[arg-type]
    assert manifest["status"] == "noop"
    assert manifest["record_count"] == 3
    assert manifest["role_counts"] == {"assistant": 1, "event": 1, "tool": 1}
    assert manifest["selected_record_count"] == 0


def test_record_boundary_fanout_is_content_addressed(tmp_path: Path) -> None:
    texts = ["a" * 900, "b" * 900, "c" * 900]
    path = _transcript_raw(
        tmp_path,
        [
            {"line": index, "role": "user", "text": text}
            for index, text in enumerate(texts, start=1)
        ],
    )

    result = project_parent_raw(
        path,
        output_dir=tmp_path / "projection",
        max_child_bytes=1_900,
    )

    assert result.kind == "children"
    assert result.child_count >= 2
    assert all(child.path.stat().st_size <= 1_900 for child in result.children)
    assert [child.index for child in result.children] == list(
        range(1, result.child_count + 1)
    )
    assert all(child.count == result.child_count for child in result.children)
    assert _reconstructed_texts(result.child_paths) == dict(enumerate(texts))
    for child in result.children:
        assert result.projection_sha256 not in child.path.name
        assert child.child_id in child.path.name


def test_single_record_uses_utf8_exact_numbered_segments(tmp_path: Path) -> None:
    text = ("あ🙂é\n" * 500) + "末尾"
    path = _transcript_raw(
        tmp_path,
        [{"line": 1, "role": "assistant", "text": text}],
    )

    result = project_parent_raw(
        path,
        output_dir=tmp_path / "projection",
        max_child_bytes=1_350,
    )

    assert result.child_count > 1
    assert all(path.stat().st_size <= 1_350 for path in result.child_paths)
    reconstructed = _reconstructed_texts(result.child_paths)[0]
    assert reconstructed.encode("utf-8") == text.encode("utf-8")
    segments = [
        row for child_path in result.child_paths for row in _child_records(child_path)
    ]
    assert [int(row["segment_index"]) for row in segments] == list(
        range(1, len(segments) + 1)
    )
    assert {int(row["segment_count"]) for row in segments} == {len(segments)}


def test_retry_reuses_exact_artifacts_and_conflict_fails_closed(tmp_path: Path) -> None:
    path = _transcript_raw(
        tmp_path,
        [{"line": 1, "role": "user", "text": "durable"}],
    )
    output = tmp_path / "projection"
    first = project_parent_raw(path, output_dir=output, max_child_bytes=2_000)
    second = project_parent_raw(path, output_dir=output, max_child_bytes=2_000)
    assert first.child_paths == second.child_paths
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()  # type: ignore[union-attr]

    first.child_paths[0].write_text("tampered", encoding="utf-8")
    with pytest.raises(ProjectionConflictError, match="artifact conflict"):
        project_parent_raw(path, output_dir=output, max_child_bytes=2_000)


@pytest.mark.parametrize(
    "audit_field",
    [
        "record_count",
        "role_counts",
        "selected_role_counts",
        "child_semantic_bytes",
        "child_source_record_indices",
    ],
)
def test_canonical_manifest_audit_tampering_fails_even_with_forged_receipt(
    tmp_path: Path,
    audit_field: str,
) -> None:
    path = _transcript_raw(
        tmp_path,
        [
            {"line": 1, "role": "user", "text": "audit me"},
            {"line": 2, "role": "assistant", "text": "exactly"},
            {"line": 3, "role": "tool", "text": "excluded"},
        ],
    )
    projected = project_parent_raw(
        path,
        output_dir=tmp_path / "projection",
        max_child_bytes=2_000,
    )
    assert projected.manifest_path is not None
    manifest = json.loads(projected.manifest_path.read_text(encoding="utf-8"))

    if audit_field == "record_count":
        manifest["record_count"] += 1
    elif audit_field == "role_counts":
        manifest["role_counts"]["tool"] += 1
    elif audit_field == "selected_role_counts":
        manifest["selected_role_counts"]["user"] += 1
    elif audit_field == "child_semantic_bytes":
        manifest["children"][0]["semantic_bytes"] += 1
    elif audit_field == "child_source_record_indices":
        manifest["children"][0]["source_record_indices"] = [999]
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(audit_field)

    _forge_bundle_receipt_for_manifest(projected.manifest_path, manifest)

    assert (
        projection_bundle_state_for_parent(
            path,
            projection_dir=projected.manifest_path.parent,
        )
        == "invalid"
    )
    with pytest.raises(RawSemanticProjectionError):
        verify_projection_bundle(projected.manifest_path)


def test_delegated_projection_requires_bundle_completion_receipt(
    tmp_path: Path,
) -> None:
    path = _transcript_raw(
        tmp_path,
        [{"line": 1, "role": "user", "text": "committed bundle"}],
    )
    projected = project_parent_raw(
        path,
        output_dir=tmp_path / "projection",
        max_child_bytes=2_000,
    )
    assert projected.manifest_path is not None
    next(projected.manifest_path.parent.glob("*-manifest-*.receipt.json")).unlink()

    with pytest.raises(RawSemanticProjectionError, match="cannot be read"):
        verify_projection_bundle(projected.manifest_path)


def test_existing_intent_reuses_same_bundle_after_child_envelope_change(
    tmp_path: Path,
) -> None:
    path = _transcript_raw(
        tmp_path,
        [{"line": 1, "role": "user", "text": "x" * 3_000}],
    )
    output = tmp_path / "projection"

    larger = project_parent_raw(path, output_dir=output, max_child_bytes=2_000)
    files_before = {path.name for path in output.iterdir()}
    retried = project_parent_raw(path, output_dir=output, max_child_bytes=3_000)

    assert larger.manifest_path == retried.manifest_path
    assert larger.child_paths == retried.child_paths
    assert {path.name for path in output.iterdir()} == files_before
    larger_manifest = verify_projection_bundle(larger.manifest_path)  # type: ignore[arg-type]
    assert larger_manifest["max_child_bytes"] == 2_000
    assert all(path.stat().st_size <= 2_000 for path in larger.child_paths)


def test_partial_child_fault_resumes_existing_intent_under_new_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import raw_semantic_projection as projection_mod

    path = _transcript_raw(
        tmp_path,
        [{"line": 1, "role": "user", "text": "x" * 5_000}],
    )
    output = tmp_path / "projection"
    original_publish = projection_mod._atomic_create_or_verify
    child_calls = 0

    def fail_before_second_child(target: Path, payload: bytes) -> bool:
        nonlocal child_calls
        if "-child-" in target.name:
            child_calls += 1
            if child_calls == 2:
                raise RuntimeError("injected child publication fault")
        return original_publish(target, payload)

    monkeypatch.setattr(
        projection_mod,
        "_atomic_create_or_verify",
        fail_before_second_child,
    )
    with pytest.raises(RuntimeError, match="injected"):
        project_parent_raw(path, output_dir=output, max_child_bytes=1_400)

    manifest_path = next(output.glob("*.manifest.json"))
    intent = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_children = {row["filename"] for row in intent["children"]}
    assert intent["max_child_bytes"] == 1_400
    assert len(expected_children) >= 2
    assert len(list(output.glob("*.md"))) == 1
    assert (
        projection_bundle_state_for_parent(path, projection_dir=output) == "incomplete"
    )

    monkeypatch.setattr(
        projection_mod,
        "_atomic_create_or_verify",
        original_publish,
    )
    resumed = project_parent_raw(path, output_dir=output, max_child_bytes=3_000)

    assert resumed.manifest_path == manifest_path
    assert {path.name for path in resumed.child_paths} == expected_children
    assert {path.name for path in output.glob("*.md")} == expected_children
    assert verify_projection_bundle(manifest_path)["max_child_bytes"] == 1_400
    assert (
        projection_bundle_state_for_parent(path, projection_dir=output) == "completed"
    )


def test_directory_fsync_failure_propagates_and_retry_resyncs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import raw_semantic_projection as projection_mod

    path = _transcript_raw(
        tmp_path,
        [{"line": 1, "role": "user", "text": "durable projection"}],
    )
    output = tmp_path / "projection"
    original_fsync_directory = projection_mod._fsync_directory
    calls = 0

    def fail_first_directory_sync(directory: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected directory fsync failure")
        original_fsync_directory(directory)

    monkeypatch.setattr(
        projection_mod,
        "_fsync_directory",
        fail_first_directory_sync,
    )

    with pytest.raises(OSError, match="directory fsync failure"):
        project_parent_raw(path, output_dir=output, max_child_bytes=2_000)

    assert projection_bundle_state_for_parent(path, projection_dir=output) in {
        "incomplete",
        "invalid",
    }

    resumed = project_parent_raw(path, output_dir=output, max_child_bytes=2_000)

    assert calls >= 2
    assert resumed.kind == "children"
    assert projection_bundle_state_for_parent(path, projection_dir=output) == (
        "completed"
    )


def test_child_reprocessing_validates_identity_then_passthrough(tmp_path: Path) -> None:
    path = _transcript_raw(
        tmp_path,
        [{"line": 1, "role": "user", "text": "one child"}],
    )
    projected = project_parent_raw(
        path,
        output_dir=tmp_path / "projection",
        max_child_bytes=2_000,
    )
    child_path = projected.child_paths[0]

    passthrough = project_parent_raw(
        child_path,
        output_dir=tmp_path / "unused",
        max_child_bytes=2_000,
    )

    assert passthrough.kind == "passthrough"
    assert passthrough.parent_sha256 == projected.parent_sha256
    assert passthrough.child_paths == (child_path,)
    assert passthrough.children[0].index == 1
    assert passthrough.children[0].count == projected.child_count

    payload = json.loads(child_path.read_text(encoding="utf-8"))
    payload["records"][0]["text"] = "changed"
    child_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProjectionConflictError, match="child artifact"):
        project_parent_raw(
            child_path,
            output_dir=tmp_path / "unused",
            max_child_bytes=2_000,
        )


def test_orphan_child_without_delegation_manifest_fails_closed(tmp_path: Path) -> None:
    path = _transcript_raw(
        tmp_path,
        [{"line": 1, "role": "user", "text": "delegated only by manifest"}],
    )
    projected = project_parent_raw(
        path,
        output_dir=tmp_path / "projection",
        max_child_bytes=2_000,
    )
    child_path = projected.child_paths[0]
    assert projected.manifest_path is not None
    projected.manifest_path.unlink()

    with pytest.raises(RawSemanticProjectionError, match="delegation manifest"):
        project_parent_raw(
            child_path,
            output_dir=tmp_path / "unused",
            max_child_bytes=2_000,
        )


def test_transcript_claim_requires_valid_save_receipt(tmp_path: Path) -> None:
    path = _transcript_raw(
        tmp_path,
        [{"line": 1, "role": "user", "text": "evidence"}],
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace("evidence", "tampered evidence"),
        encoding="utf-8",
    )

    with pytest.raises(RawSemanticProjectionError, match="save transaction receipt"):
        project_parent_raw(
            path,
            output_dir=tmp_path / "projection",
            max_child_bytes=2_000,
        )


def test_non_transcript_raw_is_passthrough_without_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "manual.md"
    path.write_text("# Ordinary raw\n\nNo saver envelope.\n", encoding="utf-8")

    result = project_parent_raw(
        path,
        output_dir=tmp_path / "projection",
        max_child_bytes=2_000,
    )

    assert result.kind == "passthrough"
    assert result.parent_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.manifest_path is None
    assert result.projection_paths == ()
    assert not (tmp_path / "projection").exists()


def _fragment_raws(tmp_path: Path, record_bytes: bytes) -> list[Path]:
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    split = len(record_bytes) // 2
    chunks = [record_bytes[:split], record_bytes[split:]]
    paths: list[Path] = []
    for index, chunk in enumerate(chunks, start=1):
        session_file = tmp_path / "fragment-session.jsonl"
        transaction = make_save_transaction(
            host="codex",
            session_file=session_file,
            session_id=f"fragment-{index}",
            after_line=8,
            until_line=9,
        )
        payload = {
            "schema": "chronovisor.raw-capture-fragment.v1",
            "host": "codex",
            "session_id": "source-session",
            "session_file": str(session_file),
            "source_line": 9,
            "record_sha256": record_sha256,
            "record_bytes": len(record_bytes),
            "fragment_index": index,
            "fragment_count": len(chunks),
            "fragment_bytes": len(chunk),
            "encoding": "base64",
            "data": base64.b64encode(chunk).decode("ascii"),
        }
        content = "\n".join(
            [
                "# Codex Oversized Transcript Record Fragment",
                "",
                "```json",
                json.dumps(payload, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        path = tmp_path / f"fragment-{index}.md"
        path.write_text(
            attach_save_transaction_marker(transaction, content),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def test_reassembled_fragments_verify_receipts_metadata_and_exact_bytes(
    tmp_path: Path,
) -> None:
    text = "fragmented semantic text🙂"
    record_bytes = json.dumps(
        [{"line": 9, "role": "assistant", "text": text}],
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    paths = _fragment_raws(tmp_path, record_bytes)

    result = project_reassembled_raws(
        paths,
        record_bytes,
        output_dir=tmp_path / "projection",
        max_child_bytes=2_000,
    )

    assert result.kind == "children"
    assert len(result.parent_paths) == 2
    assert _reconstructed_texts(result.child_paths) == {0: text}

    with pytest.raises(RawSemanticProjectionError, match="differs"):
        project_reassembled_raws(
            paths,
            record_bytes + b" ",
            output_dir=tmp_path / "other",
            max_child_bytes=2_000,
        )


def test_fragment_metadata_host_must_match_receipt_host(tmp_path: Path) -> None:
    record_bytes = json.dumps(
        [{"line": 9, "role": "user", "text": "x"}],
        separators=(",", ":"),
    ).encode("utf-8")
    paths = _fragment_raws(tmp_path, record_bytes)
    text = paths[0].read_text(encoding="utf-8")
    marker, protected = text.split("\n", 1)
    del marker
    protected = protected.replace('"host": "codex"', '"host": "claude-code"', 1)
    session_file = tmp_path / "replacement-session.jsonl"
    transaction = make_save_transaction(
        host="codex",
        session_file=session_file,
        session_id="replacement",
        after_line=0,
        until_line=1,
    )
    paths[0].write_text(
        attach_save_transaction_marker(transaction, protected.lstrip("\n")),
        encoding="utf-8",
    )

    with pytest.raises(RawSemanticProjectionError):
        project_reassembled_raws(
            paths,
            record_bytes,
            output_dir=tmp_path / "projection",
            max_child_bytes=2_000,
        )


def test_fragment_receipt_interval_must_match_metadata_source_line(
    tmp_path: Path,
) -> None:
    record_bytes = json.dumps(
        [{"line": 9, "role": "user", "text": "x"}],
        separators=(",", ":"),
    ).encode("utf-8")
    paths = _fragment_raws(tmp_path, record_bytes)
    _marker, protected = paths[0].read_text(encoding="utf-8").split("\n", 1)
    wrong_transaction = make_save_transaction(
        host="codex",
        session_file=tmp_path / "wrong-line-session.jsonl",
        session_id="wrong-line",
        after_line=7,
        until_line=8,
    )
    paths[0].write_text(
        attach_save_transaction_marker(wrong_transaction, protected.lstrip("\n")),
        encoding="utf-8",
    )

    with pytest.raises(RawSemanticProjectionError, match="interval"):
        project_reassembled_raws(
            paths,
            record_bytes,
            output_dir=tmp_path / "projection",
            max_child_bytes=2_000,
        )
