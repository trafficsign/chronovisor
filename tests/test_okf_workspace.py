from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from chronovisor.core import durable_state, okf_workspace
from chronovisor.core.canonical_document import parse_document
from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.core.okf_prepare import RawSource
from chronovisor.core.okf_v02 import ConformanceIssue, validate_pages_bundle
from chronovisor.core.okf_workspace import prepare_okf_workspace

FIXTURE = Path(__file__).parent / "fixtures" / "okf_workspace" / "source"


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source"
    shutil.copytree(FIXTURE, source)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    return source, runtime


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_not_prepared(runtime: Path, run_id: str) -> None:
    workspace = runtime / "migrations" / run_id
    for name in ("journal.json", okf_workspace.RESTART_REFUSAL_FILENAME):
        path = workspace / name
        assert not path.exists() or json.loads(path.read_bytes())["state"] != "prepared"


def test_workspace_stages_validated_namespaces_without_touching_source(
    tmp_path: Path,
) -> None:
    source, runtime = _roots(tmp_path)
    with durable_state.okf_writer_lock(source):
        pass
    before = _snapshot(source)

    workspace = prepare_okf_workspace(source, runtime, "run-001")

    assert workspace == runtime.resolve() / "migrations" / "run-001"
    assert _snapshot(source) == before
    pages = workspace / "staging" / "pages"
    system = workspace / "staging" / "system"
    assert not [issue for issue in validate_pages_bundle(pages) if issue.severity == "error"]
    assert not (pages / "schema.md").exists()
    assert (system / "schema.md").is_file()
    assert not (workspace / "rollback-backup").exists()

    index = (pages / "index.md").read_text()
    log = (pages / "log.md").read_text()
    assert "okf_version: '0.2'" in index
    assert parse_document(index.encode()).metadata == {"okf_version": "0.2"}
    assert "[Target](deep/target.md)" in index
    assert "legacy root index" not in index
    assert not log.startswith("---")
    assert log == "# Derived change history\n"
    assert "legacy operational payload" not in log
    assert "archive_reason" not in log
    assert "archive_provenance" not in log

    page = parse_document((pages / "notes" / "source.md").read_bytes())
    assert page.metadata["status"] == "stable"
    assert page.body == b"Read [the target](<../deep/target.md#Section heading>).\n"
    system_state = parse_document((system / "current-state.md").read_bytes())
    source_state = parse_document((source / "system" / "current-state.md").read_bytes())
    expected_state_metadata = dict(source_state.metadata)
    expected_state_metadata["status"] = "stable"
    expected_state_metadata.pop("chronovisor_status")
    assert system_state.metadata == expected_state_metadata
    assert system_state.metadata["summary"] == "Private state summary stays a summary."
    assert "description" not in system_state.metadata
    assert system_state.body == (
        b"System links to [the portable target](../pages/deep/target.md) "
        b"from outside the OKF bundle.\n"
    )
    schema = parse_document((system / "schema.md").read_bytes())
    source_schema = parse_document((source / "schema.md").read_bytes())
    expected_schema_metadata = dict(source_schema.metadata)
    expected_schema_metadata["status"] = "stable"
    expected_schema_metadata.pop("chronovisor_status")
    assert schema.metadata == expected_schema_metadata
    assert schema.body == source_schema.body
    for name in (
        "claude-code.md",
        "lessons-learned.md",
        "tag-changelog.md",
        "user-profile.md",
    ):
        before_system = parse_document((source / "system" / name).read_bytes())
        after_system = parse_document((system / name).read_bytes())
        assert after_system.metadata == {**before_system.metadata, "status": "stable"}
        assert after_system.body == before_system.body
    assert "type" not in parse_document((system / "claude-code.md").read_bytes()).metadata

    activity = [
        json.loads(line)
        for line in (workspace / "staging" / "activity.jsonl").read_text().splitlines()
    ]
    assert activity[0]["archive_reason"] == "merged"
    manifest_path = workspace / "dry-run-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    assert [
        (item["source_path"], item["staged_path"])
        for item in manifest["reserved_documents"]
    ] == [
        ("index.md", "pages/index.md"),
        ("log.md", "pages/log.md"),
        ("schema.md", "system/schema.md"),
    ]
    system_manifest = manifest["system_documents"]
    assert {
        (item["relative_path"], item["source_scope"])
        for item in system_manifest
    } == {
        ("claude-code.md", "system"),
        ("current-state.md", "system"),
        ("lessons-learned.md", "system"),
        ("schema.md", "root"),
        ("tag-changelog.md", "system"),
        ("user-profile.md", "system"),
    }
    assert all(
        set(item)
        == {
            "relative_path",
            "source_scope",
            "input_status",
            "output_status",
            "identity_source",
            "identity_sha256",
            "resolved_link_count",
            "source_sha256",
            "output_sha256",
        }
        for item in system_manifest
    )
    assert all(item["input_status"] == "missing" for item in system_manifest)
    assert all(item["output_status"] == "stable" for item in system_manifest)
    assert {
        item["relative_path"]: item["resolved_link_count"]
        for item in system_manifest
    } == {
        "claude-code.md": 0,
        "current-state.md": 1,
        "lessons-learned.md": 0,
        "schema.md": 0,
        "tag-changelog.md": 0,
        "user-profile.md": 0,
    }
    system_by_path = {item["relative_path"]: item for item in system_manifest}
    assert system_by_path["current-state.md"]["identity_source"] == "identity"
    assert system_by_path["current-state.md"][
        "identity_sha256"
    ] == canonical_json_sha256_strict(
        {"source": "identity", "value": "current-state"}
    )
    assert system_by_path["claude-code.md"]["identity_source"] == "relative_path"
    assert system_by_path["claude-code.md"][
        "identity_sha256"
    ] == canonical_json_sha256_strict(
        {"source": "relative_path", "value": "claude-code.md"}
    )
    missing_cohort = next(
        item
        for item in manifest["system_status_cohorts"]
        if item["input_status"] == "missing"
    )
    assert missing_cohort == {
        "input_status": "missing",
        "output_status": "stable",
        "count": 6,
        "identity_set_sha256": canonical_json_sha256_strict(
            sorted(item["identity_sha256"] for item in system_manifest)
        ),
    }
    assert len(manifest["system_status_cohorts"]) == 6
    assert manifest["unresolved_links"] == []
    assert manifest["raw_files"][0]["relative_path"] == "sessions/session.jsonl"
    assert b"merged" not in manifest_raw
    assert b"workspace-fixture" not in manifest_raw
    assert b"canonical-schema" not in manifest_raw
    assert b"Private state summary stays a summary" not in manifest_raw
    assert b"Private history body" not in manifest_raw
    assert not {
        "current-state.md",
        "schema.md",
    }.intersection(item["path"] for item in manifest["conformance"])

    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    journal = json.loads((workspace / "journal.json").read_bytes())
    sentinel = json.loads(
        (workspace / okf_workspace.RESTART_REFUSAL_FILENAME).read_bytes()
    )
    assert journal == {
        "schema": okf_workspace.JOURNAL_SCHEMA,
        "version": 1,
        "run_id": "run-001",
        "state": "prepared",
        "manifest_sha256": manifest_sha256,
    }
    assert sentinel == {
        **journal,
        "schema": okf_workspace.SENTINEL_SCHEMA,
    }


def test_system_status_mapping_preserves_private_extensions_without_page_transforms(
    tmp_path: Path,
) -> None:
    source, runtime = _roots(tmp_path)
    (source / "system" / "active.md").write_text(
        "---\n"
        "uid: private-active-id\n"
        "status: active\n"
        "summary: Keep this field\n"
        "chronovisor_status: indexed\n"
        "---\n"
        "Active private body.\n"
    )
    (source / "system" / "archived.md").write_text(
        "---\n"
        "identity: private-archived-id\n"
        "status: archived\n"
        "archive_reason: private reason remains private\n"
        "archive_provenance: private provenance remains private\n"
        "---\n"
        "Archived private body.\n"
    )

    workspace = prepare_okf_workspace(source, runtime, "system-statuses")
    system = workspace / "staging" / "system"
    active = parse_document((system / "active.md").read_bytes())
    archived = parse_document((system / "archived.md").read_bytes())

    assert active.metadata == {
        "uid": "private-active-id",
        "status": "stable",
        "summary": "Keep this field",
    }
    assert active.body == b"Active private body.\n"
    assert archived.metadata == {
        "identity": "private-archived-id",
        "status": "deprecated",
        "archive_reason": "private reason remains private",
        "archive_provenance": "private provenance remains private",
    }
    assert archived.body == b"Archived private body.\n"
    assert "description" not in active.metadata
    assert "type" not in active.metadata

    manifest_raw = (workspace / "dry-run-manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    cohorts = {
        item["input_status"]: (item["output_status"], item["count"])
        for item in manifest["system_status_cohorts"]
    }
    assert cohorts["missing"] == ("stable", 6)
    assert cohorts["active"] == ("stable", 1)
    assert cohorts["archived"] == ("deprecated", 1)
    assert b"private-active-id" not in manifest_raw
    assert b"private-archived-id" not in manifest_raw
    assert b"private reason remains private" not in manifest_raw
    assert b"Archived private body" not in manifest_raw
    activity = (workspace / "staging" / "activity.jsonl").read_text().splitlines()
    assert len(activity) == 1
    assert "private reason remains private" not in activity[0]


@pytest.mark.parametrize(
    "status_yaml",
    ["private-unknown-status", "missing", "[private-non-string-status]", "null"],
)
def test_system_invalid_status_fails_before_journal_without_private_values(
    tmp_path: Path, status_yaml: str
) -> None:
    source, runtime = _roots(tmp_path)
    (source / "system" / "bad.md").write_text(
        "---\n"
        "uid: private-system-identity\n"
        f"status: {status_yaml}\n"
        "title: Private system title\n"
        "---\n"
        "Private system body.\n"
    )
    expected_hash = canonical_json_sha256_strict(
        {"source": "uid", "value": "private-system-identity"}
    )

    with pytest.raises(ValueError, match=expected_hash) as raised:
        prepare_okf_workspace(source, runtime, "bad-system-status")

    message = str(raised.value)
    assert "bad.md" in message
    assert "private-system-identity" not in message
    assert "private-unknown-status" not in message
    assert "private-non-string-status" not in message
    assert "Private system title" not in message
    assert "Private system body" not in message
    assert not (runtime / "migrations").exists()


def test_system_identity_hash_is_idempotent_across_repeated_preparation(
    tmp_path: Path,
) -> None:
    source_a, runtime_a = _roots(tmp_path / "a")
    source_b, runtime_b = _roots(tmp_path / "b")

    workspace_a = prepare_okf_workspace(source_a, runtime_a, "repeat-a")
    workspace_b = prepare_okf_workspace(source_b, runtime_b, "repeat-b")
    manifest_a = json.loads((workspace_a / "dry-run-manifest.json").read_bytes())
    manifest_b = json.loads((workspace_b / "dry-run-manifest.json").read_bytes())

    assert _snapshot(workspace_a / "staging") == _snapshot(workspace_b / "staging")
    assert manifest_a["system_documents"] == manifest_b["system_documents"]
    assert manifest_a["system_status_cohorts"] == manifest_b[
        "system_status_cohorts"
    ]


def test_system_unresolved_link_error_does_not_disclose_private_target(
    tmp_path: Path,
) -> None:
    source, runtime = _roots(tmp_path)
    path = source / "system" / "current-state.md"
    path.write_bytes(path.read_bytes() + b"[[private-system-target]]\n")
    expected_hash = canonical_json_sha256_strict(
        {"source": "identity", "value": "current-state"}
    )

    with pytest.raises(ValueError, match=expected_hash) as raised:
        prepare_okf_workspace(source, runtime, "private-unresolved")

    message = str(raised.value)
    assert "current-state.md" in message
    assert "private-system-target" not in message
    assert not (runtime / "migrations").exists()


def test_workspace_rejects_cross_volume_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, runtime = _roots(tmp_path)
    source_device = okf_workspace._device_id(source.resolve())
    monkeypatch.setattr(
        okf_workspace,
        "_device_id",
        lambda path: source_device + 1 if path == runtime.resolve() else source_device,
    )

    with pytest.raises(ValueError, match="same volume"):
        prepare_okf_workspace(source, runtime, "cross-device")
    assert not (runtime / "migrations").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("page-to-system", "unresolved wikilinks"),
        ("invalid-system-status", "invalid system lifecycle"),
        ("unsafe-path", "bundle-relative"),
    ],
)
def test_workspace_preflight_rejects_unsafe_inputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    source, runtime = _roots(tmp_path)
    if mutation == "page-to-system":
        path = source / "pages" / "notes" / "source.md"
        path.write_bytes(path.read_bytes() + b"[[current-state]]\n")
    elif mutation == "invalid-system-status":
        path = source / "system" / "current-state.md"
        path.write_text(
            path.read_text().replace("registry_state: internal", "status: internal")
        )
    else:
        (source / "raw" / "bad\\path.jsonl").write_text("unsafe")

    with pytest.raises(ValueError, match=message):
        prepare_okf_workspace(source, runtime, mutation)
    _assert_not_prepared(runtime, mutation)


def test_workspace_raw_manifest_drift_fails_before_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, runtime = _roots(tmp_path)
    real_raw_sources = okf_workspace._raw_sources
    calls = 0

    def drifting_raw_sources(root: Path) -> tuple[RawSource, ...]:
        nonlocal calls
        calls += 1
        sources = real_raw_sources(root)
        return sources if calls == 1 else (*sources, RawSource("late.jsonl", b""))

    monkeypatch.setattr(okf_workspace, "_raw_sources", drifting_raw_sources)
    with pytest.raises(ValueError, match="raw manifest changed"):
        prepare_okf_workspace(source, runtime, "raw-drift")
    _assert_not_prepared(runtime, "raw-drift")


def test_workspace_conformance_and_roundtrip_fail_before_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, runtime = _roots(tmp_path)
    monkeypatch.setattr(
        okf_workspace,
        "validate_pages_bundle",
        lambda _root: (ConformanceIssue("error", "injected", "index.md"),),
    )
    with pytest.raises(ValueError, match="OKF conformance failed"):
        prepare_okf_workspace(source, runtime, "bad-conformance")
    _assert_not_prepared(runtime, "bad-conformance")

    monkeypatch.undo()
    source, runtime = _roots(tmp_path / "roundtrip")
    real_write = okf_workspace._write

    def corrupt_system(path: Path, data: bytes) -> None:
        real_write(path, b"not canonical" if path.name == "current-state.md" else data)

    monkeypatch.setattr(okf_workspace, "_write", corrupt_system)
    with pytest.raises(ValueError, match="semantic round-trip failed"):
        prepare_okf_workspace(source, runtime, "bad-roundtrip")
    _assert_not_prepared(runtime, "bad-roundtrip")


def test_workspace_fsyncs_staging_bottom_up_before_journal_then_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, runtime = _roots(tmp_path)
    events: list[tuple[str, Path]] = []
    real_fsync = okf_workspace.fsync_directory
    real_write = okf_workspace._write

    def tracked_fsync(path: Path) -> None:
        events.append(("fsync", Path(path)))
        real_fsync(path)

    def tracked_write(path: Path, data: bytes) -> None:
        if path.name in {"journal.json", okf_workspace.RESTART_REFUSAL_FILENAME}:
            events.append(("write", path))
        real_write(path, data)

    monkeypatch.setattr(okf_workspace, "fsync_directory", tracked_fsync)
    monkeypatch.setattr(okf_workspace, "_write", tracked_write)
    workspace = prepare_okf_workspace(source, runtime, "ordered-gate")

    journal_event = ("write", workspace / "journal.json")
    sentinel_event = (
        "write",
        workspace / okf_workspace.RESTART_REFUSAL_FILENAME,
    )
    journal_index = events.index(journal_event)
    assert journal_index < events.index(sentinel_event)
    directories = {
        runtime.resolve(),
        workspace.parent,
        workspace,
        *(path for path in (workspace / "staging").rglob("*") if path.is_dir()),
        workspace / "staging",
    }
    last_fsync = {
        directory: max(
            index
            for index, event in enumerate(events)
            if event == ("fsync", directory)
        )
        for directory in directories
    }
    assert all(index < journal_index for index in last_fsync.values())
    assert all(
        last_fsync[directory] < last_fsync[directory.parent]
        for directory in directories
        if directory.parent in directories
    )


@pytest.mark.parametrize("boundary", ["write", "replace", "fsync"])
def test_workspace_faults_never_leave_a_prepared_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    source, runtime = _roots(tmp_path)
    run_id = f"fault-{boundary}"
    expected_workspace = runtime.resolve() / "migrations" / run_id
    if boundary == "write":
        monkeypatch.setattr(
            okf_workspace,
            "atomic_write_bytes",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
        )
    elif boundary == "replace":
        real_replace = durable_state.os.replace

        def fail_sentinel_replace(source_path: object, target_path: object) -> None:
            if Path(target_path).name == okf_workspace.RESTART_REFUSAL_FILENAME:
                raise OSError("replace failed")
            real_replace(source_path, target_path)

        monkeypatch.setattr(durable_state.os, "replace", fail_sentinel_replace)
    else:
        real_fsync_directory = durable_state.fsync_directory

        def fail_sentinel_fsync(path: Path) -> None:
            if (
                Path(path) == expected_workspace
                and (expected_workspace / okf_workspace.RESTART_REFUSAL_FILENAME).exists()
            ):
                raise OSError("fsync failed")
            real_fsync_directory(path)

        monkeypatch.setattr(durable_state, "fsync_directory", fail_sentinel_fsync)

    with pytest.raises(OSError, match=f"{boundary} failed"):
        prepare_okf_workspace(source, runtime, run_id)
    _assert_not_prepared(runtime, run_id)
