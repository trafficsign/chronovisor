from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.runtime_ownership import manifests
from scripts.runtime_ownership.manifests import (
    ANALYZER_MANIFEST_KIND,
    ANALYZER_PATHS,
    FROZEN_SOURCE_REVISION,
    MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
    MACHINE_FACT_TOOLCHAIN_PATHS,
    SOURCE_MANIFEST_KIND,
    ManifestError,
    build_manifest,
    committed_snapshot,
    current_head_revision,
    resolve_full_revision,
    selected_paths_unchanged,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CURRENT_HEAD = "f90202f1d1b9b2ed44075f38b0668c91fc0f196f"
TOOLCHAIN_PRECOMMIT_HEAD = "11e2acf77a53edf520e3cce5d2e5decd16cd06c5"


def _git(repository: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    return completed.stdout


def _write(
    repository: Path, path: str, content: bytes, *, executable: bool = False
) -> None:
    target = repository / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    target.chmod(0o755 if executable else 0o644)


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", "-A")
    _git(repository, "commit", "--quiet", "-m", message)
    return _git(repository, "rev-parse", "HEAD").decode("ascii").strip()


def _commit_index(repository: Path, message: str) -> str:
    _git(repository, "commit", "--quiet", "-m", message)
    return _git(repository, "rev-parse", "HEAD").decode("ascii").strip()


def _initialize_repository(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "--quiet", "--object-format=sha1")
    _git(path, "config", "user.name", "Manifest Test")
    _git(path, "config", "user.email", "manifest@example.test")
    _git(path, "config", "core.filemode", "true")
    return path


def _populate_source(repository: Path) -> None:
    _write(repository, "pyproject.toml", b"[project]\nname='fixture'\n")
    _write(repository, "src/chronovisor/__init__.py", b"VALUE = 1\n")
    _write(repository, "launchd/com.example.fixture.plist", b"<plist/>\n")
    _write(
        repository,
        "scripts/chronovisor-fixture",
        b"#!/bin/sh\nexit 0\n",
        executable=True,
    )


@pytest.fixture
def source_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = _initialize_repository(tmp_path / "repository")
    _populate_source(repository)
    return repository, _commit(repository, "source")


@pytest.fixture
def analyzer_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = _initialize_repository(tmp_path / "repository")
    _populate_source(repository)
    for index, path in enumerate(ANALYZER_PATHS):
        _write(repository, path, f"ROW = {index}\n".encode())
    return repository, _commit(repository, "analyzer")


@pytest.fixture
def toolchain_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = _initialize_repository(tmp_path / "repository")
    for index, path in enumerate(MACHINE_FACT_TOOLCHAIN_PATHS):
        _write(repository, path, f"ROW = {index}\n".encode())
    return repository, _commit(repository, "toolchain")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reseal(manifest: dict[str, Any]) -> None:
    manifest["files_sha256"] = hashlib.sha256(
        _canonical_bytes(manifest["files"])
    ).hexdigest()
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _source_manifest(repository: Path, revision: str) -> dict[str, Any]:
    return build_manifest(
        repository,
        revision,
        manifest_kind=SOURCE_MANIFEST_KIND,
        expected_revision=revision,
    )


def _verify_source(
    repository: Path, manifest: object, revision: str
) -> manifests.CommittedSnapshot:
    return verify_manifest(
        repository,
        manifest,
        expected_kind=SOURCE_MANIFEST_KIND,
        expected_revision=revision,
    )


@pytest.mark.parametrize(
    "revision_factory",
    [
        lambda revision: revision[:12],
        lambda _revision: "HEAD",
        lambda _revision: "refs/heads/master",
        lambda revision: revision.upper(),
        lambda _revision: "f" * 40,
    ],
)
def test_resolve_full_revision_rejects_dwim_and_unknown_names(
    source_repository: tuple[Path, str], revision_factory: Any
) -> None:
    repository, revision = source_repository

    with pytest.raises(ManifestError):
        resolve_full_revision(repository, revision_factory(revision))


def test_resolve_full_revision_rejects_noncommit_object(
    source_repository: tuple[Path, str],
) -> None:
    repository, _revision = source_repository
    blob_oid = (
        _git(repository, "hash-object", "-w", "--stdin", input_bytes=b"not a commit")
        .decode("ascii")
        .strip()
    )

    with pytest.raises(ManifestError):
        resolve_full_revision(repository, blob_oid)


def test_parse_tree_rejects_unterminated_nul_framing() -> None:
    unterminated = b"100644 blob " + b"a" * 40 + b"       1\tpyproject.toml"

    assert manifests._parse_tree(b"") == ()
    with pytest.raises(ManifestError, match="missing its trailing NUL"):
        manifests._parse_tree(unterminated)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            b"100644 blob " + b"a" * 40 + b"       1\t\0",
            "empty path",
        ),
        (
            (b"100644 blob " + b"a" * 40 + b"       1\tpyproject.toml\0") * 2,
            "duplicate path",
        ),
        (
            b"100644 blob " + b"a" * 40 + b"       1\t\xff\0",
            "not valid UTF-8",
        ),
    ],
)
def test_parse_tree_rejects_empty_and_duplicate_paths(raw: bytes, message: str) -> None:
    with pytest.raises(ManifestError, match=message):
        manifests._parse_tree(raw)


@pytest.mark.parametrize("raw_size", [b"+1", b"01", b"-0"])
def test_parse_tree_rejects_noncanonical_decimal_sizes(raw_size: bytes) -> None:
    raw = b"100644 blob " + b"a" * 40 + b"    " + raw_size + b"\tpyproject.toml\0"

    with pytest.raises(ManifestError, match="canonical decimal"):
        manifests._parse_tree(raw)


@pytest.mark.parametrize("raw_size", [b"+1", b"01", b"-0"])
def test_cat_file_batch_rejects_noncanonical_decimal_sizes(
    monkeypatch: pytest.MonkeyPatch, raw_size: bytes
) -> None:
    oid = "a" * 40

    def malformed_git(
        _repository: Path,
        *arguments: str,
        input_bytes: bytes | None = None,
        allowed_returncodes: frozenset[int] = frozenset({0}),
    ) -> subprocess.CompletedProcess[bytes]:
        del allowed_returncodes
        assert arguments == ("cat-file", "--batch")
        assert input_bytes == f"{oid}\n".encode("ascii")
        stdout = oid.encode("ascii") + b" blob " + raw_size + b"\nX\n"
        return subprocess.CompletedProcess(
            ["git", *arguments], 0, stdout=stdout, stderr=b""
        )

    monkeypatch.setattr(manifests, "_git", malformed_git)

    with pytest.raises(ManifestError, match="canonical decimal"):
        manifests._cat_file_batch(ROOT, [oid])


@pytest.mark.parametrize("target", ["object-format", "revision", "head"])
def test_git_ascii_boundaries_normalize_non_ascii_stdout(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    revision = "a" * 40

    def malformed_git(
        _repository: Path,
        *arguments: str,
        input_bytes: bytes | None = None,
        allowed_returncodes: frozenset[int] = frozenset({0}),
    ) -> subprocess.CompletedProcess[bytes]:
        del input_bytes, allowed_returncodes
        if arguments == ("rev-parse", "--show-object-format"):
            stdout = b"sha\xff\n" if target == "object-format" else b"sha1\n"
        elif arguments in {
            ("rev-parse", "--verify", f"{revision}^{{commit}}"),
            ("rev-parse", "--verify", "HEAD^{commit}"),
        }:
            stdout = b"a" * 39 + b"\xff\n"
        else:
            raise AssertionError(arguments)
        return subprocess.CompletedProcess(
            ["git", *arguments], 0, stdout=stdout, stderr=b""
        )

    monkeypatch.setattr(manifests, "_git", malformed_git)

    with pytest.raises(ManifestError, match="not valid ASCII"):
        if target == "object-format":
            manifests._git_object_format(ROOT)
        elif target == "revision":
            resolve_full_revision(ROOT, revision)
        else:
            manifests._head_revision(ROOT)


def test_committed_snapshot_exposes_exact_blob_metadata_and_raw_bytes(
    source_repository: tuple[Path, str],
) -> None:
    repository, revision = source_repository
    snapshot = committed_snapshot(
        repository, revision, manifest_kind=SOURCE_MANIFEST_KIND
    )

    assert snapshot.revision == revision
    assert snapshot.git_object_format == "sha1"
    assert [row.path for row in snapshot.files] == sorted(
        row.path for row in snapshot.files
    )
    assert snapshot.read_bytes("src/chronovisor/__init__.py") == b"VALUE = 1\n"
    assert {row.git_type for row in snapshot.files} == {"blob"}
    assert {row.path: row.git_mode for row in snapshot.files}[
        "scripts/chronovisor-fixture"
    ] == "100755"


def test_snapshot_uses_required_git_plumbing_and_never_worktree_reads(
    source_repository: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, revision = source_repository
    commands: list[list[str]] = []
    real_run = subprocess.run

    def recorded_run(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        return cast(subprocess.CompletedProcess[bytes], real_run(command, **kwargs))

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("worktree filesystem selection is forbidden")

    monkeypatch.setattr(
        "scripts.runtime_ownership.manifests.subprocess.run", recorded_run
    )
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)

    _source_manifest(repository, revision)

    assert [
        "git",
        "ls-tree",
        "-r",
        "-l",
        "-z",
        "--full-tree",
        revision,
    ] in commands
    assert ["git", "cat-file", "--batch"] in commands


def test_dirty_tracked_and_untracked_matching_files_are_ignored(
    source_repository: tuple[Path, str],
) -> None:
    repository, revision = source_repository
    expected = _source_manifest(repository, revision)
    _write(repository, "src/chronovisor/__init__.py", b"DIRTY = True\n")
    _write(repository, "src/chronovisor/untracked.py", b"UNTRACKED = True\n")

    assert _source_manifest(repository, revision) == expected


def test_distinct_paths_may_share_one_blob_oid(
    source_repository: tuple[Path, str],
) -> None:
    repository, _revision = source_repository
    _write(repository, "src/chronovisor/copy.py", b"VALUE = 1\n")
    revision = _commit(repository, "shared blob")
    snapshot = committed_snapshot(
        repository, revision, manifest_kind=SOURCE_MANIFEST_KIND
    )
    matching = [
        row
        for row in snapshot.files
        if row.path in {"src/chronovisor/__init__.py", "src/chronovisor/copy.py"}
    ]

    assert len(matching) == 2
    assert len({row.blob_oid for row in matching}) == 1
    manifest = _source_manifest(repository, revision)
    _verify_source(repository, manifest, revision)


@pytest.mark.parametrize(
    ("path", "mode"),
    [
        ("src/chronovisor/__init__.py", 0o755),
        ("scripts/chronovisor-fixture", 0o644),
    ],
)
def test_source_selection_rejects_wrong_modes(
    source_repository: tuple[Path, str], path: str, mode: int
) -> None:
    repository, _revision = source_repository
    (repository / path).chmod(mode)
    revision = _commit(repository, "wrong mode")

    with pytest.raises(ManifestError, match="invalid git mode"):
        _source_manifest(repository, revision)


def test_source_selection_rejects_symlink(
    source_repository: tuple[Path, str],
) -> None:
    repository, _revision = source_repository
    target = repository / "src/chronovisor/__init__.py"
    target.unlink()
    target.symlink_to("target.py")
    revision = _commit(repository, "symlink")

    with pytest.raises(ManifestError, match="invalid git mode"):
        _source_manifest(repository, revision)


def test_source_selection_rejects_nonblob_gitlink(
    source_repository: tuple[Path, str],
) -> None:
    repository, revision = source_repository
    _git(
        repository,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{revision},src/chronovisor/vendor.py",
    )
    gitlink_revision = _commit_index(repository, "gitlink")

    with pytest.raises(ManifestError, match="not a blob"):
        _source_manifest(repository, gitlink_revision)


def test_analyzer_selection_is_exactly_the_fifteen_direct_files(
    analyzer_repository: tuple[Path, str],
) -> None:
    repository, revision = analyzer_repository
    manifest = build_manifest(
        repository,
        revision,
        manifest_kind=ANALYZER_MANIFEST_KIND,
        expected_revision=revision,
    )

    assert manifest["selection"] == {
        "python": "scripts/runtime_ownership/access*.py",
        "exact_file_count": 15,
    }
    assert [row["path"] for row in manifest["files"]] == list(ANALYZER_PATHS)


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_analyzer_selection_rejects_path_set_drift(
    analyzer_repository: tuple[Path, str], change: str
) -> None:
    repository, _revision = analyzer_repository
    if change == "extra":
        _write(repository, "scripts/runtime_ownership/access_extra.py", b"EXTRA=1\n")
    else:
        (repository / ANALYZER_PATHS[0]).unlink()
    revision = _commit(repository, change)

    with pytest.raises(ManifestError, match="exact 15 paths"):
        build_manifest(
            repository,
            revision,
            manifest_kind=ANALYZER_MANIFEST_KIND,
            expected_revision=revision,
        )


def test_analyzer_selection_rejects_executable_mode(
    analyzer_repository: tuple[Path, str],
) -> None:
    repository, _revision = analyzer_repository
    (repository / ANALYZER_PATHS[0]).chmod(0o755)
    revision = _commit(repository, "mode")

    with pytest.raises(ManifestError, match="invalid git mode"):
        build_manifest(
            repository,
            revision,
            manifest_kind=ANALYZER_MANIFEST_KIND,
            expected_revision=revision,
        )


def test_analyzer_selection_rejects_duplicate_rows_even_with_exact_path_set() -> None:
    rows = [
        manifests._TreeEntry(
            path=path,
            git_mode="100644",
            git_type="blob",
            object_oid="a" * 40,
            byte_count=1,
        )
        for path in ANALYZER_PATHS
    ]
    rows.append(rows[0])

    with pytest.raises(ManifestError, match="exact 15 paths"):
        manifests._select_entries(rows, ANALYZER_MANIFEST_KIND)


def test_manifest_schema_counts_and_hashes_are_canonical(
    source_repository: tuple[Path, str],
) -> None:
    repository, revision = source_repository
    manifest = _source_manifest(repository, revision)
    encoded = _canonical_bytes(manifest)
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }

    assert b"\n" not in encoded
    assert manifest["selection"] == {
        "pyproject": "pyproject.toml",
        "python": "src/chronovisor/**/*.py",
        "launchd": "launchd/*.plist",
        "scripts": "scripts/chronovisor-*",
        "scripts_require_executable": True,
    }
    assert manifest["counts"] == {
        "files": 4,
        "bytes": sum(row["byte_count"] for row in manifest["files"]),
    }
    assert (
        manifest["files_sha256"]
        == hashlib.sha256(_canonical_bytes(manifest["files"])).hexdigest()
    )
    assert (
        manifest["manifest_sha256"]
        == hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    )
    assert _verify_source(repository, manifest, revision).revision == revision


def test_canonical_json_normalizes_unicode_encoding_failure() -> None:
    with pytest.raises(ManifestError, match="canonical JSON"):
        manifests._canonical_bytes({"invalid": "\ud800"})


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_verify_rejects_top_level_key_drift(
    source_repository: tuple[Path, str], mutation: str
) -> None:
    repository, revision = source_repository
    manifest = _source_manifest(repository, revision)
    if mutation == "missing":
        manifest.pop("counts")
    else:
        manifest["unknown"] = True

    with pytest.raises(ManifestError, match="manifest keys mismatch"):
        _verify_source(repository, manifest, revision)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_verify_rejects_file_row_key_drift(
    source_repository: tuple[Path, str], mutation: str
) -> None:
    repository, revision = source_repository
    manifest = _source_manifest(repository, revision)
    file_row = manifest["files"][0]
    if mutation == "missing":
        file_row.pop("sha256")
    else:
        file_row["unknown"] = True
    _reseal(manifest)

    with pytest.raises(ManifestError, match=r"files\[0\] keys mismatch"):
        _verify_source(repository, manifest, revision)


@pytest.mark.parametrize("mutation", ["duplicate", "unsorted"])
def test_verify_rejects_duplicate_or_unsorted_file_rows(
    source_repository: tuple[Path, str], mutation: str
) -> None:
    repository, revision = source_repository
    manifest = _source_manifest(repository, revision)
    if mutation == "duplicate":
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
    else:
        manifest["files"][0], manifest["files"][1] = (
            manifest["files"][1],
            manifest["files"][0],
        )
    manifest["counts"] = {
        "files": len(manifest["files"]),
        "bytes": sum(row["byte_count"] for row in manifest["files"]),
    }
    _reseal(manifest)

    with pytest.raises(ManifestError, match="sorted|unique"):
        _verify_source(repository, manifest, revision)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("git_object_format", "sha256", "git_object_format"),
        ("files_sha256", "0" * 64, "files_sha256"),
        ("manifest_sha256", "0" * 64, "manifest_sha256"),
    ],
)
def test_verify_rejects_object_format_and_hash_tampering(
    source_repository: tuple[Path, str], field: str, value: str, message: str
) -> None:
    repository, revision = source_repository
    manifest = _source_manifest(repository, revision)
    manifest[field] = value

    with pytest.raises(ManifestError, match=message):
        _verify_source(repository, manifest, revision)


def test_verify_rejects_blob_oid_length_and_count_tampering(
    source_repository: tuple[Path, str],
) -> None:
    repository, revision = source_repository
    bad_oid = _source_manifest(repository, revision)
    bad_oid["files"][0]["blob_oid"] = "0" * 64
    _reseal(bad_oid)
    with pytest.raises(ManifestError, match="blob_oid"):
        _verify_source(repository, bad_oid, revision)

    bad_count = _source_manifest(repository, revision)
    bad_count["counts"]["bytes"] += 1
    _reseal(bad_count)
    with pytest.raises(ManifestError, match="counts"):
        _verify_source(repository, bad_count, revision)


def test_verify_rebuilds_tree_instead_of_trusting_resealed_rows(
    source_repository: tuple[Path, str],
) -> None:
    repository, revision = source_repository
    manifest = _source_manifest(repository, revision)
    first = manifest["files"][0]
    second = manifest["files"][1]
    first.update(
        {
            "blob_oid": second["blob_oid"],
            "byte_count": second["byte_count"],
            "sha256": second["sha256"],
        }
    )
    manifest["counts"]["bytes"] = sum(row["byte_count"] for row in manifest["files"])
    _reseal(manifest)

    with pytest.raises(ManifestError, match="independently rebuilt"):
        _verify_source(repository, manifest, revision)


def test_verify_rejects_selection_and_expected_kind_drift(
    source_repository: tuple[Path, str],
) -> None:
    repository, revision = source_repository
    selection = _source_manifest(repository, revision)
    selection["selection"]["scripts_require_executable"] = False
    _reseal(selection)
    with pytest.raises(ManifestError, match="selection object"):
        _verify_source(repository, selection, revision)

    valid = _source_manifest(repository, revision)
    with pytest.raises(ManifestError, match="expected_kind"):
        verify_manifest(
            repository,
            valid,
            expected_kind=ANALYZER_MANIFEST_KIND,
            expected_revision=revision,
        )
    with pytest.raises(ManifestError, match="expected_kind"):
        verify_manifest(
            repository,
            valid,
            expected_kind=[],
            expected_revision=revision,
        )


def test_revision_policy_is_required_and_options_are_exclusive(
    source_repository: tuple[Path, str],
) -> None:
    repository, revision = source_repository
    with pytest.raises(ManifestError, match="revision policy is required"):
        build_manifest(repository, revision, manifest_kind=SOURCE_MANIFEST_KIND)
    with pytest.raises(ManifestError, match="mutually exclusive"):
        build_manifest(
            repository,
            revision,
            manifest_kind=SOURCE_MANIFEST_KIND,
            expected_revision=revision,
            require_current_unchanged=True,
        )


def test_current_policy_allows_docs_only_descendant(
    source_repository: tuple[Path, str],
) -> None:
    repository, source_revision = source_repository
    _write(repository, "docs/note.md", b"docs only\n")
    _commit(repository, "docs")

    manifest = build_manifest(
        repository,
        source_revision,
        manifest_kind=SOURCE_MANIFEST_KIND,
        require_current_unchanged=True,
    )
    verify_manifest(
        repository,
        manifest,
        expected_kind=SOURCE_MANIFEST_KIND,
        require_current_unchanged=True,
    )


def test_current_policy_rejects_selected_path_drift(
    source_repository: tuple[Path, str],
) -> None:
    repository, source_revision = source_repository
    _write(repository, "src/chronovisor/__init__.py", b"VALUE = 2\n")
    drifted_revision = _commit(repository, "selected drift")

    assert not selected_paths_unchanged(
        repository,
        source_revision,
        drifted_revision,
        manifest_kind=SOURCE_MANIFEST_KIND,
    )
    with pytest.raises(ManifestError, match="selected committed paths changed"):
        build_manifest(
            repository,
            source_revision,
            manifest_kind=SOURCE_MANIFEST_KIND,
            require_current_unchanged=True,
        )


def test_current_policy_rejects_nonancestor_revision(
    source_repository: tuple[Path, str],
) -> None:
    repository, base = source_repository
    branch = _git(repository, "branch", "--show-current").decode("ascii").strip()
    _git(repository, "checkout", "--quiet", "-b", "side")
    _write(repository, "docs/side.md", b"side\n")
    side_revision = _commit(repository, "side")
    _git(repository, "checkout", "--quiet", branch)
    _write(repository, "docs/main.md", b"main\n")
    main_revision = _commit(repository, "main")

    assert base != side_revision != main_revision
    assert not selected_paths_unchanged(
        repository,
        side_revision,
        main_revision,
        manifest_kind=SOURCE_MANIFEST_KIND,
    )
    with pytest.raises(ManifestError, match="ancestor of HEAD"):
        build_manifest(
            repository,
            side_revision,
            manifest_kind=SOURCE_MANIFEST_KIND,
            require_current_unchanged=True,
        )


def test_raw_crlf_bytes_change_blob_and_manifest_hashes(
    source_repository: tuple[Path, str],
) -> None:
    repository, lf_revision = source_repository
    lf_manifest = _source_manifest(repository, lf_revision)
    _write(repository, "src/chronovisor/__init__.py", b"VALUE = 1\r\n")
    crlf_revision = _commit(repository, "crlf")
    crlf_manifest = _source_manifest(repository, crlf_revision)
    snapshot = committed_snapshot(
        repository, crlf_revision, manifest_kind=SOURCE_MANIFEST_KIND
    )

    assert snapshot.read_bytes("src/chronovisor/__init__.py") == b"VALUE = 1\r\n"
    assert lf_manifest["files_sha256"] != crlf_manifest["files_sha256"]
    assert lf_manifest["manifest_sha256"] != crlf_manifest["manifest_sha256"]
    assert not selected_paths_unchanged(
        repository,
        lf_revision,
        crlf_revision,
        manifest_kind=SOURCE_MANIFEST_KIND,
    )


def test_real_repository_frozen_source_is_exactly_pinned() -> None:
    manifest = build_manifest(
        ROOT,
        FROZEN_SOURCE_REVISION,
        manifest_kind=SOURCE_MANIFEST_KIND,
        expected_revision=FROZEN_SOURCE_REVISION,
    )

    assert manifest["counts"] == {"files": 296, "bytes": 7455391}
    assert manifest["files_sha256"] == (
        "6693cc159f8ab213a513225b73096a30e4ae629404d6b5b7906d63cb6a52e4ef"
    )
    assert manifest["manifest_sha256"] == (
        "8cbf1ec787ad3e10b989b4131eee0e4be469bcf7726bd18351b3db5019255343"
    )
    verify_manifest(
        ROOT,
        manifest,
        expected_kind=SOURCE_MANIFEST_KIND,
        expected_revision=FROZEN_SOURCE_REVISION,
    )
    with pytest.raises(ManifestError, match="expected_revision"):
        build_manifest(
            ROOT,
            CURRENT_HEAD,
            manifest_kind=SOURCE_MANIFEST_KIND,
            expected_revision=FROZEN_SOURCE_REVISION,
        )


def test_real_head_canonical_source_and_analyzer_manifests() -> None:
    assert resolve_full_revision(ROOT, CURRENT_HEAD) == CURRENT_HEAD
    source = build_manifest(
        ROOT,
        CURRENT_HEAD,
        manifest_kind=SOURCE_MANIFEST_KIND,
        require_current_unchanged=True,
    )
    analyzer = build_manifest(
        ROOT,
        CURRENT_HEAD,
        manifest_kind=ANALYZER_MANIFEST_KIND,
        require_current_unchanged=True,
    )

    assert source["counts"] == {"files": 296, "bytes": 7456781}
    assert source["files_sha256"] == (
        "be2ad06f687bc619a89d12ad6274d6843b26278e2094d420146105c398e73cee"
    )
    assert source["manifest_sha256"] == (
        "268a6d8ca2fbd7d4877f78a3f5c6b14fd0e7e36d760173be9ce1a05e6703f43a"
    )
    assert analyzer["counts"] == {"files": 15, "bytes": 556664}
    assert analyzer["files_sha256"] == (
        "4503080209a3a9632fe922afb2191974a6619a2f971b660c741d8b3978032b31"
    )
    assert analyzer["manifest_sha256"] == (
        "74d6671eaf0ed4a6def71b28829bdbb7b4aa6392831d01beb2384dfe7b34948b"
    )


def test_toolchain_manifest_selects_exact_three_and_ignores_worktree(
    toolchain_repository: tuple[Path, str],
) -> None:
    repository, revision = toolchain_repository
    manifest = build_manifest(
        repository,
        revision,
        manifest_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
        expected_revision=revision,
    )
    assert current_head_revision(repository) == revision
    assert manifest["selection"] == {
        "python": list(MACHINE_FACT_TOOLCHAIN_PATHS),
        "exact_file_count": 3,
    }
    assert [row["path"] for row in manifest["files"]] == sorted(
        MACHINE_FACT_TOOLCHAIN_PATHS
    )
    assert manifest["counts"]["files"] == 3
    snapshot = verify_manifest(
        repository,
        manifest,
        expected_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
        expected_revision=revision,
    )
    assert tuple(row.path for row in snapshot.files) == tuple(
        sorted(MACHINE_FACT_TOOLCHAIN_PATHS)
    )

    _write(repository, MACHINE_FACT_TOOLCHAIN_PATHS[0], b"dirty\n")
    _write(repository, "untracked.py", b"ignored\n")
    assert (
        build_manifest(
            repository,
            revision,
            manifest_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
            expected_revision=revision,
        )
        == manifest
    )


def test_toolchain_manifest_rejects_missing_and_reseeded_extra_path(
    toolchain_repository: tuple[Path, str],
) -> None:
    repository, revision = toolchain_repository
    manifest = build_manifest(
        repository,
        revision,
        manifest_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
        expected_revision=revision,
    )
    extra = copy.deepcopy(manifest)
    extra_row = {**extra["files"][0], "path": "scripts/runtime_ownership/extra.py"}
    extra["files"].append(extra_row)
    extra["files"].sort(key=lambda row: row["path"])
    extra["counts"]["files"] += 1
    extra["counts"]["bytes"] += extra_row["byte_count"]
    _reseal(extra)
    with pytest.raises(ManifestError, match="independently rebuilt"):
        verify_manifest(
            repository,
            extra,
            expected_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
            expected_revision=revision,
        )

    _git(repository, "rm", "--quiet", MACHINE_FACT_TOOLCHAIN_PATHS[0])
    missing_revision = _commit_index(repository, "missing toolchain path")
    with pytest.raises(ManifestError, match="exact 3 paths"):
        committed_snapshot(
            repository,
            missing_revision,
            manifest_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
        )


@pytest.mark.parametrize("invalid_kind", ["executable", "symlink", "gitlink"])
def test_toolchain_manifest_rejects_non_regular_or_executable_inputs(
    toolchain_repository: tuple[Path, str], invalid_kind: str
) -> None:
    repository, revision = toolchain_repository
    path = MACHINE_FACT_TOOLCHAIN_PATHS[0]
    if invalid_kind == "executable":
        _git(repository, "update-index", "--chmod=+x", path)
    elif invalid_kind == "symlink":
        target = repository / path
        target.unlink()
        target.symlink_to("declarations.py")
        _git(repository, "add", path)
    else:
        _git(
            repository,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{revision},{path}",
        )
    invalid_revision = _commit_index(repository, invalid_kind)
    with pytest.raises(ManifestError, match="not a blob|invalid git mode"):
        committed_snapshot(
            repository,
            invalid_revision,
            manifest_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
        )


def test_real_precommit_toolchain_manifest_is_git_only_and_canonical() -> None:
    assert resolve_full_revision(ROOT, TOOLCHAIN_PRECOMMIT_HEAD) == (
        TOOLCHAIN_PRECOMMIT_HEAD
    )
    manifest = build_manifest(
        ROOT,
        TOOLCHAIN_PRECOMMIT_HEAD,
        manifest_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
        expected_revision=TOOLCHAIN_PRECOMMIT_HEAD,
    )
    assert manifest["revision"] == TOOLCHAIN_PRECOMMIT_HEAD
    assert manifest["counts"] == {"files": 3, "bytes": 170_655}
    assert manifest["files_sha256"] == (
        "343d8385b93004be7f80aff326393b6965200313c124dc4e7bc0cca05854b93b"
    )
    assert manifest["manifest_sha256"] == (
        "12cd1e8b14b58e87dffce2a7541a8e152b585742e52aa884b963509be442556d"
    )
    assert [row["path"] for row in manifest["files"]] == sorted(
        MACHINE_FACT_TOOLCHAIN_PATHS
    )
    verify_manifest(
        ROOT,
        manifest,
        expected_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
        expected_revision=TOOLCHAIN_PRECOMMIT_HEAD,
    )
