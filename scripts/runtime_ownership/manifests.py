"""Strict committed-tree manifests for runtime ownership evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

FROZEN_SOURCE_REVISION = "0b8fac4609d60e85ddacc90a0768bb066591f4b8"

ANALYZER_PATHS = (
    "scripts/runtime_ownership/access.py",
    "scripts/runtime_ownership/access_bindings.py",
    "scripts/runtime_ownership/access_class_scopes.py",
    "scripts/runtime_ownership/access_control.py",
    "scripts/runtime_ownership/access_definition_execution.py",
    "scripts/runtime_ownership/access_export_flow.py",
    "scripts/runtime_ownership/access_expressions.py",
    "scripts/runtime_ownership/access_facts.py",
    "scripts/runtime_ownership/access_imports.py",
    "scripts/runtime_ownership/access_model.py",
    "scripts/runtime_ownership/access_outcome_control.py",
    "scripts/runtime_ownership/access_outcomes.py",
    "scripts/runtime_ownership/access_resolver.py",
    "scripts/runtime_ownership/access_sinks.py",
    "scripts/runtime_ownership/access_statements.py",
)

ManifestKind: TypeAlias = Literal[
    "chronovisor-source-manifest",
    "chronovisor-runtime-access-analyzer-manifest",
]

SOURCE_MANIFEST_KIND: ManifestKind = "chronovisor-source-manifest"
ANALYZER_MANIFEST_KIND: ManifestKind = (
    "chronovisor-runtime-access-analyzer-manifest"
)
_MANIFEST_KINDS = frozenset({SOURCE_MANIFEST_KIND, ANALYZER_MANIFEST_KIND})
_FULL_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CANONICAL_DECIMAL_BYTES = re.compile(rb"(?:0|[1-9][0-9]*)\Z")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "manifest_kind",
        "revision",
        "git_object_format",
        "selection",
        "files",
        "counts",
        "files_sha256",
        "manifest_sha256",
    }
)
_FILE_KEYS = frozenset(
    {"path", "git_mode", "byte_count", "blob_oid", "sha256"}
)
_COUNT_KEYS = frozenset({"files", "bytes"})


class ManifestError(ValueError):
    """Raised when committed evidence or a manifest fails closed."""


@dataclass(frozen=True)
class CommittedFile:
    """One selected blob and its exact committed bytes."""

    path: str
    git_mode: str
    git_type: str
    blob_oid: str
    raw_bytes: bytes

    @property
    def byte_count(self) -> int:
        return len(self.raw_bytes)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()


@dataclass(frozen=True)
class CommittedSnapshot:
    """A typed, worktree-independent view of selected committed blobs."""

    revision: str
    git_object_format: str
    manifest_kind: ManifestKind
    files: tuple[CommittedFile, ...]

    def read_bytes(self, path: str) -> bytes:
        """Return exact committed bytes for one selected path."""

        for row in self.files:
            if row.path == path:
                return row.raw_bytes
        raise KeyError(path)


@dataclass(frozen=True)
class _TreeEntry:
    path: str
    git_mode: str
    git_type: str
    object_oid: str
    byte_count: int | None


def _git(
    repository: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            input=input_bytes,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ManifestError(f"git execution failed: {exc}") from exc
    if completed.returncode not in allowed_returncodes:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ManifestError(
            f"git {' '.join(arguments)} failed ({completed.returncode}): {detail}"
        )
    return completed


def _decode_git_text(
    raw: bytes,
    *,
    label: str,
    encoding: Literal["ascii", "utf-8"] = "ascii",
) -> str:
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as exc:
        display_encoding = "ASCII" if encoding == "ascii" else "UTF-8"
        raise ManifestError(
            f"{label} is not valid {display_encoding}"
        ) from exc


def _git_object_format(repository: Path) -> str:
    value = _decode_git_text(
        _git(repository, "rev-parse", "--show-object-format").stdout,
        label="git object format",
    )
    value = value.strip()
    if value != "sha1":
        raise ManifestError(
            "manifest revision schema requires a sha1 repository, got "
            f"{value!r}"
        )
    return value


def resolve_full_revision(repository: Path, revision: str) -> str:
    """Validate an explicit lowercase full commit SHA without DWIM resolution."""

    _git_object_format(repository)
    if not isinstance(revision, str) or _FULL_SHA1.fullmatch(revision) is None:
        raise ManifestError(
            "revision must be an explicit lowercase full 40-character commit SHA"
        )
    completed = _git(
        repository,
        "rev-parse",
        "--verify",
        f"{revision}^{{commit}}",
    )
    resolved = _decode_git_text(
        completed.stdout, label="resolved revision"
    ).strip()
    if resolved != revision:
        raise ManifestError(
            f"revision did not resolve to itself as a commit: {revision}"
        )
    return resolved


def _head_revision(repository: Path) -> str:
    resolved = _decode_git_text(
        _git(repository, "rev-parse", "--verify", "HEAD^{commit}").stdout,
        label="resolved HEAD revision",
    )
    resolved = resolved.strip()
    return resolve_full_revision(repository, resolved)


def _parse_tree(raw: bytes) -> tuple[_TreeEntry, ...]:
    if not raw:
        return ()
    if not raw.endswith(b"\0"):
        raise ManifestError("git ls-tree -z output is missing its trailing NUL")
    rows: list[_TreeEntry] = []
    seen_paths: set[str] = set()
    for record in raw[:-1].split(b"\0"):
        if not record:
            raise ManifestError("git ls-tree -z output contains an empty record")
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, object_type, oid, raw_size = header.split(b" ", 3)
            path = _decode_git_text(
                raw_path, label="git tree path", encoding="utf-8"
            )
            git_mode = _decode_git_text(mode, label="git tree mode")
            git_type = _decode_git_text(object_type, label="git tree type")
            object_oid = _decode_git_text(oid, label="git tree object id")
            size_token = raw_size.lstrip(b" ")
            if size_token == b"-":
                byte_count = None
            elif _CANONICAL_DECIMAL_BYTES.fullmatch(size_token) is not None:
                byte_count = int(size_token)
            else:
                raise ManifestError(
                    "git ls-tree size is not '-' or a canonical decimal"
                )
        except ManifestError:
            raise
        except ValueError as exc:
            raise ManifestError("invalid git ls-tree record") from exc
        if not path:
            raise ManifestError("git ls-tree record has an empty path")
        if path in seen_paths:
            raise ManifestError(f"git ls-tree contains a duplicate path: {path}")
        seen_paths.add(path)
        rows.append(
            _TreeEntry(
                path=path,
                git_mode=git_mode,
                git_type=git_type,
                object_oid=object_oid,
                byte_count=byte_count,
            )
        )
    return tuple(rows)


def _tree_entries(repository: Path, revision: str) -> tuple[_TreeEntry, ...]:
    raw = _git(
        repository,
        "ls-tree",
        "-r",
        "-l",
        "-z",
        "--full-tree",
        revision,
    ).stdout
    return _parse_tree(raw)


def _source_category(path: str) -> str | None:
    if path == "pyproject.toml":
        return "pyproject"
    if path.startswith("src/chronovisor/") and path.endswith(".py"):
        return "python"
    if path.startswith("launchd/"):
        remainder = path.removeprefix("launchd/")
        if "/" not in remainder and remainder.endswith(".plist"):
            return "launchd"
    if path.startswith("scripts/chronovisor-"):
        remainder = path.removeprefix("scripts/")
        if "/" not in remainder:
            return "script"
    return None


def _is_direct_analyzer_path(path: str) -> bool:
    prefix = "scripts/runtime_ownership/"
    if not path.startswith(prefix):
        return False
    remainder = path.removeprefix(prefix)
    return "/" not in remainder and remainder.startswith("access") and remainder.endswith(
        ".py"
    )


def _selection(manifest_kind: ManifestKind) -> dict[str, object]:
    if manifest_kind == SOURCE_MANIFEST_KIND:
        return {
            "pyproject": "pyproject.toml",
            "python": "src/chronovisor/**/*.py",
            "launchd": "launchd/*.plist",
            "scripts": "scripts/chronovisor-*",
            "scripts_require_executable": True,
        }
    if manifest_kind == ANALYZER_MANIFEST_KIND:
        return {
            "python": "scripts/runtime_ownership/access*.py",
            "exact_file_count": 15,
        }
    raise ManifestError(f"unsupported manifest_kind: {manifest_kind!r}")


def _select_entries(
    entries: Sequence[_TreeEntry], manifest_kind: ManifestKind
) -> tuple[_TreeEntry, ...]:
    if manifest_kind == SOURCE_MANIFEST_KIND:
        selected = [row for row in entries if _source_category(row.path) is not None]
        categories = {
            category
            for row in selected
            if (category := _source_category(row.path)) is not None
        }
        missing = {"pyproject", "python", "launchd", "script"} - categories
        if missing:
            raise ManifestError(
                f"source selection categories are missing: {sorted(missing)}"
            )
    elif manifest_kind == ANALYZER_MANIFEST_KIND:
        selected = [row for row in entries if _is_direct_analyzer_path(row.path)]
        selected_paths = {row.path for row in selected}
        expected_paths = set(ANALYZER_PATHS)
        if selected_paths != expected_paths or len(selected) != len(ANALYZER_PATHS):
            raise ManifestError(
                "analyzer selection must be the exact 15 paths; "
                f"missing={sorted(expected_paths - selected_paths)}, "
                f"extra={sorted(selected_paths - expected_paths)}"
            )
    else:
        raise ManifestError(f"unsupported manifest_kind: {manifest_kind!r}")
    return tuple(sorted(selected, key=lambda row: row.path))


def _expected_mode(path: str, manifest_kind: ManifestKind) -> str:
    if manifest_kind == SOURCE_MANIFEST_KIND and path.startswith(
        "scripts/chronovisor-"
    ):
        return "100755"
    return "100644"


def _validate_selected_entries(
    rows: Sequence[_TreeEntry], manifest_kind: ManifestKind
) -> None:
    for row in rows:
        if row.git_type != "blob":
            raise ManifestError(
                f"selected path is not a blob: {row.path} ({row.git_type})"
            )
        if row.git_mode != _expected_mode(row.path, manifest_kind):
            raise ManifestError(
                f"selected path has invalid git mode: {row.path} ({row.git_mode})"
            )
        if row.byte_count is None or row.byte_count < 0:
            raise ManifestError(f"selected blob has invalid byte count: {row.path}")
        if _FULL_SHA1.fullmatch(row.object_oid) is None:
            raise ManifestError(
                f"selected blob has invalid sha1 object id: {row.path}"
            )


def _cat_file_batch(repository: Path, object_oids: Sequence[str]) -> dict[str, bytes]:
    unique_oids = sorted(set(object_oids))
    request = b"".join(f"{oid}\n".encode("ascii") for oid in unique_oids)
    response = _git(repository, "cat-file", "--batch", input_bytes=request).stdout
    offset = 0
    blobs: dict[str, bytes] = {}
    for expected_oid in unique_oids:
        line_end = response.find(b"\n", offset)
        if line_end < 0:
            raise ManifestError("truncated git cat-file --batch header")
        header = response[offset:line_end]
        offset = line_end + 1
        parts = header.split(b" ")
        if len(parts) != 3:
            raise ManifestError("invalid git cat-file --batch header")
        actual_oid = _decode_git_text(
            parts[0], label="git cat-file object id"
        )
        object_type = _decode_git_text(
            parts[1], label="git cat-file object type"
        )
        if _CANONICAL_DECIMAL_BYTES.fullmatch(parts[2]) is None:
            raise ManifestError(
                "git cat-file byte count is not a canonical decimal"
            )
        byte_count = int(parts[2])
        if actual_oid != expected_oid or object_type != "blob" or byte_count < 0:
            raise ManifestError(
                f"unexpected git cat-file object metadata for {expected_oid}"
            )
        end = offset + byte_count
        if end >= len(response) or response[end : end + 1] != b"\n":
            raise ManifestError(f"truncated git cat-file blob for {expected_oid}")
        blobs[expected_oid] = response[offset:end]
        offset = end + 1
    if offset != len(response):
        raise ManifestError("unexpected trailing git cat-file --batch output")
    return blobs


def committed_snapshot(
    repository: Path,
    revision: str,
    *,
    manifest_kind: ManifestKind,
) -> CommittedSnapshot:
    """Read selected files only from an exact committed Git tree."""

    if manifest_kind not in _MANIFEST_KINDS:
        raise ManifestError(f"unsupported manifest_kind: {manifest_kind!r}")
    full_revision = resolve_full_revision(repository, revision)
    object_format = _git_object_format(repository)
    selected = _select_entries(_tree_entries(repository, full_revision), manifest_kind)
    _validate_selected_entries(selected, manifest_kind)
    blobs = _cat_file_batch(repository, [row.object_oid for row in selected])
    files: list[CommittedFile] = []
    for row in selected:
        raw_bytes = blobs[row.object_oid]
        if len(raw_bytes) != row.byte_count:
            raise ManifestError(f"ls-tree and cat-file size mismatch: {row.path}")
        files.append(
            CommittedFile(
                path=row.path,
                git_mode=row.git_mode,
                git_type=row.git_type,
                blob_oid=row.object_oid,
                raw_bytes=raw_bytes,
            )
        )
    return CommittedSnapshot(
        revision=full_revision,
        git_object_format=object_format,
        manifest_kind=manifest_kind,
        files=tuple(files),
    )


def _is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    result = _git(
        repository,
        "merge-base",
        "--is-ancestor",
        ancestor,
        descendant,
        allowed_returncodes=frozenset({0, 1}),
    )
    return result.returncode == 0


def selected_paths_unchanged(
    repository: Path,
    ancestor_revision: str,
    descendant_revision: str,
    *,
    manifest_kind: ManifestKind,
) -> bool:
    """Return whether selected committed blobs are identical across ancestry."""

    ancestor = resolve_full_revision(repository, ancestor_revision)
    descendant = resolve_full_revision(repository, descendant_revision)
    if not _is_ancestor(repository, ancestor, descendant):
        return False
    before = committed_snapshot(
        repository, ancestor, manifest_kind=manifest_kind
    )
    after = committed_snapshot(
        repository, descendant, manifest_kind=manifest_kind
    )
    before_rows = tuple(
        (row.path, row.git_mode, row.git_type, row.blob_oid, row.raw_bytes)
        for row in before.files
    )
    after_rows = tuple(
        (row.path, row.git_mode, row.git_type, row.blob_oid, row.raw_bytes)
        for row in after.files
    )
    return before_rows == after_rows


def _enforce_revision_policy(
    repository: Path,
    revision: str,
    manifest_kind: ManifestKind,
    *,
    expected_revision: str | None,
    require_current_unchanged: bool,
) -> None:
    if expected_revision is not None and require_current_unchanged:
        raise ManifestError(
            "expected_revision and require_current_unchanged are mutually exclusive"
        )
    if expected_revision is not None:
        expected = resolve_full_revision(repository, expected_revision)
        if revision != expected:
            raise ManifestError("manifest revision does not match expected_revision")
        return
    if not require_current_unchanged:
        raise ManifestError(
            "a revision policy is required: expected_revision or "
            "require_current_unchanged"
        )
    head = _head_revision(repository)
    if not _is_ancestor(repository, revision, head):
        raise ManifestError("current manifest revision must be an ancestor of HEAD")
    if not selected_paths_unchanged(
        repository, revision, head, manifest_kind=manifest_kind
    ):
        raise ManifestError(
            "selected committed paths changed between manifest revision and HEAD"
        )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ManifestError(f"value is not canonical JSON: {exc}") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _manifest_from_snapshot(snapshot: CommittedSnapshot) -> dict[str, Any]:
    files = [
        {
            "path": row.path,
            "git_mode": row.git_mode,
            "byte_count": row.byte_count,
            "blob_oid": row.blob_oid,
            "sha256": row.sha256,
        }
        for row in snapshot.files
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": snapshot.manifest_kind,
        "revision": snapshot.revision,
        "git_object_format": snapshot.git_object_format,
        "selection": _selection(snapshot.manifest_kind),
        "files": files,
        "counts": {
            "files": len(files),
            "bytes": sum(cast(int, row["byte_count"]) for row in files),
        },
        "files_sha256": _sha256(files),
    }
    manifest["manifest_sha256"] = _sha256(manifest)
    return manifest


def _build_manifest_and_snapshot(
    repository: Path,
    revision: str,
    *,
    manifest_kind: ManifestKind,
    expected_revision: str | None,
    require_current_unchanged: bool,
) -> tuple[dict[str, Any], CommittedSnapshot]:
    snapshot = committed_snapshot(
        repository, revision, manifest_kind=manifest_kind
    )
    _enforce_revision_policy(
        repository,
        snapshot.revision,
        manifest_kind,
        expected_revision=expected_revision,
        require_current_unchanged=require_current_unchanged,
    )
    return _manifest_from_snapshot(snapshot), snapshot


def build_manifest(
    repository: Path,
    revision: str,
    *,
    manifest_kind: ManifestKind,
    expected_revision: str | None = None,
    require_current_unchanged: bool = False,
) -> dict[str, Any]:
    """Build one canonical manifest from its committed revision tree."""

    manifest, _snapshot = _build_manifest_and_snapshot(
        repository,
        revision,
        manifest_kind=manifest_kind,
        expected_revision=expected_revision,
        require_current_unchanged=require_current_unchanged,
    )
    return manifest


def _require_exact_keys(
    value: object, expected: frozenset[str], *, label: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ManifestError(f"{label} must be a JSON object")
    typed = value
    if set(typed) != expected:
        raise ManifestError(
            f"{label} keys mismatch: missing={sorted(expected - set(typed))}, "
            f"unknown={sorted(set(typed) - expected)}"
        )
    return typed


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ManifestError(f"{label} must be a lowercase sha256 hex digest")
    return value


def _validate_manifest_structure(
    repository: Path, manifest: object
) -> tuple[dict[str, Any], ManifestKind]:
    row = _require_exact_keys(manifest, _TOP_LEVEL_KEYS, label="manifest")
    if type(row["schema_version"]) is not int or row["schema_version"] != 1:
        raise ManifestError("schema_version must be integer 1")
    kind_value = row["manifest_kind"]
    if not isinstance(kind_value, str) or kind_value not in _MANIFEST_KINDS:
        raise ManifestError("manifest_kind is unsupported")
    manifest_kind: ManifestKind = kind_value
    revision = row["revision"]
    if not isinstance(revision, str) or _FULL_SHA1.fullmatch(revision) is None:
        raise ManifestError("manifest revision must be a lowercase full 40 SHA")
    object_format = row["git_object_format"]
    if object_format != _git_object_format(repository):
        raise ManifestError("git_object_format does not match the repository")
    if row["selection"] != _selection(manifest_kind):
        raise ManifestError("selection object does not match manifest_kind")

    files_value = row["files"]
    if type(files_value) is not list:
        raise ManifestError("files must be a JSON array")
    files: list[dict[str, Any]] = []
    for index, item in enumerate(files_value):
        file_row = _require_exact_keys(
            item, _FILE_KEYS, label=f"files[{index}]"
        )
        path = file_row["path"]
        if not isinstance(path, str) or not path:
            raise ManifestError(f"files[{index}].path must be non-empty")
        if file_row["git_mode"] != _expected_mode(path, manifest_kind):
            raise ManifestError(f"files[{index}].git_mode is invalid")
        byte_count = file_row["byte_count"]
        if type(byte_count) is not int or byte_count < 0:
            raise ManifestError(f"files[{index}].byte_count is invalid")
        blob_oid = file_row["blob_oid"]
        if not isinstance(blob_oid, str) or _FULL_SHA1.fullmatch(blob_oid) is None:
            raise ManifestError(f"files[{index}].blob_oid is invalid")
        _require_sha256(file_row["sha256"], label=f"files[{index}].sha256")
        files.append(file_row)
    paths = [str(item["path"]) for item in files]
    if paths != sorted(paths):
        raise ManifestError("file rows must be sorted by path")
    if len(paths) != len(set(paths)):
        raise ManifestError("file rows must have unique paths")

    counts = _require_exact_keys(row["counts"], _COUNT_KEYS, label="counts")
    if any(type(counts[key]) is not int or counts[key] < 0 for key in _COUNT_KEYS):
        raise ManifestError("manifest counts must be non-negative integers")
    if counts != {
        "files": len(files),
        "bytes": sum(cast(int, item["byte_count"]) for item in files),
    }:
        raise ManifestError("manifest counts do not match file rows")
    files_sha256 = _require_sha256(row["files_sha256"], label="files_sha256")
    if files_sha256 != _sha256(files):
        raise ManifestError("files_sha256 does not match file rows")
    manifest_sha256 = _require_sha256(
        row["manifest_sha256"], label="manifest_sha256"
    )
    unsigned = {key: value for key, value in row.items() if key != "manifest_sha256"}
    if manifest_sha256 != _sha256(unsigned):
        raise ManifestError("manifest_sha256 does not match manifest content")
    return row, manifest_kind


def verify_manifest(
    repository: Path,
    manifest: object,
    *,
    expected_kind: object,
    expected_revision: str | None = None,
    require_current_unchanged: bool = False,
) -> CommittedSnapshot:
    """Verify strict schema, hashes, policy, and an independent tree rebuild."""

    row, manifest_kind = _validate_manifest_structure(repository, manifest)
    if (
        not isinstance(expected_kind, str)
        or expected_kind not in _MANIFEST_KINDS
        or manifest_kind != expected_kind
    ):
        raise ManifestError("manifest_kind does not match expected_kind")
    expected, snapshot = _build_manifest_and_snapshot(
        repository,
        str(row["revision"]),
        manifest_kind=manifest_kind,
        expected_revision=expected_revision,
        require_current_unchanged=require_current_unchanged,
    )
    if row != expected:
        raise ManifestError(
            "manifest does not exactly match the independently rebuilt revision tree"
        )
    return snapshot


__all__ = [
    "ANALYZER_PATHS",
    "ANALYZER_MANIFEST_KIND",
    "FROZEN_SOURCE_REVISION",
    "SOURCE_MANIFEST_KIND",
    "CommittedFile",
    "CommittedSnapshot",
    "ManifestError",
    "ManifestKind",
    "build_manifest",
    "committed_snapshot",
    "resolve_full_revision",
    "selected_paths_unchanged",
    "verify_manifest",
]
