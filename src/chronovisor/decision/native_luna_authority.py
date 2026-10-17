"""Native Luna authority for the search-label review lane.

This module is deliberately a narrow adapter.  The search prompt and its
production schema remain owned by ``search_eval``; this adapter only executes
that exact request through the installed native Codex CLI and emits a
content-free receipt that the normal authority and CAS paths can verify.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
import uuid
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.decision.decision_policy import resolve_decision_policy
from chronovisor.decision.decision_router import decision_request_fingerprint_sha256
from chronovisor.decision.decision_schema_manifest import (
    production_schema_manifest,
    schema_sha256,
)
from chronovisor.decision.local_structured import validate_json

NATIVE_LUNA_AUTHORITY_ENV = "CHRONOVISOR_SEARCH_LABEL_AUTHORITY"
NATIVE_LUNA_AUTHORITY_VALUE = "native_luna"
NATIVE_LUNA_SOURCE = "native_luna_runtime"
NATIVE_LUNA_AUTHORITY_KIND = "native_luna_v1"
NATIVE_LUNA_MODEL = "gpt-5.6-luna"
NATIVE_LUNA_PROVIDER = "openai"
NATIVE_LUNA_ROUTE = "native/codex_exec"
NATIVE_LUNA_ROLE = "search_label.authority"
NATIVE_LUNA_PROTOCOL = "codex-native"
NATIVE_LUNA_REVISION_PREFIX = "native-luna-search-authority-v1-"
NATIVE_LUNA_ROLE_CONFIG_NAME = "luna_worker"
NATIVE_LUNA_AGENT_TYPE = "codex_exec"
NATIVE_LUNA_LANE = "search_label"
LOCAL_AUTHORITY_VALUE = "local"
SEARCH_LABEL_REVIEW_SECTION = "search_label_review"
NATIVE_LUNA_GLOBAL_LIMIT = 10
NATIVE_LUNA_RECEIPT_SCHEMA = "chronovisor.search-label-native-luna-receipt.v1"
NATIVE_LUNA_ROLLOUT_SCHEMA = "chronovisor.search-label-native-luna-rollout.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^native-search-[0-9a-f]{32}$")
_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "run_id",
        "status",
        "exit_code",
        "model",
        "provider",
        "route",
        "role",
        "agent_type",
        "revision",
        "request_sha256",
        "input_sha256",
        "prompt_sha256",
        "schema_sha256",
        "output_sha256",
        "role_config_sha256",
        "codex_binary_sha256",
        "codex_version_sha256",
        "session_model",
        "session_provider",
        "session_reasoning_effort",
        "session_id",
        "session_provenance_sha256",
        "rollout_sha256",
        "receipt_sha256",
    }
)


class NativeLunaIdentityError(RuntimeError):
    """The installed native Luna route cannot be proven."""


class NativeLunaCapacityUnavailable(RuntimeError):
    """The native Luna capacity reservation is busy or exhausted."""


def _runtime_root(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    configured = os.environ.get("CHRONOVISOR_ROOT", "").strip()
    return Path(configured).expanduser() if configured else CHRONOVISOR_ROOT


def _parse_authority_value(value: object, *, source: str) -> str:
    if not isinstance(value, str):
        raise NativeLunaIdentityError(
            f"native_luna_{source}_authority_invalid"
        )
    normalized = value.strip().casefold()
    if normalized not in {LOCAL_AUTHORITY_VALUE, NATIVE_LUNA_AUTHORITY_VALUE}:
        raise NativeLunaIdentityError(
            f"native_luna_{source}_authority_unsupported"
        )
    return normalized


def _search_label_review_config(root: Path | None = None) -> Mapping[str, Any]:
    path = _runtime_root(root) / "config.toml"
    try:
        snapshot = path.read_bytes()
        data = tomllib.loads(snapshot.decode("utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise NativeLunaIdentityError(
            "native_luna_config_unavailable"
        ) from exc
    section = data.get(SEARCH_LABEL_REVIEW_SECTION)
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise NativeLunaIdentityError("native_luna_config_section_invalid")
    return section


def _configured_search_label_authority(root: Path | None = None) -> str:
    section = _search_label_review_config(root)
    if "authority" not in section:
        return LOCAL_AUTHORITY_VALUE
    return _parse_authority_value(section["authority"], source="config")


def native_luna_authority_for_lane(
    lane: str,
    *,
    root: Path | None = None,
) -> str:
    """Resolve one search-label authority consistently for every process.

    The environment remains an explicit test/canary override.  Production
    processes without that override all read the same root-level config.  An
    unreadable or unsupported configured value raises instead of silently
    selecting the local route.
    """

    if lane != NATIVE_LUNA_LANE:
        return LOCAL_AUTHORITY_VALUE
    override = os.environ.get(NATIVE_LUNA_AUTHORITY_ENV)
    if override is not None:
        return _parse_authority_value(override, source="environment")
    return _configured_search_label_authority(root)


def native_luna_enabled_for_lane(
    lane: str,
    *,
    root: Path | None = None,
) -> bool:
    return (
        native_luna_authority_for_lane(lane, root=root)
        == NATIVE_LUNA_AUTHORITY_VALUE
    )


def _codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    default = Path.home() / ".config" / "codex"
    return default if default.exists() else Path.home() / ".codex"


def _role_path() -> Path:
    return _codex_home() / "agents" / "luna_worker.toml"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _read_role(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        role = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise NativeLunaIdentityError("native_luna_role_unavailable") from exc
    if (
        role.get("name") != NATIVE_LUNA_ROLE_CONFIG_NAME
        or role.get("model") != NATIVE_LUNA_MODEL
        or role.get("model_reasoning_effort") != "max"
        or role.get("model_provider", "openai") != "openai"
    ):
        raise NativeLunaIdentityError("native_luna_role_identity_mismatch")
    return role, _sha256_bytes(raw)


def _resolve_codex() -> Path:
    section = _search_label_review_config()
    override = os.environ.get("CHRONOVISOR_CODEX_EXECUTABLE", "").strip()
    try:
        if "codex_executable" in section:
            configured = section["codex_executable"]
            if not isinstance(configured, str) or not configured.strip():
                raise NativeLunaIdentityError("native_luna_config_executable_invalid")
            path = Path(configured.strip()).expanduser()
            if not path.is_absolute():
                raise NativeLunaIdentityError("native_luna_config_executable_invalid")
            if override and Path(override).expanduser().resolve() != path.resolve():
                raise NativeLunaIdentityError("native_luna_executable_override_conflict")
        else:
            path = Path(override).expanduser() if override else Path(shutil.which("codex") or "")
        if not path or not path.is_file() or not os.access(path, os.X_OK):
            raise NativeLunaIdentityError("native_luna_codex_unavailable")
        return path.resolve()
    except NativeLunaIdentityError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise NativeLunaIdentityError("native_luna_codex_unavailable") from exc


def _codex_version(path: Path) -> str:
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NativeLunaIdentityError("native_luna_codex_version_unavailable") from exc
    if completed.returncode != 0:
        raise NativeLunaIdentityError("native_luna_codex_version_unavailable")
    value = (completed.stdout or completed.stderr or "").strip()
    if not value or len(value) > 256:
        raise NativeLunaIdentityError("native_luna_codex_version_unavailable")
    return value


def _runtime_identity() -> dict[str, Any]:
    """Return the current route identity without exposing credentials."""

    # Resolve before consulting the cache so a changed or invalid canonical
    # pin cannot leave a long-lived process using its previous PATH choice.
    return _cached_runtime_identity(_resolve_codex(), _role_path())


@lru_cache(maxsize=1)
def _cached_runtime_identity(codex: Path, role_path: Path) -> dict[str, Any]:
    _role, role_sha256 = _read_role(role_path)
    version = _codex_version(codex)
    return {
        "role": NATIVE_LUNA_ROLE,
        "provider": NATIVE_LUNA_PROVIDER,
        "model": NATIVE_LUNA_MODEL,
        "location": "native",
        "protocol": NATIVE_LUNA_PROTOCOL,
        "endpoint_sha256": _sha256_bytes(codex.read_bytes()),
        "revision": f"{NATIVE_LUNA_REVISION_PREFIX}{role_sha256[:16]}",
        "ollama": None,
        "_role_config_sha256": role_sha256,
        "_codex_version_sha256": _sha256_text(version),
        "_codex_path": str(codex),
    }


def native_luna_route_identity() -> dict[str, Any]:
    """Return only the stable public route fields used by authority seals."""

    identity = _runtime_identity()
    return {key: identity[key] for key in (
        "role",
        "provider",
        "model",
        "location",
        "protocol",
        "endpoint_sha256",
        "revision",
        "ollama",
    )}


def clear_native_luna_runtime_identity_cache() -> None:
    _cached_runtime_identity.cache_clear()


def native_luna_route_provenance_error(routes: object) -> str | None:
    if not isinstance(routes, list) or len(routes) != 1:
        return "native Luna authority route is invalid"
    route = routes[0]
    if not isinstance(route, Mapping) or set(route) != {
        "role",
        "provider",
        "model",
        "location",
        "protocol",
        "endpoint_sha256",
        "revision",
        "ollama",
    }:
        return "native Luna authority route fields are invalid"
    if (
        route.get("role") != NATIVE_LUNA_ROLE
        or route.get("provider") != NATIVE_LUNA_PROVIDER
        or route.get("model") != NATIVE_LUNA_MODEL
        or route.get("location") != "native"
        or route.get("protocol") != NATIVE_LUNA_PROTOCOL
        or route.get("ollama") is not None
        or not isinstance(route.get("endpoint_sha256"), str)
        or _SHA256_RE.fullmatch(route["endpoint_sha256"]) is None
        or not isinstance(route.get("revision"), str)
        or not route["revision"].startswith(NATIVE_LUNA_REVISION_PREFIX)
    ):
        return "native Luna authority route identity is invalid"
    return None


class NativeLunaCapacityLease:
    """A shared, process-lifetime reservation for native Luna work."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        owner: str = "authority",
        slots: int = 1,
    ) -> None:
        if owner not in {"authority", "supervisor"} or slots < 1:
            raise ValueError("invalid native Luna capacity lease")
        self.root = _runtime_root(root)
        self.owner = owner
        self.slots = slots
        self._lock: Any | None = None
        self._tokens: list[Path] = []
        self._token_handles: list[Any] = []

    def _live_slots(self, lease_root: Path, owner: str) -> int:
        count = 0
        for candidate in lease_root.glob("slot-*.json"):
            handle: Any | None = None
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if data.get("owner") != owner:
                    continue
                handle = candidate.open("r+")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                count += 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            else:
                # A token left by a dead owner has no lock. Keep the cleanup
                # inside the registry critical section so a new reservation
                # cannot race a stale-file removal.
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                candidate.unlink(missing_ok=True)
            finally:
                if handle is not None:
                    handle.close()
        return count

    def _supervisor_lock_busy(self, lock_path: Path) -> bool:
        if not lock_path.exists():
            return False
        try:
            with lock_path.open("a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            return True
        except OSError as exc:
            raise NativeLunaCapacityUnavailable(
                "native_luna_supervisor_status_unavailable"
            ) from exc
        return False

    def _supervisor_status_workers(self) -> int:
        status_path = self.root / "runtime" / "recall-luna-bulk" / "supervisor-status.json"
        supervisor_lock = status_path.parent / "supervisor.lock"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            if not self._supervisor_lock_busy(supervisor_lock):
                return 0
            raise NativeLunaCapacityUnavailable(
                "native_luna_supervisor_status_unavailable"
            ) from exc
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if self._supervisor_lock_busy(supervisor_lock):
                raise NativeLunaCapacityUnavailable(
                    "native_luna_supervisor_status_unavailable"
                ) from exc
            return 0
        if not isinstance(status, Mapping):
            if self._supervisor_lock_busy(supervisor_lock):
                raise NativeLunaCapacityUnavailable("native_luna_supervisor_status_unavailable")
            return 0
        if status.get("pid") != os.getpid() and not self._supervisor_lock_busy(supervisor_lock):
            return 0
        try:
            return max(0, int(status.get("active_workers") or 0))
        except (AttributeError, ValueError, TypeError) as exc:
            raise NativeLunaCapacityUnavailable(
                "native_luna_supervisor_status_unavailable"
            ) from exc

    def __enter__(self) -> NativeLunaCapacityLease:
        lease_root = self.root / "runtime" / "native-luna-authority"
        lease_root.mkdir(parents=True, exist_ok=True)
        lock_path = lease_root / "capacity.lock"
        self._lock = lock_path.open("a+")
        try:
            deadline = time.monotonic() + 1.0
            while True:
                try:
                    fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise NativeLunaCapacityUnavailable(
                            "native_luna_capacity_busy"
                        ) from exc
                    time.sleep(0.01)
            supervisor_tokens = self._live_slots(lease_root, "supervisor")
            authority_tokens = self._live_slots(lease_root, "authority")
            supervisor_active = max(
                self._supervisor_status_workers(), supervisor_tokens
            )
            occupied = supervisor_active + authority_tokens
            if self.owner == "supervisor":
                occupied = max(supervisor_active, supervisor_tokens) + authority_tokens
            if occupied + self.slots > NATIVE_LUNA_GLOBAL_LIMIT:
                raise NativeLunaCapacityUnavailable("native_luna_capacity_exhausted")
            for _ in range(self.slots):
                token = lease_root / f"slot-{self.owner}-{os.getpid()}-{uuid.uuid4().hex}.json"
                fd = os.open(token, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                handle = os.fdopen(fd, "w+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    json.dump({"pid": os.getpid(), "owner": self.owner}, handle, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                except BaseException:
                    handle.close()
                    token.unlink(missing_ok=True)
                    raise
                self._token_handles.append(handle)
                self._tokens.append(token)
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_: object) -> None:
        for handle in self._token_handles:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        self._token_handles.clear()
        for token in self._tokens:
            token.unlink(missing_ok=True)
        self._tokens.clear()
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    @property
    def pass_fds(self) -> tuple[int, ...]:
        """Return lifetime slot locks for an inherited native child."""

        return tuple(handle.fileno() for handle in self._token_handles)


def native_luna_capacity_lease(
    root: Path | None = None,
    *,
    owner: str = "authority",
    slots: int = 1,
) -> NativeLunaCapacityLease:
    """Return a shared lease for one authority call or supervisor shard(s)."""

    return NativeLunaCapacityLease(root, owner=owner, slots=slots)


def _failure(
    summary: str, failure_class: str, *, execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "decision": "needs_retry",
        "confidence": 0.0,
        "expected_pages": [],
        "negative_pages": [],
        "stale_pages": [],
        "summary": summary,
        "notes": None,
        "reviewer": "native_luna",
        "frontier_failure": {
            "failure_class": failure_class,
            "rescue_status": "frontier_retry",
            "summary": summary,
            "human_required": False,
            "notify_user": False,
            **({"execution": {
                key: execution.get(key) for key in (
                    "run_id", "returncode", "error", "rollout_sha256",
                    "execution_sha256", "elapsed_seconds", "timeout_seconds",
                )
            }} if execution is not None else {}),
        },
    }


def _extract_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _normal_result(parsed: Mapping[str, Any]) -> dict[str, Any]:
    decision = parsed.get("decision")
    summary = parsed.get("summary")
    return {
        "decision": decision,
        "confidence": float(parsed.get("confidence")),
        "expected_pages": list(parsed.get("expected_pages") or []),
        "negative_pages": list(parsed.get("negative_pages") or []),
        "stale_pages": list(parsed.get("stale_pages") or []),
        "summary": summary if isinstance(summary, str) and summary.strip() else decision,
        "notes": parsed.get("notes") if isinstance(parsed.get("notes"), str) else None,
        "reviewer": "native_luna",
    }


def _receipt_without_digest(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_sha256"}


def _review_without_receipt(review: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in review.items()
        if key not in {"native_luna_receipt", "native_luna_receipts"}
    }


def _rollout_relative_path_is_safe(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return False
    return value in {"stdout.jsonl", "stderr.log", "output.json"} or (
        value.startswith("sessions/") and value.endswith(".jsonl")
    )


def _validate_rollout_evidence(receipt: Mapping[str, Any]) -> str | None:
    run_id = receipt.get("run_id")
    expected_digest = receipt.get("rollout_sha256")
    if (
        not isinstance(run_id, str)
        or not _RUN_ID_RE.fullmatch(run_id)
        or not isinstance(expected_digest, str)
        or _SHA256_RE.fullmatch(expected_digest) is None
    ):
        return "native Luna rollout identity is invalid"
    directory = _runtime_root() / "runtime" / "native-luna-authority" / "runs" / run_id
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        return "native Luna rollout manifest is not persisted"
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema",
        "run_id",
        "session",
        "session_file",
        "files",
    }:
        return "native Luna rollout manifest is invalid"
    if manifest.get("schema") != NATIVE_LUNA_ROLLOUT_SCHEMA or manifest.get("run_id") != run_id:
        return "native Luna rollout manifest identity is invalid"
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not {"stdout.jsonl", "stderr.log", "output.json"}.issubset(files):
        return "native Luna rollout files are invalid"
    actual_files: dict[str, dict[str, Any]] = {}
    for relative, metadata in files.items():
        if not _rollout_relative_path_is_safe(relative) or not isinstance(metadata, Mapping):
            return "native Luna rollout files are invalid"
        if set(metadata) != {"bytes", "sha256"}:
            return "native Luna rollout file metadata is invalid"
        size = metadata.get("bytes")
        digest = metadata.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            return "native Luna rollout file metadata is invalid"
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            return "native Luna rollout file metadata is invalid"
        try:
            payload = (directory / relative).read_bytes()
        except (OSError, UnicodeError):
            return "native Luna rollout file is missing"
        observed = {"bytes": len(payload), "sha256": _sha256_bytes(payload)}
        if observed != dict(metadata):
            return "native Luna rollout file changed"
        actual_files[relative] = observed
    session = manifest.get("session")
    session_file = manifest.get("session_file")
    if not isinstance(session, Mapping) or set(session) != {
        "id",
        "model",
        "provider",
        "reasoning_effort",
    }:
        return "native Luna rollout session is invalid"
    if not isinstance(session_file, str) or session_file not in actual_files:
        return "native Luna rollout session file is invalid"
    session_id = session.get("id")
    if not isinstance(session_id, str) or not session_id:
        return "native Luna rollout session is invalid"
    if dict(session) != {
        "id": receipt.get("session_id"),
        "model": receipt.get("session_model"),
        "provider": receipt.get("session_provider"),
        "reasoning_effort": receipt.get("session_reasoning_effort"),
    }:
        return "native Luna rollout session does not match receipt"
    evidence = _isolated_session_evidence(directory, session_id)
    if evidence is None:
        return "native Luna rollout session is missing or changed"
    observed_id, observed_session, observed_path = evidence
    if (
        observed_id != session_id
        or observed_path.relative_to(directory).as_posix() != session_file
        or observed_session != {
            "model": session.get("model"),
            "provider": session.get("provider"),
            "reasoning_effort": session.get("reasoning_effort"),
        }
    ):
        return "native Luna rollout session provenance changed"
    canonical = {
        "schema": NATIVE_LUNA_ROLLOUT_SCHEMA,
        "run_id": run_id,
        "session": {"id": observed_id, **observed_session},
        "session_file": session_file,
        "files": actual_files,
    }
    if dict(manifest) != canonical:
        return "native Luna rollout manifest changed"
    if canonical_json_sha256_strict(canonical) != expected_digest:
        return "native Luna rollout digest changed"
    return None


def _write_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    destination = root / "runtime" / "native-luna-authority" / "receipts"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{receipt['run_id']}.json"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=destination)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(receipt), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _prepare_rollout_directory(root: Path, run_id: str) -> Path:
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("invalid native Luna rollout id")
    directory = root / "runtime" / "native-luna-authority" / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    return directory


def _write_private_rollout_file(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return _sha256_bytes(payload)


def _copy_session_rollout(
    codex_home: Path | None,
    rollout_directory: Path,
) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    if codex_home is None:
        return files
    sessions = codex_home / "sessions"
    if not sessions.is_dir() or sessions.is_symlink():
        return files
    for current, directories, names in os.walk(sessions, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if not (current_path / name).is_symlink()
        ]
        for name in sorted(names):
            source = current_path / name
            if source.suffix != ".jsonl" or source.is_symlink() or not source.is_file():
                continue
            relative = source.relative_to(codex_home).as_posix()
            payload = source.read_bytes()
            digest = _write_private_rollout_file(
                rollout_directory / relative,
                payload,
            )
            files[relative] = {"bytes": len(payload), "sha256": digest}
    return dict(sorted(files.items()))


def _persist_rollout_evidence(
    root: Path,
    run_id: str,
    *,
    stdout: str,
    stderr: str,
    output: str,
    codex_home: Path | None,
    session_evidence: tuple[str, dict[str, str], Path] | None,
) -> str:
    directory = _prepare_rollout_directory(root, run_id)
    files = {
        "stdout.jsonl": {
            "bytes": len(stdout.encode("utf-8")),
            "sha256": _write_private_rollout_file(
                directory / "stdout.jsonl", stdout.encode("utf-8")
            ),
        },
        "stderr.log": {
            "bytes": len(stderr.encode("utf-8")),
            "sha256": _write_private_rollout_file(
                directory / "stderr.log", stderr.encode("utf-8")
            ),
        },
        "output.json": {
            "bytes": len(output.encode("utf-8")),
            "sha256": _write_private_rollout_file(
                directory / "output.json", output.encode("utf-8")
            ),
        },
    }
    files.update(_copy_session_rollout(codex_home, directory))
    session: dict[str, str] | None = None
    session_file: str | None = None
    if session_evidence is not None and codex_home is not None:
        session_id, metadata, source = session_evidence
        try:
            session_file = source.relative_to(codex_home).as_posix()
        except ValueError as exc:
            raise OSError("native Luna session path escaped isolated home") from exc
        if session_file not in files:
            raise OSError("native Luna session transcript was not preserved")
        session = {"id": session_id, **metadata}
    manifest = {
        "schema": NATIVE_LUNA_ROLLOUT_SCHEMA,
        "run_id": run_id,
        "session": session,
        "session_file": session_file,
        "files": files,
    }
    digest = canonical_json_sha256_strict(manifest)
    _write_private_rollout_file(
        directory / "manifest.json",
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )
    return digest


def _build_command(
    codex: Path,
    *,
    repo_root: Path,
    schema_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        str(codex),
        "exec",
        "--ignore-user-config",
        "-m",
        NATIVE_LUNA_MODEL,
        "-c",
        'model_provider="openai"',
        "-c",
        'approval_policy="never"',
        "-c",
        'model_reasoning_effort="max"',
        "--ignore-rules",
        "--disable",
        "hooks",
        "--sandbox",
        "read-only",
        "-C",
        str(repo_root),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "--skip-git-repo-check",
        "--json",
        "-",
    ]


def _session_provenance(output: str) -> dict[str, str] | None:
    """Extract route metadata from authenticated Codex session envelopes.

    Response content and tool payloads are deliberately ignored: a model can
    print any string it wants, so those fields cannot prove which runtime was
    used.  The CLI's ``session_meta`` and ``turn_context`` envelopes are the
    only accepted source for this binding.
    """

    providers: set[str] = set()
    models: set[str] = set()
    efforts: set[str] = set()
    for line in (output or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        event_type = event.get("type")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if event_type == "session_meta":
            provider = payload.get("model_provider")
            if isinstance(provider, str) and provider:
                providers.add(provider)
            continue
        if event_type != "turn_context":
            continue
        model = payload.get("model")
        effort = payload.get("effort")
        collaboration = payload.get("collaboration_mode")
        settings = collaboration.get("settings") if isinstance(collaboration, Mapping) else None
        settings_model = settings.get("model") if isinstance(settings, Mapping) else None
        settings_effort = settings.get("reasoning_effort") if isinstance(settings, Mapping) else None
        if isinstance(model, str) and model:
            models.add(model)
        if isinstance(settings_model, str) and settings_model:
            models.add(settings_model)
        if isinstance(effort, str) and effort:
            efforts.add(effort)
        if isinstance(settings_effort, str) and settings_effort:
            efforts.add(settings_effort)
    if (
        len(providers) != 1
        or len(models) != 1
        or len(efforts) != 1
        or next(iter(providers)) != NATIVE_LUNA_PROVIDER
        or next(iter(models)) != NATIVE_LUNA_MODEL
        or next(iter(efforts)) != "max"
    ):
        return None
    return {
        "model": next(iter(models)),
        "provider": next(iter(providers)),
        "reasoning_effort": next(iter(efforts)),
    }


def _thread_started_id(output: str) -> str | None:
    thread_ids: set[str] = set()
    for line in (output or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping) or event.get("type") != "thread.started":
            continue
        thread_id = event.get("thread_id")
        if not isinstance(thread_id, str):
            payload = event.get("payload")
            thread_id = payload.get("thread_id") if isinstance(payload, Mapping) else None
        if isinstance(thread_id, str) and thread_id:
            thread_ids.add(thread_id)
    return next(iter(thread_ids)) if len(thread_ids) == 1 else None


def _isolated_session_evidence(
    codex_home: Path,
    thread_id: str,
) -> tuple[str, dict[str, str], Path] | None:
    matches: list[tuple[str, dict[str, str], Path]] = []
    for path in codex_home.glob("sessions/**/*.jsonl"):
        try:
            transcript = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        session_ids: set[str] = set()
        for line in transcript.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping) or event.get("type") != "session_meta":
                continue
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                continue
            session_id = payload.get("id") or payload.get("session_id")
            if isinstance(session_id, str) and session_id:
                session_ids.add(session_id)
        if session_ids != {thread_id}:
            continue
        provenance = _session_provenance(transcript)
        if provenance is not None:
            matches.append((thread_id, provenance, path))
    if len(matches) != 1:
        return None
    return matches[0]


def _isolated_session_metadata(
    codex_home: Path,
    thread_id: str,
) -> tuple[str, dict[str, str]] | None:
    evidence = _isolated_session_evidence(codex_home, thread_id)
    return evidence[:2] if evidence is not None else None


def _text_output(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


def _execute_native_luna_process(
    *,
    codex: Path,
    prompt: str,
    repo_root: Path,
    schema: Mapping[str, Any],
    timeout: int,
    root: Path,
    run_id: str,
    capacity: NativeLunaCapacityLease,
) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {
        "run_id": run_id,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "output": "",
        "session_evidence": None,
        "error": None,
        "rollout_sha256": None,
        "timeout_seconds": timeout,
    }
    persisted = False

    def persist() -> None:
        nonlocal persisted
        if persisted:
            return
        record["rollout_sha256"] = _persist_rollout_evidence(
            root,
            run_id,
            stdout=record["stdout"],
            stderr=record["stderr"],
            output=record["output"],
            codex_home=record.get("codex_home"),
            session_evidence=record["session_evidence"],
        )
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        # Diagnostic process evidence is separate from the immutable authority
        # manifest: a failed process must never mint a successful verdict.
        execution = {
            key: record.get(key) for key in (
                "run_id", "returncode", "error", "timeout_seconds",
                "elapsed_seconds", "rollout_sha256",
            )
        }
        record["execution_sha256"] = _write_private_rollout_file(
            _prepare_rollout_directory(root, run_id) / "execution.json",
            (json.dumps(execution, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        persisted = True

    try:
        from chronovisor.decision.frontier_review import _isolated_codex_environment

        with tempfile.TemporaryDirectory(prefix="chronovisor-native-luna-") as td:
            directory = Path(td)
            schema_path = directory / "search-label.schema.json"
            output_path = directory / "search-label.output.json"
            schema_path.write_text(
                json.dumps(dict(schema), ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with _isolated_codex_environment() as env:
                record["codex_home"] = Path(env["CODEX_HOME"])
                try:
                    completed = subprocess.run(
                        _build_command(
                            codex,
                            repo_root=repo_root,
                            schema_path=schema_path,
                            output_path=output_path,
                        ),
                        input=prompt,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=timeout,
                        cwd=str(repo_root),
                        env=env,
                        pass_fds=capacity.pass_fds,
                    )
                    record["returncode"] = completed.returncode
                    record["stdout"] = _text_output(completed.stdout)
                    record["stderr"] = _text_output(completed.stderr)
                except subprocess.TimeoutExpired as exc:
                    record["error"] = "timeout"
                    record["stdout"] = _text_output(exc.stdout)
                    record["stderr"] = _text_output(exc.stderr)
                except (OSError, subprocess.SubprocessError) as exc:
                    record["error"] = exc.__class__.__name__
                finally:
                    try:
                        record["output"] = output_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except OSError:
                        record["output"] = record["stdout"]
                    thread_id = _thread_started_id(record["stdout"])
                    if thread_id is not None:
                        record["session_evidence"] = _isolated_session_evidence(
                            record["codex_home"], thread_id
                        )
                    persist()
    finally:
        persist()
    return record


def run_native_luna_review(
    prompt: str,
    schema: Mapping[str, Any],
    *,
    repo_root: Path,
    lane: str = NATIVE_LUNA_LANE,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Execute one exact search-label request through native Luna."""

    if not native_luna_enabled_for_lane(lane):
        return _failure("native Luna authority is disabled", "native_luna_disabled")
    policy, mode, policy_error = resolve_decision_policy(lane)
    if policy is None or mode != "enabled" or policy_error is not None:
        return _failure("native Luna lane policy is unavailable", "native_luna_policy")
    expected = production_schema_manifest().get(str(policy.schema_name))
    if expected is None or schema_sha256(schema) != expected:
        return _failure("native Luna schema is not the production schema", "native_luna_schema")
    try:
        route = native_luna_route_identity()
        runtime_identity = _runtime_identity()
        codex = Path(str(runtime_identity["_codex_path"]))
        request_sha = decision_request_fingerprint_sha256(
            prompt=prompt, schema=schema, system=None, decision_lane=lane
        )
        timeout_seconds = max(
            1,
            int(timeout or os.environ.get("CHRONOVISOR_FRONTIER_TIMEOUT_SECONDS", "3600")),
        )
    except (NativeLunaIdentityError, TypeError, ValueError):
        return _failure("native Luna route identity is unavailable", "native_luna_identity")

    run_id = f"native-search-{uuid.uuid4().hex}"
    try:
        with native_luna_capacity_lease() as capacity:
            execution = _execute_native_luna_process(
                codex=codex,
                prompt=prompt,
                repo_root=repo_root,
                schema=schema,
                timeout=timeout_seconds,
                root=_runtime_root(),
                run_id=run_id,
                capacity=capacity,
            )
    except NativeLunaCapacityUnavailable:
        return _failure("native Luna capacity is unavailable", "native_luna_capacity")
    except (OSError, subprocess.SubprocessError, ValueError):
        return _failure("native Luna execution failed", "native_luna_execution")

    rollout_sha256 = execution.get("rollout_sha256")
    if not isinstance(rollout_sha256, str) or _SHA256_RE.fullmatch(rollout_sha256) is None:
        return _failure("native Luna rollout evidence could not be persisted", "native_luna_receipt", execution=execution)
    if execution.get("error") == "timeout":
        return _failure("native Luna execution timed out", "native_luna_timeout", execution=execution)
    if execution.get("error") is not None:
        return _failure("native Luna execution failed", "native_luna_execution", execution=execution)
    if execution.get("returncode") != 0:
        return _failure("native Luna returned a nonzero status", "native_luna_execution", execution=execution)
    session_evidence = execution.get("session_evidence")
    if session_evidence is None:
        return _failure(
            "native Luna session provenance is missing or mismatched",
            "native_luna_provenance",
            execution=execution,
        )
    session_id, session, _session_path = session_evidence
    output = execution["output"]
    parsed = _extract_json(output)
    if parsed is None or validate_json(parsed, dict(schema)):
        return _failure("native Luna output failed the production schema", "native_luna_schema", execution=execution)
    result = _normal_result(parsed)
    router_policy = {"source": NATIVE_LUNA_SOURCE, "error": None, "routes": [{key: value for key, value in route.items() if not key.startswith("_")}]}
    result["decision_policy"] = {
        "lane": lane,
        "kind": policy.kind,
        "schema_name": policy.schema_name,
        "mode": mode,
        "error": policy_error,
        "router_policy": router_policy,
    }
    receipt = {
        "schema": NATIVE_LUNA_RECEIPT_SCHEMA,
        "run_id": run_id,
        "status": "completed",
        "exit_code": 0,
        "model": NATIVE_LUNA_MODEL,
        "provider": NATIVE_LUNA_PROVIDER,
        "route": NATIVE_LUNA_ROUTE,
        "role": NATIVE_LUNA_ROLE,
        "agent_type": NATIVE_LUNA_AGENT_TYPE,
        "revision": route["revision"],
        "request_sha256": request_sha,
        "input_sha256": request_sha,
        "prompt_sha256": _sha256_text(prompt),
        "schema_sha256": schema_sha256(schema),
        "output_sha256": canonical_json_sha256_strict(result),
        "role_config_sha256": runtime_identity["_role_config_sha256"],
        "codex_binary_sha256": route["endpoint_sha256"],
        "codex_version_sha256": runtime_identity["_codex_version_sha256"],
        "session_model": session["model"],
        "session_provider": session["provider"],
        "session_reasoning_effort": session["reasoning_effort"],
        "session_id": session_id,
        "session_provenance_sha256": canonical_json_sha256_strict(
            {"session_id": session_id, **session}
        ),
        "rollout_sha256": rollout_sha256,
    }
    receipt["receipt_sha256"] = canonical_json_sha256_strict(receipt)
    try:
        _write_receipt(receipt, _runtime_root())
    except OSError:
        return _failure("native Luna receipt could not be persisted", "native_luna_receipt")
    result["native_luna_receipt"] = receipt
    return result


def validate_native_luna_request(
    receipt: Mapping[str, Any],
    prompt: str,
    schema: Mapping[str, Any],
    *,
    lane: str = NATIVE_LUNA_LANE,
) -> str | None:
    if receipt.get("request_sha256") != decision_request_fingerprint_sha256(
        prompt=prompt, schema=schema, system=None, decision_lane=lane
    ):
        return "native Luna request identity changed"
    if receipt.get("prompt_sha256") != _sha256_text(prompt):
        return "native Luna prompt identity changed"
    if receipt.get("schema_sha256") != schema_sha256(schema):
        return "native Luna schema identity changed"
    return None


def _validate_receipt(
    receipt: object,
    review: Mapping[str, Any],
    route: Mapping[str, Any],
) -> str | None:
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
        return "native Luna receipt is invalid"
    run_id = receipt.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        return "native Luna receipt identity is invalid"
    if (
        receipt.get("schema") != NATIVE_LUNA_RECEIPT_SCHEMA
        or receipt.get("status") != "completed"
        or receipt.get("exit_code") != 0
        or receipt.get("model") != NATIVE_LUNA_MODEL
        or receipt.get("provider") != NATIVE_LUNA_PROVIDER
        or receipt.get("route") != NATIVE_LUNA_ROUTE
        or receipt.get("role") != NATIVE_LUNA_ROLE
        or receipt.get("agent_type") != NATIVE_LUNA_AGENT_TYPE
        or receipt.get("revision") != route.get("revision")
        or receipt.get("role_config_sha256") != route.get("_role_config_sha256")
        or receipt.get("codex_binary_sha256") != route.get("endpoint_sha256")
        or receipt.get("codex_version_sha256") != route.get("_codex_version_sha256")
        or receipt.get("session_model") != NATIVE_LUNA_MODEL
        or receipt.get("session_provider") != NATIVE_LUNA_PROVIDER
        or receipt.get("session_reasoning_effort") != "max"
        or not isinstance(receipt.get("session_id"), str)
        or not receipt.get("session_id")
        or receipt.get("session_provenance_sha256")
        != canonical_json_sha256_strict(
            {
                "session_id": receipt.get("session_id"),
                "model": receipt.get("session_model"),
                "provider": receipt.get("session_provider"),
                "reasoning_effort": receipt.get("session_reasoning_effort"),
            }
        )
    ):
        return "native Luna receipt route identity is invalid"
    for field in (
        "request_sha256",
        "input_sha256",
        "prompt_sha256",
        "schema_sha256",
        "output_sha256",
        "role_config_sha256",
        "codex_binary_sha256",
        "codex_version_sha256",
        "receipt_sha256",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            return "native Luna receipt digest is invalid"
    if receipt.get("input_sha256") != receipt.get("request_sha256"):
        return "native Luna receipt input binding is invalid"
    if receipt.get("output_sha256") != canonical_json_sha256_strict(_review_without_receipt(review)):
        return "native Luna output binding is invalid"
    if receipt.get("receipt_sha256") != canonical_json_sha256_strict(_receipt_without_digest(receipt)):
        return "native Luna receipt digest is invalid"
    receipt_path = _runtime_root() / "runtime" / "native-luna-authority" / "receipts" / f"{run_id}.json"
    try:
        persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        return "native Luna receipt is not persisted"
    if not isinstance(persisted, Mapping) or dict(persisted) != dict(receipt):
        return "native Luna persisted receipt changed"
    return _validate_rollout_evidence(receipt)


def validate_native_luna_review(
    review: object,
    authority: Mapping[str, Any],
    *,
    lane: str,
) -> str | None:
    if lane != NATIVE_LUNA_LANE or authority.get("source") != NATIVE_LUNA_SOURCE:
        return "native Luna authority lane is invalid"
    routes = authority.get("router", {}).get("routes") if isinstance(authority.get("router"), Mapping) else None
    route_error = native_luna_route_provenance_error(routes)
    if route_error is not None:
        return route_error
    if not isinstance(review, Mapping):
        return "native Luna verdict is missing"
    policy = review.get("decision_policy")
    if not isinstance(policy, Mapping) or policy.get("router_policy") != authority.get("router"):
        return "native Luna verdict authority audit is invalid"
    try:
        runtime_identity = _runtime_identity()
    except NativeLunaIdentityError:
        return "native Luna runtime identity is unavailable"
    current_public = native_luna_route_identity()
    if routes[0] != current_public:
        return "native Luna authority route changed"
    reviews: list[Mapping[str, Any]]
    receipts = review.get("native_luna_receipts")
    if isinstance(receipts, list):
        votes = review.get("votes")
        if len(receipts) != len(votes) if isinstance(votes, list) else True:
            return "native Luna vote receipts are invalid"
        reviews = [item for item in votes if isinstance(item, Mapping)] if isinstance(votes, list) else []
        if len(reviews) != len(receipts):
            return "native Luna vote receipts are invalid"
        for item, receipt in zip(reviews, receipts, strict=True):
            if item.get("reviewer") != "native_luna" or _validate_receipt(receipt, item, runtime_identity) is not None:
                return "native Luna vote proof is invalid"
        return None
    receipt = review.get("native_luna_receipt")
    if review.get("reviewer") != "native_luna":
        return "native Luna verdict reviewer is invalid"
    return _validate_receipt(receipt, review, runtime_identity)


__all__ = [
    "NATIVE_LUNA_AUTHORITY_ENV",
    "NATIVE_LUNA_AUTHORITY_KIND",
    "NATIVE_LUNA_AUTHORITY_VALUE",
    "NATIVE_LUNA_MODEL",
    "NATIVE_LUNA_SOURCE",
    "NativeLunaCapacityUnavailable",
    "NativeLunaIdentityError",
    "clear_native_luna_runtime_identity_cache",
    "native_luna_capacity_lease",
    "native_luna_authority_for_lane",
    "native_luna_enabled_for_lane",
    "native_luna_route_identity",
    "native_luna_route_provenance_error",
    "run_native_luna_review",
    "validate_native_luna_request",
    "validate_native_luna_review",
]
