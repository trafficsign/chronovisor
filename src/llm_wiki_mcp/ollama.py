"""Ollama API client for Ingest/Lint operations."""

import fcntl
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import httpx

from llm_wiki_mcp.runtime_config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INGEST_MODEL,
    IngestConfig,
    load_embedding_config,
    load_ingest_config,
)

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
MODEL = DEFAULT_INGEST_MODEL

# Health check cache
_health_cache: dict = {"status": None, "checked_at": 0.0}
HEALTH_CACHE_TTL = 900  # 15 minutes on failure

# Shared httpx.Client — one per process, reused across is_available /
# generate / embed / unload. Connection pooling avoids paying TCP setup
# and DNS lookup cost on every call. Per-call timeouts are still passed
# explicitly so the long-running /api/generate doesn't inherit the short
# health-check default.
_CLIENT_LOCK = threading.Lock()
_CLIENT: httpx.Client | None = None
_RESOURCE_LEASE_STATE = threading.local()
_MODEL_FOOTPRINT_CALIBRATION: dict[tuple[str, int, int, str, str], int] = {}
_CALIBRATION_IO_LOCK = threading.Lock()
_CALIBRATION_SCHEMA_VERSION = 2


class _ProcessResourceLock:
    """A writer-preferring reader/writer lock for this Python process."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer_thread: int | None = None
        self._waiting_writers = 0

    def acquire(self, *, exclusive: bool) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            if exclusive:
                self._waiting_writers += 1
                try:
                    self._condition.wait_for(
                        lambda: self._writer_thread is None and self._readers == 0
                    )
                    self._writer_thread = thread_id
                finally:
                    self._waiting_writers -= 1
                return
            self._condition.wait_for(
                lambda: self._writer_thread is None and self._waiting_writers == 0
            )
            self._readers += 1

    def release(self, *, exclusive: bool) -> None:
        with self._condition:
            if exclusive:
                if self._writer_thread != threading.get_ident():
                    raise RuntimeError(
                        "resource lease writer released by another thread"
                    )
                self._writer_thread = None
            else:
                if self._readers < 1:
                    raise RuntimeError("resource lease reader count underflow")
                self._readers -= 1
            self._condition.notify_all()


_PROCESS_RESOURCE_LOCK = _ProcessResourceLock()


class OutputTooLargeError(RuntimeError):
    """Raised when a structured chat response crosses its fixed char cap."""


@dataclass(frozen=True)
class ChatResponse:
    """Structured chat content plus Ollama's context accounting."""

    content: str
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    # Defaults preserve compatibility with in-process transports that created
    # ``ChatResponse`` before completion metadata was exposed.  The real HTTP
    # adapter always supplies these fields explicitly and treats an omitted
    # Ollama ``done`` flag as incomplete.
    done: bool = True
    done_reason: str | None = None


@dataclass(frozen=True)
class GenerateResponse:
    """Generate content plus Ollama's explicit completion accounting."""

    content: str
    done: bool
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    streamed: bool = False


GIB = 1024**3
RESIDENCY_UPSHIFT_MIN_HEADROOM_BYTES = 2 * GIB
RESIDENCY_UPSHIFT_HEADROOM_RATIO = 0.10
RESIDENCY_CONTEXT_FLOOR_TOLERANCE_BYTES = 256 * 1024 * 1024
RESIDENCY_CONTEXT_FLOOR_TOLERANCE_RATIO = 0.02


@dataclass(frozen=True)
class MemorySnapshot:
    total_bytes: int
    available_bytes: int
    source: str


@dataclass(frozen=True)
class ModelResidencyPlan:
    """One prompt-specific cap on concurrently resident decision runners."""

    num_ctx: int
    max_resident_models: int
    capacity_bytes: int
    reserve_bytes: int
    available_bytes: int
    total_bytes: int
    estimated_model_bytes: tuple[tuple[str, int], ...]
    role_contexts: tuple[tuple[str, int], ...]
    resident_models: tuple[str, ...]
    calibrated_models: tuple[str, ...]
    source: str
    initial_eviction_models: tuple[str, ...] = ()
    context_floor_models: tuple[str, ...] = ()
    forced_single: bool = False
    reuse_larger_context: bool = False

    def estimate(self, model: str) -> int:
        return dict(self.estimated_model_bytes).get(model, 0)

    def context_for(self, model: str) -> int:
        return dict(self.role_contexts).get(model, self.num_ctx)

    def audit_record(self) -> dict[str, Any]:
        return {
            "num_ctx": self.num_ctx,
            "max_resident_models": self.max_resident_models,
            "capacity_bytes": self.capacity_bytes,
            "reserve_bytes": self.reserve_bytes,
            "available_bytes": self.available_bytes,
            "total_bytes": self.total_bytes,
            "estimated_model_bytes": dict(self.estimated_model_bytes),
            "role_contexts": dict(self.role_contexts),
            "resident_models": list(self.resident_models),
            "calibrated_models": list(self.calibrated_models),
            "initial_eviction_models": list(self.initial_eviction_models),
            "context_floor_models": list(self.context_floor_models),
            "source": self.source,
            "forced_single": self.forced_single,
            "reuse_larger_context": self.reuse_larger_context,
            "upshift_min_headroom_bytes": RESIDENCY_UPSHIFT_MIN_HEADROOM_BYTES,
            "upshift_headroom_ratio": RESIDENCY_UPSHIFT_HEADROOM_RATIO,
        }


def _client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = httpx.Client(base_url=OLLAMA_URL)
    return _CLIENT


def _raise_for_status_with_detail(response: httpx.Response) -> None:
    """Preserve Ollama's bounded error body in runtime diagnostics."""

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            response.read()
        except Exception:
            pass
        detail = ""
        try:
            body = response.json()
        except Exception:
            body = None
        if isinstance(body, Mapping) and isinstance(body.get("error"), str):
            detail = str(body["error"]).strip()
        if not detail:
            try:
                detail = response.text.strip()
            except Exception:
                detail = ""
        detail = re.sub(r"\s+", " ", detail)[:1_000]
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Ollama HTTP {response.status_code}{suffix}") from exc


@contextmanager
def model_resource_lease(*, exclusive: bool) -> Iterator[None]:
    """Coordinate inference and runner eviction across threads and processes.

    A thread holding an exclusive lease may safely enter shared or exclusive
    code without weakening its lease.  Upgrading a shared lease to exclusive
    is rejected instead of risking an upgrade deadlock.
    """

    depth = int(getattr(_RESOURCE_LEASE_STATE, "depth", 0))
    if depth > 0:
        held_exclusive = bool(getattr(_RESOURCE_LEASE_STATE, "exclusive", False))
        if exclusive and not held_exclusive:
            raise RuntimeError(
                "cannot upgrade a shared Ollama resource lease to exclusive"
            )
        _RESOURCE_LEASE_STATE.depth = depth + 1
        try:
            yield
        finally:
            _RESOURCE_LEASE_STATE.depth -= 1
        return
    lock_path = Path(
        os.environ.get(
            "LLM_WIKI_OLLAMA_RESOURCE_LOCK",
            str(Path.home() / ".wiki/runtime/ollama-resource.lock"),
        )
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _PROCESS_RESOURCE_LOCK.acquire(exclusive=exclusive)
    try:
        with lock_path.open("a+") as handle:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
            _RESOURCE_LEASE_STATE.depth = 1
            _RESOURCE_LEASE_STATE.exclusive = exclusive
            try:
                yield
            finally:
                _RESOURCE_LEASE_STATE.depth = 0
                _RESOURCE_LEASE_STATE.exclusive = False
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        _PROCESS_RESOURCE_LOCK.release(exclusive=exclusive)


def model_resource_lease_mode() -> str | None:
    """Return the current thread's nested resource-lease mode, if any."""

    if int(getattr(_RESOURCE_LEASE_STATE, "depth", 0)) < 1:
        return None
    return (
        "exclusive"
        if bool(getattr(_RESOURCE_LEASE_STATE, "exclusive", False))
        else "shared"
    )


def is_available() -> bool:
    """Check if Ollama is running (cached on failure)."""
    now = time.time()

    # If last check failed, use cache for TTL
    if _health_cache["status"] is False:
        if now - _health_cache["checked_at"] < HEALTH_CACHE_TTL:
            return False

    try:
        resp = _client().get("/api/tags", timeout=3)
        available = resp.status_code == 200
        _health_cache["status"] = available
        _health_cache["checked_at"] = now
        return available
    except Exception:
        _health_cache["status"] = False
        _health_cache["checked_at"] = now
        return False


def model_digests(models: Sequence[str]) -> dict[str, str]:
    """Return the currently installed digest for each exact Ollama tag.

    This metadata-only request never loads a model.  Missing tags are returned
    as an empty digest so adoption callers can fail closed without guessing.
    """

    resp = _client().get("/api/tags", timeout=3)
    resp.raise_for_status()
    body = resp.json()
    rows = body.get("models") if isinstance(body, dict) else None
    rows = rows if isinstance(rows, list) else []
    result: dict[str, str] = {}
    for requested in models:
        match = next(
            (
                row
                for row in rows
                if isinstance(row, dict)
                and requested
                in {str(row.get("name") or ""), str(row.get("model") or "")}
            ),
            None,
        )
        digest = match.get("digest") if isinstance(match, dict) else None
        result[requested] = digest if isinstance(digest, str) else ""
    return result


def _ollama_daemon_process_identity() -> str:
    """Return a restart-scoped identity for the Ollama daemon.

    Footprint-affecting daemon environment cannot be recovered reliably from a
    different process on macOS.  Binding measurements to the daemon PID and
    start time still guarantees that changing those settings and restarting
    Ollama invalidates every previous calibration.  The command is included so
    switching between the CLI daemon and an app-managed daemon is also drift.
    """

    result = subprocess.run(
        ["ps", "-axo", "pid=,lstart=,command="],
        check=True,
        capture_output=True,
        text=True,
        timeout=3,
    )
    candidates: list[str] = []
    for raw_line in result.stdout.splitlines():
        parts = raw_line.strip().split(None, 6)
        if len(parts) != 7 or not parts[0].isdigit():
            continue
        pid, weekday, month, day, started_at, year, command = parts
        executable_and_args = command.strip()
        if "llama-server" in executable_and_args.lower():
            continue
        if not re.search(
            r"(?:^|/)ollama(?:\s+serve)(?:\s|$)",
            executable_and_args,
            flags=re.IGNORECASE,
        ):
            continue
        candidates.append(
            "|".join(
                (
                    pid,
                    weekday,
                    month,
                    day,
                    started_at,
                    year,
                    executable_and_args,
                )
            )
        )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one Ollama daemon process, observed {len(candidates)}"
        )
    return hashlib.sha256(candidates[0].encode("utf-8")).hexdigest()


def _ollama_engine_identity() -> str:
    """Return the exact runner identity for persisted footprint measurements."""

    response = _client().get("/api/version", timeout=3)
    response.raise_for_status()


def _post_json(
    endpoint: str,
    *,
    payload: Mapping[str, Any],
    timeout: httpx.Timeout,
) -> Any:
    """POST one non-streaming Ollama request with the shared error contract."""

    response = _client().post(endpoint, json=dict(payload), timeout=timeout)
    _raise_for_status_with_detail(response)
    return response.json()
    body = response.json()
    version = body.get("version") if isinstance(body, Mapping) else None
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError("Ollama version response is missing version")
    material = "|".join(
        (
            f"ollama-{version.strip()}",
            platform.system().lower(),
            platform.machine().lower(),
            _ollama_daemon_process_identity(),
        )
    )
    return f"ollama-engine-v2:{hashlib.sha256(material.encode('utf-8')).hexdigest()}"


def _calibration_file() -> Path:
    return Path(
        os.environ.get(
            "LLM_WIKI_OLLAMA_CALIBRATION_FILE",
            str(Path.home() / ".wiki/runtime/ollama-footprints.json"),
        )
    )


@contextmanager
def _calibration_store_lease(path: Path, *, exclusive: bool) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with _CALIBRATION_IO_LOCK, lock_path.open("a+") as handle:
        fcntl.flock(
            handle.fileno(),
            fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
        )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_calibration_payload(path: Path) -> dict[str, Any]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(body, dict):
        return {}
    if body.get("schema_version") != _CALIBRATION_SCHEMA_VERSION:
        return {}
    return body


def _atomic_write_calibration_payload(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _matching_persisted_calibrations(
    *,
    installed: Mapping[str, int],
    digests: Mapping[str, str],
    engine: str,
) -> dict[tuple[str, int], int]:
    path = _calibration_file()
    with _calibration_store_lease(path, exclusive=False):
        payload = _read_calibration_payload(path)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    result: dict[tuple[str, int], int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        model = entry.get("model")
        context = entry.get("context")
        installed_size = entry.get("installed_size")
        digest = entry.get("digest")
        entry_engine = entry.get("engine")
        size = entry.get("size_bytes")
        if (
            not isinstance(model, str)
            or not isinstance(context, int)
            or context <= 0
            or not isinstance(installed_size, int)
            or installed_size <= 0
            or not isinstance(digest, str)
            or not digest
            or not isinstance(entry_engine, str)
            or not isinstance(size, int)
            or size <= 0
        ):
            continue
        if (
            installed.get(model) != installed_size
            or digests.get(model) != digest
            or entry_engine != engine
        ):
            continue
        result[(model, context)] = size
        _MODEL_FOOTPRINT_CALIBRATION[
            (model, context, installed_size, digest, engine)
        ] = size
    return result


def _matching_memory_calibrations(
    *,
    installed: Mapping[str, int],
    digests: Mapping[str, str],
    engine: str,
) -> dict[tuple[str, int], int]:
    return {
        (model, context): size
        for (
            model,
            context,
            installed_size,
            digest,
            entry_engine,
        ), size in _MODEL_FOOTPRINT_CALIBRATION.items()
        if installed.get(model) == installed_size
        and digests.get(model) == digest
        and entry_engine == engine
    }


def _persist_model_calibration(
    *,
    model: str,
    context: int,
    installed_size: int,
    digest: str,
    engine: str,
    size_bytes: int,
) -> None:
    path = _calibration_file()
    with _calibration_store_lease(path, exclusive=True):
        payload = _read_calibration_payload(path)
        raw_entries = payload.get("entries")
        entries: list[dict[str, Any]] = []
        if isinstance(raw_entries, list):
            for entry in raw_entries:
                if not isinstance(entry, Mapping):
                    continue
                entry_model = entry.get("model")
                entry_context = entry.get("context")
                if not isinstance(entry_model, str) or not isinstance(
                    entry_context, int
                ):
                    continue
                if entry_model == model and entry_context == context:
                    continue
                entries.append(dict(entry))
        entries.append(
            {
                "model": model,
                "context": context,
                "installed_size": installed_size,
                "digest": digest,
                "engine": engine,
                "size_bytes": size_bytes,
                "observed_at": int(time.time()),
            }
        )
        entries.sort(
            key=lambda entry: (
                str(entry.get("model") or ""),
                int(entry.get("context") or 0),
            )
        )
        _atomic_write_calibration_payload(
            path,
            {
                "schema_version": _CALIBRATION_SCHEMA_VERSION,
                "entries": entries,
            },
        )


def memory_snapshot() -> MemorySnapshot:
    """Read reclaimable host memory without adding a heavyweight dependency."""

    if os.uname().sysname == "Darwin":
        try:
            total_result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            pressure_result = subprocess.run(
                ["memory_pressure", "-Q"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            total = int(total_result.stdout.strip())
            pressure_total_match = re.search(
                r"The system has\s+(\d+)(?:\s|\()",
                pressure_result.stdout,
            )
            percent_match = re.search(
                r"System-wide memory free percentage:\s*(\d+)%",
                pressure_result.stdout,
            )
            if pressure_total_match is None or percent_match is None:
                raise ValueError("memory_pressure output is incomplete")
            if int(pressure_total_match.group(1)) != total:
                raise ValueError("memory_pressure total does not match hw.memsize")
            percent_available = int(percent_match.group(1))
            if not 0 <= percent_available <= 100:
                raise ValueError("memory_pressure free percentage is out of range")
            # macOS decides pressure from more than the raw free/inactive VM
            # queues (notably file-backed cache and compressed-memory state).
            # Use the kernel's own pressure-aware availability percentage so
            # a healthy host can bootstrap one uncalibrated runner.  Rounding
            # down and the independent residency reserve keep admission
            # conservative; malformed/unavailable output falls through to the
            # lower-fidelity vm_stat probe below.
            return MemorySnapshot(
                total_bytes=total,
                available_bytes=(total * percent_available) // 100,
                source="macos_memory_pressure",
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        try:
            total_result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            vm_result = subprocess.run(
                ["vm_stat"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            total = int(total_result.stdout.strip())
            page_match = re.search(r"page size of (\d+) bytes", vm_result.stdout)
            page_size = int(page_match.group(1)) if page_match else 4096
            pages: dict[str, int] = {}
            for line in vm_result.stdout.splitlines():
                match = re.match(r"([^:]+):\s+([0-9.]+)\.?$", line.strip())
                if match:
                    pages[match.group(1)] = int(match.group(2).replace(".", ""))
            reclaimable_names = (
                "Pages free",
                "Pages inactive",
                "Pages speculative",
                "Pages purgeable",
            )
            available = (
                sum(pages.get(name, 0) for name in reclaimable_names) * page_size
            )
            return MemorySnapshot(
                total_bytes=total,
                available_bytes=min(total, max(0, available)),
                source="macos_vm_stat",
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                name, raw = line.split(":", 1)
                match = re.search(r"\d+", raw)
                if match:
                    values[name] = int(match.group()) * 1024
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", values.get("MemFree", 0))
        if total > 0:
            return MemorySnapshot(total, min(total, available), "proc_meminfo")
    except (OSError, ValueError):
        pass
    return MemorySnapshot(0, 0, "unavailable")


def _ollama_resource_rows() -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    tags_response = _client().get("/api/tags", timeout=3)
    tags_response.raise_for_status()
    ps_response = _client().get("/api/ps", timeout=3)
    ps_response.raise_for_status()
    tags_body = tags_response.json()
    ps_body = ps_response.json()
    installed: dict[str, int] = {}
    for row in tags_body.get("models", []) if isinstance(tags_body, Mapping) else []:
        if not isinstance(row, Mapping):
            continue
        size = row.get("size")
        if isinstance(size, int) and size > 0:
            for name in {str(row.get("name") or ""), str(row.get("model") or "")}:
                if name:
                    installed[name] = size
    resident: dict[str, tuple[int, int]] = {}
    for row in ps_body.get("models", []) if isinstance(ps_body, Mapping) else []:
        if not isinstance(row, Mapping):
            continue
        size_vram = row.get("size_vram")
        total_size = row.get("size")
        size = max(
            size_vram if isinstance(size_vram, int) and size_vram > 0 else 0,
            total_size if isinstance(total_size, int) and total_size > 0 else 0,
        )
        context = row.get("context_length")
        if not isinstance(size, int) or size <= 0:
            continue
        context_value = context if isinstance(context, int) and context > 0 else 0
        for name in {str(row.get("name") or ""), str(row.get("model") or "")}:
            if name:
                resident[name] = (size, context_value)
    return installed, resident


def resident_model_rows() -> dict[str, tuple[int, int]]:
    """Return a read-only snapshot of resident model size and context rows."""

    _installed, resident = _ollama_resource_rows()
    return dict(resident)


def build_model_residency_plan(
    models: Sequence[str],
    *,
    num_ctx: int,
    max_num_ctx: int,
    memory: MemorySnapshot,
    installed_sizes: Mapping[str, int],
    resident: Mapping[str, tuple[int, int]],
    calibrated_sizes: Mapping[tuple[str, int], int] | None = None,
    reserve_bytes: int,
    configured_max_resident: int,
    source: str = "measured",
    reuse_larger_context: bool = True,
    reuse_context_ceilings: Mapping[str, int] | None = None,
) -> ModelResidencyPlan:
    """Choose a 1/2/3-runner cap from context-scaled model footprints.

    ``max_num_ctx`` is the absolute ceiling. When per-model reuse ceilings are
    supplied, an omitted or invalid model entry fails closed to the requested
    context instead of inheriting another role's larger allowance.
    """

    ordered = tuple(dict.fromkeys(model for model in models if model))
    if not ordered:
        raise ValueError("at least one model is required")
    maximum = max(1, min(3, configured_max_resident, len(ordered)))
    reserve = (
        max(reserve_bytes, memory.total_bytes // 8)
        if memory.total_bytes
        else reserve_bytes
    )
    resident_candidate_bytes = sum(resident.get(model, (0, 0))[0] for model in ordered)
    exact_requested_calibrations = {
        model: int((calibrated_sizes or {}).get((model, num_ctx), 0))
        for model in ordered
    }
    context_floor_models = tuple(
        model
        for model in ordered
        for resident_size, resident_ctx in (resident.get(model, (0, 0)),)
        for exact_size in (exact_requested_calibrations[model],)
        if not reuse_larger_context
        and exact_size > 0
        and resident_size > 0
        and num_ctx < resident_ctx <= max_num_ctx
        and abs(resident_size - exact_size)
        <= max(
            RESIDENCY_CONTEXT_FLOOR_TOLERANCE_BYTES,
            int(exact_size * RESIDENCY_CONTEXT_FLOOR_TOLERANCE_RATIO),
        )
    )
    context_floor_set = set(context_floor_models)
    default_reuse_ceiling = max_num_ctx if reuse_context_ceilings is None else num_ctx
    reuse_ceiling_by_model: dict[str, int] = {}
    for model in ordered:
        candidate = default_reuse_ceiling
        if reuse_context_ceilings is not None:
            candidate = reuse_context_ceilings.get(model, default_reuse_ceiling)
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            candidate = num_ctx
        reuse_ceiling_by_model[model] = min(
            max_num_ctx,
            max(num_ctx, candidate),
        )
    selected_resident_contexts = {
        model: (
            resident_ctx
            if reuse_larger_context
            and num_ctx <= resident_ctx <= reuse_ceiling_by_model[model]
            and resident_size > 0
            else num_ctx
        )
        for model in ordered
        for resident_size, resident_ctx in (resident.get(model, (0, 0)),)
    }
    compatible_resident_models = {
        model
        for model in ordered
        for resident_size, resident_ctx in (resident.get(model, (0, 0)),)
        if resident_size > 0
        and (
            resident_ctx == selected_resident_contexts[model]
            or model in context_floor_set
        )
    }
    # Candidate runners already consume bytes excluded from available memory.
    # Count those bytes as reclaimable only together with an explicit eviction
    # requirement whenever the loaded context cannot satisfy this plan. This
    # prevents capacity from assuming a huge stale runner will disappear while
    # the router accidentally leaves it resident during new allocations.
    initial_eviction_models = tuple(
        model
        for model in ordered
        for resident_size, resident_ctx in (resident.get(model, (0, 0)),)
        if resident_size > 0 and model not in compatible_resident_models
    )
    capacity = max(0, memory.available_bytes + resident_candidate_bytes - reserve)
    estimates: list[tuple[str, int]] = []
    contexts: list[tuple[str, int]] = []
    calibrated: list[bool] = []
    # Until an exact bucket is measured, assume up to twice the installed
    # weight size for graph/KV overhead.  Exact /api/ps evidence replaces this
    # fallback and is what makes the eventual runner count context-sensitive.
    fallback = max(8 * GIB, memory.total_bytes // 3) if memory.total_bytes else 32 * GIB
    for model in ordered:
        resident_size, resident_ctx = resident.get(model, (0, 0))
        # Production uses monotonic context hysteresis: once a larger runner
        # is resident, keep it while it remains inside the configured ceiling
        # and measured capacity. The one-time adoption evaluator disables this
        # option so it can gather exact evidence for every bucket.
        selected_ctx = selected_resident_contexts[model]
        weight_size = installed_sizes.get(model, 0)
        exact_observed = (
            resident_size
            if resident_size > 0 and model in compatible_resident_models
            else int((calibrated_sizes or {}).get((model, selected_ctx), 0))
        )
        observed_lower_bound = max(
            (
                size
                for (candidate, context), size in (calibrated_sizes or {}).items()
                if candidate == model
                and context <= selected_ctx
                and isinstance(size, int)
                and size > 0
            ),
            default=0,
        )
        conservative_uncalibrated = max(
            int(weight_size * 2.0) if weight_size > 0 else 0,
            fallback,
        )
        # An exact model/context observation is the authoritative footprint for
        # admission.  Do not let a larger-context calibration inflate a 16K or
        # 32K request, otherwise the context-aware 1/2/3-runner policy quietly
        # degenerates into one fixed worst-case estimate.  An unobserved bucket
        # remains fail-safe: it uses every known lower-context allocation as a
        # floor plus the conservative single-runner fallback, and cannot admit
        # a second resident until that exact bucket has been measured.
        estimated = (
            exact_observed
            if exact_observed > 0
            else max(observed_lower_bound, conservative_uncalibrated)
        )
        estimates.append((model, estimated))
        contexts.append((model, selected_ctx))
        calibrated.append(exact_observed > 0)

    admitted = 0
    running = 0
    for index, (_model, estimated) in enumerate(estimates[:maximum]):
        running += estimated
        # File size is only a conservative single-runner fallback.  Multiple
        # residents are admitted only after this exact model/context bucket
        # has been observed through /api/ps in the current process.
        prefix_calibrated = all(calibrated[: index + 1])
        upshift_margin = (
            0
            if index == 0
            else max(
                RESIDENCY_UPSHIFT_MIN_HEADROOM_BYTES,
                int(running * RESIDENCY_UPSHIFT_HEADROOM_RATIO),
            )
        )
        # The first uncalibrated runner is the calibration bootstrap, but it
        # still has to fit in currently reclaimable capacity. Using physical
        # memory here would ignore unrelated resident models and application
        # memory, recreating the pressure spike this planner is meant to avoid.
        # A conservative 2x estimate admits one bootstrap whenever the host has
        # enough current headroom; additional unmeasured runners remain barred
        # by the calibrated-prefix rule below.
        if running + upshift_margin <= capacity and (index == 0 or prefix_calibrated):
            admitted += 1
        else:
            break
    forced_single = admitted == 0
    return ModelResidencyPlan(
        num_ctx=num_ctx,
        max_resident_models=admitted,
        capacity_bytes=capacity,
        reserve_bytes=reserve,
        available_bytes=memory.available_bytes,
        total_bytes=memory.total_bytes,
        estimated_model_bytes=tuple(estimates),
        role_contexts=tuple(contexts),
        resident_models=tuple(model for model in ordered if model in resident),
        calibrated_models=tuple(
            model for (model, _estimated), known in zip(estimates, calibrated) if known
        ),
        source=source,
        initial_eviction_models=initial_eviction_models,
        context_floor_models=context_floor_models,
        forced_single=forced_single,
        reuse_larger_context=reuse_larger_context,
    )


def plan_model_residency(
    models: Sequence[str],
    *,
    num_ctx: int,
    max_num_ctx: int,
    reserve_bytes: int,
    configured_max_resident: int,
    reuse_larger_context: bool = True,
    reuse_context_ceilings: Mapping[str, int] | None = None,
) -> ModelResidencyPlan:
    """Probe live host/Ollama state and return a fail-safe residency plan."""

    memory = memory_snapshot()
    try:
        installed, resident = _ollama_resource_rows()
        source = f"{memory.source}+ollama"
    except Exception:
        installed, resident = {}, {}
        source = f"{memory.source}+ollama_unavailable"
    calibrated: dict[tuple[str, int], int] = {}
    if installed:
        try:
            digests = model_digests(models)
            engine = _ollama_engine_identity()
            calibrated.update(
                _matching_persisted_calibrations(
                    installed=installed,
                    digests=digests,
                    engine=engine,
                )
            )
            calibrated.update(
                _matching_memory_calibrations(
                    installed=installed,
                    digests=digests,
                    engine=engine,
                )
            )
        except Exception:
            source = f"{source}+identity_unavailable"
    plan = build_model_residency_plan(
        models,
        num_ctx=num_ctx,
        max_num_ctx=max_num_ctx,
        memory=memory,
        installed_sizes=installed,
        resident=resident,
        calibrated_sizes=calibrated,
        reserve_bytes=reserve_bytes,
        configured_max_resident=configured_max_resident,
        source=source,
        reuse_larger_context=reuse_larger_context,
        reuse_context_ceilings=reuse_context_ceilings,
    )
    if (
        memory.total_bytes <= 0
        or memory.available_bytes <= 0
        or not installed
        or "identity_unavailable" in source
    ):
        return replace(plan, max_resident_models=0, forced_single=True)
    return plan


def observe_model_runtime(model: str) -> tuple[int, int] | None:
    """Calibrate one exact installed tag and context from Ollama's live runner."""

    try:
        installed, resident = _ollama_resource_rows()
    except Exception:
        return None
    row = resident.get(model)
    weight_size = installed.get(model, 0)
    if row is None or weight_size <= 0:
        return None
    size, context = row
    if size <= 0 or context <= 0:
        return None
    try:
        digest = model_digests([model]).get(model, "")
        engine = _ollama_engine_identity()
        if not digest:
            return row
        _MODEL_FOOTPRINT_CALIBRATION[(model, context, weight_size, digest, engine)] = (
            size
        )
        _persist_model_calibration(
            model=model,
            context=context,
            installed_size=weight_size,
            digest=digest,
            engine=engine,
            size_bytes=size,
        )
    except Exception:
        log.debug("failed to persist Ollama footprint calibration", exc_info=True)
    return row


def unload_named_model(model: str, *, verify_timeout: float = 30.0) -> bool:
    """Unload one known runner and verify that it disappeared from /api/ps."""

    with model_resource_lease(exclusive=True):
        try:
            response = _client().post(
                "/api/generate",
                json={"model": model, "keep_alive": 0, "prompt": ""},
                timeout=10,
            )
            if response.status_code != 200:
                return False
            deadline = time.monotonic() + max(0.0, verify_timeout)
            while True:
                try:
                    _installed, resident = _ollama_resource_rows()
                    if model not in resident:
                        return True
                except Exception:
                    return False
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.05)
        except Exception:
            return False


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]
) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        pass


def ingest_model() -> str:
    return load_ingest_config().model


def _num_ctx_for_prompt(prompt: str, system: str | None, config: IngestConfig) -> int:
    # Keep ordinary saves on a smaller MLX context, but grow for unusually long
    # raw transcripts so the old 262K ceiling remains available when needed.
    prompt_chars = len(prompt) + (len(system) if system else 0)
    estimated_prompt_tokens = max(1, (prompt_chars + 1) // 2)
    needed = estimated_prompt_tokens + config.num_predict + 1024
    return min(max(config.num_ctx, needed), config.max_num_ctx)


def _generate_unlocked(
    prompt: str,
    system: str | None = None,
    *,
    format: dict | str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    model: str | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    keep_alive: str | None = None,
    read_timeout_ms: int | None = None,
    temperature: int | float | None = None,
    seed: int | None = None,
    return_metadata: bool = False,
) -> str | GenerateResponse:
    """Call Ollama generate API.

    Uses keep_alive="5m" to keep model loaded for 5 minutes after use.
    This avoids cold-start on consecutive calls (e.g. Ingest then Lint)
    while still freeing memory after a reasonable idle period.

    When ``progress_callback`` is provided, the call uses Ollama's streaming
    response and periodically emits lightweight progress dictionaries while
    still returning the final response string for existing callers.
    """
    config = load_ingest_config()
    selected_model = (
        model.strip() if isinstance(model, str) and model.strip() else config.model
    )
    selected_num_ctx = (
        num_ctx
        if isinstance(num_ctx, int) and not isinstance(num_ctx, bool) and num_ctx > 0
        else _num_ctx_for_prompt(prompt, system, config)
    )
    selected_num_predict = (
        num_predict
        if isinstance(num_predict, int)
        and not isinstance(num_predict, bool)
        and num_predict > 0
        else config.num_predict
    )
    selected_keep_alive = (
        keep_alive
        if isinstance(keep_alive, str) and keep_alive.strip()
        else config.keep_alive
    )
    selected_read_timeout_ms = (
        read_timeout_ms
        if isinstance(read_timeout_ms, int)
        and not isinstance(read_timeout_ms, bool)
        and read_timeout_ms > 0
        else config.read_timeout_ms
    )
    selected_temperature = (
        temperature
        if isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
        else config.temperature
    )
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError("generate seed must be a non-negative integer")
    prompt_chars = len(prompt) + (len(system) if system else 0)
    log.info(
        "generate num_ctx=%d prompt_chars=%d model=%s",
        selected_num_ctx,
        prompt_chars,
        selected_model,
    )
    payload = {
        "model": selected_model,
        "prompt": prompt,
        "stream": progress_callback is not None,
        "think": False,
        # Never let Ollama silently discard the oldest input to satisfy a
        # smaller runner. Ingest performs its own fail-closed context sizing.
        "shift": False,
        "truncate": False,
        "keep_alive": selected_keep_alive,
        "options": {
            "temperature": selected_temperature,
            "num_predict": selected_num_predict,
            "num_ctx": selected_num_ctx,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed
    if system:
        payload["system"] = system
    if format is not None:
        payload["format"] = format

    # Timeout: 60s for model load + 600s for generation
    timeout = httpx.Timeout(
        connect=10.0,
        read=selected_read_timeout_ms / 1000,
        write=10.0,
        pool=10.0,
    )
    if progress_callback is not None:
        chunks = 0
        chars = 0
        started = time.monotonic()
        last_emit = 0.0
        pieces: list[str] = []
        final_payload: dict[str, Any] | None = None

        with _client().stream(
            "POST",
            "/api/generate",
            json=payload,
            timeout=timeout,
        ) as resp:
            _raise_for_status_with_detail(resp)
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                piece = data.get("response") or ""
                if piece:
                    pieces.append(piece)
                    chunks += 1
                    chars += len(piece)

                done = data.get("done") is True
                now = time.monotonic()
                elapsed = max(0.001, now - started)
                if done or now - last_emit >= 0.75:
                    update = {
                        "event": "done" if done else "chunk",
                        "active": not done,
                        "generated_chars": chars,
                        "chunks": chunks,
                        "elapsed_seconds": round(elapsed, 2),
                        "chars_per_second": round(chars / elapsed, 1),
                    }
                    for key in (
                        "total_duration",
                        "load_duration",
                        "prompt_eval_count",
                        "prompt_eval_duration",
                        "eval_count",
                        "eval_duration",
                    ):
                        if key in data:
                            update[key] = data[key]
                    _emit_progress(progress_callback, update)
                    last_emit = now

                if done:
                    final_payload = data
                    break

        if final_payload is None:
            _emit_progress(
                progress_callback,
                {
                    "event": "error",
                    "active": False,
                    "generated_chars": chars,
                    "chunks": chunks,
                    "elapsed_seconds": round(max(0.001, time.monotonic() - started), 2),
                    "error": "stream ended before done",
                },
            )
            if return_metadata:
                return GenerateResponse(
                    content="".join(pieces),
                    done=False,
                    done_reason=None,
                    streamed=True,
                )
            raise RuntimeError("Ollama stream ended before done")
        content = "".join(pieces)
        if not return_metadata:
            return content
        return GenerateResponse(
            content=content,
            done=final_payload.get("done") is True,
            done_reason=(
                str(final_payload["done_reason"])
                if isinstance(final_payload.get("done_reason"), str)
                else None
            ),
            prompt_eval_count=(
                int(final_payload["prompt_eval_count"])
                if isinstance(final_payload.get("prompt_eval_count"), int)
                and not isinstance(final_payload.get("prompt_eval_count"), bool)
                else None
            ),
            eval_count=(
                int(final_payload["eval_count"])
                if isinstance(final_payload.get("eval_count"), int)
                and not isinstance(final_payload.get("eval_count"), bool)
                else None
            ),
            streamed=True,
        )

    body = _post_json("/api/generate", payload=payload, timeout=timeout)
    if not isinstance(body, dict) or not isinstance(body.get("response"), str):
        raise RuntimeError("Ollama generate response is missing response content")
    content = str(body["response"])
    if not return_metadata:
        return content
    return GenerateResponse(
        content=content,
        done=body.get("done") is True,
        done_reason=(
            str(body["done_reason"])
            if isinstance(body.get("done_reason"), str)
            else None
        ),
        prompt_eval_count=(
            int(body["prompt_eval_count"])
            if isinstance(body.get("prompt_eval_count"), int)
            and not isinstance(body.get("prompt_eval_count"), bool)
            else None
        ),
        eval_count=(
            int(body["eval_count"])
            if isinstance(body.get("eval_count"), int)
            and not isinstance(body.get("eval_count"), bool)
            else None
        ),
    )


def generate(
    prompt: str,
    system: str | None = None,
    *,
    format: dict | str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    model: str | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    keep_alive: str | None = None,
    read_timeout_ms: int | None = None,
    temperature: int | float | None = None,
    seed: int | None = None,
    return_metadata: bool = False,
) -> str | GenerateResponse:
    with model_resource_lease(exclusive=False):
        return _generate_unlocked(
            prompt,
            system,
            format=format,
            progress_callback=progress_callback,
            model=model,
            num_ctx=num_ctx,
            num_predict=num_predict,
            keep_alive=keep_alive,
            read_timeout_ms=read_timeout_ms,
            temperature=temperature,
            seed=seed,
            return_metadata=return_metadata,
        )


def _chat_unlocked(
    messages: list[dict[str, str]],
    *,
    model: str,
    format: dict[str, Any],
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
    read_timeout_ms: int,
    max_output_chars: int,
    temperature: int | float = 0,
    seed: int = 0,
    return_metadata: bool = False,
) -> str | ChatResponse:
    """Call Ollama's chat API for one fixed-cap structured-output turn.

    Unlike :func:`generate`, this adapter never derives context size from the
    prompt.  Decision models therefore keep a stable runner allocation across
    initial and repair turns.  Only ``message.content`` is returned; any
    separate thinking field is intentionally ignored.
    """

    if not isinstance(model, str) or not model.strip():
        raise ValueError("model is required")
    if num_ctx < 1 or num_predict < 1 or max_output_chars < 1:
        raise ValueError("chat limits must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("chat seed must be a non-negative integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("chat temperature must be numeric")
    payload = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "stream": False,
        "think": False,
        "shift": False,
        "truncate": False,
        "format": format,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    timeout = httpx.Timeout(
        connect=10.0,
        read=read_timeout_ms / 1000,
        write=10.0,
        pool=10.0,
    )
    body = _post_json("/api/chat", payload=payload, timeout=timeout)
    message = body.get("message") if isinstance(body, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise RuntimeError("Ollama chat response is missing message.content")
    # Metadata callers are the bounded structured-session layer.  Return the
    # response to that layer even when it crossed the cap so it can record a
    # redacted attempt and ask the same model for a compact repair.  Plain
    # callers have no repair protocol and must continue to fail closed here.
    if len(content) > max_output_chars and not return_metadata:
        raise OutputTooLargeError(
            f"Ollama chat response exceeded max_output_chars={max_output_chars}"
        )
    if not return_metadata:
        return content
    prompt_eval_count = (
        body.get("prompt_eval_count") if isinstance(body, dict) else None
    )
    eval_count = body.get("eval_count") if isinstance(body, dict) else None
    return ChatResponse(
        content=content,
        prompt_eval_count=(
            prompt_eval_count
            if isinstance(prompt_eval_count, int)
            and not isinstance(prompt_eval_count, bool)
            else None
        ),
        eval_count=(
            eval_count
            if isinstance(eval_count, int) and not isinstance(eval_count, bool)
            else None
        ),
        done=body.get("done") is True,
        done_reason=(
            str(body["done_reason"])
            if isinstance(body.get("done_reason"), str)
            else None
        ),
    )


def chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    format: dict[str, Any],
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
    read_timeout_ms: int,
    max_output_chars: int,
    temperature: int | float = 0,
    seed: int = 0,
    return_metadata: bool = False,
) -> str | ChatResponse:
    with model_resource_lease(exclusive=False):
        return _chat_unlocked(
            messages,
            model=model,
            format=format,
            num_ctx=num_ctx,
            num_predict=num_predict,
            keep_alive=keep_alive,
            read_timeout_ms=read_timeout_ms,
            max_output_chars=max_output_chars,
            temperature=temperature,
            seed=seed,
            return_metadata=return_metadata,
        )


EMBED_MODEL = DEFAULT_EMBEDDING_MODEL


def embedding_model() -> str:
    return load_embedding_config().model


def embed(
    texts: list[str],
    *,
    model: str | None = None,
    read_timeout_ms: int | None = None,
) -> list[list[float]]:
    """Get embedding vectors via Ollama /api/embed."""
    timeout_seconds = (
        max(0.2, read_timeout_ms / 1000.0)
        if isinstance(read_timeout_ms, int)
        else 120.0
    )
    with model_resource_lease(exclusive=False):
        resp = _client().post(
            "/api/embed",
            json={"model": model or embedding_model(), "input": texts},
            timeout=httpx.Timeout(
                connect=min(10.0, timeout_seconds),
                read=timeout_seconds,
                write=min(10.0, timeout_seconds),
                pool=min(10.0, timeout_seconds),
            ),
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


def unload_model() -> None:
    """Explicitly unload model to free memory."""
    unload_named_model(ingest_model())


TRIAGE_SYSTEM_PROMPT = """\
You are a knowledge wiki triage engine. Analyze raw session data and decide \
what wiki pages to create or update. Do NOT generate page content — only output a structured plan.

Rules:
- 1 entity = 1 page
- Output valid JSON array only (no markdown fences, no explanation)
- For every new page, emit exactly `folder/kebab-case.md`; a bare filename is forbidden
- Prefer the best semantically matching folder from the provided existing-folder list
- Only when no existing folder fits, create one specific new top-level folder in
  English kebab-case and place the page there
- Do not use `misc/` merely to avoid choosing or creating a meaningful folder;
  use it only for genuinely miscellaneous knowledge
- For updates: reference the existing page ID in a field named "filename"
- Every update object MUST use "filename". Never emit a "page_id" field
- If the target page is not listed in the catalog, use create, not update
- Skip ephemeral conversation, greetings, and filler
- Include brief summary of what knowledge each page should contain
- Include keywords for finding related existing pages
- Use only these five object keys: type, filename, title, keywords, summary
- Every operation, including updates, MUST include non-empty title, keywords,
  and summary fields
- Emit exactly one operation per case/Unicode-insensitive target page ID. If
  several facts belong on one page, preserve all of them in one combined
  summary and keyword set; never emit multiple operations for that target

Output format (JSON array only):
[
  {
    "type": "create",
    "filename": "folder/kebab-case.md",
    "title": "Page Title",
    "keywords": ["keyword1", "keyword2"],
    "summary": "Brief description of what this page should cover"
  },
  {
    "type": "update",
    "filename": "existing-page.md",
    "title": "Existing Page Title",
    "keywords": ["keyword1", "keyword2"],
    "summary": "What new information to add"
  }
]

WRONG output (do NOT do these):
- Bare keyword list: ["keyword1", "keyword2"]   ← This is a list of strings, not operations
- Single object: {"type": "create", ...}        ← Must be wrapped in an array
- Code fences around the JSON                   ← Output raw JSON only
- Root-level create: {"type": "create", "filename": "topic.md", ...}
  ← Every create must use exactly one top-level folder: `folder/topic.md`

Each top-level element of the array MUST be an object with a "type" field.
"""

GENERATE_SYSTEM_PROMPT = """\
You are a knowledge wiki structuring engine. Generate content for a SINGLE NEW wiki page.

Rules:
- Frontmatter MUST include: title, updated, AND tags
- Use the exact current date supplied in the user prompt for `updated`
- Never invent or infer dates that are absent from the raw evidence
- Cross-references: use [[wiki-link]] notation (page ID only, no folder path)
- Write content in Japanese
- Focus on facts, decisions, and technical knowledge
- Use the provided context for cross-references but do not duplicate existing content

# Tag Taxonomy v0.1 (REQUIRED)

Every page must carry a ``tags:`` frontmatter list with prefixed entries
from a controlled taxonomy. Three axes:

  d/  Domain  (1-3 required) — subject area, kebab-case
       seeds: d/ai-industry, d/hardware, d/geopolitics, d/health, d/finance,
              d/personal-strategy, d/tools-config, d/japan, d/theory, d/paranormal
  t/  Type   (exactly 1 required) — content type, kebab-case
       seeds: t/analysis, t/chat-log, t/howto, t/reference, t/decision,
              t/scenario, t/news-summary
  s/  Scope  (exactly 1 required) — temporal/spatial scope
       seeds: s/2026, s/evergreen, s/historical

# Tag generation rules v1.0

1. Prefix REQUIRED (d/, t/, or s/). Never emit a tag without a prefix.
2. ASCII kebab-case body only (lowercase letters, digits, hyphens). No
   underscores, no spaces, no uppercase, no non-ASCII.
3. Maximum 2 words per tag (split by hyphen). Three+ words → keywords, not tags.
4. Singular form (analysis, not analyses).
5. NO proper nouns (product names, person names, project names) — those
   are keywords. Tags are categorical, not specific.
6. Numbers/years allowed only on the s/ axis (e.g. s/2026). The d/ and t/
   axes must start with a letter.
7. Prefer existing seed tags above when they fit. New tags should be
   genuinely novel categories, not synonyms of existing ones.

Output exactly one page block:
=== NEW PAGE: {filename} ===
---
title: Page Title
updated: YYYY-MM-DD
tags: [d/example-domain, t/analysis, s/evergreen]
---

Page content here with [[wiki-links]] to related topics.

=== END PAGE ===

The final non-whitespace line MUST be exactly `=== END PAGE ===`. Keep the
page concise enough to emit that closing line before stopping.
"""

UPDATE_SYSTEM_PROMPT = """\
You are a knowledge wiki structuring engine. Append content to an EXISTING wiki page.

Rules:
- DO NOT output frontmatter (no `---`, no title:, no updated: lines). The existing page already has frontmatter; your output is appended to its body.
- Never invent or infer dates that are absent from the raw evidence. Do not add a dated heading unless that date appears explicitly in the raw evidence.
- DO NOT repeat content that already exists on the page (it is provided in context).
- Output ONLY the new section(s) to add — Japanese prose, headings, lists, code, etc.
- Cross-references: use [[wiki-link]] notation (page ID only, no folder path)
- Focus on facts, decisions, and technical knowledge

Output exactly one block:
=== UPDATE PAGE: {filename} ===
New section(s) here. Markdown body only — NO frontmatter delimiters.

=== END PAGE ===

The final non-whitespace line MUST be exactly `=== END PAGE ===`. Keep the
update concise enough to emit that closing line before stopping.
"""
