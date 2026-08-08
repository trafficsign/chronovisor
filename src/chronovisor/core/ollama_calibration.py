"""Ollama footprint calibration and model residency planning."""

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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_MODEL_FOOTPRINT_CALIBRATION: dict[tuple[str, int, int, str, str], int] = {}
_CALIBRATION_IO_LOCK = threading.Lock()
_CALIBRATION_SCHEMA_VERSION = 2

log = logging.getLogger("chronovisor.core.ollama")

GIB = 1024**3
RESIDENCY_UPSHIFT_MIN_HEADROOM_BYTES = 2 * GIB
RESIDENCY_UPSHIFT_HEADROOM_RATIO = 0.10
RESIDENCY_CONTEXT_FLOOR_TOLERANCE_BYTES = 256 * 1024 * 1024
RESIDENCY_CONTEXT_FLOOR_TOLERANCE_RATIO = 0.02
RESIDENCY_COMPRESSED_SINGLE_MIN_BYTES = 8 * GIB
RESIDENCY_COMPRESSED_SINGLE_RATIO = 0.20
RESIDENCY_SWAP_SINGLE_MIN_BYTES = 1 * GIB
RESIDENCY_SWAP_COMPRESSED_FLOOR_BYTES = 4 * GIB
RESIDENCY_SWAP_COMPRESSED_FLOOR_RATIO = 0.0625


@dataclass(frozen=True)
class MemorySnapshot:
    total_bytes: int
    available_bytes: int
    source: str


@dataclass(frozen=True)
class MacOSPressureSnapshot:
    compressed_bytes: int
    swap_used_bytes: int
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
    pressure_forced_single: bool = False
    compressed_bytes: int = 0
    swap_used_bytes: int = 0

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
            "pressure_forced_single": self.pressure_forced_single,
            "compressed_bytes": self.compressed_bytes,
            "swap_used_bytes": self.swap_used_bytes,
            "upshift_min_headroom_bytes": RESIDENCY_UPSHIFT_MIN_HEADROOM_BYTES,
            "upshift_headroom_ratio": RESIDENCY_UPSHIFT_HEADROOM_RATIO,
        }


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


def _ollama_engine_identity(
    *,
    client: Callable[[], Any],
    daemon_identity: Callable[[], str],
) -> str:
    """Return the exact runner identity for persisted footprint measurements."""

    response = client().get("/api/version", timeout=3)
    response.raise_for_status()
    body = response.json()
    version = body.get("version") if isinstance(body, Mapping) else None
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError("Ollama version response is missing version")
    material = "|".join(
        (
            f"ollama-{version.strip()}",
            platform.system().lower(),
            platform.machine().lower(),
            daemon_identity(),
        )
    )
    return f"ollama-engine-v2:{hashlib.sha256(material.encode('utf-8')).hexdigest()}"


def _calibration_file(*, root: Path) -> Path:
    return Path(
        os.environ.get(
            "CHRONOVISOR_OLLAMA_CALIBRATION_FILE",
            str(root / "runtime/ollama-footprints.json"),
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
        with suppress(FileNotFoundError):
            temporary.unlink()


def _matching_persisted_calibrations(
    *,
    root: Path,
    installed: Mapping[str, int],
    digests: Mapping[str, str],
    engine: str,
) -> dict[tuple[str, int], int]:
    path = _calibration_file(root=root)
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
    root: Path,
    model: str,
    context: int,
    installed_size: int,
    digest: str,
    engine: str,
    size_bytes: int,
) -> None:
    path = _calibration_file(root=root)
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


def macos_pressure_snapshot() -> MacOSPressureSnapshot:
    """Read compressed-memory and swap occupancy used for residency upshifts."""

    if os.uname().sysname != "Darwin":
        return MacOSPressureSnapshot(0, 0, "not_macos")
    compressed_bytes = 0
    swap_used_bytes = 0
    sources: list[str] = []
    try:
        vm_result = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        page_match = re.search(r"page size of (\d+) bytes", vm_result.stdout)
        page_size = int(page_match.group(1)) if page_match else 4096
        compressed_match = re.search(
            r"Pages occupied by compressor:\s+([0-9.]+)\.?",
            vm_result.stdout,
        )
        if compressed_match is None:
            raise ValueError("vm_stat compressor occupancy is unavailable")
        compressed_pages = int(compressed_match.group(1).replace(".", ""))
        compressed_bytes = compressed_pages * page_size
        sources.append("vm_stat")
    except (OSError, ValueError, subprocess.SubprocessError):
        sources.append("vm_stat_unavailable")
    try:
        swap_result = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        used_match = re.search(
            r"used\s*=\s*([0-9.]+)([KMGTP])",
            swap_result.stdout,
            flags=re.IGNORECASE,
        )
        if used_match is None:
            raise ValueError("vm.swapusage used value is unavailable")
        unit_bytes = {
            "K": 1024,
            "M": 1024**2,
            "G": 1024**3,
            "T": 1024**4,
            "P": 1024**5,
        }[used_match.group(2).upper()]
        swap_used_bytes = int(float(used_match.group(1)) * unit_bytes)
        sources.append("swapusage")
    except (OSError, ValueError, subprocess.SubprocessError):
        sources.append("swapusage_unavailable")
    return MacOSPressureSnapshot(
        compressed_bytes=max(0, compressed_bytes),
        swap_used_bytes=max(0, swap_used_bytes),
        source="+".join(sources),
    )


def memory_pressure_requires_single_resident(
    memory: MemorySnapshot,
    pressure: MacOSPressureSnapshot,
) -> bool:
    """Refuse residency upshifts while macOS is compressing or swapping heavily."""

    if memory.total_bytes <= 0:
        return True
    compressed_limit = max(
        RESIDENCY_COMPRESSED_SINGLE_MIN_BYTES,
        int(memory.total_bytes * RESIDENCY_COMPRESSED_SINGLE_RATIO),
    )
    swap_compressed_floor = max(
        RESIDENCY_SWAP_COMPRESSED_FLOOR_BYTES,
        int(memory.total_bytes * RESIDENCY_SWAP_COMPRESSED_FLOOR_RATIO),
    )
    return bool(
        pressure.compressed_bytes >= compressed_limit
        or (
            pressure.swap_used_bytes >= RESIDENCY_SWAP_SINGLE_MIN_BYTES
            and pressure.compressed_bytes >= swap_compressed_floor
        )
    )


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
    pressure: MacOSPressureSnapshot | None = None,
) -> ModelResidencyPlan:
    """Choose a 1/2/3-runner cap from context-scaled model footprints.

    ``max_num_ctx`` is the absolute ceiling. When per-model reuse ceilings are
    supplied, an omitted or invalid model entry fails closed to the requested
    context instead of inheriting another role's larger allowance.
    """

    ordered = tuple(dict.fromkeys(model for model in models if model))
    if not ordered:
        raise ValueError("at least one model is required")
    pressure_snapshot = pressure or MacOSPressureSnapshot(0, 0, "not_probed")
    pressure_forced_single = memory_pressure_requires_single_resident(
        memory,
        pressure_snapshot,
    )
    configured_maximum = max(1, min(3, configured_max_resident, len(ordered)))
    maximum = 1 if pressure_forced_single else configured_maximum
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
            model for (model, _estimated), known in zip(estimates, calibrated, strict=False) if known
        ),
        source=source,
        initial_eviction_models=initial_eviction_models,
        context_floor_models=context_floor_models,
        forced_single=forced_single,
        reuse_larger_context=reuse_larger_context,
        pressure_forced_single=pressure_forced_single,
        compressed_bytes=pressure_snapshot.compressed_bytes,
        swap_used_bytes=pressure_snapshot.swap_used_bytes,
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
    root: Path,
    resource_rows: Callable[
        [], tuple[dict[str, int], dict[str, tuple[int, int]]]
    ],
    digests_for: Callable[[Sequence[str]], dict[str, str]],
    engine_identity: Callable[[], str],
    memory_snapshot_for: Callable[[], MemorySnapshot],
    macos_pressure_snapshot_for: Callable[[], MacOSPressureSnapshot],
) -> ModelResidencyPlan:
    """Probe live host/Ollama state and return a fail-safe residency plan."""

    memory = memory_snapshot_for()
    pressure = (
        macos_pressure_snapshot_for()
        if memory.source.startswith("macos_")
        else MacOSPressureSnapshot(0, 0, "not_probed")
    )
    try:
        installed, resident = resource_rows()
        source = f"{memory.source}+ollama"
    except Exception:
        installed, resident = {}, {}
        source = f"{memory.source}+ollama_unavailable"
    calibrated: dict[tuple[str, int], int] = {}
    if installed:
        try:
            digests = digests_for(models)
            engine = engine_identity()
            calibrated.update(
                _matching_persisted_calibrations(
                    root=root,
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
        pressure=pressure,
    )
    if (
        memory.total_bytes <= 0
        or memory.available_bytes <= 0
        or not installed
        or "identity_unavailable" in source
    ):
        return replace(plan, max_resident_models=0, forced_single=True)
    return plan


def observe_model_runtime(
    model: str,
    *,
    root: Path,
    resource_rows: Callable[
        [], tuple[dict[str, int], dict[str, tuple[int, int]]]
    ],
    digests_for: Callable[[Sequence[str]], dict[str, str]],
    engine_identity: Callable[[], str],
) -> tuple[int, int] | None:
    """Calibrate one exact installed tag and context from Ollama's live runner."""

    try:
        installed, resident = resource_rows()
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
        digest = digests_for([model]).get(model, "")
        engine = engine_identity()
        if not digest:
            return row
        _MODEL_FOOTPRINT_CALIBRATION[(model, context, weight_size, digest, engine)] = (
            size
        )
        _persist_model_calibration(
            root=root,
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
