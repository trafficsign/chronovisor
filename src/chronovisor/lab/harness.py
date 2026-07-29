"""Shared lifecycle primitives for reproducible Chronovisor experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from chronovisor.durable_state import read_sealed_json, write_sealed_json

MetricFunction = Callable[[Sequence[Mapping[str, Any]], str], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class LabHarness:
    """Manage the immutable selection and preregistration lifecycle."""

    root: Path
    experiment: str

    @property
    def output_root(self) -> Path:
        """Return the canonical artifact root for this experiment."""
        return self.root / "classification" / self.experiment

    def path(self, name: str) -> Path:
        """Resolve one artifact name below the experiment root."""
        return self.output_root / name

    def seal_selection(
        self, payload: Mapping[str, Any], *, backup: bool = True
    ) -> dict[str, Any]:
        """Seal and re-read the fixed selection artifact."""
        path = self.path("selection.json")
        write_sealed_json(path, dict(payload), backup=backup)
        return read_sealed_json(path)

    def read_selection(self) -> dict[str, Any]:
        """Read the fixed selection artifact."""
        return read_sealed_json(self.path("selection.json"))

    def lock_preregistration(
        self, payload: Mapping[str, Any], *, backup: bool = True
    ) -> dict[str, Any]:
        """Seal and re-read the preregistration contract."""
        path = self.path("preregistration.json")
        write_sealed_json(path, dict(payload), backup=backup)
        return read_sealed_json(path)

    def read_preregistration(self) -> dict[str, Any]:
        """Read the sealed preregistration contract."""
        return read_sealed_json(self.path("preregistration.json"))


def require_contract(
    payload: Mapping[str, Any],
    *,
    schema: str,
    exact: Mapping[str, Any] | None = None,
    error_type: type[Exception] = ValueError,
    message: str = "lab preregistration contract changed",
) -> None:
    """Validate schema and exact preregistered fields without mutation."""
    if payload.get("schema") != schema:
        raise error_type(message)
    for name, expected in (exact or {}).items():
        if payload.get(name) != expected:
            raise error_type(message)


def require_file_hashes(
    expected_hashes: Mapping[Path, str],
    *,
    digest: Callable[[Path], str],
    error_type: type[Exception] = ValueError,
    message: str = "sealed lab input changed",
) -> None:
    """Verify every preregistered input remains byte-identical."""
    for path, expected in expected_hashes.items():
        if not path.is_file() or digest(path) != expected:
            raise error_type(f"{message}: {path}")


def aggregate_channel_metrics(
    cases: Sequence[Mapping[str, Any]],
    channels: Sequence[str],
    metric: MetricFunction,
) -> dict[str, dict[str, Any]]:
    """Apply one deterministic metric function to each named channel."""
    return {channel: dict(metric(cases, channel)) for channel in channels}
