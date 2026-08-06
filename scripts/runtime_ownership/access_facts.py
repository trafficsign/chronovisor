"""Fact collection and stable identity generation for access discovery."""

from __future__ import annotations

from typing import Any

from .access_model import (
    MAX_BINDING_PATHS_PER_RESOURCE,
    FlowValue,
    RawAccess,
    RawEscape,
    _stable_id,
)


class AccessFactCollector:
    def __init__(self, resource_locators: dict[str, str]) -> None:
        self.resource_locators = dict(resource_locators)
        self.raw_accesses: set[RawAccess] = set()
        self.raw_escapes: set[RawEscape] = set()

    def record_access(
        self,
        value: FlowValue,
        *,
        actor: str,
        mode: str,
        operation: str,
        sink: str,
        path: str,
        line: int,
        ordinal: int,
    ) -> None:
        self._record_overflow(
            value,
            actor=actor,
            operation=operation,
            sink=sink,
            path=path,
            line=line,
            ordinal=ordinal,
        )
        for resource_id, chains in value.origins.items():
            for chain in chains:
                logical_actor = actor
                for step in chain:
                    if step.startswith("call:") and "->" in step:
                        logical_actor = step.removeprefix("call:").split("->", 1)[0]
                        break
                self.raw_accesses.add(
                    RawAccess(
                        resource_id=resource_id,
                        actor=logical_actor,
                        sink_actor=actor,
                        mode=mode,
                        operation=operation,
                        sink=sink,
                        binding_chain=chain,
                        path=path,
                        line=line,
                        structural_ordinal=ordinal,
                    )
                )

    def record_escape(
        self,
        value: FlowValue,
        *,
        actor: str,
        operation: str,
        sink: str,
        reason: str,
        path: str,
        line: int,
        ordinal: int,
    ) -> None:
        self._record_overflow(
            value,
            actor=actor,
            operation=operation,
            sink=sink,
            path=path,
            line=line,
            ordinal=ordinal,
        )
        for resource_id, chains in value.origins.items():
            for chain in chains:
                self.raw_escapes.add(
                    RawEscape(
                        resource_id=resource_id,
                        actor=actor,
                        operation=operation,
                        sink=sink,
                        reason=reason,
                        binding_chain=chain,
                        path=path,
                        line=line,
                        structural_ordinal=ordinal,
                    )
                )

    def _record_overflow(
        self,
        value: FlowValue,
        *,
        actor: str,
        operation: str,
        sink: str,
        path: str,
        line: int,
        ordinal: int,
    ) -> None:
        marker = (f"provenance-overflow:limit={MAX_BINDING_PATHS_PER_RESOURCE}",)
        for resource_id in value.overflowed:
            self.raw_escapes.add(
                RawEscape(
                    resource_id=resource_id,
                    actor=actor,
                    operation=operation,
                    sink=sink,
                    reason="provenance_overflow",
                    binding_chain=marker,
                    path=path,
                    line=line,
                    structural_ordinal=ordinal,
                )
            )

    def result(self) -> dict[str, Any]:
        accesses = self._finalize_accesses()
        escapes = self._finalize_escapes()
        return {
            "accesses": accesses,
            "escapes": escapes,
            "access_ids": sorted(row["access_id"] for row in accesses),
            "escape_ids": sorted(row["escape_id"] for row in escapes),
            "counts": {
                "accesses": len(accesses),
                "escapes": len(escapes),
                "read": sum(row["mode"] == "read" for row in accesses),
                "write": sum(row["mode"] == "write" for row in accesses),
                "read_write": sum(row["mode"] == "read_write" for row in accesses),
            },
        }

    def _finalize_accesses(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        grouped: dict[tuple[Any, ...], list[RawAccess]] = {}
        for row in self.raw_accesses:
            key = (
                row.resource_id,
                row.actor,
                row.sink_actor,
                row.mode,
                row.operation,
                row.sink,
                row.binding_chain,
            )
            grouped.setdefault(key, []).append(row)
        for _key, sites in sorted(grouped.items(), key=lambda item: item[0]):
            sites.sort(key=lambda row: (row.path, row.structural_ordinal, row.line))
            for occurrence, row in enumerate(sites, start=1):
                identity = {
                    "resource_id": row.resource_id,
                    "actor": row.actor,
                    "sink_actor": row.sink_actor,
                    "mode": row.mode,
                    "operation": row.operation,
                    "sink": row.sink,
                    "binding_chain": list(row.binding_chain),
                    "occurrence": occurrence,
                }
                rows.append(
                    {
                        "access_id": _stable_id("runtime-access", identity),
                        **identity,
                        "locator": self.resource_locators[row.resource_id],
                        "evidence": {"path": row.path, "line": row.line},
                        "structural_ordinal": row.structural_ordinal,
                    }
                )
        return sorted(rows, key=lambda row: str(row["access_id"]))

    def _finalize_escapes(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        grouped: dict[tuple[Any, ...], list[RawEscape]] = {}
        for row in self.raw_escapes:
            key = (
                row.resource_id,
                row.actor,
                row.operation,
                row.sink,
                row.reason,
                row.binding_chain,
            )
            grouped.setdefault(key, []).append(row)
        for _key, sites in sorted(grouped.items(), key=lambda item: item[0]):
            sites.sort(key=lambda row: (row.path, row.structural_ordinal, row.line))
            for occurrence, row in enumerate(sites, start=1):
                identity = {
                    "resource_id": row.resource_id,
                    "actor": row.actor,
                    "operation": row.operation,
                    "sink": row.sink,
                    "reason": row.reason,
                    "binding_chain": list(row.binding_chain),
                    "occurrence": occurrence,
                }
                rows.append(
                    {
                        "escape_id": _stable_id("runtime-escape", identity),
                        **identity,
                        "locator": self.resource_locators[row.resource_id],
                        "evidence": {"path": row.path, "line": row.line},
                        "structural_ordinal": row.structural_ordinal,
                    }
                )
        return sorted(rows, key=lambda row: str(row["escape_id"]))


__all__ = ["AccessFactCollector"]
