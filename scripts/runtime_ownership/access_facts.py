"""Fact collection and stable identity generation for access discovery."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .access_model import (
    MAX_BINDING_PATHS_PER_RESOURCE,
    FlowValue,
    RawAccess,
    RawEscape,
    SyntaxSite,
    _stable_id,
)

_RETENTION_POLICY = "shortest_then_lexicographic"
_SITE_ID_PATTERN = re.compile(r"\|site_id=(runtime-site:[0-9a-f]{64})")
_V1_SITE_PATTERN = re.compile(r"\|site=(?!id=)[^|]*")


def _logical_actor(actor: str, chain: tuple[str, ...]) -> str:
    for step in chain:
        if step.startswith("call:") and "->" in step:
            return step.removeprefix("call:").split("->", 1)[0]
    return actor


def _legacy_binding_chain(chain: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_SITE_ID_PATTERN.sub("", step) for step in chain)


def _v2_binding_chain(chain: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_V1_SITE_PATTERN.sub("", step) for step in chain)


def _call_site_ids(chain: tuple[str, ...]) -> list[str]:
    return sorted(
        {
            match.group(1)
            for step in chain
            for match in _SITE_ID_PATTERN.finditer(step)
        }
    )


def _access_fact_identity(
    *,
    site_id: str,
    resource_id: str,
    mode: str,
    operation: str,
    sink: str,
    sink_actor: str,
) -> dict[str, str]:
    return {
        "site_id": site_id,
        "resource_id": resource_id,
        "mode": mode,
        "operation": operation,
        "sink": sink,
        "sink_actor": sink_actor,
    }


def _escape_fact_identity(
    *,
    site_id: str,
    resource_id: str,
    operation: str,
    sink: str,
    reason: str,
) -> dict[str, str]:
    return {
        "site_id": site_id,
        "resource_id": resource_id,
        "operation": operation,
        "sink": sink,
        "reason": reason,
    }


@dataclass(frozen=True)
class _RawOverflow:
    source_kind: Literal["access", "escape"]
    source_fact_id: str
    site_id: str
    resource_id: str
    operation: str
    sink: str
    mode: str | None
    sink_actor: str | None
    source_reason: str | None


class AccessFactCollector:
    def __init__(
        self,
        resource_locators: Mapping[str, str],
        syntax_sites: Mapping[int, SyntaxSite],
    ) -> None:
        self.resource_locators = dict(resource_locators)
        self.syntax_sites = dict(syntax_sites)
        self.sites_by_id = {site.site_id: site for site in syntax_sites.values()}
        if len(self.sites_by_id) != len(syntax_sites):
            raise ValueError("runtime syntax site identities must be unique")
        self.raw_accesses: set[RawAccess] = set()
        self.raw_escapes: set[RawEscape] = set()
        self.raw_overflows: set[_RawOverflow] = set()

    def site_id(self, node: ast.AST) -> str:
        site = self.syntax_sites.get(id(node))
        if site is None:
            raise ValueError(
                "missing executable syntax site for "
                f"{type(node).__name__} at line {getattr(node, 'lineno', 0)}"
            )
        return site.site_id

    def record_access(
        self,
        value: FlowValue,
        *,
        node: ast.AST,
        actor: str,
        mode: str,
        operation: str,
        sink: str,
        path: str,
        line: int,
        ordinal: int,
    ) -> None:
        site_id = self.site_id(node)
        self._record_overflow(
            value,
            actor=actor,
            operation=operation,
            sink=sink,
            path=path,
            line=line,
            ordinal=ordinal,
            site_id=site_id,
            source_kind="access",
            mode=mode,
            sink_actor=actor,
            source_reason=None,
        )
        for resource_id, chains in value.origins.items():
            for chain in chains:
                self.raw_accesses.add(
                    RawAccess(
                        resource_id=resource_id,
                        actor=_logical_actor(actor, chain),
                        sink_actor=actor,
                        mode=mode,
                        operation=operation,
                        sink=sink,
                        binding_chain=chain,
                        path=path,
                        line=line,
                        structural_ordinal=ordinal,
                        site_id=site_id,
                    )
                )

    def record_escape(
        self,
        value: FlowValue,
        *,
        node: ast.AST,
        actor: str,
        operation: str,
        sink: str,
        reason: str,
        path: str,
        line: int,
        ordinal: int,
    ) -> None:
        site_id = self.site_id(node)
        self._record_overflow(
            value,
            actor=actor,
            operation=operation,
            sink=sink,
            path=path,
            line=line,
            ordinal=ordinal,
            site_id=site_id,
            source_kind="escape",
            mode=None,
            sink_actor=None,
            source_reason=reason,
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
                        site_id=site_id,
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
        site_id: str,
        source_kind: Literal["access", "escape"],
        mode: str | None,
        sink_actor: str | None,
        source_reason: str | None,
    ) -> None:
        marker = (f"provenance-overflow:limit={MAX_BINDING_PATHS_PER_RESOURCE}",)
        for resource_id in value.overflowed:
            if source_kind == "access":
                if mode is None or sink_actor is None:
                    raise ValueError("access overflow requires mode and sink actor")
                source_identity = _access_fact_identity(
                    site_id=site_id,
                    resource_id=resource_id,
                    mode=mode,
                    operation=operation,
                    sink=sink,
                    sink_actor=sink_actor,
                )
                source_fact_id = _stable_id(
                    "runtime-access-fact", source_identity
                )
            else:
                if source_reason is None:
                    raise ValueError("escape overflow requires its source reason")
                source_identity = _escape_fact_identity(
                    site_id=site_id,
                    resource_id=resource_id,
                    operation=operation,
                    sink=sink,
                    reason=source_reason,
                )
                source_fact_id = _stable_id(
                    "runtime-escape-fact", source_identity
                )
            self.raw_overflows.add(
                _RawOverflow(
                    source_kind=source_kind,
                    source_fact_id=source_fact_id,
                    site_id=site_id,
                    resource_id=resource_id,
                    operation=operation,
                    sink=sink,
                    mode=mode,
                    sink_actor=sink_actor,
                    source_reason=source_reason,
                )
            )
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
                    site_id=site_id,
                )
            )

    def result(self) -> dict[str, Any]:
        accesses = self._finalize_accesses()
        escapes = self._finalize_escapes()
        access_facts, access_provenances = self._finalize_v2_accesses()
        escape_facts, escape_provenances = self._finalize_v2_escapes()
        provenances = {**access_provenances, **escape_provenances}
        referenced_site_ids = {
            str(row["site_id"]) for row in [*access_facts, *escape_facts]
        }
        for row in provenances.values():
            referenced_site_ids.update(str(item) for item in row["call_site_ids"])
        return {
            "schema_version": 2,
            "legacy_identity_version": 1,
            "sites": self._finalize_sites(referenced_site_ids),
            "provenances": sorted(
                provenances.values(), key=lambda row: str(row["provenance_id"])
            ),
            "provenance_ids": sorted(provenances),
            "access_facts": access_facts,
            "escape_facts": escape_facts,
            "access_fact_ids": sorted(
                str(row["access_fact_id"]) for row in access_facts
            ),
            "escape_fact_ids": sorted(
                str(row["escape_fact_id"]) for row in escape_facts
            ),
            "accesses": accesses,
            "escapes": escapes,
            "access_ids": sorted(str(row["access_id"]) for row in accesses),
            "escape_ids": sorted(str(row["escape_id"]) for row in escapes),
            "counts": {
                "accesses": len(accesses),
                "escapes": len(escapes),
                "read": sum(row["mode"] == "read" for row in accesses),
                "write": sum(row["mode"] == "write" for row in accesses),
                "read_write": sum(
                    row["mode"] == "read_write" for row in accesses
                ),
            },
        }

    def _finalize_sites(self, site_ids: set[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for site_id in sorted(site_ids):
            site = self.sites_by_id.get(site_id)
            if site is None:
                raise ValueError(f"unknown runtime syntax site identity: {site_id}")
            rows.append(
                {
                    "site_id": site.site_id,
                    "scope": site.scope,
                    "kind": site.kind,
                    "syntax": site.syntax,
                    "occurrence": site.occurrence,
                    "evidence": {"path": site.path, "line": site.line},
                }
            )
        return rows

    def _provenance_row(
        self,
        *,
        resource_id: str,
        actor: str,
        chain: tuple[str, ...],
    ) -> dict[str, Any]:
        binding_chain = _v2_binding_chain(chain)
        identity = {
            "resource_id": resource_id,
            "actor": _logical_actor(actor, binding_chain),
            "binding_chain": list(binding_chain),
        }
        return {
            "provenance_id": _stable_id("runtime-provenance", identity),
            **identity,
            "locator": self.resource_locators[resource_id],
            "call_site_ids": _call_site_ids(binding_chain),
        }

    @staticmethod
    def _select_provenances(
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        distinct = {str(row["provenance_id"]): row for row in rows}
        retained = sorted(
            distinct.values(),
            key=lambda row: (
                len(row["binding_chain"]),
                json.dumps(row["binding_chain"], sort_keys=True),
                str(row["actor"]),
                str(row["provenance_id"]),
            ),
        )[:MAX_BINDING_PATHS_PER_RESOURCE]
        return sorted(retained, key=lambda row: str(row["provenance_id"]))

    def _finalize_v2_accesses(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        grouped: dict[tuple[str, ...], list[RawAccess]] = {}
        for row in self.raw_accesses:
            key = (
                row.site_id,
                row.resource_id,
                row.mode,
                row.operation,
                row.sink,
                row.sink_actor,
            )
            grouped.setdefault(key, []).append(row)
        overflowed_source_ids = {
            row.source_fact_id
            for row in self.raw_overflows
            if row.source_kind == "access"
        }
        facts: list[dict[str, Any]] = []
        provenances: dict[str, dict[str, Any]] = {}
        for group_key, raw_rows in sorted(grouped.items()):
            site_id, resource_id, mode, operation, sink, sink_actor = group_key
            candidates = [
                self._provenance_row(
                    resource_id=row.resource_id,
                    actor=row.actor,
                    chain=row.binding_chain,
                )
                for row in raw_rows
            ]
            distinct_candidate_count = len(
                {str(row["provenance_id"]) for row in candidates}
            )
            selected = self._select_provenances(candidates)
            provenances.update(
                {str(row["provenance_id"]): row for row in selected}
            )
            identity = _access_fact_identity(
                site_id=site_id,
                resource_id=resource_id,
                mode=mode,
                operation=operation,
                sink=sink,
                sink_actor=sink_actor,
            )
            access_fact_id = _stable_id("runtime-access-fact", identity)
            facts.append(
                {
                    "access_fact_id": access_fact_id,
                    **identity,
                    "locator": self.resource_locators[resource_id],
                    "provenance_ids": sorted(
                        str(row["provenance_id"]) for row in selected
                    ),
                    "actors": sorted({str(row["actor"]) for row in selected}),
                    "provenance_complete": (
                        access_fact_id not in overflowed_source_ids
                        and distinct_candidate_count
                        <= MAX_BINDING_PATHS_PER_RESOURCE
                    ),
                }
            )
        return (
            sorted(facts, key=lambda row: str(row["access_fact_id"])),
            provenances,
        )

    def _finalize_v2_escapes(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        grouped: dict[tuple[str, ...], list[RawEscape]] = {}
        for row in self.raw_escapes:
            if row.reason == "provenance_overflow":
                continue
            key = (
                row.site_id,
                row.resource_id,
                row.operation,
                row.sink,
                row.reason,
            )
            grouped.setdefault(key, []).append(row)
        overflowed_source_ids = {
            row.source_fact_id
            for row in self.raw_overflows
            if row.source_kind == "escape"
        }
        facts = self._finalize_v2_overflows()
        provenances: dict[str, dict[str, Any]] = {}
        for group_key, raw_rows in sorted(grouped.items()):
            site_id, resource_id, operation, sink, reason = group_key
            candidates = [
                self._provenance_row(
                    resource_id=row.resource_id,
                    actor=row.actor,
                    chain=row.binding_chain,
                )
                for row in raw_rows
            ]
            selected = self._select_provenances(candidates)
            provenances.update(
                {str(row["provenance_id"]): row for row in selected}
            )
            identity = _escape_fact_identity(
                site_id=site_id,
                resource_id=resource_id,
                operation=operation,
                sink=sink,
                reason=reason,
            )
            escape_fact_id = _stable_id("runtime-escape-fact", identity)
            facts.append(
                {
                    "escape_fact_id": escape_fact_id,
                    **identity,
                    "locator": self.resource_locators[resource_id],
                    "provenance_ids": sorted(
                        str(row["provenance_id"]) for row in selected
                    ),
                    "actors": sorted({str(row["actor"]) for row in selected}),
                    "provenance_complete": (
                        escape_fact_id not in overflowed_source_ids
                        and len(
                            {str(row["provenance_id"]) for row in candidates}
                        )
                        <= MAX_BINDING_PATHS_PER_RESOURCE
                    ),
                }
            )
        return (
            sorted(facts, key=lambda row: str(row["escape_fact_id"])),
            provenances,
        )

    def _finalize_v2_overflows(self) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        for row in sorted(
            self.raw_overflows,
            key=lambda item: (
                item.source_kind,
                item.source_fact_id,
            ),
        ):
            identity: dict[str, Any] = {
                "site_id": row.site_id,
                "resource_id": row.resource_id,
                "operation": row.operation,
                "sink": row.sink,
                "reason": "provenance_overflow",
                "source_kind": row.source_kind,
                "source_fact_id": row.source_fact_id,
                "limit": MAX_BINDING_PATHS_PER_RESOURCE,
                "retention_policy": _RETENTION_POLICY,
            }
            if row.source_kind == "access":
                identity.update(
                    {
                        "mode": row.mode,
                        "sink_actor": row.sink_actor,
                    }
                )
            else:
                identity["source_reason"] = row.source_reason
            facts.append(
                {
                    "escape_fact_id": _stable_id(
                        "runtime-escape-fact", identity
                    ),
                    **identity,
                    "locator": self.resource_locators[row.resource_id],
                    "provenance_ids": [],
                    "actors": [],
                    "provenance_complete": False,
                }
            )
        return facts

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
                _legacy_binding_chain(row.binding_chain),
            )
            grouped.setdefault(key, []).append(row)
        for key, sites in sorted(grouped.items(), key=lambda item: item[0]):
            sites.sort(key=lambda row: (row.path, row.structural_ordinal, row.line))
            binding_chain = key[-1]
            for occurrence, row in enumerate(sites, start=1):
                identity = {
                    "resource_id": row.resource_id,
                    "actor": row.actor,
                    "sink_actor": row.sink_actor,
                    "mode": row.mode,
                    "operation": row.operation,
                    "sink": row.sink,
                    "binding_chain": list(binding_chain),
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
                _legacy_binding_chain(row.binding_chain),
            )
            grouped.setdefault(key, []).append(row)
        for key, sites in sorted(grouped.items(), key=lambda item: item[0]):
            sites.sort(key=lambda row: (row.path, row.structural_ordinal, row.line))
            binding_chain = key[-1]
            for occurrence, row in enumerate(sites, start=1):
                identity = {
                    "resource_id": row.resource_id,
                    "actor": row.actor,
                    "operation": row.operation,
                    "sink": row.sink,
                    "reason": row.reason,
                    "binding_chain": list(binding_chain),
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
