"""Explicit schema-v2 joins for runtime access analyzer tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _index(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = str(row[key])
        if identity in indexed:
            raise AssertionError(f"duplicate {key}: {identity}")
        indexed[identity] = row
    return indexed


def _identities(rows: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    return [str(row[key]) for row in rows]


def _validate_result(
    result: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    sites = _index(result["sites"], "site_id")
    provenances = _index(result["provenances"], "provenance_id")
    access_facts = _index(result["access_facts"], "access_fact_id")
    escape_facts = _index(result["escape_facts"], "escape_fact_id")

    provenance_ids = [str(item) for item in result["provenance_ids"]]
    if len(provenance_ids) != len(set(provenance_ids)):
        raise AssertionError("provenance_ids must be unique")
    if provenance_ids != sorted(provenance_ids):
        raise AssertionError("provenance_ids must be sorted")
    if set(provenance_ids) != set(provenances):
        raise AssertionError(
            "provenance_ids must exactly match provenance row identities"
        )

    expected_access_fact_ids = sorted(
        _identities(result["access_facts"], "access_fact_id")
    )
    actual_access_fact_ids = [str(item) for item in result["access_fact_ids"]]
    if actual_access_fact_ids != expected_access_fact_ids:
        raise AssertionError(
            "access_fact_ids must exactly match sorted access fact identities"
        )

    expected_escape_fact_ids = sorted(
        _identities(result["escape_facts"], "escape_fact_id")
    )
    actual_escape_fact_ids = [str(item) for item in result["escape_fact_ids"]]
    if actual_escape_fact_ids != expected_escape_fact_ids:
        raise AssertionError(
            "escape_fact_ids must exactly match sorted escape fact identities"
        )

    referenced_provenance_ids: set[str] = set()
    referenced_site_ids: set[str] = set()
    for fact in [*access_facts.values(), *escape_facts.values()]:
        site_id = str(fact["site_id"])
        if site_id not in sites:
            raise AssertionError(
                f"runtime access fact references unknown site_id: {site_id}"
            )
        referenced_site_ids.add(site_id)

        fact_provenance_ids = [str(item) for item in fact["provenance_ids"]]
        if len(fact_provenance_ids) != len(set(fact_provenance_ids)):
            fact_id_key = (
                "access_fact_id"
                if "access_fact_id" in fact
                else "escape_fact_id"
            )
            raise AssertionError(
                "runtime access fact provenance_ids must be unique: "
                f"{fact[fact_id_key]}"
            )
        for provenance_id in fact_provenance_ids:
            if provenance_id not in provenances:
                raise AssertionError(
                    "runtime access fact references unknown provenance_id: "
                    f"{provenance_id}"
                )
            referenced_provenance_ids.add(provenance_id)

    if referenced_provenance_ids != set(provenances):
        orphaned = sorted(set(provenances) - referenced_provenance_ids)
        raise AssertionError(
            f"orphan top-level provenance identities: {orphaned}"
        )

    for provenance in provenances.values():
        for call_site_id in provenance["call_site_ids"]:
            normalized = str(call_site_id)
            if normalized not in sites:
                raise AssertionError(
                    "runtime provenance references unknown call_site_id: "
                    f"{normalized}"
                )
            referenced_site_ids.add(normalized)

    if referenced_site_ids != set(sites):
        orphaned = sorted(set(sites) - referenced_site_ids)
        raise AssertionError(f"orphan top-level site identities: {orphaned}")

    return sites, provenances


def validate_runtime_access_v2_result(result: Mapping[str, Any]) -> None:
    """Assert the complete relational integrity of a schema-v2 result."""

    _validate_result(result)


def _joined_fact_rows(
    result: Mapping[str, Any], *, fact_key: str
) -> list[dict[str, Any]]:
    sites, provenances = _validate_result(result)
    rows: list[dict[str, Any]] = []
    for fact in result[fact_key]:
        site_id = str(fact["site_id"])
        site = sites[site_id]
        provenance_ids = [str(item) for item in fact["provenance_ids"]]
        if not provenance_ids:
            rows.append({**fact, "evidence": site["evidence"]})
            continue
        for provenance_id in provenance_ids:
            provenance = provenances[provenance_id]
            if provenance["resource_id"] != fact["resource_id"]:
                raise AssertionError(
                    "runtime access fact joined to a different resource"
                )
            rows.append(
                {
                    **fact,
                    "provenance_id": provenance_id,
                    "actor": provenance["actor"],
                    "binding_chain": provenance["binding_chain"],
                    "evidence": site["evidence"],
                }
            )
    return rows


def joined_access_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Join access facts to their syntax-site evidence and provenances."""

    return _joined_fact_rows(result, fact_key="access_facts")


def joined_escape_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Join escape facts to their syntax-site evidence and provenances."""

    return _joined_fact_rows(result, fact_key="escape_facts")
