from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

from scripts.runtime_ownership.access import discover_access_facts
from scripts.runtime_ownership.access_facts import AccessFactCollector
from scripts.runtime_ownership.access_model import FlowValue, _collect_syntax_sites


def _discover(
    consumer: str,
    *,
    consumer_path: str = "src/chronovisor/consumer.py",
) -> dict[str, Any]:
    sources = {
        "src/chronovisor/state.py": "STATE_FILE = object()\n",
        consumer_path: consumer,
    }
    candidate = {
        "id": "runtime-resource:state",
        "module": "chronovisor.state",
        "symbol": "STATE_FILE",
        "locator": {"type": "path", "value": "$ROOT/state.json"},
    }
    return discover_access_facts(
        {path: source.encode() for path, source in sources.items()}, [candidate]
    )


def _v2_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "site_ids": sorted(row["site_id"] for row in result["sites"]),
        "provenance_ids": sorted(
            row["provenance_id"] for row in result["provenances"]
        ),
        "access_fact_ids": result["access_fact_ids"],
        "escape_fact_ids": result["escape_fact_ids"],
    }


def test_schema_v2_ids_ignore_line_shift_but_sites_keep_line_evidence() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "def writer():\n"
        "    STATE_FILE.write_text('value')\n"
    )

    base = _discover(source)
    shifted = _discover("\n\n" + source)

    assert base["schema_version"] == 2
    assert base["legacy_identity_version"] == 1
    assert base["provenance_ids"] == sorted(
        row["provenance_id"] for row in base["provenances"]
    )
    assert _v2_identity(shifted) == _v2_identity(base)
    assert base["sites"][0]["evidence"]["line"] + 2 == (
        shifted["sites"][0]["evidence"]["line"]
    )
    assert "lineno=" not in base["sites"][0]["syntax"]
    assert "col_offset=" not in base["sites"][0]["syntax"]


def test_same_module_scope_move_to_package_changes_path_bound_site_id() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "def writer():\n"
        "    STATE_FILE.write_text('value')\n"
    )

    module_file = _discover(
        source, consumer_path="src/chronovisor/foo.py"
    )
    package_file = _discover(
        source, consumer_path="src/chronovisor/foo/__init__.py"
    )

    assert module_file["sites"][0]["scope"] == (
        package_file["sites"][0]["scope"]
    )
    assert module_file["sites"][0]["kind"] == "call"
    assert module_file["sites"][0]["site_id"] != (
        package_file["sites"][0]["site_id"]
    )
    assert module_file["access_fact_ids"] != package_file["access_fact_ids"]
    assert module_file["sites"][0]["evidence"]["path"] == (
        "src/chronovisor/foo.py"
    )
    assert package_file["sites"][0]["evidence"]["path"] == (
        "src/chronovisor/foo/__init__.py"
    )


def test_shared_helper_is_one_physical_fact_with_two_provenances() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def persist(path):\n"
        "    path.write_text('value')\n"
        "def writer_b():\n"
        "    persist(STATE_FILE)\n"
        "def writer_a():\n"
        "    persist(STATE_FILE)\n"
    )

    assert len(result["access_facts"]) == 1
    fact = result["access_facts"][0]
    assert fact["actors"] == [
        "chronovisor.consumer:writer_a",
        "chronovisor.consumer:writer_b",
    ]
    assert len(fact["provenance_ids"]) == 2
    assert fact["sink_actor"] == "chronovisor.consumer:persist"
    assert len(result["accesses"]) == 2


def test_adding_writer_keeps_fact_id_and_adds_top_level_provenance_id() -> None:
    prefix = (
        "from chronovisor.state import STATE_FILE\n"
        "def persist(path):\n"
        "    path.write_text('value')\n"
        "def writer_a():\n"
        "    persist(STATE_FILE)\n"
    )

    base = _discover(prefix)
    expanded = _discover(
        prefix + "def writer_b():\n" + "    persist(STATE_FILE)\n"
    )

    assert expanded["access_fact_ids"] == base["access_fact_ids"]
    assert set(base["provenance_ids"]) < set(expanded["provenance_ids"])
    assert len(expanded["provenance_ids"]) == len(base["provenance_ids"]) + 1
    assert expanded["access_facts"][0]["provenance_ids"] == (
        expanded["provenance_ids"]
    )


def test_identical_caller_calls_have_distinct_same_shape_occurrences() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def persist(path):\n"
        "    path.write_text('value')\n"
        "def writer():\n"
        "    persist(STATE_FILE)\n"
        "    persist(STATE_FILE)\n"
    )

    fact = result["access_facts"][0]
    assert len(fact["provenance_ids"]) == 2
    provenances = {
        row["provenance_id"]: row for row in result["provenances"]
    }
    caller_site_ids = {
        site_id
        for provenance_id in fact["provenance_ids"]
        for site_id in provenances[provenance_id]["call_site_ids"]
    }
    sites = {
        row["site_id"]: row
        for row in result["sites"]
        if row["site_id"] in caller_site_ids
    }
    assert len(sites) == 2
    assert {row["scope"] for row in sites.values()} == {
        "chronovisor.consumer:writer"
    }
    assert {row["occurrence"] for row in sites.values()} == {1, 2}
    assert len({row["syntax"] for row in sites.values()}) == 1


def test_inserting_different_argument_call_does_not_renumber_v2_ids() -> None:
    prefix = (
        "from chronovisor.state import STATE_FILE\n"
        "def persist(path):\n"
        "    path.write_text('value')\n"
        "def writer():\n"
    )
    base = _discover(prefix + "    persist(STATE_FILE)\n")
    inserted = _discover(
        prefix + "    persist(None)\n" + "    persist(STATE_FILE)\n"
    )

    assert inserted["access_fact_ids"] == base["access_fact_ids"]
    assert {
        row["provenance_id"] for row in inserted["provenances"]
    } == {row["provenance_id"] for row in base["provenances"]}
    assert inserted["access_facts"][0]["provenance_ids"] == (
        base["access_facts"][0]["provenance_ids"]
    )


def _collector_result(order: tuple[str, ...]) -> dict[str, Any]:
    path = "src/chronovisor/consumer.py"
    tree = ast.parse("def persist(path):\n    path.write_text('value')\n")
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    expression = function.body[0]
    assert isinstance(expression, ast.Expr)
    call = expression.value
    assert isinstance(call, ast.Call)
    sites = _collect_syntax_sites(
        {"chronovisor.consumer": tree},
        {"chronovisor.consumer": path},
        {id(function): "chronovisor.consumer:persist"},
    )
    collector = AccessFactCollector(
        {"runtime-resource:state": "$ROOT/state.json"}, sites
    )
    for actor in order:
        caller = actor.rsplit(":", 1)[-1]
        value = FlowValue(
            {
                "runtime-resource:state": frozenset(
                    {
                        (
                            "origin:chronovisor.state:STATE_FILE",
                            f"call:{actor}->chronovisor.consumer:persist:path"
                            f"|site=persist:1|site_id={sites[id(call)].site_id}",
                        )
                    }
                )
            }
        )
        collector.record_access(
            value,
            node=call,
            actor="chronovisor.consumer:persist",
            mode="write",
            operation="path.write_text",
            sink="pathlib.Path.write_text",
            path=path,
            line=20 if caller == "writer_a" else 10,
            ordinal=1,
        )
    return collector.result()


def _mixed_overflow_projection(
    *,
    access_overflow: bool,
    escape_overflow: bool,
) -> dict[str, Any]:
    path = "src/chronovisor/consumer.py"
    tree = ast.parse("STATE_FILE.write_text('value')\n")
    expression = tree.body[0]
    assert isinstance(expression, ast.Expr)
    call = expression.value
    assert isinstance(call, ast.Call)
    sites = _collect_syntax_sites(
        {"chronovisor.consumer": tree},
        {"chronovisor.consumer": path},
        {},
    )
    collector = AccessFactCollector(
        {"runtime-resource:state": "$ROOT/state.json"}, sites
    )

    def value(*, overflow: bool) -> FlowValue:
        resource_id = "runtime-resource:state"
        return FlowValue(
            {resource_id: frozenset({("origin:chronovisor.state:STATE_FILE",)})},
            overflowed=frozenset({resource_id}) if overflow else frozenset(),
        )

    collector.record_access(
        value(overflow=access_overflow),
        node=call,
        actor="chronovisor.consumer:<module>",
        mode="write",
        operation="shared.operation",
        sink="shared.sink",
        path=path,
        line=1,
        ordinal=1,
    )
    collector.record_escape(
        value(overflow=escape_overflow),
        node=call,
        actor="chronovisor.consumer:<module>",
        operation="shared.operation",
        sink="shared.sink",
        reason="separate_escape",
        path=path,
        line=1,
        ordinal=1,
    )
    collector.record_escape(
        value(overflow=False),
        node=call,
        actor="chronovisor.consumer:<module>",
        operation="shared.operation",
        sink="shared.sink",
        reason="independent_escape",
        path=path,
        line=1,
        ordinal=1,
    )
    return collector.result()


def test_v2_projection_is_independent_of_raw_record_order_and_lines() -> None:
    forward = _collector_result(
        ("chronovisor.consumer:writer_a", "chronovisor.consumer:writer_b")
    )
    reverse = _collector_result(
        ("chronovisor.consumer:writer_b", "chronovisor.consumer:writer_a")
    )

    assert forward["sites"] == reverse["sites"]
    assert forward["provenances"] == reverse["provenances"]
    assert forward["access_facts"] == reverse["access_facts"]


def test_overflow_completeness_is_scoped_to_owning_physical_fact() -> None:
    access_only = _mixed_overflow_projection(
        access_overflow=True,
        escape_overflow=False,
    )
    escape_only = _mixed_overflow_projection(
        access_overflow=False,
        escape_overflow=True,
    )
    both = _mixed_overflow_projection(
        access_overflow=True,
        escape_overflow=True,
    )

    access_only_normal_escape = next(
        row
        for row in access_only["escape_facts"]
        if row["reason"] == "separate_escape"
    )
    escape_only_normal_escape = next(
        row
        for row in escape_only["escape_facts"]
        if row["reason"] == "separate_escape"
    )
    escape_only_independent = next(
        row
        for row in escape_only["escape_facts"]
        if row["reason"] == "independent_escape"
    )
    assert access_only["access_facts"][0]["provenance_complete"] is False
    assert access_only_normal_escape["provenance_complete"] is True
    assert escape_only["access_facts"][0]["provenance_complete"] is True
    assert escape_only_normal_escape["provenance_complete"] is False
    assert escape_only_independent["provenance_complete"] is True

    overflow_facts = [
        row
        for row in both["escape_facts"]
        if row["reason"] == "provenance_overflow"
    ]
    assert {row["source_kind"] for row in overflow_facts} == {
        "access",
        "escape",
    }
    assert len({row["source_fact_id"] for row in overflow_facts}) == 2
    assert len({row["escape_fact_id"] for row in overflow_facts}) == 2


def test_overflow_v2_is_one_stable_empty_provenance_escape() -> None:
    prefix = (
        "from chronovisor.state import STATE_FILE\n"
        "def persist(path):\n"
        "    path.write_text('value')\n"
    )
    writers = "".join(
        f"def writer_{index:02d}():\n    persist(STATE_FILE)\n"
        for index in range(65)
    )
    changed_writers = writers.replace("writer_64", "writer_99")

    first = _discover(prefix + writers)
    shifted = _discover("\n\n" + prefix + changed_writers)

    assert len(first["access_facts"]) == 1
    assert first["access_facts"][0]["provenance_complete"] is False
    assert len(first["access_facts"][0]["provenance_ids"]) == 64
    assert len(first["escape_facts"]) == 1
    overflow = first["escape_facts"][0]
    assert overflow["reason"] == "provenance_overflow"
    assert overflow["limit"] == 64
    assert overflow["retention_policy"] == "shortest_then_lexicographic"
    assert overflow["provenance_complete"] is False
    assert overflow["provenance_ids"] == []
    assert overflow["actors"] == []
    assert shifted["escape_fact_ids"] == first["escape_fact_ids"]


def test_escape_fact_collapses_shared_site_and_keeps_actor_provenance() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def leak(path):\n"
        "    opaque(path)\n"
        "def writer_a():\n"
        "    leak(STATE_FILE)\n"
        "def writer_b():\n"
        "    leak(STATE_FILE)\n"
    )

    assert len(result["escape_facts"]) == 1
    fact = result["escape_facts"][0]
    assert fact["reason"] == "registered_locator_to_unknown_callee"
    assert fact["actors"] == [
        "chronovisor.consumer:writer_a",
        "chronovisor.consumer:writer_b",
    ]
    assert len(fact["provenance_ids"]) == 2
    assert fact["provenance_complete"] is True
    assert len(result["escapes"]) == 2


def test_legacy_projection_keeps_v1_call_binding_and_identity_fields() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def persist(path):\n"
        "    path.write_text('value')\n"
        "def writer():\n"
        "    persist(STATE_FILE)\n"
    )

    legacy = result["accesses"][0]
    assert result["legacy_identity_version"] == 1
    assert result["access_ids"] == [legacy["access_id"]]
    assert any("|site=persist:1" in step for step in legacy["binding_chain"])
    assert all("|site_id=" not in step for step in legacy["binding_chain"])
    assert set(legacy) == {
        "access_id",
        "resource_id",
        "actor",
        "sink_actor",
        "mode",
        "operation",
        "sink",
        "binding_chain",
        "occurrence",
        "locator",
        "evidence",
        "structural_ordinal",
    }
