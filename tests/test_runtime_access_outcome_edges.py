from __future__ import annotations

from typing import Any

import pytest

from scripts.runtime_ownership import access_outcomes
from scripts.runtime_ownership.access import discover_access_facts
from scripts.runtime_ownership.access_model import (
    FCNTL_LOCK_MASK_OBJECT_PREFIX,
    FCNTL_UNRESOLVED_LOCK_OPERATION_OBJECT_TYPE,
    OS_FD_OBJECT_TYPE,
    PATH_OBJECT_TYPE,
    UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE,
    FlowValue,
)
from scripts.runtime_ownership.access_outcomes import (
    BlockResult,
    Outcome,
    join_states,
)
from tests.runtime_access_v2_helpers import (
    joined_access_rows,
    joined_escape_rows,
)


def _access_fixture(*, consumer: str) -> dict[str, Any]:
    sources = {
        "src/chronovisor/state.py": "STATE_FILE = object()\n",
        "src/chronovisor/consumer.py": consumer,
    }
    candidate = {
        "id": "runtime-resource:state",
        "module": "chronovisor.state",
        "symbol": "STATE_FILE",
        "locator": {"type": "path", "value": "$ROOT/state.json"},
    }
    return discover_access_facts(
        {path: text.encode() for path, text in sources.items()}, [candidate]
    )


def _operations(result: dict[str, Any]) -> set[str]:
    return {row["operation"] for row in joined_access_rows(result)}


def _marker_flow_cases() -> tuple[tuple[str, FlowValue, str], ...]:
    return (
        (
            "fcntl",
            FlowValue(
                object_types={f"{FCNTL_LOCK_MASK_OBJECT_PREFIX}2"}
            ),
            FCNTL_UNRESOLVED_LOCK_OPERATION_OBJECT_TYPE,
        ),
        (
            "fd",
            FlowValue(object_types={OS_FD_OBJECT_TYPE}),
            UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE,
        ),
        (
            "path",
            FlowValue(
                origins={
                    "runtime-resource:path": frozenset({("origin:path",)})
                },
                object_types={PATH_OBJECT_TYPE},
            ),
            UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE,
        ),
    )


@pytest.mark.parametrize(
    ("_case", "source", "marker"),
    _marker_flow_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_join_states_marks_missing_flow_and_isolates_source_env(
    _case: str,
    source: FlowValue,
    marker: str,
) -> None:
    before = source.copy()

    joined_env, joined_objects = join_states(
        (
            ({"value": source}, {"objects": {"first"}}),
            ({}, {"objects": {"second"}}),
        )
    )

    joined = joined_env["value"]
    assert marker in joined.object_types
    assert joined_objects == {"objects": {"first", "second"}}
    joined.object_types.add("mutated")
    joined.origins["runtime-resource:mutated"] = frozenset({("mutated",)})
    for variant in joined.variants:
        variant.object_types.add("mutated")
    assert source == before


def test_join_states_single_origin_adds_variant_and_retains_originless_flag() -> None:
    source = FlowValue(
        origins={"runtime-resource:path": frozenset({("single",)})},
        object_types={PATH_OBJECT_TYPE},
    )
    joined, _objects = join_states((({"value": source}, {}),))
    assert joined["value"].variants == (source.copy(),)
    assert joined["value"] is not source

    originless = FlowValue(unknown_callable=True)
    joined, _objects = join_states(
        (
            ({"value": source}, {}),
            ({"value": originless}, {}),
        )
    )
    assert joined["value"].has_originless_alternative is True


def test_join_states_only_allocates_bottom_for_missing_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = access_outcomes.FlowValue
    allocations = 0

    def counted_flow_value() -> FlowValue:
        nonlocal allocations
        allocations += 1
        return original()

    monkeypatch.setattr(access_outcomes, "FlowValue", counted_flow_value)
    present = FlowValue(object_types={"present"})

    join_states((({"value": present}, {}), ({"value": present}, {})))
    assert allocations == 0
    join_states((({"value": present}, {}), ({}, {})))
    assert allocations == 1


@pytest.mark.parametrize("companion_kind", ["bottom_return", "normal"])
@pytest.mark.parametrize(
    ("_case", "source", "marker"),
    _marker_flow_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_block_result_reuses_bottom_and_normal_outcomes_for_markers(
    _case: str,
    source: FlowValue,
    marker: str,
    companion_kind: str,
) -> None:
    before = source.copy()
    outcomes = [Outcome("return", {}, {}, source)]
    if companion_kind == "bottom_return":
        outcomes.append(Outcome("return", {}, {}, FlowValue()))
    else:
        outcomes.append(Outcome.normal({}, {}))

    returned = BlockResult(changed=False, outcomes=outcomes).returned
    assert marker in returned.object_types
    assert source == before

    exact = BlockResult(
        changed=False,
        outcomes=[Outcome("return", {}, {}, source)],
    ).returned
    assert marker not in exact.object_types


def test_block_result_normal_outcome_preserves_attribute_ambiguity() -> None:
    source = FlowValue(
        attribute_values={
            "field": FlowValue(
                origins={
                    "runtime-resource:path": frozenset({("attribute",)})
                }
            )
        },
        attribute_values_complete=True,
    )
    returned = BlockResult(
        changed=False,
        outcomes=[
            Outcome("return", {}, {}, source),
            Outcome.normal({}, {}),
        ],
    ).returned

    assert returned.attribute_values_complete is False
    assert returned.attribute_values_ambiguous is True


def test_try_handler_sees_state_before_overwriting_call_assignment() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def risky():\n"
            "    return object()\n"
            "def recover():\n"
            "    path = STATE_FILE\n"
            "    try:\n"
            "        path = risky()\n"
            "    except Exception:\n"
            "        path.write_text('resource-before-call')\n"
        )
    )

    assert _operations(result) == {"path.write_text"}
    assert joined_escape_rows(result) == []


def test_try_handler_sees_state_before_return_expression() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def risky():\n"
            "    return object()\n"
            "def recover():\n"
            "    path = STATE_FILE\n"
            "    try:\n"
            "        return risky()\n"
            "    except Exception:\n"
            "        path.write_text('resource-before-return')\n"
        )
    )

    assert _operations(result) == {"path.write_text"}
    assert joined_escape_rows(result) == []


def test_with_preserves_break_and_stops_the_body() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped(items, manager):\n"
            "    path = object()\n"
            "    for _item in items:\n"
            "        path = STATE_FILE\n"
            "        with manager:\n"
            "            break\n"
            "            path.read_text()\n"
            "    else:\n"
            "        path = object()\n"
            "    path.write_text('break-state')\n"
        )
    )

    assert _operations(result) == {"path.write_text"}
    assert joined_escape_rows(result) == []


def test_with_preserves_continue_and_stops_the_body() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped(items, manager):\n"
            "    path = STATE_FILE\n"
            "    for _item in items:\n"
            "        with manager:\n"
            "            continue\n"
            "            path.read_text()\n"
            "        path.write_text('unreachable-after-continue')\n"
        )
    )

    assert joined_access_rows(result) == []
    assert joined_escape_rows(result) == []


def test_with_preserves_return_through_finally() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped(manager):\n"
            "    path = STATE_FILE\n"
            "    try:\n"
            "        with manager:\n"
            "            return\n"
            "            path.read_text()\n"
            "    finally:\n"
            "        path.write_text('finalized')\n"
            "    path.read_bytes()\n"
        )
    )

    assert _operations(result) == {"path.write_text"}
    assert joined_escape_rows(result) == []


def test_async_with_preserves_return_through_finally() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "async def stopped(manager):\n"
            "    path = STATE_FILE\n"
            "    try:\n"
            "        async with manager:\n"
            "            return\n"
            "            path.read_text()\n"
            "    finally:\n"
            "        path.write_text('finalized')\n"
            "    path.read_bytes()\n"
        )
    )

    assert _operations(result) == {"path.write_text"}
    assert joined_escape_rows(result) == []


def test_with_retains_raise_for_handler_and_stops_the_body() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped(manager):\n"
            "    path = STATE_FILE\n"
            "    try:\n"
            "        with manager:\n"
            "            raise RuntimeError('stop')\n"
            "            path.read_text()\n"
            "    except RuntimeError:\n"
            "        path.write_text('raised')\n"
        )
    )

    assert _operations(result) == {"path.write_text"}
    assert joined_escape_rows(result) == []


def test_with_reports_unknown_suppression_for_tracked_raise() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped():\n"
            "    try:\n"
            "        with STATE_FILE:\n"
            "            raise STATE_FILE\n"
            "    except Exception:\n"
            "        pass\n"
        )
    )

    assert "unknown_context_manager_suppression" in {
        row["reason"] for row in joined_escape_rows(result)
    }


def test_irrefutable_match_return_makes_outer_sink_unreachable() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped(marker):\n"
            "    path = STATE_FILE\n"
            "    match marker:\n"
            "        case _:\n"
            "            return\n"
            "            path.read_text()\n"
            "    path.write_text('unreachable')\n"
        )
    )

    assert joined_access_rows(result) == []
    assert joined_escape_rows(result) == []


def test_irrefutable_match_preserves_break_and_skips_loop_else() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped(items, marker):\n"
            "    path = object()\n"
            "    for _item in items:\n"
            "        path = STATE_FILE\n"
            "        match marker:\n"
            "            case _:\n"
            "                break\n"
            "    else:\n"
            "        path = object()\n"
            "    path.write_text('break-state')\n"
        )
    )

    assert _operations(result) == {"path.write_text"}
    assert joined_escape_rows(result) == []


def test_irrefutable_match_preserves_continue_and_stops_following_code() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped(items, marker):\n"
            "    path = STATE_FILE\n"
            "    for _item in items:\n"
            "        match marker:\n"
            "            case _:\n"
            "                continue\n"
            "                path.read_text()\n"
            "        path.write_text('unreachable-after-continue')\n"
        )
    )

    assert joined_access_rows(result) == []
    assert joined_escape_rows(result) == []


def test_irrefutable_match_retains_raise_for_handler() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped(marker):\n"
            "    path = STATE_FILE\n"
            "    try:\n"
            "        match marker:\n"
            "            case _:\n"
            "                raise RuntimeError('stop')\n"
            "                path.read_text()\n"
            "    except RuntimeError:\n"
            "        path.write_text('raised')\n"
        )
    )

    assert _operations(result) == {"path.write_text"}
    assert joined_escape_rows(result) == []


def test_augassign_evaluates_rhs_sink_once() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def updated():\n"
            "    value = 0\n"
            "    value += STATE_FILE.read_text()\n"
        )
    )

    assert _operations(result) == {"path.read_text"}
    assert len(joined_access_rows(result)) == 1
    assert joined_escape_rows(result) == []


def test_augassign_preserves_origin_and_reports_unsupported_transform() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def updated(ordinary):\n"
            "    path = STATE_FILE\n"
            "    path += ordinary\n"
            "    path.write_text('preserved')\n"
        )
    )

    assert _operations(result) == {"path.write_text"}
    assert {row["reason"] for row in joined_escape_rows(result)} == {
        "unsupported_registered_origin_augassign"
    }


def test_augassign_attribute_origin_is_an_explicit_escape() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "class Holder:\n"
            "    def __init__(self):\n"
            "        self.path = STATE_FILE\n"
            "    def updated(self):\n"
            "        self.path += object()\n"
        )
    )

    assert joined_access_rows(result) == []
    assert {row["reason"] for row in joined_escape_rows(result)} == {
        "unsupported_registered_origin_augassign"
    }


def test_augassign_subscript_origin_is_an_explicit_escape() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def updated():\n"
            "    path = STATE_FILE\n"
            "    path[0] += object()\n"
        )
    )

    assert joined_access_rows(result) == []
    assert "unsupported_registered_origin_expression" in {
        row["reason"] for row in joined_escape_rows(result)
    }
