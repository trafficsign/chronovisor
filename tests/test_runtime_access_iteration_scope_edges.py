from __future__ import annotations

from typing import Any

import pytest

from scripts.runtime_ownership.access import discover_access_facts
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


def _operations(result: dict[str, Any]) -> list[str]:
    return [row["operation"] for row in joined_access_rows(result)]


def test_literal_for_binds_the_exact_item_value() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def iterated():\n"
            "    for path in [STATE_FILE]:\n"
            "        path.write_text('resource')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_escape_rows(result) == []


def test_literal_for_tuple_and_list_targets_bind_exact_positions() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def iterated():\n"
            "    for resource, ordinary in [(STATE_FILE, object())]:\n"
            "        resource.read_text()\n"
            "        ordinary.read_text()\n"
            "    for [ordinary, resource] in [[object(), STATE_FILE]]:\n"
            "        ordinary.write_text('ordinary')\n"
            "        resource.write_text('resource')\n"
        )
    )

    assert set(_operations(result)) == {"path.read_text", "path.write_text"}
    assert len(joined_access_rows(result)) == 2
    assert joined_escape_rows(result) == []


def test_literal_for_starred_target_binds_exact_positions() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def iterated():\n"
            "    for ordinary, *middle, resource in [\n"
            "        (object(), object(), STATE_FILE)\n"
            "    ]:\n"
            "        ordinary.read_text()\n"
            "        middle.read_text()\n"
            "        resource.write_text('resource')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_escape_rows(result) == []


def test_async_for_unknown_registered_iterable_binds_and_escapes() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "async def iterated(values=STATE_FILE):\n"
            "    async for path in values:\n"
            "        path.write_text('conservative-resource')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert {row["reason"] for row in joined_escape_rows(result)} == {
        "unsupported_registered_origin_control_flow"
    }
    assert {row["sink"] for row in joined_escape_rows(result)} == {"python.async_for"}


def test_empty_literal_for_skips_body_and_runs_else() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def iterated():\n"
            "    for _item in []:\n"
            "        STATE_FILE.write_text('unreachable-body')\n"
            "    else:\n"
            "        STATE_FILE.read_text()\n"
        )
    )

    assert _operations(result) == ["path.read_text"]
    assert joined_escape_rows(result) == []


def test_nonempty_literal_for_break_skips_else_and_carries_item() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def iterated():\n"
            "    for path in [STATE_FILE]:\n"
            "        break\n"
            "    else:\n"
            "        STATE_FILE.read_text()\n"
            "    path.write_text('resource-after-break')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_escape_rows(result) == []


def test_nonempty_literal_for_exhaustion_runs_else() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def iterated():\n"
            "    for _item in [object()]:\n"
            "        pass\n"
            "    else:\n"
            "        STATE_FILE.exists()\n"
        )
    )

    assert _operations(result) == ["path.exists"]
    assert joined_escape_rows(result) == []


@pytest.mark.parametrize(
    "assignment",
    [
        "left, right = (STATE_FILE,)",
        "left, = (STATE_FILE, object())",
        "left, *middle, right = (STATE_FILE,)",
        "left, (nested_left, nested_right) = (STATE_FILE, (object(),))",
    ],
    ids=["too-few", "too-many", "starred-minimum", "nested"],
)
def test_exact_literal_unpack_mismatch_raises_without_continuation(
    assignment: str,
) -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def mismatched():\n"
            "    left = object()\n"
            "    try:\n"
            f"        {assignment}\n"
            "        STATE_FILE.read_text()\n"
            "    except ValueError:\n"
            "        left.write_text('must-remain-ordinary')\n"
            "        STATE_FILE.exists()\n"
        )
    )

    assert _operations(result) == ["path.exists"]
    assert joined_escape_rows(result) == []


def test_literal_for_target_mismatch_raises_before_body_else_and_fallthrough() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def mismatched():\n"
            "    for left, right in [(STATE_FILE,)]:\n"
            "        STATE_FILE.read_text()\n"
            "    else:\n"
            "        STATE_FILE.exists()\n"
            "    STATE_FILE.write_text('unreachable')\n"
        )
    )

    assert joined_access_rows(result) == []
    assert joined_escape_rows(result) == []


def test_comprehension_walrus_binds_containing_function_scope() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def scoped():\n"
            "    [(path := STATE_FILE) for _item in [object()]]\n"
            "    path.write_text('resource')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_escape_rows(result) == []


def test_comprehension_walrus_binds_containing_module_scope() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "[(path := STATE_FILE) for _item in [object()]]\n"
            "path.write_text('resource')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_escape_rows(result) == []


def test_async_comprehension_walrus_binds_containing_scope() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "async def scoped(items):\n"
            "    [(path := STATE_FILE) async for _item in items]\n"
            "    path.write_text('possible-resource')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_escape_rows(result) == []


def test_comprehension_walrus_filter_binds_containing_scope() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def scoped():\n"
            "    [item for item in [object()] if (path := STATE_FILE)]\n"
            "    path.write_text('resource')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert {row["reason"] for row in joined_escape_rows(result)} == {
        "unsupported_registered_origin_control_flow"
    }
    assert {row["sink"] for row in joined_escape_rows(result)} == {"python.comprehension_if"}


def test_optional_comprehension_walrus_joins_prior_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def scoped(items):\n"
            "    path = STATE_FILE\n"
            "    [(path := object()) for _item in items]\n"
            "    path.write_text('possible-prior-resource')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_escape_rows(result) == []


def test_nested_comprehension_iteration_targets_remain_isolated() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def scoped():\n"
            "    item = STATE_FILE\n"
            "    [[item.write_text('ordinary-inner') for item in [object()]]\n"
            "     for item in [object()]]\n"
            "    item.read_text()\n"
        )
    )

    assert _operations(result) == ["path.read_text"]
    assert joined_escape_rows(result) == []
