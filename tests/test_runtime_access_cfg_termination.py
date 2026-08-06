from __future__ import annotations

import pytest

from scripts.runtime_ownership.access import discover_access_facts


def _access_fixture(*, consumer: str) -> dict:
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


def _operations(result: dict) -> list[str]:
    return [row["operation"] for row in result["accesses"]]


def test_return_terminates_block_before_sink() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped():\n"
            "    path = STATE_FILE\n"
            "    return\n"
            "    path.write_text('unreachable')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_returning_origin_branch_does_not_pollute_fallthrough() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped(flag):\n"
            "    path = object()\n"
            "    if flag:\n"
            "        path = STATE_FILE\n"
            "        return\n"
            "    path.write_text('ordinary-path')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_break_carries_origin_to_loop_exit_but_stops_body() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def carried(items):\n"
            "    path = object()\n"
            "    for _item in items:\n"
            "        path = STATE_FILE\n"
            "        break\n"
            "        path.read_text()\n"
            "    path.write_text('reachable-after-break')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_continue_carries_origin_to_later_iteration_but_stops_body() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def carried(items, choose_origin):\n"
            "    path = object()\n"
            "    for _item in items:\n"
            "        if choose_origin:\n"
            "            path = STATE_FILE\n"
            "            continue\n"
            "            path.read_text()\n"
            "        path.write_text('later-iteration')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_nested_return_in_loop_does_not_reach_fallthrough_sink() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped(items, should_return):\n"
            "    path = object()\n"
            "    for _item in items:\n"
            "        if should_return:\n"
            "            path = STATE_FILE\n"
            "            return\n"
            "        path.write_text('ordinary-path')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_raise_terminates_block_before_sink() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def stopped():\n"
            "    path = STATE_FILE\n"
            "    raise RuntimeError('stop')\n"
            "    path.write_text('unreachable')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_finally_runs_for_return_and_post_try_is_unreachable() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def finalized():\n"
            "    path = STATE_FILE\n"
            "    try:\n"
            "        return\n"
            "    finally:\n"
            "        path.write_text('finalized')\n"
            "    path.read_text()\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_terminating_finally_overrides_earlier_return_state() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def finalized():\n"
            "    path = STATE_FILE\n"
            "    try:\n"
            "        return\n"
            "    finally:\n"
            "        path = object()\n"
            "        return\n"
            "    path.write_text('unreachable')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


@pytest.mark.parametrize("control", ["break", "continue"])
def test_finally_runs_for_loop_control_and_control_stops_body(control: str) -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def finalized(items):\n"
            "    for _item in items:\n"
            "        path = STATE_FILE\n"
            "        try:\n"
            f"            {control}\n"
            "        finally:\n"
            "            path.write_text('finalized')\n"
            "        path.read_text()\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_finally_runs_for_raise_and_raise_stops_outer_block() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def finalized():\n"
            "    try:\n"
            "        raise RuntimeError('stop')\n"
            "    finally:\n"
            "        STATE_FILE.exists()\n"
            "    STATE_FILE.write_text('unreachable')\n"
        )
    )

    assert _operations(result) == ["path.exists"]
    assert result["escapes"] == []
