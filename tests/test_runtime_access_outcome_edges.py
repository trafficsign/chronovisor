from __future__ import annotations

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


def _operations(result: dict) -> set[str]:
    return {row["operation"] for row in result["accesses"]}


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
    assert result["escapes"] == []


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
    assert result["escapes"] == []


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
    assert result["escapes"] == []


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

    assert result["accesses"] == []
    assert result["escapes"] == []


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
    assert result["escapes"] == []


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
    assert result["escapes"] == []


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
    assert result["escapes"] == []


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
        row["reason"] for row in result["escapes"]
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

    assert result["accesses"] == []
    assert result["escapes"] == []


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
    assert result["escapes"] == []


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

    assert result["accesses"] == []
    assert result["escapes"] == []


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
    assert result["escapes"] == []


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
    assert len(result["accesses"]) == 1
    assert result["escapes"] == []


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
    assert {row["reason"] for row in result["escapes"]} == {
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

    assert result["accesses"] == []
    assert {row["reason"] for row in result["escapes"]} == {
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

    assert result["accesses"] == []
    assert "unsupported_registered_origin_expression" in {
        row["reason"] for row in result["escapes"]
    }
