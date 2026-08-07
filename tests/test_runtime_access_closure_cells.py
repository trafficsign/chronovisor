from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scripts.runtime_ownership.access import discover_access_facts


def _candidate(
    module: str,
    symbol: str,
    *,
    resource_id: str = "runtime-resource:state",
) -> dict[str, Any]:
    return {
        "id": resource_id,
        "module": module,
        "symbol": symbol,
        "locator": {"type": "path", "value": f"$ROOT/{symbol.lower()}.json"},
    }


def _discover(
    consumer: str,
    *,
    extra_sources: Mapping[str, str] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    sources = {
        "src/chronovisor/state.py": "STATE_FILE = object()\n",
        "src/chronovisor/consumer.py": consumer,
        **dict(extra_sources or {}),
    }
    selected = (
        list(candidates)
        if candidates is not None
        else [_candidate("chronovisor.state", "STATE_FILE")]
    )
    return discover_access_facts(
        {path: source.encode() for path, source in sources.items()}, selected
    )


def _assert_no_facts(result: Mapping[str, Any]) -> None:
    assert result["accesses"] == []
    assert result["escapes"] == []


def test_closure_late_binding_observes_kill_before_reader_call() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def outer():\n"
        "    path = STATE_FILE\n"
        "    def reader():\n"
        "        path.read_text()\n"
        "    path = None\n"
        "    reader()\n"
        "outer()\n"
    )

    _assert_no_facts(result)


def test_nonlocal_setter_before_reader_exposes_origin() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def outer():\n"
        "    path = None\n"
        "    def setter(value):\n"
        "        nonlocal path\n"
        "        path = value\n"
        "    def reader():\n"
        "        path.read_text()\n"
        "    setter(STATE_FILE)\n"
        "    reader()\n"
        "outer()\n"
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.read_text"
    assert result["escapes"] == []


def test_reader_before_nonlocal_setter_does_not_observe_future_origin() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def outer():\n"
        "    path = None\n"
        "    def setter(value):\n"
        "        nonlocal path\n"
        "        path = value\n"
        "    def reader():\n"
        "        path.read_text()\n"
        "    reader()\n"
        "    setter(STATE_FILE)\n"
        "outer()\n"
    )

    _assert_no_facts(result)


def test_mutually_exclusive_setter_and_reader_do_not_cross_branches() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def outer(flag):\n"
        "    path = None\n"
        "    def setter():\n"
        "        nonlocal path\n"
        "        path = STATE_FILE\n"
        "    def reader():\n"
        "        path.read_text()\n"
        "    if flag:\n"
        "        setter()\n"
        "    else:\n"
        "        reader()\n"
        "outer(object())\n"
    )

    _assert_no_facts(result)


def test_optional_setter_can_reach_unconditional_reader() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def outer(flag):\n"
        "    path = None\n"
        "    def setter():\n"
        "        nonlocal path\n"
        "        path = STATE_FILE\n"
        "    def reader():\n"
        "        path.read_text()\n"
        "    if flag:\n"
        "        setter()\n"
        "    reader()\n"
        "outer(object())\n"
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.read_text"
    assert result["escapes"] == []


def test_never_called_nested_reader_and_setter_produce_no_facts() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def outer():\n"
        "    path = STATE_FILE\n"
        "    def reader():\n"
        "        path.read_text()\n"
        "    def setter():\n"
        "        nonlocal path\n"
        "        path = None\n"
        "    return None\n"
        "outer()\n"
    )

    _assert_no_facts(result)


def test_global_reader_before_setter_does_not_observe_future_origin() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "current = None\n"
        "def reader():\n"
        "    current.read_text()\n"
        "def setter(value):\n"
        "    global current\n"
        "    current = value\n"
        "reader()\n"
        "setter(STATE_FILE)\n"
    )

    _assert_no_facts(result)


def test_global_setter_before_reader_exposes_origin() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "current = None\n"
        "def reader():\n"
        "    current.read_text()\n"
        "def setter(value):\n"
        "    global current\n"
        "    current = value\n"
        "setter(STATE_FILE)\n"
        "reader()\n"
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.read_text"
    assert result["escapes"] == []


def test_cross_module_reader_before_setter_does_not_observe_future_origin() -> None:
    result = _discover(
        "from chronovisor.provider import reader, setter\n"
        "from chronovisor.state import STATE_FILE\n"
        "reader()\n"
        "setter(STATE_FILE)\n",
        extra_sources={
            "src/chronovisor/provider.py": (
                "current = None\n"
                "def reader():\n"
                "    current.read_text()\n"
                "def setter(value):\n"
                "    global current\n"
                "    current = value\n"
            )
        },
    )

    _assert_no_facts(result)


def test_cross_module_setter_before_reader_exposes_origin() -> None:
    result = _discover(
        "from chronovisor.provider import reader, setter\n"
        "from chronovisor.state import STATE_FILE\n"
        "setter(STATE_FILE)\n"
        "reader()\n",
        extra_sources={
            "src/chronovisor/provider.py": (
                "current = None\n"
                "def reader():\n"
                "    current.read_text()\n"
                "def setter(value):\n"
                "    global current\n"
                "    current = value\n"
            )
        },
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.read_text"
    assert result["escapes"] == []


def test_imprecise_later_call_requires_summary_after_precise_local_call() -> None:
    result = _discover(
        "from chronovisor.resources import FIRST_FILE, SECOND_FILE\n"
        "def read(path):\n"
        "    path.read_text()\n"
        "def ignore(path):\n"
        "    return None\n"
        "read(FIRST_FILE)\n"
        "flag = False\n"
        "fn = read if flag else ignore\n"
        "fn(SECOND_FILE)\n",
        extra_sources={
            "src/chronovisor/resources.py": (
                "FIRST_FILE = object()\nSECOND_FILE = object()\n"
            )
        },
        candidates=[
            _candidate(
                "chronovisor.resources",
                "FIRST_FILE",
                resource_id="runtime-resource:first",
            ),
            _candidate(
                "chronovisor.resources",
                "SECOND_FILE",
                resource_id="runtime-resource:second",
            ),
        ],
    )

    assert {row["resource_id"] for row in result["accesses"]} == {
        "runtime-resource:first",
        "runtime-resource:second",
    }
    assert {row["operation"] for row in result["accesses"]} == {"path.read_text"}
    assert result["escapes"] == []


def test_recursive_depth_40_reaches_base_sink_without_cap() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def descend(path, depth):\n"
        "    if depth:\n"
        "        return descend(path, depth - 1)\n"
        "    path.write_text('value')\n"
        "descend(STATE_FILE, 40)\n"
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.write_text"
    assert result["escapes"] == []


def test_recursive_if_expression_base_sink_has_no_binding_cycle_escape() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def descend(path, depth):\n"
        "    return (\n"
        "        descend(path, depth - 1)\n"
        "        if depth\n"
        "        else path.write_text('value')\n"
        "    )\n"
        "descend(STATE_FILE, 40)\n"
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.write_text"
    assert result["escapes"] == []


def test_alias_only_unconditional_recursion_remains_binding_cycle_escape() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def helper(path):\n"
        "    alias = helper\n"
        "    path.write_text('value')\n"
        "    alias(path)\n"
        "helper(STATE_FILE)\n"
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.write_text"
    assert len(result["escapes"]) == 1
    assert result["escapes"][0]["reason"] == "binding_cycle"


def test_returned_closures_remain_activation_scoped() -> None:
    result = _discover(
        "from chronovisor.resources import FIRST_FILE, SECOND_FILE\n"
        "def outer(path):\n"
        "    def reader():\n"
        "        path.read_text()\n"
        "    return reader\n"
        "first_reader = outer(FIRST_FILE)\n"
        "second_reader = outer(SECOND_FILE)\n"
        "first_reader()\n",
        extra_sources={
            "src/chronovisor/resources.py": (
                "FIRST_FILE = object()\nSECOND_FILE = object()\n"
            )
        },
        candidates=[
            _candidate(
                "chronovisor.resources",
                "FIRST_FILE",
                resource_id="runtime-resource:first",
            ),
            _candidate(
                "chronovisor.resources",
                "SECOND_FILE",
                resource_id="runtime-resource:second",
            ),
        ],
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["resource_id"] == "runtime-resource:first"
    assert result["accesses"][0]["operation"] == "path.read_text"
    assert result["escapes"] == []


def test_returned_closure_observes_kill_after_definition() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def outer():\n"
        "    path = STATE_FILE\n"
        "    def reader():\n"
        "        path.read_text()\n"
        "    path = None\n"
        "    return reader\n"
        "reader = outer()\n"
        "reader()\n"
    )

    _assert_no_facts(result)


def test_returned_nonlocal_setter_updates_sibling_reader_cell() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def outer():\n"
        "    path = None\n"
        "    def setter(value):\n"
        "        nonlocal path\n"
        "        path = value\n"
        "    def reader():\n"
        "        path.read_text()\n"
        "    return setter, reader\n"
        "setter, reader = outer()\n"
        "setter(STATE_FILE)\n"
        "reader()\n"
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.read_text"
    assert result["escapes"] == []


def test_returned_reader_before_nonlocal_setter_has_no_future_cell_state() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def outer():\n"
        "    path = None\n"
        "    def setter(value):\n"
        "        nonlocal path\n"
        "        path = value\n"
        "    def reader():\n"
        "        path.read_text()\n"
        "    return setter, reader\n"
        "setter, reader = outer()\n"
        "reader()\n"
        "setter(STATE_FILE)\n"
    )

    _assert_no_facts(result)


def test_unknown_harmless_sibling_does_not_escape_dangerous_capture() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "from external_api import consume\n"
        "def outer():\n"
        "    secret = STATE_FILE\n"
        "    def dangerous():\n"
        "        secret.read_text()\n"
        "    def harmless():\n"
        "        return None\n"
        "    return harmless, dangerous\n"
        "harmless, _ = outer()\n"
        "consume(harmless)\n"
    )

    _assert_no_facts(result)


def test_registered_origin_closure_to_unknown_callee_is_explicit_escape() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "from external_api import consume\n"
        "def outer():\n"
        "    path = STATE_FILE\n"
        "    def reader():\n"
        "        return path.read_text()\n"
        "    consume(reader)\n"
        "outer()\n"
    )

    assert result["accesses"] == []
    assert len(result["escapes"]) == 1
    assert result["escapes"][0]["resource_id"] == "runtime-resource:state"
    assert result["escapes"][0]["operation"] == "call:consume"
    assert result["escapes"][0]["reason"] == "closure_to_unknown_callee"
