from __future__ import annotations

import pytest

from scripts.runtime_ownership.access import discover_access_facts


def _access_fixture(*, consumer: str, extra: dict[str, str] | None = None) -> dict:
    sources = {
        "src/chronovisor/state.py": "STATE_FILE = object()\n",
        "src/chronovisor/consumer.py": consumer,
        **(extra or {}),
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


def test_access_fixed_point_converges_for_default_and_local_alias() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE as state\n"
            "def save(path=state):\n"
            "    local = path\n"
            "    local.write_text('value')\n"
        )
    )

    assert result["counts"] == {
        "accesses": 1,
        "escapes": 0,
        "read": 0,
        "write": 1,
        "read_write": 0,
    }
    access = result["accesses"][0]
    assert access["actor"] == "chronovisor.consumer:save"
    assert access["sink_actor"] == "chronovisor.consumer:save"
    assert access["operation"] == "path.write_text"
    assert any(step.startswith("default:") for step in access["binding_chain"])
    assert any(step.startswith("alias:") for step in access["binding_chain"])


def test_access_tracks_imports_helpers_returns_and_simple_self_attrs() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE as imported_state\n"
            "import chronovisor.state as state_module\n"
            "def identity(value):\n"
            "    return value\n"
            "def append(path):\n"
            "    path.open('a')\n"
            "def read_imported():\n"
            "    return imported_state.read_text()\n"
            "def write_module_alias():\n"
            "    state_module.STATE_FILE.write_bytes(b'x')\n"
            "def through_return(path=imported_state):\n"
            "    identity(path).open('r')\n"
            "def through_helper(path=imported_state):\n"
            "    append(path)\n"
            "class Store:\n"
            "    def __init__(self, path=imported_state):\n"
            "        self.path = path\n"
            "    def load(self):\n"
            "        return self.path.read_bytes()\n"
        )
    )

    assert result["escapes"] == []
    by_actor = {row["actor"]: row for row in result["accesses"]}
    assert set(by_actor) == {
        "chronovisor.consumer:read_imported",
        "chronovisor.consumer:write_module_alias",
        "chronovisor.consumer:through_return",
        "chronovisor.consumer:through_helper",
        "chronovisor.consumer:Store.load",
    }
    helper = by_actor["chronovisor.consumer:through_helper"]
    assert helper["sink_actor"] == "chronovisor.consumer:append"
    assert any(
        "->chronovisor.consumer:append:path" in step for step in helper["binding_chain"]
    )
    returned = by_actor["chronovisor.consumer:through_return"]
    assert any(
        step.startswith("result:chronovisor.consumer:identity")
        for step in returned["binding_chain"]
    )
    assert any(
        step.startswith("attr:chronovisor.consumer:Store:path")
        for step in by_actor["chronovisor.consumer:Store.load"]["binding_chain"]
    )


def test_open_modes_are_exact_and_dynamic_mode_fails_closed() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def modes(mode):\n"
            "    open(STATE_FILE)\n"
            "    STATE_FILE.open('r')\n"
            "    STATE_FILE.open('w')\n"
            "    STATE_FILE.open('a')\n"
            "    STATE_FILE.open('r+')\n"
            "    STATE_FILE.open(mode)\n"
        )
    )

    assert result["counts"] == {
        "accesses": 5,
        "escapes": 1,
        "read": 2,
        "write": 2,
        "read_write": 1,
    }
    assert {row["operation"] for row in result["accesses"]} == {
        "builtin.open:r",
        "path.open:r",
        "path.open:w",
        "path.open:a",
        "path.open:r+",
    }
    assert result["escapes"][0]["reason"] == "dynamic_open_mode"
    assert result["escapes"][0]["sink"] == "pathlib.Path.open"


def test_registered_locator_to_unknown_callee_is_an_escape() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def leak():\n"
            "    opaque(STATE_FILE)\n"
        )
    )

    assert result["accesses"] == []
    assert len(result["escapes"]) == 1
    assert result["escapes"][0]["actor"] == "chronovisor.consumer:leak"
    assert result["escapes"][0]["reason"] == ("registered_locator_to_unknown_callee")


def test_access_ids_are_line_independent_and_detect_new_readers_and_writers() -> None:
    base_source = (
        "from chronovisor.state import STATE_FILE\n"
        "def writer_one():\n"
        "    STATE_FILE.write_text('one')\n"
    )
    base = _access_fixture(consumer=base_source)
    line_shifted = _access_fixture(consumer="\n\n" + base_source)
    second_writer = _access_fixture(
        consumer=(
            base_source + "def writer_two():\n" + "    STATE_FILE.write_text('two')\n"
        )
    )
    new_reader = _access_fixture(
        consumer=(
            base_source + "def reader():\n" + "    return STATE_FILE.read_text()\n"
        )
    )

    base_ids = set(base["access_ids"])
    assert set(line_shifted["access_ids"]) == base_ids
    assert len(set(second_writer["access_ids"]) - base_ids) == 1
    assert len(set(new_reader["access_ids"]) - base_ids) == 1
    assert {row["actor"] for row in second_writer["accesses"]} == {
        "chronovisor.consumer:writer_one",
        "chronovisor.consumer:writer_two",
    }
    assert {row["mode"] for row in new_reader["accesses"]} == {"read", "write"}


def test_shared_helper_preserves_each_logical_writer_binding() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def persist(path):\n"
            "    path.write_text('value')\n"
            "def writer_a():\n"
            "    persist(STATE_FILE)\n"
            "def writer_b():\n"
            "    persist(STATE_FILE)\n"
        )
    )

    assert result["counts"]["accesses"] == 2
    assert {row["actor"] for row in result["accesses"]} == {
        "chronovisor.consumer:writer_a",
        "chronovisor.consumer:writer_b",
    }
    assert {row["sink_actor"] for row in result["accesses"]} == {
        "chronovisor.consumer:persist"
    }
    assert len(set(result["access_ids"])) == 2


def test_recursive_binding_cycle_fails_closed_without_duplicate_access() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def helper(path):\n"
            "    path.write_text('value')\n"
            "    helper(path)\n"
            "def writer():\n"
            "    helper(STATE_FILE)\n"
        )
    )

    assert result["counts"]["accesses"] == 1
    assert result["counts"]["escapes"] == 1
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:writer"
    assert result["accesses"][0]["sink_actor"] == "chronovisor.consumer:helper"
    assert result["escapes"][0]["reason"] == "binding_cycle"
    assert result["escapes"][0]["sink"] == "chronovisor.consumer:helper"


def test_helper_access_ids_ignore_unrelated_calls_and_add_identical_calls() -> None:
    helper = (
        "from chronovisor.state import STATE_FILE\n"
        "def persist(path):\n"
        "    path.write_text('value')\n"
    )
    base = _access_fixture(
        consumer=helper + "def writer():\n" + "    persist(STATE_FILE)\n"
    )
    with_unrelated_call = _access_fixture(
        consumer=(
            helper
            + "def noop():\n"
            + "    return None\n"
            + "def writer():\n"
            + "    noop()\n"
            + "    persist(STATE_FILE)\n"
        )
    )
    with_identical_call = _access_fixture(
        consumer=(
            helper
            + "def writer():\n"
            + "    persist(STATE_FILE)\n"
            + "    persist(STATE_FILE)\n"
        )
    )

    base_ids = set(base["access_ids"])
    assert set(with_unrelated_call["access_ids"]) == base_ids
    assert len(set(with_identical_call["access_ids"]) - base_ids) == 1
    assert len(with_identical_call["access_ids"]) == 2


def test_provenance_overflow_is_a_deterministic_escape() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "def persist(path):\n"
        "    path.write_text('value')\n"
        + "".join(
            f"def writer_{index:02d}():\n    persist(STATE_FILE)\n"
            for index in range(65)
        )
    )

    first = _access_fixture(consumer=source)
    second = _access_fixture(consumer=source)

    assert first["counts"] == {
        "accesses": 64,
        "escapes": 1,
        "read": 0,
        "write": 64,
        "read_write": 0,
    }
    assert first["access_ids"] == second["access_ids"]
    assert first["escape_ids"] == second["escape_ids"]
    assert first["escapes"][0]["reason"] == "provenance_overflow"
    assert first["escapes"][0]["sink"] == "pathlib.Path.write_text"


def test_ambiguous_resource_candidates_for_one_symbol_are_rejected() -> None:
    snapshot = {
        "src/chronovisor/state.py": b"STATE_FILE = object()\n",
        "src/chronovisor/consumer.py": (b"from chronovisor.state import STATE_FILE\n"),
    }
    candidates = [
        {
            "id": "runtime-resource:first",
            "module": "chronovisor.state",
            "symbol": "STATE_FILE",
            "locator": {"type": "path", "value": "$ROOT/first.json"},
        },
        {
            "id": "runtime-resource:second",
            "module": "chronovisor.state",
            "symbol": "STATE_FILE",
            "locator": {"type": "path", "value": "$ROOT/second.json"},
        },
    ]

    with pytest.raises(ValueError, match="ambiguous runtime resource candidates"):
        discover_access_facts(snapshot, candidates)


def test_nested_writer_propagates_closure_origin_through_nested_helper() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE as state\n"
            "def outer(path=state):\n"
            "    def persist(value):\n"
            "        value.write_text('nested')\n"
            "    def nested_writer():\n"
            "        persist(path)\n"
            "    nested_writer()\n"
            "def second_writer():\n"
            "    state.write_text('second')\n"
        )
    )

    by_actor = {row["actor"]: row for row in result["accesses"]}
    nested_actor = "chronovisor.consumer:outer.<locals>.nested_writer"
    assert set(by_actor) == {
        nested_actor,
        "chronovisor.consumer:second_writer",
    }
    assert by_actor[nested_actor]["sink_actor"] == (
        "chronovisor.consumer:outer.<locals>.persist"
    )
    assert any(
        step.startswith("closure:chronovisor.consumer:outer->")
        for step in by_actor[nested_actor]["binding_chain"]
    )
    assert result["escapes"] == []


def test_generic_expressions_visit_known_sinks_with_stable_ids() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "def expressions(flag):\n"
        "    flag and STATE_FILE.read_text()\n"
        "    STATE_FILE.read_bytes() == b'x'\n"
        "    not STATE_FILE.exists()\n"
        "    STATE_FILE.stat() + 1\n"
        "    STATE_FILE.is_file() if flag else STATE_FILE.is_dir()\n"
        "    (named := STATE_FILE.iterdir())\n"
        "    STATE_FILE.read_text()[STATE_FILE.stat():STATE_FILE.stat()]\n"
        "    {STATE_FILE.read_text(): STATE_FILE.read_bytes()}\n"
        "    [STATE_FILE.exists()]\n"
        "    (STATE_FILE.is_file(),)\n"
        "    {STATE_FILE.is_dir()}\n"
        "    f'{STATE_FILE.read_text()} {STATE_FILE.stat()}'\n"
        "    [*[STATE_FILE.read_text()]]\n"
        "async def awaited():\n"
        "    await passthrough(STATE_FILE.read_bytes())\n"
    )
    shifted = source.replace(
        "def expressions(flag):\n",
        "def expressions(flag):\n    len(())\n",
        1,
    )

    result = _access_fixture(consumer=source)
    with_unrelated_expression = _access_fixture(consumer=shifted)

    assert result["counts"] == {
        "accesses": 19,
        "escapes": 0,
        "read": 19,
        "write": 0,
        "read_write": 0,
    }
    assert set(with_unrelated_expression["access_ids"]) == set(result["access_ids"])


def test_generic_origins_bind_propagate_or_fail_closed_by_semantics() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def named():\n"
            "    if (path := STATE_FILE):\n"
            "        path.read_text()\n"
            "def opaque(flag):\n"
            "    opaque_container([STATE_FILE])\n"
            "    opaque_bool(flag or STATE_FILE)\n"
            "def transformed():\n"
            "    (STATE_FILE / 'child').write_text('wrong-base')\n"
            "    STATE_FILE[0].write_text('wrong-base')\n"
        )
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.read_text"
    assert any(
        step.startswith("alias:chronovisor.consumer:named:path")
        for step in result["accesses"][0]["binding_chain"]
    )
    assert {row["reason"] for row in result["escapes"]} == {
        "unsupported_registered_origin_control_flow",
        "registered_locator_to_unknown_callee",
        "unsupported_registered_origin_expression",
    }
    assert {row["operation"] for row in result["escapes"]} == {
        "control:if",
        "call:opaque_container",
        "call:opaque_bool",
        "expression:binop",
        "expression:subscript",
    }


def test_comprehensions_visit_sinks_and_fail_closed_on_raw_controls() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "def comprehensions():\n"
        "    [STATE_FILE.read_text() for _ in STATE_FILE.iterdir() if STATE_FILE.exists()]\n"
        "    {STATE_FILE.read_bytes() for _ in range(1)}\n"
        "    {STATE_FILE.stat(): STATE_FILE.is_file() for _ in range(1) if STATE_FILE.is_dir()}\n"
        "    tuple(STATE_FILE.read_text() for _ in range(1))\n"
        "    [item for item in STATE_FILE]\n"
        "    [item for item in range(1) if STATE_FILE]\n"
    )
    shifted = source.replace(
        "def comprehensions():\n",
        "def comprehensions():\n    len(())\n",
        1,
    )

    result = _access_fixture(consumer=source)
    with_unrelated_expression = _access_fixture(consumer=shifted)

    assert result["counts"] == {
        "accesses": 8,
        "escapes": 2,
        "read": 8,
        "write": 0,
        "read_write": 0,
    }
    assert {row["operation"] for row in result["escapes"]} == {
        "control:comprehension_iter",
        "control:comprehension_if",
    }
    assert {row["reason"] for row in result["escapes"]} == {
        "unsupported_registered_origin_control_flow"
    }
    assert set(with_unrelated_expression["access_ids"]) == set(result["access_ids"])
    assert set(with_unrelated_expression["escape_ids"]) == set(result["escape_ids"])


def test_lambda_origin_accesses_are_explicit_stable_escapes() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "import chronovisor.state as state_module\n"
        "def deferred(path=STATE_FILE):\n"
        "    first = lambda: path.write_text('deferred')\n"
        "    second = lambda value=STATE_FILE: value.read_text()\n"
        "    third = lambda: state_module.STATE_FILE.exists()\n"
    )
    shifted = source.replace(
        "def deferred(path=STATE_FILE):\n",
        "def deferred(path=STATE_FILE):\n    len(())\n",
        1,
    )

    result = _access_fixture(consumer=source)
    with_unrelated_expression = _access_fixture(consumer=shifted)

    assert result["accesses"] == []
    assert len(result["escapes"]) == 3
    assert {row["reason"] for row in result["escapes"]} == {
        "unsupported_registered_origin_lambda"
    }
    assert {row["sink"] for row in result["escapes"]} == {"python.lambda"}
    assert set(with_unrelated_expression["escape_ids"]) == set(result["escape_ids"])


def test_unknown_method_receiver_is_a_stable_escape() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "def opaque_method():\n"
        "    STATE_FILE.opaque()\n"
    )
    shifted = source.replace(
        "def opaque_method():\n",
        "def opaque_method():\n    len(())\n",
        1,
    )

    result = _access_fixture(consumer=source)
    with_unrelated_call = _access_fixture(consumer=shifted)

    assert result["accesses"] == []
    assert len(result["escapes"]) == 1
    assert result["escapes"][0]["reason"] == ("registered_locator_to_unknown_callee")
    assert result["escapes"][0]["operation"] == "call:STATE_FILE.opaque"
    assert set(with_unrelated_call["escape_ids"]) == set(result["escape_ids"])


def test_await_raw_origin_escapes_while_inner_sink_is_discovered() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "async def awaited():\n"
        "    await STATE_FILE\n"
        "    await passthrough(STATE_FILE.read_text())\n"
    )
    shifted = source.replace(
        "async def awaited():\n",
        "async def awaited():\n    len(())\n",
        1,
    )

    result = _access_fixture(consumer=source)
    with_unrelated_call = _access_fixture(consumer=shifted)

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.read_text"
    assert len(result["escapes"]) == 1
    assert result["escapes"][0]["operation"] == "control:await"
    assert result["escapes"][0]["reason"] == (
        "unsupported_registered_origin_control_flow"
    )
    assert set(with_unrelated_call["access_ids"]) == set(result["access_ids"])
    assert set(with_unrelated_call["escape_ids"]) == set(result["escape_ids"])


def test_parameter_binding_kills_same_named_module_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def shadowed(STATE_FILE):\n"
            "    STATE_FILE.write_text('not-the-resource')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_local_assignment_strongly_kills_same_named_module_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def shadowed():\n"
            "    STATE_FILE = object()\n"
            "    STATE_FILE.write_text('not-the-resource')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_open_shadow_keyword_file_and_invalid_modes_are_classified_exactly() -> None:
    shadowed = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "from chronovisor.shadow import open as repo_open\n"
            "from external_api import open as external_open\n"
            "def open(path, mode='r'):\n"
            "    path.write_text('local')\n"
            "def call_local():\n"
            "    open(STATE_FILE, 'w')\n"
            "def call_imported():\n"
            "    repo_open(STATE_FILE, 'w')\n"
            "def call_external():\n"
            "    external_open(STATE_FILE)\n"
            "def call_local_shadow():\n"
            "    open = external_open\n"
            "    open(STATE_FILE)\n"
            "def call_parameter(open):\n"
            "    open(STATE_FILE)\n"
        ),
        extra={
            "src/chronovisor/shadow.py": (
                "def open(path, mode='r'):\n    path.write_bytes(b'imported')\n"
            )
        },
    )
    modes_source = (
        "from chronovisor.state import STATE_FILE\n"
        "def modes(mode):\n"
        "    open(file=STATE_FILE, mode='w')\n"
        "    open(file=STATE_FILE, mode='rw')\n"
        "    open(file=STATE_FILE, mode=mode)\n"
        "    STATE_FILE.open('rr')\n"
    )
    modes = _access_fixture(consumer=modes_source)
    shifted_modes = _access_fixture(
        consumer=modes_source.replace(
            "def modes(mode):\n",
            "def modes(mode):\n    len(())\n",
            1,
        )
    )

    assert {row["operation"] for row in shadowed["accesses"]} == {
        "path.write_text",
        "path.write_bytes",
    }
    assert len(shadowed["escapes"]) == 3
    assert {row["reason"] for row in shadowed["escapes"]} == {
        "registered_locator_to_unknown_callee"
    }
    assert modes["counts"] == {
        "accesses": 1,
        "escapes": 3,
        "read": 0,
        "write": 1,
        "read_write": 0,
    }
    assert modes["accesses"][0]["operation"] == "builtin.open:w"
    assert [row["reason"] for row in modes["escapes"]].count("invalid_open_mode") == 2
    assert [row["reason"] for row in modes["escapes"]].count("dynamic_open_mode") == 1
    assert set(shifted_modes["access_ids"]) == set(modes["access_ids"])
    assert set(shifted_modes["escape_ids"]) == set(modes["escape_ids"])


def test_async_control_flow_does_not_drop_origins() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "async def async_controls():\n"
            "    async with STATE_FILE:\n"
            "        STATE_FILE.read_text()\n"
            "    async for _item in STATE_FILE:\n"
            "        STATE_FILE.write_text('async')\n"
        )
    )

    assert {row["operation"] for row in result["accesses"]} == {
        "path.read_text",
        "path.write_text",
    }
    assert {row["operation"] for row in result["escapes"]} == {
        "control:async_with",
        "control:async_for",
    }
    assert {row["reason"] for row in result["escapes"]} == {
        "unsupported_registered_origin_control_flow"
    }


def test_match_control_flow_does_not_drop_origins() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "import chronovisor.state as state_module\n"
            "def match_control(marker):\n"
            "    match STATE_FILE:\n"
            "        case _:\n"
            "            STATE_FILE.read_bytes()\n"
            "    match marker:\n"
            "        case _ if STATE_FILE:\n"
            "            pass\n"
            "    match marker:\n"
            "        case state_module.STATE_FILE:\n"
            "            pass\n"
        )
    )

    assert {row["operation"] for row in result["accesses"]} == {"path.read_bytes"}
    assert {row["operation"] for row in result["escapes"]} == {
        "control:match_subject",
        "control:match_guard",
        "control:match_pattern",
    }
    assert {row["reason"] for row in result["escapes"]} == {
        "unsupported_registered_origin_control_flow"
    }


def test_yield_control_flow_does_not_drop_origins() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def yielded():\n"
            "    yield STATE_FILE\n"
            "    yield from STATE_FILE\n"
        )
    )

    assert result["accesses"] == []
    assert {row["operation"] for row in result["escapes"]} == {
        "control:yield",
        "control:yield_from",
    }
    assert {row["reason"] for row in result["escapes"]} == {
        "unsupported_registered_origin_control_flow"
    }


def test_sequential_assignment_strongly_kills_previous_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def killed():\n"
            "    path = STATE_FILE\n"
            "    path = object()\n"
            "    path.write_text('not-the-resource')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_possible_branch_preserves_only_feasible_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def possible(flag):\n"
            "    path = STATE_FILE\n"
            "    if flag:\n"
            "        path = object()\n"
            "    path.write_text('possible-resource')\n"
        )
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:possible"
    assert result["accesses"][0]["operation"] == "path.write_text"
    assert result["escapes"] == []


def test_branch_assignment_does_not_leak_into_mutually_exclusive_else() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def exclusive(flag):\n"
            "    path = object()\n"
            "    if flag:\n"
            "        path = STATE_FILE\n"
            "    else:\n"
            "        path.write_text('not-the-resource')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_mutually_exclusive_branches_join_each_resource_once() -> None:
    sources = {
        "src/chronovisor/state.py": "STATE_A = object()\nSTATE_B = object()\n",
        "src/chronovisor/consumer.py": (
            "from chronovisor.state import STATE_A, STATE_B\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        path = STATE_A\n"
            "    else:\n"
            "        path = STATE_B\n"
            "    path.write_text('chosen-resource')\n"
        ),
    }
    candidates = [
        {
            "id": f"runtime-resource:state-{suffix.lower()}",
            "module": "chronovisor.state",
            "symbol": f"STATE_{suffix}",
            "locator": {"type": "path", "value": f"$ROOT/state-{suffix}.json"},
        }
        for suffix in ("A", "B")
    ]

    result = discover_access_facts(
        {path: text.encode() for path, text in sources.items()}, candidates
    )

    assert len(result["accesses"]) == 2
    assert {row["resource_id"] for row in result["accesses"]} == {
        "runtime-resource:state-a",
        "runtime-resource:state-b",
    }
    assert {row["actor"] for row in result["accesses"]} == {
        "chronovisor.consumer:choose"
    }
    assert result["escapes"] == []


def test_constant_origin_without_an_io_sink_is_not_a_writer() -> None:
    result = _access_fixture(consumer="from chronovisor.state import STATE_FILE\n")

    assert result["accesses"] == []
    assert result["escapes"] == []
