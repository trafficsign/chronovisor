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


def _operations(result: dict) -> list[str]:
    return [row["operation"] for row in result["accesses"]]


def test_module_class_definition_strongly_kills_prior_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "Holder = STATE_FILE\n"
            "class Holder:\n"
            "    pass\n"
            "Holder.write_text('not-a-resource')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_nested_class_definition_strongly_kills_prior_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def outer():\n"
            "    Holder = STATE_FILE\n"
            "    class Holder:\n"
            "        pass\n"
            "    Holder.write_text('not-a-resource')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_class_header_and_body_execute_once_in_source_order() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "@STATE_FILE.exists()\n"
            "@STATE_FILE.is_file()\n"
            "class Holder(\n"
            "    STATE_FILE.read_text(),\n"
            "    metaclass=STATE_FILE.stat(),\n"
            "):\n"
            "    STATE_FILE.read_bytes()\n"
        )
    )

    ordered = sorted(result["accesses"], key=lambda row: row["evidence"]["line"])
    assert [row["operation"] for row in ordered] == [
        "path.exists",
        "path.is_file",
        "path.read_text",
        "path.stat",
        "path.read_bytes",
    ]
    assert {row["actor"] for row in ordered[:4]} == {"chronovisor.consumer:<module>"}
    assert ordered[4]["actor"] == "chronovisor.consumer:Holder.<classbody>"
    assert result["escapes"] == []


def test_method_default_executes_in_class_scope_but_body_does_not() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "class Holder:\n"
            "    path = STATE_FILE\n"
            "    def method(self, marker=path.exists()):\n"
            "        path.write_text('class-local-is-not-a-method-global')\n"
        )
    )

    assert _operations(result) == ["path.exists"]
    assert result["accesses"][0]["actor"] == ("chronovisor.consumer:Holder.<classbody>")
    assert result["escapes"] == []


def test_class_local_assignment_does_not_leak_to_module_scope() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "path = STATE_FILE\n"
            "class Holder:\n"
            "    path = object()\n"
            "    path.write_text('not-a-resource')\n"
            "path.read_text()\n"
        )
    )

    assert _operations(result) == ["path.read_text"]
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:<module>"
    assert result["escapes"] == []


def test_class_body_known_import_can_drive_sink() -> None:
    result = _access_fixture(
        consumer=(
            "class Holder:\n"
            "    from chronovisor.state import STATE_FILE as path\n"
            "    path.read_bytes()\n"
        )
    )

    assert _operations(result) == ["path.read_bytes"]
    assert result["accesses"][0]["actor"] == ("chronovisor.consumer:Holder.<classbody>")
    assert result["escapes"] == []


def test_nested_class_body_executes_recursively() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "class Outer:\n"
            "    class Nested:\n"
            "        STATE_FILE.write_bytes(b'nested')\n"
        )
    )

    assert _operations(result) == ["path.write_bytes"]
    assert result["accesses"][0]["actor"] == (
        "chronovisor.consumer:Outer.Nested.<classbody>"
    )
    assert result["escapes"] == []


def test_class_body_raise_preserves_prior_name_for_enclosing_handler() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "Holder = STATE_FILE\n"
            "try:\n"
            "    class Holder:\n"
            "        raise RuntimeError('construction failed')\n"
            "except RuntimeError:\n"
            "    Holder.write_text('prior-binding')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:<module>"
    assert result["escapes"] == []


def test_module_class_global_assignment_kills_prior_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "path = STATE_FILE\n"
            "class Holder:\n"
            "    global path\n"
            "    path = object()\n"
            "path.write_text('not-a-resource')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_module_class_global_assignment_can_bind_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "path = object()\n"
            "class Holder:\n"
            "    global path\n"
            "    path = STATE_FILE\n"
            "path.write_text('resource')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:<module>"
    assert result["escapes"] == []


def test_nested_class_nonlocal_assignment_kills_outer_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def outer():\n"
            "    path = STATE_FILE\n"
            "    class Holder:\n"
            "        nonlocal path\n"
            "        path = object()\n"
            "    path.write_text('not-a-resource')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_nested_class_nonlocal_assignment_can_bind_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def outer():\n"
            "    path = object()\n"
            "    class Holder:\n"
            "        nonlocal path\n"
            "        path = STATE_FILE\n"
            "    path.write_text('resource')\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:outer"
    assert result["escapes"] == []


def test_class_comprehension_outermost_iterable_sees_class_local() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "class Holder:\n"
            "    path = STATE_FILE\n"
            "    values = [item for item in path.iterdir()]\n"
        )
    )

    assert _operations(result) == ["path.iterdir"]
    assert result["accesses"][0]["actor"] == ("chronovisor.consumer:Holder.<classbody>")
    assert result["escapes"] == []


def test_class_comprehension_elt_ignores_class_local_shadow() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "path = STATE_FILE\n"
            "class Holder:\n"
            "    path = object()\n"
            "    values = [path.read_text() for item in [0]]\n"
        )
    )

    assert _operations(result) == ["path.read_text"]
    assert result["accesses"][0]["actor"] == ("chronovisor.consumer:Holder.<classbody>")
    assert result["escapes"] == []


def test_nested_class_comprehension_elt_sees_enclosing_function_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def outer():\n"
            "    path = STATE_FILE\n"
            "    class Holder:\n"
            "        path = object()\n"
            "        values = [path.read_bytes() for item in [0]]\n"
        )
    )

    assert _operations(result) == ["path.read_bytes"]
    assert result["accesses"][0]["actor"] == (
        "chronovisor.consumer:outer.<locals>.Holder.<classbody>"
    )
    assert result["escapes"] == []


def test_class_comprehension_later_generator_uses_implicit_scope() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "path = STATE_FILE\n"
            "class Holder:\n"
            "    path = object()\n"
            "    values = [\n"
            "        second\n"
            "        for first in [STATE_FILE]\n"
            "        for second in [first.read_text()]\n"
            "        if path.exists()\n"
            "    ]\n"
        )
    )

    assert set(_operations(result)) == {"path.read_text", "path.exists"}
    assert len(result["accesses"]) == 2
    assert {row["actor"] for row in result["accesses"]} == {
        "chronovisor.consumer:Holder.<classbody>"
    }
    assert result["escapes"] == []
