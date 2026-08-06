from __future__ import annotations

from scripts.runtime_ownership.access import discover_access_facts


def _discover(consumer: str) -> dict:
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
        {path: source.encode() for path, source in sources.items()}, [candidate]
    )


def _operations(result: dict) -> list[str]:
    return [row["operation"] for row in result["accesses"]]


def test_known_function_decorator_return_rebinds_function_name() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def expose_origin(target):\n"
        "    return STATE_FILE\n"
        "@expose_origin\n"
        "def decorated():\n"
        "    return None\n"
        "decorated.read_text()\n"
    )

    assert _operations(result) == ["path.read_text"]
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:<module>"
    assert result["escapes"] == []


def test_known_function_decorator_return_rebinds_class_name() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def expose_origin(target):\n"
        "    return STATE_FILE\n"
        "@expose_origin\n"
        "class Decorated:\n"
        "    pass\n"
        "Decorated.read_text()\n"
    )

    assert _operations(result) == ["path.read_text"]
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:<module>"
    assert result["escapes"] == []


def test_decorators_apply_bottom_up_without_false_origin() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def expose_origin(target):\n"
        "    return STATE_FILE\n"
        "def clear_origin(target):\n"
        "    return object()\n"
        "@clear_origin\n"
        "@expose_origin\n"
        "def decorated():\n"
        "    return None\n"
        "decorated.read_text()\n"
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_nested_class_global_read_uses_module_binding() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "path = STATE_FILE\n"
        "def outer():\n"
        "    path = object()\n"
        "    class Holder:\n"
        "        global path\n"
        "        path.read_text()\n"
    )

    assert _operations(result) == ["path.read_text"]
    assert result["accesses"][0]["actor"] == (
        "chronovisor.consumer:outer.<locals>.Holder.<classbody>"
    )
    assert result["escapes"] == []


def test_function_annotation_may_raise_to_enclosing_handler() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "try:\n"
        "    path = STATE_FILE\n"
        "    def annotated(value: annotation_factory()):\n"
        "        return value\n"
        "except Exception:\n"
        "    path.read_text()\n"
    )

    assert _operations(result) == ["path.read_text"]
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:<module>"
    assert result["escapes"] == []


def test_future_function_annotation_does_not_reach_enclosing_handler() -> None:
    result = _discover(
        "from __future__ import annotations\n"
        "from chronovisor.state import STATE_FILE\n"
        "try:\n"
        "    path = STATE_FILE\n"
        "    def annotated(value: annotation_factory()):\n"
        "        return value\n"
        "except Exception:\n"
        "    path.read_text()\n"
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_class_body_may_raise_to_enclosing_handler() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "try:\n"
        "    path = STATE_FILE\n"
        "    class Holder:\n"
        "        class_body_effect()\n"
        "except Exception:\n"
        "    path.read_text()\n"
    )

    assert _operations(result) == ["path.read_text"]
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:<module>"
    assert result["escapes"] == []


def test_same_line_default_calls_remain_distinct_physical_sites() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def cached(first=STATE_FILE.read_text(), second=STATE_FILE.read_text()): pass\n"
    )

    assert _operations(result) == ["path.read_text", "path.read_text"]
    assert len(set(result["access_ids"])) == 2
    assert {row["actor"] for row in result["accesses"]} == {
        "chronovisor.consumer:<module>"
    }
    assert len({row["evidence"]["line"] for row in result["accesses"]}) == 1
    assert result["escapes"] == []
