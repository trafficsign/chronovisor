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


def test_delete_name_strongly_kills_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def deleted():\n"
            "    path = STATE_FILE\n"
            "    del path\n"
            "    path.write_text('not-a-resource')\n"
        )
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_literal_tuple_and_list_destructuring_bind_exact_positions() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def destructured():\n"
            "    resource, ordinary = (STATE_FILE, object())\n"
            "    ordinary.write_text('not-a-resource')\n"
            "    resource.read_text()\n"
            "    ordinary, resource = [object(), STATE_FILE]\n"
            "    ordinary.read_text()\n"
            "    resource.write_text('resource')\n"
        )
    )

    assert set(_operations(result)) == {"path.read_text", "path.write_text"}
    assert result["escapes"] == []


def test_starred_destructuring_binds_prefix_remainder_and_suffix() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def destructured():\n"
            "    resource, *middle, ordinary = (STATE_FILE, object(), object())\n"
            "    resource.read_text()\n"
            "    ordinary.read_text()\n"
            "    ordinary, *middle, resource = [object(), object(), STATE_FILE]\n"
            "    ordinary.write_text('not-a-resource')\n"
            "    resource.write_text('resource')\n"
        )
    )

    assert set(_operations(result)) == {"path.read_text", "path.write_text"}
    assert result["escapes"] == []


def test_unknown_registered_iterable_is_bound_and_explicitly_escaped() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def destructured(values=STATE_FILE):\n"
            "    left, right = values\n"
        )
    )

    assert result["accesses"] == []
    assert len(result["escapes"]) == 1
    assert result["escapes"][0]["reason"] == (
        "unsupported_registered_origin_destructuring"
    )
    assert result["escapes"][0]["sink"] == "python.iterable-unpack"


def test_nested_function_definition_evaluates_defaults_and_kills_prior_name() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def outer():\n"
            "    nested = STATE_FILE\n"
            "    def nested(default=STATE_FILE.read_text()):\n"
            "        return default\n"
            "    nested.write_text('not-a-resource')\n"
        )
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_nested_definition_evaluates_decorator_and_kills_prior_name() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def outer():\n"
            "    decorated = STATE_FILE\n"
            "    @STATE_FILE.exists()\n"
            "    def decorated():\n"
            "        return None\n"
            "    decorated.write_text('not-a-resource')\n"
        )
    )

    assert _operations(result) == ["path.exists"]
    assert result["escapes"] == []


def test_nested_class_definition_evaluates_bases_and_kills_prior_name() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def outer():\n"
            "    Nested = STATE_FILE\n"
            "    class Nested(STATE_FILE.exists()):\n"
            "        pass\n"
            "    Nested.write_text('not-a-resource')\n"
        )
    )

    assert _operations(result) == ["path.exists"]
    assert result["escapes"] == []


def test_comprehension_target_shadows_without_leaking() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def scoped():\n"
            "    item = STATE_FILE\n"
            "    [item.write_text('not-a-resource') for item in [object()]]\n"
            "    item.read_text()\n"
        )
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_comprehension_binds_origin_before_element() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def scoped():\n"
            "    [item.write_text('resource') for item in [STATE_FILE]]\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_nested_comprehension_generators_see_prior_bindings() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def scoped():\n"
            "    [(left.read_text(), right.write_text('resource'))\n"
            "     for left in [STATE_FILE]\n"
            "     for right in [left]]\n"
        )
    )

    assert set(_operations(result)) == {"path.read_text", "path.write_text"}
    assert result["escapes"] == []


def test_comprehension_destructuring_target_uses_exact_positions() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def scoped():\n"
            "    [(resource.write_text('resource'), ordinary.write_text('ordinary'))\n"
            "     for resource, ordinary in [(STATE_FILE, object())]]\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []
