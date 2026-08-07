from __future__ import annotations

from typing import Any

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


def test_registered_origin_is_injected_when_its_assignment_executes() -> None:
    sources = {
        "src/chronovisor/state.py": (
            "def before(path=STATE_FILE):\n"
            "    path.read_text()\n"
            "STATE_FILE = object()\n"
            "def after(path=STATE_FILE):\n"
            "    path.write_text('resource-after-assignment')\n"
        )
    }
    candidate = {
        "id": "runtime-resource:state",
        "module": "chronovisor.state",
        "symbol": "STATE_FILE",
        "locator": {"type": "path", "value": "$ROOT/state.json"},
    }

    result = discover_access_facts(
        {path: text.encode() for path, text in sources.items()}, [candidate]
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_access_rows(result)[0]["actor"] == "chronovisor.state:after"
    assert joined_escape_rows(result) == []


def test_module_function_definition_strongly_kills_prior_origin() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "handler = STATE_FILE\n"
            "def handler():\n"
            "    return None\n"
            "handler.write_text('not-a-resource')\n"
        )
    )

    assert joined_access_rows(result) == []
    assert joined_escape_rows(result) == []


def test_later_function_definition_does_not_retroactively_kill_earlier_sink() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "handler = STATE_FILE\n"
            "handler.write_text('resource-before-definition')\n"
            "def handler():\n"
            "    return None\n"
        )
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_escape_rows(result) == []


def test_top_level_default_is_evaluated_once_and_cached_before_later_kill() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def cached(path=STATE_FILE, marker=STATE_FILE.read_text()):\n"
            "    path.write_text('cached-resource')\n"
            "STATE_FILE = object()\n"
        )
    )

    assert set(_operations(result)) == {"path.read_text", "path.write_text"}
    assert len(joined_access_rows(result)) == 2
    by_operation = {row["operation"]: row for row in joined_access_rows(result)}
    assert by_operation["path.read_text"]["actor"] == ("chronovisor.consumer:<module>")
    assert by_operation["path.read_text"]["sink_actor"] == (
        "chronovisor.consumer:<module>"
    )
    assert by_operation["path.write_text"]["actor"] == ("chronovisor.consumer:cached")
    assert any(
        step.startswith("default:chronovisor.consumer:cached:path")
        for step in by_operation["path.write_text"]["binding_chain"]
    )
    assert joined_escape_rows(result) == []


def test_top_level_decorator_is_evaluated_once_with_module_actor() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "@STATE_FILE.exists()\n"
            "def decorated():\n"
            "    return None\n"
            "STATE_FILE = object()\n"
        )
    )

    assert _operations(result) == ["path.exists"]
    assert len(joined_access_rows(result)) == 1
    assert joined_access_rows(result)[0]["actor"] == "chronovisor.consumer:<module>"
    assert joined_access_rows(result)[0]["sink_actor"] == ("chronovisor.consumer:<module>")
    assert joined_escape_rows(result) == []


def test_top_level_default_cannot_see_later_assignment() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def cached(path=later):\n"
            "    path.write_text('must-not-be-invented')\n"
            "later = STATE_FILE\n"
        )
    )

    assert joined_access_rows(result) == []
    assert joined_escape_rows(result) == []


def test_top_level_body_uses_final_module_state_not_definition_closure() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def reads_global():\n"
            "    STATE_FILE.write_text('must-see-later-kill')\n"
            "STATE_FILE = object()\n"
        )
    )

    assert joined_access_rows(result) == []
    assert joined_escape_rows(result) == []


def test_definition_defaults_follow_import_and_assignment_source_order() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE as imported\n"
            "before = imported\n"
            "def sees_before(path=before):\n"
            "    path.read_text()\n"
            "before = object()\n"
            "def misses_after(path=after):\n"
            "    path.write_text('must-not-be-invented')\n"
            "after = imported\n"
        )
    )

    assert _operations(result) == ["path.read_text"]
    assert joined_access_rows(result)[0]["actor"] == "chronovisor.consumer:sees_before"
    assert joined_escape_rows(result) == []


def test_definition_default_cannot_see_later_import() -> None:
    result = _access_fixture(
        consumer=(
            "def misses_import(path=imported):\n"
            "    path.write_text('must-not-be-invented')\n"
            "from chronovisor.state import STATE_FILE as imported\n"
        )
    )

    assert joined_access_rows(result) == []
    assert joined_escape_rows(result) == []


def test_definition_execution_access_fact_ids_are_line_shift_stable() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "def cached(path=STATE_FILE, marker=STATE_FILE.read_text()):\n"
        "    path.write_text('cached-resource')\n"
    )
    base = _access_fixture(consumer=source)
    shifted = _access_fixture(consumer="\n\n\n" + source)

    assert set(shifted["access_fact_ids"]) == set(base["access_fact_ids"])
    assert set(_operations(base)) == {"path.read_text", "path.write_text"}
    assert shifted["escape_fact_ids"] == base["escape_fact_ids"] == []


def test_annotations_evaluate_at_definition_time_without_future_import() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def annotated(\n"
            "    positional: STATE_FILE.read_text(),\n"
            "    /,\n"
            "    regular: STATE_FILE.read_bytes(),\n"
            "    *args: STATE_FILE.exists(),\n"
            "    keyword: STATE_FILE.stat(),\n"
            "    **kwargs: STATE_FILE.is_file(),\n"
            ") -> STATE_FILE.is_dir():\n"
            "    return None\n"
        )
    )

    assert set(_operations(result)) == {
        "path.read_text",
        "path.read_bytes",
        "path.exists",
        "path.stat",
        "path.is_file",
        "path.is_dir",
    }
    assert len(joined_access_rows(result)) == 6
    assert {row["actor"] for row in joined_access_rows(result)} == {
        "chronovisor.consumer:<module>"
    }
    assert joined_escape_rows(result) == []


def test_future_annotations_are_not_evaluated_at_definition_time() -> None:
    result = _access_fixture(
        consumer=(
            "from __future__ import annotations\n"
            "from chronovisor.state import STATE_FILE\n"
            "def annotated(\n"
            "    value: STATE_FILE.read_text(),\n"
            ") -> STATE_FILE.exists():\n"
            "    return None\n"
        )
    )

    assert joined_access_rows(result) == []
    assert joined_escape_rows(result) == []
