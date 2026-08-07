from __future__ import annotations

from typing import Any

from scripts.runtime_ownership.access import discover_access_facts
from tests.runtime_access_v2_helpers import (
    joined_access_rows,
    joined_escape_rows,
)


def _discover(
    consumer: str,
    *,
    extra_sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    sources = {
        "src/chronovisor/state.py": "STATE_FILE = object()\n",
        "src/chronovisor/consumer.py": consumer,
    }
    if extra_sources is not None:
        sources.update(extra_sources)
    candidate = {
        "id": "runtime-resource:state",
        "module": "chronovisor.state",
        "symbol": "STATE_FILE",
        "locator": {"type": "path", "value": "$ROOT/state.json"},
    }
    return discover_access_facts(
        {path: source.encode() for path, source in sources.items()}, [candidate]
    )


def _operations(result: dict[str, Any]) -> list[str]:
    return [row["operation"] for row in joined_access_rows(result)]


def _assert_known_helper_write(result: dict[str, Any]) -> None:
    assert _operations(result) == ["path.write_text"]
    assert joined_access_rows(result)[0]["actor"] == "chronovisor.consumer:run"
    assert joined_access_rows(result)[0]["sink_actor"] == "chronovisor.consumer:helper"


def _assert_unknown_helper_escape(result: dict[str, Any]) -> None:
    assert joined_access_rows(result) == []
    assert len(joined_escape_rows(result)) == 1
    assert joined_escape_rows(result)[0]["actor"] == "chronovisor.consumer:run"
    assert joined_escape_rows(result)[0]["operation"] == "call:helper"
    assert joined_escape_rows(result)[0]["reason"] == ("registered_locator_to_unknown_callee")


def test_known_repo_helper_definition_and_call_resolve() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def helper(path):\n"
        "    path.write_text('value')\n"
        "def run():\n"
        "    helper(STATE_FILE)\n"
    )

    _assert_known_helper_write(result)
    assert joined_escape_rows(result) == []


def test_external_import_shadows_repo_helper() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def helper(path):\n"
        "    path.write_text('false-repo-fact')\n"
        "from external_api import helper\n"
        "def run():\n"
        "    helper(STATE_FILE)\n"
    )

    _assert_unknown_helper_escape(result)


def test_assignment_shadows_repo_helper() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def helper(path):\n"
        "    path.write_text('false-repo-fact')\n"
        "helper = object()\n"
        "def run():\n"
        "    helper(STATE_FILE)\n"
    )

    _assert_unknown_helper_escape(result)


def test_known_and_external_branch_preserves_both_call_outcomes() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def helper(path):\n"
        "    path.write_text('known-branch')\n"
        "if object():\n"
        "    selected = helper\n"
        "else:\n"
        "    from external_api import helper as selected\n"
        "def run():\n"
        "    selected(STATE_FILE)\n"
    )

    _assert_known_helper_write(result)
    assert len(joined_escape_rows(result)) == 1
    assert joined_escape_rows(result)[0]["actor"] == "chronovisor.consumer:run"
    assert joined_escape_rows(result)[0]["operation"] == "call:selected"
    assert joined_escape_rows(result)[0]["reason"] == ("registered_locator_to_unknown_callee")


def test_repo_helper_definition_rebinds_prior_external_import() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "from external_api import helper\n"
        "def helper(path):\n"
        "    path.write_text('value')\n"
        "def run():\n"
        "    helper(STATE_FILE)\n"
    )

    _assert_known_helper_write(result)
    assert joined_escape_rows(result) == []


def test_imported_repo_helper_resolves_to_helper_sink_actor() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "from chronovisor.helpers import helper\n"
        "def run():\n"
        "    helper(STATE_FILE)\n",
        extra_sources={
            "src/chronovisor/helpers.py": (
                "def helper(path):\n    path.write_text('value')\n"
            )
        },
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_access_rows(result)[0]["actor"] == "chronovisor.consumer:run"
    assert joined_access_rows(result)[0]["sink_actor"] == "chronovisor.helpers:helper"
    assert joined_escape_rows(result) == []


def test_known_local_repo_helper_alias_resolves() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def helper(path):\n"
        "    path.write_text('value')\n"
        "selected = helper\n"
        "def run():\n"
        "    selected(STATE_FILE)\n"
    )

    _assert_known_helper_write(result)
    assert joined_escape_rows(result) == []


def test_function_parameter_shadows_module_repo_helper() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def helper(path):\n"
        "    path.write_text('false-repo-fact')\n"
        "def run(helper):\n"
        "    helper(STATE_FILE)\n"
    )

    _assert_unknown_helper_escape(result)


def test_external_module_attribute_call_is_unknown_callee_escape() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "import external_api\n"
        "def run():\n"
        "    external_api.helper(STATE_FILE)\n"
    )

    assert joined_access_rows(result) == []
    assert len(joined_escape_rows(result)) == 1
    assert joined_escape_rows(result)[0]["actor"] == "chronovisor.consumer:run"
    assert joined_escape_rows(result)[0]["operation"] == "call:external_api.helper"
    assert joined_escape_rows(result)[0]["reason"] == ("registered_locator_to_unknown_callee")


def test_branch_between_two_known_helpers_records_both_sink_actors() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def helper_a(path):\n"
        "    path.write_text('a')\n"
        "def helper_b(path):\n"
        "    path.write_text('b')\n"
        "if object():\n"
        "    selected = helper_a\n"
        "else:\n"
        "    selected = helper_b\n"
        "def run():\n"
        "    selected(STATE_FILE)\n"
    )

    assert _operations(result) == ["path.write_text", "path.write_text"]
    assert {row["actor"] for row in joined_access_rows(result)} == {"chronovisor.consumer:run"}
    assert {row["sink_actor"] for row in joined_access_rows(result)} == {
        "chronovisor.consumer:helper_a",
        "chronovisor.consumer:helper_b",
    }
    assert joined_escape_rows(result) == []


def test_known_helper_and_noncallable_branch_preserves_access_and_escape() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def helper(path):\n"
        "    path.write_text('known-branch')\n"
        "if object():\n"
        "    selected = helper\n"
        "else:\n"
        "    selected = object()\n"
        "def run():\n"
        "    selected(STATE_FILE)\n"
    )

    _assert_known_helper_write(result)
    assert len(joined_escape_rows(result)) == 1
    assert joined_escape_rows(result)[0]["actor"] == "chronovisor.consumer:run"
    assert joined_escape_rows(result)[0]["operation"] == "call:selected"
    assert joined_escape_rows(result)[0]["reason"] == ("registered_locator_to_unknown_callee")


def test_none_assignment_shadows_local_repo_helper() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def helper(path):\n"
        "    path.write_text('false-repo-fact')\n"
        "helper = None\n"
        "def run():\n"
        "    helper(STATE_FILE)\n"
    )

    assert joined_access_rows(result) == []
    assert len(joined_escape_rows(result)) == 1
    assert joined_escape_rows(result)[0]["operation"] == "call:helper"
    assert joined_escape_rows(result)[0]["reason"] == ("registered_locator_to_unknown_callee")


def test_none_assignment_shadows_imported_repo_helper() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "from chronovisor.helpers import helper\n"
        "helper = None\n"
        "def run():\n"
        "    helper(STATE_FILE)\n",
        extra_sources={
            "src/chronovisor/helpers.py": (
                "def helper(path):\n    path.write_text('false-repo-fact')\n"
            )
        },
    )

    assert joined_access_rows(result) == []
    assert len(joined_escape_rows(result)) == 1
    assert joined_escape_rows(result)[0]["operation"] == "call:helper"
    assert joined_escape_rows(result)[0]["reason"] == ("registered_locator_to_unknown_callee")


def test_none_assignment_shadows_imported_repo_module_alias() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "import chronovisor.helpers as helpers\n"
        "helpers = None\n"
        "def run():\n"
        "    helpers.helper(STATE_FILE)\n",
        extra_sources={
            "src/chronovisor/helpers.py": (
                "def helper(path):\n    path.write_text('false-repo-fact')\n"
            )
        },
    )

    assert joined_access_rows(result) == []
    assert len(joined_escape_rows(result)) == 1
    assert joined_escape_rows(result)[0]["operation"] == "call:helpers.helper"
    assert joined_escape_rows(result)[0]["reason"] == ("registered_locator_to_unknown_callee")


def test_unshadowed_local_class_constructor_binds_init_parameter() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "class Factory:\n"
        "    def __init__(self, path):\n"
        "        path.write_text('value')\n"
        "def run():\n"
        "    Factory(STATE_FILE)\n"
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_access_rows(result)[0]["actor"] == "chronovisor.consumer:run"
    assert joined_access_rows(result)[0]["sink_actor"] == (
        "chronovisor.consumer:Factory.__init__"
    )
    assert joined_escape_rows(result) == []


def test_named_imported_class_constructor_binds_init_parameter() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "from chronovisor.factories import Factory\n"
        "def run():\n"
        "    Factory(STATE_FILE)\n",
        extra_sources={
            "src/chronovisor/factories.py": (
                "class Factory:\n"
                "    def __init__(self, path):\n"
                "        path.write_text('value')\n"
            )
        },
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_access_rows(result)[0]["actor"] == "chronovisor.consumer:run"
    assert joined_access_rows(result)[0]["sink_actor"] == (
        "chronovisor.factories:Factory.__init__"
    )
    assert joined_escape_rows(result) == []


def test_imported_module_class_constructor_binds_init_parameter() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "import chronovisor.factories as factories\n"
        "def run():\n"
        "    factories.Factory(STATE_FILE)\n",
        extra_sources={
            "src/chronovisor/factories.py": (
                "class Factory:\n"
                "    def __init__(self, path):\n"
                "        path.write_text('value')\n"
            )
        },
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_access_rows(result)[0]["actor"] == "chronovisor.consumer:run"
    assert joined_access_rows(result)[0]["sink_actor"] == (
        "chronovisor.factories:Factory.__init__"
    )
    assert joined_escape_rows(result) == []


def test_none_assignment_shadows_imported_module_class_constructor() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "import chronovisor.factories as factories\n"
        "factories = None\n"
        "def run():\n"
        "    factories.Factory(STATE_FILE)\n",
        extra_sources={
            "src/chronovisor/factories.py": (
                "class Factory:\n"
                "    def __init__(self, path):\n"
                "        path.write_text('false-repo-fact')\n"
            )
        },
    )

    assert joined_access_rows(result) == []
    assert len(joined_escape_rows(result)) == 1
    assert joined_escape_rows(result)[0]["operation"] == "call:factories.Factory"
    assert joined_escape_rows(result)[0]["reason"] == ("registered_locator_to_unknown_callee")


def test_direct_star_imported_class_constructor_binds_init_parameter() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "from chronovisor.factories import *\n"
        "def run():\n"
        "    Factory(STATE_FILE)\n",
        extra_sources={
            "src/chronovisor/factories.py": (
                "class Factory:\n"
                "    def __init__(self, path):\n"
                "        path.write_text('value')\n"
            )
        },
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_access_rows(result)[0]["actor"] == "chronovisor.consumer:run"
    assert joined_access_rows(result)[0]["sink_actor"] == (
        "chronovisor.factories:Factory.__init__"
    )
    assert joined_escape_rows(result) == []


def test_two_hop_star_imported_class_constructor_binds_init_parameter() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "from chronovisor.bridge import *\n"
        "def run():\n"
        "    Factory(STATE_FILE)\n",
        extra_sources={
            "src/chronovisor/factories.py": (
                "class Factory:\n"
                "    def __init__(self, path):\n"
                "        path.write_text('value')\n"
            ),
            "src/chronovisor/bridge.py": "from chronovisor.factories import *\n",
        },
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_access_rows(result)[0]["actor"] == "chronovisor.consumer:run"
    assert joined_access_rows(result)[0]["sink_actor"] == (
        "chronovisor.factories:Factory.__init__"
    )
    assert joined_escape_rows(result) == []


def test_none_assignment_shadows_local_class_constructor() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "class Factory:\n"
        "    def __init__(self, path):\n"
        "        path.write_text('false-repo-fact')\n"
        "Factory = None\n"
        "def run():\n"
        "    Factory(STATE_FILE)\n"
    )

    assert joined_access_rows(result) == []
    assert len(joined_escape_rows(result)) == 1
    assert joined_escape_rows(result)[0]["operation"] == "call:Factory"
    assert joined_escape_rows(result)[0]["reason"] == ("registered_locator_to_unknown_callee")


def test_direct_star_imported_repo_helper_reaches_helper_sink() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "from chronovisor.helpers import *\n"
        "def run():\n"
        "    helper(STATE_FILE)\n",
        extra_sources={
            "src/chronovisor/helpers.py": (
                "def helper(path):\n    path.write_text('value')\n"
            )
        },
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_access_rows(result)[0]["sink_actor"] == "chronovisor.helpers:helper"
    assert joined_escape_rows(result) == []


def test_two_hop_star_imported_repo_helper_reaches_helper_sink() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "from chronovisor.bridge import *\n"
        "def run():\n"
        "    helper(STATE_FILE)\n",
        extra_sources={
            "src/chronovisor/helpers.py": (
                "def helper(path):\n    path.write_text('value')\n"
            ),
            "src/chronovisor/bridge.py": "from chronovisor.helpers import *\n",
        },
    )

    assert _operations(result) == ["path.write_text"]
    assert joined_access_rows(result)[0]["sink_actor"] == "chronovisor.helpers:helper"
    assert joined_escape_rows(result) == []


def test_literal_mapping_keyword_unpack_binds_helper_parameter() -> None:
    result = _discover(
        "from chronovisor.state import STATE_FILE\n"
        "def helper(path):\n"
        "    path.write_text('value')\n"
        "def run():\n"
        "    helper(**{'path': STATE_FILE})\n"
    )

    _assert_known_helper_write(result)
    assert joined_escape_rows(result) == []
