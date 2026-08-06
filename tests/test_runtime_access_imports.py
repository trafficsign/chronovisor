from __future__ import annotations

import pytest

from scripts.runtime_ownership.access import discover_access_facts


def _discover(
    sources: dict[str, str],
    *,
    module: str = "chronovisor.state",
    symbol: str = "STATE_FILE",
) -> dict:
    candidate = {
        "id": "runtime-resource:state",
        "module": module,
        "symbol": symbol,
        "locator": {"type": "path", "value": "$ROOT/state.json"},
    }
    return discover_access_facts(
        {path: source.encode() for path, source in sources.items()}, [candidate]
    )


@pytest.mark.parametrize(
    "consumer",
    [
        (
            "def save():\n"
            "    from chronovisor.state import STATE_FILE as path\n"
            "    path.write_text('value')\n"
        ),
        (
            "def save():\n"
            "    import chronovisor.state as state_module\n"
            "    state_module.STATE_FILE.write_text('value')\n"
        ),
        (
            "def save():\n"
            "    import chronovisor.state\n"
            "    chronovisor.state.STATE_FILE.write_text('value')\n"
        ),
        (
            "def save():\n"
            "    from . import state as state_module\n"
            "    state_module.STATE_FILE.write_text('value')\n"
        ),
    ],
)
def test_function_local_import_forms_resolve_registered_origin(
    consumer: str,
) -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/consumer.py": consumer,
        }
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:save"
    assert result["accesses"][0]["operation"] == "path.write_text"
    assert result["escapes"] == []


def test_package_init_relative_reexport_resolves_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/pkg/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/pkg/__init__.py": (
                "from .state import STATE_FILE as EXPORTED_STATE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.pkg import EXPORTED_STATE\n"
                "def save():\n"
                "    EXPORTED_STATE.write_text('value')\n"
            ),
        },
        module="chronovisor.pkg.state",
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.write_text"
    assert result["escapes"] == []


def test_two_hop_reexport_and_simple_module_alias_resolve_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/bridge_a.py": (
                "from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
            ),
            "src/chronovisor/bridge_b.py": (
                "import chronovisor.bridge_a as bridge\n"
                "TRANSIT_STATE = bridge.EXPORTED_STATE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.bridge_b import TRANSIT_STATE\n"
                "def save():\n"
                "    TRANSIT_STATE.write_text('value')\n"
            ),
        }
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.write_text"
    assert result["escapes"] == []


def test_reexport_fixed_point_has_no_silent_depth_cap() -> None:
    sources = {"src/chronovisor/state.py": "STATE_FILE = object()\n"}
    previous = "chronovisor.state"
    for index in range(40):
        module = f"chronovisor.hop_{index:02d}"
        sources[f"src/chronovisor/hop_{index:02d}.py"] = (
            f"from {previous} import STATE_FILE\n"
        )
        previous = module
    sources["src/chronovisor/consumer.py"] = (
        f"from {previous} import STATE_FILE\n"
        "def save():\n"
        "    STATE_FILE.write_text('value')\n"
    )

    result = _discover(sources)

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["operation"] == "path.write_text"
    assert result["escapes"] == []


def test_reexport_cycle_connected_to_registered_origin_is_explicit() -> None:
    sources = {
        "src/chronovisor/a.py": "from chronovisor.b import STATE_FILE\n",
        "src/chronovisor/b.py": "from chronovisor.a import STATE_FILE\n",
        "src/chronovisor/consumer.py": "from chronovisor.a import STATE_FILE\n",
    }

    with pytest.raises(
        ValueError,
        match=(
            r"runtime access import cycle: "
            r"chronovisor\.a:STATE_FILE -> chronovisor\.b:STATE_FILE -> "
            r"chronovisor\.a:STATE_FILE"
        ),
    ):
        _discover(sources, module="chronovisor.a")


@pytest.mark.parametrize(
    "consumer",
    [
        (
            "from chronovisor.state import STATE_FILE\n"
            "def shadowed():\n"
            "    from external_api import STATE_FILE\n"
            "    STATE_FILE.write_text('not-the-resource')\n"
        ),
        (
            "import chronovisor.state as state_module\n"
            "def shadowed():\n"
            "    import external_api as state_module\n"
            "    state_module.STATE_FILE.write_text('not-the-resource')\n"
        ),
    ],
)
def test_function_local_import_strongly_shadows_module_binding(
    consumer: str,
) -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/consumer.py": consumer,
        }
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_closure_free_and_nonlocal_names_receive_outer_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/consumer.py": (
                "from chronovisor.state import STATE_FILE\n"
                "def outer():\n"
                "    free_path = STATE_FILE\n"
                "    nonlocal_path = STATE_FILE\n"
                "    def free_reader():\n"
                "        free_path.read_text()\n"
                "    def nonlocal_writer():\n"
                "        nonlocal nonlocal_path\n"
                "        nonlocal_path.write_text('value')\n"
            ),
        }
    )

    assert {row["operation"] for row in result["accesses"]} == {
        "path.read_text",
        "path.write_text",
    }
    assert result["escapes"] == []


def test_nonlocal_assignment_kills_captured_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/consumer.py": (
                "from chronovisor.state import STATE_FILE\n"
                "def outer():\n"
                "    path = STATE_FILE\n"
                "    def inner():\n"
                "        nonlocal path\n"
                "        path = object()\n"
                "        path.write_text('not-the-resource')\n"
            ),
        }
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_nested_scope_assignment_does_not_shadow_outer_global() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/consumer.py": (
                "from chronovisor.state import STATE_FILE\n"
                "def outer():\n"
                "    def nested():\n"
                "        STATE_FILE = object()\n"
                "    STATE_FILE.write_text('value')\n"
            ),
        }
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:outer"
    assert result["accesses"][0]["operation"] == "path.write_text"
    assert result["escapes"] == []


def test_global_name_uses_module_binding_not_outer_local() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/consumer.py": (
                "from chronovisor.state import STATE_FILE\n"
                "path = STATE_FILE\n"
                "def outer():\n"
                "    path = object()\n"
                "    def global_writer():\n"
                "        global path\n"
                "        path.write_text('value')\n"
            ),
        }
    )

    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["actor"].endswith("global_writer")
    assert result["accesses"][0]["operation"] == "path.write_text"
    assert result["escapes"] == []


@pytest.mark.parametrize(
    "lambda_body",
    [
        "lambda state_module: state_module.STATE_FILE.exists()",
        "lambda: [STATE_FILE for STATE_FILE in values]",
    ],
)
def test_lambda_capture_scan_respects_nested_lexical_bindings(
    lambda_body: str,
) -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/consumer.py": (
                "from chronovisor.state import STATE_FILE\n"
                "import chronovisor.state as state_module\n"
                "def deferred(values):\n"
                f"    callback = {lambda_body}\n"
            ),
        }
    )

    assert result["accesses"] == []
    assert result["escapes"] == []
