from __future__ import annotations

from scripts.runtime_ownership.access import discover_access_facts


def _discover(sources: dict[str, str]) -> dict:
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


def test_direct_repo_star_import_exposes_known_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "from chronovisor.state import STATE_FILE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import *\n"
                "def save():\n"
                "    STATE_FILE.write_text('value')\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:save"
    assert result["escapes"] == []


def test_two_hop_star_reexport_exposes_known_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/bridge_a.py": ("from chronovisor.state import *\n"),
            "src/chronovisor/bridge_b.py": ("from chronovisor.bridge_a import *\n"),
            "src/chronovisor/consumer.py": (
                "from chronovisor.bridge_b import *\n"
                "def save():\n"
                "    STATE_FILE.write_text('value')\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:save"
    assert result["escapes"] == []


def test_static_literal_all_limits_star_exports() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "from chronovisor.state import STATE_FILE\n"
                "PUBLIC_STATE = STATE_FILE\n"
                "_PRIVATE_STATE = STATE_FILE\n"
                "__all__ = ['PUBLIC_STATE']\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import *\n"
                "def use_exports():\n"
                "    PUBLIC_STATE.write_text('value')\n"
                "    _PRIVATE_STATE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert result["accesses"][0]["actor"] == "chronovisor.consumer:use_exports"
    assert result["escapes"] == []


def test_dynamic_all_exposes_known_may_values_and_escapes() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "from chronovisor.state import STATE_FILE\n"
                "PUBLIC_STATE = STATE_FILE\n"
                "__all__ = choose_exports()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import *\n"
                "def save():\n"
                "    PUBLIC_STATE.write_text('value')\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert len(result["escapes"]) == 1
    assert result["escapes"][0]["reason"] == "dynamic_star_import"


def test_module_if_branch_import_contributes_may_export() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "FLAG = object()\n"
                "if FLAG:\n"
                "    from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
                "else:\n"
                "    EXPORTED_STATE = object()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "def save():\n"
                "    EXPORTED_STATE.write_text('value')\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_module_try_branch_import_contributes_may_export() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "try:\n"
                "    from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
                "except Exception:\n"
                "    EXPORTED_STATE = object()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "def save():\n"
                "    EXPORTED_STATE.write_text('value')\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_module_match_branch_import_contributes_may_export() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "subject = object()\n"
                "match subject:\n"
                "    case 0:\n"
                "        from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
                "    case _:\n"
                "        EXPORTED_STATE = object()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "def save():\n"
                "    EXPORTED_STATE.write_text('value')\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []
