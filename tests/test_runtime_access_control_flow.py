from __future__ import annotations

from typing import Any

import pytest

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


def test_try_handler_receives_partial_try_state() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def risky():\n"
            "    return None\n"
            "def recover():\n"
            "    path = object()\n"
            "    try:\n"
            "        path = STATE_FILE\n"
            "        risky()\n"
            "    except Exception:\n"
            "        path.write_text('recovered-resource')\n"
        )
    )

    assert len(joined_access_rows(result)) == 1
    assert joined_access_rows(result)[0]["operation"] == "path.write_text"
    assert joined_escape_rows(result) == []


def test_try_handler_keeps_origin_from_prefix_before_later_kill() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def risky():\n"
            "    return None\n"
            "def recover():\n"
            "    path = object()\n"
            "    try:\n"
            "        path = STATE_FILE\n"
            "        risky()\n"
            "        path = object()\n"
            "    except Exception:\n"
            "        path.write_text('recovered-resource')\n"
        )
    )

    assert len(joined_access_rows(result)) == 1
    assert joined_access_rows(result)[0]["operation"] == "path.write_text"
    assert joined_escape_rows(result) == []


@pytest.mark.parametrize("loop_kind", ["for", "while"])
def test_loop_body_reaches_carried_origin_on_later_iteration(loop_kind: str) -> None:
    if loop_kind == "for":
        loop = (
            "    for _item in items:\n"
            "        path.write_text('later-iteration')\n"
            "        path = STATE_FILE\n"
        )
        signature = "def carried(items):\n"
    else:
        loop = (
            "    while keep_running:\n"
            "        path.write_text('later-iteration')\n"
            "        path = STATE_FILE\n"
        )
        signature = "def carried(keep_running):\n"
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            + signature
            + "    path = object()\n"
            + loop
        )
    )

    assert len(joined_access_rows(result)) == 1
    assert joined_access_rows(result)[0]["operation"] == "path.write_text"
    assert joined_escape_rows(result) == []


def test_try_else_and_finally_keep_normal_and_exceptional_state() -> None:
    result = _access_fixture(
        consumer=(
            "from chronovisor.state import STATE_FILE\n"
            "def risky():\n"
            "    return None\n"
            "def guarded():\n"
            "    path = object()\n"
            "    try:\n"
            "        path = STATE_FILE\n"
            "        risky()\n"
            "    except Exception:\n"
            "        path.read_text()\n"
            "    else:\n"
            "        path.write_text('normal-resource')\n"
            "    finally:\n"
            "        path.exists()\n"
        )
    )

    assert {row["operation"] for row in joined_access_rows(result)} == {
        "path.read_text",
        "path.write_text",
        "path.exists",
    }
    assert joined_escape_rows(result) == []
