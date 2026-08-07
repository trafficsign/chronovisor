from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from scripts.runtime_ownership.access import discover_access_facts
from tests.runtime_access_v2_helpers import (
    joined_access_rows,
    joined_escape_rows,
)


def _candidate(symbol: str) -> dict[str, Any]:
    return {
        "id": f"runtime-resource:{symbol.lower()}",
        "module": "chronovisor.resources",
        "symbol": symbol,
        "locator": {
            "type": "path",
            "value": f"$ROOT/{symbol.lower()}.json",
        },
    }


def _discover(
    consumer: str,
    *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    extra_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    selected = list(candidates or [_candidate("STATE_FILE")])
    symbols = sorted(str(row["symbol"]) for row in selected)
    sources = {
        "src/chronovisor/resources.py": "".join(
            f"{symbol} = object()\n" for symbol in symbols
        ),
        "src/chronovisor/consumer.py": consumer,
        **dict(extra_sources or {}),
    }
    return discover_access_facts(
        {path: source.encode() for path, source in sources.items()}, selected
    )


def test_path_division_preserves_origin_without_generic_escape() -> None:
    result = _discover(
        "from chronovisor.resources import STATE_FILE\n"
        "def writer():\n"
        "    child = STATE_FILE / 'child'\n"
        "    child.write_text('value')\n"
    )

    assert [row["operation"] for row in joined_access_rows(result)] == [
        "path.write_text"
    ]
    assert any(
        step == "transform:truediv"
        for step in joined_access_rows(result)[0]["binding_chain"]
    )
    assert joined_escape_rows(result) == []


def test_path_division_with_registered_right_operand_fails_closed() -> None:
    result = _discover(
        "from chronovisor.resources import STATE_FILE\n"
        "def ambiguous():\n"
        "    return 'prefix' / STATE_FILE\n"
    )

    assert joined_access_rows(result) == []
    assert len(joined_escape_rows(result)) == 1
    assert joined_escape_rows(result)[0]["reason"] == (
        "ambiguous_registered_origin_path_division"
    )


def test_with_name_preserves_path_origin() -> None:
    result = _discover(
        "from chronovisor.resources import STATE_FILE\n"
        "def writer():\n"
        "    STATE_FILE.with_name('other.json').write_text('value')\n"
    )

    assert [row["operation"] for row in joined_access_rows(result)] == [
        "path.write_text"
    ]
    assert "transform:with_name" in joined_access_rows(result)[0]["binding_chain"]
    assert joined_escape_rows(result) == []


@pytest.mark.parametrize("method", ["rename", "replace"])
def test_path_move_records_source_destination_and_returns_destination(
    method: str,
) -> None:
    result = _discover(
        "from chronovisor.resources import SOURCE, DESTINATION\n"
        "def move():\n"
        f"    moved = SOURCE.{method}(DESTINATION)\n"
        "    moved.write_text('after')\n",
        candidates=[_candidate("SOURCE"), _candidate("DESTINATION")],
    )

    facts = {
        (row["operation"], row["resource_id"], row["mode"])
        for row in result["access_facts"]
    }
    assert facts == {
        (f"path.{method}", "runtime-resource:source", "write"),
        (
            f"path.{method}.destination",
            "runtime-resource:destination",
            "write",
        ),
        ("path.write_text", "runtime-resource:destination", "write"),
    }
    assert result["escape_facts"] == []


def test_path_move_opaque_destination_fails_closed() -> None:
    result = _discover(
        "from chronovisor.resources import SOURCE, DESTINATION\n"
        "def move():\n"
        "    SOURCE.rename(*(DESTINATION,))\n",
        candidates=[_candidate("SOURCE"), _candidate("DESTINATION")],
    )

    assert any(row["operation"] == "path.rename" for row in joined_access_rows(result))
    assert any(
        row["reason"] == "ambiguous_registered_origin_path_destination"
        for row in joined_escape_rows(result)
    )


def test_os_open_modes_and_returned_fd_keep_path_provenance() -> None:
    result = _discover(
        "import os\n"
        "from chronovisor.resources import STATE_FILE\n"
        "def open_all():\n"
        "    read_fd = os.open(STATE_FILE, os.O_RDONLY)\n"
        "    os.open(STATE_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)\n"
        "    os.open(STATE_FILE, os.O_RDWR | os.O_APPEND)\n"
        "    opaque(read_fd)\n"
    )

    assert {
        (row["mode"], row["operation"])
        for row in joined_access_rows(result)
    } == {
        ("read", "os.open:O_RDONLY"),
        ("write", "os.open:O_WRONLY|O_CREAT|O_TRUNC"),
        ("read_write", "os.open:O_RDWR|O_APPEND"),
    }
    fd_escape = next(
        row
        for row in joined_escape_rows(result)
        if row["reason"] == "registered_locator_to_unknown_callee"
    )
    assert "result:os.open:fd" in fd_escape["binding_chain"]


def test_os_open_unknown_flags_fail_closed() -> None:
    result = _discover(
        "import os\n"
        "from chronovisor.resources import STATE_FILE\n"
        "def dynamic(flags):\n"
        "    return os.open(STATE_FILE, flags)\n"
    )

    assert joined_access_rows(result) == []
    assert len(joined_escape_rows(result)) == 1
    assert joined_escape_rows(result)[0]["reason"] == "dynamic_os_open_flags"
    assert joined_escape_rows(result)[0]["operation"] == "os.open"


@pytest.mark.parametrize("keyword", ["mode", "dir_fd"])
def test_os_open_registered_auxiliary_keyword_fails_closed(
    keyword: str,
) -> None:
    result = _discover(
        "import os\n"
        "from chronovisor.resources import STATE_FILE, AUXILIARY\n"
        "def open_with_origin():\n"
        f"    os.open(STATE_FILE, os.O_RDONLY, {keyword}=AUXILIARY)\n",
        candidates=[_candidate("STATE_FILE"), _candidate("AUXILIARY")],
    )

    assert result["access_facts"] == []
    assert {
        (row["resource_id"], row["reason"])
        for row in result["escape_facts"]
    } == {
        (
            "runtime-resource:state_file",
            "ambiguous_registered_origin_os_open_arguments",
        ),
        (
            "runtime-resource:auxiliary",
            "ambiguous_registered_origin_os_open_arguments",
        ),
    }


def test_os_open_non_origin_auxiliary_values_remain_concrete() -> None:
    result = _discover(
        "import os\n"
        "from chronovisor.resources import STATE_FILE\n"
        "def open_with_auxiliary_values():\n"
        "    os.open(STATE_FILE, os.O_WRONLY, mode=0o600, dir_fd=None)\n"
    )

    assert [row["operation"] for row in result["access_facts"]] == [
        "os.open:O_WRONLY"
    ]
    assert result["escape_facts"] == []


def test_os_open_fd_proxy_is_not_dispatched_as_a_path_receiver() -> None:
    result = _discover(
        "import os\n"
        "from chronovisor.resources import STATE_FILE\n"
        "def misuse_fd():\n"
        "    descriptor = os.open(STATE_FILE, os.O_WRONLY)\n"
        "    descriptor.write_text('not a Path')\n"
    )

    assert [row["operation"] for row in result["access_facts"]] == [
        "os.open:O_WRONLY"
    ]
    assert len(result["escape_facts"]) == 1
    escape = result["escape_facts"][0]
    assert escape["reason"] == "registered_locator_to_unknown_callee"
    assert "result:os.open:fd" in joined_escape_rows(result)[0]["binding_chain"]


@pytest.mark.parametrize(
    "consumer",
    [
        "import os\n"
        "from chronovisor.resources import STATE_FILE\n"
        "def caller():\n"
        "    os.open(STATE_FILE, 0)\n",
        "from os import open as local_open, O_RDONLY\n"
        "from chronovisor.resources import STATE_FILE\n"
        "def caller():\n"
        "    local_open(STATE_FILE, O_RDONLY)\n",
    ],
)
def test_repository_local_os_module_is_not_a_concrete_stdlib_sink(
    consumer: str,
) -> None:
    result = _discover(
        consumer,
        extra_sources={
            "src/os.py": (
                "O_RDONLY = 0\n"
                "def open(path, flags):\n"
                "    path.write_text('local implementation')\n"
            )
        },
    )

    assert [row["operation"] for row in result["access_facts"]] == [
        "path.write_text"
    ]
    assert not any(
        row["sink"] == "os.open" for row in result["access_facts"]
    )
    assert result["escape_facts"] == []


def test_os_replace_and_rename_record_both_paths_through_module_alias() -> None:
    result = _discover(
        "import os as operating_system\n"
        "from chronovisor.resources import SOURCE, DESTINATION\n"
        "def move():\n"
        "    operating_system.replace(SOURCE, DESTINATION)\n"
        "    operating_system.rename(SOURCE, DESTINATION)\n",
        candidates=[_candidate("SOURCE"), _candidate("DESTINATION")],
    )

    assert {
        (row["operation"], row["resource_id"], row["mode"])
        for row in joined_access_rows(result)
    } == {
        ("os.replace.source", "runtime-resource:source", "read_write"),
        ("os.replace.destination", "runtime-resource:destination", "write"),
        ("os.rename.source", "runtime-resource:source", "read_write"),
        ("os.rename.destination", "runtime-resource:destination", "write"),
    }
    assert joined_escape_rows(result) == []


def test_from_os_import_alias_resolves_open_and_flags() -> None:
    result = _discover(
        "from os import open as low_open, O_WRONLY as WRITE_ONLY\n"
        "from chronovisor.resources import STATE_FILE\n"
        "def writer():\n"
        "    low_open(STATE_FILE, WRITE_ONLY)\n"
    )

    assert len(joined_access_rows(result)) == 1
    assert joined_access_rows(result)[0]["operation"] == "os.open:O_WRONLY"
    assert joined_escape_rows(result) == []


def test_os_local_and_external_shadows_are_not_concrete_sinks() -> None:
    result = _discover(
        "import external_os as operating_system\n"
        "from os import replace as move\n"
        "from chronovisor.resources import SOURCE, DESTINATION\n"
        "def local_shadow(operating_system):\n"
        "    operating_system.open(SOURCE, 0)\n"
        "def imported_shadow(move):\n"
        "    move(SOURCE, DESTINATION)\n"
        "def external_module():\n"
        "    operating_system.replace(SOURCE, DESTINATION)\n",
        candidates=[_candidate("SOURCE"), _candidate("DESTINATION")],
    )

    assert joined_access_rows(result) == []
    assert joined_escape_rows(result)
    assert not any(
        row["operation"].startswith("os.") for row in result["escape_facts"]
    )


def test_concrete_sink_v2_ids_ignore_line_shift() -> None:
    source = (
        "import os\n"
        "from chronovisor.resources import SOURCE, DESTINATION\n"
        "def sinks():\n"
        "    (SOURCE / 'child').write_text('value')\n"
        "    os.replace(SOURCE, DESTINATION)\n"
    )
    candidates = [_candidate("SOURCE"), _candidate("DESTINATION")]

    base = _discover(source, candidates=candidates)
    shifted = _discover("\n\n" + source, candidates=candidates)

    assert len(base["access_fact_ids"]) == 3
    assert shifted["access_fact_ids"] == base["access_fact_ids"]
    assert shifted["provenance_ids"] == base["provenance_ids"]
    assert shifted["escape_fact_ids"] == base["escape_fact_ids"]
