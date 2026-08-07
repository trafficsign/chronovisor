from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from scripts.runtime_ownership.access import discover_access_facts


def _candidate(symbol: str) -> dict[str, Any]:
    return {
        "id": f"runtime-resource:{symbol.lower()}",
        "module": "chronovisor.resources",
        "symbol": symbol,
        "locator": {
            "type": "path",
            "value": f"$ROOT/{symbol.lower()}.lock",
        },
    }


def _discover(
    consumer: str,
    *,
    symbols: Sequence[str] = ("LOCK_PATH",),
    extra_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    sources = {
        "src/chronovisor/resources.py": "".join(
            f"{symbol} = object()\n" for symbol in sorted(symbols)
        ),
        "src/chronovisor/consumer.py": consumer,
        **dict(extra_sources or {}),
    }
    return discover_access_facts(
        {path: source.encode() for path, source in sources.items()},
        [_candidate(symbol) for symbol in symbols],
    )


def _access_operations(result: Mapping[str, Any]) -> set[str]:
    return {str(row["operation"]) for row in result["access_facts"]}


def _escape_reasons(result: Mapping[str, Any]) -> set[str]:
    return {str(row["reason"]) for row in result["escape_facts"]}


def test_open_variants_return_handles_and_fileno_returns_descriptors() -> None:
    result = _discover(
        "import builtins\n"
        "import fcntl\n"
        "from builtins import open as imported_open\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def exercise():\n"
        "    with LOCK_PATH.open('r') as first:\n"
        "        fcntl.flock(first, fcntl.LOCK_SH)\n"
        "    with open(file=LOCK_PATH, mode='w') as second:\n"
        "        fcntl.flock(second.fileno(), fcntl.LOCK_EX)\n"
        "    with builtins.open(LOCK_PATH, 'a+') as third:\n"
        "        fcntl.flock(third, fcntl.LOCK_UN)\n"
        "    with imported_open(LOCK_PATH, 'x') as fourth:\n"
        "        fcntl.flock(\n"
        "            fourth.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB\n"
        "        )\n"
    )

    assert _access_operations(result) == {
        "path.open:r",
        "builtin.open:w",
        "builtin.open:a+",
        "builtin.open:x",
        "fcntl.flock:shared",
        "fcntl.flock:exclusive",
        "fcntl.flock:unlock",
        "fcntl.flock:exclusive_nonblocking",
    }
    assert result["escape_facts"] == []


def test_open_literal_keyword_unpack_and_each_argument_evaluate_once() -> None:
    result = _discover(
        "from chronovisor.resources import AUXILIARY, LOCK_PATH\n"
        "def encoding(value):\n"
        "    value.read_text()\n"
        "    return 'utf-8'\n"
        "def exercise():\n"
        "    open(**{\n"
        "        'file': LOCK_PATH,\n"
        "        'mode': 'r',\n"
        "        'encoding': encoding(AUXILIARY),\n"
        "    })\n"
        "    LOCK_PATH.open(**{'mode': 'a'})\n",
        symbols=("AUXILIARY", "LOCK_PATH"),
    )

    assert {
        (row["resource_id"], row["operation"])
        for row in result["access_facts"]
    } == {
        ("runtime-resource:auxiliary", "path.read_text"),
        ("runtime-resource:lock_path", "path.open:a"),
    }
    assert _escape_reasons(result) == {"invalid_or_ambiguous_open_options"}


@pytest.mark.parametrize(
    "call",
    [
        "open(LOCK_PATH, file=LOCK_PATH)",
        "open(*(LOCK_PATH,))",
        "open(**{'file': LOCK_PATH, 'file': LOCK_PATH})",
        "LOCK_PATH.open(unknown=LOCK_PATH)",
    ],
)
def test_open_invalid_or_ambiguous_signatures_fail_closed(call: str) -> None:
    result = _discover(
        "from chronovisor.resources import LOCK_PATH\n"
        "def exercise():\n"
        f"    {call}\n"
    )

    assert not any(
        row["operation"].startswith(("builtin.open", "path.open"))
        for row in result["access_facts"]
    )
    assert "invalid_or_ambiguous_open_signature" in _escape_reasons(result)


def test_dynamic_invalid_mode_and_opener_do_not_return_handles() -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import DYNAMIC, INVALID, OPENER\n"
        "def custom_opener(path, flags):\n"
        "    return 1\n"
        "def exercise(mode):\n"
        "    dynamic = open(DYNAMIC, mode)\n"
        "    fcntl.flock(dynamic, fcntl.LOCK_EX)\n"
        "    invalid = open(INVALID, 'rw')\n"
        "    fcntl.flock(invalid, fcntl.LOCK_EX)\n"
        "    unsupported = open(OPENER, 'r', opener=custom_opener)\n"
        "    fcntl.flock(unsupported, fcntl.LOCK_EX)\n",
        symbols=("DYNAMIC", "INVALID", "OPENER"),
    )

    assert result["access_facts"] == []
    assert _escape_reasons(result) == {
        "dynamic_open_mode",
        "invalid_open_mode",
        "unsupported_open_opener",
    }


def test_open_auxiliary_origins_are_reported_separately() -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import AUXILIARY, LOCK_PATH\n"
        "def static_auxiliary():\n"
        "    open(LOCK_PATH, 'r', encoding=AUXILIARY)\n"
        "def dynamic_mode():\n"
        "    handle = open(LOCK_PATH, AUXILIARY)\n"
        "    fcntl.flock(handle, fcntl.LOCK_SH)\n",
        symbols=("AUXILIARY", "LOCK_PATH"),
    )

    assert result["access_facts"] == []
    assert {
        (row["resource_id"], row["reason"])
        for row in result["escape_facts"]
    } == {
        (
            "runtime-resource:auxiliary",
            "ambiguous_registered_origin_open_arguments",
        ),
        ("runtime-resource:lock_path", "dynamic_open_mode"),
        (
            "runtime-resource:lock_path",
            "invalid_or_ambiguous_open_options",
        ),
    }


def test_os_fdopen_roundtrip_preserves_fd_and_avoids_duplicate_open_access() -> None:
    result = _discover(
        "import fcntl\n"
        "import os\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def exercise():\n"
        "    descriptor = os.open(LOCK_PATH, os.O_RDWR)\n"
        "    with os.fdopen(descriptor, 'a+', encoding='utf-8') as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)\n"
    )

    assert _access_operations(result) == {
        "os.open:O_RDWR",
        "fcntl.flock:exclusive",
        "fcntl.flock:unlock",
    }
    assert result["escape_facts"] == []


@pytest.mark.parametrize("mode", ["None", "1", "b'r'", "False"])
def test_static_non_string_open_modes_are_invalid(mode: str) -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def exercise():\n"
        f"    with open(LOCK_PATH, {mode}) as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_EX)\n"
    )

    assert result["access_facts"] == []
    assert _escape_reasons(result) == {"invalid_open_mode"}


def test_open_options_require_static_runtime_compatibility() -> None:
    result = _discover(
        "import fcntl\n"
        "import os\n"
        "from chronovisor.resources import (\n"
        "    BINARY_ENCODING, CLOSEFD_PATH, DYNAMIC_BUFFERING, FD_PATH,\n"
        "    FDOPEN_PATH, TEXT_BUFFERING, VALID_BINARY, VALID_TEXT,\n"
        ")\n"
        "def exercise(buffering):\n"
        "    with open(CLOSEFD_PATH, closefd=False) as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "    with open(TEXT_BUFFERING, 'r', buffering=0) as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "    with open(BINARY_ENCODING, 'rb', encoding='utf-8') as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "    with open(DYNAMIC_BUFFERING, 'r', buffering=buffering) as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "    with open(\n"
        "        VALID_TEXT, 'r', encoding='utf-8', errors='strict',\n"
        "        newline='\\n', closefd=True,\n"
        "    ) as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_SH)\n"
        "    with open(VALID_BINARY, 'rb', buffering=0) as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "    descriptor = os.open(FD_PATH, os.O_RDWR)\n"
        "    with open(descriptor, 'rb', buffering=0, closefd=False) as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_UN)\n"
        "    descriptor = os.open(FDOPEN_PATH, os.O_RDWR)\n"
        "    with os.fdopen(descriptor, 'r', closefd=False) as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_SH)\n",
        symbols=(
            "BINARY_ENCODING",
            "CLOSEFD_PATH",
            "DYNAMIC_BUFFERING",
            "FD_PATH",
            "FDOPEN_PATH",
            "TEXT_BUFFERING",
            "VALID_BINARY",
            "VALID_TEXT",
        ),
    )

    assert {
        (row["resource_id"], row["operation"])
        for row in result["access_facts"]
    } == {
        ("runtime-resource:fd_path", "os.open:O_RDWR"),
        ("runtime-resource:fd_path", "fcntl.flock:unlock"),
        ("runtime-resource:fdopen_path", "os.open:O_RDWR"),
        ("runtime-resource:fdopen_path", "fcntl.flock:shared"),
        ("runtime-resource:valid_binary", "builtin.open:rb"),
        ("runtime-resource:valid_binary", "fcntl.flock:exclusive"),
        ("runtime-resource:valid_text", "builtin.open:r"),
        ("runtime-resource:valid_text", "fcntl.flock:shared"),
    }
    assert _escape_reasons(result) == {"invalid_or_ambiguous_open_options"}


def test_signed_buffering_literals_are_precise_but_other_unary_values_are_not() -> None:
    result = _discover(
        "import fcntl\n"
        "import os\n"
        "from chronovisor.resources import (\n"
        "    BUILTIN, FDOPEN, INVALID_FLOAT, INVALID_UNARY, PATH_OPEN,\n"
        ")\n"
        "def exercise():\n"
        "    with open(BUILTIN, buffering=-1) as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_SH)\n"
        "    with PATH_OPEN.open(buffering=-1) as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "    descriptor = os.open(FDOPEN, os.O_RDWR)\n"
        "    with os.fdopen(\n"
        "        descriptor, buffering=-1, closefd=False\n"
        "    ) as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_UN)\n"
        "    open(INVALID_UNARY, buffering=~1)\n"
        "    open(INVALID_FLOAT, buffering=-1.0)\n",
        symbols=(
            "BUILTIN",
            "FDOPEN",
            "INVALID_FLOAT",
            "INVALID_UNARY",
            "PATH_OPEN",
        ),
    )

    assert {
        (row["resource_id"], row["operation"])
        for row in result["access_facts"]
    } == {
        ("runtime-resource:builtin", "builtin.open:r"),
        ("runtime-resource:builtin", "fcntl.flock:shared"),
        ("runtime-resource:fdopen", "os.open:O_RDWR"),
        ("runtime-resource:fdopen", "fcntl.flock:unlock"),
        ("runtime-resource:path_open", "path.open:r"),
        ("runtime-resource:path_open", "fcntl.flock:exclusive"),
    }
    assert _escape_reasons(result) == {"invalid_or_ambiguous_open_options"}


def test_close_and_context_exit_invalidate_handle_liveness() -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import CLOSED, CONTEXT, LIVE\n"
        "def exercise():\n"
        "    handle = open(CLOSED)\n"
        "    alias = handle\n"
        "    descriptor = handle.fileno()\n"
        "    handle.close()\n"
        "    fcntl.flock(alias, fcntl.LOCK_EX)\n"
        "    fcntl.flock(descriptor, fcntl.LOCK_EX)\n"
        "    after = handle.fileno()\n"
        "    fcntl.flock(after, fcntl.LOCK_EX)\n"
        "    with open(CONTEXT) as scoped:\n"
        "        pass\n"
        "    fcntl.flock(scoped, fcntl.LOCK_EX)\n"
        "    live = open(LIVE)\n"
        "    fcntl.flock(live, fcntl.LOCK_SH)\n",
        symbols=("CLOSED", "CONTEXT", "LIVE"),
    )

    assert {
        (row["resource_id"], row["operation"])
        for row in result["access_facts"]
    } == {
        ("runtime-resource:closed", "builtin.open:r"),
        ("runtime-resource:context", "builtin.open:r"),
        ("runtime-resource:live", "builtin.open:r"),
        ("runtime-resource:live", "fcntl.flock:shared"),
    }
    assert "ambiguous_registered_origin_flock_descriptor" in _escape_reasons(
        result
    )


def test_bound_file_methods_preserve_receiver_lifecycle_effects() -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import (\n"
        "    CALLBACK_CLOSE, DIRECT_CLOSE, FILENO, LIVE, UNKNOWN_METHOD,\n"
        ")\n"
        "def exercise(callback):\n"
        "    direct = open(DIRECT_CLOSE)\n"
        "    alias = direct\n"
        "    descriptor = direct.fileno()\n"
        "    extracted_close = direct.close\n"
        "    extracted_close()\n"
        "    fcntl.flock(alias, fcntl.LOCK_EX)\n"
        "    fcntl.flock(descriptor, fcntl.LOCK_EX)\n"
        "    via_callback = open(CALLBACK_CLOSE)\n"
        "    callback(via_callback.close)\n"
        "    fcntl.flock(via_callback, fcntl.LOCK_EX)\n"
        "    fileno_handle = open(FILENO)\n"
        "    extracted_fileno = fileno_handle.fileno\n"
        "    extracted_fileno()\n"
        "    callback(fileno_handle.fileno)\n"
        "    fcntl.flock(fileno_handle, fcntl.LOCK_SH)\n"
        "    unknown = open(UNKNOWN_METHOD)\n"
        "    callback(unknown.arbitrary)\n"
        "    fcntl.flock(unknown, fcntl.LOCK_EX)\n"
        "    live = open(LIVE)\n"
        "    fcntl.flock(live, fcntl.LOCK_SH)\n",
        symbols=(
            "CALLBACK_CLOSE",
            "DIRECT_CLOSE",
            "FILENO",
            "LIVE",
            "UNKNOWN_METHOD",
        ),
    )

    assert {
        (row["resource_id"], row["operation"])
        for row in result["access_facts"]
        if row["operation"].startswith("fcntl.flock")
    } == {
        ("runtime-resource:fileno", "fcntl.flock:shared"),
        ("runtime-resource:live", "fcntl.flock:shared"),
    }
    assert {
        row["resource_id"]
        for row in result["escape_facts"]
        if row["reason"] == "ambiguous_registered_origin_flock_descriptor"
    } == {
        "runtime-resource:callback_close",
        "runtime-resource:direct_close",
        "runtime-resource:unknown_method",
    }


def test_fdopen_closefd_ownership_controls_underlying_descriptor_liveness() -> None:
    result = _discover(
        "import fcntl\n"
        "import os\n"
        "from chronovisor.resources import (\n"
        "    CONTEXT_OWNED, CONTEXT_UNOWNED, DIRECT_OWNED, DIRECT_UNOWNED,\n"
        ")\n"
        "def exercise():\n"
        "    direct_owned = os.open(DIRECT_OWNED, os.O_RDWR)\n"
        "    owned_handle = os.fdopen(direct_owned, 'r')\n"
        "    owned_handle.close()\n"
        "    fcntl.flock(direct_owned, fcntl.LOCK_EX)\n"
        "    direct_unowned = os.open(DIRECT_UNOWNED, os.O_RDWR)\n"
        "    unowned_handle = os.fdopen(\n"
        "        direct_unowned, 'r', closefd=False\n"
        "    )\n"
        "    unowned_handle.close()\n"
        "    fcntl.flock(direct_unowned, fcntl.LOCK_SH)\n"
        "    context_owned = os.open(CONTEXT_OWNED, os.O_RDWR)\n"
        "    with os.fdopen(context_owned, 'r'):\n"
        "        pass\n"
        "    fcntl.flock(context_owned, fcntl.LOCK_EX)\n"
        "    context_unowned = os.open(CONTEXT_UNOWNED, os.O_RDWR)\n"
        "    with os.fdopen(context_unowned, 'r', closefd=False):\n"
        "        pass\n"
        "    fcntl.flock(context_unowned, fcntl.LOCK_SH)\n",
        symbols=(
            "CONTEXT_OWNED",
            "CONTEXT_UNOWNED",
            "DIRECT_OWNED",
            "DIRECT_UNOWNED",
        ),
    )

    assert {
        (row["resource_id"], row["operation"])
        for row in result["access_facts"]
        if row["operation"].startswith("fcntl.flock")
    } == {
        ("runtime-resource:context_unowned", "fcntl.flock:shared"),
        ("runtime-resource:direct_unowned", "fcntl.flock:shared"),
    }
    assert {
        row["resource_id"]
        for row in result["escape_facts"]
        if row["reason"] == "ambiguous_registered_origin_flock_descriptor"
    } == {
        "runtime-resource:context_owned",
        "runtime-resource:direct_owned",
    }


def test_class_attributes_and_allocation_activations_have_scoped_liveness() -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import CLASS_CLOSED, CLASS_LIVE, FACTORY\n"
        "class Holder:\n"
        "    def prepare(self):\n"
        "        self.closed = open(CLASS_CLOSED)\n"
        "        self.live = open(CLASS_LIVE)\n"
        "    def close(self):\n"
        "        self.closed.close()\n"
        "    def use(self):\n"
        "        fcntl.flock(self.closed, fcntl.LOCK_EX)\n"
        "        fcntl.flock(self.live, fcntl.LOCK_SH)\n"
        "def make():\n"
        "    return open(FACTORY)\n"
        "def exercise():\n"
        "    holder = Holder()\n"
        "    holder.prepare()\n"
        "    holder.close()\n"
        "    holder.use()\n"
        "    first = make()\n"
        "    second = make()\n"
        "    first.close()\n"
        "    fcntl.flock(first, fcntl.LOCK_EX)\n"
        "    fcntl.flock(second, fcntl.LOCK_SH)\n"
        "exercise()\n",
        symbols=("CLASS_CLOSED", "CLASS_LIVE", "FACTORY"),
    )

    assert {
        (row["resource_id"], row["operation"])
        for row in result["access_facts"]
        if row["operation"].startswith("fcntl.flock")
    } == {
        ("runtime-resource:class_live", "fcntl.flock:shared"),
        ("runtime-resource:factory", "fcntl.flock:shared"),
    }
    assert {
        row["resource_id"]
        for row in result["escape_facts"]
        if row["reason"] == "ambiguous_registered_origin_flock_descriptor"
    } == {
        "runtime-resource:class_closed",
        "runtime-resource:factory",
    }


def test_unknown_callbacks_and_os_close_contaminate_only_passed_objects() -> None:
    result = _discover(
        "import fcntl\n"
        "import os\n"
        "from chronovisor.resources import CLOSED_FD, FD, HANDLE, LIVE\n"
        "def exercise(callback):\n"
        "    handle = open(HANDLE)\n"
        "    live = open(LIVE)\n"
        "    callback(handle)\n"
        "    fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "    fcntl.flock(live, fcntl.LOCK_SH)\n"
        "    descriptor = os.open(FD, os.O_RDWR)\n"
        "    callback(descriptor)\n"
        "    fcntl.flock(descriptor, fcntl.LOCK_EX)\n"
        "    closed = os.open(CLOSED_FD, os.O_RDWR)\n"
        "    os.close(closed)\n"
        "    fcntl.flock(closed, fcntl.LOCK_EX)\n",
        symbols=("CLOSED_FD", "FD", "HANDLE", "LIVE"),
    )

    assert {
        (row["resource_id"], row["operation"])
        for row in result["access_facts"]
    } == {
        ("runtime-resource:closed_fd", "os.open:O_RDWR"),
        ("runtime-resource:fd", "os.open:O_RDWR"),
        ("runtime-resource:handle", "builtin.open:r"),
        ("runtime-resource:live", "builtin.open:r"),
        ("runtime-resource:live", "fcntl.flock:shared"),
    }
    assert _escape_reasons(result) == {
        "ambiguous_registered_origin_flock_descriptor",
        "registered_locator_to_unknown_callee",
    }


def test_file_handle_close_unknown_methods_and_extracted_methods_stay_non_path() -> None:
    result = _discover(
        "from chronovisor.resources import LOCK_PATH\n"
        "def exercise():\n"
        "    handle = open(LOCK_PATH)\n"
        "    handle.close()\n"
        "    handle.write_text('not a Path')\n"
        "    extracted = handle.fileno\n"
        "    extracted()\n"
        "    handle.fileno(LOCK_PATH)\n"
    )

    assert _access_operations(result) == {"builtin.open:r"}
    assert not any(
        row["operation"] == "path.write_text"
        for row in result["access_facts"]
    )
    assert {row["reason"] for row in result["escape_facts"]} == {
        "registered_locator_to_unknown_callee",
    }


def test_file_context_is_non_suppressing_and_async_body_is_unreachable() -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import ASYNC_PATH, LOCK_PATH\n"
        "def synchronous():\n"
        "    try:\n"
        "        with open(LOCK_PATH, 'a+') as handle:\n"
        "            fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "            raise RuntimeError('rollback')\n"
        "    except RuntimeError:\n"
        "        pass\n"
        "async def asynchronous():\n"
        "    async with open(ASYNC_PATH) as handle:\n"
        "        ASYNC_PATH.write_text('unreachable')\n",
        symbols=("ASYNC_PATH", "LOCK_PATH"),
    )

    assert _access_operations(result) == {
        "builtin.open:a+",
        "builtin.open:r",
        "fcntl.flock:exclusive",
    }
    assert not any(
        row["operation"] == "path.write_text"
        for row in result["access_facts"]
    )
    assert {
        (row["resource_id"], row["reason"])
        for row in result["escape_facts"]
    } == {
        (
            "runtime-resource:async_path",
            "unsupported_registered_origin_control_flow",
        )
    }


def test_flock_exact_masks_and_alternative_joins_have_precise_modes() -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def exercise(flag):\n"
        "    handle = open(LOCK_PATH, 'a+')\n"
        "    fcntl.flock(handle, fcntl.LOCK_SH)\n"
        "    fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "    fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)\n"
        "    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    fcntl.flock(handle, fcntl.LOCK_UN)\n"
        "    fcntl.flock(\n"
        "        handle, fcntl.LOCK_SH if flag else fcntl.LOCK_EX\n"
        "    )\n"
        "    fcntl.flock(\n"
        "        handle,\n"
        "        (fcntl.LOCK_SH | fcntl.LOCK_NB)\n"
        "        if flag\n"
        "        else (fcntl.LOCK_EX | fcntl.LOCK_NB),\n"
        "    )\n"
    )

    assert {
        (row["mode"], row["operation"])
        for row in result["access_facts"]
        if row["sink"] == "fcntl.flock"
    } == {
        ("read", "fcntl.flock:shared"),
        ("write", "fcntl.flock:exclusive"),
        ("read", "fcntl.flock:shared_nonblocking"),
        ("write", "fcntl.flock:exclusive_nonblocking"),
        ("read_write", "fcntl.flock:unlock"),
        ("read_write", "fcntl.flock:shared_or_exclusive"),
        (
            "read_write",
            "fcntl.flock:shared_or_exclusive_nonblocking",
        ),
    }
    assert result["escape_facts"] == []


def test_flock_helper_closure_class_and_branch_propagation_stays_exact() -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def choose(flag):\n"
        "    return fcntl.LOCK_SH if flag else fcntl.LOCK_EX\n"
        "class Locker:\n"
        "    def __init__(self, handle):\n"
        "        self.handle = handle\n"
        "    def lock(self, flag):\n"
        "        fcntl.flock(self.handle, choose(flag))\n"
        "def closure(handle, flag):\n"
        "    operation = (\n"
        "        fcntl.LOCK_SH | fcntl.LOCK_NB\n"
        "        if flag\n"
        "        else fcntl.LOCK_EX | fcntl.LOCK_NB\n"
        "    )\n"
        "    def inner():\n"
        "        fcntl.flock(handle, operation)\n"
        "    inner()\n"
        "def exercise(flag):\n"
        "    handle = open(LOCK_PATH, 'a+')\n"
        "    Locker(handle).lock(flag)\n"
        "    closure(handle, flag)\n"
    )

    assert _access_operations(result) == {
        "builtin.open:a+",
        "fcntl.flock:shared_or_exclusive",
        "fcntl.flock:shared_or_exclusive_nonblocking",
    }
    assert result["escape_facts"] == []


def test_flock_invalid_dynamic_and_unproven_inputs_fail_closed() -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import AUXILIARY, LOCK_PATH\n"
        "def maybe_operation(flag, operation):\n"
        "    return fcntl.LOCK_SH if flag else operation\n"
        "def exercise(operation, flag, unknown_descriptor):\n"
        "    handle = open(LOCK_PATH)\n"
        "    fcntl.flock(handle, 1)\n"
        "    fcntl.flock(handle, fcntl.LOCK_UN | fcntl.LOCK_NB)\n"
        "    fcntl.flock(handle, fcntl.LOCK_SH | 16)\n"
        "    fcntl.flock(handle, operation)\n"
        "    fcntl.flock(handle, maybe_operation(flag, operation))\n"
        "    joined = fcntl.LOCK_EX\n"
        "    if flag:\n"
        "        joined = operation\n"
        "    fcntl.flock(handle, joined)\n"
        "    descriptor = handle if flag else unknown_descriptor\n"
        "    fcntl.flock(descriptor, fcntl.LOCK_EX)\n"
        "    fcntl.flock(LOCK_PATH, fcntl.LOCK_EX)\n"
        "    fcntl.flock(handle, AUXILIARY)\n",
        symbols=("AUXILIARY", "LOCK_PATH"),
    )

    assert _access_operations(result) == {"builtin.open:r"}
    assert _escape_reasons(result) == {
        "invalid_flock_operation",
        "dynamic_flock_operation",
        "ambiguous_registered_origin_flock_descriptor",
        "ambiguous_registered_origin_flock_arguments",
    }


@pytest.mark.parametrize(
    "call",
    [
        "fcntl.flock(fd=handle, operation=fcntl.LOCK_EX)",
        "fcntl.flock(handle)",
        "fcntl.flock(handle, fcntl.LOCK_EX, 0)",
        "fcntl.flock(*(handle, fcntl.LOCK_EX))",
        "fcntl.flock(handle, operation=fcntl.LOCK_EX)",
    ],
)
def test_flock_requires_two_positional_only_arguments(call: str) -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def exercise():\n"
        "    handle = open(LOCK_PATH)\n"
        f"    {call}\n"
    )

    assert _access_operations(result) == {"builtin.open:r"}
    assert "invalid_or_ambiguous_flock_signature" in _escape_reasons(result)


def test_path_receiver_joins_require_only_path_alternatives() -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import FIRST, SECOND\n"
        "def exercise(flag, unknown):\n"
        "    maybe_none = FIRST if flag else None\n"
        "    with maybe_none.open() as handle:\n"
        "        fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "    maybe_unknown = FIRST if flag else unknown\n"
        "    maybe_unknown.open()\n"
        "    exact_paths = FIRST if flag else SECOND\n"
        "    exact_paths.open()\n",
        symbols=("FIRST", "SECOND"),
    )

    assert {
        (row["resource_id"], row["operation"])
        for row in result["access_facts"]
    } == {
        ("runtime-resource:first", "path.open:r"),
        ("runtime-resource:second", "path.open:r"),
    }
    assert {
        row["resource_id"] for row in result["escape_facts"]
    } == {"runtime-resource:first"}
    assert _escape_reasons(result) == {
        "registered_locator_to_unknown_callee"
    }


def test_extracted_path_open_method_fails_closed_instead_of_silent_drop() -> None:
    result = _discover(
        "import fcntl\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def exercise():\n"
        "    opener = LOCK_PATH.open\n"
        "    handle = opener('r')\n"
        "    fcntl.flock(handle, fcntl.LOCK_EX)\n"
    )

    assert result["access_facts"] == []
    assert {
        (row["resource_id"], row["operation"], row["reason"])
        for row in result["escape_facts"]
    } == {
        (
            "runtime-resource:lock_path",
            "call:opener",
            "registered_locator_to_unknown_callee",
        )
    }


def test_cross_module_bare_open_mutation_uses_canonical_escape_sink() -> None:
    result = _discover(
        "from chronovisor.mutator import mutate\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def use():\n"
        "    open(LOCK_PATH)\n"
        "def caller():\n"
        "    mutate()\n"
        "    use()\n"
        "caller()\n",
        extra_sources={
            "src/chronovisor/mutator.py": (
                "import builtins\n"
                "def fake_open(*arguments):\n"
                "    return None\n"
                "def mutate():\n"
                "    builtins.open = fake_open\n"
            )
        },
    )

    assert result["access_facts"] == []
    assert {
        (row["operation"], row["sink"], row["reason"])
        for row in result["escape_facts"]
    } == {
        (
            "call:open",
            "builtins.open",
            "registered_locator_to_unknown_callee",
        )
    }


def test_cross_module_late_from_import_observes_builtin_mutation() -> None:
    result = _discover(
        "from chronovisor.mutator import mutate\n"
        "def caller():\n"
        "    mutate()\n"
        "    import chronovisor.reader as reader\n"
        "    reader.use()\n"
        "caller()\n",
        extra_sources={
            "src/chronovisor/mutator.py": (
                "import builtins\n"
                "def fake_open(*arguments):\n"
                "    return None\n"
                "def mutate():\n"
                "    builtins.open = fake_open\n"
            ),
            "src/chronovisor/reader.py": (
                "from builtins import open as after_open\n"
                "from chronovisor.resources import LOCK_PATH\n"
                "def use():\n"
                "    after_open(LOCK_PATH)\n"
            ),
        },
    )

    assert result["access_facts"] == []
    assert {
        (row["operation"], row["sink"], row["reason"])
        for row in result["escape_facts"]
    } == {
        (
            "call:after_open",
            "builtins.open",
            "registered_locator_to_unknown_callee",
        )
    }


def test_stdlib_mutation_capture_reimport_and_repo_shadows_are_conservative() -> None:
    mutated = _discover(
        "import builtins\n"
        "import fcntl\n"
        "from builtins import open as captured_open\n"
        "from fcntl import LOCK_EX, flock as captured_flock\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def fake(*arguments):\n"
        "    return None\n"
        "builtins.open = fake\n"
        "fcntl.flock = fake\n"
        "from builtins import open as after_open\n"
        "from fcntl import flock as after_flock\n"
        "def exercise():\n"
        "    handle = captured_open(LOCK_PATH)\n"
        "    captured_flock(handle, LOCK_EX)\n"
        "    builtins.open(LOCK_PATH)\n"
        "    open(LOCK_PATH)\n"
        "    after_open(LOCK_PATH)\n"
        "    fcntl.flock(handle, LOCK_EX)\n"
        "    after_flock(handle, LOCK_EX)\n"
    )
    shadowed = _discover(
        "import fcntl\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def exercise():\n"
        "    fcntl.flock(LOCK_PATH, fcntl.LOCK_EX)\n",
        extra_sources={
            "src/fcntl.py": (
                "LOCK_EX = 2\n"
                "def flock(handle, operation):\n"
                "    handle.read_text()\n"
            )
        },
    )

    assert _access_operations(mutated) == {
        "builtin.open:r",
        "fcntl.flock:exclusive",
    }
    assert len(mutated["escape_facts"]) == 5
    assert _escape_reasons(mutated) == {
        "registered_locator_to_unknown_callee"
    }
    assert _access_operations(shadowed) == {"path.read_text"}
    assert not any(
        row["sink"] == "fcntl.flock" for row in shadowed["access_facts"]
    )


@pytest.mark.parametrize(
    ("mutation", "call", "sink"),
    [
        (
            "builtins.__dict__['open'] = fake",
            "open(LOCK_PATH)",
            "builtins.open",
        ),
        (
            "del builtins.__dict__['open']",
            "open(LOCK_PATH)",
            "builtins.open",
        ),
        (
            "vars(builtins).pop('open', None)",
            "open(LOCK_PATH)",
            "builtins.open",
        ),
        (
            "fcntl.__dict__['flock'] = fake",
            "fcntl.flock(handle, fcntl.LOCK_EX)",
            "fcntl.flock",
        ),
        (
            "del fcntl.__dict__['flock']",
            "fcntl.flock(handle, fcntl.LOCK_EX)",
            "fcntl.flock",
        ),
        (
            "vars(fcntl).pop('flock', None)",
            "fcntl.flock(handle, fcntl.LOCK_EX)",
            "fcntl.flock",
        ),
    ],
)
def test_stdlib_module_dict_mutations_invalidate_precise_calls(
    mutation: str,
    call: str,
    sink: str,
) -> None:
    result = _discover(
        "import builtins\n"
        "import fcntl\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def fake(*arguments):\n"
        "    return None\n"
        "def exercise():\n"
        "    handle = open(LOCK_PATH)\n"
        f"    {mutation}\n"
        f"    {call}\n"
    )

    assert _access_operations(result) == {"builtin.open:r"}
    assert {
        (row["sink"], row["reason"]) for row in result["escape_facts"]
    } == {(sink, "registered_locator_to_unknown_callee")}


def test_module_dict_unrelated_key_is_precise_but_dynamic_key_is_fail_closed() -> None:
    unrelated = _discover(
        "import builtins\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def fake(*arguments):\n"
        "    return None\n"
        "builtins.__dict__['unrelated'] = fake\n"
        "vars(builtins).pop('also_unrelated', None)\n"
        "open(LOCK_PATH)\n"
    )
    dynamic = _discover(
        "import builtins\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def fake(*arguments):\n"
        "    return None\n"
        "def exercise(key):\n"
        "    builtins.__dict__[key] = fake\n"
        "    open(LOCK_PATH)\n"
    )

    assert _access_operations(unrelated) == {"builtin.open:r"}
    assert unrelated["escape_facts"] == []
    assert dynamic["access_facts"] == []
    assert {
        (row["sink"], row["reason"]) for row in dynamic["escape_facts"]
    } == {
        ("builtins.open", "registered_locator_to_unknown_callee")
    }


def test_synthetic_builtins_module_does_not_shadow_cpython_builtin() -> None:
    result = _discover(
        "import builtins\n"
        "from builtins import open as imported_open\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def exercise():\n"
        "    builtins.open(LOCK_PATH)\n"
        "    imported_open(LOCK_PATH)\n",
        extra_sources={
            "src/builtins.py": (
                "def open(value):\n"
                "    value.write_text('synthetic shadow')\n"
            )
        },
    )

    assert len(result["access_facts"]) == 2
    assert _access_operations(result) == {"builtin.open:r"}
    assert result["escape_facts"] == []


def test_fcntl_ids_are_line_stable_and_production_inventory_is_complete() -> None:
    source = (
        "import fcntl\n"
        "from chronovisor.resources import LOCK_PATH\n"
        "def exercise():\n"
        "    with open(LOCK_PATH) as handle:\n"
        "        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
    )
    base = _discover(source)
    shifted = _discover("\n\n" + source)

    assert set(base["access_fact_ids"]) == set(shifted["access_fact_ids"])
    assert set(base["escape_fact_ids"]) == set(shifted["escape_fact_ids"])

    repository = Path(__file__).resolve().parents[1]
    flock_calls: list[ast.Call] = []
    fdopen_calls: list[ast.Call] = []
    for source_path in (repository / "src" / "chronovisor").rglob("*.py"):
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func,
                ast.Attribute,
            ):
                continue
            if not isinstance(node.func.value, ast.Name):
                continue
            target = (node.func.value.id, node.func.attr)
            if target == ("fcntl", "flock"):
                flock_calls.append(node)
            elif target == ("os", "fdopen"):
                fdopen_calls.append(node)

    assert len(flock_calls) == 101
    assert all(
        len(call.args) == 2
        and not call.keywords
        and not any(isinstance(argument, ast.Starred) for argument in call.args)
        for call in flock_calls
    )
    assert len(fdopen_calls) == 38
    assert len(fdopen_calls) >= 8
