from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from scripts.runtime_ownership.access import discover_access_facts


def _candidate(symbol: str, *, locator: str | None = None) -> dict[str, Any]:
    return {
        "id": f"runtime-resource:{symbol.lower()}",
        "module": "chronovisor.resources",
        "symbol": symbol,
        "locator": {
            "type": "path",
            "value": locator or f"$ROOT/{symbol.lower()}.sqlite",
        },
    }


def _discover(
    consumer: str,
    *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    extra_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    selected = list(candidates or [_candidate("DATABASE")])
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


def _operations(result: Mapping[str, Any]) -> set[str]:
    return {str(row["operation"]) for row in result["access_facts"]}


def test_sqlite_connect_default_and_proven_read_only_uri() -> None:
    result = _discover(
        "import sqlite3 as database_api\n"
        "from chronovisor.resources import DATABASE, READ_ONLY_DATABASE\n"
        "def connect_all():\n"
        "    database_api.connect(DATABASE)\n"
        "    database_api.connect(READ_ONLY_DATABASE, uri=True)\n",
        candidates=[
            _candidate("DATABASE"),
            _candidate(
                "READ_ONLY_DATABASE",
                locator="file:$ROOT/read-only.sqlite?mode=ro",
            ),
        ],
    )

    assert {
        (row["resource_id"], row["mode"], row["operation"])
        for row in result["access_facts"]
    } == {
        ("runtime-resource:database", "read_write", "sqlite.connect:rwc"),
        (
            "runtime-resource:read_only_database",
            "read",
            "sqlite.connect:ro",
        ),
    }
    assert result["escape_facts"] == []


def test_sqlite_helper_return_preserves_connection_and_cursor_tags() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def open_database(path):\n"
        "    return sqlite3.connect(path)\n"
        "def read_one():\n"
        "    connection = open_database(DATABASE)\n"
        "    cursor = connection.cursor()\n"
        "    return cursor.execute('SELECT value FROM records').fetchone()\n"
    )

    assert _operations(result) == {
        "sqlite.connect:rwc",
        "sqlite.execute:read",
    }
    assert result["escape_facts"] == []


def test_sqlite_heterogeneous_handle_and_path_join_fails_closed() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE, OTHER_PATH\n"
        "def maybe_handle(flag):\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    value = connection if flag else OTHER_PATH\n"
        "    value.execute('SELECT 1')\n",
        candidates=[_candidate("DATABASE"), _candidate("OTHER_PATH")],
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    assert {
        row["resource_id"] for row in result["escape_facts"]
    } == {"runtime-resource:database", "runtime-resource:other_path"}


def test_sqlite_handle_tags_prevent_path_false_positives() -> None:
    result = _discover(
        "from sqlite3 import connect as open_database\n"
        "from chronovisor.resources import DATABASE\n"
        "def misuse():\n"
        "    connection = open_database(DATABASE)\n"
        "    connection.write_text('not a Path')\n"
        "    connection / 'not-a-child'\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    assert not any(
        row["operation"].startswith("path.") for row in result["access_facts"]
    )
    assert result["escape_facts"]


def test_sqlite_context_records_commit_or_rollback_without_suppression() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def normal():\n"
        "    with sqlite3.connect(DATABASE) as connection:\n"
        "        connection.execute('SELECT 1').fetchall()\n"
        "def raised():\n"
        "    try:\n"
        "        with sqlite3.connect(DATABASE) as connection:\n"
        "            raise RuntimeError('rollback')\n"
        "    except RuntimeError:\n"
        "        pass\n"
    )

    assert _operations(result) == {
        "sqlite.connect:rwc",
        "sqlite.execute:read",
        "sqlite.transaction.implicit_commit",
        "sqlite.transaction.implicit_rollback",
    }
    assert not any(
        row["reason"] == "unknown_context_manager_suppression"
        for row in result["escape_facts"]
    )


@pytest.mark.parametrize(
    "consumer",
    [
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def caller():\n"
        "    sqlite3.connect(DATABASE)\n",
        "from sqlite3 import connect as local_connect\n"
        "from chronovisor.resources import DATABASE\n"
        "def caller():\n"
        "    local_connect(DATABASE)\n",
    ],
)
def test_repository_local_sqlite3_is_not_a_stdlib_sink(consumer: str) -> None:
    result = _discover(
        consumer,
        extra_sources={
            "src/sqlite3.py": (
                "def connect(path):\n"
                "    path.read_text()\n"
            )
        },
    )

    assert _operations(result) == {"path.read_text"}
    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["access_facts"]
    )
    assert result["escape_facts"] == []


def test_sqlite_connect_requires_both_uri_true_and_read_only_shape() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import READ_ONLY_SHAPE, ORDINARY\n"
        "def connect_all(uri_value):\n"
        "    sqlite3.connect(READ_ONLY_SHAPE)\n"
        "    sqlite3.connect(READ_ONLY_SHAPE, uri=uri_value)\n"
        "    sqlite3.connect(ORDINARY, uri=True)\n",
        candidates=[
            _candidate(
                "READ_ONLY_SHAPE",
                locator="file:$ROOT/read-only.sqlite?mode=ro",
            ),
            _candidate("ORDINARY"),
        ],
    )

    assert {
        (row["resource_id"], row["mode"], row["operation"])
        for row in result["access_facts"]
    } == {
        (
            "runtime-resource:read_only_shape",
            "read_write",
            "sqlite.connect:rwc",
        ),
        (
            "runtime-resource:ordinary",
            "read_write",
            "sqlite.connect:rwc",
        ),
    }


def test_sqlite_connect_accepts_static_read_only_fstring_shape() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def connect_read_only():\n"
        "    sqlite3.connect(f'file:{DATABASE}?mode=ro', uri=True)\n"
    )

    assert {
        (row["mode"], row["operation"])
        for row in result["access_facts"]
    } == {("read", "sqlite.connect:ro")}
    assert result["escape_facts"] == []


@pytest.mark.parametrize("keyword", ["timeout", "uri"])
def test_sqlite_connect_origin_bearing_auxiliary_fails_closed(
    keyword: str,
) -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE, AUXILIARY\n"
        "def connect_ambiguous():\n"
        f"    sqlite3.connect(DATABASE, {keyword}=AUXILIARY)\n",
        candidates=[_candidate("DATABASE"), _candidate("AUXILIARY")],
    )

    assert result["access_facts"] == []
    assert {
        (row["resource_id"], row["reason"])
        for row in result["escape_facts"]
    } == {
        (
            "runtime-resource:database",
            "ambiguous_registered_origin_sqlite_connect_arguments",
        ),
        (
            "runtime-resource:auxiliary",
            "ambiguous_registered_origin_sqlite_connect_arguments",
        ),
    }


def test_sqlite_connect_allows_non_origin_auxiliary_values() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def connect_configured():\n"
        "    return sqlite3.connect(\n"
        "        DATABASE, timeout=1.0, check_same_thread=False, uri=False\n"
        "    )\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    assert result["escape_facts"] == []


def test_sqlite_custom_connection_factory_fails_closed_without_handle_tag() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def custom_factory(*args):\n"
        "    return object()\n"
        "def connect_custom():\n"
        "    connection = sqlite3.connect(DATABASE, factory=custom_factory)\n"
        "    connection.execute('SELECT 1')\n"
    )

    assert result["access_facts"] == []
    assert {
        (row["resource_id"], row["reason"])
        for row in result["escape_facts"]
    } == {
        (
            "runtime-resource:database",
            "unsupported_sqlite_connect_factory",
        )
    }


def test_sqlite_standard_connection_and_cursor_factories_remain_precise() -> None:
    result = _discover(
        "from sqlite3 import Connection, Cursor, connect\n"
        "from chronovisor.resources import DATABASE\n"
        "def standard_factories():\n"
        "    connection = connect(DATABASE, factory=Connection)\n"
        "    cursor = connection.cursor(factory=Cursor)\n"
        "    cursor.execute('SELECT 1').fetchone()\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc", "sqlite.execute:read"}
    assert result["escape_facts"] == []


def test_sqlite_custom_cursor_factory_fails_closed_without_cursor_tag() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def custom_cursor(connection):\n"
        "    return object()\n"
        "def cursor_custom():\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    cursor = connection.cursor(factory=custom_cursor)\n"
        "    cursor.execute('SELECT 1')\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    assert {
        (row["resource_id"], row["reason"])
        for row in result["escape_facts"]
    } == {
        (
            "runtime-resource:database",
            "unsupported_sqlite_cursor_factory",
        )
    }


def test_sqlite_static_sql_matrix_and_script_mode_join() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def use_database():\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    connection.execute('SELECT value FROM records')\n"
        "    connection.execute('PRAGMA table_info(records)')\n"
        "    connection.execute('INSERT INTO records VALUES (1)')\n"
        "    connection.execute('UPDATE records SET value = 2')\n"
        "    connection.execute('DELETE FROM records')\n"
        "    connection.execute('REPLACE INTO records VALUES (1)')\n"
        "    connection.execute('CREATE TABLE more(value)')\n"
        "    connection.execute('DROP TABLE more')\n"
        "    connection.execute('ALTER TABLE records ADD COLUMN more')\n"
        "    connection.execute('VACUUM')\n"
        "    connection.execute('PRAGMA journal_mode = WAL')\n"
        "    connection.execute('BEGIN IMMEDIATE')\n"
        "    connection.executemany('INSERT INTO records VALUES (?)', [])\n"
        "    connection.executescript(\n"
        "        \"SELECT 'semi;colon'; INSERT INTO records VALUES (3);\"\n"
        "    )\n"
    )

    facts = result["access_facts"]
    assert sum(row["operation"] == "sqlite.execute:read" for row in facts) == 2
    assert sum(row["operation"] == "sqlite.execute:write" for row in facts) == 9
    assert {
        (row["operation"], row["mode"])
        for row in facts
    } >= {
        ("sqlite.connect:rwc", "read_write"),
        ("sqlite.transaction.begin_immediate", "write"),
        ("sqlite.executemany:write", "write"),
        ("sqlite.executescript:read_write", "read_write"),
    }
    assert result["escape_facts"] == []


@pytest.mark.parametrize(
    "sql_expression",
    [
        "statement",
        "f'{verb} value FROM records'",
        "'WITH selected AS (SELECT 1) SELECT * FROM selected'",
        "'SAVEPOINT checkpoint'",
        "\"ATTACH DATABASE 'other.sqlite' AS other\"",
    ],
)
def test_sqlite_dynamic_or_unsupported_sql_fails_closed(
    sql_expression: str,
) -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def execute_unknown(statement, verb):\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        f"    connection.execute({sql_expression})\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    sql_escape = next(
        row
        for row in result["escape_facts"]
        if row["reason"] == "dynamic_or_unsupported_sqlite_sql"
    )
    assert sql_escape["resource_id"] == "runtime-resource:database"


def test_sqlite_static_leading_fstring_verb_is_classified() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def dynamic_table(table):\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    connection.execute(f'SELECT value FROM {table}')\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc", "sqlite.execute:read"}
    assert result["escape_facts"] == []


def test_sqlite_parameter_origin_is_separate_from_database_access() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE, PAYLOAD_PATH\n"
        "def insert_path():\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    connection.execute(\n"
        "        'INSERT INTO records VALUES (?)', (PAYLOAD_PATH,)\n"
        "    )\n",
        candidates=[_candidate("DATABASE"), _candidate("PAYLOAD_PATH")],
    )

    sqlite_write = next(
        row
        for row in result["access_facts"]
        if row["operation"] == "sqlite.execute:write"
    )
    assert sqlite_write["resource_id"] == "runtime-resource:database"
    assert not any(
        row["resource_id"] == "runtime-resource:payload_path"
        for row in result["access_facts"]
    )
    assert {
        (row["resource_id"], row["reason"])
        for row in result["escape_facts"]
    } == {
        (
            "runtime-resource:payload_path",
            "ambiguous_registered_origin_sqlite_arguments",
        )
    }


def test_sqlite_cursor_iteration_fetch_and_explicit_lifecycle() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def lifecycle():\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    cursor = connection.execute('SELECT value FROM records')\n"
        "    for row in cursor:\n"
        "        consume(row)\n"
        "    cursor.fetchone()\n"
        "    cursor.fetchmany(2)\n"
        "    cursor.fetchall()\n"
        "    cursor.close()\n"
        "    connection.commit()\n"
        "    connection.rollback()\n"
        "    connection.close()\n"
    )

    assert _operations(result) == {
        "sqlite.connect:rwc",
        "sqlite.execute:read",
        "sqlite.transaction.commit",
        "sqlite.transaction.rollback",
    }
    assert result["escape_facts"] == []


def test_sqlite_explicit_context_alias_and_unhandled_raise_paths() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def returned():\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    with connection as alias:\n"
        "        alias.execute('SELECT 1')\n"
        "        return\n"
        "def unhandled():\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    with connection:\n"
        "        raise RuntimeError('stop')\n"
        "    DATABASE.read_text()\n"
    )

    assert _operations(result) == {
        "sqlite.connect:rwc",
        "sqlite.execute:read",
        "sqlite.transaction.implicit_commit",
        "sqlite.transaction.implicit_rollback",
    }
    assert "path.read_text" not in _operations(result)
    assert not any(
        row["reason"] == "unknown_context_manager_suppression"
        for row in result["escape_facts"]
    )


def test_sqlite_context_break_and_continue_paths_implicitly_commit() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def loop(items):\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    for item in items:\n"
        "        with connection:\n"
        "            if item:\n"
        "                break\n"
        "            continue\n"
    )

    assert _operations(result) == {
        "sqlite.connect:rwc",
        "sqlite.transaction.implicit_commit",
    }
    assert result["escape_facts"] == []


def test_sqlite_mixed_unknown_manager_keeps_only_unknown_suppression() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE, MANAGER\n"
        "def mixed():\n"
        "    with sqlite3.connect(DATABASE), MANAGER:\n"
        "        raise RuntimeError('maybe suppressed')\n",
        candidates=[_candidate("DATABASE"), _candidate("MANAGER")],
    )

    assert {
        "sqlite.transaction.implicit_commit",
        "sqlite.transaction.implicit_rollback",
    } <= _operations(result)
    suppression = [
        row
        for row in result["escape_facts"]
        if row["reason"] == "unknown_context_manager_suppression"
    ]
    assert {row["resource_id"] for row in suppression} == {
        "runtime-resource:manager"
    }


def test_sqlite_inner_manager_rolls_back_before_outer_unknown_suppression() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE, MANAGER\n"
        "def mixed():\n"
        "    with MANAGER, sqlite3.connect(DATABASE):\n"
        "        raise RuntimeError('maybe suppressed')\n",
        candidates=[_candidate("DATABASE"), _candidate("MANAGER")],
    )

    assert "sqlite.transaction.implicit_rollback" in _operations(result)
    assert "sqlite.transaction.implicit_commit" not in _operations(result)
    suppression = [
        row
        for row in result["escape_facts"]
        if row["reason"] == "unknown_context_manager_suppression"
    ]
    assert {row["resource_id"] for row in suppression} == {
        "runtime-resource:manager"
    }


def test_sqlite_local_and_external_shadows_are_not_concrete_sinks() -> None:
    result = _discover(
        "import external_sqlite as database_api\n"
        "from sqlite3 import connect as open_database\n"
        "from chronovisor.resources import DATABASE\n"
        "def local_module_shadow(sqlite3):\n"
        "    sqlite3.connect(DATABASE)\n"
        "def imported_shadow(open_database):\n"
        "    open_database(DATABASE)\n"
        "def external_module():\n"
        "    database_api.connect(DATABASE)\n"
    )

    assert result["access_facts"] == []
    assert result["escape_facts"]
    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["escape_facts"]
    )


@pytest.mark.parametrize(
    ("locator", "expected_operation"),
    [
        ("file:$ROOT/db.sqlite?mode=ro", "sqlite.connect:ro"),
        ("$ROOT/db.sqlite?mode=ro", "sqlite.connect:rwc"),
        ("file:$ROOT/db.sqlite#?mode=ro", "sqlite.connect:rwc"),
        ("file:$ROOT/db.sqlite?MODE=ro", "sqlite.connect:rwc"),
        ("file:$ROOT/db.sqlite?mode=RO", "sqlite.connect:rwc"),
        ("file:$ROOT/db.sqlite?mode=ro&mode=rw", "sqlite.connect:rwc"),
    ],
)
def test_sqlite_read_only_uri_requires_exact_runtime_semantics(
    locator: str,
    expected_operation: str,
) -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def connect_database():\n"
        "    sqlite3.connect(DATABASE, uri=True)\n",
        candidates=[_candidate("DATABASE", locator=locator)],
    )

    assert [row["operation"] for row in result["access_facts"]] == [
        expected_operation
    ]


@pytest.mark.parametrize(
    ("expression", "expected_operation"),
    [
        ("f'{DATABASE}?mode=ro'", "sqlite.connect:rwc"),
        ("f'file:{DATABASE}?mode=ro'", "sqlite.connect:ro"),
    ],
)
def test_sqlite_read_only_fstring_requires_literal_file_scheme(
    expression: str,
    expected_operation: str,
) -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def connect_database():\n"
        f"    sqlite3.connect({expression}, uri=True)\n"
    )

    assert [row["operation"] for row in result["access_facts"]] == [
        expected_operation
    ]


def test_sqlite_parenthesized_pragma_setters_are_never_false_read() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def pragmas():\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    connection.execute('PRAGMA journal_mode(WAL)')\n"
        "    connection.execute('PRAGMA user_version(5)')\n"
        "    connection.execute('PRAGMA application_id(1)')\n"
        "    connection.execute('PRAGMA page_size(4096)')\n"
        "    connection.execute('PRAGMA table_info(records)')\n"
        "    connection.execute('PRAGMA index_info(records_index)')\n"
        "    connection.execute('PRAGMA foreign_key_list(records)')\n"
        "    connection.execute('PRAGMA unknown_extension(value)')\n"
    )

    assert sum(
        row["operation"] == "sqlite.execute:write"
        for row in result["access_facts"]
    ) == 4
    assert sum(
        row["operation"] == "sqlite.execute:read"
        for row in result["access_facts"]
    ) == 3
    assert any(
        row["reason"] == "dynamic_or_unsupported_sqlite_sql"
        for row in result["escape_facts"]
    )


@pytest.mark.parametrize(
    "call",
    [
        "sqlite3.connect(DATABASE, database=DATABASE)",
        "sqlite3.connect(DATABASE, unknown_option=True)",
        (
            "sqlite3.connect(DATABASE, 5.0, 0, 'DEFERRED', True, "
            "sqlite3.Connection, 128, False, True)"
        ),
        "sqlite3.connect(*(DATABASE,))",
    ],
)
def test_sqlite_connect_invalid_signature_fails_closed(call: str) -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def invalid_connect():\n"
        f"    {call}\n"
    )

    assert result["access_facts"] == []
    assert {
        row["reason"] for row in result["escape_facts"]
    } == {"invalid_or_ambiguous_sqlite_signature"}


def test_sqlite_connect_valid_keyword_shape_includes_keyword_only_autocommit() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def valid_connect():\n"
        "    sqlite3.connect(database=DATABASE, autocommit=True)\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    assert result["escape_facts"] == []


@pytest.mark.parametrize(
    ("setup", "invalid_call"),
    [
        ("connection = sqlite3.connect(DATABASE)", "connection.execute()"),
        (
            "connection = sqlite3.connect(DATABASE)",
            "connection.execute('SELECT 1', (), ())",
        ),
        (
            "connection = sqlite3.connect(DATABASE)",
            "connection.execute(sql='SELECT 1')",
        ),
        (
            "connection = sqlite3.connect(DATABASE)",
            "connection.executemany('INSERT INTO records VALUES (1)')",
        ),
        (
            "connection = sqlite3.connect(DATABASE)",
            "connection.executescript('SELECT 1', 'SELECT 2')",
        ),
        ("connection = sqlite3.connect(DATABASE)", "connection.commit(1)"),
        ("connection = sqlite3.connect(DATABASE)", "connection.rollback(flag=True)"),
        ("connection = sqlite3.connect(DATABASE)", "connection.close(1)"),
        (
            "cursor = sqlite3.connect(DATABASE).cursor()",
            "cursor.fetchone(1)",
        ),
        (
            "cursor = sqlite3.connect(DATABASE).cursor()",
            "cursor.fetchall(limit=1)",
        ),
        (
            "cursor = sqlite3.connect(DATABASE).cursor()",
            "cursor.fetchmany(1, 2)",
        ),
        (
            "connection = sqlite3.connect(DATABASE)",
            "connection.cursor(factory=sqlite3.Cursor, extra=True)",
        ),
    ],
)
def test_sqlite_invalid_method_signature_fails_closed(
    setup: str,
    invalid_call: str,
) -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def invalid_method():\n"
        f"    {setup}\n"
        f"    {invalid_call}\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    assert {
        row["reason"] for row in result["escape_facts"]
    } == {"invalid_or_ambiguous_sqlite_signature"}


def test_sqlite_invalid_signature_accounts_for_argument_origin_separately() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE, PAYLOAD_PATH\n"
        "def invalid_method():\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    connection.execute('SELECT 1', (), PAYLOAD_PATH)\n",
        candidates=[_candidate("DATABASE"), _candidate("PAYLOAD_PATH")],
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    assert {
        (row["resource_id"], row["reason"])
        for row in result["escape_facts"]
    } == {
        (
            "runtime-resource:database",
            "invalid_or_ambiguous_sqlite_signature",
        ),
        (
            "runtime-resource:payload_path",
            "ambiguous_registered_origin_sqlite_arguments",
        ),
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "sqlite3.connect = fake_connect",
        "database_api = sqlite3\n    database_api.connect = fake_connect",
    ],
)
def test_sqlite_module_attribute_rebinding_disables_concrete_sink(
    mutation: str,
) -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_connect(path):\n"
        "    return object()\n"
        "def caller():\n"
        f"    {mutation}\n"
        "    sqlite3.connect(DATABASE)\n"
    )

    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


def test_sqlite_module_level_rebinding_disables_concrete_sink() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_connect(path):\n"
        "    return object()\n"
        "sqlite3.connect = fake_connect\n"
        "def caller():\n"
        "    sqlite3.connect(DATABASE)\n"
    )

    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


def test_os_module_attribute_rebinding_disables_concrete_sink() -> None:
    result = _discover(
        "import os\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_open(path, flags):\n"
        "    return 1\n"
        "def caller():\n"
        "    operating_system = os\n"
        "    operating_system.open = fake_open\n"
        "    os.open(DATABASE, os.O_RDONLY)\n"
    )

    assert not any(
        row["operation"].startswith("os.open")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


@pytest.mark.parametrize(
    "mutation",
    [
        "setattr(sqlite3, 'connect', fake_connect)",
        "delattr(sqlite3, 'connect')",
        "del sqlite3.connect",
        "database_api = sqlite3\n    setattr(database_api, 'connect', fake_connect)",
        "database_api = sqlite3\n    del database_api.connect",
    ],
)
def test_sqlite_exact_runtime_module_mutations_disable_concrete_sink(
    mutation: str,
) -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_connect(path):\n"
        "    return object()\n"
        "def caller():\n"
        f"    {mutation}\n"
        "    sqlite3.connect(DATABASE)\n"
    )

    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


def test_sqlite_runtime_module_mutation_survives_alias_branch_join() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def caller(should_mutate):\n"
        "    database_api = sqlite3\n"
        "    if should_mutate:\n"
        "        delattr(database_api, 'connect')\n"
        "    sqlite3.connect(DATABASE)\n"
    )

    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


@pytest.mark.parametrize(
    "mutation",
    [
        "setattr(sqlite3, attribute, fake_connect)",
        "delattr(sqlite3, attribute)",
    ],
)
def test_sqlite_dynamic_runtime_module_mutation_wildcard_taints(
    mutation: str,
) -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_connect(path):\n"
        "    return object()\n"
        "def caller(attribute):\n"
        f"    {mutation}\n"
        "    sqlite3.connect(DATABASE)\n"
    )

    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


@pytest.mark.parametrize("builtin_name", ["setattr", "delattr"])
def test_locally_executed_shadowed_mutation_builtin_preserves_precision(
    builtin_name: str,
) -> None:
    mutation = (
        "setattr(sqlite3, 'connect', fake_connect)"
        if builtin_name == "setattr"
        else "delattr(sqlite3, 'connect')"
    )
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_connect(path):\n"
        "    return object()\n"
        f"def {builtin_name}(*arguments):\n"
        "    return None\n"
        "def caller():\n"
        f"    {mutation}\n"
        "    sqlite3.connect(DATABASE)\n"
        "caller()\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc"}


@pytest.mark.parametrize("builtin_name", ["setattr", "delattr"])
def test_unknown_shadowed_mutation_callback_taints_sqlite(
    builtin_name: str,
) -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        f"def caller({builtin_name}):\n"
        f"    {builtin_name}(sqlite3)\n"
        "    sqlite3.connect(DATABASE)\n"
    )

    assert result["access_facts"] == []
    assert {
        row["operation"] for row in result["escape_facts"]
    } == {"call:sqlite3.connect"}


@pytest.mark.parametrize(
    "mutation",
    [
        "setattr(operating_system, 'open', fake_open)",
        "delattr(operating_system, 'open')",
        "del operating_system.open",
        "setattr(operating_system, attribute, fake_open)",
    ],
)
def test_os_runtime_module_mutations_disable_concrete_sink_after_branch(
    mutation: str,
) -> None:
    result = _discover(
        "import os\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_open(path, flags):\n"
        "    return 1\n"
        "def caller(should_mutate, attribute):\n"
        "    operating_system = os\n"
        "    if should_mutate:\n"
        f"        {mutation}\n"
        "    os.open(DATABASE, os.O_RDONLY)\n"
    )

    assert not any(
        row["operation"].startswith("os.open")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


def test_sqlite_connection_type_rebinding_invalidates_factory_proof() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "class CustomConnection:\n"
        "    pass\n"
        "def caller():\n"
        "    sqlite3.Connection = CustomConnection\n"
        "    sqlite3.connect(DATABASE, factory=sqlite3.Connection)\n"
    )

    assert result["access_facts"] == []
    assert {
        row["reason"] for row in result["escape_facts"]
    } == {"unsupported_sqlite_connect_factory"}


def test_sqlite_cursor_connection_projection_preserves_connection_semantics() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def projected_connection():\n"
        "    cursor = sqlite3.connect(DATABASE).cursor()\n"
        "    cursor.connection.commit()\n"
        "    cursor.connection.execute('INSERT INTO records VALUES (1)')\n"
        "    with cursor.connection:\n"
        "        cursor.connection.execute('SELECT 1').fetchone()\n"
    )

    assert _operations(result) == {
        "sqlite.connect:rwc",
        "sqlite.execute:read",
        "sqlite.execute:write",
        "sqlite.transaction.commit",
        "sqlite.transaction.implicit_commit",
    }
    assert result["escape_facts"] == []


def test_sqlite_scalar_attributes_do_not_leak_database_origin() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def scalar_attributes():\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    cursor = connection.execute('INSERT INTO records VALUES (1)')\n"
        "    opaque(\n"
        "        cursor.rowcount, cursor.lastrowid, cursor.description,\n"
        "        cursor.arraysize, connection.total_changes,\n"
        "        connection.in_transaction, connection.row_factory,\n"
        "    )\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc", "sqlite.execute:write"}
    assert result["escape_facts"] == []


def test_sqlite_cursor_async_iteration_fails_closed_without_body_flow() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "async def invalid_async_iteration():\n"
        "    cursor = sqlite3.connect(DATABASE).execute('SELECT 1')\n"
        "    async for row in cursor:\n"
        "        DATABASE.write_text('unreachable')\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc", "sqlite.execute:read"}
    assert "path.write_text" not in _operations(result)
    assert {
        row["operation"] for row in result["escape_facts"]
    } == {"control:async_for"}


@pytest.mark.parametrize(
    ("reimport", "call"),
    [
        ("import sqlite3", "sqlite3.connect(DATABASE)"),
        ("import sqlite3 as database_api", "database_api.connect(DATABASE)"),
    ],
)
def test_sqlite_module_mutation_survives_reimport(
    reimport: str,
    call: str,
) -> None:
    result = _discover(
        "import sqlite3 as original_database_api\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_connect(path):\n"
        "    return object()\n"
        "def caller(should_mutate):\n"
        "    if should_mutate:\n"
        "        setattr(original_database_api, 'connect', fake_connect)\n"
        f"    {reimport}\n"
        f"    {call}\n"
    )

    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


@pytest.mark.parametrize(
    "mutation",
    [
        "setattr(sqlite3, 'connect', fake_connect)",
        "setattr(sqlite3, attribute, fake_connect)",
    ],
)
def test_sqlite_module_mutation_survives_later_from_import(
    mutation: str,
) -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_connect(path):\n"
        "    return object()\n"
        "def caller(attribute):\n"
        f"    {mutation}\n"
        "    from sqlite3 import connect\n"
        "    connect(DATABASE)\n"
    )

    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


def test_os_module_mutation_survives_reimport_and_from_import() -> None:
    result = _discover(
        "import os\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_open(path, flags):\n"
        "    return 1\n"
        "def caller(should_mutate):\n"
        "    if should_mutate:\n"
        "        setattr(os, 'open', fake_open)\n"
        "    import os as operating_system\n"
        "    from os import open as raw_open\n"
        "    operating_system.open(DATABASE, operating_system.O_RDONLY)\n"
        "    raw_open(DATABASE, 0)\n"
    )

    assert not any(
        row["operation"].startswith("os.open")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


def test_repository_local_sqlite3_mutation_does_not_taint_stdlib_state() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_connect(path):\n"
        "    return object()\n"
        "def caller():\n"
        "    setattr(sqlite3, 'connect', fake_connect)\n"
        "    import sqlite3 as database_api\n"
        "    database_api.connect(DATABASE)\n",
        extra_sources={
            "src/sqlite3.py": (
                "def connect(path):\n"
                "    path.read_text()\n"
            )
        },
    )

    assert _operations(result) == {"path.read_text"}
    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["access_facts"]
    )


def test_unknown_sqlite_connection_attribute_drops_handle_semantics() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def caller():\n"
        "    connection = sqlite3.connect(DATABASE)\n"
        "    connection.foo.execute('INSERT INTO records VALUES (1)')\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    assert {
        row["operation"] for row in result["escape_facts"]
    } == {"call:connection.foo.execute"}


def test_async_with_sqlite_connection_is_unreachable_and_fails_closed() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "async def caller():\n"
        "    async with sqlite3.connect(DATABASE) as connection:\n"
        "        connection.execute('INSERT INTO records VALUES (1)')\n"
        "        DATABASE.write_text('unreachable')\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    assert not any(
        "implicit_" in row["operation"]
        for row in result["access_facts"]
    )
    assert {
        row["operation"] for row in result["escape_facts"]
    } == {"control:async_with"}


def test_known_callee_module_mutation_propagates_through_alias_and_branch() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_connect(path):\n"
        "    return object()\n"
        "def mutate(database_api, should_mutate):\n"
        "    alias = database_api\n"
        "    if should_mutate:\n"
        "        setattr(alias, 'connect', fake_connect)\n"
        "def caller(should_mutate):\n"
        "    database_api = sqlite3\n"
        "    mutate(database_api, should_mutate)\n"
        "    sqlite3.connect(DATABASE)\n"
        "caller(True)\n"
    )

    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


def test_known_nested_closure_module_mutation_propagates() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_connect(path):\n"
        "    return object()\n"
        "def caller(should_mutate):\n"
        "    database_api = sqlite3\n"
        "    def mutate():\n"
        "        if should_mutate:\n"
        "            del database_api.connect\n"
        "    mutate()\n"
        "    sqlite3.connect(DATABASE)\n"
    )

    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


def test_unknown_callee_receiving_stdlib_module_fails_closed() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def caller(callback):\n"
        "    callback(sqlite3)\n"
        "    sqlite3.connect(DATABASE)\n"
    )

    assert not any(
        row["operation"].startswith("sqlite.")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


def test_locally_executed_read_only_module_helper_preserves_precision() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def inspect(database_api):\n"
        "    return database_api.__name__\n"
        "def caller():\n"
        "    inspect(sqlite3)\n"
        "    sqlite3.connect(DATABASE)\n"
        "caller()\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    assert result["escape_facts"] == []


def test_known_callee_os_module_mutation_propagates() -> None:
    result = _discover(
        "import os\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_open(path, flags):\n"
        "    return 1\n"
        "def mutate(operating_system):\n"
        "    delattr(operating_system, 'open')\n"
        "def caller():\n"
        "    mutate(os)\n"
        "    os.open(DATABASE, os.O_RDONLY)\n"
    )

    assert not any(
        row["operation"].startswith("os.open")
        for row in result["access_facts"]
    )
    assert result["escape_facts"]


def test_unknown_callee_taints_every_branch_joined_stdlib_module() -> None:
    result = _discover(
        "import os\n"
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def caller(callback, condition):\n"
        "    module = sqlite3 if condition else os\n"
        "    callback(module)\n"
        "    sqlite3.connect(DATABASE)\n"
        "    os.open(DATABASE, os.O_RDONLY)\n"
    )

    assert result["access_facts"] == []
    assert {
        row["operation"] for row in result["escape_facts"]
    } == {"call:sqlite3.connect", "call:os.open"}


def test_unknown_callee_taints_stdlib_modules_in_nested_containers() -> None:
    result = _discover(
        "import os\n"
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def caller(callback):\n"
        "    modules = {'database': [sqlite3], 'filesystem': ((os,),)}\n"
        "    callback(modules)\n"
        "    sqlite3.connect(DATABASE)\n"
        "    os.open(DATABASE, os.O_RDONLY)\n"
    )

    assert result["access_facts"] == []
    assert {
        row["operation"] for row in result["escape_facts"]
    } == {"call:sqlite3.connect", "call:os.open"}


def test_unknown_callee_mixed_module_and_origin_taints_and_escapes() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def caller(callback, condition):\n"
        "    value = sqlite3 if condition else DATABASE\n"
        "    callback(value)\n"
        "    sqlite3.connect(DATABASE)\n"
    )

    assert result["access_facts"] == []
    assert {
        row["operation"] for row in result["escape_facts"]
    } == {"call:callback", "call:sqlite3.connect"}


@pytest.mark.parametrize(
    ("setup", "mutation", "operation", "sink"),
    [
        ("", "sqlite3.connect = DATABASE", "assignment:sqlite3.connect", "sqlite3.connect"),
        (
            "database_api = sqlite3\n    if condition:\n        ",
            "database_api.connect = DATABASE",
            "assignment:sqlite3.connect",
            "sqlite3.connect",
        ),
        ("", "sqlite3.connect: object = DATABASE", "assignment:sqlite3.connect", "sqlite3.connect"),
        ("", "os.open = DATABASE", "assignment:os.open", "os.open"),
        (
            "operating_system = os\n    if condition:\n        ",
            "operating_system.open = DATABASE",
            "assignment:os.open",
            "os.open",
        ),
    ],
)
def test_stdlib_module_attribute_assignment_with_origin_escapes(
    setup: str,
    mutation: str,
    operation: str,
    sink: str,
) -> None:
    result = _discover(
        "import os\n"
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def caller(condition):\n"
        f"    {setup}{mutation}\n"
    )

    assert result["access_facts"] == []
    assert {
        (row["operation"], row["sink"], row["reason"])
        for row in result["escape_facts"]
    } == {
        (
            operation,
            sink,
            "registered_locator_to_stdlib_module_mutation",
        )
    }


def test_stdlib_module_attribute_augassign_includes_mutation_escape() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def caller():\n"
        "    sqlite3.connect += DATABASE\n"
    )

    assert (
        "assignment:sqlite3.connect",
        "sqlite3.connect",
        "registered_locator_to_stdlib_module_mutation",
    ) in {
        (row["operation"], row["sink"], row["reason"])
        for row in result["escape_facts"]
    }


def test_sequential_known_local_callees_keep_sqlite_mutation_state() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_connect(path):\n"
        "    return object()\n"
        "def mutate():\n"
        "    sqlite3.connect = fake_connect\n"
        "def use():\n"
        "    sqlite3.connect(DATABASE)\n"
        "def caller():\n"
        "    mutate()\n"
        "    use()\n"
        "caller()\n"
    )

    assert result["access_facts"] == []
    assert {
        row["operation"] for row in result["escape_facts"]
    } == {"call:sqlite3.connect"}


def test_cross_module_mutation_state_survives_later_import() -> None:
    result = _discover(
        "from chronovisor.resources import DATABASE\n"
        "from chronovisor.mutator import mutate\n"
        "def caller():\n"
        "    mutate()\n"
        "    import sqlite3\n"
        "    sqlite3.connect(DATABASE)\n"
        "caller()\n",
        extra_sources={
            "src/chronovisor/mutator.py": (
                "import sqlite3\n"
                "def fake_connect(path):\n"
                "    return object()\n"
                "def mutate():\n"
                "    sqlite3.connect = fake_connect\n"
            )
        },
    )

    assert result["access_facts"] == []
    assert {
        row["operation"] for row in result["escape_facts"]
    } == {"call:sqlite3.connect"}


@pytest.mark.parametrize(
    ("import_statement", "call", "operation", "sink"),
    [
        (
            "import sqlite3 as database_api",
            "database_api.connect(DATABASE)",
            "call:database_api.connect",
            "sqlite3.connect",
        ),
        (
            "from sqlite3 import connect as open_database",
            "open_database(DATABASE)",
            "call:open_database",
            "sqlite3.connect",
        ),
    ],
)
def test_cross_module_sqlite_mutation_alias_uses_canonical_escape_sink(
    import_statement: str,
    call: str,
    operation: str,
    sink: str,
) -> None:
    result = _discover(
        "from chronovisor.resources import DATABASE\n"
        "from chronovisor.mutator import mutate\n"
        "def caller():\n"
        "    mutate()\n"
        f"    {import_statement}\n"
        f"    {call}\n"
        "caller()\n",
        extra_sources={
            "src/chronovisor/mutator.py": (
                "import sqlite3\n"
                "def fake_connect(path):\n"
                "    return object()\n"
                "def mutate():\n"
                "    sqlite3.connect = fake_connect\n"
            )
        },
    )

    assert result["access_facts"] == []
    assert {
        (row["operation"], row["sink"])
        for row in result["escape_facts"]
    } == {(operation, sink)}


def test_cross_module_os_mutation_alias_uses_canonical_escape_sink() -> None:
    result = _discover(
        "from chronovisor.resources import DATABASE\n"
        "from chronovisor.mutator import mutate\n"
        "def caller():\n"
        "    mutate()\n"
        "    import os as operating_system\n"
        "    operating_system.open(DATABASE, operating_system.O_RDONLY)\n"
        "caller()\n",
        extra_sources={
            "src/chronovisor/mutator.py": (
                "import os\n"
                "def fake_open(path, flags):\n"
                "    return 1\n"
                "def mutate():\n"
                "    os.open = fake_open\n"
            )
        },
    )

    assert result["access_facts"] == []
    assert {
        (row["operation"], row["sink"])
        for row in result["escape_facts"]
    } == {("call:operating_system.open", "os.open")}


def test_alias_reconciliation_preserves_unrelated_concrete_site() -> None:
    result = _discover(
        "from chronovisor.resources import DATABASE\n"
        "from chronovisor.mutator import mutate\n"
        "def caller():\n"
        "    mutate()\n"
        "    import sqlite3 as database_api\n"
        "    database_api.connect(DATABASE)\n"
        "    DATABASE.read_text()\n"
        "caller()\n",
        extra_sources={
            "src/chronovisor/mutator.py": (
                "import sqlite3\n"
                "def fake_connect(path):\n"
                "    return object()\n"
                "def mutate():\n"
                "    sqlite3.connect = fake_connect\n"
            )
        },
    )

    assert _operations(result) == {"path.read_text"}
    assert {
        (row["operation"], row["sink"])
        for row in result["escape_facts"]
    } == {("call:database_api.connect", "sqlite3.connect")}


def test_sequential_known_local_callees_keep_os_mutation_state() -> None:
    result = _discover(
        "import os\n"
        "from chronovisor.resources import DATABASE\n"
        "def fake_open(path, flags):\n"
        "    return 1\n"
        "def mutate():\n"
        "    os.open = fake_open\n"
        "def use():\n"
        "    os.open(DATABASE, os.O_RDONLY)\n"
        "def caller():\n"
        "    mutate()\n"
        "    use()\n"
        "caller()\n"
    )

    assert result["access_facts"] == []
    assert {
        row["operation"] for row in result["escape_facts"]
    } == {"call:os.open"}


def test_read_only_then_use_sequence_preserves_sqlite_precision() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def inspect():\n"
        "    return sqlite3.__name__\n"
        "def use():\n"
        "    sqlite3.connect(DATABASE)\n"
        "def caller():\n"
        "    inspect()\n"
        "    use()\n"
        "caller()\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    assert result["escape_facts"] == []


@pytest.mark.parametrize("fallback", ["opaque()", "object()"])
def test_ambiguous_sqlite_connection_does_not_emit_concrete_handle_call(
    fallback: str,
) -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def opaque():\n"
        "    return object()\n"
        "def caller(condition):\n"
        f"    connection = sqlite3.connect(DATABASE) if condition else {fallback}\n"
        "    connection.execute('SELECT 1')\n"
    )

    assert _operations(result) == {"sqlite.connect:rwc"}
    assert {
        row["operation"] for row in result["escape_facts"]
    } == {"call:connection.execute"}


def test_same_kind_sqlite_connection_join_preserves_handle_precision() -> None:
    result = _discover(
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def caller(condition):\n"
        "    left = sqlite3.connect(DATABASE)\n"
        "    right = sqlite3.connect(DATABASE)\n"
        "    connection = left if condition else right\n"
        "    connection.execute('SELECT 1')\n"
    )

    assert _operations(result) == {
        "sqlite.connect:rwc",
        "sqlite.execute:read",
    }
    assert result["escape_facts"] == []


def test_sqlite_v2_ids_ignore_line_shift() -> None:
    source = (
        "import sqlite3\n"
        "from chronovisor.resources import DATABASE\n"
        "def use_database():\n"
        "    with sqlite3.connect(DATABASE) as connection:\n"
        "        connection.execute('SELECT 1').fetchall()\n"
    )

    base = _discover(source)
    shifted = _discover("\n\n" + source)

    assert len(base["access_fact_ids"]) == 3
    assert shifted["access_fact_ids"] == base["access_fact_ids"]
    assert shifted["provenance_ids"] == base["provenance_ids"]
    assert shifted["escape_fact_ids"] == base["escape_fact_ids"]
