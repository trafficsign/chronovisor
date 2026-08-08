# ruff: noqa: F401, F403, F405
"""Runtime ownership discovery layer."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import plistlib
import re
import shlex
import subprocess
import tarfile
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from .model import *
from .source import *


def _socket_discovery(snapshot: dict[str, bytes]) -> list[dict[str, Any]]:
    specs = [
        {
            "name": "semantic-unix-socket",
            "path": "src/chronovisor/core/runtime_config.py",
            "needle": 'socket: str = "~/.chronovisor/runtime/semantic.sock"',
            "owner_symbol": "chronovisor.search.semantic_service:serve",
            "address": "unix://$HOME/.chronovisor/runtime/semantic.sock",
            "clients": ["chronovisor.search.semantic_client:request"],
            "compatibility": ["config:search.embedding.socket"],
            "socket_contract": {
                "filesystem_mode": "0600",
                "startup": "unlink-stale-then-bind",
            },
        },
        {
            "name": "reranker-unix-socket",
            "path": "src/chronovisor/core/runtime_config.py",
            "needle": 'socket: str = "~/.chronovisor/runtime/reranker.sock"',
            "owner_symbol": "chronovisor.search.reranker_service:serve",
            "address": "unix://$HOME/.chronovisor/runtime/reranker.sock",
            "clients": ["chronovisor.search.reranker_client:request"],
            "compatibility": ["config:search.reranker.service.socket"],
            "socket_contract": {
                "filesystem_mode": "0600",
                "startup": "unlink-stale-then-bind",
            },
        },
        {
            "name": "dashboard-http-socket",
            "path": "src/chronovisor/ops/dashboard.py",
            "needle": 'parser.add_argument("--port", type=int, default=8765)',
            "owner_symbol": "chronovisor.ops.dashboard:serve",
            "address": "tcp://0.0.0.0:8765",
            "clients": ["chronovisor.ops.burn_monitor:dashboard_snapshot"],
            "compatibility": [
                "bind-default:127.0.0.1:8765",
                "launchd-lan-bind:0.0.0.0:8765",
                "client-alias:127.0.0.1:8765",
                "access-token-and-credentials-required",
            ],
        },
        {
            "name": "ollama-http-socket",
            "path": "src/chronovisor/core/ollama.py",
            "needle": 'OLLAMA_URL = "http://localhost:11434"',
            "owner_symbol": "external:ollama",
            "address": "tcp://127.0.0.1:11434",
            "clients": [
                "chronovisor.core.ollama:_client",
                "chronovisor.ops.dashboard:DashboardHandler",
                "chronovisor.ops.burn_monitor:ollama_snapshot",
                "chronovisor.lab.local_model_eval:_live_transport",
            ],
            "compatibility": ["external-service:ollama", "http-api:11434"],
        },
        {
            "name": "searxng-http-socket",
            "path": "scripts/chronovisor-searxng",
            "needle": 'export SEARXNG_PORT="8888"',
            "owner_symbol": "script:scripts/chronovisor-searxng",
            "address": "tcp://127.0.0.1:8888",
            "clients": ["chronovisor.research.web_provider:search_web"],
            "compatibility": [
                "external-service:searxng",
                "launchd:com.trafficsign.chronovisor-searxng",
            ],
        },
        {
            "name": "mcp-stdio-endpoint",
            "path": "src/chronovisor/hosts/server.py",
            "needle": "def main():",
            "owner_symbol": "chronovisor.hosts.server:main",
            "address": "stdio://chronovisor-mcp",
            "clients": ["external:codex-or-claude-mcp-host"],
            "compatibility": ["entrypoint:chronovisor-mcp", "transport:stdio"],
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = str(spec["path"])
        line = _line_containing(_text(snapshot, path), str(spec["needle"]))
        owner_symbol = str(spec["owner_symbol"])
        if owner_symbol.startswith("script:"):
            module = owner_symbol
            symbol = Path(owner_symbol.removeprefix("script:")).name
        elif owner_symbol.startswith("external:"):
            module = "external"
            symbol = owner_symbol.split(":", maxsplit=1)[1]
        else:
            module, symbol = owner_symbol.split(":", maxsplit=1)
        row: dict[str, Any] = {
            "classification": "resource",
            "path": path,
            "line": line,
            "module": module,
            "symbol": symbol,
            "owner_symbol": owner_symbol,
            "kind": "socket",
            "locator": {"type": "socket", "value": spec["address"]},
            "socket": {
                "address": spec["address"],
                "server": owner_symbol,
                "clients": spec["clients"],
                **dict(spec.get("socket_contract", {})),
            },
            "compatibility": spec["compatibility"],
        }
        row["discovery_id"] = _discovery_id(row)
        rows.append(row)
    return rows


def _project_entrypoints(snapshot: dict[str, bytes]) -> dict[str, str]:
    project = tomllib.loads(_text(snapshot, "pyproject.toml"))
    scripts = project.get("project", {}).get("scripts", {})
    if not isinstance(scripts, dict):
        return {}
    return {str(name): str(target) for name, target in sorted(scripts.items())}


def _owner_package(reference: str) -> str:
    if reference.startswith("script:"):
        return "search"
    module = reference.split(":", maxsplit=1)[0]
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "chronovisor" else parts[0]


def _worker_row(
    *,
    path: str,
    line: int,
    locator_type: str,
    locator_value: str,
    owner_symbol: str,
    module: str,
    entrypoint: str,
    launchd_label: str | None,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    symbol = owner_symbol.split(":", maxsplit=1)[-1]
    worker = {"module": module, "entrypoint": entrypoint}
    if launchd_label is not None:
        worker["launchd_label"] = launchd_label
    row: dict[str, Any] = {
        "classification": "resource",
        "path": path,
        "line": line,
        "module": module,
        "symbol": symbol,
        "owner_symbol": owner_symbol,
        "kind": "worker",
        "locator": {"type": locator_type, "value": locator_value},
        "worker": worker,
        "compatibility": [f"{locator_type}:{locator_value}"],
    }
    if evidence:
        row["additional_evidence"] = evidence
    row["discovery_id"] = _discovery_id(row)
    return row


def _joined_shell_commands(
    wrapper: str,
) -> list[tuple[tuple[str, ...], tuple[tuple[int, str], ...]]]:
    commands: list[tuple[tuple[str, ...], tuple[tuple[int, str], ...]]] = []
    segments: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(wrapper.splitlines(), start=1):
        stripped = raw_line.rstrip()
        continued = stripped.endswith("\\")
        segment = stripped[:-1] if continued else stripped
        if not segments and (not segment.strip() or segment.lstrip().startswith("#")):
            continue
        segments.append((line_number, segment.strip()))
        if continued:
            continue
        logical = " ".join(text for _line, text in segments)
        try:
            tokens = tuple(shlex.split(logical, comments=True, posix=True))
        except ValueError as exc:
            raise ValueError(
                f"invalid shell command spanning line {segments[0][0]}: {exc}"
            ) from exc
        if tokens:
            commands.append((tokens, tuple(segments)))
        segments = []
    if segments:
        raise ValueError(
            f"unterminated shell continuation starting at line {segments[0][0]}"
        )
    return commands


def _entrypoint_commands(
    wrapper: str, entrypoint: str
) -> list[tuple[tuple[str, ...], tuple[tuple[int, str], ...]]]:
    return [
        command
        for command in _joined_shell_commands(wrapper)
        if entrypoint in command[0]
    ]


def _entrypoint_line(
    segments: tuple[tuple[int, str], ...], entrypoint: str
) -> int:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_-]){re.escape(entrypoint)}(?![A-Za-z0-9_-])"
    )
    matches = [line for line, text in segments if pattern.search(text)]
    if len(matches) != 1:
        raise ValueError(f"entrypoint must appear on exactly one line: {entrypoint}")
    return matches[0]


def _canonical_uvx_python(tokens: tuple[str, ...]) -> tuple[str, ...]:
    indexes = [index for index, token in enumerate(tokens) if token == "--python"]
    if len(indexes) != 1:
        raise ValueError("uvx command must have exactly one --python option")
    index = indexes[0]
    expected_index = 2 if tokens and tokens[0] == "exec" else 1
    if index != expected_index or index + 1 >= len(tokens):
        raise ValueError("uvx --python option is not in the canonical position")
    if tokens[index + 1] != "/opt/homebrew/bin/python3.14":
        raise ValueError("uvx --python must select the standard CPython executable")
    return (*tokens[:index], *tokens[index + 2 :])


def _librarian_invocations(
    wrapper: str, forwarded: list[str]
) -> list[dict[str, Any]]:
    label = "com.trafficsign.chronovisor-librarian-review"
    if forwarded:
        raise ValueError(f"{label} must not receive launchd forwarded arguments")
    entrypoint = "chronovisor-librarian"
    uvx = [
        "/Users/trafficsign/.local/bin/uvx",
        "--refresh-package",
        "chronovisor",
        "--from",
        "git+ssh://git@github.com/trafficsign/chronovisor",
        entrypoint,
    ]
    root = "/Users/trafficsign/.chronovisor"
    expected = {
        "full-sweep": tuple(
            [*uvx, "--root", root, "--full-sweep", "--json", ">/dev/null"]
        ),
        "collection-primary": tuple(
            [
                *uvx,
                "--root",
                root,
                "--review-collection-queue",
                "--limit",
                "${CHRONOVISOR_LIBRARIAN_REVIEW_LIMIT:-5}",
                "--review-model",
                "${CHRONOVISOR_LIBRARIAN_REVIEW_MODEL:-gemma4:26b}",
                "--review-role",
                "primary",
                "--json",
            ]
        ),
        "collection-challenger": tuple(
            [
                "exec",
                *uvx,
                "--root",
                root,
                "--review-collection-queue",
                "--limit",
                "${CHRONOVISOR_LIBRARIAN_CHALLENGE_LIMIT:-5}",
                "--review-model",
                "${CHRONOVISOR_LIBRARIAN_CHALLENGER_MODEL:-gpt-oss:20b}",
                "--review-role",
                "challenger",
                "--json",
            ]
        ),
    }
    discovered: dict[
        str, tuple[tuple[str, ...], tuple[tuple[int, str], ...]]
    ] = {}
    role_order: list[str] = []
    for tokens, segments in _entrypoint_commands(wrapper, entrypoint):
        role_indexes = [
            index for index, token in enumerate(tokens) if token == "--review-role"
        ]
        if "--full-sweep" in tokens:
            if role_indexes:
                raise ValueError(f"{label} full-sweep must not have a review role")
            role = "full-sweep"
        else:
            if len(role_indexes) != 1 or role_indexes[0] + 1 >= len(tokens):
                raise ValueError(f"{label} review command must have exactly one role")
            review_role = tokens[role_indexes[0] + 1]
            if review_role not in {"primary", "challenger"}:
                raise ValueError(f"{label} has unknown review role: {review_role}")
            role = f"collection-{review_role}"
        if role in discovered:
            raise ValueError(f"{label} has duplicate role: {role}")
        discovered[role] = (tokens, segments)
        role_order.append(role)
    required_order = [
        "full-sweep",
        "collection-primary",
        "collection-challenger",
    ]
    if role_order != required_order:
        raise ValueError(
            f"{label} must contain exact ordered roles: {required_order}; "
            f"found={role_order}"
        )
    for role in required_order:
        tokens, _segments = discovered[role]
        if _canonical_uvx_python(tokens) != expected[role]:
            raise ValueError(f"{label} command drifted for role: {role}")
    invocations: list[dict[str, Any]] = []
    for role in required_order:
        tokens, segments = discovered[role]
        canonical_tokens = _canonical_uvx_python(tokens)
        entrypoint_index = canonical_tokens.index(entrypoint)
        argv = list(canonical_tokens[entrypoint_index + 1 :])
        if role == "full-sweep" and (not argv or argv.pop() != ">/dev/null"):
            raise ValueError(f"{label} full-sweep redirection drifted")
        argv = ["$CHRONOVISOR_ROOT" if value == root else value for value in argv]
        invocations.append(
            {
                "entrypoint": entrypoint,
                "argv": argv,
                "role": role,
                "line": _entrypoint_line(segments, entrypoint),
            }
        )
    return invocations


def _library_evidence_invocations(
    wrapper: str, forwarded: list[str]
) -> list[dict[str, Any]]:
    label = "com.trafficsign.chronovisor-library-evidence"
    wrapper_path = "scripts/chronovisor-library-evidence"
    entrypoint = "chronovisor-lab"
    runtime_source = (
        "${CHRONOVISOR_RUNTIME_SOURCE:-"
        "git+ssh://git@github.com/trafficsign/chronovisor}"
    )
    search_path = (
        "/Users/trafficsign/.local/bin:/opt/homebrew/bin:/usr/local/bin:"
        "/usr/bin:/bin:/opt/homebrew/sbin:/usr/sbin:/sbin"
    )
    expected_wrapper = (
        "#!/bin/sh\n"
        "set -eu\n"
        "\n"
        'PROJECT_DIR="/Users/trafficsign/projects/personal/chronovisor"\n'
        f'RUNTIME_SOURCE="{runtime_source}"\n'
        f'PATH="{search_path}"\n'
        'export PATH CHRONOVISOR_REPO_ROOT="'
        '${CHRONOVISOR_REPO_ROOT:-$PROJECT_DIR}"\n'
        "\n"
        "exec uvx --python /opt/homebrew/bin/python3.14 "
        "--refresh-package chronovisor --from \"$RUNTIME_SOURCE\" \\\n"
        '  chronovisor-lab classification-library-pilot "$@"\n'
    )
    if wrapper != expected_wrapper:
        raise ValueError(f"{label} wrapper source drifted")
    commands = _joined_shell_commands(wrapper)
    expected = (
        "exec",
        "uvx",
        "--refresh-package",
        "chronovisor",
        "--from",
        "$RUNTIME_SOURCE",
        entrypoint,
        "classification-library-pilot",
        "$@",
    )
    expected_commands = (
        ("set", "-eu"),
        ("PROJECT_DIR=/Users/trafficsign/projects/personal/chronovisor",),
        (f"RUNTIME_SOURCE={runtime_source}",),
        (f"PATH={search_path}",),
        (
            "export",
            "PATH",
            "CHRONOVISOR_REPO_ROOT=${CHRONOVISOR_REPO_ROOT:-$PROJECT_DIR}",
        ),
        expected,
    )
    normalized_commands: list[tuple[str, ...]] = []
    for index, (tokens, _segments) in enumerate(commands):
        normalized_commands.append(
            _canonical_uvx_python(tokens) if index == len(commands) - 1 else tokens
        )
    if tuple(normalized_commands) != expected_commands:
        raise ValueError(f"{label} wrapper command grammar drifted")
    _tokens, runtime_source_segments = commands[2]
    _tokens, search_path_segments = commands[3]
    _tokens, segments = commands[5]
    return [
        {
            "entrypoint": entrypoint,
            "argv": ["classification-library-pilot", *forwarded],
            "role": "classification-library-pilot",
            "line": _entrypoint_line(segments, entrypoint),
            "runtime": {
                "executable": "uvx",
                "resolution": "PATH",
                "search_path": search_path,
                "source": runtime_source,
                "evidence": [
                    {
                        "path": wrapper_path,
                        "line": runtime_source_segments[0][0],
                    },
                    {"path": wrapper_path, "line": search_path_segments[0][0]},
                    {
                        "path": wrapper_path,
                        "line": _entrypoint_line(segments, "uvx"),
                    },
                ],
            },
            "linked_worker": {
                "kind": "worker",
                "locator_type": "lab_dispatch",
                "locator_value": "classification-library-pilot",
            },
        }
    ]


def _launchd_invocations(
    *,
    label: str,
    wrapper: str,
    arguments: list[Any],
    entrypoints: dict[str, str],
) -> list[dict[str, Any]]:
    special_arguments = {
        "com.trafficsign.chronovisor-librarian-review": [
            "/Users/trafficsign/projects/personal/chronovisor/"
            "scripts/chronovisor-librarian-review"
        ],
        "com.trafficsign.chronovisor-library-evidence": [
            "/Users/trafficsign/projects/personal/chronovisor/"
            "scripts/chronovisor-library-evidence",
            "run-once",
            "--repo-root",
            "/Users/trafficsign/projects/personal/chronovisor",
        ],
    }
    expected_arguments = special_arguments.get(label)
    if expected_arguments is not None and arguments != expected_arguments:
        raise ValueError(f"{label} launchd ProgramArguments drifted")
    forwarded = [str(value) for value in arguments[1:]]
    if label == "com.trafficsign.chronovisor-librarian-review":
        return _librarian_invocations(wrapper, forwarded)
    if label == "com.trafficsign.chronovisor-library-evidence":
        return _library_evidence_invocations(wrapper, forwarded)
    if label == "com.trafficsign.chronovisor-searxng":
        return [
            {
                "entrypoint": "external:granian",
                "argv": ["--host", "127.0.0.1", "--port", "8888", "--workers", "1"],
                "role": "searxng-server",
                "line": 30,
            }
        ]
    matches = [
        name
        for name in sorted(entrypoints, key=lambda item: (-len(item), item))
        if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])", wrapper)
    ]
    if not matches:
        return []
    entrypoint = matches[0]
    return [
        {
            "entrypoint": entrypoint,
            "argv": forwarded,
            "role": "launchd-service",
            "line": _line_containing(wrapper, entrypoint),
        }
    ]


def _worker_discovery(snapshot: dict[str, bytes]) -> list[dict[str, Any]]:
    entrypoints = _project_entrypoints(snapshot)
    pyproject = _text(snapshot, "pyproject.toml")
    rows: list[dict[str, Any]] = []
    for name, target in entrypoints.items():
        module, symbol = target.split(":", maxsplit=1)
        rows.append(
            _worker_row(
                path="pyproject.toml",
                line=_line_containing(pyproject, f'{name} = "{target}"'),
                locator_type="entrypoint",
                locator_value=name,
                owner_symbol=f"{module}:{symbol}",
                module=module,
                entrypoint=name,
                launchd_label=None,
            )
        )
    for path in sorted(snapshot):
        if not path.startswith("launchd/") or not path.endswith(".plist"):
            continue
        raw = snapshot[path]
        payload = plistlib.loads(raw)
        label = str(payload.get("Label") or "")
        arguments = payload.get("ProgramArguments")
        if not label or not isinstance(arguments, list) or not arguments:
            raise ValueError(f"invalid tracked launchd worker: {path}")
        wrapper_name = Path(str(arguments[0])).name
        wrapper_path = f"scripts/{wrapper_name}"
        wrapper = _text(snapshot, wrapper_path)
        invocations = _launchd_invocations(
            label=label,
            wrapper=wrapper,
            arguments=arguments,
            entrypoints=entrypoints,
        )
        matches = [
            name
            for name in sorted(entrypoints, key=lambda item: (-len(item), item))
            if re.search(
                rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])",
                wrapper,
            )
        ]
        if matches:
            entrypoint = matches[0]
            target_module, target_symbol = entrypoints[entrypoint].split(
                ":", maxsplit=1
            )
            owner_symbol = f"{target_module}:{target_symbol}"
            module = target_module
        else:
            entrypoint = wrapper_path
            module = f"script:{wrapper_path}"
            owner_symbol = module
        plist_text = raw.decode("utf-8")
        worker_row = _worker_row(
            path=path,
            line=_line_containing(plist_text, label),
            locator_type="launchd",
            locator_value=label,
            owner_symbol=owner_symbol,
            module=module,
            entrypoint=entrypoint,
            launchd_label=label,
            evidence=[
                {"path": wrapper_path, "line": int(invocation["line"])}
                for invocation in invocations
            ],
        )
        source_backed_invocations = label in {
            "com.trafficsign.chronovisor-librarian-review",
            "com.trafficsign.chronovisor-library-evidence",
        }
        worker_row["worker"]["invocations"] = []
        for invocation in invocations:
            normalized = {
                key: value for key, value in invocation.items() if key != "line"
            }
            if source_backed_invocations:
                normalized["evidence"] = {
                    "path": wrapper_path,
                    "line": int(invocation["line"]),
                }
            worker_row["worker"]["invocations"].append(normalized)
        rows.append(worker_row)
    rows.extend(_lab_dispatch_workers(snapshot))
    rows.extend(_python_module_workers(snapshot))
    return rows


def _lab_dispatch_workers(snapshot: dict[str, bytes]) -> list[dict[str, Any]]:
    path = "src/chronovisor/lab/cli.py"
    tree = ast.parse(_text(snapshot, path), filename=path)
    rows: list[dict[str, Any]] = []
    for node in tree.body:
        names, value = _assignment_names(node)
        if "COMMANDS" not in names or not isinstance(value, ast.Dict):
            continue
        for key, item in zip(value.keys, value.values, strict=True):
            if (
                not isinstance(key, ast.Constant)
                or not isinstance(key.value, str)
                or not isinstance(item, ast.Tuple)
                or not item.elts
                or not isinstance(item.elts[0], ast.Constant)
                or not isinstance(item.elts[0].value, str)
            ):
                raise ValueError(
                    "lab COMMANDS must contain literal command/module pairs"
                )
            command = key.value
            module = item.elts[0].value
            row = _worker_row(
                path=path,
                line=int(key.lineno),
                locator_type="lab_dispatch",
                locator_value=command,
                owner_symbol=f"{module}:main",
                module=module,
                entrypoint=f"chronovisor-lab {command}",
                launchd_label=None,
            )
            row["symbol"] = f"COMMANDS[{command}]"
            row["discovery_id"] = _discovery_id(row)
            rows.append(row)
    return rows


class _PythonModuleWorkerCollector(ast.NodeVisitor):
    def __init__(self, *, module: str, path: str) -> None:
        self.module = module
        self.path = path
        self.scope: list[str] = []
        self.occurrences: Counter[tuple[str, str]] = Counter()
        self.rows: list[dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_List(self, node: ast.List) -> None:
        self._record_sequence(node, node.elts)
        self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        self._record_sequence(node, node.elts)
        self.generic_visit(node)

    def _record_sequence(self, node: ast.expr, values: list[ast.expr]) -> None:
        if (
            len(values) < 3
            or not isinstance(values[0], ast.Attribute)
            or not isinstance(values[0].value, ast.Name)
            or values[0].value.id != "sys"
            or values[0].attr != "executable"
            or not isinstance(values[1], ast.Constant)
            or values[1].value != "-m"
            or not isinstance(values[2], ast.Constant)
            or not isinstance(values[2].value, str)
            or not values[2].value.startswith("chronovisor.")
        ):
            return
        target = values[2].value
        scope = ".".join(self.scope) or "<module>"
        occurrence_key = (scope, target)
        self.occurrences[occurrence_key] += 1
        row = _worker_row(
            path=self.path,
            line=int(node.lineno),
            locator_type="module_worker",
            locator_value=target,
            owner_symbol=f"{target}:main",
            module=target,
            entrypoint=f"python -m {target}",
            launchd_label=None,
        )
        row["symbol"] = (
            f"python-module:{target}:{scope}:{self.occurrences[occurrence_key]}"
        )
        row["discovery_id"] = _discovery_id(row)
        self.rows.append(row)


def _python_module_workers(snapshot: dict[str, bytes]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(snapshot):
        if not path.startswith("src/chronovisor/") or not path.endswith(".py"):
            continue
        module = _module_name(path)
        collector = _PythonModuleWorkerCollector(module=module, path=path)
        collector.visit(ast.parse(_text(snapshot, path), filename=path))
        rows.extend(collector.rows)
    return rows


class _LockProtocolCollector(ast.NodeVisitor):
    def __init__(self, *, module: str, path: str, index: _SourceIndex) -> None:
        self.module = module
        self.path = path
        self.index = index
        self.scope: list[str] = []
        self.occurrences: Counter[tuple[str, str]] = Counter()
        self.rows: list[dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        operation = ""
        protocol = ""
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "fcntl"
            and node.func.attr == "flock"
            and len(node.args) > 1
        ):
            operation = ast.unparse(node.args[1])
            if "LOCK_UN" in operation:
                self.generic_visit(node)
                return
            protocol = _flock_protocol(operation)
        else:
            helper = self._lock_helper_name(node.func)
            if helper:
                operation = f"helper:{helper}"
                protocol = "helper-managed-lock"
        if protocol:
            scope = ".".join(self.scope) or "<module>"
            occurrence_key = (scope, operation)
            self.occurrences[occurrence_key] += 1
            row: dict[str, Any] = {
                "classification": "lock_protocol",
                "path": self.path,
                "line": int(node.lineno),
                "module": self.module,
                "symbol": "flock" if not operation.startswith("helper:") else operation,
                "scope": scope,
                "operation": operation,
                "protocol": protocol,
                "occurrence": self.occurrences[occurrence_key],
            }
            row["discovery_id"] = _discovery_id(row)
            self.rows.append(row)
        self.generic_visit(node)

    def _lock_helper_name(self, expression: ast.expr) -> str:
        helpers = {
            "file_lock",
            "exclusive_text_file_lock",
            "sidecar_exclusive_lock",
            "_search_label_queue_lock",
            "_claims_ledger_lock",
        }
        if isinstance(expression, ast.Name):
            if expression.id in helpers:
                return expression.id
            imported = self.index.imports.get((self.module, expression.id))
            if imported is not None and imported[1] in helpers:
                return str(imported[1])
        if isinstance(expression, ast.Attribute) and expression.attr in helpers:
            return expression.attr
        return ""


def _flock_protocol(operation: str) -> str:
    has_exclusive = "LOCK_EX" in operation
    has_shared = "LOCK_SH" in operation
    has_nonblocking = "LOCK_NB" in operation
    if has_exclusive and has_shared:
        return "exclusive-or-shared"
    if has_exclusive and has_nonblocking:
        return "exclusive-nonblocking"
    if has_shared and has_nonblocking:
        return "shared-nonblocking"
    if has_exclusive:
        return "exclusive"
    if has_shared:
        return "shared"
    if has_nonblocking:
        return "indirect-nonblocking"
    return "indirect-operation"


def _lock_protocol_discovery(index: _SourceIndex) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module, tree in sorted(index.trees.items()):
        collector = _LockProtocolCollector(
            module=module,
            path=index.paths[module],
            index=index,
        )
        collector.visit(tree)
        rows.extend(collector.rows)
    return rows


def _planned_lock_protocol_discovery(
    snapshot: dict[str, bytes],
) -> list[dict[str, Any]]:
    """Keep the P1 sidecar-lock inventory stable across the frozen source head."""

    specs = (
        (
            "src/chronovisor/ops/golden_expand.py",
            "chronovisor.ops.golden_expand",
            "expand_golden_from_recall_questions",
            "helper:_search_label_queue_lock",
            "helper-managed-lock",
            "def expand_golden_from_recall_questions(",
        ),
        (
            "src/chronovisor/search/search_eval.py",
            "chronovisor.search.search_eval",
            "build_label_queue",
            "helper:_search_label_queue_lock",
            "helper-managed-lock",
            "def build_label_queue(",
        ),
        (
            "src/chronovisor/search/search_eval.py",
            "chronovisor.search.search_eval",
            "review_label_queue_with_frontier",
            "helper:_search_label_queue_lock",
            "helper-managed-lock",
            "def review_label_queue_with_frontier(",
        ),
        (
            "src/chronovisor/ops/golden_expand.py",
            "chronovisor.ops.golden_expand",
            "_search_label_queue_lock",
            "fcntl.LOCK_EX",
            "exclusive",
            'LABEL_QUEUE_FILE = RECALL_DIR / "search-label-queue.jsonl"',
        ),
        (
            "src/chronovisor/search/search_eval.py",
            "chronovisor.search.search_eval",
            "_search_label_queue_lock",
            "fcntl.LOCK_EX",
            "exclusive",
            'LABEL_QUEUE_FILE = RECALL_DIR / "search-label-queue.jsonl"',
        ),
        (
            "src/chronovisor/recall/claims.py",
            "chronovisor.recall.claims",
            "append_page_claims",
            "helper:_claims_ledger_lock",
            "helper-managed-lock",
            "def append_page_claims(",
        ),
        (
            "src/chronovisor/recall/claims.py",
            "chronovisor.recall.claims",
            "sanitize_claim_ledger",
            "helper:_claims_ledger_lock",
            "helper-managed-lock",
            "def sanitize_claim_ledger(",
        ),
        (
            "src/chronovisor/recall/claims.py",
            "chronovisor.recall.claims",
            "_claims_ledger_lock",
            "fcntl.LOCK_EX",
            "exclusive",
            'CLAIMS_FILE = CLAIMS_DIR / "claims.jsonl"',
        ),
    )
    rows: list[dict[str, Any]] = []
    for path, module, scope, operation, protocol, needle in specs:
        row: dict[str, Any] = {
            "classification": "lock_protocol",
            "path": path,
            "line": _line_containing(_text(snapshot, path), needle),
            "module": module,
            "symbol": operation if operation.startswith("helper:") else "flock",
            "scope": scope,
            "operation": operation,
            "protocol": protocol,
            "occurrence": 1,
        }
        row["discovery_id"] = _discovery_id(row)
        rows.append(row)
    return rows


def discover(
    snapshot: dict[str, bytes], *, include_planned: bool = False
) -> tuple[_SourceIndex, dict[str, Any]]:
    index = _SourceIndex(snapshot)
    rows = [
        *_ast_discovery(index),
        *_explicit_state_discovery(snapshot, include_planned=include_planned),
        *_socket_discovery(snapshot),
        *_worker_discovery(snapshot),
        *(_planned_lock_protocol_discovery(snapshot) if include_planned else []),
        *_lock_protocol_discovery(index),
    ]
    # A planned row is replaced by its concrete AST call after the source
    # retrofit, but keeps the same semantic identity in the frozen seed.
    rows = list({str(row["discovery_id"]): row for row in rows}.values())
    rows.sort(key=lambda row: (str(row["path"]), int(row["line"]), str(row["symbol"])))
    resources = [row for row in rows if row["classification"] == "resource"]
    exclusions = [row for row in rows if row["classification"] == "exclusion"]
    lock_protocols = [row for row in rows if row["classification"] == "lock_protocol"]
    return index, {
        "rows": rows,
        "resource_candidates": resources,
        "exclusion_candidates": exclusions,
        "lock_protocol_candidates": lock_protocols,
    }


__all__ = [
    "_socket_discovery",
    "_project_entrypoints",
    "_owner_package",
    "_worker_row",
    "_worker_discovery",
    "_lab_dispatch_workers",
    "_PythonModuleWorkerCollector",
    "_python_module_workers",
    "_LockProtocolCollector",
    "_flock_protocol",
    "_lock_protocol_discovery",
    "_planned_lock_protocol_discovery",
    "discover",
]
