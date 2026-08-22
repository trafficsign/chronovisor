"""Regression guard for every function in the 300-line refactor inventory."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = ROOT / "docs" / "refactoring" / "large-function-policy.toml"


class _Functions(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.lines: dict[str, int] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        qualname = ".".join([*self.scope, node.name])
        self.lines[qualname] = (node.end_lineno or node.lineno) - node.lineno + 1
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def _policy() -> dict[str, Any]:
    return tomllib.loads(POLICY_FILE.read_text(encoding="utf-8"))


def _source_functions(path: str) -> dict[str, int]:
    source = ROOT / path
    visitor = _Functions()
    visitor.visit(ast.parse(source.read_text(encoding="utf-8"), filename=str(source)))
    return visitor.lines


def test_large_function_policy_covers_every_current_large_function() -> None:
    policy = _policy()
    threshold = int(policy["threshold_lines"])
    entries = policy["functions"]
    assert len(entries) == int(policy["baseline_function_count"]) == 49

    retained: set[tuple[str, str]] = set()
    declared: set[tuple[str, str]] = set()
    functions_by_path: dict[str, dict[str, int]] = {}
    for entry in entries:
        key = (entry["path"], entry["qualname"])
        assert key not in declared
        declared.add(key)
        assert entry["disposition"] in {"decomposed", "retained"}
        assert len(entry["rationale"]) >= 80
        assert entry["tests"]
        assert all((ROOT / test_path).is_file() for test_path in entry["tests"])

        functions = functions_by_path.setdefault(
            entry["path"], _source_functions(entry["path"])
        )
        assert entry["qualname"] in functions
        current_lines = functions[entry["qualname"]]
        assert current_lines <= int(entry["max_lines"])
        if entry["disposition"] == "decomposed":
            assert current_lines < threshold
        else:
            retained.add(key)

    current_large: set[tuple[str, str]] = set()
    for path in sorted((ROOT / "src" / "chronovisor").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        for qualname, lines in _source_functions(relative).items():
            if lines >= threshold:
                current_large.add((relative, qualname))

    assert current_large == retained
