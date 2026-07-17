#!/usr/bin/env python3
"""Generate a deterministic repository-refactor inventory.

The report intentionally records candidates instead of declaring helpers or
entry points dead.  Runtime/config references and behavioral contracts still
need human classification before a refactor or deletion.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TEXT_SUFFIXES = {".json", ".md", ".plist", ".py", ".sh", ".toml", ".yaml", ".yml"}
IGNORED_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".venv", "__pycache__", "logs"}


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if not any(part in IGNORED_PARTS for part in path.parts)
    )


def _text_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix in TEXT_SUFFIXES or "scripts" in path.parts)
        and not any(part in IGNORED_PARTS for part in path.parts)
    )


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _call_name(call: ast.Call) -> str:
    node: ast.AST = call.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _literal_or_source(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)


class _FunctionInventory(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.scope: list[str] = []
        self.functions: list[dict[str, Any]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([*self.scope, node.name])
        calls = [item for item in ast.walk(node) if isinstance(item, ast.Call)]
        dumps_calls = [call for call in calls if _call_name(call).endswith("json.dumps")]
        sha_calls = [call for call in calls if _call_name(call).endswith("sha256")]
        flock_calls = [call for call in calls if _call_name(call).endswith("fcntl.flock")]
        replace_calls = [
            call
            for call in calls
            if _call_name(call) in {"os.replace", "Path.replace"}
            or _call_name(call).endswith(".replace")
        ]
        open_append = any(
            _call_name(call).endswith("open")
            and any(
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and "a" in arg.value
                for arg in call.args
            )
            for call in calls
        )
        json_options = []
        for call in dumps_calls:
            json_options.append(
                {
                    keyword.arg: _literal_or_source(keyword.value)
                    for keyword in call.keywords
                    if keyword.arg is not None
                }
            )
        signals: list[str] = []
        lowered = node.name.casefold()
        if "atomic" in lowered and "write" in lowered:
            signals.append("atomic-name")
        if replace_calls:
            signals.append("replace-call")
        if ("jsonl" in lowered or open_append) and dumps_calls:
            signals.append("jsonl-append")
        if sha_calls and dumps_calls:
            signals.append("canonical-json-hash")
        if flock_calls:
            signals.append("flock")
        end_lineno = node.end_lineno or node.lineno
        self.functions.append(
            {
                "path": self.path,
                "qualname": qualname,
                "line": node.lineno,
                "end_line": end_lineno,
                "lines": end_lineno - node.lineno + 1,
                "signals": signals,
                "json_dumps_options": json_options,
            }
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def _ingest_aliases(tree: ast.AST) -> set[str]:
    aliases = {"ingest"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "llm_wiki_mcp":
            for name in node.names:
                if name.name == "ingest":
                    aliases.add(name.asname or name.name)
        elif isinstance(node, ast.Import):
            for name in node.names:
                if name.name == "llm_wiki_mcp.ingest":
                    aliases.add(name.asname or "ingest")
    return aliases


def _ingest_references(tree: ast.AST) -> tuple[Counter[str], Counter[str]]:
    aliases = _ingest_aliases(tree)
    attributes: Counter[str] = Counter()
    patches: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in aliases:
                attributes[node.attr] += 1
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "setattr" or len(node.args) < 2:
            continue
        owner, name = node.args[:2]
        if (
            isinstance(owner, ast.Name)
            and owner.id in aliases
            and isinstance(name, ast.Constant)
            and isinstance(name.value, str)
        ):
            patches[name.value] += 1
        elif isinstance(owner, ast.Constant) and isinstance(owner.value, str):
            prefix = "llm_wiki_mcp.ingest."
            if owner.value.startswith(prefix):
                patches[owner.value.removeprefix(prefix).split(".", 1)[0]] += 1
    return attributes, patches


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def _reference_count(needle: str, files: Iterable[Path], *, exclude: Path | None = None) -> int:
    count = 0
    for path in files:
        if exclude is not None and path == exclude:
            continue
        try:
            count += path.read_text(encoding="utf-8", errors="ignore").count(needle)
        except OSError:
            continue
    return count


def scan_repository(root: Path) -> dict[str, Any]:
    root = root.resolve()
    src_root = root / "src" / "llm_wiki_mcp"
    python_files = _python_files(src_root)
    repository_python = _python_files(root)
    text_files = _text_files(root)
    functions: list[dict[str, Any]] = []
    ingest_attributes: Counter[str] = Counter()
    ingest_patches: Counter[str] = Counter()
    python_lines = 0

    for path in repository_python:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        if path.is_relative_to(src_root):
            python_lines += len(text.splitlines())
            visitor = _FunctionInventory(_relative(path, root))
            visitor.visit(tree)
            functions.extend(visitor.functions)
        attributes, patches = _ingest_references(tree)
        ingest_attributes.update(attributes)
        ingest_patches.update(patches)

    functions.sort(key=lambda row: (row["path"], row["line"], row["qualname"]))
    large_functions = [row for row in functions if row["lines"] >= 200]
    signal_groups = {
        signal: [
            {
                key: row[key]
                for key in ("path", "qualname", "line", "lines", "json_dumps_options")
            }
            for row in functions
            if signal in row["signals"]
        ]
        for signal in (
            "atomic-name",
            "replace-call",
            "jsonl-append",
            "canonical-json-hash",
            "flock",
        )
    }

    pyproject_path = root / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    entrypoints = pyproject.get("project", {}).get("scripts", {})
    script_rows = []
    scripts_root = root / "scripts"
    if scripts_root.exists():
        for path in sorted(item for item in scripts_root.iterdir() if item.is_file()):
            script_rows.append(
                {
                    "path": _relative(path, root),
                    "basename_references": _reference_count(path.name, text_files, exclude=path),
                    "stem_references": _reference_count(path.stem, text_files, exclude=path),
                }
            )

    return {
        "schema_version": 1,
        "baseline_commit": _git_commit(root),
        "source": {
            "python_modules": len(python_files),
            "python_lines": python_lines,
            "functions": len(functions),
            "functions_ge_200": sum(row["lines"] >= 200 for row in functions),
            "functions_ge_300": sum(row["lines"] >= 300 for row in functions),
        },
        "large_functions": large_functions,
        "candidate_signals": signal_groups,
        "ingest_seams": {
            "attribute_references": dict(sorted(ingest_attributes.items())),
            "monkeypatch_targets": dict(sorted(ingest_patches.items())),
        },
        "console_entrypoints": [
            {
                "name": name,
                "target": target,
                "repository_references": _reference_count(name, text_files, exclude=pyproject_path),
            }
            for name, target in sorted(entrypoints.items())
        ],
        "scripts": script_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.dumps(scan_repository(args.root), ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
