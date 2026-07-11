"""Structural contract: routine data-plane code cannot start frontier work."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "llm_wiki_mcp"
ROUTINE_MODULES = (
    "codex_save.py",
    "claude_code_save.py",
    "ingest.py",
    "content_correction.py",
    "recall_improvement.py",
    "sleep_cycle.py",
    "converge_worker.py",
    "convergence.py",
    "hook_dispatcher.py",
    "background_jobs.py",
)


def _tree(filename: str) -> ast.Module:
    return ast.parse((PACKAGE_ROOT / filename).read_text(encoding="utf-8"))


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


def test_routine_modules_cannot_import_or_call_frontier_execution() -> None:
    for filename in ROUTINE_MODULES:
        tree = _tree(filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = {alias.name for alias in node.names}
                assert "run_frontier_review" not in imported, filename
        called = _called_names(tree)
        assert "run_frontier_review" not in called, filename
        assert "_run_codex" not in called, filename


def test_only_self_heal_can_call_guarded_frontier_repair() -> None:
    callers: dict[str, list[ast.Call]] = {}
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        calls = [
            node
            for node in ast.walk(_tree(path.name))
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "run_frontier_review"
            )
        ]
        if calls:
            callers[path.name] = calls

    assert set(callers) == {"self_heal.py"}
    for call in callers["self_heal.py"]:
        assert "evidence" in {keyword.arg for keyword in call.keywords}


def test_codex_subprocess_is_reachable_only_inside_guarded_entrypoint() -> None:
    tree = _tree("frontier_review.py")
    calling_functions: set[str] = set()
    structured_review: ast.FunctionDef | None = None
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        called = _called_names(node)
        if "_run_codex" in called:
            calling_functions.add(node.name)
        if node.name == "run_structured_review":
            structured_review = node

    assert calling_functions == {"run_frontier_review"}
    assert structured_review is not None
    structured_calls = _called_names(structured_review)
    assert "_run_codex" not in structured_calls
    assert "run" not in {
        node.func.attr
        for node in ast.walk(structured_review)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    }
