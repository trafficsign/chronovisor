"""Structural contract: routine data-plane code cannot start frontier work."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "chronovisor"
ROUTINE_MODULES = (
    "raw/codex_record.py",
    "hosts/claude_code_record.py",
    "ingest/ingest.py",
    "recall/content_correction.py",
    "recall/recall_improvement.py",
    "ops/sleep_cycle.py",
    "ops/converge_worker.py",
    "ops/convergence.py",
    "hosts/hook_dispatcher.py",
    "core/background_jobs.py",
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
    for path in sorted(PACKAGE_ROOT.glob("*/*.py")):
        calls = [
            node
            for node in ast.walk(_tree(str(path.relative_to(PACKAGE_ROOT))))
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "run_frontier_review"
            )
        ]
        if calls:
            callers[str(path.relative_to(PACKAGE_ROOT))] = calls

    assert set(callers) == {"ops/self_heal.py"}
    for call in callers["ops/self_heal.py"]:
        assert "evidence" in {keyword.arg for keyword in call.keywords}


def test_routine_review_cannot_reach_frontier_execution() -> None:
    tree = _tree("decision/routine_review.py")
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    called = _called_names(tree)

    assert "subprocess" not in imported_modules
    assert "_run_codex" not in called
    assert "run_frontier_review" not in called


def test_codex_subprocess_is_reachable_only_inside_guarded_entrypoint() -> None:
    tree = _tree("decision/frontier_review.py")
    calling_functions: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        called = _called_names(node)
        if "_run_codex" in called:
            calling_functions.add(node.name)

    assert calling_functions == {"run_frontier_review"}
    assert all(
        not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        or node.name != "run_structured_review"
        for node in tree.body
    )
