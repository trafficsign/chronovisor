from __future__ import annotations

import ast
import importlib
from pathlib import Path

import tomllib

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "chronovisor"
DOMAIN_NAMES = {
    "classification",
    "core",
    "decision",
    "hosts",
    "ingest",
    "lab",
    "librarian",
    "ops",
    "raw",
    "recall",
    "research",
    "search",
}
LEGACY_PACKAGE_MODULES = {
    "classification": "ClassificationError",
    "ingest": "run_ingest",
    "librarian": "run_shadow",
    "search": "ScoredPage",
}
TOP_LEVEL_IMPLEMENTATION_EXCEPTIONS = {
    # Installed outside the package archive so it remains available when the
    # main Chronovisor package cannot import.
    "deadman_observer.py",
}


def test_top_level_python_files_are_compatibility_shims() -> None:
    unexpected: list[str] = []
    for path in PACKAGE_ROOT.glob("*.py"):
        if path.name == "__init__.py" or path.name in TOP_LEVEL_IMPLEMENTATION_EXCEPTIONS:
            continue
        text = path.read_text(encoding="utf-8")
        if "alias_legacy_module(__name__, _implementation)" not in text:
            unexpected.append(path.name)

    assert unexpected == []


def test_domain_implementations_do_not_import_legacy_module_paths() -> None:
    implementation_names = {
        path.stem
        for domain in DOMAIN_NAMES - {"lab"}
        for path in (PACKAGE_ROOT / domain).glob("*.py")
        if path.name not in {"__init__.py", "compat.py"}
    }
    violations: list[str] = []
    for domain in DOMAIN_NAMES:
        for path in (PACKAGE_ROOT / domain).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    parts = name.split(".")
                    if (
                        len(parts) == 2
                        and parts[0] == "chronovisor"
                        and parts[1] in implementation_names
                        and parts[1] not in DOMAIN_NAMES
                    ):
                        violations.append(f"{path}:{node.lineno}:{name}")

    assert violations == []


def test_legacy_package_modules_forward_reads_and_writes() -> None:
    for package_name, marker in LEGACY_PACKAGE_MODULES.items():
        package = importlib.import_module(f"chronovisor.{package_name}")
        implementation = importlib.import_module(
            f"chronovisor.{package_name}.{package_name}"
        )
        assert getattr(package, marker) is getattr(implementation, marker)

        sentinel = object()
        setattr(package, "_module_layout_test_sentinel", sentinel)
        assert getattr(package, "_module_layout_test_sentinel") is sentinel
        delattr(package, "_module_layout_test_sentinel")


def test_console_scripts_target_domain_modules() -> None:
    project = tomllib.loads(
        (PACKAGE_ROOT.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    legacy_targets = []
    for name, target in project["project"]["scripts"].items():
        module = target.split(":", 1)[0]
        if module == "chronovisor.lab.cli":
            continue
        if len(module.split(".")) < 3:
            legacy_targets.append((name, target))

    assert legacy_targets == []
