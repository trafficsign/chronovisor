from __future__ import annotations

import ast
import tomllib
from pathlib import Path

from chronovisor.core.module_paths import LEGACY_MODULE_PATHS

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "chronovisor"
DOMAIN_NAMES = {
    "classification",
    "core",
    "decision",
    "hosts",
    "ingest",
    "knowledge_graph",
    "lab",
    "librarian",
    "ops",
    "raw",
    "recall",
    "research",
    "search",
}
TOP_LEVEL_IMPLEMENTATION_EXCEPTIONS = {
    # Installed outside the package archive so it remains available when the
    # main Chronovisor package cannot import.
    "deadman_observer.py",
}


def test_top_level_python_files_have_no_legacy_shims() -> None:
    top_level = {
        path.name
        for path in PACKAGE_ROOT.glob("*.py")
        if path.name != "__init__.py"
    }

    assert top_level == TOP_LEVEL_IMPLEMENTATION_EXCEPTIONS


def test_domain_inventory_covers_every_package() -> None:
    packages = {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }

    assert packages == DOMAIN_NAMES


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


def test_legacy_module_map_is_retired() -> None:
    assert LEGACY_MODULE_PATHS == {}


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
