#!/usr/bin/env python3
"""Generate and verify repository architecture baselines and exception ledgers."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import plistlib
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CAMPAIGN_STARTED_AT = "2026-08-06T12:44:00+09:00"
LIVE_ONLY_EXCLUSIONS = [
    {
        "evidence": "production_runtime_archives_and_running_processes",
        "reason": "GitHub-backed uvx archives, PIDs, and health require live hosts",
    },
    {
        "evidence": "production_authority_artifact_identity",
        "reason": "active authority files live below the production CHRONOVISOR_ROOT",
    },
    {
        "evidence": "recall_save_ingest_and_repair_live_behavior",
        "reason": "behavioral smoke checks require production hooks and services",
    },
    {
        "evidence": "dashboard_cortex_dom_latency_frame_and_memory",
        "reason": "viewport, event latency, frame pacing, and memory need a live browser",
    },
]
ARCHITECTURE_EXCEPTION_LEDGER = Path("docs/refactoring/architecture-exceptions.json")
EXCEPTION_METADATA_FIELDS = ("owner", "deadline", "removal_campaign", "rationale")
PRODUCTION_PACKAGES = frozenset(
    {
        "classification",
        "core",
        "decision",
        "hosts",
        "ingest",
        "knowledge_graph",
        "librarian",
        "ops",
        "raw",
        "recall",
        "research",
        "search",
    }
)
STATEMENT_IDENTITY_FIELDS = (
    "category",
    "source_package",
    "source_module",
    "scope_kind",
    "scope",
    "statement_kind",
    "target_package",
    "target_module",
    "symbols",
    "occurrence",
)


class _FunctionCounter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.count = 0
        self.scope: list[str] = []
        self.rows: list[dict[str, Any]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.count += 1
        end_line = node.end_lineno or node.lineno
        self.rows.append(
            {
                "qualname": ".".join([*self.scope, node.name]),
                "line": node.lineno,
                "lines": end_line - node.lineno + 1,
            }
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _run_git_bytes(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return _sha256_bytes(encoded)


def _package_names(package_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in package_root.iterdir()
        if path.is_dir() and any(path.rglob("*.py"))
    )


def _import_target(
    node: ast.ImportFrom,
    *,
    relative_path: Path,
) -> str | None:
    if not node.level:
        return node.module
    package_parts = list(relative_path.parts[:-1])
    retained = len(package_parts) - node.level + 1
    if retained < 0:
        return None
    target = ["chronovisor", *package_parts[:retained]]
    if node.module:
        target.extend(node.module.split("."))
    return ".".join(target)


def _import_targets(node: ast.ImportFrom, *, relative_path: Path) -> list[str]:
    target = _import_target(node, relative_path=relative_path)
    if target is None:
        return []
    if target == "chronovisor":
        return [f"{target}.{alias.name}" for alias in node.names]
    return [target]


def _module_name(relative_path: Path) -> str:
    parts = list(relative_path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(("chronovisor", *parts))


def _semantic_identity(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("category") == "cross_domain_edge":
        return {
            key: row[key]
            for key in (
                "category",
                "source_package",
                "target_package",
            )
        }
    return {key: row[key] for key in STATEMENT_IDENTITY_FIELDS}


def _semantic_id(row: dict[str, Any]) -> str:
    return f"arch:{_canonical_sha256(_semantic_identity(row))}"


class _ImportSiteCollector(ast.NodeVisitor):
    def __init__(
        self,
        *,
        relative_path: Path,
        source_package: str,
        package_set: set[str],
    ) -> None:
        self.relative_path = relative_path
        self.source_package = source_package
        self.source_module = _module_name(relative_path)
        self.package_set = package_set
        self.scope: list[tuple[str, str]] = []
        self.rows: list[dict[str, Any]] = []
        self.occurrences: dict[str, int] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(("class", node.name))
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(("function", node.name))
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scope.append(("function", node.name))
        self.generic_visit(node)
        self.scope.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record_static(
                node=node,
                statement_kind="import",
                target_module=alias.name,
                symbols=[],
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        target = _import_target(node, relative_path=self.relative_path)
        if target is None:
            return
        if target == "chronovisor":
            for alias in node.names:
                self._record_static(
                    node=node,
                    statement_kind="from",
                    target_module=f"chronovisor.{alias.name}",
                    symbols=[alias.name],
                )
            return
        self._record_static(
            node=node,
            statement_kind="from",
            target_module=target,
            symbols=[alias.name for alias in node.names],
        )

    def visit_Call(self, node: ast.Call) -> None:
        target = ""
        statement_kind = ""
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
            statement_kind = "__import__"
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
        ):
            statement_kind = "importlib.import_module"
        if statement_kind and node.args:
            literal = node.args[0]
            if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                target = literal.value
        if target.startswith("chronovisor."):
            self._add_row(
                node=node,
                category="dynamic_import",
                statement_kind=statement_kind,
                target_module=target,
                symbols=[],
            )
        self.generic_visit(node)

    def _record_static(
        self,
        *,
        node: ast.Import | ast.ImportFrom,
        statement_kind: str,
        target_module: str,
        symbols: list[str],
    ) -> None:
        parts = target_module.split(".")
        if len(parts) < 2 or parts[0] != "chronovisor":
            return
        target_package = parts[1]
        if target_package not in self.package_set:
            return
        if target_package != self.source_package:
            self._add_row(
                node=node,
                category="cross_domain_import",
                statement_kind=statement_kind,
                target_module=target_module,
                symbols=symbols,
            )
            if any(symbol.startswith("_") for symbol in symbols) or any(
                part.startswith("_") for part in parts[2:]
            ):
                self._add_row(
                    node=node,
                    category="private_symbol_import",
                    statement_kind=statement_kind,
                    target_module=target_module,
                    symbols=symbols,
                )
        if target_module == "chronovisor.decision.decision_schema_manifest" and any(
            symbol.startswith("_") or symbol.isupper() for symbol in symbols
        ):
            self._add_row(
                node=node,
                category="schema_manifest_implementation_import",
                statement_kind=statement_kind,
                target_module=target_module,
                symbols=symbols,
            )

    def _add_row(
        self,
        *,
        node: ast.AST,
        category: str,
        statement_kind: str,
        target_module: str,
        symbols: list[str],
    ) -> None:
        target_parts = target_module.split(".")
        scope_kind = self.scope[-1][0] if self.scope else "module"
        row: dict[str, Any] = {
            "category": category,
            "source_package": self.source_package,
            "source_module": self.source_module,
            "scope_kind": scope_kind,
            "scope": ".".join(name for _kind, name in self.scope) or "<module>",
            "statement_kind": statement_kind,
            "target_package": target_parts[1] if len(target_parts) > 1 else "",
            "target_module": target_module,
            "symbols": sorted(set(symbols)),
            "line": int(getattr(node, "lineno", 0)),
        }
        occurrence_key = _canonical_sha256(
            {key: row[key] for key in STATEMENT_IDENTITY_FIELDS if key != "occurrence"}
        )
        row["occurrence"] = self.occurrences.get(occurrence_key, 0) + 1
        self.occurrences[occurrence_key] = row["occurrence"]
        row["semantic_id"] = _semantic_id(row)
        self.rows.append(row)


def _source_inventory(package_root: Path) -> tuple[dict[str, Any], list[list[str]]]:
    packages = _package_names(package_root)
    package_set = set(packages)
    totals = {"modules": 0, "lines": 0, "functions": 0}
    per_package: dict[str, dict[str, int]] = {}
    edges: set[tuple[str, str]] = set()
    source_digests: list[tuple[str, str]] = []
    module_rows: list[dict[str, Any]] = []
    function_rows: list[dict[str, Any]] = []
    import_rows: list[dict[str, Any]] = []
    for package in packages:
        metrics = {"modules": 0, "lines": 0, "functions": 0}
        for path in sorted((package_root / package).rglob("*.py")):
            raw = path.read_bytes()
            source = raw.decode("utf-8")
            tree = ast.parse(source, filename=str(path))
            counter = _FunctionCounter()
            counter.visit(tree)
            module_path = path.relative_to(package_root.parent.parent).as_posix()
            line_count = len(source.splitlines())
            metrics["modules"] += 1
            metrics["lines"] += line_count
            metrics["functions"] += counter.count
            digest = _sha256_bytes(raw)
            source_digests.append((module_path, digest))
            module_rows.append(
                {
                    "path": module_path,
                    "package": package,
                    "lines": line_count,
                    "functions": counter.count,
                    "sha256": digest,
                }
            )
            function_rows.extend({"path": module_path, **row} for row in counter.rows)
            relative_path = path.relative_to(package_root)
            collector = _ImportSiteCollector(
                relative_path=relative_path,
                source_package=package,
                package_set=package_set,
            )
            collector.visit(tree)
            import_rows.extend(collector.rows)
            for node in ast.walk(tree):
                targets: list[str] = []
                if isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    targets = _import_targets(node, relative_path=relative_path)
                for target in targets:
                    parts = target.split(".")
                    if (
                        len(parts) >= 2
                        and parts[0] == "chronovisor"
                        and parts[1] in package_set
                        and parts[1] != package
                    ):
                        edges.add((package, parts[1]))
        per_package[package] = metrics
        for key, value in metrics.items():
            totals[key] += value
    top_level = {"modules": 0, "lines": 0, "functions": 0}
    for path in sorted(package_root.glob("*.py")):
        raw = path.read_bytes()
        source = raw.decode("utf-8")
        tree = ast.parse(source, filename=str(path))
        counter = _FunctionCounter()
        counter.visit(tree)
        module_path = path.relative_to(package_root.parent.parent).as_posix()
        line_count = len(source.splitlines())
        top_level["modules"] += 1
        top_level["lines"] += line_count
        top_level["functions"] += counter.count
        digest = _sha256_bytes(raw)
        source_digests.append((module_path, digest))
        module_rows.append(
            {
                "path": module_path,
                "package": "<top-level>",
                "lines": line_count,
                "functions": counter.count,
                "sha256": digest,
            }
        )
        function_rows.extend({"path": module_path, **row} for row in counter.rows)
        collector = _ImportSiteCollector(
            relative_path=path.relative_to(package_root),
            source_package="<top-level>",
            package_set=package_set,
        )
        collector.visit(tree)
        import_rows.extend(collector.rows)
    for key, value in top_level.items():
        totals[key] += value
    inventory = {
        "package_count": len(packages),
        "packages": packages,
        "totals": totals,
        "top_level": top_level,
        "per_package": per_package,
        "namespace_packages": [
            package
            for package in packages
            if not (package_root / package / "__init__.py").is_file()
        ],
        "modules": sorted(module_rows, key=lambda row: row["path"]),
        "module_hotspots": sorted(
            module_rows, key=lambda row: (-row["lines"], row["path"])
        )[:25],
        "function_hotspots": sorted(
            function_rows,
            key=lambda row: (-row["lines"], row["path"], row["qualname"]),
        )[:50],
        "import_sites": sorted(
            import_rows,
            key=lambda row: (
                row["semantic_id"],
                row["line"],
            ),
        ),
        "import_site_counts": dict(
            sorted(
                {
                    category: sum(row["category"] == category for row in import_rows)
                    for category in {row["category"] for row in import_rows}
                }.items()
            )
        ),
        "python_source_bytes_sha256": _canonical_sha256(source_digests),
    }
    return inventory, [list(edge) for edge in sorted(edges)]


def _architecture_exception_rows(
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    import_sites = [
        dict(row) for row in source.get("import_sites", []) if isinstance(row, dict)
    ]
    cross_domain_sites: dict[tuple[str, str], list[dict[str, Any]]] = {}
    exceptions: list[dict[str, Any]] = []
    statement_categories = {
        "dynamic_import",
        "private_symbol_import",
        "schema_manifest_implementation_import",
    }
    for row in import_sites:
        category = row.get("category")
        if category == "cross_domain_import":
            edge = (str(row["source_package"]), str(row["target_package"]))
            cross_domain_sites.setdefault(edge, []).append(
                {
                    key: row[key]
                    for key in (
                        "semantic_id",
                        "source_module",
                        "scope_kind",
                        "scope",
                        "statement_kind",
                        "target_module",
                        "symbols",
                        "occurrence",
                        "line",
                    )
                }
            )
        elif category in statement_categories:
            exceptions.append(row)

    for (source_package, target_package), sites in cross_domain_sites.items():
        edge_row: dict[str, Any] = {
            "category": "cross_domain_edge",
            "source_package": source_package,
            "target_package": target_package,
            "sites": sorted(
                sites,
                key=lambda row: (row["semantic_id"], row["line"]),
            ),
        }
        edge_row["semantic_id"] = _semantic_id(edge_row)
        exceptions.append(edge_row)
    return sorted(exceptions, key=lambda row: row["semantic_id"])


def _source_inventory_at_revision(
    root: Path, revision: str
) -> tuple[dict[str, Any], list[list[str]]]:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", revision, "src/chronovisor"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    with tempfile.TemporaryDirectory(prefix="chronovisor-architecture-") as directory:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(directory, filter="data")
        return _source_inventory(Path(directory) / "src" / "chronovisor")


def _tracked_non_python_assets(root: Path, *, revision: str | None) -> dict[str, Any]:
    if revision is None:
        paths = _run_git(root, "ls-files", "src/chronovisor").splitlines()
    else:
        paths = _run_git(
            root,
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "--",
            "src/chronovisor",
        ).splitlines()
    assets: list[dict[str, Any]] = []
    for path in sorted(path for path in paths if not path.endswith(".py")):
        raw = (
            (root / path).read_bytes()
            if revision is None
            else _run_git_bytes(root, "show", f"{revision}:{path}")
        )
        row = {
            "path": path,
            "bytes": len(raw),
            "sha256": _sha256_bytes(raw),
        }
        if Path(path).suffix in {".css", ".html", ".js"}:
            row["lines"] = len(raw.decode("utf-8").splitlines())
        assets.append(row)
    frontend = [row for row in assets if "lines" in row]
    return {
        "file_count": len(assets),
        "total_bytes": sum(row["bytes"] for row in assets),
        "manifest_sha256": _canonical_sha256(assets),
        "files": assets,
        "frontend_totals": {
            "file_count": len(frontend),
            "lines": sum(row["lines"] for row in frontend),
        },
        "asset_hotspots": sorted(
            frontend, key=lambda row: (-row["lines"], row["path"])
        ),
    }


def _strongly_connected_components(
    packages: list[str], edges: list[list[str]]
) -> list[list[str]]:
    adjacency: dict[str, list[str]] = {package: [] for package in packages}
    for source, target in edges:
        adjacency[source].append(target)
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indexes[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in sorted(adjacency[node]):
            if target not in indexes:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indexes[target])
        if lowlinks[node] != indexes[node]:
            return
        component: list[str] = []
        while stack:
            target = stack.pop()
            on_stack.remove(target)
            component.append(target)
            if target == node:
                break
        components.append(sorted(component))

    for package in packages:
        if package not in indexes:
            visit(package)
    return sorted(components, key=lambda component: (-len(component), component))


def _entrypoints(pyproject: dict[str, Any]) -> list[dict[str, str]]:
    scripts = pyproject.get("project", {}).get("scripts", {})
    return [
        {"name": str(name), "target": str(target)}
        for name, target in sorted(scripts.items())
    ]


def _compatibility_semantic_id(row: dict[str, Any]) -> str:
    identity = {key: row[key] for key in ("kind", "name", "target")}
    return f"compat:{_canonical_sha256(identity)}"


def _legacy_module_paths(package_root: Path) -> dict[str, str]:
    path = package_root / "core" / "module_paths.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        if node.target.id != "LEGACY_MODULE_PATHS" or node.value is None:
            continue
        payload = ast.literal_eval(node.value)
        if not isinstance(payload, dict):
            break
        return {str(key): str(value) for key, value in payload.items()}
    raise ValueError(f"LEGACY_MODULE_PATHS literal not found in {path}")


def _compatibility_contracts(
    pyproject: dict[str, Any], package_root: Path
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for legacy, target in sorted(_legacy_module_paths(package_root).items()):
        row = {"kind": "module_string", "name": legacy, "target": target}
        row["semantic_id"] = _compatibility_semantic_id(row)
        rows.append(row)
    for entrypoint in _entrypoints(pyproject):
        row = {
            "kind": "console_entrypoint",
            "name": entrypoint["name"],
            "target": entrypoint["target"],
        }
        row["semantic_id"] = _compatibility_semantic_id(row)
        rows.append(row)
    return sorted(rows, key=lambda row: row["semantic_id"])


def _load_architecture_exception_ledger(root: Path) -> dict[str, Any]:
    path = root / ARCHITECTURE_EXCEPTION_LEDGER
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": 0,
            "baseline_semantic_ids": [],
            "exceptions": [],
            "compatibility_contracts": [],
            "load_error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "schema_version": 0,
            "baseline_semantic_ids": [],
            "exceptions": [],
            "compatibility_contracts": [],
            "load_error": "ledger root must be an object",
        }
    return payload


def build_architecture_exception_ledger(root: Path) -> dict[str, Any]:
    root = root.resolve()
    package_root = root / "src" / "chronovisor"
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    source, _edges = _source_inventory(package_root)
    detected_rows = _architecture_exception_rows(source)
    detected_by_id = {str(row["semantic_id"]): dict(row) for row in detected_rows}
    rationale = {
        "cross_domain_edge": (
            "Existing package dependency pending a published contract or use-case port."
        ),
        "private_symbol_import": (
            "Existing cross-domain private API dependency pending public API extraction."
        ),
        "dynamic_import": (
            "Existing literal dynamic import retained until composition ownership moves."
        ),
        "schema_manifest_implementation_import": (
            "Existing schema implementation constant dependency pending contract publication."
        ),
    }
    exceptions: list[dict[str, Any]] = []
    for semantic_id, detected in sorted(detected_by_id.items()):
        row = dict(detected)
        row.update(
            {
                "semantic_id": semantic_id,
                "owner": "chronovisor-architecture",
                "deadline": "2026-09-30",
                "removal_campaign": "P9",
                "rationale": rationale[str(row["category"])],
            }
        )
        exceptions.append(row)

    compatibility_contracts: list[dict[str, Any]] = []
    for detected in _compatibility_contracts(pyproject, package_root):
        compatibility_contracts.append(
            {
                **detected,
                "owner": "chronovisor-compatibility",
                "deadline": "2026-12-31",
                "removal_campaign": "V",
                "rationale": (
                    "Protected compatibility surface; retire only with an explicit migration."
                ),
            }
        )
    semantic_ids = sorted(detected_by_id)
    return {
        "schema_version": 1,
        "captured_from_head": _run_git(root, "rev-parse", "HEAD").strip(),
        "semantic_identity": (
            "cross-domain edges use category+source_package+target_package; "
            "statement exceptions use category+source_package+source_module+"
            "scope_kind+scope+statement_kind+target_package+target_module+symbols+"
            "occurrence; source line and edge sites are diagnostic only"
        ),
        "baseline_semantic_ids": semantic_ids,
        "production_to_lab_baseline_semantic_ids": sorted(
            semantic_id
            for semantic_id, row in detected_by_id.items()
            if row.get("category") == "cross_domain_edge"
            and row.get("source_package") in PRODUCTION_PACKAGES
            and row.get("target_package") == "lab"
        ),
        "counts": {
            "exceptions": len(exceptions),
            "by_category": dict(
                sorted(
                    {
                        category: sum(row["category"] == category for row in exceptions)
                        for category in {row["category"] for row in exceptions}
                    }.items()
                )
            ),
            "cross_domain_sites": source["import_site_counts"].get(
                "cross_domain_import", 0
            ),
            "compatibility_contracts": len(compatibility_contracts),
        },
        "exceptions": exceptions,
        "compatibility_contracts": compatibility_contracts,
    }


def _launchd_inventory(root: Path) -> list[dict[str, Any]]:
    paths = [
        line
        for line in _run_git(root, "ls-files", "launchd/*.plist").splitlines()
        if line
    ]
    rows: list[dict[str, Any]] = []
    for relative in paths:
        path = root / relative
        raw = path.read_bytes()
        payload = plistlib.loads(raw)
        rows.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(raw),
                "label": str(payload.get("Label") or ""),
                "program_arguments": [
                    str(value) for value in payload.get("ProgramArguments", [])
                ],
            }
        )
    return rows


def _architecture_contracts(pyproject: dict[str, Any]) -> list[dict[str, Any]]:
    contracts = pyproject.get("tool", {}).get("importlinter", {}).get("contracts", [])
    return [dict(contract) for contract in contracts]


def _contract_hashes(root: Path) -> dict[str, Any]:
    from chronovisor.decision.decision_lane_contract_cases import (
        decision_lane_contract_case_manifest_sha256,
    )
    from chronovisor.decision.decision_lane_contracts import (
        lane_contract_manifest_sha256,
    )
    from chronovisor.decision.decision_schema_manifest import (
        production_schema_manifest,
        production_signature_manifest,
    )
    from chronovisor.decision.local_structured import (
        structured_generation_policy_sha256,
    )

    fixture_paths = [
        "tests/fixtures/recall_processor_cases.json",
        "tests/fixtures/research_holdout.jsonl",
    ]
    schema_manifest = production_schema_manifest()
    signature_manifest = production_signature_manifest()
    validator_rows = [
        {"name": name, "sha256": digest}
        for name, digest in sorted(schema_manifest.items())
    ]
    return {
        "decision_authority": {
            "lane_contract_manifest_sha256": lane_contract_manifest_sha256(),
            "lane_contract_case_manifest_sha256": (
                decision_lane_contract_case_manifest_sha256()
            ),
            "structured_generation_policy_sha256": (
                structured_generation_policy_sha256()
            ),
        },
        "production_schema_manifest": {
            "entries": schema_manifest,
            "canonical_mapping_sha256": {
                "sha256": _canonical_sha256(schema_manifest),
                "semantics": "canonical name-to-schema-digest mapping",
            },
            "artifact_validator_sorted_rows_sha256": {
                "sha256": _canonical_sha256(validator_rows),
                "semantics": (
                    "DecisionRouter adoption validator sorted name/sha256 rows"
                ),
            },
        },
        "production_signature_manifest": {
            "sha256": _canonical_sha256(signature_manifest),
            "semantics": "canonical action-signature policy mapping",
        },
        "repository_fixtures": {
            path: _sha256_bytes((root / path).read_bytes()) for path in fixture_paths
        },
    }


def scan_repository(root: Path, *, captured_at: str) -> dict[str, Any]:
    root = root.resolve()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    worktree_source, worktree_edges = _source_inventory(root / "src" / "chronovisor")
    compatibility_contracts = _compatibility_contracts(
        pyproject,
        root / "src" / "chronovisor",
    )
    architecture_exception_ledger = _load_architecture_exception_ledger(root)
    worktree_status = _run_git(root, "status", "--short").splitlines()
    protected_paths = {
        "_handoff/2026-06-11_0042_recall-redesign.md",
        "logs/",
    }
    protected_status = [row for row in worktree_status if row[3:] in protected_paths]
    campaign_status = [row for row in worktree_status if row not in protected_status]
    source_base = _run_git(
        root,
        "log",
        "-1",
        "--format=%H",
        "--",
        "src",
        "scripts",
        "tests",
        "pyproject.toml",
        "docs/refactoring",
        ".github",
        "launchd",
    ).strip()
    source, edges = _source_inventory_at_revision(root, source_base)
    source["tracked_non_python_assets"] = _tracked_non_python_assets(
        root, revision=source_base
    )
    worktree_source["tracked_non_python_assets"] = _tracked_non_python_assets(
        root, revision=None
    )
    packages = list(source["packages"])
    worktree_packages = list(worktree_source["packages"])
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": "O",
        "campaign_started_at": CAMPAIGN_STARTED_AT,
        "captured_at": captured_at,
        "captured_at_semantics": (
            "Final frozen Campaign O implementation-tree capture before the "
            "authoritative isolated full suite; distinct from campaign_started_at"
        ),
        "repository": {
            "head_at_capture": _run_git(root, "rev-parse", "HEAD").strip(),
            "pre_campaign_source_head": source_base,
            "source_head_semantics": (
                "latest pre-Campaign O commit touching repository source or gates; "
                "HEAD_at_capture may additionally contain the approved plan"
            ),
            "branch": _run_git(root, "branch", "--show-current").strip(),
            "worktree": {
                "capture_phase": (
                    "Campaign O frozen pre-full, pre-commit implementation worktree"
                ),
                "campaign_o_changes": campaign_status,
                "protected_user_owned_untracked": protected_status,
            },
        },
        "source": source,
        "worktree_source": worktree_source,
        "architecture": {
            "edges": edges,
            "edge_count": len(edges),
            "strongly_connected_components": _strongly_connected_components(
                packages, edges
            ),
            "contracts": _architecture_contracts(pyproject),
        },
        "worktree_architecture": {
            "edges": worktree_edges,
            "edge_count": len(worktree_edges),
            "strongly_connected_components": _strongly_connected_components(
                worktree_packages, worktree_edges
            ),
        },
        "console_entrypoints": _entrypoints(pyproject),
        "compatibility_contracts": compatibility_contracts,
        "architecture_exception_ledger": architecture_exception_ledger,
        "tracked_launchd_plists": _launchd_inventory(root),
        "contract_hashes": _contract_hashes(root),
        "live_only_exclusions": LIVE_ONLY_EXCLUSIONS,
    }


def _keyed_rows_drift(
    baseline: list[dict[str, Any]],
    current: list[dict[str, Any]],
    *,
    key: str,
) -> dict[str, Any]:
    baseline_by_key = {str(row[key]): row for row in baseline}
    current_by_key = {str(row[key]): row for row in current}
    drift = {
        "added": sorted(set(current_by_key) - set(baseline_by_key)),
        "removed": sorted(set(baseline_by_key) - set(current_by_key)),
        "changed": [
            {
                "key": value,
                "baseline": baseline_by_key[value],
                "current": current_by_key[value],
            }
            for value in sorted(set(baseline_by_key) & set(current_by_key))
            if baseline_by_key[value] != current_by_key[value]
        ],
    }
    return {name: values for name, values in drift.items() if values}


def _missing_metadata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for row in rows:
        absent = [
            field
            for field in EXCEPTION_METADATA_FIELDS
            if not isinstance(row.get(field), str) or not str(row[field]).strip()
        ]
        if absent:
            missing.append(
                {
                    "semantic_id": str(row.get("semantic_id") or ""),
                    "missing": absent,
                }
            )
    return missing


def _architecture_exception_violations(
    current_source: dict[str, Any],
    ledger: dict[str, Any],
    current_compatibility: list[dict[str, Any]],
) -> dict[str, Any]:
    detected_rows = _architecture_exception_rows(current_source)
    detected = {str(row["semantic_id"]): row for row in detected_rows}
    exception_rows = [
        dict(row) for row in ledger.get("exceptions", []) if isinstance(row, dict)
    ]
    recorded = {str(row.get("semantic_id") or ""): row for row in exception_rows}
    baseline_ids = {str(value) for value in ledger.get("baseline_semantic_ids", [])}
    detected_ids = set(detected)
    recorded_ids = {value for value in recorded if value}

    identity_mismatches: list[str] = []
    for semantic_id, row in recorded.items():
        try:
            expected = _semantic_id(row)
        except KeyError:
            expected = ""
        if not semantic_id or semantic_id != expected:
            identity_mismatches.append(semantic_id)

    compatibility_rows = [
        dict(row)
        for row in ledger.get("compatibility_contracts", [])
        if isinstance(row, dict)
    ]
    recorded_compatibility = {
        str(row.get("semantic_id") or ""): row for row in compatibility_rows
    }
    current_compatibility_ids = {
        str(row["semantic_id"]) for row in current_compatibility
    }
    recorded_compatibility_ids = {value for value in recorded_compatibility if value}
    compatibility_identity_mismatches: list[str] = []
    for semantic_id, row in recorded_compatibility.items():
        try:
            expected = _compatibility_semantic_id(row)
        except KeyError:
            expected = ""
        if not semantic_id or semantic_id != expected:
            compatibility_identity_mismatches.append(semantic_id)

    production_lab_baseline = {
        str(value)
        for value in ledger.get("production_to_lab_baseline_semantic_ids", [])
    }
    production_lab_current = {
        semantic_id
        for semantic_id, row in detected.items()
        if row.get("category") == "cross_domain_edge"
        and row.get("source_package") in PRODUCTION_PACKAGES
        and row.get("target_package") == "lab"
    }
    return {
        "ledger_load_error": str(ledger.get("load_error") or ""),
        "ledger_schema_version": (
            [] if ledger.get("schema_version") == 1 else [ledger.get("schema_version")]
        ),
        "new_exception_ids": sorted(detected_ids - baseline_ids),
        "unrecorded_exception_ids": sorted(detected_ids - recorded_ids),
        "stale_exception_ids": sorted(recorded_ids - detected_ids),
        "baseline_semantic_id_non_subset": sorted(baseline_ids - recorded_ids),
        "exception_identity_mismatches": sorted(identity_mismatches),
        "exception_metadata_missing": _missing_metadata(exception_rows),
        "duplicate_exception_ids": sorted(
            semantic_id
            for semantic_id in recorded_ids
            if sum(
                str(row.get("semantic_id") or "") == semantic_id
                for row in exception_rows
            )
            > 1
        ),
        "production_to_lab_edge_growth": sorted(
            production_lab_current - production_lab_baseline
        ),
        "compatibility_contract_drift": {
            "unrecorded": sorted(
                current_compatibility_ids - recorded_compatibility_ids
            ),
            "stale": sorted(recorded_compatibility_ids - current_compatibility_ids),
            "identity_mismatches": sorted(compatibility_identity_mismatches),
        }
        if (
            current_compatibility_ids != recorded_compatibility_ids
            or compatibility_identity_mismatches
        )
        else {},
        "compatibility_metadata_missing": _missing_metadata(compatibility_rows),
    }


def architecture_fitness(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    baseline_architecture = baseline["architecture"]
    current_architecture = current.get("worktree_architecture", current["architecture"])
    baseline_packages = set(baseline["source"]["packages"])
    current_source = current.get("worktree_source", current["source"])
    current_packages = set(current_source["packages"])
    baseline_edges = {tuple(edge) for edge in baseline_architecture["edges"]}
    current_edges = {tuple(edge) for edge in current_architecture["edges"]}
    new_packages = sorted(current_packages - baseline_packages)
    missing_packages = sorted(baseline_packages - current_packages)
    new_edges = [list(edge) for edge in sorted(current_edges - baseline_edges)]
    baseline_components = [
        set(component)
        for component in baseline_architecture["strongly_connected_components"]
    ]
    scc_regressions = [
        component
        for component in current_architecture["strongly_connected_components"]
        if len(component) > 1
        and not any(
            set(component).issubset(baseline) for baseline in baseline_components
        )
    ]
    observations = {
        "new_packages": new_packages,
        "missing_packages": missing_packages,
        "tracked_non_python_assets": {
            "changed": baseline["source"].get("tracked_non_python_assets")
            != current_source.get("tracked_non_python_assets"),
            "baseline_manifest_sha256": baseline["source"]
            .get("tracked_non_python_assets", {})
            .get("manifest_sha256", ""),
            "current_manifest_sha256": current_source.get(
                "tracked_non_python_assets", {}
            ).get("manifest_sha256", ""),
            "baseline_file_count": baseline["source"]
            .get("tracked_non_python_assets", {})
            .get("file_count", 0),
            "current_file_count": current_source.get(
                "tracked_non_python_assets", {}
            ).get("file_count", 0),
        },
    }
    entrypoint_drift = _keyed_rows_drift(
        baseline["console_entrypoints"], current["console_entrypoints"], key="name"
    )
    launchd_drift = _keyed_rows_drift(
        baseline["tracked_launchd_plists"],
        current["tracked_launchd_plists"],
        key="path",
    )
    contract_hash_drift = (
        {
            "baseline": baseline["contract_hashes"],
            "current": current["contract_hashes"],
        }
        if baseline["contract_hashes"] != current["contract_hashes"]
        else {}
    )
    architecture_contract_drift = (
        {
            "baseline": baseline_architecture["contracts"],
            "current": current_architecture.get(
                "contracts", current["architecture"]["contracts"]
            ),
        }
        if baseline_architecture["contracts"] != current["architecture"]["contracts"]
        else {}
    )
    exception_violations = _architecture_exception_violations(
        current_source,
        current.get("architecture_exception_ledger", {}),
        [
            dict(row)
            for row in current.get("compatibility_contracts", [])
            if isinstance(row, dict)
        ],
    )
    violations = {
        "new_edges": new_edges,
        "scc_regressions": scc_regressions,
        "namespace_packages": current_source.get("namespace_packages", []),
        "entrypoint_drift": entrypoint_drift,
        "launchd_drift": launchd_drift,
        "contract_hash_drift": contract_hash_drift,
        "architecture_contract_drift": architecture_contract_drift,
        **exception_violations,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": not any(violations.values()),
        "baseline_edge_count": len(baseline_edges),
        "current_edge_count": len(current_edges),
        "baseline_scc": baseline_architecture["strongly_connected_components"],
        "current_scc": current_architecture["strongly_connected_components"],
        "observations": observations,
        "violations": violations,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--captured-at", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--generate-exceptions", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.generate_exceptions:
        payload = build_architecture_exception_ledger(args.root)
    else:
        captured_at = args.captured_at or "verification"
        payload = scan_repository(args.root, captured_at=captured_at)
    exit_code = 0
    if args.check is not None and not args.generate_exceptions:
        baseline = json.loads(args.check.read_text(encoding="utf-8"))
        payload = architecture_fitness(baseline, payload)
        exit_code = 0 if payload["passed"] else 1
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
