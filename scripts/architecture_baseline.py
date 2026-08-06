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
ARCHITECTURE_EXCEPTION_BASELINE = Path(
    "docs/refactoring/architecture-exception-baseline.json"
)
FROZEN_EXCEPTION_SOURCE_HEAD = "d404a6b20d00e3bcd1d4cdb89edfa5a718c51833"
EXCEPTION_LEDGER_SCHEMA_VERSION = 2
EXCEPTION_BASELINE_SCHEMA_VERSION = 1
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
SCHEMA_MANIFEST_MODULE = "chronovisor.decision.decision_schema_manifest"
SCHEMA_REGISTRY_SCOPES = frozenset(
    {"production_decision_schemas", "background_decision_schemas"}
)
EXCEPTION_BASELINE_ID_FIELDS = (
    "exception_semantic_ids",
    "cross_domain_site_semantic_ids",
    "production_to_lab_edge_semantic_ids",
    "production_to_lab_static_site_semantic_ids",
    "production_to_lab_dynamic_site_semantic_ids",
    "compatibility_semantic_ids",
)
CAMPAIGN_DEADLINES = {
    "P2": "2026-08-31",
    "P3": "2026-08-31",
    "P4": "2026-09-07",
    "P5": "2026-09-14",
    "P6": "2026-09-21",
    "P8": "2026-10-05",
    "Q": "2026-10-31",
    "R": "2026-11-30",
    "S": "2026-12-31",
    "V": "2027-01-31",
}


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
        schema_symbols = [
            symbol
            for symbol in symbols
            if symbol.isupper() and symbol.endswith("_SCHEMA")
        ]
        if (
            self.source_module == SCHEMA_MANIFEST_MODULE
            and self.scope
            and self.scope[-1][1] in SCHEMA_REGISTRY_SCOPES
            and schema_symbols
        ):
            self._add_row(
                node=node,
                category="schema_manifest_implementation_import",
                statement_kind=statement_kind,
                target_module=target_module,
                symbols=schema_symbols,
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
            "content_sha256": _sha256_bytes(
                ast.dump(node, include_attributes=False).encode("utf-8")
            ),
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
                        "content_sha256",
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


def _lab_dispatch_paths(package_root: Path) -> dict[str, str]:
    path = package_root / "lab" / "cli.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name) or node.target.id != "COMMANDS":
            continue
        if node.value is None:
            break
        payload = ast.literal_eval(node.value)
        if not isinstance(payload, dict):
            break
        dispatch: dict[str, str] = {}
        for command, target in payload.items():
            if (
                not isinstance(command, str)
                or not isinstance(target, tuple)
                or len(target) != 2
                or not isinstance(target[0], str)
                or not isinstance(target[1], bool)
            ):
                raise ValueError(f"invalid COMMANDS entry in {path}: {command!r}")
            dispatch[command] = target[0]
        return dispatch
    raise ValueError(f"COMMANDS literal not found in {path}")


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
    for command, target in sorted(_lab_dispatch_paths(package_root).items()):
        row = {"kind": "lab_dispatch", "name": command, "target": target}
        row["semantic_id"] = _compatibility_semantic_id(row)
        rows.append(row)
    return sorted(rows, key=lambda row: row["semantic_id"])


def _load_json_object(root: Path, relative_path: Path) -> dict[str, Any]:
    path = root / relative_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": 0,
            "load_error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "schema_version": 0,
            "load_error": f"{relative_path} root must be an object",
        }
    return payload


def _load_architecture_exception_ledger(root: Path) -> dict[str, Any]:
    payload = _load_json_object(root, ARCHITECTURE_EXCEPTION_LEDGER)
    payload.setdefault("exceptions", [])
    payload.setdefault("compatibility_contracts", [])
    return payload


def _load_architecture_exception_baseline(root: Path) -> dict[str, Any]:
    return _load_json_object(root, ARCHITECTURE_EXCEPTION_BASELINE)


def _load_previous_architecture_exception_baseline(root: Path) -> dict[str, Any]:
    relative = ARCHITECTURE_EXCEPTION_BASELINE.as_posix()
    status = _run_git(root, "status", "--porcelain", "--", relative).strip()
    revision = "HEAD" if status else "HEAD^"
    try:
        raw = _run_git_bytes(root, "show", f"{revision}:{relative}")
    except subprocess.CalledProcessError:
        return {"schema_version": 0, "absent": True}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "schema_version": 0,
            "load_error": f"JSONDecodeError: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "schema_version": 0,
            "load_error": "previous architecture exception baseline is not an object",
        }
    return payload


def _inventory_at_revision(
    root: Path, revision: str
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    archive = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            revision,
            "pyproject.toml",
            "src/chronovisor",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    with tempfile.TemporaryDirectory(
        prefix="chronovisor-exception-baseline-"
    ) as root_dir:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(root_dir, filter="data")
        extracted = Path(root_dir)
        package_root = extracted / "src" / "chronovisor"
        source, _edges = _source_inventory(package_root)
        pyproject = tomllib.loads(
            (extracted / "pyproject.toml").read_text(encoding="utf-8")
        )
        return source, _compatibility_contracts(pyproject, package_root)


def _exception_id_sets(
    source: dict[str, Any],
    compatibility: list[dict[str, Any]],
) -> dict[str, set[str]]:
    exception_rows = _architecture_exception_rows(source)
    cross_domain_sites = [
        dict(row)
        for row in source.get("import_sites", [])
        if isinstance(row, dict) and row.get("category") == "cross_domain_import"
    ]
    dynamic_sites = [
        dict(row)
        for row in source.get("import_sites", [])
        if isinstance(row, dict) and row.get("category") == "dynamic_import"
    ]
    production_lab_edges = {
        str(row["semantic_id"])
        for row in exception_rows
        if row.get("category") == "cross_domain_edge"
        and row.get("source_package") in PRODUCTION_PACKAGES
        and row.get("target_package") == "lab"
    }
    return {
        "exception_semantic_ids": {str(row["semantic_id"]) for row in exception_rows},
        "cross_domain_site_semantic_ids": {
            str(row["semantic_id"]) for row in cross_domain_sites
        },
        "production_to_lab_edge_semantic_ids": production_lab_edges,
        "production_to_lab_static_site_semantic_ids": {
            str(row["semantic_id"])
            for row in cross_domain_sites
            if row.get("source_package") in PRODUCTION_PACKAGES
            and row.get("target_package") == "lab"
        },
        "production_to_lab_dynamic_site_semantic_ids": {
            str(row["semantic_id"])
            for row in dynamic_sites
            if row.get("source_package") in PRODUCTION_PACKAGES
            and row.get("target_package") == "lab"
        },
        "compatibility_semantic_ids": {
            str(row["semantic_id"]) for row in compatibility
        },
    }


def _seed_ids(seed: dict[str, Any], field: str, state: str) -> set[str]:
    bucket = seed.get(field)
    if not isinstance(bucket, dict):
        return set()
    values = bucket.get(state)
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values}


def _exception_counts(
    source: dict[str, Any],
    compatibility: list[dict[str, Any]],
    retired: dict[str, set[str]],
) -> dict[str, Any]:
    exceptions = _architecture_exception_rows(source)
    cross_domain_sites = [
        row
        for row in source.get("import_sites", [])
        if isinstance(row, dict) and row.get("category") == "cross_domain_import"
    ]
    compatibility_kinds = {
        kind: sum(row.get("kind") == kind for row in compatibility)
        for kind in sorted({str(row.get("kind")) for row in compatibility})
    }
    schema_rows = [
        row
        for row in exceptions
        if row.get("category") == "schema_manifest_implementation_import"
    ]
    schema_counts = {
        scope: {
            "statements": sum(row.get("scope") == scope for row in schema_rows),
            "symbols": sum(
                len(row.get("symbols", []))
                for row in schema_rows
                if row.get("scope") == scope
            ),
        }
        for scope in sorted(SCHEMA_REGISTRY_SCOPES)
    }
    active_ids = _exception_id_sets(source, compatibility)
    return {
        "active": {
            "exceptions": len(exceptions),
            "by_category": dict(
                sorted(
                    {
                        category: sum(
                            row.get("category") == category for row in exceptions
                        )
                        for category in {str(row.get("category")) for row in exceptions}
                    }.items()
                )
            ),
            "cross_domain_sites": len(cross_domain_sites),
            "production_to_lab_edges": len(
                active_ids["production_to_lab_edge_semantic_ids"]
            ),
            "production_to_lab_static_sites": len(
                active_ids["production_to_lab_static_site_semantic_ids"]
            ),
            "production_to_lab_dynamic_sites": len(
                active_ids["production_to_lab_dynamic_site_semantic_ids"]
            ),
            "compatibility_contracts": len(compatibility),
            "compatibility_by_kind": compatibility_kinds,
            "schema_manifest_implementation": schema_counts,
        },
        "retired": {
            field: len(retired.get(field, set()))
            for field in EXCEPTION_BASELINE_ID_FIELDS
        },
    }


def _exception_baseline_payload(
    source: dict[str, Any],
    compatibility: list[dict[str, Any]],
    *,
    retired: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    retired_ids = retired or {field: set() for field in EXCEPTION_BASELINE_ID_FIELDS}
    active_ids = _exception_id_sets(source, compatibility)
    return {
        "schema_version": EXCEPTION_BASELINE_SCHEMA_VERSION,
        "source_baseline_head": FROZEN_EXCEPTION_SOURCE_HEAD,
        "semantic_identity": (
            "line-independent semantic IDs; every field has monotonic active and "
            "retired sets whose union equals the frozen source reference"
        ),
        **{
            field: {
                "active": sorted(active_ids[field]),
                "retired": sorted(retired_ids.get(field, set())),
            }
            for field in EXCEPTION_BASELINE_ID_FIELDS
        },
        "counts": _exception_counts(source, compatibility, retired_ids),
    }


def build_architecture_exception_baseline(root: Path) -> dict[str, Any]:
    source, compatibility = _inventory_at_revision(
        root.resolve(), FROZEN_EXCEPTION_SOURCE_HEAD
    )
    return _exception_baseline_payload(source, compatibility)


def _seed_structure_errors(seed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in EXCEPTION_BASELINE_ID_FIELDS:
        bucket = seed.get(field)
        if not isinstance(bucket, dict):
            errors.append(field)
            continue
        for state in ("active", "retired"):
            values = bucket.get(state)
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value for value in values
            ):
                errors.append(f"{field}.{state}")
    return sorted(errors)


def _seed_state_violations(
    current_source: dict[str, Any],
    current_compatibility: list[dict[str, Any]],
    seed: dict[str, Any],
    frozen_reference: dict[str, Any],
    previous_seed: dict[str, Any],
) -> dict[str, Any]:
    current_ids = _exception_id_sets(current_source, current_compatibility)
    universe_drift: dict[str, Any] = {}
    overlap: dict[str, list[str]] = {}
    current_drift: dict[str, Any] = {}
    retired_reintroductions: dict[str, list[str]] = {}
    retired_regressions: dict[str, list[str]] = {}
    active_growth: dict[str, list[str]] = {}
    duplicate_seed_ids: dict[str, list[str]] = {}
    previous_available = (
        previous_seed.get("schema_version") == EXCEPTION_BASELINE_SCHEMA_VERSION
    )
    for field in EXCEPTION_BASELINE_ID_FIELDS:
        active = _seed_ids(seed, field, "active")
        retired = _seed_ids(seed, field, "retired")
        frozen = _seed_ids(frozen_reference, field, "active") | _seed_ids(
            frozen_reference, field, "retired"
        )
        universe = active | retired
        added = sorted(universe - frozen)
        missing = sorted(frozen - universe)
        if added or missing:
            universe_drift[field] = {"added": added, "missing": missing}
        shared = sorted(active & retired)
        if shared:
            overlap[field] = shared
        current_added = sorted(current_ids[field] - active)
        current_missing = sorted(active - current_ids[field])
        if current_added or current_missing:
            current_drift[field] = {
                "unseeded_current": current_added,
                "seeded_but_absent": current_missing,
            }
        reintroduced = sorted(current_ids[field] & retired)
        if reintroduced:
            retired_reintroductions[field] = reintroduced
        bucket = seed.get(field)
        if isinstance(bucket, dict):
            duplicate_values: set[str] = set()
            for state in ("active", "retired"):
                values = bucket.get(state)
                if not isinstance(values, list):
                    continue
                duplicate_values.update(
                    str(value) for value in values if values.count(value) > 1
                )
            duplicates = sorted(duplicate_values)
            if duplicates:
                duplicate_seed_ids[field] = duplicates
        if previous_available:
            previous_active = _seed_ids(previous_seed, field, "active")
            previous_retired = _seed_ids(previous_seed, field, "retired")
            regressed = sorted(previous_retired - retired)
            grown = sorted(active - previous_active)
            if regressed:
                retired_regressions[field] = regressed
            if grown:
                active_growth[field] = grown
    retired_ids_by_field = {
        field: _seed_ids(seed, field, "retired")
        for field in EXCEPTION_BASELINE_ID_FIELDS
    }
    expected_counts = _exception_counts(
        current_source,
        current_compatibility,
        retired_ids_by_field,
    )
    seed_count_drift = (
        {"recorded": seed.get("counts"), "expected": expected_counts}
        if seed.get("counts") != expected_counts
        else {}
    )
    return {
        "seed_load_error": str(seed.get("load_error") or ""),
        "seed_schema_version": (
            []
            if seed.get("schema_version") == EXCEPTION_BASELINE_SCHEMA_VERSION
            else [seed.get("schema_version")]
        ),
        "seed_source_baseline_head_drift": (
            []
            if seed.get("source_baseline_head") == FROZEN_EXCEPTION_SOURCE_HEAD
            else [seed.get("source_baseline_head")]
        ),
        "seed_structure_errors": _seed_structure_errors(seed),
        "seed_universe_drift": universe_drift,
        "seed_active_retired_overlap": overlap,
        "seed_current_drift": current_drift,
        "retired_id_reintroductions": retired_reintroductions,
        "seed_retired_regressions": retired_regressions,
        "seed_active_growth": active_growth,
        "duplicate_seed_ids": duplicate_seed_ids,
        "seed_count_drift": seed_count_drift,
        "previous_seed_load_error": str(previous_seed.get("load_error") or ""),
        "previous_seed_schema_version": (
            []
            if previous_seed.get("absent") is True
            or previous_seed.get("schema_version") == EXCEPTION_BASELINE_SCHEMA_VERSION
            else [previous_seed.get("schema_version")]
        ),
    }


def retire_missing_architecture_exceptions(root: Path) -> dict[str, Any]:
    root = root.resolve()
    package_root = root / "src" / "chronovisor"
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    source, _edges = _source_inventory(package_root)
    compatibility = _compatibility_contracts(pyproject, package_root)
    seed = _load_architecture_exception_baseline(root)
    frozen = build_architecture_exception_baseline(root)
    current_ids = _exception_id_sets(source, compatibility)
    retired: dict[str, set[str]] = {}
    for field in EXCEPTION_BASELINE_ID_FIELDS:
        active = _seed_ids(seed, field, "active")
        already_retired = _seed_ids(seed, field, "retired")
        additions = current_ids[field] - active
        if additions:
            raise ValueError(
                f"cannot retire {field}; current contains non-active IDs: "
                f"{sorted(additions)}"
            )
        retired[field] = already_retired | (active - current_ids[field])
    updated = _exception_baseline_payload(
        source,
        compatibility,
        retired=retired,
    )
    violations = _seed_state_violations(
        source,
        compatibility,
        updated,
        frozen,
        seed,
    )
    if any(violations.values()):
        raise ValueError(f"invalid retirement transition: {violations}")
    return updated


def _exception_removal_campaign(row: dict[str, Any]) -> str:
    category = str(row.get("category") or "")
    source = str(row.get("source_package") or "")
    target = str(row.get("target_package") or "")
    if target == "lab" and source in PRODUCTION_PACKAGES:
        if source == "classification":
            return "P2" if category == "cross_domain_edge" else "P3"
        if source in {"decision", "ops", "search"}:
            return "P4"
        if source == "librarian":
            return "P5"
    if category == "schema_manifest_implementation_import":
        return "P6"
    if category == "private_symbol_import":
        return "P8"
    if category == "dynamic_import":
        if source == target:
            return "Q"
        if target in {"ingest", "librarian", "raw", "recall", "research", "search"}:
            return "R"
    return "S"


def _exception_metadata(row: dict[str, Any]) -> dict[str, str]:
    category = str(row["category"])
    campaign = _exception_removal_campaign(row)
    source = str(row.get("source_package") or "")
    target = str(row.get("target_package") or "")
    owner = str(row.get("source_module") or f"chronovisor.{source}")
    rationale = {
        "cross_domain_edge": (
            f"Remove the {source}->{target} package edge through a published contract."
        ),
        "private_symbol_import": (
            f"Replace the {source}->{target} private API dependency with a public API."
        ),
        "dynamic_import": (
            f"Move the {source}->{target} runtime import into explicit composition."
        ),
        "schema_manifest_implementation_import": (
            "Invert the schema registry so implementations publish schemas through a contract."
        ),
    }[category]
    return {
        "owner": owner,
        "deadline": CAMPAIGN_DEADLINES[campaign],
        "removal_campaign": campaign,
        "rationale": rationale,
    }


def _compatibility_metadata(row: dict[str, Any]) -> dict[str, str]:
    kind = str(row["kind"])
    if kind == "module_string":
        owner = "chronovisor.core.module_paths"
    elif kind == "lab_dispatch":
        owner = "chronovisor.lab.cli"
    else:
        owner = str(row["target"]).split(":", maxsplit=1)[0]
    return {
        "owner": owner,
        "deadline": CAMPAIGN_DEADLINES["V"],
        "removal_campaign": "V",
        "rationale": (
            "Retire after one mixed-version generation, zero observed legacy use, "
            "and a verified rollback target."
        ),
    }


def _preserved_metadata(
    existing: dict[str, Any] | None,
    fallback: dict[str, str],
    *,
    legacy_owner: str,
) -> dict[str, str]:
    if existing is None or existing.get("owner") == legacy_owner:
        return fallback
    if _missing_metadata([existing]):
        return fallback
    return {field: str(existing[field]) for field in EXCEPTION_METADATA_FIELDS}


def build_architecture_exception_ledger(root: Path) -> dict[str, Any]:
    root = root.resolve()
    package_root = root / "src" / "chronovisor"
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    source, _edges = _source_inventory(package_root)
    detected_rows = _architecture_exception_rows(source)
    compatibility = _compatibility_contracts(pyproject, package_root)
    seed = _load_architecture_exception_baseline(root)
    previous_seed = _load_previous_architecture_exception_baseline(root)
    frozen_reference = build_architecture_exception_baseline(root)
    seed_violations = _seed_state_violations(
        source,
        compatibility,
        seed,
        frozen_reference,
        previous_seed,
    )
    if any(seed_violations.values()):
        raise ValueError(f"architecture exception baseline mismatch: {seed_violations}")
    existing_ledger = _load_architecture_exception_ledger(root)
    existing_exceptions = {
        str(row.get("semantic_id") or ""): dict(row)
        for row in existing_ledger.get("exceptions", [])
        if isinstance(row, dict)
    }
    exceptions: list[dict[str, Any]] = []
    for detected in detected_rows:
        semantic_id = str(detected["semantic_id"])
        exceptions.append(
            {
                **detected,
                **_preserved_metadata(
                    existing_exceptions.get(semantic_id),
                    _exception_metadata(detected),
                    legacy_owner="chronovisor-architecture",
                ),
            }
        )
    existing_compatibility = {
        str(row.get("semantic_id") or ""): dict(row)
        for row in existing_ledger.get("compatibility_contracts", [])
        if isinstance(row, dict)
    }
    compatibility_contracts: list[dict[str, Any]] = []
    for detected in compatibility:
        semantic_id = str(detected["semantic_id"])
        compatibility_contracts.append(
            {
                **detected,
                **_preserved_metadata(
                    existing_compatibility.get(semantic_id),
                    _compatibility_metadata(detected),
                    legacy_owner="chronovisor-compatibility",
                ),
            }
        )
    return {
        "schema_version": EXCEPTION_LEDGER_SCHEMA_VERSION,
        "source_baseline_head": FROZEN_EXCEPTION_SOURCE_HEAD,
        "baseline_sha256": _canonical_sha256(seed),
        "semantic_identity": (
            "cross-domain edges use category+source_package+target_package; "
            "statement exceptions use category+source_package+source_module+"
            "scope_kind+scope+statement_kind+target_package+target_module+symbols+"
            "occurrence; source line is diagnostic and normalized statement content "
            "is verified independently"
        ),
        "counts": seed["counts"]["active"],
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
    architecture_exception_baseline = _load_architecture_exception_baseline(root)
    previous_architecture_exception_baseline = (
        _load_previous_architecture_exception_baseline(root)
    )
    frozen_architecture_exception_reference = build_architecture_exception_baseline(
        root
    )
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
        "architecture_exception_baseline": architecture_exception_baseline,
        "previous_architecture_exception_baseline": (
            previous_architecture_exception_baseline
        ),
        "frozen_architecture_exception_reference": (
            frozen_architecture_exception_reference
        ),
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


def _duplicate_row_ids(rows: list[dict[str, Any]]) -> list[str]:
    ids = [str(row.get("semantic_id") or "") for row in rows]
    return sorted({semantic_id for semantic_id in ids if ids.count(semantic_id) > 1})


def _recorded_cross_domain_sites(
    exception_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    for edge in exception_rows:
        if edge.get("category") != "cross_domain_edge":
            continue
        edge_sites = edge.get("sites")
        if not isinstance(edge_sites, list):
            continue
        for value in edge_sites:
            if not isinstance(value, dict):
                continue
            sites.append(
                {
                    "category": "cross_domain_import",
                    "source_package": edge.get("source_package"),
                    "target_package": edge.get("target_package"),
                    **value,
                }
            )
    return sites


def _architecture_exception_violations(
    current_source: dict[str, Any],
    ledger: dict[str, Any],
    current_compatibility: list[dict[str, Any]],
    seed: dict[str, Any],
    frozen_reference: dict[str, Any],
    previous_seed: dict[str, Any],
) -> dict[str, Any]:
    detected_rows = _architecture_exception_rows(current_source)
    detected = {str(row["semantic_id"]): row for row in detected_rows}
    exception_rows = [
        dict(row) for row in ledger.get("exceptions", []) if isinstance(row, dict)
    ]
    recorded = {str(row.get("semantic_id") or ""): row for row in exception_rows}
    baseline_ids = _seed_ids(seed, "exception_semantic_ids", "active")
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
    exception_content_mismatches = sorted(
        semantic_id
        for semantic_id in detected_ids & recorded_ids
        if detected[semantic_id].get("category") != "cross_domain_edge"
        and detected[semantic_id].get("content_sha256")
        != recorded[semantic_id].get("content_sha256")
    )

    detected_site_rows = [
        dict(row)
        for row in current_source.get("import_sites", [])
        if isinstance(row, dict) and row.get("category") == "cross_domain_import"
    ]
    recorded_site_rows = _recorded_cross_domain_sites(exception_rows)
    detected_sites = {
        str(row.get("semantic_id") or ""): row for row in detected_site_rows
    }
    recorded_sites = {
        str(row.get("semantic_id") or ""): row for row in recorded_site_rows
    }
    detected_site_ids = {value for value in detected_sites if value}
    recorded_site_ids = {value for value in recorded_sites if value}
    baseline_site_ids = _seed_ids(seed, "cross_domain_site_semantic_ids", "active")
    site_identity_mismatches: list[str] = []
    for semantic_id, row in recorded_sites.items():
        try:
            expected = _semantic_id(row)
        except KeyError:
            expected = ""
        if not semantic_id or semantic_id != expected:
            site_identity_mismatches.append(semantic_id)
    site_content_mismatches = sorted(
        semantic_id
        for semantic_id in detected_site_ids & recorded_site_ids
        if detected_sites[semantic_id].get("content_sha256")
        != recorded_sites[semantic_id].get("content_sha256")
    )

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

    current_id_sets = _exception_id_sets(current_source, current_compatibility)
    production_lab_baseline = _seed_ids(
        seed, "production_to_lab_edge_semantic_ids", "active"
    )
    production_lab_current = current_id_sets["production_to_lab_edge_semantic_ids"]
    static_lab_baseline = _seed_ids(
        seed, "production_to_lab_static_site_semantic_ids", "active"
    )
    dynamic_lab_baseline = _seed_ids(
        seed, "production_to_lab_dynamic_site_semantic_ids", "active"
    )
    site_counts = {
        "detected": len(detected_site_rows),
        "recorded": len(recorded_site_rows),
        "frozen_active": len(baseline_site_ids),
        "ledger_declared": ledger.get("counts", {}).get("cross_domain_sites")
        if isinstance(ledger.get("counts"), dict)
        else None,
        "seed_declared": seed.get("counts", {})
        .get("active", {})
        .get("cross_domain_sites")
        if isinstance(seed.get("counts"), dict)
        and isinstance(seed.get("counts", {}).get("active"), dict)
        else None,
    }
    site_count_drift = site_counts if len(set(site_counts.values())) != 1 else {}
    retired_ids = {
        field: _seed_ids(seed, field, "retired")
        for field in EXCEPTION_BASELINE_ID_FIELDS
    }
    expected_ledger_counts = _exception_counts(
        current_source,
        current_compatibility,
        retired_ids,
    )["active"]
    ledger_count_drift = (
        {"recorded": ledger.get("counts"), "expected": expected_ledger_counts}
        if ledger.get("counts") != expected_ledger_counts
        else {}
    )
    seed_violations = _seed_state_violations(
        current_source,
        current_compatibility,
        seed,
        frozen_reference,
        previous_seed,
    )
    violations = {
        "ledger_load_error": str(ledger.get("load_error") or ""),
        "ledger_schema_version": (
            []
            if ledger.get("schema_version") == EXCEPTION_LEDGER_SCHEMA_VERSION
            else [ledger.get("schema_version")]
        ),
        "ledger_source_baseline_head_drift": (
            []
            if ledger.get("source_baseline_head") == FROZEN_EXCEPTION_SOURCE_HEAD
            else [ledger.get("source_baseline_head")]
        ),
        "ledger_baseline_sha256_drift": (
            []
            if ledger.get("baseline_sha256") == _canonical_sha256(seed)
            else [ledger.get("baseline_sha256")]
        ),
        "new_exception_ids": sorted(detected_ids - baseline_ids),
        "unrecorded_exception_ids": sorted(detected_ids - recorded_ids),
        "stale_exception_ids": sorted(recorded_ids - detected_ids),
        "baseline_semantic_id_non_subset": sorted(baseline_ids - recorded_ids),
        "exception_identity_mismatches": sorted(identity_mismatches),
        "exception_content_mismatches": exception_content_mismatches,
        "exception_metadata_missing": _missing_metadata(exception_rows),
        "duplicate_exception_ids": sorted(
            set(_duplicate_row_ids(detected_rows))
            | set(_duplicate_row_ids(exception_rows))
        ),
        "new_cross_domain_site_ids": sorted(detected_site_ids - baseline_site_ids),
        "unrecorded_cross_domain_site_ids": sorted(
            detected_site_ids - recorded_site_ids
        ),
        "stale_cross_domain_site_ids": sorted(recorded_site_ids - detected_site_ids),
        "baseline_site_semantic_id_non_subset": sorted(
            baseline_site_ids - recorded_site_ids
        ),
        "site_identity_mismatches": sorted(site_identity_mismatches),
        "site_content_mismatches": site_content_mismatches,
        "duplicate_site_ids": sorted(
            set(_duplicate_row_ids(detected_site_rows))
            | set(_duplicate_row_ids(recorded_site_rows))
        ),
        "site_count_drift": site_count_drift,
        "ledger_count_drift": ledger_count_drift,
        "production_to_lab_edge_growth": sorted(
            production_lab_current - production_lab_baseline
        ),
        "production_to_lab_static_site_growth": sorted(
            current_id_sets["production_to_lab_static_site_semantic_ids"]
            - static_lab_baseline
        ),
        "production_to_lab_dynamic_site_growth": sorted(
            current_id_sets["production_to_lab_dynamic_site_semantic_ids"]
            - dynamic_lab_baseline
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
        "duplicate_compatibility_ids": sorted(
            set(_duplicate_row_ids(current_compatibility))
            | set(_duplicate_row_ids(compatibility_rows))
        ),
        **seed_violations,
    }
    return violations


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
        current.get("architecture_exception_baseline", {}),
        current.get("frozen_architecture_exception_reference", {}),
        current.get("previous_architecture_exception_baseline", {}),
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
    generators = parser.add_mutually_exclusive_group()
    generators.add_argument("--generate-exceptions", action="store_true")
    generators.add_argument("--bootstrap-exception-baseline", action="store_true")
    generators.add_argument("--retire-missing-exceptions", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_exception_baseline:
        payload = build_architecture_exception_baseline(args.root)
    elif args.retire_missing_exceptions:
        payload = retire_missing_architecture_exceptions(args.root)
    elif args.generate_exceptions:
        payload = build_architecture_exception_ledger(args.root)
    else:
        captured_at = args.captured_at or "verification"
        payload = scan_repository(args.root, captured_at=captured_at)
    exit_code = 0
    if args.check is not None and not any(
        (
            args.generate_exceptions,
            args.bootstrap_exception_baseline,
            args.retire_missing_exceptions,
        )
    ):
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
