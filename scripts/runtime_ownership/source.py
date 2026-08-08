# ruff: noqa: F401, F403, F405
"""Runtime ownership source layer."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import plistlib
import re
import subprocess
import tarfile
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from .model import *


def _snapshot_current(root: Path) -> dict[str, bytes]:
    paths = [root / "pyproject.toml"]
    paths.extend(sorted((root / "src" / "chronovisor").rglob("*.py")))
    paths.extend(sorted((root / "launchd").glob("*.plist")))
    paths.extend(
        path
        for path in sorted((root / "scripts").glob("chronovisor-*"))
        if path.is_file()
    )
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in paths
        if path.is_file()
    }


def _snapshot_revision(root: Path, revision: str) -> dict[str, bytes]:
    completed = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            revision,
            "pyproject.toml",
            "src/chronovisor",
            "launchd",
            "scripts",
        ],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace").strip())
    result: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                result[member.name] = handle.read()
    return result


def _text(snapshot: dict[str, bytes], path: str) -> str:
    return snapshot[path].decode("utf-8")


def _module_name(path: str) -> str:
    relative = PurePosixPath(path).relative_to("src")
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _matches_suffix(name: str) -> bool:
    return any(
        name == suffix or name.endswith(f"_{suffix}") for suffix in NAME_SUFFIXES
    )


def _is_schema_name(name: str) -> bool:
    return name == "SCHEMA" or name.endswith("_SCHEMA")


def _is_schema_version_name(name: str) -> bool:
    return name == "SCHEMA_VERSION" or name.endswith("_SCHEMA_VERSION")


def _assignment_names(node: ast.stmt) -> tuple[list[str], ast.expr | None]:
    if isinstance(node, ast.Assign):
        return (
            [target.id for target in node.targets if isinstance(target, ast.Name)],
            node.value,
        )
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id], node.value
    return [], None


def _resolve_import_module(current: str, module: str | None, level: int) -> str:
    if level == 0:
        return module or ""
    parts = current.split(".")[:-1]
    retained = max(0, len(parts) - level + 1)
    prefix = parts[:retained]
    if module:
        prefix.extend(module.split("."))
    return ".".join(prefix)


class _SourceIndex:
    def __init__(self, snapshot: dict[str, bytes]) -> None:
        self.snapshot = snapshot
        self.paths: dict[str, str] = {}
        self.trees: dict[str, ast.Module] = {}
        self.definitions: dict[tuple[str, str], tuple[ast.expr | None, int]] = {}
        self.imports: dict[tuple[str, str], tuple[str, str | None]] = {}
        self.symbols: dict[str, set[str]] = {}
        for path in sorted(snapshot):
            if not path.startswith("src/chronovisor/") or not path.endswith(".py"):
                continue
            module = _module_name(path)
            tree = ast.parse(_text(snapshot, path), filename=path)
            self.paths[module] = path
            self.trees[module] = tree
            symbols: set[str] = set()
            for node in tree.body:
                names, value = _assignment_names(node)
                for name in names:
                    symbols.add(name)
                    self.definitions[(module, name)] = (value, int(node.lineno))
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    symbols.add(node.name)
                if isinstance(node, ast.ImportFrom):
                    target_module = _resolve_import_module(
                        module, node.module, node.level
                    )
                    for alias in node.names:
                        local = alias.asname or alias.name
                        symbols.add(local)
                        self.imports[(module, local)] = (target_module, alias.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        local = alias.asname or alias.name.split(".")[0]
                        symbols.add(local)
                        self.imports[(module, local)] = (alias.name, None)
            self.symbols[module] = symbols

    def symbol_exists(self, reference: str) -> bool:
        if reference.startswith("external:"):
            return bool(reference.removeprefix("external:"))
        if reference.startswith("script:"):
            return reference.removeprefix("script:") in self.snapshot
        if ":" not in reference:
            return False
        module, symbol = reference.split(":", maxsplit=1)
        return module in self.paths and symbol in self.symbols.get(module, set())

    def evaluate_reference(
        self,
        module: str,
        name: str,
        stack: frozenset[tuple[str, str]] = frozenset(),
    ) -> str | None:
        key = (module, name)
        if key in stack:
            return None
        if name == "CHRONOVISOR_ROOT":
            return "$CHRONOVISOR_ROOT"
        imported = self.imports.get(key)
        if imported is not None and imported[1] is not None:
            return self.evaluate_reference(imported[0], imported[1], stack | {key})
        definition = self.definitions.get(key)
        if definition is None:
            return None
        return self.evaluate_expression(module, definition[0], stack | {key})

    def evaluate_expression(
        self,
        module: str,
        expression: ast.expr | None,
        stack: frozenset[tuple[str, str]],
    ) -> str | None:
        if expression is None:
            return None
        if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
            return _normalize_literal(expression.value)
        if isinstance(expression, ast.Name):
            return self.evaluate_reference(module, expression.id, stack)
        if isinstance(expression, ast.Attribute):
            if isinstance(expression.value, ast.Name):
                imported = self.imports.get((module, expression.value.id))
                if imported is not None:
                    imported_module = imported[0]
                    if imported[1] is not None:
                        submodule = f"{imported_module}.{imported[1]}"
                        imported_module = submodule if submodule in self.paths else ""
                    resolved = (
                        self.evaluate_reference(imported_module, expression.attr, stack)
                        if imported_module
                        else None
                    )
                    if resolved is not None:
                        return resolved
            base = self.evaluate_expression(module, expression.value, stack)
            if expression.attr == "parent" and base:
                return _parent_locator(base)
            return None
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Div):
            base = self.evaluate_expression(module, expression.left, stack)
            child = self.evaluate_expression(module, expression.right, stack)
            if base and child:
                return _join_locator(base, child)
            return None
        if not isinstance(expression, ast.Call):
            return None
        call_name = _call_name(expression.func)
        if call_name in {"resolve_root", "chronovisor.core.store.resolve_root"}:
            return "$CHRONOVISOR_ROOT"
        if call_name.endswith("runtime_repo_root"):
            return "$REPO_ROOT"
        if call_name.endswith("active_config_file") and not expression.args:
            return "$CHRONOVISOR_ROOT/config.toml"
        if call_name in {"Path.home", "pathlib.Path.home"}:
            return "$HOME"
        if call_name in {"Path", "pathlib.Path"} and expression.args:
            return self.evaluate_expression(module, expression.args[0], stack)
        if isinstance(expression.func, ast.Attribute):
            base = self.evaluate_expression(module, expression.func.value, stack)
            if expression.func.attr in {"expanduser", "resolve", "absolute"}:
                return base
            if expression.func.attr == "with_suffix" and base and expression.args:
                suffix = self.evaluate_expression(module, expression.args[0], stack)
                return _with_suffix(base, suffix) if suffix is not None else None
            if expression.func.attr == "joinpath" and base:
                result = base
                for argument in expression.args:
                    child = self.evaluate_expression(module, argument, stack)
                    if child is None:
                        return None
                    result = _join_locator(result, child)
                return result
        return None


def _call_name(expression: ast.expr) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        prefix = _call_name(expression.value)
        return f"{prefix}.{expression.attr}" if prefix else expression.attr
    return ""


def _normalize_literal(value: str) -> str:
    expanded = value.strip()
    if expanded.startswith("~/"):
        return f"$HOME/{expanded[2:]}"
    if expanded == "~":
        return "$HOME"
    return expanded


def _join_locator(base: str, child: str) -> str:
    if child.startswith("/"):
        return child
    return f"{base.rstrip('/')}/{child.strip('/')}"


def _parent_locator(value: str) -> str:
    head, separator, _tail = value.rstrip("/").rpartition("/")
    return head if separator else value


def _with_suffix(value: str, suffix: str) -> str:
    head, separator, tail = value.rpartition("/")
    stem = tail.rsplit(".", maxsplit=1)[0]
    replaced = f"{stem}{suffix}"
    return f"{head}/{replaced}" if separator else replaced


def _resource_kind(name: str, locator: str) -> str:
    if (
        name.endswith("_LOCK")
        or name.endswith("_LOCK_FILE")
        or locator.endswith(".lock")
    ):
        return "lock"
    if (
        name == "QUEUE_FILE"
        or name.endswith("_QUEUE")
        or name.endswith("_QUEUE_FILE")
        or name == "SEMANTIC_JOBS_DB"
    ):
        return "queue"
    return "artifact"


def _exclusion_reason(name: str, value: ast.expr | None, resolved: str | None) -> str:
    if _is_schema_version_name(name):
        return "version_only_constant"
    if name == "MIN_CASES_PER_PRODUCTION_SCHEMA":
        return "schema_suffix_quantity_false_positive"
    if _is_schema_name(name):
        return "prompt_or_in_memory_schema"
    if name in SOURCE_OR_DEPLOYMENT_NAMES or resolved == "$REPO_ROOT":
        return "source_fixture_or_deployment_path"
    if isinstance(value, ast.Constant) and value.value is None:
        return "process_local_cache_or_lock"
    if name.endswith("_STATUS") or name.endswith("_STATE"):
        return "status_enum_or_process_local_value"
    return "non_runtime_or_unresolved_origin"


def _discovery_id(row: dict[str, Any]) -> str:
    identity = {key: row[key] for key in ("classification", "path", "module", "symbol")}
    if row["classification"] == "resource":
        identity.update({"kind": row["kind"], "locator": row["locator"]})
    elif row["classification"] == "lock_protocol":
        identity.update(
            {
                "scope": row["scope"],
                "operation": row["operation"],
                "occurrence": row["occurrence"],
            }
        )
    else:
        identity["reason"] = row["reason"]
    return _semantic_id("runtime-site", identity)


def _ast_discovery(index: _SourceIndex) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (module, name), (value, line) in sorted(index.definitions.items()):
        if not _matches_suffix(name):
            continue
        path = index.paths[module]
        resolved = index.evaluate_reference(module, name)
        row: dict[str, Any] = {
            "path": path,
            "line": line,
            "module": module,
            "symbol": name,
            "owner_symbol": f"{module}:{name}",
        }
        if (
            module == "chronovisor.recall.evidence_certificate"
            and name == "CERTIFICATE_LEDGER_LOCK"
        ):
            row.update(
                {
                    "classification": "exclusion",
                    "reason": "legacy_phantom_lock_declaration",
                }
            )
        elif (
            _is_schema_name(name)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and value.value.startswith("chronovisor.")
        ):
            row.update(
                {
                    "classification": "resource",
                    "kind": "schema",
                    "locator": {"type": "schema_id", "value": value.value},
                }
            )
        elif (
            resolved
            and (
                resolved.startswith("$CHRONOVISOR_ROOT")
                or resolved.startswith("$HOME")
                or "/.chronovisor/" in resolved
            )
            and name not in SOURCE_OR_DEPLOYMENT_NAMES
        ):
            row.update(
                {
                    "classification": "resource",
                    "kind": _resource_kind(name, resolved),
                    "locator": {"type": "path", "value": resolved},
                }
            )
        else:
            row.update(
                {
                    "classification": "exclusion",
                    "reason": _exclusion_reason(name, value, resolved),
                }
            )
        row["discovery_id"] = _discovery_id(row)
        rows.append(row)
    return rows


def _explicit_state_discovery(
    snapshot: dict[str, bytes], *, include_planned: bool = False
) -> list[dict[str, Any]]:
    """Register dynamic path construction that suffix scanning cannot resolve."""

    specs: list[dict[str, Any]] = [
        {
            "path": "src/chronovisor/raw/raw_replay.py",
            "needle": 'QUEUE_FILE = CHRONOVISOR_ROOT / "review" / "raw-replay-queue.jsonl"',
            "module": "chronovisor.raw.raw_replay",
            "symbol": "QUEUE_FILE.sidecar-lock",
            "owner_symbol": "chronovisor.raw.raw_replay:QUEUE_FILE",
            "kind": "lock",
            "locator": "$CHRONOVISOR_ROOT/review/raw-replay-queue.jsonl.lock",
        },
        {
            "path": "src/chronovisor/recall/evidence_certificate.py",
            "needle": "def append_certificates(",
            "module": "chronovisor.recall.evidence_certificate",
            "symbol": "append_certificates.sidecar-lock",
            "owner_symbol": "chronovisor.recall.evidence_certificate:append_certificates",
            "kind": "lock",
            "locator": "$CHRONOVISOR_ROOT/recall/evidence-certificate-ledger.jsonl.lock",
        },
        {
            "path": "src/chronovisor/ops/convergence.py",
            "needle": "class ConvergenceStore:",
            "module": "chronovisor.ops.convergence",
            "symbol": "ConvergenceStore.state_file",
            "owner_symbol": "chronovisor.ops.convergence:ConvergenceStore",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/runtime/convergence/state.json",
        },
        {
            "path": "src/chronovisor/ops/convergence.py",
            "needle": "class ConvergenceStore:",
            "module": "chronovisor.ops.convergence",
            "symbol": "ConvergenceStore.events_file",
            "owner_symbol": "chronovisor.ops.convergence:ConvergenceStore",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/runtime/convergence/events.jsonl",
        },
        {
            "path": "src/chronovisor/ops/convergence.py",
            "needle": "class ConvergenceStore:",
            "module": "chronovisor.ops.convergence",
            "symbol": "ConvergenceStore.lock_file",
            "owner_symbol": "chronovisor.ops.convergence:ConvergenceStore",
            "kind": "lock",
            "locator": "$CHRONOVISOR_ROOT/runtime/convergence/state.lock",
        },
        {
            "path": "src/chronovisor/ingest/page_registry.py",
            "needle": "class PageRegistry:",
            "module": "chronovisor.ingest.page_registry",
            "symbol": "PageRegistry.path",
            "owner_symbol": "chronovisor.ingest.page_registry:PageRegistry",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/runtime/librarian/page-registry.json",
        },
        {
            "path": "src/chronovisor/ingest/page_registry.py",
            "needle": "class PageRegistry:",
            "module": "chronovisor.ingest.page_registry",
            "symbol": "PageRegistry.events_path",
            "owner_symbol": "chronovisor.ingest.page_registry:PageRegistry",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/runtime/librarian/page-registry-events.jsonl",
        },
        {
            "path": "src/chronovisor/ingest/page_registry.py",
            "needle": "class PageRegistry:",
            "module": "chronovisor.ingest.page_registry",
            "symbol": "PageRegistry.lock_path",
            "owner_symbol": "chronovisor.ingest.page_registry:PageRegistry",
            "kind": "lock",
            "locator": "$CHRONOVISOR_ROOT/runtime/librarian/page-registry.lock",
        },
        {
            "path": "src/chronovisor/librarian/collection_authority.py",
            "needle": "class CollectionRegistry:",
            "module": "chronovisor.librarian.collection_authority",
            "symbol": "CollectionRegistry.path",
            "owner_symbol": "chronovisor.librarian.collection_authority:CollectionRegistry",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/runtime/librarian/collection-registry.json",
        },
        {
            "path": "src/chronovisor/librarian/collection_authority.py",
            "needle": "class CollectionRegistry:",
            "module": "chronovisor.librarian.collection_authority",
            "symbol": "CollectionRegistry.lock_path",
            "owner_symbol": "chronovisor.librarian.collection_authority:CollectionRegistry",
            "kind": "lock",
            "locator": "$CHRONOVISOR_ROOT/runtime/librarian/collection-registry.lock",
        },
        {
            "path": "src/chronovisor/librarian/managed_hold.py",
            "needle": "class ManagedHoldStore:",
            "module": "chronovisor.librarian.managed_hold",
            "symbol": "ManagedHoldStore.path",
            "owner_symbol": "chronovisor.librarian.managed_hold:ManagedHoldStore",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/runtime/managed-holds/state.json",
        },
        {
            "path": "src/chronovisor/librarian/managed_hold.py",
            "needle": "class ManagedHoldStore:",
            "module": "chronovisor.librarian.managed_hold",
            "symbol": "ManagedHoldStore.lock_path",
            "owner_symbol": "chronovisor.librarian.managed_hold:ManagedHoldStore",
            "kind": "lock",
            "locator": "$CHRONOVISOR_ROOT/runtime/managed-holds/state.json.lock",
        },
        {
            "path": "src/chronovisor/knowledge_graph/store.py",
            "needle": "class KnowledgeGraphStore:",
            "module": "chronovisor.knowledge_graph.store",
            "symbol": "KnowledgeGraphStore.events_file",
            "owner_symbol": "chronovisor.knowledge_graph.store:KnowledgeGraphStore",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/knowledge-graph/relation-events.jsonl",
        },
        {
            "path": "src/chronovisor/knowledge_graph/store.py",
            "needle": "class KnowledgeGraphStore:",
            "module": "chronovisor.knowledge_graph.store",
            "symbol": "KnowledgeGraphStore.snapshot_file",
            "owner_symbol": "chronovisor.knowledge_graph.store:KnowledgeGraphStore",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/knowledge-graph/relation-snapshot.json",
        },
        {
            "path": "src/chronovisor/knowledge_graph/store.py",
            "needle": "class KnowledgeGraphStore:",
            "module": "chronovisor.knowledge_graph.store",
            "symbol": "KnowledgeGraphStore.entity_snapshot_file",
            "owner_symbol": "chronovisor.knowledge_graph.store:KnowledgeGraphStore",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/knowledge-graph/entity-snapshot.json",
        },
        {
            "path": "src/chronovisor/knowledge_graph/store.py",
            "needle": "class KnowledgeGraphStore:",
            "module": "chronovisor.knowledge_graph.store",
            "symbol": "KnowledgeGraphStore.community_snapshot_file",
            "owner_symbol": "chronovisor.knowledge_graph.store:KnowledgeGraphStore",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/knowledge-graph/community-snapshot.json",
        },
        {
            "path": "src/chronovisor/knowledge_graph/store.py",
            "needle": "class KnowledgeGraphStore:",
            "module": "chronovisor.knowledge_graph.store",
            "symbol": "KnowledgeGraphStore.builder_state_file",
            "owner_symbol": "chronovisor.knowledge_graph.store:KnowledgeGraphStore",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/knowledge-graph/builder-state.json",
        },
        {
            "path": "src/chronovisor/knowledge_graph/store.py",
            "needle": "class KnowledgeGraphStore:",
            "module": "chronovisor.knowledge_graph.store",
            "symbol": "KnowledgeGraphStore.community_summary_state_file",
            "owner_symbol": "chronovisor.knowledge_graph.store:KnowledgeGraphStore",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/knowledge-graph/community-summary-state.json",
        },
        {
            "path": "src/chronovisor/knowledge_graph/store.py",
            "needle": "class KnowledgeGraphStore:",
            "module": "chronovisor.knowledge_graph.store",
            "symbol": "KnowledgeGraphStore.lock_file",
            "owner_symbol": "chronovisor.knowledge_graph.store:KnowledgeGraphStore",
            "kind": "lock",
            "locator": "$CHRONOVISOR_ROOT/knowledge-graph/store.lock",
        },
        {
            "path": "src/chronovisor/decision/failure_supervisor.py",
            "needle": "def _failure_state_lock(",
            "module": "chronovisor.decision.failure_supervisor",
            "symbol": "_state_file",
            "owner_symbol": "chronovisor.decision.failure_supervisor:_state_file",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/runtime/failures/state.json",
        },
        {
            "path": "src/chronovisor/decision/failure_supervisor.py",
            "needle": "def _failure_state_lock(",
            "module": "chronovisor.decision.failure_supervisor",
            "symbol": "_failure_state_lock",
            "owner_symbol": "chronovisor.decision.failure_supervisor:_failure_state_lock",
            "kind": "lock",
            "locator": "$CHRONOVISOR_ROOT/runtime/failures/state.lock",
        },
        {
            "path": "src/chronovisor/ops/dashboard.py",
            "needle": 'CHRONOVISOR_ROOT / "runtime" / "dashboard-access-token"',
            "module": "chronovisor.ops.dashboard",
            "symbol": "dashboard-access-token",
            "owner_symbol": "chronovisor.ops.dashboard:serve",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/runtime/dashboard-access-token",
        },
        {
            "path": "src/chronovisor/ops/dashboard.py",
            "needle": 'CHRONOVISOR_ROOT / "runtime" / "dashboard-credentials.json"',
            "module": "chronovisor.ops.dashboard",
            "symbol": "dashboard-credentials",
            "owner_symbol": "chronovisor.ops.dashboard:serve",
            "kind": "artifact",
            "locator": "$CHRONOVISOR_ROOT/runtime/dashboard-credentials.json",
        },
        {
            "path": "scripts/chronovisor-searxng",
            "needle": 'readonly SETTINGS="$CONFIG_ROOT/settings.yml"',
            "module": "script:scripts/chronovisor-searxng",
            "symbol": "SETTINGS",
            "owner_symbol": "script:scripts/chronovisor-searxng",
            "kind": "artifact",
            "locator": "$HOME/.chronovisor/runtime/searxng/settings.yml",
        },
        {
            "path": "scripts/chronovisor-searxng",
            "needle": 'readonly SECRET_FILE="$CONFIG_ROOT/secret"',
            "module": "script:scripts/chronovisor-searxng",
            "symbol": "SECRET_FILE",
            "owner_symbol": "script:scripts/chronovisor-searxng",
            "kind": "artifact",
            "locator": "$HOME/.chronovisor/runtime/searxng/secret",
        },
        {
            "path": "scripts/chronovisor-searxng",
            "needle": 'readonly RUNTIME_ROOT="${CHRONOVISOR_SEARXNG_ROOT:-$HOME/.local/share/chronovisor/searxng}"',
            "module": "script:scripts/chronovisor-searxng",
            "symbol": "source",
            "owner_symbol": "script:scripts/chronovisor-searxng",
            "kind": "artifact",
            "locator": "$HOME/.local/share/chronovisor/searxng/source",
        },
        {
            "path": "scripts/chronovisor-searxng",
            "needle": 'readonly VENV="$RUNTIME_ROOT/.venv"',
            "module": "script:scripts/chronovisor-searxng",
            "symbol": "VENV",
            "owner_symbol": "script:scripts/chronovisor-searxng",
            "kind": "artifact",
            "locator": "$HOME/.local/share/chronovisor/searxng/.venv",
        },
        {
            "path": "src/chronovisor/search/search_eval.py",
            "needle": 'LABEL_QUEUE_FILE = RECALL_DIR / "search-label-queue.jsonl"',
            "module": "chronovisor.search.search_eval",
            "symbol": "LABEL_QUEUE_FILE.sidecar-lock",
            "owner_symbol": "chronovisor.search.search_eval:LABEL_QUEUE_FILE",
            "kind": "lock",
            "locator": "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl.lock",
        },
        {
            "path": "src/chronovisor/recall/claims.py",
            "needle": 'CLAIMS_FILE = CLAIMS_DIR / "claims.jsonl"',
            "module": "chronovisor.recall.claims",
            "symbol": "CLAIMS_FILE.sidecar-lock",
            "owner_symbol": "chronovisor.recall.claims:CLAIMS_FILE",
            "kind": "lock",
            "locator": "$CHRONOVISOR_ROOT/claims/claims.jsonl.lock",
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec["symbol"] == "LABEL_QUEUE_FILE.sidecar-lock" and not (
            include_planned
            or "_search_label_queue_lock(candidate_file)"
            in _text(snapshot, "src/chronovisor/ops/golden_expand.py")
            or "_search_label_queue_lock(output_file)"
            in _text(snapshot, "src/chronovisor/search/search_eval.py")
            or "_search_label_queue_lock(queue_file)"
            in _text(snapshot, "src/chronovisor/search/search_eval.py")
        ):
            continue
        if spec["symbol"] == "CLAIMS_FILE.sidecar-lock" and not (
            include_planned
            or "_claims_ledger_lock(CLAIMS_FILE)"
            in _text(snapshot, "src/chronovisor/recall/claims.py")
            or "_claims_ledger_lock(path)"
            in _text(snapshot, "src/chronovisor/recall/claims.py")
        ):
            continue
        path = str(spec["path"])
        row: dict[str, Any] = {
            "classification": "resource",
            "path": path,
            "line": _line_containing(_text(snapshot, path), str(spec["needle"])),
            "module": spec["module"],
            "symbol": spec["symbol"],
            "owner_symbol": spec["owner_symbol"],
            "kind": spec["kind"],
            "locator": {"type": "path", "value": spec["locator"]},
        }
        row["discovery_id"] = _discovery_id(row)
        rows.append(row)
    return rows


def _line_containing(text: str, needle: str) -> int:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return line_number
    raise ValueError(f"evidence marker not found: {needle}")


__all__ = [
    "_snapshot_current",
    "_snapshot_revision",
    "_text",
    "_module_name",
    "_matches_suffix",
    "_is_schema_name",
    "_is_schema_version_name",
    "_assignment_names",
    "_resolve_import_module",
    "_SourceIndex",
    "_call_name",
    "_normalize_literal",
    "_join_locator",
    "_parent_locator",
    "_with_suffix",
    "_resource_kind",
    "_exclusion_reason",
    "_discovery_id",
    "_ast_discovery",
    "_explicit_state_discovery",
    "_line_containing",
]
