from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "chronovisor"

_OLLAMA_OPERATIONS = frozenset({"chat", "embed", "generate"})
_MODEL_CLASSES = frozenset(
    {
        "AutoModelForSequenceClassification",
        "AutoTokenizer",
        "FlagReranker",
        "SentenceTransformer",
    }
)
_MODEL_MODULES = frozenset({"FlagEmbedding", "sentence_transformers", "transformers"})
_MODEL_METHODS = frozenset(
    {"compute_score", "encode_document", "encode_query", "from_pretrained"}
)

# These are the only production runtime boundaries allowed to touch providers.
RUNTIME_BOUNDARIES = {
    "src/chronovisor/core/nemotron_adapter.py": {
        "model_class.SentenceTransformer": 2,
        "model_method.encode_document": 1,
        "model_method.encode_query": 1,
    },
    "src/chronovisor/core/ollama_adapter.py": {
        "ollama.chat": 1,
        "ollama.embed": 1,
        "ollama.generate": 1,
    },
    "src/chronovisor/core/ollama_transport.py": {
        "ollama_http.chat": 1,
        "ollama_http.embed": 1,
        "ollama_http.generate": 3,
    },
    "src/chronovisor/core/reranker.py": {
        "model_call": 1,
        "model_class.AutoModelForSequenceClassification": 1,
        "model_class.AutoTokenizer": 1,
        "model_class.FlagReranker": 1,
        "model_method.compute_score": 2,
        "model_method.from_pretrained": 4,
    },
}

# Existing direct production access is migration debt, not an allow-list.
LEGACY_PRODUCTION_BYPASSES = {
    "src/chronovisor/ingest/ingest.py": {
        "ollama.chat": 1,
        "ollama.generate": 1,
    },
    "src/chronovisor/recall/recall_processor.py": {"ollama.chat": 1},
}

# These tools do not run in production workflows; each exception carries its reason.
NON_PRODUCTION_EXCEPTIONS = {
    "src/chronovisor/classification/classification_resource_probe.py": (
        {"ollama.embed": 1, "ollama.generate": 1},
        "Explicit local model capacity probe.",
    ),
    "src/chronovisor/decision/local_model_eval.py": (
        {"ollama.chat": 1},
        "Offline local-model evaluation entry point.",
    ),
    "src/chronovisor/lab/classification_anchor_complement_auditor.py": (
        {"ollama.chat": 1},
        "Lab-only classification auditor.",
    ),
    "src/chronovisor/lab/classification_anchor_second_auditor.py": (
        {"ollama.chat": 1},
        "Lab-only classification auditor.",
    ),
    "src/chronovisor/lab/classification_pilot.py": (
        {"ollama.chat": 1},
        "Lab-only classification pilot.",
    ),
    "src/chronovisor/librarian/librarian_burn.py": (
        {"ollama.generate": 2},
        "Explicit burn-test harness.",
    ),
    "src/chronovisor/recall/classification_resource_burn.py": (
        {"ollama.generate": 1},
        "Explicit burn-test harness.",
    ),
}


class _InferenceVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.ollama_modules: set[str] = set()
        self.ollama_functions: dict[str, str] = {}
        self.model_classes: dict[str, str] = {}
        self.counts: Counter[str] = Counter()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "chronovisor.core.ollama":
                self.ollama_modules.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "chronovisor.core":
            for alias in node.names:
                if alias.name == "ollama":
                    self.ollama_modules.add(alias.asname or alias.name)
        elif node.module == "chronovisor.core.ollama":
            for alias in node.names:
                if alias.name in _OLLAMA_OPERATIONS:
                    self.ollama_functions[alias.asname or alias.name] = alias.name
        elif node.module in _MODEL_MODULES:
            for alias in node.names:
                if alias.name in _MODEL_CLASSES:
                    self.model_classes[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        module = _qualified_name(node.value)
        if module in self.ollama_modules and node.attr in _OLLAMA_OPERATIONS:
            self.counts[f"ollama.{node.attr}"] += 1
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id in self.ollama_functions:
                self.counts[f"ollama.{self.ollama_functions[node.id]}"] += 1
            elif node.id in self.model_classes:
                self.counts[f"model_class.{self.model_classes[node.id]}"] += 1

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in _MODEL_METHODS:
            self.counts[f"model_method.{function.attr}"] += 1
        elif (
            isinstance(function, ast.Name)
            and function.id == "model"
            and any(keyword.arg is None for keyword in node.keywords)
        ):
            self.counts["model_call"] += 1
        self.generic_visit(node)


def _qualified_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _scan_inference_sites() -> dict[str, dict[str, int]]:
    found: dict[str, dict[str, int]] = {}
    for path in SOURCE_ROOT.rglob("*.py"):
        visitor = _InferenceVisitor()
        source = path.read_text(encoding="utf-8")
        visitor.visit(ast.parse(source, filename=str(path)))
        for operation in _OLLAMA_OPERATIONS:
            count = source.count(f'"/api/{operation}"') + source.count(
                f"'/api/{operation}'"
            )
            if count:
                visitor.counts[f"ollama_http.{operation}"] += count
        if visitor.counts:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            found[relative] = dict(sorted(visitor.counts.items()))
    return dict(sorted(found.items()))


def test_production_inference_inventory_matches_frozen_baseline() -> None:
    exceptions = {
        path: sites for path, (sites, _reason) in NON_PRODUCTION_EXCEPTIONS.items()
    }
    expected = RUNTIME_BOUNDARIES | LEGACY_PRODUCTION_BYPASSES | exceptions

    assert _scan_inference_sites() == dict(sorted(expected.items()))


def test_inventory_categories_are_explicit_and_disjoint() -> None:
    boundaries = set(RUNTIME_BOUNDARIES)
    bypasses = set(LEGACY_PRODUCTION_BYPASSES)
    exceptions = set(NON_PRODUCTION_EXCEPTIONS)

    assert not boundaries & bypasses
    assert not boundaries & exceptions
    assert not bypasses & exceptions
    assert all(reason.strip() for _sites, reason in NON_PRODUCTION_EXCEPTIONS.values())
