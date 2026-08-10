"""Nemotron semantic model diagnostics kept outside the runtime adapter."""

from __future__ import annotations

import importlib.metadata

from chronovisor.core.nemotron_adapter import (
    SemanticModelError,
    normalize_embeddings,
)

_normalized = normalize_embeddings

__all__ = ["SemanticModelError", "_normalized", "semantic_runtime_versions"]


def semantic_runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "sentence-transformers", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions
