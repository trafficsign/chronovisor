"""Pinned Nemotron encoder used by the dedicated semantic service."""

from __future__ import annotations

import gc
import importlib.metadata
from pathlib import Path
from typing import Sequence

import numpy as np

from chronovisor.core.runtime_config import SearchEmbeddingConfig


class SemanticModelError(RuntimeError):
    """Raised when the pinned search model cannot be loaded or validated."""


def _local_snapshot(config: SearchEmbeddingConfig) -> Path | str:
    if not config.offline:
        return config.model
    model_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{config.model.replace('/', '--')}"
        / "snapshots"
        / config.revision
    )
    if not model_dir.is_dir():
        raise SemanticModelError(
            f"pinned model snapshot is missing: {model_dir}"
        )
    return model_dir


def _normalized(vectors: object, dimensions: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != dimensions:
        raise SemanticModelError(
            f"unexpected embedding shape: {matrix.shape}; expected (*, {dimensions})"
        )
    if not np.isfinite(matrix).all():
        raise SemanticModelError("embedding contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise SemanticModelError("embedding contains a zero-norm vector")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class NemotronEncoder:
    """One lazy, device-bound SentenceTransformer instance.

    Nemotron's ``encode_query`` and ``encode_document`` methods own the
    asymmetric prompt contract.  The configured prefixes are recorded in the
    generation manifest as an encoding contract, but are deliberately not
    prepended a second time here.
    """

    def __init__(self, config: SearchEmbeddingConfig, *, device: str) -> None:
        self.config = config
        self.device = device
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SemanticModelError(
                "install the 'semantic' extra to run Nemotron retrieval"
            ) from exc

        source = _local_snapshot(config)
        kwargs = {
            "device": device,
            "model_kwargs": {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
            },
        }
        if config.offline:
            kwargs["local_files_only"] = True
        try:
            self.model = SentenceTransformer(str(source), **kwargs)
        except TypeError:
            # Older sentence-transformers accepts ``dtype`` rather than the
            # Transformers ``torch_dtype`` spelling.
            kwargs["model_kwargs"] = {
                "dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
            }
            self.model = SentenceTransformer(str(source), **kwargs)

    def encode_queries(
        self, texts: Sequence[str], batch_size: int = 8
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.config.dimensions), dtype=np.float32)
        vectors = self.model.encode_query(
            list(texts),
            batch_size=max(1, batch_size),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return _normalized(vectors, self.config.dimensions)

    def encode_documents(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.config.dimensions), dtype=np.float32)
        vectors = self.model.encode_document(
            list(texts),
            batch_size=max(1, batch_size),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return _normalized(vectors, self.config.dimensions)

    def self_test(self) -> dict[str, object]:
        query = self.encode_queries(["Chronovisorの検索インデックス"], 1)[0]
        documents = self.encode_documents(
            [
                "ChronovisorはローカルAI向けの記憶検索システムです。",
                "夕食のレシピと材料についてのメモです。",
            ],
            2,
        )
        scores = documents @ query
        if not float(scores[0]) > float(scores[1]):
            raise SemanticModelError("known-vector ranking self-test failed")
        return {
            "device": self.device,
            "dimensions": int(query.shape[0]),
            "positive_score": float(scores[0]),
            "negative_score": float(scores[1]),
        }

    def close(self) -> None:
        self.model = None
        gc.collect()
        try:
            import torch

            if self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def semantic_runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "sentence-transformers", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions
