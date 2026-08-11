"""Local Nemotron implementation of the shared embedding backend contract."""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from chronovisor.core.llm_runtime import (
    EmbeddingPurpose,
    EmbeddingRequest,
    EmbeddingResult,
    RouteLocation,
    SafeBackendError,
)
from chronovisor.core.runtime_config import SearchEmbeddingConfig


class SemanticModelError(RuntimeError):
    """Raised when the pinned search model cannot be loaded or validated."""


class _NemotronModel(Protocol):
    encode_query: Callable[..., object]
    encode_document: Callable[..., object]


def _local_snapshot(config: SearchEmbeddingConfig, model: str) -> Path | str:
    if not config.offline:
        return model
    model_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model.replace('/', '--')}"
        / "snapshots"
        / config.revision
    )
    if not model_dir.is_dir():
        raise SemanticModelError(f"pinned model snapshot is missing: {model_dir}")
    return model_dir


def normalize_embeddings(vectors: object, dimensions: int) -> np.ndarray:
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


class NemotronEmbeddingBackend:
    """One lazy, device-bound local Nemotron embedding backend."""

    provider = "nemotron"
    location = RouteLocation.LOCAL

    def __init__(
        self,
        config: SearchEmbeddingConfig,
        *,
        model: str,
        device: str,
        incremental: bool = False,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device
        self.incremental = incremental
        self._model: _NemotronModel | None = None
        self._load_lock = threading.Lock()

    def _load(self) -> _NemotronModel:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                import torch  # type: ignore[import-not-found, unused-ignore]
                from sentence_transformers import (  # type: ignore[import-not-found, unused-ignore]
                    SentenceTransformer,
                )
            except ImportError as exc:
                raise SemanticModelError(
                    "install the 'semantic' extra to run Nemotron retrieval"
                ) from exc

            kwargs: dict[str, object] = {
                "device": self.device,
                "model_kwargs": {
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "sdpa",
                },
            }
            if self.config.offline:
                kwargs["local_files_only"] = True
            try:
                self._model = cast(
                    _NemotronModel,
                    SentenceTransformer(
                        str(_local_snapshot(self.config, self.model)), **kwargs
                    ),
                )
            except TypeError:
                kwargs["model_kwargs"] = {
                    "dtype": torch.bfloat16,
                    "attn_implementation": "sdpa",
                }
                self._model = cast(
                    _NemotronModel,
                    SentenceTransformer(
                        str(_local_snapshot(self.config, self.model)), **kwargs
                    ),
                )
        return self._model

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult:
        if model != self.model:
            raise SafeBackendError("route_configuration_invalid")
        if not request.texts:
            return EmbeddingResult((), self.provider, model)
        encoder = self._load()
        batch_size = (
            self.config.foreground_max_batch
            if request.purpose is EmbeddingPurpose.QUERY
            else (
                self.config.incremental_max_batch
                if self.incremental
                else self.config.maintenance_max_batch
            )
        )
        options = {
            "batch_size": max(1, batch_size),
            "convert_to_numpy": True,
            "show_progress_bar": False,
            "normalize_embeddings": True,
        }
        encoded = (
            encoder.encode_query(list(request.texts), **options)
            if request.purpose is EmbeddingPurpose.QUERY
            else encoder.encode_document(list(request.texts), **options)
        )
        vectors = normalize_embeddings(
            encoded,
            self.config.dimensions,
        )
        return EmbeddingResult(
            tuple(tuple(float(value) for value in vector) for vector in vectors),
            self.provider,
            model,
        )

    def close(self) -> None:
        with self._load_lock:
            self._model = None
        gc.collect()
        try:
            import torch  # type: ignore[import-not-found, unused-ignore]

            if self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
