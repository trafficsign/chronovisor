"""Native MLX backend for the pinned Nemotron embedding model.

The MLX stream is thread-local.  A single worker therefore owns both the
model and every forward pass; callers only submit bounded jobs to it.
"""

from __future__ import annotations

import concurrent.futures
import gc
import importlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from chronovisor.core.llm_runtime import (
    EmbeddingPurpose,
    EmbeddingRequest,
    EmbeddingResult,
    RouteLocation,
    SafeBackendError,
)
from chronovisor.core.nemotron_adapter import (
    SemanticModelError,
    normalize_embeddings,
)

# The route keeps the NVIDIA model identity so existing semantic indexes remain
# attributable to the original model.  These constants identify the reviewed
# MLX conversion and its immutable local cache entry.
RUNTIME_MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
RUNTIME_REVISION = "a5e0f804b9e90a1ca6784ecbf6e41595774fc834"
MLX_MODEL_REPOSITORY = "mlx-community/Nemotron-3-Embed-1B-BF16"
MLX_MODEL_REVISION = "775be489a79fb17d75d27050b51ccd2298048f16"
MLX_MODEL_CACHE_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--Nemotron-3-Embed-1B-BF16"
)
MLX_SNAPSHOT_PATH = MLX_MODEL_CACHE_DIR / "snapshots" / MLX_MODEL_REVISION
MAX_TOKENS = 32_768
_NEG_INF = -1e9


class _Encoder(Protocol):
    def __call__(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> object: ...


class _Tokenizer(Protocol):
    padding_side: str

    def __call__(
        self, texts: Sequence[str], **kwargs: object
    ) -> Mapping[str, object]: ...


EncoderLoader = Callable[[Path], tuple[_Encoder, _Tokenizer]]


def mlx_snapshot_path() -> Path:
    """Return the pinned MLX snapshot path used by the production loader."""

    # Resolve at call time so tests and launchd jobs honour their active HOME.
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--mlx-community--Nemotron-3-Embed-1B-BF16"
        / "snapshots"
        / MLX_MODEL_REVISION
    )


class _MLXEncoder:
    """Convert ordinary NumPy batches to MLX arrays on the owner thread."""

    def __init__(self, model: Any, mx: Any) -> None:
        self._model = model
        self._mx = mx

    def __call__(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> object:
        mx = self._mx
        return self._model(mx.array(input_ids), mx.array(attention_mask))


def _load_encoder(snapshot: Path) -> tuple[_Encoder, _Tokenizer]:
    """Load the bidirectional encoder and tokenizer on the worker thread."""

    try:
        import mlx.core as mx
        from mlx_lm.models.ministral3 import (
            ModelArgs,
            TransformerBlock,
            _get_llama_4_attn_scale,
        )
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SemanticModelError(
            "install mlx, mlx-lm, and transformers to run Nemotron MLX"
        ) from exc

    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise SemanticModelError(f"pinned MLX model snapshot is missing: {snapshot}")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    nn = cast(Any, importlib.import_module("mlx.nn"))

    class NemotronEmbedModel(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self, args: Any) -> None:
            super().__init__()
            self.args = args
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
            self.layers = [
                TransformerBlock(args) for _ in range(args.num_hidden_layers)
            ]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, input_ids: Any, attention_mask: Any) -> Any:
            hidden = self.embed_tokens(input_ids)
            attn_scale = _get_llama_4_attn_scale(
                input_ids.shape[1],
                0,
                self.args.rope_parameters["llama_4_scaling_beta"],
                self.args.rope_parameters["original_max_position_embeddings"],
            ).astype(hidden.dtype)
            padding = (1 - attention_mask[:, None, None, :]).astype(
                hidden.dtype
            ) * _NEG_INF
            for layer in self.layers:
                hidden = layer(hidden, attn_scale, mask=padding)
            hidden = self.norm(hidden).astype(mx.float32)
            mask = attention_mask[:, :, None].astype(mx.float32)
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            return pooled / mx.linalg.norm(pooled, axis=-1, keepdims=True)

    model = NemotronEmbedModel(ModelArgs.from_dict(config))  # type: ignore[no-untyped-call]
    if "quantization" in config:
        raise SemanticModelError("pinned Nemotron MLX backend requires BF16 weights")
    model.load_weights(str(weights_path))
    model.eval()
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    tokenizer.padding_side = "right"
    return _MLXEncoder(model, mx), tokenizer


class NemotronMLXBackend:
    """Lazy, single-worker MLX embedding backend."""

    provider = "nemotron"
    location = RouteLocation.LOCAL
    device = "mlx"

    def __init__(
        self,
        config: Any,
        *,
        model: str = RUNTIME_MODEL,
        incremental: bool = False,
        snapshot: Path | str | None = None,
        loader: EncoderLoader | None = None,
    ) -> None:
        if model != RUNTIME_MODEL:
            raise SemanticModelError(
                f"Nemotron MLX backend only supports the pinned model: {RUNTIME_MODEL}"
            )
        if getattr(config, "revision", RUNTIME_REVISION) != RUNTIME_REVISION:
            raise SemanticModelError(
                f"Nemotron MLX backend requires revision {RUNTIME_REVISION}"
            )
        self.config = config
        self.model = model
        self.incremental = incremental
        self.snapshot = (
            Path(snapshot).expanduser() if snapshot is not None else mlx_snapshot_path()
        )
        self._loader = loader or _load_encoder
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()
        self._encoder: _Encoder | None = None
        self._tokenizer: _Tokenizer | None = None
        self._worker_thread_id: int | None = None
        self._closed = False

    def _executor_or_create(self) -> concurrent.futures.ThreadPoolExecutor:
        with self._executor_lock:
            if self._closed:
                raise SemanticModelError("Nemotron MLX backend is closed")
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="chronovisor-nemotron-mlx",
                )
            return self._executor

    def _load_on_worker(self) -> tuple[_Encoder, _Tokenizer]:
        current_thread = threading.get_ident()
        if (
            self._worker_thread_id is not None
            and self._worker_thread_id != current_thread
        ):
            raise RuntimeError("Nemotron MLX model is owned by another thread")
        self._worker_thread_id = current_thread
        if self._encoder is None or self._tokenizer is None:
            encoder, tokenizer = self._loader(self.snapshot)
            tokenizer.padding_side = "right"
            self._encoder = encoder
            self._tokenizer = tokenizer
        return self._encoder, self._tokenizer

    def _encode_on_worker(
        self,
        texts: Sequence[str],
        purpose: EmbeddingPurpose,
    ) -> np.ndarray:
        encoder, tokenizer = self._load_on_worker()
        batch_size = (
            int(self.config.foreground_max_batch)
            if purpose is EmbeddingPurpose.QUERY
            else (
                int(self.config.incremental_max_batch)
                if self.incremental
                else int(self.config.maintenance_max_batch)
            )
        )
        batch_size = max(1, batch_size)
        prefix = (
            str(self.config.query_prefix)
            if purpose is EmbeddingPurpose.QUERY
            else str(self.config.document_prefix)
        )
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [prefix + text for text in texts[start : start + batch_size]]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=False,
                return_tensors="np",
            )
            input_ids = np.asarray(inputs["input_ids"])
            attention_mask = np.asarray(inputs["attention_mask"])
            if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise SemanticModelError("tokenizer returned an invalid batch")
            if input_ids.shape[1] > MAX_TOKENS:
                raise SemanticModelError(
                    f"input exceeds {MAX_TOKENS} tokens; refusing silent truncation"
                )
            output = encoder(input_ids, attention_mask)
            if output.__class__.__module__.startswith("mlx"):
                import mlx.core as mx

                mlx_output = cast(Any, output)
                mx.eval(mlx_output)
                mlx_output = mlx_output.astype(mx.float32)
                mx.eval(mlx_output)
                mx.clear_cache()
            batch_vectors = np.asarray(output, dtype=np.float32)
            if batch_vectors.ndim == 1:
                batch_vectors = batch_vectors.reshape(1, -1)
            if batch_vectors.ndim != 2 or batch_vectors.shape[0] != len(batch):
                raise SemanticModelError("encoder returned an invalid batch")
            vectors.append(batch_vectors)
        return normalize_embeddings(np.vstack(vectors), int(self.config.dimensions))

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult:
        if model != self.model:
            raise SafeBackendError("route_configuration_invalid")
        if not request.texts:
            return EmbeddingResult((), self.provider, model)
        executor = self._executor_or_create()
        future = executor.submit(self._encode_on_worker, request.texts, request.purpose)
        try:
            result = future.result(
                timeout=(request.timeout_ms / 1_000)
                if request.timeout_ms is not None
                else None
            )
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise SafeBackendError("timeout", transient=True) from None
        return EmbeddingResult(
            tuple(tuple(float(value) for value in vector) for vector in result),
            self.provider,
            model,
        )

    def close(self) -> None:
        with self._executor_lock:
            if self._closed:
                return
            self._closed = True
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        self._encoder = None
        self._tokenizer = None
        self._worker_thread_id = None
        gc.collect()


__all__ = [
    "MAX_TOKENS",
    "MLX_MODEL_CACHE_DIR",
    "MLX_MODEL_REPOSITORY",
    "MLX_MODEL_REVISION",
    "MLX_SNAPSHOT_PATH",
    "NemotronMLXBackend",
    "RUNTIME_MODEL",
    "RUNTIME_REVISION",
    "mlx_snapshot_path",
]
