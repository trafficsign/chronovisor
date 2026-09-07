from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from chronovisor.core.llm_runtime import (
    EmbeddingPurpose,
    EmbeddingRequest,
    SafeBackendError,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.nemotron_adapter import SemanticModelError
from chronovisor.core.nemotron_mlx import (
    MAX_TOKENS,
    MLX_MODEL_REPOSITORY,
    MLX_MODEL_REVISION,
    RUNTIME_MODEL,
    RUNTIME_REVISION,
    NemotronMLXBackend,
)
from chronovisor.core.runtime_config import SearchEmbeddingConfig

SOURCE = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)


class FakeTokenizer:
    def __init__(self, *, token_count: int | None = None) -> None:
        self.padding_side = "left"
        self.calls: list[dict[str, object]] = []
        self.token_count = token_count

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, np.ndarray]:
        self.calls.append({"texts": texts, **kwargs})
        count = self.token_count
        lengths = [count or max(1, len(text.split())) for text in texts]
        width = max(lengths)
        ids = np.zeros((len(texts), width), dtype=np.int32)
        mask = np.zeros_like(ids)
        for row, length in enumerate(lengths):
            ids[row, :length] = row + 1
            mask[row, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


class FakeEncoder:
    def __init__(self, *, block: threading.Event | None = None) -> None:
        self.calls: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.block = block

    def __call__(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        if self.block is not None:
            self.block.wait(timeout=2)
        self.calls.append((threading.get_ident(), input_ids, attention_mask))
        return np.tile(np.array([[3.0, 4.0]], dtype=np.float32), (len(input_ids), 1))


def _config(**kwargs: object) -> SearchEmbeddingConfig:
    return SearchEmbeddingConfig(dimensions=2, **kwargs)


def test_pinned_runtime_identity_is_distinct_from_mlx_conversion() -> None:
    assert RUNTIME_MODEL == "nvidia/Nemotron-3-Embed-1B-BF16"
    assert RUNTIME_REVISION == "a5e0f804b9e90a1ca6784ecbf6e41595774fc834"
    assert MLX_MODEL_REPOSITORY == "mlx-community/Nemotron-3-Embed-1B-BF16"
    assert MLX_MODEL_REVISION == "775be489a79fb17d75d27050b51ccd2298048f16"


def test_mlx_backend_loads_and_infers_on_one_dedicated_thread(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    encoder = FakeEncoder()
    loader_threads: list[int] = []

    def loader(_snapshot: Path) -> tuple[FakeEncoder, FakeTokenizer]:
        loader_threads.append(threading.get_ident())
        return encoder, tokenizer

    backend = NemotronMLXBackend(
        _config(query_prefix="Q: ", document_prefix="D: ", foreground_max_batch=2),
        model=RUNTIME_MODEL,
        snapshot=tmp_path,
        loader=loader,
    )
    caller_thread = threading.get_ident()
    result = backend.embed(
        EmbeddingRequest(
            ("one", "two", "three"),
            SOURCE,
            purpose=EmbeddingPurpose.QUERY,
        ),
        model=RUNTIME_MODEL,
    )
    backend.close()

    assert len(result.vectors) == 3
    np.testing.assert_allclose(result.vectors, [[0.6, 0.8]] * 3)
    assert loader_threads and loader_threads[0] != caller_thread
    assert {thread_id for thread_id, _, _ in encoder.calls} == set(loader_threads)
    assert [call["texts"] for call in tokenizer.calls] == [
        ["Q: one", "Q: two"],
        ["Q: three"],
    ]
    assert all(call["padding"] is True for call in tokenizer.calls)
    assert all(call["truncation"] is False for call in tokenizer.calls)
    assert all(call["return_tensors"] == "np" for call in tokenizer.calls)
    assert tokenizer.padding_side == "right"


def test_mlx_backend_rejects_silent_truncation(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer(token_count=MAX_TOKENS + 1)
    encoder = FakeEncoder()
    backend = NemotronMLXBackend(
        _config(),
        model=RUNTIME_MODEL,
        snapshot=tmp_path,
        loader=lambda _snapshot: (encoder, tokenizer),
    )

    with pytest.raises(SemanticModelError, match="32768"):
        backend.embed(EmbeddingRequest(("long",), SOURCE), model=RUNTIME_MODEL)
    backend.close()
    assert encoder.calls == []


def test_mlx_backend_timeout_cancels_a_queued_request(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingEncoder(FakeEncoder):
        def __call__(
            self, input_ids: np.ndarray, attention_mask: np.ndarray
        ) -> np.ndarray:
            started.set()
            release.wait(timeout=2)
            return super().__call__(input_ids, attention_mask)

    encoder = BlockingEncoder()
    tokenizer = FakeTokenizer()
    backend = NemotronMLXBackend(
        _config(),
        model=RUNTIME_MODEL,
        snapshot=tmp_path,
        loader=lambda _snapshot: (encoder, tokenizer),
    )

    first_error: list[BaseException] = []

    def first() -> None:
        try:
            backend.embed(
                EmbeddingRequest(("first",), SOURCE),
                model=RUNTIME_MODEL,
            )
        except BaseException as exc:  # pragma: no cover - assertion below
            first_error.append(exc)

    first_thread = threading.Thread(target=first)
    first_thread.start()
    assert started.wait(timeout=1)
    with pytest.raises(SafeBackendError, match="timeout"):
        backend.embed(
            EmbeddingRequest(("queued",), SOURCE, timeout_ms=1),
            model=RUNTIME_MODEL,
        )
    release.set()
    first_thread.join(timeout=2)
    backend.close()

    assert first_error == []
    assert len(encoder.calls) == 1


def test_mlx_backend_rejects_wrong_route_and_close_is_idempotent(
    tmp_path: Path,
) -> None:
    backend = NemotronMLXBackend(
        _config(),
        model=RUNTIME_MODEL,
        snapshot=tmp_path,
        loader=lambda _snapshot: (FakeEncoder(), FakeTokenizer()),
    )
    with pytest.raises(SafeBackendError, match="route_configuration_invalid"):
        backend.embed(EmbeddingRequest(("text",), SOURCE), model="other-model")
    backend.close()
    backend.close()
    with pytest.raises(SemanticModelError, match="closed"):
        backend.embed(EmbeddingRequest(("text",), SOURCE), model=RUNTIME_MODEL)
