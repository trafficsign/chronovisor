import sys
from types import SimpleNamespace

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
from chronovisor.core.nemotron_adapter import NemotronEmbeddingBackend
from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.search.semantic_model import SemanticModelError, _normalized


def test_normalized_returns_contiguous_float32_unit_vectors() -> None:
    result = _normalized([[3.0, 4.0], [0.0, 2.0]], 2)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0, 1.0])


def test_normalized_rejects_bad_shapes_and_zero_vectors() -> None:
    with pytest.raises(SemanticModelError):
        _normalized([[1.0, 2.0]], 3)
    with pytest.raises(SemanticModelError):
        _normalized([[0.0, 0.0]], 2)


def test_nemotron_backend_selects_query_or_document_without_eager_loading() -> None:
    calls: list[str] = []

    class Model:
        def encode_query(self, _texts: list[str], **_kwargs: object) -> object:
            calls.append("query")
            return [[3.0, 4.0]]

        def encode_document(self, _texts: list[str], **_kwargs: object) -> object:
            calls.append("document")
            return [[0.0, 2.0]]

    config = SearchEmbeddingConfig(dimensions=2)
    backend = NemotronEmbeddingBackend(config, model="test-model", device="mps")
    assert backend._model is None
    backend._model = Model()
    source = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)

    query = backend.embed(
        EmbeddingRequest(("q",), source, purpose=EmbeddingPurpose.QUERY),
        model="test-model",
    )
    document = backend.embed(EmbeddingRequest(("d",), source), model="test-model")

    assert calls == ["query", "document"]
    np.testing.assert_allclose(query.vectors[0], [0.6, 0.8])
    np.testing.assert_allclose(document.vectors[0], [0.0, 1.0])

    with pytest.raises(SafeBackendError, match="route_configuration_invalid"):
        backend.embed(EmbeddingRequest(("d",), source), model="other-model")


def test_nemotron_mps_backend_flushes_cache_after_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases: list[bool] = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: True),
            ),
            mps=SimpleNamespace(empty_cache=lambda: releases.append(True)),
        ),
    )

    class Model:
        def encode_query(self, _texts: list[str], **_kwargs: object) -> object:
            return [[3.0, 4.0]]

        def encode_document(self, _texts: list[str], **_kwargs: object) -> object:
            raise RuntimeError("encode failed")

    backend = NemotronEmbeddingBackend(
        SearchEmbeddingConfig(dimensions=2), model="test-model", device="mps"
    )
    backend._model = Model()
    source = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)

    backend.embed(
        EmbeddingRequest(("q",), source, purpose=EmbeddingPurpose.QUERY),
        model="test-model",
    )
    with pytest.raises(RuntimeError, match="encode failed"):
        backend.embed(EmbeddingRequest(("d",), source), model="test-model")

    assert releases == [True, True]
