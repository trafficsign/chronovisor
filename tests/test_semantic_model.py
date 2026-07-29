import numpy as np
import pytest

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
