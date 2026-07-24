
import numpy as np
import pytest

from chronovisor.semantic_service import (
    QueryBatcher,
    ServiceBusy,
    _drifted_page_ids,
)


def test_drifted_page_ids_are_deduplicated_across_states() -> None:
    assert _drifted_page_ids(
        {
            "missing_page_ids": ["new", "shared"],
            "stale_page_ids": ["changed", "shared"],
            "deleted_page_ids": ["gone"],
        }
    ) == ["changed", "gone", "new", "shared"]


def test_query_batcher_combines_concurrent_requests() -> None:
    calls: list[list[str]] = []

    def encode(texts: list[str], _batch_size: int) -> np.ndarray:
        calls.append(texts)
        return np.asarray([[float(len(text)), 1.0] for text in texts])

    batcher = QueryBatcher(
        encode=encode,
        search=lambda vector, _top_n: [("page", float(vector[0]))],
        window_ms=5,
        max_batch=8,
        available=lambda: True,
    )
    try:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(batcher.submit, f"q{index}", 1, 1.0)
                for index in range(4)
            ]
            assert all(future.result()[0][0] == "page" for future in futures)
        assert len(calls) == 1
        assert len(calls[0]) == 4
    finally:
        batcher.close()


def test_query_batcher_rejects_when_generation_is_unavailable() -> None:
    batcher = QueryBatcher(
        encode=lambda texts, _batch: np.zeros((len(texts), 2)),
        search=lambda _vector, _top_n: [],
        window_ms=0,
        max_batch=1,
        available=lambda: False,
    )
    try:
        with pytest.raises(ServiceBusy):
            batcher.submit("query", 1, 1.0)
    finally:
        batcher.close()
