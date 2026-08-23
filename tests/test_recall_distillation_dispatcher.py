from __future__ import annotations

import threading
import time

import pytest

from chronovisor.recall.recall_distillation_dispatcher import (
    DispatchFailure,
    SingleTeacherDispatcher,
    dispatch_claimed_work,
)


def test_dispatch_preserves_input_order_and_bounds_inflight() -> None:
    active = 0
    max_active = 0
    early_ramp: list[int] = []
    lock = threading.Lock()

    def evaluate(item: int) -> int:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if item < 20 and active > 1:
                early_ramp.append(item)
        time.sleep(0.001)
        with lock:
            active -= 1
        return item * 2

    results = dispatch_claimed_work(range(25), evaluate)

    assert [result.value for result in results] == [item * 2 for item in range(25)]
    assert all(result.status == "ok" for result in results)
    assert early_ramp == []
    assert max_active <= 2


def test_ramp_reaches_ten_only_after_twenty_valid_results_per_cap() -> None:
    active = 0
    max_active = 0
    early_ramp: list[int] = []
    lock = threading.Lock()

    def evaluate(item: int) -> int:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if item < 20 and active > 1:
                early_ramp.append(item)
        time.sleep(0.01)
        with lock:
            active -= 1
        return item

    results = dispatch_claimed_work(range(70), evaluate)

    assert [result.value for result in results] == list(range(70))
    assert early_ramp == []
    assert max_active == 10


def test_ramp_counts_valid_labels_inside_each_batch() -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def evaluate(item: int) -> dict[str, list[int]]:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.002)
        with lock:
            active -= 1
        return {"labels": [item] * 16}

    results = dispatch_claimed_work(
        list(range(19)),
        evaluate,
        valid_result_count=lambda value: len(value["labels"]),
    )

    assert all(result.status == "ok" for result in results)
    assert max_active == 10


def test_retry_backoff_is_bounded_and_injected() -> None:
    attempts = 0
    sleeps: list[float] = []

    def evaluate(_: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DispatchFailure("http_429")
        if attempts == 2:
            raise DispatchFailure("http_5xx")
        return "ok"

    result = SingleTeacherDispatcher(
        evaluate,
        max_retries=2,
        backoff_base_seconds=0.5,
        backoff_max_seconds=0.75,
        jitter_ratio=0,
        sleep=sleeps.append,
    ).dispatch(["work"])[0]

    assert result.status == "ok"
    assert result.value == "ok"
    assert result.attempts == 3
    assert result.rate_limited is True
    assert sleeps == [0.5, 0.75]


def test_redacted_teacher_failure_envelope_uses_same_retry_policy() -> None:
    attempts = 0

    def evaluate(_: str) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {"_failure": {"class": "http_429", "labelable": False}}
        return {"labels": []}

    result = dispatch_claimed_work(
        ["work"],
        evaluate,
        backoff_base_seconds=0,
        jitter_ratio=0,
        sleep=lambda _delay: None,
    )[0]

    assert result.status == "ok"
    assert result.attempts == 2
    assert result.rate_limited is True


def test_redacted_failure_preserves_safe_stage_without_changing_category() -> None:
    result = dispatch_claimed_work(
        ["work"],
        lambda _: {
            "_failure": {
                "class": "invalid_response",
                "stage": "teacher_json_parse",
                "request_id": "ox_req_1",
                "labelable": False,
            }
        },
    )[0]

    assert result.status == "failed"
    assert result.category == "invalid_response"
    assert isinstance(result.error, DispatchFailure)
    assert result.error.stage == "teacher_json_parse"
    assert result.error.request_id == "ox_req_1"


def test_redacted_failure_drops_unknown_stage() -> None:
    result = dispatch_claimed_work(
        ["work"],
        lambda _: {
            "_failure": {
                "class": "invalid_response",
                "stage": "secret_token",
                "labelable": False,
            }
        },
    )[0]

    assert isinstance(result.error, DispatchFailure)
    assert result.error.stage is None


def test_rate_limit_halves_future_window_after_ramp() -> None:
    calls: dict[int, int] = {}
    active = 0
    max_active = 0
    post_limit_max_active = 0
    lock = threading.Lock()

    def evaluate(item: int) -> int:
        nonlocal active, max_active, post_limit_max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if item in {9, 10}:
                post_limit_max_active = max(post_limit_max_active, active)
            calls[item] = calls.get(item, 0) + 1
            item_attempt = calls[item]
        time.sleep(0.001)
        with lock:
            active -= 1
        if item == 6 and item_attempt == 1:
            raise DispatchFailure("http_429")
        return item

    results = SingleTeacherDispatcher(
        evaluate,
        max_retries=0,
        backoff_base_seconds=0,
        jitter_ratio=0,
        sleep=lambda _delay: None,
        min_valid_results_per_cap=2,
    ).dispatch(list(range(16)))

    assert [result.value for result in results[:6]] == list(range(6))
    assert results[6].status == "failed"
    assert results[6].category == "http_429"
    assert calls[6] == 1
    assert post_limit_max_active <= 2
    assert max_active <= 5


@pytest.mark.parametrize(
    "category",
    ["http_402", "paid-fallback", "model-unavailable", "kill-switch"],
)
def test_stop_defers_unstarted_work_in_order(category: str) -> None:
    calls: list[int] = []

    def evaluate(item: int) -> int:
        calls.append(item)
        if item == 0:
            raise DispatchFailure(category)
        return item

    results = dispatch_claimed_work(
        list(range(8)),
        evaluate,
        sleep=lambda _delay: None,
    )

    assert calls == [0]
    assert [result.work for result in results] == list(range(8))
    assert results[0].status == "stopped"
    expected_category = category.replace("-", "_")
    assert results[0].category == expected_category
    assert all(result.status == "deferred" for result in results[1:])
    assert all(result.category == results[0].category for result in results[1:])
    assert all(result.attempts == 0 for result in results[1:])


def test_stop_in_cap5_defers_completed_siblings_and_cancels_tail() -> None:
    sibling_returned = threading.Event()
    calls: list[int] = []
    calls_lock = threading.Lock()

    def evaluate(item: int) -> int:
        with calls_lock:
            calls.append(item)
        if item == 3:
            assert sibling_returned.wait(1)
            raise DispatchFailure("model-unavailable")
        if item == 4:
            sibling_returned.set()
        return item

    results = dispatch_claimed_work(
        list(range(13)),
        evaluate,
        min_valid_results_per_cap=1,
        sleep=lambda _delay: None,
    )

    assert [result.work for result in results] == list(range(13))
    assert results[3].status == "stopped"
    assert results[3].category == "model_unavailable"
    assert all(result.status == "deferred" for result in results[4:])
    assert all(result.category == "model_unavailable" for result in results[4:])
    assert 4 in calls
    assert all(item < 8 for item in calls)


def test_stop_in_cap10_defers_completed_siblings_and_preserves_order() -> None:
    sibling_returned = threading.Event()
    calls: list[int] = []
    calls_lock = threading.Lock()

    def evaluate(item: int) -> int:
        with calls_lock:
            calls.append(item)
        if item == 8:
            assert sibling_returned.wait(1)
            raise DispatchFailure("paid-fallback")
        if item == 9:
            sibling_returned.set()
        return item

    results = dispatch_claimed_work(
        list(range(18)),
        evaluate,
        min_valid_results_per_cap=1,
        sleep=lambda _delay: None,
    )

    assert [result.work for result in results] == list(range(18))
    assert results[8].status == "stopped"
    assert results[8].category == "paid_fallback"
    assert all(result.status == "deferred" for result in results[9:])
    assert all(result.category == "paid_fallback" for result in results[9:])
    assert 9 in calls
    assert all(item < 18 for item in calls)


def test_timeout_retries_only_within_bounded_attempt_budget() -> None:
    attempts = 0

    def evaluate(_: None) -> None:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("teacher timeout")

    result = dispatch_claimed_work(
        [None],
        evaluate,
        max_retries=2,
        backoff_base_seconds=0,
        sleep=lambda _delay: None,
    )[0]

    assert result.status == "failed"
    assert result.category == "timeout"
    assert result.attempts == 3
    assert attempts == 3
