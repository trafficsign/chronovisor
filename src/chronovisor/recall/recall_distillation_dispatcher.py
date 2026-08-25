"""Bounded, single-teacher execution for claimed distillation work.

The dispatcher owns only remote/model calls.  Callers receive ordered results
and remain the sole writer for SQLite and ledgers.
"""

from __future__ import annotations

import random as _random
import re
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast

from chronovisor.core.llm_runtime import safe_metadata_identifier

T = TypeVar("T")
R = TypeVar("R")
Status = Literal["ok", "failed", "stopped", "deferred"]

_RAMP = (1, 2, 5, 10)
_RETRYABLE = {"http_429", "http_5xx", "timeout"}
_STOPPED = {
    "http_402",
    "payment_required",
    "paid_fallback",
    "model_unavailable",
    "route_model_drift",
    "kill_switch",
    "ox_guard_denied",
}
_FAILURE_STAGES = frozenset(
    {
        "transport_response",
        "http_json",
        "choices_shape",
        "choice_shape",
        "message_shape",
        "content_shape",
        "finish_reason",
        "usage_shape",
        "usage_tokens",
        "teacher_json_parse",
        "teacher_response_shape",
        "teacher_label_count",
        "teacher_label_schema",
    }
)


def _safe_stage(value: object) -> str | None:
    return value if isinstance(value, str) and value in _FAILURE_STAGES else None


class DispatchFailure(RuntimeError):
    """An evaluator failure with a stable category for retry/stop policy."""

    def __init__(
        self,
        category: str,
        message: str | None = None,
        *,
        status_code: int | None = None,
        stage: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.category = _normalize_category(category)
        self.status_code = status_code
        self.stage = _safe_stage(stage)
        self.request_id = safe_metadata_identifier(request_id)
        super().__init__(message or self.category)


class DispatchStopped(RuntimeError):
    """A work item was not evaluated because the cohort was stopped."""

    def __init__(self, category: str = "kill_switch") -> None:
        self.category = _normalize_category(category)
        super().__init__(self.category)


class DispatchGuardDenied(DispatchStopped):
    """The egress guard denied an attempt before provider work began."""

    def __init__(self, message: str | None = None) -> None:
        self.category = "ox_guard_denied"
        RuntimeError.__init__(self, message or self.category)


@dataclass(frozen=True)
class DispatchResult(Generic[T, R]):
    """One ordered result returned to the caller's single-writer boundary."""

    work: T
    status: Status
    value: R | None = None
    error: BaseException | None = None
    category: str | None = None
    attempts: int = 0
    rate_limited: bool = False


@dataclass(frozen=True)
class _Attempt(Generic[T, R]):
    result: DispatchResult[T, R]


def _normalize_category(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    aliases = {
        "429": "http_429",
        "http429": "http_429",
        "rate_limited": "http_429",
        "rate_limit": "http_429",
        "too_many_requests": "http_429",
        "5xx": "http_5xx",
        "http5xx": "http_5xx",
        "server_error": "http_5xx",
        "timeout_error": "timeout",
        "402": "http_402",
        "http402": "http_402",
        "payment_required": "http_402",
        "paid": "paid_fallback",
        "paid_fallback_detected": "paid_fallback",
        "model_unavailable_error": "model_unavailable",
        "kill_switch_triggered": "kill_switch",
    }
    return aliases.get(text, text)


def _status_code(error: BaseException) -> int | None:
    for candidate in (
        getattr(error, "status_code", None),
        getattr(error, "status", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int):
            return candidate
    return None


def _category(error: BaseException) -> str:
    if isinstance(error, DispatchFailure):
        return error.category
    if isinstance(error, DispatchStopped):
        return error.category
    status = _status_code(error)
    if status == 429:
        return "http_429"
    if status == 402:
        return "http_402"
    if status is not None and 500 <= status <= 599:
        return "http_5xx"
    if isinstance(error, TimeoutError):
        return "timeout"
    for value in (
        getattr(error, "category", None),
        getattr(error, "safe_category", None),
        str(error),
    ):
        normalized = _normalize_category(value)
        if normalized in _RETRYABLE or normalized in _STOPPED:
            return normalized
        if "model_unavailable" in normalized:
            return "model_unavailable"
        if "paid_fallback" in normalized or "paid_route" in normalized:
            return "paid_fallback"
        if "kill_switch" in normalized:
            return "kill_switch"
    return "error"


def _result_failure(value: object) -> tuple[str, str | None, str | None] | None:
    """Read the redacted failure envelope used by teacher adapters."""
    if not isinstance(value, Mapping):
        return None
    failure = value.get("_failure")
    if not isinstance(failure, Mapping):
        return None
    category = failure.get("class")
    if not isinstance(category, str) or not category:
        return "error", None, None
    normalized = _normalize_category(category)
    if normalized == "remote_teacher_disabled":
        normalized = "kill_switch"
    stage = failure.get("stage")
    request_id = failure.get("request_id")
    return (
        normalized,
        _safe_stage(stage),
        safe_metadata_identifier(request_id),
    )


class SingleTeacherDispatcher(Generic[T, R]):
    """Run claimed work with bounded parallelism and a conservative ramp."""

    def __init__(
        self,
        evaluate: Callable[[T], R],
        *,
        max_inflight: int = 10,
        max_retries: int = 2,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 30.0,
        jitter_ratio: float = 0.1,
        min_valid_results_per_cap: int = 20,
        initial_cap: int = 1,
        initial_valid_results: int = 0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        random_fn: Callable[[], float] = _random.random,
        valid_result_count: Callable[[R], int] | None = None,
        before_attempt: Callable[[], None] | None = None,
    ) -> None:
        if not 1 <= max_inflight <= 10:
            raise ValueError("max_inflight must be between 1 and 10")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if backoff_base_seconds < 0 or backoff_max_seconds < 0:
            raise ValueError("backoff values must be non-negative")
        if jitter_ratio < 0:
            raise ValueError("jitter_ratio must be non-negative")
        if before_attempt is not None and not callable(before_attempt):
            raise ValueError("before_attempt must be callable or None")
        if min_valid_results_per_cap < 1:
            raise ValueError("min_valid_results_per_cap must be positive")
        if (
            isinstance(initial_cap, bool)
            or not isinstance(initial_cap, int)
            or not 1 <= initial_cap <= max_inflight
        ):
            raise ValueError("initial_cap must be between 1 and max_inflight")
        if (
            isinstance(initial_valid_results, bool)
            or not isinstance(initial_valid_results, int)
            or initial_valid_results < 0
            or initial_valid_results > min_valid_results_per_cap
            or (
                initial_valid_results == min_valid_results_per_cap
                and initial_cap < max_inflight
            )
        ):
            raise ValueError("initial_valid_results is outside the current ramp stage")
        self.evaluate = evaluate
        self.max_inflight = max_inflight
        self.max_retries = max_retries
        self.backoff_base_seconds: float = float(backoff_base_seconds)
        self.backoff_max_seconds: float = float(
            max(backoff_base_seconds, backoff_max_seconds)
        )
        self.jitter_ratio: float = float(jitter_ratio)
        self.min_valid_results_per_cap = min_valid_results_per_cap
        self.current_cap = initial_cap
        self.valid_results_at_cap = initial_valid_results
        self.sleep = sleep
        self.clock = clock
        self.random_fn: Callable[[], float] = random_fn
        self.valid_result_count = valid_result_count or (lambda _value: 1)
        self.before_attempt = before_attempt

    def dispatch(self, claimed_work: Sequence[T]) -> list[DispatchResult[T, R]]:
        """Evaluate all work in input order; never writes caller-owned state."""
        work = list(claimed_work)
        if not work:
            return []
        results: list[DispatchResult[T, R] | None] = [None] * len(work)
        stop_event = threading.Event()
        stop_reason: list[str] = []
        stop_reason_lock = threading.Lock()
        current_cap = self.current_cap
        valid_results_at_cap = self.valid_results_at_cap
        offset = 0
        executor = ThreadPoolExecutor(max_workers=self.max_inflight)
        futures: dict[Future[_Attempt[T, R]], int] = {}
        try:
            while offset < len(work) and not stop_event.is_set():
                cap = min(current_cap, self.max_inflight, len(work) - offset)
                indexes = range(offset, offset + cap)
                futures = {
                    executor.submit(
                        self._evaluate_with_retry,
                        index,
                        work[index],
                        stop_event,
                        stop_reason,
                        stop_reason_lock,
                    ): index
                    for index in indexes
                }
                wave_results: dict[int, DispatchResult[T, R]] = {}
                stop_index: int | None = None
                cancelled_category: str | None = None
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        result = future.result().result
                    except CancelledError:
                        cancelled_category = (
                            stop_reason[0] if stop_reason else "kill_switch"
                        )
                        result = DispatchResult(
                            work[index],
                            "deferred",
                            error=DispatchStopped(cancelled_category),
                            category=cancelled_category,
                            attempts=0,
                        )
                    wave_results[index] = result
                    if result.status == "stopped" and stop_index is None:
                        stop_index = index
                    if result.status == "stopped" or stop_event.is_set():
                        # A running HTTP call cannot be force-cancelled, but queued
                        # work should not start after the stop signal is observed.
                        for pending in futures:
                            if pending is not future:
                                pending.cancel()

                if stop_event.is_set():
                    stop_category = (
                        stop_reason[0]
                        if stop_reason
                        else next(
                            (
                                result.category
                                for result in wave_results.values()
                                if result.status == "stopped"
                            ),
                            cancelled_category or "kill_switch",
                        )
                    )
                    stop_category = stop_category or "kill_switch"
                    if stop_index is not None:
                        for index, result in wave_results.items():
                            if index == stop_index:
                                results[index] = DispatchResult(
                                    result.work,
                                    "stopped",
                                    error=result.error,
                                    category=stop_category,
                                    attempts=result.attempts,
                                    rate_limited=result.rate_limited,
                                )
                            else:
                                results[index] = self._defer_after_stop(
                                    result, stop_category
                                )
                    else:
                        for index, result in wave_results.items():
                            results[index] = self._defer_after_stop(
                                result, stop_category
                            )
                    offset += cap
                    break

                for index, result in wave_results.items():
                    results[index] = result
                offset += cap

                if any(item.rate_limited for item in wave_results.values()):
                    next_cap = max(1, current_cap // 2)
                    current_cap = next_cap
                    valid_results_at_cap = 0
                else:
                    for item in wave_results.values():
                        if item.status != "ok":
                            continue
                        count = self.valid_result_count(cast(R, item.value))
                        if (
                            isinstance(count, bool)
                            or not isinstance(count, int)
                            or count < 0
                        ):
                            raise ValueError(
                                "valid_result_count must return a non-negative integer"
                            )
                        valid_results_at_cap += count
                    if valid_results_at_cap >= self.min_valid_results_per_cap:
                        next_cap = self._next_cap(current_cap)
                        if next_cap != current_cap:
                            current_cap = next_cap
                            valid_results_at_cap = 0
                        else:
                            valid_results_at_cap = self.min_valid_results_per_cap

            if stop_event.is_set():
                stop_category = (
                    stop_reason[0]
                    if stop_reason
                    else next(
                        (
                            result.category
                            for result in results
                            if result is not None and result.status == "stopped"
                        ),
                        "kill_switch",
                    )
                )
                stop_category = stop_category or "kill_switch"
                stopped = DispatchStopped(stop_category)
                for index, stored in enumerate(results):
                    if stored is None:
                        results[index] = DispatchResult(
                            work[index],
                            "deferred",
                            error=stopped,
                            category=stop_category,
                            attempts=0,
                        )
        except BaseException:
            # The Recall deadline is a BaseException; do not wait for a hung call.
            stop_event.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        self.current_cap = current_cap
        self.valid_results_at_cap = valid_results_at_cap
        return [item for item in results if item is not None]

    @staticmethod
    def _defer_after_stop(
        result: DispatchResult[T, R],
        category: str,
    ) -> DispatchResult[T, R]:
        return DispatchResult(
            result.work,
            "deferred",
            error=DispatchStopped(category),
            category=category,
            attempts=result.attempts,
            rate_limited=result.rate_limited,
        )

    def _evaluate_with_retry(
        self,
        index: int,
        work: T,
        stop_event: threading.Event,
        stop_reason: list[str],
        stop_reason_lock: threading.Lock,
    ) -> _Attempt[T, R]:
        del index  # Indexing is owned by the caller; evaluation stays payload-only.
        rate_limited = False
        for attempt_number in range(1, self.max_retries + 2):
            if stop_event.is_set():
                category = stop_reason[0] if stop_reason else "kill_switch"
                return _Attempt(
                    DispatchResult(
                        work,
                        "deferred",
                        error=DispatchStopped(category),
                        category=category,
                        attempts=attempt_number - 1,
                        rate_limited=rate_limited,
                    )
                )
            try:
                if self.before_attempt is not None:
                    self.before_attempt()
            except DispatchGuardDenied as error:
                with stop_reason_lock:
                    if not stop_reason:
                        stop_reason.append(error.category)
                stop_event.set()
                return _Attempt(
                    DispatchResult(
                        work,
                        "stopped",
                        error=error,
                        category=error.category,
                        attempts=attempt_number - 1,
                        rate_limited=rate_limited,
                    )
                )
            try:
                value = self.evaluate(work)
                failure = _result_failure(value)
                if failure is not None:
                    failure_category, stage, request_id = failure
                    raise DispatchFailure(
                        failure_category,
                        stage=stage,
                        request_id=request_id,
                    )
            except DispatchGuardDenied as error:
                with stop_reason_lock:
                    if not stop_reason:
                        stop_reason.append(error.category)
                stop_event.set()
                return _Attempt(
                    DispatchResult(
                        work,
                        "stopped",
                        error=error,
                        category=error.category,
                        attempts=attempt_number - 1,
                        rate_limited=rate_limited,
                    )
                )
            except Exception as error:
                category = _category(error)
                if category in _STOPPED:
                    with stop_reason_lock:
                        if not stop_reason:
                            stop_reason.append(category)
                    stop_event.set()
                    return _Attempt(
                        DispatchResult(
                            work,
                            "stopped",
                            error=error,
                            category=category,
                            attempts=attempt_number,
                            rate_limited=rate_limited,
                        )
                    )
                if category == "http_429":
                    rate_limited = True
                if category not in _RETRYABLE or attempt_number > self.max_retries:
                    return _Attempt(
                        DispatchResult(
                            work,
                            "failed",
                            error=error,
                            category=category,
                            attempts=attempt_number,
                            rate_limited=rate_limited,
                        )
                    )
                self.sleep(self._backoff(attempt_number))
                continue
            return _Attempt(
                DispatchResult(
                    work,
                    "ok",
                    value=value,
                    attempts=attempt_number,
                    rate_limited=rate_limited,
                )
            )
        raise AssertionError("retry loop must return")

    def _backoff(self, attempt_number: int) -> float:
        delay: float = min(
            self.backoff_max_seconds,
            self.backoff_base_seconds * (2 ** (attempt_number - 1)),
        )
        if not delay or not self.jitter_ratio:
            return delay
        jitter: float = (self.random_fn() * 2.0 - 1.0) * self.jitter_ratio
        return max(0.0, min(self.backoff_max_seconds, delay * (1.0 + jitter)))

    def _next_cap(self, current_cap: int) -> int:
        for cap in _RAMP:
            if cap > current_cap:
                return min(cap, self.max_inflight)
        return min(current_cap, self.max_inflight)


def dispatch_claimed_work(
    claimed_work: Sequence[T],
    evaluate: Callable[[T], R],
    *,
    ramp_state: MutableMapping[str, int] | None = None,
    **kwargs: Any,
) -> list[DispatchResult[T, R]]:
    """Convenience wrapper for a one-shot single-teacher dispatch."""
    dispatcher = SingleTeacherDispatcher(evaluate, **kwargs)
    results = dispatcher.dispatch(claimed_work)
    if ramp_state is not None:
        ramp_state.update(
            current_cap=dispatcher.current_cap,
            valid_results_at_cap=dispatcher.valid_results_at_cap,
        )
    return results


__all__ = [
    "DispatchFailure",
    "DispatchGuardDenied",
    "DispatchResult",
    "DispatchStopped",
    "SingleTeacherDispatcher",
    "dispatch_claimed_work",
]
