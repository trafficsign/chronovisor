"""Bounded, single-teacher execution for claimed distillation work.

The dispatcher owns only remote/model calls.  Callers receive ordered results
and remain the sole writer for SQLite and ledgers.
"""

from __future__ import annotations

import random as _random
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

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
    "kill_switch",
}


class DispatchFailure(RuntimeError):
    """An evaluator failure with a stable category for retry/stop policy."""

    def __init__(
        self,
        category: str,
        message: str | None = None,
        *,
        status_code: int | None = None,
    ) -> None:
        self.category = _normalize_category(category)
        self.status_code = status_code
        super().__init__(message or self.category)


class DispatchStopped(RuntimeError):
    """A work item was not evaluated because the cohort was stopped."""

    def __init__(self, category: str = "kill_switch") -> None:
        self.category = _normalize_category(category)
        super().__init__(self.category)


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
class _Attempt:
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


def _result_category(value: object) -> str | None:
    """Read the redacted failure envelope used by teacher adapters."""
    if not isinstance(value, Mapping):
        return None
    failure = value.get("_failure")
    if not isinstance(failure, Mapping):
        return None
    category = failure.get("class")
    if not isinstance(category, str) or not category:
        return "error"
    normalized = _normalize_category(category)
    if normalized == "remote_teacher_disabled":
        return "kill_switch"
    return normalized


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
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        random_fn: Callable[[], float] = _random.random,
    ) -> None:
        if not 1 <= max_inflight <= 10:
            raise ValueError("max_inflight must be between 1 and 10")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if backoff_base_seconds < 0 or backoff_max_seconds < 0:
            raise ValueError("backoff values must be non-negative")
        if jitter_ratio < 0:
            raise ValueError("jitter_ratio must be non-negative")
        if min_valid_results_per_cap < 1:
            raise ValueError("min_valid_results_per_cap must be positive")
        self.evaluate = evaluate
        self.max_inflight = max_inflight
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = max(backoff_base_seconds, backoff_max_seconds)
        self.jitter_ratio = jitter_ratio
        self.min_valid_results_per_cap = min_valid_results_per_cap
        self.sleep = sleep
        self.clock = clock
        self.random_fn = random_fn

    def dispatch(self, claimed_work: Sequence[T]) -> list[DispatchResult[T, R]]:
        """Evaluate all work in input order; never writes caller-owned state."""
        work = list(claimed_work)
        if not work:
            return []
        results: list[DispatchResult[T, R] | None] = [None] * len(work)
        stop_event = threading.Event()
        stop_reason: list[str] = []
        stop_reason_lock = threading.Lock()
        current_cap = 1
        valid_results_at_cap = 0
        offset = 0
        with ThreadPoolExecutor(max_workers=self.max_inflight) as executor:
            while offset < len(work) and not stop_event.is_set():
                cap = min(current_cap, self.max_inflight, len(work) - offset)
                indexes = range(offset, offset + cap)
                futures: dict[Future[_Attempt], int] = {
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
                wave_results: list[DispatchResult[T, R]] = []
                for future, index in futures.items():
                    attempt = future.result()
                    results[index] = attempt.result
                    wave_results.append(attempt.result)
                offset += cap

                if stop_event.is_set():
                    break
                if any(item.rate_limited for item in wave_results):
                    next_cap = max(1, current_cap // 2)
                    current_cap = next_cap
                    valid_results_at_cap = 0
                else:
                    valid_results_at_cap += sum(
                        item.status == "ok" for item in wave_results
                    )
                    if valid_results_at_cap >= self.min_valid_results_per_cap:
                        next_cap = self._next_cap(current_cap)
                        if next_cap != current_cap:
                            current_cap = next_cap
                            valid_results_at_cap = 0

            if stop_event.is_set():
                stop_category = stop_reason[0] if stop_reason else next(
                    (
                        result.category
                        for result in results
                        if result is not None and result.status == "stopped"
                    ),
                    "kill_switch",
                )
                stopped = DispatchStopped(stop_category)
                for index, item in enumerate(results):
                    if item is None:
                        results[index] = DispatchResult(
                            work[index],
                            "deferred",
                            error=stopped,
                            category=stop_category,
                            attempts=0,
                        )

        return [item for item in results if item is not None]

    def _evaluate_with_retry(
        self,
        index: int,
        work: T,
        stop_event: threading.Event,
        stop_reason: list[str],
        stop_reason_lock: threading.Lock,
    ) -> _Attempt:
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
                value = self.evaluate(work)
                failure_category = _result_category(value)
                if failure_category is not None:
                    raise DispatchFailure(failure_category)
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
        delay = min(
            self.backoff_max_seconds,
            self.backoff_base_seconds * (2 ** (attempt_number - 1)),
        )
        if not delay or not self.jitter_ratio:
            return delay
        jitter = (self.random_fn() * 2.0 - 1.0) * self.jitter_ratio
        return max(0.0, min(self.backoff_max_seconds, delay * (1.0 + jitter)))

    def _next_cap(self, current_cap: int) -> int:
        for cap in _RAMP:
            if cap > current_cap:
                return min(cap, self.max_inflight)
        return min(current_cap, self.max_inflight)


def dispatch_claimed_work(
    claimed_work: Sequence[T],
    evaluate: Callable[[T], R],
    **kwargs: Any,
) -> list[DispatchResult[T, R]]:
    """Convenience wrapper for a one-shot single-teacher dispatch."""
    return SingleTeacherDispatcher(evaluate, **kwargs).dispatch(claimed_work)


__all__ = [
    "DispatchFailure",
    "DispatchResult",
    "DispatchStopped",
    "SingleTeacherDispatcher",
    "dispatch_claimed_work",
]
