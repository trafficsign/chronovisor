from __future__ import annotations

import gc
import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

from scripts.runtime_ownership.access import (
    _AccessAnalysis,
    discover_access_facts,
)
from scripts.runtime_ownership.access_model import (
    AnalysisLimits,
    AnalysisNonConvergenceError,
    AnalysisProgress,
    _call_ordinals,
)
from scripts.runtime_ownership.access_statements import _analyze_legacy_block
from tests.runtime_access_v2_helpers import joined_access_rows, joined_escape_rows

RESOURCE_ID = "runtime-resource:state"


def _candidate(
    *,
    symbol: str = "STATE_FILE",
    resource_id: str = RESOURCE_ID,
) -> dict[str, Any]:
    return {
        "id": resource_id,
        "module": "chronovisor.state",
        "symbol": symbol,
        "locator": {"type": "path", "value": f"$ROOT/{symbol.lower()}.json"},
    }


def _snapshot(consumer: str, *, state: str = "STATE_FILE = object()\n") -> dict[str, bytes]:
    return {
        "src/chronovisor/state.py": state.encode(),
        "src/chronovisor/consumer.py": consumer.encode(),
    }


def _discover(
    consumer: str,
    *,
    limits: AnalysisLimits | None = None,
    progress: AnalysisProgress | None = None,
    optimize_gc: bool | None = False,
) -> dict[str, Any]:
    return discover_access_facts(
        _snapshot(consumer),
        [_candidate()],
        limits=limits,
        progress=progress,
        optimize_gc=optimize_gc,
    )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


@pytest.mark.parametrize(
    "field",
    [
        "max_module_export_iterations",
        "max_outer_iterations",
        "max_function_summary_iterations",
        "max_cfg_loop_iterations",
        "max_legacy_loop_iterations",
        "max_known_call_depth",
        "max_work_units",
    ],
)
@pytest.mark.parametrize("invalid", [True, 1.0, 0, -1])
def test_limits_require_positive_exact_integers(field: str, invalid: object) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a positive exact integer"):
        AnalysisLimits(**{field: invalid})  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [True, 1.0, 0, -1])
def test_progress_interval_requires_a_positive_exact_integer(invalid: object) -> None:
    with pytest.raises(
        ValueError,
        match="event_interval_work_units must be a positive exact integer",
    ):
        AnalysisProgress(event_interval_work_units=invalid)  # type: ignore[arg-type]


def test_module_export_limit_is_stable_before_limit_and_40_hops_take_43() -> None:
    sources = {"src/chronovisor/state.py": b"STATE_FILE = object()\n"}
    previous = "chronovisor.state"
    for index in range(40):
        module = f"chronovisor.hop_{index:02d}"
        sources[f"src/chronovisor/hop_{index:02d}.py"] = (
            f"from {previous} import STATE_FILE\n".encode()
        )
        previous = module
    sources["src/chronovisor/consumer.py"] = (
        f"from {previous} import STATE_FILE\n"
        "def save():\n"
        "    STATE_FILE.write_text('value')\n"
    ).encode()

    with pytest.raises(AnalysisNonConvergenceError) as raised:
        discover_access_facts(
            sources,
            [_candidate()],
            limits=replace(
                AnalysisLimits(), max_module_export_iterations=42
            ),
            optimize_gc=False,
        )
    assert raised.value.payload["phase"] == "module_exports"
    assert raised.value.payload["iteration"] == 42

    default_progress = AnalysisProgress()
    default = discover_access_facts(
        sources,
        [_candidate()],
        progress=default_progress,
        optimize_gc=False,
    )
    exact = discover_access_facts(
        sources,
        [_candidate()],
        limits=replace(AnalysisLimits(), max_module_export_iterations=43),
        optimize_gc=False,
    )
    assert default_progress.module_export_iterations == 43
    assert _canonical_bytes(exact) == _canonical_bytes(default)
    assert joined_access_rows(default)[0]["operation"] == "path.write_text"


def test_outer_limit_is_checked_only_after_nonstable_iteration() -> None:
    source = "from chronovisor.state import STATE_FILE\nSTATE_FILE.exists()\n"
    with pytest.raises(AnalysisNonConvergenceError) as raised:
        _discover(
            source,
            limits=replace(AnalysisLimits(), max_outer_iterations=1),
        )
    assert raised.value.payload["phase"] == "outer"
    assert raised.value.payload["iteration"] == 1

    progress = AnalysisProgress()
    result = _discover(
        source,
        limits=replace(AnalysisLimits(), max_outer_iterations=2),
        progress=progress,
    )
    assert progress.outer_iterations == 2
    assert joined_access_rows(result)[0]["operation"] == "path.exists"


def test_function_summary_limit_allows_convergence_exactly_at_three() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "def leaf(path):\n"
        "    path.write_text('value')\n"
        "def middle(path):\n"
        "    leaf(path)\n"
        "def top():\n"
        "    middle(STATE_FILE)\n"
    )
    with pytest.raises(AnalysisNonConvergenceError) as raised:
        _discover(
            source,
            limits=replace(
                AnalysisLimits(), max_function_summary_iterations=2
            ),
        )
    assert raised.value.payload["phase"] == "function_summary"
    assert raised.value.payload["iteration"] == 2

    result = _discover(
        source,
        limits=replace(AnalysisLimits(), max_function_summary_iterations=3),
    )
    assert joined_access_rows(result)[0]["operation"] == "path.write_text"


def test_cfg_loop_limit_allows_convergence_exactly_at_two() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "def save(items):\n"
        "    path = object()\n"
        "    for _item in items:\n"
        "        path = STATE_FILE\n"
        "    path.write_text('value')\n"
    )
    with pytest.raises(AnalysisNonConvergenceError) as raised:
        _discover(
            source,
            limits=replace(AnalysisLimits(), max_cfg_loop_iterations=1),
        )
    assert raised.value.payload["phase"] == "cfg_loop"
    assert raised.value.payload["iteration"] == 1

    result = _discover(
        source,
        limits=replace(AnalysisLimits(), max_cfg_loop_iterations=2),
    )
    assert joined_access_rows(result)[0]["operation"] == "path.write_text"


def test_legacy_loop_limit_allows_convergence_exactly_at_two() -> None:
    snapshot = _snapshot(
        "from chronovisor.state import STATE_FILE\n"
        "path = object()\n"
        "for _item in items:\n"
        "    path = STATE_FILE\n"
    )

    def analyze(cap: int) -> AnalysisProgress:
        limits = replace(AnalysisLimits(), max_legacy_loop_iterations=cap)
        progress = AnalysisProgress()
        analysis = _AccessAnalysis(
            snapshot,
            [_candidate()],
            limits=limits,
            progress=progress,
        )
        tree = analysis.trees["chronovisor.consumer"]
        _analyze_legacy_block(
            analysis,
            tree.body,
            module="chronovisor.consumer",
            actor="chronovisor.consumer:<module>",
            class_ref=None,
            env={},
            object_env={},
            call_ordinals=_call_ordinals(tree),
        )
        return progress

    with pytest.raises(AnalysisNonConvergenceError) as raised:
        analyze(1)
    assert raised.value.payload["phase"] == "legacy_loop"
    assert raised.value.payload["iteration"] == 1
    assert analyze(2).legacy_loop_iterations == 2


def test_known_call_depth_is_bounded_and_restored() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "def leaf(path):\n"
        "    path.write_text('value')\n"
        "def middle(path):\n"
        "    leaf(path)\n"
        "def top(path):\n"
        "    middle(path)\n"
        "top(STATE_FILE)\n"
    )
    progress = AnalysisProgress()
    with pytest.raises(AnalysisNonConvergenceError) as raised:
        _discover(
            source,
            limits=replace(AnalysisLimits(), max_known_call_depth=2),
            progress=progress,
        )
    assert raised.value.payload["phase"] == "known_call_depth"
    assert raised.value.payload["iteration"] == 3
    assert progress.known_call_depth == 0

    exact_progress = AnalysisProgress()
    result = _discover(
        source,
        limits=replace(AnalysisLimits(), max_known_call_depth=3),
        progress=exact_progress,
    )
    assert exact_progress.max_observed_known_call_depth == 3
    assert exact_progress.known_call_depth == 0
    assert joined_access_rows(result)[0]["operation"] == "path.write_text"


def test_global_work_limit_is_finite_and_exact_limit_succeeds() -> None:
    source = "from chronovisor.state import STATE_FILE\nSTATE_FILE.exists()\n"
    baseline_progress = AnalysisProgress()
    baseline = _discover(source, progress=baseline_progress)
    exact_work = baseline_progress.work_units

    with pytest.raises(AnalysisNonConvergenceError) as raised:
        _discover(
            source,
            limits=replace(AnalysisLimits(), max_work_units=exact_work - 1),
        )
    assert raised.value.payload["phase"] == "global_work"
    assert raised.value.payload["iteration"] == exact_work
    assert raised.value.payload["limit"] == exact_work - 1

    exact = _discover(
        source,
        limits=replace(AnalysisLimits(), max_work_units=exact_work),
    )
    assert _canonical_bytes(exact) == _canonical_bytes(baseline)


def test_nonconvergence_payload_is_canonical_and_has_only_stable_fields() -> None:
    source = "from chronovisor.state import STATE_FILE\nSTATE_FILE.exists()\n"
    payloads: list[dict[str, object]] = []
    messages: list[str] = []
    for _ in range(2):
        with pytest.raises(AnalysisNonConvergenceError) as raised:
            _discover(
                source,
                limits=replace(AnalysisLimits(), max_outer_iterations=1),
            )
        payloads.append(raised.value.payload)
        messages.append(str(raised.value))

    assert payloads[0] == payloads[1]
    assert messages[0] == messages[1]
    assert json.loads(messages[0]) == payloads[0]
    assert set(payloads[0]) == {
        "phase",
        "subject",
        "iteration",
        "limit",
        "counters",
    }


def test_progress_and_output_are_deterministic_under_reversed_inputs() -> None:
    snapshot = _snapshot(
        "from chronovisor.state import FIRST_FILE, SECOND_FILE\n"
        "FIRST_FILE.write_text('value')\n"
        "SECOND_FILE.read_text()\n",
        state="FIRST_FILE = object()\nSECOND_FILE = object()\n",
    )
    candidates = [
        _candidate(symbol="FIRST_FILE", resource_id="runtime-resource:first"),
        _candidate(symbol="SECOND_FILE", resource_id="runtime-resource:second"),
    ]
    progress_a = AnalysisProgress(event_interval_work_units=1)
    progress_b = AnalysisProgress(event_interval_work_units=1)
    first = discover_access_facts(
        snapshot,
        candidates,
        progress=progress_a,
        optimize_gc=False,
    )
    second = discover_access_facts(
        dict(reversed(list(snapshot.items()))),
        list(reversed(candidates)),
        progress=progress_b,
        optimize_gc=False,
    )

    assert progress_a.events == progress_b.events
    assert progress_a.counters() == progress_b.counters()
    assert _canonical_bytes(first) == _canonical_bytes(second)
    assert not set(first).intersection(
        {"progress", "work_units", "module_analyses", "function_analyses"}
    )


def test_two_argument_api_remains_byte_identical() -> None:
    snapshot = _snapshot(
        "from chronovisor.state import STATE_FILE\nSTATE_FILE.exists()\n"
    )
    candidates = [_candidate()]
    legacy_call = discover_access_facts(snapshot, candidates)
    explicit_call = discover_access_facts(
        snapshot,
        candidates,
        limits=AnalysisLimits(),
        progress=AnalysisProgress(),
        optimize_gc=False,
    )
    assert _canonical_bytes(legacy_call) == _canonical_bytes(explicit_call)


@pytest.mark.parametrize("invalid", [0, 1, "yes"])
def test_optimize_gc_requires_an_exact_bool(invalid: object) -> None:
    with pytest.raises(ValueError, match="optimize_gc must be an exact bool"):
        discover_access_facts({}, [], optimize_gc=invalid)  # type: ignore[arg-type]


class _ProgressCallbackFailure(RuntimeError):
    pass


def test_callback_failure_is_explicit_and_unwinds_known_call_depth() -> None:
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "def leaf(path):\n"
        "    path.write_text('value')\n"
        "def middle(path):\n"
        "    leaf(path)\n"
        "def top(path):\n"
        "    middle(path)\n"
        "top(STATE_FILE)\n"
    )

    def fail(_event: Mapping[str, object]) -> None:
        raise _ProgressCallbackFailure("observer failed")

    original_enabled = gc.isenabled()
    try:
        gc.enable()
        progress = AnalysisProgress(callback=fail, event_interval_work_units=8)
        with pytest.raises(_ProgressCallbackFailure, match="observer failed"):
            _discover(source, progress=progress, optimize_gc=True)
        assert progress.known_call_depth == 0
        assert gc.isenabled()
    finally:
        if original_enabled:
            gc.enable()
        else:
            gc.disable()


@pytest.mark.parametrize("initially_enabled", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_gc_configuration_is_restored_after_success_and_error(
    initially_enabled: bool,
    fail: bool,
) -> None:
    original_enabled = gc.isenabled()
    original_thresholds = gc.get_threshold()
    original_debug = gc.get_debug()
    original_callbacks = tuple(gc.callbacks)

    def callback(_phase: str, _info: dict[str, int]) -> None:
        return

    try:
        gc.set_threshold(1_000_003, 101, 103)
        gc.callbacks[:] = [*original_callbacks, callback]
        if initially_enabled:
            gc.enable()
        else:
            gc.disable()
        expected_thresholds = gc.get_threshold()
        expected_debug = gc.get_debug()
        expected_callbacks = tuple(gc.callbacks)
        limits = (
            replace(AnalysisLimits(), max_outer_iterations=1)
            if fail
            else AnalysisLimits()
        )

        if fail:
            with pytest.raises(AnalysisNonConvergenceError):
                _discover(
                    "from chronovisor.state import STATE_FILE\n"
                    "STATE_FILE.exists()\n",
                    limits=limits,
                    optimize_gc=True,
                )
        else:
            _discover(
                "from chronovisor.state import STATE_FILE\n"
                "STATE_FILE.exists()\n",
                limits=limits,
                optimize_gc=True,
            )

        assert gc.isenabled() is initially_enabled
        assert gc.get_threshold() == expected_thresholds
        assert gc.get_debug() == expected_debug
        assert tuple(gc.callbacks) == expected_callbacks
    finally:
        gc.set_threshold(*original_thresholds)
        gc.set_debug(original_debug)
        gc.callbacks[:] = original_callbacks
        if original_enabled:
            gc.enable()
        else:
            gc.disable()


def test_gc_optimization_is_scoped_to_run_not_analysis_construction() -> None:
    original_enabled = gc.isenabled()
    observed: list[tuple[str, bool]] = []

    def observe(event: Mapping[str, object]) -> None:
        observed.append((str(event["phase"]), gc.isenabled()))

    try:
        gc.enable()
        _discover(
            "from chronovisor.state import STATE_FILE\nSTATE_FILE.exists()\n",
            progress=AnalysisProgress(
                callback=observe, event_interval_work_units=1
            ),
            optimize_gc=True,
        )
        construction_states = [
            enabled for phase, enabled in observed if phase == "module_exports"
        ]
        run_states = [
            enabled for phase, enabled in observed if phase != "module_exports"
        ]
        assert construction_states and all(construction_states)
        assert run_states and not any(run_states)
        assert gc.isenabled()
    finally:
        if original_enabled:
            gc.enable()
        else:
            gc.disable()


def test_free_threaded_default_keeps_gc_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    original_enabled = gc.isenabled()
    observed: list[bool] = []
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: False, raising=False)

    def observe(event: Mapping[str, object]) -> None:
        if event["phase"] == "outer":
            observed.append(gc.isenabled())

    try:
        gc.enable()
        _discover(
            "from chronovisor.state import STATE_FILE\nSTATE_FILE.exists()\n",
            progress=AnalysisProgress(
                callback=observe, event_interval_work_units=1
            ),
            optimize_gc=None,
        )
        assert observed and all(observed)
        assert gc.isenabled()
    finally:
        if original_enabled:
            gc.enable()
        else:
            gc.disable()


def test_gil_enabled_cpython_default_disables_gc_during_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.implementation.name != "cpython":
        pytest.skip("automatic GC optimization is CPython-specific")
    original_enabled = gc.isenabled()
    observed: list[bool] = []
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: True, raising=False)

    def observe(event: Mapping[str, object]) -> None:
        if event["phase"] == "outer":
            observed.append(gc.isenabled())

    try:
        gc.enable()
        _discover(
            "from chronovisor.state import STATE_FILE\nSTATE_FILE.exists()\n",
            progress=AnalysisProgress(
                callback=observe, event_interval_work_units=1
            ),
            optimize_gc=None,
        )
        assert observed and not any(observed)
        assert gc.isenabled()
    finally:
        if original_enabled:
            gc.enable()
        else:
            gc.disable()


@pytest.mark.parametrize(
    "consumer",
    [
        (
            "from chronovisor.state import STATE_FILE\n"
            "def outer(path):\n"
            "    def inner():\n"
            "        path.write_text('value')\n"
            "    return inner\n"
            "writer = outer(STATE_FILE)\n"
            "writer()\n"
        ),
        (
            "from chronovisor.state import STATE_FILE\n"
            "class Writer:\n"
            "    def __init__(self, path):\n"
            "        self.path = path\n"
            "    def save(self):\n"
            "        self.path.write_text('value')\n"
            "Writer(STATE_FILE).save()\n"
        ),
        (
            "from chronovisor.state import STATE_FILE\n"
            "def save(pair):\n"
            "    path, _unused = pair\n"
            "    path.write_text('value')\n"
            "save((STATE_FILE, object()))\n"
        ),
        (
            "from chronovisor.state import STATE_FILE\n"
            "def descend(path, depth):\n"
            "    if depth:\n"
            "        return descend(path, depth - 1)\n"
            "    path.write_text('value')\n"
            "descend(STATE_FILE, 40)\n"
        ),
    ],
    ids=["closure", "class", "structured", "recursive"],
)
def test_gc_optimized_and_unoptimized_results_are_canonical_bytes(
    consumer: str,
) -> None:
    results = [
        _canonical_bytes(_discover(consumer, optimize_gc=optimize_gc))
        for optimize_gc in (False, True)
        for _ in range(3)
    ]
    assert len(set(results)) == 1


def test_65_writer_overflow_remains_canonical_at_limit_64() -> None:
    writers = "".join(
        f"def writer_{index:02d}():\n    persist(STATE_FILE)\n"
        for index in range(65)
    )
    source = (
        "from chronovisor.state import STATE_FILE\n"
        "def persist(path):\n"
        "    path.write_text('value')\n"
        f"{writers}"
    )
    unoptimized = _discover(source, optimize_gc=False)
    optimized = _discover(source, optimize_gc=True)

    assert _canonical_bytes(optimized) == _canonical_bytes(unoptimized)
    assert len(unoptimized["access_facts"]) == 1
    assert len(unoptimized["access_facts"][0]["provenance_ids"]) == 64
    assert unoptimized["access_facts"][0]["provenance_complete"] is False
    overflow = joined_escape_rows(unoptimized)[0]
    assert overflow["reason"] == "provenance_overflow"
    assert overflow["limit"] == 64
    assert overflow["retention_policy"] == "shortest_then_lexicographic"
