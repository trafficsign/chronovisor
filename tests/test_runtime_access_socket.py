from __future__ import annotations

import ast
import builtins
import contextvars
import json
import random
import sys
import threading
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from scripts.runtime_ownership import access_model
from scripts.runtime_ownership.access import discover_access_facts
from scripts.runtime_ownership.access_facts import AccessFactCollector
from scripts.runtime_ownership.access_model import (
    _MAX_FLOW_REPR_CACHE_BYTES,
    _MAX_FLOW_REPR_CACHE_ENTRIES,
    _MAX_FLOW_REPR_CACHE_WITNESSES,
    MAX_FLOW_VARIANTS,
    FlowValue,
    SyntaxSite,
    _current_flow_repr_cache,
    _flow_key_repr,
    _flow_value_key,
    _FlowReprCache,
    _normalize_flow_variants,
    _scoped_flow_repr_cache,
    _without_variants,
    candidate_flow_variants,
    exclusive_flow_join,
    simultaneous_flow_merge,
    structured_flow_value,
)
from tests.runtime_access_v2_helpers import (
    joined_escape_rows,
    validate_runtime_access_v2_result,
)

RESOURCE_ID = "runtime-resource:unix-socket"


def _reference_normalize_flow_variants(
    variants: Sequence[FlowValue],
) -> tuple[tuple[FlowValue, ...], bool]:
    """Preserve the pre-optimization normalization for exact comparisons."""

    resource_variants: dict[tuple[Any, ...], FlowValue] = {}
    for variant in variants:
        normalized = _without_variants(variant)
        if not normalized.has_origins:
            continue
        resource_variants.setdefault(_flow_value_key(normalized), normalized)
    ordered_resources = [
        resource_variants[key]
        for key in sorted(resource_variants, key=repr)
    ]
    return (
        tuple(ordered_resources[:MAX_FLOW_VARIANTS]),
        len(ordered_resources) > MAX_FLOW_VARIANTS,
    )


def _reference_exclusive_flow_join(
    alternatives: Sequence[FlowValue],
) -> FlowValue:
    """Preserve the pre-optimization exclusive join for exact comparisons."""

    materialized = list(alternatives)
    result = FlowValue()
    variants: list[FlowValue] = []
    has_originless_alternative = False
    tainted_resources: set[str] = set()
    for alternative in materialized:
        result = result.merged(
            access_model._without_top_variants(alternative)
        )
        variants.extend(access_model._all_flow_variants(alternative))
        has_originless_alternative |= (
            alternative.has_originless_alternative
            or (not alternative.has_origins and alternative.has_analysis_state)
        )
        tainted_resources.update(alternative.variant_tainted_resource_ids)
    result.has_originless_alternative = has_originless_alternative
    result.variant_tainted_resource_ids = frozenset(tainted_resources)
    result.variants, variant_overflow = _reference_normalize_flow_variants(
        variants
    )
    if variant_overflow:
        result.overflowed = frozenset(
            set(result.overflowed)
            | {
                resource_id
                for variant in variants
                for resource_id in variant.origins
            }
        )
    return result


def _normalized_key_result(
    values: Sequence[FlowValue],
) -> tuple[tuple[tuple[Any, ...], ...], bool]:
    normalized, overflowed = _normalize_flow_variants(values)
    return tuple(map(_flow_value_key, normalized)), overflowed


def _recursive_retained_size(value: object, seen: set[int]) -> int:
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    retained = sys.getsizeof(value)
    if isinstance(value, dict):
        return retained + sum(
            _recursive_retained_size(key, seen)
            + _recursive_retained_size(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, deque, set, frozenset)):
        return retained + sum(
            _recursive_retained_size(item, seen) for item in value
        )
    namespace = getattr(value, "__dict__", None)
    if isinstance(namespace, dict):
        retained += _recursive_retained_size(namespace, seen)
    return retained


def _flow_key(token: str) -> tuple[Any, ...]:
    return _flow_value_key(
        FlowValue(
            origins={RESOURCE_ID: frozenset({(token,)})},
            object_types={f"object:{token}"},
        )
    )


def _random_flow_value(
    generator: random.Random,
    *,
    depth: int = 0,
) -> FlowValue:
    token = generator.randrange(8)
    secondary_resource = "runtime-resource:secondary-socket"
    resource_id = RESOURCE_ID if token % 2 else secondary_resource
    origins: dict[str, frozenset[tuple[str, ...]]] = (
        {
            resource_id: frozenset(
                {(f"random:{token}", f"depth:{depth}")}
            )
        }
        if generator.randrange(3)
        else {}
    )
    structured_items = None
    attribute_values: dict[str, FlowValue] = {}
    variants: tuple[FlowValue, ...] = ()
    if depth < 2:
        if generator.randrange(3) == 0:
            structured_items = (
                _random_flow_value(generator, depth=depth + 1),
            )
        if generator.randrange(3) == 0:
            attribute_values = {
                f"attribute:{token % 3}": _random_flow_value(
                    generator, depth=depth + 1
                )
            }
        if generator.randrange(2) == 0:
            variants = (
                _random_flow_value(generator, depth=depth + 1),
            )
    return FlowValue(
        origins=origins,
        object_types={f"object:{token % 4}"} if token % 2 else set(),
        overflowed=(frozenset({resource_id}) if origins and token % 5 == 0 else frozenset()),
        module_refs={f"module:{token % 3}"} if token % 3 else set(),
        call_targets={f"call:{token % 3}"} if token % 4 else set(),
        class_targets={f"class:{token % 3}"} if token % 5 else set(),
        unknown_callable=bool(token % 2),
        closure_instances={(f"closure:{token}", f"actor:{depth}")}
        if token % 3 == 1
        else set(),
        structured_items=structured_items,
        runtime_object_ids={f"object-id:{token}"} if token % 3 == 2 else set(),
        runtime_close_ids={f"close-id:{token}"} if token % 4 == 2 else set(),
        runtime_descriptor_ids={f"descriptor-id:{token}"}
        if token % 5 == 2
        else set(),
        instance_ids={f"instance:{token}"} if token % 3 == 0 else set(),
        attribute_values=attribute_values,
        attribute_values_complete=bool(attribute_values) and token % 2 == 0,
        attribute_values_ambiguous=bool(attribute_values) and token % 3 == 0,
        variants=variants,
        has_originless_alternative=token % 5 == 3,
        variant_tainted_resource_ids=(
            frozenset({resource_id}) if origins and token % 4 == 3 else frozenset()
        ),
    )


def _mutate_mutable_flow_graph(value: FlowValue) -> None:
    """Mutate every mutable container recursively to expose shared graphs."""

    value.origins["runtime-resource:mutated"] = frozenset({("mutated",)})
    value.object_types.add("mutated")
    value.module_refs.add("mutated")
    value.call_targets.add("mutated")
    value.class_targets.add("mutated")
    value.closure_instances.add(("mutated", "mutated"))
    value.runtime_object_ids.add("mutated")
    value.runtime_close_ids.add("mutated")
    value.runtime_descriptor_ids.add("mutated")
    value.instance_ids.add("mutated")
    for item in value.structured_items or ():
        _mutate_mutable_flow_graph(item)
    for attribute in value.attribute_values.values():
        _mutate_mutable_flow_graph(attribute)
    for variant in value.variants:
        _mutate_mutable_flow_graph(variant)


def _candidate(
    *,
    resource_id: str = RESOURCE_ID,
    module: str = "chronovisor.resources",
    symbol: str = "SOCKET_PATH",
    locator: str = "unix://$HOME/.chronovisor/runtime/test.sock",
) -> dict[str, Any]:
    return {
        "id": resource_id,
        "module": module,
        "symbol": symbol,
        "locator": {"type": "socket", "value": locator},
    }


def _discover(
    consumer: str,
    *,
    locator: str = "unix://$HOME/.chronovisor/runtime/test.sock",
    extra_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    sources = {
        "src/chronovisor/resources.py": "SOCKET_PATH = object()\n",
        "src/chronovisor/consumer.py": consumer,
        **dict(extra_sources or {}),
    }
    return discover_access_facts(
        {path: source.encode() for path, source in sources.items()},
        [_candidate(locator=locator)],
    )


def _operations(result: Mapping[str, Any]) -> list[str]:
    return sorted(str(row["operation"]) for row in result["access_facts"])


def test_candidate_variants_are_bounded_deduplicated_and_order_stable() -> None:
    alternatives = [
        FlowValue(
            origins={
                RESOURCE_ID: frozenset({(f"origin:variant:{index}",)})
            },
            object_types={f"variant:{index}"},
        )
        for index in range(MAX_FLOW_VARIANTS + 1)
    ]
    forward = exclusive_flow_join([*alternatives, alternatives[0]])
    reverse = exclusive_flow_join(list(reversed(alternatives)))

    assert len(forward.variants) == MAX_FLOW_VARIANTS
    assert forward.variants == reverse.variants
    assert forward.overflowed == frozenset({RESOURCE_ID})

    with_originless_state = exclusive_flow_join(
        [FlowValue(unknown_callable=True), *reversed(alternatives)]
    )
    assert len(with_originless_state.variants) == MAX_FLOW_VARIANTS
    assert all(variant.has_origins for variant in with_originless_state.variants)
    assert with_originless_state.overflowed == frozenset({RESOURCE_ID})


def test_candidate_overflow_is_intersected_per_variant_in_both_orders() -> None:
    second_resource = "runtime-resource:second-socket"
    third_resource = "runtime-resource:third-socket"
    orphan_resource = "runtime-resource:orphan-parent-overflow"
    variants = (
        FlowValue(
            origins={
                RESOURCE_ID: frozenset({("first",)}),
                third_resource: frozenset({("shared",)}),
            },
            overflowed=frozenset({third_resource}),
        ),
        FlowValue(origins={second_resource: frozenset({("second",)})}),
        FlowValue(origins={third_resource: frozenset({("third",)})}),
    )
    parent_overflow = frozenset(
        {RESOURCE_ID, second_resource, orphan_resource}
    )
    expected = {
        frozenset({RESOURCE_ID, third_resource}): frozenset(
            {RESOURCE_ID, third_resource}
        ),
        frozenset({second_resource}): frozenset({second_resource}),
        frozenset({third_resource}): frozenset(),
    }

    for ordered in (variants, tuple(reversed(variants))):
        candidates = candidate_flow_variants(
            FlowValue(overflowed=parent_overflow, variants=ordered)
        )
        assert [frozenset(candidate.origins) for candidate in candidates] == [
            frozenset(candidate.origins) for candidate in ordered
        ]
        for candidate in candidates:
            origin_ids = frozenset(candidate.origins)
            assert candidate.overflowed == expected[origin_ids]
            assert candidate.overflowed <= origin_ids

    expression = ast.parse("sink()\n").body[0]
    assert isinstance(expression, ast.Expr)
    assert isinstance(expression.value, ast.Call)
    site = SyntaxSite(
        site_id=f"runtime-site:{'1' * 64}",
        scope="chronovisor.consumer:exercise",
        kind="Call",
        syntax="sink()",
        occurrence=1,
        path="src/chronovisor/consumer.py",
        line=1,
    )
    locators = {
        RESOURCE_ID: "unix://$HOME/.chronovisor/runtime/first.sock",
        second_resource: "unix://$HOME/.chronovisor/runtime/second.sock",
        third_resource: "unix://$HOME/.chronovisor/runtime/third.sock",
    }
    candidates = candidate_flow_variants(
        FlowValue(overflowed=parent_overflow, variants=variants)
    )
    for candidate in candidates:
        collector = AccessFactCollector(locators, {id(expression.value): site})
        collector.record_access(
            candidate,
            node=expression.value,
            actor="chronovisor.consumer:exercise",
            mode="read",
            operation="path.exists",
            sink="pathlib.Path.exists",
            path="src/chronovisor/consumer.py",
            line=1,
            ordinal=1,
        )
        result = collector.result()
        validate_runtime_access_v2_result(result)
        access_ids = set(result["access_fact_ids"])
        for overflow in result["escape_facts"]:
            if overflow["reason"] == "provenance_overflow":
                assert overflow["resource_id"] in candidate.origins
                assert overflow["source_fact_id"] in access_ids


def test_nested_variants_do_not_split_canonical_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FlowValue(
        origins={RESOURCE_ID: frozenset({("shared",)})},
        object_types={"shared-object"},
        variants=(
            FlowValue(origins={RESOURCE_ID: frozenset({("nested:first",)})}),
        ),
    )
    second = FlowValue(
        origins={RESOURCE_ID: frozenset({("shared",)})},
        object_types={"shared-object"},
        variants=(
            FlowValue(origins={RESOURCE_ID: frozenset({("nested:second",)})}),
        ),
    )
    expected = _without_variants(first)
    stripped_ids: list[int] = []
    original_without_variants = access_model._without_variants

    def tracked_without_variants(value: FlowValue) -> FlowValue:
        stripped_ids.append(id(value))
        return original_without_variants(value)

    monkeypatch.setattr(access_model, "_without_variants", tracked_without_variants)

    normalized, overflowed = _normalize_flow_variants((first, second))

    assert normalized == (expected,)
    assert stripped_ids == [id(first)]
    assert overflowed is False


def test_flow_repr_cache_is_exact_for_random_reverse_and_overflow_cases() -> None:
    generator = random.Random(0xCACE)
    cases: list[list[FlowValue]] = []
    for _ in range(48):
        values = [
            _random_flow_value(generator)
            for _ in range(generator.randrange(2, 24))
        ]
        cases.extend((values, list(reversed(values))))
    cases.append(
        [
            FlowValue(
                origins={RESOURCE_ID: frozenset({(f"overflow:{index}",)})}
            )
            for index in range(MAX_FLOW_VARIANTS + 11)
        ]
    )

    for values in cases:
        uncached = _normalized_key_result(values)
        with _scoped_flow_repr_cache() as cache:
            cached = _normalized_key_result(values)
            cached_again = _normalized_key_result(values)
            cached_third = _normalized_key_result(values)
            stats = cache.stats()
        assert repr(cached).encode() == repr(uncached).encode()
        assert repr(cached_again).encode() == repr(uncached).encode()
        assert repr(cached_third).encode() == repr(uncached).encode()
        assert stats.peak_entries <= _MAX_FLOW_REPR_CACHE_ENTRIES
        assert stats.peak_witnesses <= _MAX_FLOW_REPR_CACHE_WITNESSES
        assert stats.peak_bytes <= _MAX_FLOW_REPR_CACHE_BYTES


def test_flow_repr_cache_fastpaths_never_consult_cache() -> None:
    originless = FlowValue(object_types={"originless"})
    single = FlowValue(origins={RESOURCE_ID: frozenset({("single",)})})
    duplicate = single.copy()
    duplicate.variants = (FlowValue(object_types={"ignored"}),)

    with _scoped_flow_repr_cache() as cache:
        assert _normalize_flow_variants(()) == ((), False)
        assert _normalize_flow_variants((originless,)) == ((), False)
        assert _normalize_flow_variants((single,)) == (
            (_without_variants(single),),
            False,
        )
        assert _normalize_flow_variants((single, duplicate)) == (
            (_without_variants(single),),
            False,
        )
        stats = cache.stats()

    assert stats.hits == stats.misses == stats.admissions == 0
    assert stats.current_entries == stats.current_witnesses == 0


def test_flow_repr_cache_retains_immutable_snapshot_not_mutable_flow_graph() -> None:
    value = FlowValue(
        origins={RESOURCE_ID: frozenset({("before",)})},
        object_types={"before"},
        structured_items=(FlowValue(object_types={"nested-before"}),),
    )
    original_key = _flow_value_key(value)
    with _scoped_flow_repr_cache() as cache:
        assert _flow_key_repr(original_key) == repr(original_key)
        assert _flow_key_repr(original_key) == repr(original_key)
        assert _flow_key_repr(original_key) == repr(original_key)
        _mutate_mutable_flow_graph(value)
        mutated_key = _flow_value_key(value)
        assert mutated_key != original_key
        assert _flow_key_repr(mutated_key) == repr(mutated_key)
        stats = cache.stats()
        retained_keys = [entry[1] for entry in cache._fifo]

    assert stats.hits == 1
    assert stats.misses == 3
    assert retained_keys == [original_key]
    assert all(
        not isinstance(item, FlowValue)
        for key in retained_keys
        for item in key
    )


def test_flow_repr_cache_evicts_witnesses_and_entries_then_relearns_phase() -> None:
    def values(prefix: str) -> tuple[FlowValue, ...]:
        return tuple(
            FlowValue(
                origins={RESOURCE_ID: frozenset({(f"{prefix}:{index}",)})}
            )
            for index in range(4)
        )

    phase_a = values("phase-a")
    phase_b = values("phase-b")
    with _scoped_flow_repr_cache(
        max_entries=4,
        max_witnesses=4,
        max_bytes=1_000_000,
    ) as cache:
        expected_a = _normalized_key_result(phase_a)
        assert _normalized_key_result(phase_a) == expected_a
        assert _normalized_key_result(phase_a) == expected_a
        for index in range(12):
            churn = values(f"churn:{index}")
            expected_churn, expected_overflow = (
                _reference_normalize_flow_variants(churn)
            )
            assert _normalized_key_result(churn) == (
                tuple(map(_flow_value_key, expected_churn)),
                expected_overflow,
            )
        before_b = cache.stats()
        expected_b = _normalized_key_result(phase_b)
        after_first = cache.stats()
        assert _normalized_key_result(phase_b) == expected_b
        after_second = cache.stats()
        assert _normalized_key_result(phase_b) == expected_b
        after_third = cache.stats()

    assert after_first.misses - before_b.misses == 4
    assert after_second.misses - after_first.misses == 4
    assert after_third.hits - after_second.hits == 4
    assert after_second.entry_evictions - before_b.entry_evictions == 4
    assert after_third.witness_evictions > 0
    assert after_third.current_entries == 4
    assert after_third.current_witnesses == 0


def test_flow_repr_cache_unique_stream_retains_only_bounded_witnesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        access_model,
        "hash",
        lambda key: key[0],
        raising=False,
    )
    with _scoped_flow_repr_cache() as cache:
        for index in range(10_000):
            key = (index,)
            assert _flow_key_repr(key) == repr(key)
        stats = cache.stats()

    assert stats.hits == 0
    assert stats.misses == 10_000
    assert stats.admissions == 0
    assert stats.current_entries == 0
    assert stats.current_witnesses == _MAX_FLOW_REPR_CACHE_WITNESSES
    assert (
        stats.witness_evictions
        == 10_000 - _MAX_FLOW_REPR_CACHE_WITNESSES
    )
    assert stats.peak_bytes <= _MAX_FLOW_REPR_CACHE_BYTES


def test_flow_repr_cache_constant_hash_collision_is_exact_and_linearly_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(access_model, "hash", lambda _key: 7, raising=False)
    keys = [_flow_key(f"collision:{index}") for index in range(65)]
    with _scoped_flow_repr_cache() as cache:
        for key in keys[:64]:
            assert _flow_key_repr(key) == builtins.repr(key)
            assert _flow_key_repr(key) == builtins.repr(key)
        bucket = cache._buckets[7]
        assert len(bucket) == 64
        assert all(
            entry[1] == key
            for entry, key in zip(bucket, keys[:64], strict=True)
        )
        hits_before = cache.stats().hits
        for key in keys[:64]:
            assert _flow_key_repr(key) == builtins.repr(key)
        assert cache.stats().hits - hits_before == 64
        assert _flow_key_repr(keys[64]) == builtins.repr(keys[64])
        assert _flow_key_repr(keys[64]) == builtins.repr(keys[64])
        stats = cache.stats()
        bucket_after = list(cache._buckets[7])

    assert len(bucket_after) == 64
    assert [entry[1] for entry in bucket_after] == keys[1:]
    assert stats.entry_evictions == 1
    assert stats.peak_entries == 64


@pytest.mark.parametrize(
    "token",
    [
        "ascii:" + "x" * 10_000,
        "cjk:" + "漢字" * 5_000,
        "emoji:" + "🙂" * 5_000,
    ],
)
def test_flow_repr_cache_accounting_bounds_entire_retained_graph(
    token: str,
) -> None:
    key = _flow_key(token)
    with _scoped_flow_repr_cache() as cache:
        assert _current_flow_repr_cache() is cache
        for _ in range(3):
            assert _flow_key_repr(key) == repr(key)
        stats = cache.stats()
        actual_graph_bytes = _recursive_retained_size(
            (cache, _current_flow_repr_cache()), set()
        )

    assert actual_graph_bytes <= stats.current_bytes
    assert stats.current_bytes <= _MAX_FLOW_REPR_CACHE_BYTES
    assert stats.peak_bytes <= _MAX_FLOW_REPR_CACHE_BYTES
    assert cache.stats().current_bytes == 0


def test_flow_repr_cache_rejects_tiny_cap_oversized_and_unknown_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="too small"):
        _FlowReprCache(max_bytes=1)

    hot = ("hot",)
    oversized = ("x" * 20_000,)
    with _scoped_flow_repr_cache(max_bytes=16_384) as cache:
        assert _flow_key_repr(hot) == repr(hot)
        assert _flow_key_repr(hot) == repr(hot)
        assert _flow_key_repr(hot) == repr(hot)
        assert _flow_key_repr(oversized) == repr(oversized)
        assert _flow_key_repr(oversized) == repr(oversized)
        assert _flow_key_repr(hot) == repr(hot)
        oversized_stats = cache.stats()
        retained_hot = [entry[1] for entry in cache._fifo]
    assert retained_hot == [hot]
    assert oversized_stats.current_entries == 1
    assert oversized_stats.skips == 1
    assert oversized_stats.peak_bytes <= 16_384

    class UnknownHashable:
        def __init__(self) -> None:
            self.payload = "x" * (2 * _MAX_FLOW_REPR_CACHE_BYTES)

        def __hash__(self) -> int:
            return 29

        def __sizeof__(self) -> int:
            raise AssertionError("custom __sizeof__ must not run")

    unknown = ("unknown", UnknownHashable())
    with _scoped_flow_repr_cache() as cache:
        expected = repr(unknown)
        assert _flow_key_repr(unknown) == expected
        assert _flow_key_repr(unknown) == expected
        unknown_stats = cache.stats()
    assert unknown_stats.current_entries == 0
    assert unknown_stats.current_witnesses == 0
    assert unknown_stats.misses == 1
    assert unknown_stats.skips == 1
    assert unknown_stats.peak_witnesses == 1
    assert unknown_stats.peak_bytes <= _MAX_FLOW_REPR_CACHE_BYTES

    class RaisingEquality:
        def __repr__(self) -> str:
            return "raising-equality"

        def __eq__(self, _other: object) -> bool:
            raise AssertionError("custom __eq__ must not run")

        def __hash__(self) -> int:
            raise AssertionError("custom __hash__ must not run")

        def __sizeof__(self) -> int:
            raise AssertionError("custom __sizeof__ must not run")

    monkeypatch.setattr(access_model, "hash", lambda _key: 7, raising=False)
    safe = ("same-prefix", "safe")
    unsupported = ("same-prefix", RaisingEquality())
    with _scoped_flow_repr_cache() as cache:
        assert _flow_key_repr(safe) == repr(safe)
        assert _flow_key_repr(safe) == repr(safe)
        assert _flow_key_repr(unsupported) == repr(unsupported)
        protocol_stats = cache.stats()
    assert protocol_stats.current_entries == 1
    assert protocol_stats.skips == 1


def test_flow_repr_cache_unsafe_prefix_never_invokes_custom_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsafePrefix:
        def __repr__(self) -> str:
            return "unsafe-prefix"

        def __hash__(self) -> int:
            raise AssertionError("custom __hash__ must not run")

        def __eq__(self, _other: object) -> bool:
            raise AssertionError("custom __eq__ must not run")

        def __sizeof__(self) -> int:
            raise AssertionError("custom __sizeof__ must not run")

    def forbidden_hash(_value: object) -> int:
        raise AssertionError("unsafe prefix must not reach fingerprint hash")

    monkeypatch.setattr(access_model, "hash", forbidden_hash, raising=False)
    key = (UnsafePrefix(), "tail")
    expected = repr(key)
    with _scoped_flow_repr_cache() as cache:
        assert _flow_key_repr(key) == expected
        assert _flow_key_repr(key) == expected
        stats = cache.stats()

    assert stats.hits == stats.misses == stats.admissions == 0
    assert stats.skips == 2
    assert stats.current_entries == stats.current_witnesses == 0


def test_flow_repr_cache_unsafe_tail_stays_baseline_and_cannot_false_hit() -> None:
    class UnsafeTail:
        def __repr__(self) -> str:
            return "unsafe-tail"

        def __hash__(self) -> int:
            raise AssertionError("custom __hash__ must not run")

        def __eq__(self, _other: object) -> bool:
            raise AssertionError("custom __eq__ must not run")

        def __sizeof__(self) -> int:
            raise AssertionError("custom __sizeof__ must not run")

    prefix = tuple(f"prefix:{index}" for index in range(8))
    unsafe = (*prefix, UnsafeTail())
    expected_unsafe = repr(unsafe)
    with _scoped_flow_repr_cache() as cache:
        assert _flow_key_repr(unsafe) == expected_unsafe
        after_first = cache.stats()
        assert _flow_key_repr(unsafe) == expected_unsafe
        after_second = cache.stats()

        safe = (*prefix, "safe-tail")
        assert _flow_key_repr(safe) == repr(safe)
        assert _flow_key_repr(safe) == repr(safe)
        after_safe = cache.stats()
        assert _flow_key_repr(unsafe) == expected_unsafe
        final = cache.stats()

    assert after_first.misses == 1
    assert after_first.current_witnesses == 1
    assert after_second.skips == 1
    assert after_second.current_witnesses == 0
    assert after_safe.admissions == 1
    assert after_safe.current_entries == 1
    assert final.hits == 0
    assert final.skips == 2
    assert final.current_entries == 1


def test_flow_repr_cache_reentrant_tail_repr_keeps_witness_accounting_exact() -> None:
    active_cache: list[_FlowReprCache] = []
    nested_outputs: list[str] = []
    safe = ("reentrant-prefix", "safe-tail")

    class ReentrantTail:
        reenter = True

        def __repr__(self) -> str:
            if self.reenter and active_cache:
                self.reenter = False
                nested_outputs.append(active_cache[0].render(safe))
            return "reentrant-tail"

    unsafe = ("reentrant-prefix", ReentrantTail())
    expected = repr(unsafe)
    with _scoped_flow_repr_cache() as cache:
        active_cache.append(cache)
        assert cache.render(unsafe) == expected
        after_reentry = cache.stats()
        assert cache._witness_bytes == sum(
            sys.getsizeof(fingerprint) for fingerprint in cache._witnesses
        )

        assert cache.render(unsafe) == expected
        after_discard = cache.stats()
        assert cache._witness_bytes == 0
        assert cache._witnesses == {}

        assert cache.render(unsafe) == expected
        assert cache.render(unsafe) == expected
        after_repeat = cache.stats()
        assert cache._witness_bytes == sum(
            sys.getsizeof(fingerprint) for fingerprint in cache._witnesses
        )
        active_cache.clear()

    assert nested_outputs == [repr(safe)]
    assert after_reentry.current_witnesses == 1
    assert after_discard.current_witnesses == 0
    assert after_repeat.current_witnesses == 0
    assert after_repeat.current_bytes <= _MAX_FLOW_REPR_CACHE_BYTES
    assert after_repeat.peak_bytes <= _MAX_FLOW_REPR_CACHE_BYTES
    assert cache.stats().current_bytes == 0

    closing_cache: list[_FlowReprCache] = []

    class ClosingTail:
        def __repr__(self) -> str:
            if closing_cache:
                closing_cache[0].close()
            return "closing-tail"

    closing = ("closing-prefix", ClosingTail())
    expected_closing = repr(closing)
    with _scoped_flow_repr_cache() as cache:
        closing_cache.append(cache)
        assert cache.render(closing) == expected_closing
        assert cache.render(closing) == expected_closing
        closed_stats = cache.stats()
        closing_cache.clear()

    assert closed_stats.closed is True
    assert closed_stats.current_entries == 0
    assert closed_stats.current_witnesses == 0
    assert closed_stats.current_bytes == 0
    assert cache._witness_bytes == 0
    assert cache._witnesses == {}


def test_flow_repr_cache_same_prefix_collisions_are_exact_and_bounded() -> None:
    prefix = tuple(f"shared-prefix:{index}" for index in range(8))
    keys = [(*prefix, f"tail:{index}") for index in range(200)]
    with _scoped_flow_repr_cache() as cache:
        for key in keys:
            assert _flow_key_repr(key) == repr(key)
        before_probes = cache.stats()
        assert _flow_key_repr(keys[0]) == repr(keys[0])
        assert cache.stats().hits == before_probes.hits
        assert _flow_key_repr(keys[-1]) == repr(keys[-1])
        final = cache.stats()
        bucket_lengths = [len(bucket) for bucket in cache._buckets.values()]

    assert before_probes.hits == 0
    assert before_probes.admissions == 100
    assert before_probes.current_entries == _MAX_FLOW_REPR_CACHE_ENTRIES
    assert before_probes.entry_evictions == 36
    assert final.hits == 1
    assert max(bucket_lengths) <= _MAX_FLOW_REPR_CACHE_ENTRIES
    assert final.peak_entries <= _MAX_FLOW_REPR_CACHE_ENTRIES
    assert final.peak_bytes <= _MAX_FLOW_REPR_CACHE_BYTES


@pytest.mark.parametrize("max_bytes", [11_578, 11_579, 11_580, 11_581, 11_582])
def test_flow_repr_cache_admission_boundary_never_exceeds_cap(
    max_bytes: int,
) -> None:
    key = ("x" * 1_000,)
    with _scoped_flow_repr_cache(max_bytes=max_bytes) as cache:
        assert _flow_key_repr(key) == repr(key)
        assert _flow_key_repr(key) == repr(key)
        assert _flow_key_repr(key) == repr(key)
        stats = cache.stats()

    assert stats.current_bytes <= max_bytes
    assert stats.peak_bytes <= max_bytes
    assert stats.current_entries in {0, 1}


def test_flow_repr_cache_nested_and_copied_contexts_are_thread_safe() -> None:
    key = _flow_key("context")
    assert _current_flow_repr_cache() is None
    with _scoped_flow_repr_cache() as outer:
        assert _current_flow_repr_cache() is outer
        with _scoped_flow_repr_cache() as inner:
            assert _current_flow_repr_cache() is inner
            assert _flow_key_repr(key) == repr(key)
        assert inner.stats().closed is True
        assert _current_flow_repr_cache() is outer

        contexts = [contextvars.copy_context(), contextvars.copy_context()]
        barrier = threading.Barrier(3)

        def exercise() -> list[str]:
            barrier.wait()
            return [_flow_key_repr(key) for _ in range(4)]

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(context.run, exercise) for context in contexts
            ]
            barrier.wait()
            results = [future.result() for future in futures]
        stats = outer.stats()

    assert results == [[repr(key)] * 4, [repr(key)] * 4]
    assert stats.admissions == 1
    assert stats.hits == 6
    assert stats.misses == 2
    assert outer.stats().closed is True
    assert outer.stats().current_entries == 0
    assert _current_flow_repr_cache() is None
    assert all(
        context.run(_current_flow_repr_cache) is outer for context in contexts
    )
    assert all(
        context.run(_flow_key_repr, key) == repr(key) for context in contexts
    )
    assert outer.stats().current_entries == 0


def test_nested_only_originless_variant_is_skipped() -> None:
    nested_only = FlowValue(
        structured_items=(
            FlowValue(origins={RESOURCE_ID: frozenset({("structured",)})}),
        ),
        attribute_values={
            "resource": FlowValue(
                origins={RESOURCE_ID: frozenset({("attribute",)})}
            )
        },
        variants=(
            FlowValue(origins={RESOURCE_ID: frozenset({("variant",)})}),
        ),
    )
    direct = FlowValue(origins={RESOURCE_ID: frozenset({("direct",)})})

    normalized, overflowed = _normalize_flow_variants((nested_only, direct))

    assert normalized == (_without_variants(direct),)
    assert overflowed is False


def test_normalized_winner_is_copy_isolated_from_input() -> None:
    winner = FlowValue(
        origins={RESOURCE_ID: frozenset({("winner",)})},
        object_types={"outer"},
        structured_items=(FlowValue(object_types={"structured"}),),
        attribute_values={"field": FlowValue(object_types={"attribute"})},
        variants=(FlowValue(object_types={"discarded-variant"}),),
    )
    original = winner.copy()

    normalized, overflowed = _normalize_flow_variants((winner,))
    normalized[0].origins[RESOURCE_ID] = frozenset({("mutated",)})
    normalized[0].object_types.add("mutated")
    assert normalized[0].structured_items is not None
    normalized[0].structured_items[0].object_types.add("mutated")
    normalized[0].attribute_values["field"].object_types.add("mutated")

    assert winner == original
    assert normalized[0].variants == ()
    assert overflowed is False


def test_optimized_normalization_matches_reference_on_bounded_cases() -> None:
    generator = random.Random(0xC0FFEE)
    cases: list[list[FlowValue]] = [
        [],
        [FlowValue()],
        [
            FlowValue(
                origins={RESOURCE_ID: frozenset({(f"unique:{index}",)})},
                variants=(FlowValue(object_types={f"nested:{index}"}),),
            )
            for index in range(MAX_FLOW_VARIANTS + 1)
        ],
    ]
    for _ in range(32):
        values = [
            _random_flow_value(generator)
            for _ in range(generator.randrange(1, 20))
        ]
        if values:
            duplicate = values[0].copy()
            duplicate.variants = (FlowValue(object_types={"duplicate-only"}),)
            values.append(duplicate)
        cases.append(values)

    for values in cases:
        expected, expected_overflow = _reference_normalize_flow_variants(values)
        actual, actual_overflow = _normalize_flow_variants(values)
        assert actual == expected
        assert tuple(map(_flow_value_key, actual)) == tuple(
            map(_flow_value_key, expected)
        )
        assert actual_overflow is expected_overflow


def test_normalization_trivial_and_unique_paths_do_not_render_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    originless = FlowValue(object_types={"originless"})
    single = FlowValue(
        origins={RESOURCE_ID: frozenset({("single",)})},
        structured_items=(FlowValue(object_types={"nested"}),),
    )
    duplicate = single.copy()
    duplicate.variants = (FlowValue(object_types={"ignored"}),)
    builtin_repr = repr
    rendered: list[object] = []
    keyed: list[int] = []
    original_flow_value_key = access_model._flow_value_key
    key_in_progress = False

    def forbidden_repr(value: object) -> str:
        rendered.append(value)
        raise AssertionError("trivial normalization rendered a flow key")

    def forbidden_flow_value_key(value: FlowValue) -> tuple[Any, ...]:
        raise AssertionError(f"trivial normalization keyed {value!r}")

    with monkeypatch.context() as context:
        context.setattr(
            access_model, "repr", forbidden_repr, raising=False
        )
        context.setattr(
            access_model, "_flow_value_key", forbidden_flow_value_key
        )
        assert _normalize_flow_variants(()) == ((), False)
        assert _normalize_flow_variants((originless,)) == ((), False)
        assert _normalize_flow_variants((single,)) == (
            (_without_variants(single),),
            False,
        )

    def tracked_flow_value_key(value: FlowValue) -> tuple[Any, ...]:
        nonlocal key_in_progress
        if key_in_progress:
            return original_flow_value_key(value)
        keyed.append(id(value))
        key_in_progress = True
        try:
            return original_flow_value_key(value)
        finally:
            key_in_progress = False

    with monkeypatch.context() as context:
        context.setattr(
            access_model, "repr", forbidden_repr, raising=False
        )
        context.setattr(
            access_model, "_flow_value_key", tracked_flow_value_key
        )
        assert _normalize_flow_variants((single, duplicate)) == (
            (_without_variants(single),),
            False,
        )
    assert rendered == []
    assert keyed == [id(single), id(duplicate)]

    first = FlowValue(origins={RESOURCE_ID: frozenset({("first",)})})
    second = FlowValue(origins={RESOURCE_ID: frozenset({("second",)})})

    def tracked_repr(value: object) -> str:
        rendered.append(value)
        return builtin_repr(value)

    with monkeypatch.context() as context:
        context.setattr(access_model, "repr", tracked_repr, raising=False)
        normalized, overflowed = _normalize_flow_variants((first, second))
    assert len(normalized) == 2
    assert overflowed is False
    assert len(rendered) == 2


def test_exclusive_join_matches_reference_for_adversarial_values() -> None:
    generator = random.Random(0xA11CE)
    cases: list[list[FlowValue]] = [
        [],
        [FlowValue()],
        [FlowValue(origins={RESOURCE_ID: frozenset({("single",)})})],
        [
            FlowValue(
                variants=(
                    FlowValue(
                        origins={
                            RESOURCE_ID: frozenset({("variant-only",)})
                        }
                    ),
                )
            ),
            FlowValue(
                origins={RESOURCE_ID: frozenset({("later-direct",)})}
            ),
        ],
    ]
    for _ in range(64):
        values = [
            _random_flow_value(generator)
            for _ in range(generator.randrange(1, 18))
        ]
        if values:
            duplicate = values[0].copy()
            duplicate.variants = (
                FlowValue(object_types={"ignored-duplicate-variant"}),
            )
            values.append(duplicate)
        cases.extend((values, list(reversed(values))))

    for alternatives in cases:
        before = [alternative.copy() for alternative in alternatives]
        expected = _reference_exclusive_flow_join(alternatives)
        actual = exclusive_flow_join(alternatives)
        assert actual == expected
        assert _flow_value_key(actual) == _flow_value_key(expected)
        assert alternatives == before


def test_exclusive_join_only_strips_truthy_top_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = FlowValue(
        origins={RESOURCE_ID: frozenset({("direct",)})},
        structured_items=(FlowValue(object_types={"direct-nested"}),),
    )
    branched = FlowValue(
        origins={RESOURCE_ID: frozenset({("outer",)})},
        variants=(
            FlowValue(origins={RESOURCE_ID: frozenset({("variant",)})}),
        ),
    )
    expected = _reference_exclusive_flow_join((direct, branched))
    expected_reverse = _reference_exclusive_flow_join((branched, direct))
    stripped_ids: list[int] = []
    original = access_model._without_top_variants

    def tracked(value: FlowValue) -> FlowValue:
        stripped_ids.append(id(value))
        return original(value)

    monkeypatch.setattr(access_model, "_without_top_variants", tracked)

    assert exclusive_flow_join((direct, branched)) == expected
    assert stripped_ids == [id(branched)]
    stripped_ids.clear()
    assert exclusive_flow_join((branched, direct)) == expected_reverse
    assert stripped_ids == [id(branched)]


def test_exclusive_join_preserves_shortcut_counterexamples_and_copy_isolation() -> None:
    nested = FlowValue(
        origins={RESOURCE_ID: frozenset({("outer",)})},
        object_types={"outer"},
        overflowed=frozenset({RESOURCE_ID}),
        module_refs={"module"},
        call_targets={"call"},
        class_targets={"class"},
        closure_instances={("closure", "actor")},
        structured_items=(
            FlowValue(
                origins={RESOURCE_ID: frozenset({("structured",)})},
                object_types={"structured"},
            ),
        ),
        runtime_object_ids={"object-id"},
        runtime_close_ids={"close-id"},
        runtime_descriptor_ids={"descriptor-id"},
        instance_ids={"instance"},
        attribute_values={
            "field": FlowValue(
                origins={RESOURCE_ID: frozenset({("attribute",)})},
                object_types={"attribute"},
            )
        },
        attribute_values_complete=True,
        variants=(
            FlowValue(
                origins={RESOURCE_ID: frozenset({("variant",)})},
                object_types={"variant"},
            ),
        ),
        has_originless_alternative=True,
        variant_tainted_resource_ids=frozenset({RESOURCE_ID}),
    )
    direct = FlowValue(
        origins={RESOURCE_ID: frozenset({("direct",)})},
        object_types={"direct"},
    )
    alternatives = [nested, direct]
    before = [alternative.copy() for alternative in alternatives]

    result = exclusive_flow_join(alternatives)
    assert result == _reference_exclusive_flow_join(alternatives)
    _mutate_mutable_flow_graph(result)
    assert alternatives == before

    single = exclusive_flow_join((direct,))
    equal = exclusive_flow_join((direct, direct.copy()))
    assert len(single.variants) == 1
    assert len(equal.variants) == 1
    assert single == _reference_exclusive_flow_join((direct,))
    assert equal == _reference_exclusive_flow_join((direct, direct.copy()))


def test_originless_exclusive_state_is_retained_for_simultaneous_merges() -> None:
    first = exclusive_flow_join(
        [
            FlowValue(origins={RESOURCE_ID: frozenset({("first",)})}),
            FlowValue(unknown_callable=True),
        ]
    )
    second = exclusive_flow_join(
        [
            FlowValue(origins={RESOURCE_ID: frozenset({("second",)})}),
            FlowValue(object_types={"opaque-runtime-type"}),
        ]
    )

    assert len(first.variants) == 1
    assert first.has_originless_alternative is True
    assert len(candidate_flow_variants(first)) == 1
    simultaneous = simultaneous_flow_merge((first, second))
    assert simultaneous.variant_tainted_resource_ids == frozenset({RESOURCE_ID})
    assert all(
        variant.unknown_callable
        for variant in candidate_flow_variants(simultaneous)
    )


def test_variant_overflow_survives_transforms_rejoin_and_fact_collection() -> None:
    alternatives = [
        FlowValue(
            origins={
                RESOURCE_ID: frozenset({(f"origin:variant:{index}",)})
            }
        )
        for index in range(MAX_FLOW_VARIANTS + 1)
    ]

    def transformed(values: list[FlowValue]) -> FlowValue:
        joined = exclusive_flow_join(values)
        for step in (
            "constructor:pathlib.Path",
            "transform:expanduser",
            "transform:resolve",
            "representation:builtins.str",
        ):
            joined = exclusive_flow_join(
                [
                    variant.bound(step)
                    for variant in candidate_flow_variants(joined)
                ]
            )
        return joined

    forward = transformed(alternatives)
    reverse = transformed(list(reversed(alternatives)))
    assert forward == reverse
    assert forward.overflowed == frozenset({RESOURCE_ID})

    expression = ast.parse("sink()\n").body[0]
    assert isinstance(expression, ast.Expr)
    assert isinstance(expression.value, ast.Call)
    site = SyntaxSite(
        site_id=f"runtime-site:{'0' * 64}",
        scope="chronovisor.consumer:exercise",
        kind="Call",
        syntax="sink()",
        occurrence=1,
        path="src/chronovisor/consumer.py",
        line=1,
    )
    projected_results: list[dict[str, Any]] = []
    for joined in (forward, reverse):
        collector = AccessFactCollector(
            {RESOURCE_ID: "unix://$HOME/.chronovisor/runtime/test.sock"},
            {id(expression.value): site},
        )
        collector.record_access(
            joined,
            node=expression.value,
            actor="chronovisor.consumer:exercise",
            mode="read",
            operation="path.exists",
            sink="pathlib.Path.exists",
            path="src/chronovisor/consumer.py",
            line=1,
            ordinal=1,
        )
        projected_results.append(collector.result())
    projected = projected_results[0]
    assert json.dumps(
        projected_results[0], sort_keys=True, separators=(",", ":")
    ).encode() == json.dumps(
        projected_results[1], sort_keys=True, separators=(",", ":")
    ).encode()
    assert projected["access_facts"][0]["provenance_complete"] is False
    assert any(
        row["reason"] == "provenance_overflow"
        and row["source_kind"] == "access"
        for row in projected["escape_facts"]
    )


def test_structured_candidates_keep_outer_shape_and_unknown_taint() -> None:
    candidate = exclusive_flow_join(
        [
            FlowValue(origins={RESOURCE_ID: frozenset({("origin",)})}),
            FlowValue(),
        ]
    ).merged(FlowValue(unknown_callable=True))
    wrapped = structured_flow_value((candidate,))

    variants = candidate_flow_variants(wrapped)
    assert len(variants) == 1
    assert variants[0].structured_items is not None
    assert variants[0].unknown_callable is True


def test_raw_unix_socket_module_from_import_alias_and_star_import() -> None:
    module_alias = _discover(
        "import socket as network\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "FAMILY = network.AF_UNIX\n"
        "TYPE = network.SOCK_STREAM\n"
        "def exercise():\n"
        "    client = network.socket(FAMILY, TYPE, proto=0, fileno=None)\n"
        "    alias = client\n"
        "    alias.connect(SOCKET_PATH)\n"
        "    client.sendall(b'request')\n"
        "    alias.recv(1024)\n"
        "    client.close()\n"
    )
    from_import = _discover(
        "from socket import AF_UNIX as FAMILY, SOCK_STREAM as TYPE\n"
        "from socket import socket as Socket\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def exercise():\n"
        "    client = Socket(family=FAMILY, type=TYPE)\n"
        "    client.connect(SOCKET_PATH)\n"
        "    client.send(b'x')\n"
        "    client.recv(1)\n"
    )
    star_import = _discover(
        "from socket import *\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def exercise():\n"
        "    client = socket(AF_UNIX, SOCK_STREAM)\n"
        "    client.connect(SOCKET_PATH)\n"
        "    client.recv(1)\n"
    )

    assert _operations(module_alias) == [
        "socket.connect",
        "socket.recv",
        "socket.sendall",
    ]
    assert _operations(from_import) == [
        "socket.connect",
        "socket.recv",
        "socket.send",
    ]
    assert _operations(star_import) == ["socket.connect", "socket.recv"]


def test_unix_stream_constructor_defaults_type_and_rejects_default_family() -> None:
    result = _discover(
        "import socket\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def exercise():\n"
        "    implicit = socket.socket(socket.AF_UNIX)\n"
        "    implicit.connect(SOCKET_PATH)\n"
        "    negative_one = socket.socket(socket.AF_UNIX, -1, -1)\n"
        "    negative_one.connect(SOCKET_PATH)\n"
        "    socket.socket().connect(SOCKET_PATH)\n"
    )

    assert Counter(_operations(result)) == Counter({"socket.connect": 2})


def test_socket_module_mutation_reimport_and_prior_from_import_follow_python() -> None:
    prior_binding = _discover(
        "import socket\n"
        "from socket import socket as original_socket\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "socket.socket = object()\n"
        "import socket as reimported\n"
        "def preserved_prior_binding():\n"
        "    client = original_socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "    client.connect(SOCKET_PATH)\n"
        "def mutated_reimport():\n"
        "    client = reimported.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "    client.connect(SOCKET_PATH)\n"
    )
    mutation_forms = [
        "socket.__dict__['socket'] = object()",
        "setattr(socket, 'socket', object())",
        "delattr(socket, 'socket')",
        "socket.__dict__.update({'socket': object()})",
    ]

    assert _operations(prior_binding) == ["socket.connect"]
    for mutation in mutation_forms:
        result = _discover(
            "import socket\n"
            "from chronovisor.resources import SOCKET_PATH\n"
            f"{mutation}\n"
            "def exercise():\n"
            "    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            "    client.connect(SOCKET_PATH)\n"
        )
        assert result["access_facts"] == [], mutation


def test_raw_bind_listen_accept_and_accepted_socket_lifetimes_are_separate() -> None:
    result = _discover(
        "import socket\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def serve_once():\n"
        "    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "    listener_alias = listener\n"
        "    listener_alias.bind(SOCKET_PATH)\n"
        "    listener.listen(8)\n"
        "    accepted, address = listener_alias.accept()\n"
        "    accepted.recv(1024)\n"
        "    accepted.sendall(b'ok')\n"
        "    accepted.close()\n"
        "    listener.accept()\n"
        "    listener.close()\n"
    )

    assert Counter(_operations(result)) == Counter(
        {
            "socket.bind": 1,
            "socket.listen": 1,
            "socket.accept": 2,
            "socket.recv": 1,
            "socket.sendall": 1,
        }
    )
    accept_provenance = [
        row
        for row in result["provenances"]
        if any("result:socket.accept:socket" in step for step in row["binding_chain"])
    ]
    assert accept_provenance
    assert all("address" not in step for row in accept_provenance for step in row["binding_chain"])


def test_raw_socket_context_exit_close_and_unknown_escape_stop_later_facts() -> None:
    context = _discover(
        "import socket\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def exercise():\n"
        "    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:\n"
        "        client.connect(SOCKET_PATH)\n"
        "        client.recv(1)\n"
        "    client.sendall(b'closed')\n"
    )
    unrelated = _discover(
        "import socket\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def exercise(callback):\n"
        "    first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "    second = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "    first.connect(SOCKET_PATH)\n"
        "    second.connect(SOCKET_PATH)\n"
        "    callback(first)\n"
        "    first.recv(1)\n"
        "    second.recv(1)\n"
    )

    assert _operations(context) == ["socket.connect", "socket.recv"]
    assert Counter(_operations(unrelated)) == Counter(
        {"socket.connect": 2, "socket.recv": 1}
    )
    assert any(
        row["reason"] == "registered_locator_to_unknown_callee"
        for row in unrelated["escape_facts"]
    )


def test_raw_socket_invalid_cross_state_methods_escape_and_contaminate() -> None:
    result = _discover(
        "import socket\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def exercise():\n"
        "    connected = socket.socket(socket.AF_UNIX)\n"
        "    connected.connect(SOCKET_PATH)\n"
        "    connected.listen()\n"
        "    bound = socket.socket(socket.AF_UNIX)\n"
        "    bound.bind(SOCKET_PATH)\n"
        "    bound.sendall(b'not-connected')\n"
        "    fresh = socket.socket(socket.AF_UNIX)\n"
        "    fresh.recv(1)\n"
    )

    assert Counter(_operations(result)) == Counter(
        {"socket.connect": 1, "socket.bind": 1}
    )
    assert sum(
        row["reason"] == "invalid_unix_socket_lifecycle_transition"
        for row in result["escape_facts"]
    ) == 2


def test_socket_constructor_method_and_address_negatives_fail_closed() -> None:
    result = _discover(
        "import socket\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def negatives(family, socket_type):\n"
        "    socket.socket().connect(SOCKET_PATH)\n"
        "    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(SOCKET_PATH)\n"
        "    socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM).connect(SOCKET_PATH)\n"
        "    socket.socket(socket.AF_UNIX, socket.SOCK_RAW).connect(SOCKET_PATH)\n"
        "    socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET).connect(SOCKET_PATH)\n"
        "    socket.socket(family, socket.SOCK_STREAM).connect(SOCKET_PATH)\n"
        "    socket.socket(socket.AF_UNIX, socket_type).connect(SOCKET_PATH)\n"
        "    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 1).connect(SOCKET_PATH)\n"
        "    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=3).connect(SOCKET_PATH)\n"
        "    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "    client.connect_ex(SOCKET_PATH)\n"
        "    client.recv(1)\n"
        "    socket.socketpair()\n"
    )

    assert result["access_facts"] == []
    assert all(
        row["operation"] != "socket.connect"
        for row in result["access_facts"]
    )


def test_abstract_nul_unix_address_is_never_a_filesystem_fact() -> None:
    abstract = _discover(
        "import socket\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def exercise():\n"
        "    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "    client.connect(SOCKET_PATH)\n",
        locator="unix://\0chronovisor-test",
    )

    assert abstract["access_facts"] == []
    assert any(
        row["reason"] == "dynamic_or_unsupported_unix_socket_endpoint"
        for row in abstract["escape_facts"]
    )


def test_socketserver_constructor_true_false_and_lifecycle() -> None:
    result = _discover(
        "import socketserver\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "class Handler(socketserver.StreamRequestHandler):\n"
        "    pass\n"
        "class Server(socketserver.ThreadingUnixStreamServer):\n"
        "    daemon_threads = True\n"
        "    allow_reuse_address = True\n"
        "def immediate():\n"
        "    server = Server(SOCKET_PATH, Handler)\n"
        "    server.serve_forever(poll_interval=0.25)\n"
        "    server.shutdown()\n"
        "    server.server_close()\n"
        "def delayed():\n"
        "    server = socketserver.UnixStreamServer(\n"
        "        SOCKET_PATH, Handler, bind_and_activate=False\n"
        "    )\n"
        "    server.server_bind()\n"
        "    server.server_activate()\n"
        "    server.serve_forever()\n"
        "    server.shutdown()\n"
        "    server.server_close()\n"
    )

    assert Counter(_operations(result)) == Counter(
        {
            "socketserver.constructor.server_bind": 1,
            "socketserver.constructor.server_activate": 1,
            "socketserver.server_bind": 1,
            "socketserver.server_activate": 1,
            "socketserver.serve_forever": 2,
        }
    )
    assert sum(
        row["operation"] == "socketserver.request_handler_boundary"
        for row in result["escape_facts"]
    ) == 2
    assert sum(
        row["reason"] == "synchronous_socketserver_shutdown_would_deadlock"
        for row in result["escape_facts"]
    ) == 2


def test_socketserver_thread_shutdown_boundary_preserves_main_lifecycle() -> None:
    result = _discover(
        "import socketserver\n"
        "import threading\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "class Handler:\n"
        "    pass\n"
        "class Server(socketserver.ThreadingUnixStreamServer):\n"
        "    daemon_threads = True\n"
        "def exercise():\n"
        "    server = Server(SOCKET_PATH, Handler)\n"
        "    threading.Thread(target=server.shutdown, daemon=True).start()\n"
        "    server.serve_forever()\n"
        "    server.server_close()\n"
    )

    assert Counter(_operations(result)) == Counter(
        {
            "socketserver.constructor.server_bind": 1,
            "socketserver.constructor.server_activate": 1,
            "socketserver.shutdown": 1,
            "socketserver.serve_forever": 1,
        }
    )
    assert any(
        row["reason"] == "unix_socket_shutdown_thread_callback_boundary"
        for row in result["escape_facts"]
    )


def test_signal_handler_shutdown_is_isolated_may_access() -> None:
    result = _discover(
        "import signal\n"
        "import socketserver\n"
        "import threading\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "class Handler:\n"
        "    pass\n"
        "def exercise():\n"
        "    server = socketserver.UnixStreamServer(SOCKET_PATH, Handler)\n"
        "    def stop(_signum, _frame):\n"
        "        threading.Thread(target=server.shutdown, daemon=True).start()\n"
        "    signal.signal(signal.SIGTERM, stop)\n"
        "    server.serve_forever()\n"
        "    server.server_close()\n"
    )

    assert Counter(_operations(result)) == Counter(
        {
            "socketserver.constructor.server_bind": 1,
            "socketserver.constructor.server_activate": 1,
            "socketserver.shutdown": 1,
            "socketserver.serve_forever": 1,
        }
    )
    shutdown = next(
        row
        for row in result["access_facts"]
        if row["operation"] == "socketserver.shutdown"
    )
    assert shutdown["actors"] == ["chronovisor.consumer:exercise.<locals>.stop"]
    assert any(
        row["reason"] == "unix_socket_shutdown_thread_callback_boundary"
        for row in result["escape_facts"]
    )


def test_signal_unknown_handler_and_immediate_shutdown_remain_fail_closed() -> None:
    result = _discover(
        "import signal\n"
        "import socketserver\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "class Handler:\n"
        "    pass\n"
        "def exercise(callback):\n"
        "    server = socketserver.UnixStreamServer(SOCKET_PATH, Handler)\n"
        "    def stop(_signum, _frame):\n"
        "        callback(server)\n"
        "    signal.signal(signal.SIGTERM, stop)\n"
        "    server.shutdown()\n"
        "    server.serve_forever()\n"
    )

    assert "socketserver.shutdown" not in _operations(result)
    assert "socketserver.serve_forever" not in _operations(result)
    assert {
        row["reason"]
        for row in result["escape_facts"]
        if row["operation"].startswith("socketserver.shutdown")
    } >= {
        "unknown_or_ambiguous_signal_handler_callback",
        "synchronous_socketserver_shutdown_would_deadlock",
    }


def test_default_socket_branch_is_may_access_but_override_alone_is_not() -> None:
    possible_default = _discover(
        "from pathlib import Path\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def choose(flag):\n"
        "    return SOCKET_PATH if flag else 'custom.sock'\n"
        "def exercise(flag):\n"
        "    Path(choose(flag)).exists()\n"
    )
    override_only = _discover(
        "from pathlib import Path\n"
        "def exercise():\n"
        "    Path('custom.sock').exists()\n"
    )

    assert _operations(possible_default) == ["path.exists"]
    assert override_only["access_facts"] == []


def test_structured_wrappers_never_leak_scalar_path_candidates() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def choose(flag):\n"
        "    return SOCKET_PATH if flag else 'custom.sock'\n"
        "def exercise(flag):\n"
        "    Path([choose(flag)]).exists()\n"
        "    Path((choose(flag),)).exists()\n"
        "    Path({choose(flag)}).exists()\n"
        "    Path({'socket': choose(flag)}).exists()\n"
        "    Path(list([choose(flag)])).exists()\n"
        "    Path(tuple((choose(flag),))).exists()\n"
        "    Path(set({choose(flag)})).exists()\n"
        "    Path(dict({'socket': choose(flag)})).exists()\n"
    )

    assert result["access_facts"] == []
    assert sum(
        row["reason"] == "invalid_or_ambiguous_path_constructor_signature"
        for row in result["escape_facts"]
    ) == 8


def test_simultaneous_variant_expressions_and_fstrings_fail_closed() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def opaque():\n"
        "    return mystery()\n"
        "def choose(flag):\n"
        "    return SOCKET_PATH if flag else opaque()\n"
        "def exercise(first, second):\n"
        "    combined = choose(first) + choose(second)\n"
        "    Path(combined).exists()\n"
        "    formatted = f'{choose(first)}{choose(second)}'\n"
        "    Path(formatted).exists()\n"
    )

    assert result["access_facts"] == []
    assert any(
        row["reason"] == "unsupported_registered_origin_expression"
        for row in result["escape_facts"]
    )


def test_socketserver_subclass_mutation_invalidates_all_aliases() -> None:
    mutations = [
        "Server.server_bind = lambda self: None",
        "setattr(Server, 'server_bind', lambda self: None)",
        "delattr(Server, 'server_bind')",
        "del Server.server_bind",
        "Server.__dict__.update({'server_bind': lambda self: None})",
        "vars(Server).update({'server_bind': lambda self: None})",
    ]
    for mutation in mutations:
        result = _discover(
            "import socketserver\n"
            "from chronovisor.resources import SOCKET_PATH\n"
            "class Handler:\n"
            "    pass\n"
            "class Server(socketserver.UnixStreamServer):\n"
            "    pass\n"
            "Captured = Server\n"
            f"{mutation}\n"
            "def exercise():\n"
            "    Captured(SOCKET_PATH, Handler)\n"
        )
        assert result["access_facts"] == [], mutation
        assert any(
            row["reason"] == "unsafe_or_mutated_unix_stream_server_subclass"
            for row in result["escape_facts"]
        ), mutation


def test_socketserver_local_shadow_is_not_stdlib_but_prior_capture_is_exact() -> None:
    shadowed = _discover(
        "import socketserver\n"
        "from socketserver import UnixStreamServer\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "class Handler:\n"
        "    pass\n"
        "def exercise():\n"
        "    socketserver.UnixStreamServer(SOCKET_PATH, Handler)\n"
        "    UnixStreamServer(SOCKET_PATH, Handler)\n",
        extra_sources={
            "src/socketserver.py": (
                "class UnixStreamServer:\n"
                "    def __init__(self, address, handler):\n"
                "        pass\n"
            )
        },
    )
    captured = _discover(
        "import socketserver\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "class Handler:\n"
        "    pass\n"
        "Captured = socketserver.UnixStreamServer\n"
        "socketserver.UnixStreamServer = object()\n"
        "def exercise():\n"
        "    Captured(SOCKET_PATH, Handler)\n"
    )

    assert shadowed["access_facts"] == []
    assert Counter(_operations(captured)) == Counter(
        {
            "socketserver.constructor.server_bind": 1,
            "socketserver.constructor.server_activate": 1,
        }
    )


def test_socketserver_arbitrary_subclasses_tcp_and_http_are_not_concrete() -> None:
    result = _discover(
        "import socketserver\n"
        "from http.server import ThreadingHTTPServer\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "class Handler:\n"
        "    pass\n"
        "class Unsafe(socketserver.ThreadingUnixStreamServer):\n"
        "    def server_bind(self):\n"
        "        super().server_bind()\n"
        "def exercise():\n"
        "    Unsafe(SOCKET_PATH, Handler)\n"
        "    socketserver.TCPServer(SOCKET_PATH, Handler)\n"
        "    ThreadingHTTPServer(SOCKET_PATH, Handler)\n"
    )

    assert result["access_facts"] == []


def test_os_chmod_and_path_chmod_are_same_resource_writes() -> None:
    result = _discover(
        "import os\n"
        "from pathlib import Path\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def exercise():\n"
        "    os.chmod(SOCKET_PATH, 0o600)\n"
        "    Path(SOCKET_PATH).chmod(0o600)\n"
    )

    assert _operations(result) == ["os.chmod", "path.chmod"]
    assert {row["resource_id"] for row in result["access_facts"]} == {RESOURCE_ID}


def test_socket_ids_are_line_stable() -> None:
    source = (
        "import socket\n"
        "from chronovisor.resources import SOCKET_PATH\n"
        "def exercise():\n"
        "    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "    client.connect(SOCKET_PATH)\n"
        "    client.sendall(b'x')\n"
        "    client.recv(1)\n"
    )
    base = _discover(source)
    shifted = _discover("\n\n" + source)

    assert base["access_fact_ids"] == shifted["access_fact_ids"]
    assert base["escape_fact_ids"] == shifted["escape_fact_ids"]


def test_production_socket_inventory_is_exact_and_excludes_tcp_helpers() -> None:
    repository = Path(__file__).resolve().parents[1]
    selected = [
        "src/chronovisor/core/runtime_config.py",
        "src/chronovisor/search/semantic_client.py",
        "src/chronovisor/search/reranker_client.py",
        "src/chronovisor/search/semantic_service.py",
        "src/chronovisor/search/reranker_service.py",
    ]
    snapshot = {path: (repository / path).read_bytes() for path in selected}
    candidates = [
        _candidate(
            resource_id="runtime-resource:semantic-socket",
            module="chronovisor.search.semantic_service",
            symbol="serve",
            locator="unix://$HOME/.chronovisor/runtime/semantic.sock",
        ),
        _candidate(
            resource_id="runtime-resource:reranker-socket",
            module="chronovisor.search.reranker_service",
            symbol="serve",
            locator="unix://$HOME/.chronovisor/runtime/reranker.sock",
        ),
    ]
    result = discover_access_facts(snapshot, candidates)

    counts = Counter(row["resource_id"] for row in result["access_facts"])
    assert counts == {
        "runtime-resource:semantic-socket": 11,
        "runtime-resource:reranker-socket": 11,
    }, (
        result["counts"],
        Counter(
            (tuple(row["actors"]), row["operation"], row["reason"])
            for row in result["escape_facts"]
        ),
        [
            (row["actor"], row["operation"], row["binding_chain"])
            for row in joined_escape_rows(result)
            if row["operation"] == "path.constructor"
            and row["actor"].endswith(":serve")
        ][:10],
    )
    expected = Counter(
        {
            "path.exists": 1,
            "socket.connect": 1,
            "socket.sendall": 1,
            "socket.recv": 1,
            "path.unlink": 2,
            "os.chmod": 1,
            "socketserver.constructor.server_bind": 1,
            "socketserver.constructor.server_activate": 1,
            "socketserver.shutdown": 1,
            "socketserver.serve_forever": 1,
        }
    )
    for resource_id in counts:
        assert Counter(
            row["operation"]
            for row in result["access_facts"]
            if row["resource_id"] == resource_id
        ) == expected
    assert all(
        not str(row["sink"]).startswith(("http.", "socket.get"))
        for row in result["access_facts"]
    )

    raw_calls: Counter[str] = Counter()
    for path in selected[1:]:
        tree = ast.parse(snapshot[path], filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                raw_calls[node.func.attr] += 1
    assert raw_calls["connect"] == 2
    assert raw_calls["sendall"] == 2
    assert raw_calls["recv"] == 2
    assert raw_calls["serve_forever"] == 2
    assert raw_calls["server_close"] == 2
