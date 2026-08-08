from __future__ import annotations

import ast
import builtins
import contextvars
import random
import sys
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from scripts.runtime_ownership import access_model
from scripts.runtime_ownership.access import discover_access_facts
from scripts.runtime_ownership.access_facts import AccessFactCollector
from scripts.runtime_ownership.access_model import (
    MAX_FLOW_REPR_CACHE_BUCKET_ENTRIES,
    MAX_FLOW_REPR_CACHE_BYTES,
    MAX_FLOW_VARIANTS,
    FlowValue,
    SyntaxSite,
    _current_flow_repr_cache,
    _flow_key_repr,
    _flow_value_key,
    _FlowReprCacheStats,
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


def _random_flow_value(
    generator: random.Random,
    *,
    depth: int = 0,
) -> FlowValue:
    token = generator.randrange(8)
    origins: dict[str, frozenset[tuple[str, ...]]] = (
        {
            RESOURCE_ID: frozenset(
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
        module_refs={f"module:{token % 3}"} if token % 3 else set(),
        unknown_callable=bool(token % 2),
        structured_items=structured_items,
        attribute_values=attribute_values,
        attribute_values_complete=bool(attribute_values) and token % 2 == 0,
        attribute_values_ambiguous=bool(attribute_values) and token % 3 == 0,
        variants=variants,
    )


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


def test_flow_key_repr_cache_hit_reuses_admitted_builtin_repr() -> None:
    value = FlowValue(
        origins={RESOURCE_ID: frozenset({("shared",)})},
        object_types={"counted"},
    )
    key = _flow_value_key(value)
    expected = repr(key)
    repr_calls: list[object] = []
    hash_calls: list[object] = []
    real_repr = builtins.repr
    real_hash = builtins.hash

    def counted_repr(value: object) -> str:
        repr_calls.append(value)
        return real_repr(value)

    def counted_hash(value: object) -> int:
        hash_calls.append(value)
        return real_hash(value)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "repr", counted_repr)
        patch.setattr(builtins, "hash", counted_hash)
        with _scoped_flow_repr_cache() as cache:
            first = _flow_key_repr(key)
            second = _flow_key_repr(key)
            third = _flow_key_repr(key)
            active_stats = cache.stats()

    assert first == second == third == expected
    assert repr_calls == [key, key]
    assert hash_calls == [key, key, key]
    assert active_stats.hits == 1
    assert active_stats.misses == 2
    assert active_stats.skips == 1
    assert cache.stats().closed is True
    assert cache.stats().current_entries == 0

    with pytest.MonkeyPatch.context() as patch:
        repr_calls.clear()
        patch.setattr(builtins, "repr", counted_repr)
        _flow_key_repr(key)
        _flow_key_repr(key)
    assert repr_calls == [key, key]


def test_flow_key_cache_misses_for_every_key_field_except_variants() -> None:
    nested = FlowValue(object_types={"nested"})
    base = FlowValue(
        origins={RESOURCE_ID: frozenset({("base",)})},
        object_types={"object"},
        overflowed=frozenset({"overflow"}),
        module_refs={"module"},
        call_targets={"call"},
        class_targets={"class"},
        unknown_callable=True,
        closure_instances={("closure", "instance")},
        structured_items=(nested,),
        runtime_object_ids={"runtime-object"},
        runtime_close_ids={"runtime-close"},
        runtime_descriptor_ids={"runtime-descriptor"},
        instance_ids={"instance"},
        attribute_values={"attribute": nested.copy()},
        attribute_values_complete=True,
        attribute_values_ambiguous=True,
        variants=(FlowValue(object_types={"ignored-variant"}),),
        has_originless_alternative=True,
        variant_tainted_resource_ids=frozenset({"tainted"}),
    )
    replacements: dict[str, object] = {
        "origins": {RESOURCE_ID: frozenset({("changed",)})},
        "object_types": {"changed"},
        "overflowed": frozenset({"changed"}),
        "module_refs": {"changed"},
        "call_targets": {"changed"},
        "class_targets": {"changed"},
        "unknown_callable": False,
        "closure_instances": {("changed", "instance")},
        "structured_items": (FlowValue(object_types={"changed"}),),
        "runtime_object_ids": {"changed"},
        "runtime_close_ids": {"changed"},
        "runtime_descriptor_ids": {"changed"},
        "instance_ids": {"changed"},
        "attribute_values": {"attribute": FlowValue(object_types={"changed"})},
        "attribute_values_complete": False,
        "attribute_values_ambiguous": False,
        "has_originless_alternative": False,
        "variant_tainted_resource_ids": frozenset({"changed"}),
    }
    assert set(FlowValue.__dataclass_fields__) == {*replacements, "variants"}
    base_key = _flow_value_key(base)

    with _scoped_flow_repr_cache() as cache:
        _flow_key_repr(base_key)
        variants_only = base.copy()
        variants_only.variants = (FlowValue(object_types={"changed"}),)
        assert _flow_value_key(variants_only) == base_key
        _flow_key_repr(_flow_value_key(variants_only))
        _flow_key_repr(base_key)
        for field, replacement in replacements.items():
            changed = base.copy()
            setattr(changed, field, replacement)
            changed_key = _flow_value_key(changed)
            assert changed_key != base_key, field
            _flow_key_repr(changed_key)
        stats = cache.stats()

    assert stats.hits == 1
    assert stats.misses == len(replacements) + 2
    assert stats.skips == len(replacements) + 1


def test_flow_key_cache_entry_and_byte_caps_skip_without_eviction() -> None:
    values = [
        FlowValue(origins={RESOURCE_ID: frozenset({(f"entry:{index}",)})})
        for index in range(3)
    ]
    expected = [_normalize_flow_variants((value,)) for value in values]

    with _scoped_flow_repr_cache(max_entries=2, max_bytes=1_000_000) as cache:
        actual = [_normalize_flow_variants((value,)) for value in values]
        _normalize_flow_variants((values[0],))
        _normalize_flow_variants((values[1],))
        _normalize_flow_variants((values[2],))
        _normalize_flow_variants((values[0],))
        _normalize_flow_variants((values[2],))
        entry_stats = cache.stats()
    assert actual == expected
    assert entry_stats.hits == 1
    assert entry_stats.misses == 7
    assert entry_stats.skips == 5
    assert entry_stats.peak_entries == 2
    assert entry_stats.peak_bytes <= 1_000_000

    with _scoped_flow_repr_cache(max_entries=2, max_bytes=1) as cache:
        first = _normalize_flow_variants((values[0],))
        second = _normalize_flow_variants((values[0],))
        third = _normalize_flow_variants((values[0],))
        byte_stats = cache.stats()
    assert first == second == third == expected[0]
    assert byte_stats.hits == 0
    assert byte_stats.misses == 3
    assert byte_stats.skips == 3
    assert byte_stats.peak_entries == 0
    assert byte_stats.peak_bytes == 0

    with _scoped_flow_repr_cache(
        max_entries=2,
        max_bytes=1_000_000,
        max_fingerprints=1,
    ) as cache:
        _normalize_flow_variants((values[0],))
        _normalize_flow_variants((values[1],))
        _normalize_flow_variants((values[1],))
        fingerprint_stats = cache.stats()
    assert fingerprint_stats.current_fingerprints == 1
    assert fingerprint_stats.peak_fingerprints == 1
    assert fingerprint_stats.peak_entries == 0
    assert fingerprint_stats.misses == 3
    assert fingerprint_stats.skips == 3


def test_flow_key_cache_collision_never_reuses_a_distinct_key_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _flow_value_key(
        FlowValue(origins={RESOURCE_ID: frozenset({("collision:first",)})})
    )
    second = _flow_value_key(
        FlowValue(origins={RESOURCE_ID: frozenset({("collision:second",)})})
    )
    assert first != second
    monkeypatch.setattr(builtins, "hash", lambda _value: 7)

    with _scoped_flow_repr_cache() as cache:
        first_rendered = _flow_key_repr(first)
        second_rendered = _flow_key_repr(second)
        first_again = _flow_key_repr(first)
        second_again = _flow_key_repr(second)
        stats = cache.stats()

    assert first_rendered == first_again == repr(first)
    assert second_rendered == second_again == repr(second)
    assert first_rendered != second_rendered
    assert stats.peak_fingerprints == 1
    assert stats.peak_entries == 2
    assert stats.hits == 1
    assert stats.misses == 3
    assert stats.skips == 1


def test_flow_key_cache_collision_bucket_is_exactly_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = [
        _flow_value_key(
            FlowValue(origins={RESOURCE_ID: frozenset({(f"bucket:{index}",)})})
        )
        for index in range(MAX_FLOW_REPR_CACHE_BUCKET_ENTRIES + 4)
    ]
    expected = [repr(key) for key in keys]
    monkeypatch.setattr(builtins, "hash", lambda _value: 11)

    with _scoped_flow_repr_cache() as cache:
        first = [_flow_key_repr(key) for key in keys]
        second = [_flow_key_repr(key) for key in keys]
        stats = cache.stats()

    assert first == second == expected
    assert stats.peak_fingerprints == 1
    assert stats.peak_entries == MAX_FLOW_REPR_CACHE_BUCKET_ENTRIES
    assert stats.hits == MAX_FLOW_REPR_CACHE_BUCKET_ENTRIES
    assert stats.misses == 2 * len(keys) - stats.hits
    assert stats.skips == 8


def test_flow_key_cache_accounting_bounds_retained_container_graph() -> None:
    def recursive_size(value: object, seen: set[int]) -> int:
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(value)
        if isinstance(value, dict):
            return size + sum(
                recursive_size(key, seen) + recursive_size(item, seen)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple, set, frozenset)):
            return size + sum(recursive_size(item, seen) for item in value)
        return size

    def retained_graph_size(cache: access_model._FlowReprCache) -> int:
        seen: set[int] = set()
        return recursive_size(cache._seen_fingerprints, seen) + recursive_size(
            cache._buckets, seen
        )

    for token in (
        "ascii:" + "x" * 10_000,
        "unicode:" + "漢字🙂" * 3_000,
    ):
        value = FlowValue(
            origins={RESOURCE_ID: frozenset({(token,)})},
            object_types={token},
        )
        with _scoped_flow_repr_cache() as cache:
            _normalize_flow_variants((value,))
            _normalize_flow_variants((value,))
            retained_graph_bytes = retained_graph_size(cache)
            stats = cache.stats()
        assert retained_graph_bytes <= stats.current_bytes
        assert stats.current_bytes <= MAX_FLOW_REPR_CACHE_BYTES
        assert stats.current_fingerprints == 1
        assert stats.current_entries == 1


def test_flow_key_cache_rejects_unknown_objects_from_full_key_retention() -> None:
    class HugeHashable:
        def __init__(self) -> None:
            self.payload = "x" * (MAX_FLOW_REPR_CACHE_BYTES * 2)

        def __hash__(self) -> int:
            return 29

    huge = HugeHashable()
    key = ("unknown", huge)
    expected = repr(key)
    with _scoped_flow_repr_cache() as cache:
        assert _flow_key_repr(key) == expected
        assert _flow_key_repr(key) == expected
        assert _flow_key_repr(key) == expected
        stats = cache.stats()

    assert stats.hits == 0
    assert stats.misses == 3
    assert stats.skips == 3
    assert stats.current_entries == 0
    assert stats.current_fingerprints == 1
    assert stats.current_bytes < MAX_FLOW_REPR_CACHE_BYTES


def test_flow_key_cache_scope_resets_context_when_close_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CloseFailure(BaseException):
        pass

    captured: list[access_model._FlowReprCache] = []

    def fail_close(cache: access_model._FlowReprCache) -> None:
        captured.append(cache)
        raise CloseFailure("close failed")

    assert _current_flow_repr_cache() is None
    with monkeypatch.context() as close_patch:
        close_patch.setattr(access_model._FlowReprCache, "close", fail_close)
        with pytest.raises(CloseFailure, match="close failed"):
            with _scoped_flow_repr_cache() as cache:
                assert _current_flow_repr_cache() is cache
    assert _current_flow_repr_cache() is None
    assert captured == [cache]
    cache.close()


def test_flow_key_cache_nested_scope_restores_outer() -> None:
    value = FlowValue(origins={RESOURCE_ID: frozenset({("nested-scope",)})})
    assert _current_flow_repr_cache() is None
    with _scoped_flow_repr_cache() as outer:
        assert _current_flow_repr_cache() is outer
        _normalize_flow_variants((value,))
        with _scoped_flow_repr_cache() as inner:
            assert _current_flow_repr_cache() is inner
            _normalize_flow_variants((value,))
        assert inner.stats().closed is True
        assert _current_flow_repr_cache() is outer
        _normalize_flow_variants((value,))
        _normalize_flow_variants((value,))
        assert outer.stats().hits == 1
        assert outer.stats().misses == 2
        assert outer.stats().skips == 1
    assert outer.stats().closed is True
    assert outer.stats().current_entries == 0
    assert _current_flow_repr_cache() is None


def test_copied_contexts_share_locked_cache_and_retain_no_entries_after_close() -> None:
    value = FlowValue(origins={RESOURCE_ID: frozenset({("shared-context",)})})
    barrier = threading.Barrier(3)

    def exercise() -> tuple[tuple[FlowValue, ...], bool]:
        barrier.wait()
        result = _normalize_flow_variants((value,))
        for _ in range(7):
            assert _normalize_flow_variants((value,)) == result
        return result

    with _scoped_flow_repr_cache() as cache:
        contexts = [contextvars.copy_context(), contextvars.copy_context()]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(context.run, exercise) for context in contexts]
            barrier.wait()
            results = [future.result() for future in futures]
        stats = cache.stats()

    assert results[0] == results[1]
    assert stats.misses == 2
    assert stats.hits == 14
    assert stats.skips == 1
    assert cache.stats().closed is True
    assert cache.stats().current_entries == 0
    assert cache.stats().current_fingerprints == 0
    assert all(context.run(_current_flow_repr_cache) is cache for context in contexts)
    for context in contexts:
        assert context.run(_normalize_flow_variants, (value,)) == results[0]
    assert cache.stats().current_entries == 0


def test_flow_key_cache_stats_and_outputs_are_reversed_input_deterministic() -> None:
    values = [
        FlowValue(origins={RESOURCE_ID: frozenset({(f"reverse:{index}",)})})
        for index in range(MAX_FLOW_VARIANTS + 5)
    ]

    def normalize(
        sequence: Sequence[FlowValue],
    ) -> tuple[tuple[tuple[FlowValue, ...], bool], _FlowReprCacheStats]:
        with _scoped_flow_repr_cache() as cache:
            result = _normalize_flow_variants(sequence)
            stats = cache.stats()
        return result, stats

    forward, forward_stats = normalize(values)
    reverse, reverse_stats = normalize(tuple(reversed(values)))
    assert forward == reverse
    assert forward_stats == reverse_stats
    assert forward_stats.current_bytes > 0


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
        with _scoped_flow_repr_cache():
            cached, cached_overflow = _normalize_flow_variants(values)
        assert actual == expected
        assert cached == expected
        assert tuple(map(_flow_value_key, actual)) == tuple(
            map(_flow_value_key, expected)
        )
        assert actual_overflow is expected_overflow
        assert cached_overflow is expected_overflow


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
    collector = AccessFactCollector(
        {RESOURCE_ID: "unix://$HOME/.chronovisor/runtime/test.sock"},
        {id(expression.value): site},
    )
    collector.record_access(
        forward,
        node=expression.value,
        actor="chronovisor.consumer:exercise",
        mode="read",
        operation="path.exists",
        sink="pathlib.Path.exists",
        path="src/chronovisor/consumer.py",
        line=1,
        ordinal=1,
    )
    projected = collector.result()
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
