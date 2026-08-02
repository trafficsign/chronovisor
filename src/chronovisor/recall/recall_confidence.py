"""Deterministic, fail-closed confidence evidence for Recall promotion."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any


def manifest_sha256(values: Sequence[str]) -> str:
    """Seal ordered privacy-safe sample identities."""

    encoded = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def wilson_interval(
    successes: int,
    samples: int,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return a strict Wilson interval; invalid input has no point fallback."""

    if (
        isinstance(successes, bool)
        or isinstance(samples, bool)
        or not isinstance(successes, int)
        or not isinstance(samples, int)
        or samples <= 0
        or successes < 0
        or successes > samples
        or not 0.0 < confidence < 1.0
    ):
        return {
            "valid": False,
            "method": "wilson-score",
            "confidence": confidence,
            "samples": max(0, samples) if isinstance(samples, int) else 0,
            "successes": successes,
            "point": None,
            "lower": None,
            "upper": None,
        }
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    point = successes / samples
    denominator = 1.0 + z * z / samples
    center = (point + z * z / (2.0 * samples)) / denominator
    radius = (
        z
        * math.sqrt(
            point * (1.0 - point) / samples + z * z / (4.0 * samples * samples)
        )
        / denominator
    )
    return {
        "valid": True,
        "method": "wilson-score",
        "confidence": confidence,
        "samples": samples,
        "successes": successes,
        "point": round(point, 9),
        "lower": round(max(0.0, center - radius), 9),
        "upper": round(min(1.0, center + radius), 9),
    }


def cluster_bootstrap_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    cluster_keys: Sequence[str] = ("session_hash", "query_sha256", "content_sha256"),
    confidence: float = 0.95,
    seed: int = 1729,
    draws: int = 4000,
    value_floor: float = -1.0,
    value_ceiling: float = 1.0,
) -> dict[str, Any]:
    """Bootstrap independent connected clusters, never individual rows.

    Session, query, and content identities form union-find components. Sampling
    those components prevents a repeated session/query/page revision from being
    mistaken for independent evidence.
    """

    if not 0.0 < confidence < 1.0 or draws < 100 or not rows:
        return _invalid_bootstrap(confidence, seed, draws)
    clustered = _connected_cluster_values(
        rows,
        value_key=value_key,
        cluster_keys=cluster_keys,
        value_floor=value_floor,
        value_ceiling=value_ceiling,
    )
    if clustered is None:
        return _invalid_bootstrap(confidence, seed, draws)
    cluster_values, sample_count = clustered
    if len(cluster_values) < 2:
        return _invalid_bootstrap(
            confidence,
            seed,
            draws,
            clusters=len(cluster_values),
            samples=sample_count,
        )
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(cluster_values) for _ in cluster_values) / len(cluster_values)
        for _draw in range(draws)
    )
    alpha = (1.0 - confidence) / 2.0
    lower_index = min(len(estimates) - 1, max(0, int(alpha * len(estimates))))
    upper_index = min(
        len(estimates) - 1,
        max(0, int((1.0 - alpha) * len(estimates)) - 1),
    )
    point = sum(cluster_values) / len(cluster_values)
    return {
        "valid": True,
        "method": "connected-cluster-bootstrap-percentile",
        "confidence": confidence,
        "seed": seed,
        "draws": draws,
        "samples": sample_count,
        "clusters": len(cluster_values),
        "cluster_keys": list(cluster_keys),
        "point": round(point, 9),
        "lower": round(estimates[lower_index], 9),
        "upper": round(estimates[upper_index], 9),
    }


def cluster_rate_wilson_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    success_threshold: float,
    cluster_keys: Sequence[str] = (
        "session_hash",
        "query_sha256",
        "content_sha256",
    ),
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return a conservative Wilson bound over independent clusters.

    A cluster counts as successful only when its mean rate meets the declared
    point threshold. Perfect observed clusters therefore retain finite-sample
    uncertainty instead of receiving a lower bound of exactly 1.0.
    """

    if not 0.0 <= success_threshold <= 1.0:
        return {
            **wilson_interval(0, 0, confidence=confidence),
            "valid": False,
            "method": "connected-cluster-wilson-score",
            "clusters": 0,
            "success_threshold": success_threshold,
        }
    clustered = _connected_cluster_values(
        rows,
        value_key=value_key,
        cluster_keys=cluster_keys,
        value_floor=0.0,
        value_ceiling=1.0,
    )
    if clustered is None:
        return {
            **wilson_interval(0, 0, confidence=confidence),
            "valid": False,
            "method": "connected-cluster-wilson-score",
            "clusters": 0,
            "success_threshold": success_threshold,
        }
    cluster_values, sample_count = clustered
    successes = sum(value >= success_threshold for value in cluster_values)
    bound = wilson_interval(successes, len(cluster_values), confidence=confidence)
    return {
        **bound,
        "method": "connected-cluster-wilson-score",
        "samples": sample_count,
        "clusters": len(cluster_values),
        "cluster_successes": successes,
        "success_threshold": success_threshold,
        "cluster_point_mean": round(
            sum(cluster_values) / len(cluster_values), 9
        )
        if cluster_values
        else None,
    }


def _connected_cluster_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    cluster_keys: Sequence[str],
    value_floor: float,
    value_ceiling: float,
) -> tuple[list[float], int] | None:
    """Return component means, rejecting every row without durable identity."""

    if not rows:
        return None
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    row_nodes: list[list[str]] = []
    values: list[float] = []
    for row in rows:
        raw_value = row.get(value_key)
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, int | float)
            or not math.isfinite(float(raw_value))
            or not value_floor <= float(raw_value) <= value_ceiling
        ):
            return None
        values.append(float(raw_value))
        explicit_nodes = row.get("cluster_nodes")
        nodes = [
            f"{key}:{row.get(key)}"
            for key in cluster_keys
            if isinstance(row.get(key), str) and str(row.get(key)).strip()
        ]
        if isinstance(explicit_nodes, list):
            nodes.extend(
                f"node:{value}"
                for value in explicit_nodes
                if isinstance(value, str) and value.strip()
            )
        nodes = list(dict.fromkeys(nodes))
        if not nodes:
            return None
        for node in nodes:
            find(node)
        for node in nodes[1:]:
            union(nodes[0], node)
        row_nodes.append(nodes)

    clusters: dict[str, list[float]] = defaultdict(list)
    for nodes, value in zip(row_nodes, values, strict=True):
        clusters[find(nodes[0])].append(value)
    cluster_values = [
        sum(cluster) / len(cluster)
        for _root, cluster in sorted(clusters.items())
    ]
    return cluster_values, len(rows)


def _invalid_bootstrap(
    confidence: float,
    seed: int,
    draws: int,
    *,
    clusters: int = 0,
    samples: int = 0,
) -> dict[str, Any]:
    return {
        "valid": False,
        "method": "connected-cluster-bootstrap-percentile",
        "confidence": confidence,
        "seed": seed,
        "draws": draws,
        "samples": samples,
        "clusters": clusters,
        "point": None,
        "lower": None,
        "upper": None,
    }
