"""Recall-owned port for optional evidence reconstruction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

Observer = Callable[..., tuple[Any | None, dict[str, Any]]]
PayloadBuilder = Callable[[Any, Mapping[str, Any], int], dict[str, Any]]

_observer: Observer | None = None
_payload_builder: PayloadBuilder | None = None


def bind(observer: Observer, payload_builder: PayloadBuilder) -> None:
    global _observer, _payload_builder
    _observer = observer
    _payload_builder = payload_builder


def observe(**kwargs: Any) -> tuple[Any | None, dict[str, Any]] | None:
    return _observer(**kwargs) if _observer is not None else None


def publication_payload(
    packet: Any, metadata: Mapping[str, Any], max_chars: int
) -> dict[str, Any]:
    if _payload_builder is None:
        return {}
    return _payload_builder(packet, metadata, max_chars)
