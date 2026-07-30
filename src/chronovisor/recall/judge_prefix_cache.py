"""Bounded cache contract for decoder-judge prefix handles."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefixCacheKey:
    model_revision: str
    template_sha256: str
    position: int
    support_sha256: str

    @classmethod
    def build(
        cls,
        *,
        model_revision: str,
        fixed_prompt: str,
        support_span: str,
        position: int,
    ) -> PrefixCacheKey:
        return cls(
            model_revision=model_revision,
            template_sha256=hashlib.sha256(fixed_prompt.encode()).hexdigest(),
            position=max(0, position),
            support_sha256=hashlib.sha256(support_span.encode()).hexdigest(),
        )


class PrefixHandleCache:
    """LRU for backend-owned handles; page IDs and full pages are forbidden."""

    def __init__(self, max_entries: int = 32) -> None:
        self.max_entries = max(1, max_entries)
        self._items: OrderedDict[PrefixCacheKey, Any] = OrderedDict()

    def get(self, key: PrefixCacheKey) -> Any | None:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: PrefixCacheKey, handle: Any) -> None:
        self._items[key] = handle
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)
