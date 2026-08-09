"""Host-neutral helpers for parsing source-native transcript records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                yield line_no, parsed


def content_has_capture_payload(content: Any) -> bool:
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return bool(content)
    return True
