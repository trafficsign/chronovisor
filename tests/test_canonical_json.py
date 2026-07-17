from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from llm_wiki_mcp.canonical_json import (
    canonical_json_bytes_stringifying,
    canonical_json_bytes_strict,
    canonical_json_line_bytes_strict,
    canonical_json_permissive,
    canonical_json_sha256_strict,
    canonical_json_sha256_stringifying,
    canonical_json_sha256_stringifying_strict,
    canonical_json_strict,
    canonical_json_stringifying,
    canonical_json_stringifying_strict,
)


def test_strict_contract_has_stable_utf8_bytes_and_digest() -> None:
    value = {"z": [3, 2, 1], "日本語": "記憶", "a": {"b": True}}
    expected = '{"a":{"b":true},"z":[3,2,1],"日本語":"記憶"}'

    assert canonical_json_strict(value) == expected
    assert canonical_json_sha256_strict(value) == hashlib.sha256(
        expected.encode("utf-8")
    ).hexdigest()
    assert canonical_json_line_bytes_strict(value) == (expected + "\n").encode(
        "utf-8"
    )
    assert canonical_json_bytes_strict(value) == expected.encode("utf-8")


def test_strict_contract_rejects_unknown_values_and_nonfinite_numbers() -> None:
    with pytest.raises(TypeError):
        canonical_json_strict({"path": Path("wiki")})
    with pytest.raises(ValueError):
        canonical_json_strict({"score": math.nan})


def test_stringifying_contract_preserves_legacy_default_str_semantics() -> None:
    value = {"path": Path("wiki/page.md"), "日本語": "記憶"}
    expected = '{"path":"wiki/page.md","日本語":"記憶"}'

    assert canonical_json_stringifying(value) == expected
    assert canonical_json_sha256_stringifying(value) == hashlib.sha256(
        expected.encode("utf-8")
    ).hexdigest()
    assert canonical_json_bytes_stringifying(value) == expected.encode("utf-8")


def test_stringifying_strict_contract_rejects_nonfinite_numbers() -> None:
    value = {"path": Path("wiki/page.md")}
    expected = '{"path":"wiki/page.md"}'

    assert canonical_json_stringifying_strict(value) == expected
    assert canonical_json_sha256_stringifying_strict(value) == hashlib.sha256(
        expected.encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError):
        canonical_json_stringifying_strict({"score": math.inf})


def test_permissive_contract_preserves_legacy_nonfinite_encoding() -> None:
    assert canonical_json_permissive({"score": math.inf}) == '{"score":Infinity}'
