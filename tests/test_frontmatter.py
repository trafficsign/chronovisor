"""Tests for the frontmatter parse/patch module."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from chronovisor.core.canonical_document import CanonicalDocumentError
from chronovisor.core.frontmatter import (
    canonicalize,
    parse,
    patch,
    propose_nested_resolution,
    review_value,
)
from chronovisor.core.legacy_frontmatter import parse as parse_legacy

# ---------------------------------------------------------------------------
# parse() — empty / no frontmatter
# ---------------------------------------------------------------------------

def test_parse_empty_returns_empty_meta_and_text():
    meta, body = parse("")
    assert meta == {}
    assert body == ""


def test_parse_no_frontmatter_returns_text_as_body():
    text = "# Hello\nbody only\n"
    meta, body = parse(text)
    assert meta == {}
    assert body == text


def test_parse_unclosed_frontmatter_fails_closed():
    text = "---\ntitle: x\n# body\n"
    with pytest.raises(CanonicalDocumentError, match="closing frontmatter"):
        parse(text)


# ---------------------------------------------------------------------------
# parse() — scalar values
# ---------------------------------------------------------------------------

def test_parse_scalar_basic():
    text = "---\ntitle: Hello\nupdated: 2026-05-08\n---\nbody text\n"
    meta, body = parse(text)
    assert meta == {"title": "Hello", "updated": date(2026, 5, 8)}
    assert body == "body text\n"


def test_parse_scalar_with_quotes():
    text = '---\ntitle: "Hello, World"\nauthor: \'Alice\'\n---\nbody\n'
    meta, _ = parse(text)
    assert meta == {"title": "Hello, World", "author": "Alice"}


def test_parse_scalar_strip_whitespace():
    text = "---\ntitle:    spaced value   \n---\nbody\n"
    meta, _ = parse(text)
    assert meta == {"title": "spaced value"}


def test_parse_scalar_value_with_colon():
    text = '---\nnote: "foo: bar: baz"\n---\n'
    meta, _ = parse(text)
    assert meta == {"note": "foo: bar: baz"}


def test_parse_skips_blank_lines_and_lines_without_colon():
    text = "---\ntitle: Hi\n\n# random comment-like line\nupdated: 2026-05-08\n---\n"
    meta, _ = parse(text)
    assert meta == {"title": "Hi", "updated": date(2026, 5, 8)}


# ---------------------------------------------------------------------------
# parse() — inline list
# ---------------------------------------------------------------------------

def test_parse_inline_list():
    text = "---\nraw_keywords: [Anthropic, NVIDIA, Groq]\n---\nbody\n"
    meta, _ = parse(text)
    assert meta == {"raw_keywords": ["Anthropic", "NVIDIA", "Groq"]}


def test_parse_inline_list_empty():
    text = "---\nraw_keywords: []\n---\n"
    meta, _ = parse(text)
    assert meta == {"raw_keywords": []}


def test_parse_inline_list_with_quoted_items():
    text = '---\nraw_keywords: ["Hello, World", \'Alice\']\n---\n'
    meta, _ = parse(text)
    assert meta == {"raw_keywords": ["Hello, World", "Alice"]}


# ---------------------------------------------------------------------------
# parse() — block list
# ---------------------------------------------------------------------------

def test_parse_block_list():
    text = (
        "---\n"
        "raw_keywords:\n"
        "  - Anthropic\n"
        "  - NVIDIA\n"
        "  - Groq\n"
        "---\n"
        "body\n"
    )
    meta, _ = parse(text)
    assert meta == {"raw_keywords": ["Anthropic", "NVIDIA", "Groq"]}


def test_parse_block_list_quoted_items():
    text = (
        "---\n"
        "tags:\n"
        '  - "with, comma"\n'
        "  - 'with quotes'\n"
        "---\n"
    )
    meta, _ = parse(text)
    assert meta == {"tags": ["with, comma", "with quotes"]}


def test_parse_mixed_scalar_and_list():
    text = (
        "---\n"
        "title: Anthropic Analysis\n"
        "updated: 2026-05-08\n"
        "raw_keywords: [Anthropic, NVIDIA]\n"
        "---\n"
    )
    meta, _ = parse(text)
    assert meta == {
        "title": "Anthropic Analysis",
        "updated": date(2026, 5, 8),
        "raw_keywords": ["Anthropic", "NVIDIA"],
    }


# ---------------------------------------------------------------------------
# parse() — preserve unknown keys
# ---------------------------------------------------------------------------

def test_parse_preserves_unknown_keys():
    text = (
        "---\n"
        "title: X\n"
        "custom_field: custom value\n"
        "another: 42\n"
        "---\n"
    )
    meta, _ = parse(text)
    assert meta == {"title": "X", "custom_field": "custom value", "another": 42}


# ---------------------------------------------------------------------------
# patch() — basic add / replace / delete
# ---------------------------------------------------------------------------

def test_patch_add_to_empty():
    text = "body only\n"
    out = patch(text, {"title": "New"})
    assert out == "---\ntitle: New\n---\nbody only\n"


def test_patch_serializes_yaml_unsafe_scalars_and_flow_items():
    text = "---\ntitle: Old\n---\nbody\n"

    out = patch(
        text,
        {
            "title": "Agents-A1: Model Analysis",
            "recall_questions": [
                "What changed?",
                "Which value contains a comma, and why?",
            ],
        },
    )

    meta, body = parse(out)
    assert meta["title"] == "Agents-A1: Model Analysis"
    assert meta["recall_questions"] == [
        "What changed?",
        "Which value contains a comma, and why?",
    ]
    assert body == "body\n"


def test_canonicalize_preserves_body_and_full_yaml_values():
    body = "# Heading\n\nUnchanged body.\n"
    legacy = (
        "---\n"
        'title: "Agents-A1: Model Analysis"\n'
        'recall_questions: ["What changed?", "Why now?"]\n'
        "---\n"
        + body
    )

    normalized = canonicalize(legacy)

    assert parse(normalized)[0] == {
        "title": "Agents-A1: Model Analysis",
        "recall_questions": ["What changed?", "Why now?"],
    }
    assert parse(normalized)[1] == body


def test_patch_replace_existing():
    text = "---\ntitle: Old\nupdated: 2025-01-01\n---\nbody\n"
    out = patch(text, {"title": "New"})
    meta, body = parse(out)
    assert meta["title"] == "New"
    assert meta["updated"] == date(2025, 1, 1)
    assert body == "body\n"


def test_patch_add_new_key():
    text = "---\ntitle: X\n---\nbody\n"
    out = patch(text, {"raw_keywords": ["a", "b"]})
    meta, _ = parse(out)
    assert meta == {"title": "X", "raw_keywords": ["a", "b"]}


def test_patch_delete_key():
    text = "---\ntitle: X\nold: keep_no\n---\nbody\n"
    out = patch(text, {}, deletes=["old"])
    meta, _ = parse(out)
    assert meta == {"title": "X"}


def test_patch_preserves_body_verbatim():
    body = "# Title\n\nMulti-line body\nwith [[wiki-link]] inside.\n```\ncode\n```\n"
    text = "---\ntitle: X\n---\n" + body
    out = patch(text, {"updated": "2026-05-08"})
    _, out_body = parse(out)
    assert out_body == body


def test_patch_no_frontmatter_no_changes_returns_body():
    text = "just body\n"
    out = patch(text, {})
    # No updates, no existing meta — the function returns just the body.
    assert out == "just body\n"


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------

def test_round_trip_scalar_and_list():
    text = (
        "---\n"
        "title: Round Trip\n"
        "updated: 2026-05-08\n"
        "raw_keywords: [Anthropic, NVIDIA]\n"
        "---\n"
        "body content here\n"
    )
    meta, body = parse(text)
    rebuilt = patch(body, meta)  # apply the parsed meta to a body-only doc
    meta2, body2 = parse(rebuilt)
    assert meta2 == meta
    assert body2 == body


def test_full_yaml_parse_patch_and_canonicalize_preserve_body():
    body = "# Body\r\n\r\nExact bytes stay here.  \r\n日本語\r\n"
    text = (
        "---\n"
        "routing:\n"
        "  primary: local\n"
        "  fallback: [cloud, edge]\n"
        "policy: {mode: strict, retries: 2}\n"
        "description: |-\n"
        "  first line\n"
        "  second: line\n"
        "---\n"
        + body
    )

    meta, parsed_body = parse(text)
    assert meta == {
        "routing": {"primary": "local", "fallback": ["cloud", "edge"]},
        "policy": {"mode": "strict", "retries": 2},
        "description": "first line\nsecond: line",
    }
    assert parsed_body == body

    patched = patch(text, {"status": "stable"})
    patched_meta, patched_body = parse(patched)
    assert patched_meta == {**meta, "status": "stable"}
    assert patched_body == body
    assert parse(canonicalize(text))[1] == body


def test_malformed_present_frontmatter_fails_closed():
    text = "---\nrouting: {primary: local\n---\nbody\n"

    for operation in (parse, canonicalize, lambda value: patch(value, {"x": 1})):
        with pytest.raises(CanonicalDocumentError, match="not valid YAML"):
            operation(text)


def test_unhashable_yaml_key_is_normalized_to_canonical_document_error() -> None:
    text = "---\n? [complex, key]\n: value\n---\nbody\n"

    with pytest.raises(CanonicalDocumentError, match="not valid YAML"):
        parse(text)


def test_parse_legacy_retains_historical_raw_keyword_strings():
    text = (
        "---\n"
        "raw_keywords: [historical, 2026, true, 2026-08-11]\n"
        "---\n"
        "body\n"
    )

    legacy, legacy_body = parse_legacy(text)
    canonical, canonical_body = parse(text)

    assert legacy == {
        "raw_keywords": ["historical", "2026", "true", "2026-08-11"]
    }
    assert canonical == {
        "raw_keywords": ["historical", 2026, True, date(2026, 8, 11)]
    }
    assert legacy_body == canonical_body == "body\n"


def test_nested_resolution_review_values_are_json_safe_and_hash_stable() -> None:
    text = (
        "---\n"
        "title: Typed review\n"
        "updated: 2026-08-11\n"
        "policy: {cutoff: 2026-08-11}\n"
        "milestones: [2026-08-11, 2026-08-12]\n"
        "---\n"
        "---\n"
        "title: Typed review\n"
        "updated: 2026-08-10\n"
        "policy: {cutoff: 2026-08-10}\n"
        "milestones: [2026-08-10]\n"
        "---\n"
        "Body.\n"
    )

    proposed, details = propose_nested_resolution(text)
    metadata, body = parse(proposed)

    assert metadata["updated"] == date(2026, 8, 11)
    assert metadata["policy"] == {"cutoff": date(2026, 8, 11)}
    assert metadata["milestones"] == [
        date(2026, 8, 11),
        date(2026, 8, 12),
        date(2026, 8, 10),
    ]
    assert body == "Body.\n"
    assert details["conflicts"]["updated"]["outer"] == "2026-08-11"
    assert details["conflicts"]["policy"]["outer"] == {"cutoff": "2026-08-11"}
    assert details["conflicts"]["milestones"]["outer"] == {
        "kind": "list",
        "count": 2,
        "sha256": "648bb38e618228875065c9241358affb13f15bea69997f03664f5318e9e26148",
        "sample": ["2026-08-11", "2026-08-12"],
    }
    json.dumps(details, ensure_ascii=False, allow_nan=False)


def test_review_value_stringifies_dates_and_receipts_yaml_sets() -> None:
    assert review_value(
        {
            "updated": date(2026, 8, 11),
            "policy": {"cutoff": date(2026, 8, 12)},
        }
    ) == {
        "policy": {"cutoff": "2026-08-12"},
        "updated": "2026-08-11",
    }
    assert review_value(
        {
            "updated": date(2026, 8, 11),
            "features": {"gamma", "alpha", "beta"},
        }
    ) == {
        "kind": "canonical_yaml",
        "utf8_bytes": 102,
        "sha256": "67d2a003893c27aaa1bc5ad144cca74776f325db4e1d42fecec43b06ce13207a",
    }


def test_nested_resolution_handles_mapping_lists_and_mixed_key_receipts() -> None:
    text = (
        "---\n"
        "title: Complex review\n"
        "rules: [{kind: shared}, {kind: outer}]\n"
        "settings: {1: current, label: current}\n"
        "---\n"
        "---\n"
        "title: Complex review\n"
        "rules: [{kind: shared}, {kind: inner}]\n"
        "settings: {1: previous, label: previous}\n"
        "---\n"
        "Body.\n"
    )

    proposed, details = propose_nested_resolution(text)
    _second_proposed, second_details = propose_nested_resolution(text)
    metadata, body = parse(proposed)

    assert metadata["rules"] == [
        {"kind": "shared"},
        {"kind": "outer"},
        {"kind": "inner"},
    ]
    assert metadata["settings"] == {1: "current", "label": "current"}
    assert body == "Body.\n"
    assert details == second_details
    assert details["conflicts"]["settings"]["outer"] == {
        "kind": "canonical_yaml",
        "utf8_bytes": 45,
        "sha256": "6baf188966acd5c0adf08a46b30db2a81caad983b3930efe2d9c5a9bc2653007",
    }
    json.dumps(details, ensure_ascii=False, allow_nan=False)


def test_yaml_set_serialization_and_review_receipt_ignore_hash_seed() -> None:
    source = (
        "---\n"
        "title: Set order\n"
        "features: !!set\n"
        "  ? gamma\n"
        "  ? alpha\n"
        "  ? beta\n"
        "---\n"
        "Body.\n"
    )
    nested = (
        "---\n"
        "title: Set review\n"
        "features: !!set\n"
        "  ? gamma\n"
        "  ? alpha\n"
        "  ? beta\n"
        "---\n"
        "---\n"
        "title: Set review\n"
        "features: !!set\n"
        "  ? old\n"
        "---\n"
        "Body.\n"
    )
    script = (
        "import json\n"
        "from chronovisor.core.canonical_json import "
        "canonical_json_sha256_stringifying_strict\n"
        "from chronovisor.core.frontmatter import canonicalize, "
        "propose_nested_resolution\n"
        f"source = {source!r}\n"
        f"nested = {nested!r}\n"
        "canonical = canonicalize(source)\n"
        "_, details = propose_nested_resolution(nested)\n"
        "receipt = details['conflicts']['features']['outer']\n"
        "print(json.dumps({"
        "'canonical_hex': canonical.encode('utf-8').hex(), "
        "'receipt': receipt, "
        "'receipt_sha256': canonical_json_sha256_stringifying_strict(receipt)"
        "}, sort_keys=True))\n"
    )
    outputs = []
    for seed in ("1", "2", "3"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        outputs.append(completed.stdout.strip())

    assert len(set(outputs)) == 1
    payload = json.loads(outputs[0])
    expected = (
        "---\n"
        "title: Set order\n"
        "features: !!set\n"
        "  alpha: null\n"
        "  beta: null\n"
        "  gamma: null\n"
        "---\n"
        "Body.\n"
    )
    assert payload == {
        "canonical_hex": expected.encode("utf-8").hex(),
        "receipt": {
            "kind": "canonical_yaml",
            "utf8_bytes": 62,
            "sha256": "e6190c20b69f1fc88b14270f6667fa5c89344e05f01a746e81d1bdc06688a357",
        },
        "receipt_sha256": (
            "31a287b3e84a2aaa9ca7036a9045afb32af2a4c0dc1fbfacacb59c7042616310"
        ),
    }


def test_legacy_frontmatter_imports_are_limited_to_raw_and_offline_migration():
    source_root = Path(__file__).parents[1] / "src"
    allowed = {
        "chronovisor/core/okf_prepare.py",
        "chronovisor/core/raw_store.py",
        "chronovisor/ingest/orchestrator.py",
        "chronovisor/ingest/raw_replay.py",
        "chronovisor/ops/memory_integrity.py",
        "chronovisor/raw/record_raw.py",
    }
    importers: set[str] = set()
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            (
                isinstance(node, ast.ImportFrom)
                and (
                    node.module == "chronovisor.core.legacy_frontmatter"
                    or (
                        node.module == "chronovisor.core"
                        and any(
                            alias.name == "legacy_frontmatter" for alias in node.names
                        )
                    )
                )
            )
            or (
                isinstance(node, ast.Import)
                and any(
                    alias.name == "chronovisor.core.legacy_frontmatter"
                    for alias in node.names
                )
            )
            for node in ast.walk(tree)
        ):
            importers.add(path.relative_to(source_root).as_posix())

    assert importers == allowed
