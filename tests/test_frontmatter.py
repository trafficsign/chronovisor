"""Tests for the frontmatter parse/patch module."""

from __future__ import annotations

from chronovisor.core.frontmatter import canonicalize, parse, patch


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


def test_parse_unclosed_frontmatter_returns_empty():
    # A leading "---" without a matching close is treated as no frontmatter.
    text = "---\ntitle: x\n# body\n"
    meta, body = parse(text)
    assert meta == {}
    assert body == text


# ---------------------------------------------------------------------------
# parse() — scalar values (legacy parity)
# ---------------------------------------------------------------------------

def test_parse_scalar_basic():
    text = "---\ntitle: Hello\nupdated: 2026-05-08\n---\nbody text\n"
    meta, body = parse(text)
    assert meta == {"title": "Hello", "updated": "2026-05-08"}
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
    # `key: value: with: colons` — only the first colon splits.
    text = "---\nnote: foo: bar: baz\n---\n"
    meta, _ = parse(text)
    assert meta == {"note": "foo: bar: baz"}


def test_parse_skips_blank_lines_and_lines_without_colon():
    text = "---\ntitle: Hi\n\n# random comment-like line\nupdated: 2026-05-08\n---\n"
    meta, _ = parse(text)
    assert meta == {"title": "Hi", "updated": "2026-05-08"}


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
        "updated": "2026-05-08",
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
    assert meta == {"title": "X", "custom_field": "custom value", "another": "42"}


# ---------------------------------------------------------------------------
# patch() — basic add / replace / delete
# ---------------------------------------------------------------------------

def test_patch_add_to_empty():
    text = "body only\n"
    out = patch(text, {"title": "New"})
    assert out == "---\ntitle: New\n---\nbody only\n"


def test_patch_quotes_yaml_unsafe_scalars_and_flow_items():
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

    assert 'title: "Agents-A1: Model Analysis"' in out
    assert (
        'recall_questions: ["What changed?", '
        '"Which value contains a comma, and why?"]'
    ) in out
    meta, body = parse(out)
    assert meta["title"] == "Agents-A1: Model Analysis"
    assert meta["recall_questions"] == [
        "What changed?",
        "Which value contains a comma, and why?",
    ]
    assert body == "body\n"


def test_canonicalize_preserves_body_and_quotes_legacy_unsafe_yaml():
    body = "# Heading\n\nUnchanged body.\n"
    legacy = (
        "---\n"
        "title: Agents-A1: Model Analysis\n"
        "recall_questions: [What changed?, Why now?]\n"
        "---\n"
        + body
    )

    normalized = canonicalize(legacy)

    assert normalized.startswith(
        '---\ntitle: "Agents-A1: Model Analysis"\n'
        'recall_questions: ["What changed?", "Why now?"]\n---\n'
    )
    assert parse(normalized)[1] == body


def test_patch_replace_existing():
    text = "---\ntitle: Old\nupdated: 2025-01-01\n---\nbody\n"
    out = patch(text, {"title": "New"})
    meta, body = parse(out)
    assert meta["title"] == "New"
    assert meta["updated"] == "2025-01-01"
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
