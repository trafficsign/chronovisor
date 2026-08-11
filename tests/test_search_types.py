from __future__ import annotations

from chronovisor.core.search_types import tokenize


def test_tokenize_treats_malformed_leading_frontmatter_as_query_text() -> None:
    query = "---\ntitle: [unterminated\n---\nneedle query"

    assert tokenize(query) == ["title", "unterminated", "needle", "query"]


def test_tokenize_falls_back_for_unhashable_complex_yaml_key() -> None:
    query = "---\n? [complex, key]\n: value\n---\nneedle query"

    assert tokenize(query) == ["complex", "key", "value", "needle", "query"]


def test_tokenize_strips_valid_canonical_frontmatter_from_page_text() -> None:
    page = "---\ntitle: Metadata Needle\n---\nBody token\n"

    assert tokenize(page) == ["body", "token"]
