"""Tests for server-side validation helpers."""

from __future__ import annotations

from chronovisor.server import _validate_raw_keyword


def test_accepts_basic_keywords():
    assert _validate_raw_keyword("Anthropic")
    assert _validate_raw_keyword("multi word")
    assert _validate_raw_keyword("a")
    assert _validate_raw_keyword("kebab-case-ok")


def test_rejects_empty_or_whitespace_only():
    assert not _validate_raw_keyword("")
    assert not _validate_raw_keyword(" ")
    assert not _validate_raw_keyword("   ")
    assert not _validate_raw_keyword("\t")


def test_rejects_forbidden_chars():
    for ch in ",[]:#{}":
        assert not _validate_raw_keyword(f"foo{ch}bar"), f"should reject {ch!r}"


def test_rejects_newline_and_cr():
    assert not _validate_raw_keyword("foo\nbar")
    assert not _validate_raw_keyword("foo\rbar")


def test_rejects_control_chars():
    # 0x00–0x1F are control characters; 0x7F is DEL but is allowed here
    # (callers are unlikely to pass it, and the YAML parser handles it).
    assert not _validate_raw_keyword("foo\x01bar")
    assert not _validate_raw_keyword("foo\x1fbar")


def test_rejects_non_string():
    assert not _validate_raw_keyword(None)
    assert not _validate_raw_keyword(123)
    assert not _validate_raw_keyword(["a"])
    assert not _validate_raw_keyword({"k": "v"})
