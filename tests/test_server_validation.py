"""Tests for server-side validation helpers."""

from __future__ import annotations

from chronovisor.raw.record_raw import validate_raw_keyword


def test_accepts_basic_keywords():
    assert validate_raw_keyword("Anthropic")
    assert validate_raw_keyword("multi word")
    assert validate_raw_keyword("a")
    assert validate_raw_keyword("kebab-case-ok")


def test_rejects_empty_or_whitespace_only():
    assert not validate_raw_keyword("")
    assert not validate_raw_keyword(" ")
    assert not validate_raw_keyword("   ")
    assert not validate_raw_keyword("\t")


def test_rejects_forbidden_chars():
    for ch in ",[]:#{}":
        assert not validate_raw_keyword(f"foo{ch}bar"), f"should reject {ch!r}"


def test_rejects_newline_and_cr():
    assert not validate_raw_keyword("foo\nbar")
    assert not validate_raw_keyword("foo\rbar")


def test_rejects_control_chars():
    # 0x00–0x1F are control characters; 0x7F is DEL but is allowed here
    # (callers are unlikely to pass it, and the YAML parser handles it).
    assert not validate_raw_keyword("foo\x01bar")
    assert not validate_raw_keyword("foo\x1fbar")


def test_rejects_non_string():
    assert not validate_raw_keyword(None)
    assert not validate_raw_keyword(123)
    assert not validate_raw_keyword(["a"])
    assert not validate_raw_keyword({"k": "v"})
