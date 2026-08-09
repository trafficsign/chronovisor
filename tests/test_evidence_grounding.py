from __future__ import annotations

import pytest

from chronovisor.raw.evidence_grounding import (
    ProtectedLiteralGroundingError,
    protected_literals,
    validate_protected_literals,
)


def test_protected_literals_focuses_on_identity_like_values() -> None:
    assert protected_literals(
        "Q-KUN の 32GP と Qwen の 32GB。ordinary lowercase words are ignored.",
        context_texts=("Qwen was mentioned by the assistant.",),
    ) == ["Q-KUN", "32GP", "Qwen", "32GB"]
    assert protected_literals(
        "Installed RAM via JSON API, auto-apply, and src/model2.py."
    ) == []


def test_exact_user_literals_pass_but_normalized_substitutions_fail() -> None:
    validate_protected_literals(
        {"content": "Q-KUNの32GPレビュー"},
        evidence_quotes=("Q-KUNの32GPレビュー",),
        context_texts=("Qwenの32GBレビュー",),
    )

    with pytest.raises(ProtectedLiteralGroundingError) as caught:
        validate_protected_literals(
            {"content": "Qwenの32GBレビュー"},
            evidence_quotes=("Q-KUNの32GPレビュー",),
            context_texts=("Qwenの32GBレビュー",),
        )

    assert {item.literal for item in caught.value.violations} >= {"Qwen", "32GB"}

    with pytest.raises(ProtectedLiteralGroundingError):
        validate_protected_literals(
            {"content": "Qwen with 32GB"},
            evidence_quotes=("Qwen2 with 132GB",),
        )


def test_bounded_edit_may_carry_forward_existing_literals() -> None:
    validate_protected_literals(
        {"new_text": "Q-KUN has 32GB RAM."},
        evidence_quotes=("正しくは32GB",),
        allowed_texts=("Q-KUN has 16GB RAM.",),
    )


@pytest.mark.parametrize(
    ("text", "literal"),
    [
        ("P24Uを2台購入した", "2台"),
        ("価格は55,399円だった", "55,399円"),
        ("2026年7月11日に届く", "2026年7月11日"),
        ("7月11日に届く", "7月11日"),
        ("24インチのモニター", "24インチ"),
        ("解像度は1920x1080", "1920x1080"),
        ("at 12:30 PM", "12:30 PM"),
    ],
)
def test_protected_literals_include_concrete_numeric_facts(
    text: str,
    literal: str,
) -> None:
    assert literal in protected_literals(text)


@pytest.mark.parametrize(
    ("output", "expected_literal"),
    [
        ("Kuycon P24Uを3台購入した。", "3台"),
        ("Kuycon P24Uは合計114,110円だった。", "114,110円"),
        ("Kuycon P24Uは2026年7月12日に届いた。", "2026年7月12日"),
        ("Kuycon P24Uは27インチである。", "27インチ"),
    ],
)
def test_numeric_fact_without_user_evidence_is_rejected(
    output: str,
    expected_literal: str,
) -> None:
    with pytest.raises(ProtectedLiteralGroundingError) as caught:
        validate_protected_literals(
            {"content": output},
            evidence_quotes=(
                "Kuycon P24Uを2台、1台55,399円で2026年7月11日に購入した。24インチ。",
            ),
        )

    assert expected_literal in {item.literal for item in caught.value.violations}


def test_exact_numeric_facts_and_product_brand_pass() -> None:
    quote = "Kuycon P24Uを2台、1台55,399円で2026年7月11日に購入した。24インチ。"
    validate_protected_literals(
        {
            "content": (
                "2026年7月11日にKuycon P24Uを2台購入。"
                "24インチで1台55,399円。"
            )
        },
        evidence_quotes=(quote,),
    )
    validate_protected_literals(
        {"content": "7月11日に届く。"},
        evidence_quotes=(quote,),
    )


def test_partial_date_substitution_is_rejected() -> None:
    with pytest.raises(ProtectedLiteralGroundingError) as caught:
        validate_protected_literals(
            {"content": "8月11日に届く。"},
            evidence_quotes=("2026年7月11日に届く。",),
        )

    assert "8月11日" in {item.literal for item in caught.value.violations}


def test_assistant_only_product_brand_replacement_is_rejected() -> None:
    with pytest.raises(ProtectedLiteralGroundingError) as caught:
        validate_protected_literals(
            {"content": "Samsung P24Uを2台購入した。"},
            evidence_quotes=("Kuycon P24Uを2台購入した。",),
            context_texts=("Samsung P24Uを2台購入したんですね。",),
        )

    assert "Samsung" in {item.literal for item in caught.value.violations}


def test_tight_assistant_product_pair_protects_unknown_brand() -> None:
    with pytest.raises(ProtectedLiteralGroundingError) as caught:
        validate_protected_literals(
            {"content": "Viewson P24Uを購入した。"},
            evidence_quotes=("Kuycon P24Uを購入した。",),
            context_texts=("Viewson P24Uを購入したんですね。",),
        )

    assert "Viewson" in {item.literal for item in caught.value.violations}


def test_general_words_paths_and_api_identifiers_are_not_overprotected() -> None:
    text = (
        "ordinary phase2 notes use JSON APIv2, REST API v2.0, HTTP/2, "
        "/api/v2.0/responses, docs/v3.1/setup, 3 API calls, and src/model2.py"
    )
    assert protected_literals(text) == []
    validate_protected_literals(
        {"content": text},
        evidence_quotes=("ordinary notes",),
    )
