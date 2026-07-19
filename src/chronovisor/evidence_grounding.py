"""Deterministic grounding for high-risk literals in generated memory.

This module deliberately protects a narrow class of strings where an LLM's
"helpful" normalization is especially damaging: model/product names, version
identifiers, and concrete numeric facts.  It is not a general fact checker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


_ASCII_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([A-Za-z0-9](?:[A-Za-z0-9_.+/-]{0,62}[A-Za-z0-9])?)"
    r"(?![A-Za-z0-9])"
)
_CAPACITY_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(\d+(?:\.\d+)?\s*[KMGTPE]i?B)"
    r"(?![A-Za-z0-9])"
)
_VERSION_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(v?\d+\.\d+(?:\.\d+)*(?:[-+][A-Za-z0-9.]+)?)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_NUMBER_PATTERN = r"(?:\d{1,3}(?:[,，_]\d{3})+|\d+)(?:\.\d+)?"
_CALENDAR_DATE_RE = re.compile(
    rf"(?<![A-Za-z0-9])({_NUMBER_PATTERN}(?:年|[-/.])"
    rf"{_NUMBER_PATTERN}(?:月|[-/.]){_NUMBER_PATTERN}日?)"
    r"(?![A-Za-z0-9])"
)
_JAPANESE_MONTH_DAY_RE = re.compile(
    rf"(?<![A-Za-z0-9])({_NUMBER_PATTERN}月{_NUMBER_PATTERN}日)"
    r"(?![A-Za-z0-9])"
)
_CLOCK_TIME_RE = re.compile(
    rf"(?<![A-Za-z0-9])({_NUMBER_PATTERN}:[0-5]\d(?:\s*(?:AM|PM))?)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_DIMENSION_RE = re.compile(
    rf"(?<![A-Za-z0-9])({_NUMBER_PATTERN}\s*[xX×✕]\s*{_NUMBER_PATTERN}"
    r"(?:\s*(?:px|mm|cm|m|inch(?:es)?|インチ))?)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_PREFIX_CURRENCY_RE = re.compile(
    rf"(?<![A-Za-z0-9])([¥￥$€£]\s*{_NUMBER_PATTERN})"
    r"(?![A-Za-z0-9])"
)
_NUMERIC_FACT_RE = re.compile(
    rf"(?<![A-Za-z0-9])({_NUMBER_PATTERN}(?:\s*|\s*[-–—]\s*)"
    r"(?:"
    r"円|万円|千円|ドル|米ドル|台|個|枚|本|人|件|回|基|機|冊|箱|セット|つ|"
    r"年|か月|ヶ月|月|日|時間|時|分|秒|"
    r"インチ|型|％|%|"
    r"px|mm|cm|km|mg|kg|oz|lb|"
    r"Hz|kHz|MHz|GHz|W|kW|V|A|"
    r"bytes?|bits?|items?|units?|copies|people|times?|runs?|episodes?|"
    r"pages?|queries|models?|inch(?:es)?|years?|months?|days?|hours?|minutes?|seconds?"
    r"))"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Letter-only product names cannot be identified from shape alone.  Keep this
# list intentionally focused on common model/runtime brands; context-derived
# TitleCase tokens below cover new names that appear in assistant prose.
_KNOWN_PRODUCT_NAMES = frozenset(
    name.casefold()
    for name in (
        "Anthropic",
        "Acer",
        "AMD",
        "Apple",
        "ASUS",
        "BenQ",
        "BOE",
        "ChatGPT",
        "Claude",
        "Codex",
        "Copilot",
        "Dell",
        "DeepSeek",
        "Eizo",
        "Gemini",
        "GitHub",
        "GPT",
        "Grok",
        "HP",
        "Huawei",
        "Intel",
        "Kuycon",
        "Lenovo",
        "LG",
        "Llama",
        "Microsoft",
        "Mistral",
        "Notion",
        "NVIDIA",
        "Ollama",
        "OpenAI",
        "Qwen",
        "Samsung",
        "Sony",
        "Xiaomi",
    )
)
_PRODUCT_VERSION_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"((?:Anthropic|ChatGPT|Claude|Codex|Copilot|DeepSeek|Gemini|GPT|Grok|"
    r"Llama|Mistral|Ollama|OpenAI|Qwen)"
    r"[ \t]+v?\d+(?:\.\d+)*(?:[A-Za-z]{0,4})?)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_TITLECASE_STOPWORDS = frozenset(
    {
        "A",
        "An",
        "And",
        "As",
        "At",
        "For",
        "From",
        "I",
        "If",
        "In",
        "It",
        "No",
        "On",
        "Or",
        "That",
        "The",
        "This",
        "To",
        "User",
        "We",
        "With",
        "You",
    }
)
_FILE_SUFFIXES = (
    ".go",
    ".js",
    ".json",
    ".jsx",
    ".log",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
)
_GENERIC_VERSIONED_TOKEN_RE = re.compile(
    r"(?:api|cli|cpu|css|csv|dom|gpu|html|http|https|id|ip|json|ram|rest|"
    r"rpc|sdk|sql|ssh|tcp|tls|udp|ui|uri|url|xml)v?\d+\Z",
    re.IGNORECASE,
)
_TECHNICAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:https?://)?(?:[A-Za-z_.~-][A-Za-z0-9_.~-]*/)+[A-Za-z0-9_.~/-]+",
    re.IGNORECASE,
)
_GENERIC_API_VERSION_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:(?:JSON|REST|HTTP|RPC)\s+)?(?:API|SDK|CLI)\s+v?\d+(?:\.\d+)*"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_CONTEXT_PRODUCT_PAIR_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Z][a-z]{2,})[ \t]+"
    r"([A-Za-z0-9][A-Za-z0-9_.+-]{1,31})(?![A-Za-z0-9])"
)


@dataclass(frozen=True)
class GroundingViolation:
    field: str
    literal: str


class ProtectedLiteralGroundingError(ValueError):
    """Raised when generated identity-like text lacks exact USER evidence."""

    def __init__(self, violations: Sequence[GroundingViolation]) -> None:
        self.violations = tuple(violations)
        detail = ", ".join(
            f"{violation.field}={violation.literal!r}" for violation in self.violations[:8]
        )
        super().__init__(f"ungrounded protected literal: {detail}")


def _mask_trusted_literals(text: str, trusted_literals: Iterable[str]) -> str:
    masked = text
    for literal in sorted(
        {value for value in trusted_literals if isinstance(value, str) and value},
        key=len,
        reverse=True,
    ):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(literal)}(?![A-Za-z0-9])"
        )
        masked = pattern.sub(lambda match: " " * len(match.group(0)), masked)
    return masked


def _mask_technical_syntax(text: str) -> str:
    """Mask path/API syntax that resembles a product version or dimension."""
    masked = text
    for pattern in (_TECHNICAL_PATH_RE, _GENERIC_API_VERSION_RE):
        masked = pattern.sub(lambda match: " " * len(match.group(0)), masked)
    return masked


def _context_titlecase_tokens(context_texts: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for text in context_texts:
        if not isinstance(text, str):
            continue
        for match in _ASCII_TOKEN_RE.finditer(text):
            token = match.group(1)
            if token in _TITLECASE_STOPWORDS:
                continue
            if token[:1].isupper() and token[1:].islower():
                tokens.add(token)
    return tokens


def _context_product_brand_tokens(context_texts: Iterable[str]) -> set[str]:
    """Infer an unknown brand only from a tight ``Brand Model123`` pair."""
    tokens: set[str] = set()
    for text in context_texts:
        if not isinstance(text, str):
            continue
        for match in _CONTEXT_PRODUCT_PAIR_RE.finditer(text):
            brand, model = match.groups()
            if brand in _TITLECASE_STOPWORDS:
                continue
            if not any(char.isdigit() for char in model):
                continue
            if not any(char.isascii() and char.isalpha() for char in model):
                continue
            tokens.add(brand)
    return tokens


def _contains_bounded_literal(text: str, literal: str) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(literal)}(?![A-Za-z0-9])",
            text,
        )
    )


def _token_is_protected(
    token: str,
    *,
    contextual_titlecase: set[str],
    contextual_product_brands: set[str],
    protect_contextual_titlecase: bool,
) -> bool:
    letters = [char for char in token if char.isascii() and char.isalpha()]
    if not letters:
        return False
    parts = re.split(r"[-+/]", token)
    known_part = any(part.casefold() in _KNOWN_PRODUCT_NAMES for part in parts)
    if "/" in token and not known_part:
        return False
    if token.casefold().endswith(_FILE_SUFFIXES) and not known_part:
        return False
    if _GENERIC_VERSIONED_TOKEN_RE.fullmatch(token):
        return False
    if token.casefold() in _KNOWN_PRODUCT_NAMES:
        return True
    if token in contextual_product_brands:
        return True
    if any(char.isdigit() for char in token) and any(
        char.isupper() for char in letters
    ):
        return True
    if any(char in token for char in "-+/"):
        if any(char.isupper() for char in token) or known_part:
            return True
    if any(char.islower() for char in letters) and any(
        char.isupper() for char in token[1:]
    ):
        return True
    return protect_contextual_titlecase and token in contextual_titlecase


def protected_literals(
    text: str,
    *,
    context_texts: Iterable[str] = (),
    trusted_literals: Iterable[str] = (),
    protect_contextual_titlecase: bool = False,
) -> list[str]:
    """Return exact, ordered identity-like literals from ``text``.

    Ordinary words, technical acronyms, paths, and bare numbers are ignored.
    Product identities and numeric facts with an explicit unit/counter/date
    shape are protected. TitleCase words are protected only when known
    model/product names, unless the caller explicitly enables context-derived
    protection (used for keyword fields).
    """

    context = tuple(context_texts)
    contextual_titlecase = _context_titlecase_tokens(context)
    contextual_product_brands = _context_product_brand_tokens(context)
    masked = _mask_technical_syntax(_mask_trusted_literals(text, trusted_literals))
    matches: list[tuple[int, int, str]] = []
    for pattern in (
        _PRODUCT_VERSION_RE,
        _CALENDAR_DATE_RE,
        _JAPANESE_MONTH_DAY_RE,
        _CLOCK_TIME_RE,
        _DIMENSION_RE,
        _PREFIX_CURRENCY_RE,
        _NUMERIC_FACT_RE,
        _CAPACITY_RE,
        _VERSION_RE,
    ):
        for match in pattern.finditer(masked):
            matches.append((match.start(1), match.end(1), match.group(1)))
    for match in _ASCII_TOKEN_RE.finditer(masked):
        token = match.group(1)
        if _token_is_protected(
            token,
            contextual_titlecase=contextual_titlecase,
            contextual_product_brands=contextual_product_brands,
            protect_contextual_titlecase=protect_contextual_titlecase,
        ):
            matches.append((match.start(1), match.end(1), token))

    ordered: list[str] = []
    seen: set[str] = set()
    ordered_matches = sorted(
        matches,
        key=lambda item: (item[0], -(item[1] - item[0])),
    )
    for _start, _end, literal in ordered_matches:
        if literal not in seen:
            ordered.append(literal)
            seen.add(literal)
    return ordered


def validate_protected_literals(
    fields: Mapping[str, str | Iterable[str]],
    *,
    evidence_quotes: Iterable[str],
    context_texts: Iterable[str] = (),
    allowed_texts: Iterable[str] = (),
    trusted_literals: Iterable[str] = (),
) -> None:
    """Require every protected output literal to exist verbatim in evidence.

    ``allowed_texts`` is for literals that a bounded edit carries forward from
    its exact preimage.  It must not be used as a substitute for USER evidence
    when introducing a new identity.
    """

    evidence = tuple(value for value in evidence_quotes if isinstance(value, str))
    allowed = tuple(value for value in allowed_texts if isinstance(value, str))
    context = tuple(value for value in context_texts if isinstance(value, str))
    evidence_literals = {
        literal
        for value in evidence
        for literal in protected_literals(
            value,
            context_texts=context,
            protect_contextual_titlecase=True,
        )
    }
    allowed_literals = {
        literal
        for value in allowed
        for literal in protected_literals(
            value,
            context_texts=context,
            protect_contextual_titlecase=True,
        )
    }
    violations: list[GroundingViolation] = []
    for field, raw_values in fields.items():
        values = (raw_values,) if isinstance(raw_values, str) else tuple(raw_values)
        for index, value in enumerate(values):
            if not isinstance(value, str):
                continue
            label = field if len(values) == 1 else f"{field}[{index}]"
            for literal in protected_literals(
                value,
                context_texts=context,
                trusted_literals=trusted_literals,
                protect_contextual_titlecase=field == "keywords",
            ):
                if literal in evidence_literals:
                    continue
                if any(_contains_bounded_literal(value, literal) for value in evidence):
                    continue
                if literal in allowed_literals:
                    continue
                if any(_contains_bounded_literal(value, literal) for value in allowed):
                    continue
                violations.append(GroundingViolation(field=label, literal=literal))
    if violations:
        raise ProtectedLiteralGroundingError(violations)
