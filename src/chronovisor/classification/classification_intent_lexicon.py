"""Deterministic high-precision intent lane for CVO complementary anchors."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronovisor.classification.classification import ClassificationError
from chronovisor.classification.classification_anchor import (
    UNRESOLVED_ANCHOR_ID,
    AnchorSet,
    load_anchor_set,
)

LEXICON_SCHEMA = "chronovisor.cvo-intent-lexicon.v1"


@dataclass(frozen=True)
class IntentSignal:
    field: str
    pattern: str


@dataclass(frozen=True)
class IntentRule:
    term_id: str
    label_ja: str
    anchor_id: str
    priority: int
    signals: tuple[IntentSignal, ...]


@dataclass(frozen=True)
class IntentLexicon:
    epoch: str
    checksum: str
    rules: tuple[IntentRule, ...]


def default_intent_lexicon_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "cvo-intent-lexicon-v2.json"
    )


def _checksum(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def load_intent_lexicon(
    path: Path | None = None,
    *,
    anchor_set: AnchorSet | None = None,
) -> IntentLexicon:
    source = path or default_intent_lexicon_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != LEXICON_SCHEMA
        or not str(payload.get("epoch") or "")
    ):
        raise ClassificationError("CVO intent lexicon contract mismatch")
    anchors = anchor_set or load_anchor_set()
    rules = []
    seen_terms: set[str] = set()
    for row in payload.get("rules") or []:
        if not isinstance(row, Mapping):
            raise ClassificationError("CVO intent lexicon rule is invalid")
        term_id = str(row.get("term_id") or "")
        anchor_id = str(row.get("anchor_id") or "")
        priority = int(row.get("priority") or 0)
        signals = tuple(
            IntentSignal(
                field=str(signal.get("field") or ""),
                pattern=str(signal.get("pattern") or ""),
            )
            for signal in row.get("signals") or []
            if isinstance(signal, Mapping)
        )
        if (
            not term_id
            or term_id in seen_terms
            or anchor_id not in anchors.by_id
            or anchor_id == UNRESOLVED_ANCHOR_ID
            or not 1 <= priority <= 100
            or not signals
            or any(
                signal.field not in {"title", "summary", "excerpt"}
                or not signal.pattern
                for signal in signals
            )
        ):
            raise ClassificationError("CVO intent lexicon rule is incomplete")
        for signal in signals:
            try:
                re.compile(signal.pattern)
            except re.error as exc:
                raise ClassificationError(
                    f"CVO intent regex is invalid: {term_id}"
                ) from exc
        seen_terms.add(term_id)
        rules.append(
            IntentRule(
                term_id=term_id,
                label_ja=str(row.get("label_ja") or ""),
                anchor_id=anchor_id,
                priority=priority,
                signals=signals,
            )
        )
    if not rules:
        raise ClassificationError("CVO intent lexicon has no rules")
    return IntentLexicon(
        epoch=str(payload["epoch"]),
        checksum=_checksum(payload),
        rules=tuple(rules),
    )


def classify_complement(
    page: Mapping[str, Any],
    *,
    core_anchor_id: str,
    lexicon: IntentLexicon | None = None,
) -> dict[str, Any]:
    active = lexicon or load_intent_lexicon()
    matches = []
    for rule in active.rules:
        if rule.anchor_id == core_anchor_id:
            continue
        for signal in rule.signals:
            value = str(page.get(signal.field) or "")
            matched = re.search(signal.pattern, value)
            if matched is None:
                continue
            matches.append(
                {
                    "term_id": rule.term_id,
                    "label_ja": rule.label_ja,
                    "anchor_id": rule.anchor_id,
                    "priority": rule.priority,
                    "field": signal.field,
                    "evidence": matched.group(0)[:160],
                }
            )
            break
    by_anchor: dict[str, dict[str, Any]] = {}
    for match in matches:
        anchor_id = str(match["anchor_id"])
        current = by_anchor.get(anchor_id)
        if current is None or int(match["priority"]) > int(current["priority"]):
            by_anchor[anchor_id] = match
    ranked = sorted(
        by_anchor.values(),
        key=lambda match: (-int(match["priority"]), str(match["anchor_id"])),
    )
    if not ranked:
        selected = None
        reason = "no_high_precision_intent_match"
    elif len(ranked) > 1 and int(ranked[0]["priority"]) == int(
        ranked[1]["priority"]
    ):
        selected = None
        reason = "ambiguous_equal_priority_matches"
    else:
        selected = ranked[0]
        reason = "unique_highest_priority_match"
    return {
        "schema": "chronovisor.cvo-intent-complement.v1",
        "lexicon_epoch": active.epoch,
        "lexicon_checksum": active.checksum,
        "core_anchor_id": core_anchor_id,
        "second_anchor_id": (
            str(selected["anchor_id"]) if selected is not None else "NONE"
        ),
        "selected_match": selected,
        "all_matches": ranked,
        "reason": reason,
    }
