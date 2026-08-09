"""Versioned controlled vocabulary used by the CVO two-arm classifier."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronovisor.recall.classification import ClassificationError
from chronovisor.recall.classification_anchor import UNRESOLVED_ANCHOR_ID

VOCABULARY_SCHEMA = "chronovisor.controlled-vocabulary.v1"


@dataclass(frozen=True)
class ControlledTerm:
    term_id: str
    label_ja: str
    label_en: str
    definition_ja: str
    definition_en: str
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    anchor_id: str
    source_kind: str
    source_ids: tuple[str, ...]

    def model_card(self) -> dict[str, Any]:
        return {
            "id": self.term_id,
            "label_ja": self.label_ja,
            "label_en": self.label_en,
            "definition_ja": self.definition_ja,
            "definition_en": self.definition_en,
            "includes": list(self.includes),
            "excludes": list(self.excludes),
        }


@dataclass(frozen=True)
class ControlledVocabulary:
    epoch: str
    checksum: str
    terms: tuple[ControlledTerm, ...]
    ambiguous_registry: tuple[dict[str, Any], ...]
    gap_policy: dict[str, Any]
    derivation: dict[str, Any]

    @property
    def by_id(self) -> dict[str, ControlledTerm]:
        return {term.term_id: term for term in self.terms}

    def model_cards(self) -> list[dict[str, Any]]:
        cards = []
        for term in self.terms:
            card = term.model_card()
            if term.anchor_id == UNRESOLVED_ANCHOR_ID:
                card["id"] = UNRESOLVED_ANCHOR_ID
            cards.append(card)
        return cards

    def anchors_for_terms(self, term_ids: list[str]) -> list[str]:
        if UNRESOLVED_ANCHOR_ID in term_ids:
            return [UNRESOLVED_ANCHOR_ID]
        anchors = sorted(
            {
                self.by_id[term_id].anchor_id
                for term_id in term_ids
                if term_id in self.by_id
            }
        )
        if not anchors or len(anchors) > 2:
            return [UNRESOLVED_ANCHOR_ID]
        return anchors


def default_vocabulary_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "cvo-controlled-vocabulary-v1.json"
    )


def _checksum(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("checksum", None)
    return "sha256:" + hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def load_controlled_vocabulary(
    path: Path | None = None,
) -> ControlledVocabulary:
    source = path or default_vocabulary_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != VOCABULARY_SCHEMA
        or payload.get("status") != "frozen"
        or payload.get("mapping_audit") != "all-terms-reviewed"
    ):
        raise ClassificationError("controlled vocabulary contract is invalid")
    expected_checksum = _checksum(payload)
    if payload.get("checksum") != expected_checksum:
        raise ClassificationError("controlled vocabulary checksum mismatch")
    terms: list[ControlledTerm] = []
    seen: set[str] = set()
    seen_labels: set[tuple[str, str]] = set()
    allowed_source_kinds = {
        "cvo-anchor",
        "existing-intent-lexicon",
        "governance",
        "raw-keyword",
        "tag",
        "wiki-derived",
        "wikilink",
    }
    for row in payload.get("terms") or []:
        if not isinstance(row, dict):
            raise ClassificationError("controlled vocabulary term is invalid")
        term_id = str(row.get("id") or "")
        anchor_id = str(row.get("anchor_id") or "")
        label_pair = (
            str(row.get("label_ja") or "").strip(),
            str(row.get("label_en") or "").strip(),
        )
        source_kind = str(row.get("source_kind") or "")
        source_ids = tuple(
            str(item).strip()
            for item in row.get("source_ids") or []
            if str(item).strip()
        )
        if (
            not term_id
            or term_id in seen
            or not term_id.startswith("cvo:term:")
            or not anchor_id.startswith("cvo:anchor:")
            or not all(label_pair)
            or label_pair in seen_labels
            or source_kind not in allowed_source_kinds
            or not source_ids
        ):
            raise ClassificationError(
                "controlled vocabulary term identity is invalid"
            )
        seen.add(term_id)
        seen_labels.add(label_pair)
        terms.append(
            ControlledTerm(
                term_id=term_id,
                label_ja=str(row.get("label_ja") or ""),
                label_en=str(row.get("label_en") or ""),
                definition_ja=str(row.get("definition_ja") or ""),
                definition_en=str(row.get("definition_en") or ""),
                includes=tuple(str(item) for item in row.get("includes") or []),
                excludes=tuple(str(item) for item in row.get("excludes") or []),
                anchor_id=anchor_id,
                source_kind=source_kind,
                source_ids=source_ids,
            )
        )
    if (
        not 30 <= len(terms) <= 60
        or sum(term.anchor_id == UNRESOLVED_ANCHOR_ID for term in terms) != 1
        or not any(term.source_kind == "raw-keyword" for term in terms)
        or not any(term.source_kind == "wikilink" for term in terms)
        or not any(term.source_kind == "tag" for term in terms)
    ):
        raise ClassificationError(
            "controlled vocabulary coverage contract is invalid"
        )
    ambiguous = payload.get("ambiguous_registry")
    gap_policy = payload.get("gap_policy")
    derivation = payload.get("derivation")
    if (
        not isinstance(ambiguous, list)
        or not isinstance(gap_policy, dict)
        or not isinstance(derivation, dict)
        or derivation.get("page_label_associations_used") is not False
        or derivation.get("literal_regex_matching_used") is not False
        or gap_policy.get("unknown_term_result") != "hold"
        or gap_policy.get("production_mutation") is not False
    ):
        raise ClassificationError(
            "controlled vocabulary governance metadata is missing"
        )
    corpus_snapshot = derivation.get("corpus_snapshot")
    if (
        not isinstance(corpus_snapshot, dict)
        or int(corpus_snapshot.get("page_count") or 0) <= 0
        or int(corpus_snapshot.get("specific_in_domain_terms") or 0)
        + int(corpus_snapshot.get("broad_fallback_terms") or 0)
        != len(terms) - 1
    ):
        raise ClassificationError(
            "controlled vocabulary derivation receipt is invalid"
        )
    for item in ambiguous:
        if (
            not isinstance(item, dict)
            or not str(item.get("surface") or "").strip()
            or item.get("standalone_forbidden") is not True
            or not str(item.get("reason") or "").strip()
        ):
            raise ClassificationError(
                "controlled vocabulary ambiguity registry is invalid"
            )
    return ControlledVocabulary(
        epoch=str(payload.get("epoch") or ""),
        checksum=expected_checksum,
        terms=tuple(terms),
        ambiguous_registry=tuple(dict(item) for item in ambiguous),
        gap_policy=dict(gap_policy),
        derivation=dict(derivation),
    )
