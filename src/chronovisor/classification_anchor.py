"""Versioned Chronovisor-specific operational anchor definitions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronovisor.classification import ClassificationError

ANCHOR_SET_SCHEMA = "chronovisor.cvo-anchor-set.v1"
ANCHOR_EPOCH = "cvo-anchor-v0"
UNRESOLVED_ANCHOR_ID = "cvo:anchor:0099"


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    family: str
    label_ja: str
    label_en: str
    definition_ja: str
    definition_en: str
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    udc_scope: tuple[str, ...]

    def model_card(self) -> dict[str, Any]:
        return {
            "id": self.anchor_id,
            "label_ja": self.label_ja,
            "label_en": self.label_en,
            "definition_ja": self.definition_ja,
            "definition_en": self.definition_en,
            "includes": list(self.includes),
            "excludes": list(self.excludes),
        }


@dataclass(frozen=True)
class AnchorSet:
    schema: str
    epoch: str
    status: str
    checksum: str
    anchors: tuple[Anchor, ...]
    by_id: Mapping[str, Anchor]

    def model_cards(self) -> list[dict[str, Any]]:
        return [anchor.model_card() for anchor in self.anchors]


def default_anchor_set_path() -> Path:
    return Path(__file__).parent / "data" / "cvo-anchor-set-v0.json"


def default_anchor_gold_path() -> Path:
    return Path(__file__).parent / "data" / "cvo-anchor-dev-gold-v0.json"


def load_anchor_set(path: Path | None = None) -> AnchorSet:
    source = path or default_anchor_set_path()
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClassificationError(f"invalid CVO anchor JSON: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != ANCHOR_SET_SCHEMA
        or payload.get("epoch") != ANCHOR_EPOCH
    ):
        raise ClassificationError("unsupported CVO anchor set")
    rows = payload.get("anchors")
    if not isinstance(rows, list) or not 30 <= len(rows) <= 60:
        raise ClassificationError("CVO anchor set must contain 30 to 60 anchors")
    anchors = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ClassificationError("CVO anchor must be an object")
        anchor_id = str(row.get("id") or "")
        family = str(row.get("family") or "")
        if (
            not anchor_id.startswith("cvo:anchor:")
            or anchor_id in seen
            or not family
            or not str(row.get("label_ja") or "")
            or not str(row.get("label_en") or "")
            or not str(row.get("definition_ja") or "")
            or not str(row.get("definition_en") or "")
        ):
            raise ClassificationError("CVO anchor is incomplete or duplicated")
        seen.add(anchor_id)
        anchors.append(
            Anchor(
                anchor_id=anchor_id,
                family=family,
                label_ja=str(row["label_ja"]),
                label_en=str(row["label_en"]),
                definition_ja=str(row["definition_ja"]),
                definition_en=str(row["definition_en"]),
                includes=tuple(str(value) for value in row.get("includes") or []),
                excludes=tuple(str(value) for value in row.get("excludes") or []),
                udc_scope=tuple(str(value) for value in row.get("udc_scope") or []),
            )
        )
    if UNRESOLVED_ANCHOR_ID not in seen:
        raise ClassificationError("CVO anchor set has no unresolved exit")
    anchors.sort(key=lambda anchor: anchor.anchor_id)
    return AnchorSet(
        schema=ANCHOR_SET_SCHEMA,
        epoch=ANCHOR_EPOCH,
        status=str(payload.get("status") or ""),
        checksum="sha256:" + hashlib.sha256(raw).hexdigest(),
        anchors=tuple(anchors),
        by_id={anchor.anchor_id: anchor for anchor in anchors},
    )


def validate_anchor_gold(
    payload: Mapping[str, Any],
    anchor_set: AnchorSet,
    expected_uids: Sequence[str],
) -> dict[str, list[str]]:
    if (
        payload.get("schema") != "chronovisor.cvo-anchor-dev-gold.v1"
        or payload.get("anchor_epoch") != anchor_set.epoch
    ):
        raise ClassificationError("CVO anchor gold contract mismatch")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ClassificationError("CVO anchor gold cases are missing")
    output: dict[str, list[str]] = {}
    for row in cases:
        if not isinstance(row, Mapping):
            raise ClassificationError("CVO anchor gold case is invalid")
        uid = str(row.get("uid") or "")
        expected = list(
            dict.fromkeys(
                str(value)
                for value in row.get("expected_primary_anchor_ids") or []
                if str(value)
            )
        )
        if (
            not uid
            or uid in output
            or not expected
            or any(value not in anchor_set.by_id for value in expected)
        ):
            raise ClassificationError("CVO anchor gold case is incomplete")
        output[uid] = expected
    if set(output) != set(expected_uids):
        raise ClassificationError("CVO anchor gold UIDs do not match dev fixture")
    return output
