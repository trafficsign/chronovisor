"""Versioned UDC-backed classification records and deterministic shadow seeds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

CLASSIFICATION_SCHEMA = "chronovisor.classification.v1"
PACKAGE_SCHEMA = "chronovisor.udcs-package.v1"
VALID_FORMS = {
    "decision",
    "event",
    "howto",
    "reference",
    "architecture",
    "analysis",
    "state",
    "profile",
    "knowledge",
}
VALID_LIFECYCLES = {"active", "historical", "superseded", "experimental", "held"}
VALID_EVIDENCE = {"raw-grounded", "derived", "external", "mixed"}
SENSITIVITY_ORDER = {"normal": 0, "personal": 1, "restricted": 2, "high": 2}
CVO_SUBJECT_SCHEMA = "chronovisor.cvo-subject.v1"
NDC_OVERLAY_SCHEMA = "chronovisor.ndc-crosswalk.v1"
CALIBRATION_SCHEMA = "chronovisor.classification-calibration.v1"


class ClassificationError(ValueError):
    """Raised when a classification record or taxonomy package is invalid."""


@dataclass(frozen=True)
class Subject:
    concept_uri: str
    notation: str
    label: str
    label_source: str


@dataclass(frozen=True)
class ClassificationRecord:
    schema: str
    subject_scheme: str
    subject_release: str
    subject_checksum: str
    primary: Subject
    secondary: tuple[Subject, ...]
    facets: Mapping[str, Any]
    confidence: float
    evidence_refs: tuple[str, ...]
    classifier_authority_epoch: int
    status: str = "proposed"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["secondary"] = [asdict(value) for value in self.secondary]
        payload["primary"] = asdict(self.primary)
        payload["facets"] = dict(self.facets)
        payload["evidence_refs"] = list(self.evidence_refs)
        return payload


@dataclass(frozen=True)
class UDCPackage:
    release: str
    checksum: str
    source_url: str
    license: str
    attribution: str
    complete: bool
    concepts: Mapping[str, Mapping[str, Any]]

    @classmethod
    def load(cls, path: Path) -> "UDCPackage":
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ClassificationError(f"invalid UDC package JSON: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema") != PACKAGE_SCHEMA:
            raise ClassificationError("unsupported UDC package schema")
        rows = payload.get("concepts")
        if not isinstance(rows, list) or not rows:
            raise ClassificationError("UDC package concepts must be a non-empty list")
        concepts: dict[str, Mapping[str, Any]] = {}
        notation_seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ClassificationError("UDC concept must be an object")
            uri = str(row.get("uri") or "").strip()
            notation = str(row.get("notation") or "").strip()
            label = str(row.get("label_en") or row.get("label") or "").strip()
            if not uri or not notation or not label:
                raise ClassificationError("UDC concept requires uri, notation, label")
            if uri in concepts or notation in notation_seen:
                raise ClassificationError("UDC concept URI/notation must be unique")
            concepts[uri] = dict(row)
            notation_seen.add(notation)
        return cls(
            release=str(payload.get("release") or ""),
            checksum="sha256:" + hashlib.sha256(raw).hexdigest(),
            source_url=str(payload.get("source_url") or ""),
            license=str(payload.get("license") or ""),
            attribution=str(payload.get("attribution") or ""),
            complete=bool(payload.get("complete")),
            concepts=concepts,
        )

    def by_notation(self, notation: str) -> Mapping[str, Any] | None:
        return next(
            (
                row
                for row in self.concepts.values()
                if str(row.get("notation")) == notation
            ),
            None,
        )


@dataclass(frozen=True)
class ControlledSubject:
    concept_uri: str
    broader_udc_uri: str
    label: str
    definition: str
    inclusion_examples: tuple[str, ...]
    exclusion_examples: tuple[str, ...]
    version: str
    authority_epoch: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_udc_package() -> UDCPackage:
    return UDCPackage.load(
        Path(__file__).parent / "data" / "udc-summary-top-level.json"
    )


def load_udc_package(root: Path | None = None) -> UDCPackage:
    """Load an operator-installed full package or the bundled bootstrap."""

    if root is not None:
        installed = root / "classification" / "udc-package.json"
        if installed.exists():
            return UDCPackage.load(installed)
    return default_udc_package()


def classification_authority_status(
    root: Path,
    *,
    package: UDCPackage | None = None,
) -> dict[str, Any]:
    package = package or load_udc_package(root)
    calibration_path = root / "classification" / "calibration.json"
    calibration: dict[str, Any] = {}
    if calibration_path.exists():
        try:
            loaded = json.loads(calibration_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            calibration = loaded
    try:
        forced_misclassification = float(
            calibration.get("forced_misclassification_rate", 1.0)
        )
    except (TypeError, ValueError):
        forced_misclassification = 1.0
    calibrated = bool(
        calibration.get("schema") == CALIBRATION_SCHEMA
        and calibration.get("status") == "adopted"
        and calibration.get("package_checksum") == package.checksum
        and calibration.get("fixture_locked") is True
        and forced_misclassification <= 0.01
    )
    active = bool(package.complete and calibrated)
    reasons = []
    if not package.complete:
        reasons.append("complete_udc_package_missing")
    if not calibrated:
        reasons.append("locked_calibration_not_adopted")
    return {
        "active": active,
        "package_complete": package.complete,
        "calibrated": calibrated,
        "release": package.release,
        "checksum": package.checksum,
        "license": package.license,
        "calibration_path": str(calibration_path),
        "reason": ",".join(reasons) if reasons else "adopted",
    }


def validate_record(
    record: ClassificationRecord,
    *,
    package: UDCPackage,
    require_complete_package: bool = False,
) -> None:
    if record.schema != CLASSIFICATION_SCHEMA:
        raise ClassificationError("classification schema mismatch")
    if record.subject_scheme != "udcs":
        raise ClassificationError("subject_scheme must be udcs")
    if record.subject_release != package.release:
        raise ClassificationError("subject release mismatch")
    if record.subject_checksum != package.checksum:
        raise ClassificationError("subject checksum mismatch")
    if require_complete_package and not package.complete:
        raise ClassificationError(
            "classification authority requires a complete UDC package"
        )
    if record.primary.concept_uri not in package.concepts:
        raise ClassificationError("primary concept is not in the UDC package")
    if len(record.secondary) > 3:
        raise ClassificationError("secondary subjects exceed limit")
    uris = [
        record.primary.concept_uri,
        *(item.concept_uri for item in record.secondary),
    ]
    if len(uris) != len(set(uris)):
        raise ClassificationError("primary and secondary subjects must be unique")
    if not 0.0 <= float(record.confidence) <= 1.0:
        raise ClassificationError("confidence must be within [0, 1]")
    if record.classifier_authority_epoch < 0:
        raise ClassificationError("authority epoch must be non-negative")
    facets = dict(record.facets)
    if facets.get("form") not in VALID_FORMS:
        raise ClassificationError("invalid form facet")
    if facets.get("lifecycle") not in VALID_LIFECYCLES:
        raise ClassificationError("invalid lifecycle facet")
    if facets.get("evidence") not in VALID_EVIDENCE:
        raise ClassificationError("invalid evidence facet")
    if facets.get("sensitivity") not in SENSITIVITY_ORDER:
        raise ClassificationError("invalid sensitivity facet")
    if not record.evidence_refs:
        raise ClassificationError("classification evidence is required")


def classification_frontmatter(record: ClassificationRecord) -> dict[str, Any]:
    """Return flat frontmatter fields compatible with the existing parser."""

    payload = record.to_dict()
    return {
        "classification_schema": CLASSIFICATION_SCHEMA,
        "classification_primary": record.primary.concept_uri,
        "classification_notation": record.primary.notation,
        "classification_status": record.status,
        "classification_confidence": f"{record.confidence:.6f}",
        "classification_authority_epoch": str(record.classifier_authority_epoch),
        "classification_json": json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    }


def record_from_frontmatter(meta: Mapping[str, Any]) -> ClassificationRecord | None:
    raw = meta.get("classification_json")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClassificationError("invalid classification_json") from exc
    return record_from_dict(payload)


def record_from_dict(payload: Mapping[str, Any]) -> ClassificationRecord:
    try:
        primary = Subject(**dict(payload["primary"]))
        secondary = tuple(
            Subject(**dict(value)) for value in payload.get("secondary") or []
        )
        return ClassificationRecord(
            schema=str(payload["schema"]),
            subject_scheme=str(payload["subject_scheme"]),
            subject_release=str(payload["subject_release"]),
            subject_checksum=str(payload["subject_checksum"]),
            primary=primary,
            secondary=secondary,
            facets=dict(payload["facets"]),
            confidence=float(payload["confidence"]),
            evidence_refs=tuple(str(value) for value in payload["evidence_refs"]),
            classifier_authority_epoch=int(payload["classifier_authority_epoch"]),
            status=str(payload.get("status") or "proposed"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ClassificationError(f"invalid classification record: {exc}") from exc


_DOMAIN_PREFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ai", "software", "computer", "information", "library", "memory"), "0"),
    (("philosophy", "psychology"), "1"),
    (("religion", "theology"), "2"),
    (("career", "business", "econom", "law", "politic", "society", "management"), "3"),
    (("math", "physics", "science", "biology", "chemistry"), "5"),
    (("engineering", "health", "medicine", "technology", "hardware"), "6"),
    (("art", "music", "sport", "game", "recreation"), "7"),
    (("language", "linguistic", "literature", "writing"), "8"),
    (("history", "geography", "biography"), "9"),
)


def _tag_tokens(tags: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for raw in tags:
        text = str(raw).casefold()
        tokens.add(text)
        tokens.update(part for part in text.replace("/", "-").split("-") if part)
    return tokens


def _form_from_tags(tags: Iterable[str], page_type: str) -> str:
    combined = " ".join(str(value).casefold() for value in tags)
    for candidate in VALID_FORMS:
        if f"t/{candidate}" in combined or candidate in combined.split():
            return candidate
    return page_type if page_type in VALID_FORMS else "knowledge"


def propose_from_legacy_metadata(
    *,
    tags: Iterable[str],
    page_type: str,
    lifecycle: str,
    sensitivity: str,
    evidence_ref: str,
    package: UDCPackage | None = None,
) -> ClassificationRecord:
    """Create a deterministic *shadow* proposal from legacy metadata.

    This deliberately cannot activate authority.  It is a migration seed used
    for distribution measurement and later local-model review.
    """

    package = package or default_udc_package()
    tokens = _tag_tokens(tags)
    scores: dict[str, int] = {}
    for prefixes, notation in _DOMAIN_PREFIXES:
        score = sum(
            1
            for token in tokens
            if any(token.startswith(prefix) for prefix in prefixes)
        )
        if score:
            scores[notation] = score
    notation = max(scores, key=lambda key: (scores[key], key)) if scores else "0"
    concept = package.by_notation(notation)
    if concept is None:
        raise ClassificationError(f"bootstrap package lacks notation {notation}")
    confidence = min(0.79, 0.45 + 0.08 * scores.get(notation, 0))
    status = "proposed" if scores else "held"
    record = ClassificationRecord(
        schema=CLASSIFICATION_SCHEMA,
        subject_scheme="udcs",
        subject_release=package.release,
        subject_checksum=package.checksum,
        primary=Subject(
            concept_uri=str(concept["uri"]),
            notation=notation,
            label=str(concept.get("label_en") or concept.get("label") or notation),
            label_source="udcs-top-level-bootstrap-en",
        ),
        secondary=(),
        facets={
            "project": [],
            "form": _form_from_tags(tags, page_type),
            "lifecycle": (lifecycle if lifecycle in VALID_LIFECYCLES else "active"),
            "temporal": {"kind": "evergreen"},
            "evidence": "derived",
            "sensitivity": (
                sensitivity if sensitivity in SENSITIVITY_ORDER else "normal"
            ),
        },
        confidence=confidence,
        evidence_refs=(evidence_ref,),
        classifier_authority_epoch=0,
        status=status,
    )
    validate_record(record, package=package)
    return record


def strongest_sensitivity(values: Iterable[str]) -> str:
    normalized = [str(value) for value in values]
    if not normalized:
        return "normal"
    return max(normalized, key=lambda value: SENSITIVITY_ORDER.get(value, 2))


def validate_controlled_subject(
    subject: ControlledSubject,
    *,
    package: UDCPackage,
) -> None:
    if not subject.concept_uri.startswith("cvo:subject/"):
        raise ClassificationError("controlled subject must use cvo:subject namespace")
    if subject.broader_udc_uri not in package.concepts:
        raise ClassificationError("controlled subject broader UDC concept is unknown")
    if not subject.label.strip() or not subject.definition.strip():
        raise ClassificationError("controlled subject requires label and definition")
    if not subject.version.strip() or subject.authority_epoch < 1:
        raise ClassificationError(
            "controlled subject requires version and positive authority epoch"
        )
    if not subject.inclusion_examples or not subject.exclusion_examples:
        raise ClassificationError(
            "controlled subject requires inclusion and exclusion examples"
        )


_FORM_ABBREVIATIONS = {
    "decision": "DEC",
    "event": "EVT",
    "howto": "HOW",
    "reference": "REF",
    "architecture": "ARC",
    "analysis": "ANL",
    "state": "STA",
    "profile": "PRO",
    "knowledge": "KNW",
}


def render_call_number(
    record: ClassificationRecord,
    *,
    project: str | None = None,
) -> str:
    """Render a mutable human-facing call number (never a page identity)."""

    facets = dict(record.facets)
    form = _FORM_ABBREVIATIONS.get(str(facets.get("form") or ""), "KNW")
    temporal = facets.get("temporal")
    temporal = temporal if isinstance(temporal, Mapping) else {}
    temporal_value = str(temporal.get("value") or "")
    if temporal.get("kind") == "date" and len(temporal_value) >= 4:
        temporal_value = temporal_value[:4]
    project_value = project
    if not project_value:
        projects = facets.get("project")
        if isinstance(projects, list) and projects:
            project_value = str(projects[0]).rsplit("/", 1)[-1]
    parts = [
        f"UDCS {record.primary.notation}",
        str(project_value or "").upper(),
        form,
        temporal_value,
    ]
    return " · ".join(value for value in parts if value)


def load_ndc_overlay(root: Path) -> dict[str, Any] | None:
    """Load an optional user-provided licensed NDC crosswalk.

    The repository never bundles NDC data.  A local overlay is accepted only
    when its schema and explicit license acknowledgement are present.
    """

    path = root / "classification" / "overlays" / "ndc-crosswalk.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ClassificationError(f"invalid NDC overlay JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != NDC_OVERLAY_SCHEMA:
        raise ClassificationError("unsupported NDC overlay schema")
    if payload.get("license_acknowledged") is not True:
        raise ClassificationError("NDC overlay requires license acknowledgement")
    mappings = payload.get("mappings")
    if not isinstance(mappings, list):
        raise ClassificationError("NDC overlay mappings must be a list")
    return payload
