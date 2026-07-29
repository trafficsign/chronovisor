"""Deterministic source coverage gates and append-only merge receipts."""

from __future__ import annotations

from chronovisor.hashutil import sha256_text as _sha256_text

import fcntl
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from chronovisor.classification import strongest_sensitivity

LEDGER_SCHEMA = "chronovisor.merge-ledger.v1"
RECEIPT_SCHEMA = "chronovisor.merge-receipt.v1"
_SENTENCE_RE = re.compile(r".+?(?:[。！？!?]+(?:[」』）)]*)|\n{2,}|$)", re.S)
_URL_RE = re.compile(r"https?://[^\s<>()]+")
_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}(?:[-/.年](?:0?[1-9]|1[0-2])(?:[-/.月](?:0?[1-9]|[12]\d|3[01])日?)?)?\b"
)
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*(?:%|ms|s|GB|MB|B)?")
_CODE_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\.|-)[A-Za-z0-9_]+)+\b")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)\b")


class MergeCoverageError(ValueError):
    """Raised when a merge proposal loses deterministic source evidence."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")




def split_source_spans(text: str) -> list[dict[str, Any]]:
    """Split a page into stable, byte-addressable semantic spans."""

    spans: list[dict[str, Any]] = []
    for index, match in enumerate(_SENTENCE_RE.finditer(text)):
        value = match.group(0)
        if not value:
            continue
        stripped = value.strip()
        if not stripped:
            kind = "boilerplate"
        elif stripped == "---" or stripped.startswith(("title:", "updated:", "uid:")):
            kind = "boilerplate"
        else:
            kind = "claim"
        spans.append(
            {
                "index": index,
                "start": match.start(),
                "end": match.end(),
                "sha256": _sha256_text(value),
                "kind": kind,
                "text": value,
            }
        )
    return spans


def extract_fingerprints(text: str) -> dict[str, list[str]]:
    """Extract exact high-risk facts that must survive synthesis."""

    return {
        "urls": sorted(set(_URL_RE.findall(text))),
        "dates": sorted(set(_DATE_RE.findall(text))),
        "numbers": sorted(set(_NUMBER_RE.findall(text))),
        "code_identifiers": sorted(set(_CODE_RE.findall(text))),
        "entities": sorted(set(_ENTITY_RE.findall(text))),
    }


def build_source_inventory(sources: Mapping[str, str]) -> dict[str, Any]:
    return {
        uid: {
            "content_sha256": _sha256_text(text),
            "spans": split_source_spans(text),
            "fingerprints": extract_fingerprints(text),
        }
        for uid, text in sorted(sources.items())
    }


def verify_merge_coverage(
    *,
    inventory: Mapping[str, Any],
    mappings: Iterable[Mapping[str, Any]],
    output_text: str,
    ledger_dispositions: Iterable[Mapping[str, Any]] = (),
    input_sensitivities: Iterable[str] = (),
    output_sensitivity: str,
    require_raw_refs: bool = False,
) -> dict[str, Any]:
    """Fail closed unless every span and deterministic fingerprint is mapped."""

    mapping_rows = [dict(value) for value in mappings]
    dispositions = [dict(value) for value in ledger_dispositions]
    missing_raw_refs = (
        [
            {
                "source_uid": str(row.get("source_uid") or ""),
                "span_sha256": str(row.get("span_sha256") or ""),
            }
            for row in mapping_rows
            if row.get("action") in {"output", "ledger"}
            and (
                not isinstance(row.get("raw_refs"), list)
                or not any(str(value).strip() for value in row.get("raw_refs") or [])
            )
        ]
        if require_raw_refs
        else []
    )
    span_lookup: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for source_uid, source in inventory.items():
        if not isinstance(source, Mapping):
            raise MergeCoverageError(f"invalid source inventory for {source_uid}")
        for span in source.get("spans") or []:
            if isinstance(span, Mapping):
                span_lookup.setdefault(
                    (str(source_uid), str(span.get("sha256") or "")), []
                ).append(span)

    mapped: set[tuple[str, int, str]] = set()
    invalid_mappings: list[dict[str, Any]] = []
    for row in mapping_rows:
        action = str(row.get("action") or "")
        if action not in {"output", "ledger", "boilerplate"}:
            continue
        source_uid = str(row.get("source_uid") or "")
        span_sha256 = str(row.get("span_sha256") or "")
        candidates = span_lookup.get((source_uid, span_sha256), [])
        if row.get("span_index") is not None:
            try:
                wanted_index = int(row["span_index"])
            except (TypeError, ValueError):
                candidates = []
            else:
                candidates = [
                    span
                    for span in candidates
                    if int(span.get("index") or 0) == wanted_index
                ]
        if len(candidates) != 1:
            invalid_mappings.append(
                {
                    "source_uid": source_uid,
                    "span_sha256": span_sha256,
                    "span_index": row.get("span_index"),
                    "reason": (
                        "unknown_span" if not candidates else "ambiguous_repeated_span"
                    ),
                }
            )
            continue
        span = candidates[0]
        if action == "boilerplate" and span.get("kind") != "boilerplate":
            invalid_mappings.append(
                {
                    "source_uid": source_uid,
                    "span_sha256": span_sha256,
                    "span_index": span.get("index"),
                    "reason": "claim_cannot_be_declared_boilerplate",
                }
            )
            continue
        mapped.add(
            (
                source_uid,
                int(span.get("index") or 0),
                span_sha256,
            )
        )

    missing_spans: list[dict[str, Any]] = []
    all_fingerprints: dict[str, set[str]] = {}
    for source_uid, source in inventory.items():
        for span in source.get("spans") or []:
            if not isinstance(span, Mapping):
                continue
            key = (
                str(source_uid),
                int(span.get("index") or 0),
                str(span.get("sha256")),
            )
            if key not in mapped:
                missing_spans.append(
                    {
                        "source_uid": str(source_uid),
                        "span_index": int(span.get("index") or 0),
                        "span_sha256": str(span.get("sha256") or ""),
                    }
                )
        fingerprints = source.get("fingerprints")
        if isinstance(fingerprints, Mapping):
            for kind, values in fingerprints.items():
                all_fingerprints.setdefault(str(kind), set()).update(
                    str(value) for value in values if str(value)
                )
    disposition_text = json.dumps(
        dispositions, ensure_ascii=False, sort_keys=True, default=str
    )
    missing_output_spans: list[dict[str, Any]] = []
    missing_output_anchors: list[dict[str, Any]] = []
    missing_ledger_spans: list[dict[str, Any]] = []
    for row in mapping_rows:
        action = str(row.get("action") or "")
        if action not in {"output", "ledger"}:
            continue
        source_uid = str(row.get("source_uid") or "")
        span_sha256 = str(row.get("span_sha256") or "")
        candidates = span_lookup.get((source_uid, span_sha256), [])
        if row.get("span_index") is not None:
            try:
                wanted_index = int(row["span_index"])
            except (TypeError, ValueError):
                candidates = []
            else:
                candidates = [
                    span
                    for span in candidates
                    if int(span.get("index") or 0) == wanted_index
                ]
        if len(candidates) != 1:
            continue
        span_text = str(candidates[0].get("text") or "")
        if action == "output" and span_text not in output_text:
            missing_output_spans.append(
                {
                    "source_uid": source_uid,
                    "span_index": candidates[0].get("index"),
                    "span_sha256": span_sha256,
                }
            )
        if action == "output" and require_raw_refs:
            output_anchor = str(row.get("output_anchor") or "").strip()
            anchor_present = bool(
                output_anchor
                and (
                    f"^{output_anchor}" in output_text
                    or any(
                        line.startswith("#")
                        and line.lstrip("#").strip().casefold().replace(" ", "-")
                        == output_anchor.casefold().replace(" ", "-")
                        for line in output_text.splitlines()
                    )
                )
            )
            if not anchor_present:
                missing_output_anchors.append(
                    {
                        "source_uid": source_uid,
                        "span_index": candidates[0].get("index"),
                        "span_sha256": span_sha256,
                        "output_anchor": output_anchor,
                    }
                )
        if (
            action == "ledger"
            and span_text not in disposition_text
            and span_sha256 not in disposition_text
        ):
            missing_ledger_spans.append(
                {
                    "source_uid": source_uid,
                    "span_index": candidates[0].get("index"),
                    "span_sha256": span_sha256,
                }
            )
    missing_fingerprints = {
        kind: sorted(
            value
            for value in values
            if value not in output_text and value not in disposition_text
        )
        for kind, values in all_fingerprints.items()
    }
    missing_fingerprints = {
        kind: values for kind, values in missing_fingerprints.items() if values
    }
    required_sensitivity = strongest_sensitivity(input_sensitivities)
    sensitivity_ok = (
        strongest_sensitivity([required_sensitivity, output_sensitivity])
        == output_sensitivity
    )
    if (
        missing_spans
        or missing_output_spans
        or missing_output_anchors
        or missing_ledger_spans
        or missing_fingerprints
        or missing_raw_refs
        or invalid_mappings
        or not sensitivity_ok
    ):
        raise MergeCoverageError(
            json.dumps(
                {
                    "missing_spans": missing_spans,
                    "missing_output_spans": missing_output_spans,
                    "missing_output_anchors": missing_output_anchors,
                    "missing_ledger_spans": missing_ledger_spans,
                    "missing_fingerprints": missing_fingerprints,
                    "missing_raw_refs": missing_raw_refs,
                    "invalid_mappings": invalid_mappings,
                    "required_sensitivity": required_sensitivity,
                    "output_sensitivity": output_sensitivity,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return {
        "status": "verified",
        "source_count": len(inventory),
        "span_count": sum(
            len(source.get("spans") or [])
            for source in inventory.values()
            if isinstance(source, Mapping)
        ),
        "fingerprint_count": sum(len(values) for values in all_fingerprints.values()),
        "raw_ref_count": sum(
            len(row.get("raw_refs") or [])
            for row in mapping_rows
            if isinstance(row.get("raw_refs"), list)
        ),
        "sensitivity": output_sensitivity,
    }


class MergeLedger:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / "runtime" / "librarian" / "merge-ledger.jsonl"
        self.lock_path = self.path.with_suffix(".lock")

    def append(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        row = {
            "schema": RECEIPT_SCHEMA,
            "recorded_at": _now_iso(),
            **dict(receipt),
        }
        encoded = (
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            os.chmod(self.lock_path, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    os.chmod(self.path, 0o600)
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return row

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines()[
            -max(0, limit) :
        ]:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows
