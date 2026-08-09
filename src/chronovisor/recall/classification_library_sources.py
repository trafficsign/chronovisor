"""Rights-aware acquisition and normalization for library evidence sources.

Downloaded records remain runtime artifacts.  This module deliberately does
not bundle catalog data in the wheel and does not infer rights for unknown
providers.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.jsonl_write import atomic_replace_bytes as atomic_write
from chronovisor.core.timeutil import utc_iso_milliseconds as _now
from chronovisor.recall.classification import ClassificationError
from chronovisor.recall.classification_fixture_contract import (
    sha256_bytes,
    sha256_file,
)

EXTERNAL_PACKAGE_SCHEMA = "chronovisor.external-library-package.v1"
EXTERNAL_RECORD_SCHEMA = "chronovisor.external-library-record.v1"
OAI_NS = "http://www.openarchives.org/OAI/2.0/"
MARC_NS = "http://www.loc.gov/MARC21/slim"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
MAX_DOWNLOAD_BYTES = 2 * 1024**3

_atomic_write = atomic_write






def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class SourceContract:
    source_name: str
    source_url: str
    record_license: str
    scheme_license: str
    vocabulary_license: str
    redistribution_policy: Mapping[str, bool]
    attribution: str
    field_allowlist: tuple[str, ...]
    rights_by_field: Mapping[str, str]
    provider_allowlist: tuple[str, ...] = ()
    software_license: str = "not-applicable"
    model_license: str = "not-applicable"
    training_corpus_license: str = "not-applicable"
    size_cap_bytes: int = MAX_DOWNLOAD_BYTES

    def validate(self) -> None:
        required = {
            "source_name": self.source_name,
            "source_url": self.source_url,
            "record_license": self.record_license,
            "scheme_license": self.scheme_license,
            "vocabulary_license": self.vocabulary_license,
            "software_license": self.software_license,
            "model_license": self.model_license,
            "training_corpus_license": self.training_corpus_license,
            "attribution": self.attribution,
        }
        missing = sorted(key for key, value in required.items() if not value)
        if missing:
            raise ClassificationError(
                f"source contract lacks required rights fields: {missing}"
            )
        if not self.field_allowlist or not self.rights_by_field:
            raise ClassificationError("source contract must bind fields to rights")
        if self.size_cap_bytes <= 0 or self.size_cap_bytes > MAX_DOWNLOAD_BYTES:
            raise ClassificationError("source contract size cap is unsafe")


def czech_bibliography_contract(source_url: str) -> SourceContract:
    return SourceContract(
        source_name="czech-national-bibliography",
        source_url=source_url,
        record_license="CC0-1.0",
        scheme_license="UDC notation rights retained by source; no schedule copied",
        vocabulary_license="Czech National Library open-data terms",
        redistribution_policy={
            "download": True,
            "runtime_index": True,
            "export_normalized_records": True,
            "bundle_in_repo_or_wheel": False,
        },
        attribution="National Library of the Czech Republic, Czech National Bibliography",
        field_allowlist=("001", "008", "020", "022", "041", "080", "245", "650", "883"),
        rights_by_field={
            "descriptive_metadata": "CC0-1.0",
            "bibliographic_subjects": "CC0-1.0",
            "udc_assignments": "observed notation only; no UDC captions",
        },
    )


def czech_authority_contract(source_url: str) -> SourceContract:
    return SourceContract(
        source_name="czech-topical-authorities",
        source_url=source_url,
        record_license="CC0-1.0",
        scheme_license="UDC notation rights retained by source; no schedule copied",
        vocabulary_license="Czech National Library authority open-data terms",
        redistribution_policy={
            "download": True,
            "runtime_index": True,
            "export_normalized_records": True,
            "bundle_in_repo_or_wheel": False,
        },
        attribution="National Library of the Czech Republic, Topical Authorities",
        field_allowlist=("001", "150", "450", "550", "089"),
        rights_by_field={
            "authority_labels": "CC0-1.0",
            "authority_links": "CC0-1.0",
            "udc_assignments": "089$a observed notation only",
        },
    )


def ndl_bibliography_contract(
    source_url: str,
    *,
    provider_allowlist: Sequence[str],
) -> SourceContract:
    if not provider_allowlist:
        raise ClassificationError("NDL bibliography requires a provider allowlist")
    return SourceContract(
        source_name="ndl-created-bibliography",
        source_url=source_url,
        record_license="provider-specific-allowlist",
        scheme_license="NDC/NDLC diagnostic references only",
        vocabulary_license="provider-specific-allowlist",
        redistribution_policy={
            "download": True,
            "runtime_index": True,
            "export_normalized_records": False,
            "bundle_in_repo_or_wheel": False,
        },
        attribution="National Diet Library Search and the allowlisted record creator",
        field_allowlist=(
            "provider_id",
            "record_creator",
            "provider_terms_url",
            "record_license_class",
            "title",
            "language",
            "subjects",
            "classifications",
            "identifiers",
        ),
        rights_by_field={
            "all": "provider-specific; reject unless every rights field is allowlisted"
        },
        provider_allowlist=tuple(sorted(set(provider_allowlist))),
    )


def ndlsh_contract(source_url: str) -> SourceContract:
    return SourceContract(
        source_name="ndlsh-authority",
        source_url=source_url,
        record_license="NDL open data terms",
        scheme_license="NDC/NDLC diagnostic references only",
        vocabulary_license="NDL open data terms",
        redistribution_policy={
            "download": True,
            "runtime_index": True,
            "export_normalized_records": True,
            "bundle_in_repo_or_wheel": False,
        },
        attribution="National Diet Library, Web NDL Authorities",
        field_allowlist=(
            "prefLabel",
            "altLabel",
            "broader",
            "narrower",
            "related",
            "closeMatch",
            "notation",
        ),
        rights_by_field={
            "authority_labels": "NDL open data terms",
            "relations": "preserve source predicate; no inferred exactMatch",
            "representative_classification": "diagnostic only",
        },
    )


def _stable_bucket(record_id: str, modulus: int) -> int:
    return int(hashlib.sha256(record_id.encode("utf-8")).hexdigest(), 16) % modulus


def stable_sample(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    strata: Sequence[str] = ("language", "major_class", "year_bucket"),
) -> list[dict[str, Any]]:
    """Select a deterministic interleaved sample instead of a leading prefix."""

    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        record_id = str(row.get("source_record_id") or "")
        if not record_id:
            continue
        key = tuple(str(row.get(field) or "unknown") for field in strata)
        buckets.setdefault(key, []).append(row)
    for _key, values in buckets.items():
        values.sort(
            key=lambda row: (
                _stable_bucket(str(row["source_record_id"]), 2**31 - 1),
                str(row["source_record_id"]),
            )
        )
    output: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while keys and len(output) < limit:
        remaining = []
        for key in keys:
            values = buckets[key]
            if values and len(output) < limit:
                output.append(values.pop(0))
            if values:
                remaining.append(key)
        keys = remaining
    return output


def _parse_link_key(values: Sequence[str]) -> str | None:
    for value in values:
        head = value.strip().split("/", 1)[0].split("\\", 1)[0]
        if head:
            return head
    return None


def _subfields(field: ET.Element) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for child in field.findall(f"{{{MARC_NS}}}subfield"):
        code = str(child.attrib.get("code") or "")
        text = (child.text or "").strip()
        if code and text:
            values.setdefault(code, []).append(text)
    return values


def _control(record: ET.Element, tag: str) -> str:
    field = record.find(f"{{{MARC_NS}}}controlfield[@tag='{tag}']")
    return (field.text or "").strip() if field is not None else ""


def _datafields(record: ET.Element, tag: str) -> list[ET.Element]:
    return list(record.findall(f"{{{MARC_NS}}}datafield[@tag='{tag}']"))


def _assignment_883(record: ET.Element) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for field in _datafields(record, "883"):
        values = _subfields(field)
        key = _parse_link_key(values.get("8") or [])
        if not key:
            continue
        generation = str(field.attrib.get("ind1") or "").strip()
        method = {
            "0": "fully_machine_generated",
            "1": "partially_machine_generated",
            "2": "not_machine_generated",
        }.get(generation, "unknown")
        output[key] = {
            "raw_subfield_8": (values.get("8") or [None])[0],
            "parsed_field_link_key": key,
            "first_indicator": generation or None,
            "generation_process": (values.get("a") or [None])[0],
            "agency": (values.get("q") or [None])[0],
            "generation_method": method,
        }
    return output


def normalize_czech_bibliographic_record(
    record: ET.Element,
) -> dict[str, Any]:
    record_id = _control(record, "001")
    if not record_id:
        raise ClassificationError("Czech bibliographic record lacks 001")
    links_883 = _assignment_883(record)
    assignments: list[dict[str, Any]] = []
    for field in _datafields(record, "080"):
        values = _subfields(field)
        link_key = _parse_link_key(values.get("8") or [])
        provenance = links_883.get(link_key or "")
        assignments.append(
            {
                "scheme": "UDC",
                "role": "bibliographic_assignment",
                "notation_or_uri": (values.get("a") or [""])[0],
                "raw_field": ET.tostring(field, encoding="unicode"),
                "authority_record_id": None,
                "authority_status": None,
                "source_field": "080",
                "indicators": {
                    "first": field.attrib.get("ind1"),
                    "second": field.attrib.get("ind2"),
                },
                "components": {
                    "a": values.get("a") or [],
                    "x": values.get("x") or [],
                    "subfield_2": values.get("2") or [],
                    "subfield_8": values.get("8") or [],
                },
                "edition": (values.get("2") or [None])[0],
                "assigned_by": None,
                "generation_method": (
                    provenance.get("generation_method") if provenance else "unknown"
                ),
                "intellectual_assignment": (
                    "confirmed"
                    if provenance
                    and provenance.get("generation_method") == "not_machine_generated"
                    else "unconfirmed"
                ),
                "provenance_883": provenance,
            }
        )
    title_fields = _datafields(record, "245")
    title_values = _subfields(title_fields[0]) if title_fields else {}
    subjects = []
    for field in _datafields(record, "650"):
        values = _subfields(field)
        label = " -- ".join(values.get("a", []) + values.get("x", []))
        if label:
            subjects.append(
                {
                    "uri": None,
                    "pref_label": label,
                    "alt_labels": [],
                    "relation": "bibliographic-subject",
                }
            )
    fixed = _control(record, "008")
    language = fixed[35:38].strip() if len(fixed) >= 38 else ""
    output = {
        "schema": EXTERNAL_RECORD_SCHEMA,
        "source_record_id": record_id,
        "provider_id": "cz-nkp",
        "record_creator": "National Library of the Czech Republic",
        "provider_terms_url": "https://www.nkp.cz/en/about-us/professional-activities/open-data",
        "record_license_class": "CC0-1.0",
        "rights_ref": "czech-national-bibliography",
        "title": " ".join(title_values.get("a", []) + title_values.get("b", [])).strip(
            " /:"
        ),
        "language": language or "unknown",
        "subject_headings": subjects,
        "source_assignments": assignments,
        "identifiers": {},
        "major_class": (
            str(assignments[0]["notation_or_uri"])[:1] if assignments else "unknown"
        ),
        "year_bucket": fixed[7:11] if len(fixed) >= 11 else "unknown",
    }
    output["record_sha256"] = sha256_bytes(_canonical_json(output))
    return output


def normalize_czech_authority_record(record: ET.Element) -> dict[str, Any]:
    record_id = _control(record, "001")
    if not record_id:
        raise ClassificationError("Czech authority record lacks 001")
    preferred_fields = _datafields(record, "150")
    preferred = _subfields(preferred_fields[0]) if preferred_fields else {}
    alt = []
    for field in _datafields(record, "450"):
        alt.extend(_subfields(field).get("a") or [])
    relations = []
    for field in _datafields(record, "550"):
        values = _subfields(field)
        for label in values.get("a") or []:
            relations.append(
                {
                    "uri": None,
                    "pref_label": label,
                    "alt_labels": [],
                    "relation": (values.get("w") or ["related"])[0],
                }
            )
    assignments = []
    for field in _datafields(record, "089"):
        values = _subfields(field)
        for notation in values.get("a") or []:
            assignments.append(
                {
                    "scheme": "UDC",
                    "role": "authority_representative_classification",
                    "notation_or_uri": notation,
                    "raw_field": None,
                    "authority_record_id": record_id,
                    "authority_status": "source-record",
                    "source_field": "089$a",
                    "indicators": {
                        "first": field.attrib.get("ind1"),
                        "second": field.attrib.get("ind2"),
                    },
                    "components": {
                        "a": [notation],
                        "x": [],
                        "subfield_2": [],
                        "subfield_8": [],
                    },
                    "edition": None,
                    "assigned_by": None,
                    "generation_method": "unknown",
                    "intellectual_assignment": "unconfirmed",
                    "provenance_883": None,
                }
            )
    output = {
        "schema": EXTERNAL_RECORD_SCHEMA,
        "source_record_id": record_id,
        "provider_id": "cz-nkp-authority",
        "record_creator": "National Library of the Czech Republic",
        "provider_terms_url": "https://www.nkp.cz/en/about-us/professional-activities/open-data",
        "record_license_class": "CC0-1.0",
        "rights_ref": "czech-topical-authorities",
        "title": " ".join(preferred.get("a") or []),
        "language": "cze",
        "subject_headings": [
            {
                "uri": None,
                "pref_label": " ".join(preferred.get("a") or []),
                "alt_labels": alt,
                "relation": "preferred",
            },
            *relations,
        ],
        "source_assignments": assignments,
        "identifiers": {},
        "major_class": (
            str(assignments[0]["notation_or_uri"])[:1] if assignments else "unknown"
        ),
        "year_bucket": "authority",
    }
    output["record_sha256"] = sha256_bytes(_canonical_json(output))
    return output


def parse_marcxml_records(
    source: bytes | Path | Any,
    *,
    authority: bool,
) -> Iterator[dict[str, Any]]:
    if isinstance(source, Path):
        stream: Any = source
    elif isinstance(source, bytes):
        stream = io.BytesIO(source)
    else:
        stream = source
    for _event, element in ET.iterparse(stream, events=("end",)):
        if element.tag != f"{{{MARC_NS}}}record":
            continue
        try:
            yield (
                normalize_czech_authority_record(element)
                if authority
                else normalize_czech_bibliographic_record(element)
            )
        finally:
            element.clear()


def normalize_ndlsh_rdf(source: bytes | Path | Any) -> Iterator[dict[str, Any]]:
    stream: Any = (
        source
        if isinstance(source, Path)
        else (io.BytesIO(source) if isinstance(source, bytes) else source)
    )
    predicates = {
        f"{{{SKOS_NS}}}prefLabel": "pref_label",
        f"{{{SKOS_NS}}}altLabel": "alt_labels",
        f"{{{SKOS_NS}}}broader": "broader",
        f"{{{SKOS_NS}}}narrower": "narrower",
        f"{{{SKOS_NS}}}related": "related",
        f"{{{SKOS_NS}}}closeMatch": "close_match",
        f"{{{SKOS_NS}}}notation": "notations",
    }
    for _event, element in ET.iterparse(stream, events=("end",)):
        uri = element.attrib.get(f"{{{RDF_NS}}}about")
        if not uri:
            continue
        row: dict[str, Any] = {
            "schema": EXTERNAL_RECORD_SCHEMA,
            "source_record_id": uri,
            "provider_id": "ndlsh",
            "record_creator": "National Diet Library",
            "provider_terms_url": "https://id.ndl.go.jp/information/download_en/",
            "record_license_class": "NDL-open-data-terms",
            "rights_ref": "ndlsh-authority",
            "title": "",
            "language": "jpn",
            "subject_headings": [],
            "source_assignments": [],
            "identifiers": {},
            "diagnostic_relations": [],
            "major_class": "authority",
            "year_bucket": "authority",
        }
        labels: dict[str, list[str]] = {
            "pref_label": [],
            "alt_labels": [],
        }
        for child in list(element):
            field = predicates.get(child.tag)
            if not field:
                continue
            value = (
                child.attrib.get(f"{{{RDF_NS}}}resource") or (child.text or "").strip()
            )
            if not value:
                continue
            if field in labels:
                labels[field].append(value)
            else:
                row["diagnostic_relations"].append(
                    {"predicate": field, "target": value}
                )
        if not labels["pref_label"]:
            continue
        row["title"] = labels["pref_label"][0]
        row["subject_headings"] = [
            {
                "uri": uri,
                "pref_label": labels["pref_label"][0],
                "alt_labels": labels["alt_labels"],
                "relation": "preferred",
            }
        ]
        row["record_sha256"] = sha256_bytes(_canonical_json(row))
        yield row
        element.clear()


def validate_ndl_provider(
    row: Mapping[str, Any],
    contract: SourceContract,
) -> tuple[bool, str | None]:
    values = (
        str(row.get("provider_id") or ""),
        str(row.get("record_creator") or ""),
        str(row.get("provider_terms_url") or ""),
        str(row.get("record_license_class") or ""),
    )
    if any(not value for value in values):
        return False, "missing_provider_rights"
    if values[0] not in contract.provider_allowlist:
        return False, "provider_not_allowlisted"
    return True, None


def fetch_oai_window(
    *,
    base_url: str,
    metadata_prefix: str,
    from_date: str,
    until_date: str,
    set_spec: str | None = None,
    timeout_seconds: float = 60.0,
    size_cap_bytes: int = MAX_DOWNLOAD_BYTES,
    extract_oai_records: bool = False,
    checkpoint_dir: Path | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Fetch a fixed OAI-PMH window while preserving every response page."""

    if size_cap_bytes <= 0 or size_cap_bytes > MAX_DOWNLOAD_BYTES:
        raise ClassificationError("OAI size cap is unsafe")
    identity = {
        "base_url": base_url,
        "metadata_prefix": metadata_prefix,
        "from": from_date,
        "until": until_date,
        "set_spec": set_spec,
        "extract_oai_records": extract_oai_records,
    }
    identity_sha256 = sha256_bytes(_canonical_json(identity))
    checkpoint_path = (
        checkpoint_dir / "checkpoint.json" if checkpoint_dir is not None else None
    )
    checkpoint = (
        read_sealed_json(checkpoint_path)
        if checkpoint_path is not None and checkpoint_path.exists()
        else {
            "schema": "chronovisor.oai-checkpoint.v1",
            "identity_sha256": identity_sha256,
            "pages": [],
            "next_token": None,
            "complete": False,
            "response_bytes": 0,
        }
    )
    if checkpoint.get("identity_sha256") != identity_sha256:
        raise ClassificationError("OAI checkpoint belongs to another source window")
    pages = list(checkpoint.get("pages") or [])
    resumed_pages = len(pages)
    for page in pages:
        path = Path(str(page.get("path") or ""))
        if (
            not path.is_file()
            or sha256_file(path) != page.get("sha256")
            or path.stat().st_size != int(page.get("bytes") or -1)
        ):
            raise ClassificationError("OAI checkpoint page is missing or corrupt")
    token = checkpoint.get("next_token")
    total = int(checkpoint.get("response_bytes") or 0)
    requests = len(pages)
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        while not checkpoint.get("complete"):
            params = (
                {"verb": "ListRecords", "resumptionToken": token}
                if token
                else {
                    "verb": "ListRecords",
                    "metadataPrefix": metadata_prefix,
                    "from": from_date,
                    "until": until_date,
                    **({"set": set_spec} if set_spec else {}),
                }
            )
            response = client.get(base_url, params=params)
            response.raise_for_status()
            content = response.content
            total += len(content)
            if total > size_cap_bytes:
                raise ClassificationError("OAI response exceeded package size cap")
            requests += 1
            xml = ET.fromstring(content)
            record_tag = OAI_NS if extract_oai_records else MARC_NS
            records = [
                ET.tostring(record, encoding="utf-8")
                for record in xml.findall(f".//{{{record_tag}}}record")
            ]
            page_bytes = b"\n".join(records)
            node = xml.find(f".//{{{OAI_NS}}}resumptionToken")
            token = (node.text or "").strip() if node is not None else ""
            if checkpoint_dir is not None:
                page_path = checkpoint_dir / "pages" / f"{requests:06d}.xml"
                _atomic_write(page_path, page_bytes)
                pages.append(
                    {
                        "path": str(page_path),
                        "sha256": sha256_file(page_path),
                        "bytes": len(page_bytes),
                        "record_count": len(records),
                    }
                )
                checkpoint.update(
                    {
                        "pages": pages,
                        "next_token": token or None,
                        "complete": not bool(token),
                        "response_bytes": total,
                    }
                )
                write_sealed_json(checkpoint_path, checkpoint, backup=True)
            else:
                pages.append(
                    {
                        "bytes_data": page_bytes,
                        "record_count": len(records),
                    }
                )
            if not token:
                checkpoint["complete"] = True
                break
    record_pages = [
        (
            Path(str(page["path"])).read_bytes()
            if "path" in page
            else bytes(page.get("bytes_data") or b"")
        )
        for page in pages
    ]
    record_count = sum(int(page.get("record_count") or 0) for page in pages)
    if extract_oai_records:
        joined = (
            f'<OAI-PMH xmlns="{OAI_NS}"><ListRecords>'.encode()
            + b"\n".join(record_pages)
            + b"</ListRecords></OAI-PMH>"
        )
    else:
        joined = (
            f'<collection xmlns="{MARC_NS}">'.encode()
            + b"\n".join(record_pages)
            + b"</collection>"
        )
    return joined, {
        "request_count": requests,
        "record_count": record_count,
        "response_bytes": total,
        "response_sha256": sha256_bytes(joined),
        "from": from_date,
        "until": until_date,
        "metadata_prefix": metadata_prefix,
        "set_spec": set_spec,
        "resumption_complete": True,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "resumed_pages": resumed_pages,
    }


def download_file(
    url: str,
    target: Path,
    *,
    size_cap_bytes: int,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    if size_cap_bytes <= 0 or size_cap_bytes > MAX_DOWNLOAD_BYTES:
        raise ClassificationError("download size cap is unsafe")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if target.stat().st_size > size_cap_bytes:
            raise ClassificationError("existing download exceeds source size cap")
        return {
            "url": url,
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "etag": None,
            "last_modified": None,
            "content_type": None,
            "resumed_from_bytes": target.stat().st_size,
            "reused_complete_file": True,
        }
    partial = target.with_name(f".{target.name}.part")
    checkpoint_path = target.with_name(f".{target.name}.part.json")
    offset = partial.stat().st_size if partial.exists() else 0
    checkpoint = read_sealed_json(checkpoint_path) if checkpoint_path.exists() else {}
    if offset and checkpoint.get("url") != url:
        raise ClassificationError("partial download belongs to another URL")
    request_headers = {}
    if offset:
        request_headers["Range"] = f"bytes={offset}-"
        if checkpoint.get("etag"):
            request_headers["If-Range"] = str(checkpoint["etag"])
    with httpx.stream(
        "GET",
        url,
        headers=request_headers,
        timeout=timeout_seconds,
        follow_redirects=True,
    ) as response:
        response.raise_for_status()
        append = bool(offset and response.status_code == 206)
        if offset and response.status_code not in {200, 206}:
            raise ClassificationError("download server returned unsafe range response")
        if append and not str(response.headers.get("content-range") or "").startswith(
            f"bytes {offset}-"
        ):
            raise ClassificationError("download server returned mismatched range")
        total = offset if append else 0
        headers = {
            "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
            "content_type": response.headers.get("content-type"),
        }
        write_sealed_json(
            checkpoint_path,
            {
                "schema": "chronovisor.download-checkpoint.v1",
                "url": url,
                "etag": headers["etag"],
                "last_modified": headers["last_modified"],
                "partial_path": str(partial),
                "resumed_from_bytes": offset if append else 0,
                "complete": False,
            },
            backup=True,
        )
        with partial.open("ab" if append else "wb") as handle:
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > size_cap_bytes:
                    raise ClassificationError("download exceeded source size cap")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(partial, target)
    os.chmod(target, 0o600)
    write_sealed_json(
        checkpoint_path,
        {
            "schema": "chronovisor.download-checkpoint.v1",
            "url": url,
            **headers,
            "partial_path": str(partial),
            "resumed_from_bytes": offset if append else 0,
            "complete": True,
            "target_sha256": sha256_file(target),
            "bytes": total,
        },
        backup=True,
    )
    return {
        "url": url,
        "path": str(target),
        "bytes": total,
        "sha256": sha256_file(target),
        "resumed_from_bytes": offset if append else 0,
        "reused_complete_file": False,
        **headers,
    }


def normalize_ndl_oai_records(source: bytes | Path | Any) -> Iterator[dict[str, Any]]:
    """Normalize allowlisted NDL-created OAI records as diagnostic evidence."""

    stream: Any = (
        source
        if isinstance(source, Path)
        else (io.BytesIO(source) if isinstance(source, bytes) else source)
    )
    for _event, element in ET.iterparse(stream, events=("end",)):
        if element.tag != f"{{{OAI_NS}}}record":
            continue
        header = element.find(f"{{{OAI_NS}}}header")
        if header is None:
            element.clear()
            continue
        identifier = header.findtext(f"{{{OAI_NS}}}identifier") or ""
        set_specs = [
            (node.text or "").strip()
            for node in header.findall(f"{{{OAI_NS}}}setSpec")
            if (node.text or "").strip()
        ]
        provider_id = next(
            (value for value in set_specs if value.startswith("iss-ndl-opac")),
            "",
        )
        metadata = element.find(f"{{{OAI_NS}}}metadata")
        if not identifier or not provider_id or metadata is None:
            element.clear()
            continue
        titles: list[str] = []
        subjects: list[str] = []
        languages: list[str] = []
        diagnostics: list[dict[str, str]] = []
        for node in metadata.iter():
            local = node.tag.rsplit("}", 1)[-1]
            text = (node.text or "").strip()
            resource = node.attrib.get(f"{{{RDF_NS}}}resource") or ""
            value = text or resource
            if not value:
                continue
            if local in {"title", "alternative"}:
                titles.append(value)
            elif local == "subject":
                subjects.append(value)
            elif local == "language":
                languages.append(value)
            elif local in {"classification", "subjectScheme"}:
                diagnostics.append({"predicate": local, "target": value})
        output = {
            "schema": EXTERNAL_RECORD_SCHEMA,
            "source_record_id": identifier,
            "provider_id": provider_id,
            "record_creator": "National Diet Library",
            "provider_terms_url": "https://ndlsearch.ndl.go.jp/help/api/terms",
            "record_license_class": "NDL-created-bibliography-open-data",
            "rights_ref": "ndl-created-bibliography",
            "title": titles[0] if titles else "",
            "language": languages[0] if languages else "jpn",
            "subject_headings": [
                {
                    "uri": value if value.startswith("http") else None,
                    "pref_label": value,
                    "alt_labels": [],
                    "relation": "bibliographic-subject",
                }
                for value in subjects
            ],
            "source_assignments": [],
            "identifiers": {},
            "diagnostic_relations": diagnostics,
            "major_class": "diagnostic",
            "year_bucket": "oai-window",
        }
        output["record_sha256"] = sha256_bytes(_canonical_json(output))
        yield output
        element.clear()


def write_external_package(
    package_root: Path,
    *,
    contract: SourceContract,
    source_release: str,
    rows: Sequence[Mapping[str, Any]],
    acquisition: Mapping[str, Any],
    rejected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    contract.validate()
    records_path = package_root / "records.jsonl"
    records = b"".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True).encode("utf-8")
        + b"\n"
        for row in rows
    )
    _atomic_write(records_path, records)
    payload = {
        "schema": EXTERNAL_PACKAGE_SCHEMA,
        **asdict(contract),
        "source_release": source_release,
        "fetched_at": _now(),
        "package_sha256": sha256_file(records_path),
        "record_count": len(rows),
        "records_path": str(records_path),
        "rejected_counts_by_reason": dict(rejected_counts or {}),
        "acquisition": dict(acquisition),
        "repo_or_wheel_bundled": False,
    }
    write_sealed_json(package_root / "manifest.json", payload, backup=True)
    return payload


def rejection_counts(reasons: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(reasons).items()))
