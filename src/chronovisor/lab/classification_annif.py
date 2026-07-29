"""Annif-first pilot for UDC classification.

Annif is kept in an isolated Python 3.13 environment launched through ``uvx``.
No Annif dependency or downloaded library corpus is bundled in Chronovisor's
wheel.  The pilot deliberately starts with a ten-case kill gate before any
larger evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.classification import (
    ClassificationError,
    UDCPackage,
    load_udc_package,
)
from chronovisor.lab.classification_fixture_set import read_jsonl, sha256_file
from chronovisor.lab.classification_library_evidence import (
    COMPOSITE_UDC_RE,
    split_for_group,
)
from chronovisor.classification_library_sources import (
    MARC_NS,
    czech_bibliography_contract,
    fetch_oai_window,
    parse_marcxml_records,
    stable_sample,
    write_external_package,
)
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.store import CHRONOVISOR_ROOT

ANNIF_VERSION = "1.4.1"
ANNIF_DISTRIBUTION = f"annif[fasttext]=={ANNIF_VERSION}"
ANNIF_PYTHON = "3.13"
ANNIF_STATE_SCHEMA = "chronovisor.classification-annif-pilot-state.v1"
EARLY_REVIEW_SCHEMA = "chronovisor.classification-early-review.v1"
ANNIF_EVALUATION_SCHEMA = "chronovisor.classification-annif-early-evaluation.v1"
SOURCE_RELEASE = "2026-07-25"
FIXTURE_EPOCH = "epoch-3-library-evidence-v1"
EARLY_CASE_NUMBERS = (3, 4, 9, 11, 13, 18, 23, 25, 38, 50)
EARLY_CASE_EXPECTATIONS: Mapping[str, tuple[str, ...]] = {
    "Japan National Football Team: 2026 World Cup Matches and Personal Context": (
        "7",
        "79",
        "796",
        "796.3",
        "796.33",
    ),
    "Critique of the Horse-Operator Analogy in AI Labor Displacement": (
        "004.8",
        "331",
        "331.5",
    ),
    "Semantic Projection v75: Ingest-Drain Staging Tactic for Lock Contention": (
        "004.05",
        "004.4",
        "004.8",
        "005",
    ),
    "Interview Preparation: Handling 'Why-Why' Questions": (
        "331",
        "331.5",
        "37.04",
        "377",
    ),
    "Job Change Preparation Tactics: SPI, Interviews, and Agent Strategy": (
        "005.95/.96",
        "331",
        "331.5",
        "37.04",
        "377",
    ),
    "Chronovisor Long-Term Value: From Points to Connected Knowledge": (
        "005.94",
        "021",
    ),
    "Defense Industry Appeal: Stability and Target Clarity Arguments": (
        "331",
        "331.5",
        "355/359",
    ),
    "Interview Question Consolidation: June 2026 Preparation": (
        "331",
        "331.5",
        "37.04",
        "377",
    ),
    "Chronovisor Self-Heal Deadline Fix for Duplicate Lane Bottleneck": (
        "004.05",
        "004.4",
        "005.6",
    ),
    "職場ストレスと後輩指導の課題": (
        "005.95/.96",
        "331.4",
        "658.3",
    ),
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def pilot_root(root: Path) -> Path:
    return root / "classification" / "annif-pilot"


def library_fixture_path(root: Path) -> Path:
    return (
        root
        / "classification"
        / "fixtures"
        / "epochs"
        / FIXTURE_EPOCH
        / "adjudication.jsonl"
    )


def czech_source_root(root: Path) -> Path:
    return (
        root
        / "classification"
        / "library-evidence"
        / "sources"
        / "czech-national-bibliography"
        / SOURCE_RELEASE
    )


def _state(root: Path, *, status: str, stage: str, **detail: Any) -> dict[str, Any]:
    payload = {
        "schema": ANNIF_STATE_SCHEMA,
        "status": status,
        "stage": stage,
        "updated_at": _now(),
        "annif_version": ANNIF_VERSION,
        "annif_distribution": ANNIF_DISTRIBUTION,
        "annif_python": ANNIF_PYTHON,
        **detail,
    }
    write_sealed_json(pilot_root(root) / "state.json", payload, backup=True)
    return payload


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        os.chmod(path, 0o600)
    finally:
        temp.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    _atomic_write_text(path, content)


def _notation_matches(actual: str, expected: Sequence[str]) -> bool:
    actual = actual.strip()
    if not actual:
        return False
    return any(
        actual == candidate
        or actual.startswith(candidate + ".")
        or candidate.startswith(actual + ".")
        for candidate in expected
    )


def write_council_early_review(root: Path) -> dict[str, Any]:
    """Freeze a deterministic ten-case review of already generated labels."""

    source = library_fixture_path(root)
    rows = read_jsonl(source)
    if len(rows) < max(EARLY_CASE_NUMBERS):
        raise ClassificationError(
            f"early review requires {max(EARLY_CASE_NUMBERS)} completed rows"
        )
    reviewed = []
    for number in EARLY_CASE_NUMBERS:
        row = rows[number - 1]
        title = str(row.get("title") or "")
        expected = EARLY_CASE_EXPECTATIONS.get(title)
        if expected is None:
            raise ClassificationError(f"early review expectation missing for {title!r}")
        council = str(row.get("gold_primary_notation") or "")
        reviewed.append(
            {
                "case_number": number,
                "uid": str(row.get("uid") or ""),
                "title": title,
                "expected_primary_notations": list(expected),
                "council_primary_notation": council,
                "council_hit": _notation_matches(council, expected),
                "council_status": str(row.get("gold_expected_status") or ""),
                "council_quorum": int(row.get("gold_quorum") or 0),
            }
        )
    hit_count = sum(bool(row["council_hit"]) for row in reviewed)
    payload = {
        "schema": EARLY_REVIEW_SCHEMA,
        "created_at": _now(),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "source_completed_rows": len(rows),
        "case_numbers": list(EARLY_CASE_NUMBERS),
        "review_basis": (
            "fixed diverse cases; semantic accept sets chosen independently of "
            "the generated council label"
        ),
        "cases": reviewed,
        "council_hit_count": hit_count,
        "council_hit_rate": hit_count / len(reviewed),
        "decision": "reject-council" if hit_count < 6 else "continue-council",
        "additional_council_rows_authorized": 0,
    }
    output = pilot_root(root) / "early-council-review.json"
    write_sealed_json(output, payload, backup=True)

    legacy_state_path = (
        root / "classification" / "library-evidence" / "state.json"
    )
    if legacy_state_path.exists():
        legacy = read_sealed_json(legacy_state_path)
        legacy.update(
            {
                "status": "rejected",
                "stage": "e0_early_sample_rejected",
                "last_error": None,
                "decision_receipt": str(output),
                "decision": payload["decision"],
            }
        )
        write_sealed_json(legacy_state_path, legacy, backup=True)
    return payload


def acquire_czech_bibliography(
    root: Path,
    *,
    from_date: str = "2025-01-01",
    until_date: str = "2026-07-25",
    record_limit: int = 100_000,
) -> Path:
    """Acquire the labelled UDC bibliography without running council fixtures."""

    target = czech_source_root(root)
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        manifest = read_sealed_json(manifest_path)
        records_path = Path(str(manifest.get("records_path") or ""))
        if (
            records_path.is_file()
            and sha256_file(records_path) == manifest.get("package_sha256")
        ):
            return manifest_path
    _state(root, status="running", stage="download-czech-bibliography")
    raw, acquisition = fetch_oai_window(
        base_url="https://aleph.nkp.cz/OAI",
        metadata_prefix="marc21",
        from_date=from_date,
        until_date=until_date,
        set_spec="CNB",
        size_cap_bytes=1024**3,
        checkpoint_dir=target / "oai-checkpoint",
    )
    rows = stable_sample(
        parse_marcxml_records(raw, authority=False),
        limit=record_limit,
    )
    manifest = write_external_package(
        target,
        contract=czech_bibliography_contract(
            "https://aleph.nkp.cz/OAI?set=CNB&metadataPrefix=marc21"
        ),
        source_release=SOURCE_RELEASE,
        rows=rows,
        acquisition=acquisition,
    )
    if int(manifest.get("record_count") or 0) < 1:
        raise ClassificationError("Czech bibliography returned no records")
    return manifest_path


def finalize_czech_checkpoint(
    root: Path,
    *,
    minimum_records: int = 12_000,
    record_limit: int = 100_000,
) -> Path:
    """Freeze a sufficient OAI prefix instead of downloading the full window."""

    target = czech_source_root(root)
    checkpoint_path = target / "oai-checkpoint" / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise ClassificationError("Czech OAI checkpoint is missing")
    checkpoint = read_sealed_json(checkpoint_path)
    pages = []
    accumulated = 0
    for page in checkpoint.get("pages") or []:
        path = Path(str(page.get("path") or ""))
        if (
            not path.is_file()
            or sha256_file(path) != page.get("sha256")
            or path.stat().st_size != int(page.get("bytes") or -1)
        ):
            raise ClassificationError("Czech OAI checkpoint page is corrupt")
        pages.append(page)
        accumulated += int(page.get("record_count") or 0)
        if accumulated >= minimum_records:
            break
    if accumulated < minimum_records:
        raise ClassificationError(
            f"Czech checkpoint has {accumulated} records; "
            f"{minimum_records} are required"
        )
    raw = (
        f'<collection xmlns="{MARC_NS}">'.encode()
        + b"\n".join(Path(str(page["path"])).read_bytes() for page in pages)
        + b"</collection>"
    )
    rows = stable_sample(
        parse_marcxml_records(raw, authority=False),
        limit=record_limit,
    )
    manifest = write_external_package(
        target,
        contract=czech_bibliography_contract(
            "https://aleph.nkp.cz/OAI?set=CNB&metadataPrefix=marc21"
        ),
        source_release=SOURCE_RELEASE,
        rows=rows,
        acquisition={
            "request_count": len(pages),
            "record_count": accumulated,
            "response_bytes": sum(int(page.get("bytes") or 0) for page in pages),
            "from": "2025-01-01",
            "until": "2026-07-25",
            "metadata_prefix": "marc21",
            "set_spec": "CNB",
            "resumption_complete": False,
            "sampling_stop": "minimum-labelled-pilot-records",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
    )
    _state(
        root,
        status="acquired",
        stage="czech-bibliography-ready",
        source_records=manifest["record_count"],
        oai_full_window_complete=False,
        pilot_prefix_intentional=True,
    )
    return target / "manifest.json"


def export_vocabulary(package: UDCPackage, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("uri", "label_en", "label_ja", "notation"),
            )
            writer.writeheader()
            for uri, concept in sorted(package.concepts.items()):
                label_en = str(concept.get("label_en") or concept.get("label") or "")
                label_ja = str(concept.get("label_ja") or label_en)
                writer.writerow(
                    {
                        "uri": uri,
                        "label_en": label_en,
                        "label_ja": label_ja,
                        "notation": str(concept.get("notation") or ""),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, output_path)
        os.chmod(output_path, 0o600)
    finally:
        temp.unlink(missing_ok=True)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "concept_count": len(package.concepts),
        "license": package.license,
        "attribution": package.attribution,
    }


def _mapped_notations(value: str, package: UDCPackage) -> list[str]:
    """Map detailed/composite assignments to explicit UDC Summary ancestors."""

    notation = value.strip()
    if not notation:
        return []
    if not COMPOSITE_UDC_RE.search(notation) and package.by_notation(notation):
        return [notation]
    output = []
    for component in re.split(r"[:+]", notation):
        match = re.match(r"\s*(\d+(?:\.\d+)*)", component)
        if match is None:
            continue
        candidate = match.group(1)
        variants = [candidate]
        while "." in candidate:
            candidate = candidate.rsplit(".", 1)[0]
            variants.append(candidate)
        mapped = next(
            (variant for variant in variants if package.by_notation(variant)),
            None,
        )
        if mapped and mapped not in output:
            output.append(mapped)
    return output[:3]


def _record_text(row: Mapping[str, Any]) -> str:
    values = [str(row.get("title") or "")]
    for subject in row.get("subject_headings") or []:
        if not isinstance(subject, Mapping):
            continue
        values.append(str(subject.get("pref_label") or ""))
        values.extend(str(value) for value in subject.get("alt_labels") or [])
    return " ".join(value.strip() for value in values if value.strip())


def export_corpus(
    manifest_path: Path,
    package: UDCPackage,
    output_root: Path,
    *,
    maximum_documents: int = 50_000,
) -> dict[str, Any]:
    manifest = read_sealed_json(manifest_path)
    records_path = Path(str(manifest.get("records_path") or ""))
    if not records_path.is_file():
        raise ClassificationError("Czech source records are missing")
    if sha256_file(records_path) != manifest.get("package_sha256"):
        raise ClassificationError("Czech source package checksum mismatch")

    grouped: list[dict[str, Any]] = []
    rejected = Counter()
    for row in read_jsonl(records_path):
        record_id = str(row.get("source_record_id") or "")
        text = _record_text(row)
        if not record_id or not text:
            rejected["missing_record_id_or_text"] += 1
            continue
        subjects: dict[str, dict[str, str]] = {}
        for assignment in row.get("source_assignments") or []:
            if not isinstance(assignment, Mapping):
                continue
            if (
                assignment.get("role") != "bibliographic_assignment"
                or assignment.get("source_field") != "080"
            ):
                continue
            raw_notation = str(assignment.get("notation_or_uri") or "")
            notations = _mapped_notations(raw_notation, package)
            if not notations:
                rejected["unresolvable_or_composite_udc"] += 1
                continue
            if notations != [raw_notation.strip()]:
                rejected["broadened_to_udc_summary_ancestor"] += 1
            for notation in notations:
                concept = package.by_notation(notation)
                if concept is None:
                    continue
                uri = str(concept.get("uri") or "")
                subjects[uri] = {
                    "uri": uri,
                    "label": str(
                        concept.get("label_ja")
                        or concept.get("label_en")
                        or concept.get("label")
                        or notation
                    ),
                }
        if not subjects:
            rejected["no_safe_udc_assignment"] += 1
            continue
        grouped.append(
            {
                "document_id": record_id,
                "text": text,
                "metadata": {
                    "language": str(row.get("language") or "unknown"),
                    "title": str(row.get("title") or ""),
                },
                "subjects": [subjects[uri] for uri in sorted(subjects)],
                "_split": split_for_group(record_id),
            }
        )
    grouped.sort(
        key=lambda row: (
            str(row["_split"]),
            str(row["document_id"]),
        )
    )
    if maximum_documents > 0:
        train = [row for row in grouped if row["_split"] == "train"][
            :maximum_documents
        ]
        test = [row for row in grouped if row["_split"] == "test"][
            : max(1000, maximum_documents // 5)
        ]
    else:
        train = [row for row in grouped if row["_split"] == "train"]
        test = [row for row in grouped if row["_split"] == "test"]
    for row in (*train, *test):
        row.pop("_split", None)

    train_path = output_root / "corpus" / "train.jsonl"
    test_path = output_root / "corpus" / "test.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(test_path, test)
    return {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "train_documents": len(train),
        "test_path": str(test_path),
        "test_sha256": sha256_file(test_path),
        "test_documents": len(test),
        "rejected_counts": dict(sorted(rejected.items())),
    }


def write_projects(output_path: Path) -> dict[str, Any]:
    projects = """\
[udc-tfidf-ja]
name=UDC TF-IDF Japanese
language=ja
backend=tfidf
analyzer=simple(token_min_length=1)
limit=20
vocab=udc

[udc-fasttext-ja]
name=UDC fastText Japanese
language=ja
backend=fasttext
analyzer=simple(token_min_length=1)
dim=128
lr=0.25
epoch=15
loss=hs
minn=2
maxn=6
minCount=1
wordNgrams=2
thread=8
chunksize=24
limit=20
vocab=udc

[udc-ensemble-ja]
name=UDC TF-IDF plus fastText
language=ja
backend=ensemble
sources=udc-tfidf-ja,udc-fasttext-ja
limit=20
vocab=udc
"""
    _atomic_write_text(output_path, projects)
    return {"path": str(output_path), "sha256": sha256_file(output_path)}


def prepare_annif(
    root: Path,
    *,
    source_manifest: Path | None = None,
    maximum_documents: int = 50_000,
) -> dict[str, Any]:
    target = pilot_root(root)
    _state(root, status="running", stage="prepare-artifacts")
    package = load_udc_package(root)
    source_manifest = source_manifest or czech_source_root(root) / "manifest.json"
    if not source_manifest.is_file():
        raise ClassificationError("Annif source manifest is missing")
    vocabulary = export_vocabulary(package, target / "udc.csv")
    corpus = export_corpus(
        source_manifest,
        package,
        target,
        maximum_documents=maximum_documents,
    )
    projects = write_projects(target / "projects.cfg")
    payload = {
        "schema": "chronovisor.classification-annif-artifacts.v1",
        "created_at": _now(),
        "vocabulary": vocabulary,
        "corpus": corpus,
        "projects": projects,
        "external_data_bundled": False,
    }
    write_sealed_json(target / "artifacts.json", payload, backup=True)
    _state(
        root,
        status="prepared",
        stage="artifacts-ready",
        train_documents=corpus["train_documents"],
        test_documents=corpus["test_documents"],
    )
    return payload


def _annif_command(
    root: Path,
    arguments: Sequence[str],
    *,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    target = pilot_root(root)
    environment = os.environ.copy()
    environment["ANNIF_DATADIR"] = str(target / "data")
    command = [
        "uvx",
        "--python",
        ANNIF_PYTHON,
        "--from",
        ANNIF_DISTRIBUTION,
        "annif",
        *arguments,
        "-p",
        str(target / "projects.cfg"),
    ]
    result = subprocess.run(
        command,
        cwd=target,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    log = {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "finished_at": _now(),
    }
    log_name = arguments[0] + "-" + str(
        int(datetime.now(UTC).timestamp() * 1000)
    )
    write_sealed_json(target / "logs" / f"{log_name}.json", log, backup=False)
    if result.returncode != 0:
        raise ClassificationError(
            f"Annif {arguments[0]} failed: {result.stderr.strip()}"
        )
    return result


def train_annif(root: Path) -> dict[str, Any]:
    target = pilot_root(root)
    artifacts_path = target / "artifacts.json"
    if not artifacts_path.is_file():
        raise ClassificationError("Annif artifacts are not prepared")
    artifacts = read_sealed_json(artifacts_path)
    vocabulary = Path(str(artifacts["vocabulary"]["path"]))
    train_path = Path(str(artifacts["corpus"]["train_path"]))
    _state(root, status="running", stage="load-vocabulary")
    _annif_command(
        root,
        ["load-vocab", "udc", str(vocabulary), "--force"],
        timeout_seconds=300,
    )
    trained = []
    for project in ("udc-tfidf-ja", "udc-fasttext-ja"):
        _state(root, status="running", stage=f"train-{project}")
        _annif_command(
            root,
            ["train", project, str(train_path), "--jobs", "8"],
            timeout_seconds=14_400,
        )
        trained.append(project)
    payload = {
        "schema": "chronovisor.classification-annif-training.v1",
        "trained_at": _now(),
        "projects": trained,
        "ensemble": "udc-ensemble-ja",
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "annif_version": ANNIF_VERSION,
        "annif_distribution": ANNIF_DISTRIBUTION,
    }
    write_sealed_json(target / "training.json", payload, backup=True)
    _state(root, status="trained", stage="training-complete", projects=trained)
    return payload


def export_early_documents(root: Path) -> Path:
    review = write_council_early_review(root)
    fixture_rows = read_jsonl(library_fixture_path(root))
    rows = []
    for case in review["cases"]:
        source = fixture_rows[int(case["case_number"]) - 1]
        rows.append(
            {
                "document_id": str(source.get("uid") or ""),
                "text": str(source.get("excerpt") or ""),
                "metadata": {
                    "title": str(source.get("title") or ""),
                    "page_type": str(source.get("page_type") or ""),
                },
            }
        )
    output = pilot_root(root) / "early-cases.jsonl"
    _write_jsonl(output, rows)
    return output


def _evaluate_project(
    root: Path,
    project: str,
    documents_path: Path,
    review: Mapping[str, Any],
) -> dict[str, Any]:
    output_path = pilot_root(root) / f"early-{project}.jsonl"
    _annif_command(
        root,
        [
            "index",
            project,
            str(documents_path),
            "--output",
            str(output_path),
            "--force",
            "--limit",
            "10",
        ],
        timeout_seconds=1200,
    )
    predictions = {
        str(row.get("document_id") or ""): row for row in read_jsonl(output_path)
    }
    cases = []
    for case in review["cases"]:
        uid = str(case["uid"])
        prediction = predictions.get(uid, {})
        results = prediction.get("results") or []
        notations = [
            str(result.get("notation") or "")
            for result in results
            if isinstance(result, Mapping)
        ]
        expected = list(case["expected_primary_notations"])
        cases.append(
            {
                "uid": uid,
                "title": str(case["title"]),
                "expected_primary_notations": expected,
                "top_notations": notations,
                "top1_hit": bool(notations)
                and _notation_matches(notations[0], expected),
                "top5_hit": any(
                    _notation_matches(notation, expected)
                    for notation in notations[:5]
                ),
            }
        )
    top1 = sum(bool(case["top1_hit"]) for case in cases)
    top5 = sum(bool(case["top5_hit"]) for case in cases)
    return {
        "project": project,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "top1_hit_count": top1,
        "top5_hit_count": top5,
        "top1_hit_rate": top1 / len(cases),
        "top5_hit_rate": top5 / len(cases),
        "cases": cases,
    }


def evaluate_early(root: Path) -> dict[str, Any]:
    target = pilot_root(root)
    if not (target / "training.json").is_file():
        raise ClassificationError("Annif has not been trained")
    review = write_council_early_review(root)
    documents = export_early_documents(root)
    _state(root, status="running", stage="early-ten-case-evaluation")
    projects = [
        _evaluate_project(root, project, documents, review)
        for project in (
            "udc-tfidf-ja",
            "udc-fasttext-ja",
            "udc-ensemble-ja",
        )
    ]
    best = max(
        projects,
        key=lambda row: (
            int(row["top1_hit_count"]),
            int(row["top5_hit_count"]),
        ),
    )
    council_hits = int(review["council_hit_count"])
    viable = (
        int(best["top1_hit_count"]) > council_hits
        and int(best["top1_hit_count"]) >= 4
        and int(best["top5_hit_count"]) >= 6
    )
    payload = {
        "schema": ANNIF_EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "case_count": len(review["cases"]),
        "council_hit_count": council_hits,
        "council_hit_rate": review["council_hit_rate"],
        "projects": projects,
        "best_project": best["project"],
        "best_top1_hit_count": best["top1_hit_count"],
        "best_top5_hit_count": best["top5_hit_count"],
        "viability_gate": {
            "top1_strictly_better_than_council": int(best["top1_hit_count"])
            > council_hits,
            "top1_at_least_4_of_10": int(best["top1_hit_count"]) >= 4,
            "top5_at_least_6_of_10": int(best["top5_hit_count"]) >= 6,
        },
        "decision": "continue-annif" if viable else "reject-annif",
        "larger_evaluation_authorized": viable,
    }
    write_sealed_json(target / "early-evaluation.json", payload, backup=True)
    _state(
        root,
        status="passed" if viable else "rejected",
        stage="early-evaluation-complete",
        decision=payload["decision"],
        best_project=payload["best_project"],
        best_top1_hit_count=payload["best_top1_hit_count"],
        best_top5_hit_count=payload["best_top5_hit_count"],
    )
    return payload


def run_pilot(root: Path, *, maximum_documents: int = 50_000) -> dict[str, Any]:
    write_council_early_review(root)
    manifest = acquire_czech_bibliography(root)
    prepare_annif(
        root,
        source_manifest=manifest,
        maximum_documents=maximum_documents,
    )
    train_annif(root)
    return evaluate_early(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("review-council")
    acquire = sub.add_parser("acquire")
    acquire.add_argument("--from-date", default="2025-01-01")
    acquire.add_argument("--until-date", default="2026-07-25")
    acquire.add_argument("--record-limit", type=int, default=100_000)
    finalize = sub.add_parser("finalize-checkpoint")
    finalize.add_argument("--minimum-records", type=int, default=12_000)
    finalize.add_argument("--record-limit", type=int, default=100_000)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source-manifest", type=Path)
    prepare.add_argument("--maximum-documents", type=int, default=50_000)
    sub.add_parser("train")
    sub.add_parser("evaluate")
    run = sub.add_parser("run")
    run.add_argument("--maximum-documents", type=int, default=50_000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "review-council":
        result = write_council_early_review(args.root)
    elif args.command == "acquire":
        path = acquire_czech_bibliography(
            args.root,
            from_date=args.from_date,
            until_date=args.until_date,
            record_limit=args.record_limit,
        )
        result = {"manifest_path": str(path)}
    elif args.command == "finalize-checkpoint":
        path = finalize_czech_checkpoint(
            args.root,
            minimum_records=args.minimum_records,
            record_limit=args.record_limit,
        )
        result = {"manifest_path": str(path)}
    elif args.command == "prepare":
        result = prepare_annif(
            args.root,
            source_manifest=args.source_manifest,
            maximum_documents=args.maximum_documents,
        )
    elif args.command == "train":
        result = train_annif(args.root)
    elif args.command == "evaluate":
        result = evaluate_early(args.root)
    else:
        result = run_pilot(
            args.root,
            maximum_documents=args.maximum_documents,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
