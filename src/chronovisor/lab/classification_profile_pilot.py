"""Retrieval-only pilot for deterministic UDC hierarchy profiles.

This module deliberately stops before classification judgment.  It builds a
small persistent dense index from official multilingual UDC Summary captions
and their official ancestors, then compares candidate recall on the already
fixed ten-case diagnostic set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from chronovisor.classification.classification import (
    ClassificationError,
    UDCPackage,
    default_udc_package,
)
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.jsonl_write import write_jsonl_atomic as _write_jsonl
from chronovisor.core.runtime_config import load_embedding_config
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.core.timeutil import utc_iso_milliseconds as _now
from chronovisor.lab.classification_fixture_set import read_jsonl, sha256_file
from chronovisor.lab.classification_library_evidence import (
    DENSE_MODEL_LICENSE,
    DENSE_TRAINING_CORPUS_LICENSE,
    embed_texts_cancellable,
)

PROFILE_SCHEMA = "chronovisor.classification-profile-index.v1"
EVALUATION_SCHEMA = "chronovisor.classification-profile-evaluation.v1"
STATE_SCHEMA = "chronovisor.classification-profile-pilot-state.v1"
DEFAULT_LIMIT = 12
PASS_HITS = 8
FIXTURE_EPOCH = "epoch-3-library-evidence-v1"




def profile_pilot_root(root: Path) -> Path:
    return root / "classification" / "profile-retrieval-pilot"


def _write_state(
    root: Path,
    *,
    status: str,
    stage: str,
    **detail: Any,
) -> dict[str, Any]:
    payload = {
        "schema": STATE_SCHEMA,
        "status": status,
        "stage": stage,
        "updated_at": _now(),
        **detail,
    }
    return write_sealed_json(
        profile_pilot_root(root) / "state.json",
        payload,
        backup=True,
    )


def _fixture_path(root: Path) -> Path:
    return (
        root
        / "classification"
        / "fixtures"
        / "epochs"
        / FIXTURE_EPOCH
        / "adjudication.jsonl"
    )


def _review_path(root: Path) -> Path:
    return (
        root
        / "classification"
        / "annif-pilot"
        / "early-council-review.json"
    )




def _atomic_save_numpy(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        os.chmod(path, 0o600)
    finally:
        temp.unlink(missing_ok=True)




def _valid_primary_notation(notation: str, label: str) -> bool:
    return bool(
        notation
        and notation[0] in "012356789"
        and not any(char in notation for char in ('"', "'", "`", "(", ")", "="))
        and "special auxiliary" not in label.casefold()
    )


def _path_rows(
    package: UDCPackage,
    row: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    output = [row]
    seen = {str(row.get("uri") or "")}
    current = row
    while current.get("broader_uri"):
        uri = str(current["broader_uri"])
        if uri in seen:
            break
        seen.add(uri)
        parent = package.concepts.get(uri)
        if parent is None:
            break
        output.append(parent)
        current = parent
    return output


def build_profile_rows(package: UDCPackage) -> list[dict[str, Any]]:
    """Build official-caption-only profiles without mined corpus aliases."""

    candidates = []
    for row in package.concepts.values():
        notation = str(row.get("notation") or "")
        label_en = str(row.get("label_en") or row.get("label") or "")
        if _valid_primary_notation(notation, label_en):
            candidates.append(row)
    candidates.sort(key=lambda row: str(row.get("notation") or ""))

    profiles = []
    for row in candidates:
        path = _path_rows(package, row)
        lineage = []
        for value in reversed(path):
            lineage.append(
                {
                    "notation": str(value.get("notation") or ""),
                    "label_en": str(
                        value.get("label_en") or value.get("label") or ""
                    ),
                    "label_ja": str(value.get("label_ja") or ""),
                }
            )
        leaf = lineage[-1]
        broader = lineage[:-1]
        text_parts = [
            (
                f"UDC subject {leaf['notation']}: "
                f"{leaf['label_en']} / {leaf['label_ja']}"
            ).strip()
        ]
        if broader:
            text_parts.append(
                "Broader subjects: "
                + " > ".join(
                    (
                        f"{value['notation']} "
                        f"{value['label_en']} / {value['label_ja']}"
                    ).strip()
                    for value in broader
                )
            )
        profiles.append(
            {
                "notation": leaf["notation"],
                "concept_uri": str(row.get("uri") or ""),
                "label_en": leaf["label_en"],
                "label_ja": leaf["label_ja"],
                "lineage": lineage,
                "profile_text": "\n".join(text_parts),
                "profile_sources": [
                    "official_udc_caption_en",
                    "official_udc_caption_ja",
                    "official_udc_ancestry",
                ],
            }
        )
    return profiles


def _default_embed_many(texts: list[str]) -> list[list[float]]:
    _model, vectors = embed_texts_cancellable(texts, purpose="explicit")
    return vectors


def _normalized_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ClassificationError("profile embedding matrix is invalid")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix),
        where=norms != 0,
    )


def build_profile_index(
    root: Path,
    *,
    package: UDCPackage | None = None,
    embed_many: Callable[[list[str]], list[list[float]]] | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Build or reuse a sealed profile/vector pair."""

    package = package or default_udc_package()
    embed_many = embed_many or _default_embed_many
    output_root = profile_pilot_root(root)
    profiles_path = output_root / "profiles.jsonl"
    vectors_path = output_root / "vectors.npy"
    manifest_path = output_root / "manifest.json"
    profiles = build_profile_rows(package)
    _write_state(
        root,
        status="running",
        stage="preparing-official-profiles",
        profile_count=len(profiles),
        embedded_count=0,
        llm_calls=0,
    )
    _write_jsonl(profiles_path, profiles)
    profiles_sha256 = sha256_file(profiles_path)
    model = load_embedding_config().model
    if "bge-m3" not in model.casefold():
        raise ClassificationError(
            f"profile pilot requires bge-m3, configured embedding model is {model!r}"
        )

    if manifest_path.is_file():
        existing = read_sealed_json(manifest_path)
        if (
            existing.get("schema") == PROFILE_SCHEMA
            and existing.get("package_checksum") == package.checksum
            and existing.get("profiles_sha256") == profiles_sha256
            and existing.get("embedding_model") == model
            and int(existing.get("profile_count") or 0) == len(profiles)
            and vectors_path.is_file()
            and existing.get("vectors_sha256") == sha256_file(vectors_path)
        ):
            matrix = np.load(vectors_path, allow_pickle=False, mmap_mode="r")
            if matrix.shape[0] == len(profiles):
                _write_state(
                    root,
                    status="running",
                    stage="reusing-profile-index",
                    profile_count=len(profiles),
                    embedded_count=len(profiles),
                    llm_calls=0,
                )
                return existing

    embedded: list[list[float]] = []
    for offset in range(0, len(profiles), max(1, batch_size)):
        batch = profiles[offset : offset + max(1, batch_size)]
        embedded.extend(embed_many([str(row["profile_text"]) for row in batch]))
        _write_state(
            root,
            status="running",
            stage="embedding-official-profiles",
            profile_count=len(profiles),
            embedded_count=len(embedded),
            llm_calls=0,
        )
    if len(embedded) != len(profiles):
        raise ClassificationError("profile embedding backend returned wrong count")
    matrix = _normalized_matrix(embedded)
    _atomic_save_numpy(vectors_path, matrix)
    manifest = {
        "schema": PROFILE_SCHEMA,
        "created_at": _now(),
        "package_release": package.release,
        "package_checksum": package.checksum,
        "package_license": package.license,
        "package_attribution": package.attribution,
        "profile_policy": "official captions plus ancestors; no siblings or corpus aliases",
        "profile_count": len(profiles),
        "profiles_path": str(profiles_path),
        "profiles_sha256": profiles_sha256,
        "embedding_model": model,
        "embedding_model_license": DENSE_MODEL_LICENSE,
        "embedding_training_corpus_license": DENSE_TRAINING_CORPUS_LICENSE,
        "dimensions": int(matrix.shape[1]),
        "vectors_path": str(vectors_path),
        "vectors_sha256": sha256_file(vectors_path),
        "working_set_bytes": profiles_path.stat().st_size + vectors_path.stat().st_size,
        "external_library_records_used": 0,
        "local_page_label_associations_used": 0,
        "llm_calls": 0,
    }
    write_sealed_json(manifest_path, manifest, backup=True)
    return read_sealed_json(manifest_path)


def _page_query(page: Mapping[str, Any]) -> str:
    return "\n".join(
        value
        for value in (
            str(page.get("title") or ""),
            str(page.get("summary") or ""),
            str(page.get("excerpt") or "")[:2_400],
        )
        if value
    )


def query_profile_index(
    root: Path,
    page: Mapping[str, Any],
    *,
    limit: int = DEFAULT_LIMIT,
    embed_many: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[dict[str, Any]]:
    output_root = profile_pilot_root(root)
    manifest = read_sealed_json(output_root / "manifest.json")
    profiles_path = Path(str(manifest.get("profiles_path") or ""))
    vectors_path = Path(str(manifest.get("vectors_path") or ""))
    if (
        manifest.get("schema") != PROFILE_SCHEMA
        or not profiles_path.is_file()
        or not vectors_path.is_file()
        or sha256_file(profiles_path) != manifest.get("profiles_sha256")
        or sha256_file(vectors_path) != manifest.get("vectors_sha256")
    ):
        raise ClassificationError("profile index manifest is missing or corrupt")
    profiles = read_jsonl(profiles_path)
    matrix = np.load(vectors_path, allow_pickle=False, mmap_mode="r")
    if matrix.ndim != 2 or matrix.shape[0] != len(profiles):
        raise ClassificationError("profile vector matrix shape is invalid")
    embed_many = embed_many or _default_embed_many
    query = _normalized_matrix(embed_many([_page_query(page)]))[0]
    scores = np.asarray(matrix @ query, dtype=np.float32)
    order = sorted(
        range(len(profiles)),
        key=lambda index: (-float(scores[index]), str(profiles[index]["notation"])),
    )
    return [
        {
            "rank": rank,
            "notation": str(profiles[index]["notation"]),
            "label_en": str(profiles[index].get("label_en") or ""),
            "label_ja": str(profiles[index].get("label_ja") or ""),
            "concept_uri": str(profiles[index].get("concept_uri") or ""),
            "semantic_score": round(float(scores[index]), 6),
        }
        for rank, index in enumerate(order[: max(1, limit)], start=1)
    ]


def notation_matches(actual: str, expected: Sequence[str]) -> bool:
    """Use strict target/descendant matching; a broad parent is not enough."""

    actual = actual.strip()
    if not actual:
        return False
    for raw_candidate in expected:
        candidate = raw_candidate.strip()
        if not candidate:
            continue
        if actual == candidate or actual.startswith(candidate + "."):
            return True
        if "/." in candidate:
            start, suffix = candidate.split("/.", 1)
            base = start.rsplit(".", 1)[0] if "." in start else start
            end = f"{base}.{suffix}"
            if actual in {start, end}:
                return True
            if actual.startswith((start + ".", end + ".")):
                return True
        integer_range = re.fullmatch(r"(\d+)/(\d+)", candidate)
        actual_root = re.match(r"(\d+)", actual)
        if integer_range and actual_root:
            lower = int(integer_range.group(1))
            upper = int(integer_range.group(2))
            if lower <= int(actual_root.group(1)) <= upper:
                return True
    return False


def _first_matching_rank(
    rows: Sequence[Mapping[str, Any]],
    expected: Sequence[str],
) -> int | None:
    for rank, row in enumerate(rows, start=1):
        if notation_matches(str(row.get("notation") or ""), expected):
            return rank
    return None


def evaluate_fixed_cases(
    root: Path,
    *,
    limit: int = DEFAULT_LIMIT,
    embed_many: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict[str, Any]:
    review = read_sealed_json(_review_path(root))
    source_rows = read_jsonl(_fixture_path(root))
    source_by_uid = {str(row.get("uid") or ""): row for row in source_rows}
    cases = []
    for reviewed in review.get("cases") or []:
        uid = str(reviewed.get("uid") or "")
        source = source_by_uid.get(uid)
        if source is None:
            raise ClassificationError(f"profile pilot source row missing for {uid}")
        expected = [
            str(value)
            for value in reviewed.get("expected_primary_notations") or []
            if str(value)
        ]
        baseline = [dict(row) for row in source.get("candidates") or []][:limit]
        profile = query_profile_index(
            root,
            source,
            limit=limit,
            embed_many=embed_many,
        )
        baseline_rank = _first_matching_rank(baseline, expected)
        profile_rank = _first_matching_rank(profile, expected)
        cases.append(
            {
                "case_number": int(reviewed.get("case_number") or 0),
                "uid": uid,
                "title": str(source.get("title") or ""),
                "expected_primary_notations": expected,
                "baseline_hit": baseline_rank is not None,
                "baseline_rank": baseline_rank,
                "baseline_candidates": [
                    {
                        "notation": str(row.get("notation") or ""),
                        "label_en": str(row.get("label_en") or ""),
                        "retrieval_score": row.get("retrieval_score"),
                    }
                    for row in baseline
                ],
                "profile_hit": profile_rank is not None,
                "profile_rank": profile_rank,
                "profile_candidates": profile,
            }
        )
    baseline_hits = sum(bool(case["baseline_hit"]) for case in cases)
    profile_hits = sum(bool(case["profile_hit"]) for case in cases)
    baseline_mrr = sum(
        1 / int(case["baseline_rank"])
        for case in cases
        if case["baseline_rank"] is not None
    ) / max(1, len(cases))
    profile_mrr = sum(
        1 / int(case["profile_rank"])
        for case in cases
        if case["profile_rank"] is not None
    ) / max(1, len(cases))
    decision = (
        "qualify-profile-retrieval"
        if profile_hits >= PASS_HITS
        else "reject-profile-retrieval"
    )
    receipt = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "source_review_path": str(_review_path(root)),
        "source_review_sha256": sha256_file(_review_path(root)),
        "source_fixture_path": str(_fixture_path(root)),
        "source_fixture_sha256": sha256_file(_fixture_path(root)),
        "match_policy": "exact target, target descendant, or explicit UDC range",
        "candidate_limit": limit,
        "case_count": len(cases),
        "baseline_hit_count": baseline_hits,
        "baseline_recall_at_12": baseline_hits / max(1, len(cases)),
        "baseline_mrr": baseline_mrr,
        "profile_hit_count": profile_hits,
        "profile_recall_at_12": profile_hits / max(1, len(cases)),
        "profile_mrr": profile_mrr,
        "minimum_profile_hits": PASS_HITS,
        "decision": decision,
        "larger_evaluation_authorized": decision == "qualify-profile-retrieval",
        "llm_calls": 0,
        "cases": cases,
    }
    output_root = profile_pilot_root(root)
    write_sealed_json(output_root / "evaluation.json", receipt, backup=True)
    _write_state(
        root,
        status=(
            "qualified"
            if decision == "qualify-profile-retrieval"
            else "rejected"
        ),
        stage="fixed-ten-retrieval-complete",
        decision=decision,
        baseline_hit_count=baseline_hits,
        profile_hit_count=profile_hits,
        case_count=len(cases),
        larger_evaluation_authorized=receipt["larger_evaluation_authorized"],
        llm_calls=0,
    )
    return read_sealed_json(output_root / "evaluation.json")


def run_pilot(
    root: Path,
    *,
    batch_size: int = 64,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    try:
        build_profile_index(root, batch_size=batch_size)
        _write_state(
            root,
            status="running",
            stage="evaluating-fixed-ten-retrieval",
            llm_calls=0,
        )
        return evaluate_fixed_cases(root, limit=limit)
    except Exception as exc:
        _write_state(
            root,
            status="failed",
            stage="pilot-failed",
            error=f"{type(exc).__name__}: {exc}",
            llm_calls=0,
        )
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic UDC profiles and run the fixed ten-case gate"
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    arguments = parser.parse_args(argv)
    result = run_pilot(
        arguments.root,
        batch_size=max(1, arguments.batch_size),
        limit=max(1, arguments.limit),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
