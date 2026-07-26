"""Single resolver for production classification candidate behavior."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chronovisor.classification import ClassificationError, UDCPackage
from chronovisor.classification_bundle import (
    CANDIDATE_BUNDLE_SCHEMA,
    resolve_authority,
)
from chronovisor.classification_engine import CandidateIndex
from chronovisor.classification_fixture_set import inference_dto
from chronovisor.classification_library_evidence import (
    LibraryEvidenceIndex,
    LibraryEvidenceProvider,
)
from chronovisor.durable_state import read_sealed_json


class ResolvedCandidateIndex:
    def __init__(self, root: Path, package: UDCPackage) -> None:
        self.root = root
        self.package = package
        self.resolution = resolve_authority(root)
        self.official = CandidateIndex(package)
        self.provider: LibraryEvidenceProvider | None = None
        if self.resolution["status"] == "active":
            target = self.resolution["target"]
            candidate_path = Path(str(target.get("candidate_bundle_path") or ""))
            candidate = read_sealed_json(candidate_path)
            if candidate.get("schema") != CANDIDATE_BUNDLE_SCHEMA:
                raise ClassificationError("adopted candidate bundle is invalid")
            provider_manifest = Path(str(candidate.get("provider_manifest_path") or ""))
            self.provider = LibraryEvidenceProvider(
                package=package,
                evidence_index=LibraryEvidenceIndex(provider_manifest),
            )
            self.arms = tuple(
                candidate.get("run_config", {}).get("provider_arms") or ("B1b", "B3")
            )
        elif (
            self.resolution["status"] == "disabled"
            or self.resolution.get("reason") == "missing_active_pointer"
        ):
            self.arms = ()
        else:
            raise ClassificationError(
                f"classification authority resolution failed: {self.resolution}"
            )

    def candidates(
        self,
        page: Mapping[str, Any],
        *,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        if self.provider is None:
            return self.official.candidates(page, limit=limit)
        result = self.provider.candidates(
            inference_dto(page),
            arms=self.arms,
            limit=max(20, limit),
        )
        return list(result["union"])[:limit]


def production_candidate_index(
    root: Path,
    package: UDCPackage,
) -> ResolvedCandidateIndex:
    return ResolvedCandidateIndex(root, package)
