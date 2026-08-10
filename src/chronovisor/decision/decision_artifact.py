"""Content-addressed replay for local semantic decisions.

Sampler settings reduce variance but cannot make model inference a durable
deterministic function.  The runtime therefore defines reproducibility as:
one exact execution fingerprint publishes one sealed canonical decision, and
all subsequent executions replay that artifact without loading a model.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import (
    DEFAULT_MIN_FREE_BYTES,
    canonical_bytes,
    canonical_sha256,
    file_lock,
    seal_object,
    verify_sealed_object,
)
from chronovisor.core.sealed_artifact_decoder import schema_matches

EXECUTION_FINGERPRINT_VERSION = 2
DECISION_ARTIFACT_SCHEMA = "chronovisor.canonical-decision-artifact.v2"


class DecisionArtifactError(RuntimeError):
    """A canonical artifact is malformed, stale, or conflicts with its CAS."""


def execution_fingerprint(
    *,
    request_sha256: str,
    lane: str,
    context_tier: int,
    authority: Mapping[str, Any],
    router_policy: Mapping[str, Any],
    generation_policy_sha256: str,
    model_runtime: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    identity = {
        "fingerprint_version": EXECUTION_FINGERPRINT_VERSION,
        "request_sha256": request_sha256,
        "lane": lane,
        "context_tier": context_tier,
        "authority_sha256": canonical_sha256(authority),
        "router_policy_sha256": canonical_sha256(router_policy),
        "generation_policy_sha256": generation_policy_sha256,
        "model_runtime_sha256": canonical_sha256(model_runtime),
    }
    return canonical_sha256(identity), identity


class DecisionArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve(strict=False)
        self.lock_root = self.root / "locks"

    def path_for(self, fingerprint: str) -> Path:
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
        ):
            raise ValueError("execution fingerprint must be a lowercase sha256")
        return self.root / fingerprint[:2] / f"{fingerprint}.json"

    def _lock_path(self, fingerprint: str) -> Path:
        return self.lock_root / f"{fingerprint}.lock"

    def load(self, fingerprint: str) -> dict[str, Any] | None:
        path = self.path_for(fingerprint)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise DecisionArtifactError(f"cannot read decision artifact: {exc}") from exc
        try:
            payload = verify_sealed_object(json.loads(raw.decode("utf-8")))
        except Exception as exc:
            raise DecisionArtifactError("decision artifact seal is invalid") from exc
        if not schema_matches(payload.get("schema"), DECISION_ARTIFACT_SCHEMA):
            raise DecisionArtifactError("decision artifact schema is invalid")
        if payload.get("execution_fingerprint") != fingerprint:
            raise DecisionArtifactError("decision artifact path identity mismatch")
        identity = payload.get("execution_identity")
        if not isinstance(identity, dict) or canonical_sha256(identity) != fingerprint:
            raise DecisionArtifactError("decision artifact execution identity mismatch")
        if payload.get("decision_sha256") != canonical_sha256(payload.get("decision")):
            raise DecisionArtifactError("decision artifact payload digest mismatch")
        proof = payload.get("quorum_proof")
        if not isinstance(proof, list) or len(proof) < 2:
            raise DecisionArtifactError("decision artifact has no two-vote proof")
        agreement = payload.get("agreement_sha256")
        roles: set[str] = set()
        voters: set[str] = set()
        for row in proof:
            if not isinstance(row, dict):
                raise DecisionArtifactError("decision artifact quorum proof is malformed")
            role = row.get("role")
            model = row.get("model")
            provider = row.get("provider")
            route_provenance = row.get("route_provenance")
            returned_model = row.get("returned_model")
            signature = row.get("signature_sha256")
            if (
                set(row)
                != {
                    "role",
                    "provider",
                    "model",
                    "route_provenance",
                    "returned_model",
                    "signature_sha256",
                }
                or not isinstance(role, str)
                or not role
                or not isinstance(provider, str)
                or not provider
                or not isinstance(model, str)
                or not model
                or not isinstance(route_provenance, dict)
                or route_provenance.get("provider") != provider
                or route_provenance.get("model") != model
                or (
                    route_provenance.get("location") == "remote"
                    and returned_model != model
                )
                or signature != agreement
            ):
                raise DecisionArtifactError("decision artifact quorum proof is invalid")
            roles.add(role)
            voters.add(canonical_sha256(route_provenance))
        if len(roles) < 2 or len(voters) < 2:
            raise DecisionArtifactError(
                "decision artifact quorum proof lacks independent voters"
            )
        return payload

    def publish(
        self,
        *,
        fingerprint: str,
        identity: Mapping[str, Any],
        decision: Any,
        agreement_sha256: str,
        quorum_proof: list[dict[str, Any]],
        provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        if canonical_sha256(identity) != fingerprint:
            raise DecisionArtifactError("execution identity does not match fingerprint")
        payload = seal_object(
            {
                "schema": DECISION_ARTIFACT_SCHEMA,
                "execution_fingerprint": fingerprint,
                "execution_identity": dict(identity),
                "decision": decision,
                "decision_sha256": canonical_sha256(decision),
                "agreement_sha256": agreement_sha256,
                "quorum_proof": quorum_proof,
                "provenance": dict(provenance),
                "mutation_authority": "configured_route_quorum",
                "frontier_calls": 0,
            }
        )
        encoded = canonical_bytes(payload)
        path = self.path_for(fingerprint)
        with file_lock(self._lock_path(fingerprint), exclusive=True):
            existing = self.load(fingerprint)
            if existing is not None:
                if canonical_bytes(existing) != encoded:
                    raise DecisionArtifactError(
                        "conflicting canonical decision for one execution fingerprint"
                    )
                return existing
            path.parent.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(path.parent).free
            required = DEFAULT_MIN_FREE_BYTES + len(encoded) * 2
            if free < required:
                raise DecisionArtifactError(
                    f"insufficient free space for decision artifact ({free}<{required})"
                )
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                with contextlib.suppress(FileExistsError):
                    os.link(temporary, path)
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temporary is not None:
                    with contextlib.suppress(FileNotFoundError):
                        temporary.unlink()
            published = self.load(fingerprint)
            if published is None or canonical_bytes(published) != encoded:
                raise DecisionArtifactError("decision artifact read-back mismatch")
            return published


def default_store_root(chronovisor_root: Path) -> Path:
    return chronovisor_root / "runtime" / "decision-artifacts"
