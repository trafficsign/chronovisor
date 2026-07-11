"""Local-only semantic decision routing with a two-vote quorum."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from llm_wiki_mcp.decision_schema_manifest import (
    NON_DECISION_FIELDS,
    decision_signature_value,
    default_decision_value,
)
from llm_wiki_mcp.local_structured import (
    ChatTransport,
    LocalConsensusAuditStore,
    LocalStructuredResult,
    LocalStructuredSession,
    structured_request_sha256,
)
from llm_wiki_mcp.runtime_config import (
    DecisionRouterConfig,
    load_decision_router_config,
)

AgreementKey = Callable[[Any], Any]
ModelIdentityProvider = Callable[[Sequence[str]], Mapping[str, str]]
AUDIT_ROLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
ADOPTION_ARTIFACT_SCHEMA_VERSION = 2
MIN_ADOPTION_USABLE_CASES = 100
MIN_CASES_PER_PRODUCTION_SCHEMA = 5
REQUIRED_ADOPTION_CHECKS = frozenset(
    {
        "full_usable_corpus",
        "minimum_usable_cases",
        "role_coverage",
        "historical_decision_coverage",
        "production_schema_coverage",
        "minimum_cases_per_production_schema",
        "first_pass_schema_success",
        "final_schema_success",
        "pair_valid_vote",
        "pair_agreement",
        "three_model_majority_resolution",
        "historical_signature_match",
        "invalid_output_accepted",
        "unsafe_decision_flips",
    }
)
MINIMUM_QUALITY_THRESHOLDS = {
    "first_pass_schema_rate": 0.98,
    "final_schema_rate": 1.0,
    "pair_valid_rate": 0.99,
    "pair_agreement_rate": 0.75,
    "majority_resolution_rate": 0.99,
    "historical_signature_match_rate": 0.90,
}

def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def default_agreement_value(value: Any) -> Any:
    """Remove prose/confidence fields while preserving decision structure."""

    return default_decision_value(value)


def _has_decision_signal(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_decision_signal(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_decision_signal(item) for item in value)
    return value is not None


def canonical_agreement_signature(
    value: Any,
    agreement_key: AgreementKey | None = None,
    *,
    schema: Mapping[str, Any] | None = None,
) -> str:
    """Return a stable action signature or raise for a metadata-only result."""

    if agreement_key is not None:
        selected = agreement_key(value)
    elif schema is not None:
        selected = decision_signature_value(schema, value)
    else:
        selected = default_agreement_value(value)
    if not _has_decision_signal(selected):
        raise ValueError("agreement key produced no decision-bearing value")
    return _canonical_json(selected)


@dataclass(frozen=True)
class DecisionVote:
    role: str
    model: str
    result: LocalStructuredResult
    signature: str | None = None
    signature_sha256: str | None = None
    invalid_reason: str | None = None

    @property
    def valid(self) -> bool:
        return bool(self.result.ok and self.signature is not None and self.invalid_reason is None)

    def audit_record(self) -> dict[str, Any]:
        """Describe the vote without recording prompt, raw text, or payload."""

        return {
            "role": self.role,
            "model": self.model,
            "valid": self.valid,
            "signature_sha256": self.signature_sha256,
            "invalid_reason": self.invalid_reason,
            "session": self.result.audit_record(),
        }


@dataclass(frozen=True)
class DecisionRouterResult:
    status: str
    value: Any = None
    agreement_sha256: str | None = None
    votes: tuple[DecisionVote, ...] = ()
    failure_class: str | None = None
    quarantine_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "agreed"

    @property
    def decision(self) -> Any:
        return self.value

    def audit_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "agreement_sha256": self.agreement_sha256,
            "failure_class": self.failure_class,
            "quarantine_reason": self.quarantine_reason,
            "votes": [vote.audit_record() for vote in self.votes],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a caller-friendly result plus a redacted audit envelope."""

        return {
            "status": self.status,
            "ok": self.ok,
            "decision": self.value if self.ok else None,
            "failure_class": self.failure_class,
            "quarantine_reason": self.quarantine_reason,
            "agreement_sha256": self.agreement_sha256,
            "audit": self.audit_record(),
        }


def _config_error(config: DecisionRouterConfig) -> str | None:
    models = (
        config.primary_model.strip(),
        config.challenger_model.strip(),
        config.tie_break_model.strip(),
    )
    if not all(models):
        return "all three decision model tags are required"
    if len(set(models)) != len(models):
        return "primary, challenger, and tie-break models must be distinct"
    if config.quorum != 2:
        return "decision router quorum must be exactly 2"
    integer_minimums = {
        "num_ctx": 2_048,
        "num_predict": 128,
        "read_timeout_ms": 1_000,
        "max_input_chars": 4_096,
        "max_output_chars": 256,
        "max_feedback_chars": 512,
    }
    for name, minimum in integer_minimums.items():
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            return f"{name} must be an integer >= {minimum}"
    return None


@dataclass(frozen=True)
class RouterPolicyResolution:
    """The exact model policy selected for this router process."""

    config: DecisionRouterConfig
    source: str
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    error: str | None = None

    def audit_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "artifact_sha256": self.artifact_sha256,
            "error": self.error,
            "models": [
                self.config.primary_model,
                self.config.challenger_model,
                self.config.tie_break_model,
            ],
        }


def _candidate_config(value: Any) -> DecisionRouterConfig:
    if not isinstance(value, Mapping):
        raise ValueError("artifact config must be an object")
    field_names = {item.name for item in fields(DecisionRouterConfig)}
    required = field_names - {"adoption_artifact"}
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - field_names)
    if missing:
        raise ValueError(f"artifact config is missing fields: {','.join(missing)}")
    if unknown:
        raise ValueError(f"artifact config has unknown fields: {','.join(unknown)}")
    string_fields = {
        "primary_model",
        "challenger_model",
        "tie_break_model",
        "primary_keep_alive",
        "challenger_keep_alive",
        "tie_break_keep_alive",
    }
    integer_fields = required - string_fields
    kwargs: dict[str, Any] = {}
    for name in string_fields:
        item = value.get(name)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"artifact config field {name} must be non-empty")
        kwargs[name] = item.strip()
    for name in integer_fields:
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"artifact config field {name} must be an integer")
        kwargs[name] = item
    kwargs["adoption_artifact"] = ""
    candidate = DecisionRouterConfig(**kwargs)
    if error := _config_error(candidate):
        raise ValueError(error)
    return candidate


def _validated_adoption_artifact(
    path: Path,
) -> tuple[DecisionRouterConfig, str, dict[str, str]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read adoption artifact: {exc}") from exc
    try:
        artifact = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"adoption artifact is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(artifact, Mapping):
        raise ValueError("adoption artifact must be an object")
    if artifact.get("schema_version") != ADOPTION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("adoption artifact schema version is unsupported")
    if artifact.get("status") != "complete" or artifact.get("adopted") is not True:
        raise ValueError("adoption artifact is not complete and adopted")

    identity = artifact.get("identity")
    if not isinstance(identity, Mapping) or artifact.get("run_key") != _sha256_json(identity):
        raise ValueError("adoption artifact run identity is inconsistent")
    config_payload = artifact.get("config")
    config_sha256 = _sha256_json(config_payload)
    if (
        artifact.get("config_sha256") != config_sha256
        or identity.get("config_sha256") != config_sha256
    ):
        raise ValueError("adoption artifact config hash is inconsistent")
    thresholds = artifact.get("thresholds")
    if identity.get("thresholds_sha256") != _sha256_json(thresholds):
        raise ValueError("adoption artifact threshold hash is inconsistent")
    if not isinstance(thresholds, Mapping):
        raise ValueError("adoption artifact thresholds are missing")
    for name, minimum in MINIMUM_QUALITY_THRESHOLDS.items():
        observed = thresholds.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or observed < minimum
        ):
            raise ValueError(f"adoption threshold {name} is weaker than runtime policy")
    for name in ("max_invalid_output_accepted", "max_unsafe_decision_flips"):
        observed = thresholds.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed > 0
        ):
            raise ValueError(f"adoption threshold {name} is weaker than runtime policy")
    metadata = artifact.get("model_metadata")
    if (
        not isinstance(metadata, Mapping)
        or identity.get("model_metadata_sha256")
        != artifact.get("model_metadata_sha256")
        or artifact.get("model_metadata_sha256") != _sha256_json(metadata)
    ):
        raise ValueError("adoption artifact model identity is inconsistent")

    source = artifact.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("adoption artifact source is missing")
    from llm_wiki_mcp.decision_schema_manifest import (
        production_schema_manifest,
        production_signature_manifest,
    )

    current_schema_manifest_mapping = production_schema_manifest()
    current_schema_manifest = [
        {"name": name, "sha256": digest}
        for name, digest in sorted(current_schema_manifest_mapping.items())
    ]
    current_schema_manifest_sha256 = _sha256_json(current_schema_manifest)
    current_signature_manifest_sha256 = _sha256_json(
        production_signature_manifest()
    )
    if (
        identity.get("source_sha256") != source.get("source_sha256")
        or identity.get("selected_case_ids_sha256")
        != source.get("selected_case_ids_sha256")
        or identity.get("schema_manifest_sha256")
        != (
            source.get("coverage", {}).get("schema_manifest_sha256")
            if isinstance(source.get("coverage"), Mapping)
            else None
        )
        or identity.get("schema_manifest_sha256")
        != current_schema_manifest_sha256
        or identity.get("signature_manifest_sha256")
        != (
            source.get("coverage", {}).get("signature_manifest_sha256")
            if isinstance(source.get("coverage"), Mapping)
            else None
        )
        or identity.get("signature_manifest_sha256")
        != current_signature_manifest_sha256
    ):
        raise ValueError("adoption artifact source identity is inconsistent")
    usable = source.get("usable_cases")
    selected = source.get("selected_cases")
    processed = artifact.get("processed_cases")
    if (
        isinstance(usable, bool)
        or not isinstance(usable, int)
        or usable < MIN_ADOPTION_USABLE_CASES
        or source.get("full_usable_selection") is not True
        or selected != usable
        or artifact.get("selected_cases") != usable
        or processed != usable
    ):
        raise ValueError("adoption artifact does not cover the full usable corpus")
    coverage = source.get("coverage")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("role_coverage_rate") != 1.0
        or coverage.get("decision_coverage_rate") != 1.0
        or coverage.get("production_schema_coverage_rate") != 1.0
        or not isinstance(coverage.get("minimum_production_schema_cases"), int)
        or coverage.get("minimum_production_schema_cases")
        < MIN_CASES_PER_PRODUCTION_SCHEMA
    ):
        raise ValueError("adoption artifact is not representative of usable evidence")
    required_schemas = coverage.get("required_schemas")
    if not isinstance(required_schemas, list) or not required_schemas:
        raise ValueError("adoption artifact schema evidence is missing")
    observed_schema_manifest: dict[str, str] = {}
    for row in required_schemas:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("names"), list)
            or not row.get("names")
            or not all(isinstance(name, str) and name for name in row["names"])
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or isinstance(row.get("selected_cases"), bool)
            or not isinstance(row.get("selected_cases"), int)
            or row["selected_cases"] < MIN_CASES_PER_PRODUCTION_SCHEMA
        ):
            raise ValueError("adoption artifact has incomplete production schema evidence")
        for name in row["names"]:
            if name in observed_schema_manifest:
                raise ValueError("adoption artifact repeats a production schema name")
            observed_schema_manifest[name] = str(row["sha256"])
    if observed_schema_manifest != current_schema_manifest_mapping:
        raise ValueError("adoption artifact schema evidence does not match runtime")

    gate = artifact.get("adoption_gate")
    checks = gate.get("checks") if isinstance(gate, Mapping) else None
    if (
        not isinstance(checks, Mapping)
        or not checks
        or not REQUIRED_ADOPTION_CHECKS.issubset(checks)
        or gate.get("passed") is not True
        or any(
            not isinstance(check, Mapping) or check.get("passed") is not True
            for check in checks.values()
        )
    ):
        raise ValueError("adoption artifact gate did not fully pass")
    metrics = artifact.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("adoption artifact metrics are missing")
    for name, minimum in MINIMUM_QUALITY_THRESHOLDS.items():
        observed = metrics.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or observed < minimum
        ):
            raise ValueError(f"adoption metric {name} is below runtime policy")
    for name in ("invalid_output_accepted", "unsafe_decision_flips"):
        observed = metrics.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed > 0
        ):
            raise ValueError(f"adoption metric {name} exceeds runtime policy")

    candidate = _candidate_config(config_payload)
    model_records = metadata.get("models")
    if not isinstance(model_records, Mapping):
        raise ValueError("adoption artifact model metadata is missing")
    expected_digests: dict[str, str] = {}
    for model in (
        candidate.primary_model,
        candidate.challenger_model,
        candidate.tie_break_model,
    ):
        record = model_records.get(model)
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("digest"), str)
            or not record.get("digest")
            or record.get("status") == "missing"
        ):
            raise ValueError(f"model {model!r} has no evaluated identity")
        expected_digests[model] = str(record["digest"])
    return candidate, hashlib.sha256(raw).hexdigest(), expected_digests


def resolve_router_policy(
    config: DecisionRouterConfig,
    *,
    model_identity_provider: ModelIdentityProvider | None = None,
) -> RouterPolicyResolution:
    """Use an adopted candidate only after validating its complete artifact.

    The configured triplet is deliberately retained as the bootstrap/current
    policy.  A missing or corrupt nominated artifact therefore cannot stop the
    running system and cannot partially switch individual model roles.
    """

    nominated = config.adoption_artifact.strip()
    if not nominated:
        return RouterPolicyResolution(
            config=config,
            source="bootstrap_current_policy",
        )
    path = Path(nominated).expanduser()
    try:
        candidate, artifact_sha256, expected_digests = (
            _validated_adoption_artifact(path)
        )
        if model_identity_provider is None:
            from llm_wiki_mcp.ollama import model_digests

            model_identity_provider = model_digests
        observed_digests = model_identity_provider(tuple(expected_digests))
        if not isinstance(observed_digests, Mapping) or any(
            observed_digests.get(model) != digest
            for model, digest in expected_digests.items()
        ):
            raise ValueError(
                "installed model digests differ from the evaluated candidate"
            )
    except Exception as exc:
        return RouterPolicyResolution(
            config=config,
            source="bootstrap_current_policy",
            artifact_path=str(path),
            error=f"adoption_artifact_invalid:{exc}",
        )
    return RouterPolicyResolution(
        config=candidate,
        source="adopted_artifact",
        artifact_path=str(path),
        artifact_sha256=artifact_sha256,
    )


class DecisionRouter:
    """Reach a local two-model quorum without any frontier fallback."""

    def __init__(
        self,
        *,
        config: DecisionRouterConfig | None = None,
        transport: ChatTransport | None = None,
        agreement_key: AgreementKey | None = None,
        audit_root: Path | None = None,
        audit_role: str = "routine",
        resolve_adoption: bool = True,
        model_identity_provider: ModelIdentityProvider | None = None,
        record_replay: bool = True,
        replay_path: Path | None = None,
    ) -> None:
        if not isinstance(audit_role, str) or not AUDIT_ROLE_RE.fullmatch(audit_role):
            raise ValueError("audit_role must be a safe identifier of at most 80 chars")
        baseline_config = config or load_decision_router_config()
        self.policy = (
            resolve_router_policy(
                baseline_config,
                model_identity_provider=model_identity_provider,
            )
            if resolve_adoption
            else RouterPolicyResolution(
                config=baseline_config,
                source="evaluation_candidate",
            )
        )
        self.config = self.policy.config
        self.transport = transport
        self.agreement_key = agreement_key
        self.audit_root = audit_root
        self.audit_role = audit_role
        self.record_replay = record_replay and audit_role != "model_eval"
        self.replay_path = replay_path
        self.audit_store = LocalConsensusAuditStore(audit_root)
        self.config_error = _config_error(self.config)

    def _session(
        self,
        model: str,
        keep_alive: str,
        role: str,
    ) -> LocalStructuredSession:
        return LocalStructuredSession(
            model=model,
            transport=self.transport,
            role=f"{self.audit_role}:{role}",
            audit_root=self.audit_root,
            num_ctx=self.config.num_ctx,
            num_predict=self.config.num_predict,
            keep_alive=keep_alive,
            read_timeout_ms=self.config.read_timeout_ms,
            max_input_chars=self.config.max_input_chars,
            max_output_chars=self.config.max_output_chars,
            max_feedback_chars=self.config.max_feedback_chars,
        )

    def _vote(
        self,
        *,
        role: str,
        model: str,
        keep_alive: str,
        prompt: str,
        schema: Mapping[str, Any],
        system: str | None,
        agreement_key: AgreementKey | None,
    ) -> DecisionVote:
        result = self._session(model, keep_alive, role).run(prompt, schema, system=system)
        if not result.ok:
            return DecisionVote(
                role=role,
                model=model,
                result=result,
                invalid_reason=result.failure_class or "structured_session_failed",
            )
        try:
            signature = canonical_agreement_signature(
                result.value,
                agreement_key,
                schema=schema,
            )
        except Exception as exc:
            return DecisionVote(
                role=role,
                model=model,
                result=result,
                invalid_reason=f"agreement_key_error:{type(exc).__name__}",
            )
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        return DecisionVote(
            role=role,
            model=model,
            result=result,
            signature=signature,
            signature_sha256=digest,
        )

    @staticmethod
    def _winner(votes: Sequence[DecisionVote]) -> str | None:
        counts = Counter(vote.signature for vote in votes if vote.valid)
        for signature, count in counts.items():
            if signature is not None and count >= 2:
                return signature
        return None

    @staticmethod
    def _agreed(votes: Sequence[DecisionVote], signature: str) -> DecisionRouterResult:
        selected = next(vote for vote in votes if vote.valid and vote.signature == signature)
        return DecisionRouterResult(
            status="agreed",
            value=selected.result.value,
            agreement_sha256=hashlib.sha256(signature.encode("utf-8")).hexdigest(),
            votes=tuple(votes),
        )

    @staticmethod
    def _quarantined(votes: Sequence[DecisionVote], reason: str) -> DecisionRouterResult:
        return DecisionRouterResult(
            status="quarantined",
            votes=tuple(votes),
            failure_class="local_consensus_failed",
            quarantine_reason=reason,
        )

    def decide(
        self,
        prompt: str,
        schema: Mapping[str, Any],
        *,
        system: str | None = None,
        agreement_key: AgreementKey | None = None,
    ) -> DecisionRouterResult:
        started = time.monotonic()
        request_sha256 = structured_request_sha256(prompt, schema, system)

        def finalize(result: DecisionRouterResult) -> DecisionRouterResult:
            try:
                self.audit_store.append(
                    {
                        "kind": "decision",
                        "request_sha256": request_sha256,
                        "role": self.audit_role,
                        "status": result.status,
                        "failure_class": result.failure_class,
                        "quarantine_reason": result.quarantine_reason,
                        "pair_agreement": bool(result.ok and len(result.votes) == 2),
                        "tie_break_used": len(result.votes) == 3,
                        "unresolved_quarantine": result.status == "quarantined",
                        "vote_count": len(result.votes),
                        "valid_votes": sum(vote.valid for vote in result.votes),
                        "first_pass_valid_votes": sum(
                            vote.result.first_pass_valid for vote in result.votes
                        ),
                        "repaired_votes": sum(
                            bool(vote.result.ok and vote.result.repair_turns > 0)
                            for vote in result.votes
                        ),
                        "repair_turns": sum(
                            vote.result.repair_turns for vote in result.votes
                        ),
                        "models": [vote.model for vote in result.votes],
                        "policy": self.policy.audit_record(),
                    }
                )
            except Exception:
                # Durable audit failures must not alter a local decision.
                pass
            if self.record_replay and result.ok and isinstance(result.value, Mapping):
                try:
                    from llm_wiki_mcp import wiki
                    from llm_wiki_mcp.model_lab import record_local_replay_case

                    replay_path = self.replay_path
                    if replay_path is None:
                        if self.audit_root is not None:
                            replay_path = (
                                Path(self.audit_root).parent
                                / "model-lab"
                                / "replay.jsonl"
                            )
                        else:
                            replay_path = (
                                wiki.WIKI_ROOT
                                / "runtime"
                                / "model-lab"
                                / "replay.jsonl"
                            )
                    record_local_replay_case(
                        role=self.audit_role,
                        prompt=prompt,
                        schema=schema,
                        result=result.value,
                        models=[vote.model for vote in result.votes],
                        latency_seconds=time.monotonic() - started,
                        system=system,
                        replay_file=replay_path,
                    )
                except Exception:
                    # Replay evidence is observational and must never alter the
                    # already-reached local decision.
                    pass
            return result

        if self.config_error:
            return finalize(
                self._quarantined((), f"router_config_invalid:{self.config_error}")
            )

        key = agreement_key if agreement_key is not None else self.agreement_key
        votes: list[DecisionVote] = []
        votes.append(
            self._vote(
                role="primary",
                model=self.config.primary_model,
                keep_alive=self.config.primary_keep_alive,
                prompt=prompt,
                schema=schema,
                system=system,
                agreement_key=key,
            )
        )
        votes.append(
            self._vote(
                role="challenger",
                model=self.config.challenger_model,
                keep_alive=self.config.challenger_keep_alive,
                prompt=prompt,
                schema=schema,
                system=system,
                agreement_key=key,
            )
        )

        winner = self._winner(votes)
        if winner is not None:
            return finalize(self._agreed(votes, winner))
        if not any(vote.valid for vote in votes):
            return finalize(
                self._quarantined(votes, "primary_and_challenger_invalid")
            )

        votes.append(
            self._vote(
                role="tie_break",
                model=self.config.tie_break_model,
                keep_alive=self.config.tie_break_keep_alive,
                prompt=prompt,
                schema=schema,
                system=system,
                agreement_key=key,
            )
        )
        winner = self._winner(votes)
        if winner is not None:
            return finalize(self._agreed(votes, winner))

        valid_count = sum(vote.valid for vote in votes)
        if valid_count < self.config.quorum:
            return finalize(
                self._quarantined(votes, "fewer_than_two_valid_local_votes")
            )
        return finalize(
            self._quarantined(votes, "local_models_did_not_reach_two_vote_quorum")
        )


__all__ = [
    "AgreementKey",
    "DecisionRouter",
    "DecisionRouterResult",
    "DecisionVote",
    "NON_DECISION_FIELDS",
    "ModelIdentityProvider",
    "RouterPolicyResolution",
    "canonical_agreement_signature",
    "default_agreement_value",
    "resolve_router_policy",
]
