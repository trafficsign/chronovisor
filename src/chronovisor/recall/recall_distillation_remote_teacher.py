"""Temporary, free-only OpenCode Go teacher adapter for Recall distillation.

The adapter deliberately has one narrow seam: ``Teacher.evaluate`` receives a
compact, already-selected batch and returns labels or a redacted failure
classification.  Provider authentication and HTTP remain in the shared
OpenAI-compatible adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from chronovisor.core.llm_runtime import (
    GenerationRequest,
    RouteLocation,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.openai_compatible_adapter import OpenAICompatibleAdapter
from chronovisor.core.provider_profiles import (
    ProviderAdapterError,
    ProviderFailureCategory,
)

OX_ALPHA_PROVIDER = "opencode-go"
OX_ALPHA_ROUTE_MODEL = "opencode-go/ox-alpha-free"
OX_ALPHA_REQUEST_MODEL = "ox-alpha-free"
OX_ALPHA_ENDPOINT = "https://opencode.ai/zen/go/v1"
TEACHER_BATCH_SCHEMA = "chronovisor.recall-distill-teacher-batch.v1"
MAX_TEACHER_CANDIDATES = 16
MAX_PAYLOAD_BYTES = 12_000
MAX_REQUEST_BYTES = 18_000
MAX_TEXT_CHARS = 8_000
MAX_TIMEOUT_MS = 660_000

_CANDIDATE_KEYS = frozenset(
    {"candidate_id", "rally_id", "query", "context", "evidence"}
)
_FAILURE_TRANSIENT = frozenset(
    {
        ProviderFailureCategory.RATE_LIMITED.value,
        ProviderFailureCategory.SERVER_ERROR.value,
        ProviderFailureCategory.TIMEOUT.value,
        ProviderFailureCategory.TRANSPORT_ERROR.value,
    }
)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}\Z")
_SECRET_TEXT = re.compile(
    r"(?ix)"
    r"(?:api[_ -]?key|access[_ -]?token|authorization|bearer|password|"
    r"secret|credential|token|private[_ -]?key|client[_ -]?secret)\s*[:=]"
    r"|(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_-]{12,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{12,})"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)
_PII_TEXT = re.compile(
    r"(?ix)"
    r"(?:\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b)"
    r"|(?:\+?\d[\d .()\-]{7,}\d)"
    r"|(?:\b\d{3}-\d{2}-\d{4}\b)"
    r"|(?:\b\d{3}-\d{4}\b)"
    r"|(?:(?:full\s+name|real\s+name|氏名|個人情報)\s*[:：=])"
)
_PRIVATE_WORK_TEXT = re.compile(
    r"(?i)(?:\binternal\b|\bconfidential\b|\bnon[ -]?public\b|"
    r"\bprivate\b|\bcustomer\b|\bclient\b|\bemployer\b|"
    r"\bcompany\b|社内|非公開|顧客|取引先|機密|業務情報)"
)
_LOCAL_PATH_TEXT = re.compile(
    r"(?ix)"
    r"(?:^|[\s(=:\"'])"
    r"(?:~[/\\]|/(?:users|home|private|tmp|var|etc|opt|volumes|system)/"
    r"|file:(?:/{2,3})(?:users|home|private|tmp|var|etc|opt|volumes|system)/"
    r"|[a-z]:[/\\]|\\\\[^/\\\s]+[/\\])"
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def resolve_ox_alpha_model(provider: str, configured_model: str) -> str:
    """Resolve the request model from the existing provider/model route.

    Configurations may identify a model as either ``ox-alpha-free`` (the
    normal route model) or ``opencode-go/ox-alpha-free`` (the catalog identity).
    No alternative or paid model is accepted.
    """

    if provider != OX_ALPHA_PROVIDER or not isinstance(configured_model, str):
        raise ValueError("invalid OX Alpha route")
    if configured_model == OX_ALPHA_ROUTE_MODEL:
        return OX_ALPHA_REQUEST_MODEL
    if configured_model == OX_ALPHA_REQUEST_MODEL:
        return configured_model
    prefix = f"{OX_ALPHA_PROVIDER}/"
    if configured_model.startswith(prefix):
        request_model = configured_model[len(prefix) :]
        if request_model == OX_ALPHA_REQUEST_MODEL:
            return request_model
    raise ValueError("invalid OX Alpha model")


def _teacher_schema(candidate_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["labels"],
        "properties": {
            "labels": {
                "type": "array",
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "candidate_id",
                        "verdict",
                        "confidence",
                        "rationale",
                        "minimal_atom_ids",
                        "missing_slots",
                        "changing_claim",
                    ],
                    "properties": {
                        "candidate_id": {
                            "type": "string",
                            "enum": list(candidate_ids),
                        },
                        "verdict": {
                            "type": "string",
                            "enum": ["relevant", "irrelevant", "uncertain"],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "rationale": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 600,
                        },
                        "minimal_atom_ids": {
                            "type": "array",
                            "maxItems": 8,
                            "uniqueItems": True,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 160,
                            },
                        },
                        "missing_slots": {
                            "type": "array",
                            "maxItems": 5,
                            "uniqueItems": True,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 160,
                            },
                        },
                        "changing_claim": {
                            "type": "string",
                            "maxLength": 600,
                        },
                    },
                },
            }
        },
    }


def _contains_forbidden_text(value: str) -> bool:
    return bool(
        _SECRET_TEXT.search(value)
        or _PII_TEXT.search(value)
        or _PRIVATE_WORK_TEXT.search(value)
        or _LOCAL_PATH_TEXT.search(value)
    )


def _valid_text(value: object, *, required: bool = True) -> bool:
    return (
        isinstance(value, str)
        and (bool(value) if required else True)
        and len(value) <= MAX_TEXT_CHARS
        and not _contains_forbidden_text(value)
    )


def _validate_payload(
    payload: Mapping[str, Any], *, max_input_bytes: int
) -> tuple[dict[str, Any], tuple[str, ...]] | None:
    if set(payload) != {"schema", "candidates"}:
        return None
    if payload.get("schema") != TEACHER_BATCH_SCHEMA:
        return None
    candidates = payload.get("candidates")
    if (
        not isinstance(candidates, list)
        or not 1 <= len(candidates) <= MAX_TEACHER_CANDIDATES
    ):
        return None
    normalized: list[dict[str, Any]] = []
    candidate_ids: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or set(candidate) != _CANDIDATE_KEYS:
            return None
        candidate_id = candidate.get("candidate_id")
        rally_id = candidate.get("rally_id")
        context = candidate.get("context")
        if (
            not isinstance(candidate_id, str)
            or _SAFE_ID.fullmatch(candidate_id) is None
            or _contains_forbidden_text(candidate_id)
            or not isinstance(rally_id, str)
            or _SAFE_ID.fullmatch(rally_id) is None
            or _contains_forbidden_text(rally_id)
            or not _valid_text(candidate.get("query"))
            or not _valid_text(candidate.get("evidence"))
            or not isinstance(context, list)
            or len(context) > MAX_TEACHER_CANDIDATES
            or any(not _valid_text(item) for item in context)
        ):
            return None
        if candidate_id in candidate_ids:
            return None
        candidate_ids.append(candidate_id)
        normalized.append(
            {
                "candidate_id": candidate_id,
                "rally_id": rally_id,
                "query": candidate["query"],
                "context": list(context),
                "evidence": candidate["evidence"],
            }
        )
    result = {"schema": TEACHER_BATCH_SCHEMA, "candidates": normalized}
    try:
        if len(_json_bytes(result)) > max_input_bytes:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return result, tuple(candidate_ids)


def _safe_label(label: object, candidate_ids: frozenset[str]) -> dict[str, Any] | None:
    if not isinstance(label, Mapping):
        return None
    allowed = {
        "candidate_id",
        "verdict",
        "confidence",
        "rationale",
        "minimal_atom_ids",
        "missing_slots",
        "changing_claim",
    }
    if set(label) != allowed:
        return None
    candidate_id = label.get("candidate_id")
    verdict = label.get("verdict")
    confidence = label.get("confidence")
    rationale = label.get("rationale")
    atoms = label.get("minimal_atom_ids")
    missing = label.get("missing_slots")
    changing = label.get("changing_claim")
    if (
        not isinstance(candidate_id, str)
        or candidate_id not in candidate_ids
        or verdict not in {"relevant", "irrelevant", "uncertain"}
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
        or not _valid_text(rationale)
        or not isinstance(atoms, list)
        or len(atoms) > 8
        or any(not _valid_text(item) for item in atoms)
        or len(set(atoms)) != len(atoms)
        or not isinstance(missing, list)
        or len(missing) > 5
        or any(not _valid_text(item) for item in missing)
        or len(set(missing)) != len(missing)
        or not _valid_text(changing, required=False)
    ):
        return None
    return {
        "candidate_id": candidate_id,
        "verdict": verdict,
        "confidence": confidence,
        "rationale": rationale,
        "minimal_atom_ids": list(atoms),
        "missing_slots": list(missing),
        "changing_claim": changing,
    }


class OpenCodeOxAlphaTeacher:
    """A free-only remote adapter satisfying the Recall ``Teacher`` seam."""

    local = False
    location = RouteLocation.REMOTE

    def __init__(
        self,
        backend: OpenAICompatibleAdapter,
        *,
        configured_model: str = OX_ALPHA_ROUTE_MODEL,
        enabled: bool = True,
        free_only: bool = True,
        allow_paid_fallback: bool = False,
        max_input_bytes: int = MAX_PAYLOAD_BYTES,
        timeout_ms: int = 60_000,
    ) -> None:
        if not isinstance(backend, OpenAICompatibleAdapter):
            raise TypeError("OpenCode Ox Alpha requires OpenAICompatibleAdapter")
        profile = getattr(backend, "_profile", None)
        endpoint = getattr(profile, "endpoint", None)
        if (
            backend.provider != OX_ALPHA_PROVIDER
            or backend.location is not RouteLocation.REMOTE
            or endpoint != OX_ALPHA_ENDPOINT
        ):
            raise ValueError("invalid OX Alpha route")
        if free_only is not True or allow_paid_fallback is not False:
            raise ValueError("paid fallback is forbidden for the temporary route")
        if (
            isinstance(max_input_bytes, bool)
            or not isinstance(max_input_bytes, int)
            or not 1 <= max_input_bytes <= MAX_PAYLOAD_BYTES
        ):
            raise ValueError("invalid teacher input budget")
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or not 1 <= timeout_ms <= MAX_TIMEOUT_MS
        ):
            raise ValueError("invalid teacher timeout")
        self._backend = backend
        self._request_model = resolve_ox_alpha_model(backend.provider, configured_model)
        if not backend.capabilities_for(self._request_model).structured_output:
            raise ValueError("O× Alpha route lacks structured output")
        self.role = "recall.distill.teacher.ox-alpha"
        self.provider = OX_ALPHA_PROVIDER
        self.model = OX_ALPHA_ROUTE_MODEL
        self.enabled = bool(enabled)
        self.max_input_bytes = max_input_bytes
        self.timeout_ms = timeout_ms
        self._route_identity = {
            "provider": self.provider,
            "model": self.model,
            "location": RouteLocation.REMOTE.value,
        }
        self._model_digest = hashlib.sha256(self.model.encode("utf-8")).hexdigest()
        self._route_digest = _sha256(self._route_identity)

    def disable(self) -> None:
        """Trip the temporary route kill switch without touching provider state."""

        self.enabled = False

    def _metadata(
        self, *, prompt_digest: str = "", schema_digest: str = ""
    ) -> dict[str, Any]:
        return {
            "_route_identity": dict(self._route_identity),
            "_model_digest": self._model_digest,
            "_route_digest": self._route_digest,
            "_prompt_digest": prompt_digest,
            "_schema_digest": schema_digest,
        }

    def _failure(
        self,
        category: str,
        *,
        prompt_digest: str = "",
        schema_digest: str = "",
    ) -> dict[str, Any]:
        failure: dict[str, Any] = {
            "class": category,
            "retryable": category in _FAILURE_TRANSIENT,
            "labelable": False,
        }
        return {
            "_failure": failure,
            **self._metadata(
                prompt_digest=prompt_digest,
                schema_digest=schema_digest,
            ),
        }

    def evaluate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.enabled:
            return self._failure("remote_teacher_disabled")
        if not isinstance(payload, Mapping):
            return self._failure("remote_payload_rejected")
        validated = _validate_payload(payload, max_input_bytes=self.max_input_bytes)
        if validated is None:
            return self._failure("remote_payload_rejected")
        normalized, candidate_ids = validated
        schema = _teacher_schema(candidate_ids)
        prompt_digest = ""
        schema_digest = ""
        try:
            prompt_json = _json_bytes(normalized).decode("utf-8")
            system = (
                "You are a temporary Recall relevance teacher. Judge only the "
                "supplied point-in-time evidence. Return schema-valid JSON; use "
                "uncertain when evidence is insufficient."
            )
            prompt = (
                "Label every candidate exactly once. Do not add facts or repeat "
                "secrets. Input:\n" + prompt_json
            )
            if (
                len(system.encode("utf-8"))
                + len(prompt.encode("utf-8"))
                + len(_json_bytes(schema))
                > MAX_REQUEST_BYTES
            ):
                return self._failure("remote_payload_rejected")
            prompt_digest = _sha256({"system": system, "prompt": prompt})
            schema_digest = _sha256(schema)
            result = self._backend.generate(
                GenerationRequest(
                    prompt=prompt,
                    source=SourceDataClassification(
                        SourceDataClass.DERIVED_SNIPPET,
                        SourceSensitivity.NORMAL,
                    ),
                    system=system,
                    format=schema,
                    max_output_tokens=4_000,
                    timeout_ms=self.timeout_ms,
                    temperature=0,
                ),
                model=self._request_model,
            )
        except ProviderAdapterError as exc:
            return self._failure(
                exc.category.value,
                prompt_digest=prompt_digest,
                schema_digest=schema_digest,
            )
        except Exception:
            return self._failure(
                "backend_error",
                prompt_digest=prompt_digest,
                schema_digest=schema_digest,
            )
        try:
            decoded = json.loads(result.content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return self._failure(
                ProviderFailureCategory.INVALID_RESPONSE.value,
                prompt_digest=prompt_digest,
                schema_digest=schema_digest,
            )
        if not isinstance(decoded, Mapping) or set(decoded) != {"labels"}:
            return self._failure(
                ProviderFailureCategory.INVALID_RESPONSE.value,
                prompt_digest=prompt_digest,
                schema_digest=schema_digest,
            )
        labels = decoded.get("labels")
        if not isinstance(labels, list) or len(labels) != len(candidate_ids):
            return self._failure(
                ProviderFailureCategory.INVALID_RESPONSE.value,
                prompt_digest=prompt_digest,
                schema_digest=schema_digest,
            )
        safe_labels = [
            _safe_label(label, frozenset(candidate_ids)) for label in labels
        ]
        if (
            any(label is None for label in safe_labels)
            or {label["candidate_id"] for label in safe_labels if label is not None}
            != set(candidate_ids)
        ):
            return self._failure(
                ProviderFailureCategory.INVALID_RESPONSE.value,
                prompt_digest=prompt_digest,
                schema_digest=schema_digest,
            )
        return {
            "labels": [label for label in safe_labels if label is not None],
            **self._metadata(
                prompt_digest=prompt_digest,
                schema_digest=schema_digest,
            ),
        }


OXAlphaRemoteTeacher = OpenCodeOxAlphaTeacher
