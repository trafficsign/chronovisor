"""Temporary, free-only OpenCode Go teacher adapter for Recall distillation.

The adapter deliberately has one narrow seam: ``Teacher.evaluate`` receives a
compact, already-selected batch and returns labels or a redacted failure
classification.  Provider authentication and HTTP remain in the shared
OpenAI-compatible adapter.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core import runtime_config
from chronovisor.core.egress_policy import guard_egress_query
from chronovisor.core.llm_config import is_openai_compatible_adapter
from chronovisor.core.llm_runtime import (
    GenerationBackend,
    GenerationRequest,
    RouteLocation,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
    safe_metadata_identifier,
)
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
OX_RATIONALE_CODES = (
    "direct_match",
    "partial_match",
    "insufficient_evidence",
    "contradictory_evidence",
    "not_relevant",
)

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
_SINGLE_JSON_FENCE = re.compile(r"\A```json\n((?:(?!```).)*)\n```\Z", re.DOTALL)
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
_RELATIVE_PATH_TEXT = re.compile(
    r"(?ix)(?:^|[\s(=:\"'])"
    r"(?:\.\.?[/\\])?[A-Za-z0-9_.-]+(?:[/\\][A-Za-z0-9_.-]+)+"
    r"(?:$|[\s)\"',;:])"
)
_PROMPT_INJECTION_TEXT = re.compile(
    r"(?ix)"
    r"\b(?:ignore|disregard|override)\s+(?:all\s+)?previous\s+"
    r"(?:instruction|instructions|message|messages|prompt|prompts)\b"
    r"|\b(?:system|developer)\s+(?:prompt|message|instruction|instructions)\b"
    r"|\btool\s+(?:call|invocation)\b"
    r"|(?:命令|指示)(?:を|は)?無視"
    r"|システムプロンプト|開発者メッセージ|ツール(?:呼び出し|コール)"
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
                            "enum": list(OX_RATIONALE_CODES),
                        },
                    },
                },
            }
        },
    }


OX_ALPHA_FIXED_IDENTITY_REVISION = "ox-alpha-fixed-identity-v1"
_SYSTEM_PROMPT = (
    "You are a temporary Recall relevance teacher. Judge only the "
    "supplied point-in-time evidence. Return schema-valid JSON; use "
    "uncertain when evidence is insufficient. The rationale field "
    "must contain one fixed snake_case code, never free text."
)
_PROMPT_PREFIX = (
    "Label every candidate exactly once. Return only one JSON object; "
    "do not add facts, markdown, prose, or repeat secrets. Use only "
    "the fixed rationale codes in the schema. Output schema:\n"
)
_PROMPT_INPUT_SEPARATOR = "\nInput:\n"
_SCHEMA_CANDIDATE_PLACEHOLDER = "{candidate_id}"


def _prompt_template_digest(
    system: str = _SYSTEM_PROMPT,
    prefix: str = _PROMPT_PREFIX,
    separator: str = _PROMPT_INPUT_SEPARATOR,
) -> str:
    return _sha256({"system": system, "prefix": prefix, "separator": separator})


def _schema_revision_digest(
    schema: Mapping[str, Any] | None = None,
) -> str:
    return _sha256(
        schema
        if schema is not None
        else _teacher_schema((_SCHEMA_CANDIDATE_PLACEHOLDER,))
    )


OX_ALPHA_FIXED_IDENTITY = {
    "revision": OX_ALPHA_FIXED_IDENTITY_REVISION,
    "route_identity": {
        "provider": OX_ALPHA_PROVIDER,
        "model": OX_ALPHA_ROUTE_MODEL,
        "location": RouteLocation.REMOTE.value,
    },
    "model_digest": hashlib.sha256(OX_ALPHA_ROUTE_MODEL.encode("utf-8")).hexdigest(),
    "route_digest": _sha256(
        {
            "provider": OX_ALPHA_PROVIDER,
            "model": OX_ALPHA_ROUTE_MODEL,
            "location": RouteLocation.REMOTE.value,
        }
    ),
    "prompt_template_sha256": _prompt_template_digest(),
    "schema_revision_sha256": _schema_revision_digest(),
}


_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


def _git_snapshot_entries(
    source: Path, command: list[str]
) -> tuple[tuple[bytes, bytes, bytes], ...]:
    """Return the exact mode/blob/path snapshot, rejecting non-stage-zero index rows."""

    output = subprocess.run(
        command,
        cwd=source,
        check=True,
        capture_output=True,
        timeout=5,
    ).stdout.split(b"\0")
    entries: list[tuple[bytes, bytes, bytes]] = []
    for entry in output:
        if not entry:
            continue
        try:
            metadata, path = entry.split(b"\t", 1)
            fields = metadata.split(b" ")
            if command[2] == "-s":
                mode, blob, stage = fields
                if stage != b"0":
                    raise ValueError("index contains an unresolved entry")
            else:
                mode, _kind, blob = fields
        except ValueError as exc:
            raise ValueError("installed OX source index is invalid") from exc
        entries.append((mode, blob, path))
    return tuple(sorted(entries))


def _has_symlink_component(path: Path) -> bool:
    current = path.expanduser().absolute()
    while True:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def ox_alpha_source_binding() -> dict[str, str]:
    """Return the clean, installed source identity required for OX egress.

    The worktree is deliberately required to match the installed archive and
    origin/main.  A local edit must never acquire remote labels that later look
    like production evidence.
    """

    runtime = runtime_config.runtime_identity()
    commit = runtime.get("commit_id")
    expected = runtime.get("expected_commit")
    if (
        not isinstance(commit, str)
        or _COMMIT_RE.fullmatch(commit) is None
        or commit != expected
        or runtime.get("drift") is not False
    ):
        raise ValueError("installed OX source identity is unavailable")
    try:
        source_root = runtime_config.runtime_repo_root()
        if _has_symlink_component(source_root):
            raise ValueError("installed OX source root contains a symlink")
        source = source_root.resolve(strict=True)
        root_stat = source.stat()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        origin = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        # `status` honors assume-unchanged.  These two index comparisons and
        # the blob check below deliberately do not trust that optimization.
        dirty = [
            subprocess.run(
                command,
                cwd=source,
                capture_output=True,
                timeout=5,
            ).returncode
            for command in (
                ["git", "diff-files", "--quiet", "--no-ext-diff"],
                ["git", "diff-index", "--cached", "--quiet", "HEAD", "--"],
            )
        ]
        index_entries = _git_snapshot_entries(source, ["git", "ls-files", "-s", "-z"])
        head_entries = _git_snapshot_entries(
            source, ["git", "ls-tree", "-r", "-z", "HEAD"]
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("installed OX source identity is unavailable") from exc
    if (
        head != commit
        or origin != commit
        or status
        or any(dirty)
        or index_entries != head_entries
    ):
        raise ValueError("installed OX source checkout is not exact")
    module_path = Path(__file__).resolve(strict=True)
    installed_bytes = module_path.read_bytes()
    installed_digest = hashlib.sha256(installed_bytes).hexdigest()
    digest = hashlib.sha256()
    remote_sha256 = ""
    for _mode, blob, raw_path in index_entries:
        relative = os.fsdecode(raw_path)
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("installed OX source index is unsafe")
        path = source / relative
        try:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("installed OX source contains a non-file")
            content = path.read_bytes()
            after = path.lstat()
        except OSError as exc:
            raise ValueError("installed OX source is unreadable") from exc
        if (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("installed OX source changed while reading")
        content_sha256 = hashlib.sha256(content).hexdigest()
        git_blob = (
            hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content)
            .hexdigest()
            .encode("ascii")
        )
        if git_blob != blob:
            raise ValueError("installed OX source blob differs from index")
        digest.update(
            _json_bytes(
                {
                    "kind": "file",
                    "path": relative,
                    "size": int(before.st_size),
                    "sha256": content_sha256,
                }
            )
        )
        digest.update(b"\n")
        if relative == "src/chronovisor/recall/recall_distillation_remote_teacher.py":
            remote_sha256 = content_sha256
    after_root = source.stat()
    if (after_root.st_dev, after_root.st_ino) != (root_stat.st_dev, root_stat.st_ino):
        raise ValueError("installed OX source root changed while reading")
    try:
        index_after = _git_snapshot_entries(source, ["git", "ls-files", "-s", "-z"])
        head_tree_after = _git_snapshot_entries(
            source, ["git", "ls-tree", "-r", "-z", "HEAD"]
        )
        head_after = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        origin_after = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("installed OX source identity is unavailable") from exc
    if (
        index_after != index_entries
        or head_tree_after != head_entries
        or head_after != head
        or index_after != head_tree_after
    ):
        raise ValueError("installed OX source index changed while reading")
    if not hmac.compare_digest(origin, origin_after) or origin_after != commit:
        raise ValueError("installed OX origin/main changed while reading")
    if not remote_sha256 or remote_sha256 != installed_digest:
        raise ValueError("installed OX source identity is incomplete")
    return {
        "source_commit": commit,
        "source_tree_sha256": digest.hexdigest(),
        "source_ox_identity_sha256": remote_sha256,
    }


def _contains_forbidden_text(value: str) -> bool:
    return bool(
        _SECRET_TEXT.search(value)
        or _PII_TEXT.search(value)
        or _PRIVATE_WORK_TEXT.search(value)
        or _LOCAL_PATH_TEXT.search(value)
        or _RELATIVE_PATH_TEXT.search(value)
        or _PROMPT_INJECTION_TEXT.search(value)
    )


def _safe_text(
    value: object, *, required: bool = True, max_chars: int = MAX_TEXT_CHARS
) -> str | None:
    if not isinstance(value, str):
        return None
    if not value and not required:
        return ""
    decision = guard_egress_query(value, max_chars=max_chars)
    if not decision.allowed or _contains_forbidden_text(decision.normalized):
        return None
    return decision.normalized


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
        query = _safe_text(candidate.get("query"))
        evidence = _safe_text(candidate.get("evidence"))
        safe_context = (
            [_safe_text(item) for item in context] if isinstance(context, list) else []
        )
        if (
            not isinstance(candidate_id, str)
            or _SAFE_ID.fullmatch(candidate_id) is None
            or _contains_forbidden_text(candidate_id)
            or not isinstance(rally_id, str)
            or _SAFE_ID.fullmatch(rally_id) is None
            or _contains_forbidden_text(rally_id)
            or query is None
            or evidence is None
            or not isinstance(context, list)
            or len(context) > MAX_TEACHER_CANDIDATES
            or any(item is None for item in safe_context)
        ):
            return None
        if candidate_id in candidate_ids:
            return None
        candidate_ids.append(candidate_id)
        normalized.append(
            {
                "candidate_id": candidate_id,
                "rally_id": rally_id,
                "query": query,
                "context": safe_context,
                "evidence": evidence,
            }
        )
    result = {"schema": TEACHER_BATCH_SCHEMA, "candidates": normalized}
    try:
        if len(_json_bytes(result)) > max_input_bytes:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return result, tuple(candidate_ids)


def _prepare_request(
    payload: Mapping[str, Any], *, max_input_bytes: int
) -> tuple[tuple[str, ...], dict[str, Any], str, str] | None:
    validated = _validate_payload(payload, max_input_bytes=max_input_bytes)
    if validated is None:
        return None
    normalized, candidate_ids = validated
    schema = _teacher_schema(candidate_ids)
    try:
        prompt_json = _json_bytes(normalized).decode("utf-8")
        schema_json = _json_bytes(schema).decode("utf-8")
        system = _SYSTEM_PROMPT
        prompt = _PROMPT_PREFIX + schema_json + _PROMPT_INPUT_SEPARATOR + prompt_json
        if (
            len(system.encode("utf-8")) + len(prompt.encode("utf-8"))
            > MAX_REQUEST_BYTES
        ):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return candidate_ids, schema, system, prompt


def ox_alpha_response_metadata(
    payload: Mapping[str, Any], *, max_input_bytes: int = MAX_PAYLOAD_BYTES
) -> dict[str, Any] | None:
    """Return fixed identity plus the digest of one normalized egress request."""

    prepared = _prepare_request(payload, max_input_bytes=max_input_bytes)
    if prepared is None:
        return None
    _candidate_ids, schema, system, prompt = prepared
    return {
        "_identity_revision": OX_ALPHA_FIXED_IDENTITY["revision"],
        "_route_identity": dict(OX_ALPHA_FIXED_IDENTITY["route_identity"]),
        "_model_digest": OX_ALPHA_FIXED_IDENTITY["model_digest"],
        "_route_digest": OX_ALPHA_FIXED_IDENTITY["route_digest"],
        "_prompt_digest": OX_ALPHA_FIXED_IDENTITY["prompt_template_sha256"],
        "_schema_digest": OX_ALPHA_FIXED_IDENTITY["schema_revision_sha256"],
        "_request_digest": _sha256(
            {"system": system, "prompt": prompt, "format": schema}
        ),
    }


def _safe_label(label: object, candidate_ids: frozenset[str]) -> dict[str, Any] | None:
    if not isinstance(label, Mapping):
        return None
    allowed = {
        "candidate_id",
        "verdict",
        "confidence",
        "rationale",
    }
    if set(label) != allowed:
        return None
    candidate_id = label.get("candidate_id")
    verdict = label.get("verdict")
    confidence = label.get("confidence")
    rationale = label.get("rationale")
    safe_rationale = _safe_text(rationale, max_chars=600)
    if (
        not isinstance(candidate_id, str)
        or candidate_id not in candidate_ids
        or verdict not in {"relevant", "irrelevant", "uncertain"}
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
        or rationale not in OX_RATIONALE_CODES
        or safe_rationale is None
    ):
        return None
    return {
        "candidate_id": candidate_id,
        "verdict": verdict,
        "confidence": confidence,
        "rationale": safe_rationale,
    }


def validate_ox_alpha_labels(
    labels: object, candidate_ids: tuple[str, ...]
) -> list[dict[str, Any]] | None:
    """Validate the exact OX label contract at every trust boundary."""

    if not isinstance(labels, list) or len(labels) != len(candidate_ids):
        return None
    safe_labels = [_safe_label(label, frozenset(candidate_ids)) for label in labels]
    if any(label is None for label in safe_labels) or {
        label["candidate_id"] for label in safe_labels if label is not None
    } != set(candidate_ids):
        return None
    return [label for label in safe_labels if label is not None]


class OpenCodeOxAlphaTeacher:
    """A free-only remote adapter satisfying the Recall ``Teacher`` seam."""

    local = False
    location = RouteLocation.REMOTE

    def __init__(
        self,
        backend: GenerationBackend,
        *,
        configured_model: str = OX_ALPHA_ROUTE_MODEL,
        enabled: bool = True,
        free_only: bool = True,
        allow_paid_fallback: bool = False,
        test_only: bool = False,
        simulation_attestation: Path | None = None,
        owned_root: Path | None = None,
        max_input_bytes: int = MAX_PAYLOAD_BYTES,
        timeout_ms: int = 60_000,
    ) -> None:
        profile = getattr(backend, "_profile", None)
        endpoint = getattr(profile, "endpoint", None)
        if (
            not callable(getattr(backend, "generate", None))
            or backend.provider != OX_ALPHA_PROVIDER
            or backend.location is not RouteLocation.REMOTE
            or endpoint != OX_ALPHA_ENDPOINT
        ):
            raise ValueError("invalid OX Alpha route")
        if test_only is not True and not is_openai_compatible_adapter(backend):
            raise ValueError("untrusted OX Alpha backend")
        if test_only is not True and (
            simulation_attestation is not None or owned_root is not None
        ):
            raise ValueError("simulation attestation is test-only")
        if (simulation_attestation is None) != (owned_root is None):
            raise ValueError("simulation attestation root is incomplete")
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
        capabilities_for = getattr(backend, "capabilities_for", None)
        if not callable(capabilities_for) or (
            getattr(capabilities_for(self._request_model), "structured_output", False)
            is not True
        ):
            raise ValueError("O× Alpha route lacks structured output")
        self.role = "recall.distill.teacher.ox-alpha"
        self.provider = OX_ALPHA_PROVIDER
        self.model = OX_ALPHA_ROUTE_MODEL
        self.enabled = bool(enabled)
        self.test_only = bool(test_only)
        self._simulation_attestation = simulation_attestation
        self._owned_root = owned_root
        self.max_input_bytes = max_input_bytes
        self.timeout_ms = timeout_ms
        self._route_identity = dict(OX_ALPHA_FIXED_IDENTITY["route_identity"])

    def disable(self) -> None:
        """Trip the temporary route kill switch without touching provider state."""

        self.enabled = False

    def receipt_binding(self) -> dict[str, str]:
        """Bind remote labels to the exact installed source before egress."""
        if self.test_only and self._simulation_attestation is not None:
            path = self._simulation_attestation
            root = self._owned_root
            if (
                root is None
                or path.is_symlink()
                or not path.is_file()
                or root.is_symlink()
            ):
                raise ValueError("simulation attestation is unavailable")
            before = path.stat()
            root_stat = root.resolve(strict=True).stat()
            try:
                payload = json.loads(path.read_bytes())
            except (OSError, ValueError, UnicodeError) as exc:
                raise ValueError("simulation attestation is invalid") from exc
            after = path.stat()
            if before != after or not isinstance(payload, Mapping):
                raise ValueError("simulation attestation changed during read")
            unsigned = {
                key: value for key, value in payload.items() if key != "seal_sha256"
            }
            binding = payload.get("source_binding")
            try:
                expires = datetime.fromisoformat(
                    str(payload.get("expires_at")).replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise ValueError("simulation attestation expiry is invalid") from exc
            if (
                payload.get("schema")
                != "chronovisor.recall-r4-simulation-attestation.v1"
                or payload.get("namespace") != "recall-distillation"
                or payload.get("seal_sha256") != _sha256(unsigned)
                or payload.get("owned_root")
                != {"st_dev": root_stat.st_dev, "st_ino": root_stat.st_ino}
                or expires.tzinfo is None
                or expires <= datetime.now(UTC)
                or expires > datetime.now(UTC) + timedelta(minutes=10)
                or not isinstance(binding, Mapping)
                or set(binding)
                != {"source_commit", "source_tree_sha256", "source_ox_identity_sha256"}
                or not isinstance(binding.get("source_commit"), str)
                or re.fullmatch(r"[0-9a-f]{40}", binding["source_commit"]) is None
                or any(
                    not isinstance(binding.get(key), str)
                    or re.fullmatch(r"[0-9a-f]{64}", str(binding.get(key))) is None
                    for key in ("source_tree_sha256", "source_ox_identity_sha256")
                )
            ):
                raise ValueError("simulation attestation binding is invalid")
            try:
                expected_binding = ox_alpha_source_binding()
            except ValueError as exc:
                raise ValueError("simulation attestation binding is invalid") from exc
            if dict(binding) != expected_binding:
                raise ValueError("simulation attestation binding is invalid")
            # This short-lived schema is explicitly non-certifying: callers
            # retain `_test_only` metadata and the production collector rejects
            # every simulation artifact.  Formal OX identity remains available
            # only through `ox_alpha_source_binding()` on the non-test path.
            return {key: str(value) for key, value in binding.items()}
        return ox_alpha_source_binding()

    def _metadata(self, *, request_digest: str = "") -> dict[str, Any]:
        return {
            "_identity_revision": OX_ALPHA_FIXED_IDENTITY["revision"],
            "_route_identity": dict(self._route_identity),
            "_model_digest": OX_ALPHA_FIXED_IDENTITY["model_digest"],
            "_route_digest": OX_ALPHA_FIXED_IDENTITY["route_digest"],
            "_prompt_digest": OX_ALPHA_FIXED_IDENTITY["prompt_template_sha256"],
            "_schema_digest": OX_ALPHA_FIXED_IDENTITY["schema_revision_sha256"],
            "_request_digest": request_digest,
            "_test_only": self.test_only,
        }

    def _failure(
        self,
        category: str,
        *,
        request_digest: str = "",
        stage: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        failure: dict[str, Any] = {
            "class": category,
            "retryable": category in _FAILURE_TRANSIENT,
            "labelable": False,
        }
        if safe_metadata_identifier(stage) is not None:
            failure["stage"] = stage
        safe_request_id = safe_metadata_identifier(request_id)
        if safe_request_id is not None:
            failure["request_id"] = safe_request_id
        return {
            "_failure": failure,
            **self._metadata(request_digest=request_digest),
        }

    def accepts_egress_payload(self, payload: Mapping[str, Any]) -> bool:
        """Return whether the exact adapter checks can reach HTTP."""

        return (
            isinstance(payload, Mapping)
            and _prepare_request(payload, max_input_bytes=self.max_input_bytes)
            is not None
        )

    def evaluate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.enabled:
            return self._failure("remote_teacher_disabled")
        if not isinstance(payload, Mapping):
            return self._failure("remote_payload_rejected")
        prepared = _prepare_request(payload, max_input_bytes=self.max_input_bytes)
        if prepared is None:
            return self._failure("remote_payload_rejected")
        candidate_ids, schema, system, prompt = prepared
        response_metadata = ox_alpha_response_metadata(
            payload, max_input_bytes=self.max_input_bytes
        )
        if response_metadata is None:
            return self._failure("remote_payload_rejected")
        request_digest = str(response_metadata["_request_digest"])
        try:
            result = self._backend.generate(
                GenerationRequest(
                    prompt=prompt,
                    source=SourceDataClassification(
                        SourceDataClass.DERIVED_SNIPPET,
                        SourceSensitivity.NORMAL,
                    ),
                    system=system,
                    format=schema,
                    max_output_tokens=16_000,
                    timeout_ms=self.timeout_ms,
                    temperature=0,
                ),
                model=self._request_model,
            )
        except ProviderAdapterError as exc:
            return self._failure(
                exc.category.value,
                request_digest=request_digest,
                stage=exc.stage,
                request_id=exc.request_id,
            )
        except Exception:
            return self._failure(
                "backend_error",
                request_digest=request_digest,
            )
        returned_model = None
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, Mapping):
            returned_model = metadata.get("returned_model")
        request_id = (
            safe_metadata_identifier(metadata.get("request_id"))
            if isinstance(metadata, Mapping)
            else None
        )
        if returned_model != OX_ALPHA_REQUEST_MODEL:
            return self._failure(
                "model_unavailable",
                request_digest=request_digest,
                request_id=request_id,
            )
        if result.finish_reason != "stop":
            return self._failure(
                ProviderFailureCategory.INVALID_RESPONSE.value,
                request_digest=request_digest,
                stage="teacher_finish_reason",
                request_id=request_id,
            )
        content = result.content
        if isinstance(content, str):
            fenced = _SINGLE_JSON_FENCE.fullmatch(content.strip())
            if fenced is not None:
                content = fenced.group(1)
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return self._failure(
                ProviderFailureCategory.INVALID_RESPONSE.value,
                request_digest=request_digest,
                stage="teacher_json_parse",
                request_id=request_id,
            )
        if not isinstance(decoded, Mapping) or set(decoded) != {"labels"}:
            return self._failure(
                ProviderFailureCategory.INVALID_RESPONSE.value,
                request_digest=request_digest,
                stage="teacher_response_shape",
                request_id=request_id,
            )
        labels = decoded.get("labels")
        if not isinstance(labels, list) or len(labels) != len(candidate_ids):
            return self._failure(
                ProviderFailureCategory.INVALID_RESPONSE.value,
                request_digest=request_digest,
                stage="teacher_label_count",
                request_id=request_id,
            )
        safe_labels = validate_ox_alpha_labels(labels, candidate_ids)
        if safe_labels is None:
            return self._failure(
                ProviderFailureCategory.INVALID_RESPONSE.value,
                request_digest=request_digest,
                stage="teacher_label_schema",
                request_id=request_id,
            )
        return {
            "labels": safe_labels,
            **response_metadata,
            "_test_only": self.test_only,
        }


OXAlphaRemoteTeacher = OpenCodeOxAlphaTeacher
