"""Entity backfill proposal and validation contracts."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chronovisor.core.frontmatter import parse, patch
from chronovisor.core.hashutil import sha256_text as _sha256_text
from chronovisor.core.runtime_config import runtime_repo_root
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.decision.lint_mutation_contract import review_packet_error

ENTITY_DIR = CHRONOVISOR_ROOT / "entities"
ENTITY_REGISTRY_FILE = ENTITY_DIR / "registry.json"
ENTITY_REVIEW_DIR = CHRONOVISOR_ROOT / "runtime" / "entity-backfill"
REPO_ROOT = runtime_repo_root()
ENTITY_PROPOSAL_VERSION = 2

DEFAULT_ALIASES: dict[str, list[str]] = {
    "chronovisor": [
        "Chronovisor",
        "クロノバイザー",
        "ウィキ",
    ],
    "codex": ["Codex", "コードエクス", "コーデックス"],
    "claude-code": ["Claude Code", "クラウドコード"],
    "ollama": ["Ollama"],
    "qwen": ["Qwen", "クエン"],
    "gemma": ["Gemma", "ジェンマ"],
    "mhi": ["MHI", "三菱重工", "三菱重工業"],
    "khi": ["KHI", "川崎重工", "川重"],
    "mazda": ["Mazda", "マツダ"],
}

ENTITY_ID_RE = re.compile(r"[^a-z0-9]+")


def normalize_entity_id(value: str) -> str:
    normalized = value.strip().casefold()
    normalized = ENTITY_ID_RE.sub("-", normalized).strip("-")
    return normalized[:80]


def load_registry(path: Path = ENTITY_REGISTRY_FILE) -> dict[str, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_ALIASES)
    aliases: dict[str, list[str]] = dict(DEFAULT_ALIASES)
    if isinstance(data, dict):
        raw_entities = data.get("entities", data)
        if isinstance(raw_entities, dict):
            for key, value in raw_entities.items():
                entity_id = normalize_entity_id(str(key))
                if not entity_id:
                    continue
                values: list[str] = []
                if isinstance(value, list):
                    values = [v for v in value if isinstance(v, str) and v.strip()]
                elif isinstance(value, dict):
                    raw_aliases = value.get("aliases")
                    if isinstance(raw_aliases, list):
                        values = [
                            v for v in raw_aliases if isinstance(v, str) and v.strip()
                        ]
                    label = value.get("label")
                    if isinstance(label, str) and label.strip():
                        values.insert(0, label)
                aliases[entity_id] = list(dict.fromkeys([entity_id, *values]))
    return aliases


def extract_entities(
    text: str, *, registry: dict[str, list[str]] | None = None
) -> list[str]:
    registry = registry or load_registry()
    haystack = text.casefold()
    found: list[str] = []
    for entity_id, aliases in registry.items():
        for alias in aliases:
            if alias and alias.casefold() in haystack:
                found.append(entity_id)
                break
    return list(dict.fromkeys(found))[:20]


def patch_entities_frontmatter(
    text: str,
    *,
    registry: dict[str, list[str]] | None = None,
) -> str:
    meta, body = parse(text)
    title = meta.get("title")
    current = meta.get("entities")
    existing = current if isinstance(current, list) else []
    extracted = extract_entities(
        "\n".join([title if isinstance(title, str) else "", body]),
        registry=registry,
    )
    merged = list(dict.fromkeys([*existing, *extracted]))
    if not merged or merged == existing:
        return text
    return patch(text, {"entities": merged})


def _review_evidence(
    text: str,
    new_text: str,
    *,
    registry: Mapping[str, list[str]],
) -> dict[str, Any]:
    meta, body = parse(text)
    new_meta, _new_body = parse(new_text)
    title = meta.get("title") if isinstance(meta.get("title"), str) else ""
    haystack = f"{title}\n{body}".casefold()
    before = meta.get("entities") if isinstance(meta.get("entities"), list) else []
    after = (
        new_meta.get("entities") if isinstance(new_meta.get("entities"), list) else []
    )
    before_ids = [item for item in before if isinstance(item, str)]
    after_ids = [item for item in after if isinstance(item, str)]
    added = [item for item in after_ids if item not in before_ids]
    alias_evidence = []
    for entity_id in added:
        aliases = [
            alias for alias in registry.get(entity_id, []) if isinstance(alias, str)
        ]
        alias_evidence.append(
            {
                "entity_id": entity_id,
                "matched_aliases": [
                    alias for alias in aliases if alias and alias.casefold() in haystack
                ],
                "aliases_considered": aliases,
            }
        )
    registry_payload = json.dumps(
        {key: list(value) for key, value in registry.items()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "proposal_generator_version": ENTITY_PROPOSAL_VERSION,
        "existing_entities": before_ids,
        "proposed_entities": after_ids,
        "added_entities": added,
        "alias_evidence": alias_evidence,
        "registry_sha256": hashlib.sha256(registry_payload.encode("utf-8")).hexdigest(),
    }


review_evidence = _review_evidence




def validate_entity_backfill_proposal(
    proposal: Mapping[str, Any],
    *,
    expected_text: str | None = None,
    updated_text: str | None = None,
    registry: Mapping[str, list[str]] | None = None,
) -> bool:
    """Prove that an entity proposal is an alias-backed frontmatter-only edit."""

    if (
        proposal.get("schema_version") != 1
        or proposal.get("kind") != "lint_safe_fix_proposal"
        or proposal.get("operation") != "backfill_entities_frontmatter"
        or proposal.get("unified_diff_truncated") is not False
    ):
        return False
    hash_fields = (
        "expected_sha256",
        "updated_sha256",
        "unified_diff_sha256",
        "full_unified_diff_sha256",
    )
    if any(
        not isinstance(proposal.get(name), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(proposal.get(name))) is None
        for name in hash_fields
    ):
        return False
    review_packet = proposal.get("review_packet")
    if not isinstance(review_packet, Mapping):
        return False
    mode = review_packet.get("mode")
    unified_diff = proposal.get("unified_diff")
    if mode == "full":
        if not isinstance(unified_diff, str) or not unified_diff:
            return False
        if _sha256_text(unified_diff) != proposal.get(
            "unified_diff_sha256"
        ) or proposal.get("unified_diff_sha256") != proposal.get(
            "full_unified_diff_sha256"
        ):
            return False
    elif mode == "changed_spans":
        coverage = review_packet.get("coverage")
        if (
            unified_diff is not None
            or proposal.get("unified_diff_repacket") is not True
            or not isinstance(coverage, Mapping)
            or coverage.get("all_changed_spans_rendered") is not True
        ):
            return False
    else:
        return False

    details = proposal.get("details")
    if (
        not isinstance(details, Mapping)
        or details.get("proposal_generator_version") != ENTITY_PROPOSAL_VERSION
    ):
        return False
    existing = details.get("existing_entities")
    proposed = details.get("proposed_entities")
    added = details.get("added_entities")
    evidence = details.get("alias_evidence")
    if not all(
        isinstance(value, list) and all(isinstance(item, str) for item in value)
        for value in (existing, proposed, added)
    ):
        return False
    assert (
        isinstance(existing, list)
        and isinstance(proposed, list)
        and isinstance(added, list)
    )
    if (
        not added
        or len(existing) != len(set(existing))
        or any(normalize_entity_id(item) != item for item in existing)
        or len(proposed) != len(set(proposed))
        or any(normalize_entity_id(item) != item for item in proposed)
        or any(item not in proposed for item in existing)
        or added != [item for item in proposed if item not in existing]
        or not isinstance(evidence, list)
        or len(evidence) != len(added)
    ):
        return False
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for row in evidence:
        if not isinstance(row, Mapping) or not isinstance(row.get("entity_id"), str):
            return False
        entity_id = str(row["entity_id"])
        matched = row.get("matched_aliases")
        considered = row.get("aliases_considered")
        if (
            entity_id in evidence_by_id
            or entity_id not in added
            or not isinstance(matched, list)
            or not matched
            or not all(isinstance(alias, str) and alias for alias in matched)
            or not isinstance(considered, list)
            or not all(isinstance(alias, str) and alias for alias in considered)
            or any(alias not in considered for alias in matched)
        ):
            return False
        evidence_by_id[entity_id] = row
    if set(evidence_by_id) != set(added):
        return False
    registry_sha = details.get("registry_sha256")
    if (
        not isinstance(registry_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", registry_sha) is None
    ):
        return False

    if (expected_text is None) != (updated_text is None):
        return False
    if expected_text is not None and updated_text is not None:
        if _sha256_text(expected_text) != proposal.get(
            "expected_sha256"
        ) or _sha256_text(updated_text) != proposal.get("updated_sha256"):
            return False
        exact_diff = "".join(
            difflib.unified_diff(
                expected_text.splitlines(keepends=True),
                updated_text.splitlines(keepends=True),
                fromfile=f"{proposal.get('page_id')}:before",
                tofile=f"{proposal.get('page_id')}:after",
                n=5,
            )
        )
        if (
            not exact_diff
            or _sha256_text(exact_diff) != proposal.get("unified_diff_sha256")
            or proposal.get("unified_diff_sha256")
            != proposal.get("full_unified_diff_sha256")
        ):
            return False
        before_meta, before_body = parse(expected_text)
        after_meta, after_body = parse(updated_text)
        if before_body != after_body:
            return False
        before_entities = before_meta.get("entities")
        if "entities" not in before_meta:
            if existing:
                return False
        elif (
            not isinstance(before_entities, list)
            or not all(isinstance(item, str) for item in before_entities)
            or before_entities != existing
        ):
            return False
        after_entities = after_meta.get("entities")
        if (
            not isinstance(after_entities, list)
            or not all(isinstance(item, str) for item in after_entities)
            or after_entities != proposed
        ):
            return False
        before_without = {
            key: value for key, value in before_meta.items() if key != "entities"
        }
        after_without = {
            key: value for key, value in after_meta.items() if key != "entities"
        }
        if before_without != after_without:
            return False
        expected_details = (
            _review_evidence(expected_text, updated_text, registry=registry)
            if registry is not None
            else None
        )
        observed_details = dict(details)
        observed_details.pop("review_receipt", None)
        if (
            expected_details is None
            or observed_details != expected_details
            or review_packet_error(
                proposal,
                expected_text=expected_text,
                updated_text=updated_text,
            )
            is not None
        ):
            return False
    return True
