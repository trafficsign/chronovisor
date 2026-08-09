"""CAS-bound merge transaction engine with isolated TTL preimages."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import atomic_write_bytes, write_sealed_json
from chronovisor.core.hashutil import sha256_bytes as _sha256_bytes
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.page_mutation import chronovisor_mutation_lock
from chronovisor.core.timeutil import utc_now as _now
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.recall.merge_ledger import (
    MergeCoverageError,
    MergeLedger,
    build_source_inventory,
    verify_merge_coverage,
)

PLAN_SCHEMA = "chronovisor.merge-plan.v1"
PREIMAGE_SCHEMA = "chronovisor.merge-preimage.v1"






def _heading_exists(text: str, anchor: str) -> bool:
    if not anchor:
        return True
    normalized = anchor.strip().casefold().replace(" ", "-")
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        heading = line.lstrip("#").strip().casefold().replace(" ", "-")
        if heading == normalized:
            return True
    return False


def prepare_merge_plan(
    root: Path,
    *,
    source_keys: list[str],
    canonical_key: str,
    canonical_content: str,
    mappings: list[Mapping[str, Any]],
    ledger_dispositions: list[Mapping[str, Any]] | None = None,
    output_sensitivity: str,
    affected_page_updates: Mapping[str, str] | None = None,
    anchor_maps: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Create an exact, read-only merge plan and run deterministic gates."""

    registry = PageRegistry(root)
    state = registry.load()
    resolved = {}
    for key in source_keys:
        row = registry.resolve(key)
        if row is None:
            raise KeyError(key)
        resolved[str(row["uid"])] = row
    canonical = registry.resolve(canonical_key)
    if canonical is None:
        raise KeyError(canonical_key)
    canonical_uid = str(canonical["uid"])
    if canonical_uid not in resolved:
        resolved[canonical_uid] = canonical
    source_texts: dict[str, str] = {}
    inputs: list[dict[str, Any]] = []
    sensitivities: list[str] = []
    for uid, row in sorted(resolved.items()):
        path = root / str(row["path"])
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        source_texts[uid] = text
        sensitivities.append(str(row.get("sensitivity") or "normal"))
        inputs.append(
            {
                "uid": uid,
                "path": str(path.relative_to(root)),
                "content_sha256": _sha256_bytes(raw),
                "sensitivity": str(row.get("sensitivity") or "normal"),
            }
        )
    inventory = build_source_inventory(source_texts)
    coverage = verify_merge_coverage(
        inventory=inventory,
        mappings=mappings,
        output_text=canonical_content,
        ledger_dispositions=ledger_dispositions or [],
        input_sensitivities=sensitivities,
        output_sensitivity=output_sensitivity,
        require_raw_refs=True,
    )
    redirects = []
    anchor_maps = anchor_maps or {}
    for uid in sorted(resolved):
        if uid == canonical_uid:
            continue
        anchor_map = dict(anchor_maps.get(uid) or {})
        missing_anchors = [
            value
            for value in anchor_map.values()
            if not _heading_exists(canonical_content, str(value))
        ]
        if missing_anchors:
            raise MergeCoverageError(
                f"canonical output lacks redirect anchors: {missing_anchors}"
            )
        redirects.append(
            {
                "from_uid": uid,
                "to_uid": canonical_uid,
                "anchor_map": anchor_map,
            }
        )
    affected = []
    for key, content in sorted((affected_page_updates or {}).items()):
        row = registry.resolve(key)
        if row is None:
            raise KeyError(key)
        path = root / str(row["path"])
        raw = path.read_bytes()
        affected.append(
            {
                "uid": str(row["uid"]),
                "path": str(path.relative_to(root)),
                "before_sha256": _sha256_bytes(raw),
                "after_sha256": _sha256_bytes(content.encode("utf-8")),
                "content": content,
            }
        )
    transaction_id = "merge_" + uuid.uuid4().hex
    return {
        "schema": PLAN_SCHEMA,
        "transaction_id": transaction_id,
        "status": "prepared",
        "registry_generation": int(state.get("generation") or 0),
        "inputs": inputs,
        "output": {
            "uid": canonical_uid,
            "path": str(canonical["path"]),
            "content_sha256": _sha256_bytes(canonical_content.encode("utf-8")),
            "content": canonical_content,
            "sensitivity": output_sensitivity,
        },
        "inventory": inventory,
        "claim_map": [dict(value) for value in mappings],
        "ledger_dispositions": [dict(value) for value in (ledger_dispositions or [])],
        "redirects": redirects,
        "link_rewrites": affected,
        "verification_receipt": coverage,
        "prepared_at": _now().isoformat(timespec="milliseconds"),
    }


def _write_preimage(root: Path, plan: Mapping[str, Any], ttl_days: int) -> Path:
    transaction_id = str(plan["transaction_id"])
    path = root / "runtime" / "librarian" / "transaction-preimages" / transaction_id
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    os.chmod(path, 0o700)
    files: list[dict[str, Any]] = []
    for row in [
        *(plan.get("inputs") or []),
        *(plan.get("link_rewrites") or []),
    ]:
        relative = str(row["path"])
        source = root / relative
        raw = source.read_bytes()
        destination = path / "payload" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(destination, raw, backup=False)
        files.append(
            {
                "path": relative,
                "size": len(raw),
                "sha256": _sha256_bytes(raw),
            }
        )
    now = _now()
    write_sealed_json(
        path / "manifest.json",
        {
            "schema": PREIMAGE_SCHEMA,
            "transaction_id": transaction_id,
            "input_uids": [
                str(row.get("uid") or "") for row in plan.get("inputs") or []
            ],
            "canonical_uid": str((plan.get("output") or {}).get("uid") or ""),
            "created_at": now.isoformat(timespec="milliseconds"),
            "expires_at": (now + timedelta(days=max(0, ttl_days))).isoformat(
                timespec="milliseconds"
            ),
            "files": files,
        },
        backup=False,
    )
    return path


def apply_merge_plan(
    root: Path,
    plan: Mapping[str, Any],
    *,
    activate: bool = False,
    preimage_ttl_days: int = 0,
) -> dict[str, Any]:
    """Apply a prepared plan only after explicit activation."""

    if not activate:
        return {
            "status": "blocked",
            "reason": "explicit_activation_required",
            "transaction_id": plan.get("transaction_id"),
        }
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "prepared":
        raise ValueError("unsupported or non-prepared merge plan")
    transaction_id = str(plan["transaction_id"])
    registry = PageRegistry(root)
    ledger = MergeLedger(root)
    preimage = _write_preimage(root, plan, preimage_ttl_days)
    ledger.append(
        {
            "transaction_id": transaction_id,
            "operation": "merge",
            "status": "pending",
            "inputs": plan.get("inputs"),
            "output": {
                key: value
                for key, value in dict(plan.get("output") or {}).items()
                if key != "content"
            },
            "claim_map": plan.get("claim_map"),
            "redirects": plan.get("redirects"),
            "verification_receipt": plan.get("verification_receipt"),
            "temporary_preimage": str(preimage),
        }
    )
    original_bytes: dict[Path, bytes] = {}
    owned_bytes: dict[Path, bytes | None] = {}
    registry_preimage = registry.path.read_bytes()
    registry_owned: bytes | None = None
    try:
        with chronovisor_mutation_lock():
            for row in plan.get("inputs") or []:
                path = root / str(row["path"])
                raw = path.read_bytes()
                if _sha256_bytes(raw) != row["content_sha256"]:
                    raise RuntimeError(f"source CAS mismatch: {path}")
                original_bytes[path] = raw
            for row in plan.get("link_rewrites") or []:
                path = root / str(row["path"])
                raw = path.read_bytes()
                if _sha256_bytes(raw) != row["before_sha256"]:
                    raise RuntimeError(f"link rewrite CAS mismatch: {path}")
                original_bytes[path] = raw

            output = dict(plan["output"])
            output_path = root / str(output["path"])
            output_raw = str(output["content"]).encode("utf-8")
            atomic_write(output_path, output_raw.decode("utf-8"))
            owned_bytes[output_path] = output_raw
            for row in plan.get("link_rewrites") or []:
                path = root / str(row["path"])
                raw = str(row["content"]).encode("utf-8")
                atomic_write(path, raw.decode("utf-8"))
                owned_bytes[path] = raw
            canonical_uid = str(output["uid"])
            for row in plan.get("inputs") or []:
                if str(row["uid"]) == canonical_uid:
                    continue
                path = root / str(row["path"])
                path.unlink()
                owned_bytes[path] = None

            registry_result = registry.add_redirects(
                [dict(value) for value in plan.get("redirects") or []],
                expected_generation=int(plan["registry_generation"]),
            )
            registry_owned = registry.path.read_bytes()

            if output_path.read_bytes() != output_raw:
                raise RuntimeError("canonical output read-back mismatch")
            for row in plan.get("link_rewrites") or []:
                path = root / str(row["path"])
                if _sha256_bytes(path.read_bytes()) != row["after_sha256"]:
                    raise RuntimeError(f"link rewrite read-back mismatch: {path}")
            for row in plan.get("inputs") or []:
                if str(row["uid"]) == canonical_uid:
                    continue
                if (root / str(row["path"])).exists():
                    raise RuntimeError(
                        f"superseded source still exists: {row['path']}"
                    )
            for redirect in plan.get("redirects") or []:
                resolved = registry.resolve(str(redirect["from_uid"]))
                if (
                    resolved is None
                    or str(resolved.get("uid") or "") != canonical_uid
                ):
                    raise RuntimeError("redirect postflight resolution mismatch")
            postflight = verify_merge_coverage(
                inventory=dict(plan.get("inventory") or {}),
                mappings=list(plan.get("claim_map") or []),
                output_text=output_raw.decode("utf-8"),
                ledger_dispositions=list(plan.get("ledger_dispositions") or []),
                input_sensitivities=[
                    str(row.get("sensitivity") or "normal")
                    for row in plan.get("inputs") or []
                ],
                output_sensitivity=str(output.get("sensitivity") or "normal"),
                require_raw_refs=True,
            )
    except Exception as exc:
        rollback: dict[str, bool] = {}
        with chronovisor_mutation_lock():
            for path, original in original_bytes.items():
                current = path.read_bytes() if path.exists() else None
                owned = owned_bytes.get(path, current)
                if current != owned:
                    rollback[str(path)] = False
                    continue
                atomic_write(path, original.decode("utf-8"))
                rollback[str(path)] = path.read_bytes() == original
            if (
                registry_owned is not None
                and registry.path.read_bytes() == registry_owned
            ):
                atomic_write_bytes(registry.path, registry_preimage, backup=False)
                rollback[str(registry.path)] = (
                    registry.path.read_bytes() == registry_preimage
                )
                registry._append_event(
                    {
                        "event": "redirect_batch_rolled_back",
                        "transaction_id": transaction_id,
                        "restored": rollback[str(registry.path)],
                    }
                )
        ledger.append(
            {
                "transaction_id": transaction_id,
                "operation": "merge",
                "status": "rolled_back",
                "error": f"{type(exc).__name__}: {exc}",
                "rollback": rollback,
            }
        )
        return {
            "status": "rolled_back",
            "transaction_id": transaction_id,
            "error": f"{type(exc).__name__}: {exc}",
            "rollback": rollback,
            "preimage": str(preimage),
        }
    receipt = ledger.append(
        {
            "transaction_id": transaction_id,
            "operation": "merge",
            "status": "committed",
            "redirects": plan.get("redirects"),
            "link_rewrites": [
                {key: value for key, value in dict(row).items() if key != "content"}
                for row in plan.get("link_rewrites") or []
            ],
            "verification_receipt": plan.get("verification_receipt"),
            "postflight_receipt": postflight,
            "registry": registry_result,
            "temporary_preimage": str(preimage),
            "committed_at": _now().isoformat(timespec="milliseconds"),
        }
    )
    if preimage_ttl_days <= 0:
        shutil.rmtree(preimage)
    return {
        "status": "committed",
        "transaction_id": transaction_id,
        "registry": registry_result,
        "receipt": receipt,
        "preimage": str(preimage) if preimage.exists() else None,
    }


def cleanup_expired_preimages(
    root: Path,
    *,
    now: datetime | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Remove valid transaction preimages after TTL or verified release."""

    current = now or _now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    base = root / "runtime" / "librarian" / "transaction-preimages"
    deleted: list[str] = []
    retained: list[str] = []
    if not base.exists():
        return {"deleted": deleted, "retained": retained}
    for path in sorted(value for value in base.iterdir() if value.is_dir()):
        try:
            payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if payload.get("schema") != PREIMAGE_SCHEMA:
                raise ValueError("schema")
            expires = datetime.fromisoformat(str(payload["expires_at"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            retained.append(path.name)
            continue
        if force or expires <= current:
            shutil.rmtree(path)
            deleted.append(path.name)
        else:
            retained.append(path.name)
    return {"deleted": deleted, "retained": retained}
