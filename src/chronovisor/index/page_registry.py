"""Read-only resolution of existing page identities and redirects.

The registry is metadata only. Page bodies remain under ``pages/`` and
``system/``; redirect source bodies are not copied here. Read resolution is pure:
redirect following never rewrites or compresses the registry.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chronovisor.common import canonical_document
from chronovisor.index import index_store
from chronovisor.pages import store
from chronovisor.pages.page_identity import normalize_page_uid

REGISTRY_SCHEMA = "chronovisor.page-registry.v1"
EVENT_SCHEMA = "chronovisor.page-registry-event.v1"
MAX_REDIRECT_HOPS = 8


class PageRegistryError(RuntimeError):
    """Raised when the identity registry is malformed or unsafe to mutate."""


def _normalize_key(value: object) -> str:
    text = str(value or "").strip()
    text = text.removesuffix(".md")
    return text.casefold()


def _compose_anchor_maps(
    incoming: Mapping[str, str],
    downstream: Mapping[str, str],
) -> dict[str, str]:
    """Compose redirect anchor maps from source through the final target."""

    normalized_downstream = {
        str(source): str(target) for source, target in downstream.items()
    }
    if not incoming:
        return normalized_downstream
    return {
        **normalized_downstream,
        **{
            str(source): normalized_downstream.get(str(middle), str(middle))
            for source, middle in incoming.items()
        },
    }


class PageRegistry:
    """Reader for the existing UID and redirect registry."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runtime_dir = self.root / "runtime" / "librarian"
        self.path = self.runtime_dir / "page-registry.json"

    @staticmethod
    def empty() -> dict[str, Any]:
        return {
            "schema": REGISTRY_SCHEMA,
            "generation": 0,
            "updated_at": None,
            "pages": {},
            "keys": {},
            "ambiguous_keys": {},
            "redirects": {},
        }

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self.empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PageRegistryError(f"cannot read page registry: {exc}") from exc
        if not isinstance(data, dict) or data.get("schema") != REGISTRY_SCHEMA:
            raise PageRegistryError("unsupported page registry schema")
        data.setdefault("ambiguous_keys", {})
        for field in ("pages", "keys", "ambiguous_keys", "redirects"):
            if not isinstance(data.get(field), dict):
                raise PageRegistryError(f"registry field {field!r} must be an object")
        for uid, row in data["pages"].items():
            if not isinstance(row, dict) or row.get("uid") != uid:
                raise PageRegistryError(f"registry page {uid!r} is malformed")
            canonical_uid = row.get("canonical_uid")
            redirect = data["redirects"].get(uid)
            if canonical_uid is not None and (
                not isinstance(redirect, dict)
                or redirect.get("to_uid") != canonical_uid
            ):
                raise PageRegistryError(
                    f"registry redirect state is inconsistent for {uid!r}"
                )
        return data

    def _canonical_path(
        self,
        row: Mapping[str, Any],
        *,
        require_stable: bool,
    ) -> Path | None:
        raw_path = str(row.get("path") or "")
        if raw_path.startswith("pages/"):
            directory = self.root / "pages"
            namespace: canonical_document.Namespace = "pages"
            reserved = store.PAGE_RESERVED_FILENAMES
        elif raw_path.startswith("system/"):
            directory = self.root / "system"
            namespace = "system"
            reserved = store.SYSTEM_RESERVED_FILENAMES
        else:
            return None
        path = store.canonical_document_path(
            self.root / raw_path,
            directory,
            namespace=namespace,
            reserved_filenames=reserved,
            require_stable=require_stable,
        )
        if path is None:
            return None
        data = index_store.canonical_document_bytes(path, directory)
        if data is None:
            return None
        try:
            document_status = canonical_document.parse_document(data).metadata.get(
                "status"
            )
        except canonical_document.CanonicalDocumentError:
            return None
        return path if document_status == row.get("status") else None

    @staticmethod
    def resolve_from_state(
        state: Mapping[str, Any],
        key: object,
        *,
        max_hops: int = MAX_REDIRECT_HOPS,
    ) -> dict[str, Any] | None:
        """Resolve a key against one already validated registry snapshot."""

        normalized_key = _normalize_key(key)
        ambiguous = state.get("ambiguous_keys", {}).get(normalized_key)
        if isinstance(ambiguous, list) and ambiguous:
            raw_key = str(key or "").strip().casefold()
            exact = [
                uid
                for uid in ambiguous
                if str((state["pages"].get(uid) or {}).get("page_id") or "").casefold()
                == raw_key
            ]
            if len(exact) == 1:
                uid = exact[0]
            else:
                raise PageRegistryError(
                    f"ambiguous page key {key!r}; use UID or relative path"
                )
        else:
            uid = state["keys"].get(normalized_key)
        if uid is None:
            try:
                uid = normalize_page_uid(key)
            except ValueError:
                return None
        seen: list[str] = []
        anchor_map: dict[str, str] = {}
        for _hop in range(max_hops + 1):
            if uid in seen:
                raise PageRegistryError(f"redirect cycle: {' -> '.join([*seen, uid])}")
            seen.append(uid)
            redirect = state["redirects"].get(uid)
            if not isinstance(redirect, dict):
                row = state["pages"].get(uid)
                if not isinstance(row, dict):
                    return None
                return {
                    **row,
                    "requested": str(key),
                    "redirect_chain": seen[:-1],
                    "anchor_map": anchor_map,
                }
            target = normalize_page_uid(redirect.get("to_uid"))
            raw_anchor_map = redirect.get("anchor_map")
            if isinstance(raw_anchor_map, dict):
                anchor_map = _compose_anchor_maps(
                    anchor_map,
                    {
                        str(source): str(target_anchor)
                        for source, target_anchor in raw_anchor_map.items()
                    },
                )
            uid = target
        raise PageRegistryError(f"redirect hop limit exceeded for {key!r}")

    def resolve(
        self, key: object, *, max_hops: int = MAX_REDIRECT_HOPS
    ) -> dict[str, Any] | None:
        """Resolve a UID/slug/path through bounded redirects without writes."""

        return self.resolve_from_state(self.load(), key, max_hops=max_hops)

    def path_for(self, key: object, *, require_stable: bool = True) -> Path | None:
        resolved = self.resolve(key)
        status = resolved.get("status") if resolved is not None else None
        if (
            resolved is None
            or status not in canonical_document.PAGE_STATUSES
            or (require_stable and status != "stable")
            or (resolved.get("redirect_chain") and status != "stable")
            or resolved.get("canonical_uid")
        ):
            return None
        if (
            not require_stable
            and status != "stable"
            and str(key)
            not in {
                str(resolved.get("uid") or ""),
                str(resolved.get("page_id") or ""),
            }
        ):
            return None
        return self._canonical_path(resolved, require_stable=require_stable)
