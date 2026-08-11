"""Durable page identity, redirect, and classification registry.

The registry is metadata only. Page bodies remain under ``pages/`` and
``system/``; redirect source bodies are not copied here. Read resolution is pure:
redirect following never rewrites or compresses the registry.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core import canonical_document, frontmatter, index_store
from chronovisor.core.hashutil import sha256_bytes as _sha256
from chronovisor.core.page_identity import new_page_uid, normalize_page_uid

REGISTRY_SCHEMA = "chronovisor.page-registry.v1"
EVENT_SCHEMA = "chronovisor.page-registry-event.v1"
MAX_REDIRECT_HOPS = 8


class PageRegistryError(RuntimeError):
    """Raised when the identity registry is malformed or unsafe to mutate."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")




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


def _legacy_aliases(root: Path) -> dict[str, str]:
    path = root / "runtime" / "page-aliases.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PageRegistryError(f"cannot read legacy aliases: {exc}") from exc
    rows = payload.get("aliases") if isinstance(payload, dict) else None
    if not isinstance(rows, dict):
        raise PageRegistryError("legacy alias file has no aliases object")
    aliases: dict[str, str] = {}
    for key, value in rows.items():
        target = value.get("target") if isinstance(value, dict) else value
        if not isinstance(target, str) or not target.strip():
            raise PageRegistryError(f"legacy alias {key!r} has no target")
        aliases[str(key)] = target.removesuffix(".md")
    return aliases


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    finally:
        tmp.unlink(missing_ok=True)


class PageRegistry:
    """Atomic UID and redirect registry rooted in one Chronovisor store."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runtime_dir = self.root / "runtime" / "librarian"
        self.path = self.runtime_dir / "page-registry.json"
        self.events_path = self.runtime_dir / "page-registry-events.jsonl"
        self.lock_path = self.runtime_dir / "page-registry.lock"

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            os.chmod(self.lock_path, 0o600)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

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

    def stable_pages(
        self,
        state: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return current stable, non-redirect canonical page rows."""

        snapshot = state if state is not None else self.load()
        redirects = snapshot.get("redirects")
        redirect_uids = set(redirects) if isinstance(redirects, Mapping) else set()
        return {
            str(uid): dict(row)
            for uid, row in snapshot["pages"].items()
            if isinstance(row, Mapping)
            and row.get("status") == "stable"
            and not row.get("canonical_uid")
            and uid not in redirect_uids
            and self._canonical_path(row, require_stable=True) is not None
        }

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
            reserved = index_store.PAGE_RESERVED_FILENAMES
        elif raw_path.startswith("system/"):
            directory = self.root / "system"
            namespace = "system"
            reserved = index_store.SYSTEM_RESERVED_FILENAMES
        else:
            return None
        path = index_store.canonical_document_path(
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

    def _append_event(self, event: Mapping[str, Any]) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": EVENT_SCHEMA,
            "timestamp": _now_iso(),
            **dict(event),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            os.chmod(self.events_path, 0o600)
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _page_paths(root: Path, *, include_system: bool) -> list[Path]:
        try:
            return index_store.canonical_document_paths(
                root / "pages",
                system_dir=root / "system" if include_system else None,
                strict=True,
            )
        except index_store.CanonicalDocumentError as exc:
            raise PageRegistryError(str(exc)) from exc

    def ensure_manifest(
        self,
        *,
        include_system: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        """Assign one persistent UUIDv7 to every current Markdown page."""

        with self._lock() if write else nullcontext():
            state = self.load()
            pages = dict(state["pages"])
            previous_by_path = {
                str(row.get("path")): uid
                for uid, row in pages.items()
                if isinstance(row, dict) and row.get("path")
            }
            previous_page_id_candidates: dict[str, list[str]] = {}
            for uid, row in pages.items():
                if isinstance(row, dict) and row.get("page_id"):
                    previous_page_id_candidates.setdefault(
                        str(row["page_id"]), []
                    ).append(uid)
            previous_by_page_id = {
                page_id: uids[0]
                for page_id, uids in previous_page_id_candidates.items()
                if len(uids) == 1
            }
            seen_uids: set[str] = set()
            created: list[str] = []
            updated: list[str] = []
            legacy_aliases = _legacy_aliases(self.root)

            for uid, redirect in state["redirects"].items():
                row = pages.get(uid)
                if not isinstance(row, dict) or not isinstance(redirect, dict):
                    raise PageRegistryError(f"registry redirect {uid!r} is malformed")
                canonical_uid = normalize_page_uid(redirect.get("to_uid"))
                if (
                    row.get("status") not in canonical_document.PAGE_STATUSES
                    or row.get("canonical_uid") != canonical_uid
                ):
                    pages[uid] = {
                        **row,
                        "status": "deprecated",
                        "canonical_uid": canonical_uid,
                        "updated_at": _now_iso(),
                    }
                    updated.append(uid)

            for path in self._page_paths(self.root, include_system=include_system):
                raw = path.read_bytes()
                stat = path.stat()
                text = raw.decode("utf-8")
                meta, _body = frontmatter.parse(text)
                page_id = path.stem
                rel = str(path.relative_to(self.root))
                candidate = meta.get("uid")
                uid: str | None = None
                if candidate:
                    try:
                        uid = normalize_page_uid(candidate)
                    except ValueError as exc:
                        raise PageRegistryError(
                            f"invalid UID in {rel}: {candidate!r}"
                        ) from exc
                uid = uid or previous_by_path.get(rel)
                rename_uid = previous_by_page_id.get(page_id)
                if uid is None and rename_uid:
                    previous_row = pages.get(rename_uid)
                    previous_path = (
                        self.root / str(previous_row.get("path") or "")
                        if isinstance(previous_row, dict)
                        else None
                    )
                    if previous_path is not None and not previous_path.exists():
                        uid = rename_uid
                uid = normalize_page_uid(uid) if uid else new_page_uid()
                if uid in seen_uids:
                    raise PageRegistryError(f"duplicate page UID {uid}")
                seen_uids.add(uid)

                aliases = meta.get("aliases")
                legacy_keys = (
                    [str(item) for item in aliases if isinstance(item, str)]
                    if isinstance(aliases, list)
                    else []
                )
                relative_without_suffix = str(
                    path.relative_to(self.root).with_suffix("")
                )
                pages_relative = (
                    str(path.relative_to(self.root / "pages").with_suffix(""))
                    if self.root / "pages" in path.parents
                    else relative_without_suffix
                )
                legacy_keys.extend(
                    alias
                    for alias, target in legacy_aliases.items()
                    if target
                    in {
                        page_id,
                        relative_without_suffix,
                        pages_relative,
                    }
                )
                previous = pages.get(uid)
                previous_updated_at = (
                    previous.get("updated_at") if isinstance(previous, dict) else None
                )
                row = {
                    "uid": uid,
                    "page_id": page_id,
                    "path": rel,
                    "legacy_keys": sorted({page_id, *legacy_keys}),
                    "status": str(meta["status"]),
                    "canonical_uid": None,
                    "content_sha256": _sha256(raw),
                    "content_size": len(raw),
                    "content_mtime_ns": stat.st_mtime_ns,
                    "classification_status": (
                        "classified"
                        if meta.get("classification_schema")
                        else "unclassified"
                    ),
                    "classification": None,
                    "collection_uid": None,
                    "collection_status": "unclassified",
                    "collection_generation": None,
                    "sensitivity": str(meta.get("sensitivity") or "normal"),
                    "updated_at": previous_updated_at or _now_iso(),
                }
                if previous is None:
                    created.append(uid)
                elif isinstance(previous, dict):
                    row["classification"] = previous.get("classification")
                    if row["classification"] is not None:
                        row["classification_status"] = str(
                            previous.get("classification_status") or "proposed"
                        )
                    row["collection_uid"] = previous.get("collection_uid")
                    row["collection_status"] = str(
                        previous.get("collection_status") or "unclassified"
                    )
                    row["collection_generation"] = previous.get(
                        "collection_generation"
                    )
                    row["canonical_uid"] = previous.get("canonical_uid")
                    if previous != row:
                        row["updated_at"] = _now_iso()
                        updated.append(uid)
                pages[uid] = row

            # Historical entries remain metadata-only. Missing pages are never
            # silently deleted.
            keys: dict[str, str] = {}
            ambiguous_keys: dict[str, list[str]] = {}
            for uid, row in pages.items():
                if not isinstance(row, dict):
                    raise PageRegistryError(f"registry page {uid!r} is malformed")
                for key in [
                    uid,
                    row.get("page_id"),
                    row.get("path"),
                    *(row.get("legacy_keys") or []),
                ]:
                    normalized = _normalize_key(key)
                    if not normalized:
                        continue
                    if normalized in ambiguous_keys:
                        ambiguous_keys[normalized] = sorted(
                            {*ambiguous_keys[normalized], uid}
                        )
                        continue
                    existing = keys.get(normalized)
                    if existing and existing != uid:
                        keys.pop(normalized, None)
                        ambiguous_keys[normalized] = sorted({existing, uid})
                        continue
                    keys[normalized] = uid

            next_state = {
                **state,
                "generation": int(state.get("generation") or 0)
                + int(bool(created or updated)),
                "updated_at": _now_iso(),
                "pages": pages,
                "keys": keys,
                "ambiguous_keys": ambiguous_keys,
            }
            if write and (created or updated or not self.path.exists()):
                _atomic_json(self.path, next_state)
                self._append_event(
                    {
                        "event": "manifest_refreshed",
                        "generation": next_state["generation"],
                        "created": len(created),
                        "updated": len(updated),
                        "observed": len(seen_uids),
                    }
                )
            return {
                "status": "ok",
                "write": write,
                "generation": next_state["generation"],
                "observed": len(seen_uids),
                "created": len(created),
                "updated": len(updated),
                "registry": next_state,
            }

    def apply_page_updates(
        self,
        updates: Mapping[str, Mapping[str, Any]],
        *,
        expected_generation: int | None = None,
        event: str = "page_metadata_batch_updated",
    ) -> dict[str, Any]:
        """Apply a metadata-only batch under one registry generation CAS."""

        normalized = {
            normalize_page_uid(uid): dict(value) for uid, value in updates.items()
        }
        if not normalized:
            state = self.load()
            return {
                "status": "unchanged",
                "generation": int(state.get("generation") or 0),
                "updated": 0,
            }
        with self._lock():
            state = self.load()
            generation = int(state.get("generation") or 0)
            if expected_generation is not None and generation != expected_generation:
                raise PageRegistryError(
                    f"registry generation changed: {generation} != {expected_generation}"
                )
            missing = sorted(uid for uid in normalized if uid not in state["pages"])
            if missing:
                raise PageRegistryError(
                    f"registry batch references missing UIDs: {', '.join(missing[:5])}"
                )
            changed = 0
            for uid, patch in normalized.items():
                row = dict(state["pages"][uid])
                candidate = {**row, **patch}
                if candidate == row:
                    continue
                candidate["updated_at"] = _now_iso()
                state["pages"][uid] = candidate
                changed += 1
            if changed:
                state["generation"] = generation + 1
                state["updated_at"] = _now_iso()
                _atomic_json(self.path, state)
                self._append_event(
                    {
                        "event": event,
                        "generation": state["generation"],
                        "updated": changed,
                    }
                )
            return {
                "status": "updated" if changed else "unchanged",
                "generation": int(state.get("generation") or generation),
                "updated": changed,
            }

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
                if str((state["pages"].get(uid) or {}).get("page_id") or "")
                .casefold()
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
        if not require_stable and status != "stable" and str(key) not in {
            str(resolved.get("uid") or ""),
            str(resolved.get("page_id") or ""),
        }:
            return None
        return self._canonical_path(resolved, require_stable=require_stable)

    def update_page(
        self,
        uid: object,
        updates: Mapping[str, Any],
        *,
        expected_generation: int | None = None,
        event: str = "page_metadata_updated",
    ) -> dict[str, Any]:
        normalized_uid = normalize_page_uid(uid)
        with self._lock():
            state = self.load()
            generation = int(state.get("generation") or 0)
            if expected_generation is not None and generation != expected_generation:
                raise PageRegistryError(
                    f"registry generation changed: {generation} != {expected_generation}"
                )
            row = state["pages"].get(normalized_uid)
            if not isinstance(row, dict):
                raise KeyError(normalized_uid)
            state["pages"][normalized_uid] = {
                **row,
                **dict(updates),
                "updated_at": _now_iso(),
            }
            state["generation"] = generation + 1
            state["updated_at"] = _now_iso()
            _atomic_json(self.path, state)
            self._append_event(
                {
                    "event": event,
                    "uid": normalized_uid,
                    "generation": state["generation"],
                }
            )
            return dict(state["pages"][normalized_uid])

    def add_redirect(
        self,
        from_uid: object,
        to_uid: object,
        *,
        anchor_map: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        source = normalize_page_uid(from_uid)
        target = normalize_page_uid(to_uid)
        if source == target:
            raise PageRegistryError("self redirect is forbidden")
        with self._lock():
            state = self.load()
            if source not in state["pages"] or target not in state["pages"]:
                raise PageRegistryError("redirect endpoint is not registered")

            # Eagerly flatten the target.  Read resolution itself remains pure.
            current = target
            seen = {source}
            combined = dict(anchor_map or {})
            for _hop in range(MAX_REDIRECT_HOPS):
                if current in seen:
                    raise PageRegistryError("redirect would create a cycle")
                seen.add(current)
                redirect = state["redirects"].get(current)
                if not isinstance(redirect, dict):
                    break
                raw_map = redirect.get("anchor_map")
                if isinstance(raw_map, dict):
                    combined = _compose_anchor_maps(
                        combined,
                        {str(key): str(value) for key, value in raw_map.items()},
                    )
                current = normalize_page_uid(redirect.get("to_uid"))
            else:
                raise PageRegistryError("redirect chain exceeds hop limit")

            state["redirects"][source] = {
                "to_uid": current,
                "anchor_map": combined,
                "created_at": _now_iso(),
            }
            source_row = dict(state["pages"][source])
            source_row["canonical_uid"] = current
            source_row["updated_at"] = _now_iso()
            state["pages"][source] = source_row
            state["generation"] = int(state.get("generation") or 0) + 1
            state["updated_at"] = _now_iso()
            _atomic_json(self.path, state)
            self._append_event(
                {
                    "event": "redirect_added",
                    "from_uid": source,
                    "to_uid": current,
                    "generation": state["generation"],
                }
            )
            return dict(state["redirects"][source])

    def add_redirects(
        self,
        redirects: list[Mapping[str, Any]],
        *,
        expected_generation: int | None = None,
    ) -> dict[str, Any]:
        """Atomically add and flatten a set of redirects."""

        normalized: list[tuple[str, str, dict[str, str]]] = []
        for row in redirects:
            source = normalize_page_uid(row.get("from_uid"))
            target = normalize_page_uid(row.get("to_uid"))
            if source == target:
                raise PageRegistryError("self redirect is forbidden")
            raw_anchor_map = row.get("anchor_map")
            anchor_map = (
                {str(key): str(value) for key, value in raw_anchor_map.items()}
                if isinstance(raw_anchor_map, Mapping)
                else {}
            )
            normalized.append((source, target, anchor_map))
        if len({source for source, _target, _map in normalized}) != len(normalized):
            raise PageRegistryError("duplicate redirect source")
        with self._lock():
            state = self.load()
            generation = int(state.get("generation") or 0)
            if expected_generation is not None and generation != expected_generation:
                raise PageRegistryError(
                    f"registry generation changed: {generation} != {expected_generation}"
                )
            for source, target, anchor_map in normalized:
                if source not in state["pages"] or target not in state["pages"]:
                    raise PageRegistryError("redirect endpoint is not registered")
                current = target
                seen = {source}
                combined = dict(anchor_map)
                for _hop in range(MAX_REDIRECT_HOPS):
                    if current in seen:
                        raise PageRegistryError("redirect would create a cycle")
                    seen.add(current)
                    redirect = state["redirects"].get(current)
                    if not isinstance(redirect, dict):
                        break
                    raw_map = redirect.get("anchor_map")
                    if isinstance(raw_map, dict):
                        combined = _compose_anchor_maps(
                            combined,
                            {str(key): str(value) for key, value in raw_map.items()},
                        )
                    current = normalize_page_uid(redirect.get("to_uid"))
                else:
                    raise PageRegistryError("redirect chain exceeds hop limit")
                state["redirects"][source] = {
                    "to_uid": current,
                    "anchor_map": combined,
                    "created_at": _now_iso(),
                }
                source_row = dict(state["pages"][source])
                source_row["canonical_uid"] = current
                source_row["updated_at"] = _now_iso()
                state["pages"][source] = source_row
            if normalized:
                state["generation"] = generation + 1
                state["updated_at"] = _now_iso()
                _atomic_json(self.path, state)
                self._append_event(
                    {
                        "event": "redirect_batch_added",
                        "count": len(normalized),
                        "generation": state["generation"],
                    }
                )
            return {
                "status": "updated" if normalized else "unchanged",
                "count": len(normalized),
                "generation": int(state.get("generation") or generation),
            }
