"""CAS-bound split of an oversized page into a hub plus verbatim child pages.

Ingest's compact update path is append-only, so pages past the local model's
context budget only grow. A split keeps the parent UID as a small hub (its
frontmatter and preface plus links to the children) and moves every H1/H2
section verbatim into ``<stem>-part-NN.md`` children in the same collection
folder. Exact duplicate sections (identical bytes) are kept once. Anchor links
into moved headings are rewritten to the child that now holds the heading.

Like merges, a plan is prepared read-only and applied only on explicit
activation, with preimages, ledger receipts, CAS checks, and rollback.
"""

from __future__ import annotations

import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path, PurePath
from typing import Any
from urllib.parse import unquote

from chronovisor.core import frontmatter
from chronovisor.core.canonical_document import (
    CanonicalDocument,
    CanonicalDocumentError,
    ResolvedMarkdownLink,
    rewrite_internal_markdown_links,
    serialize_document,
    validate_canonical_document,
)
from chronovisor.core.durable_state import atomic_write_bytes
from chronovisor.core.hashutil import sha256_bytes as _sha256_bytes
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.markdown_sections import (
    MARKDOWN_FENCE_RE,
    MarkdownSection,
    markdown_sections,
    split_children,
)
from chronovisor.core.page_mutation import chronovisor_mutation_lock
from chronovisor.core.timeutil import utc_now as _now
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.recall.merge_ledger import MergeLedger
from chronovisor.recall.merge_transaction import _write_preimage

PLAN_SCHEMA = "chronovisor.split-plan.v1"
CHILD_TARGET_BYTES = 30_000
HUB_HEADING = "## 分割先"
_HEADING_RE = re.compile(r"^ {0,3}#{1,6}[\t ]+(?P<title>.+?)(?:[\t ]+#+)?[\t ]*$")
_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?(?:\s|$)")
_SAME_DOC_LINK_RE = re.compile(r"\]\(<?#([^)\s>]+)>?\)")


class SplitPlanError(ValueError):
    """A split plan failed a deterministic gate."""


def unique_sections(
    text: str,
) -> tuple[MarkdownSection | None, list[MarkdownSection], int]:
    """Return (preface, first occurrence of each headed section, duplicates)."""

    sections = markdown_sections(text)
    preface = sections[0] if sections and sections[0].heading is None else None
    seen: set[str] = set()
    unique: list[MarkdownSection] = []
    duplicates = 0
    for section in sections:
        if section.heading is None:
            continue
        if section.sha256 in seen:
            duplicates += 1
            continue
        seen.add(section.sha256)
        unique.append(section)
    return preface, unique, duplicates


def group_sections(
    sections: Sequence[MarkdownSection],
    target_bytes: int = CHILD_TARGET_BYTES,
) -> list[list[int]]:
    """Greedily pack consecutive sections into children of ~target bytes."""

    groups: list[list[int]] = []
    size = 0
    for index, section in enumerate(sections):
        section_bytes = len(section.content.encode("utf-8"))
        if groups and size + section_bytes <= target_bytes:
            groups[-1].append(index)
            size += section_bytes
        else:
            groups.append([index])
            size = section_bytes
    return groups


def _anchor_keys(title: str) -> set[str]:
    title = title.strip()
    folded = title.casefold()
    keys = {
        folded.replace(" ", "-"),
        re.sub(r"[^\w\- ]", "", folded).replace(" ", "-"),  # GitHub-style slug
    }
    number = _NUMBER_RE.match(title)
    if number:
        keys.add(number.group(1))
    return keys


def _heading_anchor_keys(content: str) -> set[str]:
    keys: set[str] = set()
    fence: str | None = None
    for line in content.splitlines():
        marker = MARKDOWN_FENCE_RE.match(line)
        if marker:
            token = marker.group("marker")
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None and (match := _HEADING_RE.match(line)):
            keys |= _anchor_keys(match.group("title"))
    return keys


def _anchor_target(anchor_targets: Mapping[str, str], fragment: str) -> str | None:
    for key in (fragment, unquote(fragment)):
        target = anchor_targets.get(key) or anchor_targets.get(key.casefold())
        if target is not None:
            return target
    return None


def _rewrite_same_doc_anchors(
    text: str, *, own_path: str, anchor_targets: Mapping[str, str]
) -> str:
    """Point bare ``#frag`` links at the sibling that now holds the heading."""

    def replace(match: re.Match[str]) -> str:
        target = _anchor_target(anchor_targets, match.group(1))
        if target is None or target == own_path:
            return match.group(0)
        return f"](<{PurePath(target).name}#{match.group(1)}>)"

    return _SAME_DOC_LINK_RE.sub(replace, text)


def _with_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _rewrite_anchor_links(
    text: str,
    *,
    source_path: str,
    parent_path: str,
    anchor_targets: Mapping[str, str],
) -> tuple[str, int]:
    """Point ``parent#anchor`` links at the child now holding that heading."""

    def rewrite(link: ResolvedMarkdownLink, _label: str) -> ResolvedMarkdownLink | None:
        if link.namespace != "pages" or link.path != parent_path or not link.fragment:
            return None
        target = _anchor_target(anchor_targets, link.fragment)
        if target is None:
            return None
        return ResolvedMarkdownLink("pages", target, link.fragment)

    return rewrite_internal_markdown_links(
        text,
        source_namespace="pages",
        source_path=source_path,
        rewrite=rewrite,
        on_invalid=lambda _target, _label, _exc: None,
    )


def prepare_split_plan(
    root: Path,
    *,
    page_key: str,
    groups: Sequence[Mapping[str, Any]] | None = None,
    target_bytes: int = CHILD_TARGET_BYTES,
) -> dict[str, Any]:
    """Build an exact, read-only split plan and run the deterministic gates.

    ``groups`` is ``[{"title": str | None, "sections": [unique section index]}]``;
    when omitted, consecutive sections are packed to ``target_bytes``.
    """

    pages_dir = root / "pages"
    registry = PageRegistry(root)
    state = registry.load()
    resolved = registry.resolve_from_state(state, page_key)
    row = registry.stable_page(str(resolved["uid"]), state) if resolved else None
    if row is None:
        raise KeyError(page_key)
    parent_uid = str(row["uid"])
    path = root / str(row["path"])
    if pages_dir not in path.parents:
        raise SplitPlanError("only pages/ documents can be split")
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if split_children(text):
        raise SplitPlanError(f"{page_key} is already a split hub")
    meta, _body = frontmatter.parse(text)
    preface, sections, duplicates = unique_sections(text)
    # A single section larger than a child is broken at its H3 boundaries.
    sections = [
        piece
        for section in sections
        for piece in (
            markdown_sections(section.content, max_level=3)
            if len(section.content.encode("utf-8")) > target_bytes
            else (section,)
        )
    ]
    if preface is None or not meta:
        raise SplitPlanError("page lacks frontmatter to keep on the hub")
    # A child's leading "# title" line is synthetic; peeling it off alone would
    # nest a useless hub, so it does not count as a section of its own.
    real = [s for s in sections if s.content.strip() != (s.heading or "").strip()]
    if len(real) < 2:
        raise SplitPlanError(
            f"{page_key} is a split child without two real sections"
            if meta.get("split_from")
            else "page has fewer than two unique sections"
        )

    if groups is None:
        groups = [
            {"title": None, "sections": value}
            for value in group_sections(sections, target_bytes)
        ]
    assigned = sorted(
        index for group in groups for index in group.get("sections") or []
    )
    if assigned != list(range(len(sections))) or any(
        not group.get("sections") for group in groups
    ):
        raise SplitPlanError("groups must assign every unique section exactly once")
    if len(groups) < 2:
        raise SplitPlanError("a split needs at least two children")

    parent_rel = path.relative_to(pages_dir).as_posix()
    parent_title = str(meta.get("title") or path.stem)
    today = date.today().isoformat()
    width = max(2, len(str(len(groups))))
    children: list[dict[str, Any]] = []
    anchor_targets: dict[str, str] = {}
    for number, group in enumerate(groups, start=1):
        child_path = path.with_name(f"{path.stem}-part-{number:0{width}d}.md")
        if child_path.exists():
            raise SplitPlanError(f"child path already exists: {child_path.name}")
        indices = sorted(group["sections"])
        members = [sections[index] for index in indices]
        first_heading = (members[0].heading or "").lstrip("#").strip()
        title = (
            str(group.get("title") or "").strip()
            or (f"{parent_title} ({number}/{len(groups)}): {first_heading}"[:160])
        )
        child_rel = child_path.relative_to(pages_dir).as_posix()
        for member in members:
            for key in _heading_anchor_keys(member.content):
                anchor_targets.setdefault(key, child_rel)
        child_meta: dict[str, Any] = {
            "title": title,
            "updated": today,
            "type": meta.get("type"),
            "status": "stable",
            "description": f"Part {number} of {len(groups)} split from {parent_title}.",
            "split_from": parent_uid,
        }
        for key in ("tags", "sensitivity", "entities"):
            if meta.get(key) is not None:
                child_meta[key] = meta[key]
        body = f"# {title}\n\n" + "".join(_with_newline(m.content) for m in members)
        children.append(
            {
                "path": str(child_path.relative_to(root)),
                "pages_path": child_rel,
                "title": title,
                "metadata": child_meta,
                "body": body,
                "section_sha256s": [member.sha256 for member in members],
            }
        )

    # Coverage gate on the verbatim assembly, before any link retargeting.
    covered = [sha for child in children for sha in child["section_sha256s"]]
    if sorted(covered) != sorted(section.sha256 for section in sections):
        raise SplitPlanError("section coverage mismatch")
    for child in children:
        for sha in child["section_sha256s"]:
            content = next(s.content for s in sections if s.sha256 == sha)
            if _with_newline(content) not in child["body"]:
                raise SplitPlanError("child body lost verbatim section bytes")

    same_doc_anchors = 0
    for child in children:
        body, _changed = _rewrite_anchor_links(
            child["body"],
            source_path=child["pages_path"],
            parent_path=parent_rel,
            anchor_targets=anchor_targets,
        )
        body = _rewrite_same_doc_anchors(
            body, own_path=child["pages_path"], anchor_targets=anchor_targets
        )
        same_doc_anchors += sum(
            _anchor_target(anchor_targets, frag) is None
            for frag in _SAME_DOC_LINK_RE.findall(body)
        )
        content = serialize_document(
            CanonicalDocument(metadata=child.pop("metadata"), body=body.encode("utf-8"))
        ).decode("utf-8")
        child.pop("body")
        child["content"] = content
        child["content_sha256"] = _sha256_bytes(content.encode("utf-8"))

    links = "".join(
        f"- [{child['title']}]({PurePath(child['pages_path']).name})\n"
        for child in children
    )
    preface_text, _changed = _rewrite_anchor_links(
        _rewrite_same_doc_anchors(
            preface.content, own_path=parent_rel, anchor_targets=anchor_targets
        ),
        source_path=parent_rel,
        parent_path=parent_rel,
        anchor_targets=anchor_targets,
    )
    hub_head = frontmatter.patch(
        _with_newline(preface_text),
        {
            "split_children": [PurePath(c["pages_path"]).name for c in children],
            "updated": today,
        },
    )
    hub = f"{_with_newline(hub_head)}\n{HUB_HEADING}\n\n{links}"

    link_rewrites = []
    for other in sorted(pages_dir.rglob("*.md")):
        if other == path:
            continue
        other_raw = other.read_bytes()
        if parent_rel.rsplit("/", 1)[-1].removesuffix(".md").encode() not in other_raw:
            continue
        other_rel = other.relative_to(pages_dir).as_posix()
        try:
            rewritten, changed = _rewrite_anchor_links(
                other_raw.decode("utf-8"),
                source_path=other_rel,
                parent_path=parent_rel,
                anchor_targets=anchor_targets,
            )
        except CanonicalDocumentError, UnicodeDecodeError:
            continue
        if changed:
            link_rewrites.append(
                {
                    "path": str(other.relative_to(root)),
                    "before_sha256": _sha256_bytes(other_raw),
                    "after_sha256": _sha256_bytes(rewritten.encode("utf-8")),
                    "content": rewritten,
                }
            )

    return {
        "schema": PLAN_SCHEMA,
        "transaction_id": "split_" + uuid.uuid4().hex,
        "status": "prepared",
        "registry_generation": int(state.get("generation") or 0),
        "inputs": [
            {
                "uid": parent_uid,
                "path": str(path.relative_to(root)),
                "content_sha256": _sha256_bytes(raw),
                "sensitivity": str(
                    row.get("sensitivity") or meta.get("sensitivity") or "normal"
                ),
            }
        ],
        "output": {
            "uid": parent_uid,
            "path": str(path.relative_to(root)),
            "content": hub,
            "content_sha256": _sha256_bytes(hub.encode("utf-8")),
        },
        "children": children,
        "link_rewrites": link_rewrites,
        "verification_receipt": {
            "input_bytes": len(raw),
            "hub_bytes": len(hub.encode("utf-8")),
            "child_bytes": [len(c["content"].encode("utf-8")) for c in children],
            "unique_sections": len(sections),
            "duplicate_sections_removed": duplicates,
            "anchor_keys": len(anchor_targets),
            "unrewritten_same_doc_anchors": same_doc_anchors,
        },
        "prepared_at": _now().isoformat(timespec="milliseconds"),
    }


def _validate_written(root: Path, relative: str) -> None:
    path = root / relative
    validate_canonical_document(
        path.read_bytes(),
        namespace="pages",
        path=path.relative_to(root / "pages").as_posix(),
        require_stable=True,
    )


def apply_split_plan(
    root: Path,
    plan: Mapping[str, Any],
    *,
    activate: bool = False,
    preimage_ttl_days: int = 0,
) -> dict[str, Any]:
    """Apply a prepared split plan only after explicit activation."""

    if not activate:
        return {
            "status": "blocked",
            "reason": "explicit_activation_required",
            "transaction_id": plan.get("transaction_id"),
        }
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "prepared":
        raise ValueError("unsupported or non-prepared split plan")
    transaction_id = str(plan["transaction_id"])
    registry = PageRegistry(root)
    ledger = MergeLedger(root)
    preimage = _write_preimage(root, plan, preimage_ttl_days)
    ledger.append(
        {
            "transaction_id": transaction_id,
            "operation": "split",
            "status": "pending",
            "inputs": plan.get("inputs"),
            "children": [
                {key: value for key, value in child.items() if key != "content"}
                for child in plan.get("children") or []
            ],
            "verification_receipt": plan.get("verification_receipt"),
            "temporary_preimage": str(preimage),
        }
    )
    original_bytes: dict[Path, bytes] = {}
    owned_bytes: dict[Path, bytes] = {}
    created: list[Path] = []
    # The whole apply, including rollback, runs under one lock hold so nothing
    # can land between a failure and its compensation.
    with chronovisor_mutation_lock(pages_dir=root / "pages"):
        registry_preimage = (
            registry.path.read_bytes() if registry.path.exists() else None
        )
        try:
            receipt = _apply_locked(
                root,
                plan,
                registry,
                ledger,
                preimage,
                original_bytes,
                owned_bytes,
                created,
            )
        except Exception as exc:
            rollback = _rollback_locked(
                registry, registry_preimage, original_bytes, owned_bytes, created
            )
            ledger.append(
                {
                    "transaction_id": transaction_id,
                    "operation": "split",
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
    if preimage_ttl_days <= 0:
        shutil.rmtree(preimage, ignore_errors=True)  # a GC may beat us to it
    return {
        "status": "committed",
        "transaction_id": transaction_id,
        "receipt": receipt,
        "preimage": str(preimage) if preimage.exists() else None,
    }


def _apply_locked(
    root: Path,
    plan: Mapping[str, Any],
    registry: PageRegistry,
    ledger: MergeLedger,
    preimage: Path,
    original_bytes: dict[Path, bytes],
    owned_bytes: dict[Path, bytes],
    created: list[Path],
) -> dict[str, Any]:
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
    for child in plan.get("children") or []:
        if (root / str(child["path"])).exists():
            raise RuntimeError(f"child path appeared: {child['path']}")

    writes = [(root / str(c["path"]), str(c["content"])) for c in plan["children"]]
    for path, content in writes:
        created.append(path)  # before the write: a late fsync error still owns it
        owned_bytes[path] = content.encode("utf-8")
        atomic_write(path, content)
    output = dict(plan["output"])
    hub_path = root / str(output["path"])
    writes = [(hub_path, str(output["content"]))] + [
        (root / str(row["path"]), str(row["content"]))
        for row in plan.get("link_rewrites") or []
    ]
    for path, content in writes:
        owned_bytes[path] = content.encode("utf-8")
        atomic_write(path, content)

    registry_result = registry.ensure_manifest(write=True)

    if _sha256_bytes(hub_path.read_bytes()) != output["content_sha256"]:
        raise RuntimeError("hub read-back mismatch")
    _validate_written(root, str(output["path"]))
    for child in plan.get("children") or []:
        path = root / str(child["path"])
        if _sha256_bytes(path.read_bytes()) != child["content_sha256"]:
            raise RuntimeError(f"child read-back mismatch: {child['path']}")
        _validate_written(root, str(child["path"]))
        resolved = registry.resolve(path.stem)
        if (
            resolved is None
            or resolved.get("status") != "stable"
            or resolved.get("path") != str(child["path"])
        ):
            raise RuntimeError(f"child not registered as itself: {child['path']}")
    for row in plan.get("link_rewrites") or []:
        if _sha256_bytes((root / str(row["path"])).read_bytes()) != row["after_sha256"]:
            raise RuntimeError(f"link rewrite read-back mismatch: {row['path']}")
    parent = registry.resolve(str(output["uid"]))
    if parent is None or parent.get("path") != str(output["path"]):
        raise RuntimeError("hub lost its registry identity")
    return ledger.append(
        {
            "transaction_id": str(plan["transaction_id"]),
            "operation": "split",
            "status": "committed",
            "children": [str(child["path"]) for child in plan["children"]],
            "link_rewrites": [
                str(row["path"]) for row in plan.get("link_rewrites") or []
            ],
            "verification_receipt": plan.get("verification_receipt"),
            "registry_generation": int(
                (registry_result.get("registry") or {}).get("generation") or 0
            ),
            "temporary_preimage": str(preimage),
            "committed_at": _now().isoformat(timespec="milliseconds"),
        }
    )


def _rollback_locked(
    registry: PageRegistry,
    registry_preimage: bytes | None,
    original_bytes: Mapping[Path, bytes],
    owned_bytes: Mapping[Path, bytes],
    created: Sequence[Path],
) -> dict[str, bool]:
    """Undo only bytes this transaction wrote; the caller still holds the lock."""

    rollback: dict[str, bool] = {}
    for path in created:
        if path.exists() and path.read_bytes() == owned_bytes.get(path):
            path.unlink()
        rollback[str(path)] = not path.exists()
    for path, original in original_bytes.items():
        if path not in owned_bytes:
            continue  # never written by us
        atomic_write(path, original.decode("utf-8"))
        rollback[str(path)] = path.read_bytes() == original
    current = registry.path.read_bytes() if registry.path.exists() else None
    if current != registry_preimage:
        if registry_preimage is None:
            registry.path.unlink()
        else:
            atomic_write_bytes(registry.path, registry_preimage, backup=False)
        rollback[str(registry.path)] = (
            registry.path.read_bytes() if registry.path.exists() else None
        ) == registry_preimage
    return rollback
