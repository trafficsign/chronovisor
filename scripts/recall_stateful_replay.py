"""Prepare isolated as-of Recall on/off/restore diagnostics.

Byte bindings establish reproducibility, not the truth of a label.  These
diagnostics cannot authorize promotion; formal R5/R6/R7 evidence is separate.
No live corpus or live session is used to fill a missing historical snapshot.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import canonical_sha256, seal_object
from chronovisor.recall.recall_answer_eval import _source_span_item_error

SCHEMA = "chronovisor.recall-stateful-diagnostic.v1"
ARMS = ("on", "off", "restore")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("historical timestamp missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("historical timestamp must include a timezone")
    return parsed


def _archived_bytes(root: Path, relative: Any) -> bytes:
    if not isinstance(relative, str) or not relative:
        raise ValueError("archive relative path missing")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("archive path escapes snapshot")
    selected = root / path
    if any(
        parent.is_symlink()
        for parent in (selected, *selected.parents)
        if parent != root.parent
    ):
        raise ValueError("archive symlink is forbidden")
    resolved = selected.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("archive path escapes snapshot")
    return resolved.read_bytes()


def _evidence_context(spans: Any, pages: Mapping[str, bytes]) -> tuple[str, list[str]]:
    if not isinstance(spans, list):
        raise ValueError("historical evidence spans missing")
    rendered: list[str] = []
    page_ids: list[str] = []
    for span in spans:
        if _source_span_item_error(span):
            raise ValueError("invalid historical source span")
        page_id = span["page_id"]
        content = pages.get(page_id)
        if content is None or _sha(content) != span["content_sha256"]:
            raise ValueError("historical page hash mismatch")
        if len(content) != span["content_byte_length"]:
            raise ValueError("historical page length mismatch")
        excerpt = content[span["byte_start"] : span["byte_end"]].decode("utf-8")
        if excerpt != span["excerpt"]:
            raise ValueError("historical source bytes mismatch")
        rendered.append(f"[PAGE {page_id}]\n{excerpt}")
        if page_id not in page_ids:
            page_ids.append(page_id)
    return "\n\n".join(rendered), page_ids


def prepare_replay(episode: Mapping[str, Any], snapshot_root: Path) -> dict[str, Any]:
    """Validate a frozen episode and construct the three reader environments."""

    if episode.get("schema") != SCHEMA:
        raise ValueError("unsupported historical episode schema")
    for key in (
        "episode_id",
        "host",
        "session_hash",
        "prompt",
        "index_generation",
        "policy_sha256",
    ):
        if not isinstance(episode.get(key), str) or not episode[key]:
            raise ValueError(f"historical {key} missing")
    as_of = _timestamp(episode.get("as_of"))
    state = episode.get("state_snapshot")
    if not isinstance(state, Mapping) or not isinstance(state.get("state"), dict):
        raise ValueError("historical state snapshot missing")
    if _timestamp(state.get("captured_at")) > as_of:
        raise ValueError("future state snapshot")
    if state.get("sha256") != canonical_sha256(state["state"]):
        raise ValueError("historical state hash mismatch")
    history = episode.get("history")
    if not isinstance(history, list):
        raise ValueError("historical conversation context missing")
    for message in history:
        if not isinstance(message, dict) or message.get("role") not in {
            "user",
            "assistant",
        }:
            raise ValueError("invalid historical context message")
        if (
            not isinstance(message.get("content"), str)
            or _timestamp(message.get("at")) > as_of
        ):
            raise ValueError("future or invalid conversation context")
    root = snapshot_root.resolve(strict=True)
    if snapshot_root.is_symlink() or not root.is_dir():
        raise ValueError("snapshot must be an isolated directory")
    pages: dict[str, bytes] = {}
    inventory = episode.get("pages")
    if not isinstance(inventory, list):
        raise ValueError("historical page inventory missing")
    for page in inventory:
        if not isinstance(page, dict) or not isinstance(page.get("page_id"), str):
            raise ValueError("invalid historical page identity")
        page_id = page["page_id"]
        if not page_id or page_id in pages:
            raise ValueError("duplicate historical page identity")
        if _timestamp(page.get("captured_at")) > as_of:
            raise ValueError("future page snapshot")
        content = _archived_bytes(root, page.get("relative_path"))
        if _sha(content) != page.get("content_sha256"):
            raise ValueError("historical page snapshot changed")
        pages[page_id] = content
    candidates = episode.get("candidate_page_ids")
    if not isinstance(candidates, list) or any(
        not isinstance(page, str) or page not in pages for page in candidates
    ):
        raise ValueError("historical candidate inventory missing")
    if len(candidates) != len(set(candidates)):
        raise ValueError("duplicate historical candidate")
    on_context, selected = _evidence_context(episode.get("selected_evidence"), pages)
    restore_context, gold = _evidence_context(episode.get("gold_evidence"), pages)
    if not set(selected).issubset(candidates):
        raise ValueError("selected evidence was absent from captured candidates")
    required = episode.get("memory_required")
    if type(required) is not bool or required != bool(gold):
        raise ValueError("independent memory requirement and gold evidence disagree")
    if episode.get("health") not in {"healthy", "degraded"}:
        raise ValueError("historical health stratum missing")
    return {
        "schema": SCHEMA,
        "episode_sha256": canonical_sha256(dict(episode)),
        "episode_id": episode["episode_id"],
        "as_of": episode["as_of"],
        "prompt": episode["prompt"],
        "host": episode["host"],
        "session_hash": episode["session_hash"],
        "history": copy.deepcopy(history),
        "state": copy.deepcopy(state["state"]),
        "base_state_sha256": state["sha256"],
        "index_generation": episode["index_generation"],
        "policy_sha256": episode["policy_sha256"],
        "health": episode["health"],
        "candidate_page_ids": list(candidates),
        "selected_page_ids": selected,
        "gold_page_ids": gold,
        "memory_required": required,
        "arms": {"on": on_context, "off": "", "restore": restore_context},
        "promotion_eligible": False,
        "truth_authority": "external_independent_labels_required",
    }


def failure_categories(
    prepared: Mapping[str, Any], correct: Mapping[str, bool]
) -> list[str]:
    """Separate retrieval/selection failures from the reader and recoverability."""

    if set(correct) != set(ARMS) or any(
        type(value) is not bool for value in correct.values()
    ):
        raise ValueError("three exact boolean correctness outcomes required")
    selected = set(prepared["selected_page_ids"])
    gold = set(prepared["gold_page_ids"])
    categories: list[str] = []
    if not prepared["memory_required"]:
        if selected:
            categories.append("unnecessary_injection")
        return categories
    if gold - set(prepared["candidate_page_ids"]):
        categories.append("candidate_missing")
    elif gold - selected:
        categories.append("candidate_not_selected")
    elif not correct["on"]:
        categories.append("reader_failure_after_correct_injection")
    if not correct["on"] and correct["restore"]:
        categories.append("recoverable_with_restored_gold")
    return categories


def sample_historical_cohort(
    episodes: Sequence[Mapping[str, Any]],
    *,
    start: str,
    end: str,
    per_stratum: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    """Uniform seeded sampling within health strata, including zero-card turns."""

    lower, upper = _timestamp(start), _timestamp(end)
    if lower >= upper or type(per_stratum) is not int or per_stratum <= 0:
        raise ValueError("invalid historical sampling window or count")
    groups: dict[str, list[Mapping[str, Any]]] = {"healthy": [], "degraded": []}
    seen: set[str] = set()
    for episode in episodes:
        identity = episode.get("episode_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("missing or duplicate historical episode identity")
        seen.add(identity)
        if lower <= _timestamp(episode.get("as_of")) < upper:
            health = episode.get("health")
            if health not in groups:
                raise ValueError("historical health stratum missing")
            groups[health].append(episode)
    return [
        episode
        for group in groups.values()
        for episode in sorted(
            group, key=lambda row: _sha(f"{seed}:{row['episode_id']}".encode())
        )[:per_stratum]
    ]


def replay(
    prepared: Mapping[str, Any],
    *,
    reader: Callable[[Mapping[str, Any]], str],
    score: Callable[[str], bool],
    seed: int = 0,
) -> dict[str, Any]:
    """Run isolated diagnostic adapters; a score is never promotion authority.

    Adapters are supplied by an explicit offline caller, never loaded from an
    episode. Production callers must use calibrated local reader/judge routes.
    """

    base = {
        key: prepared[key]
        for key in (
            "prompt",
            "host",
            "session_hash",
            "history",
            "state",
            "as_of",
            "index_generation",
            "policy_sha256",
        )
    }
    order = sorted(
        ARMS,
        key=lambda arm: _sha(f"{seed}:{prepared['episode_sha256']}:{arm}".encode()),
    )
    results: dict[str, Any] = {}
    for arm in order:
        request = copy.deepcopy(base)
        request.update(context=prepared["arms"][arm], seed=seed)
        answer = reader(request)
        if not isinstance(answer, str):
            raise ValueError("diagnostic reader returned a non-string answer")
        correct = score(answer)
        if type(correct) is not bool:
            raise ValueError("diagnostic scorer must return an exact boolean")
        results[arm] = {
            "answer_sha256": _sha(answer.encode()),
            "correct": correct,
            "post_state_sha256": canonical_sha256(request["state"]),
        }
    return seal_object(
        {
            "schema": SCHEMA,
            "status": "diagnostic_completed",
            "episode_sha256": prepared["episode_sha256"],
            "base_state_sha256": prepared["base_state_sha256"],
            "arm_order": order,
            "results": results,
            "failure_categories": failure_categories(
                prepared, {arm: row["correct"] for arm, row in results.items()}
            ),
            "promotion_eligible": False,
            "truth_authority": "external_independent_labels_required",
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--episode-sha256", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.episode.read_bytes()
    if _sha(raw) != args.episode_sha256:
        raise ValueError("episode artifact hash mismatch")
    prepared = prepare_replay(json.loads(raw), args.snapshot_root)
    # Context bytes remain private; the CLI only prepares and never calls a model.
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(
        args.output,
        "x",
        encoding="utf-8",
        opener=lambda path, flags: os.open(path, flags, 0o600),
    ) as handle:
        handle.write(
            json.dumps(seal_object(prepared), ensure_ascii=False, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {"status": "prepared", "arms": list(ARMS), "promotion_eligible": False}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
