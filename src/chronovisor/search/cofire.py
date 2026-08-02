"""Recall co-fire graph.

Pages that repeatedly appear together in recall contexts become weak edges.
Search graph expansion can then use actual recall behavior, not only wiki links.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.store import CHRONOVISOR_ROOT, find_page
from chronovisor.recall.feedback_ledger import (
    feedback_row_sha256,
    trusted_negative_feedback_rows,
)
from chronovisor.recall.recall_answer_eval import (
    TRAIN_ANSWER_EVAL_ARTIFACT,
    validate_answer_outcome_artifact,
)
from chronovisor.recall.recall_field_schema import session_hash
from chronovisor.recall.recall_log_schema import (
    canonicalize_page_ids,
    join_used_recall_episodes,
    page_ids_from_record,
)
from chronovisor.recall.recall_runtime_paths import RECALL_DIR

RECALL_LOG_FILE = RECALL_DIR / "recall-log.jsonl"
RECALL_PULL_LOG_FILE = RECALL_DIR / "pull-log.jsonl"
COFIRE_FILE = RECALL_DIR / "cofire.json"
RECALL_FEEDBACK_FILE = RECALL_DIR / "feedback.jsonl"


def _read_recent_jsonl(path: Path, *, limit: int = 5000) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=max(1, limit))
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def build_cofire_graph(
    *,
    log_file: Path = RECALL_LOG_FILE,
    pull_log_file: Path = RECALL_PULL_LOG_FILE,
    output_file: Path = COFIRE_FILE,
    limit: int = 5000,
    min_count: int = 2,
    answer_outcome_file: Path = TRAIN_ANSWER_EVAL_ARTIFACT,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    minimum_outcome_clusters: int = 20,
    write: bool = True,
) -> dict[str, Any]:
    rows = _read_recent_jsonl(log_file, limit=limit)
    joined = join_used_recall_episodes(
        rows,
        _read_recent_jsonl(pull_log_file, limit=limit),
    )
    from chronovisor.core.alias_store import load_aliases

    aliases = load_aliases()
    try:
        from chronovisor.ingest.page_registry import PageRegistry

        registry = PageRegistry(CHRONOVISOR_ROOT)
        uid_by_page = {
            page_id: str((registry.resolve(page_id) or {}).get("uid") or "")
            for row in rows
            for page_id in page_ids_from_record(row)
        }
    except Exception:
        uid_by_page = {}

    def compile_graph(page_sets: list[list[str]]) -> tuple[dict[str, list[dict[str, Any]]], int]:
        pair_counts: Counter[tuple[str, str]] = Counter()
        node_counts: Counter[str] = Counter()
        episodes = 0
        for page_ids in page_sets:
            if len(page_ids) < 2:
                continue
            episodes += 1
            for page_id in page_ids:
                node_counts[page_id] += 1
            for left, right in itertools.combinations(sorted(page_ids), 2):
                pair_counts[(left, right)] += 1
        graph: dict[str, list[dict[str, Any]]] = {}
        for (left, right), count in pair_counts.items():
            if count < min_count:
                continue
            denom = max(node_counts[left], node_counts[right], 1)
            weight = count / denom
            edge = {"count": count, "weight": round(weight, 4)}
            graph.setdefault(left, []).append(
                {
                    "page_id": right,
                    **({"page_uid": uid_by_page[right]} if uid_by_page.get(right) else {}),
                    **edge,
                }
            )
            graph.setdefault(right, []).append(
                {
                    "page_id": left,
                    **({"page_uid": uid_by_page[left]} if uid_by_page.get(left) else {}),
                    **edge,
                }
            )
        for page_id, items in graph.items():
            items.sort(
                key=lambda item: (float(item["weight"]), int(item["count"])),
                reverse=True,
            )
            graph[page_id] = items[:20]
        return graph, episodes

    exposure_graph, exposure_episodes = compile_graph(
        [canonicalize_page_ids(page_ids_from_record(row), aliases) for row in rows]
    )
    usage_graph, usage_episodes = compile_graph(
        [
            canonicalize_page_ids(episode["page_ids"], aliases)
            for episode in joined["episodes"]
        ]
    )
    outcome = validate_answer_outcome_artifact(
        answer_outcome_file,
        required_split="train",
        minimum_independent_samples=minimum_outcome_clusters,
    )
    trusted_negatives = trusted_negative_feedback_rows(
        feedback_file, recall_log_file=log_file
    )
    excluded_by_episode: dict[tuple[str, str], set[str]] = {}
    negative_producers: list[dict[str, Any]] = []
    for row in trusted_negatives:
        snapshot = row["snapshot"]
        episode_key = (
            str(row["ref"]),
            session_hash(
                str(row.get("host") or snapshot.get("host") or ""),
                str(snapshot["session_id"]),
            ),
        )
        excluded_by_episode.setdefault(episode_key, set()).update(
            canonicalize_page_ids(
                [str(page_id) for page_id in row["negative_pages"]], aliases
            )
        )
        negative_producers.append(
            {
                "producer_key": str(
                    row.get("content_correction_key") or row.get("producer_key")
                ),
                "feedback_sha256": feedback_row_sha256(row),
                "decision_id": str(row["ref"]),
                "session_hash": episode_key[1],
                "page_ids": list(dict.fromkeys(row["negative_pages"])),
            }
        )
    for penalty in outcome.get("page_penalties", []):
        if not isinstance(penalty, dict):
            continue
        decision_id = str(penalty.get("decision_id") or "")
        reward_session = str(penalty.get("session_hash") or "")
        page_ids = canonicalize_page_ids(
            [str(penalty.get("page_id") or "")], aliases
        )
        if decision_id and reward_session and page_ids:
            excluded_by_episode.setdefault((decision_id, reward_session), set()).update(
                page_ids
            )
    outcome_groups: dict[tuple[str, str, str], list[str]] = {}
    for reward in outcome.get("page_rewards", []):
        if not isinstance(reward, dict):
            continue
        decision_id = str(reward.get("decision_id") or "")
        episode_id = str(reward.get("episode_id") or "")
        reward_session = str(reward.get("session_hash") or "")
        page_id = str(reward.get("page_id") or "")
        expected_content_sha = str(reward.get("content_sha256") or "")
        if not decision_id or not episode_id or not reward_session or not page_id:
            continue
        path = find_page(page_id)
        try:
            current_content_sha = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path is not None
                else ""
            )
        except OSError:
            current_content_sha = ""
        if current_content_sha != expected_content_sha:
            continue
        canonical_ids = canonicalize_page_ids([page_id], aliases)
        if not canonical_ids or canonical_ids[0] in excluded_by_episode.get(
            (decision_id, reward_session), set()
        ):
            continue
        outcome_groups.setdefault((episode_id, decision_id, reward_session), []).append(
            canonical_ids[0]
        )
    positive_page_sets = [
        canonicalize_page_ids(list(dict.fromkeys(page_ids)), aliases)
        for _key, page_ids in sorted(outcome_groups.items())
    ]
    positive_graph, positive_episodes = compile_graph(positive_page_sets)

    payload = {
        "schema_version": 3,
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "log_file": str(log_file),
        "episodes": exposure_episodes,
        "positive_episodes": positive_episodes,
        "nodes": len(exposure_graph),
        "edges": sum(len(items) for items in exposure_graph.values()),
        "positive_nodes": len(positive_graph),
        "positive_edges": sum(len(items) for items in positive_graph.values()),
        "usage_diagnostic_episodes": usage_episodes,
        "min_count": min_count,
        "node_uids": {
            page_id: uid for page_id, uid in sorted(uid_by_page.items()) if uid
        },
        "join": {key: value for key, value in joined.items() if key != "episodes"},
        "graphs": {
            "positive_used": {
                "supervision": "verified_train_answer_outcome",
                "graph": positive_graph,
            },
            "usage_diagnostic": {
                "supervision": "explicit_used_receipt_not_learning",
                "graph": usage_graph,
            },
            "exposure": {
                "supervision": "recalled_not_confirmed_used",
                "graph": exposure_graph,
            },
        },
        # Compatibility alias: explicitly the exposure graph, never a positive
        # usage label.
        "graph": exposure_graph,
        "answer_outcome": {
            key: value
            for key, value in outcome.items()
            if key not in {"page_rewards", "page_penalties"}
        },
        "negative_retractions": negative_producers,
    }
    if write:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def neighbors(
    page_id: str,
    *,
    path: Path = COFIRE_FILE,
    limit: int = 8,
    positive_weight: float = 4.0,
    exposure_weight: float = 1.0,
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    scored: dict[str, dict[str, Any]] = {}

    def add_graph(graph: Any, supervision: str, multiplier: float) -> None:
        if multiplier <= 0 or not isinstance(graph, dict):
            return
        rows = graph.get(page_id, [])
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("page_id"), str):
                continue
            target = row["page_id"]
            score = float(row.get("weight") or 0.0) * multiplier
            current = scored.setdefault(
                target,
                {"page_id": target, "count": 0, "weight": 0.0, "signals": []},
            )
            current["count"] += int(row.get("count") or 0)
            current["weight"] += score
            current["signals"].append(supervision)

    graphs = payload.get("graphs")
    if isinstance(graphs, dict):
        positive = graphs.get("positive_used")
        exposure = graphs.get("exposure")
        add_graph(
            positive.get("graph") if isinstance(positive, dict) else None,
            "positive_used",
            positive_weight,
        )
        add_graph(
            exposure.get("graph") if isinstance(exposure, dict) else None,
            "exposure",
            exposure_weight,
        )
    else:
        add_graph(payload.get("graph"), "legacy_exposure", exposure_weight)
    ranked = sorted(
        scored.values(),
        key=lambda row: (float(row["weight"]), int(row["count"])),
        reverse=True,
    )
    for row in ranked:
        row["weight"] = round(float(row["weight"]), 4)
    return ranked[:limit]


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-cofire`` command-line entry point."""
    parser = argparse.ArgumentParser(description="Build recall co-fire graph.")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = build_cofire_graph(
        limit=max(1, args.limit),
        min_count=max(1, args.min_count),
        write=not args.no_write,
    )
    if args.json:
        print(
            json.dumps(
                {k: v for k, v in payload.items() if k not in {"graph", "graphs"}},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"episodes\t{payload['episodes']}")
        print(f"nodes\t{payload['nodes']}")
        print(f"edges\t{payload['edges']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
