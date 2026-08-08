"""Recall co-fire graph.

Pages that repeatedly appear together in recall contexts become weak edges.
Search graph expansion can then use actual recall behavior, not only wiki links.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chronovisor.core.store import CHRONOVISOR_ROOT

COFIRE_FILE = CHRONOVISOR_ROOT / "recall" / "cofire.json"


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
    from chronovisor.recall.cofire import build_cofire_graph

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
