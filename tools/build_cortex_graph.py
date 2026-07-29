#!/usr/bin/env python3
"""Export the local Wiki graph used by the Synaptic Cortex dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chronovisor.core.runtime_config import runtime_identity
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.ops.cortex import build_cortex_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("cortex_graph.json"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=CHRONOVISOR_ROOT,
        help="Chronovisor root containing pages/ and system/",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    identity = runtime_identity()
    commit = str(
        identity.get("commit_id")
        or identity.get("expected_commit")
        or ""
    )
    graph = build_cortex_graph(
        args.root,
        commit=commit,
        use_cache=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(graph, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        "Built graph: "
        f"{len(graph['nodes'])} neurons, "
        f"{len(graph['links'])} rendered synapses, "
        f"{graph['meta']['deferred']} deferred targets → {args.output}"
    )


if __name__ == "__main__":
    main()
