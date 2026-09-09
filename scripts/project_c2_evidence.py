"""Build an isolated, source-bound C2 artifact using existing ingest triage.

The supplied child must have an opt-in native source map. This command never
marks a Raw processed or applies the page operations returned by triage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Isolated Chronovisor root with its own runtime configuration",
    )
    parser.add_argument(
        "--raw-dir", type=Path, required=True, help="Source Raw directory (read only)"
    )
    parser.add_argument("--child", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--triage-result",
        type=Path,
        help="Replay an already fixed triage_c2 result without a model call",
    )
    source.add_argument(
        "--generate",
        action="store_true",
        help="Call the configured ingest model once, with bounded repairs",
    )
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    production = (Path.home() / ".chronovisor").resolve()
    if root in {production, raw_dir.parent} or root.is_relative_to(raw_dir):
        parser.error(
            "--root must be an isolated output root, distinct from the source root"
        )
    if not root.is_dir() or not (root / "config.toml").is_file():
        parser.error("--root must already contain its isolated config.toml")
    # Set the root before importing modules with root-bound stores/configuration.
    os.environ["CHRONOVISOR_ROOT"] = str(root)

    from chronovisor.core.store import CHRONOVISOR_ROOT

    if CHRONOVISOR_ROOT.resolve() != root:
        parser.error("Chronovisor was already imported with a different root")

    from chronovisor.ingest.raw_semantic_projection import read_native_c2_source_records
    from chronovisor.research.c2_semantic_projection import (
        load_c2_semantic_envelope,
        materialize_c2_semantic_envelope,
        store_c2_semantic_envelope,
    )
    from chronovisor.search.research_store import ResearchStore

    try:
        sources = read_native_c2_source_records(args.child, raw_dir=raw_dir)
        if args.generate:
            from chronovisor.ingest.ingest_triage import triage_c2

            triage = triage_c2(sources["source_records"], raise_on_failure=True)
        else:
            triage = json.loads(args.triage_result.read_text(encoding="utf-8"))
        if not isinstance(triage, dict):
            raise ValueError("C2 triage did not produce a structured result")
        envelope = materialize_c2_semantic_envelope(
            triage_result=triage,
            source_records=sources["source_records"],
            source_bindings=sources["bindings"],
            projection_id=sources["projection_id"],
            source_sha256=sources["source_sha256"],
        )
        store = ResearchStore()
        artifact = store_c2_semantic_envelope(
            store, envelope, source_uri=f"projection:{sources['projection_id']}"
        )
        restored = load_c2_semantic_envelope(store, artifact.artifact_id)
        if restored != envelope:
            raise ValueError("C2 artifact read-back differs from fixed materialization")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        # Keep private source text and model responses out of terminal receipts.
        print(json.dumps({"status": "held", "error_type": type(exc).__name__}))
        return 2
    print(
        json.dumps(
            {
                "status": "stored",
                "artifact_id": artifact.artifact_id,
                "projection_id": sources["projection_id"],
                "source_record_count": len(sources["source_records"]),
                "model_called": args.generate,
                "page_operations_applied": False,
                "production_c2_enabled": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
