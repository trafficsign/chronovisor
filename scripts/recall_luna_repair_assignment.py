#!/usr/bin/env python3
"""Derive a canonical bundle from the legacy non-probe assignment typo.

The original bundle and native responses remain immutable. This explicit repair
does not make the production importer accept the legacy assignment revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronovisor.recall import recall_distillation_luna as luna
from chronovisor.recall import recall_distillation_native as native

LEGACY_REVISION = "native-luna-assignment-v1"
REPAIR_SCHEMA = "chronovisor.native-luna-assignment-repair.v1"


def _frozen_hashes(path: Path) -> dict[str, str]:
    files = [path / "manifest.json", *sorted((path / "snapshot").glob("*.json*")),
             *sorted(path.glob("shards/*/source.jsonl")),
             *sorted(path.glob("shards/*/batches/*.json"))]
    files += [path / name for name in ("labels.jsonl", "completion.json")
              if (path / name).exists()]
    return {str(item.relative_to(path)): hashlib.sha256(luna._read_bytes(item)).hexdigest()
            for item in files}


def repair_assignment_bundle(source: Path, destination: Path) -> dict[str, Any]:
    """Repair only the proven five-field, non-probe producer metadata error."""
    if destination.exists() or destination.is_symlink():
        raise ValueError("repair destination already exists")
    before = _frozen_hashes(source)
    original = luna.LunaBulkBundle.open(source)
    manifest = original.manifest
    if original.status()["remaining"] or manifest["cohort"] != native.NATIVE_COHORT:
        raise ValueError("repair requires a complete native cohort")
    # Validate every source and response before creating any repair output.
    sources = native._source_rows(original)
    batches = native._batch_records(original)
    source_rows = luna._read_jsonl(source / "snapshot/source.jsonl")
    repaired_rows = []
    for row in source_rows:
        expected = {"revision": LEGACY_REVISION,
                    "profile": luna.DEFAULT_TEACHER_PROFILE, "split": row["split"],
                    "probe": False, "owner": "native-luna"}
        if row.get("assignment") != expected or row.get("repeat_pair_id"):
            raise ValueError("source is outside the exact legacy assignment repair")
        changed = {**row, "assignment": {**expected, "revision": native.NATIVE_ASSIGNMENT_REVISION}}
        changed.pop("source_record_sha256")
        repaired_rows.append(changed)
    labels = {label["task_id"]: label for _, label, _ in batches}
    if len(labels) != len(sources):
        raise ValueError("repair response cardinality mismatch")
    proof = {"schema": REPAIR_SCHEMA, "original_manifest_sha256": before["manifest.json"],
             "original_source_snapshot_sha256": manifest["source_snapshot_sha256"],
             "from_revision": LEGACY_REVISION, "to_revision": native.NATIVE_ASSIGNMENT_REVISION,
             "records": len(sources), "quarantined": 0}
    derived = luna.LunaBulkBundle.create(
        destination, repaired_rows, teacher_profile=manifest["teacher_profile"],
        cohort=manifest["cohort"], frozen_as_of_us=manifest["frozen_as_of_us"],
        split_plan_id=manifest["split_plan_id"], shard_count=len(manifest["shards"]),
        batch_size=manifest["batch_size"], route=manifest["teacher"]["route"],
        metadata={"assignment_repair": proof},
    )
    mappings = []
    for spec in derived.manifest["shards"]:
        shard = spec["shard"]
        for row in derived._source_rows(shard):
            old = sources[row["task_id"]]
            # The teacher-visible payload and all other metadata must be exact.
            stripped = {key: value for key, value in row.items()
                        if key not in {"assignment", "source_record_sha256"}}
            if stripped != {key: value for key, value in old.items()
                            if key not in {"assignment", "source_record_sha256"}}:
                raise ValueError("repair changed fields outside assignment revision")
            mappings.append({"task_id": row["task_id"], "input_sha256": row["input_sha256"],
                             "original_source_record_sha256": old["source_record_sha256"],
                             "repaired_source_record_sha256": row["source_record_sha256"]})
        while True:
            request = derived.next_batch(shard)
            if not request["records"]:
                break
            response = [labels[row["task_id"]] for row in request["records"]]
            derived.submit(shard, cursor=request["cursor"], records=response)
    completion = derived.finalize()
    if _frozen_hashes(source) != before:
        raise ValueError("original frozen bundle changed during repair")
    if {label["task_id"]: label for label in derived.labels()} != labels:
        raise ValueError("repair changed native responses")
    proof = {**proof, "status": "repaired", "original_frozen_sha256": before,
             "repaired_manifest_sha256": hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest(),
             "repaired_source_snapshot_sha256": derived.manifest["source_snapshot_sha256"],
             "labels_sha256": completion["labels_sha256"],
             "verdicts": completion["verdicts"], "mapping_sha256": luna._sha256_json(mappings)}
    luna._publish_immutable(destination / "assignment-repair-mapping.jsonl", luna._jsonl_bytes(mappings))
    luna._publish_immutable(destination / "assignment-repair-receipt.json", luna._json_bytes(proof) + b"\n")
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = repair_assignment_bundle(args.source, args.destination)
    print(json.dumps({key: value for key, value in result.items() if key != "original_frozen_sha256"}, sort_keys=True))


if __name__ == "__main__":
    main()
