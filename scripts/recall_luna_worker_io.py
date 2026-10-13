#!/usr/bin/env python3
"""Stream one frozen shard to a native Luna worker and persist its decisions.

The process never chooses a verdict. The native worker reads each request and
supplies task_id, verdict, confidence and reason_code for every record. Raw
requests travel over stdin/stdout only; durable batches stay in the private
bundle and every committed response is bound to its exact source identity.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import select
import sys
import termios
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronovisor.recall.recall_distillation_luna import (
    LunaBulkBundle,
    native_output_sha256,
)


def bind_decisions(
    request: Mapping[str, Any],
    decisions: object,
    *,
    assignment_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    """Bind decisions without guessing, correcting or filling any verdict."""

    records = request["records"]
    if not isinstance(decisions, list) or len(decisions) != len(records):
        raise ValueError("response cardinality mismatch")
    by_id: dict[str, Mapping[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping) or set(decision) != {
            "task_id",
            "verdict",
            "confidence",
            "reason_code",
        }:
            raise ValueError("decision fields mismatch")
        task_id = decision["task_id"]
        if not isinstance(task_id, str) or task_id in by_id:
            raise ValueError("duplicate or invalid response identity")
        by_id[task_id] = decision
    if set(by_id) != {record["task_id"] for record in records}:
        raise ValueError("response identities do not match the frozen request")
    labels: list[dict[str, Any]] = []
    for record in records:
        label = {
            **by_id[record["task_id"]],
            "input_sha256": record["input_sha256"],
            "teacher_profile": request["teacher_profile"],
            "native_luna": True,
            "rationale": None,
        }
        label["native_receipt"] = {
            "task_id": label["task_id"],
            "output_sha256": native_output_sha256(label),
            "assignment_id": assignment_id,
            "run_id": run_id,
        }
        labels.append(label)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--idle-timeout", type=float, default=900)
    args = parser.parse_args()
    # macOS canonical TTY input truncates long JSON lines before Python reads
    # them. Pipes need no adjustment; the dedicated worker TTY must be raw.
    if sys.stdin.isatty():
        settings = termios.tcgetattr(sys.stdin.fileno())
        settings[3] &= ~(termios.ICANON | termios.ECHO)
        settings[6][termios.VMIN] = 1
        settings[6][termios.VTIME] = 0
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, settings)
    bundle = LunaBulkBundle.open(args.bundle)
    path = bundle.root / "shards" / f"{args.shard:02d}" / ".native-worker.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "shard_busy", "shard": args.shard}), flush=True)
            return 2
        return _serve_shard(bundle, args)
    finally:
        os.close(descriptor)


def _serve_shard(bundle: LunaBulkBundle, args: argparse.Namespace) -> int:
    committed = 0
    while True:
        if args.max_records > 0 and committed >= args.max_records:
            print(json.dumps({"status": "round_complete", "committed": committed}), flush=True)
            return 0
        # Show the complete query, context and candidate together. Ten full
        # inputs exceed common tool-output limits and invite partial reading.
        # The frozen batch size is an upper bound, so single-record commits
        # retain the same identities, hash validation and resume contract.
        request = bundle.next_batch(args.shard, limit=1)
        if not request["records"]:
            print(
                json.dumps({"status": "shard_complete", "shard": args.shard}),
                flush=True,
            )
            return 0
        print(
            json.dumps(request, ensure_ascii=False, separators=(",", ":")), flush=True
        )
        while True:
            if args.idle_timeout > 0 and not select.select([sys.stdin], [], [], args.idle_timeout)[0]:
                print(json.dumps({"status": "worker_input_timeout"}), flush=True)
                return 3
            response = sys.stdin.readline()
            if not response:
                return 0
            try:
                labels = bind_decisions(
                    request,
                    json.loads(response),
                    assignment_id=args.assignment_id,
                    run_id=args.run_id,
                )
                bundle.submit(args.shard, cursor=request["cursor"], records=labels)
            except (ValueError, KeyError) as exc:
                print(
                    json.dumps(
                        {
                            "status": "response_rejected",
                            "error_type": type(exc).__name__,
                        }
                    ),
                    flush=True,
                )
                continue
            print(
                json.dumps(
                    {
                        "status": "committed",
                        "count": len(labels),
                        "cursor": request["cursor"],
                    }
                ),
                flush=True,
            )
            committed += len(labels)
            break


if __name__ == "__main__":
    raise SystemExit(main())
