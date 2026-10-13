from __future__ import annotations

import json
import os
import pty
import select
import subprocess
import sys
from pathlib import Path

import pytest

from chronovisor.recall.recall_distillation_luna import (
    LunaBulkBundle,
    native_output_sha256,
)
from scripts.recall_luna_worker_io import bind_decisions


def test_native_decisions_are_bound_to_exact_frozen_ids() -> None:
    request = {
        "teacher_profile": "native-luna-single-v1",
        "records": [{"task_id": "a" * 64, "input_sha256": "b" * 64}],
    }
    decisions = [
        {
            "task_id": "a" * 64,
            "verdict": "uncertain",
            "confidence": 0.4,
            "reason_code": "insufficient_context",
        }
    ]
    row = bind_decisions(request, decisions, assignment_id="worker-0", run_id="run-0")[
        0
    ]
    assert row["verdict"] == "uncertain"
    assert row["input_sha256"] == "b" * 64
    assert row["rationale"] is None
    assert row["native_receipt"]["output_sha256"] == native_output_sha256(row)
    with pytest.raises(ValueError, match="cardinality"):
        bind_decisions(request, [], assignment_id="worker-0", run_id="run-0")
    with pytest.raises(ValueError, match="identities"):
        bind_decisions(
            request,
            [{**decisions[0], "task_id": "c" * 64}],
            assignment_id="worker-0",
            run_id="run-0",
        )


def test_worker_tty_accepts_json_larger_than_canonical_input_buffer(tmp_path: Path) -> None:
    bundle = LunaBulkBundle.create(
        tmp_path / "bundle",
        [{
            "rally_id": "synthetic-rally",
            "candidate_id": "synthetic-candidate",
            "as_of_us": 1_000,
            "input": {"query": "synthetic", "candidate": "unrelated"},
            "split": "train",
            "group_id": "synthetic-group",
        }],
        shard_count=1,
    )
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).parents[1] / "scripts/recall_luna_worker_io.py"),
         "--bundle", str(bundle.root), "--shard", "0", "--assignment-id", "tty-test",
         "--run-id", "tty-test-run"],
        stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    os.close(slave)
    try:
        assert process.stdout is not None
        assert select.select([process.stdout], [], [], 10)[0]
        request = json.loads(process.stdout.readline())
        response = [{
            "task_id": request["records"][0]["task_id"],
            "verdict": "irrelevant", "confidence": 0.9,
            "reason_code": "synthetic_unrelated",
        }]
        payload = (" " * 8_192 + json.dumps(response) + "\n").encode()
        while payload:
            payload = payload[os.write(master, payload):]
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        replies = [json.loads(line) for line in stdout.splitlines()]
        assert replies[0]["status"] == "committed"
        assert replies[0]["count"] == 1
        assert replies[1]["status"] == "shard_complete"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
