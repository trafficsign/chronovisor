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
from scripts.recall_luna_worker_io import bind_decisions, display_request


def test_worker_display_keeps_every_input_byte_and_omits_only_bound_metadata() -> None:
    payload = {"query": "synthetic query", "context": ["first", "", "last"], "candidate": "whole text"}
    request = {"prompt": "frozen prompt", "records": [{
        "task_id": "a" * 64, "input_sha256": "b" * 64, "input": payload,
        "context_sha256": ["c" * 64] * 10_000,
    }]}
    displayed = display_request(request)
    assert displayed["prompt"] == request["prompt"]
    assert displayed["records"][0] == {"task_id": "a" * 64, "input_sha256": "b" * 64, "input": payload}
    assert len(request["records"][0]["context_sha256"]) == 10_000
    assert len(json.dumps(displayed)) < 1000


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


def test_worker_presents_one_complete_input_before_accepting_its_decision(tmp_path: Path) -> None:
    inputs = [
        {"query": f"synthetic-{index}", "context": ["first-context", "final-context"],
         "candidate": {"text": "whole-candidate"}}
        for index in range(2)
    ]
    bundle = LunaBulkBundle.create(
        tmp_path / "bundle",
        [{"rally_id": f"rally-{index}", "candidate_id": f"candidate-{index}",
          "as_of_us": 1000 + index, "input": value, "split": "train", "group_id": "group"}
         for index, value in enumerate(inputs)],
        shard_count=1, batch_size=10,
    )
    process = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).parents[1] / "scripts/recall_luna_worker_io.py"),
         "--bundle", str(bundle.root), "--shard", "0", "--assignment-id", "full-input",
         "--run-id", "full-input-run"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout is not None and process.stdin is not None
        request = json.loads(process.stdout.readline())
        assert len(request["records"]) == 1
        assert request["records"][0]["input"] == inputs[0]
        duplicate = subprocess.run(
            process.args, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10
        )
        assert duplicate.returncode == 2
        assert json.loads(duplicate.stdout) == {"status": "shard_busy", "shard": 0}
        process.stdin.write(json.dumps([{
            "task_id": request["records"][0]["task_id"], "verdict": "uncertain",
            "confidence": 0.5, "reason_code": "synthetic_only",
        }]) + "\n")
        process.stdin.flush()
        assert json.loads(process.stdout.readline())["count"] == 1
        second = json.loads(process.stdout.readline())
        assert second["cursor"] == 1
        assert len(second["records"]) == 1
        assert second["records"][0]["input"] == inputs[1]
        _stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert LunaBulkBundle.open(bundle.root).status()["completed"] == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_worker_idle_timeout_releases_lease_without_writing_a_label(tmp_path: Path) -> None:
    bundle = LunaBulkBundle.create(tmp_path / "bundle", [{
        "rally_id": "rally", "candidate_id": "candidate", "as_of_us": 1000,
        "input": {"query": "synthetic"}, "split": "train", "group_id": "group",
    }], shard_count=1)
    command = [sys.executable, "-u", str(Path(__file__).parents[1] / "scripts/recall_luna_worker_io.py"),
               "--bundle", str(bundle.root), "--shard", "0", "--assignment-id", "timeout-test",
               "--run-id", "timeout-run", "--idle-timeout", "0.05"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout is not None
    assert json.loads(process.stdout.readline())["cursor"] == 0
    assert json.loads(process.stdout.readline())["status"] == "worker_input_timeout"
    process.wait(timeout=5)
    assert process.returncode == 3
    assert LunaBulkBundle.open(bundle.root).status()["completed"] == 0
    restarted = subprocess.run(command, input="", capture_output=True, text=True, timeout=5)
    assert restarted.returncode == 0
    assert json.loads(restarted.stdout.splitlines()[0])["cursor"] == 0


def test_worker_round_limit_preserves_the_next_unread_record(tmp_path: Path) -> None:
    bundle = LunaBulkBundle.create(tmp_path / "bundle", [{
        "rally_id": f"rally-{i}", "candidate_id": f"candidate-{i}", "as_of_us": 1000 + i,
        "input": {"query": "synthetic"}, "split": "train", "group_id": "group",
    } for i in range(2)], shard_count=1)
    process = subprocess.Popen([
        sys.executable, "-u", str(Path(__file__).parents[1] / "scripts/recall_luna_worker_io.py"),
        "--bundle", str(bundle.root), "--shard", "0", "--assignment-id", "round-test",
        "--run-id", "round-run", "--max-records", "1",
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout is not None
    request = json.loads(process.stdout.readline())
    stdout, stderr = process.communicate(json.dumps([{
        "task_id": request["records"][0]["task_id"], "verdict": "uncertain",
        "confidence": 0.5, "reason_code": "synthetic_only",
    }]) + "\n", timeout=5)
    assert process.returncode == 0, stderr
    assert [json.loads(line)["status"] for line in stdout.splitlines()] == ["committed", "round_complete"]
    assert LunaBulkBundle.open(bundle.root).next_batch(0, limit=1)["cursor"] == 1
