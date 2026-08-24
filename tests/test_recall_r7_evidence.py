from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation_store as store

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_r7_evidence_test", ROOT / "src/chronovisor/recall/recall_r7_evidence.py"
)
assert SPEC is not None and SPEC.loader is not None
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


def _id(number: int) -> str:
    return f"{number:064x}"


def _poll(root: Path, stage: str, when: datetime, observation_id: str) -> str:
    poll_id, _, _ = store.write_immutable(
        root / "polls",
        {
            "kind": "r7-live-poll",
            "stage": stage,
            "run_id": _id(100),
            "captured_at": when.isoformat(),
            "monotonic_ns": 1,
            "identities": {},
            "source": {},
            "runtime": {},
            "process": {},
            "health": {},
            "api": {},
            "dom_sha256": _id(200),
            "observation_chain": {"records": 0, "head_sha256": ""},
            "observations_sha256": _id(201),
            "observations": [
                {
                    "observation_id": observation_id,
                    "host": "host-a",
                    "cohort": "cohort-a",
                }
            ],
            "producer": {
                "name": "chronovisor-r7-evidence",
                "version": 1,
                "synthetic_fixture": False,
            },
        },
        schema=EVIDENCE.POLL_SCHEMA,
    )
    return poll_id


def _ledger(root: Path, entries: list[tuple[str, str, datetime, int]]) -> None:
    prior = ""
    lines = []
    for poll_id, stage, when, monotonic_ns in entries:
        row = {
            "schema": EVIDENCE.LEDGER_SCHEMA,
            "namespace": "recall-distillation",
            "poll_id": poll_id,
            "stage": stage,
            "observed_at": when.isoformat(),
            "monotonic_ns": monotonic_ns,
            "previous_sha256": prior,
        }
        row["entry_sha256"] = EVIDENCE._digest(row)
        prior = row["entry_sha256"]
        lines.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    (root / "poll-ledger.jsonl").write_text("\n".join(lines) + "\n")
    store.write_sealed_state(
        root / "poll-ledger-state.json",
        {
            "kind": "r7-poll-ledger-state",
            "count": len(entries),
            "head_sha256": prior,
        },
    )


def test_empty_or_short_real_collector_is_not_certified(tmp_path: Path) -> None:
    assert EVIDENCE.validate_collector(tmp_path)["certification"] is False
    now = datetime(2026, 8, 24, tzinfo=UTC)
    poll_id = _poll(tmp_path, "shadow", now, _id(1))
    _ledger(tmp_path, [(poll_id, "shadow", now, 1)])
    result = EVIDENCE.validate_collector(tmp_path)
    assert result["certification"] is False
    assert result["stages"]["shadow"]["paired"] == 1


def test_ledger_tamper_backward_clock_and_cross_stage_reuse_fail_closed(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 24, tzinfo=UTC)
    first = _poll(tmp_path, "shadow", now, _id(1))
    second = _poll(tmp_path, "5", now + timedelta(seconds=1), _id(1))
    _ledger(
        tmp_path,
        [(first, "shadow", now, 2), (second, "5", now + timedelta(seconds=1), 1)],
    )
    assert (
        EVIDENCE.validate_collector(tmp_path)["certification_reason"]
        == "collector_ledger_invalid"
    )
    _ledger(
        tmp_path,
        [(first, "shadow", now, 1), (second, "5", now + timedelta(seconds=1), 2)],
    )
    result = EVIDENCE.validate_collector(tmp_path)
    assert result["certification_reason"] == "collector_bundle_invalid"
    assert all(stage["certified"] is False for stage in result["stages"].values())
    text = (tmp_path / "poll-ledger.jsonl").read_text()
    (tmp_path / "poll-ledger.jsonl").write_text(
        text.replace('"stage":"shadow"', '"stage":"25"', 1)
    )
    assert (
        EVIDENCE.validate_collector(tmp_path)["certification_reason"]
        == "collector_ledger_invalid"
    )


def test_safe_input_rejects_symlink_and_tamper(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(EVIDENCE.EvidenceError, match="unsafe"):
        EVIDENCE._safe_json(link, "input")
    source.write_text("[]")
    with pytest.raises(EVIDENCE.EvidenceError, match="not object"):
        EVIDENCE._safe_json(source, "input")


def test_readonly_snapshot_does_not_create_missing_runtime_lock(tmp_path: Path) -> None:
    ledger = tmp_path / "shadow-observation-receipts.jsonl"
    lock = ledger.with_suffix(ledger.suffix + ".lock")
    with pytest.raises(EVIDENCE.EvidenceError, match="lock"):
        EVIDENCE._readonly_chain_snapshot(ledger)
    assert not lock.exists()


def test_missing_poll_timestamp_holds_every_stage(tmp_path: Path) -> None:
    now = datetime(2026, 8, 24, tzinfo=UTC)
    poll_id = _poll(tmp_path, "shadow", now, _id(1))
    _ledger(tmp_path, [(poll_id, "shadow", now, 1)])
    path = tmp_path / "polls" / f"{poll_id}.json"
    poll = json.loads(path.read_text())
    poll.pop("captured_at")
    poll["seal_sha256"] = EVIDENCE._digest(
        {key: value for key, value in poll.items() if key != "seal_sha256"}
    )
    path.write_text(json.dumps(poll, sort_keys=True, separators=(",", ":")))
    result = EVIDENCE.validate_collector(tmp_path)
    assert result["certification"] is False
    assert result["certification_reason"] == "collector_poll_timestamp_invalid"
    assert all(stage["certified"] is False for stage in result["stages"].values())


@pytest.mark.parametrize(
    "mutation", ["host", "observations", "producer", "run_id", "stage"]
)
def test_malformed_sealed_poll_never_escapes_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / mutation
    root.mkdir()
    now = datetime(2026, 8, 24, tzinfo=UTC)
    poll_id = _poll(root, "shadow", now, _id(1))
    _ledger(root, [(poll_id, "shadow", now, 1)])
    path = root / "polls" / f"{poll_id}.json"
    poll = json.loads(path.read_text())
    if mutation == "host":
        poll["observations"][0].pop("host")
    elif mutation == "observations":
        poll["observations"] = None
    elif mutation == "producer":
        poll.pop("producer")
    elif mutation == "run_id":
        poll["run_id"] = []
    else:
        poll["stage"] = "not-a-stage"
    poll["seal_sha256"] = EVIDENCE._digest(
        {key: value for key, value in poll.items() if key != "seal_sha256"}
    )
    path.write_text(json.dumps(poll, sort_keys=True, separators=(",", ":")))
    result = EVIDENCE.validate_collector(root)
    assert result["certification"] is False
    assert all(stage["certified"] is False for stage in result["stages"].values())


def test_collector_rejects_nonproduction_root_before_any_runtime_read(
    tmp_path: Path,
) -> None:
    with pytest.raises(EVIDENCE.EvidenceError, match="not the production"):
        EVIDENCE.collect_poll(
            root=tmp_path,
            source_root=tmp_path,
            evidence_root=tmp_path / "evidence",
            stage="shadow",
            run_id=_id(1),
            dashboard_url="http://127.0.0.1:1",
            dom_capture_path=tmp_path / "dom.json",
            direct_url_path=tmp_path / "direct-url.json",
            executable=tmp_path / "worker",
            pid=1,
        )


def test_forged_full_window_and_resealed_flags_never_certify(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    entries: list[tuple[str, str, datetime, int]] = []
    number = 1
    for stage_index, stage in enumerate(EVIDENCE.STAGES):
        stage_start = start + timedelta(days=stage_index * 8)
        for _poll_index, when in enumerate(
            (stage_start, stage_start + timedelta(days=7))
        ):
            rows = [
                {
                    "observation_id": _id(number + offset),
                    "host": "host-a",
                    "cohort": "cohort-a",
                    "decision_sha256": _id(30_000 + number + offset),
                    "session_sha256": _id(40_000 + number + offset),
                    "query_sha256": _id(50_000 + number + offset),
                    "candidate_pool_sha256": _id(60_000 + number + offset),
                    "feature_bytes_sha256": _id(70_000),
                }
                for offset in range(250)
            ]
            poll_id, _, _ = store.write_immutable(
                tmp_path / "polls",
                {
                    "kind": "r7-live-poll",
                    "stage": stage,
                    "run_id": _id(10_000 + stage_index),
                    "captured_at": when.isoformat(),
                    "monotonic_ns": number,
                    "identities": {},
                    "source": {},
                    "runtime": {},
                    "process": {},
                    "health": {},
                    "api": {},
                    "dom_sha256": _id(20_000),
                    "observation_chain": {"records": 2_000, "head_sha256": _id(20_002)},
                    "observations_sha256": _id(20_001),
                    "observations": rows,
                    "producer": {
                        "name": "forged",
                        "version": 99,
                        "synthetic_fixture": False,
                    },
                },
                schema=EVIDENCE.POLL_SCHEMA,
            )
            entries.append((poll_id, stage, when, number))
            number += 250
    _ledger(tmp_path, entries)
    result = EVIDENCE.validate_collector(tmp_path)
    assert result["certification"] is False
    assert (
        result["certification_reason"]
        == "authoritative_runtime_observation_chain_unavailable"
    )
    assert (
        EVIDENCE.validate_collector(tmp_path, root=tmp_path)["certification"] is False
    )
    ledger_path = tmp_path / "poll-ledger.jsonl"
    ledger_path.write_text(ledger_path.read_text().splitlines()[0] + "\n")
    assert (
        EVIDENCE.validate_collector(tmp_path)["certification_reason"]
        == "collector_ledger_invalid"
    )


def test_rollback_never_accepts_forged_post_state_without_r7_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EVIDENCE.store, "CHRONOVISOR_ROOT", tmp_path)
    with pytest.raises(EVIDENCE.EvidenceError, match="authoritative R7 binding"):
        EVIDENCE.validate_rollback(tmp_path, tmp_path / "forged-receipt.json")
