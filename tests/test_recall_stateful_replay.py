from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import stat
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "stateful_replay_test",
    Path(__file__).parents[1] / "scripts/recall_stateful_replay.py",
)
assert SPEC and SPEC.loader
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def fixture_episode(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    content = "合言葉は翡翠です。\n".encode()
    (root / "note.md").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    span = {
        "page_id": "note",
        "page_uid": "fixture-note",
        "content_sha256": digest,
        "content_byte_length": len(content),
        "byte_start": 0,
        "byte_end": len(content),
        "excerpt": content.decode(),
        "excerpt_sha256": digest,
        "truncated": False,
    }
    state = {"topic": "fictional project", "turn": 4}
    return {
        "schema": REPLAY.SCHEMA,
        "episode_id": "fixture",
        "host": "fixture",
        "session_hash": "fixture-session",
        "prompt": "合言葉は何だっけ？",
        "as_of": "2026-08-03T00:00:00Z",
        "index_generation": "fixture-asof-index",
        "policy_sha256": "a" * 64,
        "health": "healthy",
        "memory_required": True,
        "history": [
            {
                "role": "user",
                "content": "計画の続きをお願い",
                "at": "2026-08-02T23:59:59Z",
            }
        ],
        "state_snapshot": {
            "state": state,
            "sha256": REPLAY.canonical_sha256(state),
            "captured_at": "2026-08-02T23:59:59Z",
        },
        "pages": [
            {
                "page_id": "note",
                "relative_path": "note.md",
                "content_sha256": digest,
                "captured_at": "2026-08-01T00:00:00Z",
            }
        ],
        "candidate_page_ids": [],
        "selected_evidence": [],
        "gold_evidence": [span],
    }


def test_on_off_restore_clone_state_and_preserve_source(tmp_path: Path) -> None:
    episode = fixture_episode(tmp_path)
    before = copy.deepcopy(episode)
    prepared = REPLAY.prepare_replay(episode, tmp_path)
    observations = []

    def reader(request: dict) -> str:
        observations.append(copy.deepcopy(request))
        request["state"]["turn"] += 1
        request["history"][0]["content"] = "adapter-local mutation"
        return "翡翠" if "翡翠" in request["context"] else "分からない"

    report = REPLAY.replay(
        prepared, reader=reader, score=lambda answer: answer == "翡翠", seed=12
    )
    assert len(observations) == 3
    assert all(row["state"]["turn"] == 4 for row in observations)
    assert all(row["history"] == before["history"] for row in observations)
    assert all(row["as_of"] == episode["as_of"] for row in observations)
    assert report["failure_categories"] == [
        "candidate_missing",
        "recoverable_with_restored_gold",
    ]
    assert report["promotion_eligible"] is False
    assert report["status"] == "diagnostic_completed"
    assert episode == before
    assert prepared["state"]["turn"] == 4
    assert "翡翠" not in json.dumps(report, ensure_ascii=False)


@pytest.mark.parametrize(
    "damage",
    [
        "future_state",
        "future_history",
        "changed_page",
        "missing_page",
        "bad_span",
        "escape",
    ],
)
def test_historical_binding_never_falls_back_to_current_data(
    tmp_path: Path, damage: str
) -> None:
    episode = fixture_episode(tmp_path)
    if damage == "future_state":
        episode["state_snapshot"]["captured_at"] = "2026-08-04T00:00:00Z"
    elif damage == "future_history":
        episode["history"][0]["at"] = "2026-08-04T00:00:00Z"
    elif damage == "changed_page":
        (tmp_path / "note.md").write_text("new live value")
    elif damage == "missing_page":
        (tmp_path / "note.md").unlink()
    elif damage == "bad_span":
        episode["gold_evidence"][0]["excerpt"] = "different evidence"
    else:
        episode["pages"][0]["relative_path"] = "../note.md"
    with pytest.raises((ValueError, FileNotFoundError)):
        REPLAY.prepare_replay(episode, tmp_path)


def test_failure_categories_distinguish_retrieval_selection_and_reader(
    tmp_path: Path,
) -> None:
    episode = fixture_episode(tmp_path)
    episode["candidate_page_ids"] = ["note"]
    prepared = REPLAY.prepare_replay(episode, tmp_path)
    wrong = {"on": False, "off": False, "restore": False}
    assert REPLAY.failure_categories(prepared, wrong) == ["candidate_not_selected"]
    episode["selected_evidence"] = copy.deepcopy(episode["gold_evidence"])
    prepared = REPLAY.prepare_replay(episode, tmp_path)
    assert REPLAY.failure_categories(prepared, wrong) == [
        "reader_failure_after_correct_injection"
    ]
    episode.update(memory_required=False, gold_evidence=[])
    prepared = REPLAY.prepare_replay(episode, tmp_path)
    assert REPLAY.failure_categories(prepared, wrong) == ["unnecessary_injection"]
    with pytest.raises(ValueError, match="boolean"):
        REPLAY.failure_categories(prepared, {"on": 1, "off": False, "restore": True})


def test_sampling_includes_zero_card_and_degraded_turns(tmp_path: Path) -> None:
    episode = fixture_episode(tmp_path)
    degraded = {**episode, "episode_id": "degraded", "health": "degraded"}
    outside = {**episode, "episode_id": "outside", "as_of": "2026-09-24T00:00:00Z"}
    args = dict(
        start="2026-08-02T00:00:00Z", end="2026-09-24T00:00:00Z", per_stratum=1, seed=4
    )
    result = REPLAY.sample_historical_cohort([outside, degraded, episode], **args)
    assert {row["episode_id"] for row in result} == {"fixture", "degraded"}
    assert all(row["selected_evidence"] == [] for row in result)
    assert (
        REPLAY.sample_historical_cohort([episode, degraded, outside], **args) == result
    )


def test_prepare_cli_private_and_no_overwrite(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    episode = fixture_episode(snapshot)
    source = tmp_path / "episode.json"
    source.write_text(json.dumps(episode, ensure_ascii=False))
    output = tmp_path / "output" / "prepared.json"
    args = [
        "--episode",
        str(source),
        "--episode-sha256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "--snapshot-root",
        str(snapshot),
        "--output",
        str(output),
    ]
    assert REPLAY.main(args) == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    assert json.loads(output.read_text())["promotion_eligible"] is False
    with pytest.raises(FileExistsError):
        REPLAY.main(args)
