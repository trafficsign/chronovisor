from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chronovisor.core.cofire import neighbors
from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.feedback_ledger import feedback_row_sha256
from chronovisor.recall import cofire
from chronovisor.recall.cofire import build_cofire_graph
from chronovisor.recall.recall_field_schema import session_hash


def test_build_cofire_graph_counts_repeated_context_pairs(tmp_path: Path) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    out_file = tmp_path / "cofire.json"
    rows = [
        {"pages": ["a", "b"]},
        {"context_items": [{"page_id": "a"}, {"page_id": "b"}, {"page_id": "c"}]},
    ]
    log_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    payload = build_cofire_graph(
        log_file=log_file,
        pull_log_file=tmp_path / "missing-pull-log.jsonl",
        output_file=out_file,
        min_count=2,
    )

    assert payload["episodes"] == 2
    assert payload["edges"] == 2
    assert neighbors("a", path=out_file)[0]["page_id"] == "b"


def test_cofire_keeps_used_graph_separate_from_exposure_graph(tmp_path: Path) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    pull_file = tmp_path / "pull-log.jsonl"
    out_file = tmp_path / "cofire.json"
    log_file.write_text(
        json.dumps(
            {
                "decision_id": "decision-1",
                "session_id": "session-1",
                "pages": ["exposure-a", "exposure-b"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pull_file.write_text(
        json.dumps(
            {
                "type": "used",
                "event_id": "event-1",
                "decision_id": "decision-1",
                "session_id": "session-1",
                "page_ids": ["used-a", "used-b"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_cofire_graph(
        log_file=log_file,
        pull_log_file=pull_file,
        output_file=out_file,
        min_count=1,
    )

    assert payload["graphs"]["positive_used"]["graph"] == {}
    assert payload["graphs"]["usage_diagnostic"]["graph"]["used-a"][0]["page_id"] == "used-b"
    assert payload["graphs"]["exposure"]["graph"]["exposure-a"][0]["page_id"] == "exposure-b"


def _write_train_answer_artifact(path: Path, page_hashes: dict[str, str]) -> None:
    manifest = {"split": "train"}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rewards = [
        {
            "episode_id": "episode-1",
            "decision_id": "decision-1",
            "page_id": page_id,
            "content_sha256": digest,
            "reward": 0.2,
            "producer": "verified_answer_pair_v1",
            "session_hash": session_hash("codex", "session-1"),
            "query_sha256": "a" * 64,
            "observed_at": "2026-08-01T00:00:00Z",
        }
        for page_id, digest in page_hashes.items()
    ]
    write_sealed_json(
        path,
        {
            "schema_version": 1,
            "artifact_kind": "locked-answer-on-off-evaluation",
            "status": "passed",
            "production_host_exact_replay_claimed": False,
            "manifest": manifest,
            "samples": 2,
            "confidence_bound": {
                "valid": True,
                "method": "connected-cluster-bootstrap-percentile",
                "clusters": 2,
                "confidence": 0.95,
                "seed": 1729,
                "point": 0.2,
                "lower": 0.1,
            },
            "gates": {"verified": True},
            "page_rewards": rewards,
        },
    )


def test_verified_train_outcomes_build_positive_graph_and_exact_negative_retracts(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.core import feedback_ledger

    pages = {}
    hashes = {}
    for page_id in ("positive-a", "positive-b"):
        page = tmp_path / f"{page_id}.md"
        page.write_text(page_id, encoding="utf-8")
        pages[page_id] = page
        hashes[page_id] = hashlib.sha256(page.read_bytes()).hexdigest()
    monkeypatch.setattr(feedback_ledger, "find_page", lambda page_id: pages.get(page_id))
    monkeypatch.setattr(cofire, "find_page", lambda page_id: pages.get(page_id))
    artifact = tmp_path / "train-answer.json"
    _write_train_answer_artifact(artifact, hashes)
    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text("", encoding="utf-8")

    baseline = build_cofire_graph(
        log_file=tmp_path / "missing-recall.jsonl",
        pull_log_file=tmp_path / "missing-pull.jsonl",
        output_file=tmp_path / "cofire.json",
        answer_outcome_file=artifact,
        feedback_file=feedback,
        minimum_outcome_clusters=2,
        min_count=1,
        write=False,
    )
    assert baseline["graphs"]["positive_used"]["graph"] == {}
    assert baseline["answer_outcome"]["passed"] is False
    monkeypatch.setattr(
        cofire,
        "validate_answer_outcome_artifact",
        lambda *_args, **_kwargs: {
            "passed": True,
            "page_rewards": [
                    {
                        "episode_id": "episode-a",
                        "decision_id": "decision-1",
                        "session_hash": session_hash("codex", "session-1"),
                    "page_id": "positive-a",
                    "page_uid": "uid-a",
                    "content_sha256": hashes["positive-a"],
                    "reward": 0.2,
                },
                    {
                        "episode_id": "episode-a",
                        "decision_id": "decision-1",
                        "session_hash": session_hash("codex", "session-1"),
                    "page_id": "positive-b",
                    "page_uid": "uid-b",
                    "content_sha256": hashes["positive-b"],
                    "reward": 0.2,
                },
            ],
        },
    )
    monkeypatch.setattr(
        cofire,
        "trusted_negative_feedback_rows",
        lambda path, **_kwargs: feedback_ledger.active_feedback_rows(path),
    )

    negative = {
        "kind": "page_ignored",
        "host": "codex",
        "frontier_reviewed": True,
        "label_quality": "strong",
        "content_correction_key": "correction-1",
        "ref": "decision-1",
        "snapshot": {"decision_id": "decision-1", "session_id": "session-1"},
        "negative_pages": ["positive-b"],
        "negative_page_hashes": {"positive-b": hashes["positive-b"]},
    }
    feedback.write_text(json.dumps(negative) + "\n", encoding="utf-8")
    suppressed = build_cofire_graph(
        log_file=tmp_path / "missing-recall.jsonl",
        pull_log_file=tmp_path / "missing-pull.jsonl",
        answer_outcome_file=artifact,
        feedback_file=feedback,
        minimum_outcome_clusters=2,
        min_count=1,
        write=False,
    )
    assert suppressed["graphs"]["positive_used"]["graph"] == {}
    assert suppressed["negative_retractions"][0]["feedback_sha256"] == feedback_row_sha256(negative)

    malformed = {
        "kind": "page_ignored_retracted",
        "target_kind": "page_ignored",
        "content_correction_key": "correction-1",
        "target_feedback_sha256": "0" * 64,
    }
    feedback.write_text(
        json.dumps(negative) + "\n" + json.dumps(malformed) + "\n",
        encoding="utf-8",
    )
    still_suppressed = build_cofire_graph(
        log_file=tmp_path / "missing-recall.jsonl",
        pull_log_file=tmp_path / "missing-pull.jsonl",
        answer_outcome_file=artifact,
        feedback_file=feedback,
        minimum_outcome_clusters=2,
        min_count=1,
        write=False,
    )
    assert still_suppressed["graphs"]["positive_used"]["graph"] == {}

    exact = {
        **malformed,
        "target_feedback_sha256": feedback_row_sha256(negative),
    }
    feedback.write_text(
        json.dumps(negative) + "\n" + json.dumps(exact) + "\n",
        encoding="utf-8",
    )
    restored = build_cofire_graph(
        log_file=tmp_path / "missing-recall.jsonl",
        pull_log_file=tmp_path / "missing-pull.jsonl",
        answer_outcome_file=artifact,
        feedback_file=feedback,
        minimum_outcome_clusters=2,
        min_count=1,
        write=False,
    )
    assert restored["graphs"]["positive_used"]["graph"]["positive-a"][0]["page_id"] == "positive-b"
