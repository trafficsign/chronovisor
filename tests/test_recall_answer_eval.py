from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core.durable_state import seal_object
from chronovisor.raw.raw_segment import append_capture
from chronovisor.recall import recall_answer_eval
from chronovisor.recall.recall_runtime import stable_prompt_hash


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_pair_protocol_is_independent_of_evaluation_run_time() -> None:
    evidence = {
        "seed": 1729,
        "episode_id": "episode-1",
        "episode_sha256": "1" * 64,
        "split_manifest_sha256": "2" * 64,
        "gold_manifest_sha256": "3" * 64,
        "adapter_registry_sha256": "4" * 64,
        "evaluation_kind": "field-e2e-replay",
    }

    first = recall_answer_eval._preregistered_pair_protocol(**evidence)
    second = recall_answer_eval._preregistered_pair_protocol(**evidence)

    assert first == second
    assert first["arm_order"] in (
        ["field_on", "field_off"],
        ["field_off", "field_on"],
    )


def test_field_environment_receipt_rejects_unrendered_bindings(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.recall import recall_runtime

    monkeypatch.setattr(
        recall_runtime,
        "_retained_context_page_ids",
        lambda _context: ["page-0"],
    )
    identity = {
        "adapter_id": "field-test",
        "version": "1",
        "model_sha256": "1" * 64,
        "policy_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "corpus_sha256": "4" * 64,
        "index_sha256": "5" * 64,
        "clone_protocol_sha256": "6" * 64,
        "lkg_base_artifact_sha256": "7" * 64,
        "lkg_base_snapshot_sha256": "8" * 64,
        "effective_field_config_sha256": "9" * 64,
        "candidate_policy_delta_sha256": "a" * 64,
    }
    base_sha = "b" * 64
    bindings = [
        {
            "page_id": f"page-{index}",
            "page_uid": f"uid-{index}",
            "content_sha256": f"{index + 12:064x}",
            "rank": index + 1,
        }
        for index in range(2)
    ]

    def arm(name: str) -> dict:
        effective = (
            recall_answer_eval._canonical_sha(
                {
                    "base_policy_sha256": identity["policy_sha256"],
                    "candidate_policy_delta_sha256": identity[
                        "candidate_policy_delta_sha256"
                    ],
                }
            )
            if name == "candidate_field"
            else identity["policy_sha256"]
        )
        certificates = ["certificate-0", "certificate-1"]
        post = "c" * 64
        return {
            "context": "rendered page-0 only",
            "context_sha256": recall_answer_eval._sha_text("rendered page-0 only"),
            "pre_state_sha256": base_sha,
            "post_state_sha256": post,
            "rollback_state_sha256": base_sha,
            "clone_sha256": "d" * 64,
            "topic_epoch": 1,
            "policy_sha256": identity["policy_sha256"],
            "effective_policy_sha256": effective,
            "config_sha256": identity["config_sha256"],
            "corpus_sha256": identity["corpus_sha256"],
            "index_sha256": identity["index_sha256"],
            "retrieved_page_bindings": bindings,
            "certificate_ids": certificates,
            "commit_ids": [
                recall_answer_eval._field_replay_commit_id(
                    arm_name=name,
                    base_state_sha256=base_sha,
                    post_state_sha256=post,
                    topic_epoch=1,
                    bindings=bindings,
                    certificate_ids=certificates,
                    effective_policy_sha256=effective,
                )
            ],
        }

    def adapter(_prompt: str, _episode: dict, _seed: int) -> dict:
        return {
            "identity": identity,
            "base_state_sha256": base_sha,
            "arms": {
                "candidate_field": arm("candidate_field"),
                "production_teacher": arm("production_teacher"),
            },
        }

    contexts, evidence, error = recall_answer_eval._field_environment_contexts(
        adapter,
        prompt="prompt",
        episode={"episode_id": "episode-1"},
        pair_seed=1,
        identity=identity,
        parent_run_id="e" * 64,
        execution_ledger_file=tmp_path / "execution.jsonl",
    )

    assert contexts == {}
    assert evidence == {}
    assert error == "field_environment_arm_receipt_invalid"


def test_receipt_chunk_requires_segment_v2_commit_identity(
    tmp_path: Path, monkeypatch
) -> None:
    transaction = recall_answer_eval.make_save_transaction(
        host="codex",
        session_file=tmp_path / "session.jsonl",
        session_id="session",
        after_line=0,
        until_line=1,
    )
    legacy_unit = SimpleNamespace(
        is_segment=False,
        commit=None,
        storage="legacy_archive",
        offset=0,
        length=1,
        path=tmp_path / "legacy.tar.zst",
    )

    class FakeStore:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def resolve(self, _raw_id: str):
            return legacy_unit

    monkeypatch.setattr(recall_answer_eval, "RawStore", FakeStore)
    error = recall_answer_eval._durable_receipt_chunk_error(
        {
            "raw_id": f"save-{transaction.idempotency_key}.md",
            "raw_dir": str(tmp_path),
            "storage": "segment-v2",
            "segment_storage": "segment_open",
            "commit": {"idempotency_key": transaction.idempotency_key},
            "offset": 0,
            "length": 1,
            "logical_sha256": hashlib.sha256(b"x").hexdigest(),
            "path": str(legacy_unit.path),
        },
        expected=transaction,
    )

    assert error == "segment_identity_mismatch"


def _scorer_calibration(
    scorer_identity: dict,
    *,
    review_ledger: Path,
    execution_ledger: Path,
    count: int = 40,
    overlap_session: str = "",
) -> dict:
    assert count % 2 == 0
    frozen_at = "2026-06-01T00:00:00Z"
    protocol_sha = scorer_identity["calibration_protocol_sha256"]
    identity_sha = recall_answer_eval._canonical_sha(
        recall_answer_eval._calibration_scorer_identity(scorer_identity)
    )
    calibration_run_id = recall_answer_eval._canonical_sha(
        {
            "artifact_kind": "preregistered-answer-scorer-calibration",
            "frozen_at": frozen_at,
            "scorer_identity_sha256": identity_sha,
            "review_protocol_sha256": protocol_sha,
        }
    )
    cases = []
    for index in range(count):
        pair_arm = "a" if index % 2 == 0 else "b"
        scores = {
            dimension: 0.8 if pair_arm == "a" else 0.2
            for dimension in recall_answer_eval.ANSWER_DIMENSIONS
        }
        case = {
            "case_id": f"calibration-{index}",
            "session_hash": overlap_session
            if index == 0 and overlap_session
            else f"calibration-session-{index}",
            "query_sha256": f"{index + 1000:064x}",
            "evidence_sha256": f"{index + 2000:064x}",
            "pair_id": f"pair-{index // 2}",
            "pair_arm": pair_arm,
            "human_reviewed": scores,
            "scorer_scores": scores,
        }
        review = recall_answer_eval.append_answer_review_receipt(
            kind="scorer_calibration_case_review",
            payload={
                "case_id": case["case_id"],
                "session_hash": case["session_hash"],
                "query_sha256": case["query_sha256"],
                "evidence_sha256": case["evidence_sha256"],
                "human_reviewed": scores,
            },
            reviewer_kind="human_reviewer",
            reviewed_at="2026-05-01T00:00:00Z",
            protocol_sha256=protocol_sha,
            ledger_file=review_ledger,
        )
        execution = recall_answer_eval.append_answer_execution_receipt(
            kind="calibration_scorer_call",
            adapter_identity_sha256=identity_sha,
            parent_run_id=calibration_run_id,
            input_payload={
                "case_id": case["case_id"],
                "query_sha256": case["query_sha256"],
                "evidence_sha256": case["evidence_sha256"],
            },
            output_payload={"dimensions": scores},
            started_at="2026-05-01T00:00:01Z",
            completed_at="2026-05-01T00:00:02Z",
            ledger_file=execution_ledger,
        )
        cases.append(
            {
                **case,
                "review_receipt_sha256": review["receipt_sha256"],
                "execution_receipt_sha256": execution["receipt_sha256"],
            }
        )
    return recall_answer_eval.build_scorer_calibration_artifact(
        cases=cases,
        scorer_identity=scorer_identity,
        frozen_at=frozen_at,
        review_protocol_sha256=protocol_sha,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
    )


def _install_deterministic_builtin_field_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict[str, list[dict]]]:
    """Keep the builtin replay real while replacing its external search seams."""

    from chronovisor.recall import recall_field, recall_field_candidate, recall_runtime
    from chronovisor.recall.recall_field_store import RecallFieldStore

    calls: dict[str, list[dict]] = {"turn": [], "queue": []}
    pages: dict[str, Path] = {}

    class ReplayState:
        def __init__(self, session_hash: str) -> None:
            self.session_hash = session_hash
            self.commits: list[list[str]] = []
            self.updated_at_epoch = 0.0

        def to_dict(self) -> dict:
            return {
                "session_hash": self.session_hash,
                "commits": self.commits,
                "updated_at_epoch": self.updated_at_epoch,
            }

    def page_id_for_prompt(prompt: str) -> str:
        page_id = f"replay-{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"
        if page_id not in pages:
            path = tmp_path / f"{page_id}.md"
            path.write_text(f"# {page_id}\n\nverified replay evidence\n", encoding="utf-8")
            pages[page_id] = path
        return page_id

    def run_field_turn(**kwargs) -> dict:
        calls["turn"].append(dict(kwargs))
        return {
            "candidate_page_ids": [page_id_for_prompt(str(kwargs["prompt"]))],
            "topic_epoch": 7,
        }

    def queue_teacher_commits(**kwargs) -> dict:
        page_ids = list(kwargs["page_ids"])
        certificate_ids = dict(kwargs["certificate_ids"])
        calls["queue"].append(
            {"page_ids": page_ids, "certificate_ids": certificate_ids}
        )
        kwargs["store"].state.commits.append(page_ids)
        return {"queued": len(page_ids)}

    def candidate_verify(prompt: str, page_ids: list[str], **_kwargs):
        del prompt
        return (
            [SimpleNamespace(page_id=page_ids[0], replay_arm="candidate")],
            {"status": "verified"},
        )

    def teacher_search(queries: list[str], *_args, **_kwargs):
        return (
            [
                SimpleNamespace(
                    page_id=page_id_for_prompt(queries[0]),
                    replay_arm="teacher",
                )
            ],
            "test",
        )

    def collect_context(*_args, candidates: list[object], **_kwargs):
        candidate = candidates[0]
        page_id = str(candidate.page_id)  # type: ignore[attr-defined]
        arm = str(candidate.replay_arm)  # type: ignore[attr-defined]
        return (
            [
                recall_runtime.ContextItem(
                    page_id=page_id,
                    title=f"{arm} evidence",
                    updated="2026-05-01T00:00:00Z",
                    score=1.0,
                    uid=f"uid-{page_id}",
                    snippets=[f"{arm} support"],
                    certificate_id=f"cert-{arm}-{page_id}",
                    evidence_kind="rich",
                    source_line=1,
                )
            ],
            {"status": "verified"},
        )

    monkeypatch.setattr(
        RecallFieldStore,
        "load",
        lambda _self, session_hash: ReplayState(session_hash),
    )
    monkeypatch.setattr(recall_field, "run_field_turn", run_field_turn)
    monkeypatch.setattr(recall_field, "queue_teacher_commits", queue_teacher_commits)
    monkeypatch.setattr(recall_field_candidate, "_verify", candidate_verify)
    monkeypatch.setattr(recall_runtime, "search_candidates", teacher_search)
    monkeypatch.setattr(recall_runtime, "collect_certified_context", collect_context)
    monkeypatch.setattr(
        recall_runtime, "page_uid_for_id", lambda page_id: f"uid-{page_id}"
    )
    monkeypatch.setattr(recall_answer_eval, "find_page", lambda page_id: pages.get(page_id))
    identity = recall_answer_eval.builtin_field_environment_identity()
    return identity, calls


def test_builtin_field_environment_replay_uses_shared_production_seams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, calls = _install_deterministic_builtin_field_replay(tmp_path, monkeypatch)
    execution_ledger = tmp_path / "execution.jsonl"
    prompt = "retained replay prompt"
    contexts, evidence, error = recall_answer_eval._field_environment_contexts(
        recall_answer_eval.builtin_field_environment_replay,
        prompt=prompt,
        episode={"episode_id": "locked-episode", "session_hash": "session-locked"},
        pair_seed=1729,
        identity=identity,
        parent_run_id="a" * 64,
        execution_ledger_file=execution_ledger,
    )

    page_id = f"replay-{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"
    assert error == ""
    assert len(calls["turn"]) == 2
    assert len(calls["queue"]) == 2
    assert all(call["page_ids"] == [page_id] for call in calls["queue"])
    assert all(call["certificate_ids"] for call in calls["queue"])
    assert set(contexts) == {"candidate_field", "production_teacher"}
    for arm in evidence["arms"].values():
        assert arm["retrieved_page_bindings"][0]["page_id"] == page_id
        assert len(arm["certificate_ids"]) == 1
        assert len(arm["commit_ids"]) == 1
    assert (
        evidence["arms"]["candidate_field"]["effective_policy_sha256"]
        != evidence["arms"]["production_teacher"]["effective_policy_sha256"]
    )
    assert recall_answer_eval._execution_receipt_error(
        receipt_sha256=evidence["execution_receipt_sha256"],
        expected_kind="field_environment_replay",
        expected_adapter_identity_sha256=recall_answer_eval._canonical_sha(identity),
        expected_parent_run_id="a" * 64,
        expected_input_payload={
            "episode_id": "locked-episode",
            "prompt_sha256": recall_answer_eval._sha_text(prompt),
            "pair_seed": 1729,
        },
        expected_output_payload={
            key: value
            for key, value in evidence.items()
            if key != "execution_receipt_sha256"
        },
        ledger_file=execution_ledger,
    ) == ""


def test_real_train_and_locked_artifact_set_reaches_growth_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_growth

    evaluated_at = "2026-08-01T12:00:00Z"
    monkeypatch.setattr(recall_answer_eval, "_now_utc", lambda: evaluated_at)
    field_identity, _calls = _install_deterministic_builtin_field_replay(
        tmp_path, monkeypatch
    )
    episodes = tmp_path / "episodes.jsonl"
    review_ledger = tmp_path / "review.jsonl"
    execution_ledger = tmp_path / "execution.jsonl"
    turns: dict[str, SimpleNamespace] = {}
    rows: list[dict] = []
    observed = datetime(2024, 1, 1, tzinfo=UTC)
    for index in range(210):
        episode_id = f"episode-{index:03d}"
        prompt = f"artifact-set prompt {index}"
        page_id = f"history-page-{index:03d}"
        unsigned = {
            "schema_version": 1,
            "episode_id": episode_id,
            "binding_status": "verified",
            "exact_used_subset": True,
            "session_hash": f"session-{index:03d}",
            "prompt_sha256": recall_answer_eval._sha_text(prompt),
            "decision_id": f"decision-{index:03d}",
            "used_page_ids": [page_id],
            "injected_page_ids": [page_id],
            "page_content_sha256": {
                page_id: recall_answer_eval._sha_text(f"history content {index}")
            },
            "page_uids": {page_id: f"uid-{page_id}"},
            "observed_at": (observed + timedelta(days=3 * index))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "answer_sha256": recall_answer_eval._sha_text(
                f"production answer {index}"
            ),
            "raw_ref": {},
        }
        unsigned["episode_sha256"] = recall_answer_eval._canonical_sha(unsigned)
        rows.append(unsigned)
        turns[episode_id] = SimpleNamespace(
            prompt=prompt,
            assistant_response=f"production answer {index}",
        )
    _write_rows(episodes, rows)
    monkeypatch.setattr(
        recall_answer_eval,
        "_load_bound_turn",
        lambda episode: (turns[str(episode["episode_id"])], ""),
    )
    monkeypatch.setattr(
        recall_answer_eval,
        "_context_for_episode",
        lambda _episode: ("historical candidate evidence", ""),
    )

    entries = [recall_answer_eval._episode_manifest_entry(row) for row in rows]
    assignments = recall_answer_eval._assign_episode_splits(entries)
    split_entries = [
        {**entry, "split": assignments[str(entry["episode_id"])]}
        for entry in entries
    ]
    assert sum(entry["split"] == "train" for entry in split_entries) == 147
    assert sum(entry["split"] == "locked-test" for entry in split_entries) == 20
    split_payload = {
        "schema_version": recall_answer_eval.ANSWER_SPLIT_SCHEMA_VERSION,
        "artifact_kind": "answer-preregistered-split-manifest",
        "frozen_at": "2026-06-01T00:00:00Z",
        "embargo_seconds": recall_answer_eval.AUTHORITY_EMBARGO_SECONDS,
        "strategy": "connected-components-chronological-70-20-10",
        "component_keys": [
            "session_hash",
            "query_sha256",
            "page_id",
            "page_uid",
            "content_sha256",
        ],
        "episode_ledger_manifest_sha256": recall_answer_eval.manifest_sha256(
            [entry["episode_sha256"] for entry in split_entries]
        ),
        "entries": split_entries,
    }
    split_payload["epoch_id"] = recall_answer_eval._split_epoch_id(split_payload)
    split_manifest = seal_object(split_payload)
    rubric_sha = "c" * 64
    review_protocol_sha = "e" * 64

    def gold_manifest(split: str) -> dict:
        gold_entries = []
        for entry in split_entries:
            if entry["split"] != split:
                continue
            episode_id = str(entry["episode_id"])
            evidence = {"facts": [f"reviewed fact for {episode_id}"]}
            gold_answer = f"gold answer {episode_id}"
            evidence_sha = recall_answer_eval._canonical_sha(
                {
                    "episode_id": episode_id,
                    "gold_answer": gold_answer,
                    "evidence": evidence,
                    "rubric_sha256": rubric_sha,
                }
            )
            receipt = recall_answer_eval.append_answer_review_receipt(
                kind="gold_entry_review",
                payload={
                    "episode_id": episode_id,
                    "gold_answer_sha256": recall_answer_eval._sha_text(gold_answer),
                    "evidence_sha256": evidence_sha,
                    "rubric_sha256": rubric_sha,
                },
                reviewer_kind="human_reviewer",
                reviewed_at="2026-05-01T00:00:00Z",
                protocol_sha256=review_protocol_sha,
                ledger_file=review_ledger,
            )
            gold_entries.append(
                {
                    "episode_id": episode_id,
                    "gold_answer": gold_answer,
                    "evidence": evidence,
                    "evidence_sha256": evidence_sha,
                    "review_provenance": {
                        "source_kind": "human_review",
                        "reviewer_receipt_sha256": receipt["receipt_sha256"],
                        "reviewed_at": "2026-05-01T00:00:00Z",
                    },
                }
            )
        return seal_object(
            {
                "schema_version": 1,
                "artifact_kind": "immutable-answer-gold-manifest",
                "frozen_at": "2026-06-01T00:00:00Z",
                "gold_id": f"artifact-set-{split}",
                "gold_family_id": "artifact-set-family",
                "version": "1",
                "review_protocol_sha256": review_protocol_sha,
                "rubric_sha256": rubric_sha,
                "entries": gold_entries,
            }
        )

    train_gold = gold_manifest("train")
    locked_gold = gold_manifest("locked-test")
    runner_identity = {
        "runner_id": "artifact-set-runner",
        "model": "fixture-model",
        "system_sha256": "a" * 64,
        "sampler_sha256": "b" * 64,
        "policy_sha256": "d" * 64,
    }
    scorer_base = {
        "scorer_id": "artifact-set-scorer",
        "version": "1",
        "model": "fixture-scorer",
        "system_sha256": "1" * 64,
        "sampler_sha256": "2" * 64,
        "policy_sha256": "3" * 64,
        "rubric_sha256": rubric_sha,
        "calibration_protocol_sha256": review_protocol_sha,
    }
    train_scorer_identity = {
        **scorer_base,
        "evidence_manifest_sha256": train_gold["seal_sha256"],
    }
    locked_scorer_identity = {
        **scorer_base,
        "evidence_manifest_sha256": locked_gold["seal_sha256"],
    }
    active_scorer_identity = dict(train_scorer_identity)

    def runner(_prompt: str, context: str, generation: dict) -> dict:
        return {
            "answer": "better" if "candidate" in context else "worse",
            "identity": runner_identity,
            "reset_receipt": {
                "seed": generation["seed"],
                "base_state_sha256": generation["base_state_sha256"],
                "reset_protocol_sha256": runner_identity["policy_sha256"],
            },
        }

    def scorer(_prompt: str, answer: str, gold: dict, scoring: dict) -> dict:
        score = 1.0 if answer == "better" else 0.0
        return {
            "identity": active_scorer_identity,
            "evidence_sha256": gold["evidence_sha256"],
            "reset_receipt": {
                "seed": scoring["seed"],
                "base_state_sha256": scoring["base_state_sha256"],
                "reset_protocol_sha256": active_scorer_identity["policy_sha256"],
            },
            "dimensions": {
                dimension: score for dimension in recall_answer_eval.ANSWER_DIMENSIONS
            },
        }

    scorer_calibration = _scorer_calibration(
        train_scorer_identity,
        review_ledger=review_ledger,
        execution_ledger=execution_ledger,
    )
    registry_entries = []
    for kind, adapter_id, adapter, identity in (
        ("runner", runner_identity["runner_id"], runner, runner_identity),
        ("scorer", scorer_base["scorer_id"], scorer, train_scorer_identity),
        (
            "field_environment",
            field_identity["adapter_id"],
            recall_answer_eval.builtin_field_environment_replay,
            field_identity,
        ),
    ):
        registry_entry = {
            "kind": kind,
            "adapter_id": adapter_id,
            "callable_sha256": recall_answer_eval.adapter_callable_sha256(adapter),
            "identity_sha256": recall_answer_eval._canonical_sha(
                recall_answer_eval._adapter_registry_identity(kind, identity)
            ),
        }
        registry_entry["entry_sha256"] = recall_answer_eval._canonical_sha(
            registry_entry
        )
        registry_entries.append(registry_entry)
    adapter_registry = seal_object(
        {
            "schema_version": 1,
            "artifact_kind": "answer-authority-adapter-registry",
            "frozen_at": "2026-06-01T00:00:00Z",
            "entries": registry_entries,
        }
    )
    registry_file = tmp_path / "adapter-registry.json"
    registry_file.write_text(json.dumps(adapter_registry), encoding="utf-8")
    train_file = tmp_path / "train-answer.json"
    locked_file = tmp_path / "locked-answer.json"
    train = recall_answer_eval.evaluate_answer_episodes(
        runner=runner,
        scorer=scorer,
        runner_identity=runner_identity,
        scorer_identity=train_scorer_identity,
        episode_file=episodes,
        output_file=train_file,
        split_manifest=split_manifest,
        gold_manifest=train_gold,
        scorer_calibration=scorer_calibration,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=registry_file,
        min_independent_samples=20,
        split="train",
        evaluation_kind="historical-context-utility",
    )
    active_scorer_identity.clear()
    active_scorer_identity.update(locked_scorer_identity)
    locked = recall_answer_eval.evaluate_answer_episodes(
        runner=runner,
        scorer=scorer,
        runner_identity=runner_identity,
        scorer_identity=locked_scorer_identity,
        field_environment_replay=recall_answer_eval.builtin_field_environment_replay,
        field_environment_identity=field_identity,
        episode_file=episodes,
        output_file=locked_file,
        split_manifest=split_manifest,
        gold_manifest=locked_gold,
        scorer_calibration=scorer_calibration,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=registry_file,
        min_independent_samples=20,
        split="locked-test",
        evaluation_kind="field-e2e-replay",
    )

    assert train["status"] == "passed"
    assert locked["status"] == "passed"
    artifact_set = recall_answer_eval.validate_answer_artifact_set(
        train=train_file,
        locked=locked_file,
        episode_file=episodes,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=registry_file,
    )
    assert artifact_set["passed"] is True
    assert artifact_set["harness_same"] is True
    assert artifact_set["train_samples"] == 147
    assert artifact_set["locked_samples"] == 20

    recall_log = tmp_path / "recall.jsonl"
    pull_log = tmp_path / "pull.jsonl"
    candidate_trace = tmp_path / "candidate.jsonl"
    retrieval_artifact = tmp_path / "retrieval.json"
    for path in (recall_log, pull_log, candidate_trace):
        path.write_text("", encoding="utf-8")
    retrieval_artifact.write_text("{}", encoding="utf-8")
    label_result = {
        "labels": [],
        "counts": {
            "scope": "train",
            "strong_positive": 0,
            "strong_positive_sessions": 0,
            "total": 0,
        },
        "gates": {"field_learning_allowed": False},
    }
    integrity = {
        "passed": True,
        "session_leakage": 0,
        "query_leakage": 0,
        "page_leakage": 0,
        "content_leakage": 0,
        "timestamp_leakage": 0,
        "embargo_leakage": 0,
    }
    monkeypatch.setattr(
        recall_growth, "materialize_label_ledger", lambda **_kwargs: label_result
    )
    monkeypatch.setattr(recall_growth, "split_integrity", lambda _rows: integrity)
    monkeypatch.setattr(
        recall_growth,
        "decide_learning_update",
        lambda *, current, **_kwargs: {
            "status": "held",
            "reason": "test",
            "policy": current,
            "field_learning_allowed": False,
            "calibration_allowed": False,
        },
    )
    monkeypatch.setattr(
        recall_growth,
        "retrieval_locked_e2e_status",
        lambda _path: {
            "passed": True,
            "environment_epoch_sha256": recall_answer_eval._canonical_sha(
                field_identity
            ),
        },
    )
    growth = recall_growth.run_growth_cycle(
        state_file=tmp_path / "growth-state.json",
        history_file=tmp_path / "growth-history.jsonl",
        candidate_trace_file=candidate_trace,
        promotion_file=tmp_path / "promotion.json",
        locked_e2e_file=retrieval_artifact,
        locked_answer_eval_file=locked_file,
        train_answer_eval_file=train_file,
        answer_episode_file=episodes,
        answer_review_ledger_file=review_ledger,
        answer_execution_ledger_file=execution_ledger,
        answer_adapter_registry=registry_file,
        label_inputs={
            "recall_log_file": recall_log,
            "pull_log_file": pull_log,
        },
        now=datetime(2026, 8, 1, 13, tzinfo=UTC),
    )
    assert growth["gates"]["train_answer_e2e"] is True
    assert growth["gates"]["locked_answer_e2e"] is True
    assert growth["gates"]["answer_artifact_set"] is True


@pytest.mark.parametrize("host", ["codex", "claude-code"])
def test_multichunk_receipt_binds_exact_chunks_and_survives_segment_append(
    tmp_path: Path,
    host: str,
) -> None:
    raw_dir = tmp_path / "raw"
    session = tmp_path / f"{host}.jsonl"
    session.write_bytes(b"{}\n{}\n{}\n{}\n")
    session_id = f"{host}-session"
    session_key = recall_answer_eval.save_session_key(
        host=host, session_file=session, session_id=session_id
    )
    results = []
    for after, until, source in ((0, 2, b"{}\n{}\n"), (2, 4, b"{}\n{}\n")):
        transaction = recall_answer_eval.make_save_transaction(
            host=host,
            session_file=session,
            session_id=session_id,
            after_line=after,
            until_line=until,
        )
        results.append(
            append_capture(
                raw_dir=raw_dir,
                raw_id=f"save-{transaction.idempotency_key}.md",
                idempotency_key=transaction.idempotency_key,
                host=host,
                session_key=session_key,
                session_id=session_id,
                source_file=session,
                after_line=after,
                until_line=until,
                source_bytes=source,
                record_count=2,
            ).to_result()
        )
    receipt, error = recall_answer_eval._verified_save_receipt(
        host=host,
        save_output={
            "status": "saved",
            "session_file": str(session),
            "session_id": session_id,
            "after_line": 0,
            "scanned_until_line": 4,
            "chunk_count": 2,
            "save_result": results[0],
            "save_results": results,
        },
        session_file=session,
        session_id=session_id,
        raw_dir=raw_dir,
    )
    assert error == ""
    assert [(row["after_line"], row["until_line"]) for row in receipt["chunks"]] == [
        (0, 2),
        (2, 4),
    ]
    assert recall_answer_eval._receipt_error(
        receipt,
        host=host,
        session_file=session,
        session_id=session_id,
        user_line=3,
        assistant_line=4,
    ) == ""

    recovered_chain, recovered_error = recall_answer_eval._verified_save_receipt(
        host=host,
        save_output={
            "status": "saved",
            "session_file": str(session),
            "session_id": session_id,
            "after_line": 2,
            "scanned_until_line": 4,
            "chunk_count": 1,
            "save_result": results[1],
            "recovered_save": {
                "idempotency_key": receipt["chunks"][0]["idempotency_key"],
                "until_line": 2,
            },
        },
        session_file=session,
        session_id=session_id,
        raw_dir=raw_dir,
    )
    assert recovered_error == ""
    assert [row["after_line"] for row in recovered_chain["chunks"]] == [0, 2]

    later = recall_answer_eval.make_save_transaction(
        host=host,
        session_file=session,
        session_id=session_id,
        after_line=4,
        until_line=6,
    )
    append_capture(
        raw_dir=raw_dir,
        raw_id=f"save-{later.idempotency_key}.md",
        idempotency_key=later.idempotency_key,
        host=host,
        session_key=session_key,
        session_id=session_id,
        source_file=session,
        after_line=4,
        until_line=6,
        source_bytes=b"{}\n{}\n",
        record_count=2,
    )
    assert recall_answer_eval._receipt_error(
        receipt,
        host=host,
        session_file=session,
        session_id=session_id,
        user_line=1,
        assistant_line=2,
    ) == ""


def test_capture_all_complete_turns_is_exact_once_and_binds_used_subset(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "session.jsonl"
    session.write_text("u1\na1\nu2\na2\n", encoding="utf-8")
    page_a = tmp_path / "a.md"
    page_b = tmp_path / "b.md"
    page_a.write_text("alpha", encoding="utf-8")
    page_b.write_text("beta", encoding="utf-8")
    records = [
        SimpleNamespace(role="user", line=1, text="prompt one", timestamp="2026-08-01T00:00:00Z"),
        SimpleNamespace(role="assistant", line=2, text="answer one", timestamp="2026-08-01T00:00:01Z"),
        SimpleNamespace(role="user", line=3, text="prompt two", timestamp="2026-08-01T00:01:00Z"),
        SimpleNamespace(role="assistant", line=4, text="answer two", timestamp="2026-08-01T00:01:01Z"),
    ]
    monkeypatch.setattr(
        recall_answer_eval,
        "_extract",
        lambda _host, _path: SimpleNamespace(
            records=records, session_id="session", cwd="/repo"
        ),
    )
    monkeypatch.setattr(
        recall_answer_eval,
        "find_page",
        lambda page_id: {"a": page_a, "b": page_b}.get(page_id),
    )
    recalls = tmp_path / "recall.jsonl"
    pulls = tmp_path / "pull.jsonl"
    _write_rows(
        recalls,
        [
            {
                "decision_id": "d1",
                "host": "codex",
                "session_id": "session",
                "prompt_hash": stable_prompt_hash("prompt one"),
                "ts": "2026-08-01T00:00:00Z",
                "pages": ["a"],
                "context_items": [
                    {
                        "page_id": "a",
                        "page_uid": "uid-a",
                        "content_sha256": hashlib.sha256(page_a.read_bytes()).hexdigest(),
                    }
                ],
            },
            {
                "decision_id": "d2",
                "host": "codex",
                "session_id": "session",
                "prompt_hash": stable_prompt_hash("prompt two"),
                "ts": "2026-08-01T00:01:00Z",
                "pages": ["b"],
                "context_items": [
                    {
                        "page_id": "b",
                        "page_uid": "uid-b",
                        "content_sha256": hashlib.sha256(page_b.read_bytes()).hexdigest(),
                    }
                ],
            },
        ],
    )
    _write_rows(
        pulls,
        [
            {"type": "used", "event_id": "e1", "decision_id": "d1", "session_id": "session", "page_ids": ["a"]},
            {"type": "used", "event_id": "e2", "decision_id": "d2", "session_id": "session", "page_ids": ["b"]},
        ],
    )
    episodes = tmp_path / "episodes.jsonl"
    cursor = tmp_path / "cursor.json"
    transaction = recall_answer_eval.make_save_transaction(
        host="codex",
        session_file=session,
        session_id="session",
        after_line=0,
        until_line=4,
    )
    receipt = {
        "status": "saved",
        "session_key": transaction.session_key,
        "after_line": 0,
        "until_line": 4,
        "chunks": [
            {
                "idempotency_key": transaction.idempotency_key,
                "session_key": transaction.session_key,
                "after_line": 0,
                "until_line": 4,
                "raw_id": f"save-{transaction.idempotency_key}.md",
                "raw_dir": str(tmp_path),
                "storage": "legacy_file",
                "offset": 0,
                "length": 1,
                "logical_sha256": "f" * 64,
                "path": str(tmp_path / "receipt.md"),
            }
        ],
    }
    receipt["receipt_manifest_sha256"] = recall_answer_eval._canonical_sha(receipt)
    monkeypatch.setattr(
        recall_answer_eval,
        "_verified_save_receipt",
        lambda **_kwargs: (receipt, ""),
    )
    monkeypatch.setattr(
        recall_answer_eval, "_durable_receipt_chunk_error", lambda *_args, **_kwargs: ""
    )
    save_output = {
        "status": "saved",
        "session_file": str(session),
        "session_id": "session",
    }

    first = recall_answer_eval.capture_session_answer_episodes(
        host="codex",
        session_file=session,
        episode_file=episodes,
        cursor_file=cursor,
        recall_log_file=recalls,
        pull_log_file=pulls,
        save_output=save_output,
    )
    second = recall_answer_eval.capture_session_answer_episodes(
        host="codex",
        session_file=session,
        episode_file=episodes,
        cursor_file=cursor,
        recall_log_file=recalls,
        pull_log_file=pulls,
        save_output=save_output,
    )

    rows = [json.loads(line) for line in episodes.read_text().splitlines()]
    assert first["captured"] == 2
    assert second["captured"] == 0
    assert {row["decision_id"] for row in rows} == {"d1", "d2"}
    assert all(row["exact_used_subset"] is True for row in rows)
    assert all("answer one" not in json.dumps(row) for row in rows)


def test_paired_fake_runner_is_sealed_and_missing_seams_hold(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        recall_answer_eval, "_now_utc", lambda: "2026-08-01T12:00:00Z"
    )
    episodes = tmp_path / "episodes.jsonl"
    review_ledger = tmp_path / "review.jsonl"
    execution_ledger = tmp_path / "execution.jsonl"
    rows = []
    turns = {}
    for index in range(2):
        unsigned = {
            "schema_version": 1,
            "episode_id": f"episode-{index}",
            "binding_status": "verified",
            "exact_used_subset": True,
            "session_hash": f"session-{index}",
            "prompt_sha256": f"{index + 1:064x}",
            "decision_id": f"decision-{index}",
            "used_page_ids": [f"page-{index}"],
            "injected_page_ids": [f"page-{index}"],
            "page_content_sha256": {f"page-{index}": f"{index + 2:064x}"},
            "page_uids": {f"page-{index}": f"uid-{index}"},
            "observed_at": f"2026-07-{index + 1:02d}T00:00:00Z",
            "answer_sha256": f"{index + 3:064x}",
            "raw_ref": {},
        }
        unsigned["episode_sha256"] = recall_answer_eval._canonical_sha(unsigned)
        rows.append(unsigned)
        turns[unsigned["episode_id"]] = SimpleNamespace(
            prompt=f"prompt-{index}", assistant_response="reference"
        )
    _write_rows(episodes, rows)
    monkeypatch.setattr(
        recall_answer_eval,
        "_load_bound_turn",
        lambda episode: (turns[episode["episode_id"]], ""),
    )
    monkeypatch.setattr(recall_answer_eval, "_context_for_episode", lambda _episode: ("context", ""))
    runner_identity = {
        "runner_id": "fake",
        "model": "fake-model",
        "system_sha256": "a" * 64,
        "sampler_sha256": "b" * 64,
        "policy_sha256": "d" * 64,
    }
    split_entries = [
        {
            **recall_answer_eval._episode_manifest_entry(row),
            "split": "train",
        }
        for row in rows
    ]
    split_payload = {
        "schema_version": recall_answer_eval.ANSWER_SPLIT_SCHEMA_VERSION,
        "artifact_kind": "answer-preregistered-split-manifest",
        "frozen_at": "2026-06-01T00:00:00Z",
        "embargo_seconds": recall_answer_eval.AUTHORITY_EMBARGO_SECONDS,
        "strategy": "connected-components-chronological-70-20-10",
        "component_keys": [
            "session_hash", "query_sha256", "page_id", "page_uid", "content_sha256"
        ],
        "episode_ledger_manifest_sha256": recall_answer_eval.manifest_sha256(
            [entry["episode_sha256"] for entry in split_entries]
        ),
        "entries": split_entries,
    }
    split_payload["epoch_id"] = recall_answer_eval._split_epoch_id(split_payload)
    split_manifest = seal_object(split_payload)
    rubric_sha = "c" * 64
    review_protocol_sha = "e" * 64
    gold_entries = []
    for row in rows:
        evidence = {"facts": [f"fact-{row['episode_id']}"]}
        gold_answer = f"gold-{row['episode_id']}"
        evidence_sha = recall_answer_eval._canonical_sha(
            {
                "episode_id": row["episode_id"],
                "gold_answer": gold_answer,
                "evidence": evidence,
                "rubric_sha256": rubric_sha,
            }
        )
        review = recall_answer_eval.append_answer_review_receipt(
            kind="gold_entry_review",
            payload={
                "episode_id": row["episode_id"],
                "gold_answer_sha256": recall_answer_eval._sha_text(gold_answer),
                "evidence_sha256": evidence_sha,
                "rubric_sha256": rubric_sha,
            },
            reviewer_kind="human_reviewer",
            reviewed_at="2026-05-01T00:00:00Z",
            protocol_sha256=review_protocol_sha,
            ledger_file=review_ledger,
        )
        gold_entries.append(
            {
                "episode_id": row["episode_id"],
                "gold_answer": gold_answer,
                "evidence": evidence,
                "evidence_sha256": evidence_sha,
                "review_provenance": {
                    "source_kind": "human_review",
                    "reviewer_receipt_sha256": review["receipt_sha256"],
                    "reviewed_at": "2026-05-01T00:00:00Z",
                },
            }
        )
    gold_manifest = seal_object(
        {
            "schema_version": 1,
            "artifact_kind": "immutable-answer-gold-manifest",
            "frozen_at": "2026-06-01T00:00:00Z",
                "gold_id": "test-gold",
                "gold_family_id": "test-gold-family",
                "version": "1",
                "review_protocol_sha256": review_protocol_sha,
            "rubric_sha256": rubric_sha,
            "entries": gold_entries,
        }
    )
    scorer_identity = {
        "scorer_id": "fake",
        "version": "1",
        "model": "fake-scorer-model",
        "system_sha256": "1" * 64,
        "sampler_sha256": "2" * 64,
        "policy_sha256": "3" * 64,
        "rubric_sha256": rubric_sha,
        "evidence_manifest_sha256": gold_manifest["seal_sha256"],
        "calibration_protocol_sha256": review_protocol_sha,
    }
    scorer_calibration = _scorer_calibration(
        scorer_identity,
        review_ledger=review_ledger,
        execution_ledger=execution_ledger,
    )
    generations: list[dict] = []

    def runner(_prompt: str, context: str, generation: dict) -> dict:
        generations.append(dict(generation))
        return {
            "answer": "better" if context else "worse",
            "identity": runner_identity,
            "reset_receipt": {
                "seed": generation["seed"],
                "base_state_sha256": generation["base_state_sha256"],
                "reset_protocol_sha256": runner_identity["policy_sha256"],
            },
        }

    def scorer(_prompt: str, answer: str, gold: dict, scoring: dict) -> dict:
        score = 1.0 if answer == "better" else 0.0
        return {
            "identity": scorer_identity,
            "evidence_sha256": gold["evidence_sha256"],
            "reset_receipt": {
                "seed": scoring["seed"],
                "base_state_sha256": scoring["base_state_sha256"],
                "reset_protocol_sha256": scorer_identity["policy_sha256"],
            },
            "dimensions": {dimension: score for dimension in recall_answer_eval.ANSWER_DIMENSIONS},
        }

    registry_entries = []
    for kind, adapter_id, adapter, identity in (
        ("runner", runner_identity["runner_id"], runner, runner_identity),
        ("scorer", scorer_identity["scorer_id"], scorer, scorer_identity),
    ):
        entry = {
            "kind": kind,
            "adapter_id": adapter_id,
            "callable_sha256": recall_answer_eval.adapter_callable_sha256(adapter),
            "identity_sha256": recall_answer_eval._canonical_sha(
                recall_answer_eval._adapter_registry_identity(kind, identity)
            ),
        }
        entry["entry_sha256"] = recall_answer_eval._canonical_sha(entry)
        registry_entries.append(entry)
    adapter_registry = seal_object(
        {
            "schema_version": 1,
            "artifact_kind": "answer-authority-adapter-registry",
            "frozen_at": "2026-06-01T00:00:00Z",
            "entries": registry_entries,
        }
    )

    first = recall_answer_eval.evaluate_answer_episodes(
        runner=runner,
        scorer=scorer,
        runner_identity=runner_identity,
        scorer_identity=scorer_identity,
        episode_file=episodes,
        output_file=None,
        min_independent_samples=2,
        split_manifest=split_manifest,
        gold_manifest=gold_manifest,
        scorer_calibration=scorer_calibration,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=adapter_registry,
        split="train",
    )
    second = recall_answer_eval.evaluate_answer_episodes(
        runner=runner,
        scorer=scorer,
        runner_identity=runner_identity,
        scorer_identity=scorer_identity,
        episode_file=episodes,
        output_file=None,
        min_independent_samples=2,
        split_manifest=split_manifest,
        gold_manifest=gold_manifest,
        scorer_calibration=scorer_calibration,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=adapter_registry,
        split="train",
    )
    held = recall_answer_eval.evaluate_answer_episodes(
        runner=None,
        scorer=scorer,
        runner_identity=runner_identity,
        scorer_identity=scorer_identity,
        episode_file=episodes,
        output_file=None,
        min_independent_samples=2,
        split_manifest=split_manifest,
        gold_manifest=gold_manifest,
        scorer_calibration=scorer_calibration,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=adapter_registry,
        split="train",
    )

    assert first["status"] == "passed"
    assert first["seal_sha256"] == second["seal_sha256"]
    assert len(generations) == 8
    assert all(
        generations[index] == generations[index + 1]
        for index in range(0, len(generations), 2)
    )
    assert len(first["page_rewards"]) == 2
    assert held["status"] == "held"
    assert held["page_rewards"] == []
    assert recall_answer_eval.validate_answer_outcome_artifact(
        first,
        required_split="train",
        minimum_independent_samples=2,
        episode_file=episodes,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=adapter_registry,
    )["passed"] is True

    uncalibrated = recall_answer_eval.evaluate_answer_episodes(
        runner=runner,
        scorer=scorer,
        runner_identity=runner_identity,
        scorer_identity=scorer_identity,
        episode_file=episodes,
        output_file=None,
        min_independent_samples=2,
        split_manifest=split_manifest,
        gold_manifest=gold_manifest,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=adapter_registry,
        split="train",
    )
    assert uncalibrated["status"] == "held"
    assert uncalibrated["reason"] == "missing_scorer_calibration"

    scalar_tamper = json.loads(json.dumps({
        key: value for key, value in first.items() if key != "seal_sha256"
    }))
    scalar_tamper["results"][0]["score_delta"] = 0.5
    assert recall_answer_eval.validate_answer_outcome_artifact(
        seal_object(scalar_tamper),
        required_split="train",
        minimum_independent_samples=2,
        episode_file=episodes,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=adapter_registry,
    )["passed"] is False

    calibration_tamper = {
        key: value for key, value in scorer_calibration.items() if key != "seal_sha256"
    }
    calibration_tamper["metrics"] = {
        **calibration_tamper["metrics"],
        "cases": 999,
    }
    assert recall_answer_eval.validate_scorer_calibration_artifact(
        seal_object(calibration_tamper),
        scorer_identity=scorer_identity,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
    )["passed"] is False

    mismatched_identity = {**scorer_identity, "version": "different"}
    assert recall_answer_eval.validate_scorer_calibration_artifact(
        scorer_calibration,
        scorer_identity=mismatched_identity,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
    )["passed"] is False

    insufficient = _scorer_calibration(
        scorer_identity,
        count=recall_answer_eval.SCORER_CALIBRATION_MIN_CASES - 2,
        review_ledger=review_ledger,
        execution_ledger=execution_ledger,
    )
    assert recall_answer_eval.validate_scorer_calibration_artifact(
        insufficient,
        scorer_identity=scorer_identity,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
    )["passed"] is False

    identity_tamper = json.loads(json.dumps({
        key: value for key, value in first.items() if key != "seal_sha256"
    }))
    identity_tamper["runner_identity"]["model"] = "different-model"
    assert recall_answer_eval.validate_answer_outcome_artifact(
        seal_object(identity_tamper),
        required_split="train",
        minimum_independent_samples=2,
        episode_file=episodes,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=adapter_registry,
    )["passed"] is False

    invalid_manifest = {
        key: value for key, value in first.items() if key != "seal_sha256"
    }
    invalid_manifest["manifest"] = {
        **invalid_manifest["manifest"],
        "manifest_sha256": "0" * 64,
    }
    assert recall_answer_eval.validate_answer_outcome_artifact(
        seal_object(invalid_manifest),
        required_split="train",
        minimum_independent_samples=2,
        episode_file=episodes,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=adapter_registry,
    )["passed"] is False

    malformed = json.loads(
        json.dumps({key: value for key, value in first.items() if key != "seal_sha256"})
    )
    malformed["results"][0]["field_on"] = None
    assert recall_answer_eval.validate_answer_outcome_artifact(
        seal_object(malformed),
        required_split="train",
        minimum_independent_samples=2,
        episode_file=episodes,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=adapter_registry,
    )["passed"] is False
    assert recall_answer_eval.validate_answer_artifact_set(
        train=seal_object({"manifest": None}),
        locked=seal_object({"manifest": None}),
        minimum_independent_samples=2,
        episode_file=episodes,
    )["passed"] is False

    overlap_calibration = _scorer_calibration(
        scorer_identity,
        review_ledger=review_ledger,
        execution_ledger=execution_ledger,
        overlap_session=rows[0]["session_hash"],
    )
    overlap_check = recall_answer_eval.validate_scorer_calibration_artifact(
        overlap_calibration,
        scorer_identity=scorer_identity,
        answer_split_manifest=split_manifest,
        evaluated_at="2026-08-01T12:00:00Z",
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
    )
    assert overlap_check["passed"] is False
    assert overlap_check["reason"] == "calibration_answer_split_overlap"
    assert recall_answer_eval.validate_scorer_calibration_artifact(
        scorer_calibration,
        scorer_identity=scorer_identity,
        answer_split_manifest=split_manifest,
        evaluated_at="2026-05-01T00:00:00Z",
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
    )["passed"] is False

    subset_payload = {
        **split_payload,
        "episode_ledger_manifest_sha256": recall_answer_eval.manifest_sha256(
            [split_entries[0]["episode_sha256"]]
        ),
        "entries": [split_entries[0]],
    }
    subset_payload["epoch_id"] = recall_answer_eval._split_epoch_id(subset_payload)
    subset = recall_answer_eval.evaluate_answer_episodes(
        runner=runner,
        scorer=scorer,
        runner_identity=runner_identity,
        scorer_identity=scorer_identity,
        episode_file=episodes,
        output_file=None,
        min_independent_samples=1,
        split_manifest=seal_object(subset_payload),
        gold_manifest=gold_manifest,
        scorer_calibration=scorer_calibration,
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=adapter_registry,
        split="train",
    )
    assert subset["status"] == "held"
    assert subset["reason"] == "episode_ledger_mismatch"

    self_rows = json.loads(json.dumps(rows))
    for row in self_rows:
        row["answer_sha256"] = recall_answer_eval._sha_text("reference")
        row["episode_sha256"] = recall_answer_eval._canonical_sha(
            {key: value for key, value in row.items() if key != "episode_sha256"}
        )
    _write_rows(episodes, self_rows)
    self_split_entries = [
        {**recall_answer_eval._episode_manifest_entry(row), "split": "train"}
        for row in self_rows
    ]
    self_split_payload = {
        **split_payload,
        "episode_ledger_manifest_sha256": recall_answer_eval.manifest_sha256(
            [entry["episode_sha256"] for entry in self_split_entries]
        ),
        "entries": self_split_entries,
    }
    self_split_payload["epoch_id"] = recall_answer_eval._split_epoch_id(
        self_split_payload
    )
    self_split = seal_object(self_split_payload)
    self_gold_entries = []
    for entry in gold_entries:
        updated = {**entry, "gold_answer": "reference"}
        updated["evidence_sha256"] = recall_answer_eval._canonical_sha(
            {
                "episode_id": updated["episode_id"],
                "gold_answer": updated["gold_answer"],
                "evidence": updated["evidence"],
                "rubric_sha256": rubric_sha,
            }
        )
        self_review = recall_answer_eval.append_answer_review_receipt(
            kind="gold_entry_review",
            payload={
                "episode_id": updated["episode_id"],
                "gold_answer_sha256": recall_answer_eval._sha_text("reference"),
                "evidence_sha256": updated["evidence_sha256"],
                "rubric_sha256": rubric_sha,
            },
            reviewer_kind="human_reviewer",
            reviewed_at="2026-05-02T00:00:00Z",
            protocol_sha256=review_protocol_sha,
            ledger_file=review_ledger,
        )
        updated["review_provenance"] = {
            "source_kind": "human_review",
            "reviewer_receipt_sha256": self_review["receipt_sha256"],
            "reviewed_at": "2026-05-02T00:00:00Z",
        }
        self_gold_entries.append(updated)
    self_gold = seal_object(
        {
            **{key: value for key, value in gold_manifest.items() if key != "seal_sha256"},
            "entries": self_gold_entries,
        }
    )
    self_scorer_identity = {
        **scorer_identity,
        "evidence_manifest_sha256": self_gold["seal_sha256"],
    }

    def self_scorer(_prompt: str, answer: str, gold: dict, scoring: dict) -> dict:
        score = 1.0 if answer == "better" else 0.0
        return {
            "identity": self_scorer_identity,
            "evidence_sha256": gold["evidence_sha256"],
            "reset_receipt": {
                "seed": scoring["seed"],
                "base_state_sha256": scoring["base_state_sha256"],
                "reset_protocol_sha256": self_scorer_identity["policy_sha256"],
            },
            "dimensions": {
                dimension: score
                for dimension in recall_answer_eval.ANSWER_DIMENSIONS
            },
        }

    self_registry_entries = []
    for kind, adapter_id, adapter, identity in (
        ("runner", runner_identity["runner_id"], runner, runner_identity),
        (
            "scorer",
            self_scorer_identity["scorer_id"],
            self_scorer,
            self_scorer_identity,
        ),
    ):
        item = {
            "kind": kind,
            "adapter_id": adapter_id,
            "callable_sha256": recall_answer_eval.adapter_callable_sha256(adapter),
            "identity_sha256": recall_answer_eval._canonical_sha(
                recall_answer_eval._adapter_registry_identity(kind, identity)
            ),
        }
        item["entry_sha256"] = recall_answer_eval._canonical_sha(item)
        self_registry_entries.append(item)
    self_registry = seal_object(
        {
            "schema_version": 1,
            "artifact_kind": "answer-authority-adapter-registry",
            "frozen_at": "2026-06-01T00:00:00Z",
            "entries": self_registry_entries,
        }
    )
    self_result = recall_answer_eval.evaluate_answer_episodes(
        runner=runner,
        scorer=self_scorer,
        runner_identity=runner_identity,
        scorer_identity=self_scorer_identity,
        episode_file=episodes,
        output_file=None,
        min_independent_samples=2,
        split_manifest=self_split,
        gold_manifest=self_gold,
        scorer_calibration=_scorer_calibration(
            self_scorer_identity,
            review_ledger=review_ledger,
            execution_ledger=execution_ledger,
        ),
        review_ledger_file=review_ledger,
        execution_ledger_file=execution_ledger,
        adapter_registry=self_registry,
        split="train",
    )
    assert self_result["status"] == "held"
    assert {row["reason"] for row in self_result["results"]} == {
        "gold_reuses_production_answer"
    }


def test_split_connects_uid_present_and_legacy_page_id_and_fixes_embargo() -> None:
    same_page = [
        {
            "episode_id": f"same-{index}",
            "episode_sha256": f"{index + 1:064x}",
            "observed_at": f"2026-07-{index + 1:02d}T00:00:00Z",
            "session_hash": f"session-{index}",
            "query_sha256": f"{index + 20:064x}",
            "page_bindings": [
                {
                    "page_id": "shared-page",
                    "page_uid": "shared-uid" if index == 0 else "",
                    "content_sha256": f"{index + 40:064x}",
                }
            ],
        }
        for index in range(2)
    ]
    assignments = recall_answer_eval._assign_episode_splits(same_page)
    assert assignments["same-0"] == assignments["same-1"]

    entries = [
        {
            "episode_id": f"episode-{index}",
            "episode_sha256": f"{index + 100:064x}",
            "observed_at": f"2026-07-01T{index:02d}:00:00Z",
            "session_hash": f"session-{index}",
            "query_sha256": f"{index + 200:064x}",
            "page_bindings": [
                {
                    "page_id": f"page-{index}",
                    "page_uid": "",
                    "content_sha256": f"{index + 300:064x}",
                }
            ],
        }
        for index in range(10)
    ]
    assigned = recall_answer_eval._assign_episode_splits(entries)
    assert set(assigned.values()) == {"embargo"}

    payload = {
        "schema_version": 1,
        "artifact_kind": "answer-preregistered-split-manifest",
        "embargo_seconds": -1,
        "strategy": "connected-components-chronological-70-20-10",
        "component_keys": [],
        "episode_ledger_manifest_sha256": recall_answer_eval.manifest_sha256(
            [entry["episode_sha256"] for entry in entries]
        ),
        "entries": [
            {**entry, "split": assigned[entry["episode_id"]]} for entry in entries
        ],
    }
    assert recall_answer_eval.validate_split_manifest(seal_object(payload))["passed"] is False


def test_unknown_capture_cursor_waits_and_upgrades_same_episode(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "session.jsonl"
    session.write_text("user\nassistant\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text("page", encoding="utf-8")
    records = [
        SimpleNamespace(
            role="user", line=1, text="prompt", timestamp="2026-08-01T00:00:00Z"
        ),
        SimpleNamespace(
            role="assistant",
            line=2,
            text="answer",
            timestamp="2026-08-01T00:00:01Z",
        ),
    ]
    monkeypatch.setattr(
        recall_answer_eval,
        "_extract",
        lambda _host, _path: SimpleNamespace(
            records=records, session_id="session", cwd="/repo"
        ),
    )
    monkeypatch.setattr(recall_answer_eval, "find_page", lambda _page_id: page)
    transaction = recall_answer_eval.make_save_transaction(
        host="codex",
        session_file=session,
        session_id="session",
        after_line=0,
        until_line=2,
    )
    raw_dir = tmp_path / "raw"
    save_result = append_capture(
        raw_dir=raw_dir,
        raw_id=f"save-{transaction.idempotency_key}.md",
        idempotency_key=transaction.idempotency_key,
        host="codex",
        session_key=transaction.session_key,
        session_id="session",
        source_file=session,
        after_line=0,
        until_line=2,
        source_bytes=session.read_bytes(),
        record_count=2,
    ).to_result()
    receipt, receipt_error = recall_answer_eval._verified_save_receipt(
        host="codex",
        save_output={
            "status": "saved",
            "session_file": str(session),
            "session_id": "session",
            "after_line": 0,
            "scanned_until_line": 2,
            "save_result": save_result,
        },
        session_file=session,
        session_id="session",
        raw_dir=raw_dir,
    )
    assert receipt_error == ""
    monkeypatch.setattr(
        recall_answer_eval,
        "_verified_save_receipt",
        lambda **_kwargs: (receipt, ""),
    )
    source = {
        "decision_id": "decision",
        "session_id": "session",
        "host": "codex",
        "prompt_hash": stable_prompt_hash("prompt"),
        "pages": ["page"],
        "context_items": [
            {
                "page_id": "page",
                "page_uid": "uid-page",
                "content_sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
            }
        ],
    }
    context_receipt = {
        "schema_version": 1,
        "renderer_protocol": "recall-result-context-v1",
        "context_style": "compact",
        "rendered_context": "exact injected card",
        "rendered_context_sha256": recall_answer_eval._sha_text(
            "exact injected card"
        ),
        "page_bindings": source["context_items"],
    }
    context_receipt["receipt_sha256"] = recall_answer_eval._canonical_sha(
        context_receipt
    )
    source["context_receipt"] = context_receipt
    current = {"value": None}
    monkeypatch.setattr(
        recall_answer_eval,
        "source_recall_record",
        lambda *_args, **_kwargs: current["value"],
    )
    recalls = tmp_path / "recall.jsonl"
    pulls = tmp_path / "pull.jsonl"
    _write_rows(recalls, [source])
    _write_rows(
        pulls,
        [
            {
                "type": "used",
                "event_id": "used",
                "decision_id": "decision",
                "session_id": "session",
                "page_ids": ["page"],
            }
        ],
    )
    kwargs = {
        "host": "codex",
        "session_file": session,
        "episode_file": tmp_path / "episodes.jsonl",
        "cursor_file": tmp_path / "cursor.json",
        "recall_log_file": recalls,
        "pull_log_file": pulls,
        "save_output": {
            "status": "saved",
            "session_file": str(session),
            "session_id": "session",
        },
    }

    unknown = recall_answer_eval.capture_session_answer_episodes(**kwargs)
    current["value"] = source
    upgraded = recall_answer_eval.capture_session_answer_episodes(**kwargs)
    duplicate = recall_answer_eval.capture_session_answer_episodes(**kwargs)
    rows = [
        json.loads(line)
        for line in (tmp_path / "episodes.jsonl").read_text().splitlines()
    ]

    assert unknown["cursor_line"] == 0
    assert upgraded["cursor_line"] == 2
    assert duplicate["captured"] == 0
    assert [row["binding_status"] for row in rows] == ["unknown", "verified"]
    assert rows[0]["episode_id"] == rows[1]["episode_id"]


def test_recall_time_page_identity_is_mandatory_and_uid_drift_holds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from chronovisor.ingest import page_registry

    page = tmp_path / "page.md"
    page.write_text("historical", encoding="utf-8")
    digest = hashlib.sha256(page.read_bytes()).hexdigest()
    monkeypatch.setattr(recall_answer_eval, "find_page", lambda _page_id: page)
    monkeypatch.setattr(
        page_registry,
        "PageRegistry",
        lambda _root: SimpleNamespace(
            resolve=lambda _page_id: {"uid": "uid-current"}
        ),
    )

    hashes, uids, errors = recall_answer_eval._page_hashes(
        ["page"],
        {
            "context_items": [
                {
                    "page_id": "page",
                    "page_uid": "uid-at-recall",
                    "content_sha256": digest,
                }
            ]
        },
    )
    assert hashes == {"page": digest}
    assert uids == {"page": "uid-at-recall"}
    assert errors == {"page": "page_uid_changed_since_recall"}

    hashes, _uids, errors = recall_answer_eval._page_hashes(
        ["page"], {"pages": ["page"]}
    )
    assert hashes == {}
    assert errors["page"] in {
        "missing_recall_time_content_sha256",
        "missing_recall_time_page_uid",
    }


def test_capture_hook_rejects_non_receipt_hook_payload() -> None:
    assert recall_answer_eval.capture_hook_only(
        host="codex", stdin_text='{"session_id":"ordinary-hook"}'
    ) == {
        "status": "held",
        "reason": "exact_save_output_required",
        "captured": 0,
    }


def test_authority_confidence_and_seed_are_not_caller_selectable() -> None:
    base = {
        "runner": None,
        "scorer": None,
        "runner_identity": {},
        "scorer_identity": {},
        "output_file": None,
        "split": "train",
    }
    with pytest.raises(ValueError, match="confidence is fixed"):
        recall_answer_eval.evaluate_answer_episodes(**base, confidence=0.01)
    with pytest.raises(ValueError, match="bootstrap seed is fixed"):
        recall_answer_eval.evaluate_answer_episodes(**base, seed=7)
