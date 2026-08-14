from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from chronovisor.core.durable_state import write_sealed_json
from chronovisor.recall import recall_growth


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _label_inputs(tmp_path: Path) -> dict[str, Path]:
    return {
        "certificate_file": tmp_path / "certificates.jsonl",
        "recall_log_file": tmp_path / "recall.jsonl",
        "pull_log_file": tmp_path / "pull.jsonl",
        "golden_file": tmp_path / "golden.jsonl",
    }


def _locked_gate(tmp_path: Path) -> Path:
    path = tmp_path / "locked-e2e.json"
    gates = {
        "sealed_manual_94": True,
        "rerank_recall_at_5": True,
        "negative_hit_rate": True,
        "processor_precision": True,
        "processor_related_recall": True,
        "rich_precision": True,
        "pointer_precision": True,
        "latency": True,
    }
    unsigned = {
        "schema_version": 1,
        "status": "passed",
        "examples": 94,
        "manifest_sha256": "a" * 64,
        "gates": gates,
        "metrics": {
            "recall_at_5": 0.6,
            "negative_hit_rate_at_20": 0.1,
            "latency_ms": {"max": 100},
            "processor": {
                "precision": 0.95,
                "related_recall": 0.6,
                "evidence_kind": {
                    "rich": {"precision": 0.95},
                    "pointer": {"precision": 0.95},
                },
            },
        },
    }
    sealed = {
        **unsigned,
        "snapshot_sha256": hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    path.write_text(json.dumps(sealed), encoding="utf-8")
    return path


def _manual94_case_gate(
    tmp_path: Path, monkeypatch
) -> Path:
    from chronovisor.recall import recall_runtime, search_label_contract
    from chronovisor.search import search_eval

    pages: dict[str, Path] = {}
    environment = {
        "adapter_id": "test-field",
        "version": 1,
        "policy_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "corpus_sha256": "3" * 64,
        "index_sha256": "4" * 64,
        "model_sha256": "5" * 64,
        "clone_protocol_sha256": "6" * 64,
        "candidate_policy_delta_sha256": "7" * 64,
        "lkg_base_artifact_sha256": "8" * 64,
        "lkg_base_snapshot_sha256": "9" * 64,
        "effective_field_config_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        recall_growth, "builtin_field_environment_identity", lambda: environment
    )
    monkeypatch.setattr(
        recall_runtime, "page_uid_for_id", lambda page_id: f"uid-{page_id}"
    )
    monkeypatch.setattr(recall_growth, "find_page", lambda page_id: pages.get(page_id))
    entries = []
    cases = []
    cohort_rows = []
    for index in range(94):
        page_id = f"manual-page-{index}"
        page = tmp_path / f"{page_id}.md"
        page.write_text(page_id, encoding="utf-8")
        pages[page_id] = page
        query = f"manual query {index}"
        query_sha = hashlib.sha256(query.encode()).hexdigest()
        entry = {
            "query_sha256": query_sha,
            "ref": f"review-{index}",
            "source": "human",
            "split": "locked-test",
            "language": "en",
            "kind": "manual",
            "reviewed": True,
            "expected_pages": [page_id],
            "negative_pages": [],
            "stale_pages": [],
        }
        entry["entry_sha256"] = hashlib.sha256(
            json.dumps(
                entry,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        entries.append(entry)
        cohort_rows.append(
            {
                "query": query,
                "expected_pages": [page_id],
                "negative_pages": [],
                "stale_pages": [],
                "split": "locked-test",
                "language": "en",
                "kind": "manual",
                "source": "human",
                "ref": f"review-{index}",
                "ts": "2026-07-31T00:00:00Z",
                "reviewed": True,
            }
        )
        certificate_id = f"certificate-{index}"
        committed = [page_id]
        certificates = [certificate_id]
        selected = [
            {
                "page_id": page_id,
                "certificate_id": certificate_id,
                "evidence_kind": "rich" if index % 2 == 0 else "pointer",
            }
        ]
        case = {
            "manifest_entry_sha256": entry["entry_sha256"],
            "review_receipt_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "kind": "manual94-human-review-v1",
                        "entry_sha256": entry["entry_sha256"],
                        "ref": entry["ref"],
                        "source": entry["source"],
                        "reviewed": True,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "query_sha256": query_sha,
            "expected_pages": [page_id],
            "bad_pages": [],
            "reviewed": True,
            "ranked_page_bindings": [
                {
                    "page_id": page_id,
                    "page_uid": f"uid-{page_id}",
                    "content_sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
                    "rank": 1,
                }
            ],
            "committed_page_ids": committed,
            "certificate_ids": certificates,
            "commit_ids": [
                hashlib.sha256(
                    json.dumps(
                        {
                            "query_sha256": query_sha,
                            "committed_page_ids": committed,
                            "certificate_ids": certificates,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            ],
            "selected_evidence": selected,
            "latency_ms": 100,
        }
        case["case_sha256"] = hashlib.sha256(
            json.dumps(
                case,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        cases.append(case)
    cohort_file = tmp_path / "manual94-cohort.jsonl"
    _write_jsonl(cohort_file, cohort_rows)
    frozen_epoch = datetime(2026, 7, 31).timestamp()
    os.utime(cohort_file, (frozen_epoch, frozen_epoch))
    canonical_manifest_file = tmp_path / "manual-94-manifest.json"
    search_eval.write_sealed_manifest(
        search_label_contract.load_examples(cohort_file),
        canonical_manifest_file,
        review_ledger_file=cohort_file,
    )
    monkeypatch.setattr(
        search_label_contract,
        "MANUAL_MANIFEST_FILE",
        canonical_manifest_file,
    )
    manifest = json.loads(canonical_manifest_file.read_text(encoding="utf-8"))
    manifest_sha = str(manifest["manifest_sha256"])
    gates = {
        "sealed_manual_94": True,
        "rerank_recall_at_5": True,
        "negative_hit_rate": True,
        "processor_precision": True,
        "processor_related_recall": True,
        "rich_precision": True,
        "pointer_precision": True,
        "latency": True,
    }
    unsigned = {
        "schema_version": 2,
        "generated_at": "2026-08-01T00:00:00Z",
        "frozen_at": manifest["frozen_at"],
        "variant": "hybrid-rerank",
        "manifest_sha256": manifest_sha,
        "manifest": manifest,
        "frozen_manifest_sha256": hashlib.sha256(
            canonical_manifest_file.read_bytes()
        ).hexdigest(),
        "examples": 94,
        "environment_epoch": environment,
        "environment_epoch_sha256": hashlib.sha256(
            json.dumps(
                environment,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "cases": cases,
        "status": "passed",
        "gates": gates,
        "metrics": {
            "recall_at_5": 1.0,
            "negative_hit_rate_at_20": 0.0,
            "latency_ms": {"max": 100.0},
            "processor": {
                "precision": 1.0,
                "related_recall": 1.0,
                "evidence_kind": {
                    "rich": {"precision": 1.0},
                    "pointer": {"precision": 1.0},
                },
            },
        },
    }
    path = tmp_path / "manual94-v2.json"
    path.write_text(
        json.dumps(
            {
                **unsigned,
                "snapshot_sha256": hashlib.sha256(
                    json.dumps(
                        unsigned,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return path


def _train_outcome_gate(tmp_path: Path, count: int) -> Path:
    path = tmp_path / "train-answer-eval.json"
    manifest = {
        "split": "train",
        "episode_ids": [f"episode-{index}" for index in range(count)],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    write_sealed_json(
        path,
        {
            "schema_version": 1,
            "artifact_kind": "locked-answer-on-off-evaluation",
            "status": "passed",
            "production_host_exact_replay_claimed": False,
            "manifest": manifest,
            "samples": count,
            "confidence_bound": {
                "valid": True,
                "method": "connected-cluster-bootstrap-percentile",
                "confidence": 0.95,
                "seed": 1729,
                "samples": count,
                "clusters": 20,
                "point": 0.1,
                "lower": 0.05,
                "upper": 0.15,
            },
            "gates": {"verified": True},
            "page_rewards": [
                {
                    "episode_id": f"episode-{index}",
                    "decision_id": f"decision-{index}",
                    "page_id": f"page-{index}",
                    "content_sha256": f"{index + 1:064x}",
                    "reward": 0.1,
                    "producer": "verified_answer_pair_v1",
                    "session_hash": hashlib.sha256(
                        f"session-{index % 20}".encode()
                    ).hexdigest()[:16],
                    "query_sha256": f"{index + 1:064x}",
                    "observed_at": "2026-07-01T00:00:00Z",
                }
                for index in range(count)
            ],
        },
    )
    return path


def _qualified_label_inputs(tmp_path: Path, *, count: int = 200) -> dict[str, Path]:
    inputs = _label_inputs(tmp_path)
    recalls = []
    pulls = []
    for index in range(count):
        page = f"page-{index}"
        session = f"session-{index}"
        content_sha = f"{index + 1:064x}"
        recalls.append(
            {
                "schema_version": 2,
                "decision_id": f"decision-{index}",
                "session_id": session,
                "prompt_hash": f"{index + 1:064x}",
                "ts": "2026-07-01T00:00:00Z",
                "context_items": [
                    {
                        "page_id": page,
                        "page_uid": f"uid-{index}",
                        "content_sha256": content_sha,
                    }
                ],
                "evidence_features": {
                    "processor_shadow": {"committed_page_ids": [page]}
                },
            }
        )
        pulls.append(
            {
                "type": "used",
                "event_id": f"used-{index}",
                "decision_id": f"decision-{index}",
                "session_id": session,
                "page_ids": [page],
                "ts": "2026-07-01T00:00:01Z",
            }
        )
    _write_jsonl(inputs["recall_log_file"], recalls)
    _write_jsonl(inputs["pull_log_file"], pulls)
    _write_jsonl(inputs["certificate_file"], [])
    _write_jsonl(inputs["golden_file"], [])
    inputs["answer_outcome_file"] = _train_outcome_gate(tmp_path, count)
    return inputs


def _qualified_candidate_traces(count: int = 100) -> list[dict]:
    rows = []
    previous = "0" * 64
    for index in range(count):
        page_id = f"page-{index}"
        session = f"candidate-{index}"
        query_sha = f"{index + 1000:064x}"
        content_sha = f"{index + 2000:064x}"
        row = {
            "schema_version": 3,
            "session_hash": session,
            "query_sha256": query_sha,
            "page_content_sha256": {page_id: content_sha},
            "page_uids": {page_id: f"uid-{index}"},
            "cluster_nodes": [
                f"session:{session}",
                f"query:{query_sha}",
                f"page:{page_id}",
                f"uid:uid-{index}",
                f"content:{content_sha}",
            ],
            "status": "observed",
            "field_page_ids": [page_id],
            "teacher_page_ids": [page_id],
            "committed_page_ids": [page_id],
            "latency_ms": 120,
            "field_latency_ms": 20,
            "teacher_latency_ms": 120,
            "full_search_required": False,
            "over_4s": False,
            "authority": "teacher",
            "quality_eligible": True,
            "previous_record_sha256": previous,
            "cumulative_eligible_trace_count": index + 1,
        }
        row["record_sha256"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        rows.append(row)
        previous = row["record_sha256"]
    return rows


def test_growth_cycle_collects_without_authorizing_weak_evidence(
    tmp_path: Path,
) -> None:
    inputs = _label_inputs(tmp_path)
    for path in inputs.values():
        _write_jsonl(path, [])
    state = tmp_path / "growth-state.json"
    promotion = tmp_path / "promotion.json"

    result = recall_growth.run_growth_cycle(
        state_file=state,
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=tmp_path / "candidate.jsonl",
        promotion_file=promotion,
        label_inputs=inputs,
        now=datetime(2026, 7, 31, 20, 0, 0),
    )

    assert result["stage"] == "collecting_labels"
    assert result["effective_mode"] == "candidate"
    assert result["canary_percent"] == 100
    assert result["field_learning_allowed"] is False
    assert result["authority_enabled"] is False
    assert json.loads(promotion.read_text())["status"] == "held"


def test_growth_cycle_reads_default_locked_e2e_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _label_inputs(tmp_path)
    for path in inputs.values():
        _write_jsonl(path, [])
    locked = _locked_gate(tmp_path)
    monkeypatch.setattr(recall_growth, "LOCKED_E2E_ARTIFACT", locked)

    result = recall_growth.run_growth_cycle(
        state_file=tmp_path / "growth-state.json",
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=tmp_path / "candidate.jsonl",
        promotion_file=tmp_path / "promotion.json",
        label_inputs=inputs,
    )

    assert result["locked_e2e"]["passed"] is False
    assert result["gates"]["locked_e2e"] is False


def test_manual94_recomputes_all_case_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = _manual94_case_gate(tmp_path, monkeypatch)
    assert recall_growth.retrieval_locked_e2e_status(artifact)["passed"] is True

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["cases"][0]["ranked_page_bindings"][0]["rank"] = 2
    unsigned = {key: value for key, value in payload.items() if key != "snapshot_sha256"}
    payload["snapshot_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert recall_growth.retrieval_locked_e2e_status(artifact)["passed"] is False


def test_manual94_counts_every_selected_nonexpected_page_as_false_positive(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = _manual94_case_gate(tmp_path, monkeypatch)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    case = payload["cases"][0]
    expected_binding = dict(case["ranked_page_bindings"][0])
    expected_binding["rank"] = 2
    foreign_binding = dict(payload["cases"][1]["ranked_page_bindings"][0])
    foreign_binding["rank"] = 1
    foreign_page = foreign_binding["page_id"]
    case["ranked_page_bindings"] = [foreign_binding, expected_binding]
    case["selected_evidence"][0]["page_id"] = foreign_page
    case["committed_page_ids"] = [foreign_page]
    case["commit_ids"] = [
        hashlib.sha256(
            json.dumps(
                {
                    "query_sha256": case["query_sha256"],
                    "committed_page_ids": case["committed_page_ids"],
                    "certificate_ids": case["certificate_ids"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    ]
    case_unsigned = {key: value for key, value in case.items() if key != "case_sha256"}
    case["case_sha256"] = hashlib.sha256(
        json.dumps(
            case_unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    # Preserve the metrics that the previous bad-pages-only FP accounting
    # accepted; the exact recomputation must now expose the unlabeled FP.
    payload["metrics"]["processor"]["related_recall"] = 93 / 94
    unsigned = {key: value for key, value in payload.items() if key != "snapshot_sha256"}
    payload["snapshot_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    assert recall_growth.retrieval_locked_e2e_status(artifact)["passed"] is False


def test_manual94_rejects_ghost_selection_and_changed_canonical_cohort(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.recall import search_label_contract

    artifact = _manual94_case_gate(tmp_path, monkeypatch)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    case = payload["cases"][0]
    case["selected_evidence"][0]["page_id"] = "ghost-page"
    case["committed_page_ids"] = ["ghost-page"]
    case["commit_ids"] = [
        hashlib.sha256(
            json.dumps(
                {
                    "query_sha256": case["query_sha256"],
                    "committed_page_ids": case["committed_page_ids"],
                    "certificate_ids": case["certificate_ids"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    ]
    case_unsigned = {key: value for key, value in case.items() if key != "case_sha256"}
    case["case_sha256"] = hashlib.sha256(
        json.dumps(
            case_unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    unsigned = {key: value for key, value in payload.items() if key != "snapshot_sha256"}
    payload["snapshot_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert recall_growth.retrieval_locked_e2e_status(artifact)["passed"] is False

    artifact = _manual94_case_gate(tmp_path, monkeypatch)
    manifest_path = search_label_contract.MANUAL_MANIFEST_FILE
    canonical = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical["entries"][0]["expected_pages"] = ["changed-label"]
    unsigned_manifest = {
        key: value for key, value in canonical.items() if key != "manifest_sha256"
    }
    canonical["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(canonical), encoding="utf-8")
    assert recall_growth.retrieval_locked_e2e_status(artifact)["passed"] is False


def test_candidate_trace_chain_rejects_one_malformed_record(tmp_path: Path) -> None:
    path = tmp_path / "candidate.jsonl"
    row = _qualified_candidate_traces(1)[0]
    path.write_text(json.dumps(row) + "\n{malformed\n", encoding="utf-8")

    rows, error = recall_growth._validated_candidate_trace_rows(path)

    assert rows == [row]
    assert error == "candidate_trace_chain_invalid"


def test_candidate_confidence_connects_partial_page_overlap() -> None:
    def row(index: int, pages: list[str]) -> dict:
        session = f"session-{index}"
        query = f"{index + 1:064x}"
        hashes = {page: hashlib.sha256(page.encode()).hexdigest() for page in pages}
        uids = {page: f"uid-{page}" for page in pages}
        nodes = [f"session:{session}", f"query:{query}"]
        nodes.extend(f"page:{page}" for page in sorted(pages))
        nodes.extend(f"uid:{uids[page]}" for page in sorted(pages))
        nodes.extend(f"content:{hashes[page]}" for page in sorted(pages))
        return {
            "schema_version": 3,
            "session_hash": session,
            "query_sha256": query,
            "page_content_sha256": hashes,
            "page_uids": uids,
            "cluster_nodes": nodes,
            "quality_eligible": True,
            "status": "observed",
            "field_page_ids": pages,
            "teacher_page_ids": pages,
            "committed_page_ids": pages,
        }

    metrics = recall_growth.candidate_metrics(
        [row(0, ["a", "b"]), row(1, ["b", "c"])]
    )
    assert metrics["confidence"]["samples"] == 2
    assert metrics["confidence"]["clusters"] == 1


def test_split_integrity_tracks_page_id_and_uid_independently() -> None:
    result = recall_growth.split_integrity(
        [
            {
                "split": "train",
                "page_id": "same-page",
                "page_uid": "uid-same",
                "observed_at": "2026-01-01T00:00:00Z",
            },
            {
                "split": "locked-test",
                "page_id": "same-page",
                "page_uid": "",
                "observed_at": "2026-01-02T00:00:00Z",
            },
        ]
    )
    assert result["page_leakage"] == 1
    assert result["passed"] is False


def test_usage_receipts_alone_never_unlock_positive_learning(tmp_path: Path) -> None:
    inputs = _label_inputs(tmp_path)
    recalls: list[dict] = []
    pulls: list[dict] = []
    for index in range(200):
        decision = f"decision-{index}"
        session = f"session-{index % 20}"
        page = f"page-{index}"
        recalls.append(
            {
                "decision_id": decision,
                "session_id": session,
                "prompt_hash": f"{index:064x}",
            }
        )
        pulls.append(
            {
                "type": "used",
                "event_id": f"used-{index}",
                "decision_id": decision,
                "session_id": session,
                "page_ids": [page],
            }
        )
    _write_jsonl(inputs["recall_log_file"], recalls)
    _write_jsonl(inputs["pull_log_file"], pulls)
    _write_jsonl(inputs["certificate_file"], [])
    _write_jsonl(inputs["golden_file"], [])
    state = tmp_path / "growth-state.json"
    last_known_good = tmp_path / "last-known-good.json"

    result = recall_growth.run_growth_cycle(
        state_file=state,
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=tmp_path / "candidate.jsonl",
        promotion_file=tmp_path / "promotion.json",
        last_known_good_file=last_known_good,
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    assert result["stage"] == "collecting_labels"
    assert result["positive_learning_allowed"] is False
    assert result["field_learning_allowed"] is False
    assert result["policy_update_allowed"] is False
    assert result["authority_enabled"] is False
    assert not recall_growth.automatic_learning_allowed(
        enabled=True,
        state_file=state,
    )
    assert not last_known_good.exists()


def test_verified_train_outcomes_unlock_learning_before_authority(
    tmp_path: Path,
) -> None:
    inputs = _qualified_label_inputs(tmp_path)
    state = tmp_path / "growth-state.json"

    result = recall_growth.run_growth_cycle(
        state_file=state,
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=tmp_path / "candidate.jsonl",
        promotion_file=tmp_path / "promotion.json",
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    assert result["stage"] == "collecting_labels"
    assert result["positive_learning_allowed"] is False
    assert result["field_learning_allowed"] is False
    assert result["policy_update_allowed"] is False
    assert result["authority_enabled"] is False
    assert not recall_growth.automatic_learning_allowed(enabled=True, state_file=state)


def test_train_learning_is_independent_from_locked_authority_gate(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _label_inputs(tmp_path)
    for path in inputs.values():
        _write_jsonl(path, [])
    labels = {
        "labels": [],
        "counts": {
            "scope": "train",
            "strong_positive": 200,
            "strong_positive_sessions": 20,
            "total": 200,
        },
        "gates": {"field_learning_allowed": True},
    }
    train = {
        "passed": True,
        "method": "connected-cluster-bootstrap-percentile",
        "confidence": 0.95,
        "seed": 1729,
        "manifest_sha256": "a" * 64,
        "samples": 20,
        "distinct_clusters": 20,
        "point": 0.1,
        "lower": 0.05,
        "upper": 0.15,
    }
    monkeypatch.setattr(recall_growth, "build_label_ledger", lambda **_kwargs: labels)
    monkeypatch.setattr(
        recall_growth,
        "split_integrity",
        lambda _rows: {
            "passed": True,
            "session_leakage": 0,
            "query_leakage": 0,
            "page_leakage": 0,
            "content_leakage": 0,
            "timestamp_leakage": 0,
            "embargo_leakage": 0,
        },
    )
    monkeypatch.setattr(
        recall_growth,
        "retrieval_locked_e2e_status",
        lambda _path: {"passed": True, "examples": 94},
    )
    monkeypatch.setattr(
        recall_growth,
        "validate_answer_outcome_artifact",
        lambda *_args, **_kwargs: train,
    )
    monkeypatch.setattr(
        recall_growth,
        "validate_locked_answer_artifact",
        lambda _path, **_kwargs: {"passed": False},
    )
    monkeypatch.setattr(
        recall_growth,
        "validate_answer_artifact_set",
        lambda **_kwargs: {"passed": False},
    )

    result = recall_growth.run_growth_cycle(
        dry_run=True,
        state_file=tmp_path / "state.json",
        candidate_trace_file=tmp_path / "candidate.jsonl",
        label_inputs=inputs,
    )

    assert result["positive_learning_allowed"] is True
    assert result["gates"]["locked_answer_e2e"] is False
    assert result["authority_enabled"] is False


def test_processor_used_metrics_penalize_unused_shadow_cards() -> None:
    metrics = recall_growth.processor_used_metrics(
        [
            {
                "decision_id": "decision-1",
                "session_id": "session-1",
                "prompt_hash": "a" * 64,
                "evidence_features": {
                    "processor_shadow": {"committed_page_ids": ["used", "unused"]}
                },
            }
        ],
        [
            {
                "type": "used",
                "event_id": "used-1",
                "decision_id": "decision-1",
                "session_id": "session-1",
                "page_ids": ["used"],
            }
        ],
    )

    assert metrics["used_page_coverage"] == 1.0
    assert metrics["used_precision_proxy"] == 0.5


def test_candidate_metrics_separate_stable_quality_from_fallback_e2e() -> None:
    metrics = recall_growth.candidate_metrics(
        [
            {
                "session_hash": "stable-observed",
                "status": "observed",
                "authority": "teacher",
                "field_page_ids": ["page-a"],
                "teacher_page_ids": ["page-a"],
                "committed_page_ids": ["page-a"],
                "latency_ms": 100,
                "field_latency_ms": 20,
                "teacher_latency_ms": 100,
                "full_search_required": False,
                "over_4s": False,
            },
            {
                "session_hash": "stable-active",
                "status": "active",
                "authority": "field",
                "field_page_ids": ["page-b"],
                "teacher_page_ids": ["page-b"],
                "committed_page_ids": ["page-b"],
                "latency_ms": 110,
                "field_latency_ms": 30,
                "teacher_latency_ms": 110,
                "full_search_required": False,
                "over_4s": False,
            },
            {
                "session_hash": "legacy-observed",
                "status": "observed",
                "authority": "teacher",
                "field_page_ids": ["page-legacy"],
                "teacher_page_ids": ["page-legacy"],
                "committed_page_ids": ["page-legacy"],
                "latency_ms": 120,
                "field_latency_ms": 40,
                "teacher_latency_ms": 120,
                "over_4s": False,
            },
            {
                "session_hash": "topic-reset",
                "status": "fallback",
                "fallback_reason": "topic_reset",
                "authority": "teacher",
                "field_page_ids": [],
                "teacher_page_ids": ["fallback-page"],
                "committed_page_ids": ["fallback-page"],
                "latency_ms": 5_000,
                "teacher_latency_ms": 90,
                "full_search_required": True,
                "over_4s": True,
            },
            {
                "session_hash": "not-verified",
                "status": "fallback",
                "authority": "teacher",
                "quality_eligible": True,
                "field_attempted": True,
                "field_verified": False,
                "field_page_ids": [],
                "teacher_page_ids": ["other-page"],
                "committed_page_ids": ["other-page"],
                "latency_ms": 400,
                "field_latency_ms": 50,
                "teacher_latency_ms": 400,
                "full_search_required": True,
                "over_4s": False,
            },
            {
                "session_hash": "malformed-full-search-flag",
                "status": "observed",
                "authority": "teacher",
                "field_page_ids": ["malformed-page"],
                "teacher_page_ids": ["malformed-page"],
                "committed_page_ids": ["malformed-page"],
                "latency_ms": 100,
                "field_latency_ms": 10,
                "teacher_latency_ms": 100,
                "full_search_required": "false",
                "over_4s": False,
            },
        ]
    )

    # Stable-topic attempts are quality evidence; verifier failures stay misses.
    assert metrics["traces"] == 6
    assert metrics["sessions"] == 6
    assert metrics["stable_traces"] == 4
    assert metrics["stable_sessions"] == 4
    assert metrics["coverage_evidence_traces"] == 4
    assert metrics["coverage_evidence_sessions"] == 4
    assert metrics["commit_evidence_traces"] == 4
    assert metrics["commit_evidence_sessions"] == 4
    assert metrics["paired_latency_traces"] == 4
    assert metrics["paired_latency_sessions"] == 4
    assert metrics["teacher_top30_coverage"] == 0.75
    assert metrics["teacher_commit_coverage"] == 0.75
    assert metrics["field_precision_against_teacher"] == 1.0
    assert metrics["field_latency_ms"]["p95"] == 50.0
    assert metrics["teacher_latency_ms"]["p95"] == 400.0
    assert metrics["p95_improvement_ms"] == 350.0
    assert metrics["active_traces"] == 1

    # Fallbacks remain visible to whole-request safety and cost metrics.
    assert metrics["latency_ms"]["p95"] == 5_000.0
    assert metrics["over_4s"] == 1
    assert metrics["fallbacks"] == 2
    assert metrics["fallback_rate"] == 0.333333
    assert metrics["full_searches"] == 3
    assert metrics["full_search_rate"] == 0.5


def test_candidate_precision_counts_field_only_false_positives() -> None:
    metrics = recall_growth.candidate_metrics(
        [
            {
                "session_hash": "perfect",
                "status": "observed",
                "full_search_required": False,
                "field_page_ids": ["page-a"],
                "teacher_page_ids": ["page-a"],
            },
            {
                "session_hash": "field-only",
                "status": "observed",
                "full_search_required": False,
                "field_page_ids": ["false-positive"],
                "teacher_page_ids": [],
            },
        ]
    )

    assert metrics["field_pages"] == 2
    assert metrics["field_teacher_overlap"] == 1
    assert metrics["field_precision_against_teacher"] == 0.5
    assert metrics["teacher_top30_coverage"] == 1.0
    assert metrics["coverage_evidence_traces"] == 1


def test_all_fallback_candidate_evidence_fails_quality_gates(tmp_path: Path) -> None:
    inputs = _label_inputs(tmp_path)
    recalls: list[dict] = []
    pulls: list[dict] = []
    for index in range(200):
        page = f"page-{index}"
        recalls.append(
            {
                "decision_id": f"decision-{index}",
                "session_id": f"session-{index % 20}",
                "prompt_hash": f"{index:064x}",
                "evidence_features": {
                    "processor_shadow": {"committed_page_ids": [page]}
                },
            }
        )
        pulls.append(
            {
                "type": "used",
                "event_id": f"used-{index}",
                "decision_id": f"decision-{index}",
                "session_id": f"session-{index % 20}",
                "page_ids": [page],
            }
        )
    _write_jsonl(inputs["recall_log_file"], recalls)
    _write_jsonl(inputs["pull_log_file"], pulls)
    _write_jsonl(inputs["certificate_file"], [])
    _write_jsonl(inputs["golden_file"], [])
    fallback_rows = [
        {
            "session_hash": f"fallback-{index % 20}",
            "status": "fallback",
            "fallback_reason": "topic_reset",
            "authority": "teacher",
            "field_page_ids": [],
            "teacher_page_ids": [f"page-{index}"],
            "committed_page_ids": [f"page-{index}"],
            "latency_ms": 5_000 if index == 0 else 100,
            "teacher_latency_ms": 100,
            "full_search_required": True,
            "over_4s": index == 0,
        }
        for index in range(100)
    ]
    candidate_trace = tmp_path / "candidate.jsonl"
    _write_jsonl(candidate_trace, fallback_rows)

    result = recall_growth.run_growth_cycle(
        state_file=tmp_path / "growth-state.json",
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=candidate_trace,
        promotion_file=tmp_path / "promotion.json",
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    candidate = result["metrics"]["candidate"]
    assert candidate["traces"] == 100
    assert candidate["sessions"] == 20
    assert candidate["stable_traces"] == 0
    assert candidate["stable_sessions"] == 0
    assert candidate["quality_window_traces"] == 100
    assert candidate["quality_window_stable_traces"] == 0
    assert candidate["quality_window_stable_sessions"] == 0
    assert candidate["teacher_top30_coverage"] == 0.0
    assert candidate["teacher_commit_coverage"] == 0.0
    assert candidate["field_latency_ms"]["p95"] is None
    assert candidate["p95_improvement_ms"] is None
    assert candidate["full_search_rate"] == 1.0
    assert candidate["over_4s"] == 1
    assert result["gates"]["candidate_samples"] is False
    assert result["gates"]["candidate_sessions"] is False
    assert result["gates"]["candidate_coverage_evidence"] is False
    assert result["gates"]["candidate_commit_evidence"] is False
    assert result["gates"]["candidate_latency_evidence"] is False
    assert result["gates"]["teacher_top30_coverage"] is False
    assert result["gates"]["teacher_commit_coverage"] is False
    assert result["gates"]["p95_improvement"] is False
    assert result["authority_enabled"] is False


def test_sparse_quality_evidence_cannot_satisfy_candidate_gates(
    tmp_path: Path,
) -> None:
    inputs = _label_inputs(tmp_path)
    for path in inputs.values():
        _write_jsonl(path, [])
    empty_rows = [
        {
            "session_hash": f"empty-{index}",
            "status": "observed",
            "full_search_required": False,
            "latency_ms": 100,
        }
        for index in range(99)
    ]
    perfect = {
        "session_hash": "perfect",
        "status": "observed",
        "field_page_ids": ["page"],
        "teacher_page_ids": ["page"],
        "committed_page_ids": ["page"],
        "latency_ms": 100,
        "field_latency_ms": 10,
        "teacher_latency_ms": 100,
        "full_search_required": False,
    }
    candidate_trace = tmp_path / "candidate.jsonl"
    _write_jsonl(candidate_trace, [*empty_rows, perfect])

    result = recall_growth.run_growth_cycle(
        state_file=tmp_path / "growth-state.json",
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=candidate_trace,
        promotion_file=tmp_path / "promotion.json",
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    candidate = result["metrics"]["candidate"]
    assert candidate["quality_window_stable_traces"] == 100
    assert candidate["teacher_top30_coverage"] == 1.0
    assert candidate["teacher_commit_coverage"] == 1.0
    assert candidate["p95_improvement_ms"] == 90.0
    assert candidate["coverage_evidence_traces"] == 1
    assert candidate["commit_evidence_traces"] == 1
    assert candidate["paired_latency_traces"] == 1
    assert result["gates"]["candidate_samples"] is True
    assert result["gates"]["candidate_coverage_evidence"] is False
    assert result["gates"]["candidate_commit_evidence"] is False
    assert result["gates"]["candidate_latency_evidence"] is False
    assert result["authority_enabled"] is False


def test_canary_counter_migration_rebases_stable_units() -> None:
    assert recall_growth._advance_rollout(
        {
            "effective_mode": "active",
            "canary_percent": 25,
            "stage_started_trace_count": 500,
        },
        authority_eligible=True,
        candidate_trace_count=100,
    ) == ("active", 25, 100)
    assert recall_growth._advance_rollout(
        {
            "effective_mode": "active",
            "canary_percent": 5,
            "stage_started_stable_trace_count": 0,
        },
        authority_eligible=True,
        candidate_trace_count=100,
    ) == ("active", 5, 100)


def test_growth_cycle_promotes_qualified_evidence_through_canary(
    tmp_path: Path,
) -> None:
    inputs = _qualified_label_inputs(tmp_path)
    traces = _qualified_candidate_traces()
    candidate_trace = tmp_path / "candidate.jsonl"
    _write_jsonl(candidate_trace, traces)
    state = tmp_path / "growth-state.json"
    promotion = tmp_path / "promotion.json"

    result = recall_growth.run_growth_cycle(
        state_file=state,
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=candidate_trace,
        promotion_file=promotion,
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    assert result["stage"] == "collecting_labels"
    assert result["effective_mode"] == "candidate"
    assert result["field_learning_allowed"] is False
    assert result["authority_enabled"] is False
    assert json.loads(promotion.read_text())["status"] == "held"

    mode, percent, started = recall_growth._advance_rollout(
        {}, authority_eligible=True, candidate_trace_count=100
    )
    assert (mode, percent, started) == ("active", 5, 100)
    mode, percent, started = recall_growth._advance_rollout(
        {
            "effective_mode": mode,
            "canary_percent": percent,
            "stage_started_confidence_sample_count": started,
        },
        authority_eligible=True,
        candidate_trace_count=200,
    )
    assert (mode, percent, started) == ("active", 25, 200)
    mode, percent, started = recall_growth._advance_rollout(
        {
            "effective_mode": mode,
            "canary_percent": percent,
            "stage_started_confidence_sample_count": started,
        },
        authority_eligible=True,
        candidate_trace_count=300,
    )
    assert (mode, percent, started) == ("active", 100, 300)
    assert recall_growth._advance_rollout(
        {
            "effective_mode": mode,
            "canary_percent": percent,
            "stage_started_confidence_sample_count": started,
        },
        authority_eligible=False,
        candidate_trace_count=303,
    ) == ("candidate", 100, 303)


def test_growth_cycle_real_promotion_rejoins_live_evidence_and_advances_canary(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.recall import recall_answer_eval, recall_field_candidate

    inputs = _qualified_label_inputs(tmp_path)
    manual = _manual94_case_gate(tmp_path, monkeypatch)
    environment = json.loads(manual.read_text(encoding="utf-8"))[
        "environment_epoch"
    ]
    environment_sha = hashlib.sha256(
        json.dumps(
            environment,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    split_sha = "d" * 64
    train_check = {
        "passed": True,
        "method": "connected-cluster-bootstrap-percentile",
        "confidence": 0.95,
        "seed": 1729,
        "manifest_sha256": "b" * 64,
        "samples": 20,
        "distinct_clusters": 20,
        "point": 0.1,
        "lower": 0.05,
        "upper": 0.15,
        "split_manifest_sha256": split_sha,
    }
    locked_check = {
        **train_check,
        "manifest_sha256": "c" * 64,
        "environment_epoch_sha256": environment_sha,
    }
    set_check = {"passed": True, "split_manifest_sha256": split_sha}
    labels = {
        "labels": [],
        "counts": {
            "scope": "train",
            "strong_positive": 200,
            "strong_positive_sessions": 20,
            "total": 200,
        },
        "gates": {"field_learning_allowed": True},
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
    monkeypatch.setattr(recall_growth, "materialize_label_ledger", lambda **_: labels)
    monkeypatch.setattr(recall_growth, "split_integrity", lambda _rows: integrity)
    monkeypatch.setattr(
        recall_growth,
        "decide_learning_update",
        lambda *, current, **_: {
            "status": "candidate",
            "reason": "verified-test-candidate",
            "policy": current,
            "field_learning_allowed": True,
            "calibration_allowed": False,
        },
    )
    monkeypatch.setattr(
        recall_growth,
        "validate_answer_outcome_artifact",
        lambda *_args, **_kwargs: train_check,
    )
    monkeypatch.setattr(
        recall_growth,
        "validate_locked_answer_artifact",
        lambda *_args, **_kwargs: locked_check,
    )
    monkeypatch.setattr(
        recall_growth,
        "validate_answer_artifact_set",
        lambda **_kwargs: set_check,
    )
    monkeypatch.setattr(
        recall_answer_eval,
        "builtin_field_environment_identity",
        lambda: environment,
    )
    monkeypatch.setattr(
        recall_answer_eval,
        "validate_answer_outcome_artifact",
        lambda *_args, **_kwargs: train_check,
    )
    monkeypatch.setattr(
        recall_answer_eval,
        "validate_locked_answer_artifact",
        lambda *_args, **_kwargs: locked_check,
    )
    monkeypatch.setattr(
        recall_answer_eval,
        "validate_answer_artifact_set",
        lambda **_kwargs: set_check,
    )

    source_paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "train",
            "locked",
            "episodes",
            "reviews",
            "executions",
            "registry",
        )
    }
    for source in source_paths.values():
        source.write_text("{}\n", encoding="utf-8")
    state = tmp_path / "growth-state.json"
    promotion = tmp_path / "promotion.json"
    trace_file = tmp_path / "candidate.jsonl"
    expected_percents = ((100, 5), (200, 25), (300, 100))
    for count, expected_percent in expected_percents:
        _write_jsonl(trace_file, _qualified_candidate_traces(count))
        result = recall_growth.run_growth_cycle(
            state_file=state,
            history_file=tmp_path / "history.jsonl",
            candidate_trace_file=trace_file,
            promotion_file=promotion,
            policy_history_file=tmp_path / "policy-history.jsonl",
            last_known_good_file=tmp_path / "last-known-good-policy.json",
            locked_e2e_file=manual,
            locked_answer_eval_file=source_paths["locked"],
            train_answer_eval_file=source_paths["train"],
            answer_episode_file=source_paths["episodes"],
            answer_review_ledger_file=source_paths["reviews"],
            answer_execution_ledger_file=source_paths["executions"],
            answer_adapter_registry=source_paths["registry"],
            label_inputs=inputs,
            now=datetime.now(UTC),
        )
        assert result["authority_enabled"] is True
        assert result["canary_percent"] == expected_percent
        assert recall_field_candidate.authority_allowed(promotion) is True

    original_promotion = promotion.read_bytes()
    _write_jsonl(trace_file, _qualified_candidate_traces(301))
    assert recall_field_candidate.authority_allowed(promotion) is True
    with inputs["recall_log_file"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "later-recall"}) + "\n")
    with inputs["pull_log_file"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "later-pull"}) + "\n")
    assert recall_field_candidate.authority_allowed(promotion) is True

    changed_environment = {
        **environment,
        "lkg_base_artifact_sha256": "f" * 64,
        "effective_field_config_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        recall_answer_eval,
        "builtin_field_environment_identity",
        lambda: changed_environment,
    )
    assert recall_field_candidate.authority_allowed(promotion) is False
    monkeypatch.setattr(
        recall_answer_eval,
        "builtin_field_environment_identity",
        lambda: environment,
    )

    payload = json.loads(original_promotion)
    payload["confidence_evidence"]["candidate"] = recall_growth._candidate_growth_metrics(
        _qualified_candidate_traces(100)
    )["confidence"]
    unsigned = {key: value for key, value in payload.items() if key != "snapshot_sha256"}
    payload["snapshot_sha256"] = recall_field_candidate._canonical_sha256(unsigned)
    promotion.write_text(json.dumps(payload), encoding="utf-8")
    assert recall_field_candidate.authority_allowed(promotion) is False

    promotion.write_bytes(original_promotion)
    payload = json.loads(original_promotion)
    generated = datetime.fromisoformat(payload["generated_at"].replace("Z", "+00:00"))
    payload["expires_at"] = (
        generated.replace(year=generated.year + 1)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    unsigned = {key: value for key, value in payload.items() if key != "snapshot_sha256"}
    payload["snapshot_sha256"] = recall_field_candidate._canonical_sha256(unsigned)
    promotion.write_text(json.dumps(payload), encoding="utf-8")
    assert recall_field_candidate.authority_allowed(promotion) is False

    promotion.write_bytes(original_promotion)
    payload = json.loads(original_promotion)
    payload["generated_at"] = "2026-08-02T00:00:00"
    unsigned = {key: value for key, value in payload.items() if key != "snapshot_sha256"}
    payload["snapshot_sha256"] = recall_field_candidate._canonical_sha256(unsigned)
    promotion.write_text(json.dumps(payload), encoding="utf-8")
    assert recall_field_candidate.authority_allowed(promotion) is False

    incomplete_rows = _qualified_candidate_traces(301)
    incomplete = incomplete_rows[-1]
    incomplete["page_uids"] = {}
    incomplete["cumulative_eligible_trace_count"] = 300
    incomplete.pop("record_sha256")
    incomplete["record_sha256"] = hashlib.sha256(
        json.dumps(incomplete, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_jsonl(trace_file, incomplete_rows)
    incomplete_result = recall_growth.run_growth_cycle(
        state_file=state,
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=trace_file,
        promotion_file=promotion,
        policy_history_file=tmp_path / "policy-history.jsonl",
        last_known_good_file=tmp_path / "last-known-good-policy.json",
        locked_e2e_file=manual,
        locked_answer_eval_file=source_paths["locked"],
        train_answer_eval_file=source_paths["train"],
        answer_episode_file=source_paths["episodes"],
        answer_review_ledger_file=source_paths["reviews"],
        answer_execution_ledger_file=source_paths["executions"],
        answer_adapter_registry=source_paths["registry"],
        label_inputs=inputs,
        now=datetime.now(UTC),
    )
    assert incomplete_result["authority_enabled"] is False
    assert incomplete_result["gates"]["candidate_confidence_complete"] is False
    assert incomplete_result["gates"]["candidate_cumulative_exact"] is True

    promotion.write_bytes(original_promotion)
    with trace_file.open("a", encoding="utf-8") as handle:
        handle.write("{malformed\n")
    assert recall_field_candidate.authority_allowed(promotion) is False

    _write_jsonl(trace_file, _qualified_candidate_traces(301))
    payload = json.loads(original_promotion)
    inputs["recall_log_file"].write_text("", encoding="utf-8")
    inputs["pull_log_file"].write_text("", encoding="utf-8")
    empty_sha = hashlib.sha256(b"").hexdigest()
    payload["source_artifacts"]["recall_log"]["file_sha256"] = empty_sha
    payload["source_artifacts"]["pull_log"]["file_sha256"] = empty_sha
    unsigned = {key: value for key, value in payload.items() if key != "snapshot_sha256"}
    payload["snapshot_sha256"] = recall_field_candidate._canonical_sha256(unsigned)
    promotion.write_text(json.dumps(payload), encoding="utf-8")
    assert recall_field_candidate.authority_allowed(promotion) is False


def test_automatic_rollout_and_learning_fail_closed(tmp_path: Path) -> None:
    state = tmp_path / "growth-state.json"
    assert recall_growth.automatic_rollout(
        enabled=True,
        state_file=state,
    ) == ("candidate", 100)
    assert not recall_growth.automatic_learning_allowed(
        enabled=True,
        state_file=state,
    )
    state.write_text(
        json.dumps(
            {
                "effective_mode": "active",
                "canary_percent": 25,
                "authority_enabled": True,
                "field_learning_allowed": True,
            }
        ),
        encoding="utf-8",
    )

    assert recall_growth.automatic_rollout(
        enabled=True,
        state_file=state,
    ) == ("candidate", 100)
    assert not recall_growth.automatic_learning_allowed(
        enabled=True,
        state_file=state,
    )


def test_growth_quality_gate_uses_recent_window(tmp_path: Path) -> None:
    inputs = _qualified_label_inputs(tmp_path, count=220)
    bad = [
        {
            "schema_version": 2,
            "session_hash": f"bad-{index}",
            "query_sha256": f"{index + 7000:064x}",
            "content_sha256": f"{index + 8000:064x}",
            "status": "observed",
            "field_page_ids": [],
            "teacher_page_ids": [f"old-{index}"],
            "committed_page_ids": [f"old-{index}"],
            "latency_ms": 5_000,
            "full_search_required": False,
            "over_4s": True,
            "authority": "teacher",
        }
        for index in range(20)
    ]
    good = [
        {
            "schema_version": 2,
            "session_hash": f"good-{index % 20}",
            "query_sha256": f"{index + 9000:064x}",
            "content_sha256": f"{index + 10000:064x}",
            "status": "observed",
            "field_page_ids": [f"page-{index}"],
            "teacher_page_ids": [f"page-{index}"],
            "committed_page_ids": [f"page-{index}"],
            "latency_ms": 100,
            "field_latency_ms": 20,
            "teacher_latency_ms": 100,
            "full_search_required": False,
            "over_4s": False,
            "authority": "teacher",
        }
        for index in range(200)
    ]
    fallback = [
        {
            "session_hash": f"fallback-{index}",
            "status": "fallback",
            "fallback_reason": "topic_reset",
            "field_page_ids": [],
            "teacher_page_ids": [f"fallback-page-{index}"],
            "committed_page_ids": [f"fallback-page-{index}"],
            "latency_ms": 100,
            "teacher_latency_ms": 100,
            "full_search_required": True,
            "over_4s": False,
            "authority": "teacher",
        }
        for index in range(50)
    ]
    candidate_trace = tmp_path / "candidate.jsonl"
    _write_jsonl(candidate_trace, bad + good + fallback)

    result = recall_growth.run_growth_cycle(
        state_file=tmp_path / "growth-state.json",
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=candidate_trace,
        promotion_file=tmp_path / "promotion.json",
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    assert result["authority_enabled"] is False
    assert result["metrics"]["candidate"]["traces"] == 270
    assert result["metrics"]["candidate"]["stable_traces"] == 220
    assert result["metrics"]["candidate"]["quality_window_traces"] == 200
    assert result["metrics"]["candidate"]["quality_window_stable_traces"] == 200
    assert result["metrics"]["candidate"]["quality_window_stable_sessions"] == 20
    assert result["metrics"]["candidate"]["teacher_top30_coverage"] == 1.0
    assert result["metrics"]["candidate"]["full_search_rate"] == 0.25
    assert result["metrics"]["candidate"]["over_4s"] == 0
    assert result["metrics"]["processor_used"]["episodes"] == 220
    assert result["metrics"]["processor_used"]["quality_window_episodes"] == 200


def test_growth_writer_is_hard_off_for_distillation_single_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_file = tmp_path / "growth-state.json"
    history_file = tmp_path / "growth-history.jsonl"
    monkeypatch.setattr(
        recall_growth, "_distillation_single_writer_active", lambda: True
    )

    result = recall_growth.run_growth_cycle(
        state_file=state_file,
        history_file=history_file,
    )

    assert result == {
        "status": "hard_off",
        "reason": "distillation_single_writer",
        "dry_run": False,
    }
    assert not state_file.exists()
    assert not history_file.exists()
