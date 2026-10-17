from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_ledger as ledger
from chronovisor.recall import recall_distillation_store as store


@pytest.fixture(autouse=True)
def _clear_ledger_cache() -> Generator[None]:
    ledger.clear_cache()
    yield
    ledger.clear_cache()


def _payloads() -> list[dict[str, object]]:
    return [
        {
            "kind": "label",
            "rally_id": "rally-a",
            "candidate_id": "candidate-a1",
            "receipt_sha256": "receipt-a1",
            "work_id": "work-a1",
            "profile": "profile-a",
            "profile_contract_id": "contract-a",
            "authority": "teacher-only",
            "assignment": {"probe": True},
        },
        {
            "kind": "label",
            "rally_id": "rally-a",
            "candidate_id": "candidate-a2",
            "receipt_sha256": "receipt-a2",
            "work_id": "work-a2",
            "profile": "profile-a",
            "profile_contract_id": "contract-a",
            "authority": "teacher-only",
            "assignment": {"probe": False},
        },
        {
            "kind": "receipt",
            "rally_id": "rally-b",
            "candidate_id": "candidate-b1",
            "receipt_sha256": "receipt-b1",
            "work_id": "work-b1",
            "profile": "profile-b",
            "profile_contract_id": "contract-b",
            "authority": "verified",
        },
    ]


def _make_ledger(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    path = tmp_path / "labels.jsonl"
    rows = store.append_chain_batch(path, _payloads())
    return path, rows


def test_projection_preserves_complete_ordered_rows_and_keyed_views(
    tmp_path: Path,
) -> None:
    path, expected = _make_ledger(tmp_path)
    index = ledger.ledger_index(path)

    assert index.rows() == expected
    assert index.rows(after=1) == expected[1:]
    assert index.rows(kind="label") == expected[:2]
    assert index.distinct("rally_id") == {"rally-a", "rally-b"}
    assert index.distinct("kind") == {"label", "receipt"}
    assert index.lookup("rally_id", "rally-a") == expected[1]
    assert index.lookup("rally_id", "missing") is None

    mapping = ledger.LedgerMapping(index, "rally_id")
    assert set(mapping) == {"rally-a", "rally-b"}
    assert len(mapping) == 2
    assert mapping["rally-a"] == expected[1]
    with pytest.raises(KeyError):
        _ = mapping["missing"]
    with pytest.raises(ValueError, match="unknown ledger projection field"):
        index.distinct("row_json")


def test_clear_cache_unchanged_snapshot_does_not_decode_source_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, expected = _make_ledger(tmp_path)
    assert ledger.ledger_index(path).rows() == expected
    ledger.clear_cache()

    def unexpected_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unchanged projection must use its verified cursor")

    monkeypatch.setattr(store, "_decode_chain_bytes", unexpected_decode)
    reopened = ledger.ledger_index(path)
    assert reopened.rows() == expected


def test_append_reopens_from_persisted_cursor_and_decodes_suffix_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, expected = _make_ledger(tmp_path)
    assert ledger.ledger_index(path).rows() == expected
    ledger.clear_cache()
    prefix_size = path.stat().st_size
    appended = store.append_chain(
        path,
        {
            "kind": "label",
            "rally_id": "rally-c",
            "candidate_id": "candidate-c1",
            "receipt_sha256": "receipt-c1",
            "authority": "teacher-only",
        },
    )
    # The old projection's cursor is recovered from SQLite; this spy must see
    # only the newly appended JSONL suffix, never the old ledger prefix.
    decoded_sizes: list[int] = []
    real_decode = store._decode_chain_bytes

    def track_decode(data: bytes, **kwargs: object) -> object:
        decoded_sizes.append(len(data))
        return real_decode(data, **kwargs)

    monkeypatch.setattr(store, "_decode_chain_bytes", track_decode)
    reopened = ledger.ledger_index(path)
    assert reopened.rows() == [*expected, appended]
    assert decoded_sizes
    assert decoded_sizes[-1] == len(
        json.dumps(appended, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    )
    assert decoded_sizes[-1] == path.stat().st_size - prefix_size


def test_projection_database_tamper_rebuilds_from_verified_source(
    tmp_path: Path,
) -> None:
    path, expected = _make_ledger(tmp_path)
    index = ledger.ledger_index(path)
    projection = index.index_path
    ledger.clear_cache()

    with sqlite3.connect(projection) as connection:
        connection.execute(
            "UPDATE rows SET row_json=? WHERE sequence=1",
            (json.dumps({"kind": "forged"}, separators=(",", ":")),),
        )

    rebuilt = ledger.ledger_index(path)
    assert rebuilt.rows() == expected


def test_corrupt_projection_checkpoint_does_not_leave_stale_rows(
    tmp_path: Path,
) -> None:
    path, expected = _make_ledger(tmp_path)
    index = ledger.ledger_index(path)
    checkpoint = index.index_path.with_suffix(index.index_path.suffix + ".checkpoint.json")
    ledger.clear_cache()
    checkpoint.write_bytes(b"not a sealed checkpoint")

    rebuilt = ledger.ledger_index(path)
    assert rebuilt.rows() == expected


def test_source_checkpoint_corruption_forces_full_verified_reset(
    tmp_path: Path,
) -> None:
    path, expected = _make_ledger(tmp_path)
    assert ledger.ledger_index(path).rows() == expected
    ledger.clear_cache()
    source_checkpoint = store._chain_checkpoint_path(path)
    source_checkpoint.write_bytes(b"not a sealed checkpoint")

    rebuilt = ledger.ledger_index(path)
    assert rebuilt.rows() == expected
    assert source_checkpoint.read_bytes() == b"not a sealed checkpoint"


def test_source_head_tamper_fails_closed_without_stale_projection(
    tmp_path: Path,
) -> None:
    path, _expected = _make_ledger(tmp_path)
    assert ledger.ledger_index(path).rows()
    ledger.clear_cache()
    head = store.chain_head(path)
    store._write_chain_checkpoint(
        path, {"records": head["records"], "head_sha256": "0" * 64}
    )

    with pytest.raises(store.DistillationStoreError, match="checkpoint|chain"):
        ledger.ledger_index(path)


def test_source_rewrite_and_truncation_are_rejected(
    tmp_path: Path,
) -> None:
    path, _expected = _make_ledger(tmp_path)
    assert ledger.ledger_index(path).rows()
    ledger.clear_cache()
    raw = path.read_bytes()
    assert b'"rally_id":"rally-a"' in raw
    path.write_bytes(raw.replace(b'"rally_id":"rally-a"', b'"rally_id":"rally-z"', 1))
    with pytest.raises(store.DistillationStoreError, match="chain|checkpoint"):
        ledger.ledger_index(path)

    ledger.clear_cache()
    path.write_bytes(raw)
    assert ledger.ledger_index(path).rows()
    ledger.clear_cache()
    path.write_bytes(raw.splitlines(keepends=True)[0])
    with pytest.raises(store.DistillationStoreError, match="truncated|checkpoint|chain"):
        ledger.ledger_index(path)


def test_pinned_view_rejects_append_during_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _expected = _make_ledger(tmp_path)
    index = ledger.ledger_index(path)
    original_current = index.current
    calls = 0

    def append_between_checks() -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            store.append_chain(
                path,
                {
                    "kind": "label",
                    "rally_id": "rally-c",
                    "candidate_id": "candidate-c1",
                    "authority": "teacher-only",
                },
            )
        return bool(original_current())

    monkeypatch.setattr(index, "current", append_between_checks)
    with pytest.raises(store.DistillationStoreError, match="changed"):
        index.rows()
    assert calls == 2


def test_verified_counts_keep_authority_and_verdict_semantics_without_full_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    label_path = store.distillation_dir(tmp_path) / store.LABEL_LEDGER_FILE
    store.append_chain_batch(
        label_path,
        (
            {
                "kind": "teacher-label",
                "rally_id": "rally-a",
                "candidate_id": "candidate-a",
                "authority": "verified",
                "dimension": "answer_utility",
                "verdict": "helpful",
            },
            {
                "kind": "teacher-label",
                "rally_id": "rally-b",
                "candidate_id": "candidate-b",
                "authority": "verified",
                "dimension": "answer_utility",
                "verdict": "neutral",
            },
            {
                "kind": "teacher-label",
                "rally_id": "rally-c",
                "candidate_id": "candidate-c",
                "authority": "verified",
                "dimension": "answer_utility",
                "verdict": "harmful",
            },
            {
                "kind": "teacher-label",
                "rally_id": "rally-d",
                "candidate_id": "candidate-d",
                "authority": "teacher-only",
                "dimension": "answer_utility",
                "verdict": "helpful",
            },
            {
                "kind": "teacher-label",
                "rally_id": "rally-e",
                "candidate_id": "candidate-e",
                "authority": "verified",
                "dimension": "relevance",
                "verdict": "relevant",
            },
        ),
    )
    real_read_chain = store.read_chain

    def reject_full_label_decode(path: Path) -> list[dict[str, object]]:
        if path == label_path:
            raise AssertionError("verified counts must use the label projection")
        return real_read_chain(path)

    monkeypatch.setattr(distill.store, "read_chain", reject_full_label_decode)
    assert distill._verified_counts(tmp_path) == {
        "helpful": 1,
        "neutral": 1,
        "harmful": 1,
    }


def test_automatic_structural_metric_uses_verified_projection_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    label_path = store.distillation_dir(tmp_path) / store.LABEL_LEDGER_FILE
    store.append_chain_batch(
        label_path,
        (
            {
                "kind": "teacher-label",
                "rally_id": "rally-a",
                "candidate_id": "candidate-a",
                "authority": "verified",
                "dimension": "relevance",
                "verdict": "relevant",
                "closed_predicate": "exact_claim_supported",
            },
            {
                "kind": "teacher-label",
                "rally_id": "rally-b",
                "candidate_id": "candidate-b",
                "authority": "verified",
                "dimension": "relevance",
                "verdict": "irrelevant",
                "closed_predicate": "exact_path_overlap",
            },
            {
                "kind": "teacher-label",
                "rally_id": "rally-c",
                "candidate_id": "candidate-c",
                "authority": "teacher-only",
                "dimension": "relevance",
                "verdict": "relevant",
                "closed_predicate": "exact_claim_supported",
            },
            {
                "kind": "teacher-label",
                "rally_id": "rally-d",
                "candidate_id": "candidate-d",
                "authority": "verified",
                "dimension": "answer_utility",
                "verdict": "helpful",
                "closed_predicate": "exact_claim_supported",
            },
        ),
    )
    monkeypatch.setattr(distill, "_exposure_map", lambda _root: {})
    real_read_chain = distill._read_chain

    def reject_label_scan(path: Path) -> list[dict[str, object]]:
        if path == label_path:
            raise AssertionError("structural metrics must use the label projection")
        return real_read_chain(path)

    monkeypatch.setattr(distill, "_read_chain", reject_label_scan)
    assert distill._automatic_baseline_metrics(tmp_path)["exact_structural_links"] == 2
