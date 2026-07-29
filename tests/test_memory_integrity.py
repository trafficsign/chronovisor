from __future__ import annotations

import json
from pathlib import Path

from chronovisor.ops import memory_integrity


class _Result:
    page_id = "target"
    title = "Target"
    score = 1.0


def test_evaluate_raw_passes_when_search_finds_page(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "20260706-codex-target.md"
    raw.write_text("Target project decision about qwen mlx", encoding="utf-8")
    monkeypatch.setattr(memory_integrity, "run_search", lambda **kwargs: ([_Result()], "hybrid"))

    row = memory_integrity.evaluate_raw(raw, claimed=set())

    assert row["status"] == "pass"
    assert row["search_present"] is True
    assert row["top_pages"][0]["page_id"] == "target"


def test_evaluate_raw_does_not_pass_from_claim_presence_only(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "20260706-codex-target.md"
    raw.write_text(
        "---\nraw_keywords: [target]\nentities: [target]\n---\n"
        "Independent body fact that search will miss.",
        encoding="utf-8",
    )
    monkeypatch.setattr(memory_integrity, "run_search", lambda **kwargs: ([], "hybrid"))

    row = memory_integrity.evaluate_raw(raw, claimed={raw.name})

    assert row["claim_present"] is True
    assert row["claim_present_is_audit_only"] is True
    assert row["status"] == "miss"
    assert "target" not in row["terms"]


def test_claimed_raw_names_strips_replay_prefix(tmp_path: Path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    claims = chronovisor_root / "claims"
    claims.mkdir(parents=True)
    (claims / "claims.jsonl").write_text(
        json.dumps({"source_raw": "replay:20260706-codex-a.md"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(memory_integrity, "CHRONOVISOR_ROOT", chronovisor_root)

    assert memory_integrity.claimed_raw_names() == {"20260706-codex-a.md"}


def test_claimed_raw_names_streams_escaped_values_and_skips_bad_rows(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "claims.jsonl"
    ledger.write_bytes(
        b'{"source_raw":"replay:folder\\\\raw-a.md","value":"ok"}\n'
        b'{"source_raw":broken}\n'
        b'{"source_raw":"","value":"empty"}\n'
    )

    assert memory_integrity.claimed_raw_names(path=ledger) == {"folder\\raw-a.md"}
