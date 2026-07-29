from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.librarian.merge_ledger import (
    MergeCoverageError,
    MergeLedger,
    build_source_inventory,
    verify_merge_coverage,
)
from chronovisor.librarian.merge_transaction import apply_merge_plan, prepare_merge_plan


def _mappings(inventory: dict) -> list[dict[str, str]]:
    return [
        {
            "source_uid": uid,
            "span_sha256": span["sha256"],
            "action": "output" if span["kind"] == "claim" else "boilerplate",
        }
        for uid, source in inventory.items()
        for span in source["spans"]
    ]


def test_merge_preflight_requires_spans_fingerprints_and_sensitivity() -> None:
    inventory = build_source_inventory(
        {
            "uid-a": (
                "Release 2026-07-25 uses 42GB. "
                "See https://example.com and Chronovisor.PageRegistry."
            )
        }
    )
    mappings = _mappings(inventory)
    output = (
        "Release 2026-07-25 uses 42GB. "
        "See https://example.com and Chronovisor.PageRegistry."
    )
    result = verify_merge_coverage(
        inventory=inventory,
        mappings=mappings,
        output_text=output,
        input_sensitivities=["high"],
        output_sensitivity="high",
    )
    assert result["status"] == "verified"

    with pytest.raises(MergeCoverageError, match="missing_fingerprints"):
        verify_merge_coverage(
            inventory=inventory,
            mappings=mappings,
            output_text="Release summary.",
            input_sensitivities=["high"],
            output_sensitivity="high",
        )
    with pytest.raises(MergeCoverageError, match="required_sensitivity"):
        verify_merge_coverage(
            inventory=inventory,
            mappings=mappings,
            output_text=output,
            input_sensitivities=["high"],
            output_sensitivity="normal",
        )


def test_merge_preflight_rejects_claim_as_boilerplate_and_ambiguous_repeat() -> None:
    inventory = build_source_inventory({"uid-a": "維持する。維持する。"})
    spans = inventory["uid-a"]["spans"]
    assert len(spans) == 2
    with pytest.raises(
        MergeCoverageError,
        match="claim_cannot_be_declared_boilerplate",
    ):
        verify_merge_coverage(
            inventory=inventory,
            mappings=[
                {
                    "source_uid": "uid-a",
                    "span_sha256": span["sha256"],
                    "span_index": span["index"],
                    "action": "boilerplate",
                }
                for span in spans
            ],
            output_text="維持する。",
            output_sensitivity="normal",
        )
    with pytest.raises(MergeCoverageError, match="ambiguous_repeated_span"):
        verify_merge_coverage(
            inventory=inventory,
            mappings=[
                {
                    "source_uid": "uid-a",
                    "span_sha256": spans[0]["sha256"],
                    "action": "output",
                }
            ],
            output_text="維持する。",
            output_sensitivity="normal",
        )


def test_merge_ledger_is_append_only(tmp_path: Path) -> None:
    ledger = MergeLedger(tmp_path)
    ledger.append({"transaction_id": "tx-1", "operation": "keep-both"})
    ledger.append({"transaction_id": "tx-2", "operation": "merge"})
    assert [row["transaction_id"] for row in ledger.recent()] == ["tx-1", "tx-2"]


def test_merge_transaction_requires_activation_and_redirects_old_uid(
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    alpha_text = "---\ntitle: Alpha\n---\n\nAlpha uses 42GB.\n"
    beta_text = "---\ntitle: Beta\n---\n\nBeta date is 2026-07-25.\n"
    (pages / "alpha.md").write_text(alpha_text, encoding="utf-8")
    (pages / "beta.md").write_text(beta_text, encoding="utf-8")
    registry = PageRegistry(tmp_path)
    registry.ensure_manifest()
    alpha = registry.resolve("alpha")
    beta = registry.resolve("beta")
    assert alpha and beta
    combined = (
        alpha_text
        + f"\n^chronovisor-source-uid-{alpha['uid']}\n"
        + beta_text
        + f"\n^chronovisor-source-uid-{beta['uid']}\n"
    )
    inventory = build_source_inventory(
        {alpha["uid"]: alpha_text, beta["uid"]: beta_text}
    )
    mappings = [
        {
            "source_uid": uid,
                "span_sha256": span["sha256"],
                "action": "output" if span["kind"] == "claim" else "boilerplate",
                "output_anchor": (
                    f"chronovisor-source-uid-{uid}"
                    if span["kind"] == "claim"
                    else None
                ),
                "raw_refs": [f"raw:test#{uid}"],
        }
        for uid, source in inventory.items()
        for span in source["spans"]
    ]
    plan = prepare_merge_plan(
        tmp_path,
        source_keys=["alpha", "beta"],
        canonical_key="alpha",
        canonical_content=combined,
        mappings=mappings,
        output_sensitivity="normal",
    )

    assert apply_merge_plan(tmp_path, plan)["status"] == "blocked"
    committed = apply_merge_plan(tmp_path, plan, activate=True)

    assert committed["status"] == "committed"
    assert not (pages / "beta.md").exists()
    assert (pages / "alpha.md").read_text(encoding="utf-8") == combined
    resolved = registry.resolve("beta")
    assert resolved and resolved["uid"] == alpha["uid"]
