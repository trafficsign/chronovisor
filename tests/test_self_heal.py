from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def isolated_wiki(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    wiki_root = tmp_path / "wiki"
    pages = wiki_root / "pages"
    raw = wiki_root / "raw"
    system = wiki_root / "system"
    runtime = wiki_root / "runtime"
    for d in (pages, raw, system, runtime):
        d.mkdir(parents=True, exist_ok=True)

    from llm_wiki_mcp import wiki, runtime_status

    monkeypatch.setattr(wiki, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages)
    monkeypatch.setattr(wiki, "RAW_DIR", raw)
    monkeypatch.setattr(wiki, "SYSTEM_DIR", system)
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime / "metrics.jsonl")
    return wiki_root


def _seed_page(wiki_root: Path, rel: str) -> None:
    path = wiki_root / "pages" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\ntitle: T\nupdated: 2026-01-01\n---\nbody\n")


def _write_packet(wiki_root: Path) -> Path:
    packet = {
        "failure_id": "f1",
        "raw_file": "broken.md",
        "failure_class": "apply.update_target_not_found",
        "fingerprint": "apply.update_target_not_found:model-made-up-target",
        "attempts": 3,
        "error": "update target not found for page_id 'model-made-up-target'",
        "requested_page_id": "model-made-up-target",
        "similar_existing_pages": ["ai/canonical-target"],
        "status": "pending_local_repair",
    }
    path = wiki_root / "runtime" / "failures" / "packets" / "f1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet))
    return path


def test_local_repair_adds_alias_restores_raw_and_retries(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_wiki_mcp import self_heal
    from llm_wiki_mcp.alias_store import load_aliases

    _seed_page(isolated_wiki, "ai/canonical-target.md")
    packet_path = _write_packet(isolated_wiki)
    quarantined = isolated_wiki / "runtime" / "failures" / "quarantined-raw" / "broken.md"
    quarantined.parent.mkdir(parents=True, exist_ok=True)
    quarantined.write_text("raw body")

    monkeypatch.setattr(
        self_heal,
        "_retry_ingest",
        lambda *, dry_run: {"triggered": True, "files_processed": ["broken.md"]},
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=False,
        dry_run=False,
    )

    assert result["status"] == "local_repair_applied"
    assert load_aliases()["model-made-up-target"] == "ai/canonical-target"
    assert not quarantined.exists()
    assert (isolated_wiki / "raw" / "broken.md").exists()
    updated_packet = json.loads(packet_path.read_text())
    assert updated_packet["status"] == "local_repair_applied"


def test_drill_returns_local_repair_decision(isolated_wiki: Path) -> None:
    from llm_wiki_mcp.self_heal import run_drill

    result = run_drill(use_qwen=False)

    assert result["decision"]["status"] == "resolved"
    assert result["decision"]["action"] == "resolve_update_target"


def test_auto_apply_error_packet_escalates_to_frontier_deterministically(isolated_wiki: Path) -> None:
    from llm_wiki_mcp.local_repair import propose_repair

    packet = {
        "failure_class": "recall.auto_apply_error",
        "fingerprint": "recall.auto_apply_error:page_tag:invalid_page_tag",
        "attempts": 425,
        "auto_apply_error": {"error_kind": "page_tag:invalid_page_tag"},
    }

    decision = propose_repair(packet, use_qwen=False)

    assert decision.status == "escalate"
    assert decision.action == "escalate_to_frontier"
    assert decision.confidence >= 0.85


def test_sandbox_drill_runs_pending_raw_to_self_heal(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_wiki_mcp.self_heal import run_sandbox_drill

    monkeypatch.setenv("LLM_WIKI_SELF_HEAL_AUTORUN", "0")

    result = run_sandbox_drill(use_qwen=False)

    assert result["status"] == "ok"
    assert result["packet_paths"]
    assert result["heal_result"]["status"] == "local_repair_applied"
    assert result["pending_after"] == []
    assert (
        result["aliases"]["opus-4-7-evaluation-and-industry-geopolitics"]
        == "ai/opus-4.7-evaluation-and-industry-geopolitics"
    )


def test_frontier_human_required_sends_notification(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_wiki_mcp import self_heal
    from llm_wiki_mcp.local_repair import LocalRepairDecision

    packet = {
        "failure_id": "auth1",
        "raw_file": "auth-broken.md",
        "failure_class": "triage.parse_failed",
        "fingerprint": "triage.parse_failed",
        "attempts": 3,
        "error": "triage parse failed",
        "status": "pending_local_repair",
    }
    packet_path = isolated_wiki / "runtime" / "failures" / "packets" / "auth1.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    sent: list[tuple[str, str]] = []

    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: LocalRepairDecision(
            status="escalate",
            action="escalate_to_frontier",
            confidence=0.9,
            reason="needs frontier",
            source="deterministic",
        ),
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: {
            "decision": "needs_retry",
            "summary": "codex auth missing",
            "tests_run": [],
            "commit": None,
            "committed": False,
            "pushed": False,
            "risk": None,
            "notes": None,
            "raw_output": "401 Unauthorized",
            "frontier_failure": {
                "failure_class": "auth_required",
                "rescue_status": "human_required",
                "summary": "frontier API authentication is missing or invalid",
                "human_required": True,
                "notify_user": True,
            },
            "rescue_status": "human_required",
            "human_required": True,
            "notify_user": True,
            "rescue_attempt": None,
        },
    )
    monkeypatch.setattr(
        self_heal,
        "_send_mac_notification",
        lambda title, body: sent.append((title, body)) or {"sent": True},
    )

    result = self_heal.handle_packet(packet_path, use_qwen=False)

    assert result["status"] == "human_required"
    assert sent == [
        (
            self_heal.MAC_NOTIFICATION_TITLE,
            "Codex の認証が切れている可能性があります。ログイン確認が必要です。",
        )
    ]
    updated_packet = json.loads(packet_path.read_text())
    assert updated_packet["status"] == "human_required"
    assert updated_packet["human_notification"]["delivery"]["sent"] is True


def test_frontier_pending_review_writes_artifact(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_wiki_mcp import self_heal
    from llm_wiki_mcp.local_repair import LocalRepairDecision

    packet = {
        "failure_id": "schema1",
        "raw_file": "schema-broken.md",
        "failure_class": "triage.parse_failed",
        "fingerprint": "triage.parse_failed",
        "attempts": 3,
        "error": "triage parse failed",
        "status": "pending_local_repair",
    }
    packet_path = isolated_wiki / "runtime" / "failures" / "packets" / "schema1.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: LocalRepairDecision(
            status="escalate",
            action="escalate_to_frontier",
            confidence=0.9,
            reason="needs frontier",
            source="deterministic",
        ),
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: {
            "decision": "needs_retry",
            "summary": "codex schema needs review",
            "tests_run": [],
            "commit": None,
            "committed": False,
            "pushed": False,
            "risk": None,
            "notes": None,
            "frontier_failure": {
                "failure_class": "schema_invalid",
                "rescue_status": "pending_frontier_review",
                "summary": "schema repaired but needs review",
                "human_required": False,
                "notify_user": False,
            },
            "rescue_status": "pending_frontier_review",
            "human_required": False,
            "notify_user": False,
            "access_repair": {
                "applied": True,
                "repairs": [{"type": "schema_strictness_autofix"}],
            },
        },
    )

    result = self_heal.handle_packet(packet_path, use_qwen=False)

    assert result["status"] == "pending_frontier_review"
    pending_path = Path(result["pending_frontier_review_path"])
    assert pending_path.exists()
    pending = json.loads(pending_path.read_text())
    assert pending["status"] == "pending_frontier_review"
    assert pending["access_repair"]["applied"] is True
    updated_packet = json.loads(packet_path.read_text())
    assert updated_packet["pending_frontier_review_path"] == str(pending_path)


def test_human_required_notification_cooldown(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timedelta

    from llm_wiki_mcp import self_heal

    packet = {
        "failure_id": "auth1",
        "raw_file": "auth-broken.md",
        "fingerprint": "triage.parse_failed",
    }
    frontier_result = {
        "human_required": True,
        "notify_user": True,
        "frontier_failure": {"failure_class": "auth_required"},
        "rescue_status": "human_required",
    }
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        self_heal,
        "_send_mac_notification",
        lambda title, body: sent.append((title, body)) or {"sent": True},
    )
    now = datetime(2026, 6, 4, 22, 0, 0)

    first = self_heal.maybe_notify_human_required(packet, frontier_result, now=now)
    second = self_heal.maybe_notify_human_required(
        packet,
        frontier_result,
        now=now + timedelta(seconds=10),
    )

    assert first["delivery"]["sent"] is True
    assert second["reason"] == "cooldown"
    assert len(sent) == 1
