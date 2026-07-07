from __future__ import annotations

from pathlib import Path

from llm_wiki_mcp import autonomy


def test_duplicate_decision_defers_uncertain_pair(monkeypatch) -> None:
    monkeypatch.setattr(autonomy, "_page_meta", lambda page_id: {"page_id": page_id})
    decision = autonomy.decide_duplicate(
        {
            "left": "a",
            "right": "b",
            "left_title": "Alpha",
            "right_title": "Beta",
            "score": 0.999,
            "method": "embedding",
        }
    )

    assert decision["action"] == "defer"
    assert decision["reason"] == "title_mismatch"


def test_duplicate_resolution_supersedes_exact_high_confidence_pair(monkeypatch, tmp_path: Path) -> None:
    pages = {
        "rich": {
            "page_id": "rich",
            "title": "Same",
            "summary": "Useful",
            "recall_questions": ["q1", "q2"],
            "path": str(tmp_path / "rich.md"),
        },
        "thin": {
            "page_id": "thin",
            "title": "Same",
            "summary": "",
            "recall_questions": [],
            "path": str(tmp_path / "thin.md"),
        },
    }
    (tmp_path / "thin.md").write_text("---\ntitle: Same\nupdated: 2026-07-06\n---\nThin\n", encoding="utf-8")
    monkeypatch.setattr(autonomy, "_page_meta", lambda page_id: pages.get(page_id, {}))
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md")
    monkeypatch.setattr("llm_wiki_mcp.index_store.get_store", lambda: None)
    monkeypatch.setattr(autonomy, "_page_quality", lambda page_id, meta=None: 5.0 if page_id == "rich" else 1.0)
    writes: list[dict] = []
    monkeypatch.setattr(autonomy, "_append_jsonl", lambda path, row: writes.append(row))

    payload = autonomy.resolve_duplicate_candidates(
        [
            {
                "left": "rich",
                "right": "thin",
                "left_title": "Same",
                "right_title": "Same",
                "score": 1.0,
                "method": "title",
            }
        ],
        apply=True,
        write=True,
    )

    assert payload["applied"] == 1
    text = (tmp_path / "thin.md").read_text(encoding="utf-8")
    assert "status: deprecated" in text
    assert "superseded_by: rich" in text
    assert writes[0]["action"] == "supersede"


def test_watchdog_alerts_when_sleep_never_ran(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_wiki_mcp.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.95},
            "queues": {"duplicate_candidates": 0, "lint_repair": 0},
        },
    )
    monkeypatch.setattr(autonomy, "_latest_jsonl", lambda path: {})
    writes: list[tuple[Path, dict]] = []
    monkeypatch.setattr(autonomy, "_write_json", lambda path, payload: writes.append((path, payload)))
    monkeypatch.setattr(autonomy, "_append_jsonl", lambda path, row: None)

    payload = autonomy.watchdog_snapshot(write=True)

    assert payload["status"] == "alert"
    assert payload["alerts"][0]["type"] == "sleep_never_ran"
    assert writes[0][0] == autonomy.WATCHDOG_FILE


def test_install_launchd_dry_run_builds_sleep_and_watchdog_plists(monkeypatch) -> None:
    monkeypatch.setattr(autonomy, "_uv_path", lambda: "/opt/homebrew/bin/uv")

    payload = autonomy.install_launchd(dry_run=True, load=False)

    assert payload["status"] == "ok"
    assert payload["dry_run"] is True
    labels = {item["label"] for item in payload["plists"]}
    assert autonomy.SLEEP_LABEL in labels
    assert autonomy.WATCHDOG_LABEL in labels
    programs = {item["label"]: item["program"] for item in payload["plists"]}
    assert Path(programs[autonomy.SLEEP_LABEL][0]).name == "llm-wiki-sleep"
    assert Path(programs[autonomy.WATCHDOG_LABEL][0]).name == "llm-wiki-watchdog"
    assert payload["wrappers"][0]["command"][0] == "/opt/homebrew/bin/uv"
