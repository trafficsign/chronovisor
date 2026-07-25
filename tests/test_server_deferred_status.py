from __future__ import annotations

import json
from pathlib import Path

import pytest


class _Store:
    def refresh(self) -> None:
        return None

    def page_count(self, *, include_system: bool) -> int:
        assert include_system is False
        return 2

    def all_pages_meta(self, *, include_system: bool) -> list[dict[str, str]]:
        assert include_system is False
        return [
            {"page_id": "new", "page_type": "knowledge"},
            {"page_id": "old", "page_type": "event"},
        ]

    def orphans(self, *, include_system: bool) -> list[str]:
        assert include_system is False
        return ["old"]

    def outlinks(self, _page_id: str) -> list[str]:
        return []

    def backlinks(self, _page_id: str) -> list[str]:
        return []


def _seed_raws(root: Path) -> Path:
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    for name in ("pending-a.md", "pending-b.md", "semantic.md", "operational.md"):
        (raw_dir / name).write_text(name, encoding="utf-8")
    return raw_dir


def _patch_deferred_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> list[list[str]]:
    from chronovisor import failure_supervisor

    scans: list[list[str]] = []

    def deferred(raw_paths: list[Path]) -> dict[str, str]:
        scans.append([path.name for path in raw_paths])
        return {
            "semantic.md": "semantic_no_quorum",
            "operational.md": "pending_local_repair",
        }

    monkeypatch.setattr(
        failure_supervisor,
        "operational_deferred_raw_files",
        deferred,
    )
    return scans


def test_chronovisor_status_separates_runnable_and_deferred_raws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor import health, ollama, orchestrator, server

    raw_dir = _seed_raws(tmp_path)
    scans = _patch_deferred_statuses(monkeypatch)
    monkeypatch.setattr(server, "RAW_DIR", raw_dir)
    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(server, "get_store", lambda: _Store())
    monkeypatch.setattr(
        orchestrator,
        "get_pending_raw_files",
        lambda: [raw_dir / "pending-a.md", raw_dir / "pending-b.md"],
    )
    monkeypatch.setattr(health, "health_snapshot", lambda: {"status": "ok"})

    class Response:
        status_code = 200

    class Client:
        def get(self, *_args: object, **_kwargs: object) -> Response:
            return Response()

    monkeypatch.setattr(ollama, "_client", lambda: Client())

    payload = json.loads(server.chronovisor_status())

    assert scans == [["operational.md", "pending-a.md", "pending-b.md", "semantic.md"]]
    assert payload["raw_total"] == 4
    assert payload["raw_pending"] == 2
    assert payload["semantic_deferred"] == 1
    assert payload["operational_deferred"] == 1
    assert payload["raw_outstanding"] == 4
    assert payload["page_count"] == 2
    assert payload["ollama_status"] == "running"


def test_chronovisor_init_reports_deferred_counts_in_parallel_bootstrap_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor import ollama, orchestrator, server

    raw_dir = _seed_raws(tmp_path)
    system_dir = tmp_path / "system"
    system_dir.mkdir()
    for page_id in ("user-profile", "current-state", "lessons-learned"):
        (system_dir / f"{page_id}.md").write_text(
            f"---\ntitle: {page_id}\n---\n{page_id}",
            encoding="utf-8",
        )
    scans = _patch_deferred_statuses(monkeypatch)
    monkeypatch.setattr(server, "RAW_DIR", raw_dir)
    monkeypatch.setattr(server, "SYSTEM_DIR", system_dir)
    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(server, "get_store", lambda: _Store())
    monkeypatch.setattr(
        orchestrator,
        "get_pending_raw_files",
        lambda: [raw_dir / "pending-a.md", raw_dir / "pending-b.md"],
    )
    monkeypatch.setattr(ollama, "is_available", lambda: False)

    payload = json.loads(server.chronovisor_init())

    assert scans == [["operational.md", "pending-a.md", "pending-b.md", "semantic.md"]]
    assert payload["status"] | {"librarian": None} == {
        "page_count": 2,
        "raw_total": 4,
        "raw_pending": 2,
        "semantic_deferred": 1,
        "operational_deferred": 1,
        "raw_outstanding": 4,
        "ollama_status": "stopped",
        "chronovisor_root": str(tmp_path),
        "librarian": None,
    }
    assert payload["status"]["librarian"]["state"] == "NOT_READY"
    assert (
        payload["status"]["librarian"]["authority"]["reason"]
        == "locked_calibration_not_adopted"
    )
    assert set(payload["system_pages"]) == {
        "user-profile",
        "current-state",
        "lessons-learned",
    }
