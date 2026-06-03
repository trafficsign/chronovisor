from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp import cli, runtime_config, runtime_status, wiki
from llm_wiki_mcp import recall_runtime


def patch_wiki(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "wiki"
    raw = root / "raw"
    pages = root / "pages"
    system = root / "system"
    recall = root / "recall"
    runtime = root / "runtime"
    for path in (raw, pages, system, recall, runtime):
        path.mkdir(parents=True, exist_ok=True)
    (raw / "r.md").write_text("raw", encoding="utf-8")
    (pages / "p.md").write_text("---\ntitle: P\n---\n", encoding="utf-8")
    (system / "user-profile.md").write_text("# User\n", encoding="utf-8")
    (recall / "recall-log.jsonl").write_text(
        json.dumps({"decision": "read"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (recall / "feedback.jsonl").write_text(
        json.dumps({"kind": "missed_candidate"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config = root / "config.toml"
    config.write_text("[hooks.stop]\nsave = true\naudit = true\n", encoding="utf-8")

    monkeypatch.setattr(wiki, "WIKI_ROOT", root)
    monkeypatch.setattr(wiki, "RAW_DIR", raw)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages)
    monkeypatch.setattr(wiki, "SYSTEM_DIR", system)
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)
    monkeypatch.setattr(runtime_config, "LEGACY_RECALL_CONFIG_FILE", root / "recall.toml")
    monkeypatch.setattr(recall_runtime, "RECALL_DIR", recall)
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", recall / "recall-log.jsonl")
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", recall / "feedback.jsonl")
    monkeypatch.setattr(cli, "RECALL_DIR", recall)
    monkeypatch.setattr(cli, "RECALL_LOG_FILE", recall / "recall-log.jsonl")
    monkeypatch.setattr(cli, "RECALL_FEEDBACK_FILE", recall / "feedback.jsonl")
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime / "metrics.jsonl")


def test_status_json_reports_wiki_and_recall_counts(tmp_path, monkeypatch, capsys) -> None:
    patch_wiki(tmp_path, monkeypatch)

    assert cli.main(["status", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["wiki"]["raw_files"] == 1
    assert output["wiki"]["pages"] == 1
    assert output["config"]["mode"] == "unified"
    assert output["recall"]["decisions"] == {"read": 1}
    assert output["recall"]["feedback"] == {"missed_candidate": 1}


def test_hooks_inspect_json_handles_missing_host_files(tmp_path, monkeypatch, capsys) -> None:
    patch_wiki(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "CODEX_HOOKS_FILE", tmp_path / "missing-hooks.json")
    monkeypatch.setattr(cli, "CLAUDE_SETTINGS_FILE", tmp_path / "missing-settings.json")

    assert cli.main(["hooks", "inspect", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["codex"]["entries"] == []
    assert output["claude_code"]["entries"] == []
    assert output["hook_policy"]["stop_audit"] is True
