"""Capture-only hooks/jobs and the four read-only MCP tools."""

from __future__ import annotations

import asyncio
import io
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core import background_jobs, llm_config
from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.hosts import hook_dispatcher, server

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(background_jobs, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    return background_jobs


@pytest.mark.parametrize("module", sorted(background_jobs.CAPTURE_MODULES))
def test_only_capture_jobs_execute_without_legacy_followups(jobs, module, monkeypatch):
    job = jobs.enqueue_job(
        name="codex-save", module=module, args=[], env={}, stdin_text="{}"
    )
    state = jobs._load()
    state["jobs"][job["job_id"]]["on_success"] = [
        {"module": "chronovisor.ops.recall_auditor"}
    ]
    jobs._save(state)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["env"]["CHRONOVISOR_ROOT"] == str(jobs.CHRONOVISOR_ROOT)
        assert kwargs["env"]["PYTHONPATH"] == str(SRC)
        return SimpleNamespace(returncode=0, stdout="receipt", stderr="")

    monkeypatch.setattr(jobs.subprocess, "run", run)
    assert jobs.retry_due()["results"][0]["status"] == "completed"
    assert commands == [[sys.executable, "-m", module]]
    stored = jobs._load()["jobs"]
    assert list(stored) == [job["job_id"]]
    assert "on_success" not in stored[job["job_id"]]


def test_retired_jobs_are_terminal_even_when_future_or_retry_limit_zero(
    jobs, monkeypatch
):
    state = {"schema_version": 1, "jobs": {}}
    for name in ("recall_answer_eval", "recall_auditor", "recall_shadow"):
        state["jobs"][name] = {
            "module": f"chronovisor.ops.{name}",
            "status": "retry_wait",
            "next_retry_at": "2099-01-01T00:00:00+00:00",
            "stdin": "private",
            "on_success": [{"module": "unexpected"}],
        }
    jobs._save(state)
    monkeypatch.setattr(
        jobs.subprocess, "run", lambda *a, **k: pytest.fail("retired job executed")
    )
    assert jobs.retry_due(limit=0)["results"] == []
    for job in jobs._load()["jobs"].values():
        assert job["status"] in jobs.TERMINAL_STATUSES
        assert job["reason"] == "retired_module"
        assert not job["on_success"] and not job["stdin"]
        assert job["next_retry_at"] is None
    direct = jobs.enqueue_job(
        name="retired",
        module="chronovisor.ops.recall_shadow",
        args=[],
        env={},
        stdin_text="",
    )
    assert jobs.run_job(direct["job_id"])["reason"] == "retired_module"


def test_abandoned_capture_requeues_and_snapshot_is_read_only(jobs, monkeypatch):
    job = jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=[],
        env={},
        stdin_text="{}",
    )
    jobs._claim(job["job_id"])
    monkeypatch.setattr(jobs, "_pid_alive", lambda pid: False)
    before = jobs.STATE_FILE.read_bytes(), jobs.STATE_FILE.stat().st_mtime_ns
    assert jobs.snapshot()["by_status"] == {"running": 1}
    assert (jobs.STATE_FILE.read_bytes(), jobs.STATE_FILE.stat().st_mtime_ns) == before
    assert jobs.repair_stale()["requeued"] == 1
    assert jobs._load()["jobs"][job["job_id"]]["stdin"] == "{}"


@pytest.mark.parametrize("host,output", [("codex", "{}\n"), ("claude-code", "")])
def test_prompt_hook_is_noop_without_reading_input_or_config(
    host, output, monkeypatch, capsys
):
    class Unreadable(io.StringIO):
        def read(self, *args, **kwargs):
            pytest.fail("prompt input read")

    monkeypatch.setattr(sys, "stdin", Unreadable())
    monkeypatch.setattr(
        hook_dispatcher, "load_hook_policy", lambda *a: pytest.fail("config read")
    )
    assert (
        hook_dispatcher.main(["--host", host, "--event", "UserPromptSubmit", "--hook"])
        == 0
    )
    assert capsys.readouterr().out == output


@pytest.mark.parametrize("host", ["codex", "claude-code"])
def test_stop_hook_enqueues_one_save_only(host, monkeypatch, capsys):
    monkeypatch.delenv("CHRONOVISOR_INTERNAL_FRONTIER", raising=False)
    monkeypatch.setattr(hook_dispatcher, "save_enabled", lambda *a: True)
    assert (
        hook_dispatcher.main(
            ["--host", host, "--event", "Stop", "--dry-run", "--format", "json"]
        )
        == 0
    )
    tasks = json.loads(capsys.readouterr().out)["tasks"]
    assert len(tasks) == 1 and tasks[0]["name"] == f"{host}-save"
    assert tasks[0]["module"] in background_jobs.CAPTURE_MODULES
    assert "on_success" not in tasks[0]


def test_mcp_exposes_exactly_five_tools_and_retained_parameters():
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    assert set(tools) == {
        "chronovisor_search",
        "chronovisor_read",
        "chronovisor_init",
        "chronovisor_status",
        "chronovisor_record",
    }
    assert set(tools["chronovisor_search"].inputSchema["properties"]) == {
        "query",
        "depth",
        "folder",
        "updated_after",
        "updated_before",
        "sort_by",
        "semantic",
        "tags",
        "tag_match",
    }
    instructions = server.mcp.instructions.lower()
    assert "recall_used" not in instructions and "ollama" not in instructions
    assert "chronovisor_record" in instructions
    assert set(tools["chronovisor_record"].inputSchema["properties"]) == {
        "content",
        "session_id",
        "keywords",
        "idempotency_key",
    }


@pytest.mark.parametrize("layout", ["legacy", "v2"])
def test_record_publishes_raw_with_only_valid_keywords(tmp_path, monkeypatch, layout):
    from chronovisor.raw import record_raw

    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", layout)
    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(record_raw, "RAW_DIR", tmp_path / "raw")
    content = "---\ntitle: Remember\n---\n覚えて: explicit memory\n"
    rejected = ["", "  ", *[f"bad{char}word" for char in ",[]:#{}\n\r\x00\t"]]
    result = json.loads(
        server.chronovisor_record(
            content,
            session_id="memory-writer",
            keywords=["local-memory", "日本語", *rejected],
        )
    )
    path = tmp_path / result["path"]
    assert (
        path.read_bytes()
        == (
            "---\ntitle: Remember\nraw_keywords: [local-memory, 日本語]\n"
            "---\n覚えて: explicit memory\n"
        ).encode()
    )
    assert path.is_relative_to(tmp_path / "raw")
    assert result == {"saved": path.name, "path": path.relative_to(tmp_path).as_posix()}
    assert list(tmp_path.iterdir()) == [tmp_path / "raw"]
    assert list((tmp_path / "raw").rglob("*.md")) == [path]


@pytest.mark.parametrize("layout", ["legacy", "v2"])
def test_record_idempotency_reuses_first_raw_without_overwriting(
    tmp_path, monkeypatch, layout
):
    from chronovisor.raw import record_raw

    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", layout)
    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(record_raw, "RAW_DIR", tmp_path / "raw")
    args = {
        "content": "first memory\n",
        "keywords": ["memory"],
        "idempotency_key": "remember-test",
    }
    first = json.loads(server.chronovisor_record(**args))
    path = tmp_path / first["path"]
    original = path.read_bytes(), path.stat().st_mtime_ns
    repeated = json.loads(server.chronovisor_record(**args))
    assert first == {
        "saved": "save-remember-test.md",
        "path": "raw/save-remember-test.md",
        "deduplicated": False,
    }
    assert repeated == {**first, "deduplicated": True}
    with pytest.raises(RuntimeError, match="idempotency key collision"):
        server.chronovisor_record(**{**args, "content": "different memory\n"})
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    assert list((tmp_path / "raw").iterdir()) == [path]


def test_record_metadata_omits_paths_outside_root(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    monkeypatch.setattr(
        server,
        "record_raw",
        lambda *a, **k: {
            "saved": "memory.md",
            "path": str(tmp_path / "outside.md"),
            "content": "private body",
            "accepted_keywords": ["private"],
            "rejected_keywords": ["private:invalid"],
            "raw_slug": "private",
        },
    )
    assert json.loads(server.chronovisor_record("private body")) == {
        "saved": "memory.md"
    }


def test_init_reads_only_two_pages_and_capture_count(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_store",
        lambda: SimpleNamespace(refresh=lambda: None, page_count=lambda **k: 7),
    )
    monkeypatch.setattr(server, "snapshot", lambda: {"by_status": {"queued": 2}})
    seen = []

    def read(page):
        seen.append(page)
        return json.dumps({"page_id": page, "content": page})

    monkeypatch.setattr(server, "chronovisor_read", read)
    result = json.loads(server.chronovisor_init())
    assert seen == ["user-profile", "lessons-learned"]
    assert result["status"] == {"page_count": 7, "queued_capture_jobs": 2}


def test_obsolete_config_sections_and_roles_are_ignored(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("""[llm.providers.retired]
kind = "not-a-provider"
[llm.roles."recall.filter"]
provider = "missing"
capability = "invalid"
[llm.roles."search.rerank"]
provider = "startlux"
capability = "rerank"
model = "decision"
[llm.providers.startlux]
kind = "local-decision-mlx"
path = "/unused/model"
[llm.roles."search.semantic.foreground"]
provider = "embedding"
capability = "embedding"
model = "nemotron"
[llm.providers.embedding]
kind = "nemotron"
device = "cpu"
[recall]
enabled = true
[decision]
invalid = true
[classification]
provider = "retired"
""")
    monkeypatch.setattr(
        llm_config,
        "load_search_embedding_config",
        lambda path: SearchEmbeddingConfig(query_device="cpu"),
    )
    runtime = llm_config.load_llm_runtime(path)
    assert runtime.resolve_rerank("search.rerank").provider == "local-decision"
    assert set(runtime._embedding) == {"search.semantic.foreground"}
    path.write_text(
        path.read_text().replace('kind = "local-decision-mlx"', 'kind = "remote"')
    )
    with pytest.raises(llm_config.LLMConfigError, match="local StartLux"):
        llm_config.load_llm_runtime(path)


def test_sweeper_recovers_both_hosts_and_appends_byte_exact_tails(tmp_path):
    root = tmp_path / "wiki"
    root.mkdir()
    codex_root = tmp_path / "codex" / "sessions"
    claude_root = tmp_path / ".claude" / "projects"
    codex_root.mkdir(parents=True)
    claude_root.mkdir(parents=True)
    sessions = {
        "codex": codex_root / "user.jsonl",
        "claude-code": claude_root / "user.jsonl",
    }
    rows = {
        "codex": [
            {
                "type": "session_meta",
                "payload": {
                    "id": "sweeper-codex",
                    "source": "vscode",
                    "thread_source": "user",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "日本語と\u2028行"}],
                },
            },
        ],
        "claude-code": [
            {
                "type": "user",
                "sessionId": "sweeper-claude",
                "message": {"content": "日本語と\u2028行"},
            },
        ],
    }
    for host, session in sessions.items():
        session.write_bytes(
            b"".join(
                (json.dumps(row, ensure_ascii=False) + "\n").encode()
                for row in rows[host]
            )
        )
    env = {
        **os.environ,
        "CHRONOVISOR_ROOT": str(root),
        "CHRONOVISOR_RAW_LAYOUT": "v2",
        "CODEX_HOME": str(codex_root.parent),
        "HOME": str(tmp_path),
    }
    env.pop("CHRONOVISOR_READ_ONLY", None)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")

    def sweep():
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "chronovisor.ops.session_sweeper",
                "--limit",
                "8",
                "--idle-seconds",
                "0",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert child.returncode == 0, child.stderr
        return json.loads(child.stdout)

    first = sweep()
    assert first["processed"] == 2, first
    assert {item["host"] for item in first["results"]} == set(sessions)
    segments = {}
    for item in first["results"]:
        assert item["status"] == "saved", item
        segments[item["host"]] = Path(item["save_result"]["path"])
        assert (
            segments[item["host"]].read_bytes() == sessions[item["host"]].read_bytes()
        )
    for host, session in sessions.items():
        with session.open("ab") as stream:
            stream.write(
                (json.dumps(rows[host][-1], ensure_ascii=False) + "\n").encode()
            )
    second = sweep()
    assert second["processed"] == 2
    for host, session in sessions.items():
        assert segments[host].read_bytes() == session.read_bytes()
        state = json.loads((root / f"{host}-save-state.json").read_text())
        assert state["files"][str(session)]["last_saved_line"] == len(rows[host]) + 1
    assert sweep()["processed"] == 0
    assert not (root / "pages").exists() and not (root / "system").exists()


def test_capture_wrapper_runs_one_entrypoint(tmp_path):
    scripts = SRC.parent / "scripts"
    wrapper = tmp_path / "chronovisor-capture"
    wrapper.write_bytes((scripts / wrapper.name).read_bytes())
    runtime_env = tmp_path / "chronovisor-runtime-env"
    runtime_env.write_text(
        'chronovisor_exec_uvx() { printf "%s\\n" "$*"; }\n'
        "CHRONOVISOR_PYTHON=python3.14\n"
        "RUNTIME_SOURCE=test-source\n"
        "CHRONOVISOR_ROOT=$CAPTURE_TEST_ROOT\n"
    )
    completed = subprocess.run(
        ["/bin/sh", str(wrapper)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "CAPTURE_TEST_ROOT": str(tmp_path / "root")},
    )
    assert completed.stdout.splitlines() == [
        "--python python3.14 --from test-source chronovisor-capture"
    ]
    template = SRC.parent / "launchd" / "com.trafficsign.chronovisor-capture.plist"
    assert plistlib.loads(template.read_bytes())["StartInterval"] == 120


def test_capture_init_does_not_chmod_a_raw_symlink(tmp_path):
    from chronovisor.core.store import RuntimeContext, init_chronovisor

    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "raw").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        init_chronovisor(RuntimeContext(root))
    assert outside.stat().st_mode & 0o777 == 0o755


def test_read_links_and_search_expansion_keep_read_only_indexes(tmp_path, monkeypatch):
    from chronovisor.core.index_store import IndexStore

    pages = tmp_path / "pages"
    pages.mkdir()
    for page, body in {"a": "[b](b.md)", "b": "[c](c.md)", "c": "leaf"}.items():
        (pages / f"{page}.md").write_text(
            f"---\ntitle: {page}\nstatus: stable\ntype: knowledge\nupdated: 2026-11-08\ntags: [topic]\n---\n{body}\n"
        )
    index = IndexStore(tmp_path)
    index.refresh()
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    monkeypatch.setenv("CHRONOVISOR_READ_ONLY", "1")
    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(server, "get_store", lambda: index)
    read = json.loads(server.chronovisor_read("b", max_chars=5))
    assert read["backlinks"] == ["a"] and read["outlinks"] == ["c"]
    assert read["next_offset"] == 5
    direct = [{"page_id": "a", "score": 1}]
    assert server._expanded_search_hits(
        direct, depth=0, store=index, tag_filter=[], tag_match="all"
    ) == ([], [])
    expanded, edges = server._expanded_search_hits(
        direct, depth=1, store=index, tag_filter=["topic"], tag_match="all"
    )
    assert [(p["page_id"], p["distance"], p["score"]) for p in expanded] == [
        ("b", 1, 0.5)
    ]
    assert edges[0]["from"] == "a" and edges[0]["to"] == "b"
    assert {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    } == before


@pytest.mark.parametrize("method", ["judge", "recall_filter"])
def test_reranker_rejects_retired_methods_without_loading_models(method):
    from chronovisor.search.reranker_service import RerankerServiceState

    state = object.__new__(RerankerServiceState)
    with pytest.raises(ValueError, match="unknown reranker method"):
        state.handle({"method": method})


def test_existing_page_registry_redirects_remain_read_only_and_contained(
    tmp_path, monkeypatch
):
    from chronovisor.core.page_registry import PageRegistry

    uid = "019f0000-0000-7000-8000-000000000001"
    retired_uid = "019f0000-0000-7000-8000-000000000002"
    registry = PageRegistry(tmp_path)
    registry.path.parent.mkdir(parents=True)
    state = registry.empty()
    state["pages"] = {
        uid: {
            "uid": uid,
            "page_id": "current",
            "path": "pages/current.md",
            "status": "stable",
        },
        retired_uid: {
            "uid": retired_uid,
            "page_id": "old",
            "path": "pages/old.md",
            "status": "deprecated",
            "canonical_uid": uid,
        },
    }
    state["keys"] = {"old": retired_uid, "current": uid}
    state["redirects"] = {retired_uid: {"to_uid": uid, "anchor_map": {}}}
    registry.path.write_text(json.dumps(state))
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "current.md").write_text(
        "---\ntitle: Current\nstatus: stable\ntype: knowledge\n---\nbody\n"
    )
    before = registry.path.read_bytes(), registry.path.stat().st_mtime_ns
    assert registry.resolve("old")["uid"] == uid
    assert registry.path_for("old") == pages / "current.md"
    assert (registry.path.read_bytes(), registry.path.stat().st_mtime_ns) == before
    state["pages"][uid]["path"] = "pages/../../outside.md"
    registry.path.write_text(json.dumps(state))
    assert registry.path_for("old") is None
