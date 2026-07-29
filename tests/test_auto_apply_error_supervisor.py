from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def isolated_wiki(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    chronovisor_root = tmp_path / "wiki"
    pages = chronovisor_root / "pages"
    raw = chronovisor_root / "raw"
    system = chronovisor_root / "system"
    runtime = chronovisor_root / "runtime"
    recall = chronovisor_root / "recall"
    for path in (pages, raw, system, runtime, recall):
        path.mkdir(parents=True, exist_ok=True)

    from chronovisor.core import store
    from chronovisor.ops import runtime_status

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages)
    monkeypatch.setattr(store, "RAW_DIR", raw)
    monkeypatch.setattr(store, "SYSTEM_DIR", system)
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime / "metrics.jsonl")
    return chronovisor_root


def _auto_apply_error(index: int) -> dict[str, object]:
    return {
        "ts": f"2026-06-20T22:4{index}:00",
        "apply_key": f"page_tag:key-{index}",
        "normalize_key": "gate_missed:bad-tag:page",
        "action_type": "page_tag",
        "source_ref": f"decision-{index}",
        "dry_run": False,
        "status": "error",
        "result": {
            "status": "error",
            "error": (
                "ValueError: invalid page tag 'Assistant wrote prose': "
                "missing required prefix (one of ('d/', 't/', 's/'))"
            ),
        },
    }


def test_auto_apply_errors_create_self_heal_packet_after_threshold(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.ops import auto_apply_error_supervisor as supervisor

    started: list[Path] = []
    monkeypatch.setattr(
        "chronovisor.ops.self_heal.start_background",
        lambda path: started.append(path),
    )

    result = supervisor.supervise_error_records(
        [_auto_apply_error(1), _auto_apply_error(2), _auto_apply_error(3)],
        threshold=3,
    )

    assert result["status"] == "ok"
    assert len(result["packets_created"]) == 1
    packet_path = Path(result["packets_created"][0])
    assert started == [packet_path]
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["failure_class"] == "recall.auto_apply_error"
    assert packet["status"] == "pending_local_repair"
    assert packet["local_model"] == "qwen"
    assert packet["auto_apply_error"]["count"] == 3
    assert packet["auto_apply_error"]["error_kind"] == "page_tag:invalid_page_tag.missing_required_prefix"


def test_auto_apply_errors_below_threshold_do_not_create_packet(isolated_wiki: Path) -> None:
    from chronovisor.ops import auto_apply_error_supervisor as supervisor

    result = supervisor.supervise_error_records(
        [_auto_apply_error(1), _auto_apply_error(2)],
        threshold=3,
    )

    assert result["packets_created"] == []
    assert list((isolated_wiki / "runtime" / "failures" / "packets").glob("*.json")) == []


def test_auto_apply_errors_accumulate_across_runs(isolated_wiki: Path) -> None:
    from chronovisor.ops import auto_apply_error_supervisor as supervisor

    first = supervisor.supervise_error_records([_auto_apply_error(1)], threshold=3, start_background=False)
    second = supervisor.supervise_error_records([_auto_apply_error(2)], threshold=3, start_background=False)
    third = supervisor.supervise_error_records([_auto_apply_error(3)], threshold=3, start_background=False)

    assert first["clusters"][0]["count"] == 1
    assert second["clusters"][0]["count"] == 2
    assert len(third["packets_created"]) == 1
    packet = json.loads(Path(third["packets_created"][0]).read_text(encoding="utf-8"))
    assert packet["auto_apply_error"]["count"] == 3
    assert packet["auto_apply_error"]["observed_count"] == 1


def test_auto_apply_error_supervisor_dry_run_writes_nothing(isolated_wiki: Path) -> None:
    from chronovisor.ops import auto_apply_error_supervisor as supervisor

    result = supervisor.supervise_error_records(
        [_auto_apply_error(1), _auto_apply_error(2), _auto_apply_error(3)],
        threshold=3,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["packets_created"] == []
    assert result["clusters"][0]["would_create_packet"] is True
    assert result["clusters"][0]["packet_created"] is False
    assert not (isolated_wiki / "runtime" / "failures" / "auto-apply-error-state.json").exists()
    assert list((isolated_wiki / "runtime" / "failures" / "packets").glob("*.json")) == []


def test_auto_apply_error_self_heal_cli_path_uses_existing_pipeline(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.ops import self_heal

    log_file = isolated_wiki / "recall" / "auto-apply.jsonl"
    log_file.write_text(
        "\n".join(json.dumps(_auto_apply_error(i), ensure_ascii=False) for i in range(1, 4))
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        self_heal,
        "handle_packet",
        lambda path, **kwargs: {"status": "handled", "packet": str(path), "kwargs": kwargs},
    )

    result = self_heal.run_auto_apply_error_self_heal(
        threshold=3,
        max_packets=1,
        use_qwen=True,
        enable_frontier=False,
        dry_run=False,
    )

    assert result["status"] == "ok"
    assert result["supervision"]["errors_seen"] == 3
    assert result["packets_seen"] == 1
    assert result["results"][0]["status"] == "handled"
    assert result["results"][0]["kwargs"]["use_qwen"] is True
    assert result["results"][0]["kwargs"]["enable_frontier"] is False
