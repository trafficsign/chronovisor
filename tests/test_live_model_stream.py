from __future__ import annotations

import stat
import time
from pathlib import Path

from chronovisor.core.live_model_stream import (
    LiveModelStreamReceiver,
    model_stream_socket_path,
    publish_model_stream,
)
from chronovisor.ingest import ingest_transport


def _wait_for_output(receiver: LiveModelStreamReceiver, expected: str) -> dict:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = receiver.snapshot()
        if snapshot["output"] == expected:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"live stream did not reach {expected!r}")


def test_live_model_stream_fans_out_without_persisting_content(tmp_path: Path) -> None:
    local = LiveModelStreamReceiver(tmp_path, lan=False)
    lan = LiveModelStreamReceiver(tmp_path, lan=True)
    local.start()
    lan.start()
    try:
        publish_model_stream(
            tmp_path,
            {"event": "start", "job_id": "job-1", "phase": "generate"},
        )
        publish_model_stream(
            tmp_path,
            {
                "event": "delta",
                "job_id": "job-1",
                "channel": "thinking",
                "text": "plan ",
            },
        )
        publish_model_stream(
            tmp_path,
            {
                "event": "delta",
                "job_id": "job-1",
                "channel": "output",
                "text": "answer",
            },
        )
        publish_model_stream(
            tmp_path,
            {
                "event": "progress",
                "job_id": "job-1",
                "output_tokens": 7,
                "tokens_per_second": 14.0,
                "generation_seconds": 0.5,
            },
        )

        local_snapshot = _wait_for_output(local, "answer")
        lan_snapshot = _wait_for_output(lan, "answer")
        assert local_snapshot["thinking"] == "plan "
        assert lan_snapshot["thinking"] == "plan "
        assert local_snapshot["output_tokens"] == 7
        assert local_snapshot["active"] is True
        for path in (
            model_stream_socket_path(tmp_path, lan=False),
            model_stream_socket_path(tmp_path, lan=True),
        ):
            assert stat.S_ISSOCK(path.lstat().st_mode)
        assert not any(path.is_file() for path in tmp_path.rglob("*"))
    finally:
        local.close()
        lan.close()

    assert not model_stream_socket_path(tmp_path, lan=False).exists()
    assert not model_stream_socket_path(tmp_path, lan=True).exists()


def test_ingest_progress_publishes_live_delta_but_persists_only_counters(
    tmp_path: Path, monkeypatch
) -> None:
    writes: list[dict] = []
    published: list[dict] = []

    class RuntimeStatus:
        def now_iso(self) -> str:
            return "2026-08-30T16:00:00"

        def safe_write_status(self, **fields) -> None:
            writes.append(fields)

    monkeypatch.setattr(ingest_transport, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(
        ingest_transport,
        "publish_model_stream",
        lambda _root, event: published.append(dict(event)),
    )
    callback = ingest_transport.llm_progress_callback(
        RuntimeStatus(),
        phase="generate",
        target="page.md",
        job_id="job-1",
        source_raw="raw.md",
    )

    callback.stream_delta(channel="thinking", text="private plan")  # type: ignore[attr-defined]
    callback.stream_delta(channel="output", text="visible output")  # type: ignore[attr-defined]
    callback(
        {
            "output_tokens": 8,
            "tokens_per_second": 20.0,
            "generation_seconds": 0.4,
        }
    )

    assert published[0]["event"] == "start"
    assert [
        event.get("channel") for event in published if event["event"] == "delta"
    ] == [
        "thinking",
        "output",
    ]
    assert all("private plan" not in str(write) for write in writes)
    assert all("visible output" not in str(write) for write in writes)
    assert writes[-1]["llm"]["output_tokens"] == 8
