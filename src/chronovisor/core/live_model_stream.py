"""Ephemeral model deltas for the live dashboard.

Model output is intentionally kept out of durable runtime status.  Ingest fans
out small Unix datagrams to the local and LAN dashboard processes; each
dashboard keeps only its own bounded in-memory view.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MODEL_STREAM_MAX_CHARS = 128 * 1024
_MODEL_STREAM_DELTA_CHARS = 2 * 1024
_MODEL_STREAM_RECEIVE_BYTES = 8 * 1024
_MODEL_STREAM_EVENTS = {"start", "delta", "progress", "done", "error"}
_MODEL_STREAM_CHANNELS = {"thinking", "output"}


def model_stream_socket_path(root: Path, *, lan: bool) -> Path:
    suffix = "lan" if lan else "local"
    namespace = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path("/tmp") / f"chronovisor-{os.getuid()}-{namespace}-{suffix}.sock"


def _normalized_event(event: Mapping[str, Any]) -> dict[str, Any] | None:
    event_name = str(event.get("event") or "")
    if event_name not in _MODEL_STREAM_EVENTS:
        return None
    payload: dict[str, Any] = {"event": event_name, "sent_at": time.time()}
    for key in ("job_id", "phase", "target", "model"):
        value = event.get(key)
        if isinstance(value, str):
            payload[key] = value[:512]
    if event_name == "delta":
        channel = str(event.get("channel") or "")
        text = event.get("text")
        if (
            channel not in _MODEL_STREAM_CHANNELS
            or not isinstance(text, str)
            or not text
        ):
            return None
        payload["channel"] = channel
        payload["text"] = text
    for key in ("output_tokens", "tokens_per_second", "generation_seconds"):
        value = event.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            payload[key] = value
    return payload


def publish_model_stream(root: Path, event: Mapping[str, Any]) -> None:
    """Best-effort fan-out; a missing or backlogged dashboard never blocks inference."""

    payload = _normalized_event(event)
    if payload is None:
        return
    text = payload.pop("text", None)
    parts = (
        [
            text[index : index + _MODEL_STREAM_DELTA_CHARS]
            for index in range(0, len(text), _MODEL_STREAM_DELTA_CHARS)
        ]
        if isinstance(text, str)
        else [None]
    )
    datagrams = [
        json.dumps(
            {**payload, **({"text": part} if part is not None else {})},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        for part in parts
    ]
    for lan in (False, True):
        path = model_stream_socket_path(root, lan=lan)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
                sender.setblocking(False)
                for datagram in datagrams:
                    sender.sendto(datagram, str(path))
        except (BlockingIOError, FileNotFoundError, ConnectionRefusedError, OSError):
            continue


class LiveModelStreamBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._revision = 0
        self._state = self._empty_state()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "active": False,
            "status": "idle",
            "job_id": None,
            "phase": None,
            "target": None,
            "model": None,
            "thinking": "",
            "output": "",
            "last_channel": None,
            "output_tokens": 0,
            "tokens_per_second": 0.0,
            "generation_seconds": 0.0,
            "updated_at": None,
            "truncated": False,
        }

    def apply(self, event: Mapping[str, Any]) -> None:
        event_name = str(event.get("event") or "")
        if event_name not in _MODEL_STREAM_EVENTS:
            return
        job_id = event.get("job_id") if isinstance(event.get("job_id"), str) else None
        with self._lock:
            if event_name == "start" or (
                job_id is not None and job_id != self._state.get("job_id")
            ):
                self._state = self._empty_state()
            for key in ("job_id", "phase", "target", "model"):
                value = event.get(key)
                if isinstance(value, str):
                    self._state[key] = value
            if event_name == "delta":
                channel = str(event.get("channel") or "")
                text = event.get("text")
                if channel in _MODEL_STREAM_CHANNELS and isinstance(text, str):
                    combined = str(self._state[channel]) + text
                    if len(combined) > MODEL_STREAM_MAX_CHARS:
                        combined = combined[-MODEL_STREAM_MAX_CHARS:]
                        self._state["truncated"] = True
                    self._state[channel] = combined
                    self._state["last_channel"] = channel
            for key in ("output_tokens", "tokens_per_second", "generation_seconds"):
                value = event.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._state[key] = value
            self._state["active"] = event_name not in {"done", "error"}
            self._state["status"] = (
                "error"
                if event_name == "error"
                else "complete"
                if event_name == "done"
                else "streaming"
            )
            sent_at = event.get("sent_at")
            self._state["updated_at"] = (
                float(sent_at)
                if isinstance(sent_at, (int, float)) and not isinstance(sent_at, bool)
                else time.time()
            )
            self._revision += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"revision": self._revision, **self._state}


def empty_model_stream_snapshot() -> dict[str, Any]:
    return {"revision": 0, **LiveModelStreamBuffer._empty_state()}


class LiveModelStreamReceiver:
    def __init__(self, root: Path, *, lan: bool) -> None:
        self.path = model_stream_socket_path(root, lan=lan)
        self._buffer = LiveModelStreamBuffer()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.path.is_symlink():
            mode = self.path.lstat().st_mode
            if self.path.is_symlink() or not stat.S_ISSOCK(mode):
                raise RuntimeError(f"model stream path is not a socket: {self.path}")
            self.path.unlink()
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        receiver.bind(str(self.path))
        os.chmod(self.path, 0o600)
        receiver.settimeout(0.2)
        self._socket = receiver
        self._thread = threading.Thread(
            target=self._run,
            name=f"chronovisor-model-stream-{self.path.stem}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        receiver = self._socket
        if receiver is None:
            return
        while not self._stop.is_set():
            try:
                encoded = receiver.recv(_MODEL_STREAM_RECEIVE_BYTES)
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                event = json.loads(encoded)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(event, dict):
                self._buffer.apply(event)

    def snapshot(self) -> dict[str, Any]:
        return self._buffer.snapshot()

    def close(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        try:
            mode = self.path.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(mode):
            self.path.unlink()
