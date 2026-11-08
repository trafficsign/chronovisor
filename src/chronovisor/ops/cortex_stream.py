"""WebSocket framing for the Synaptic Cortex dashboard view."""

import base64
import hashlib
import json
from typing import Any

_WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def websocket_accept(key: str) -> str:
    """Return the RFC 6455 accept token for a validated browser key."""

    try:
        raw = base64.b64decode(key, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid Sec-WebSocket-Key") from exc
    if len(raw) != 16:
        raise ValueError("invalid Sec-WebSocket-Key")
    digest = hashlib.sha1(f"{key}{_WEBSOCKET_GUID}".encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def websocket_text_frame(payload: dict[str, Any]) -> bytes:
    """Encode one unmasked server-to-browser JSON text frame."""

    body = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    size = len(body)
    if size < 126:
        header = bytes((0x81, size))
    elif size <= 0xFFFF:
        header = bytes((0x81, 126)) + size.to_bytes(2, "big")
    else:
        header = bytes((0x81, 127)) + size.to_bytes(8, "big")
    return header + body
