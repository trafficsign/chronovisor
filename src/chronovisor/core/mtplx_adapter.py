"""MTPLX generation via the existing local OpenAI-compatible transport."""

from collections.abc import Mapping
from typing import Any

import httpx

from chronovisor.core.omlx_adapter import OMLXAdapter

MTPLX_BASE_URL = "http://127.0.0.1:18145/v1"


def _transport_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Leave uniqueness to the original schema's response validator.

    MTPLX's llguidance rejects uniqueItems before inference. Only the wire
    grammar omits it; prompts and the caller's validation schema stay intact.
    Traverse schema nodes, never enum/const values or property names.
    """
    result = {key: value for key, value in schema.items() if key != "uniqueItems"}
    properties = result.get("properties")
    if isinstance(properties, Mapping):
        result["properties"] = {
            key: _transport_schema(value) if isinstance(value, Mapping) else value
            for key, value in properties.items()
        }
    for key in ("items", "additionalProperties"):
        value = result.get(key)
        if isinstance(value, Mapping):
            result[key] = _transport_schema(value)
    return result


class MTPLXAdapter(OMLXAdapter):
    """Keep prompts intact; constrain only the structured response envelope."""

    provider = "mtplx"

    def __init__(
        self,
        *,
        base_url: str = MTPLX_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(base_url=base_url, transport=transport)

    @staticmethod
    def _uses_dflash(model: str) -> bool:
        return False

    @staticmethod
    def _response_format(value: Mapping[str, Any] | str) -> dict[str, object]:
        if isinstance(value, Mapping):
            value = _transport_schema(value)
            if value.get("type") == "object":
                # Changing the schema in the prompt changed Gate decisions in
                # the acceptance benchmark. Close only the wire grammar.
                value = {**value, "additionalProperties": False}
        return OMLXAdapter._response_format(value)
