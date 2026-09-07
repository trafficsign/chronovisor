"""MTPLX generation via the existing local OpenAI-compatible transport."""

from collections.abc import Mapping
from typing import Any

import httpx

from chronovisor.core.omlx_adapter import OMLXAdapter

MTPLX_BASE_URL = "http://127.0.0.1:18145/v1"


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
        if isinstance(value, Mapping) and value.get("type") == "object":
            # Transport-only constraint: changing the schema embedded in the
            # system prompt changed Gate decisions in the acceptance benchmark.
            value = {**value, "additionalProperties": False}
        return OMLXAdapter._response_format(value)
