"""Isolated structured-model worker terminated by the sync-first parent."""

from __future__ import annotations

import json
import sys

from llm_wiki_mcp.local_structured import LocalStructuredSession


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        result = LocalStructuredSession(
            model=str(request["model"]),
            role=str(request.get("role") or "research_planner"),
            num_ctx=int(request["num_ctx"]),
            num_predict=int(request["num_predict"]),
            keep_alive=str(request.get("keep_alive") or "2m"),
            read_timeout_ms=int(request["read_timeout_ms"]),
            max_input_chars=int(request["max_input_chars"]),
            max_output_chars=int(request["max_output_chars"]),
            max_feedback_chars=int(request["max_feedback_chars"]),
        ).run(
            str(request["prompt"]),
            dict(request["schema"]),
            system=str(request.get("system") or ""),
            format_schema=(
                dict(request["format_schema"])
                if isinstance(request.get("format_schema"), dict)
                else None
            ),
        )
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "value": result.value,
                    "first_pass_valid": result.first_pass_valid,
                    "repair_turns": result.repair_turns,
                    "failure_class": result.failure_class,
                    "failure_reason": result.failure_reason,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        print(f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
