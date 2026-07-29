"""Long-running local model probes used only by the resource overlap burn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chronovisor.core import ollama


def _mark_ready(path: Path, *, kind: str, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"kind": kind, "model": model}, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("llm", "embed"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    _mark_ready(args.ready_file, kind=args.kind, model=args.model)
    if args.kind == "embed":
        text = (
            "Chronovisor library evidence multilingual dense classification "
            "resource overlap cancellation probe. "
        ) * 512
        vectors = ollama.embed(
            [f"{index}: {text}" for index in range(128)],
            model=args.model,
            read_timeout_ms=600_000,
        )
        print(json.dumps({"completed": True, "vectors": len(vectors)}))
        return
    value = ollama.generate(
        (
            "Generate a long numbered technical analysis of knowledge "
            "classification. Continue until the token limit."
        ),
        model=args.model,
        num_ctx=8_192,
        num_predict=8_192,
        keep_alive="24h",
        read_timeout_ms=600_000,
        progress_callback=lambda _event: None,
        temperature=0,
        seed=0,
    )
    print(json.dumps({"completed": True, "chars": len(str(value))}))


if __name__ == "__main__":
    main()
