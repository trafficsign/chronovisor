"""Drain pending raw LLM Wiki entries through the ingest orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp import orchestrator, runtime_status
from llm_wiki_mcp.wiki import WIKI_ROOT, init_wiki

DEFAULT_MAX_BATCHES = 24
DEFAULT_MAX_UNITS = orchestrator.MAX_INGEST_BATCH_UNITS
DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_IDLE_SLEEP_SECONDS = 60.0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _default_log_file() -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    return WIKI_ROOT / "logs" / f"ingest-drain-{stamp}.jsonl"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def drain(
    *,
    max_batches: int = DEFAULT_MAX_BATCHES,
    max_units: int = DEFAULT_MAX_UNITS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Run forced ingest batches until the queue drains or progress stops."""
    if max_batches < 0:
        raise ValueError("max_batches must be >= 0")
    if not isinstance(max_units, int) or isinstance(max_units, bool):
        raise ValueError("max_units must be an integer between 1 and 10")
    if not 1 <= max_units <= orchestrator.MAX_INGEST_BATCH_UNITS:
        raise ValueError(
            f"max_units must be between 1 and {orchestrator.MAX_INGEST_BATCH_UNITS}"
        )
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be >= 0")

    init_wiki()
    orchestrator.reset_stale_lock()
    runtime_status.reset_stale_runtime_status()
    try:
        from llm_wiki_mcp.managed_hold import sync_ingest_semantic_holds

        managed_holds = sync_ingest_semantic_holds(wiki_root=WIKI_ROOT)
    except Exception as exc:
        managed_holds = {"status": "error", "error": str(exc)}

    started = time.time()
    log_path = log_file or _default_log_file()
    pending_start = len(orchestrator.get_pending_raw_files())
    batches: list[dict[str, Any]] = []
    total_processed = 0
    stop_reason = "max_batches reached"

    if max_batches == 0:
        return {
            "status": "checked",
            "pending_start": pending_start,
            "pending_after": pending_start,
            "batches_run": 0,
            "files_processed": 0,
            "elapsed_seconds": 0.0,
            "managed_holds": managed_holds,
        }

    for batch_index in range(1, max_batches + 1):
        pending_before = len(orchestrator.get_pending_raw_files())
        if pending_before == 0:
            stop_reason = "no pending raws"
            break

        result = orchestrator.run_pending_ingest(
            force=True,
            max_units=max_units,
        )
        processed = len(result.get("files_processed", []))
        total_processed += processed
        try:
            from llm_wiki_mcp.self_heal import run_pending as run_pending_self_heal

            self_heal_result = run_pending_self_heal(
                max_packets=1,
                enable_frontier=False,
            )
        except Exception as exc:
            self_heal_result = {"status": "error", "error": str(exc)}
        pending_after = len(orchestrator.get_pending_raw_files())
        record = {
            "timestamp": datetime.now().isoformat(),
            "batch": batch_index,
            "pending_before": pending_before,
            "pending_after": pending_after,
            "files_processed": processed,
            "result": result,
            "self_heal": self_heal_result,
        }
        try:
            managed_holds = sync_ingest_semantic_holds(wiki_root=WIKI_ROOT)
        except Exception as exc:
            managed_holds = {"status": "error", "error": str(exc)}
        record["managed_holds"] = managed_holds
        batches.append(record)
        _append_jsonl(log_path, record)

        if not result.get("triggered"):
            stop_reason = result.get("reason", "ingest did not trigger")
            break
        if processed == 0 and pending_after >= pending_before:
            stop_reason = "no batch progress"
            break
        if pending_after == 0:
            stop_reason = "drained"
            break
        if sleep_seconds:
            time.sleep(sleep_seconds)

    pending_final = len(orchestrator.get_pending_raw_files())
    if pending_final == 0:
        status = "drained"
    elif not batches and pending_start == 0:
        status = "idle"
    elif stop_reason == "no batch progress":
        status = "stalled"
    elif batches and not batches[-1]["result"].get("triggered"):
        status = "blocked"
    else:
        status = "partial"

    return {
        "status": status,
        "pending_start": pending_start,
        "pending_after": pending_final,
        "batches_run": len(batches),
        "files_processed": total_processed,
        "stop_reason": stop_reason,
        "elapsed_seconds": round(time.time() - started, 2),
        "log_file": str(log_path),
        "managed_holds": managed_holds,
    }


def watch(
    *,
    max_batches: int = DEFAULT_MAX_BATCHES,
    max_units: int = DEFAULT_MAX_UNITS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    idle_sleep_seconds: float = DEFAULT_IDLE_SLEEP_SECONDS,
    log_file: Path | None = None,
) -> None:
    """Run drain cycles forever for launchd KeepAlive mode."""
    if idle_sleep_seconds < 0:
        raise ValueError("idle_sleep_seconds must be >= 0")

    while True:
        try:
            result = drain(
                max_batches=max_batches,
                max_units=max_units,
                sleep_seconds=sleep_seconds,
                log_file=log_file,
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error": str(exc),
                        "timestamp": datetime.now().isoformat(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if idle_sleep_seconds:
            time.sleep(idle_sleep_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drain LLM Wiki pending raw ingest backlog."
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=_env_int("LLM_WIKI_INGEST_DRAIN_MAX_BATCHES", DEFAULT_MAX_BATCHES),
        help="Maximum orchestrator batches to run in this process.",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=_env_int("LLM_WIKI_INGEST_DRAIN_MAX_UNITS", DEFAULT_MAX_UNITS),
        help="Semantic work units per orchestrator batch (1-10).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=_env_float(
            "LLM_WIKI_INGEST_DRAIN_SLEEP_SECONDS", DEFAULT_SLEEP_SECONDS
        ),
        help="Delay between successful batches.",
    )
    parser.add_argument("--log-file", type=Path)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Stay alive and run drain cycles repeatedly. Intended for launchd KeepAlive.",
    )
    parser.add_argument(
        "--idle-sleep-seconds",
        type=float,
        default=_env_float(
            "LLM_WIKI_INGEST_DRAIN_IDLE_SLEEP_SECONDS", DEFAULT_IDLE_SLEEP_SECONDS
        ),
        help="Delay between drain cycles in --watch mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.watch:
        watch(
            max_batches=args.max_batches,
            max_units=args.max_units,
            sleep_seconds=args.sleep_seconds,
            idle_sleep_seconds=args.idle_sleep_seconds,
            log_file=args.log_file,
        )
        return 0
    try:
        result = drain(
            max_batches=args.max_batches,
            max_units=args.max_units,
            sleep_seconds=args.sleep_seconds,
            log_file=args.log_file,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
