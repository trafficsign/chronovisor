"""Autonomous self-healing loop for LLM Wiki failure packets."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp import runtime_status, wiki
from llm_wiki_mcp.alias_store import add_alias
from llm_wiki_mcp.local_repair import LocalRepairDecision, propose_repair


SELF_HEAL_STATUSES = {
    "pending_local_repair",
    "local_repair_failed",
    "pending_frontier",
    "frontier_retry",
}


def _repo_root() -> Path:
    configured = os.environ.get("LLM_WIKI_REPO_ROOT")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2]


def _failures_dir() -> Path:
    return wiki.WIKI_ROOT / "runtime" / "failures"


def _packet_dir() -> Path:
    return _failures_dir() / "packets"


def _local_repair_dir() -> Path:
    return _failures_dir() / "local-repair"


def _frontier_queue_dir() -> Path:
    return _failures_dir() / "frontier-queue"


def _frontier_decision_dir() -> Path:
    return _failures_dir() / "frontier-decisions"


def _applied_actions_dir() -> Path:
    return _failures_dir() / "applied-actions"


def _rejected_actions_dir() -> Path:
    return _failures_dir() / "rejected-actions"


def _registry_file() -> Path:
    return _failures_dir() / "failure-registry.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _append_registry(record: dict[str, Any]) -> None:
    path = _registry_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _update_packet(path: Path, packet: dict[str, Any], **updates: Any) -> None:
    packet.update(updates)
    packet["updated_at"] = datetime.now().isoformat()
    _write_json(path, packet)


def pending_packets() -> list[Path]:
    if not _packet_dir().exists():
        return []
    out: list[Path] = []
    for path in sorted(_packet_dir().glob("*.json")):
        try:
            packet = _read_json(path)
        except Exception:
            continue
        if packet.get("status") in SELF_HEAL_STATUSES:
            out.append(path)
    return out


def _raw_candidate_paths(packet: dict[str, Any]) -> list[Path]:
    raw_file = packet.get("raw_file")
    if not isinstance(raw_file, str) or not raw_file:
        return []
    return [
        _failures_dir() / "quarantined-raw" / raw_file,
        wiki.RAW_DIR / raw_file,
    ]


def _restore_quarantined_raw(packet: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    candidates = _raw_candidate_paths(packet)
    if not candidates:
        return {"restored": False, "reason": "packet has no raw_file"}
    quarantine_path, raw_path = candidates[0], candidates[1]
    if raw_path.exists():
        return {"restored": False, "reason": "raw already pending", "path": str(raw_path)}
    if not quarantine_path.exists():
        return {"restored": False, "reason": "quarantined raw not found"}
    if dry_run:
        return {
            "restored": False,
            "dry_run": True,
            "source": str(quarantine_path),
            "target": str(raw_path),
        }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(quarantine_path), str(raw_path))
    return {"restored": True, "source": str(quarantine_path), "target": str(raw_path)}


def _retry_ingest(*, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"triggered": False, "dry_run": True}
    from llm_wiki_mcp import orchestrator

    return orchestrator.run_pending_ingest(force=True)


def apply_local_decision(
    packet: dict[str, Any],
    decision: LocalRepairDecision,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply a whitelisted local repair action."""

    if decision.action == "resolve_update_target":
        if not decision.requested_page_id or not decision.target_page_id:
            raise ValueError("resolve_update_target requires requested and target ids")
        if not dry_run:
            add_alias(
                decision.requested_page_id,
                decision.target_page_id,
                source=packet.get("failure_id"),
            )
        restore = _restore_quarantined_raw(packet, dry_run=dry_run)
        retry = _retry_ingest(dry_run=dry_run)
        return {
            "action": decision.action,
            "alias": {
                "requested": decision.requested_page_id,
                "target": decision.target_page_id,
                "dry_run": dry_run,
            },
            "restore": restore,
            "retry": retry,
        }

    if decision.action == "retry_raw":
        restore = _restore_quarantined_raw(packet, dry_run=dry_run)
        retry = _retry_ingest(dry_run=dry_run)
        return {"action": decision.action, "restore": restore, "retry": retry}

    if decision.action == "quarantine_raw":
        return {"action": decision.action, "kept_quarantined": True}

    raise ValueError(f"local action requires frontier or is not directly applicable: {decision.action}")


def _save_local_decision(packet_path: Path, decision: LocalRepairDecision) -> Path:
    path = _local_repair_dir() / packet_path.name
    _write_json(path, decision.to_dict())
    return path


def _save_action(packet_path: Path, action: dict[str, Any], *, applied: bool) -> Path:
    target_dir = _applied_actions_dir() if applied else _rejected_actions_dir()
    path = target_dir / packet_path.name
    _write_json(path, action)
    return path


def _queue_frontier(packet_path: Path, packet: dict[str, Any], decision: dict[str, Any] | None) -> Path:
    target = _frontier_queue_dir() / packet_path.name
    payload = {
        "queued_at": datetime.now().isoformat(),
        "packet_path": str(packet_path),
        "packet": packet,
        "local_decision": decision,
    }
    _write_json(target, payload)
    return target


def _run_frontier(
    packet_path: Path,
    packet: dict[str, Any],
    local_decision: dict[str, Any] | None,
    *,
    execute_patch: bool,
) -> dict[str, Any]:
    from llm_wiki_mcp.frontier_review import run_frontier_review

    result = run_frontier_review(
        packet,
        local_decision,
        repo_root=_repo_root(),
        execute_patch=execute_patch,
    )
    payload = result.to_dict()
    _write_json(_frontier_decision_dir() / packet_path.name, payload)
    return payload


def handle_packet(
    packet_path: Path,
    *,
    use_qwen: bool = True,
    enable_frontier: bool = True,
    execute_frontier_patch: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    packet = _read_json(packet_path)
    _update_packet(packet_path, packet, status="local_repairing")
    decision = propose_repair(packet, use_qwen=use_qwen)
    decision_path = _save_local_decision(packet_path, decision)

    result: dict[str, Any] = {
        "packet": str(packet_path),
        "failure_id": packet.get("failure_id"),
        "local_decision": decision.to_dict(),
        "local_decision_path": str(decision_path),
    }

    try:
        if decision.status == "resolved" and decision.action in {
            "resolve_update_target",
            "retry_raw",
            "quarantine_raw",
        }:
            action = apply_local_decision(packet, decision, dry_run=dry_run)
            action_path = _save_action(packet_path, action, applied=True)
            _update_packet(
                packet_path,
                packet,
                status="local_repair_applied" if not dry_run else "local_repair_dry_run",
                local_decision=decision.to_dict(),
                applied_action_path=str(action_path),
            )
            _append_registry({
                "timestamp": datetime.now().isoformat(),
                "failure_id": packet.get("failure_id"),
                "failure_class": packet.get("failure_class"),
                "fingerprint": packet.get("fingerprint"),
                "resolution": "local",
                "decision": decision.to_dict(),
                "action": action,
            })
            runtime_status.safe_append_event(
                "success",
                f"self-heal | local repair applied for {packet.get('raw_file')}",
                source="self-heal",
                packet=str(packet_path),
                action=decision.action,
            )
            result["status"] = "local_repair_applied"
            result["action"] = action
            return result
    except Exception as exc:
        action = {
            "action": decision.action,
            "error": str(exc),
            "decision": decision.to_dict(),
        }
        _save_action(packet_path, action, applied=False)
        _update_packet(
            packet_path,
            packet,
            status="local_repair_failed",
            local_decision=decision.to_dict(),
            local_error=str(exc),
        )
        result["local_error"] = str(exc)

    queue_path = _queue_frontier(packet_path, packet, decision.to_dict())
    result["frontier_queue_path"] = str(queue_path)
    if not enable_frontier:
        _update_packet(
            packet_path,
            packet,
            status="pending_frontier",
            local_decision=decision.to_dict(),
            frontier_queue_path=str(queue_path),
        )
        result["status"] = "pending_frontier"
        return result

    _update_packet(
        packet_path,
        packet,
        status="frontier_running",
        local_decision=decision.to_dict(),
        frontier_queue_path=str(queue_path),
    )
    frontier_result = _run_frontier(
        packet_path,
        packet,
        decision.to_dict(),
        execute_patch=execute_frontier_patch and not dry_run,
    )
    final_status = (
        "frontier_approved"
        if frontier_result.get("decision") == "approved"
        else "frontier_retry"
        if frontier_result.get("decision") == "needs_retry"
        else "frontier_quarantined"
        if frontier_result.get("decision") == "quarantined"
        else "frontier_rejected"
    )
    _update_packet(
        packet_path,
        packet,
        status=final_status,
        frontier_result=frontier_result,
    )
    _append_registry({
        "timestamp": datetime.now().isoformat(),
        "failure_id": packet.get("failure_id"),
        "failure_class": packet.get("failure_class"),
        "fingerprint": packet.get("fingerprint"),
        "resolution": "frontier",
        "decision": decision.to_dict(),
        "frontier": frontier_result,
    })
    result["status"] = final_status
    result["frontier_result"] = frontier_result
    return result


def run_pending(
    *,
    max_packets: int = 3,
    use_qwen: bool = True,
    enable_frontier: bool = True,
    execute_frontier_patch: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    packets = pending_packets()[:max_packets]
    results = [
        handle_packet(
            packet,
            use_qwen=use_qwen,
            enable_frontier=enable_frontier,
            execute_frontier_patch=execute_frontier_patch,
            dry_run=dry_run,
        )
        for packet in packets
    ]
    return {
        "status": "ok",
        "packets_seen": len(packets),
        "results": results,
    }


def start_background(packet_path: Path) -> None:
    """Launch self-heal asynchronously after quarantine."""

    if os.environ.get("LLM_WIKI_SELF_HEAL_AUTORUN", "1") in {"0", "false", "False"}:
        return
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "llm_wiki_mcp.self_heal",
            "--packet",
            str(packet_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def drill_packet() -> dict[str, Any]:
    return {
        "failure_id": "drill-update-target-not-found",
        "created_at": datetime.now().isoformat(),
        "raw_file": "drill.md",
        "job_id": "drill",
        "failure_class": "apply.update_target_not_found",
        "fingerprint": (
            "apply.update_target_not_found:"
            "opus-4-7-evaluation-and-industry-geopolitics"
        ),
        "attempts": 3,
        "error": (
            "update target not found for page_id "
            "'opus-4-7-evaluation-and-industry-geopolitics'"
        ),
        "requested_page_id": "opus-4-7-evaluation-and-industry-geopolitics",
        "similar_existing_pages": [
            "ai/opus-4.7-evaluation-and-industry-geopolitics"
        ],
        "status": "pending_local_repair",
    }


def run_drill(*, use_qwen: bool = True) -> dict[str, Any]:
    packet = drill_packet()
    decision = propose_repair(packet, use_qwen=use_qwen)
    return {"packet": packet, "decision": decision.to_dict()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM Wiki self-healing.")
    parser.add_argument("--packet", type=Path, help="Process one failure packet.")
    parser.add_argument("--max-packets", type=int, default=3)
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument("--no-frontier", action="store_true")
    parser.add_argument("--review-only", action="store_true", help="Frontier may review but not patch.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--drill", action="store_true", help="Run a synthetic local repair drill.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.drill:
        print(json.dumps(run_drill(use_qwen=not args.no_qwen), ensure_ascii=False, indent=2))
        return 0
    if args.packet:
        result = handle_packet(
            args.packet,
            use_qwen=not args.no_qwen,
            enable_frontier=not args.no_frontier,
            execute_frontier_patch=not args.review_only,
            dry_run=args.dry_run,
        )
    else:
        result = run_pending(
            max_packets=args.max_packets,
            use_qwen=not args.no_qwen,
            enable_frontier=not args.no_frontier,
            execute_frontier_patch=not args.review_only,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
