"""Autonomous frontier-model discovery, replay evaluation, and rollback."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import exclusive_text_file_lock
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
)
from chronovisor.core.store import (
    MODEL_LAB_REPLAY_FILE as REPLAY_FILE,
)
from chronovisor.core.timeutil import utc_iso_seconds as _now
from chronovisor.decision.decision_schema_manifest import (
    decision_signature_value,
    default_decision_value,
)

LAB_DIR = CHRONOVISOR_ROOT / "runtime" / "model-lab"
POLICY_FILE = LAB_DIR / "active-policy.json"
STATE_FILE = LAB_DIR / "state.json"
HISTORY_FILE = LAB_DIR / "history.jsonl"
LOCK_FILE = LAB_DIR / "model-lab.lock"
REPLAY_PROMPT_LIMIT = 50_000

ROLE_SPECS: dict[str, dict[str, str]] = {
    "code_repair": {
        "tier": "sol",
        "effort": "high",
        "fallback_model": "gpt-5.5",
        "fallback_effort": "high",
    },
}

MODEL_RE = re.compile(r"^gpt-(\d+)\.(\d+)(?:-(sol|terra|luna|mini))?$")
_lock = partial(exclusive_text_file_lock, LOCK_FILE)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def cache_paths() -> list[Path]:
    paths: list[Path] = []
    if os.environ.get("CODEX_HOME"):
        paths.append(Path(os.environ["CODEX_HOME"]).expanduser() / "models_cache.json")
    paths.extend(
        [
            Path.home() / ".config" / "codex" / "models_cache.json",
            Path.home() / ".codex" / "models_cache.json",
        ]
    )
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def discover_models(paths: list[Path] | None = None) -> dict[str, Any]:
    source: Path | None = None
    payload: dict[str, Any] = {}
    for path in paths or cache_paths():
        value = _read_json(path, {})
        if isinstance(value, dict) and isinstance(value.get("models"), list):
            payload = value
            source = path
            break
    models: list[dict[str, Any]] = []
    for row in payload.get("models", []):
        if not isinstance(row, dict):
            continue
        slug = str(row.get("slug") or "")
        match = MODEL_RE.match(slug)
        if not match or str(row.get("visibility") or "list") not in {"list", "visible"}:
            continue
        efforts = [
            str(item.get("effort"))
            for item in row.get("supported_reasoning_levels", [])
            if isinstance(item, dict) and item.get("effort")
        ]
        models.append(
            {
                "slug": slug,
                "tier": match.group(3) or "flagship",
                "version": [int(match.group(1)), int(match.group(2))],
                "efforts": efforts,
                "priority": int(row.get("priority") or 0),
            }
        )
    models.sort(key=lambda row: (row["version"], row["priority"]), reverse=True)
    latest: dict[str, dict[str, Any]] = {}
    for row in models:
        latest.setdefault(str(row["tier"]), row)
    return {
        "status": "ok" if source else "unavailable",
        "source": str(source) if source else None,
        "fetched_at": payload.get("fetched_at"),
        "models": models,
        "latest": latest,
    }


def _selection(role: str, discovery: dict[str, Any]) -> dict[str, Any]:
    spec = ROLE_SPECS[role]
    preferences = {
        "code_repair": ("sol", "flagship", "terra", "luna", "mini"),
    }
    latest = discovery.get("latest", {})
    candidate = next(
        (latest.get(tier) for tier in preferences[role] if latest.get(tier)), None
    )
    model = spec["fallback_model"]
    effort = spec["fallback_effort"]
    source = "fallback"
    if isinstance(candidate, dict):
        model = str(candidate["slug"])
        efforts = candidate.get("efforts", [])
        effort = (
            spec["effort"]
            if spec["effort"] in efforts or not efforts
            else str(efforts[0])
        )
        source = "codex-model-cache"
    return {
        "model": model,
        "effort": effort,
        "tier": str(candidate.get("tier"))
        if isinstance(candidate, dict)
        else spec["tier"],
        "source": source,
    }


def bootstrap_policy(
    *, write: bool = False, discovery: dict[str, Any] | None = None
) -> dict[str, Any]:
    discovery = discovery or discover_models()
    policy = {
        "schema_version": 1,
        "updated_at": _now(),
        "bootstrap": False,
        "bootstrap_completed_at": _now(),
        "roles": {role: _selection(role, discovery) for role in ROLE_SPECS},
        "canaries": {},
    }
    if write:
        with _lock():
            existing = _read_json(POLICY_FILE, {})
            if isinstance(existing, dict) and isinstance(existing.get("roles"), dict):
                return existing
            _atomic_json(POLICY_FILE, policy)
            _append_jsonl(
                HISTORY_FILE,
                {"event": "bootstrap", "timestamp": _now(), "roles": policy["roles"]},
            )
    return policy


def load_policy() -> dict[str, Any]:
    value = _read_json(POLICY_FILE, {})
    if isinstance(value, dict) and isinstance(value.get("roles"), dict):
        return value
    return bootstrap_policy(write=False)


def resolve_role(role: str) -> tuple[str, str]:
    from chronovisor.decision.local_model_eval import resolve_frontier_role

    return resolve_frontier_role(
        role,
        policy=load_policy(),
        discovery=discover_models(),
    )


def decision_signature(
    value: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = (
        decision_signature_value(schema, value)
        if schema is not None
        else default_decision_value(value)
    )
    return dict(selected) if isinstance(selected, Mapping) else {}


def _model_version(model: Any) -> tuple[int, int]:
    match = MODEL_RE.match(str(model or ""))
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def record_replay_case(
    *,
    role: str,
    prompt: str,
    schema: dict[str, Any],
    result: dict[str, Any],
    model: str,
    effort: str,
    latency_seconds: float,
) -> None:
    if role not in ROLE_SPECS or os.environ.get("CHRONOVISOR_MODEL_LAB_REPLAY") == "1":
        return
    if result.get("frontier_failure"):
        return
    prompt_truncated = len(prompt) > REPLAY_PROMPT_LIMIT
    _append_jsonl(
        REPLAY_FILE,
        {
            "timestamp": _now(),
            "role": role,
            "model": model,
            "effort": effort,
            "prompt": prompt[-REPLAY_PROMPT_LIMIT:],
            "prompt_truncated": prompt_truncated,
            "prompt_original_chars": len(prompt),
            "expected": decision_signature(result, schema),
            "latency_seconds": round(latency_seconds, 3),
        },
    )


def record_local_replay_case(
    *,
    role: str,
    prompt: str,
    schema: Mapping[str, Any],
    result: Mapping[str, Any],
    models: Sequence[str],
    latency_seconds: float,
    system: str | None = None,
    policy_source: str | None = None,
    policy_artifact_sha256: str | None = None,
    decision_lane: str | None = None,
    lane_contract_sha256: str | None = None,
    lane_contract_effect: str | None = None,
    effective_request_sha256: str | None = None,
    effective_model_prompt: str | None = None,
    effective_model_system: str | None = None,
    replay_file: Path | None = None,
) -> bool:
    from chronovisor.decision.local_model_eval import (
        record_local_replay_case as _record_local_replay_case,
    )

    return _record_local_replay_case(
        role=role,
        prompt=prompt,
        schema=schema,
        result=result,
        models=models,
        latency_seconds=latency_seconds,
        system=system,
        policy_source=policy_source,
        policy_artifact_sha256=policy_artifact_sha256,
        decision_lane=decision_lane,
        lane_contract_sha256=lane_contract_sha256,
        lane_contract_effect=lane_contract_effect,
        effective_request_sha256=effective_request_sha256,
        effective_model_prompt=effective_model_prompt,
        effective_model_system=effective_model_system,
        replay_file=replay_file or REPLAY_FILE,
    )


def _load_replays(role: str, limit: int = 40) -> list[dict[str, Any]]:
    try:
        lines = REPLAY_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("role") == role:
            rows.append(row)
        if len(rows) >= limit:
            break
    return list(reversed(rows))


def _is_unsafe(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    expected_decision = str(expected.get("decision") or "")
    actual_decision = str(actual.get("decision") or "")
    return (
        expected_decision in {"rejected", "needs_retry", "quarantined"}
        and actual_decision == "approved"
    )


def record_live_result(
    *, role: str, model: str, ok: bool, failure_class: str | None = None
) -> dict[str, Any] | None:
    with _lock():
        policy = _read_json(POLICY_FILE, {})
        canary = (
            policy.get("canaries", {}).get(role) if isinstance(policy, dict) else None
        )
        if not isinstance(canary, dict) or model != canary.get("model"):
            return None
        canary["calls"] = int(canary.get("calls") or 0) + 1
        canary["failures"] = int(canary.get("failures") or 0) + (0 if ok else 1)
        canary["updated_at"] = _now()
        calls = int(canary["calls"])
        failures = int(canary["failures"])
        rollback = failures >= 2 and calls >= 5 and failures / calls > 0.10
        if rollback:
            previous = canary.get("previous")
            if isinstance(previous, dict):
                policy["roles"][role] = previous
            policy["canaries"].pop(role, None)
            policy["updated_at"] = _now()
            _append_jsonl(
                HISTORY_FILE,
                {
                    "event": "rollback",
                    "timestamp": _now(),
                    "role": role,
                    "model": model,
                    "failure_class": failure_class,
                    "calls": calls,
                    "failures": failures,
                },
            )
        elif calls >= int(canary.get("target_calls") or 20):
            policy["canaries"].pop(role, None)
            _append_jsonl(
                HISTORY_FILE,
                {
                    "event": "canary_complete",
                    "timestamp": _now(),
                    "role": role,
                    "model": model,
                    "calls": calls,
                },
            )
        _atomic_json(POLICY_FILE, policy)
        return {"rollback": rollback, "calls": calls, "failures": failures}


def run_due(*, dry_run: bool = False, max_evaluations: int = 0) -> dict[str, Any]:
    """Track the newest repair model without running routine model reviews.

    The frontier model is no longer a data-plane judge, so replay evaluation
    here would itself create forbidden subscription traffic.  Discovery may
    promote only the single ``code_repair`` role by monotonically newer model
    version; the durable repair guard and the incident's tests remain the
    actual execution gate.
    """
    discovery = discover_models()
    if not POLICY_FILE.exists():
        policy = bootstrap_policy(write=not dry_run, discovery=discovery)
        return {
            "status": "bootstrapped",
            "dry_run": dry_run,
            "roles": policy["roles"],
            "evaluated": 0,
        }
    policy = load_policy()
    active = policy.get("roles", {}).get("code_repair", {})
    desired = _selection("code_repair", discovery)
    newer = _model_version(desired["model"]) > _model_version(active.get("model"))
    changed = newer or set(policy.get("roles", {})) != {"code_repair"}
    if not changed:
        return {"status": "current", "dry_run": dry_run, "evaluated": 0, "promoted": []}
    if dry_run:
        return {
            "status": "would_promote" if newer else "would_migrate",
            "dry_run": True,
            "evaluated": 0,
            "candidate": desired,
        }
    previous = dict(active)
    policy["roles"] = {"code_repair": desired if newer else active}
    policy["canaries"] = (
        {
            "code_repair": {
                "model": desired["model"],
                "effort": desired["effort"],
                "previous": previous,
                "calls": 0,
                "failures": 0,
                "target_calls": 5,
                "started_at": _now(),
            }
        }
        if newer
        else {}
    )
    policy["bootstrap"] = False
    policy["updated_at"] = _now()
    with _lock():
        _atomic_json(POLICY_FILE, policy)
        _append_jsonl(
            HISTORY_FILE,
            {
                "event": "repair_model_promoted" if newer else "repair_roles_migrated",
                "timestamp": _now(),
                "role": "code_repair",
                "candidate": desired,
                "previous": previous,
                "gate": "version_discovery_plus_guarded_incident_tests",
            },
        )
    return {
        "status": "promoted" if newer else "migrated",
        "dry_run": False,
        "evaluated": 0,
        "promoted": ["code_repair"] if newer else [],
    }


def snapshot() -> dict[str, Any]:
    policy = load_policy()
    state = _read_json(STATE_FILE, {"candidates": {}})
    history: list[dict[str, Any]] = []
    try:
        for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines()[-20:]:
            value = json.loads(line)
            if isinstance(value, dict):
                history.append(value)
    except (OSError, json.JSONDecodeError):
        pass
    discovery = discover_models()
    return {
        "status": "ok",
        "policy": policy,
        "discovery": discovery,
        "candidates": list(state.get("candidates", {}).values()),
        "history": history,
        "replay_cases": sum(1 for _ in REPLAY_FILE.open(encoding="utf-8"))
        if REPLAY_FILE.exists()
        else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Autonomous frontier Model Lab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("discover")
    run = sub.add_parser("run-due")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--max-evaluations", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "discover":
        result = discover_models()
    elif args.command == "run-due":
        result = run_due(
            dry_run=args.dry_run, max_evaluations=max(0, args.max_evaluations)
        )
    else:
        result = snapshot()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
