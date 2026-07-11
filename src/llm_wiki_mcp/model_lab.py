"""Autonomous frontier-model discovery, replay evaluation, and rollback."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from llm_wiki_mcp.wiki import WIKI_ROOT

LAB_DIR = WIKI_ROOT / "runtime" / "model-lab"
POLICY_FILE = LAB_DIR / "active-policy.json"
STATE_FILE = LAB_DIR / "state.json"
REPLAY_FILE = LAB_DIR / "replay.jsonl"
HISTORY_FILE = LAB_DIR / "history.jsonl"
LOCK_FILE = LAB_DIR / "model-lab.lock"

ROLE_SPECS: dict[str, dict[str, str]] = {
    "raw_writer": {"tier": "luna", "effort": "low", "fallback_model": "gpt-5.4-mini", "fallback_effort": "low"},
    "semantic_judge": {"tier": "terra", "effort": "medium", "fallback_model": "gpt-5.5", "fallback_effort": "medium"},
    "mutation_approver": {"tier": "sol", "effort": "low", "fallback_model": "gpt-5.5", "fallback_effort": "low"},
    "mutation_escalation": {"tier": "sol", "effort": "medium", "fallback_model": "gpt-5.5", "fallback_effort": "medium"},
    "code_repair": {"tier": "sol", "effort": "high", "fallback_model": "gpt-5.5", "fallback_effort": "high"},
}

MODEL_RE = re.compile(r"^gpt-(\d+)\.(\d+)(?:-(sol|terra|luna|mini))?$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _lock():
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
        try:
            os.unlink(tmp)
        except OSError:
            pass


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
    paths.extend([
        Path.home() / ".config" / "codex" / "models_cache.json",
        Path.home() / ".codex" / "models_cache.json",
    ])
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
        models.append({
            "slug": slug,
            "tier": match.group(3) or "flagship",
            "version": [int(match.group(1)), int(match.group(2))],
            "efforts": efforts,
            "priority": int(row.get("priority") or 0),
        })
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
        "raw_writer": ("luna", "mini", "terra", "flagship", "sol"),
        "semantic_judge": ("terra", "flagship", "sol", "luna", "mini"),
        "mutation_approver": ("sol", "flagship", "terra", "luna", "mini"),
        "mutation_escalation": ("sol", "flagship", "terra", "luna", "mini"),
        "code_repair": ("sol", "flagship", "terra", "luna", "mini"),
    }
    latest = discovery.get("latest", {})
    candidate = next((latest.get(tier) for tier in preferences[role] if latest.get(tier)), None)
    model = spec["fallback_model"]
    effort = spec["fallback_effort"]
    source = "fallback"
    if isinstance(candidate, dict):
        model = str(candidate["slug"])
        efforts = candidate.get("efforts", [])
        effort = spec["effort"] if spec["effort"] in efforts or not efforts else str(efforts[0])
        source = "codex-model-cache"
    return {"model": model, "effort": effort, "tier": str(candidate.get("tier")) if isinstance(candidate, dict) else spec["tier"], "source": source}


def bootstrap_policy(*, write: bool = False, discovery: dict[str, Any] | None = None) -> dict[str, Any]:
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
            _append_jsonl(HISTORY_FILE, {"event": "bootstrap", "timestamp": _now(), "roles": policy["roles"]})
    return policy


def load_policy() -> dict[str, Any]:
    value = _read_json(POLICY_FILE, {})
    if isinstance(value, dict) and isinstance(value.get("roles"), dict):
        return value
    return bootstrap_policy(write=False)


def resolve_role(role: str) -> tuple[str, str]:
    if role not in ROLE_SPECS:
        role = "semantic_judge"
    env_key = re.sub(r"[^A-Z0-9]", "_", role.upper())
    policy = load_policy()
    selected = policy.get("roles", {}).get(role, _selection(role, discover_models()))
    model = os.environ.get(f"LLM_WIKI_MODEL_{env_key}", str(selected.get("model") or "")).strip()
    effort = os.environ.get(f"LLM_WIKI_EFFORT_{env_key}", str(selected.get("effort") or "")).strip()
    return model, effort


def decision_signature(value: dict[str, Any]) -> dict[str, Any]:
    keys = ("decision", "classification", "action", "approved", "ignored_pages")
    return {key: value.get(key) for key in keys if key in value}


def _model_version(model: Any) -> tuple[int, int]:
    match = MODEL_RE.match(str(model or ""))
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def record_replay_case(
    *, role: str, prompt: str, schema: dict[str, Any], result: dict[str, Any], model: str,
    effort: str, latency_seconds: float,
) -> None:
    if role not in ROLE_SPECS or os.environ.get("LLM_WIKI_MODEL_LAB_REPLAY") == "1":
        return
    if result.get("frontier_failure"):
        return
    _append_jsonl(REPLAY_FILE, {
        "timestamp": _now(), "role": role, "model": model, "effort": effort,
        "prompt": prompt[-50_000:], "schema": schema,
        "expected": decision_signature(result), "latency_seconds": round(latency_seconds, 3),
    })


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
    return expected_decision in {"rejected", "needs_retry", "quarantined"} and actual_decision == "approved"


def record_live_result(*, role: str, model: str, ok: bool, failure_class: str | None = None) -> dict[str, Any] | None:
    with _lock():
        policy = _read_json(POLICY_FILE, {})
        canary = policy.get("canaries", {}).get(role) if isinstance(policy, dict) else None
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
            _append_jsonl(HISTORY_FILE, {"event": "rollback", "timestamp": _now(), "role": role, "model": model, "failure_class": failure_class, "calls": calls, "failures": failures})
        elif calls >= int(canary.get("target_calls") or 20):
            policy["canaries"].pop(role, None)
            _append_jsonl(HISTORY_FILE, {"event": "canary_complete", "timestamp": _now(), "role": role, "model": model, "calls": calls})
        _atomic_json(POLICY_FILE, policy)
        return {"rollback": rollback, "calls": calls, "failures": failures}


Reviewer = Callable[[dict[str, Any], dict[str, str]], dict[str, Any]]


def _default_reviewer(case: dict[str, Any], candidate: dict[str, str]) -> dict[str, Any]:
    from llm_wiki_mcp.frontier_review import run_structured_review
    old = os.environ.get("LLM_WIKI_MODEL_LAB_REPLAY")
    os.environ["LLM_WIKI_MODEL_LAB_REPLAY"] = "1"
    try:
        return run_structured_review(
            str(case["prompt"]), dict(case["schema"]), repo_root=Path.cwd(), timeout=300,
            model_role=str(case["role"]), model_override=candidate["model"],
            reasoning_effort_override=candidate["effort"], record_replay=False,
        )
    finally:
        if old is None:
            os.environ.pop("LLM_WIKI_MODEL_LAB_REPLAY", None)
        else:
            os.environ["LLM_WIKI_MODEL_LAB_REPLAY"] = old


def run_due(*, dry_run: bool = False, max_evaluations: int = 2, reviewer: Reviewer | None = None) -> dict[str, Any]:
    discovery = discover_models()
    if not POLICY_FILE.exists():
        policy = bootstrap_policy(write=not dry_run, discovery=discovery)
        return {"status": "bootstrapped", "dry_run": dry_run, "roles": policy["roles"], "evaluated": 0}
    policy = load_policy()
    # One-time migration for policies written before model discovery became
    # available. This is not a future promotion bypass: the flag is cleared
    # atomically, and every later model change must pass replay gates.
    if policy.get("bootstrap") is True and not REPLAY_FILE.exists():
        policy["roles"] = {role: _selection(role, discovery) for role in ROLE_SPECS}
        policy["bootstrap"] = False
        policy["bootstrap_completed_at"] = _now()
        policy["updated_at"] = _now()
        if not dry_run:
            with _lock():
                _atomic_json(POLICY_FILE, policy)
                _append_jsonl(HISTORY_FILE, {"event": "bootstrap_migrated", "timestamp": _now(), "roles": policy["roles"]})
        return {"status": "bootstrapped", "dry_run": dry_run, "roles": policy["roles"], "evaluated": 0}
    reviewer = reviewer or _default_reviewer
    state = _read_json(STATE_FILE, {"schema_version": 1, "candidates": {}})
    candidates = state.setdefault("candidates", {})
    evaluated = 0
    promoted: list[str] = []
    pending: list[dict[str, Any]] = []
    for role in ROLE_SPECS:
        desired = _selection(role, discovery)
        active = policy.get("roles", {}).get(role, {})
        if desired["model"] == active.get("model") and desired["effort"] == active.get("effort"):
            continue
        if _model_version(desired["model"]) <= _model_version(active.get("model")):
            continue
        key = f"{role}:{desired['model']}:{desired['effort']}"
        candidate = candidates.setdefault(key, {"role": role, "model": desired["model"], "effort": desired["effort"], "cases": 0, "matches": 0, "schema_failures": 0, "unsafe": 0, "created_at": _now()})
        replays = _load_replays(role)
        seen = int(candidate.get("cases") or 0)
        available = replays[seen:]
        for case in available[: max(0, max_evaluations - evaluated)]:
            started = time.monotonic()
            actual = reviewer(case, {"model": desired["model"], "effort": desired["effort"]})
            actual_sig = decision_signature(actual)
            expected = dict(case.get("expected") or {})
            candidate["cases"] = int(candidate.get("cases") or 0) + 1
            candidate["matches"] = int(candidate.get("matches") or 0) + int(actual_sig == expected)
            candidate["schema_failures"] = int(candidate.get("schema_failures") or 0) + int(bool(actual.get("frontier_failure")))
            candidate["unsafe"] = int(candidate.get("unsafe") or 0) + int(_is_unsafe(expected, actual_sig))
            candidate["last_latency_seconds"] = round(time.monotonic() - started, 3)
            candidate["updated_at"] = _now()
            evaluated += 1
        minimum = max(3, int(os.environ.get("LLM_WIKI_MODEL_LAB_MIN_REPLAYS", "8")))
        cases = int(candidate.get("cases") or 0)
        agreement = int(candidate.get("matches") or 0) / cases if cases else 0.0
        passed = cases >= minimum and candidate.get("schema_failures") == 0 and candidate.get("unsafe") == 0 and agreement >= 0.95
        if passed and not dry_run:
            previous = dict(active)
            policy["roles"][role] = desired
            policy.setdefault("canaries", {})[role] = {"model": desired["model"], "effort": desired["effort"], "previous": previous, "calls": 0, "failures": 0, "target_calls": 20, "started_at": _now()}
            policy["updated_at"] = _now()
            promoted.append(role)
            _append_jsonl(HISTORY_FILE, {"event": "promoted", "timestamp": _now(), "role": role, "candidate": desired, "previous": previous, "cases": cases, "agreement": agreement})
        else:
            pending.append({"role": role, "candidate": desired, "cases": cases, "minimum": minimum, "agreement": round(agreement, 3), "unsafe": candidate.get("unsafe", 0), "schema_failures": candidate.get("schema_failures", 0), "reason": "no_replay_cases" if not replays else "replay_gate"})
        if evaluated >= max_evaluations:
            break
    if not dry_run:
        with _lock():
            _atomic_json(STATE_FILE, state)
            if promoted:
                _atomic_json(POLICY_FILE, policy)
    return {"status": "promoted" if promoted else "pending" if pending else "current", "dry_run": dry_run, "evaluated": evaluated, "promoted": promoted, "pending": pending}


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
    return {"status": "ok", "policy": policy, "discovery": discovery, "candidates": list(state.get("candidates", {}).values()), "history": history, "replay_cases": sum(1 for _ in REPLAY_FILE.open(encoding="utf-8")) if REPLAY_FILE.exists() else 0}


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
        result = run_due(dry_run=args.dry_run, max_evaluations=max(0, args.max_evaluations))
    else:
        result = snapshot()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
