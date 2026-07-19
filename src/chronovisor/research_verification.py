"""Independent, temp-only adversarial verification for the research lane."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from chronovisor.research_config import ResearchConfig, WebConfig
from chronovisor.research_orchestrator import PlannerResponse, run_research
from chronovisor.research_scheduler import foreground_lane, research_lane
from chronovisor.research_security import guard_egress_query, guard_url
from chronovisor.research_store import ResearchStore, reduce_events
from chronovisor.research_types import Action, ActionType, parse_action
from chronovisor.web_fetch import fetch_web
from chronovisor.web_provider import HttpSearchProvider, search_web


def _check(name: str, command: str, function: Callable[[], None]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        function()
        return {"name": name, "status": "PASS", "command": command, "elapsed_ms": round((time.monotonic() - started) * 1000)}
    except Exception as exc:
        return {
            "name": name,
            "status": "FAIL",
            "command": command,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def run_verification() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="chronovisor-research-verify-") as temporary:
        root = Path(temporary)

        def strict_action() -> None:
            assert parse_action({"type": "unknown", "arguments": {}, "rationale": "x"}, epoch=0).action is None
            assert parse_action({"type": "finish", "arguments": {}, "rationale": "x", "extra": True}, epoch=0).action is None

        checks.append(_check("malformed-and-unknown-action", "uv run pytest -q tests/test_research_types.py", strict_action))

        def terminal_ledger() -> None:
            reduced = reduce_events(
                [
                    {"kind": "action", "epoch": 0, "iteration": 1, "action": {"type": "chronovisor_read"}},
                    {"kind": "stop", "epoch": 1, "stop_reason": "interrupted"},
                ]
            )
            assert reduced["orphan_actions"]
            assert reduced["terminal"] is True
            assert reduced["epoch"] == 1
            assert Action(ActionType.WIKI_SEARCH, {"query": "x"}).canonical_key() == Action(
                ActionType.WIKI_SEARCH, {"query": "x"}
            ).canonical_key()

        checks.append(_check("late-result-interrupt-orphan", "uv run pytest -q tests/test_research_orchestrator.py", terminal_ledger))

        def security() -> None:
            assert not guard_egress_query("token sk-abcdefghijklmnopqrstuvwxyz").allowed
            assert not guard_egress_query("safe\u200bquery").allowed
            policy, _ = guard_url(
                "http://metadata.test/",
                resolver=lambda _host, _port: ["169.254.169.254"],
            )
            assert not policy.allowed

        checks.append(_check("secret-unicode-ssrf", "uv run pytest -q tests/test_research_security.py", security))

        def outage() -> None:
            from chronovisor import web_provider

            web_provider.WEB_TRACE = root / "web-trace.jsonl"
            provider = HttpSearchProvider(
                name="searxng",
                endpoint="https://example.com",
                client=httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(503))),
            )
            response = search_web(
                "safe query",
                config=WebConfig(adapter_enabled=True, live_egress_enabled=True, provider="searxng", endpoint="https://example.com"),
                provider=provider,
            )
            assert response.status == "degraded"

        checks.append(_check("provider-outage-fallback", "uv run pytest -q tests/test_web_provider.py", outage))

        def redirect_and_oversize() -> None:
            resolver = lambda _host, _port: ["93.184.216.34"]

            def redirect_handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(302, headers={"location": "/loop"}, request=request)

            redirect = fetch_web(
                "https://example.com/loop",
                config=WebConfig(adapter_enabled=True, live_egress_enabled=True, max_fetch_bytes=100),
                client=httpx.Client(transport=httpx.MockTransport(redirect_handler)),
                resolver=resolver,
                cache_dir=root / "redirect-cache",
            )
            assert redirect.status == "blocked" and redirect.error == "redirect_loop"

            oversized = fetch_web(
                "https://example.com/big",
                config=WebConfig(adapter_enabled=True, live_egress_enabled=True, max_fetch_bytes=10),
                client=httpx.Client(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(
                            200,
                            headers={"content-type": "text/plain", "content-length": "100"},
                            content=b"x" * 100,
                            request=request,
                        )
                    )
                ),
                resolver=resolver,
                cache_dir=root / "oversized-cache",
            )
            assert oversized.status == "blocked" and oversized.error == "declared_body_too_large"

        checks.append(_check("redirect-loop-and-oversized-content", "uv run pytest -q tests/test_web_fetch.py", redirect_and_oversize))

        def outage_terminal_and_sync_overlap() -> None:
            from chronovisor import research_scheduler

            scheduler_root = root / "scheduler"
            research_scheduler.RUNTIME_DIR = scheduler_root
            research_scheduler.SYNC_DIR = scheduler_root / "sync-pending"
            research_scheduler.RESEARCH_LOCK = scheduler_root / "research.lock"
            research_scheduler.ACTIVE_FILE = scheduler_root / "active.json"
            research_scheduler.SCHEDULER_LOG = scheduler_root / "scheduler.jsonl"

            class OutagePlanner:
                needs_model = False

                def plan(self, *_args, **_kwargs):
                    return PlannerResponse(None, status="error", error="Ollama unavailable")

            store = ResearchStore(root=root / "outage-research")
            result = run_research(
                "outage",
                config=ResearchConfig(enabled=True, mode="explicit"),
                planner=OutagePlanner(),
                store=store,
                run_id="outage",
            )
            assert result["status"] == "terminal"
            assert any(event.get("terminal") is True for event in store.events("outage"))
            with research_lane(
                "overlap", enabled=True, mode="explicit", purpose="explicit", needs_model=True
            ) as lease:
                with foreground_lane(preempt_grace_ms=0) as receipt:
                    assert receipt.research_overlap is True
                    assert lease.cancelled() is True

        checks.append(_check("ollama-outage-and-sync-overlap", "uv run pytest -q tests/test_research_scheduler.py", outage_terminal_and_sync_overlap))

        def checkpoint_protection() -> None:
            store = ResearchStore(root=root / "research")
            store.checkpoints = root / "checkpoints"
            active = store.checkpoint("active", {}, active=True, durable_receipt=False)
            unreceipted = store.checkpoint("unreceipted", {}, active=False, durable_receipt=False)
            result = store.gc_checkpoints(ttl_seconds=0, max_total_bytes=0)
            assert active.exists() and unreceipted.exists()
            assert len(result["protected"]) == 2

        checks.append(_check("checkpoint-gc-protection", "uv run pytest -q tests/test_research_store.py", checkpoint_protection))

    passed = sum(row["status"] == "PASS" for row in checks)
    return {
        "schema_version": 1,
        "status": "PASS" if passed == len(checks) else "FAIL",
        "mutation_scope": "temporary-directory-only",
        "passed": passed,
        "failed": len(checks) - passed,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the research lane without mutating the Wiki")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_verification()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Research verification: {report['status']} ({report['passed']} passed, {report['failed']} failed)")
        for row in report["checks"]:
            print(f"- {row['status']} {row['name']}: {row['command']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
