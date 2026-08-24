"""P0-A recall baseline artifact fixtures.

The anonymous baseline captured at b0484f3 is fixed in
``_handoff/2026-08-13_1446_recall-baseline.json`` so that P3 paired replay
can compare post-change behavior against a stable cohort definition without
recording prompt/query/page bodies.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from chronovisor.recall.recall_runtime import ContextItem

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_FILE = REPO_ROOT / "_handoff" / "2026-08-13_1446_recall-baseline.json"
PARITY_SCRIPT = REPO_ROOT / "scripts" / "recall_parity.py"
R0_SCRIPT = REPO_ROOT / "scripts" / "recall_r0_harness.py"


def _load_baseline() -> dict:
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))


def _load_parity():
    spec = importlib.util.spec_from_file_location("recall_parity", PARITY_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_r0():
    spec = importlib.util.spec_from_file_location("recall_r0_harness", R0_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log_row(parity, index: int, prompt: str) -> dict:
    return {
        "schema_version": 2,
        "event": "UserPromptSubmit",
        "status": "ok",
        "decision": "read",
        "prompt_hash": parity._stable_prompt_hash(prompt),
        "decision_id": f"decision-{index:04d}",
        "prompt_preview": prompt,
        "prompt_chars": len(prompt),
    }


def _identity(commit: str, tree: str, module: str) -> dict[str, str]:
    return {
        "source_commit": commit * 40,
        "source_tree": tree * 40,
        "runtime_module_sha256": module * 64,
    }


def _assert_parity_artifact_whitelist(value, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        allowed = {
            (): {"paired_replay"},
            ("paired_replay",): {
                "schema",
                "protocol",
                "protocol_controls_sha256",
                "runtime_identity",
                "input_snapshot",
                "candidate_count",
                "source_cohort_sha256",
                "input_cohort_sha256",
                "baseline_stage_latency_ms",
                "projections",
            },
            ("paired_replay", "baseline_stage_latency_ms"): {
                "teacher",
                "reranker",
                "context",
                "total",
                "field",
            },
            ("paired_replay", "runtime_identity"): {
                "source_commit",
                "source_tree",
                "runtime_module_sha256",
            },
            ("paired_replay", "input_snapshot"): {"sha256", "file_count"},
        }.get(path)
        if len(path) == 3 and path[:2] == (
            "paired_replay",
            "baseline_stage_latency_ms",
        ):
            allowed = {"count", "p50", "p95", "p99"}
        if path == ("paired_replay", "projections", "[]"):
            parity = _load_parity()
            allowed = set(parity.RECEIPT_FIELDS)
        assert allowed is not None
        assert set(value) == allowed
        for key, child in value.items():
            _assert_parity_artifact_whitelist(child, (*path, key))
    elif isinstance(value, list):
        for child in value:
            _assert_parity_artifact_whitelist(child, (*path, "[]"))
    else:
        assert isinstance(value, str | int)


def test_baseline_artifact_schema() -> None:
    data = _load_baseline()
    assert data["schema"] == "chronovisor.recall-baseline.v1"
    assert data["captured_at"]
    assert data["head_commit"]
    cohort = data["cohort_definition"]
    assert cohort["source"].endswith("recall-log.jsonl")
    assert cohort["eligible_event"] == "UserPromptSubmit"
    assert data["status_distribution"]["last400"]
    assert data["latency_ms"]
    paired = data["paired_replay"]
    assert paired["candidate_count"] == 100
    assert len(paired["projections"]) == 100
    assert paired["runtime_identity"]["source_commit"] == data["head_commit"]


def test_baseline_anonymized() -> None:
    raw = BASELINE_FILE.read_text(encoding="utf-8")
    # Bodies must never be recorded; only aggregate numbers and metadata.
    for forbidden in ("prompt_body", "query_body", "page_body"):
        assert forbidden not in raw
    data = _load_baseline()
    assert isinstance(data["status_distribution"], dict)
    assert isinstance(data["latency_ms"], dict)
    assert isinstance(data["injection"], dict)
    _assert_parity_artifact_whitelist({"paired_replay": data["paired_replay"]})


def test_baseline_known_gaps_cover_stage_timing() -> None:
    data = _load_baseline()
    gaps = " ".join(data["known_gaps"]).lower()
    assert "stage" in gaps
    assert "p0-b" in gaps


def test_stage_timings_ms_anonymous_schema() -> None:
    # P0-B adds an anonymous stage->ms map to recall log records. The map
    # carries stage names and integer millisecond values only - never text
    # bodies, page ids, or query strings.
    timings = {
        "prepare": 10,
        "cleanup": 12,
        "session_load": 5,
        "field": 338,
        "compiler": 2,
        "teacher": 248,
        "bm25_build": 120,
        "bm25_query": 80,
        "semantic": 60,
        "graph": 40,
        "verify": 30,
        "rewrite": 15,
        "reranker": 643,
        "context": 30,
        "finalize": 10,
    }
    assert all(isinstance(ms, int) and ms >= 0 for ms in timings.values())
    joined = "|".join(timings.keys())
    # stage names stay short identifiers, never raw text
    assert all(part.isidentifier() for part in joined.split("|"))


def test_parity_cohort_selects_sorted_100_complete_read_rows(tmp_path: Path) -> None:
    parity = _load_parity()
    rows = [
        _log_row(parity, index, f"private prompt {index}")
        for index in reversed(range(101))
    ]
    rows.extend(
        [
            {**rows[0], "status": "timeout", "prompt_hash": "bad-status"},
            {**rows[0], "prompt_chars": 999, "prompt_hash": "truncated"},
        ]
    )
    log_file = tmp_path / "recall-log.jsonl"
    log_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    selected = parity.select_cohort(log_file)

    assert len(selected) == 100
    assert (
        selected
        == sorted(
            parity.eligible_rows(log_file),
            key=lambda row: (row["prompt_hash"], row["decision_id"]),
        )[:100]
    )


def test_parity_capture_schema_is_receipts_only(tmp_path: Path) -> None:
    parity = _load_parity()
    secret = "RAW_PROMPT_QUERY_PAGE_CONTEXT_SECRET"
    rows = [
        {
            **_log_row(parity, index, f"{secret}-{index}"),
            "evidence_features": {"field_shadow": {"latency_ms": index}},
        }
        for index in range(100)
    ]
    log_file = tmp_path / "recall-log.jsonl"
    log_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def runner(row):
        return {
            "case_id": parity._case_id(row),
            "status": "ok",
            "decision": "read",
            "search_mode": "hybrid",
            **{
                field: parity._sha([field, row["prompt_preview"]])
                for field in parity.PROJECTION_FIELDS
            },
            "_stage_latency_ms": {
                "teacher": 10,
                "reranker": 20,
                "context": 30,
                "total": 60,
            },
        }

    artifact = parity.capture(log_file, runner=runner)
    encoded = json.dumps(artifact, ensure_ascii=False)
    paired = artifact["paired_replay"]
    _assert_parity_artifact_whitelist(artifact)
    assert set(artifact) == {"paired_replay"}
    assert paired["schema"] == parity.SCHEMA
    assert paired["protocol"] == parity.PROTOCOL
    assert paired["candidate_count"] == 100
    assert len(paired["projections"]) == 100
    assert secret not in encoded
    assert set(paired["projections"][0]) == set(parity.RECEIPT_FIELDS)
    for receipt in paired["projections"]:
        assert set(receipt) == set(parity.RECEIPT_FIELDS)
        assert all(isinstance(value, str) for value in receipt.values())
        for field in parity.PROJECTION_FIELDS:
            assert len(receipt[field]) == 64
            assert set(receipt[field]) <= set("0123456789abcdef")
    assert paired["baseline_stage_latency_ms"]["teacher"] == {
        "count": 100,
        "p50": 10,
        "p95": 10,
        "p99": 10,
    }
    assert paired["baseline_stage_latency_ms"]["field"] == {
        "count": 100,
        "p50": 49,
        "p95": 94,
        "p99": 98,
    }

    assert set(paired) == {
        "schema",
        "protocol",
        "protocol_controls_sha256",
        "runtime_identity",
        "input_snapshot",
        "candidate_count",
        "source_cohort_sha256",
        "input_cohort_sha256",
        "baseline_stage_latency_ms",
        "projections",
    }
    assert set(paired["baseline_stage_latency_ms"]) == {
        "teacher",
        "reranker",
        "context",
        "total",
        "field",
    }
    for aggregate in paired["baseline_stage_latency_ms"].values():
        assert set(aggregate) == {"count", "p50", "p95", "p99"}


def test_parity_comparator_rejects_projection_mutation(tmp_path: Path) -> None:
    parity = _load_parity()
    projections = [
        {
            "case_id": parity._sha(["case", index]),
            "input_sha256": parity._sha(["input", index]),
            "status": "ok",
            "decision": "read",
            "search_mode": "hybrid",
            **{
                field: parity._sha([field, index]) for field in parity.PROJECTION_FIELDS
            },
        }
        for index in range(100)
    ]
    baseline = {
        "paired_replay": {
            "schema": parity.SCHEMA,
            "protocol": parity.PROTOCOL,
            "protocol_controls_sha256": parity._protocol_controls_sha256(),
            "runtime_identity": _identity("a", "b", "c"),
            "input_snapshot": {"sha256": "d" * 64, "file_count": 1},
            "candidate_count": 100,
            "source_cohort_sha256": parity._sha(
                [projection["case_id"] for projection in projections]
            ),
            "input_cohort_sha256": parity._sha(
                [projection["input_sha256"] for projection in projections]
            ),
            "baseline_stage_latency_ms": {},
            "projections": projections,
        }
    }
    candidate = json.loads(json.dumps(baseline))
    candidate["paired_replay"]["runtime_identity"] = _identity("e", "f", "1")
    assert parity.comparison_errors(baseline, candidate) == []
    candidate["paired_replay"]["input_snapshot"] = {
        "sha256": "a" * 64,
        "file_count": 1,
    }
    assert parity.comparison_errors(baseline, candidate) == ["input snapshot mismatch"]
    candidate = json.loads(json.dumps(baseline))
    candidate["paired_replay"]["runtime_identity"] = _identity("e", "f", "1")
    candidate["paired_replay"]["projections"][42]["rendered_context_sha256"] = "b" * 64
    assert parity.comparison_errors(baseline, candidate) == [
        "projection 42 rendered_context_sha256 mismatch"
    ]


def test_parity_baseline_fixes_first_100_successful_replays(tmp_path: Path) -> None:
    parity = _load_parity()
    rows = [_log_row(parity, index, f"private-{index}") for index in range(102)]
    log_file = tmp_path / "recall-log.jsonl"
    log_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def runner(row):
        if row["decision_id"] == "decision-0000":
            raise RuntimeError("private failure")
        status = "degraded" if row["decision_id"] == "decision-0001" else "ok"
        return {
            "case_id": parity._case_id(row),
            "status": status,
            "decision": "read",
            "search_mode": "hybrid",
            **{
                field: parity._sha([field, row["prompt_hash"]])
                for field in parity.PROJECTION_FIELDS
            },
        }

    artifact = parity.capture(log_file, runner=runner)
    projections = artifact["paired_replay"]["projections"]

    assert len(projections) == 100
    expected = [
        row
        for row in sorted(
            rows, key=lambda row: (row["prompt_hash"], row["decision_id"])
        )
        if row["decision_id"] not in {"decision-0000", "decision-0001"}
    ][:100]
    assert [row["case_id"] for row in projections] == [
        parity._case_id(row) for row in expected
    ]
    assert all(
        row["status"] == "ok" and row["decision"] == "read" for row in projections
    )


def test_parity_candidate_preserves_failed_receipt(tmp_path: Path) -> None:
    parity = _load_parity()
    rows = [_log_row(parity, index, f"private-{index}") for index in range(100)]
    log_file = tmp_path / "recall-log.jsonl"
    log_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def successful(row):
        return {
            "case_id": parity._case_id(row),
            "status": "ok",
            "decision": "read",
            "search_mode": "hybrid",
            **{
                field: parity._sha([field, row["prompt_hash"]])
                for field in parity.PROJECTION_FIELDS
            },
        }

    baseline = parity.capture(log_file, runner=successful)
    baseline["paired_replay"]["runtime_identity"] = _identity("a", "b", "c")
    failed_case = baseline["paired_replay"]["projections"][42]["case_id"]

    def candidate(row):
        if parity._case_id(row) == failed_case:
            raise RuntimeError("private failure")
        return successful(row)

    replay = parity.capture(log_file, cases=baseline, runner=candidate)
    replay["paired_replay"]["runtime_identity"] = _identity("d", "e", "f")

    assert replay["paired_replay"]["projections"][42]["status"] == "error"
    assert (
        parity.comparison_errors(baseline, replay)[0] == "projection 42 status mismatch"
    )


def test_parity_context_receipt_covers_every_public_context_field() -> None:
    parity = _load_parity()
    base = ContextItem(
        page_id="raw-page-id",
        uid="uid-safe",
        title="Private title",
        updated="2026-08-13",
        score=0.75,
        snippets=["Private snippet"],
        sensitivity="normal",
        certificate_id="certificate",
        evidence_kind="rich",
        source_line=42,
    )
    baseline = parity._context_hash([base], lambda _page_id: "")
    mutations = (
        replace(base, page_id="other-raw-page-id"),
        replace(base, uid="uid-other"),
        replace(base, title="Other title"),
        replace(base, updated="2026-08-12"),
        replace(base, score=0.5),
        replace(base, snippets=["Other snippet"]),
        replace(base, sensitivity="high"),
        replace(base, certificate_id="other-certificate"),
        replace(base, evidence_kind="legacy"),
        replace(base, source_line=43),
    )

    assert all(
        parity._context_hash([mutation], lambda _page_id: "") != baseline
        for mutation in mutations
    )


def test_parity_runtime_identity_rejects_another_source_root(
    tmp_path: Path, monkeypatch
) -> None:
    parity = _load_parity()
    source_root = tmp_path / "source"
    module = source_root / "src/chronovisor/recall/recall_runtime.py"
    module.parent.mkdir(parents=True)
    module.write_text("# sealed test module\n", encoding="utf-8")
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "Recall Test"],
        ["git", "config", "user.email", "recall@example.invalid"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "fixture"],
    ):
        parity.subprocess.run(command, cwd=source_root, check=True)
    head = parity.subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    from chronovisor.recall import recall_runtime

    monkeypatch.setattr(recall_runtime, "__file__", str(module))
    identity = parity._runtime_identity(source_root, head)

    assert identity["source_commit"] == head
    assert len(identity["source_tree"]) == 40
    assert len(identity["runtime_module_sha256"]) == 64
    try:
        parity._runtime_identity(tmp_path / "other", head)
    except ValueError as exc:
        assert "outside --source-root" in str(exc)
    else:
        raise AssertionError("runtime imports from another source tree must fail")


def test_parity_restores_read_only_environment(monkeypatch) -> None:
    parity = _load_parity()
    monkeypatch.setenv("CHRONOVISOR_READ_ONLY", "before")
    monkeypatch.setattr(parity, "_run_production", lambda _row: {"status": "ok"})

    assert parity.run_production({}) == {"status": "ok"}
    assert parity.os.environ["CHRONOVISOR_READ_ONLY"] == "before"


def test_r0_harness_metrics_and_fail_closed(monkeypatch) -> None:
    harness = _load_r0()
    samples = iter(
        (
            {
                "rusage_uuid": "a",
                "resident_bytes": 10,
                "footprint_bytes": 20,
                "disk_read_bytes": 30,
                "disk_write_bytes": 40,
            },
            {
                "rusage_uuid": "a",
                "resident_bytes": 11,
                "footprint_bytes": 25,
                "disk_read_bytes": 33,
                "disk_write_bytes": 44,
            },
        )
    )
    monkeypatch.setattr(harness, "_proc_pid_rusage_v2", lambda: next(samples))
    metrics = {}
    assert harness._measure_stage(
        harness.STAGES[0], lambda: "ok", metrics
    ) == "ok"
    metric = metrics[harness.STAGES[0]][0]
    assert metric["footprint_before_bytes"] == 20
    assert metric["footprint_after_bytes"] == 25
    assert metric["disk_read_bytes"] == 3
    assert metric["disk_write_bytes"] == 4
    assert "rss" not in metric
    assert "peak_rss" not in metric

    incomplete = {name: [] for name in harness.STAGES}
    incomplete.pop(harness.STAGES[-1])
    with pytest.raises(ValueError, match="missing stages"):
        harness._require_complete_stages(incomplete)

    class RemoteTeacher:
        role = "recall.distill.teacher.a"
        local = False

    with pytest.raises(ValueError, match="provider/OX attempt"):
        harness._assert_local_workers(
            {role: RemoteTeacher() for role in (
                "recall.distill.teacher.a",
                "recall.distill.teacher.b",
                "recall.distill.teacher.c",
            )},
            type("LocalCounterfactual", (), {"local": True})(),
        )
    with pytest.raises(ValueError, match="runtime identity drift"):
        harness._assert_identity_stable({"source_commit": "a"}, {"source_commit": "b"})
