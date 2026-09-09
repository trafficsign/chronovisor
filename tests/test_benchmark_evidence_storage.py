"""P4 runner tests exercise the published context, not candidate objects."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from chronovisor.core.recall_context import render_recall_payload

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_evidence_storage.py"
SPEC = importlib.util.spec_from_file_location("evidence_storage_benchmark", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)

SOURCE = (
    b"---\n"
    b"title: Page One\nstatus: stable\ntype: note\nuid: uid-1\n---\n"
    b"required evidence with a \"quoted\" value\n"
    b"unrelated proper evidence\nforbidden evidence\n"
)
OTHER_SOURCE = b"---\ntitle: Other\nstatus: stable\ntype: note\nuid: uid-2\n---\nother page evidence\n"
COPY_SOURCE = b"---\ntitle: Copy\nstatus: stable\ntype: note\nuid: uid-3\n---\nrequired evidence with a \"quoted\" value\n"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _span(source: bytes, text: bytes) -> tuple[int, int]:
    start = source.index(text)
    return start, start + len(text)


def _corpus(tmp_path: Path) -> tuple[Path, dict[str, dict[str, object]]]:
    base = tmp_path / "frozen"
    root = base / "root" / "pages"
    root.mkdir(parents=True)
    (root / "one.md").write_bytes(SOURCE)
    (root / "other.md").write_bytes(OTHER_SOURCE)
    (root / "copy.md").write_bytes(COPY_SOURCE)
    rows = [
        {"page_id": "page-1", "path": "pages/one.md", "content_sha256": _sha(SOURCE), "content_byte_length": len(SOURCE), "page_uid": "uid-1", "status": "stable"},
        {"page_id": "page-other", "path": "pages/other.md", "content_sha256": _sha(OTHER_SOURCE), "content_byte_length": len(OTHER_SOURCE), "page_uid": "uid-2", "status": "stable"},
        {"page_id": "page-copy", "path": "pages/copy.md", "content_sha256": _sha(COPY_SOURCE), "content_byte_length": len(COPY_SOURCE), "page_uid": "uid-3", "status": "stable"},
    ]
    root_identity = benchmark.canonical_sha256([
        {"page_id": row["page_id"], "path": row["path"], "content_sha256": row["content_sha256"], "status": row["status"], "error": None}
        for row in rows
    ])
    manifest = {"schema": "chronovisor.evidence-source-freeze.v1", "entries": rows, "source_root_sha256": root_identity}
    manifest_path = base / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    snapshot = {"schema": "chronovisor.evidence-index-input-freeze.v1", "source_root_sha256": root_identity, "source_manifest_sha256": _sha(manifest_path.read_bytes())}
    (base / "index-input-snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    return base, {str(row["page_id"]): row for row in rows}


def _payload(corpus: Path, rows: dict[str, dict[str, object]], count: int = 120) -> dict:
    required = b'required evidence with a "quoted" value'
    forbidden = b"forbidden evidence"
    required_start, required_end = _span(SOURCE, required)
    forbidden_start, forbidden_end = _span(SOURCE, forbidden)
    source_root = json.loads((corpus / "source-manifest.json").read_text())["source_root_sha256"]
    required_span = {"page_id": "page-1", "page_uid": "uid-1", "content_sha256": rows["page-1"]["content_sha256"], "byte_start": required_start, "byte_end": required_end, "excerpt": required.decode(), "excerpt_sha256": _sha(required)}
    forbidden_span = {"page_id": "page-1", "page_uid": "uid-1", "content_sha256": rows["page-1"]["content_sha256"], "byte_start": forbidden_start, "byte_end": forbidden_end, "excerpt_sha256": _sha(forbidden)}
    return {
        "schema_version": 3,
        "artifact_kind": "independent-answer-benchmark-manifest",
        "entries": [
            {
                "case_id": f"case-{index}", "prompt": f"question {index}", "split": "holdout", "evidence_chunks": [required_span],
                "source_span": {"source_root_sha256": source_root, "index_snapshot_sha256": "b" * 64, "required_spans": [required_span], "forbidden_spans": [forbidden_span], "language": ("ja", "en", "cross")[index % 3]},
                "source_authority_sha256": "c" * 64,
            }
            for index in range(count)
        ],
    }


def _protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int = 120) -> tuple[Path, dict, benchmark.FrozenCorpus]:
    corpus_dir, rows = _corpus(tmp_path)
    payload = _payload(corpus_dir, rows, count)
    monkeypatch.setattr(benchmark, "_load_gold", lambda path, chronovisor_root=None: payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    path = tmp_path / "protocol.json"
    benchmark.preregister_protocol(manifest_path=manifest, output_path=path, arm_codes={"A": "d652252", "B": "winning", "C": "section-v1"}, config_identity={"profile": "frozen"}, seed=19, minimum_samples=count, frozen_corpus_dir=corpus_dir, now="2026-01-01T00:00:00Z")
    return path, payload, benchmark.FrozenCorpus.load(corpus_dir)


def _context(items: list[dict]) -> str:
    return render_recall_payload({"trace": {}, "decision": "read", "items": items}, 3000)


def _item(page_id: str, evidence: str, row: dict[str, object], *, source_ref: bool = True) -> dict:
    item = {"page_id": page_id, "title": page_id, "evidence": evidence}
    if source_ref:
        source = {"page-1": SOURCE, "page-other": OTHER_SOURCE, "page-copy": COPY_SOURCE}[page_id]
        start, end = _span(source, evidence.encode())
        item["source_ref"] = {"doc_id": page_id, "uid": row["page_uid"], "sha256": row["content_sha256"], "generation_id": "frozen", "byte_start": start, "byte_end": end}
    return item


def test_digest_and_seeded_order_are_stable() -> None:
    assert benchmark.canonical_sha256({"b": 2, "a": 1}) == benchmark.canonical_sha256({"a": 1, "b": 2})
    assert benchmark.pair_order("case-1", 19) == benchmark.pair_order("case-1", 19)
    assert set(benchmark.pair_order("case-1", 19)) == set(benchmark.ARMS)


def test_preregister_binds_frozen_manifest_and_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol, _payload_value, _corpus_value = _protocol(tmp_path, monkeypatch)
    loaded = benchmark.load_protocol(protocol)
    assert loaded["frozen"]["source_manifest_sha256"]
    manifest = Path(loaded["frozen"]["manifest_path"])
    manifest.write_text("changed", encoding="utf-8")
    with pytest.raises(benchmark.BenchmarkHeld, match="manifest_changed"):
        benchmark.load_protocol(protocol)


def test_p4_protocol_uses_only_the_fixed_holdout_slices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus_dir, rows = _corpus(tmp_path)
    payload = _payload(corpus_dir, rows)
    payload["entries"].extend([{**payload["entries"][0], "case_id": "dev", "split": "train"}, {**payload["entries"][0], "case_id": "locked", "split": "locked-test"}])
    monkeypatch.setattr(benchmark, "_load_gold", lambda path, chronovisor_root=None: payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    protocol = benchmark.preregister_protocol(manifest_path=manifest, output_path=tmp_path / "protocol.json", arm_codes={"A": "d652252", "B": "winning", "C": "section-v1"}, config_identity={"profile": "frozen"}, frozen_corpus_dir=corpus_dir)
    assert protocol["dataset"]["case_count"] == 120
    assert protocol["dataset"]["split"] == "holdout"


def test_subprocess_adapter_binds_checkout_module_and_private_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkouts = {arm: tmp_path / f"checkout-{arm}" for arm in benchmark.ARMS}
    roots = {arm: tmp_path / f"root-{arm}" for arm in benchmark.ARMS}
    sockets = {arm: tmp_path / f"socket-{arm}" for arm in benchmark.ARMS}
    commits = {arm: f"{arm.lower()}" * 40 for arm in benchmark.ARMS}
    for arm in benchmark.ARMS:
        module = checkouts[arm] / "src/chronovisor/recall/recall_runtime.py"
        module.parent.mkdir(parents=True)
        module.write_text(f"# {arm}\n", encoding="utf-8")
        roots[arm].mkdir()
        roots[arm].joinpath("config.toml").write_text(f'[search.embedding.service]\nsocket = "{sockets[arm]}"\n', encoding="utf-8")

    def fake_run(command: tuple[str, ...], **kwargs: object) -> CompletedProcess[str]:
        if command[0] == "git":
            arm = next(arm for arm, checkout in checkouts.items() if kwargs["cwd"] == checkout)
            return CompletedProcess(command, 0, stdout=commits[arm], stderr="")
        arm = next(arm for arm, checkout in checkouts.items() if kwargs["cwd"] == checkout)
        module = checkouts[arm] / "src/chronovisor/recall/recall_runtime.py"
        assert json.loads(str(kwargs["input"])) == {"case_id": "case", "prompt": "prompt"}
        assert kwargs["env"]["PYTHONPATH"] == str(checkouts[arm] / "src")
        return CompletedProcess(command, 0, stdout=json.dumps({"code_binding": {"commit": commits[arm], "module": str(module), "module_sha256": _sha(module.read_bytes())}}), stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    adapters = benchmark.make_subprocess_adapters({arm: ("runner",) for arm in benchmark.ARMS}, checkouts, roots, arm_codes=commits, semantic_sockets=sockets)
    assert adapters["A"]({"case_id": "case", "prompt": "prompt", "evidence_chunks": ["not sent"]})["code_binding"]["commit"] == commits["A"]


def test_final_renderer_drop_is_zero_coverage_not_candidate_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _protocol_path, payload, corpus = _protocol(tmp_path, monkeypatch)
    rows = {page.page_id: {"page_uid": page.page_uid, "content_sha256": page.content_sha256} for page in corpus.pages.values()}
    first = _item("page-other", "other page evidence", rows["page-other"])
    required = _item("page-1", 'required evidence with a "quoted" value', rows["page-1"])
    first_only = _context([first])
    rendered = render_recall_payload({"trace": {}, "decision": "read", "items": [first, required]}, len(first_only))
    checked = benchmark.validate_context(payload["entries"][0], {"rendered_context": rendered, "timing": {"queue_ms": 1, "service_ms": 2, "recall_wall_ms": 3, "warm": True}}, budget_chars=3000, corpus=corpus)
    assert checked["source_consistent"] is True
    assert checked["required_coverage"]["full"] is False


def test_escaped_final_source_ref_and_non_gold_page_are_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _protocol_path, payload, corpus = _protocol(tmp_path, monkeypatch)
    rows = {page.page_id: {"page_uid": page.page_uid, "content_sha256": page.content_sha256} for page in corpus.pages.values()}
    rendered = _context([_item("page-1", 'required evidence with a "quoted" value', rows["page-1"]), _item("page-other", "other page evidence", rows["page-other"])])
    assert r'\"quoted\"' in rendered
    checked = benchmark.validate_context(payload["entries"][0], {"rendered_context": rendered, "timing": {"queue_ms": 1, "service_ms": 2, "recall_wall_ms": 3, "warm": True}}, budget_chars=3000, corpus=corpus)
    assert checked["status"] == "verified"
    assert checked["required_coverage"]["full"] is True


def test_legacy_a_excerpt_is_checked_against_frozen_text_without_fabricating_a_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _protocol_path, payload, corpus = _protocol(tmp_path, monkeypatch)
    rows = {page.page_id: {"page_uid": page.page_uid, "content_sha256": page.content_sha256} for page in corpus.pages.values()}
    legacy = _item("page-1", 'required evidence with a "quoted" value', rows["page-1"], source_ref=False)
    checked = benchmark.validate_context(payload["entries"][0], {"rendered_context": _context([legacy]), "timing": {"queue_ms": 1, "service_ms": 2, "recall_wall_ms": 3, "warm": True}}, budget_chars=3000, corpus=corpus)
    assert checked["status"] == "verified"
    assert checked["source_ref_available"] is False
    assert checked["required_coverage"]["full"] is True
    partial_text = 'required evidence with a "quoted" value'[:10]
    partial = _item("page-1", partial_text, rows["page-1"], source_ref=False)
    partial_checked = benchmark.validate_context(payload["entries"][0], {"rendered_context": _context([partial]), "timing": {"queue_ms": 1, "service_ms": 2, "recall_wall_ms": 3, "warm": True}}, budget_chars=3000, corpus=corpus)
    assert partial_checked["required_coverage"]["full"] is False


def test_same_required_text_from_a_different_page_does_not_cover_the_gold_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _protocol_path, payload, corpus = _protocol(tmp_path, monkeypatch)
    rows = {page.page_id: {"page_uid": page.page_uid, "content_sha256": page.content_sha256} for page in corpus.pages.values()}
    copied = _item("page-copy", 'required evidence with a "quoted" value', rows["page-copy"])
    checked = benchmark.validate_context(payload["entries"][0], {"rendered_context": _context([copied]), "timing": {"queue_ms": 1, "service_ms": 2, "recall_wall_ms": 3, "warm": True}}, budget_chars=3000, corpus=corpus)
    assert checked["status"] == "verified"
    assert checked["required_coverage"]["full"] is False


def test_required_range_needs_full_interval_union_not_one_overlap_byte(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _protocol_path, payload, corpus = _protocol(tmp_path, monkeypatch)
    required = payload["entries"][0]["source_span"]["required_spans"][0]
    start, end = required["byte_start"], required["byte_end"]
    midpoint = start + (end - start) // 2

    def ranged_item(left: int, right: int) -> dict:
        evidence = SOURCE[left:right].decode()
        return {
            "page_id": "page-1", "title": "page-1", "evidence": evidence,
            "source_ref": {"doc_id": "page-1", "uid": "uid-1", "sha256": _sha(SOURCE), "generation_id": "frozen", "byte_start": left, "byte_end": right},
        }

    timing = {"queue_ms": 1, "service_ms": 2, "recall_wall_ms": 3, "warm": True}
    one_byte = benchmark.validate_context(payload["entries"][0], {"rendered_context": _context([ranged_item(start, start + 1)]), "timing": timing}, budget_chars=3000, corpus=corpus)
    half = benchmark.validate_context(payload["entries"][0], {"rendered_context": _context([ranged_item(start, midpoint)]), "timing": timing}, budget_chars=3000, corpus=corpus)
    whole = benchmark.validate_context(payload["entries"][0], {"rendered_context": _context([ranged_item(start, midpoint), ranged_item(midpoint, end)]), "timing": timing}, budget_chars=3000, corpus=corpus)
    assert one_byte["required_coverage"]["full"] is False
    assert half["required_coverage"]["full"] is False
    assert whole["required_coverage"]["full"] is True


def test_normal_empty_recall_has_zero_coverage_without_claiming_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _protocol_path, payload, corpus = _protocol(tmp_path, monkeypatch)
    checked = benchmark.validate_context(payload["entries"][0], {"decision": "none", "timing": {"queue_ms": 1, "service_ms": 2, "recall_wall_ms": 3, "warm": True}}, budget_chars=3000, corpus=corpus)
    assert checked["status"] == "empty"
    assert checked["source_consistent"] is True
    assert checked["required_coverage"]["rate"] == 0.0


def test_bad_source_ref_and_forbidden_source_hold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _protocol_path, payload, corpus = _protocol(tmp_path, monkeypatch)
    rows = {page.page_id: {"page_uid": page.page_uid, "content_sha256": page.content_sha256} for page in corpus.pages.values()}
    bad = _item("page-1", 'required evidence with a "quoted" value', rows["page-1"])
    bad["source_ref"]["sha256"] = "0" * 64
    invalid = benchmark.validate_context(payload["entries"][0], {"rendered_context": _context([bad]), "timing": {"queue_ms": 1, "service_ms": 2, "recall_wall_ms": 3, "warm": True}}, budget_chars=3000, corpus=corpus)
    assert invalid["status"] == "unknown"
    forbidden = _item("page-1", "forbidden evidence", rows["page-1"])
    checked = benchmark.validate_context(payload["entries"][0], {"rendered_context": _context([forbidden]), "timing": {"queue_ms": 1, "service_ms": 2, "recall_wall_ms": 3, "warm": True}}, budget_chars=3000, corpus=corpus)
    assert checked["forbidden_hit_count"] == 1


def test_paired_runner_proves_a_b_coverage_difference_and_holds_without_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol, _payload_value, corpus = _protocol(tmp_path, monkeypatch)
    rows = {page.page_id: {"page_uid": page.page_uid, "content_sha256": page.content_sha256} for page in corpus.pages.values()}

    def adapter(entry: dict, *, arm: str) -> dict:
        item = _item("page-other", "other page evidence", rows["page-other"]) if arm == "A" else _item("page-1", 'required evidence with a "quoted" value', rows["page-1"])
        return {"rendered_context": _context([item]), "timing": {"queue_ms": 1, "service_ms": 2, "recall_wall_ms": 3, "warm": True}}

    report = benchmark.run_paired_evaluation(protocol, adapters={arm: (lambda entry, arm=arm: adapter(entry, arm=arm)) for arm in benchmark.ARMS}, now="2026-01-02T00:00:00Z", allow_in_process=True)
    assert report["status"] == "held"
    assert "missing_answer_scores" in report["reason"]
    assert report["paired"]["coverage_B_vs_A"]["point"] == 1.0


def test_c_failure_does_not_block_b_adoption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol, _payload_value, corpus = _protocol(tmp_path, monkeypatch)
    rows = {page.page_id: {"page_uid": page.page_uid, "content_sha256": page.content_sha256} for page in corpus.pages.values()}

    def adapter(entry: dict, *, arm: str) -> dict:
        item = _item("page-other", "other page evidence", rows["page-other"]) if arm == "A" else _item("page-1", 'required evidence with a "quoted" value', rows["page-1"])
        return {"rendered_context": _context([item]), "timing": {"queue_ms": 1, "service_ms": 2, "recall_wall_ms": 3, "warm": True}}

    adapters = {arm: (lambda entry, arm=arm: adapter(entry, arm=arm)) for arm in benchmark.ARMS}
    for callback in adapters.values():
        callback._isolated_process = True

    def score(*, entry: dict, arm: str, value: object) -> dict:
        if arm == "C":
            raise RuntimeError("C scorer unavailable")
        return {"accuracy": 0.9}

    report = benchmark.run_paired_evaluation(protocol, adapters=adapters, answer_evaluator=score, now="2026-01-02T00:00:00Z")
    assert report["status"] == "passed"
    assert report["recommended_arm"] == "B"
    assert report["gates"]["B"]["passed"] is True
    assert report["gates"]["C"]["passed"] is False
