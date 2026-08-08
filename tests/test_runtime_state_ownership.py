from __future__ import annotations

import copy
import fcntl
import json
import os
import plistlib
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from chronovisor.ops import golden_expand
from chronovisor.recall import claims, evidence_certificate
from chronovisor.search import search_eval, semantic_index
from scripts.runtime_ownership import gate as ownership_gate
from scripts.runtime_ownership import registry as ownership_registry
from scripts.runtime_ownership.discovery import _launchd_invocations, discover
from scripts.runtime_ownership.gate import (
    _registry_structure_errors,
    _resource_contract_drift,
    _resource_validation_errors,
    runtime_state_fitness,
)
from scripts.runtime_ownership.model import (
    BASELINE_PATH,
    LOCK_PROTECTS_LOCATORS,
    REGISTRY_PATH,
    _json_document_bytes,
    _same_json_value,
)
from scripts.runtime_ownership.registry import _registry_payload
from scripts.runtime_ownership.seed import (
    _base_resources,
    _id_sets,
    _load_json,
    _load_previous_baseline,
    _seed_ids,
    _seed_state_violations,
    _seed_structure_errors,
    build_runtime_state_baseline,
)
from scripts.runtime_ownership.source import _ast_discovery, _snapshot_current

ROOT = Path(__file__).parents[1]


def _inventory() -> tuple[dict[str, bytes], object, dict, list[dict]]:
    snapshot = _snapshot_current(ROOT)
    index, detection = discover(snapshot)
    return snapshot, index, detection, _base_resources(detection)


def _resource(resources: list[dict], kind: str, locator: str) -> dict:
    return next(
        row
        for row in resources
        if row["kind"] == kind and row["locator"]["value"] == locator
    )


def test_ownership_documents_are_canonical_and_structurally_exact() -> None:
    baseline_path = ROOT / BASELINE_PATH
    registry_path = ROOT / REGISTRY_PATH
    baseline = _load_json(baseline_path)
    registry = _load_json(registry_path)

    assert baseline_path.read_bytes() == _json_document_bytes(baseline)
    assert registry_path.read_bytes() == _json_document_bytes(registry)
    assert _seed_structure_errors(baseline) == []
    assert _registry_structure_errors(registry) == []


@pytest.mark.parametrize(
    "raw",
    [
        b'{"key": 1, "key": 2}\n',
        b'{"key": NaN}\n',
        b'{"key":1}\n',
        b"[]\n",
    ],
)
def test_ownership_document_loader_rejects_ambiguous_json(
    tmp_path: Path, raw: bytes
) -> None:
    path = tmp_path / "ownership.json"
    path.write_bytes(raw)
    assert "load_error" in _load_json(path)


def test_ownership_document_structure_rejects_extra_or_reordered_keys() -> None:
    baseline = _load_json(ROOT / BASELINE_PATH)
    malformed_baseline = copy.deepcopy(baseline)
    malformed_baseline["unexpected"] = True
    assert "root_keys" in _seed_structure_errors(malformed_baseline)
    wrong_schema_type = copy.deepcopy(baseline)
    wrong_schema_type["schema_version"] = True
    assert "schema_version" in _seed_structure_errors(wrong_schema_type)
    assert not _same_json_value(True, 1)
    reordered_bucket = copy.deepcopy(baseline)
    active = reordered_bucket["discovery_ids"].pop("active")
    reordered_bucket["discovery_ids"]["active"] = active
    assert "discovery_ids.keys" in _seed_structure_errors(reordered_bucket)

    registry = _load_json(ROOT / REGISTRY_PATH)
    malformed_registry = copy.deepcopy(registry)
    malformed_registry["unexpected"] = True
    assert "root_keys" in _registry_structure_errors(malformed_registry)
    wrong_registry_schema_type = copy.deepcopy(registry)
    wrong_registry_schema_type["schema_version"] = True
    assert "schema_version" in _registry_structure_errors(wrong_registry_schema_type)
    changed_policy = copy.deepcopy(registry)
    changed_policy["policy"]["new_resources"] = "allow"
    assert "policy.values" in _registry_structure_errors(changed_policy)
    reordered_row = copy.deepcopy(registry)
    evidence = reordered_row["resources"][0].pop("evidence")
    reordered_row["resources"][0]["evidence"] = evidence
    assert "resources[0].keys" in _registry_structure_errors(reordered_row)


def test_generated_worker_contract_drift_includes_launchd_argv() -> None:
    _snapshot, _index, _detection, resources = _inventory()
    registered = copy.deepcopy(resources)
    librarian = _resource(
        registered,
        "worker",
        "com.trafficsign.chronovisor-librarian-review",
    )
    librarian["worker"]["invocations"][0]["argv"].append("--unreviewed")
    assert _resource_contract_drift(resources, registered) == {
        librarian["id"]: ["worker"]
    }


def test_generated_owner_and_discovery_grouping_drift_fail_closed() -> None:
    _snapshot, _index, _detection, resources = _inventory()
    registered = copy.deepcopy(resources)
    target = registered[0]
    target["owner_symbol"] = "chronovisor.invalid:owner"
    target["discovery_ids"] = [*target["discovery_ids"], "runtime-site:" + "0" * 64]
    assert _resource_contract_drift(resources, registered)[target["id"]] == [
        "owner_symbol",
        "discovery_ids",
    ]


def test_generated_kind_and_locator_contract_drift_fail_closed() -> None:
    _snapshot, _index, _detection, resources = _inventory()
    changed_kind = copy.deepcopy(resources)
    changed_kind[0]["kind"] = "queue"
    assert _resource_contract_drift(resources, changed_kind)[changed_kind[0]["id"]] == [
        "kind"
    ]

    changed_locator = copy.deepcopy(resources)
    changed_locator[0]["locator"]["type"] = "socket"
    assert _resource_contract_drift(resources, changed_locator)[
        changed_locator[0]["id"]
    ] == ["locator"]


def test_invalid_kind_locator_pair_fails_full_fitness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, index, _detection, resources = _inventory()
    registry = _load_json(ROOT / REGISTRY_PATH)
    target = next(row for row in registry["resources"] if row["kind"] == "artifact")
    target["locator"]["type"] = "socket"

    errors = _resource_validation_errors(index, snapshot, registry["resources"])
    assert any(
        row["id"] == target["id"] and "locator type is invalid" in row["error"]
        for row in errors
    )
    original_load = ownership_gate._load_json

    def load_with_invalid_locator(path: Path) -> dict:
        if path == ROOT / REGISTRY_PATH:
            return registry
        return original_load(path)

    monkeypatch.setattr(ownership_gate, "_load_json", load_with_invalid_locator)
    fitness = ownership_gate.runtime_state_fitness(ROOT)
    assert fitness["passed"] is False
    assert fitness["violations"]["resource_contract_drift"][target["id"]] == [
        "locator"
    ]
    assert any(
        row["id"] == target["id"]
        and "locator type is invalid" in row["error"]
        for row in fitness["violations"]["resource_validation_errors"]
    )


def test_existing_registry_cannot_self_authorize_an_owner() -> None:
    _snapshot, _index, detection, resources = _inventory()
    generated = resources[0]
    existing = _load_json(ROOT / REGISTRY_PATH)
    edited = copy.deepcopy(existing)
    registered = next(row for row in edited["resources"] if row["id"] == generated["id"])
    registered["owner_symbol"] = "chronovisor.invalid:self_authorized"
    registered["writers"] = [registered["owner_symbol"]]
    edited["exclusions"][0]["reason"] = "self_authorized"

    payload = _registry_payload(
        detection,
        _load_json(ROOT / BASELINE_PATH),
        edited,
    )
    rebuilt = next(row for row in payload["resources"] if row["id"] == generated["id"])
    assert rebuilt["owner_symbol"] == generated["owner_symbol"]
    assert rebuilt["writers"] == generated["writers"]
    assert payload["exclusions"][0]["reason"] != "self_authorized"


def test_cli_check_emits_exact_canonical_fitness_document() -> None:
    expected = runtime_state_fitness(ROOT)
    assert expected["passed"] is True
    violations = expected["violations"]
    assert all(not value for value in violations.values())
    completed = subprocess.run(
        [sys.executable, "scripts/runtime_state_ownership.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == (0 if expected["passed"] else 1)
    assert completed.stderr == b""
    assert completed.stdout == _json_document_bytes(expected)


def test_quality_workflow_runs_exact_runtime_ownership_check() -> None:
    workflow = (ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    command = "uv run --frozen python scripts/runtime_state_ownership.py --check"
    assert workflow.count(command) == 1
    assert workflow.count("fetch-depth: 0") == 1


def test_frozen_inventory_matches_seed_and_current_drift_is_exact() -> None:
    baseline = build_runtime_state_baseline(ROOT)
    stored_baseline = _load_json(ROOT / BASELINE_PATH)
    snapshot, index, detection, resources = _inventory()
    current = _id_sets(detection)

    assert stored_baseline == baseline
    for field in ("discovery_ids", "resource_ids"):
        assert current[field] == _seed_ids(baseline, field, "active")

    counts = baseline["counts"]["active"]
    assert counts["resources"] == 454
    assert counts["discoveries"] == 733
    assert counts["exclusions"] == 138
    assert counts["by_kind"] == {
        "artifact": 182,
        "lock": 21,
        "queue": 5,
        "schema": 154,
        "socket": 6,
        "worker": 86,
    }
    assert counts["entrypoint_workers"] == 51
    assert counts["launchd_workers"] == 7
    assert counts["lock_protocol_sites"] == 103
    assert counts["direct_flock_acquisitions"] == 51
    assert counts["direct_flock_modules"] == 38
    assert counts["direct_flock_functions"] == 50

    suffix_rows = _ast_discovery(index)
    assert len(suffix_rows) == 507
    assert (
        sum(
            row["symbol"] == "SCHEMA"
            or row["symbol"].endswith("_SCHEMA")
            or row["symbol"] == "SCHEMA_VERSION"
            or row["symbol"].endswith("_SCHEMA_VERSION")
            for row in suffix_rows
        )
        == 244
    )
    assert len(resources) == counts["resources"]
    assert snapshot


def test_previous_seed_history_survives_delete_and_unrelated_commit(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "history"
    repository.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=repository,
            capture_output=True,
            check=True,
        )

    git("init", "-q")
    git("config", "user.name", "Runtime Ownership Test")
    git("config", "user.email", "runtime-ownership@example.invalid")
    (repository / "README.md").write_text("history\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-q", "-m", "initial")

    active = _load_json(ROOT / BASELINE_PATH)
    baseline_path = repository / BASELINE_PATH
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_bytes(_json_document_bytes(active))
    assert _load_previous_baseline(repository) == {
        "latest": {"absent": True},
        "historical_retired": {
            "discovery_ids": [],
            "resource_ids": [],
        },
        "history_errors": [],
    }

    retired = copy.deepcopy(active)
    retired_id = retired["discovery_ids"]["active"].pop(0)
    retired["discovery_ids"]["retired"] = [retired_id]
    retired["counts"]["active"]["discoveries"] -= 1
    retired["counts"]["retired"]["discovery_ids"] = 1
    baseline_path.write_bytes(_json_document_bytes(retired))
    git("add", BASELINE_PATH.as_posix())
    git("commit", "-q", "-m", "retire runtime site")

    baseline_path.unlink()
    git("add", "-A", BASELINE_PATH.as_posix())
    git("commit", "-q", "-m", "delete baseline")
    (repository / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
    git("add", "unrelated.txt")
    git("commit", "-q", "-m", "unrelated change")

    baseline_path.write_bytes(_json_document_bytes(active))
    previous = _load_previous_baseline(repository)
    assert previous["latest"] == retired
    assert previous["historical_retired"]["discovery_ids"] == [retired_id]
    assert previous["history_errors"] == []

    _snapshot, _index, detection, _resources = _inventory()
    violations = _seed_state_violations(detection, active, active, previous)
    assert violations["seed_active_growth"]["discovery_ids"] == [retired_id]
    assert violations["seed_retired_regressions"]["discovery_ids"] == [retired_id]

    git("add", BASELINE_PATH.as_posix())
    git("commit", "-q", "-m", "reintroduce runtime site")
    (repository / "after-readd.txt").write_text("unrelated\n", encoding="utf-8")
    git("add", "after-readd.txt")
    git("commit", "-q", "-m", "unrelated after readd")

    committed_readd_history = _load_previous_baseline(repository)
    assert committed_readd_history["latest"] == active
    assert committed_readd_history["historical_retired"]["discovery_ids"] == [
        retired_id
    ]
    committed_violations = _seed_state_violations(
        detection, active, active, committed_readd_history
    )
    assert committed_violations["seed_active_growth"] == {}
    assert committed_violations["seed_retired_regressions"]["discovery_ids"] == [
        retired_id
    ]


def test_previous_seed_history_unions_retired_ids_from_merge_parents(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "merge-history"
    repository.mkdir()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
        )
        return completed.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Runtime Ownership Test")
    git("config", "user.email", "runtime-ownership@example.invalid")
    active = _load_json(ROOT / BASELINE_PATH)
    baseline_path = repository / BASELINE_PATH
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_bytes(_json_document_bytes(active))
    git("add", BASELINE_PATH.as_posix())
    git("commit", "-q", "-m", "active baseline")
    main_branch = git("branch", "--show-current")
    git("branch", "retired-side")

    git("checkout", "-q", "retired-side")
    retired = copy.deepcopy(active)
    retired_id = retired["discovery_ids"]["active"].pop(0)
    retired["discovery_ids"]["retired"] = [retired_id]
    retired["counts"]["active"]["discoveries"] -= 1
    retired["counts"]["retired"]["discovery_ids"] = 1
    baseline_path.write_bytes(_json_document_bytes(retired))
    git("add", BASELINE_PATH.as_posix())
    git("commit", "-q", "-m", "retire runtime site on side branch")

    git("checkout", "-q", main_branch)
    git("commit", "-q", "--allow-empty", "-m", "main keeps site active")
    git(
        "merge",
        "-q",
        "--no-ff",
        "-s",
        "ours",
        "retired-side",
        "-m",
        "merge with active resolution",
    )

    history = _load_previous_baseline(repository)
    assert history["latest"] == active
    assert history["historical_retired"]["discovery_ids"] == [retired_id]
    assert history["history_errors"] == []
    _snapshot, _index, detection, _resources = _inventory()
    violations = _seed_state_violations(detection, active, active, history)
    assert violations["seed_active_growth"] == {}
    assert violations["seed_retired_regressions"]["discovery_ids"] == [retired_id]


def test_previous_seed_history_survives_path_rename_and_readd(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "rename-history"
    repository.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=repository,
            capture_output=True,
            check=True,
        )

    git("init", "-q")
    git("config", "user.name", "Runtime Ownership Test")
    git("config", "user.email", "runtime-ownership@example.invalid")
    active = _load_json(ROOT / BASELINE_PATH)
    retired = copy.deepcopy(active)
    retired_id = retired["discovery_ids"]["active"].pop(0)
    retired["discovery_ids"]["retired"] = [retired_id]
    retired["counts"]["active"]["discoveries"] -= 1
    retired["counts"]["retired"]["discovery_ids"] = 1
    baseline_path = repository / BASELINE_PATH
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_bytes(_json_document_bytes(retired))
    git("add", BASELINE_PATH.as_posix())
    git("commit", "-q", "-m", "retired baseline")

    archived = baseline_path.with_name("runtime-state-baseline.archived.json")
    git("mv", BASELINE_PATH.as_posix(), archived.relative_to(repository).as_posix())
    git("commit", "-q", "-m", "archive baseline path")
    baseline_path.write_bytes(_json_document_bytes(active))
    git("add", BASELINE_PATH.as_posix())
    git("commit", "-q", "-m", "readd active baseline")
    (repository / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
    git("add", "unrelated.txt")
    git("commit", "-q", "-m", "unrelated after readd")

    history = _load_previous_baseline(repository)
    assert history["latest"] == active
    assert history["historical_retired"]["discovery_ids"] == [retired_id]
    assert history["history_errors"] == []
    _snapshot, _index, detection, _resources = _inventory()
    violations = _seed_state_violations(detection, active, active, history)
    assert violations["seed_retired_regressions"]["discovery_ids"] == [retired_id]


@pytest.mark.parametrize(
    "malformation",
    [
        "invalid-json",
        "noncanonical",
        "schema",
        "counts-shape",
        "counts-bool",
        "counts-999",
    ],
)
def test_invalid_historical_seed_fails_closed(
    tmp_path: Path,
    malformation: str,
) -> None:
    repository = tmp_path / malformation
    repository.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=repository,
            capture_output=True,
            check=True,
        )

    git("init", "-q")
    git("config", "user.name", "Runtime Ownership Test")
    git("config", "user.email", "runtime-ownership@example.invalid")
    active = _load_json(ROOT / BASELINE_PATH)
    baseline_path = repository / BASELINE_PATH
    baseline_path.parent.mkdir(parents=True)
    if malformation == "invalid-json":
        baseline_path.write_bytes(b"{invalid\n")
    elif malformation == "noncanonical":
        baseline_path.write_text(json.dumps(active), encoding="utf-8")
    elif malformation == "schema":
        drifted = copy.deepcopy(active)
        drifted["schema_version"] = 2
        baseline_path.write_bytes(_json_document_bytes(drifted))
    else:
        drifted = copy.deepcopy(active)
        if malformation == "counts-shape":
            drifted["counts"]["active"]["unexpected"] = 0
        elif malformation == "counts-bool":
            drifted["counts"]["active"]["direct_flock_modules"] = True
        else:
            drifted["counts"]["active"]["resources"] = 999
        baseline_path.write_bytes(_json_document_bytes(drifted))
    git("add", BASELINE_PATH.as_posix())
    git("commit", "-q", "-m", "invalid historical baseline")
    baseline_path.unlink()
    git("add", "-A", BASELINE_PATH.as_posix())
    git("commit", "-q", "-m", "delete invalid baseline")
    baseline_path.write_bytes(_json_document_bytes(active))

    history = _load_previous_baseline(repository)
    assert history["history_errors"]
    _snapshot, _index, detection, _resources = _inventory()
    violations = _seed_state_violations(detection, active, active, history)
    assert violations["previous_seed_history_errors"] == history["history_errors"]


def test_from_import_submodule_aliases_resolve_runtime_paths() -> None:
    _snapshot, _index, detection, _resources = _inventory()
    by_symbol = {
        (row["module"], row["symbol"]): row for row in detection["resource_candidates"]
    }
    assert (
        by_symbol[("chronovisor.ingest.read_back_repair", "FAILURE_FILE")]["locator"][
            "value"
        ]
        == "$CHRONOVISOR_ROOT/runtime/ingest-read-back-failures.jsonl"
    )
    assert (
        by_symbol[("chronovisor.ingest.read_back_repair", "LEDGER_FILE")]["locator"][
            "value"
        ]
        == "$CHRONOVISOR_ROOT/runtime/ingest-read-back-repair.json"
    )
    assert (
        by_symbol[("chronovisor.recall.recall_hints", "QUERY_HINTS_FILE")]["locator"][
            "value"
        ]
        == "$CHRONOVISOR_ROOT/recall/query-hints.json"
    )


def test_reviewed_owners_formats_and_transaction_contracts_are_exact() -> None:
    snapshot, index, _detection, resources = _inventory()
    assert _resource_validation_errors(index, snapshot, resources) == []

    promotion = _resource(
        resources,
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/recall-field/promotion.json",
    )
    assert promotion["owner_symbol"] == (
        "chronovisor.recall.recall_growth:PROMOTION_ARTIFACT"
    )
    assert promotion["format"]["version"] == 4

    locked = _resource(
        resources,
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/search-eval/recall-field-locked-e2e.json",
    )
    assert (
        locked["owner_symbol"] == "chronovisor.search.search_eval:LOCKED_E2E_ARTIFACT"
    )
    assert locked["format"]["version"] == 2

    raw_queue = _resource(
        resources,
        "queue",
        "$CHRONOVISOR_ROOT/review/raw-replay-queue.jsonl",
    )
    assert raw_queue["format"]["version"] == 2
    assert raw_queue["coordination"]["lock_id"]

    semantic = _resource(
        resources, "queue", "$CHRONOVISOR_ROOT/runtime/semantic-jobs.sqlite"
    )
    assert semantic["format"]["status"] == "unversioned"
    assert "version" not in semantic["format"]
    assert semantic["coordination"] == {
        "protocol": "sqlite-wal-transactions",
        "transaction": "BEGIN IMMEDIATE",
    }

    generations = _resource(
        resources,
        "artifact",
        "$CHRONOVISOR_ROOT/.index/semantic/generations",
    )
    assert generations["format"]["version"] == 3
    for locator, owner in (
        (
            "chronovisor.classification-disabled-baseline.v1",
            "chronovisor.lab.classification_fixture_set:DISABLED_BASELINE_SCHEMA",
        ),
        (
            "chronovisor.classification-inference-dto.v1",
            "chronovisor.lab.classification_fixture_set:INFERENCE_DTO_SCHEMA",
        ),
    ):
        schema = _resource(resources, "schema", locator)
        assert schema["owner_symbol"] == owner
        assert schema["writers"] == [owner]
        assert schema["readers"] == [owner]
    assert all(
        row["format"]["status"] in {"versioned", "unversioned"}
        and not (
            row["format"]["status"] == "unversioned"
            and ("version" in row["format"] or "schema_id" in row["format"])
        )
        for row in resources
        if row["kind"] in {"artifact", "queue"}
    )


def test_uncoordinated_multiwriter_fails_but_sqlite_transactions_pass() -> None:
    snapshot, index, _detection, resources = _inventory()
    assert _resource_validation_errors(index, snapshot, resources) == []

    broken = copy.deepcopy(resources)
    semantic = _resource(
        broken, "queue", "$CHRONOVISOR_ROOT/runtime/semantic-jobs.sqlite"
    )
    semantic["coordination"] = {"protocol": "atomic-replace"}
    errors = _resource_validation_errors(index, snapshot, broken)
    assert any("multiple writers require" in row["error"] for row in errors)

    broken = copy.deepcopy(resources)
    broken[0]["readers"].append(broken[0]["readers"][0])
    errors = _resource_validation_errors(index, snapshot, broken)
    assert any("readers must not contain duplicates" in row["error"] for row in errors)

    broken = copy.deepcopy(resources)
    labels = _resource(
        broken, "queue", "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl"
    )
    labels["coordination"].pop("lock_id")
    errors = _resource_validation_errors(index, snapshot, broken)
    assert any("multiple writers require" in row["error"] for row in errors)


def test_locks_have_reviewed_scopes_and_no_certificate_phantom() -> None:
    _snapshot, _index, _detection, resources = _inventory()
    locks = [row for row in resources if row["kind"] == "lock"]
    assert {row["locator"]["value"] for row in locks} == set(LOCK_PROTECTS_LOCATORS)
    assert all(
        row["scope"] in {"artifact_sidecar", "worker_lease", "global_protocol"}
        for row in locks
    )
    assert all(row["protects"] for row in locks)
    assert not any(
        row["locator"]["value"]
        == "$CHRONOVISOR_ROOT/recall/evidence-certificate-ledger.lock"
        for row in locks
    )
    assert _resource(
        resources,
        "lock",
        "$CHRONOVISOR_ROOT/recall/evidence-certificate-ledger.jsonl.lock",
    )["protects"] == [
        _resource(
            resources,
            "artifact",
            "$CHRONOVISOR_ROOT/recall/evidence-certificate-ledger.jsonl",
        )["id"]
    ]


def test_removing_writer_lock_calls_breaks_frozen_discovery_ceiling() -> None:
    baseline = build_runtime_state_baseline(ROOT)
    snapshot = _snapshot_current(ROOT)
    replacements = {
        "src/chronovisor/ops/golden_expand.py": (
            b"_search_label_queue_lock(candidate_file)",
            b"_missing_queue_lock(candidate_file)",
        ),
        "src/chronovisor/search/search_eval.py": (
            b"_search_label_queue_lock(output_file)",
            b"_missing_queue_lock(output_file)",
        ),
    }
    for path, (before, after) in replacements.items():
        assert before in snapshot[path]
        snapshot[path] = snapshot[path].replace(before, after, 1)

    _index, changed = discover(snapshot)
    active = _seed_ids(baseline, "discovery_ids", "active")
    assert _id_sets(changed)["discovery_ids"] != active

    search_eval_path = "src/chronovisor/search/search_eval.py"
    snapshot[search_eval_path] = snapshot[search_eval_path].replace(
        b"_search_label_queue_lock(queue_file)",
        b"_missing_queue_lock(queue_file)",
        1,
    )
    _index, changed = discover(snapshot)
    assert not any(
        row["kind"] == "lock"
        and row["locator"]["value"]
        == "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl.lock"
        for row in changed["resource_candidates"]
    )


def test_launchd_invocations_preserve_wrapper_roles_and_links() -> None:
    _snapshot, _index, _detection, resources = _inventory()
    librarian = _resource(
        resources,
        "worker",
        "com.trafficsign.chronovisor-librarian-review",
    )
    invocations = librarian["worker"]["invocations"]
    assert invocations == [
        {
            "entrypoint": "chronovisor-librarian",
            "argv": [
                "--root",
                "$CHRONOVISOR_ROOT",
                "--full-sweep",
                "--json",
            ],
            "role": "full-sweep",
            "evidence": {
                "path": "scripts/chronovisor-librarian-review",
                "line": 11,
            },
        },
        {
            "entrypoint": "chronovisor-librarian",
            "argv": [
                "--root",
                "$CHRONOVISOR_ROOT",
                "--review-collection-queue",
                "--limit",
                "${CHRONOVISOR_LIBRARIAN_REVIEW_LIMIT:-5}",
                "--review-model",
                "${CHRONOVISOR_LIBRARIAN_REVIEW_MODEL:-gemma4:26b}",
                "--review-role",
                "primary",
                "--json",
            ],
            "role": "collection-primary",
            "evidence": {
                "path": "scripts/chronovisor-librarian-review",
                "line": 20,
            },
        },
        {
            "entrypoint": "chronovisor-librarian",
            "argv": [
                "--root",
                "$CHRONOVISOR_ROOT",
                "--review-collection-queue",
                "--limit",
                "${CHRONOVISOR_LIBRARIAN_CHALLENGE_LIMIT:-5}",
                "--review-model",
                "${CHRONOVISOR_LIBRARIAN_CHALLENGER_MODEL:-gpt-oss:20b}",
                "--review-role",
                "challenger",
                "--json",
            ],
            "role": "collection-challenger",
            "evidence": {
                "path": "scripts/chronovisor-librarian-review",
                "line": 31,
            },
        },
    ]

    evidence = _resource(
        resources,
        "worker",
        "com.trafficsign.chronovisor-library-evidence",
    )["worker"]["invocations"]
    assert evidence == [
        {
            "entrypoint": "chronovisor-lab",
            "argv": [
                "classification-library-pilot",
                "run-once",
                "--repo-root",
                "/Users/trafficsign/projects/personal/chronovisor",
            ],
            "role": "classification-library-pilot",
            "runtime": {
                "executable": "uvx",
                "resolution": "PATH",
                "search_path": (
                    "/Users/trafficsign/.local/bin:/opt/homebrew/bin:/usr/local/bin:"
                    "/usr/bin:/bin:/opt/homebrew/sbin:/usr/sbin:/sbin"
                ),
                "source": (
                    "${CHRONOVISOR_RUNTIME_SOURCE:-"
                    "git+ssh://git@github.com/trafficsign/chronovisor}"
                ),
                "evidence": [
                    {
                        "path": "scripts/chronovisor-library-evidence",
                        "line": 5,
                    },
                    {
                        "path": "scripts/chronovisor-library-evidence",
                        "line": 6,
                    },
                    {
                        "path": "scripts/chronovisor-library-evidence",
                        "line": 9,
                    },
                ],
            },
            "linked_worker": {
                "kind": "worker",
                "locator_type": "lab_dispatch",
                "locator_value": "classification-library-pilot",
            },
            "evidence": {
                "path": "scripts/chronovisor-library-evidence",
                "line": 10,
            },
        }
    ]
    dashboard = _resource(resources, "worker", "com.trafficsign.chronovisor-dashboard")
    assert dashboard["worker"]["invocations"][0]["argv"] == [
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
        "--lan",
    ]


@pytest.mark.parametrize(
    ("label", "mutation"),
    [
        ("com.trafficsign.chronovisor-librarian-review", "executable"),
        ("com.trafficsign.chronovisor-librarian-review", "extra"),
        ("com.trafficsign.chronovisor-library-evidence", "executable"),
        ("com.trafficsign.chronovisor-library-evidence", "forwarded"),
        ("com.trafficsign.chronovisor-library-evidence", "extra"),
    ],
)
def test_special_launchd_program_arguments_fail_full_gate_and_generation(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    mutation: str,
) -> None:
    snapshot = _snapshot_current(ROOT)
    plist_path = f"launchd/{label}.plist"
    payload = plistlib.loads(snapshot[plist_path])
    arguments = payload["ProgramArguments"]
    if mutation == "executable":
        arguments[0] = f"/tmp/{Path(arguments[0]).name}"
    elif mutation == "forwarded":
        arguments[1] = "attacker-run"
    else:
        arguments.append("--attacker")
    changed_snapshot = dict(snapshot)
    changed_snapshot[plist_path] = plistlib.dumps(payload, sort_keys=False)

    with pytest.raises(ValueError, match="ProgramArguments drifted"):
        discover(changed_snapshot)
    monkeypatch.setattr(
        ownership_gate, "_snapshot_current", lambda _root: changed_snapshot
    )
    with pytest.raises(ValueError, match="ProgramArguments drifted"):
        ownership_gate.runtime_state_fitness(ROOT)
    monkeypatch.setattr(
        ownership_registry, "_snapshot_current", lambda _root: changed_snapshot
    )
    with pytest.raises(ValueError, match="ProgramArguments drifted"):
        ownership_registry.build_runtime_state_registry(ROOT)


@pytest.mark.parametrize(
    "mutation",
    [
        "delete-primary",
        "duplicate-primary",
        "unknown-role",
        "limit",
        "model",
        "extra-argv",
        "redirection",
    ],
)
def test_librarian_wrapper_command_drift_fails_closed(mutation: str) -> None:
    wrapper = (ROOT / "scripts/chronovisor-librarian-review").read_text(
        encoding="utf-8"
    )
    if mutation in {"delete-primary", "duplicate-primary"}:
        lines = wrapper.splitlines(keepends=True)
        assert "--review-role primary" in "".join(lines[16:26])
        primary = "".join(lines[16:26])
        wrapper = (
            "".join([*lines[:16], *lines[26:]])
            if mutation == "delete-primary"
            else f"{wrapper}\n{primary}"
        )
    else:
        replacements = {
            "unknown-role": ("--review-role primary", "--review-role observer"),
            "limit": (
                "${CHRONOVISOR_LIBRARIAN_REVIEW_LIMIT:-5}",
                "${CHRONOVISOR_LIBRARIAN_REVIEW_LIMIT:-6}",
            ),
            "model": (
                "${CHRONOVISOR_LIBRARIAN_REVIEW_MODEL:-gemma4:26b}",
                "${CHRONOVISOR_LIBRARIAN_REVIEW_MODEL:-gemma4:27b}",
            ),
            "extra-argv": (
                "  --review-role primary \\\n",
                "  --review-role primary \\\n  --unexpected \\\n",
            ),
            "redirection": (">/dev/null", "2>/dev/null"),
        }
        before, after = replacements[mutation]
        assert wrapper.count(before) == 1
        wrapper = wrapper.replace(before, after, 1)

    with pytest.raises(ValueError):
        _launchd_invocations(
            label="com.trafficsign.chronovisor-librarian-review",
            wrapper=wrapper,
            arguments=[
                "/Users/trafficsign/projects/personal/chronovisor/"
                "scripts/chronovisor-librarian-review"
            ],
            entrypoints={},
        )


@pytest.mark.parametrize(
    ("label", "wrapper_name", "arguments"),
    [
        (
            "com.trafficsign.chronovisor-librarian-review",
            "chronovisor-librarian-review",
            [
                "/Users/trafficsign/projects/personal/chronovisor/"
                "scripts/chronovisor-librarian-review"
            ],
        ),
        (
            "com.trafficsign.chronovisor-library-evidence",
            "chronovisor-library-evidence",
            [
                "/Users/trafficsign/projects/personal/chronovisor/"
                "scripts/chronovisor-library-evidence",
                "run-once",
                "--repo-root",
                "/Users/trafficsign/projects/personal/chronovisor",
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("--python /opt/homebrew/bin/python3.14 ", ""),
        ("--python /opt/homebrew/bin/python3.14", "--python 3.14"),
    ],
)
def test_source_backed_wrappers_require_exact_standard_python(
    label: str,
    wrapper_name: str,
    arguments: list[str],
    before: str,
    after: str,
) -> None:
    wrapper = (ROOT / "scripts" / wrapper_name).read_text(encoding="utf-8")
    assert before in wrapper
    wrapper = wrapper.replace(before, after, 1)
    with pytest.raises(ValueError):
        _launchd_invocations(
            label=label,
            wrapper=wrapper,
            arguments=arguments,
            entrypoints={},
        )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("chronovisor-lab", "chronovisor-dashboard"),
        ("classification-library-pilot", "classification-library-pilot-v2"),
        ('"$@"', '--unexpected "$@"'),
    ],
)
def test_library_evidence_wrapper_dispatch_drift_fails_closed(
    before: str, after: str
) -> None:
    wrapper = (ROOT / "scripts/chronovisor-library-evidence").read_text(
        encoding="utf-8"
    )
    assert wrapper.count(before) == 1
    wrapper = wrapper.replace(before, after, 1)
    with pytest.raises(ValueError):
        _launchd_invocations(
            label="com.trafficsign.chronovisor-library-evidence",
            wrapper=wrapper,
            arguments=[
                "/Users/trafficsign/projects/personal/chronovisor/"
                "scripts/chronovisor-library-evidence",
                "run-once",
                "--repo-root",
                "/Users/trafficsign/projects/personal/chronovisor",
            ],
            entrypoints={},
        )


@pytest.mark.parametrize(
    "mutation",
    ["runtime-source", "path", "duplicate-runtime-source", "duplicate-path"],
)
def test_library_evidence_runtime_resolution_drift_fails_closed(
    mutation: str,
) -> None:
    wrapper = (ROOT / "scripts/chronovisor-library-evidence").read_text(
        encoding="utf-8"
    )
    source = (
        'RUNTIME_SOURCE="${CHRONOVISOR_RUNTIME_SOURCE:-'
        'git+ssh://git@github.com/trafficsign/chronovisor}"'
    )
    path = (
        'PATH="/Users/trafficsign/.local/bin:/opt/homebrew/bin:/usr/local/bin:'
        '/usr/bin:/bin:/opt/homebrew/sbin:/usr/sbin:/sbin"'
    )
    if mutation == "runtime-source":
        wrapper = wrapper.replace(
            source,
            'RUNTIME_SOURCE="${CHRONOVISOR_RUNTIME_SOURCE:-https://attacker.invalid}"',
            1,
        )
    elif mutation == "path":
        wrapper = wrapper.replace(path, 'PATH="/tmp/attacker:/usr/bin:/bin"', 1)
    elif mutation == "duplicate-runtime-source":
        wrapper = f"{wrapper}\n{source}\n"
    else:
        wrapper = f"{wrapper}\n{path}\n"

    with pytest.raises(ValueError):
        _launchd_invocations(
            label="com.trafficsign.chronovisor-library-evidence",
            wrapper=wrapper,
            arguments=[
                "/Users/trafficsign/projects/personal/chronovisor/"
                "scripts/chronovisor-library-evidence",
                "run-once",
                "--repo-root",
                "/Users/trafficsign/projects/personal/chronovisor",
            ],
            entrypoints={},
        )


@pytest.mark.parametrize(
    "injected_command",
    [
        'uvx() { /tmp/attacker "$@"; }',
        "alias uvx=/tmp/attacker",
        "/tmp/attacker --prepare",
        "set -eu",
    ],
)
def test_library_evidence_rejects_commands_outside_exact_wrapper_grammar(
    injected_command: str,
) -> None:
    wrapper = (ROOT / "scripts/chronovisor-library-evidence").read_text(
        encoding="utf-8"
    )
    marker = "\nexec uvx"
    assert wrapper.count(marker) == 1
    wrapper = wrapper.replace(marker, f"\n{injected_command}\n\nexec uvx", 1)
    with pytest.raises(ValueError):
        _launchd_invocations(
            label="com.trafficsign.chronovisor-library-evidence",
            wrapper=wrapper,
            arguments=[
                "/Users/trafficsign/projects/personal/chronovisor/"
                "scripts/chronovisor-library-evidence",
                "run-once",
                "--repo-root",
                "/Users/trafficsign/projects/personal/chronovisor",
            ],
            entrypoints={},
        )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("#!/bin/sh", "#!/bin/bash"),
        ("#!/bin/sh", "#!/bin/zsh"),
        ("#!/bin/sh", "#!/usr/bin/env bash"),
        ("#!/bin/sh", "#!/bin/sh -i"),
        ('"$RUNTIME_SOURCE"', "$RUNTIME_SOURCE"),
        ('"$@"', "$@"),
    ],
)
def test_library_evidence_rejects_shebang_and_quote_drift(
    before: str,
    after: str,
) -> None:
    wrapper = (ROOT / "scripts/chronovisor-library-evidence").read_text(
        encoding="utf-8"
    )
    assert wrapper.count(before) == 1
    wrapper = wrapper.replace(before, after, 1)
    with pytest.raises(ValueError):
        _launchd_invocations(
            label="com.trafficsign.chronovisor-library-evidence",
            wrapper=wrapper,
            arguments=[
                "/Users/trafficsign/projects/personal/chronovisor/"
                "scripts/chronovisor-library-evidence",
                "run-once",
                "--repo-root",
                "/Users/trafficsign/projects/personal/chronovisor",
            ],
            entrypoints={},
        )


@pytest.mark.parametrize("field", ["search_path", "source"])
def test_library_evidence_runtime_contract_registry_drift(field: str) -> None:
    _snapshot, _index, _detection, resources = _inventory()
    registered = copy.deepcopy(resources)
    evidence = _resource(
        registered,
        "worker",
        "com.trafficsign.chronovisor-library-evidence",
    )
    evidence["worker"]["invocations"][0]["runtime"][field] = "attacker"
    assert _resource_contract_drift(resources, registered)[evidence["id"]] == [
        "worker"
    ]


def test_source_backed_invocation_line_drift_rejects_recorded_worker_contract() -> None:
    snapshot, _index, detection, resources = _inventory()
    wrapper_path = "scripts/chronovisor-librarian-review"
    changed_snapshot = dict(snapshot)
    changed_snapshot[wrapper_path] = b"\n" + snapshot[wrapper_path]
    _changed_index, changed_detection = discover(changed_snapshot)
    changed_resources = _base_resources(changed_detection)

    current = _resource(
        resources,
        "worker",
        "com.trafficsign.chronovisor-librarian-review",
    )
    changed = _resource(
        changed_resources,
        "worker",
        "com.trafficsign.chronovisor-librarian-review",
    )
    current_lines = [
        invocation["evidence"]["line"]
        for invocation in current["worker"]["invocations"]
    ]
    changed_lines = [
        invocation["evidence"]["line"]
        for invocation in changed["worker"]["invocations"]
    ]
    assert current_lines == [11, 20, 31]
    assert changed_lines == [line + 1 for line in current_lines]
    assert _resource_contract_drift(changed_resources, resources)[changed["id"]] == [
        "worker"
    ]


def test_endpoint_roles_and_sensitive_config_are_registered() -> None:
    _snapshot, _index, _detection, resources = _inventory()
    dashboard = _resource(resources, "socket", "tcp://0.0.0.0:8765")
    assert dashboard["socket"]["clients"] == [
        "chronovisor.ops.burn_monitor:dashboard_snapshot"
    ]
    mcp = _resource(resources, "socket", "stdio://chronovisor-mcp")
    assert mcp["socket"]["server"] not in mcp["socket"]["clients"]
    semantic = _resource(
        resources, "socket", "unix://$HOME/.chronovisor/runtime/semantic.sock"
    )
    assert semantic["socket"]["filesystem_mode"] == "0600"
    assert semantic["socket"]["startup"] == "unlink-stale-then-bind"

    for locator in (
        "$CHRONOVISOR_ROOT/runtime/dashboard-access-token",
        "$CHRONOVISOR_ROOT/runtime/dashboard-credentials.json",
        "$HOME/.chronovisor/runtime/searxng/secret",
    ):
        resource = _resource(resources, "artifact", locator)
        assert "secret" in resource["compatibility"]
    for locator in (
        "$HOME/.chronovisOR/runtime/searxng/settings.yml".replace("visOR", "visor"),
        "$HOME/.local/share/chronovisor/searxng/source",
        "$HOME/.local/share/chronovisor/searxng/.venv",
    ):
        _resource(resources, "artifact", locator)


def test_queue_helpers_share_one_physical_sidecar(tmp_path: Path) -> None:
    queue = tmp_path / "search-label-queue.jsonl"
    lock_path = queue.with_suffix(".jsonl.lock")
    with golden_expand._search_label_queue_lock(queue):
        descriptor = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
    first_inode = lock_path.stat().st_ino
    with search_eval._search_label_queue_lock(queue):
        assert lock_path.stat().st_ino == first_inode
    assert lock_path.stat().st_mode & 0o777 == 0o600


def test_golden_expand_rereads_queue_under_lock(tmp_path: Path, monkeypatch) -> None:
    queue = tmp_path / "search-label-queue.jsonl"
    golden = tmp_path / "search-golden.jsonl"
    candidate = {"query": "q", "expected_pages": ["p"], "negative": False}
    monkeypatch.setattr(
        golden_expand,
        "rows_from_recall_questions",
        lambda **_kwargs: [candidate],
    )
    real_lock = golden_expand._search_label_queue_lock

    @contextmanager
    def competing_writer(path: Path):
        path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
        with real_lock(path):
            yield

    monkeypatch.setattr(golden_expand, "_search_label_queue_lock", competing_writer)
    result = golden_expand.expand_golden_from_recall_questions(
        golden_file=golden,
        candidate_file=queue,
        write=True,
    )
    assert result["added"] == 0
    assert len(golden_expand._read_jsonl(queue)) == 1


def test_claim_writers_use_the_same_sidecar(tmp_path: Path, monkeypatch) -> None:
    ledger = tmp_path / "claims.jsonl"
    calls: list[Path] = []

    @contextmanager
    def recording_lock(path: Path):
        calls.append(path)
        yield

    monkeypatch.setattr(claims, "CLAIMS_FILE", ledger)
    monkeypatch.setattr(claims, "_claims_ledger_lock", recording_lock)
    monkeypatch.setattr(
        claims,
        "page_claims",
        lambda *_args, **_kwargs: [
            {
                "source_raw": "raw/x.json",
                "source_page": "page",
                "value": "durable fact",
            }
        ],
    )
    claims.append_page_claims(["page"], source_raw="raw/x.json")
    monkeypatch.setattr(claims, "find_page", lambda _page: tmp_path / "page.md")
    claims.sanitize_claim_ledger(path=ledger, write=True)
    assert calls == [ledger, ledger]


def test_constant_lock_paths_match_runtime_defaults() -> None:
    assert (
        Path(f"{evidence_certificate.CERTIFICATE_LEDGER}.lock")
        == evidence_certificate.CERTIFICATE_LEDGER_LOCK
    )
    assert semantic_index.ACTIVATION_LOCK == semantic_index.SEMANTIC_ROOT / (
        "activation.lock"
    )
