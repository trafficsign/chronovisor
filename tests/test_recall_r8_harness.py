from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_r8_harness_test", ROOT / "scripts" / "recall_r8_harness.py"
)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def _id(number: int) -> str:
    return f"{number:064x}"


def _sealed(value: dict[str, Any]) -> dict[str, Any]:
    value["seal_sha256"] = HARNESS._digest(value)
    return value


def _sealed_artifact(directory: Path, schema: str, payload: dict[str, Any]) -> Path:
    unsigned = {"schema": schema, "namespace": "recall-distillation", **payload}
    value = {"artifact_id": HARNESS._digest(unsigned), **unsigned}
    value["seal_sha256"] = HARNESS._digest(value)
    path = directory / f"{value['artifact_id']}.json"
    path.write_bytes(HARNESS._canonical(value) + b"\n")
    return path


def _r4(directory: Path, source_commit: str, *, passed: bool = True) -> Path:
    snapshot = {
        "commit": source_commit,
        "clean": True,
        "status_sha256": _id(20),
        "status_count": 0,
        "tree_sha256": _id(21),
        "file_count": 1,
        "symlink_count": 0,
        "ox_identity_sha256": _id(22),
        "account_uid": 1,
        "account_home": "/fixture",
    }
    return _sealed_artifact(
        directory,
        "chronovisor.recall-r4.v1",
        {
            "captured_at": "2026-08-25T00:00:00+00:00",
            "source": snapshot,
            "source_after": dict(snapshot),
            "source_final": dict(snapshot),
            "source_contract": {
                "schema": "chronovisor.recall-source-contract.v1",
                "passed": True,
                "local": {},
                "ox": {},
            },
            "production_certification": {
                "passed": passed,
                "reasons": [] if passed else ["fixture_false"],
                "collector": "fixture",
                "provider_calls": 0,
                "workset": {"sha256": _id(23)},
            },
            "receipt_files": {
                name: {"files": [], "count": 0}
                for name in ("local", "ox", "production")
            },
            "provider_calls": 0,
            "production_root_used": False,
        },
    )


def _r2(directory: Path, source_commit: str, *, passed: bool = True) -> Path:
    return _sealed_artifact(
        directory,
        "chronovisor.recall-r2.v1",
        {
            "runtime_identity": {"source_commit": source_commit},
            "full_rebuild_parity": {"passed": passed},
            "full_rebuild_parity_independent": {"passed": passed},
            "production_unchanged": passed,
            "cleanup": {"remaining": 0 if passed else 1},
        },
    )


class _ReadOnlyStore:
    def read_sealed(self, path: Path, *, schema: str | None = None) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["schema"] == schema
        unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
        assert value["seal_sha256"] == HARNESS._digest(unsigned)
        return cast(dict[str, Any], value)


class _FakeR3:
    R3_SCHEMA = "chronovisor.recall-r3.v1"

    @staticmethod
    def _assert_formal_acceptance(
        result: dict[str, Any], source: dict[str, Any], *, require_completion: bool
    ) -> None:
        return None


def _r3(directory: Path, source_commit: str) -> Path:
    return _sealed_artifact(
        directory,
        "chronovisor.recall-r3.v1",
        {
            "runtime": {"source_commit": source_commit},
            "source": {"source_commit": source_commit},
            "result": {
                "clone_workset": {"final_status": {"leased": 0}},
                "duplicates": 0,
                "production_workset_unchanged": True,
                "cleanup": {"remaining": 0},
            },
        },
    )


def _r3_completion(directory: Path, source_commit: str, main: Path) -> Path:
    main_value = json.loads(main.read_text(encoding="utf-8"))
    main_sha = HARNESS._hash_file(
        main, HARNESS._file_state(main, label="fixture"), label="fixture"
    )
    assert main_sha is not None
    return _sealed_artifact(
        directory,
        "chronovisor.recall-r3-completion.v1",
        {
            "main_artifact_id": main_value["artifact_id"],
            "main_artifact_sha256": main_sha,
            "sealed_artifact_id": main_value["artifact_id"],
            "sealed_artifact_sha256": main_sha,
            "source_commit": source_commit,
            "readback_verified": True,
        },
    )


def _r7(
    path: Path,
    *,
    source_commit: str = "a" * 40,
    certified: bool = True,
    complete: bool = True,
) -> Path:
    stages = [
        {
            "stage": name,
            "days": 7,
            "paired": 500,
            "certified": certified,
        }
        for name in ("shadow", "5", "25", "100")
    ]
    if not complete:
        stages = stages[:2]
    value: dict[str, Any] = {
        "schema": HARNESS.R7_SCHEMA,
        "namespace": "recall-distillation",
        "captured_at": "2026-08-25T00:00:00+00:00",
        "certification": certified,
        "synthetic_fixture": False,
        "source_before": {"source_commit": source_commit, "source_clean": "true"},
        "source_after": {"source_commit": source_commit, "source_clean": "true"},
        "stages": stages,
        "forced_failure": {
            "deterministic_failure": True,
            "rolled_back": True,
            "learning_halted": True,
            "rollout_percent": 0,
            "rollback_state": {"active_policy_id": _id(3), "lkg_policy_id": _id(3)},
        },
    }
    unsigned = dict(value)
    value["artifact_id"] = HARNESS._digest(unsigned)
    value["seal_sha256"] = HARNESS._digest(value)
    artifact = path / f"{value['artifact_id']}.json"
    artifact.write_bytes(HARNESS._canonical(value) + b"\n")
    return artifact


def _r7_live_receipt(
    directory: Path, source_commit: str, *, certification: bool = True
) -> tuple[Path, str]:
    live_dir = directory / "r7-live-attestations"
    live_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "artifact_id": _id(30),
        "file_sha256": _id(31),
        "seal_sha256": _id(32),
        "source_commit": source_commit,
    }
    payload: dict[str, Any] = {
        "schema": HARNESS.R7_LIVE_ATTESTATION_SCHEMA,
        "namespace": "recall-distillation",
        "kind": "r7-live-attestation",
        "source_commit": source_commit,
        "certification": certification,
        "source": {
            "source_commit": source_commit,
            "source_tree_sha256": _id(33),
            "source_clean": True,
        },
        "run": {"run_id": _id(34), "source_commit": source_commit},
    }
    payload.update({name: dict(identity) for name in (
        "collector", "rollback", "process", "archive", "direct_url",
        "health", "api", "dom", "stage100",
    )})
    artifact_id = HARNESS._digest(payload)
    value = {"artifact_id": artifact_id, **payload}
    value["seal_sha256"] = HARNESS._digest(value)
    path = live_dir / f"{artifact_id}.json"
    path.write_bytes(HARNESS._canonical(value) + b"\n")
    file_sha = HARNESS._hash_file(path, HARNESS._file_state(path, label="fixture"), label="fixture")
    assert file_sha is not None
    return path, file_sha


def _r7_structurally_valid(
    directory: Path, source_commit: str, *, live_certification: bool = True
) -> Path:
    live_path, live_sha = _r7_live_receipt(
        directory, source_commit, certification=live_certification
    )
    live_value = json.loads(live_path.read_text(encoding="utf-8"))
    live_id = live_path.stem
    live_seal = live_value["seal_sha256"]
    live_identity = {
        "live_attestation_artifact_id": live_id,
        "live_attestation_file_sha256": live_sha,
        "live_attestation_seal_sha256": live_seal,
        "live_attestation_source_commit": source_commit,
        "live_attestation_run_id": live_value["run"]["run_id"],
        "live_attestation_stage100_artifact_id": live_value["stage100"]["artifact_id"],
        "live_attestation_rollback_artifact_id": live_value["rollback"]["artifact_id"],
    }
    stages = [
        {
            "stage": name,
            "rollout_percent": percent,
            "days": 7,
            "paired": 500,
            "certified": True,
            "run_id": _id(40 + index),
            "stage_seal_sha256": _id(50 + index),
            "last_poll_seal_sha256": _id(60 + index),
        }
        for index, (name, percent) in enumerate(
            (("shadow", 0), ("5", 5), ("25", 25), ("100", 100))
        )
    ]
    stages[-1]["run_id"] = live_value["run"]["run_id"]
    forced = {
        "kind": "forced-failure-receipt",
        "stage": "100",
        "run_id": stages[-1]["run_id"],
        "source_commit": source_commit,
        "stage_seal_sha256": stages[-1]["stage_seal_sha256"],
        "last_poll_seal_sha256": stages[-1]["last_poll_seal_sha256"],
        "poll_artifact_id": _id(70),
        "process_artifact_id": _id(71),
        "archive_artifact_id": _id(72),
        "final_stage_artifact_id": _id(73),
        "deterministic_failure": True,
        "rolled_back": True,
        "learning_halted": True,
        "rollout_percent": 0,
        "rollback_state": {"active_policy_id": _id(3), "lkg_policy_id": _id(3)},
    }
    value: dict[str, Any] = {
        "schema": HARNESS.R7_SCHEMA,
        "namespace": "recall-distillation",
        "captured_at": "2026-08-25T00:00:00+00:00",
        "certification": True,
        "synthetic_fixture": False,
        "source_before": {"source_commit": source_commit, "source_clean": "true"},
        "source_after": {"source_commit": source_commit, "source_clean": "true"},
        "stage_matrix": [
            {"stage": name, "rollout_percent": percent}
            for name, percent in (("shadow", 0), ("5", 5), ("25", 25), ("100", 100))
        ],
        "stages": stages,
        "forced_failure": forced,
        "collector": {
            "artifact_id": live_value["collector"]["artifact_id"],
            "file_sha256": live_value["collector"]["file_sha256"],
            "seal_sha256": live_value["collector"]["seal_sha256"],
            "source_commit": source_commit,
            "run_id": live_value["run"]["run_id"],
            **live_identity,
        },
        **live_identity,
    }
    value["forced_failure"].update(
        {
            **live_identity,
            "rollback_artifact_id": live_value["rollback"]["artifact_id"],
            "rollback_file_sha256": live_value["rollback"]["file_sha256"],
            "rollback_seal_sha256": live_value["rollback"]["seal_sha256"],
        }
    )
    value["artifact_id"] = HARNESS._digest(value)
    value["seal_sha256"] = HARNESS._digest(value)
    artifact = directory / f"{value['artifact_id']}.json"
    artifact.write_bytes(HARNESS._canonical(value) + b"\n")
    return artifact


def _source(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=test",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source, commit


def _source_tree_digest(source: Path) -> str:
    indexed = subprocess.run(
        ["git", "ls-files", "-s", "-z"],
        cwd=source,
        check=True,
        capture_output=True,
    ).stdout
    tree: list[tuple[str, str, str]] = []
    for record in indexed.split(b"\0"):
        if not record:
            continue
        header, separator, path_bytes = record.partition(b"\t")
        fields = header.split()
        assert separator == b"\t" and len(fields) == 3
        tree.append((fields[0].decode(), fields[1].decode(), path_bytes.decode()))
    return str(HARNESS._digest(tree))


def _entry() -> dict[str, Any]:
    return {
        "present": True,
        "count": 1,
        "bytes": 1,
        "sha256": _id(9),
        "file_state": {
            "st_dev": 1,
            "st_ino": 1,
            "st_mode": 0o600,
            "st_uid": 1,
            "st_size": 1,
            "st_mtime_ns": 1,
            "st_ctime_ns": 1,
        },
        "bounded": True,
    }


def _observation(*, enabled: bool = False, leased: int = 0) -> dict[str, Any]:
    return {
        "ox": {
            "enabled": enabled,
            "provider_calls": 0,
            "leased": leased,
            "process_lock": False,
            **{name: _entry() for name in HARNESS._OX_FILES},
        },
        "pointers": {name: _entry() for name in HARNESS._POINTER_FILES},
        "legacy": {name: _entry() for name in HARNESS._LEGACY_FILES},
        "phase_receipts": {
            phase["name"]: _sealed(
                {
                    "schema": HARNESS.R8_PHASE_RECEIPT_SCHEMA,
                    "namespace": "recall-distillation",
                    "kind": "r8-phase-receipt",
                    "phase": phase["name"],
                    "status": "sealed",
                    "rollback_artifact": phase["rollback_artifact"],
                }
            )
            for phase in HARNESS.PHASES
        },
    }


def _phase_receipt_file(
    directory: Path,
    phase: dict[str, Any],
    source_commit: str,
    inventory_sha256: str,
    *,
    previous: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Produce one immutable phase receipt using the non-circular contract."""

    value: dict[str, Any] = {
        "schema": HARNESS.R8_PHASE_RECEIPT_SCHEMA,
        "namespace": "recall-distillation",
        "kind": "r8-phase-receipt",
        "phase": phase["name"],
        "order": phase["order"],
        "status": "sealed",
        "rollback_artifact": phase["rollback_artifact"],
        "source_commit": source_commit,
        "inventory_sha256": inventory_sha256,
        "previous_artifact_id": previous["artifact_id"] if previous else None,
        "previous_artifact_sha256": previous["file_sha256"] if previous else None,
        "official_validation": {
            "validator": "chronovisor.recall.r8.phase-validator.v1",
            "certified": True,
            "source_commit": source_commit,
            "inventory_sha256": inventory_sha256,
            "prerequisites": {
                prerequisite: True for prerequisite in phase["prerequisites"]
            },
        },
    }
    value["file_sha256"] = HARNESS._embedded_content_digest(
        value, label=f"fixture phase {phase['name']}"
    )
    value["artifact_id"] = HARNESS._digest(value)
    value["seal_sha256"] = HARNESS._digest(value)
    path = directory / f"{value['artifact_id']}.json"
    path.write_bytes(HARNESS._canonical(value) + b"\n")
    state = HARNESS._file_state(path, label="fixture phase")
    file_sha = HARNESS._hash_file(path, state, label="fixture phase")
    assert file_sha is not None
    return path, {
        "path": str(path),
        "artifact_id": value["artifact_id"],
        "file_sha256": file_sha,
        "seal_sha256": value["seal_sha256"],
        "schema": HARNESS.R8_PHASE_RECEIPT_SCHEMA,
    }


def _production_observation_file(
    directory: Path,
    production_root: Path,
    source_commit: str,
    phase_refs: dict[str, dict[str, Any]],
    source_tree_sha256: str,
) -> Path:
    actual = HARNESS._observe_directory(production_root)
    source_state = HARNESS._file_state(directory, label="fixture source")
    snapshots = {
        name: {
            "source_commit": source_commit,
            "source_tree_sha256": source_tree_sha256,
            "source_clean": True,
            "file_state": source_state,
        }
        for name in ("before", "after", "final")
    }
    root_state = HARNESS._file_state(production_root, label="fixture production root")
    value: dict[str, Any] = {
        "schema": HARNESS.R8_OBSERVATION_SCHEMA,
        "namespace": "recall-distillation",
        "kind": "r8-production-read-only-observation",
        "source_commit": source_commit,
        "source_before": snapshots["before"],
        "source_after": snapshots["after"],
        "source_final": snapshots["final"],
        "production": {
            "ox": actual["ox"],
            "pointers": actual["pointers"],
            "legacy": actual["legacy"],
            "locks": actual["locks"],
            "phase_receipts": phase_refs,
        },
        "production_root": str(production_root),
        "production_root_state": root_state,
        "production_before": actual,
        "production_after": actual,
        "production_final": actual,
    }
    value["file_sha256"] = HARNESS._embedded_content_digest(
        value, label="fixture production observation"
    )
    value["artifact_id"] = HARNESS._digest(value)
    value["seal_sha256"] = HARNESS._digest(value)
    path = directory / f"{value['artifact_id']}.json"
    path.write_bytes(HARNESS._canonical(value) + b"\n")
    return path


def test_missing_r7_publishes_false_sealed_artifact_and_nonzero(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    result = HARNESS.run(
        source_root=source,
        source_commit=commit,
        r7_artifact=tmp_path / "missing.json",
        output=tmp_path / "out",
        production_observation=None,
    )
    assert result["cleanup_authorized"] is False
    artifact = HARNESS.read_artifact(Path(result["path"]))
    assert artifact["cleanup_authorized"] is False
    assert list((tmp_path / "out").glob("*.json")) == [Path(result["path"])]


def test_resealed_blocked_report_cannot_claim_cleanup_authority(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    result = HARNESS.run(
        source_root=source,
        source_commit=commit,
        r7_artifact=tmp_path / "missing.json",
        output=tmp_path / "out",
        production_observation=None,
    )
    original = Path(result["path"])
    value = json.loads(original.read_text(encoding="utf-8"))
    value["cleanup_authorized"] = True
    unsigned = {key: item for key, item in value.items() if key not in {"artifact_id", "seal_sha256"}}
    value["artifact_id"] = HARNESS._digest(unsigned)
    value["seal_sha256"] = HARNESS._digest(
        {key: item for key, item in value.items() if key != "seal_sha256"}
    )
    forged = original.with_name(f"{value['artifact_id']}.json")
    original.unlink()
    forged.write_bytes(HARNESS._canonical(value) + b"\n")
    with pytest.raises(HARNESS.R8Error, match="authorization"):
        HARNESS.read_artifact(forged)


def test_bound_resealed_report_requires_all_underlying_files(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    result = HARNESS.run(
        source_root=source,
        source_commit=commit,
        r7_artifact=tmp_path / "missing.json",
        output=tmp_path / "out",
        production_observation=None,
    )
    original = Path(result["path"])
    value = json.loads(original.read_text(encoding="utf-8"))
    missing = tmp_path / "does-not-exist.json"
    ref = {
        "path": str(missing),
        "artifact_id": _id(80),
        "file_sha256": _id(81),
        "seal_sha256": _id(82),
        "schema": "chronovisor.recall-r8-readiness.v1",
    }
    value["evidence"] = {
        "status": "bound",
        "source_root": str(source),
        "observation": dict(ref),
        "r7": dict(ref),
        "auxiliary": {name: dict(ref) for name in (
            "r4", "r2", "r2_completion", "r2_external", "r3", "r3_completion", "r3_external"
        )},
        "phases": {phase["name"]: dict(ref) for phase in HARNESS.PHASES},
    }
    value["cleanup_authorized"] = True
    unsigned = {key: item for key, item in value.items() if key not in {"artifact_id", "seal_sha256"}}
    value["artifact_id"] = HARNESS._digest(unsigned)
    value["seal_sha256"] = HARNESS._digest(
        {key: item for key, item in value.items() if key != "seal_sha256"}
    )
    forged = original.with_name(f"{value['artifact_id']}.json")
    original.unlink()
    forged.write_bytes(HARNESS._canonical(value) + b"\n")
    with pytest.raises(HARNESS.R8Error, match="unavailable|evidence"):
        HARNESS.read_artifact(forged)


def _evidence_ref(path: Path, value: dict[str, Any], schema: str) -> dict[str, Any]:
    state = HARNESS._file_state(path, label="fixture evidence")
    file_sha = HARNESS._hash_file(path, state, label="fixture evidence")
    assert file_sha is not None
    return {
        "path": str(path),
        "artifact_id": value["artifact_id"],
        "file_sha256": file_sha,
        "seal_sha256": value["seal_sha256"],
        "schema": schema,
    }


def _bound_evidence(tmp_path: Path) -> dict[str, Any]:
    refs: dict[str, dict[str, Any]] = {}
    for index, (name, schema) in enumerate(HARNESS._EVIDENCE_REF_SCHEMAS.items()):
        directory = tmp_path / str(index)
        directory.mkdir()
        path = _sealed_artifact(directory, schema, {"kind": name})
        value = json.loads(path.read_text(encoding="utf-8"))
        refs[name] = _evidence_ref(path, value, schema)
    return {
        "status": "bound",
        "source_root": str(tmp_path),
        "observation": refs["observation"],
        "r7": refs["r7"],
        "auxiliary": {
            name: refs[name]
            for name in (
                "r4",
                "r2",
                "r2_completion",
                "r2_external",
                "r3",
                "r3_completion",
                "r3_external",
            )
        },
        "phases": {
            phase: refs[phase]
            for phase in HARNESS._R8_PHASE_NAMES
        },
    }


def _evidence_ref_for(value: dict[str, Any], name: str) -> dict[str, Any]:
    if name in {"r7", "observation"}:
        return value[name]
    if name in HARNESS._R8_PHASE_NAMES:
        return value["phases"][name]
    return value["auxiliary"][name]


def test_evidence_refs_bind_exact_kind_schema(tmp_path: Path) -> None:
    evidence = _bound_evidence(tmp_path)
    HARNESS._validate_evidence_shape(evidence)

    for name, expected_schema in HARNESS._EVIDENCE_REF_SCHEMAS.items():
        ref = _evidence_ref_for(evidence, name)
        path = Path(ref["path"])
        artifact, _state = HARNESS._read_json(path, label="fixture evidence")
        loaded = HARNESS._read_bound_ref(
            ref, label=f"{name} evidence", expected_schema=expected_schema
        )
        assert loaded[0] == path

        for replacement in (
            "evil",
            next(
                schema
                for schema in HARNESS._EVIDENCE_REF_SCHEMAS.values()
                if schema != expected_schema
            ),
            "chronovisor.recall-unknown.v1",
        ):
            ref["schema"] = replacement
            with pytest.raises(HARNESS.R8Error, match="schema"):
                HARNESS._validate_evidence_shape(evidence)
            with pytest.raises(HARNESS.R8Error, match="schema"):
                HARNESS._read_bound_ref(
                    ref, label=f"{name} evidence", expected_schema=expected_schema
                )
            ref["schema"] = expected_schema

            forged_dir = tmp_path / f"forged-{name}-{replacement.replace('.', '_')}"
            forged_dir.mkdir()
            forged_path = _sealed_artifact(forged_dir, replacement, {"kind": name})
            forged_value = json.loads(forged_path.read_text(encoding="utf-8"))
            forged_ref = _evidence_ref(forged_path, forged_value, replacement)
            with pytest.raises(HARNESS.R8Error, match="schema"):
                HARNESS._read_bound_ref(
                    forged_ref,
                    label=f"{name} evidence",
                    expected_schema=expected_schema,
                )

        assert artifact["schema"] == expected_schema


def test_absent_inventory_metadata_is_preserved() -> None:
    value = HARNESS._safe_metadata(HARNESS._empty_inventory(), label="fixture")
    assert value["present"] is False
    assert value["file_state"] is None
    assert value["sidecars"] == {}


def test_phase_and_observation_external_hashes_are_non_circular(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _source(tmp_path)
    production_root = tmp_path / "production"
    production_root.mkdir()
    for relative in HARNESS._POINTER_FILES.values():
        pointer = production_root / relative
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        HARNESS, "_fixed_production_root", lambda _source_root: production_root
    )
    phase_dir = tmp_path / "phases"
    phase_dir.mkdir()
    actual = HARNESS._observe_directory(production_root)
    source_tree_sha256 = _source_tree_digest(source)
    inventory, _inventory_reasons = HARNESS._inventory_summary(
        {
            "ox": actual["ox"],
            "pointers": actual["pointers"],
            "legacy": actual["legacy"],
            "locks": actual["locks"],
        }
    )
    inventory_sha = HARNESS._digest(inventory)
    refs: dict[str, dict[str, Any]] = {}
    previous_value: dict[str, Any] | None = None
    for phase in HARNESS.PHASES:
        path, ref = _phase_receipt_file(
            phase_dir, phase, commit, inventory_sha, previous=previous_value
        )
        refs[phase["name"]] = ref
        previous_value = json.loads(path.read_text(encoding="utf-8"))
    observation_path = _production_observation_file(
        tmp_path, production_root, commit, refs, source_tree_sha256
    )

    phase_value, _state = HARNESS._read_json(
        Path(refs[HARNESS.PHASES[0]["name"]]["path"]), label="fixture phase"
    )
    assert phase_value["file_sha256"] != refs[HARNESS.PHASES[0]["name"]]["file_sha256"]
    observation_value = json.loads(observation_path.read_text(encoding="utf-8"))
    observation_state = HARNESS._file_state(
        observation_path, label="fixture observation"
    )
    observation_file_sha = HARNESS._hash_file(
        observation_path, observation_state, label="fixture observation"
    )
    assert observation_file_sha is not None
    assert observation_value["file_sha256"] != observation_file_sha
    loaded = HARNESS._read_phase_receipt_ref(
        refs[HARNESS.PHASES[0]["name"]], source_root=source, phase_name="r7_receipt"
    )
    assert loaded is not None
    observation, _ = HARNESS._read_observation(observation_path, source_root=source)
    assert observation["_sealed"] is True
    inventory, inventory_reasons = HARNESS._inventory_summary(observation)
    phases, phase_reasons = HARNESS._phase_summary(
        observation,
        inventory_reasons,
        source_commit=commit,
        inventory_sha256=HARNESS._digest(inventory),
        source_root=source,
        r7_ready=True,
    )
    assert inventory_reasons == []
    assert phase_reasons == []
    assert all(item["status"] == "ready" for item in phases)


def test_phase_ref_and_observation_content_hash_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _source(tmp_path)
    production_root = tmp_path / "production"
    production_root.mkdir()
    monkeypatch.setattr(
        HARNESS, "_fixed_production_root", lambda _source_root: production_root
    )
    phase_dir = tmp_path / "phases"
    phase_dir.mkdir()
    phase_path, phase_ref = _phase_receipt_file(
        phase_dir, HARNESS.PHASES[0], commit, _id(93)
    )
    phase_value, _state = HARNESS._read_json(phase_path, label="fixture phase")
    forged_ref = dict(phase_ref)
    forged_ref["file_sha256"] = phase_value["file_sha256"]
    assert (
        HARNESS._read_phase_receipt_ref(
            forged_ref, source_root=source, phase_name="r7_receipt"
        )
        is None
    )
    refs = {phase["name"]: dict(phase_ref) for phase in HARNESS.PHASES}
    observation_path = _production_observation_file(
        tmp_path, production_root, commit, refs, _source_tree_digest(source)
    )
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    actual_file_sha = HARNESS._hash_file(
        observation_path,
        HARNESS._file_state(observation_path, label="fixture observation"),
        label="fixture observation",
    )
    assert actual_file_sha is not None
    observation["file_sha256"] = actual_file_sha
    observation["artifact_id"] = HARNESS._digest(
        {
            key: item
            for key, item in observation.items()
            if key not in {"artifact_id", "seal_sha256"}
        }
    )
    observation.pop("seal_sha256", None)
    observation["seal_sha256"] = HARNESS._digest(observation)
    forged_path = observation_path.with_name(f"{observation['artifact_id']}.json")
    forged_path.write_bytes(HARNESS._canonical(observation) + b"\n")
    with pytest.raises(HARNESS.R8Error, match="content hash"):
        HARNESS._read_observation(forged_path, source_root=source)


def test_forged_resealed_r7_cannot_pass(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    r7_dir = tmp_path / "r7"
    r7_dir.mkdir()
    artifact = _r7(r7_dir, source_commit=commit, certified=True, complete=False)
    result = HARNESS.run(
        source_root=source,
        source_commit=commit,
        r7_artifact=artifact,
        output=tmp_path / "out",
        production_observation=_observation(),
    )
    assert result["cleanup_authorized"] is False
    assert any(
        "r7" in reason.lower()
        or "official" in reason.lower()
        for reason in HARNESS.read_artifact(Path(result["path"]))["reasons"]
    )


@pytest.mark.parametrize("mode", ["missing", "false", "tampered"])
def test_r7_live_attestation_is_external_and_fail_closed(
    tmp_path: Path, mode: str
) -> None:
    source, commit = _source(tmp_path)
    r7_dir = tmp_path / "r7"
    r7_dir.mkdir()
    if mode == "missing":
        artifact = _r7_structurally_valid(r7_dir, commit)
        next((r7_dir / "r7-live-attestations").glob("*.json")).unlink()
    else:
        artifact = _r7_structurally_valid(
            r7_dir, commit, live_certification=(mode != "false")
        )
        if mode == "tampered":
            live_path = next((r7_dir / "r7-live-attestations").glob("*.json"))
            live = json.loads(live_path.read_text(encoding="utf-8"))
            live["source_commit"] = "b" * 40
            live_path.write_bytes(HARNESS._canonical(live) + b"\n")
    result = HARNESS.run(
        source_root=source,
        source_commit=commit,
        r7_artifact=artifact,
        output=tmp_path / "out",
        production_observation=_observation(),
    )
    report = HARNESS.read_artifact(Path(result["path"]))
    assert result["cleanup_authorized"] is False
    assert any("r7" in reason.lower() or "official" in reason.lower() for reason in report["reasons"])


def test_r7_inline_live_attestation_is_not_a_gate(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    r7_dir = tmp_path / "r7"
    r7_dir.mkdir()
    artifact = _r7_structurally_valid(r7_dir, commit)
    value = json.loads(artifact.read_text(encoding="utf-8"))
    value["live_attestation"] = {"certification": True}
    unsigned = {key: item for key, item in value.items() if key not in {"artifact_id", "seal_sha256"}}
    value["artifact_id"] = HARNESS._digest(unsigned)
    value["seal_sha256"] = HARNESS._digest(
        {key: item for key, item in value.items() if key != "seal_sha256"}
    )
    artifact.unlink()
    artifact = r7_dir / f"{value['artifact_id']}.json"
    artifact.write_bytes(HARNESS._canonical(value) + b"\n")
    result = HARNESS.run(
        source_root=source,
        source_commit=commit,
        r7_artifact=artifact,
        output=tmp_path / "out",
        production_observation=_observation(),
    )
    report = HARNESS.read_artifact(Path(result["path"]))
    assert result["cleanup_authorized"] is False
    assert any("r7" in reason.lower() or "official" in reason.lower() for reason in report["reasons"])


def test_r7_live_receipt_postpublish_drift_removes_new_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _source(tmp_path)
    r7_dir = tmp_path / "r7"
    r7_dir.mkdir()
    artifact = _r7_structurally_valid(r7_dir, commit)
    live_path = next((r7_dir / "r7-live-attestations").glob("*.json"))
    original = HARNESS._input_boundary
    calls = 0

    def drift(path: Path, *, label: str) -> object:
        nonlocal calls
        if label == "r7_live input":
            calls += 1
            if calls == 3:
                path.write_bytes(path.read_bytes() + b"\n")
        return original(path, label=label)

    monkeypatch.setattr(HARNESS, "_input_boundary", drift)
    with pytest.raises(HARNESS.R8Error, match="r7_live input changed"):
        HARNESS.run(
            source_root=source,
            source_commit=commit,
            r7_artifact=artifact,
            output=tmp_path / "out",
            production_observation=_observation(),
        )
    assert not list((tmp_path / "out").glob("*.json"))
    assert live_path.exists()


def test_symlink_and_root_overlap_are_rejected(tmp_path: Path) -> None:
    source, _ = _source(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(HARNESS.R8Error, match="symlink"):
        HARNESS._assert_paths(link, tmp_path / "r7.json", tmp_path / "out")
    with pytest.raises(HARNESS.R8Error, match="overlap"):
        HARNESS._assert_paths(source, tmp_path / "r7.json", source / "out")


@pytest.mark.parametrize(
    ("enabled", "leased", "reason"),
    [(True, 0, "ox_enabled"), (False, 1, "leased_work_present")],
)
def test_ox_safety_vetoes_block_readiness(
    tmp_path: Path, enabled: bool, leased: int, reason: str
) -> None:
    source, commit = _source(tmp_path)
    r7_dir = tmp_path / "r7"
    r7_dir.mkdir()
    r7 = _r7(r7_dir, source_commit=commit)
    result = HARNESS.run(
        source_root=source,
        source_commit=commit,
        r7_artifact=r7,
        output=tmp_path / "out",
        production_observation=_observation(enabled=enabled, leased=leased),
    )
    artifact = HARNESS.read_artifact(Path(result["path"]))
    assert result["cleanup_authorized"] is False
    assert reason in artifact["reasons"]


def test_sensitive_observation_is_not_copied(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    r7_dir = tmp_path / "r7"
    r7_dir.mkdir()
    observation = _observation()
    observation["legacy"]["provider_payload"] = "DO-NOT-LEAK"
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    result = HARNESS.run(
        source_root=source,
        source_commit=commit,
        r7_artifact=_r7(r7_dir, source_commit=commit),
        output=tmp_path / "out",
        production_observation=observation_path,
    )
    raw = Path(result["path"]).read_text(encoding="utf-8")
    assert "DO-NOT-LEAK" not in raw
    assert result["cleanup_authorized"] is False


def test_idempotent_immutable_write(tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        "captured_at": "1970-01-01T00:00:00+00:00",
        "source": {},
        "r7": {},
        "inventory": {},
        "observation_state": None,
        "phases": [],
        "reasons": ["test"],
        "cleanup_authorized": False,
        "cleanup_performed": False,
        "provider_calls": 0,
        "production_write_performed": False,
    }
    first = HARNESS._write_immutable(tmp_path / "out", payload)
    second = HARNESS._write_immutable(tmp_path / "out", payload)
    assert first[:2] == second[:2]
    assert len(list((tmp_path / "out").glob("*.json"))) == 1


def test_phase_receipts_alone_do_not_authorize_without_r4_r2_r3(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    r7_dir = tmp_path / "r7"
    r7_dir.mkdir()
    result = HARNESS.run(
        source_root=source,
        source_commit=commit,
        r7_artifact=_r7(r7_dir, source_commit=commit),
        output=tmp_path / "out",
        production_observation=_observation(),
    )
    assert result["cleanup_authorized"] is False
    artifact = HARNESS.read_artifact(Path(result["path"]))
    assert artifact["cleanup_performed"] is False
    assert artifact["production_write_performed"] is False
    assert "r4_artifact_missing" in artifact["reasons"]
    assert "r2_artifact_missing" in artifact["reasons"]
    assert "r3_artifact_missing" in artifact["reasons"]


def test_auxiliary_receipts_close_r4_r2_r3_identity_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _source(tmp_path)
    aux = tmp_path / "aux"
    aux.mkdir()
    r7_dir = tmp_path / "r7"
    r7_dir.mkdir()
    r4 = _r4(aux, commit)
    r2 = _r2(aux, commit)
    r3 = _r3(aux, commit)
    r3_completion = _r3_completion(aux, commit, r3)
    r2_value = json.loads(r2.read_text(encoding="utf-8"))
    r2_completion = aux / "r2-completion.json"
    r2_completion.write_bytes(
        HARNESS._canonical(
            _sealed(
                {
                    "schema": "chronovisor.recall-distillation.v1",
                    "namespace": "recall-distillation",
                    "kind": "chronovisor.recall-r2-completion",
                    "r2_artifact_id": r2_value["artifact_id"],
                    "r2_artifact_seal_sha256": r2_value["seal_sha256"],
                    "source_commit": commit,
                }
            )
        )
        + b"\n"
    )
    original_loader = HARNESS._load_sibling

    def load(name: str) -> Any:
        if name == "recall_r3_harness.py":
            return _FakeR3
        return original_loader(name)

    monkeypatch.setattr(HARNESS, "_load_sibling", load)
    monkeypatch.setattr(HARNESS, "_load_runtime_store", lambda _root: _ReadOnlyStore())
    result = HARNESS.run(
        source_root=source,
        source_commit=commit,
        r7_artifact=_r7(r7_dir, source_commit=commit),
        output=tmp_path / "out",
        production_observation=_observation(),
        r4_artifact=r4,
        r2_artifact=r2,
        r2_completion=r2_completion,
        r3_artifact=r3,
        r3_completion=r3_completion,
    )
    assert result["cleanup_authorized"] is False
    artifact = HARNESS.read_artifact(Path(result["path"]))
    assert any("r7" in reason.lower() or "official" in reason.lower() for reason in artifact["reasons"])


def test_auxiliary_false_r4_receipt_blocks_readiness(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    aux = tmp_path / "aux"
    aux.mkdir()
    _, reasons = HARNESS._validate_auxiliary(
        source_root=source,
        source_commit=commit,
        r4_artifact=_r4(aux, commit, passed=False),
        r2_artifact=None,
        r2_completion=None,
        r3_artifact=None,
        r3_completion=None,
    )
    assert any(reason.startswith("r4_invalid:") for reason in reasons)


def test_auxiliary_tampered_r4_receipt_blocks_readiness(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    aux = tmp_path / "aux"
    aux.mkdir()
    r4 = _r4(aux, commit)
    value = json.loads(r4.read_text(encoding="utf-8"))
    value["production_certification"]["passed"] = False
    r4.write_bytes(HARNESS._canonical(value) + b"\n")
    _, reasons = HARNESS._validate_auxiliary(
        source_root=source,
        source_commit=commit,
        r4_artifact=r4,
        r2_artifact=None,
        r2_completion=None,
        r3_artifact=None,
        r3_completion=None,
    )
    assert any(reason.startswith("r4_invalid:") for reason in reasons)


def test_observation_toctou_is_sealed_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, commit = _source(tmp_path)
    r7_dir = tmp_path / "r7"
    r7_dir.mkdir()
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(
        json.dumps(_observation(), sort_keys=True), encoding="utf-8"
    )
    original = HARNESS._observation_boundary
    calls = 0

    def drift(path: Path) -> object:
        nonlocal calls
        boundary = original(path)
        calls += 1
        if calls == 1:
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return boundary

    monkeypatch.setattr(HARNESS, "_observation_boundary", drift)
    with pytest.raises(HARNESS.R8Error, match="observation changed"):
        HARNESS.run(
            source_root=source,
            source_commit=commit,
            r7_artifact=_r7(r7_dir, source_commit=commit),
            output=tmp_path / "out",
            production_observation=observation_path,
        )


def test_plain_mapping_is_test_only_and_never_authorizes(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    r7_dir = tmp_path / "r7"
    r7_dir.mkdir()
    result = HARNESS.run(
        source_root=source,
        source_commit=commit,
        r7_artifact=_r7(r7_dir, source_commit=commit),
        output=tmp_path / "out",
        production_observation=_observation(),
        test_only=True,
    )
    artifact = HARNESS.read_artifact(Path(result["path"]))
    assert result["cleanup_authorized"] is False
    assert any("r7" in reason.lower() or "official" in reason.lower() for reason in artifact["reasons"])


def test_r2_compact_boolean_parity_is_rejected() -> None:
    with pytest.raises(HARNESS.R8Error, match="compact boolean"):
        HARNESS._r2_parity_projection(True, label="fixture")


def test_minimal_phase_receipt_cannot_be_ready() -> None:
    observation = _observation()
    phases, reasons = HARNESS._phase_summary(
        observation,
        (),
        source_commit="a" * 40,
        inventory_sha256=HARNESS._digest({}),
    )
    assert all(item["status"] == "blocked" for item in phases)
    assert reasons


def test_directory_observation_never_reads_raw_or_writes_production(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    production = tmp_path / "production"
    raw = production / "raw"
    raw.mkdir(parents=True)
    secret = raw / "secret.txt"
    secret.write_text("provider payload", encoding="utf-8")
    before = secret.read_bytes()
    r7_dir = tmp_path / "r7"
    r7_dir.mkdir()
    result = HARNESS.run(
        source_root=source,
        source_commit=commit,
        r7_artifact=_r7(r7_dir, source_commit=commit),
        output=tmp_path / "out",
        production_observation=production,
    )
    assert result["cleanup_authorized"] is False
    assert secret.read_bytes() == before
    assert not list(raw.glob("*.json"))
