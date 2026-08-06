from __future__ import annotations

from scripts.runtime_ownership.access import discover_access_facts


def _discover(
    sources: dict[str, str],
    *,
    module: str = "chronovisor.state",
) -> dict:
    candidate = {
        "id": "runtime-resource:state",
        "module": module,
        "symbol": "STATE_FILE",
        "locator": {"type": "path", "value": "$ROOT/state.json"},
    }
    return discover_access_facts(
        {path: source.encode() for path, source in sources.items()}, [candidate]
    )


def _operations(result: dict) -> list[str]:
    return [row["operation"] for row in result["accesses"]]


def test_default_star_exports_public_but_excludes_private_name() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "from chronovisor.state import STATE_FILE\n"
                "PUBLIC_STATE = STATE_FILE\n"
                "_PRIVATE_STATE = STATE_FILE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import *\n"
                "PUBLIC_STATE.write_text('value')\n"
                "_PRIVATE_STATE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_static_tuple_all_exactly_limits_star_exports() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "from chronovisor.state import STATE_FILE\n"
                "PUBLIC_STATE = STATE_FILE\n"
                "_PRIVATE_STATE = STATE_FILE\n"
                "__all__ = ('PUBLIC_STATE',)\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import *\n"
                "PUBLIC_STATE.write_text('value')\n"
                "_PRIVATE_STATE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_named_private_import_ignores_empty_all() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "from chronovisor.state import STATE_FILE\n"
                "_PRIVATE_STATE = STATE_FILE\n"
                "__all__ = []\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import _PRIVATE_STATE\n"
                "_PRIVATE_STATE.write_text('value')\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_all_declared_before_binding_selects_final_binding() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "__all__ = ['PUBLIC_STATE']\n"
                "from chronovisor.state import STATE_FILE\n"
                "PUBLIC_STATE = STATE_FILE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import *\nPUBLIC_STATE.write_text('value')\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_named_import_observes_origin_then_object_strong_rebind() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "import chronovisor.state as state_module\n"
                "EXPORTED_STATE = state_module.STATE_FILE\n"
                "EXPORTED_STATE = object()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "EXPORTED_STATE.write_text('value')\n"
            ),
        }
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_named_import_observes_object_then_origin_strong_rebind() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "import chronovisor.state as state_module\n"
                "EXPORTED_STATE = object()\n"
                "EXPORTED_STATE = state_module.STATE_FILE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "EXPORTED_STATE.write_text('value')\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_star_import_observes_origin_then_object_strong_rebind() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "import chronovisor.state as state_module\n"
                "EXPORTED_STATE = state_module.STATE_FILE\n"
                "EXPORTED_STATE = object()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import *\n"
                "EXPORTED_STATE.write_text('value')\n"
            ),
        }
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_star_import_observes_object_then_origin_strong_rebind() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "import chronovisor.state as state_module\n"
                "EXPORTED_STATE = object()\n"
                "EXPORTED_STATE = state_module.STATE_FILE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import *\n"
                "EXPORTED_STATE.write_text('value')\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_relative_single_dot_star_reexport_resolves_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/pkg/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/pkg/exports.py": "from .state import *\n",
            "src/chronovisor/consumer.py": (
                "from chronovisor.pkg.exports import *\n"
                "STATE_FILE.write_text('value')\n"
            ),
        },
        module="chronovisor.pkg.state",
    )

    assert _operations(result) == ["path.write_text"]
    assert result["escapes"] == []


def test_function_local_import_does_not_become_module_export() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "def load():\n"
                "    from chronovisor.state import STATE_FILE as FUNCTION_STATE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import FUNCTION_STATE\n"
                "FUNCTION_STATE.write_text('value')\n"
            ),
        }
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_dynamic_all_may_export_private_origin_with_one_escape() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "from chronovisor.state import STATE_FILE\n"
                "_PRIVATE_STATE = STATE_FILE\n"
                "__all__ = choose_exports()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import *\n"
                "_PRIVATE_STATE.write_text('value')\n"
            ),
        }
    )

    assert _operations(result) == ["path.write_text"]
    assert len(result["escapes"]) == 1
    assert result["escapes"][0]["reason"] == "dynamic_star_import"


def test_dynamic_star_sites_have_two_deterministic_escapes() -> None:
    sources = {
        "src/chronovisor/state.py": "STATE_FILE = object()\n",
        "src/chronovisor/exports.py": (
            "from chronovisor.state import STATE_FILE\n__all__ = choose_exports()\n"
        ),
        "src/chronovisor/consumer.py": (
            "from chronovisor.exports import *\nfrom chronovisor.exports import *\n"
        ),
    }

    first = _discover(sources)
    second = _discover(sources)

    assert first["accesses"] == []
    assert len(first["escapes"]) == 2
    assert {row["reason"] for row in first["escapes"]} == {"dynamic_star_import"}
    assert sorted(row["evidence"]["line"] for row in first["escapes"]) == [1, 2]
    assert len(set(first["escape_ids"])) == 2
    assert first["escape_ids"] == second["escape_ids"]


def test_dynamic_star_weakly_joins_existing_origin_with_object_export() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "STATE_FILE = object()\n__all__ = choose_exports()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.state import STATE_FILE\n"
                "from chronovisor.exports import *\n"
                "STATE_FILE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert len(result["escapes"]) == 1
    assert result["escapes"][0]["reason"] == "dynamic_star_import"


def test_module_if_join_followed_by_object_strong_rebind_kills_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "if object():\n"
                "    from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
                "else:\n"
                "    EXPORTED_STATE = object()\n"
                "EXPORTED_STATE = object()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "EXPORTED_STATE.read_text()\n"
            ),
        }
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_try_finally_object_strong_rebind_kills_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "try:\n"
                "    from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
                "finally:\n"
                "    EXPORTED_STATE = object()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "EXPORTED_STATE.read_text()\n"
            ),
        }
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_try_prefix_origin_survives_possible_raise_and_pass_handler() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "try:\n"
                "    from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
                "    may_raise()\n"
                "except Exception:\n"
                "    pass\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "EXPORTED_STATE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_try_handler_origin_joins_normal_object_export() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "try:\n"
                "    may_raise()\n"
                "    EXPORTED_STATE = object()\n"
                "except Exception:\n"
                "    from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "EXPORTED_STATE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_try_else_normal_object_keeps_handler_origin_joined() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "try:\n"
                "    may_raise()\n"
                "except Exception:\n"
                "    from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
                "else:\n"
                "    EXPORTED_STATE = object()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "EXPORTED_STATE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_try_finally_object_kills_origin_from_every_path() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "try:\n"
                "    from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
                "    may_raise()\n"
                "except Exception:\n"
                "    from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
                "finally:\n"
                "    EXPORTED_STATE = object()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "EXPORTED_STATE.read_text()\n"
            ),
        }
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_try_finally_origin_bind_exposes_origin_from_every_path() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "try:\n"
                "    may_raise()\n"
                "    EXPORTED_STATE = object()\n"
                "except Exception:\n"
                "    EXPORTED_STATE = object()\n"
                "finally:\n"
                "    from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "EXPORTED_STATE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_try_star_handler_origin_joins_normal_object_export() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "try:\n"
                "    may_raise()\n"
                "    EXPORTED_STATE = object()\n"
                "except* Exception:\n"
                "    from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "EXPORTED_STATE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_non_exhaustive_match_preserves_fallthrough_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
                "match object():\n"
                "    case 0:\n"
                "        EXPORTED_STATE = object()\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "EXPORTED_STATE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_anchored_two_module_star_cycle_converges_with_one_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/a.py": (
                "from chronovisor.state import STATE_FILE\n"
                "from chronovisor.b import *\n"
            ),
            "src/chronovisor/b.py": "from chronovisor.a import *\n",
            "src/chronovisor/consumer.py": (
                "from chronovisor.b import *\nSTATE_FILE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert result["accesses"][0]["binding_chain"] == [
        "origin:chronovisor.state:STATE_FILE",
        "alias:chronovisor.state:<module>:STATE_FILE",
        ("import:chronovisor.a:<module>:STATE_FILE->chronovisor.state:STATE_FILE"),
        "import:chronovisor.b:<module>:STATE_FILE->chronovisor.a:*",
        "import:chronovisor.consumer:<module>:STATE_FILE->chronovisor.b:*",
    ]
    assert result["escapes"] == []


def test_pure_two_module_star_cycle_terminates_without_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/a.py": "from chronovisor.b import *\n",
            "src/chronovisor/b.py": "from chronovisor.a import *\n",
            "src/chronovisor/consumer.py": (
                "from chronovisor.a import *\nSTATE_FILE.read_text()\n"
            ),
        }
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_class_body_import_does_not_become_module_export() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "class Holder:\n"
                "    from chronovisor.state import STATE_FILE as CLASS_STATE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import CLASS_STATE\nCLASS_STATE.read_text()\n"
            ),
        }
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_module_delete_removes_prior_origin_export() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
                "del EXPORTED_STATE\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import EXPORTED_STATE\n"
                "EXPORTED_STATE.read_text()\n"
            ),
        }
    )

    assert result["accesses"] == []
    assert result["escapes"] == []


def test_public_star_optional_name_preserves_consumer_origin_fallthrough() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": ("if object():\n    STATE_FILE = object()\n"),
            "src/chronovisor/consumer.py": (
                "from chronovisor.state import STATE_FILE\n"
                "from chronovisor.exports import *\n"
                "STATE_FILE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_augmented_all_exports_private_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "from chronovisor.state import STATE_FILE as _PRIVATE\n"
                "__all__ = []\n"
                "__all__ += ['_PRIVATE']\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import *\n_PRIVATE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_deleting_all_restores_public_default_exports() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "from chronovisor.state import STATE_FILE as PUBLIC\n"
                "__all__ = []\n"
                "del __all__\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.exports import *\nPUBLIC.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_future_annotations_does_not_shadow_consumer_origin_via_star() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": "from __future__ import annotations\n",
            "src/chronovisor/consumer.py": (
                "from chronovisor.state import STATE_FILE as annotations\n"
                "from chronovisor.exports import *\n"
                "annotations.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_except_as_implicit_delete_preserves_consumer_origin_via_star() -> None:
    result = _discover(
        {
            "src/chronovisor/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/exports.py": (
                "try:\n    may_raise()\nexcept Exception as EXPORTED_STATE:\n    pass\n"
            ),
            "src/chronovisor/consumer.py": (
                "from chronovisor.state import STATE_FILE as EXPORTED_STATE\n"
                "from chronovisor.exports import *\n"
                "EXPORTED_STATE.read_text()\n"
            ),
        }
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []


def test_relative_parent_star_reexport_resolves_origin() -> None:
    result = _discover(
        {
            "src/chronovisor/pkg/state.py": "STATE_FILE = object()\n",
            "src/chronovisor/pkg/sub/exports.py": "from ..state import *\n",
            "src/chronovisor/consumer.py": (
                "from chronovisor.pkg.sub.exports import *\nSTATE_FILE.read_text()\n"
            ),
        },
        module="chronovisor.pkg.state",
    )

    assert _operations(result) == ["path.read_text"]
    assert result["escapes"] == []
