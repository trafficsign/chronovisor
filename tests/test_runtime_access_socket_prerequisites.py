from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.runtime_ownership.access import discover_access_facts
from scripts.runtime_ownership.access_model import FlowValue

SEMANTIC_ID = "runtime-resource:semantic-socket"
RERANKER_ID = "runtime-resource:reranker-socket"


def _candidate(resource_id: str, *, semantic: bool) -> dict[str, object]:
    service = "semantic" if semantic else "reranker"
    return {
        "id": resource_id,
        "module": f"chronovisor.search.{service}_service",
        "symbol": "serve",
        "locator": {
            "type": "socket",
            "value": f"unix://$HOME/.chronovisor/runtime/{service}.sock",
        },
    }


def _config_source() -> str:
    return (
        "from dataclasses import dataclass, field\n"
        "@dataclass(frozen=True)\n"
        "class SearchEmbeddingConfig:\n"
        "    socket: str = '~/.chronovisor/runtime/semantic.sock'\n"
        "@dataclass(frozen=True)\n"
        "class RerankerServiceConfig:\n"
        "    socket: str = '~/.chronovisor/runtime/reranker.sock'\n"
        "@dataclass(frozen=True)\n"
        "class RerankerConfig:\n"
        "    service: RerankerServiceConfig = field(\n"
        "        default_factory=RerankerServiceConfig\n"
        "    )\n"
    )


def _discover(
    consumer: str,
    *,
    config_source: str | None = None,
) -> dict[str, Any]:
    sources = {
        "src/chronovisor/core/runtime_config.py": (
            config_source if config_source is not None else _config_source()
        ).encode(),
        "src/chronovisor/consumer.py": consumer.encode(),
    }
    return discover_access_facts(
        sources,
        [
            _candidate(SEMANTIC_ID, semantic=True),
            _candidate(RERANKER_ID, semantic=False),
        ],
    )


def _assert_no_concrete_socket_facts(result: dict[str, Any]) -> None:
    assert result["accesses"] == []
    assert all(
        not str(row["sink"]).startswith(("socket.", "socketserver."))
        for row in result["accesses"]
    )


def test_flow_value_recursively_preserves_nested_analysis_state() -> None:
    nested = FlowValue(
        origins={SEMANTIC_ID: frozenset({("origin:test",)})},
    )
    parent = FlowValue(
        object_types={"example:Config"},
        attribute_values={"socket": nested},
        attribute_values_complete=True,
    )

    assert not parent.has_origins
    assert parent.has_analysis_state
    assert parent.copy().attribute_values["socket"].has_origins
    assert parent.bound("summary").attribute_values["socket"].has_origins
    safe, cyclic = parent.partition_call_cycles(target="example:helper")
    assert safe.attribute_values["socket"].has_origins
    assert not cyclic.attribute_values["socket"].has_origins

    override = FlowValue(
        object_types={"example:Config"},
        attribute_values={"socket": FlowValue()},
        attribute_values_complete=True,
    )
    joined = parent.merged(override)
    assert joined.attribute_values["socket"].has_origins
    assert joined.attribute_values["socket"].attribute_values_ambiguous


def test_dataclass_defaults_factory_class_projection_and_override_are_exact() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import (\n"
        "    RerankerConfig, RerankerServiceConfig, SearchEmbeddingConfig,\n"
        ")\n"
        "def semantic_default():\n"
        "    opaque(str(Path(SearchEmbeddingConfig().socket).expanduser()))\n"
        "def semantic_override():\n"
        "    opaque(str(Path(SearchEmbeddingConfig(socket='custom').socket).expanduser()))\n"
        "def parent_only():\n"
        "    opaque(SearchEmbeddingConfig())\n"
        "def reranker_factory():\n"
        "    opaque(str(Path(RerankerConfig().service.socket).expanduser()))\n"
        "def reranker_class_default():\n"
        "    opaque(str(Path(RerankerServiceConfig.socket).expanduser()))\n"
    )

    _assert_no_concrete_socket_facts(result)
    escapes = result["escapes"]
    assert {(row["actor"], row["resource_id"]) for row in escapes} == {
        ("chronovisor.consumer:semantic_default", SEMANTIC_ID),
        ("chronovisor.consumer:reranker_factory", RERANKER_ID),
        ("chronovisor.consumer:reranker_class_default", RERANKER_ID),
    }


def test_dataclass_branch_unknown_and_missing_attributes_fail_closed() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "def branch(flag):\n"
        "    config = (\n"
        "        SearchEmbeddingConfig()\n"
        "        if flag\n"
        "        else SearchEmbeddingConfig(socket='custom')\n"
        "    )\n"
        "    opaque(str(Path(config.socket).expanduser()))\n"
        "def unknown(flag):\n"
        "    config = SearchEmbeddingConfig() if flag else object()\n"
        "    opaque(Path(config.socket))\n"
        "def missing():\n"
        "    opaque(Path(SearchEmbeddingConfig().missing))\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert {(row["actor"], row["resource_id"]) for row in result["escapes"]} == {
        ("chronovisor.consumer:branch", SEMANTIC_ID),
        ("chronovisor.consumer:unknown", SEMANTIC_ID),
    }
    # Access facts are may-path evidence: the registered default alternatives
    # remain concrete, while the origin-bearing invalid projection still
    # escapes and the surviving path reaches the deliberately unknown boundary.
    assert {row["reason"] for row in result["escapes"]} == {
        "invalid_or_ambiguous_path_constructor_signature",
        "registered_locator_to_unknown_callee"
    }


def test_generated_dataclass_constructor_rejects_invalid_binding() -> None:
    result = _discover(
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "def duplicate():\n"
        "    SearchEmbeddingConfig(\n"
        "        SearchEmbeddingConfig.socket, socket='custom'\n"
        "    )\n"
        "def unexpected():\n"
        "    SearchEmbeddingConfig(unknown=SearchEmbeddingConfig.socket)\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert {row["actor"] for row in result["escapes"]} == {
        "chronovisor.consumer:duplicate",
        "chronovisor.consumer:unexpected",
    }
    assert {row["reason"] for row in result["escapes"]} == {
        "invalid_or_ambiguous_dataclass_signature"
    }


def test_path_expanduser_and_str_preserve_locator_for_both_import_forms() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "import pathlib\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "def direct():\n"
        "    value = Path(SearchEmbeddingConfig.socket).expanduser()\n"
        "    opaque(str(value))\n"
        "def qualified():\n"
        "    value = pathlib.Path(SearchEmbeddingConfig().socket).expanduser()\n"
        "    opaque(str(value))\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert {(row["actor"], row["resource_id"]) for row in result["escapes"]} == {
        ("chronovisor.consumer:direct", SEMANTIC_ID),
        ("chronovisor.consumer:qualified", SEMANTIC_ID),
    }
    assert all(
        "representation:builtins.str" in row["binding_chain"]
        for row in result["escapes"]
    )


def test_qualified_path_mutations_and_reimport_escape_with_the_locator() -> None:
    mutations = (
        "pathlib.Path = opaque\n",
        "setattr(pathlib, 'Path', opaque)\n",
        "del pathlib.Path\n",
        "pathlib.__dict__['Path'] = opaque\n",
        "attribute = 'Path' if opaque() else 'Other'\n"
        "setattr(pathlib, attribute, opaque)\n",
    )

    for mutation in mutations:
        result = _discover(
            "import pathlib\n"
            "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
            f"{mutation}"
            "import pathlib\n"
            "def inspect():\n"
            "    pathlib.Path(SearchEmbeddingConfig.socket)\n"
        )

        _assert_no_concrete_socket_facts(result)
        assert len(result["escapes"]) == 1
        assert result["escapes"][0]["actor"] == "chronovisor.consumer:inspect"
        assert result["escapes"][0]["resource_id"] == SEMANTIC_ID
        assert result["escapes"][0]["reason"] == (
            "registered_locator_to_unknown_callee"
        )


def test_path_reference_captured_before_mutation_retains_exact_binding() -> None:
    result = _discover(
        "import pathlib\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "captured_path = pathlib.Path\n"
        "pathlib.Path = opaque\n"
        "import pathlib\n"
        "def inspect():\n"
        "    value = captured_path(SearchEmbeddingConfig.socket).expanduser()\n"
        "    opaque(str(value))\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert len(result["escapes"]) == 1
    escape = result["escapes"][0]
    assert escape["actor"] == "chronovisor.consumer:inspect"
    assert escape["resource_id"] == SEMANTIC_ID
    assert "constructor:pathlib.Path" in escape["binding_chain"]
    assert any("expanduser" in step for step in escape["binding_chain"])


def test_unrelated_pathlib_mutation_does_not_taint_path() -> None:
    result = _discover(
        "import pathlib\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "pathlib.Other = opaque\n"
        "import pathlib\n"
        "def inspect():\n"
        "    value = pathlib.Path(SearchEmbeddingConfig.socket).expanduser()\n"
        "    opaque(str(value))\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert len(result["escapes"]) == 1
    escape = result["escapes"][0]
    assert escape["actor"] == "chronovisor.consumer:inspect"
    assert escape["resource_id"] == SEMANTIC_ID
    assert "constructor:pathlib.Path" in escape["binding_chain"]
    assert any("expanduser" in step for step in escape["binding_chain"])


def test_shadowed_and_invalid_path_operations_escape_without_concrete_facts() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "def shadow_path(Path):\n"
        "    Path(SearchEmbeddingConfig.socket)\n"
        "def shadow_str(str):\n"
        "    str(Path(SearchEmbeddingConfig.socket))\n"
        "def dynamic_path(flag):\n"
        "    constructor = Path if flag else opaque\n"
        "    constructor(SearchEmbeddingConfig.socket)\n"
        "def invalid_path():\n"
        "    Path(SearchEmbeddingConfig.socket, 'extra')\n"
        "def invalid_expanduser():\n"
        "    Path(SearchEmbeddingConfig.socket).expanduser('extra')\n"
        "def invalid_str():\n"
        "    str(Path(SearchEmbeddingConfig.socket), 'utf-8')\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert {row["actor"] for row in result["escapes"]} == {
        "chronovisor.consumer:shadow_path",
        "chronovisor.consumer:shadow_str",
        "chronovisor.consumer:dynamic_path",
        "chronovisor.consumer:invalid_path",
        "chronovisor.consumer:invalid_expanduser",
        "chronovisor.consumer:invalid_str",
    }
    assert {row["reason"] for row in result["escapes"]} >= {
        "invalid_or_ambiguous_path_constructor_signature",
        "invalid_or_ambiguous_path_transform_signature",
        "invalid_or_ambiguous_path_representation_signature",
    }


def test_nested_only_state_crosses_helper_summary_and_closure() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "def identity(value):\n"
        "    return value\n"
        "def through_summary():\n"
        "    config = identity(SearchEmbeddingConfig())\n"
        "    opaque(str(Path(config.socket).expanduser()))\n"
        "def outer():\n"
        "    config = SearchEmbeddingConfig()\n"
        "    def capture():\n"
        "        return config\n"
        "    return capture()\n"
        "def through_closure():\n"
        "    opaque(str(Path(outer().socket).expanduser()))\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert {(row["actor"], row["resource_id"]) for row in result["escapes"]} == {
        ("chronovisor.consumer:through_summary", SEMANTIC_ID),
        ("chronovisor.consumer:through_closure", SEMANTIC_ID),
    }


def test_production_runtime_config_fields_seed_exact_socket_origins() -> None:
    repository = Path(__file__).resolve().parents[1]
    runtime_config = (
        repository / "src/chronovisor/core/runtime_config.py"
    ).read_text()
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import (\n"
        "    RerankerConfig, SearchEmbeddingConfig, load_reranker_config,\n"
        "    load_search_embedding_config,\n"
        ")\n"
        "def semantic():\n"
        "    opaque(str(Path(SearchEmbeddingConfig().socket).expanduser()))\n"
        "def reranker():\n"
        "    opaque(str(Path(RerankerConfig().service.socket).expanduser()))\n"
        "def semantic_fallback():\n"
        "    config = load_search_embedding_config()\n"
        "    opaque(str(Path(config.socket).expanduser()))\n"
        "def reranker_fallback():\n"
        "    config = load_reranker_config()\n"
        "    opaque(str(Path(config.service.socket).expanduser()))\n",
        config_source=runtime_config,
    )

    _assert_no_concrete_socket_facts(result)
    assert {(row["actor"], row["resource_id"]) for row in result["escapes"]} == {
        ("chronovisor.consumer:semantic", SEMANTIC_ID),
        ("chronovisor.consumer:reranker", RERANKER_ID),
        ("chronovisor.consumer:semantic_fallback", SEMANTIC_ID),
        ("chronovisor.consumer:reranker_fallback", RERANKER_ID),
    }


def test_socket_alias_requires_the_config_default_to_match_the_candidate() -> None:
    mismatched = _config_source().replace(
        "~/.chronovisor/runtime/semantic.sock",
        "~/.chronovisor/runtime/not-semantic.sock",
    )
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "def inspect():\n"
        "    opaque(Path(SearchEmbeddingConfig.socket))\n"
        "    opaque(Path(SearchEmbeddingConfig().socket))\n",
        config_source=mismatched,
    )

    _assert_no_concrete_socket_facts(result)
    assert result["escapes"] == []


def test_post_init_and_bound_method_use_the_exact_instance_state() -> None:
    config_source = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class SearchEmbeddingConfig:\n"
        "    socket: str = '~/.chronovisor/runtime/semantic.sock'\n"
        "    def get_socket(self):\n"
        "        return self.socket\n"
        "@dataclass\n"
        "class ResetConfig:\n"
        "    socket: str = SearchEmbeddingConfig.socket\n"
        "    def __post_init__(self):\n"
        "        self.socket = 'custom'\n"
    )
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import (\n"
        "    ResetConfig, SearchEmbeddingConfig,\n"
        ")\n"
        "def kept():\n"
        "    opaque(Path(SearchEmbeddingConfig().get_socket()))\n"
        "def reset():\n"
        "    opaque(Path(ResetConfig().socket))\n",
        config_source=config_source,
    )

    _assert_no_concrete_socket_facts(result)
    assert {(row["actor"], row["resource_id"]) for row in result["escapes"]} == {
        ("chronovisor.consumer:kept", SEMANTIC_ID),
    }


def test_explicit_init_state_is_not_shared_between_instances() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "class Holder:\n"
        "    def __init__(self, socket):\n"
        "        self.socket = socket\n"
        "def inspect():\n"
        "    tracked = Holder(SearchEmbeddingConfig.socket)\n"
        "    custom = Holder('custom')\n"
        "    opaque(Path(custom.socket))\n"
        "    opaque(Path(tracked.socket))\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert len(result["escapes"]) == 1
    assert result["escapes"][0]["resource_id"] == SEMANTIC_ID


def test_post_construction_attribute_assignments_use_strong_exact_updates() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "def generated_origin_to_custom():\n"
        "    item = SearchEmbeddingConfig()\n"
        "    item.socket = 'custom'\n"
        "    opaque(Path(item.socket))\n"
        "def generated_custom_to_origin():\n"
        "    item = SearchEmbeddingConfig(socket='custom')\n"
        "    item.socket = SearchEmbeddingConfig.socket\n"
        "    opaque(Path(item.socket))\n"
        "def class_roundtrip():\n"
        "    original = SearchEmbeddingConfig.socket\n"
        "    SearchEmbeddingConfig.socket = 'custom'\n"
        "    opaque(Path(SearchEmbeddingConfig.socket))\n"
        "    SearchEmbeddingConfig.socket = original\n"
        "    opaque(Path(SearchEmbeddingConfig.socket))\n"
        "def instance_class_separation():\n"
        "    item = SearchEmbeddingConfig()\n"
        "    item.socket = 'custom'\n"
        "    opaque(Path(SearchEmbeddingConfig.socket))\n"
        "def instance_alias_observes_update():\n"
        "    item = SearchEmbeddingConfig(socket='custom')\n"
        "    alias = item\n"
        "    item.socket = SearchEmbeddingConfig.socket\n"
        "    opaque(Path(alias.socket))\n"
        "def class_alias_observes_update():\n"
        "    config_type = SearchEmbeddingConfig\n"
        "    original = SearchEmbeddingConfig.socket\n"
        "    config_type.socket = 'custom'\n"
        "    opaque(Path(SearchEmbeddingConfig.socket))\n"
        "    config_type.socket = original\n"
        "    opaque(Path(SearchEmbeddingConfig.socket))\n"
        "def unknown_receiver_is_conservative(flag):\n"
        "    item = SearchEmbeddingConfig() if flag else opaque()\n"
        "    item.socket = 'custom'\n"
        "    opaque(Path(item.socket))\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert {row["actor"] for row in result["escapes"]} == {
        "chronovisor.consumer:generated_custom_to_origin",
        "chronovisor.consumer:class_roundtrip",
        "chronovisor.consumer:instance_class_separation",
        "chronovisor.consumer:instance_alias_observes_update",
        "chronovisor.consumer:class_alias_observes_update",
        "chronovisor.consumer:unknown_receiver_is_conservative",
    }


def test_explicit_instance_attribute_assignments_update_both_directions() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "class Holder:\n"
        "    def __init__(self, socket='custom'):\n"
        "        self.socket = socket\n"
        "def origin_to_custom():\n"
        "    item = Holder(SearchEmbeddingConfig.socket)\n"
        "    item.socket = 'custom'\n"
        "    opaque(Path(item.socket))\n"
        "def custom_to_origin():\n"
        "    item = Holder()\n"
        "    item.socket = SearchEmbeddingConfig.socket\n"
        "    opaque(Path(item.socket))\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert {row["actor"] for row in result["escapes"]} == {
        "chronovisor.consumer:custom_to_origin",
    }


def test_explicit_init_executes_for_default_and_body_origin_inputs() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "class DefaultHolder:\n"
        "    def __init__(self, socket=SearchEmbeddingConfig.socket):\n"
        "        self.socket = socket\n"
        "class BodyHolder:\n"
        "    def __init__(self):\n"
        "        self.socket = SearchEmbeddingConfig.socket\n"
        "def default_origin():\n"
        "    opaque(Path(DefaultHolder().socket))\n"
        "def default_override():\n"
        "    opaque(Path(DefaultHolder('custom').socket))\n"
        "def body_origin():\n"
        "    opaque(Path(BodyHolder().socket))\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert {row["actor"] for row in result["escapes"]} == {
        "chronovisor.consumer:default_origin",
        "chronovisor.consumer:body_origin",
    }


def test_init_var_binds_post_init_without_becoming_an_instance_field() -> None:
    config_source = (
        "from dataclasses import InitVar, KW_ONLY, dataclass\n"
        "@dataclass\n"
        "class SearchEmbeddingConfig:\n"
        "    socket: str = '~/.chronovisor/runtime/semantic.sock'\n"
        "@dataclass\n"
        "class InitVarConfig:\n"
        "    socket: str = 'custom'\n"
        "    source: InitVar[str] = SearchEmbeddingConfig.socket\n"
        "    def __post_init__(self, source):\n"
        "        self.socket = source\n"
        "@dataclass\n"
        "class KeywordInitVarConfig:\n"
        "    socket: str = 'custom'\n"
        "    _: KW_ONLY\n"
        "    source: InitVar[str] = SearchEmbeddingConfig.socket\n"
        "    def __post_init__(self, source):\n"
        "        self.socket = source\n"
        "@dataclass\n"
        "class RequiredInitVarConfig:\n"
        "    source: InitVar[str]\n"
        "    def __post_init__(self, source):\n"
        "        pass\n"
    )
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import (\n"
        "    InitVarConfig, KeywordInitVarConfig, RequiredInitVarConfig,\n"
        "    SearchEmbeddingConfig,\n"
        ")\n"
        "def default_origin():\n"
        "    opaque(Path(InitVarConfig().socket))\n"
        "def positional_override():\n"
        "    opaque(Path(InitVarConfig('custom', 'custom').socket))\n"
        "def keyword_default_origin():\n"
        "    opaque(Path(KeywordInitVarConfig().socket))\n"
        "def keyword_override():\n"
        "    opaque(Path(KeywordInitVarConfig(source='custom').socket))\n"
        "def class_default_fallback():\n"
        "    opaque(Path(InitVarConfig().source))\n"
        "def override_uses_class_default():\n"
        "    opaque(Path(InitVarConfig(source='custom').source))\n"
        "def no_default_has_no_class_fallback():\n"
        "    opaque(Path(RequiredInitVarConfig('custom').source))\n"
        "def class_mutation():\n"
        "    InitVarConfig.source = 'custom'\n"
        "    opaque(Path(InitVarConfig().source))\n"
        "    InitVarConfig.source = SearchEmbeddingConfig.socket\n"
        "    opaque(Path(InitVarConfig(source='custom').source))\n"
        "def instance_custom_shadow():\n"
        "    item = InitVarConfig()\n"
        "    item.source = 'custom'\n"
        "    opaque(Path(item.source))\n"
        "def instance_origin_shadow():\n"
        "    item = InitVarConfig(source='custom')\n"
        "    item.source = SearchEmbeddingConfig.socket\n"
        "    opaque(Path(item.source))\n"
        "def ambiguous(flag):\n"
        "    source = SearchEmbeddingConfig.socket if flag else opaque()\n"
        "    opaque(Path(InitVarConfig(source=source).socket))\n",
        config_source=config_source,
    )

    _assert_no_concrete_socket_facts(result)
    assert {row["actor"] for row in result["escapes"]} == {
        "chronovisor.consumer:default_origin",
        "chronovisor.consumer:keyword_default_origin",
        "chronovisor.consumer:class_default_fallback",
        "chronovisor.consumer:override_uses_class_default",
        "chronovisor.consumer:class_mutation",
        "chronovisor.consumer:instance_origin_shadow",
        "chronovisor.consumer:ambiguous",
    }


def test_inherited_kw_only_and_function_lambda_factories_preserve_defaults() -> None:
    config_source = (
        "from dataclasses import KW_ONLY, dataclass, field\n"
        "@dataclass\n"
        "class SearchEmbeddingConfig:\n"
        "    socket: str = '~/.chronovisor/runtime/semantic.sock'\n"
        "@dataclass\n"
        "class RerankerServiceConfig:\n"
        "    socket: str = '~/.chronovisor/runtime/reranker.sock'\n"
        "def make_service():\n"
        "    return RerankerServiceConfig()\n"
        "@dataclass\n"
        "class FunctionFactoryConfig:\n"
        "    service: RerankerServiceConfig = field(default_factory=make_service)\n"
        "@dataclass\n"
        "class LambdaFactoryConfig:\n"
        "    service: RerankerServiceConfig = field(\n"
        "        default_factory=lambda: RerankerServiceConfig()\n"
        "    )\n"
        "@dataclass\n"
        "class DerivedConfig(SearchEmbeddingConfig):\n"
        "    _: KW_ONLY\n"
        "    label: str = 'derived'\n"
    )
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import (\n"
        "    DerivedConfig, FunctionFactoryConfig, LambdaFactoryConfig,\n"
        ")\n"
        "def inherited():\n"
        "    opaque(Path(DerivedConfig().socket))\n"
        "def function_factory():\n"
        "    opaque(Path(FunctionFactoryConfig().service.socket))\n"
        "def lambda_factory():\n"
        "    opaque(Path(LambdaFactoryConfig().service.socket))\n",
        config_source=config_source,
    )

    _assert_no_concrete_socket_facts(result)
    assert {(row["actor"], row["resource_id"]) for row in result["escapes"]} == {
        ("chronovisor.consumer:inherited", SEMANTIC_ID),
        ("chronovisor.consumer:function_factory", RERANKER_ID),
        ("chronovisor.consumer:lambda_factory", RERANKER_ID),
    }


def test_extracted_path_transforms_preserve_the_receiver_origin() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "def attribute():\n"
        "    transform = Path(SearchEmbeddingConfig.socket).expanduser\n"
        "    opaque(transform())\n"
        "def reflected():\n"
        "    transform = getattr(Path(SearchEmbeddingConfig.socket), 'expanduser')\n"
        "    opaque(transform())\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert {(row["actor"], row["resource_id"]) for row in result["escapes"]} == {
        ("chronovisor.consumer:attribute", SEMANTIC_ID),
        ("chronovisor.consumer:reflected", SEMANTIC_ID),
    }


def test_main_guard_with_candidate_state_keeps_the_function_facts() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "def boot():\n"
        "    opaque(Path(SearchEmbeddingConfig.socket))\n"
        "if __name__ == '__main__':\n"
        "    boot()\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert {(row["actor"], row["resource_id"]) for row in result["escapes"]} == {
        ("chronovisor.consumer:boot", SEMANTIC_ID),
    }


def test_main_guard_local_origin_import_keeps_module_side_effects() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "configured = None\n"
        "def boot():\n"
        "    global configured\n"
        "    from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "    configured = SearchEmbeddingConfig.socket\n"
        "if __name__ == '__main__':\n"
        "    boot()\n"
        "def inspect():\n"
        "    opaque(Path(configured))\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert {(row["actor"], row["resource_id"]) for row in result["escapes"]} == {
        ("chronovisor.consumer:inspect", SEMANTIC_ID),
    }


def test_general_module_level_candidate_call_keeps_local_execution() -> None:
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "def consume(path):\n"
        "    opaque(Path(path))\n"
        "consume(SearchEmbeddingConfig.socket)\n"
    )

    _assert_no_concrete_socket_facts(result)
    assert {(row["actor"], row["resource_id"]) for row in result["escapes"]} == {
        ("chronovisor.consumer:consume", SEMANTIC_ID),
    }


def test_unknown_dataclass_base_and_factory_fail_closed_with_the_locator() -> None:
    config_source = (
        "from dataclasses import dataclass, field\n"
        "from external_factory import make_service\n"
        "class UnknownBase:\n"
        "    pass\n"
        "@dataclass\n"
        "class SearchEmbeddingConfig(UnknownBase):\n"
        "    socket: str = '~/.chronovisor/runtime/semantic.sock'\n"
        "@dataclass\n"
        "class RerankerServiceConfig:\n"
        "    socket: str = '~/.chronovisor/runtime/reranker.sock'\n"
        "@dataclass\n"
        "class RerankerConfig:\n"
        "    service: RerankerServiceConfig = field(default_factory=make_service)\n"
    )
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import (\n"
        "    RerankerConfig, SearchEmbeddingConfig,\n"
        ")\n"
        "def unknown_base():\n"
        "    opaque(Path(SearchEmbeddingConfig().socket))\n"
        "def unknown_factory():\n"
        "    opaque(Path(RerankerConfig().service.socket))\n",
        config_source=config_source,
    )

    _assert_no_concrete_socket_facts(result)
    assert {(row["actor"], row["resource_id"]) for row in result["escapes"]} == {
        ("chronovisor.consumer:unknown_base", SEMANTIC_ID),
        ("chronovisor.consumer:unknown_factory", RERANKER_ID),
    }


def test_python_313_field_signature_rejects_doc_keyword() -> None:
    config_source = (
        "from dataclasses import dataclass, field\n"
        "@dataclass\n"
        "class SearchEmbeddingConfig:\n"
        "    socket: str = field(\n"
        "        default='~/.chronovisor/runtime/semantic.sock', doc='unsupported'\n"
        "    )\n"
    )
    result = _discover(
        "from pathlib import Path\n"
        "from chronovisor.core.runtime_config import SearchEmbeddingConfig\n"
        "def inspect():\n"
        "    opaque(Path(SearchEmbeddingConfig().socket))\n",
        config_source=config_source,
    )

    _assert_no_concrete_socket_facts(result)
    assert result["escapes"] == []
