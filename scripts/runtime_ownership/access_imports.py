"""Import binding and module re-export resolution for runtime access analysis."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Protocol

from .access_export_flow import (
    ModuleExportSummary,
    ModuleExportTable,
    StarExportPolicy,
)
from .access_model import (
    DATACLASS_INIT_VAR_MARKER,
    DATACLASS_KW_ONLY_MARKER,
    FCNTL_LOCK_FLAGS,
    OS_FLAG_OBJECT_PREFIX,
    OS_OPEN_ACCESS_FLAGS,
    OS_OPEN_MODIFIER_FLAGS,
    SQLITE_TYPE_OBJECT_PREFIX,
    STDLIB_BUILTINS_CALLS,
    STDLIB_DATACLASSES_CALLS,
    STDLIB_FCNTL_CALLS,
    STDLIB_MODULE_WILDCARD_ATTRIBUTE,
    STDLIB_OS_CALLS,
    STDLIB_PATHLIB_CALLS,
    STDLIB_SQLITE3_CALLS,
    STDLIB_SQLITE3_TYPES,
    SUPPORTED_STDLIB_MODULES,
    FlowValue,
    fcntl_lock_mask_value,
    is_precise_stdlib_module,
    precise_stdlib_module_name,
    stdlib_call_target_marker,
    stdlib_module_mutation_attributes,
    stdlib_module_mutation_marker,
    stdlib_module_state_name,
)

SymbolKey = tuple[str, str]
ExportState = tuple[
    dict[str, FlowValue],
    dict[str, set[str]],
    StarExportPolicy,
    set[str],
    set[str],
]


class ImportEngine(Protocol):
    module_exports: dict[str, dict[str, FlowValue]]
    module_star_exports: dict[str, dict[str, FlowValue]]
    module_star_definite: dict[str, frozenset[str]]
    module_star_policies: dict[str, StarExportPolicy]
    known_modules: frozenset[str]
    package_modules: frozenset[str]

    def record_dynamic_star_import(
        self,
        statement: ast.ImportFrom,
        *,
        module: str,
        actor: str,
        target_module: str,
        value: FlowValue,
    ) -> None: ...


class _ExportView:
    def __init__(
        self,
        module_exports: Mapping[str, Mapping[str, FlowValue]],
        module_star_exports: Mapping[str, Mapping[str, FlowValue]],
        module_star_definite: Mapping[str, frozenset[str]],
        module_star_policies: Mapping[str, StarExportPolicy],
        *,
        known_modules: frozenset[str],
        package_modules: frozenset[str],
    ) -> None:
        self.module_exports = {
            name: dict(values) for name, values in module_exports.items()
        }
        self.module_star_exports = {
            name: dict(values) for name, values in module_star_exports.items()
        }
        self.module_star_definite = dict(module_star_definite)
        self.module_star_policies = dict(module_star_policies)
        self.known_modules = known_modules
        self.package_modules = package_modules

    def record_dynamic_star_import(
        self,
        statement: ast.ImportFrom,
        *,
        module: str,
        actor: str,
        target_module: str,
        value: FlowValue,
    ) -> None:
        return


def resolve_import_from(
    module: str,
    *,
    level: int,
    imported_module: str | None,
    is_package: bool,
) -> str:
    if level == 0:
        return imported_module or ""
    package = module.split(".") if is_package else module.split(".")[:-1]
    parent_hops = level - 1
    if parent_hops > len(package):
        return imported_module or ""
    prefix = package[: len(package) - parent_hops]
    if imported_module:
        prefix.extend(imported_module.split("."))
    return ".".join(prefix)


def bind_import_statement(
    engine: ImportEngine,
    statement: ast.Import | ast.ImportFrom,
    *,
    module: str,
    actor: str,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    definite_names: set[str] | None = None,
) -> None:
    if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
        return
    if isinstance(statement, ast.Import):
        for alias in statement.names:
            local = alias.asname or alias.name.split(".")[0]
            imported = alias.name if alias.asname else alias.name.split(".")[0]
            value = _module_import_value(
                env,
                imported=imported,
                known_modules=engine.known_modules,
            )
            _strong_bind(
                env,
                object_env,
                local,
                value,
                step=f"import:{actor}:{local}->{alias.name}",
            )
            if definite_names is not None:
                definite_names.add(local)
        return

    target_module = resolve_import_from(
        module,
        level=statement.level,
        imported_module=statement.module,
        is_package=module in engine.package_modules,
    )
    for alias in statement.names:
        if alias.name == "*":
            star_values = engine.module_star_exports.get(target_module, {})
            star_definite = engine.module_star_definite.get(target_module, frozenset())
            policy = engine.module_star_policies.get(
                target_module, StarExportPolicy.public()
            )
            dynamic = policy.kind == "dynamic"
            implicated = _dynamic_star_value(
                star_values,
                env,
                target_module=target_module,
            )
            for local in sorted(star_values.keys() | star_definite):
                value = star_values.get(local, FlowValue())
                step = f"import:{actor}:{local}->{target_module}:*"
                if dynamic or local not in star_definite:
                    _weak_bind(env, object_env, local, value, step=step)
                else:
                    _strong_bind(env, object_env, local, value, step=step)
                    if definite_names is not None:
                        definite_names.add(local)
            if dynamic:
                engine.record_dynamic_star_import(
                    statement,
                    module=module,
                    actor=actor,
                    target_module=target_module,
                    value=implicated,
                )
            continue
        local = alias.asname or alias.name
        value = engine.module_exports.get(target_module, {}).get(
            alias.name, FlowValue()
        )
        child_module = f"{target_module}.{alias.name}" if target_module else alias.name
        stdlib_os = target_module == "os" and "os" not in engine.known_modules
        stdlib_sqlite3 = (
            target_module == "sqlite3"
            and "sqlite3" not in engine.known_modules
        )
        stdlib_fcntl = (
            target_module == "fcntl" and "fcntl" not in engine.known_modules
        )
        stdlib_builtins = (
            target_module == "builtins"
            and "builtins" not in engine.known_modules
        )
        stdlib_dataclasses = (
            target_module == "dataclasses"
            and "dataclasses" not in engine.known_modules
        )
        stdlib_pathlib = (
            target_module == "pathlib"
            and "pathlib" not in engine.known_modules
        )
        stdlib_os_attribute = stdlib_os and _stdlib_attribute_is_unmutated(
            env, module="os", attribute=alias.name
        )
        stdlib_sqlite3_attribute = (
            stdlib_sqlite3
            and _stdlib_attribute_is_unmutated(
                env, module="sqlite3", attribute=alias.name
            )
        )
        stdlib_fcntl_attribute = stdlib_fcntl and _stdlib_attribute_is_unmutated(
            env, module="fcntl", attribute=alias.name
        )
        stdlib_builtins_attribute = (
            stdlib_builtins
            and _stdlib_attribute_is_unmutated(
                env, module="builtins", attribute=alias.name
            )
        )
        stdlib_dataclasses_attribute = (
            stdlib_dataclasses
            and _stdlib_attribute_is_unmutated(
                env, module="dataclasses", attribute=alias.name
            )
        )
        stdlib_pathlib_attribute = (
            stdlib_pathlib
            and _stdlib_attribute_is_unmutated(
                env, module="pathlib", attribute=alias.name
            )
        )
        if stdlib_builtins_attribute and alias.name in STDLIB_BUILTINS_CALLS:
            value = FlowValue(call_targets={f"builtins:{alias.name}"})
        elif (
            stdlib_dataclasses_attribute
            and alias.name in STDLIB_DATACLASSES_CALLS
        ):
            value = FlowValue(call_targets={f"dataclasses:{alias.name}"})
        elif stdlib_dataclasses_attribute and alias.name == "KW_ONLY":
            value = FlowValue(object_types={DATACLASS_KW_ONLY_MARKER})
        elif stdlib_dataclasses_attribute and alias.name == "InitVar":
            value = FlowValue(object_types={DATACLASS_INIT_VAR_MARKER})
        elif stdlib_fcntl_attribute and alias.name in STDLIB_FCNTL_CALLS:
            value = FlowValue(call_targets={f"fcntl:{alias.name}"})
        elif stdlib_os_attribute and alias.name in STDLIB_OS_CALLS:
            value = FlowValue(call_targets={f"os:{alias.name}"})
        elif stdlib_pathlib_attribute and alias.name in STDLIB_PATHLIB_CALLS:
            value = FlowValue(call_targets={f"pathlib:{alias.name}"})
        elif stdlib_sqlite3_attribute and alias.name in STDLIB_SQLITE3_CALLS:
            value = FlowValue(call_targets={f"sqlite3:{alias.name}"})
        elif stdlib_sqlite3_attribute and alias.name in STDLIB_SQLITE3_TYPES:
            value = FlowValue(
                object_types={f"{SQLITE_TYPE_OBJECT_PREFIX}{alias.name}"}
            )
        elif stdlib_os_attribute and alias.name in (
            OS_OPEN_ACCESS_FLAGS | OS_OPEN_MODIFIER_FLAGS
        ):
            value = FlowValue(
                object_types={f"{OS_FLAG_OBJECT_PREFIX}{alias.name}"}
            )
        elif stdlib_fcntl_attribute and alias.name in FCNTL_LOCK_FLAGS:
            value = fcntl_lock_mask_value(alias.name)
        elif stdlib_builtins and alias.name in STDLIB_BUILTINS_CALLS:
            value = FlowValue(
                object_types={
                    stdlib_call_target_marker("builtins", alias.name)
                },
                unknown_callable=True,
            )
        elif stdlib_fcntl and alias.name in STDLIB_FCNTL_CALLS:
            value = FlowValue(
                object_types={stdlib_call_target_marker("fcntl", alias.name)},
                unknown_callable=True,
            )
        elif stdlib_os and alias.name in STDLIB_OS_CALLS:
            value = FlowValue(
                object_types={stdlib_call_target_marker("os", alias.name)},
                unknown_callable=True,
            )
        elif stdlib_sqlite3 and alias.name in STDLIB_SQLITE3_CALLS:
            value = FlowValue(
                object_types={
                    stdlib_call_target_marker("sqlite3", alias.name)
                },
                unknown_callable=True,
            )
        elif (
            not value.has_origins
            and not value.module_refs
            and child_module in engine.known_modules
        ):
            value = FlowValue(module_refs={child_module})
        elif target_module not in engine.known_modules:
            value = FlowValue(unknown_callable=True)
        _strong_bind(
            env,
            object_env,
            local,
            value,
            step=f"import:{actor}:{local}->{target_module}:{alias.name}",
        )
        if definite_names is not None:
            definite_names.add(local)


def _module_import_value(
    env: Mapping[str, FlowValue],
    *,
    imported: str,
    known_modules: frozenset[str],
) -> FlowValue:
    value = FlowValue(module_refs={imported})
    if imported not in SUPPORTED_STDLIB_MODULES or imported in known_modules:
        return value
    state_name = stdlib_module_state_name(imported)
    candidates = [
        candidate
        for name, candidate in env.items()
        if name == state_name or candidate.module_refs == {imported}
    ]
    for candidate in candidates:
        for attribute in stdlib_module_mutation_attributes(
            candidate, module=imported
        ):
            value.object_types.add(
                stdlib_module_mutation_marker(imported, attribute)
            )
    return value


def _stdlib_attribute_is_unmutated(
    env: Mapping[str, FlowValue],
    *,
    module: str,
    attribute: str,
) -> bool:
    marker = stdlib_module_mutation_marker(module, attribute)
    wildcard_marker = stdlib_module_mutation_marker(
        module, STDLIB_MODULE_WILDCARD_ATTRIBUTE
    )
    return not any(
        value.module_refs == {module}
        and bool({marker, wildcard_marker}.intersection(value.object_types))
        for value in env.values()
    )


def build_module_exports(
    trees: Mapping[str, ast.Module],
    *,
    package_modules: frozenset[str],
    known_modules: frozenset[str],
    origin_symbols: Mapping[SymbolKey, FlowValue],
) -> ModuleExportTable:
    dependencies = _export_dependencies(
        trees,
        package_modules=package_modules,
        known_modules=known_modules,
    )
    _reject_origin_cycles(dependencies, frozenset(origin_symbols))

    summaries = {module: ModuleExportSummary.empty() for module in trees}
    while True:
        exports = {module: summary.bindings for module, summary in summaries.items()}
        star_exports = {
            module: summary.star_bindings for module, summary in summaries.items()
        }
        star_definite = {
            module: summary.star_definite for module, summary in summaries.items()
        }
        star_policies = {
            module: summary.star_policy for module, summary in summaries.items()
        }
        evaluated_summaries = {
            module: _evaluate_module_exports(
                module,
                tree,
                exports=exports,
                star_exports=star_exports,
                star_definite=star_definite,
                star_policies=star_policies,
                package_modules=package_modules,
                known_modules=known_modules,
                origin_symbols=origin_symbols,
            )
            for module, tree in sorted(trees.items())
        }
        next_summaries = {
            module: _merge_module_export_summary(
                summaries[module], evaluated_summaries[module]
            )
            for module in sorted(trees)
        }
        if next_summaries == summaries:
            return ModuleExportTable(next_summaries)
        summaries = next_summaries


def resolve_module_attribute(
    value: FlowValue,
    attribute: str,
    *,
    module_exports: Mapping[str, Mapping[str, FlowValue]],
    known_modules: frozenset[str],
    step: str,
) -> FlowValue:
    resolved = FlowValue()
    for module_ref in sorted(value.module_refs):
        exported = module_exports.get(module_ref, {}).get(attribute)
        if exported is not None:
            resolved = resolved.merged(exported.bound(step))
        child_module = f"{module_ref}.{attribute}"
        if child_module in known_modules:
            resolved = resolved.merged(FlowValue(module_refs={child_module}))
        if module_ref in known_modules:
            continue
        precise_module = precise_stdlib_module_name(value)
        if precise_module != module_ref:
            continue
        if is_precise_stdlib_module(
            value,
            module=module_ref,
            attribute=attribute,
        ):
            if module_ref == "fcntl" and attribute in FCNTL_LOCK_FLAGS:
                resolved = resolved.merged(fcntl_lock_mask_value(attribute))
            elif module_ref == "os" and attribute in (
                OS_OPEN_ACCESS_FLAGS | OS_OPEN_MODIFIER_FLAGS
            ):
                resolved = resolved.merged(
                    FlowValue(
                        object_types={f"{OS_FLAG_OBJECT_PREFIX}{attribute}"}
                    )
                )
            elif module_ref == "sqlite3" and attribute in STDLIB_SQLITE3_TYPES:
                resolved = resolved.merged(
                    FlowValue(
                        object_types={f"{SQLITE_TYPE_OBJECT_PREFIX}{attribute}"}
                    )
                )
            elif module_ref == "dataclasses" and attribute == "KW_ONLY":
                resolved = resolved.merged(
                    FlowValue(object_types={DATACLASS_KW_ONLY_MARKER})
                )
            elif module_ref == "dataclasses" and attribute == "InitVar":
                resolved = resolved.merged(
                    FlowValue(object_types={DATACLASS_INIT_VAR_MARKER})
                )
            elif _is_supported_stdlib_call(module_ref, attribute):
                resolved = resolved.merged(
                    FlowValue(call_targets={f"{module_ref}:{attribute}"})
                )
        elif _is_supported_stdlib_call(module_ref, attribute):
            resolved = resolved.merged(
                FlowValue(
                    object_types={
                        stdlib_call_target_marker(module_ref, attribute)
                    },
                    unknown_callable=True,
                )
            )
    return resolved


def _is_supported_stdlib_call(module: str, attribute: str) -> bool:
    calls = {
        "builtins": STDLIB_BUILTINS_CALLS,
        "dataclasses": STDLIB_DATACLASSES_CALLS,
        "fcntl": STDLIB_FCNTL_CALLS,
        "os": STDLIB_OS_CALLS,
        "pathlib": STDLIB_PATHLIB_CALLS,
        "sqlite3": STDLIB_SQLITE3_CALLS,
    }
    return attribute in calls.get(module, frozenset())


def _merge_module_export_summary(
    previous: ModuleExportSummary,
    evaluated: ModuleExportSummary,
) -> ModuleExportSummary:
    bindings: dict[str, FlowValue] = {}
    for name in previous.bindings.keys() | evaluated.bindings.keys():
        prior_value = previous.bindings.get(name)
        evaluated_value = evaluated.bindings.get(name)
        if prior_value is None:
            assert evaluated_value is not None
            bindings[name] = evaluated_value.copy()
        elif evaluated_value is None:
            bindings[name] = prior_value.copy()
        else:
            bindings[name] = _merge_fixed_point_value(prior_value, evaluated_value)
    definite_bindings = previous.definite_bindings | evaluated.definite_bindings
    policy = evaluated.star_policy
    return ModuleExportSummary(
        bindings,
        definite_bindings,
        policy.select(bindings),
        policy.select_definite_names(definite_bindings),
        policy,
    )


def _merge_fixed_point_value(previous: FlowValue, evaluated: FlowValue) -> FlowValue:
    merged = previous.merged(evaluated)
    for resource_id, paths in merged.origins.items():
        shortest = min(len(path) for path in paths)
        merged.origins[resource_id] = frozenset(
            path for path in paths if len(path) == shortest
        )
    return merged


def _evaluate_module_exports(
    module: str,
    tree: ast.Module,
    *,
    exports: Mapping[str, Mapping[str, FlowValue]],
    star_exports: Mapping[str, Mapping[str, FlowValue]],
    star_definite: Mapping[str, frozenset[str]],
    star_policies: Mapping[str, StarExportPolicy],
    package_modules: frozenset[str],
    known_modules: frozenset[str],
    origin_symbols: Mapping[SymbolKey, FlowValue],
) -> ModuleExportSummary:
    view = _ExportView(
        exports,
        star_exports,
        star_definite,
        star_policies,
        known_modules=known_modules,
        package_modules=package_modules,
    )

    def evaluate_block(statements: list[ast.stmt], state: ExportState) -> ExportState:
        env, object_env, star_policy, definite_names, deleted_names = state
        for statement in statements:
            if isinstance(statement, ast.If):
                incoming: ExportState = (
                    env,
                    object_env,
                    star_policy,
                    definite_names,
                    deleted_names,
                )
                body = evaluate_block(statement.body, _copy_export_state(incoming))
                orelse = evaluate_block(
                    statement.orelse,
                    _copy_export_state(incoming),
                )
                (
                    env,
                    object_env,
                    star_policy,
                    definite_names,
                    deleted_names,
                ) = _join_export_states([body, orelse])
            elif isinstance(statement, ast.Match):
                incoming = (
                    env,
                    object_env,
                    star_policy,
                    definite_names,
                    deleted_names,
                )
                case_states = [
                    evaluate_block(case.body, _copy_export_state(incoming))
                    for case in statement.cases
                ]
                final_case = statement.cases[-1]
                if final_case.guard is not None or not _is_irrefutable_match_pattern(
                    final_case.pattern
                ):
                    case_states.append(_copy_export_state(incoming))
                (
                    env,
                    object_env,
                    star_policy,
                    definite_names,
                    deleted_names,
                ) = _join_export_states(case_states)
            elif isinstance(statement, (ast.Try, ast.TryStar)):
                incoming = (
                    env,
                    object_env,
                    star_policy,
                    definite_names,
                    deleted_names,
                )
                exceptional_prefixes = [_copy_export_state(incoming)]
                normal_state = _copy_export_state(incoming)
                for body_statement in statement.body:
                    if _module_statement_may_raise(body_statement):
                        exceptional_prefixes.append(_copy_export_state(normal_state))
                    normal_state = evaluate_block([body_statement], normal_state)
                if statement.orelse:
                    normal_state = evaluate_block(statement.orelse, normal_state)

                outcomes = [normal_state]
                if statement.handlers:
                    exceptional_state = _join_export_states(exceptional_prefixes)
                    for handler in statement.handlers:
                        handler_state = _copy_export_state(exceptional_state)
                        if handler.name is not None:
                            (
                                handler_env,
                                handler_object_env,
                                _handler_policy,
                                handler_definite,
                                _handler_deleted,
                            ) = handler_state
                            _strong_bind(
                                handler_env,
                                handler_object_env,
                                handler.name,
                                FlowValue(),
                                step=f"except:{module}:<module>:{handler.name}",
                            )
                            handler_definite.add(handler.name)
                        handler_outcome = evaluate_block(handler.body, handler_state)
                        if handler.name is not None:
                            (
                                handler_env,
                                handler_object_env,
                                _handler_policy,
                                handler_definite,
                                handler_deleted,
                            ) = handler_outcome
                            handler_env.pop(handler.name, None)
                            handler_object_env.pop(handler.name, None)
                            handler_definite.discard(handler.name)
                            handler_deleted.add(handler.name)
                        outcomes.append(handler_outcome)
                if statement.finalbody:
                    outcomes = [
                        evaluate_block(
                            statement.finalbody,
                            _copy_export_state(outcome),
                        )
                        for outcome in outcomes
                    ]
                (
                    env,
                    object_env,
                    star_policy,
                    definite_names,
                    deleted_names,
                ) = _join_export_states(outcomes)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                bind_import_statement(
                    view,
                    statement,
                    module=module,
                    actor=f"{module}:<module>",
                    env=env,
                    object_env=object_env,
                    definite_names=definite_names,
                )
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value_node = statement.value
                if value_node is None:
                    continue
                value = _evaluate_export_expression(
                    value_node,
                    env=env,
                    exports=exports,
                    known_modules=known_modules,
                    step=f"module-alias:{module}",
                )
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                for target in targets:
                    for name in _target_names(target):
                        if name == "__all__":
                            star_policy = StarExportPolicy.from_all_expression(
                                value_node
                            )
                        direct = origin_symbols.get((module, name))
                        _strong_bind(
                            env,
                            object_env,
                            name,
                            direct.copy() if direct is not None else value,
                            step=f"alias:{module}:<module>:{name}",
                        )
                        definite_names.add(name)
            elif isinstance(statement, ast.AugAssign):
                for name in _target_names(statement.target):
                    if name == "__all__":
                        added_policy = StarExportPolicy.from_all_expression(
                            statement.value
                        )
                        if (
                            isinstance(statement.op, ast.Add)
                            and star_policy.kind == "static"
                            and added_policy.kind == "static"
                        ):
                            star_policy = StarExportPolicy(
                                "static",
                                (*star_policy.names, *added_policy.names),
                            )
                        else:
                            star_policy = StarExportPolicy("dynamic")
                    _strong_bind(
                        env,
                        object_env,
                        name,
                        FlowValue(),
                        step=f"augassign:{module}:<module>:{name}",
                    )
                    definite_names.add(name)
            elif isinstance(statement, ast.Delete):
                for target in statement.targets:
                    for name in _target_names(target):
                        env.pop(name, None)
                        object_env.pop(name, None)
                        definite_names.discard(name)
                        deleted_names.add(name)
                        if name == "__all__":
                            star_policy = StarExportPolicy.public()
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _strong_bind(
                    env,
                    object_env,
                    statement.name,
                    FlowValue(
                        call_targets={f"{module}:{statement.name}"},
                        unknown_callable=bool(statement.decorator_list),
                    ),
                    step=f"definition:{module}:<module>:{statement.name}",
                )
                definite_names.add(statement.name)
            elif isinstance(statement, ast.ClassDef):
                class_ref = f"{module}:{statement.name}"
                _strong_bind(
                    env,
                    object_env,
                    statement.name,
                    FlowValue(
                        object_types={class_ref},
                        class_targets={class_ref},
                        unknown_callable=bool(statement.decorator_list),
                    ),
                    step=f"definition:{module}:<module>:{statement.name}",
                )
                definite_names.add(statement.name)
        return env, object_env, star_policy, definite_names, deleted_names

    env, _object_env, star_policy, definite_names, deleted_names = evaluate_block(
        tree.body,
        ({}, {}, StarExportPolicy.public(), set(), set()),
    )
    for (origin_module, symbol), value in origin_symbols.items():
        if (
            origin_module == module
            and symbol not in env
            and symbol not in deleted_names
        ):
            env[symbol] = value.copy()
            definite_names.add(symbol)
    return ModuleExportSummary(
        env,
        frozenset(definite_names),
        star_policy.select(env),
        star_policy.select_definite_names(definite_names),
        star_policy,
    )


def _copy_export_state(state: ExportState) -> ExportState:
    env, object_env, star_policy, definite_names, deleted_names = state
    return (
        {name: value.copy() for name, value in env.items()},
        {name: set(values) for name, values in object_env.items()},
        star_policy,
        set(definite_names),
        set(deleted_names),
    )


def _is_irrefutable_match_pattern(pattern: ast.pattern) -> bool:
    if not isinstance(pattern, ast.MatchAs):
        return False
    return pattern.pattern is None or _is_irrefutable_match_pattern(pattern.pattern)


def _module_statement_may_raise(statement: ast.stmt) -> bool:
    return not isinstance(statement, ast.Pass)


def _join_export_states(states: list[ExportState]) -> ExportState:
    env: dict[str, FlowValue] = {}
    names = set().union(
        *(state_env for state_env, _objects, _policy, _definite, _deleted in states)
    )
    for name in names:
        value = FlowValue()
        for state_env, _objects, _policy, _definite, _deleted in states:
            if name in state_env:
                value = value.merged(state_env[name])
        env[name] = value
    object_env = {
        name: set(value.object_types)
        for name, value in env.items()
        if value.object_types
    }
    definite_names = set(states[0][3])
    deleted_names = set(states[0][4])
    for (
        _env,
        _objects,
        _policy,
        state_definite_names,
        state_deleted_names,
    ) in states[1:]:
        definite_names.intersection_update(state_definite_names)
        deleted_names.intersection_update(state_deleted_names)
    return (
        env,
        object_env,
        StarExportPolicy.joined(
            policy for _env, _objects, policy, _definite, _deleted in states
        ),
        definite_names,
        deleted_names,
    )


def _evaluate_export_expression(
    node: ast.expr,
    *,
    env: Mapping[str, FlowValue],
    exports: Mapping[str, Mapping[str, FlowValue]],
    known_modules: frozenset[str],
    step: str,
) -> FlowValue:
    if isinstance(node, ast.Name):
        return env.get(node.id, FlowValue()).copy()
    if isinstance(node, ast.Attribute):
        base = _evaluate_export_expression(
            node.value,
            env=env,
            exports=exports,
            known_modules=known_modules,
            step=step,
        )
        return resolve_module_attribute(
            base,
            node.attr,
            module_exports=exports,
            known_modules=known_modules,
            step=f"{step}.{node.attr}",
        )
    return FlowValue()


def _strong_bind(
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    name: str,
    value: FlowValue,
    *,
    step: str,
) -> None:
    env[name] = value.bound(step)
    if value.object_types:
        object_env[name] = set(value.object_types)
    else:
        object_env.pop(name, None)


def _weak_bind(
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    name: str,
    value: FlowValue,
    *,
    step: str,
) -> None:
    env[name] = env.get(name, FlowValue()).merged(value.bound(step))
    if env[name].object_types:
        object_env[name] = set(env[name].object_types)
    else:
        object_env.pop(name, None)


def _dynamic_star_value(
    star_values: Mapping[str, FlowValue],
    existing: Mapping[str, FlowValue],
    *,
    target_module: str,
) -> FlowValue:
    resource_ids = {
        resource_id
        for value in (*star_values.values(), *existing.values())
        for resource_id in value.origins
    }
    marker = (f"dynamic-star-import:{target_module}",)
    return FlowValue(
        {resource_id: frozenset({marker}) for resource_id in sorted(resource_ids)}
    )


def _target_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _target_names(item)]
    return []


def _export_dependencies(
    trees: Mapping[str, ast.Module],
    *,
    package_modules: frozenset[str],
    known_modules: frozenset[str],
) -> dict[SymbolKey, SymbolKey]:
    dependencies: dict[SymbolKey, SymbolKey] = {}
    for module, tree in sorted(trees.items()):
        module_aliases: dict[str, str] = {}
        symbol_aliases: dict[str, SymbolKey] = {}
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    local = alias.asname or alias.name.split(".")[0]
                    module_aliases[local] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
                    symbol_aliases.pop(local, None)
                    dependencies.pop((module, local), None)
            elif isinstance(statement, ast.ImportFrom):
                if statement.module == "__future__":
                    continue
                target_module = resolve_import_from(
                    module,
                    level=statement.level,
                    imported_module=statement.module,
                    is_package=module in package_modules,
                )
                for alias in statement.names:
                    if alias.name == "*":
                        continue
                    local = alias.asname or alias.name
                    child = (
                        f"{target_module}.{alias.name}" if target_module else alias.name
                    )
                    if child in known_modules:
                        module_aliases[local] = child
                        symbol_aliases.pop(local, None)
                        dependencies.pop((module, local), None)
                    else:
                        source = (target_module, alias.name)
                        symbol_aliases[local] = source
                        module_aliases.pop(local, None)
                        dependencies[(module, local)] = source
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                if statement.value is None:
                    continue
                assignment_source = _dependency_expression(
                    statement.value,
                    module=module,
                    module_aliases=module_aliases,
                    symbol_aliases=symbol_aliases,
                )
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                for target in targets:
                    for name in _target_names(target):
                        module_aliases.pop(name, None)
                        if assignment_source is None:
                            symbol_aliases.pop(name, None)
                            dependencies.pop((module, name), None)
                        else:
                            symbol_aliases[name] = assignment_source
                            dependencies[(module, name)] = assignment_source
    return dependencies


def _dependency_expression(
    node: ast.expr,
    *,
    module: str,
    module_aliases: Mapping[str, str],
    symbol_aliases: Mapping[str, SymbolKey],
) -> SymbolKey | None:
    if isinstance(node, ast.Name):
        return symbol_aliases.get(node.id, (module, node.id))
    if not isinstance(node, ast.Attribute):
        return None
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name) or not parts:
        return None
    base_module = module_aliases.get(current.id)
    if base_module is None:
        return None
    parts.reverse()
    if len(parts) == 1:
        return base_module, parts[0]
    return ".".join([base_module, *parts[:-1]]), parts[-1]


def _reject_origin_cycles(
    dependencies: Mapping[SymbolKey, SymbolKey], origins: frozenset[SymbolKey]
) -> None:
    for origin in sorted(origins):
        path: list[SymbolKey] = []
        positions: dict[SymbolKey, int] = {}
        current = origin
        while current in dependencies:
            if current in positions:
                cycle = [*path[positions[current] :], current]
                rendered = " -> ".join(f"{module}:{symbol}" for module, symbol in cycle)
                raise ValueError(f"runtime access import cycle: {rendered}")
            positions[current] = len(path)
            path.append(current)
            current = dependencies[current]


__all__ = [
    "ImportEngine",
    "bind_import_statement",
    "build_module_exports",
    "resolve_import_from",
    "resolve_module_attribute",
]
