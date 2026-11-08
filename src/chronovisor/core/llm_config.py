"""Compose the local embedding and search rerank routes from existing config."""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

from chronovisor.core.decision_mlx import DecisionMLXBackend, DecisionModelConfig
from chronovisor.core.llm_runtime import (
    EmbeddingBackend,
    EmbeddingRoute,
    LLMRuntime,
    RerankRoute,
)
from chronovisor.core.nemotron_adapter import NemotronEmbeddingBackend
from chronovisor.core.runtime_config import load_search_embedding_config, load_toml_file


class LLMConfigError(ValueError):
    """A retained local model route is invalid."""


def load_llm_runtime(path: Path | str | None = None) -> LLMRuntime:
    data = load_toml_file(path)
    llm = data.get("llm", {})
    if not isinstance(llm, dict):
        raise LLMConfigError("invalid local model configuration")
    roles, providers = llm.get("roles", {}), llm.get("providers", {})
    if not isinstance(roles, dict) or not isinstance(providers, dict):
        raise LLMConfigError("invalid local model configuration")
    config = load_search_embedding_config(path)
    embedding: dict[str, EmbeddingRoute] = {}
    rerank: dict[str, RerankRoute] = {}
    mlx_backends: dict[str, EmbeddingBackend] = {}
    for name in (
        "search.semantic.foreground",
        "search.semantic.incremental",
        "search.rerank",
    ):
        role = roles.get(name)
        if role is None:
            continue
        if not isinstance(role, dict):
            raise LLMConfigError(f"invalid role: {name}")
        provider_id, model = role.get("provider"), role.get("model")
        if (
            not isinstance(provider_id, str)
            or not isinstance(model, str)
            or not model.strip()
        ):
            raise LLMConfigError(f"invalid role: {name}")
        provider = providers.get(provider_id)
        if not isinstance(provider, dict):
            raise LLMConfigError(f"missing local provider for {name}")
        if name == "search.rerank":
            if (
                role.get("capability") != "rerank"
                or provider.get("kind") != "local-decision-mlx"
            ):
                raise LLMConfigError("search.rerank requires local StartLux")
            folder = provider.get("path")
            revision = provider.get("revision")
            max_chars = provider.get("max_chars", DecisionModelConfig.max_chars)
            if (
                not isinstance(folder, str)
                or not folder.strip()
                or any(ord(ch) < 32 for ch in folder)
                or type(max_chars) is not int
                or max_chars <= 0
                or (
                    revision is not None
                    and (
                        not isinstance(revision, str)
                        or re.fullmatch(r"[0-9a-f]{64}", revision) is None
                    )
                )
            ):
                raise LLMConfigError("invalid StartLux artifact")
            backend = DecisionMLXBackend(
                DecisionModelConfig(str(Path(folder).expanduser()), max_chars, revision)
            )
            rerank[name] = RerankRoute(backend, model, revision)
        else:
            expected = (
                config.query_device
                if name.endswith("foreground")
                else config.incremental_device
            )
            device = provider.get("device")
            if (
                role.get("capability") != "embedding"
                or provider.get("kind") != "nemotron"
                or device not in {"mlx", "cpu", "mps"}
                or device != expected
            ):
                raise LLMConfigError(f"invalid local embedding route: {name}")
            if device == "mlx":
                from chronovisor.core.nemotron_mlx import NemotronMLXBackend

                if model not in mlx_backends:
                    mlx_backends[model] = NemotronMLXBackend(config, model=model)
                embedding_backend = mlx_backends[model]
            else:
                embedding_backend = NemotronEmbeddingBackend(
                    config,
                    model=model,
                    device=device,
                    incremental=name.endswith("incremental"),
                )
            embedding[name] = EmbeddingRoute(embedding_backend, model)
    return LLMRuntime(embedding=embedding, rerank=rerank)


@cache
def load_default_llm_runtime() -> LLMRuntime:
    return load_llm_runtime()
