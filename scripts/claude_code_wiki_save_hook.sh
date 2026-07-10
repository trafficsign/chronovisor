#!/bin/sh
set -eu

PROJECT_DIR="/Users/trafficsign/projects/personal/llm-wiki-mcp"
RUNTIME_SOURCE="${LLM_WIKI_RUNTIME_SOURCE:-git+ssh://git@github.com/trafficsign/llm-wiki-mcp}"
export LLM_WIKI_REPO_ROOT="${LLM_WIKI_REPO_ROOT:-$PROJECT_DIR}"

uvx --from "$RUNTIME_SOURCE" llm-wiki-hook \
  --host claude-code \
  --event Stop \
  --hook \
  --only save
