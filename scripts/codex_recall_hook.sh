#!/bin/sh
set -eu

PROJECT_DIR="/Users/trafficsign/projects/personal/llm-wiki-mcp"

if [ "${LLM_WIKI_RECALL_ENABLED:-1}" = "0" ]; then
  printf '%s\n' '{}'
  exit 0
fi

uv run --project "$PROJECT_DIR" llm-wiki-recall \
  --host codex \
  --event UserPromptSubmit \
  --hook \
  --format codex
