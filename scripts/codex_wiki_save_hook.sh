#!/bin/sh
set -eu

PROJECT_DIR="/Users/trafficsign/projects/personal/llm-wiki-mcp"
MODEL="${CODEX_WIKI_SAVE_MODEL:-gpt-5.4-mini}"

if [ "${CODEX_WIKI_SAVE_ENABLED:-0}" != "1" ]; then
  printf '%s\n' '{"status":"disabled","reason":"CODEX_WIKI_SAVE_ENABLED=1 is required"}'
  exit 0
fi

exec uv run --project "$PROJECT_DIR" llm-wiki-codex-save --hook --save --model "$MODEL"
