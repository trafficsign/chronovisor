#!/bin/sh
set -eu

PROJECT_DIR="/Users/trafficsign/projects/personal/llm-wiki-mcp"
MODEL="${CLAUDE_CODE_WIKI_SAVE_MODEL:-}"

if [ "${CLAUDE_CODE_WIKI_SAVE_ENABLED:-0}" != "1" ]; then
  printf '%s\n' '{"status":"disabled","reason":"CLAUDE_CODE_WIKI_SAVE_ENABLED=1 is required"}'
  exit 0
fi

cmd="llm-wiki-claude-code-save --hook --save"
[ -n "$MODEL" ] && cmd="$cmd --model $MODEL"

exec uv run --project "$PROJECT_DIR" $cmd
