#!/bin/sh
set -eu

PROJECT_DIR="/Users/trafficsign/projects/personal/llm-wiki-mcp"
MODEL="${CODEX_WIKI_SAVE_MODEL:-gpt-5.4-mini}"

if [ "${CODEX_WIKI_SAVE_ENABLED:-0}" != "1" ]; then
  printf '%s\n' '{"status":"disabled","reason":"CODEX_WIKI_SAVE_ENABLED=1 is required"}'
  exit 0
fi

STDIN_DATA="$(cat)"
LOG_DIR="${HOME}/.wiki/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/codex-save-$(date +%Y%m%d).log"

printf '%s\n' "$STDIN_DATA" \
  | nohup uv run --project "$PROJECT_DIR" llm-wiki-codex-save --hook --save --model "$MODEL" \
  >> "$LOG_FILE" 2>&1 &

printf '%s\n' '{"status":"launched","pid":'"$!"'}'
