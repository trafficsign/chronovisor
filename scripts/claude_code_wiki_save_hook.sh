#!/bin/sh
set -eu

PROJECT_DIR="/Users/trafficsign/projects/personal/llm-wiki-mcp"

if [ "${CLAUDE_CODE_WIKI_SAVE_ENABLED:-0}" != "1" ]; then
  printf '%s\n' '{}'
  exit 0
fi

STDIN_DATA="$(cat)"
LOG_DIR="${HOME}/.wiki/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/claude-code-save-$(date +%Y%m%d).log"

printf '%s\n' "$STDIN_DATA" \
  | nohup uv run --project "$PROJECT_DIR" llm-wiki-claude-code-save --hook --save --trigger-ingest \
  >> "$LOG_FILE" 2>&1 &

printf '%s\n' '{}'
