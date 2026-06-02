#!/bin/sh
set -eu

PROJECT_DIR="/Users/trafficsign/projects/personal/llm-wiki-mcp"

if [ "${LLM_WIKI_RECALL_AUDIT_ENABLED:-0}" != "1" ]; then
  printf '%s\n' '{}'
  exit 0
fi

STDIN_DATA="$(cat)"
LOG_DIR="${HOME}/.wiki/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/codex-recall-audit-$(date +%Y%m%d).log"

printf '%s\n' "$STDIN_DATA" \
  | nohup uv run --project "$PROJECT_DIR" llm-wiki-recall-audit --host codex --hook \
  >> "$LOG_FILE" 2>&1 &

printf '%s\n' '{}'
