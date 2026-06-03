#!/bin/sh
set -eu

PROJECT_DIR="/Users/trafficsign/projects/personal/llm-wiki-mcp"

uv run --project "$PROJECT_DIR" llm-wiki-hook \
  --host claude-code \
  --event UserPromptSubmit \
  --hook
