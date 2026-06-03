#!/bin/sh
set -eu

PROJECT_DIR="/Users/trafficsign/projects/personal/llm-wiki-mcp"

uv run --project "$PROJECT_DIR" llm-wiki-hook \
  --host codex \
  --event Stop \
  --hook \
  --only audit
