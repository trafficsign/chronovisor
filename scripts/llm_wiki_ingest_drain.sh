#!/bin/sh
set -eu

PROJECT_DIR="/Users/trafficsign/projects/personal/llm-wiki-mcp"
PATH="/Users/trafficsign/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/opt/homebrew/sbin:/usr/sbin:/sbin"
export PATH

exec uv run --project "$PROJECT_DIR" llm-wiki-ingest-drain "$@"
