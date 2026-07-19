#!/bin/sh
set -eu

PROJECT_DIR="/Users/trafficsign/projects/personal/chronovisor"
RUNTIME_SOURCE="${CHRONOVISOR_RUNTIME_SOURCE:-git+ssh://git@github.com/trafficsign/chronovisor}"
export CHRONOVISOR_REPO_ROOT="${CHRONOVISOR_REPO_ROOT:-$PROJECT_DIR}"

uvx --from "$RUNTIME_SOURCE" chronovisor-hook \
  --host codex \
  --event Stop \
  --hook \
  --only save
