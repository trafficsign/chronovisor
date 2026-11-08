"""Chronovisor directory management."""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".chronovisor"


def resolve_root() -> Path:
    """Return the one authoritative Chronovisor data root."""

    configured = os.environ.get("CHRONOVISOR_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve(strict=False)

    return DEFAULT_ROOT


@dataclass(frozen=True)
class RuntimeContext:
    """Immutable paths for one Chronovisor data root."""

    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def pages_dir(self) -> Path:
        return self.root / "pages"

    @property
    def system_dir(self) -> Path:
        return self.root / "system"

    @property
    def config_file(self) -> Path:
        return self.root / "config.toml"

    @property
    def index_file(self) -> Path:
        return self.pages_dir / "index.md"

    @property
    def log_file(self) -> Path:
        return self.pages_dir / "log.md"

    @property
    def schema_file(self) -> Path:
        return self.system_dir / "schema.md"

    @property
    def activity_file(self) -> Path:
        return self.root / "runtime" / "activity.jsonl"

    @property
    def codex_state_file(self) -> Path:
        return self.root / "codex-save-state.json"

    @property
    def claude_code_state_file(self) -> Path:
        return self.root / "claude-code-save-state.json"


DEFAULT_CONTEXT = RuntimeContext(resolve_root())
CHRONOVISOR_ROOT = DEFAULT_CONTEXT.root
RAW_DIR = DEFAULT_CONTEXT.raw_dir
PAGES_DIR = DEFAULT_CONTEXT.pages_dir
SYSTEM_DIR = DEFAULT_CONTEXT.system_dir
INDEX_FILE = DEFAULT_CONTEXT.index_file
LOG_FILE = DEFAULT_CONTEXT.log_file
SCHEMA_FILE = DEFAULT_CONTEXT.schema_file
ACTIVITY_FILE = DEFAULT_CONTEXT.activity_file


def init_chronovisor(context: RuntimeContext | None = None) -> None:
    """Create only the directories needed to capture transcripts."""
    if os.environ.get("CHRONOVISOR_READ_ONLY") == "1":
        return
    raw_dir = RAW_DIR if context is None else context.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(raw_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)
