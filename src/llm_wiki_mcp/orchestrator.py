"""Orchestrator - deterministic control flow for Ingest/Lint scheduling.

NOT an LLM. Pure code logic. Sonnet handles content structuring,
this module handles when to trigger it.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from llm_wiki_mcp.wiki import RAW_DIR, WIKI_ROOT
from llm_wiki_mcp.ollama import is_available

# Config
INGEST_THRESHOLD = 5  # Trigger ingest after N raw files
LINT_INTERVAL_HOURS = 24  # Run lint every N hours

# State file
STATE_FILE = WIKI_ROOT / ".orchestrator_state.json"


def _load_state() -> dict:
    """Load orchestrator state."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "last_ingest": None,
        "last_lint": None,
        "processed_raw_files": [],
        "ollama_health": {"status": None, "checked_at": None},
    }


def _save_state(state: dict) -> None:
    """Save orchestrator state."""
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def get_pending_raw_files() -> list[Path]:
    """Get raw files that haven't been processed yet."""
    state = _load_state()
    processed = set(state.get("processed_raw_files", []))
    pending = []
    for f in sorted(RAW_DIR.glob("*.md")):
        if f.name not in processed:
            pending.append(f)
    return pending


def should_ingest() -> tuple[bool, str]:
    """Check if ingest should be triggered. Returns (should_run, reason)."""
    pending = get_pending_raw_files()
    if len(pending) >= INGEST_THRESHOLD:
        return True, f"{len(pending)} pending raw files (threshold: {INGEST_THRESHOLD})"
    return False, f"Only {len(pending)} pending (threshold: {INGEST_THRESHOLD})"


def should_lint() -> tuple[bool, str]:
    """Check if lint should be triggered. Returns (should_run, reason)."""
    state = _load_state()
    last_lint = state.get("last_lint")

    if last_lint is None:
        return True, "Lint has never been run"

    last_lint_dt = datetime.fromisoformat(last_lint)
    hours_since = (datetime.now() - last_lint_dt).total_seconds() / 3600

    if hours_since >= LINT_INTERVAL_HOURS:
        return True, f"{hours_since:.1f} hours since last lint (threshold: {LINT_INTERVAL_HOURS}h)"
    return False, f"Only {hours_since:.1f}h since last lint (threshold: {LINT_INTERVAL_HOURS}h)"


def mark_raw_processed(filenames: list[str]) -> None:
    """Mark raw files as processed."""
    state = _load_state()
    processed = set(state.get("processed_raw_files", []))
    processed.update(filenames)
    state["processed_raw_files"] = sorted(processed)
    state["last_ingest"] = datetime.now().isoformat()
    _save_state(state)


def mark_lint_complete() -> None:
    """Mark lint as completed."""
    state = _load_state()
    state["last_lint"] = datetime.now().isoformat()
    _save_state(state)


def get_ollama_status() -> dict:
    """Get Ollama status with caching."""
    available = is_available()
    return {
        "available": available,
        "processor": "ollama" if available else "sonnet",
    }


def run_pending_ingest() -> dict:
    """Run ingest on all pending raw files if threshold is met.

    Returns result dict with status and details.
    """
    should, reason = should_ingest()
    if not should:
        return {"triggered": False, "reason": reason}

    pending = get_pending_raw_files()

    # Limit batch size to avoid overwhelming LLM
    MAX_BATCH = 10
    pending = pending[:MAX_BATCH]

    # Collect keywords and content from pending raw files
    contents = []
    filenames = []
    all_keywords = []
    for f in pending:
        raw_text = f.read_text()
        contents.append(f"--- Source: {f.name} ---\n{raw_text}")
        filenames.append(f.name)
        # Extract keywords from individual file frontmatter
        from llm_wiki_mcp.ingest import _extract_keywords_from_raw
        all_keywords.extend(_extract_keywords_from_raw(raw_text))

    # Deduplicate keywords preserving order
    seen = set()
    unique_keywords = []
    for kw in all_keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    # Prepend aggregated keywords as frontmatter for the combined content
    combined_body = "\n\n".join(contents)
    if unique_keywords:
        kw_line = ", ".join(unique_keywords)
        combined = f"---\nkeywords: [{kw_line}]\n---\n\n{combined_body}"
    else:
        combined = combined_body

    # Start ingest
    from llm_wiki_mcp.ingest import start_ingest
    job_id = start_ingest(combined)

    # Mark as processed
    mark_raw_processed(filenames)

    return {
        "triggered": True,
        "reason": reason,
        "job_id": job_id,
        "files_processed": filenames,
        "processor": get_ollama_status()["processor"],
    }


def run_lint_if_due() -> dict:
    """Run lint check + safe apply if due.

    Returns result dict with status and details.
    """
    should, reason = should_lint()
    if not should:
        return {"triggered": False, "reason": reason}

    from llm_wiki_mcp.lint import check, apply_safe_fixes

    issues = check()
    actions = apply_safe_fixes(issues)
    remaining = [i for i in issues if not i.get("auto_fixable")]

    mark_lint_complete()

    return {
        "triggered": True,
        "reason": reason,
        "total_issues": len(issues),
        "actions_taken": actions,
        "remaining_issues": len(remaining),
    }


def tick() -> dict:
    """Main orchestration tick. Call this periodically.

    Checks if ingest or lint should run, and triggers them if needed.
    Returns summary of what happened.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "ollama": get_ollama_status(),
        "ingest": run_pending_ingest(),
        "lint": run_lint_if_due(),
    }
    return results
