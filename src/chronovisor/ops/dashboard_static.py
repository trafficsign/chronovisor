"""Static asset path resolution for the local dashboard."""

from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parents[1] / "dashboard_static"


def _resolve_static_path(static_dir: Path, request_path: str) -> Path | None:
    rel = request_path.removeprefix("/static/").lstrip("/")
    target = (static_dir / rel).resolve()
    try:
        target.relative_to(static_dir.resolve())
    except ValueError:
        return None
    return target
