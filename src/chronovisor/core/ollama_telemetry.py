"""Redacted local model activity telemetry."""

import logging
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from types import FrameType

from chronovisor.core.jsonl_write import atomic_write_json_payload

_MODEL_ACTIVITY_SCHEMA_VERSION = 1


def _model_activity_caller(facade_module: str) -> tuple[str, str]:
    """Identify the first caller outside the telemetry facade."""

    frame: FrameType | None = sys._getframe(1)
    skipped_modules = {__name__, facade_module, "contextlib"}
    while frame is not None:
        component = str(frame.f_globals.get("__name__") or "")
        if component and component not in skipped_modules:
            return component, frame.f_code.co_name
        frame = frame.f_back
    return "unknown", "unknown"


def _model_activity_pipeline(component: str) -> str:
    normalized = component.casefold()
    if ".knowledge_graph." in normalized:
        return "typed_graph"
    if ".recall." in normalized:
        return "recall"
    if ".ingest." in normalized:
        return "ingest"
    if "repair" in normalized:
        return "repair"
    if ".lab." in normalized or ".autonomy." in normalized:
        return "improve"
    return "audit"


@contextmanager
def model_activity(
    *,
    model: str,
    operation: str,
    root: Path,
    facade_module: str,
    pipeline: str | None = None,
) -> Iterator[None]:
    """Publish one redacted live marker for a local model inference."""

    component, caller = _model_activity_caller(facade_module)
    selected_pipeline = pipeline or _model_activity_pipeline(component)
    started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
    activity_id = f"{os.getpid()}-{threading.get_ident()}-{time.time_ns()}"
    active_dir = root / "runtime" / "model-activity" / "active"
    marker_path = active_dir / f"{activity_id}.json"
    marker_created = False
    status = "done"
    logger = logging.getLogger(facade_module)
    try:
        active_dir.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            active_dir.chmod(0o700)
        atomic_write_json_payload(
            marker_path,
            {
                "schema_version": _MODEL_ACTIVITY_SCHEMA_VERSION,
                "activity_id": activity_id,
                "pipeline": selected_pipeline,
                "component": component,
                "caller": caller,
                "operation": operation,
                "model": model,
                "pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "status": "active",
                "started_at": started_at,
                "updated_at": started_at,
            },
        )
        marker_created = True
    except (OSError, TypeError, ValueError):
        logger.debug("could not publish model activity marker", exc_info=True)
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        if marker_created:
            with suppress(OSError):
                marker_path.unlink(missing_ok=True)
            finished_at = datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )
            recent_path = active_dir.parent / "recent" / f"{selected_pipeline}.json"
            try:
                recent_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json_payload(
                    recent_path,
                    {
                        "schema_version": _MODEL_ACTIVITY_SCHEMA_VERSION,
                        "activity_id": activity_id,
                        "pipeline": selected_pipeline,
                        "component": component,
                        "caller": caller,
                        "operation": operation,
                        "model": model,
                        "pid": os.getpid(),
                        "thread_id": threading.get_ident(),
                        "status": status,
                        "started_at": started_at,
                        "updated_at": finished_at,
                        "finished_at": finished_at,
                    },
                )
            except (OSError, TypeError, ValueError):
                logger.debug("could not publish recent model activity", exc_info=True)
