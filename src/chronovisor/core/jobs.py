"""Job tracking for async operations."""

import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus
    processor: str  # "ollama" or "unavailable"
    created_at: str
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    pages_created: list[str] = field(default_factory=list)
    pages_updated: list[str] = field(default_factory=list)
    stage: str | None = None  # "triage" | "generate" | None
    total_ops: int = 0
    completed_ops: int = 0


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, processor: str) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            processor=processor,
            created_at=datetime.now().isoformat(),
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)

    def recent(self, limit: int = 10) -> list[Job]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]


job_store = JobStore()
