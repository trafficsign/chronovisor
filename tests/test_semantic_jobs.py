from datetime import UTC, datetime, timedelta
from pathlib import Path

from chronovisor.core.semantic_jobs import (
    claim_next,
    complete,
    enqueue_page,
    enqueue_pages,
    enqueue_rebuild,
    fail,
    job_status,
    prune_completed_jobs,
)


def test_page_jobs_coalesce_only_while_pending(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite"
    first = enqueue_page("ai/example", source_sha256="old", path=path)
    assert enqueue_page("ai/example", source_sha256="new", path=path) == first

    leased = claim_next(path=path)
    assert leased is not None
    assert leased.source_sha256 == "new"

    # A newer update arriving while the first job is leased must be a distinct
    # durable job; resetting the lease would let the old worker win.
    second = enqueue_page("ai/example", source_sha256="newest", path=path)
    assert second != first
    complete(first, path=path)
    next_job = claim_next(path=path)
    assert next_job is not None
    assert next_job.job_id == second
    assert next_job.source_sha256 == "newest"


def test_failure_retries_then_can_be_terminal(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite"
    job_id = enqueue_page("ai/example", path=path)
    leased = claim_next(path=path)
    assert leased is not None
    fail(job_id, "temporary", path=path)
    status = job_status(path=path)
    assert status["counts"] == {"pending": 1}

    # Make the retry immediately eligible without sleeping.
    import sqlite3

    connection = sqlite3.connect(path)
    with connection:
        connection.execute(
            "UPDATE jobs SET next_attempt_at = ? WHERE job_id = ?",
            (
                (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
                job_id,
            ),
        )
    connection.close()
    assert claim_next(path=path) is not None
    fail(job_id, "permanent", terminal=True, path=path)
    assert job_status(path=path)["counts"] == {"dead": 1}


def test_rebuild_and_page_batches_are_durable(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite"
    ids = enqueue_pages(["b", "a", "a"], path=path)
    assert len(ids) == 2
    rebuild = enqueue_rebuild(path=path)
    assert enqueue_rebuild(path=path) == rebuild
    assert job_status(path=path)["counts"] == {"pending": 3}


def test_rebuild_request_coalesces_with_running_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite"
    rebuild = enqueue_rebuild(path=path)
    leased = claim_next(kinds=("rebuild",), path=path)
    assert leased is not None
    assert leased.job_id == rebuild
    assert enqueue_rebuild(path=path) == rebuild
    assert job_status(path=path)["counts"] == {"leased": 1}


def test_old_completed_jobs_are_pruned(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite"
    job_id = enqueue_page("page", path=path)
    assert claim_next(path=path) is not None
    complete(job_id, path=path)

    import sqlite3

    connection = sqlite3.connect(path)
    with connection:
        connection.execute(
            "UPDATE jobs SET completed_at = '2000-01-01T00:00:00+00:00'"
        )
    connection.close()
    assert prune_completed_jobs(path=path) == 1
    assert job_status(path=path)["counts"] == {}
