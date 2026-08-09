"""Executable forwarder for the recall-owned collection anomaly worker."""

from chronovisor.recall.collection_anomaly_worker import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
