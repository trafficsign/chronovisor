"""Chronovisor ingest domain implementations and legacy API."""

from chronovisor.core.compat import install_legacy_package

install_legacy_package(__name__, f"{__name__}.ingest")
