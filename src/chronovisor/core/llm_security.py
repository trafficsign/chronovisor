"""Allowlisted environment for capture subprocesses."""

from __future__ import annotations

import os
from collections.abc import Mapping

MAX_REQUEST_TIMEOUT_MS = 900_000
_CHILD_ENV_NAMES = frozenset(
    {
        "COMSPEC",
        "CURL_CA_BUNDLE",
        "DBUS_SESSION_BUS_ADDRESS",
        "HOME",
        "LANG",
        "LANGUAGE",
        "LC_ADDRESS",
        "LC_ALL",
        "LC_COLLATE",
        "LC_CTYPE",
        "LC_IDENTIFICATION",
        "LC_MEASUREMENT",
        "LC_MESSAGES",
        "LC_MONETARY",
        "LC_NAME",
        "LC_NUMERIC",
        "LC_PAPER",
        "LC_TELEPHONE",
        "LC_TIME",
        "PATH",
        "PATHEXT",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "TZDIR",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "XDG_RUNTIME_DIR",
    }
)


def build_child_process_env(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy only non-secret runtime settings into an isolated child."""

    source = os.environ if environ is None else environ
    return {name: value for name, value in source.items() if name in _CHILD_ENV_NAMES}
