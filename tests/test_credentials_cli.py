from __future__ import annotations

import json

import pytest

from chronovisor.core.llm_security import (
    CredentialFailureCategory,
    CredentialRef,
    CredentialSecurityError,
    CredentialStoreStatus,
)
from chronovisor.hosts import cli

CANARY = "sk-CANARY-CREDENTIAL-CLI"


class FakeStdin:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class FakeCredentialStore:
    def __init__(self) -> None:
        self.set_calls: list[tuple[CredentialRef, str]] = []
        self.status_result = CredentialStoreStatus(True, "present")
        self.delete_result = CredentialStoreStatus(False, "missing")

    def set(self, ref: CredentialRef, secret: str) -> CredentialStoreStatus:
        self.set_calls.append((ref, secret))
        return CredentialStoreStatus(True, "present")

    def status(self, _ref: CredentialRef) -> CredentialStoreStatus:
        return self.status_result

    def delete(self, _ref: CredentialRef) -> CredentialStoreStatus:
        return self.delete_result


def test_credentials_set_reads_only_hidden_tty_input_and_never_prints_value(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = FakeCredentialStore()
    monkeypatch.setattr(cli, "_credential_store", lambda: store)
    monkeypatch.setattr(cli.sys, "stdin", FakeStdin(True))
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: CANARY)

    assert cli.main(["credentials", "set", "openai/default", "--json"]) == 0

    output = capsys.readouterr()
    assert json.loads(output.out) == {"present": True, "category": "present"}
    assert output.err == ""
    assert store.set_calls == [
        (CredentialRef.parse("oskeyring:openai/default"), CANARY)
    ]
    assert CANARY not in output.out
    assert CANARY not in output.err


def test_credentials_set_rejects_non_tty_before_reading_secret(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = FakeCredentialStore()
    prompted = False

    def prompt(_message: str) -> str:
        nonlocal prompted
        prompted = True
        return CANARY

    monkeypatch.setattr(cli, "_credential_store", lambda: store)
    monkeypatch.setattr(cli.sys, "stdin", FakeStdin(False))
    monkeypatch.setattr(cli.getpass, "getpass", prompt)

    assert cli.main(["credentials", "set", "openai/default", "--json"]) == 1

    output = capsys.readouterr()
    assert json.loads(output.out) == {"present": False, "category": "tty_required"}
    assert not prompted
    assert store.set_calls == []


@pytest.mark.parametrize(
    "command,present,category",
    [("status", True, "present"), ("delete", False, "missing")],
)
def test_credentials_status_and_delete_expose_only_presence_and_category(
    command: str,
    present: bool,
    category: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = FakeCredentialStore()
    monkeypatch.setattr(cli, "_credential_store", lambda: store)

    assert cli.main(["credentials", command, "openai/default", "--json"]) == 0

    output = capsys.readouterr()
    assert json.loads(output.out) == {"present": present, "category": category}
    assert set(json.loads(output.out)) == {"present", "category"}
    assert CANARY not in output.out


def test_credentials_backend_failure_is_safe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable() -> FakeCredentialStore:
        raise CredentialSecurityError(CredentialFailureCategory.STORE_UNAVAILABLE)

    monkeypatch.setattr(cli, "_credential_store", unavailable)

    assert cli.main(["credentials", "status", "openai/default", "--json"]) == 1

    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "present": False,
        "category": "store_unavailable",
    }
    assert CANARY not in output.out + output.err


def test_credentials_has_no_get_or_secret_argument_and_parser_error_is_redacted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as get_error:
        cli.main(["credentials", "get", "openai/default"])
    assert get_error.value.code == 2

    with pytest.raises(SystemExit) as secret_error:
        cli.main(["credentials", "set", "openai/default", CANARY])
    assert secret_error.value.code == 2

    output = capsys.readouterr()
    assert CANARY not in output.out + output.err
