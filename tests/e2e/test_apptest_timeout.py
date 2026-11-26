"""Tests for the shared AppTest run timeout helper in ``tests/e2e/conftest.py``.

These cover the contract every e2e run call relies on: a shared 30s default, an
environment override that wins over any default, and a loud failure when the
override is misconfigured.
"""

from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    APPTTEST_TIMEOUT_ENV_VAR,
    DEFAULT_APPTEST_TIMEOUT,
    apptest_timeout,
)


@pytest.fixture(autouse=True)
def _unset_timeout_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any inherited override so tests start from the shared default."""
    monkeypatch.delenv(APPTTEST_TIMEOUT_ENV_VAR, raising=False)


def test_default_is_thirty_seconds() -> None:
    """Without an override the shared default is used unchanged."""
    assert DEFAULT_APPTEST_TIMEOUT == 30.0
    assert apptest_timeout() == 30.0


def test_environment_variable_overrides_every_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MUNASIB_APPTEST_TIMEOUT`` wins over the shared and the per-call default."""
    monkeypatch.setenv(APPTTEST_TIMEOUT_ENV_VAR, "45")
    assert apptest_timeout() == 45.0
    assert apptest_timeout(DEFAULT_APPTEST_TIMEOUT) == 45.0
    assert apptest_timeout(60.0) == 45.0


def test_fractional_override_and_empty_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fractional override is honoured; an empty variable falls back to the default."""
    monkeypatch.setenv(APPTTEST_TIMEOUT_ENV_VAR, "12.5")
    assert apptest_timeout() == 12.5
    monkeypatch.setenv(APPTTEST_TIMEOUT_ENV_VAR, "")
    assert apptest_timeout() == DEFAULT_APPTEST_TIMEOUT


@pytest.mark.parametrize("bad_value", ["abc", "0", "-5", "5s"])
def test_misconfigured_override_raises(monkeypatch: pytest.MonkeyPatch, bad_value: str) -> None:
    """A bad override fails loudly rather than silently changing the budget."""
    monkeypatch.setenv(APPTTEST_TIMEOUT_ENV_VAR, bad_value)
    with pytest.raises(ValueError, match=APPTTEST_TIMEOUT_ENV_VAR):
        apptest_timeout()
