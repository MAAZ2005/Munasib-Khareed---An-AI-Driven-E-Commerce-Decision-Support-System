"""Tests for the ``analysis`` configuration block added for the API layer.

Covers the three behaviours the API depends on: the documented ``stub`` default,
the ``MUNASIB_ANALYSIS_SERVICE`` environment override, and the rejection of an
unknown value (so a typo cannot silently fall back to the stub and present
placeholder numbers as a real analysis).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from munasib.core.config import Settings, load_settings
from munasib.core.exceptions import ConfigError

REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def test_analysis_service_defaults_to_stub() -> None:
    """The default is the stub, so a fresh checkout serves placeholders knowingly."""
    assert Settings().analysis.service == "stub"


def test_repo_config_declares_stub_service() -> None:
    """The shipped config agrees with the documented default."""
    text = REPO_CONFIG.read_text(encoding="utf-8")
    assert 'service: "stub"' in text


def test_analysis_service_accepts_real_value() -> None:
    """'real' is a legal selection (it fails later, not at config load)."""
    assert Settings.model_validate({"analysis": {"service": "real"}}).analysis.service == "real"


def test_analysis_service_rejects_unknown_value() -> None:
    """An unknown service name is rejected instead of falling back to the stub."""
    with pytest.raises(ValidationError):
        Settings.model_validate({"analysis": {"service": "banana"}})


def test_env_override_selects_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUNASIB_ANALYSIS_SERVICE overrides the YAML value."""
    monkeypatch.delenv("MUNASIB_ANALYSIS_SERVICE", raising=False)
    monkeypatch.setenv("MUNASIB_CONFIG_PATH", str(REPO_CONFIG))
    monkeypatch.setenv("MUNASIB_ANALYSIS_SERVICE", "real")
    try:
        assert load_settings().analysis.service == "real"
    finally:
        os.environ.pop("MUNASIB_ANALYSIS_SERVICE", None)


def test_env_override_can_select_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same override can pin the stub back on."""
    monkeypatch.delenv("MUNASIB_ANALYSIS_SERVICE", raising=False)
    monkeypatch.setenv("MUNASIB_CONFIG_PATH", str(REPO_CONFIG))
    monkeypatch.setenv("MUNASIB_ANALYSIS_SERVICE", "stub")
    try:
        assert load_settings().analysis.service == "stub"
    finally:
        os.environ.pop("MUNASIB_ANALYSIS_SERVICE", None)


def test_env_override_with_invalid_value_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad override surfaces as ConfigError, never as a silent fallback."""
    monkeypatch.setenv("MUNASIB_CONFIG_PATH", str(REPO_CONFIG))
    monkeypatch.setenv("MUNASIB_ANALYSIS_SERVICE", "banana")
    try:
        with pytest.raises(ConfigError):
            load_settings()
    finally:
        os.environ.pop("MUNASIB_ANALYSIS_SERVICE", None)
