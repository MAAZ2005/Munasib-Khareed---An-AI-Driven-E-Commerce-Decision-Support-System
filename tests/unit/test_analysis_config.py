"""Tests for the ``analysis`` configuration block added for the API layer.

Covers the behaviours the API depends on: the documented ``hybrid`` default,
the ``MUNASIB_ANALYSIS_SERVICE`` environment override, and the rejection of an
unknown value (so a typo cannot silently fall back to the stub and present
placeholder numbers as a real analysis).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from munasib.api.deps import get_analysis_service
from munasib.core.config import Settings, load_settings
from munasib.core.exceptions import ConfigError, ModelNotLoadedError

REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def test_analysis_service_defaults_to_hybrid() -> None:
    """The default is hybrid, serving real stored data with clearly labelled placeholders."""
    assert Settings().analysis.service == "hybrid"


def test_repo_config_declares_hybrid_service() -> None:
    """The shipped config agrees with the hybrid default."""
    text = REPO_CONFIG.read_text(encoding="utf-8")
    assert 'service: "hybrid"' in text


def test_analysis_service_accepts_valid_values() -> None:
    """'stub', 'hybrid', and 'real' are legal selections."""
    for service_name in ("stub", "hybrid", "real"):
        assert (
            Settings.model_validate({"analysis": {"service": service_name}}).analysis.service
            == service_name
        )


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


def test_get_analysis_service_with_real_raises_model_not_loaded() -> None:
    """Selecting 'real' raises ModelNotLoadedError in the dependency injection layer."""
    settings = Settings.model_validate({"analysis": {"service": "real"}})
    session = MagicMock()
    with pytest.raises(ModelNotLoadedError) as exc_info:
        get_analysis_service(session=session, settings=settings)
    assert "fuzzy + TOPSIS + SHAP engine is not implemented yet" in str(exc_info.value)
    assert exc_info.value.details == {"service": "real"}
