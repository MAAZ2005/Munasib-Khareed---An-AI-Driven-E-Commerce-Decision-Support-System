"""Smoke tests verifying package importability, configuration loading, and core modules."""

import pytest

import munasib
from munasib.core.config import Settings, get_settings, load_settings
from munasib.core.exceptions import (
    DataError,
    InsufficientEvidenceError,
    ModelError,
    MunasibError,
    ProductNotFoundError,
)
from munasib.core.logging import get_logger, setup_logging


def test_package_metadata() -> None:
    """Verify that munasib package is successfully imported and accessible."""
    assert munasib is not None
    assert hasattr(munasib, "__doc__")


def test_load_settings() -> None:
    """Verify loading configuration settings from YAML and default model validity."""
    settings = load_settings("configs/config.yaml")
    assert isinstance(settings, Settings)
    assert settings.project.name == "munasib"
    assert settings.project.environment == "development"
    assert settings.dataset.category == "Appliances"
    assert settings.review_intel.sentiment.model_name == "distilbert-base-uncased"
    assert settings.review_intel.aspects.spacy_model == "en_core_web_sm"
    assert settings.review_intel.sentiment.rating_mapping.negative_max == 2
    assert settings.review_intel.sentiment.rating_mapping.positive_min == 4
    assert settings.risk.scale_max == 100.0


def test_get_settings_singleton() -> None:
    """Verify that get_settings provides a cached singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_logging_setup() -> None:
    """Verify logger configuration and retrieval."""
    setup_logging(level="DEBUG")
    logger = get_logger("munasib.test")
    assert logger is not None
    logger.debug("Smoke test logging verification")


def test_custom_exceptions_hierarchy() -> None:
    """Verify exception hierarchy and string formatting."""
    with pytest.raises(MunasibError) as exc_info:
        raise ProductNotFoundError(asin="B000TEST01")
    assert "B000TEST01" in str(exc_info.value)
    assert exc_info.value.asin == "B000TEST01"

    with pytest.raises(DataError) as data_exc:
        raise InsufficientEvidenceError(asin="B000TEST02", review_count=2, min_required=5)
    assert data_exc.value.review_count == 2
    assert data_exc.value.min_required == 5

    with pytest.raises(MunasibError):
        raise ModelError("Model failed to load weights")
