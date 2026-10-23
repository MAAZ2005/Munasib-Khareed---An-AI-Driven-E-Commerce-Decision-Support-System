"""Tests for the ``product_intel`` configuration block.

Covers the validation that keeps the profiling stage honest: the documented
defaults, the shipped config agreeing with them, and the rejection of every
value that would otherwise produce a silently wrong indicator (a zero target
would make every completeness score 0, a typo'd metadata field would vanish
from the denominator).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from munasib.core.config import METADATA_FIELD_VOCABULARY, ProductIntelConfig, Settings

REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def test_documented_defaults() -> None:
    """The defaults match those documented in docs/product_intelligence.md."""
    config = ProductIntelConfig()
    assert config.target_description_words == 100
    assert config.target_feature_count == 5
    assert config.min_brand_peers == 5
    assert config.min_rank_peers == 5
    assert config.metadata_fields == list(METADATA_FIELD_VOCABULARY)


def test_shipped_config_declares_every_threshold() -> None:
    """The repo config file names each key, so nothing falls back silently."""
    text = REPO_CONFIG.read_text(encoding="utf-8")
    for key in (
        "target_description_words",
        "target_feature_count",
        "min_brand_peers",
        "min_rank_peers",
        "metadata_fields",
    ):
        assert f"{key}:" in text


def test_settings_exposes_the_block() -> None:
    """``Settings`` surfaces the block under ``product_intel``."""
    assert Settings().product_intel.target_description_words == 100


@pytest.mark.parametrize("field", ["target_description_words", "target_feature_count"])
def test_zero_completeness_target_is_rejected(field: str) -> None:
    """A zero target would divide by zero / zero every score - rejected."""
    with pytest.raises(ValidationError, match=field):
        ProductIntelConfig(**{field: 0})


@pytest.mark.parametrize("field", ["min_brand_peers", "min_rank_peers"])
def test_zero_peer_minimum_is_rejected(field: str) -> None:
    """A zero peer minimum would report percentiles over no peers - rejected."""
    with pytest.raises(ValidationError, match=field):
        ProductIntelConfig(**{field: 0})


def test_negative_target_is_rejected() -> None:
    """Negative thresholds are as invalid as zero ones."""
    with pytest.raises(ValidationError, match="target_description_words"):
        ProductIntelConfig(target_description_words=-5)


def test_empty_metadata_field_list_is_rejected() -> None:
    """An empty field list would divide by zero in metadata_completeness."""
    with pytest.raises(ValidationError, match="metadata_fields"):
        ProductIntelConfig(metadata_fields=[])


def test_unknown_metadata_field_is_rejected() -> None:
    """A typo'd field must fail loudly instead of shrinking the denominator."""
    with pytest.raises(ValidationError, match="unknown fields"):
        ProductIntelConfig(metadata_fields=["title", "weight"])


def test_repeated_metadata_field_is_rejected() -> None:
    """Repeating a field would double-count it in the completeness mean."""
    with pytest.raises(ValidationError, match="repeat"):
        ProductIntelConfig(metadata_fields=["title", "title"])


def test_reduced_metadata_field_list_is_accepted() -> None:
    """A narrower list is legal - it simply scores over fewer fields."""
    config = ProductIntelConfig(metadata_fields=["title", "price"])
    assert config.metadata_fields == ["title", "price"]


def test_shipped_config_loads_with_the_repo_defaults() -> None:
    """The shipped YAML matches the code defaults exactly (no drift)."""
    from munasib.core.config import load_settings

    loaded = load_settings(REPO_CONFIG).product_intel
    assert loaded == ProductIntelConfig()
