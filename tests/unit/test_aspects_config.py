"""Tests for the aspect configuration block.

Covers the requested stoplist additions, the service-aspect exceptions, the
vocabulary cutoff, and the validation of the four-condition pain-point rule
(including the requirement that strength and pain-point thresholds cannot
overlap).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from munasib.core.config import (
    DEFAULT_GENERIC_NOUNS,
    AspectsConfig,
    Settings,
    load_settings,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def test_shipped_config_loads_the_expected_aspect_values() -> None:
    """The repository config declares the new cutoff and aggregation keys."""
    settings = load_settings(str(REPO_CONFIG))
    aspects = settings.review_intel.aspects
    assert aspects.min_review_count == 50
    assert aspects.top_k_aspects == 10
    assert aspects.pain_point_min_negative_mentions == 3
    assert aspects.pain_point_min_negative_share == 0.25
    assert aspects.strength_min_positive_share == 0.75
    assert aspects.strength_max_negative_lift == 0.75
    assert aspects.candidate_parts_dir == Path("data/interim/aspect_candidates.parts")


def test_stoplist_includes_person_family_and_review_meta_words() -> None:
    """The requested person/family/review words are filtered by default."""
    stoplist = set(AspectsConfig().generic_nouns)
    for word in (
        "husband",
        "wife",
        "son",
        "daughter",
        "mom",
        "dad",
        "friend",
        "customer",
        "buyer",
        "review",
        "star",
        "rating",
        "amazon",
        "stuff",
        "bit",
        "way",
        "kind",
        "issue",
        "family",
        "person",
    ):
        assert word in stoplist
    assert set(DEFAULT_GENERIC_NOUNS) == stoplist


def test_stoplist_was_extended_after_the_vocabulary_inspection() -> None:
    """The --vocabulary inspections found person/meta junk; these words were added.

    The first inspection (``data/interim/aspect_vocabulary.log``) still listed
    ``thank``, ``fiance``, ``baby``, ``mother``, ``other`` and similar entries
    at >= 50 reviews. The second inspection of the rebuilt top-100 / near-cutoff
    lists found the date abbreviations ``jan``/``feb``/``nov``/``dec``/``may``
    plus the person words ``phillip`` and ``woman``. A third pass over the
    printed product examples found ``reviewer``, ``poster``, ``purchaser``,
    ``owner`` and ``user``. The extended stoplist plus the phrase-level filter
    remove all of them from the rebuilt vocabulary (docs/aspects.md).
    """
    stoplist = set(AspectsConfig().generic_nouns)
    added = (
        "adult",
        "advise",
        "baby",
        "boss",
        "boyfriend",
        "brother",
        "buy",
        "buying",
        "child",
        "claim",
        "co",
        "dec",
        "describe",
        "described",
        "description",
        "father",
        "feedback",
        "feb",
        "fiance",
        "following",
        "girlfriend",
        "jan",
        "kid",
        "listing",
        "may",
        "mine",
        "mother",
        "neighbor",
        "none",
        "nov",
        "other",
        "owner",
        "phillip",
        "poster",
        "purchaser",
        "recommend",
        "reviewer",
        "roommate",
        "sister",
        "teenager",
        "thank",
        "user",
        "woman",
    )
    assert set(added) <= stoplist
    assert len(stoplist) == 118


def test_service_aspects_are_deliberately_kept() -> None:
    """Delivery/shipping/seller/packaging/box are legitimate service aspects."""
    stoplist = set(AspectsConfig().generic_nouns)
    for word in ("delivery", "shipping", "seller", "packaging", "box", "service"):
        assert word not in stoplist


def test_stoplist_is_normalised_and_duplicates_rejected() -> None:
    """Stoplist entries are lowercased/stripped; duplicates fail loudly."""
    config = AspectsConfig(generic_nouns=["  Thing ", "THING2"])
    assert config.generic_nouns == ["thing", "thing2"]
    with pytest.raises(ValidationError, match="must not repeat"):
        AspectsConfig(generic_nouns=["thing", " Thing "])


def test_required_spacy_components_cannot_be_disabled() -> None:
    """Disabling the parser or lemmatizer would silently change extraction."""
    with pytest.raises(ValidationError, match="cannot disable"):
        AspectsConfig(spacy_disable_components=["parser"])
    with pytest.raises(ValidationError, match="cannot disable"):
        AspectsConfig(spacy_disable_components=["lemmatizer"])


def test_threshold_validation_rejects_invalid_bounds() -> None:
    """Counts and shares must stay in their documented ranges."""
    with pytest.raises(ValidationError, match="must be >= 1"):
        AspectsConfig(min_review_count=0)
    with pytest.raises(ValidationError, match="must be >= 1"):
        AspectsConfig(pain_point_min_negative_mentions=0)
    with pytest.raises(ValidationError, match="must be in \\[0, 1\\]"):
        AspectsConfig(pain_point_min_negative_share=1.5)
    with pytest.raises(ValidationError, match="must be in \\[0, 1\\]"):
        AspectsConfig(strength_min_positive_share=-0.1)
    with pytest.raises(ValidationError, match="must be below pain_point_min_lift"):
        AspectsConfig(pain_point_min_lift=1.0, strength_max_negative_lift=1.0)
    with pytest.raises(ValidationError, match="max_aspect_tokens"):
        AspectsConfig(max_aspect_tokens=3)


def test_paths_must_not_be_empty() -> None:
    """An empty artifact path is a configuration error, not a cwd write."""
    with pytest.raises(ValidationError, match="vocabulary_path must not be empty"):
        AspectsConfig(vocabulary_path=Path("   "))


def test_settings_default_has_the_new_cutoff() -> None:
    """A Settings object without YAML uses the same validated defaults."""
    assert Settings().review_intel.aspects.min_review_count == 50


def test_self_reference_exclusion_defaults_to_on() -> None:
    """The shipped config keeps the rule enabled and caches its token sets."""
    aspects = load_settings(str(REPO_CONFIG)).review_intel.aspects
    assert aspects.exclude_self_reference is True
    assert aspects.self_reference_cache_path == Path("data/interim/product_self_tokens.json")
    assert "exclude_self_reference: true" in REPO_CONFIG.read_text(encoding="utf-8")
    assert AspectsConfig().exclude_self_reference is True


def test_self_reference_cache_path_must_not_be_empty() -> None:
    """An empty cache path would silently write to the working directory."""
    with pytest.raises(
        ValidationError, match="self_reference_cache_path must not be empty"
    ):
        AspectsConfig(self_reference_cache_path=Path("   "))
