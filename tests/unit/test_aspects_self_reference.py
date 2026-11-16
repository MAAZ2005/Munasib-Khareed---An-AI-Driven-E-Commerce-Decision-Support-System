"""Tests for the self-reference exclusion rule, its cache and its report.

The rule is a token-set test: an aspect is excluded only when **every** token
of the aspect appears in the product's lemmatized title/brand/category token
set. These tests pin the excluded/kept boundary, the config flag that turns
the rule off, determinism and idempotence, and the cache fingerprint.
"""

from __future__ import annotations

from pathlib import Path

from munasib.core.config import AspectsConfig
from munasib.review_intel.aspects.self_reference import (
    ProductIdentity,
    build_self_token_sets,
    excluded_aspect_summary,
    filter_self_reference,
    is_self_reference,
    load_self_token_cache,
    save_self_token_cache,
    self_reference_fingerprint,
    spacy_batch_lemmatizer,
)
from munasib.review_intel.aspects.types import (
    ASPECT_NEUTRAL,
    ASPECT_PAIN_POINT,
    AspectLabelCounts,
    ProductAspectStats,
)

#: Title, brand and category of the worked example in docs/aspects.md.
WATER_FILTER = ProductIdentity(
    asin="B00000020B",
    title_clean="General Electric MWF Refrigerator Water Filter",
    brand="GE",
    category_leaf="Water Filter",
)


def _lowercase_split(texts: list[str]) -> list[list[str]]:
    """Deterministic stand-in for spaCy's lemmatizer (lowercase + split).

    Args:
        texts: Metadata strings.

    Returns:
        One token list per input string, in input order.
    """
    return [text.lower().split() for text in texts]


def _stats(
    asin: str,
    aspect: str,
    *,
    negative: int = 0,
    neutral: int = 0,
    positive: int = 0,
    classification: str = ASPECT_NEUTRAL,
) -> ProductAspectStats:
    """Build one aggregated row for the filter tests.

    Args:
        asin: Product ASIN.
        aspect: Aspect phrase.
        negative: Negative review mentions.
        neutral: Neutral review mentions.
        positive: Positive review mentions.
        classification: Stored classification label.

    Returns:
        One product-aspect statistic.
    """
    return ProductAspectStats(
        asin=asin,
        aspect=aspect,
        counts=AspectLabelCounts(negative=negative, neutral=neutral, positive=positive),
        product_labeled_reviews=10,
        product_negative_share=0.5,
        negativity_lift=None,
        classification=classification,
    )


def _water_filter_sets() -> dict[str, frozenset[str]]:
    """Build the token set of :data:`WATER_FILTER` with the fake lemmatizer.

    Returns:
        Token sets keyed by ASIN.
    """
    return build_self_token_sets([WATER_FILTER], AspectsConfig(), _lowercase_split)


def test_all_title_tokens_exclude_the_aspect_and_partial_overlap_is_kept() -> None:
    """The documented example: filter/water filter/mwf filter out, water flow in."""
    token_sets = _water_filter_sets()
    tokens = token_sets[WATER_FILTER.asin]
    # The set really comes from title, brand and category tokens.
    assert {"refrigerator", "water", "filter", "ge", "mwf"} <= tokens

    for aspect in ("filter", "water filter", "mwf filter", "ge", "ge filter", "refrigerator"):
        assert is_self_reference(aspect, tokens), aspect
    for aspect in ("water flow", "fit", "box", "noise", "filter life", "geyser"):
        assert not is_self_reference(aspect, tokens), aspect


def test_excluded_and_kept_rows_are_partitioned_by_the_rule() -> None:
    """The filter drops only rows whose tokens all name the product itself."""
    token_sets = _water_filter_sets()
    rows = [
        _stats(WATER_FILTER.asin, "filter", negative=6, classification=ASPECT_PAIN_POINT),
        _stats(WATER_FILTER.asin, "water filter", negative=4, classification=ASPECT_PAIN_POINT),
        _stats(WATER_FILTER.asin, "water flow", positive=8),
        _stats(WATER_FILTER.asin, "fit", positive=7),
    ]
    result = filter_self_reference(rows, token_sets)
    assert [stat.aspect for stat in result.excluded] == ["filter", "water filter"]
    assert [stat.aspect for stat in result.kept] == ["water flow", "fit"]


def test_flag_off_keeps_every_row() -> None:
    """With exclude_self_reference disabled nothing is dropped."""
    token_sets = _water_filter_sets()
    rows = [
        _stats(WATER_FILTER.asin, "filter", negative=6),
        _stats(WATER_FILTER.asin, "water filter", negative=4),
    ]
    result = filter_self_reference(rows, token_sets, enabled=False)
    assert result.kept == rows
    assert result.excluded == []


def test_missing_metadata_keeps_every_row() -> None:
    """An empty token set proves nothing, so nothing is excluded."""
    token_sets = build_self_token_sets(
        [
            ProductIdentity(
                asin="B000000000", title_clean=None, brand=None, category_leaf=None
            )
        ],
        AspectsConfig(),
        _lowercase_split,
    )
    assert token_sets["B000000000"] == frozenset()
    result = filter_self_reference([_stats("B000000000", "filter")], token_sets)
    assert result.kept and not result.excluded


def test_build_is_deterministic_and_filter_is_idempotent() -> None:
    """Identical inputs give identical sets, and re-filtering excludes nothing."""
    first = _water_filter_sets()
    second = _water_filter_sets()
    assert first == second

    rows = [
        _stats(WATER_FILTER.asin, "filter", negative=6),
        _stats(WATER_FILTER.asin, "water flow", positive=8),
        _stats(WATER_FILTER.asin, "fit", positive=7),
    ]
    once = filter_self_reference(rows, first)
    twice = filter_self_reference(once.kept, first)
    assert twice.kept == once.kept
    assert twice.excluded == []


def test_excluded_summary_ranks_rows_then_mentions_then_aspect() -> None:
    """The report ordering is total: rows, then mentions, then aspect name."""
    rows = [
        _stats("B1", "filter", negative=10),
        _stats("B2", "filter", negative=3),
        _stats("B3", "water filter", positive=30),
        _stats("B4", "water filter", positive=2),
    ]
    summary = excluded_aspect_summary(rows, limit=2)
    # Both have two rows, so excluded mentions decide: water filter (32) first.
    assert [(entry.aspect, entry.rows, entry.mentions) for entry in summary] == [
        ("water filter", 2, 32),
        ("filter", 2, 13),
    ]
    assert excluded_aspect_summary(rows, limit=2) == summary
    # A pure tie (same rows, same mentions) resolves on the aspect name.
    tied = [
        _stats("B1", "zebra", positive=5),
        _stats("B2", "apple", positive=5),
    ]
    assert [entry.aspect for entry in excluded_aspect_summary(tied, limit=2)] == [
        "apple",
        "zebra",
    ]


def test_cache_roundtrip_and_fingerprint_mismatch(tmp_path: Path) -> None:
    """A matching fingerprint reads the sets back; anything else rebuilds."""
    config = AspectsConfig()
    token_sets = _water_filter_sets()
    fingerprint = self_reference_fingerprint(config, [WATER_FILTER])
    path = tmp_path / "self_tokens.json"

    assert load_self_token_cache(path, fingerprint) is None  # missing file

    save_self_token_cache(path, token_sets, fingerprint)
    assert load_self_token_cache(path, fingerprint) == token_sets

    other = dict(fingerprint)
    other["products_sha256"] = "0" * 64
    assert load_self_token_cache(path, other) is None  # metadata changed

    path.write_text("{not json", encoding="utf-8")
    assert load_self_token_cache(path, fingerprint) is None  # corrupt file


def test_fingerprint_is_order_independent_and_tracks_metadata() -> None:
    """Hashing sorts by ASIN, and a title change changes the fingerprint."""
    config = AspectsConfig()
    other = ProductIdentity(
        asin="B00000020C",
        title_clean="Samsung Refrigerator Water Filter",
        brand="Samsung",
        category_leaf="Water Filter",
    )
    forward = self_reference_fingerprint(config, [WATER_FILTER, other])
    reverse = self_reference_fingerprint(config, [other, WATER_FILTER])
    assert forward == reverse

    changed = ProductIdentity(
        asin=WATER_FILTER.asin,
        title_clean="General Electric MWF Refrigerator Filter",
        brand=WATER_FILTER.brand,
        category_leaf=WATER_FILTER.category_leaf,
    )
    assert self_reference_fingerprint(config, [changed]) != forward


def test_spacy_batch_lemmatizer_reads_token_lemmas() -> None:
    """The adapter surfaces ``token.lemma_`` per document, in input order."""

    class _Token:
        def __init__(self, lemma: str) -> None:
            self.lemma_ = lemma

    class _Doc:
        def __init__(self, lemmas: list[str]) -> None:
            self._tokens = [_Token(lemma) for lemma in lemmas]

        def __iter__(self):
            return iter(self._tokens)

    class _Nlp:
        def pipe(self, texts):
            return [_Doc(list(text)) for text in texts]

    lemmatize = spacy_batch_lemmatizer(_Nlp())
    assert list(lemmatize(["ab", "cd"])) == [["a", "b"], ["c", "d"]]


def test_min_token_chars_filters_title_tokens() -> None:
    """Single-letter title tokens never enter the set (same rule as aspects)."""
    identity = ProductIdentity(
        asin="B00000000X",
        title_clean="A 3d Printing Pen",
        brand=None,
        category_leaf=None,
    )
    token_sets = build_self_token_sets([identity], AspectsConfig(), _lowercase_split)
    tokens = token_sets[identity.asin]
    assert "a" not in tokens  # too short
    assert "3d" not in tokens  # not alphabetic
    assert {"printing", "pen"} <= tokens
