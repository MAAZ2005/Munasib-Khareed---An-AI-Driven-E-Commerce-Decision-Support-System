"""Tests for spaCy noun-chunk canonicalisation and extraction behaviour.

The tests use tiny fake token/chunk objects, so they exercise the real
canonicaliser without loading a model or requiring a GPU/network.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from munasib.core.config import AspectsConfig
from munasib.review_intel.aspects.extraction import (
    ExtractionTiming,
    canonical_aspect,
    extract_doc_aspects,
    extract_text_aspects,
    filter_aspect_phrase,
    measure_extraction_speed,
)


@dataclass(frozen=True)
class FakeToken:
    """Minimal token surface read by the canonicaliser."""

    text: str
    lemma: str
    pos: str = "NOUN"
    alpha: bool = True
    numeric: bool = False

    @property
    def pos_(self) -> str:
        """Return the POS tag."""
        return self.pos

    @property
    def lemma_(self) -> str:
        """Return the lemma."""
        return self.lemma

    @property
    def is_alpha(self) -> bool:
        """Return whether the token is alphabetic."""
        return self.alpha

    @property
    def like_num(self) -> bool:
        """Return whether the token looks numeric."""
        return self.numeric


class FakeDoc:
    """Minimal document with noun chunks and a token list."""

    def __init__(self, chunks: list[list[FakeToken]], extra: list[FakeToken] | None = None) -> None:
        self.noun_chunks = chunks
        self._tokens = [token for chunk in chunks for token in chunk] + list(extra or [])

    def __iter__(self):
        """Iterate over every token."""
        return iter(self._tokens)


def tok(text: str, lemma: str | None = None, pos: str = "NOUN", **kwargs: object) -> FakeToken:
    """Build a fake token.

    Args:
        text: Surface text.
        lemma: Lemma (defaults to lowercased text).
        pos: POS tag.
        **kwargs: Overrides for alpha/numeric.

    Returns:
        Fake token.
    """
    return FakeToken(
        text=text,
        lemma=lemma or text.lower(),
        pos=pos,
        alpha=bool(kwargs.get("alpha", True)),
        numeric=bool(kwargs.get("numeric", False)),
    )


def test_canonical_aspect_keeps_the_head_and_nearest_modifier() -> None:
    """A long chunk collapses to its two rightmost nominal words."""
    tokens = [
        tok("the", "the", pos="DET"),
        tok("stainless", "stainless", pos="ADJ"),
        tok("steel", "steel"),
        tok("ice", "ice"),
        tok("maker", "maker"),
    ]
    assert canonical_aspect(tokens, AspectsConfig()) == "ice maker"


def test_canonical_aspect_filters_generic_nouns_but_keeps_service() -> None:
    """A generic modifier is skipped; the legitimate head survives."""
    config = AspectsConfig()
    tokens = [tok("customer"), tok("service")]
    assert canonical_aspect(tokens, config) == "service"
    assert canonical_aspect(tokens, config, apply_stoplist=False) == "customer service"
    assert canonical_aspect([tok("delivery")], config) == "delivery"


def test_canonical_aspect_rejects_short_numeric_and_non_alpha_tokens() -> None:
    """Single letters, punctuation-like tokens and numbers are not aspects."""
    config = AspectsConfig(min_token_chars=3)
    assert canonical_aspect([tok("tv")], config) is None
    assert canonical_aspect([tok("100", numeric=True)], config) is None
    assert canonical_aspect([tok("a-b", alpha=False)], config) is None
    assert canonical_aspect([tok("fan")], config) == "fan"


def test_canonical_aspect_honours_max_aspect_tokens_one() -> None:
    """With one-token phrases only the head is kept."""
    config = AspectsConfig(max_aspect_tokens=1)
    assert canonical_aspect([tok("water"), tok("filter")], config) == "filter"


def test_filter_aspect_phrase_recanonicalises_cached_phrases() -> None:
    """Cached phrases get the same stoplist treatment as live extraction."""
    config = AspectsConfig()
    assert filter_aspect_phrase("ice maker", config) == "ice maker"
    # Generic modifier dropped, legitimate head kept - exactly what
    # canonical_aspect would have produced during a live extraction.
    assert filter_aspect_phrase("customer service", config) == "service"
    assert filter_aspect_phrase("year warranty", config) == "warranty"
    assert filter_aspect_phrase("baby clothe", config) == "clothe"
    # Person/meta tokens only: the phrase must not enter the vocabulary.
    assert filter_aspect_phrase("amazon review", config) is None
    assert filter_aspect_phrase("star rating", config) is None
    assert filter_aspect_phrase("product description", config) is None
    assert filter_aspect_phrase("husband", config) is None
    # Short and non-alphabetic tokens are dropped like in canonical_aspect.
    assert filter_aspect_phrase("a fan", config) == "fan"
    assert filter_aspect_phrase("3 way", config) is None


def test_filter_aspect_phrase_honours_max_aspect_tokens() -> None:
    """A two-token cap keeps the rightmost (head) tokens of a longer phrase."""
    assert filter_aspect_phrase("washing machine part", AspectsConfig()) == "machine part"
    assert (
        filter_aspect_phrase("washing machine part", AspectsConfig(max_aspect_tokens=1))
        == "part"
    )


def test_extract_doc_aspects_deduplicates_and_adds_unclaimed_nouns() -> None:
    """Repeated phrases collapse and a bare unclaimed noun is still captured."""
    chunk_a = [tok("the", "the", pos="DET"), tok("noise")]
    chunk_b = [tok("noise")]
    extra = tok("motor")
    doc = FakeDoc([chunk_a, chunk_b], extra=[extra])
    assert extract_doc_aspects(doc, AspectsConfig()) == ["noise", "motor"]


def test_extract_doc_aspects_applies_stoplist_only_when_asked() -> None:
    """The cached extraction can retain raw generic candidates for later filtering."""
    doc = FakeDoc([[tok("husband")], [tok("filter")]])
    assert extract_doc_aspects(doc, AspectsConfig()) == ["filter"]
    assert extract_doc_aspects(doc, AspectsConfig(), apply_stoplist=False) == [
        "husband",
        "filter",
    ]


def test_extract_text_aspects_truncates_before_parsing() -> None:
    """The configured character cap is applied to the text handed to spaCy."""
    received: list[str] = []

    class FakeNlp:
        def __call__(self, text: str) -> FakeDoc:
            received.append(text)
            return FakeDoc([[tok("filter")]])

    config = AspectsConfig(max_text_chars=5)
    assert extract_text_aspects("filter and motor", FakeNlp(), config) == ["filter"]
    assert received == ["filte"]


def test_extract_text_aspects_returns_empty_for_blank_text() -> None:
    """Blank prefixes never reach spaCy."""

    class ExplodingNlp:
        def __call__(self, text: str) -> FakeDoc:
            raise AssertionError("blank text must not be parsed")

    assert extract_text_aspects("   ", ExplodingNlp(), AspectsConfig()) == []


def test_measure_extraction_speed_uses_the_pipe_and_reports_throughput() -> None:
    """The timing helper counts parsed documents and computes a rate."""
    class FakeNlp:
        def pipe(self, texts, batch_size: int, n_process: int):
            for _text in texts:
                yield FakeDoc([[tok("filter")]])

    timing = measure_extraction_speed(FakeNlp(), ["a", "b"], AspectsConfig())
    assert timing.documents == 2
    assert timing.seconds >= 0.0
    if timing.seconds > 0:
        assert timing.documents_per_second > 0


def test_extraction_timing_never_invents_a_rate() -> None:
    """A zero-second sample has no finite estimate."""
    timing = ExtractionTiming(documents=0, seconds=0.0)
    assert timing.documents_per_second == 0.0
    assert timing.estimated_seconds(100) is None


def test_load_nlp_rejects_a_missing_model() -> None:
    """A missing model raises DataError with the download command."""
    from munasib.core.exceptions import DataError
    from munasib.review_intel.aspects.extraction import load_nlp

    with pytest.raises(DataError, match="spacy download"):
        load_nlp(AspectsConfig(spacy_model="definitely_missing_model_xyz"))
