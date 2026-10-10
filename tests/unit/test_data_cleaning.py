"""Unit tests for :mod:`munasib.data.cleaning` (pure text/field cleaning).

Covers the format variations seen in the real Appliances files: HTML in product
text, encoding artifacts, URLs in reviews, brand casing/punctuation, and the
rank-string variants verified against ``data/raw/meta_Appliances.json.gz``.

Tests are hermetic: no network, no GPU, no raw data files required.
"""

from __future__ import annotations

import pytest

from munasib.data.cleaning import (
    clean_description,
    clean_features,
    clean_review_text,
    clean_title,
    collapse_whitespace,
    fix_encoding_artifacts,
    normalize_brand,
    parse_rank_fields,
    remove_urls,
    strip_html,
)


class TestStripHtml:
    """HTML removal with the stdlib parser (no regex guessing)."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("plain text", "plain text"),
            ("<b>Bold</b> and <i>italic</i>", "Bold and italic"),
            ("a &amp; b", "a & b"),
            ("<br/>line one<br/>line two", " line one line two"),
            ("<img src='x.jpg' alt='a > b'>text", "text"),
        ],
    )
    def test_strips_tags_and_decodes_entities(self, raw: str, expected: str) -> None:
        """Tags are removed, entities decoded, non-tag content preserved."""
        assert strip_html(raw) == expected

    def test_script_content_is_dropped(self) -> None:
        """Inline script/style content is code, not product text."""
        result = strip_html("<p>Hi <script>var x = '<b>no</b>';</script>there</p>")
        assert "var x" not in result
        assert "<b>no</b>" not in result
        assert "Hi" in result and "there" in result

    def test_no_markup_input_only_unescapes(self) -> None:
        """Input without '<' is entity-decoded but otherwise untouched."""
        assert strip_html("Tom & Jerry") == "Tom & Jerry"


class TestFixEncodingArtifacts:
    """Encoding repairs that never invent characters."""

    def test_nfc_normalisation(self) -> None:
        """Decomposed unicode is composed (NFC)."""
        assert fix_encoding_artifacts("Cafe\u0301") == "Café"

    def test_replacement_chars_removed(self) -> None:
        """U+FFFD artifacts from decoding bad bytes are dropped, not kept."""
        assert fix_encoding_artifacts("abc\ufffddef") == "abcdef"

    def test_html_entities_unescaped(self) -> None:
        """Stray entities are decoded."""
        assert fix_encoding_artifacts("hot &amp; cold") == "hot & cold"

    def test_case_and_punctuation_preserved(self) -> None:
        """Review text keeps natural casing and punctuation."""
        assert fix_encoding_artifacts("GREAT product!!!") == "GREAT product!!!"


class TestRemoveUrls:
    """URL removal for review text."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("see https://amazon.com/dp/B00TEST now", "see now"),
            ("check http://example.com/a?b=1 ok", "check ok"),
            ("visit www.example.com now", "visit now"),
            ("no link here", "no link here"),
            ("dot.com/path is not a url", "dot.com/path is not a url"),
        ],
    )
    def test_url_patterns(self, raw: str, expected: str) -> None:
        """Scheme/www URLs are removed; ordinary text is untouched."""
        # remove_urls substitutes a space, so whitespace is normalised for the check.
        assert collapse_whitespace(remove_urls(raw)) == expected


class TestCollapseWhitespace:
    """Whitespace normalisation."""

    def test_collapses_runs_and_trims(self) -> None:
        """Newlines/tabs/runs become single spaces and edges are trimmed."""
        assert collapse_whitespace("  a \n\t b  ") == "a b"

    def test_empty_stays_empty(self) -> None:
        """Whitespace-only input collapses to an empty string."""
        assert collapse_whitespace("   ") == ""


class TestCleanReviewText:
    """Model-input cleaning for review text (case/punctuation preserved)."""

    @pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
    def test_missing_or_empty_returns_none(self, raw: object) -> None:
        """Absent/empty text nulls out instead of becoming ''."""
        assert clean_review_text(raw) is None

    def test_preserves_case_and_punctuation(self) -> None:
        """No lowercasing or punctuation stripping - the transformer needs it."""
        assert clean_review_text("GREAT product!!! Loved IT.") == "GREAT product!!! Loved IT."

    def test_removes_urls_and_collapses(self) -> None:
        """URLs are dropped and whitespace collapsed."""
        assert clean_review_text("Loved it https://x.co/abc\n\nreally") == "Loved it really"

    def test_fixes_entities(self) -> None:
        """HTML entities in raw text are decoded."""
        assert clean_review_text("hot &amp; cold") == "hot & cold"


class TestCleanTitleDescriptionFeatures:
    """Product text cleaning."""

    def test_title_strips_html(self) -> None:
        """Title tags are removed."""
        assert clean_title("<b>Acme</b> Toaster") == "Acme Toaster"

    @pytest.mark.parametrize("raw", [None, "", "  "])
    def test_title_missing_returns_none(self, raw: object) -> None:
        """Missing/empty title nulls out."""
        assert clean_title(raw) is None

    def test_description_list_joined(self) -> None:
        """List-of-paragraph descriptions join into one string."""
        assert clean_description(["First.", "<p>Second.</p>"]) == "First. Second."

    def test_description_plain_string(self) -> None:
        """Plain-string descriptions (older format) pass through cleaned."""
        assert clean_description("plain <b>string</b>") == "plain string"

    @pytest.mark.parametrize("raw", [None, "", []])
    def test_description_missing_returns_none(self, raw: object) -> None:
        """Absent/empty description nulls out."""
        assert clean_description(raw) is None

    def test_features_list_cleaned(self) -> None:
        """Feature bullets stay separate and get HTML-stripped."""
        assert clean_features(["F1", " <b>F2</b> "]) == ["F1", "F2"]

    @pytest.mark.parametrize("raw", [None, "", []])
    def test_features_missing_returns_none(self, raw: object) -> None:
        """Absent/empty features null out."""
        assert clean_features(raw) is None


class TestNormalizeBrand:
    """Brand comparison-key normalisation (original kept separately)."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Acme. ", "acme"),
            ("ACME", "acme"),
            ("acme", "acme"),
            ("BLACK+DECKER", "black+decker"),
            ("Acme Co.", "acme co"),
            ("Tom   &   Jerry", "tom & jerry"),
        ],
    )
    def test_normalises(self, raw: str, expected: str) -> None:
        """Trim, collapse, edge punctuation removed, lower-cased."""
        assert normalize_brand(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "..."])
    def test_missing_returns_none(self, raw: object) -> None:
        """Absent/empty/punctuation-only brands null out."""
        assert normalize_brand(raw) is None


class TestParseRankFields:
    """Rank-string parsing against the real Appliances formats."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Verified against data/raw/meta_Appliances.json.gz:
            (">#43,077 in Appliances (See top 100)", (43077, "Appliances")),
            (">#963 in Appliances (See top 100)", (963, "Appliances")),
            (
                ">#1,617 in Tools &amp; Home Improvement &gt; Appliances &gt; "
                "Large Appliance Accessories &gt; Range Parts &amp; Accessories",
                (
                    1617,
                    "Tools & Home Improvement > Appliances > Large Appliance "
                    "Accessories > Range Parts & Accessories",
                ),
            ),
            ("66,593 in Industrial & Scientific (", (66593, "Industrial & Scientific")),
            ("484 in Appliances (", (484, "Appliances")),
            ("#99 in Appliances", (99, "Appliances")),
            (">#46,879 in Home & Kitchen (See Top 100 in Home & Kitchen)", (46879, "Home & Kitchen")),
            ("1,234", (1234, None)),  # legacy salesRank value
            ("2,345", (2345, None)),
            ("►#7 in Appliances", (7, "Appliances")),  # U+25BA prefix form
        ],
    )
    def test_parses(self, raw: str, expected: tuple[int | None, str | None]) -> None:
        """Known formats parse into (rank_value, rank_category)."""
        assert parse_rank_fields(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "Appliances", "see top 100", "n/a"])
    def test_unparseable_returns_none_none(self, raw: object) -> None:
        """Missing/unparseable input never fabricates a rank."""
        assert parse_rank_fields(raw) == (None, None)
