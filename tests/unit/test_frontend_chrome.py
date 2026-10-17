"""Unit tests for the new marketplace-chrome pure helpers.

Covers:
* :mod:`app.branding` - the offline logo loader (a missing asset is ``None``,
  never an exception);
* :func:`app.components.sidebar.is_stub_analysis` - the one place the "is this
  stub output?" rule lives;
* :mod:`app.components.product_card` - the escaped, absence-aware card builders.
"""

from __future__ import annotations

from pathlib import Path

from app.branding import (
    APP_NAME,
    load_logo_bytes,
    logo_is_available,
    logo_path,
)
from app.components.product_card import (
    card_brand_html,
    card_price_html,
    card_rating_html,
    card_title_html,
)
from app.components.sidebar import is_stub_analysis
from app.formatting import BRAND_NOT_LISTED, PRICE_NOT_LISTED
from munasib.api.schemas import AnalysisResponse
from munasib.api.schemas.analysis import ANALYSIS_EXAMPLE

# --------------------------------------------------------------------------- #
# app.branding
# --------------------------------------------------------------------------- #


def test_missing_logo_returns_none(tmp_path: Path) -> None:
    """A missing logo file is a normal None, not an error."""
    assert load_logo_bytes(tmp_path / "absent.png") is None
    assert logo_is_available(tmp_path / "absent.png") is False


def test_directory_is_not_a_logo(tmp_path: Path) -> None:
    """A directory path yields None rather than raising."""
    assert load_logo_bytes(tmp_path) is None


def test_readable_file_returns_its_bytes(tmp_path: Path) -> None:
    """A readable file returns its non-empty bytes."""
    target = tmp_path / "logo.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert load_logo_bytes(target) == b"\x89PNG\r\n\x1a\n"
    assert logo_is_available(target) is True


def test_committed_logo_is_available() -> None:
    """The committed logo asset exists and is readable."""
    assert logo_path().is_file()
    assert logo_is_available() is True
    assert APP_NAME == "Munasib Khareed"


# --------------------------------------------------------------------------- #
# is_stub_analysis
# --------------------------------------------------------------------------- #


def _analysis(*, version: str, warnings: list[str]) -> AnalysisResponse:
    """Build an analysis with a chosen pipeline version and warnings."""
    payload = dict(ANALYSIS_EXAMPLE)
    payload["pipeline_version"] = version
    payload["warnings"] = warnings
    return AnalysisResponse.model_validate(payload)


def test_is_stub_analysis_detects_version_and_warning() -> None:
    """Stub is flagged by the version prefix or a STUB: warning."""
    assert is_stub_analysis(None) is False
    assert is_stub_analysis(_analysis(version="stub-0", warnings=[])) is True
    assert is_stub_analysis(_analysis(version="0.1.0", warnings=["STUB: placeholder"])) is True
    assert is_stub_analysis(_analysis(version="0.1.0", warnings=["info"])) is False


# --------------------------------------------------------------------------- #
# product_card builders
# --------------------------------------------------------------------------- #


def test_card_price_html_marks_a_missing_price() -> None:
    """A missing price is the explicit absence, never a fabricated value."""
    assert PRICE_NOT_LISTED in card_price_html(None)
    assert "mk-price-missing" in card_price_html(None)
    assert "$12.66" in card_price_html(12.66)
    assert "mk-price" in card_price_html(12.66)


def test_card_rating_html_always_carries_the_number_and_count() -> None:
    """The star row prints the numeric rating and the review count beside the stars."""
    html = card_rating_html(2.63, 52)
    assert "★★★☆☆" in html
    assert "2.63 / 5" in html
    assert "52 reviews" in html


def test_card_title_and_brand_escape_html() -> None:
    """Product-derived strings are HTML-escaped before they reach the page."""
    title_html = card_title_html('<script>alert(1)</script>')
    assert "<script>" not in title_html
    assert "&lt;script&gt;" in title_html

    brand_html = card_brand_html("Brand <b>X</b>")
    assert "<b>X</b>" not in brand_html
    assert "&lt;b&gt;X&lt;/b&gt;" in brand_html


def test_card_brand_html_shows_absence() -> None:
    """A missing brand renders its explicit absence."""
    assert BRAND_NOT_LISTED in card_brand_html(None)
