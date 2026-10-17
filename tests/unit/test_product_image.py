"""Unit tests for the product-image component's pure helpers.

``product_image`` itself needs Streamlit (covered by the e2e tests); the two pure
helpers decide *whether* an image is drawn and *what letter* the placeholder tile
shows, so they are tested directly here.
"""

from __future__ import annotations

from app.components.product_image import (
    NO_IMAGE_LABEL,
    has_renderable_image,
    placeholder_initial,
)


def test_has_renderable_image_requires_flag_and_bytes() -> None:
    """An image is drawn only when the flag is true and the bytes arrived."""
    assert has_renderable_image(True, b"\xff\xd8\xff") is True
    assert has_renderable_image(False, b"\xff\xd8\xff") is False
    assert has_renderable_image(True, None) is False
    assert has_renderable_image(True, b"") is False
    assert has_renderable_image(False, None) is False


def test_placeholder_initial_uses_the_first_alphanumeric_character() -> None:
    """The tile shows the first alphanumeric character of the title."""
    assert placeholder_initial("Range Kleen 8234", "B00009V3UF") == "R"
    assert placeholder_initial("  123 knobs", "B00009V3UF") == "1"


def test_placeholder_initial_falls_back_to_the_asin() -> None:
    """A title with no alphanumeric character falls back to the ASIN."""
    assert placeholder_initial("---", "B00009V3UF") == "B"
    assert placeholder_initial(None, "b00009v3uf") == "B"


def test_placeholder_initial_handles_empty_inputs() -> None:
    """Nothing usable in title or ASIN yields the explicit "?"."""
    assert placeholder_initial(None, "") == "?"
    assert placeholder_initial("", "") == "?"


def test_no_image_label_text() -> None:
    """The placeholder's absence text is stable (the e2e tests assert on it)."""
    assert NO_IMAGE_LABEL == "No image available"
