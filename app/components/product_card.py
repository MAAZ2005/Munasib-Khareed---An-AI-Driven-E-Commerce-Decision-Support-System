"""Marketplace-style product card shared by the selection and alternatives grids.

The card follows the conventions of a large online shop - a fixed-height image box,
a title clamped to two lines, a star row that always carries the numeric rating and
the review count, a prominent price (or its explicit absence) and one primary
button - while rendering everything with native Streamlit elements plus a small,
**HTML-escaped** amount of markup.

Escaping rule: every product-derived string (title, brand, price label) passes
through :func:`html.escape` before it is written into HTML, so a title that
contains ``<`` or ``&`` can never inject markup. The two pure builders below
(:func:`card_price_html`, :func:`card_rating_html`) are unit-tested directly.
"""

from __future__ import annotations

from html import escape

from app.formatting import (
    BRAND_NOT_LISTED,
    PRICE_NOT_LISTED,
    clamp_text,
    format_count,
    format_price,
    format_rating,
    format_stars,
)

#: Nominal card width in pixels: fixed so a wrapping horizontal container behaves
#: like a responsive grid (4 across on a wide screen, fewer on a narrow one).
CARD_WIDTH = 300

#: Fixed height of the image box so every card lines up.
IMAGE_BOX_HEIGHT = 190

#: Rendered image edge inside the box.
IMAGE_WIDTH = 150

#: Hard character cap on the (two-line) card title.
TITLE_MAX_CHARS = 90


def card_price_html(
    price: float | None,
    price_max: float | None = None,
    price_status: str | None = None,
) -> str:
    """Build the price line of a card as escaped HTML.

    A listed price is shown in large type; a missing price is the explicit
    "Price not listed", never a fabricated value.

    Args:
        price: The price, or None when the source published none.
        price_max: Upper end of a price range, when applicable.
        price_status: ``single``, ``range``, ``missing`` or ``unparseable``.

    Returns:
        A ``<div>`` fragment safe to render with ``unsafe_allow_html``.
    """
    if price is None:
        return f'<div class="mk-price-missing">{escape(PRICE_NOT_LISTED)}</div>'
    return f'<div class="mk-price">{escape(format_price(price, price_max, price_status))}</div>'


def card_rating_html(mean_rating: float | None, n_reviews: int | None) -> str:
    """Build the star row: glyphs plus the exact rating and review count.

    The stars are a summary; the numeric rating and the review count are always
    printed next to them, so the rating is never signalled by the stars alone.

    Args:
        mean_rating: The mean rating, or None.
        n_reviews: The number of reviews, or None.

    Returns:
        A ``<div>`` fragment safe to render with ``unsafe_allow_html``.
    """
    stars = format_stars(mean_rating)
    rating = format_rating(mean_rating)
    reviews = format_count(n_reviews)
    return (
        f'<div><span class="mk-stars" aria-hidden="true">{escape(stars)}</span> '
        f'<span class="mk-review-count">{escape(rating)} · {escape(reviews)} reviews</span></div>'
    )


def card_title_html(title: str | None) -> str:
    """Build the clamped, escaped two-line title.

    Args:
        title: The product title, or None.

    Returns:
        A ``<div>`` fragment safe to render with ``unsafe_allow_html``.
    """
    return f'<div class="mk-card-title">{escape(clamp_text(title, TITLE_MAX_CHARS))}</div>'


def card_brand_html(brand: str | None) -> str:
    """Build the escaped brand line, or its explicit absence.

    Args:
        brand: The product brand, or None.

    Returns:
        A ``<div>`` fragment safe to render with ``unsafe_allow_html``.
    """
    return f'<div class="mk-brand">{escape(brand or BRAND_NOT_LISTED)}</div>'


__all__ = [
    "CARD_WIDTH",
    "IMAGE_BOX_HEIGHT",
    "IMAGE_WIDTH",
    "TITLE_MAX_CHARS",
    "card_brand_html",
    "card_price_html",
    "card_rating_html",
    "card_title_html",
]
