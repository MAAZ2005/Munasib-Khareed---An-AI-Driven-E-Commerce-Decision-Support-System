"""Page 2: Product Overview.

Renders the full profile of the selected product from ``GET /products/{asin}``
(image, title, brand, price, sales rank, review counts, rating, review date range,
features and description) plus the related-product counts and the data-basis block
from the composed analysis. Only real fields are shown: a missing price, rank,
brand or date is displayed as its explicit absence, never as a fabricated value.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path so 'app.*' imports resolve
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from app.cache import get_cached_product  # noqa: E402
from app.components import metric_card, product_image  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from app.errors import render_api_error  # noqa: E402
from app.formatting import (  # noqa: E402
    NOT_AVAILABLE,
    format_count,
    format_date,
    format_price,
    format_rank,
    format_rating,
)
from app.state import get_or_load_analysis, require_product  # noqa: E402
from app.theme import apply_theme  # noqa: E402
from munasib.api.schemas import ProductDetail  # noqa: E402

#: Larger thumbnail edge for the profile view.
IMAGE_WIDTH = 220


def _render_profile(asin: str) -> ProductDetail | None:
    """Render the product identity and profile fields.

    Args:
        asin: The selected product ASIN.

    Returns:
        The loaded product detail, or None when the profile could not be fetched
        (in which case a friendly message has already been rendered).
    """
    try:
        product = get_cached_product(asin)
    except Exception as exc:
        render_api_error(exc)
        return None

    col_image, col_info = st.columns([1, 2.4])
    with col_image:
        product_image(asin, product.has_image, width=IMAGE_WIDTH, title=product.title)

    with col_info:
        st.subheader(product.title or NOT_AVAILABLE)
        st.markdown(
            f"**Brand:** {product.brand or NOT_AVAILABLE}  \n"
            f"**Price:** {format_price(product.price, product.price_max, product.price_status)}  \n"
            f"**Sales rank:** {format_rank(product.rank_missing, product.rank_value, product.rank_category)}"
        )

    st.divider()

    col_reviews, col_rating, col_first, col_last = st.columns(4)
    with col_reviews:
        metric_card("Reviews in subset", product.n_reviews)
    with col_rating:
        metric_card("Mean rating", format_rating(product.mean_rating))
    with col_first:
        metric_card("First review", format_date(product.first_review_time))
    with col_last:
        metric_card("Last review", format_date(product.last_review_time))

    return product


def _render_features(product_description: str | None, features: list[str] | None) -> None:
    """Render the feature bullets and the long description in an expander.

    Args:
        product_description: Cleaned description, or None when absent.
        features: Cleaned feature bullets, or None when absent.
    """
    st.subheader("Features")
    if features:
        st.markdown("\n".join(f"- {feature}" for feature in features))
    else:
        st.info("No feature bullets were published for this product.")

    with st.expander("Full product description"):
        st.write(product_description or "No description was published for this product.")


def _render_related(asin: str) -> None:
    """Render the related-product counts and the data-basis provenance block.

    Args:
        asin: The selected product ASIN.
    """
    try:
        analysis = get_or_load_analysis(asin)
    except Exception as exc:
        # The profile above is still valid; only the provenance block is missing.
        render_api_error(exc)
        return

    alternatives = analysis.alternatives
    st.subheader("Related products")
    st.markdown(
        f"- **Related products in the catalogue:** {format_count(alternatives.total_related)}\n"
        f"- **Usable in-subset alternatives (with an analysis):** {format_count(alternatives.usable)}"
    )

    basis = analysis.data_basis
    st.caption(
        "Data basis · "
        f"category `{basis.category}` · "
        f"{format_count(basis.n_reviews)} reviews · "
        f"{format_date(basis.date_range_start)} to {format_date(basis.date_range_end)} · "
        f"reference date {format_date(basis.reference_date)} · "
        f"pipeline `{analysis.pipeline_version}`"
    )


def main() -> None:
    """Render the Product Overview page."""
    apply_theme("Product Overview")
    render_sidebar()

    asin = require_product()

    st.title("📦 2. Product Overview")
    st.caption(f"Active subject ASIN: `{asin}`")

    product = _render_profile(asin)
    if product is None:
        return

    _render_features(product.description, product.features)
    st.divider()
    _render_related(asin)


if __name__ == "__main__":
    main()
