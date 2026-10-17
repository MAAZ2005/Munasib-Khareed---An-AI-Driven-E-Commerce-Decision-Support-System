"""Pure display helpers shared by the dashboard pages.

Nothing here touches Streamlit, the API client or the database: these functions
turn validated schema values into display strings (and build the page-7 CSV
summary), so the pages stay thin and the rules can be unit-tested directly.

Missing data is always rendered as an explicit absence - ``"Price not listed"``,
``"Rank not listed"`` or ``"n/a"`` - never as a fabricated zero, in line with the
project rule that the UI must not invent a value the backend did not compute.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from munasib.api.schemas import (
        AnalysisResponse,
        EvidenceIndicator,
        EvidenceIndicators,
        ReviewSnippet,
    )

#: Rendered when a scalar is genuinely missing.
NOT_AVAILABLE = "n/a"

#: Rendered when the source published no parsable price.
PRICE_NOT_LISTED = "Price not listed"

#: Rendered when the source published no best-seller rank.
RANK_NOT_LISTED = "Rank not listed"

#: Rendered when the source published no parsable brand.
BRAND_NOT_LISTED = "Brand not listed"

#: Display order of the indicator rows built by :func:`build_indicator_rows`.
INDICATOR_COLUMNS: tuple[str, ...] = (
    "Group",
    "Indicator",
    "Value (0-1)",
    "Raw value",
    "Direction",
    "Description",
)


def _number(value: float, digits: int = 4) -> str:
    """Format a float with trailing zeros trimmed (whole values stay whole).

    Args:
        value: The number to format.
        digits: Maximum number of decimal places.

    Returns:
        A compact decimal string, e.g. ``0.9423``, ``206.31`` or ``1``.
    """
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def format_price(
    price: float | None,
    price_max: float | None = None,
    price_status: str | None = None,
) -> str:
    """Format a price, showing a range when the source published one.

    Args:
        price: Lower (or only) price in USD, or None when missing/unparseable.
        price_max: Upper end of a price range, when ``price_status`` is ``range``.
        price_status: ``single``, ``range``, ``missing`` or ``unparseable``.

    Returns:
        ``"$12.66"``, ``"$5.00 - $9.00"`` for a range, or
        :data:`PRICE_NOT_LISTED` when the price is missing.
    """
    if price is None:
        return PRICE_NOT_LISTED
    if price_status == "range" and price_max is not None:
        return f"${price:,.2f} - ${price_max:,.2f}"
    return f"${price:,.2f}"


def format_rank(
    rank_missing: bool,
    rank_value: int | None,
    rank_category: str | None = None,
) -> str:
    """Format a best-seller rank, or the explicit "not listed" absence.

    Args:
        rank_missing: True when the source published no rank.
        rank_value: Parsed rank integer (smaller is better), or None.
        rank_category: Category the rank was measured in, when known.

    Returns:
        ``"#4,165 in Appliances"`` (or ``"#4,165"`` without a category), or
        :data:`RANK_NOT_LISTED`.
    """
    if rank_missing or rank_value is None:
        return RANK_NOT_LISTED
    if rank_category:
        return f"#{rank_value:,} in {rank_category}"
    return f"#{rank_value:,}"


def format_rating(mean_rating: float | None) -> str:
    """Format a 1-5 mean rating, or ``"n/a"`` when no rated review exists.

    Args:
        mean_rating: Mean star rating, or None.

    Returns:
        ``"2.63 / 5"`` or :data:`NOT_AVAILABLE`.
    """
    return NOT_AVAILABLE if mean_rating is None else f"{mean_rating:.2f} / 5"


def format_count(value: int | None) -> str:
    """Format an integer count with thousands separators.

    Args:
        value: Count, or None.

    Returns:
        ``"425,993"`` or :data:`NOT_AVAILABLE`.
    """
    return NOT_AVAILABLE if value is None else f"{value:,}"


def format_date(value: datetime | None) -> str:
    """Format a UTC timestamp as an ISO date.

    Args:
        value: Timestamp, or None.

    Returns:
        ``"2018-02-03"`` or :data:`NOT_AVAILABLE`.
    """
    return NOT_AVAILABLE if value is None else value.strftime("%Y-%m-%d")


def format_unit(value: float | None, digits: int = 2) -> str:
    """Format a unit-interval (or other small) number.

    Args:
        value: The number, or None.
        digits: Number of decimal places.

    Returns:
        The fixed-point string, or :data:`NOT_AVAILABLE`.
    """
    return NOT_AVAILABLE if value is None else f"{value:.{digits}f}"


def format_signed(value: float | None, digits: int = 2) -> str:
    """Format a signed delta, showing absence instead of a fabricated zero.

    Args:
        value: The delta, or None when it could not be computed.
        digits: Number of decimal places.

    Returns:
        The delta with an explicit sign (e.g. ``"-34.60"``), or
        :data:`NOT_AVAILABLE`.
    """
    return NOT_AVAILABLE if value is None else f"{value:+.{digits}f}"


#: Filled and empty star glyphs. They are ordinary text characters rendered by the
#: operating system's fonts, so no icon font or CDN is needed (works offline).
STAR_FULL = "★"
STAR_EMPTY = "☆"


def format_stars(value: float | None, maximum: float = 5.0, stars: int = 5) -> str:
    """Render a rating as a row of filled/empty stars.

    The glyph row is an at-a-glance summary only: pages always print the exact
    numeric rating next to it, so a rating is never conveyed by the stars alone.

    Args:
        value: The mean rating, or None when the product has no rated review.
        maximum: The top of the rating scale (``5.0`` for Amazon-style ratings).
        stars: How many glyphs to draw.

    Returns:
        ``stars`` glyphs (filled then empty), or :data:`NOT_AVAILABLE` when the
        rating is missing.
    """
    if value is None or maximum <= 0:
        return NOT_AVAILABLE
    clamped = max(0.0, min(float(maximum), float(value)))
    filled = max(0, min(stars, int(round(clamped / maximum * stars))))
    return STAR_FULL * filled + STAR_EMPTY * (stars - filled)


def clamp_text(text: str | None, max_chars: int = 120, ellipsis: str = "…") -> str:
    """Trim a long label to ``max_chars`` for a compact card.

    Used for product titles, which are clamped to two lines in the card CSS and
    also hard-capped here so an unusually long title cannot balloon the layout. A
    missing title renders as the explicit :data:`NOT_AVAILABLE`.

    Args:
        text: The text to clamp, or None when it is missing.
        max_chars: Maximum length of the returned string.
        ellipsis: Marker appended when the text is clipped.

    Returns:
        The text unchanged when short, a clipped ``...`` version when long, or
        :data:`NOT_AVAILABLE` when there is nothing to show.
    """
    if text is None:
        return NOT_AVAILABLE
    stripped = text.strip()
    if not stripped:
        return NOT_AVAILABLE
    if max_chars <= 0:
        return ellipsis
    if len(stripped) <= max_chars:
        return stripped
    keep = max(1, max_chars - len(ellipsis))
    return stripped[:keep].rstrip() + ellipsis


def _indicator_row(indicator: EvidenceIndicator, group: str) -> dict[str, str]:
    """Build one display row for one evidence or risk indicator.

    Args:
        indicator: The indicator to render.
        group: ``"evidence"`` or ``"risk"``; the section it belongs to.

    Returns:
        A mapping in :data:`INDICATOR_COLUMNS` order, with ``"n/a"`` for a
        missing raw value.
    """
    return {
        "Group": group,
        "Indicator": indicator.name,
        "Value (0-1)": _number(indicator.value),
        "Raw value": NOT_AVAILABLE if indicator.raw_value is None else _number(indicator.raw_value),
        "Direction": indicator.direction.value,
        "Description": indicator.description,
    }


def build_indicator_rows(
    evidence: EvidenceIndicators | None,
    risk_indicators: Sequence[EvidenceIndicator] = (),
) -> list[dict[str, str]]:
    """Build the page-4 table rows for every evidence and risk indicator.

    The three fixed evidence indicators come first, then any extra evidence
    ``details``, then the risk-stage indicators - so the table shows the same
    indicators the radar draws, each with its own value, raw value, direction and
    plain-language description.

    Args:
        evidence: Evidence indicators, or None when none were computed.
        risk_indicators: The risk stage's indicators (possibly empty).

    Returns:
        One display mapping per indicator, in :data:`INDICATOR_COLUMNS` order.
    """
    rows: list[dict[str, str]] = []
    if evidence is not None:
        fixed = (evidence.verified_purchase, evidence.helpfulness, evidence.review_depth)
        for indicator in (*fixed, *evidence.details):
            rows.append(_indicator_row(indicator, "evidence"))
    for indicator in risk_indicators:
        rows.append(_indicator_row(indicator, "risk"))
    return rows


#: Display order of the review rows built by :func:`build_review_rows`.
REVIEW_COLUMNS: tuple[str, ...] = (
    "Rating",
    "Sentiment",
    "Verified",
    "Helpful votes",
    "Date",
    "Summary",
    "Review text",
)


def build_review_rows(
    reviews: Sequence[ReviewSnippet],
    max_text: int = 240,
) -> list[dict[str, str]]:
    """Build the page-3 review-table rows from review snippets.

    The weak-supervision sentiment label and the tri-state verified flag are shown
    exactly as the API emitted them: a review whose source did not state the
    verified flag reads ``"unknown"``, never ``"no"``. Long review bodies are
    truncated for the table (the full text stays in the CSV/JSON downloads).

    Args:
        reviews: The review snippets of the current page.
        max_text: Maximum characters of review text kept per row.

    Returns:
        One display mapping per review, in :data:`REVIEW_COLUMNS` order.
    """
    rows: list[dict[str, str]] = []
    for review in reviews:
        text = review.review_text or ""
        if len(text) > max_text:
            text = text[: max_text - 1].rstrip() + "…"
        rows.append(
            {
                "Rating": NOT_AVAILABLE if review.rating is None else f"{int(review.rating)} ★",
                "Sentiment": (
                    review.sentiment_label.value if review.sentiment_label else NOT_AVAILABLE
                ),
                "Verified": (
                    "yes" if review.verified is True else "no" if review.verified is False else "unknown"
                ),
                "Helpful votes": f"{review.helpful_votes:,}",
                "Date": format_date(review.review_time),
                "Summary": review.summary or NOT_AVAILABLE,
                "Review text": text or NOT_AVAILABLE,
            }
        )
    return rows


def build_summary_rows(analysis: AnalysisResponse) -> list[tuple[str, str]]:
    """Flatten an analysis into ``(field, value)`` pairs for the CSV summary.

    Only values the analysis actually carries are emitted; a missing value is the
    explicit ``"n/a"`` (or ``"Price not listed"`` for the price), never a zero.

    Args:
        analysis: The composed analysis response.

    Returns:
        The flat summary, in a stable documented order.
    """
    verdict = analysis.verdict
    product = analysis.product
    basis = analysis.data_basis
    return [
        ("asin", analysis.asin),
        ("title", product.title or NOT_AVAILABLE),
        ("brand", product.brand or BRAND_NOT_LISTED),
        ("price", format_price(product.price, product.price_max, product.price_status)),
        ("n_reviews", format_count(product.n_reviews)),
        ("mean_rating", format_rating(product.mean_rating)),
        ("verdict", verdict.label.value if verdict is not None else NOT_AVAILABLE),
        ("risk_score_0_100", format_unit(analysis.risk_score)),
        ("evidence_strength_0_1", format_unit(analysis.evidence_strength)),
        ("category", basis.category),
        ("reviews_in_analysis", format_count(basis.n_reviews)),
        ("date_range_start", format_date(basis.date_range_start)),
        ("date_range_end", format_date(basis.date_range_end)),
        ("reference_date", format_date(basis.reference_date)),
        ("pipeline_version", analysis.pipeline_version),
        ("generated_at", format_date(analysis.generated_at)),
        ("major_concerns", format_count(len(verdict.major_concerns)) if verdict else "0"),
        ("usable_alternatives", format_count(analysis.alternatives.usable)),
        ("total_related_products", format_count(analysis.alternatives.total_related)),
    ]


def build_summary_csv(analysis: AnalysisResponse) -> str:
    """Build the flat CSV summary of an analysis (page 7 download).

    Args:
        analysis: The composed analysis response.

    Returns:
        CSV text with a ``field,value`` header, one row per summary field,
        ``\\n`` line endings and no trailing blank line beyond the final row.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["field", "value"])
    writer.writerows(build_summary_rows(analysis))
    return buffer.getvalue()


def summary_csv_filename(asin: str) -> str:
    """Return the download filename for a product's CSV summary.

    Args:
        asin: Product ASIN.

    Returns:
        ``"<asin>_summary.csv"``.
    """
    return f"{asin}_summary.csv"


def analysis_json_filename(asin: str) -> str:
    """Return the download filename for a product's full analysis JSON.

    Args:
        asin: Product ASIN.

    Returns:
        ``"<asin>_analysis.json"``.
    """
    return f"{asin}_analysis.json"


def analysis_json(analysis: AnalysisResponse) -> str:
    """Serialise the full analysis as pretty JSON (page 7's JSON download).

    Emits exactly the API contract, so the downloaded file parses straight back
    into :class:`~munasib.api.schemas.AnalysisResponse`.

    Args:
        analysis: The composed analysis response.

    Returns:
        The indented JSON text of the analysis.
    """
    return analysis.model_dump_json(indent=2)
