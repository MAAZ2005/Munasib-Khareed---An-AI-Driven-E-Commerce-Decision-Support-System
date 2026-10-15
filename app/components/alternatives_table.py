"""Alternatives table component: side-by-side comparison of the alternatives."""

from __future__ import annotations

from munasib.api.schemas import AlternativeProduct, AlternativeRelation, AlternativesResult

#: Rendered when there is no alternatives block at all. Explicit absence.
NO_ALTERNATIVES_MESSAGE = "No alternatives were produced for this product yet."

#: Icon per relation, always shown next to the relation word so the comparison
#: is never signalled by color alone.
RELATION_ICONS: dict[str, str] = {
    AlternativeRelation.BETTER.value: "✅",
    AlternativeRelation.COMPARABLE.value: "➖",
    AlternativeRelation.WORSE.value: "⚠️",
}

#: Column order of the rendered table.
COLUMNS: tuple[str, ...] = (
    "Relation",
    "ASIN",
    "Title",
    "Price",
    "Reviews",
    "Rating",
    "Verdict",
    "Risk score",
    "Δ Risk",
    "Δ Evidence",
    "Δ Rating",
)


def _signed(value: float | None) -> str:
    """Format a signed delta, showing absence instead of a fabricated zero.

    Args:
        value: The delta, or None when it could not be computed.

    Returns:
        The delta with an explicit sign, or "n/a" when it is missing.
    """
    return "n/a" if value is None else f"{value:+.2f}"


def _row(product: AlternativeProduct) -> dict[str, str]:
    """Build one display row for one alternative.

    Every cell is either a real computed value or the explicit "n/a"; missing
    prices and ratings are never imputed.

    Args:
        product: The alternative to render.

    Returns:
        Mapping of column label to display text, in :data:`COLUMNS` order.
    """
    summary = product.summary
    relation = product.relation.value
    return {
        "Relation": f"{RELATION_ICONS.get(relation, 'ℹ️')} {relation}",
        "ASIN": summary.asin,
        "Title": summary.title or "not available",
        "Price": f"${summary.price:,.2f}" if summary.price is not None else "n/a",
        "Reviews": f"{summary.n_reviews:,}",
        "Rating": f"{summary.mean_rating:.2f}" if summary.mean_rating is not None else "n/a",
        "Verdict": product.verdict.value,
        "Risk score": f"{product.risk_score:.1f}",
        "Δ Risk": _signed(product.deltas.risk_score_delta),
        "Δ Evidence": _signed(product.deltas.evidence_strength_delta),
        "Δ Rating": _signed(product.deltas.mean_rating_delta),
    }


def alternatives_table(result: AlternativesResult | None) -> None:
    """Render the alternatives comparison table.

    Deltas are measured against the selected product: a negative "Δ Risk" means
    the alternative is safer than the product on screen.

    Args:
        result: The alternatives block, or None when it was not produced. An
            empty result shows its own ``empty_reason``.
    """
    import streamlit as st

    if result is None:
        st.info(NO_ALTERNATIVES_MESSAGE)
        return

    if not result.items:
        st.info(result.empty_reason or NO_ALTERNATIVES_MESSAGE)
        return

    rows = [_row(product) for product in result.items]
    st.dataframe([{column: row[column] for column in COLUMNS} for row in rows], hide_index=True)

    legend = " · ".join(
        f"{RELATION_ICONS.get(relation.value, 'ℹ️')} {relation.value}" for relation in AlternativeRelation
    )
    st.caption(f"Relations: {legend}. Deltas compare each alternative with the selected product.")
    st.caption(
        f"{result.usable} of {result.total_related} related product(s) are usable in this subset; "
        f"{len(result.items)} shown, ranked by the pipeline."
    )


def build_alternatives_rows(result: AlternativesResult | None) -> list[dict[str, str]]:
    """Build the comparison rows without rendering them.

    Args:
        result: The alternatives block, or None.

    Returns:
        One display mapping per alternative, empty when there is nothing to
        compare.
    """
    if result is None:
        return []
    return [_row(product) for product in result.items]
