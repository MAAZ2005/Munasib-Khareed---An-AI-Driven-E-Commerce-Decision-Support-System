"""Unit tests for the frontend's pure display helpers (``app.formatting``).

These functions have no Streamlit dependency, so they are tested directly: price
and rank formatting (including the explicit "not listed" absences), the CSV
summary builder and the review/indicator table row builders. Missing values must
never become a fabricated zero.
"""

from __future__ import annotations

import copy
import csv
import io
from datetime import UTC, datetime

from app.formatting import (
    NOT_AVAILABLE,
    PRICE_NOT_LISTED,
    RANK_NOT_LISTED,
    analysis_json,
    analysis_json_filename,
    build_indicator_rows,
    build_review_rows,
    build_summary_csv,
    build_summary_rows,
    format_count,
    format_date,
    format_price,
    format_rank,
    format_rating,
    format_signed,
    format_unit,
    summary_csv_filename,
)
from munasib.api.schemas import AnalysisResponse
from munasib.api.schemas.analysis import ANALYSIS_EXAMPLE
from munasib.api.schemas.reviews import REVIEW_SNIPPET_EXAMPLE


def _analysis() -> AnalysisResponse:
    """Return the composed analysis example.

    Returns:
        A validated analysis built from the schema example.
    """
    return AnalysisResponse.model_validate(ANALYSIS_EXAMPLE)


def test_format_price_shows_absence_for_a_missing_price() -> None:
    """A missing price must render "Price not listed", never "$0.00"."""
    assert format_price(None) == PRICE_NOT_LISTED


def test_format_price_formats_a_single_price() -> None:
    """A single price is formatted with two decimals and a dollar sign."""
    assert format_price(12.66) == "$12.66"
    assert format_price(12.66, price_max=19.99, price_status="single") == "$12.66"


def test_format_price_formats_a_range() -> None:
    """A range price shows both ends when the status says so."""
    assert format_price(5.0, price_max=9.0, price_status="range") == "$5.00 - $9.00"


def test_format_rank_shows_absence_when_missing() -> None:
    """A rank flagged missing is never rendered as a number."""
    assert format_rank(True, 4165, "Appliances") == RANK_NOT_LISTED
    assert format_rank(False, None, "Appliances") == RANK_NOT_LISTED


def test_format_rank_formats_with_and_without_category() -> None:
    """A rank is formatted with thousands separators and an optional category."""
    assert format_rank(False, 4165, "Appliances") == "#4,165 in Appliances"
    assert format_rank(False, 4165, None) == "#4,165"


def test_format_rating_and_count_show_absence() -> None:
    """A missing rating or count is an explicit "n/a"."""
    assert format_rating(None) == NOT_AVAILABLE
    assert format_rating(2.634615) == "2.63 / 5"
    assert format_count(None) == NOT_AVAILABLE
    assert format_count(425993) == "425,993"


def test_format_date_uses_iso_date() -> None:
    """A datetime renders as an ISO date; None is "n/a"."""
    assert format_date(None) == NOT_AVAILABLE
    assert format_date(datetime(2018, 2, 3, tzinfo=UTC)) == "2018-02-03"


def test_format_unit_and_signed() -> None:
    """Unit numbers keep their digits; signed deltas keep their sign."""
    assert format_unit(None) == NOT_AVAILABLE
    assert format_unit(0.7173) == "0.72"
    assert format_unit(62.5, digits=1) == "62.5"
    assert format_signed(None) == NOT_AVAILABLE
    assert format_signed(-34.6) == "-34.60"
    assert format_signed(0.06) == "+0.06"


def test_build_review_rows_marks_verified_tristate() -> None:
    """A review whose source did not state verification reads "unknown"."""
    from munasib.api.schemas import ReviewSnippet

    snippet = ReviewSnippet.model_validate(REVIEW_SNIPPET_EXAMPLE)
    rows = build_review_rows([snippet])
    assert len(rows) == 1
    row = rows[0]
    assert row["Rating"] == "1 ★"
    assert row["Sentiment"] == "negative"
    assert row["Verified"] == "yes"
    assert row["Date"] == "2018-02-03"
    assert row["Review text"].startswith("These knobs are supposed")


def test_build_review_rows_truncates_long_text() -> None:
    """Long review bodies are truncated for the table."""
    from munasib.api.schemas import ReviewSnippet

    snippet = ReviewSnippet.model_validate(REVIEW_SNIPPET_EXAMPLE)
    row = build_review_rows([snippet], max_text=10)[0]
    assert len(row["Review text"]) <= 10
    assert row["Review text"].endswith("…")


def test_build_indicator_rows_orders_and_formats() -> None:
    """The indicator table lists the fixed indicators first, then extras."""
    evidence = _analysis().evidence
    rows = build_indicator_rows(evidence)
    names = [row["Indicator"] for row in rows]
    assert names[:3] == ["verified_purchase", "helpfulness_ratio", "review_depth"]
    assert rows[0]["Value (0-1)"] == "0.9423"
    assert rows[0]["Raw value"] == "0.9423"
    assert rows[0]["Direction"] == "benefit"


def test_build_indicator_rows_marks_a_missing_raw_value() -> None:
    """An indicator without a raw value shows "n/a", never 0."""
    analysis = _analysis()
    indicator = analysis.evidence.helpfulness.model_copy(update={"raw_value": None})
    rows = build_indicator_rows(None, [indicator])
    assert rows[0]["Raw value"] == NOT_AVAILABLE
    assert rows[0]["Group"] == "risk"


def test_build_summary_rows_and_csv_are_explicit_about_absences() -> None:
    """The flat summary cites the real numbers and uses explicit absences."""
    analysis = _analysis()
    rows = dict(build_summary_rows(analysis))
    assert rows["asin"] == "B00009V3UF"
    assert rows["verdict"] == "CONSIDER WITH CAUTION"
    assert rows["risk_score_0_100"] == "62.50"
    assert rows["mean_rating"] == "2.63 / 5"

    csv_text = build_summary_csv(analysis)
    parsed = list(csv.reader(io.StringIO(csv_text)))
    assert parsed[0] == ["field", "value"]
    parsed_map = dict(parsed[1:])
    assert parsed_map["asin"] == "B00009V3UF"


def test_build_summary_csv_writes_explicit_absence_for_missing_values() -> None:
    """A missing price, verdict or score is an explicit absence, never a zero."""
    payload = copy.deepcopy(ANALYSIS_EXAMPLE)
    payload["product"]["price"] = None
    payload["product"]["price_missing"] = True
    payload["verdict"] = None
    payload["risk_score"] = None
    payload["evidence_strength"] = None
    analysis = AnalysisResponse.model_validate(payload)

    rows = dict(build_summary_rows(analysis))
    assert rows["price"] == PRICE_NOT_LISTED
    assert rows["verdict"] == NOT_AVAILABLE
    assert rows["risk_score_0_100"] == NOT_AVAILABLE
    assert rows["mean_rating"] == "2.63 / 5"


def test_analysis_json_round_trips() -> None:
    """The JSON download parses straight back into AnalysisResponse."""
    analysis = _analysis()
    restored = AnalysisResponse.model_validate_json(analysis_json(analysis))
    assert restored.asin == analysis.asin
    assert restored.verdict == analysis.verdict


def test_download_filenames() -> None:
    """Download filenames carry the ASIN."""
    assert analysis_json_filename("B00009V3UF") == "B00009V3UF_analysis.json"
    assert summary_csv_filename("B00009V3UF") == "B00009V3UF_summary.csv"
