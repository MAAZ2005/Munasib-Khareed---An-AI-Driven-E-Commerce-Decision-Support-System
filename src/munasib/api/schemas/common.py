"""Cross-cutting API models: shared value types, pagination, error envelope.

These models carry no product or analysis semantics; every other schema module
imports its numeric bounds from here so the same rule is enforced once:

* ``UnitFloat`` - every indicator, evidence strength and sub-score is ``0.0-1.0``.
* ``RiskScore`` - the consumer risk score is ``0.0-100.0`` (higher = riskier).
* ``StarRating`` - a star rating is an integer ``1-5``.
* ``UtcDatetime`` - every timestamp the API emits is UTC; a naive value is
  interpreted as UTC (the SQLite serving cache stores naive UTC).

Also home to the pagination envelope (:class:`Page`) and the error envelope
(:class:`ErrorResponse`) used by every endpoint, plus the two small response
models for the ``/health`` and ``/meta/category`` endpoints (they describe the
service itself rather than a product, so they do not belong to a product,
review or analysis module).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Generic, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

from munasib.core.constants import SPLIT_ORDER

T = TypeVar("T")


def _ensure_utc(value: datetime) -> datetime:
    """Normalise a datetime to UTC, interpreting naive input as UTC.

    Args:
        value: Datetime with or without an offset.

    Returns:
        The same instant expressed in UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


#: Any ratio-like quantity on the unit interval (indicators, sub-scores, shares).
UnitFloat = Annotated[
    float,
    Field(ge=0.0, le=1.0, description="Ratio on the unit interval, 0.0-1.0."),
]

#: Consumer risk on the project's 0-100 scale (higher means riskier).
RiskScore = Annotated[
    float,
    Field(
        ge=0.0,
        le=100.0,
        description="Consumer risk on the project's 0-100 scale; higher is riskier.",
    ),
]

#: Star rating, the source of the weak supervision labels.
StarRating = Annotated[int, Field(ge=1, le=5, description="Star rating from 1 to 5.")]

#: UTC instant; naive input is interpreted as UTC.
UtcDatetime = Annotated[
    datetime,
    AfterValidator(_ensure_utc),
    Field(description="UTC timestamp (naive input is interpreted as UTC)."),
]


class Page(BaseModel, Generic[T]):
    """One page of results plus the unpaginated total for the same filter.

    ``total`` counts every row matching the filter (before ``limit``/``offset``),
    so a client can render "showing ``offset + len(items)`` of ``total``"
    without a second request. Concrete pages (``ProductPage``,
    ``ReviewPage``) extend this class so their OpenAPI schema can carry its own
    example.
    """

    model_config = ConfigDict(from_attributes=True)

    total: int = Field(
        ..., ge=0, description="Rows matching the query across all pages."
    )
    limit: int = Field(..., ge=0, description="Page size that was applied.")
    offset: int = Field(..., ge=0, description="Number of rows skipped before this page.")
    items: list[T] = Field(..., description="The page itself, in a stable documented order.")


class ErrorResponse(BaseModel):
    """Error envelope returned for every non-2xx response.

    ``code`` is a stable machine token (``PRODUCT_NOT_FOUND``, ``VALIDATION_ERROR``,
    ``INTERNAL_ERROR``); the HTTP status repeats it for convenience so a client
    can branch on either. ``detail`` carries the machine-readable particulars
    (for example the offending field and its limit).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "PRODUCT_NOT_FOUND",
                "message": "Product 'B0000000000' was not found in the serving cache.",
                "detail": {"asin": "B0000000000"},
                "request_id": "req-5f0a1c2d",
            }
        }
    )

    code: str = Field(
        description="Stable error token, e.g. PRODUCT_NOT_FOUND, VALIDATION_ERROR, "
        "INTERNAL_ERROR.",
    )
    message: str = Field(description="Human-readable summary, safe to render.")
    detail: dict[str, object] | None = Field(
        default=None,
        description="Optional machine-readable particulars (unknown ASIN, "
        "failing field name and bound).",
    )
    request_id: str | None = Field(
        default=None, description="Correlation id for logs, when available."
    )


class HealthResponse(BaseModel):
    """``GET /health`` payload: service identity and reachability of the cache."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "service": "Munasib Khareed",
                "version": "0.1.0",
                "database": "reachable",
            }
        }
    )

    status: str = Field(description="'ok' when the API process is serving.")
    service: str = Field(description="Project display name from the config.")
    version: str = Field(description="Project version from the config.")
    database: str = Field(
        description="'reachable' when the SQLite serving cache answers, "
        "'unavailable' when it does not.",
    )


class SplitSizes(BaseModel):
    """Row counts for one product-level split (reviews inherit the product's split)."""

    model_config = ConfigDict(from_attributes=True)

    products: int = Field(..., ge=0, description="Products assigned to this split.")
    reviews: int = Field(..., ge=0, description="Reviews belonging to those products.")


class CategoryMetaResponse(BaseModel):
    """``GET /meta/category`` payload: what the serving cache actually contains.

    Every number is counted from the serving cache, and the date range is taken
    from the review data itself - which ends on 2018-10-04, so downstream
    "recency" features are measured against ``reference_date``, never today.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "Appliances",
                "n_products": 2066,
                "n_reviews": 425993,
                "n_related_edges": 84885,
                "n_related_edges_in_subset": 22349,
                "date_range_start": "2001-11-08T00:00:00Z",
                "date_range_end": "2018-10-04T00:00:00Z",
                "reference_date": "2018-10-04T00:00:00Z",
                "split_sizes": {
                    "train": {"products": 1444, "reviews": 303317},
                    "val": {"products": 310, "reviews": 62961},
                    "test": {"products": 312, "reviews": 59715},
                },
            }
        }
    )

    category: str = Field(description="UCSD Amazon Review Data (2018) category being served.")
    n_products: int = Field(..., ge=0, description="Products in the processed subset.")
    n_reviews: int = Field(..., ge=0, description="Reviews in the processed subset.")
    n_related_edges: int = Field(
        ..., ge=0, description="Related-product edges (also_bought/also_viewed)."
    )
    n_related_edges_in_subset: int = Field(
        ...,
        ge=0,
        description="Edges whose target is itself a processed subset product.",
    )
    date_range_start: UtcDatetime = Field(description="Earliest review timestamp in the subset.")
    date_range_end: UtcDatetime = Field(description="Latest review timestamp in the subset.")
    reference_date: UtcDatetime = Field(
        description="The date 'recent' is measured from: the latest review in the "
        "data (2018-10-04 for this dataset), never today.",
    )
    split_sizes: dict[str, SplitSizes] = Field(
        description="Products and reviews per product-level split."
    )

    @field_validator("split_sizes")
    @classmethod
    def _validate_split_names(cls, value: dict[str, SplitSizes]) -> dict[str, SplitSizes]:
        """Reject split names outside the shared split vocabulary.

        Args:
            value: Parsed ``{split: sizes}`` mapping.

        Returns:
            The unchanged mapping.

        Raises:
            ValueError: If a key is not one of ``train``/``val``/``test``.
        """
        unknown = sorted(set(value) - set(SPLIT_ORDER))
        if unknown:
            raise ValueError(f"unexpected split name(s) {unknown}; expected {list(SPLIT_ORDER)}")
        return value
