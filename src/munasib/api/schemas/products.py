"""Product contract: :class:`ProductSummary` for listings, :class:`ProductDetail` for one product.

Both models map straight onto the ``products`` table of the SQLite serving cache
(``from_attributes=True``), so the API never re-derives values that already
exist. Fields that the source data can genuinely lack stay ``None`` and carry a
companion ``*_missing`` flag - the API never invents a price, rank or brand
value (locked project rule). Text is returned exactly as stored after
preprocessing, including HTML entities such as ``&ndash;`` that appear in the
UCSD Amazon source.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .common import Page, UtcDatetime

PRODUCT_SUMMARY_EXAMPLE: dict[str, Any] = {
    "asin": "B00009V3UF",
    "title": "Range Kleen 8234 Gas Range Knobs &ndash; 4 White Gas Range Safety Knobs "
    "Gas Stove Accessories",
    "brand": "Range Kleen",
    "price": 12.66,
    "n_reviews": 52,
    "mean_rating": 2.634615,
    "split": "train",
}


class ProductSummary(BaseModel):
    """Compact product identity used in search results and comparisons."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"example": PRODUCT_SUMMARY_EXAMPLE},
    )

    asin: str = Field(description="Amazon Standard Identification Number (stable key).")
    title: str | None = Field(
        default=None, description="Product title as stored; None only when the source lacked one."
    )
    brand: str | None = Field(
        default=None,
        description="Brand as stored; None when no brand could be parsed "
        "(see brand_missing on the detail view).",
    )
    price: float | None = Field(
        default=None,
        description="Price in USD when parsable; None when missing or unparseable "
        "(never imputed).",
    )
    n_reviews: int = Field(
        ..., ge=0, description="Reviews stored for this product in the processed subset."
    )
    mean_rating: float | None = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Average star rating (1-5); None when no rated review exists.",
    )
    split: str = Field(
        description="Product-level split (train/val/test) assigned by the build; "
        "reviews inherit it."
    )


class ProductDetail(ProductSummary):
    """Full product profile for one ASIN (every column of the ``products`` table)."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "asin": "B00009V3UF",
                "title": "Range Kleen 8234 Gas Range Knobs &ndash; 4 White Gas Range Safety "
                "Knobs Gas Stove Accessories",
                "brand": "Range Kleen",
                "price": 12.66,
                "n_reviews": 52,
                "mean_rating": 2.634615,
                "split": "train",
                "brand_missing": False,
                "price_status": "single",
                "price_missing": False,
                "price_max": None,
                "rank": ">#4,165 in Appliances (See top 100)",
                "rank_value": 4165,
                "rank_category": "Appliances",
                "rank_missing": False,
                "description": "Want a fresh start with your stove burner knobs? Replace them "
                "with the Range Kleen 8234 Pack of 4 white gas range replacement knobs! Use "
                "these gas stove accessoriesto replace broken gas burner knobsthat have become "
                "a nuisance and a safety hazard to you and your loved ones. This pack of 4 gas "
                "range safety knobs includes four gas range knobs, eight adapters, and four "
                "overlays. The set of gas range knobs replacements are neutral white in "
                "color,making them compatible with any kitchen style or motif. No need to do "
                "the heavy lifting when it comes to replacing an appliance; just give your "
                "stove a facelift with new burner knobs! NOTE: Some gas range knobs "
                "replacements may require a longer shaft for replacement. Item 82001 Knob "
                "Shaft Extender may be required to replace thestove burner knobs on your gas "
                "stove.",
                "features": [
                    "SAFE GAS RANGE REPLACEMENT KNOBS: Gas range knobs give you piece of mind; "
                    "stove burner knobs protect and help gauge stove settings.",
                    "PRISTINE WHITE BURNER KNOB: Range Kleen 8234 gas burner knobs are "
                    "immaculate white color; providing maximum design compatibility.",
                    "COMPLETE BURNER KNOBS SET: 4 knobs, 8 adapters and 4 overlays included in "
                    "this burner knobs for gas stove set; update all of your gas burner knobs "
                    "with Range Kleen.",
                    "REPLACEMENT BURNER KNOB FOR GAS STOVE: No need to buy a new stove, just "
                    "freshen up the burner knob covers for gas range stoves for a fresh look.",
                    "EASY-TURN BURNER KNOB: Gas range knob replacements reset your stoves look; "
                    "these brand-new burner gas stove accessories restore your stove to that "
                    "factory setting feel.",
                ],
                "first_review_time": "2009-07-20T00:00:00Z",
                "last_review_time": "2018-02-03T00:00:00Z",
            }
        },
    )

    brand_missing: bool = Field(
        description="True when the source had no parsable brand (brand is None)."
    )
    price_status: str | None = Field(
        default=None,
        description="How the price was obtained: 'single', 'range' (price_max holds the "
        "upper end), 'missing' or 'unparseable'.",
    )
    price_missing: bool = Field(description="True when price is None for any reason.")
    price_max: float | None = Field(
        default=None,
        description="Upper end of a price range when price_status == 'range'; otherwise None.",
    )
    rank: str | None = Field(
        default=None, description="Raw best-seller-rank string exactly as published."
    )
    rank_value: int | None = Field(
        default=None, ge=1, description="Best-seller rank as an integer (smaller is better)."
    )
    rank_category: str | None = Field(
        default=None, description="Category the best-seller rank was measured in."
    )
    rank_missing: bool = Field(description="True when the source published no rank.")
    description: str | None = Field(
        default=None, description="Cleaned product description as stored; None when absent."
    )
    features: list[str] | None = Field(
        default=None, description="Cleaned feature bullets; None when absent."
    )
    first_review_time: UtcDatetime | None = Field(
        default=None, description="UTC timestamp of this product's oldest stored review."
    )
    last_review_time: UtcDatetime | None = Field(
        default=None, description="UTC timestamp of this product's newest stored review."
    )


class ProductPage(Page[ProductSummary]):
    """Paginated product search results (ordered by ASIN)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 2,
                "limit": 20,
                "offset": 0,
                "items": [
                    {
                        "asin": "B00QO4ZX30",
                        "title": "FilterBuy Replacement Humidifier Filter Compatible with "
                        "Holmes HWF62. Replacement for part # HWF62, HWF62D, HWF-62 &amp; "
                        "Hamilton Beach Part # 05910",
                        "brand": "FilterBuy",
                        "price": 10.99,
                        "n_reviews": 152,
                        "mean_rating": 4.473684,
                        "split": "train",
                    },
                    {
                        "asin": "B01FQR5RZE",
                        "title": "Hamilton Beach PIM-1-1A Portable Ice Maker, Black with "
                        "Stainless Steel",
                        "brand": "Hamilton Beach",
                        "price": 18.29,
                        "n_reviews": 114,
                        "mean_rating": 3.605263,
                        "split": "test",
                    },
                ],
            }
        }
    )


__all__ = ["ProductDetail", "ProductPage", "ProductSummary"]
