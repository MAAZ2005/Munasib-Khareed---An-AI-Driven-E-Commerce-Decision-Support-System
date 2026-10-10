"""SQLAlchemy 2.0 ORM models for the SQLite serving cache.

The database is a **cache of the processed parquet subset**, not a product
database (locked design decision #6): ``scripts/ingest_db.py`` fills
``products``/``reviews``/``related_products`` from ``data/processed/*.parquet``
and the pipeline fills ``review_features``/``product_features``/
``analysis_results`` later. Column names mirror the parquet columns, with three
documented renames: ``description_clean`` -> ``description`` (the cleaned text
*is* the description of the cache), ``features_clean`` -> ``features`` (stored
as a JSON array), and the review date columns ``review_datetime``/
``review_year``/``review_month`` -> ``review_time``/``year``/``month``.

Data-integrity rules encoded in the schema (verified from the parquet files,
never assumed):

* ``verified`` is a **nullable** boolean - NULL means "unknown", which is not
  the same as an explicit ``False``.
* ``sentiment_label_weak`` uses the shared codes from
  :mod:`munasib.core.constants` (0 negative / 1 neutral / 2 positive - weak
  supervision, not true sentiment) and stays nullable for unlabelled rows.
* Price, rank and brand stay NULL when missing/unparseable; the ``*_missing``
  boolean flags record why. Nothing is imputed.
* All datetime columns are **UTC** (review dates run 2001-11-08 .. 2018-10-04);
  SQLite stores them without an offset, so every value in this database is UTC
  by contract. Nothing in this layer ever uses today's date.
* ``related_products.related_asin`` has **no** foreign key: an edge may point
  to a product outside the processed subset (the ``in_subset`` flag records it).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: SHA1-prefix length of ``review_id`` produced by the preprocessing pipeline.
REVIEW_ID_LENGTH = 16

#: Valid relation types of the related-products edge table.
RELATION_ALSO_BOUGHT = "also_bought"
RELATION_ALSO_VIEWED = "also_viewed"
RELATION_TYPES = (RELATION_ALSO_BOUGHT, RELATION_ALSO_VIEWED)


class Base(DeclarativeBase):
    """Declarative base carrying the shared ``metadata`` for all tables."""


class Product(Base):
    """One processed candidate product (cache row mirrors ``products.parquet``)."""

    __tablename__ = "products"

    asin: Mapped[str] = mapped_column(String, primary_key=True)
    #: Product-level split; reviews inherit it (leakage-safe assignment).
    split: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    title_clean: Mapped[str | None] = mapped_column(Text)
    brand: Mapped[str | None] = mapped_column(Text)
    brand_clean: Mapped[str | None] = mapped_column(Text)
    brand_missing: Mapped[bool] = mapped_column(Boolean, nullable=False)
    #: NULL when the raw price is absent or unparseable (see ``price_status``).
    price: Mapped[float | None] = mapped_column(Float)
    price_max: Mapped[float | None] = mapped_column(Float)
    price_status: Mapped[str | None] = mapped_column(String)
    price_missing: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rank: Mapped[str | None] = mapped_column(Text)
    rank_value: Mapped[int | None] = mapped_column(Integer)
    rank_category: Mapped[str | None] = mapped_column(String)
    rank_missing: Mapped[bool] = mapped_column(Boolean, nullable=False)
    #: Cleaned description (parquet ``description_clean``).
    description: Mapped[str | None] = mapped_column(Text)
    #: Cleaned feature bullets (parquet ``features_clean``) as a JSON array.
    features: Mapped[list | None] = mapped_column(JSON)
    n_reviews: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_rating: Mapped[float | None] = mapped_column(Float)
    first_review_time: Mapped[datetime | None] = mapped_column(DateTime)
    last_review_time: Mapped[datetime | None] = mapped_column(DateTime)

    __table_args__ = (
        Index("ix_products_brand", "brand"),
        Index("ix_products_title", "title"),
    )


class Review(Base):
    """One processed review (cache row mirrors ``reviews.parquet``)."""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    #: Deterministic sha1-prefix id from the preprocessing pipeline; the stable
    #: business key (target of ``review_features.review_id``).
    review_id: Mapped[str] = mapped_column(String(REVIEW_ID_LENGTH), nullable=False, unique=True)
    asin: Mapped[str] = mapped_column(ForeignKey("products.asin"), nullable=False)
    split: Mapped[str] = mapped_column(String, nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String, nullable=False)
    #: Star rating 1.0-5.0 (weak-label source); NULL only if the source lacked it.
    rating: Mapped[float | None] = mapped_column(Float)
    #: Original text kept for display; ``*_clean`` feeds the models.
    review_text_raw: Mapped[str | None] = mapped_column(Text)
    review_text_clean: Mapped[str | None] = mapped_column(Text)
    summary_raw: Mapped[str | None] = mapped_column(Text)
    summary_clean: Mapped[str | None] = mapped_column(Text)
    #: False for rows kept without usable text (never dropped).
    has_text: Mapped[bool] = mapped_column(Boolean, nullable=False)
    text_length: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Missing vote means 0 recorded helpful votes (documented zero-fill).
    helpful_votes: Mapped[int] = mapped_column(Integer, nullable=False)
    #: NULL = unknown (field absent in the source), which is NOT False.
    verified: Mapped[bool | None] = mapped_column(Boolean)
    #: UTC review time (source data spans 2001-2018; never today's date).
    review_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    year: Mapped[int | None] = mapped_column(Integer)
    month: Mapped[int | None] = mapped_column(Integer)
    #: Weak-supervision label code from :mod:`munasib.core.constants`.
    sentiment_label_weak: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        Index("ix_reviews_asin", "asin"),
        Index("ix_reviews_asin_review_time", "asin", "review_time"),
        Index("ix_reviews_split", "split"),
    )


class RelatedProduct(Base):
    """Related-products edge (mirrors ``related_products.parquet``).

    The composite primary key doubles as the required unique constraint on
    ``(asin, related_asin, relation_type)``.
    """

    __tablename__ = "related_products"

    asin: Mapped[str] = mapped_column(ForeignKey("products.asin"), primary_key=True)
    #: May point outside the processed subset - deliberately **not** a foreign key.
    related_asin: Mapped[str] = mapped_column(String, primary_key=True)
    relation_type: Mapped[str] = mapped_column(String, primary_key=True)
    #: True when ``related_asin`` is itself a processed candidate product.
    in_subset: Mapped[bool] = mapped_column(Boolean, nullable=False)

    __table_args__ = (
        CheckConstraint(
            f"relation_type IN ('{RELATION_ALSO_BOUGHT}', '{RELATION_ALSO_VIEWED}')",
            name="ck_related_products_relation_type",
        ),
        Index("ix_related_products_asin", "asin"),
        Index("ix_related_products_asin_in_subset", "asin", "in_subset"),
    )


class ReviewFeature(Base):
    """Model outputs per review (filled by later prompts; empty at ingest).

    ``predicted_label`` uses the *model's* own label space - the weak labels in
    ``reviews.sentiment_label_weak`` are the training targets, not predictions.
    """

    __tablename__ = "review_features"

    review_id: Mapped[str] = mapped_column(
        ForeignKey("reviews.review_id"), primary_key=True
    )
    predicted_label: Mapped[int | None] = mapped_column(Integer)
    prob_negative: Mapped[float | None] = mapped_column(Float)
    prob_neutral: Mapped[float | None] = mapped_column(Float)
    prob_positive: Mapped[float | None] = mapped_column(Float)
    aspects_json: Mapped[list | None] = mapped_column(JSON)


class ProductFeature(Base):
    """Composite per-product indicators (filled by later prompts; empty at ingest)."""

    __tablename__ = "product_features"

    asin: Mapped[str] = mapped_column(ForeignKey("products.asin"), primary_key=True)
    indicators: Mapped[dict | None] = mapped_column(JSON)
    #: When the indicators were computed (pipeline-set; never read from data).
    computed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AnalysisResult(Base):
    """Persisted pipeline output per (product, pipeline version).

    The composite primary key makes re-runs of the same pipeline version
    idempotent via upsert instead of accumulating duplicate rows.
    """

    __tablename__ = "analysis_results"

    asin: Mapped[str] = mapped_column(ForeignKey("products.asin"), primary_key=True)
    pipeline_version: Mapped[str] = mapped_column(String, primary_key=True)
    #: Fuzzy+TOPSIS consumer risk on the configured scale (NULL until computed).
    risk_score: Mapped[float | None] = mapped_column(Float)
    evidence_strength: Mapped[float | None] = mapped_column(Float)
    #: Verdict label from :mod:`munasib.core.constants` (CONSIDER / ... / CONSUMER RISK).
    verdict: Mapped[str | None] = mapped_column(String)
    payload: Mapped[dict | None] = mapped_column(JSON)
    #: Last time this row was written (upsert refreshes it); pipeline-set UTC.
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
