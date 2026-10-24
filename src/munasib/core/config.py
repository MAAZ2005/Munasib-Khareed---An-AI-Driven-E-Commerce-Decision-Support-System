"""Configuration management with typed Pydantic models and YAML/environment loading."""

import os
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from munasib.core.exceptions import ConfigError

# Load any .env variables if present
load_dotenv()

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


# ------------------------------------------------------------------------------
# Pydantic Configuration Models
# ------------------------------------------------------------------------------


class ProjectConfig(BaseModel):
    """General project identity and runtime configuration."""

    name: str = "munasib"
    display_name: str = "Munasib Khareed"
    version: str = "0.1.0"
    environment: str = "development"
    random_seed: int = 42


class PathsConfig(BaseModel):
    """Resolved directory and cache paths."""

    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    interim_data_dir: Path = Path("data/interim")
    processed_data_dir: Path = Path("data/processed")
    db_path: Path = Path("data/db/munasib.sqlite")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    logs_dir: Path = Path("logs")


class DatasetSourceConfig(BaseModel):
    """Remote source files for one UCSD Amazon Review Data (2018) category."""

    reviews_url: str
    metadata_url: str


class AuditRubricConfig(BaseModel):
    """Thresholds for the dataset acceptance rubric printed by notebooks/01_dataset_audit.py."""

    candidate_min_reviews: int = 50
    min_price_coverage: float = 0.70
    min_candidate_products: int = 500
    min_related_links_share: float = 0.60
    fallback_category: str = "Luxury_Beauty"


class SplitConfig(BaseModel):
    """Product-level train/val/test split (leakage-safe: reviews inherit splits).

    Fractions must sum to 1.0. Stratification uses the interior bin edges below:
    a product's average rating and review count are binned, and every product in a
    stratum is shuffled with ``seed`` and allocated to splits by the fractions.
    """

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    # Interior cut points (strictly ascending). Bins: (-inf, e0], (e0, e1], ... (en, inf).
    rating_bin_edges: list[float] = Field(default_factory=lambda: [3.5, 4.5])
    review_count_bin_edges: list[float] = Field(default_factory=lambda: [75.0, 150.0, 400.0])

    @model_validator(mode="after")
    def _check_fractions_and_edges(self) -> "SplitConfig":
        """Validate that fractions sum to 1.0 and bin edges ascend strictly.

        Returns:
            The validated configuration.

        Raises:
            ValueError: When fractions are outside (0, 1], do not sum to 1.0, or
                bin edges are not strictly ascending.
        """
        fractions = (self.train_fraction, self.val_fraction, self.test_fraction)
        if any(f <= 0.0 or f > 1.0 for f in fractions):
            raise ValueError(
                "dataset.split fractions must be in (0, 1], got "
                f"train={self.train_fraction}, val={self.val_fraction}, "
                f"test={self.test_fraction}"
            )
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError(f"dataset.split fractions must sum to 1.0, got {sum(fractions)}")
        for name, edges in (
            ("rating_bin_edges", self.rating_bin_edges),
            ("review_count_bin_edges", self.review_count_bin_edges),
        ):
            if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
                raise ValueError(f"dataset.split.{name} must be strictly ascending, got {edges}")
        return self


class DatasetConfig(BaseModel):
    """Target category and sampling constraints from UCSD Amazon Review Data (2018)."""

    category: str = "Appliances"
    sample_size: int | None = None
    min_reviews_per_product: int = 5
    max_reviews_per_product: int = 500
    sources: dict[str, DatasetSourceConfig] = Field(default_factory=dict)
    audit: AuditRubricConfig = Field(default_factory=AuditRubricConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)


class RatingMappingConfig(BaseModel):
    """Weak supervision mapping boundaries for rating-to-sentiment labels."""

    negative_max: int = 2  # Ratings 1-2: negative
    neutral_val: int = 3   # Rating 3: neutral
    positive_min: int = 4  # Ratings 4-5: positive


#: Review text columns a sentiment model may read. Kept beside the config
#: validator so a typo cannot silently select a column the review schema does
#: not have (the vocabulary mirrors ``munasib.data.preprocessing.REVIEW_SCHEMA``).
SENTIMENT_TEXT_COLUMN_VOCABULARY: tuple[str, ...] = (
    "review_text_clean",
    "review_text_raw",
)

#: Review columns that carry the weak-supervision sentiment target.
SENTIMENT_LABEL_COLUMN_VOCABULARY: tuple[str, ...] = ("sentiment_label_weak",)

#: Class-weighting schemes accepted by the neural sentiment trainers. ``none``
#: uses the raw class frequencies, ``balanced`` weights each class by inverse
#: frequency, and ``sqrt_inverse`` takes the square root of those weights - a
#: middle path between raw accuracy (which ignores the ~5% neutral class) and
#: full balancing (which over-states negativity in the downstream risk stage).
ClassWeighting = Literal["none", "sqrt_inverse", "balanced"]
CLASS_WEIGHTING_CHOICES: tuple[str, ...] = get_args(ClassWeighting)

#: Recurrent cells the RNN sentiment model can use.
RnnCell = Literal["gru", "lstm"]
RNN_CELL_CHOICES: tuple[str, ...] = get_args(RnnCell)


class TfidfConfig(BaseModel):
    """TF-IDF vectoriser settings for the sentiment baseline.

    ``ngram_range`` is a closed interval of word n-gram sizes, ``min_df`` drops
    terms seen in fewer documents, ``max_features`` caps the vocabulary and
    ``sublinear_tf`` applies ``1 + log(tf)`` so a repeated word cannot dominate.
    """

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 3
    max_features: int = 200_000
    sublinear_tf: bool = True

    @model_validator(mode="after")
    def _check_tfidf_values(self) -> "TfidfConfig":
        """Validate the vectoriser settings.

        Returns:
            The validated configuration.

        Raises:
            ValueError: When the n-gram range is descending or non-positive, or
                ``min_df``/``max_features`` is not positive.
        """
        low, high = self.ngram_range
        if low < 1 or high < low:
            raise ValueError(
                "review_intel.sentiment.tfidf.ngram_range must be ascending and "
                f">= 1, got {self.ngram_range}"
            )
        if self.min_df < 1:
            raise ValueError(
                f"review_intel.sentiment.tfidf.min_df must be >= 1, got {self.min_df}"
            )
        if self.max_features < 1:
            raise ValueError(
                "review_intel.sentiment.tfidf.max_features must be >= 1, got "
                f"{self.max_features}"
            )
        return self


class LogRegConfig(BaseModel):
    """Logistic-regression settings for the sentiment baseline.

    ``C`` is the inverse regularisation strength (smaller = stronger) and
    ``max_iter`` caps the L-BFGS iterations; a run that fails to converge says so
    in the evaluation report rather than silently reporting partial weights.
    """

    C: float = 1.0
    max_iter: int = 1000

    @model_validator(mode="after")
    def _check_logreg_values(self) -> "LogRegConfig":
        """Validate the classifier settings.

        Returns:
            The validated configuration.

        Raises:
            ValueError: When ``C`` is not positive or ``max_iter`` is < 1.
        """
        if self.C <= 0.0:
            raise ValueError(f"review_intel.sentiment.logreg.C must be > 0, got {self.C}")
        if self.max_iter < 1:
            raise ValueError(
                f"review_intel.sentiment.logreg.max_iter must be >= 1, got {self.max_iter}"
            )
        return self


class TransformerConfig(BaseModel):
    """DistilBERT fine-tuning settings for the three-class sentiment classifier.

    ``learning_rate`` is the peak AdamW rate (linear warmup then decay),
    ``warmup_ratio`` is the share of total steps spent warming up and
    ``early_stopping_patience`` counts epochs without a macro-F1 improvement
    before training stops. ``output_dir`` lives under the git-ignored ``models/``
    tree so checkpoints never enter the repository.
    """

    model_name: str = "distilbert-base-uncased"
    learning_rate: float = 3e-5
    batch_size: int = 32
    epochs: int = 3
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 1
    output_dir: Path = Path("models/sentiment_distilbert")

    @model_validator(mode="after")
    def _check_transformer_values(self) -> "TransformerConfig":
        """Validate the fine-tuning schedule and the artifact path.

        Returns:
            The validated configuration.

        Raises:
            ValueError: When the model name or output directory is empty, the
                learning rate is outside ``(0, 1)``, a count is below one, the
                warmup ratio is outside ``[0, 1)``, or a regularisation value is
                negative.
        """
        if not self.model_name.strip():
            raise ValueError("review_intel.sentiment.transformer.model_name must not be empty")
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError(
                "review_intel.sentiment.transformer.learning_rate must be in (0, 1), "
                f"got {self.learning_rate}"
            )
        if self.batch_size < 1:
            raise ValueError(
                "review_intel.sentiment.transformer.batch_size must be >= 1, "
                f"got {self.batch_size}"
            )
        if self.epochs < 1:
            raise ValueError(
                f"review_intel.sentiment.transformer.epochs must be >= 1, got {self.epochs}"
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(
                "review_intel.sentiment.transformer.warmup_ratio must be in [0, 1), "
                f"got {self.warmup_ratio}"
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                "review_intel.sentiment.transformer.weight_decay must be >= 0, "
                f"got {self.weight_decay}"
            )
        if self.max_grad_norm <= 0.0:
            raise ValueError(
                "review_intel.sentiment.transformer.max_grad_norm must be > 0, "
                f"got {self.max_grad_norm}"
            )
        if self.early_stopping_patience < 0:
            raise ValueError(
                "review_intel.sentiment.transformer.early_stopping_patience must be >= 0, "
                f"got {self.early_stopping_patience}"
            )
        if not str(self.output_dir).strip():
            raise ValueError("review_intel.sentiment.transformer.output_dir must not be empty")
        return self


class RnnConfig(BaseModel):
    """BiGRU/BiLSTM sentiment model settings.

    The vocabulary is fitted on the training sample only (terms seen at least
    ``min_freq`` times, capped at ``vocab_size``), so no evaluation-split token
    can reach the embedding table. ``learning_rate`` is separate from the
    transformer's because a model trained from scratch needs a much larger step
    than a fine-tune; ``output_dir`` keeps its checkpoints under the git-ignored
    ``models/`` tree.
    """

    embedding_dim: int = 128
    hidden_size: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.3
    vocab_size: int = 50_000
    min_freq: int = 2
    cell: RnnCell = "gru"
    learning_rate: float = 1e-3
    output_dir: Path = Path("models/sentiment_rnn")

    @model_validator(mode="after")
    def _check_rnn_values(self) -> "RnnConfig":
        """Validate the recurrent model dimensions and vocabulary caps.

        Returns:
            The validated configuration.

        Raises:
            ValueError: When a dimension is below one, the vocabulary cap is
                below two (it must hold padding and unknown tokens), ``min_freq``
                is below one, dropout is outside ``[0, 1)``, the learning rate is
                outside ``(0, 1)`` or the output directory is empty.
        """
        for name, value in (
            ("embedding_dim", self.embedding_dim),
            ("hidden_size", self.hidden_size),
            ("num_layers", self.num_layers),
            ("min_freq", self.min_freq),
        ):
            if value < 1:
                raise ValueError(f"review_intel.sentiment.rnn.{name} must be >= 1, got {value}")
        if self.vocab_size < 2:
            raise ValueError(
                "review_intel.sentiment.rnn.vocab_size must be >= 2 (padding + unknown), "
                f"got {self.vocab_size}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"review_intel.sentiment.rnn.dropout must be in [0, 1), got {self.dropout}"
            )
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError(
                "review_intel.sentiment.rnn.learning_rate must be in (0, 1), "
                f"got {self.learning_rate}"
            )
        if not str(self.output_dir).strip():
            raise ValueError("review_intel.sentiment.rnn.output_dir must not be empty")
        return self


class SentimentConfig(BaseModel):
    """Sentiment model settings: weak-supervision sample, models and reports.

    The first group configures the **training sample** every model learns from:
    which review column holds the text and the weak label, how many reviews per
    product may enter the sample, the overall row budget, the shortest usable
    review and the seed. The second group configures the TF-IDF + logistic
    regression baseline, the DistilBERT fine-tuning schedule
    (``transformer``) and the recurrent model (``rnn``). ``max_length_tokens`` is
    the single token budget every neural model truncates and pads to.
    """

    rating_mapping: RatingMappingConfig = Field(default_factory=RatingMappingConfig)

    #: Review column holding the text the model reads (cleaned text by default).
    text_column: str = "review_text_clean"
    #: Review column holding the weak-supervision label (0/1/2).
    label_column: str = "sentiment_label_weak"
    #: Hard cap on how many reviews one product may contribute to the sample, so
    #: a single 6,510-review product cannot dominate the training set.
    max_reviews_per_product: int = 50
    #: Overall row budget for the training sample (positive reviews are
    #: subsampled to reach it; negatives and neutrals are never dropped for it).
    max_train_rows: int = 100_000
    #: Shortest review text, in characters, that may enter training or evaluation.
    min_chars: int = 3
    #: Seed for the deterministic review selection (never the global RNG).
    seed: int = 42
    #: Token budget every neural model truncates and pads to.
    max_length_tokens: int = 128
    #: How the neural trainers weight the three classes in the loss.
    class_weighting: ClassWeighting = "sqrt_inverse"
    tfidf: TfidfConfig = Field(default_factory=TfidfConfig)
    logreg: LogRegConfig = Field(default_factory=LogRegConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    rnn: RnnConfig = Field(default_factory=RnnConfig)
    #: Where the fitted baseline artifact is written (git-ignored ``models/``).
    model_path: Path = Path("models/sentiment_tfidf_logreg.joblib")
    #: Where the JSON/Markdown/CSV evaluation reports are written.
    report_dir: Path = Path("reports/evaluation")

    @model_validator(mode="after")
    def _check_sentiment_values(self) -> "SentimentConfig":
        """Validate the sample, vocabulary and path settings.

        Returns:
            The validated configuration.

        Raises:
            ValueError: When a column name is outside the review vocabulary, a
                count is not positive, or an artifact path is empty.
        """
        if self.text_column not in SENTIMENT_TEXT_COLUMN_VOCABULARY:
            raise ValueError(
                "review_intel.sentiment.text_column must be one of "
                f"{list(SENTIMENT_TEXT_COLUMN_VOCABULARY)}, got {self.text_column!r}"
            )
        if self.label_column not in SENTIMENT_LABEL_COLUMN_VOCABULARY:
            raise ValueError(
                "review_intel.sentiment.label_column must be one of "
                f"{list(SENTIMENT_LABEL_COLUMN_VOCABULARY)}, got {self.label_column!r}"
            )
        for name, value in (
            ("max_reviews_per_product", self.max_reviews_per_product),
            ("max_train_rows", self.max_train_rows),
            ("min_chars", self.min_chars),
            ("max_length_tokens", self.max_length_tokens),
        ):
            if value < 1:
                raise ValueError(f"review_intel.sentiment.{name} must be >= 1, got {value}")
        for name, path in (
            ("model_path", self.model_path),
            ("report_dir", self.report_dir),
        ):
            if not str(path).strip():
                raise ValueError(f"review_intel.sentiment.{name} must not be empty")
        return self


class AspectsConfig(BaseModel):
    """Aspect-based attribute extraction configuration."""

    spacy_model: str = "en_core_web_sm"
    min_aspect_frequency: int = 3
    top_k_aspects: int = 10


class EvidenceWeightsConfig(BaseModel):
    """Weights for composite evidence-strength scoring."""

    verified_purchase: float = 0.50
    helpfulness_ratio: float = 0.30
    review_depth: float = 0.20


class EvidenceConfig(BaseModel):
    """Review credibility and evidence-strength configuration."""

    min_length_chars: int = 20
    length_scale_chars: int = 200
    weights: EvidenceWeightsConfig = Field(default_factory=EvidenceWeightsConfig)


class ReviewIntelConfig(BaseModel):
    """Review intelligence module settings."""

    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    aspects: AspectsConfig = Field(default_factory=AspectsConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)


class RiskConfig(BaseModel):
    """Fuzzy logic risk inference configuration."""

    fuzzy_rules_file: Path = Path("configs/fuzzy_rules.yaml")
    scale_min: float = 0.0
    scale_max: float = 100.0
    default_defuzzification_method: str = "centroid"


class DecisionConfig(BaseModel):
    """TOPSIS multi-criteria decision ranking configuration."""

    criteria_weights_file: Path = Path("configs/criteria_weights.yaml")
    peer_sample_size: int = 20
    normalize_method: str = "vector"


class XAIConfig(BaseModel):
    """Surrogate regressor and SHAP explainability configuration."""

    surrogate_model_type: str = "random_forest"
    n_estimators: int = 100
    max_depth: int = 6
    min_fidelity_r2: float = 0.80
    background_sample_size: int = 50


class VerdictConfig(BaseModel):
    """Rule-enforced verdict synthesis configuration."""

    thresholds_file: Path = Path("configs/verdict_thresholds.yaml")


class AnalysisConfig(BaseModel):
    """Which analysis implementation the API layer should serve.

    ``stub`` is the deterministic placeholder in
    :mod:`munasib.pipeline.service` (real evidence numbers, clearly labelled
    placeholder risk scores) and is the default. ``real`` is the fuzzy + TOPSIS
    engine; selecting it before that engine exists raises a documented error from
    the API dependency instead of silently falling back.
    """

    service: Literal["stub", "real"] = "stub"


class APIConfig(BaseModel):
    """FastAPI REST service settings."""

    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True
    workers: int = 1
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501", "http://127.0.0.1:8501"])


class LoggingConfig(BaseModel):
    """Application logging settings."""

    level: str = "INFO"
    file_path: Path = Path("logs/munasib.log")
    max_bytes: int = 10485760
    backup_count: int = 5
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"


class ImagesConfig(BaseModel):
    """Local product-thumbnail cache (display only; the dashboard never hotlinks).

    ``scripts/download_images.py`` downloads a small, EXIF-stripped JPEG per ASIN
    into ``directory`` and the API serves it from disk, so the dashboard keeps
    working offline. ``cache_max_age_seconds`` is the ``Cache-Control`` lifetime
    the image endpoint advertises for an immutable cached thumbnail.
    """

    directory: Path = Path("data/images")
    thumbnail_max_side: int = 320
    download_delay_seconds: float = 0.5
    timeout_seconds: float = 15.0
    user_agent: str = (
        "MunasibKhareed/0.1 (academic FYP; local product-thumbnail cache; "
        "contact: munasib-khareed-fyp@example.invalid)"
    )
    cache_max_age_seconds: int = 86400

    @model_validator(mode="after")
    def _check_positive_values(self) -> "ImagesConfig":
        """Validate the numeric thumbnail settings.

        Returns:
            The validated configuration.

        Raises:
            ValueError: When a size/timeout is not positive, the delay is
                negative, or the user agent is empty.
        """
        if self.thumbnail_max_side < 1:
            raise ValueError(
                f"images.thumbnail_max_side must be >= 1, got {self.thumbnail_max_side}"
            )
        if self.download_delay_seconds < 0.0:
            raise ValueError(
                f"images.download_delay_seconds must be >= 0, got {self.download_delay_seconds}"
            )
        if self.timeout_seconds <= 0.0:
            raise ValueError(f"images.timeout_seconds must be > 0, got {self.timeout_seconds}")
        if self.cache_max_age_seconds < 0:
            raise ValueError(
                f"images.cache_max_age_seconds must be >= 0, got {self.cache_max_age_seconds}"
            )
        if not self.user_agent.strip():
            raise ValueError("images.user_agent must not be empty (polite downloads)")
        return self


#: Metadata fields whose share of presence is reported as
#: ``metadata_completeness`` by ``munasib.product_intel``. Kept here so the
#: config validator and the profiler agree on the vocabulary.
METADATA_FIELD_VOCABULARY: tuple[str, ...] = (
    "title",
    "brand",
    "price",
    "rank",
    "description",
    "features",
)


class ProductIntelConfig(BaseModel):
    """Product-intelligence thresholds (see ``docs/product_intelligence.md``).

    These turn raw product metadata into 0-1 indicators: the completeness
    targets (a description is "complete" at ``target_description_words`` words,
    a feature list at ``target_feature_count`` bullets), the minimum peer counts
    that keep a brand, rank or category price percentile meaningful, and the
    exact field list whose share of presence is reported as
    ``metadata_completeness``. Nothing here imputes a value - a missing field
    scores 0 for completeness and its ``*_missing`` flag is set.
    """

    target_description_words: int = 100
    target_feature_count: int = 5
    #: Other priced products a brand needs before ``price_percentile_brand`` is
    #: reported (small brands would otherwise give a meaningless percentile).
    min_brand_peers: int = 5
    #: Other ranked products a rank category needs before ``rank_percentile`` is
    #: reported.
    min_rank_peers: int = 5
    #: Priced products a category-breadcrumb prefix must hold before it is used
    #: as the price peer group of a product (see
    #: ``docs/product_intelligence.md``); below it the peer group falls back one
    #: level towards the whole candidate set.
    min_price_peers: int = 10
    metadata_fields: list[str] = Field(
        default_factory=lambda: list(METADATA_FIELD_VOCABULARY)
    )

    @model_validator(mode="after")
    def _check_product_intel_values(self) -> "ProductIntelConfig":
        """Validate the product-intelligence thresholds.

        Returns:
            The validated configuration.

        Raises:
            ValueError: When a target or minimum is not positive, or the
                metadata field list is empty or names an unknown field.
        """
        for name, value in (
            ("target_description_words", self.target_description_words),
            ("target_feature_count", self.target_feature_count),
            ("min_brand_peers", self.min_brand_peers),
            ("min_rank_peers", self.min_rank_peers),
            ("min_price_peers", self.min_price_peers),
        ):
            if value < 1:
                raise ValueError(f"product_intel.{name} must be >= 1, got {value}")
        if not self.metadata_fields:
            raise ValueError("product_intel.metadata_fields must not be empty")
        unknown = sorted(set(self.metadata_fields) - set(METADATA_FIELD_VOCABULARY))
        if unknown:
            raise ValueError(
                f"product_intel.metadata_fields contains unknown fields {unknown}; "
                f"expected a subset of {list(METADATA_FIELD_VOCABULARY)}"
            )
        if len(set(self.metadata_fields)) != len(self.metadata_fields):
            raise ValueError("product_intel.metadata_fields must not repeat a field")
        return self


class WellReviewedConfig(BaseModel):
    """Thresholds for "well-reviewed" demo products."""

    mean_rating_min: float = 4.5
    min_reviews: int = 100


class PolarizedConfig(BaseModel):
    """Thresholds for "polarized" demo products (both camps materially present)."""

    five_star_share_min: float = 0.25
    low_star_share_min: float = 0.25


class LowRatedConfig(BaseModel):
    """Thresholds for "low-rated" demo products."""

    mean_rating_max: float = 3.0


class DemoProductsConfig(BaseModel):
    """Selection thresholds for ``scripts/suggest_demo_products.py``.

    The suggester prints candidate ASINs for a live demo: well-reviewed,
    polarized and low-rated products that also have a cached image candidate and
    enough usable in-subset alternatives to compare against.
    """

    per_group: int = 10
    min_usable_alternatives: int = 3
    well_reviewed: WellReviewedConfig = Field(default_factory=WellReviewedConfig)
    polarized: PolarizedConfig = Field(default_factory=PolarizedConfig)
    low_rated: LowRatedConfig = Field(default_factory=LowRatedConfig)

    @model_validator(mode="after")
    def _check_selection_values(self) -> "DemoProductsConfig":
        """Validate the demo-suggestion thresholds.

        Returns:
            The validated configuration.

        Raises:
            ValueError: When a count is not positive or a rating/share bound is
                outside its 0-5 / (0, 1] range.
        """
        if self.per_group < 1:
            raise ValueError(f"demo_products.per_group must be >= 1, got {self.per_group}")
        if self.min_usable_alternatives < 1:
            raise ValueError(
                "demo_products.min_usable_alternatives must be >= 1, got "
                f"{self.min_usable_alternatives}"
            )
        if self.well_reviewed.min_reviews < 1:
            raise ValueError(
                "demo_products.well_reviewed.min_reviews must be >= 1, got "
                f"{self.well_reviewed.min_reviews}"
            )
        for name, value in (
            ("well_reviewed.mean_rating_min", self.well_reviewed.mean_rating_min),
            ("low_rated.mean_rating_max", self.low_rated.mean_rating_max),
        ):
            if not 0.0 <= value <= 5.0:
                raise ValueError(f"demo_products.{name} must be in [0, 5], got {value}")
        for name, value in (
            ("polarized.five_star_share_min", self.polarized.five_star_share_min),
            ("polarized.low_star_share_min", self.polarized.low_star_share_min),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"demo_products.{name} must be in (0, 1], got {value}")
        return self


class Settings(BaseModel):
    """Master application settings model."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    review_intel: ReviewIntelConfig = Field(default_factory=ReviewIntelConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    xai: XAIConfig = Field(default_factory=XAIConfig)
    verdict: VerdictConfig = Field(default_factory=VerdictConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    images: ImagesConfig = Field(default_factory=ImagesConfig)
    demo_products: DemoProductsConfig = Field(default_factory=DemoProductsConfig)
    product_intel: ProductIntelConfig = Field(default_factory=ProductIntelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ------------------------------------------------------------------------------
# Loader Functions
# ------------------------------------------------------------------------------


def _apply_env_overrides(raw_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides (prefixed with MUNASIB_) onto config dict.

    Args:
        raw_dict: Dictionary loaded from YAML file.

    Returns:
        Updated dictionary with environment variable values applied.
    """
    # Mixed tuple shapes (2 items = plain override, 3 items = override + cast
    # function): mypy joins them to `tuple[str | type[int], ...]`, which it then
    # refuses to index into `dict[str, Any]`. Widening the value type to
    # `tuple[Any, ...]` documents the heterogeneous table without changing it.
    env_mappings: dict[str, tuple[Any, ...]] = {
        "MUNASIB_ENVIRONMENT": ("project", "environment"),
        "MUNASIB_RANDOM_SEED": ("project", "random_seed", int),
        "MUNASIB_DB_PATH": ("paths", "db_path"),
        "MUNASIB_API_HOST": ("api", "host"),
        "MUNASIB_API_PORT": ("api", "port", int),
        "MUNASIB_LOG_LEVEL": ("logging", "level"),
        "MUNASIB_DATASET_CATEGORY": ("dataset", "category"),
        "MUNASIB_ANALYSIS_SERVICE": ("analysis", "service"),
    }

    for env_var, mapping in env_mappings.items():
        val = os.getenv(env_var)
        if val is not None:
            section = mapping[0]
            field = mapping[1]
            cast_type = mapping[2] if len(mapping) > 2 else str
            if section not in raw_dict:
                raw_dict[section] = {}
            raw_dict[section][field] = cast_type(val)

    return raw_dict


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load, merge, and validate settings from YAML config file and environment variables.

    Args:
        config_path: Path to configuration YAML file. Defaults to 'configs/config.yaml'
            or environment variable MUNASIB_CONFIG_PATH if set.

    Returns:
        Validated Settings instance.

    Raises:
        ConfigError: If config file is missing or invalid.
    """
    selected_path = (
        Path(config_path)
        if config_path
        else Path(os.getenv("MUNASIB_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    )

    if not selected_path.exists():
        raise ConfigError(f"Configuration file not found at: {selected_path}")

    try:
        with open(selected_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML configuration at {selected_path}: {e}") from e

    merged_data = _apply_env_overrides(raw_data)

    try:
        return Settings.model_validate(merged_data)
    except Exception as e:
        raise ConfigError(f"Validation error loading configuration: {e}") from e


_settings_instance: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Return the cached singleton Settings instance.

    Args:
        reload: Force reloading of configuration from disk and environment.

    Returns:
        Settings instance.
    """
    global _settings_instance
    if _settings_instance is None or reload:
        _settings_instance = load_settings()
    return _settings_instance
