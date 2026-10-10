"""Configuration management with typed Pydantic models and YAML/environment loading."""

import os
from pathlib import Path
from typing import Any

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
    db_path: Path = Path("data/db/serving_cache.sqlite")
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


class SentimentConfig(BaseModel):
    """Transformer sentiment inference configuration."""

    model_name: str = "distilbert-base-uncased"
    max_sequence_length: int = 256
    batch_size: int = 32
    rating_mapping: RatingMappingConfig = Field(default_factory=RatingMappingConfig)


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
    env_mappings = {
        "MUNASIB_ENVIRONMENT": ("project", "environment"),
        "MUNASIB_RANDOM_SEED": ("project", "random_seed", int),
        "MUNASIB_DB_PATH": ("paths", "db_path"),
        "MUNASIB_API_HOST": ("api", "host"),
        "MUNASIB_API_PORT": ("api", "port", int),
        "MUNASIB_LOG_LEVEL": ("logging", "level"),
        "MUNASIB_DATASET_CATEGORY": ("dataset", "category"),
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
