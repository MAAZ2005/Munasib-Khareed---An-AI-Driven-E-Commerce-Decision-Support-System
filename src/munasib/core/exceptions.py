"""Domain exceptions for Munasib Khareed."""



class MunasibError(Exception):
    """Base exception for all Munasib Khareed errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ------------------------------------------------------------------------------
# Configuration Exceptions
# ------------------------------------------------------------------------------


class ConfigError(MunasibError):
    """Raised when application configuration is missing, unreadable, or invalid."""


# ------------------------------------------------------------------------------
# Data & Database Exceptions
# ------------------------------------------------------------------------------


class DataError(MunasibError):
    """Base exception for dataset ingestion, parsing, or transformation failures."""


class DataNotFoundError(DataError):
    """Raised when an expected dataset file, directory, or partition is missing."""


class SchemaValidationError(DataError):
    """Raised when dataset records fail schema validation checks."""


class ProductNotFoundError(DataError):
    """Raised when a queried product (ASIN) does not exist in dataset or database."""

    def __init__(self, asin: str, message: str | None = None) -> None:
        msg = message or f"Product with ASIN '{asin}' was not found in catalog or cache."
        super().__init__(msg, details={"asin": asin})
        self.asin = asin


class InsufficientEvidenceError(DataError):
    """Raised when a product lacks sufficient reviews or verification for reliable analysis."""

    def __init__(
        self,
        asin: str,
        review_count: int,
        min_required: int,
        message: str | None = None,
    ) -> None:
        msg = (
            message
            or f"Product '{asin}' has insufficient evidence: {review_count} review(s) found, "
            f"minimum required is {min_required}."
        )
        super().__init__(
            msg,
            details={
                "asin": asin,
                "review_count": review_count,
                "min_required": min_required,
            },
        )
        self.asin = asin
        self.review_count = review_count
        self.min_required = min_required


class DatabaseError(MunasibError):
    """Raised when SQLite database operations fail."""


# ------------------------------------------------------------------------------
# Model & Inference Exceptions
# ------------------------------------------------------------------------------


class ModelError(MunasibError):
    """Base exception for model training, weights loading, or inference errors."""


class ModelNotLoadedError(ModelError):
    """Raised when an inference request is made to a model that is not yet loaded."""


class InferenceError(ModelError):
    """Raised when model inference or NLP pipeline fails during batch execution."""


class FuzzyEvaluationError(MunasibError):
    """Raised when fuzzy risk inference or defuzzification fails."""


class DecisionError(MunasibError):
    """Raised when TOPSIS ranking or multi-criteria matrix calculation encounters an error."""


class SurrogateFidelityError(ModelError):
    """Raised when the surrogate regressor for SHAP explanation fails minimum fidelity criteria."""

    def __init__(
        self,
        r2_score: float,
        min_required: float,
        message: str | None = None,
    ) -> None:
        msg = (
            message
            or f"Surrogate risk regressor fidelity R2 ({r2_score:.4f}) is below acceptable "
            f"threshold ({min_required:.4f}). Explanation cannot be certified."
        )
        super().__init__(
            msg,
            details={"r2_score": r2_score, "min_required": min_required},
        )
        self.r2_score = r2_score
        self.min_required = min_required
