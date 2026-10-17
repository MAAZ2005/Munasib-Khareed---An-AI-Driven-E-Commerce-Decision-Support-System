"""Shared named constants used across every layer of Munasib Khareed.

Project rule: fixed vocabularies (weak-supervision label codes, split names,
verdict labels) are defined **once** here and imported everywhere - no bare
``0``/``1``/``2`` or inline ``"train"`` strings anywhere else in the codebase.
"""

from __future__ import annotations

import re

# ------------------------------------------------------------------------------
# Weak-supervision sentiment labels (locked FYP design decision #2).
# `sentiment_label_weak` in reviews is derived from star ratings:
# ratings 1-2 -> negative (0), rating 3 -> neutral (1), ratings 4-5 -> positive (2).
# These are WEAK LABELS, not true sentiment - every metric that uses them
# must say so (see `review_intel.sentiment.rating_mapping` in configs/config.yaml).
# ------------------------------------------------------------------------------
SENTIMENT_LABEL_NEGATIVE = 0
SENTIMENT_LABEL_NEUTRAL = 1
SENTIMENT_LABEL_POSITIVE = 2

#: Human-readable names for the weak label codes (reporting/UI only).
SENTIMENT_LABEL_NAMES: dict[int, str] = {
    SENTIMENT_LABEL_NEGATIVE: "negative",
    SENTIMENT_LABEL_NEUTRAL: "neutral",
    SENTIMENT_LABEL_POSITIVE: "positive",
}

#: The complete set of valid weak label codes.
SENTIMENT_LABELS: tuple[int, ...] = (
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABEL_POSITIVE,
)

# ------------------------------------------------------------------------------
# Product-level train/val/test splits (leakage-safe: reviews inherit splits;
# assignment is seeded via dataset.split in configs/config.yaml).
# ------------------------------------------------------------------------------
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"

#: Canonical split order used for reporting and iteration.
SPLIT_ORDER: tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# ------------------------------------------------------------------------------
# Verdict labels (locked FYP design decision #5). A product with very weak
# evidence can never get a clean CONSIDER.
# ------------------------------------------------------------------------------
VERDICT_CONSIDER = "CONSIDER"
VERDICT_CAUTION = "CONSIDER WITH CAUTION"
VERDICT_RISK = "CONSUMER RISK"

#: All valid verdict labels, in increasing severity order.
VERDICT_LABELS: tuple[str, ...] = (VERDICT_CONSIDER, VERDICT_CAUTION, VERDICT_RISK)

# ------------------------------------------------------------------------------
# Product identity and the local image cache (display-only; no image analysis).
# ------------------------------------------------------------------------------
#: An Amazon ASIN is exactly 10 alphanumeric characters. Both the image endpoint
#: and the image downloader validate against this before any filesystem access,
#: so a crafted ``asin`` can never escape the thumbnail-cache directory.
ASIN_PATTERN = re.compile(r"^[A-Za-z0-9]{10}$")

#: Attribution rendered wherever a cached product image is shown. Images are the
#: Amazon listing photos carried by the UCSD Amazon Review Data (2018); they are
#: served from the local cache, never hotlinked.
IMAGE_ATTRIBUTION = "Images: Amazon product listings, UCSD Amazon Review Data 2018"

#: Suffix of the locally cached thumbnails (``<asin>.jpg``). Written once by
#: ``scripts/download_images.py``, served by ``GET /products/{asin}/image`` and
#: re-scanned by ``scripts/ingest_db.py`` so ``products.image_cached`` always
#: reflects the files that actually exist in the git-ignored cache directory.
IMAGE_FILENAME_SUFFIX = ".jpg"
