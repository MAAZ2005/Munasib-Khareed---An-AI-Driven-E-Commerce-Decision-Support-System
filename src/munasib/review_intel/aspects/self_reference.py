"""Self-reference exclusion: aspects that only restate the product itself.

Reviews name the product when they complain, so an aspect such as ``filter``
or ``mwf filter`` for a water-filter product correlates with complaints
without describing any feature. The exclusion drops such aspects from a
product's ``product_aspects`` rows.

The rule is a token-set test:

1. Lemmatize the product's ``title_clean``, ``brand`` and ``category_leaf``
   with the same spaCy pipeline the extractor uses (``load_nlp`` +
   ``token.lemma_``). Lowercase alphabetic lemmas of at least
   ``min_token_chars`` form the product's **self-reference token set**.
2. An aspect is excluded when **every** token of the aspect is in that set.

For a product titled ``...Refrigerator Water Filter`` the set contains
``refrigerator``, ``water`` and ``filter``: ``filter`` and ``water filter``
are excluded, while ``water flow`` (``flow`` is absent) and ``fit`` are kept.

Limits (documented in ``docs/aspects.md``): the test is lexical, not
semantic. A feature word that also occurs in the title is hidden for that
product, brand tokens drop brand discussion, and category tokens drop
category-generic words for products whose category names them. Only stored
product rows are filtered; ``review_features.aspects_json`` is never
modified.

The lemmatized sets are cached in ``self_reference_cache_path`` under a
fingerprint of the lemmatizer settings plus a hash of the product metadata,
so a re-run reads the cache instead of loading spaCy again.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from munasib.core.config import AspectsConfig
from munasib.review_intel.aspects.types import ProductAspectStats

#: Per-product set of lemmatized title/brand/category tokens, keyed by ASIN.
SelfTokenSets = dict[str, frozenset[str]]

#: Batch lemmatizer: one lemma list per input text, in input order.
BatchLemmatizer = Callable[[Sequence[str]], Sequence[Sequence[str]]]


@dataclass(frozen=True)
class ProductIdentity:
    """The metadata fields that name a product (self-reference input).

    Attributes:
        asin: Product ASIN (cache key and aggregation join key).
        title_clean: Cleaned product title, or None when missing.
        brand: Brand name, or None when missing.
        category_leaf: Category breadcrumb leaf, or None when missing.
    """

    asin: str
    title_clean: str | None
    brand: str | None
    category_leaf: str | None


@dataclass(frozen=True)
class SelfReferenceFilter:
    """Partition of aggregated rows produced by one exclusion pass.

    Attributes:
        kept: Rows that survive the rule (written to ``product_aspects``).
        excluded: Rows whose aspect tokens all name the product itself.
    """

    kept: list[ProductAspectStats]
    excluded: list[ProductAspectStats]


@dataclass(frozen=True)
class ExcludedAspectSummary:
    """Aggregated view of one excluded aspect across products.

    Attributes:
        aspect: Excluded aspect phrase.
        rows: Number of product-aspect rows removed for it.
        mentions: Total mentions across those rows.
    """

    aspect: str
    rows: int
    mentions: int


def spacy_batch_lemmatizer(nlp: Any) -> BatchLemmatizer:
    """Adapt a spaCy pipeline to the batch-lemmatizer protocol.

    This is the same lemmatization the extractor performs
    (``token.lemma_`` over the configured model), so title tokens and aspect
    tokens are comparable.

    Args:
        nlp: Loaded pipeline from ``load_nlp``.

    Returns:
        Callable mapping texts to one lemma list per text.
    """

    def lemmatize(texts: Sequence[str]) -> list[list[str]]:
        return [[token.lemma_ for token in doc] for doc in nlp.pipe(texts)]

    return lemmatize


def build_self_token_sets(
    products: Sequence[ProductIdentity],
    config: AspectsConfig,
    lemmatize: BatchLemmatizer,
) -> SelfTokenSets:
    """Lemmatize product metadata into per-product self-reference token sets.

    Every non-empty metadata field of every product is lemmatized in one
    batch. A token enters the set when it is alphabetic, at least
    ``min_token_chars`` long after lowercasing (the same token filters the
    aspect extractor applies).

    Args:
        products: Product identities (any order; results keyed by ASIN).
        config: Aspect configuration (``min_token_chars``).
        lemmatize: Batch lemmatizer applied to every non-empty field.

    Returns:
        One token set per product; an empty set when all its metadata fields
        are missing (then nothing can be excluded for that product).

    Raises:
        ValueError: If the lemmatizer returns a different number of lemma
            lists than texts it was given.
    """
    owners: list[str] = []
    texts: list[str] = []
    for product in products:
        for value in (product.title_clean, product.brand, product.category_leaf):
            if value and value.strip():
                owners.append(product.asin)
                texts.append(value)
    lemma_lists = list(lemmatize(texts)) if texts else []
    tokens: defaultdict[str, set[str]] = defaultdict(set)
    for asin, lemmas in zip(owners, lemma_lists, strict=True):
        for lemma in lemmas:
            token = (lemma or "").strip().lower()
            if len(token) >= config.min_token_chars and token.isalpha():
                tokens[asin].add(token)
    return {product.asin: frozenset(tokens.get(product.asin, ())) for product in products}


def is_self_reference(aspect: str, tokens: Set[str]) -> bool:
    """Return True when every aspect token names the product itself.

    Args:
        aspect: Aspect phrase (lowercase lemmas separated by spaces).
        tokens: Product self-reference token set.

    Returns:
        False when the token set is empty (nothing matches) or any aspect
        token is absent from the set.
    """
    if not tokens:
        return False
    parts = aspect.split()
    return bool(parts) and all(part in tokens for part in parts)


def filter_self_reference(
    stats: Sequence[ProductAspectStats],
    token_sets: Mapping[str, Set[str]],
    *,
    enabled: bool = True,
) -> SelfReferenceFilter:
    """Split aggregated rows into kept and self-reference-excluded rows.

    The pass preserves input order and is idempotent: filtering ``kept``
    again excludes nothing more, so re-running the aggregation over an
    already-filtered table changes nothing.

    Args:
        stats: Aggregated rows (any order).
        token_sets: Self-reference token sets keyed by ASIN.
        enabled: When False every row is kept (config flag off).

    Returns:
        Partition of ``stats``.
    """
    if not enabled:
        return SelfReferenceFilter(kept=list(stats), excluded=[])
    kept: list[ProductAspectStats] = []
    excluded: list[ProductAspectStats] = []
    for stat in stats:
        if is_self_reference(stat.aspect, token_sets.get(stat.asin, frozenset())):
            excluded.append(stat)
        else:
            kept.append(stat)
    return SelfReferenceFilter(kept=kept, excluded=excluded)


def excluded_aspect_summary(
    excluded: Sequence[ProductAspectStats], limit: int
) -> list[ExcludedAspectSummary]:
    """Rank excluded aspects for the aggregate report.

    Args:
        excluded: Rows removed by the filter.
        limit: Maximum number of entries (the report prints 20).

    Returns:
        Up to ``limit`` summaries ordered by excluded rows descending, then
        excluded mentions descending, then aspect ascending - deterministic
        for identical inputs.
    """
    rows: defaultdict[str, int] = defaultdict(int)
    mentions: defaultdict[str, int] = defaultdict(int)
    for stat in excluded:
        rows[stat.aspect] += 1
        mentions[stat.aspect] += stat.counts.mentions
    ranked = sorted(rows, key=lambda aspect: (-rows[aspect], -mentions[aspect], aspect))
    return [
        ExcludedAspectSummary(aspect=aspect, rows=rows[aspect], mentions=mentions[aspect])
        for aspect in ranked[:limit]
    ]


def self_reference_fingerprint(
    config: AspectsConfig, products: Sequence[ProductIdentity]
) -> dict[str, object]:
    """Return the cache fingerprint for the lemmatizer settings and inputs.

    The fingerprint covers everything that changes the token sets: the spaCy
    model, the token-length rule and a hash of every product's metadata.
    Products are hashed in ASIN order so the value is order-independent.

    Args:
        config: Aspect configuration (``spacy_model``, ``min_token_chars``).
        products: Product identities.

    Returns:
        JSON-serialisable mapping stored with the cached token sets.
    """
    digest = hashlib.sha256()
    for product in sorted(products, key=lambda identity: identity.asin):
        digest.update(
            json.dumps(
                [
                    product.asin,
                    product.title_clean,
                    product.brand,
                    product.category_leaf,
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "spacy_model": config.spacy_model,
        "min_token_chars": config.min_token_chars,
        "products_sha256": digest.hexdigest(),
    }


def load_self_token_cache(
    path: Path, fingerprint: Mapping[str, object]
) -> SelfTokenSets | None:
    """Read cached token sets when the fingerprint still matches.

    Args:
        path: Cache file (``self_reference_cache_path``).
        fingerprint: Current fingerprint from :func:`self_reference_fingerprint`.

    Returns:
        Token sets, or None when the file is missing, corrupt, or was built
        for other settings/metadata (the caller rebuilds it).
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("fingerprint") != dict(fingerprint):
        return None
    raw_tokens = payload.get("tokens")
    if not isinstance(raw_tokens, dict):
        return None
    token_sets: SelfTokenSets = {}
    for asin, tokens in raw_tokens.items():
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            return None
        token_sets[str(asin)] = frozenset(tokens)
    return token_sets


def save_self_token_cache(
    path: Path,
    token_sets: Mapping[str, Iterable[str]],
    fingerprint: Mapping[str, object],
) -> None:
    """Write token sets deterministically (sorted ASINs and sorted tokens).

    Args:
        path: Destination file (parent directories are created).
        token_sets: Token sets keyed by ASIN.
        fingerprint: Fingerprint stored next to the sets.

    Raises:
        OSError: If the file cannot be written.
    """
    payload = {
        "fingerprint": dict(fingerprint),
        "tokens": {
            asin: sorted(tokens) for asin, tokens in sorted(token_sets.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=1), encoding="utf-8")


__all__ = [
    "BatchLemmatizer",
    "ExcludedAspectSummary",
    "ProductIdentity",
    "SelfReferenceFilter",
    "SelfTokenSets",
    "build_self_token_sets",
    "excluded_aspect_summary",
    "filter_self_reference",
    "is_self_reference",
    "load_self_token_cache",
    "save_self_token_cache",
    "self_reference_fingerprint",
    "spacy_batch_lemmatizer",
]
