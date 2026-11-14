"""spaCy noun-chunk extraction of aspect candidates.

The extractor keeps nominal words from spaCy noun chunks (plus unclaimed nouns
that the chunker missed) and canonicalises them into lowercase lemmas. A phrase
has at most ``max_aspect_tokens`` tokens, taken from the rightmost (head) end of
the chunk, so ``"the stainless steel ice maker"`` becomes ``"ice maker"``.

Two filtering layers exist deliberately:

* ``extract_doc_aspects(..., apply_stoplist=True)`` applies the configured
  generic-noun stoplist. This is the public extraction behaviour.
* The resumable candidate cache calls it with ``apply_stoplist=False`` so the
  cached parts stay raw. The vocabulary and per-review stages apply the current
  stoplist afterwards; extending the stoplist therefore rebuilds the vocabulary
  without another 20-minute spaCy pass. The final stored aspects are always
  stoplist-filtered.

Aspect polarity is **review-level**, not sentence-level: the caller pairs each
extracted aspect with the review's stored predicted sentiment. A long review
that mentions a good motor and a bad filter in different sentences therefore
gives both aspects the review's one label. This limitation is documented in
``docs/aspects.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from munasib.core.config import AspectsConfig
from munasib.core.exceptions import DataError

#: POS tags that may contribute a token to an aspect phrase. Noun chunks also
#: contain determiners, adjectives and particles; only nominal heads survive.
ASPECT_POS_TAGS: frozenset[str] = frozenset({"NOUN", "PROPN"})


class TokenLike(Protocol):
    """Minimal spaCy ``Token`` surface the canonicaliser reads."""

    pos_: str
    lemma_: str
    text: str
    is_alpha: bool
    like_num: bool


@dataclass(frozen=True)
class ExtractionTiming:
    """Measured throughput of one speed sample."""

    documents: int
    seconds: float

    @property
    def documents_per_second(self) -> float:
        """Return documents processed per second.

        Returns:
            Throughput, or 0.0 for an empty/zero-second sample.
        """
        return self.documents / self.seconds if self.seconds > 0 else 0.0

    def estimated_seconds(self, total_documents: int) -> float | None:
        """Estimate a full pass from this sample.

        Args:
            total_documents: Number of reviews to process.

        Returns:
            Estimated seconds, or None when the sample measured no throughput
            (the caller must not invent an estimate).
        """
        rate = self.documents_per_second
        return total_documents / rate if rate > 0 else None


def load_nlp(config: AspectsConfig) -> Any:
    """Load the configured spaCy pipeline and verify its required components.

    Args:
        config: Aspect configuration (model name and disabled components).

    Returns:
        The loaded spaCy ``Language`` object.

    Raises:
        DataError: If spaCy or the model is unavailable, or a component the
            extractor depends on was disabled.
    """
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - dependency is installed in this project
        raise DataError(
            "spaCy is required for aspect extraction; install it from requirements"
        ) from exc

    try:
        nlp = spacy.load(config.spacy_model, disable=config.spacy_disable_components)
    except OSError as exc:
        raise DataError(
            f"spaCy model {config.spacy_model!r} is not installed; run "
            f"'python -m spacy download {config.spacy_model}'"
        ) from exc

    missing = {
        "tagger",
        "parser",
        "attribute_ruler",
        "lemmatizer",
    } - set(nlp.pipe_names)
    if missing:
        raise DataError(
            f"spaCy model {config.spacy_model!r} lacks required component(s) "
            f"{sorted(missing)}; the aspect extractor needs noun chunks and lemmas"
        )
    return nlp


def aspect_stoplist(config: AspectsConfig) -> frozenset[str]:
    """Return the configured generic-noun stoplist as a set.

    Args:
        config: Aspect configuration.

    Returns:
        Lowercase generic lemmas that may never become aspects.
    """
    return frozenset(config.generic_nouns)


def canonical_aspect(
    tokens: Sequence[TokenLike],
    config: AspectsConfig,
    *,
    apply_stoplist: bool = True,
) -> str | None:
    """Canonicalise one noun-chunk token sequence into an aspect phrase.

    Nominal tokens are collected from right to left (the head first), so a long
    chunk contributes its most specific tail. Tokens shorter than
    ``min_token_chars``, non-alphabetic tokens, numerals and configured generic
    nouns are skipped.

    Args:
        tokens: Tokens of one chunk (or one unclaimed token).
        config: Aspect configuration.
        apply_stoplist: When True, drop configured generic nouns. The candidate
            cache calls this with False and filters later.

    Returns:
        Lowercase aspect phrase, or None when no usable nominal token remains.
    """
    stoplist = aspect_stoplist(config) if apply_stoplist else frozenset()
    kept: list[str] = []
    for token in reversed(tokens):
        if token.pos_ not in ASPECT_POS_TAGS:
            continue
        if token.like_num or not token.is_alpha:
            continue
        lemma = (token.lemma_ or token.text).strip().lower()
        if len(lemma) < config.min_token_chars or not lemma:
            continue
        if lemma in stoplist:
            continue
        kept.append(lemma)
        if len(kept) >= config.max_aspect_tokens:
            break
    if not kept:
        return None
    return " ".join(reversed(kept))


def filter_aspect_phrase(aspect: str, config: AspectsConfig) -> str | None:
    """Re-canonicalise a cached aspect phrase under the current stoplist.

    Candidate parts are raw (stoplist not applied), so a stoplist extension must
    also reach phrases that *contain* a generic token: ``"amazon review"`` is
    dropped, ``"customer service"`` becomes ``"service"``, and
    ``"year warranty"`` becomes ``"warranty"`` - exactly what
    :func:`canonical_aspect` would have produced during a live extraction.

    Args:
        aspect: Cached aspect phrase (lowercase lemmas separated by spaces).
        config: Aspect configuration (stoplist, token length, phrase length).

    Returns:
        Canonicalised phrase, or None when every token is generic/too short/
        non-alphabetic (the phrase must not enter the vocabulary).
    """
    stoplist = aspect_stoplist(config)
    kept: list[str] = []
    for token in reversed(aspect.split()):
        lemma = token.strip().lower()
        if len(lemma) < config.min_token_chars or not lemma.isalpha():
            continue
        if lemma in stoplist:
            continue
        kept.append(lemma)
        if len(kept) >= config.max_aspect_tokens:
            break
    if not kept:
        return None
    return " ".join(reversed(kept))


def extract_doc_aspects(
    doc: Any,
    config: AspectsConfig,
    *,
    apply_stoplist: bool = True,
) -> list[str]:
    """Extract unique aspect candidates from one parsed spaCy document.

    Noun chunks are the primary source. Nouns that the chunker did not claim
    (for example a bare noun in an odd coordination) are added as single-token
    candidates only when they are not already covered. Result order is the
    document order of first appearance; duplicates collapse because a review
    mention is counted once.

    Args:
        doc: Parsed spaCy ``Doc`` (needs ``noun_chunks`` and tokens).
        config: Aspect configuration.
        apply_stoplist: Whether to apply the generic-noun stoplist.

    Returns:
        Unique aspect phrases in first-seen order.
    """
    found: list[str] = []
    seen: set[str] = set()
    covered: set[int] = set()

    for chunk in doc.noun_chunks:
        chunk_tokens = list(chunk)
        covered.update(id(token) for token in chunk_tokens)
        aspect = canonical_aspect(chunk_tokens, config, apply_stoplist=apply_stoplist)
        if aspect is not None and aspect not in seen:
            seen.add(aspect)
            found.append(aspect)

    for token in doc:
        if id(token) in covered or token.pos_ not in ASPECT_POS_TAGS:
            continue
        aspect = canonical_aspect([token], config, apply_stoplist=apply_stoplist)
        if aspect is not None and aspect not in seen:
            seen.add(aspect)
            found.append(aspect)
    return found


def extract_text_aspects(
    text: str,
    nlp: Any,
    config: AspectsConfig,
    *,
    apply_stoplist: bool = True,
) -> list[str]:
    """Parse one review prefix and extract its aspect candidates.

    Args:
        text: Review text.
        nlp: Loaded spaCy pipeline.
        config: Aspect configuration (``max_text_chars``, token rules).
        apply_stoplist: Whether to apply the generic-noun stoplist.

    Returns:
        Unique aspect phrases in first-seen order; empty when the text is blank.
    """
    prefix = text[: config.max_text_chars]
    if not prefix.strip():
        return []
    return extract_doc_aspects(
        nlp(prefix), config, apply_stoplist=apply_stoplist
    )


def measure_extraction_speed(
    nlp: Any,
    texts: Sequence[str],
    config: AspectsConfig,
    *,
    apply_stoplist: bool = False,
) -> ExtractionTiming:
    """Measure spaCy throughput on a sample of review texts.

    Args:
        nlp: Loaded spaCy pipeline.
        texts: Sample review texts.
        config: Aspect configuration (batch size, process count, prefix cap).
        apply_stoplist: Whether to apply the stoplist while measuring. The
            sample is only used for timing, so the default leaves it raw.

    Returns:
        Measured document count and elapsed wall-clock seconds.
    """
    prefixes = [text[: config.max_text_chars] for text in texts]
    start = perf_counter()
    documents = 0
    for doc in nlp.pipe(
        prefixes,
        batch_size=config.batch_size,
        n_process=config.n_process,
    ):
        extract_doc_aspects(doc, config, apply_stoplist=apply_stoplist)
        documents += 1
    return ExtractionTiming(documents=documents, seconds=perf_counter() - start)


def iter_extracted_aspects(
    nlp: Any,
    texts: Iterable[str],
    config: AspectsConfig,
    *,
    apply_stoplist: bool = False,
) -> Iterable[list[str]]:
    """Yield one aspect list per input text using spaCy's batched pipe.

    Args:
        nlp: Loaded spaCy pipeline.
        texts: Review texts in input order.
        config: Aspect configuration (batch size and process count).
        apply_stoplist: Whether to apply the generic-noun stoplist.

    Yields:
        Aspect lists aligned with ``texts``.
    """
    prefixes = (text[: config.max_text_chars] for text in texts)
    for doc in nlp.pipe(
        prefixes,
        batch_size=config.batch_size,
        n_process=config.n_process,
    ):
        yield extract_doc_aspects(doc, config, apply_stoplist=apply_stoplist)


__all__ = [
    "ASPECT_POS_TAGS",
    "ExtractionTiming",
    "TokenLike",
    "aspect_stoplist",
    "canonical_aspect",
    "extract_doc_aspects",
    "extract_text_aspects",
    "iter_extracted_aspects",
    "load_nlp",
    "measure_extraction_speed",
]
