"""Extract review aspects, persist them and aggregate per-product statistics.

The stage is split so the ~20-minute spaCy pass happens **once** and can resume
after an interruption:

``--vocabulary``
    Extract candidates into resumable parquet parts (skipping complete parts),
    then build ``data/interim/aspect_vocabulary.parquet`` and
    ``reports/aspect_vocabulary.csv``. Prints the top 100 aspects and the 50
    nearest the ``min_review_count`` cutoff for inspection.

``--reviews``
    Filter the cached candidates against the vocabulary and upsert
    ``review_features.aspects_json``. This reads no transformer and runs no
    spaCy. ``--limit N`` processes only the first N reviews (development aid).

``--aggregate``
    Read the database only and replace ``product_aspects`` with the per-product
    counts, shares, smoothed lift, classification and examples. Requires a
    complete per-review aspect pass.

``--all``
    Run vocabulary, reviews and aggregation in that order without repeating the
    extraction. ``--limit`` is rejected here because a partial aspect pass would
    make the whole-category aggregation misleading.

Exit codes:
  0 - stage completed.
  1 - configuration/data failure.
  2 - resumable-cache conflict (missing parts without ``--resume``, or a
      fingerprint mismatch). The message names the directory to remove.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from munasib.core.config import AspectsConfig, Settings, load_settings
from munasib.core.exceptions import ConfigError, DataError
from munasib.db.repositories import (
    ProductAspectRepository,
    ProductRepository,
    ReviewFeatureRepository,
)
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)
from munasib.product_intel.features import read_build_timestamp
from munasib.review_intel.aspects.aggregation import iter_product_aspects
from munasib.review_intel.aspects.cache import (
    AspectCacheIncomplete,
    AspectCacheMismatch,
    CandidateCache,
    extraction_fingerprint,
    iter_part_records,
    source_signature,
)
from munasib.review_intel.aspects.extraction import (
    filter_aspect_phrase,
    iter_extracted_aspects,
    load_nlp,
    measure_extraction_speed,
)
from munasib.review_intel.aspects.storage import (
    ReviewAspectWriter,
    count_text_reviews,
    iter_prediction_rows,
    iter_text_reviews,
)
from munasib.review_intel.aspects.types import (
    ASPECT_NEUTRAL,
    ASPECT_PAIN_POINT,
    ASPECT_STRENGTH,
    ProductAspectStats,
)
from munasib.review_intel.aspects.vocabulary import (
    build_vocabulary,
    load_vocabulary,
    vocabulary_preview,
    write_vocabulary,
    write_vocabulary_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("munasib.compute_aspects")

#: Cache conflict exit code (documented in the module docstring).
EXIT_CACHE_CONFLICT = 2
#: Rows per database transaction while persisting per-review aspects.
DB_BATCH_SIZE = 1000
#: Products whose pain points/strengths are printed after aggregation.
DEFAULT_EXAMPLE_PRODUCTS = 3


@dataclass(frozen=True)
class VocabularyRun:
    """Summary of the vocabulary stage."""

    total_reviews: int
    extracted_now: int
    reused_parts: int
    vocabulary_size: int
    raw_candidates: int


@dataclass(frozen=True)
class ReviewRun:
    """Summary of the per-review persistence stage."""

    processed: int
    with_aspects: int
    mentions: int
    vocabulary_size: int


@dataclass(frozen=True)
class AggregateRun:
    """Summary of the product aggregation stage."""

    rows: int
    products: int
    pain_points: int
    strengths: int
    neutral: int
    examples: list[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract aspects from cached review predictions and aggregate per-product "
            "pain points/strengths."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration (default: configs/config.yaml).",
    )
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument(
        "--vocabulary",
        action="store_true",
        help="Extract/reuse candidate parts and build the category vocabulary.",
    )
    stage.add_argument(
        "--reviews",
        action="store_true",
        help="Filter cached candidates into review_features.aspects_json.",
    )
    stage.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate stored aspects into product_aspects.",
    )
    stage.add_argument(
        "--all",
        action="store_true",
        help="Run vocabulary, reviews and aggregate without repeating spaCy.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted extraction; complete parts are reused.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N reviews (only valid with --reviews).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=DEFAULT_EXAMPLE_PRODUCTS,
        help=(
            "Print this many products' pain points/strengths after aggregation "
            f"(default: {DEFAULT_EXAMPLE_PRODUCTS})."
        ),
    )
    return parser.parse_args(argv)


def _fingerprint_payload(config: AspectsConfig, source: dict[str, object]) -> dict[str, object]:
    """Return the human-readable fingerprint payload stored in the manifest.

    Args:
        config: Aspect configuration.
        source: Review-source signature.

    Returns:
        Mapping with the settings that affect extraction.
    """
    return {
        "spacy_model": config.spacy_model,
        "spacy_disable_components": list(config.spacy_disable_components),
        "max_text_chars": config.max_text_chars,
        "min_token_chars": config.min_token_chars,
        "max_aspect_tokens": config.max_aspect_tokens,
        "chunk_rows": config.chunk_rows,
        "source": source,
    }


def _format_seconds(seconds: float | None) -> str:
    """Format an elapsed/estimated duration for the log.

    Args:
        seconds: Duration in seconds, or None when unmeasurable.

    Returns:
        Human-readable duration or ``"unknown"``.
    """
    if seconds is None:
        return "unknown"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes}m {remainder:02d}s"


def run_vocabulary(settings: Settings, *, resume: bool) -> VocabularyRun:
    """Extract/reuse candidates and write the category vocabulary.

    Args:
        settings: Loaded project settings.
        resume: Continue an interrupted extraction instead of failing.

    Returns:
        Vocabulary-stage summary.

    Raises:
        AspectCacheMismatch: If the cache belongs to other settings/source.
        AspectCacheIncomplete: If parts are missing and ``resume`` is False.
        DataError: If there is no usable review text or no candidate mentions.
    """
    config = settings.review_intel.aspects
    engine = create_engine_from_settings(settings)
    init_db(engine)
    factory = create_session_factory(engine)
    session = factory()
    extracted_now = 0
    try:
        total_reviews = count_text_reviews(session)
        if total_reviews == 0:
            raise DataError(
                "no reviews with usable text in the serving cache; run "
                "scripts/build_dataset.py and scripts/ingest_db.py first"
            )
        source = source_signature(session)
        fingerprint = extraction_fingerprint(config, source)
        cache = CandidateCache(Path(config.candidate_parts_dir), config.chunk_rows)
        status = cache.prepare(
            fingerprint=fingerprint,
            fingerprint_payload=_fingerprint_payload(config, source),
            total_reviews=total_reviews,
            resume=resume,
        )
        logger.info(
            "candidate_cache parts=%d complete=%d missing=%d source_text_reviews=%d",
            status.manifest.part_count,
            len(status.completed_parts),
            len(status.missing_parts),
            total_reviews,
        )

        if status.missing_parts:
            nlp = load_nlp(config)
            sample = iter_text_reviews(
                session,
                limit=min(config.speed_sample_size, total_reviews),
            )
            timing = measure_extraction_speed(
                nlp,
                [row.text for row in sample],
                config,
                apply_stoplist=False,
            )
            estimate = timing.estimated_seconds(total_reviews)
            logger.info(
                "speed_sample docs=%d seconds=%.2f docs_per_second=%.1f estimated=%s",
                timing.documents,
                timing.seconds,
                timing.documents_per_second,
                _format_seconds(estimate),
            )
            after: str | None = None
            for index in range(status.manifest.part_count):
                if index in status.completed_parts:
                    after = cache.last_review_id(index)
                    continue
                rows = iter_text_reviews(
                    session,
                    after_review_id=after,
                    limit=config.chunk_rows,
                )
                expected = cache.expected_rows(index, total_reviews)
                if len(rows) != expected:
                    raise DataError(
                        f"review source changed while extracting part {index}: expected "
                        f"{expected} rows, found {len(rows)}; re-run without --resume after "
                        "removing the candidate cache"
                    )
                aspect_lists = iter_extracted_aspects(
                    nlp,
                    [row.text for row in rows],
                    config,
                    apply_stoplist=False,
                )
                records = [
                    {
                        "review_id": row.review_id,
                        "asin": row.asin,
                        "aspects": aspects,
                    }
                    for row, aspects in zip(rows, aspect_lists, strict=True)
                ]
                cache.write_part(index, records)
                extracted_now += len(records)
                after = rows[-1].review_id
                logger.info(
                    "extracted part=%d/%d reviews=%d total_new=%d",
                    index + 1,
                    status.manifest.part_count,
                    len(records),
                    extracted_now,
                )
            del nlp

        result = build_vocabulary(cache.iter_parts(), config)
        write_vocabulary(result.entries, config.vocabulary_path)
        write_vocabulary_csv(result.entries, config.vocabulary_csv_path)
        print()
        for line in vocabulary_preview(result):
            print(line)
        print()
        print(f"vocabulary parquet          : {config.vocabulary_path}")
        print(f"vocabulary csv              : {config.vocabulary_csv_path}")
        return VocabularyRun(
            total_reviews=total_reviews,
            extracted_now=extracted_now,
            reused_parts=len(status.completed_parts),
            vocabulary_size=len(result.entries),
            raw_candidates=len(result.all_review_counts),
        )
    finally:
        session.close()
        engine.dispose()


def _validate_cache_for_reviews(
    session,
    config: AspectsConfig,
) -> tuple[CandidateCache, int]:
    """Verify that a complete candidate cache exists for the current source.

    Args:
        session: Open session.
        config: Aspect configuration.

    Returns:
        Cache and total text-review count.

    Raises:
        DataError: If no manifest exists.
        AspectCacheMismatch: If the cache belongs to another source/settings.
        AspectCacheIncomplete: If any part is missing.
    """
    total_reviews = count_text_reviews(session)
    if total_reviews == 0:
        raise DataError("no reviews with usable text in the serving cache")
    source = source_signature(session)
    fingerprint = extraction_fingerprint(config, source)
    cache = CandidateCache(Path(config.candidate_parts_dir), config.chunk_rows)
    if cache.load_manifest() is None:
        raise DataError(
            f"candidate cache not found at {config.candidate_parts_dir}; run "
            "'python scripts/compute_aspects.py --vocabulary' first"
        )
    status = cache.prepare(
        fingerprint=fingerprint,
        fingerprint_payload=_fingerprint_payload(config, source),
        total_reviews=total_reviews,
        resume=True,
    )
    if not status.complete:
        raise AspectCacheIncomplete(
            f"{len(status.missing_parts)} candidate part(s) are missing in "
            f"{config.candidate_parts_dir}; run --vocabulary --resume first"
        )
    return cache, total_reviews


def run_reviews(settings: Settings, *, limit: int | None) -> ReviewRun:
    """Filter cached candidates and upsert per-review aspects.

    Args:
        settings: Loaded project settings.
        limit: Maximum reviews to process (None = all).

    Returns:
        Per-review stage summary.

    Raises:
        DataError: If the vocabulary or candidate cache is missing/incomplete.
        AspectCacheMismatch: If the cache belongs to another source/settings.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    config = settings.review_intel.aspects
    vocabulary = load_vocabulary(config.vocabulary_path)
    vocabulary_terms = {entry.aspect for entry in vocabulary}

    engine = create_engine_from_settings(settings)
    init_db(engine)
    factory = create_session_factory(engine)
    session = factory()
    try:
        cache, _total = _validate_cache_for_reviews(session, config)
        if limit is None:
            with session_scope(factory) as write_session:
                cleared = ReviewFeatureRepository(write_session).clear_aspects()
                logger.info("cleared_existing_aspects rows=%d", cleared)

        processed = 0
        with_aspects = 0
        mentions = 0
        pending: list[dict[str, object]] = []
        with ReviewAspectWriter(config.review_aspects_path) as writer:
            for frame in cache.iter_parts():
                for review_id, _asin, aspects in iter_part_records(frame):
                    if limit is not None and processed >= limit:
                        break
                    # Same re-canonicalisation the vocabulary builder applied,
                    # so stored mentions and vocabulary counts agree.
                    rewritten = [
                        phrase
                        for aspect in aspects
                        if (phrase := filter_aspect_phrase(aspect, config)) is not None
                    ]
                    kept = [
                        aspect
                        for aspect in dict.fromkeys(rewritten)
                        if aspect in vocabulary_terms
                    ]
                    writer.write(review_id, kept)
                    pending.append(
                        {"review_id": review_id, "aspects_json": kept}
                    )
                    processed += 1
                    with_aspects += bool(kept)
                    mentions += len(kept)
                    if len(pending) >= DB_BATCH_SIZE:
                        with session_scope(factory) as write_session:
                            ReviewFeatureRepository(write_session).upsert_aspects(pending)
                        pending.clear()
                if limit is not None and processed >= limit:
                    break
        if pending:
            with session_scope(factory) as write_session:
                ReviewFeatureRepository(write_session).upsert_aspects(pending)
            pending.clear()

        coverage_ok = True
        if limit is None:
            with session_scope(factory) as read_session:
                uncovered = ReviewFeatureRepository(
                    read_session
                ).count_text_reviews_without_aspects()
            coverage_ok = uncovered == 0
            if not coverage_ok:
                raise DataError(
                    f"{uncovered:,} text review(s) still have no aspect payload after "
                    "the full pass; the database and candidate cache disagree"
                )
        logger.info(
            "reviews_written processed=%d with_aspects=%d mentions=%d coverage_ok=%s",
            processed,
            with_aspects,
            mentions,
            coverage_ok,
        )
        return ReviewRun(
            processed=processed,
            with_aspects=with_aspects,
            mentions=mentions,
            vocabulary_size=len(vocabulary),
        )
    finally:
        session.close()
        engine.dispose()


def _render_product_examples(
    rows: Sequence[ProductAspectStats],
    titles: dict[str, str],
    limit: int,
) -> list[str]:
    """Render pain points and strengths for a few products.

    Args:
        rows: All aggregated product-aspect statistics.
        titles: Product ASIN -> title.
        limit: Number of products to print.

    Returns:
        Lines ready to print. Products are chosen by total mentions of their
        non-neutral aspects, then ASIN, so a run is reproducible.
    """
    by_asin: defaultdict[str, list[ProductAspectStats]] = defaultdict(list)
    for row in rows:
        by_asin[row.asin].append(row)
    candidates = [
        (asin, stats)
        for asin, stats in by_asin.items()
        if any(
            stat.classification in (ASPECT_PAIN_POINT, ASPECT_STRENGTH)
            for stat in stats
        )
    ]
    candidates.sort(
        key=lambda item: (
            -sum(
                stat.counts.mentions
                for stat in item[1]
                if stat.classification in (ASPECT_PAIN_POINT, ASPECT_STRENGTH)
            ),
            item[0],
        )
    )
    lines: list[str] = []
    for asin, stats in candidates[:limit]:
        title = titles.get(asin) or "(title not available)"
        lines.append(f"{asin} | {title}")
        for classification, heading in (
            (ASPECT_PAIN_POINT, "pain points"),
            (ASPECT_STRENGTH, "strengths"),
        ):
            selected = sorted(
                (stat for stat in stats if stat.classification == classification),
                key=lambda stat: (-stat.counts.mentions, stat.aspect),
            )
            if not selected:
                lines.append(f"  {heading}: none")
                continue
            lines.append(f"  {heading}:")
            for stat in selected:
                lift = "n/a" if stat.negativity_lift is None else f"{stat.negativity_lift:.3f}"
                lines.append(
                    f"    - {stat.aspect}: mentions={stat.counts.mentions}, "
                    f"negative={stat.counts.negative} "
                    f"({stat.counts.negative_share:.1%}), "
                    f"positive_share={stat.counts.positive_share:.1%}, lift={lift}"
                )
        lines.append("")
    if not lines:
        lines.append("no product has a stored pain point or strength")
    return lines


def run_aggregate(settings: Settings, *, examples: int) -> AggregateRun:
    """Aggregate stored per-review aspects into ``product_aspects``.

    Args:
        settings: Loaded project settings.
        examples: Number of products whose pain points/strengths are printed.

    Returns:
        Aggregation summary.

    Raises:
        DataError: If aspect coverage is incomplete or there is nothing to
            aggregate.
    """
    if examples < 1:
        raise ValueError(f"examples must be >= 1, got {examples}")
    config = settings.review_intel.aspects
    engine = create_engine_from_settings(settings)
    init_db(engine)
    factory = create_session_factory(engine)
    computed_at = read_build_timestamp(settings)
    try:
        with session_scope(factory) as session:
            feature_repository = ReviewFeatureRepository(session)
            uncovered = feature_repository.count_text_reviews_without_aspects()
            if uncovered:
                raise DataError(
                    f"{uncovered:,} text review(s) have no aspect payload; run "
                    "'python scripts/compute_aspects.py --reviews' first"
                )
            if feature_repository.count_aspect_rows() == 0:
                raise DataError(
                    "no stored aspect payloads; run "
                    "'python scripts/compute_aspects.py --reviews' first"
                )
            stats = list(iter_product_aspects(iter_prediction_rows(session), config))
            payloads = [stat.to_row(computed_at) for stat in stats]
            repository = ProductAspectRepository(session)
            written = repository.replace_all(payloads)
            counts = repository.count_by_classification()
            products = repository.count_products()
            asins = sorted({stat.asin for stat in stats})
            titles = {
                product.asin: product.title or ""
                for product in ProductRepository(session).get_products(asins)
            }
        logger.info(
            "aggregate_written rows=%d products=%d computed_at=%s",
            written,
            products,
            computed_at.isoformat(),
        )
        return AggregateRun(
            rows=written,
            products=products,
            pain_points=counts.get(ASPECT_PAIN_POINT, 0),
            strengths=counts.get(ASPECT_STRENGTH, 0),
            neutral=counts.get(ASPECT_NEUTRAL, 0),
            examples=_render_product_examples(stats, titles, examples),
        )
    finally:
        engine.dispose()


def _print_vocabulary_summary(summary: VocabularyRun) -> None:
    """Print the vocabulary-stage summary.

    Args:
        summary: Stage summary.
    """
    print()
    print(f"text reviews                : {summary.total_reviews:,}")
    print(f"reviews extracted now       : {summary.extracted_now:,}")
    print(f"complete parts reused       : {summary.reused_parts:,}")
    print(f"raw candidates              : {summary.raw_candidates:,}")
    print(f"vocabulary size             : {summary.vocabulary_size:,}")


def _print_review_summary(summary: ReviewRun) -> None:
    """Print the per-review-stage summary.

    Args:
        summary: Stage summary.
    """
    print()
    print(f"reviews processed           : {summary.processed:,}")
    print(f"reviews with aspects        : {summary.with_aspects:,}")
    print(f"aspect mentions             : {summary.mentions:,}")
    print(f"vocabulary size             : {summary.vocabulary_size:,}")


def _print_aggregate_summary(summary: AggregateRun) -> None:
    """Print the aggregation-stage summary and example products.

    Args:
        summary: Stage summary.
    """
    print()
    print(f"product-aspect rows         : {summary.rows:,}")
    print(f"products with aspects       : {summary.products:,}")
    print(f"pain points                 : {summary.pain_points:,}")
    print(f"strengths                   : {summary.strengths:,}")
    print(f"neutral aspects             : {summary.neutral:,}")
    print()
    print("Example products:")
    for line in summary.examples:
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested aspect stage.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 success, 1 failure, 2 cache/resume conflict).
    """
    args = parse_args(argv)
    if args.limit is not None and not args.reviews:
        logger.error(
            "invalid_limit detail=--limit is only valid with --reviews (not --all)"
        )
        return 1
    if args.limit is not None and args.limit < 1:
        logger.error("invalid_limit value=%d minimum=1", args.limit)
        return 1
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        logger.error("config_error path=%s detail=%s", args.config, exc)
        return 1

    try:
        if args.vocabulary:
            _print_vocabulary_summary(run_vocabulary(settings, resume=args.resume))
        elif args.reviews:
            _print_review_summary(run_reviews(settings, limit=args.limit))
        elif args.aggregate:
            _print_aggregate_summary(run_aggregate(settings, examples=args.examples))
        else:
            vocabulary = run_vocabulary(settings, resume=args.resume)
            _print_vocabulary_summary(vocabulary)
            reviews = run_reviews(settings, limit=None)
            _print_review_summary(reviews)
            aggregate = run_aggregate(settings, examples=args.examples)
            _print_aggregate_summary(aggregate)
        return 0
    except (AspectCacheMismatch, AspectCacheIncomplete) as exc:
        logger.error("cache_conflict detail=%s", exc)
        return EXIT_CACHE_CONFLICT
    except (DataError, ValueError) as exc:
        logger.error("run_failed detail=%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
