"""Download small product thumbnails into a local, git-ignored cache.

Images are **display-only** (no image analysis) and must work **offline**, so the
dashboard never hotlinks Amazon: this one-time script downloads the product image
of every selected candidate into ``images.directory`` and
``GET /products/{asin}/image`` serves it from disk. Only products that carry a
stored ``image_url`` are considered - a product with no image URL is simply not a
candidate, and nothing is fabricated for it.

Behaviour:

* **Polite**: one request at a time, a configurable pause between requests
  (``images.download_delay_seconds``), a project ``User-Agent`` and a single retry
  on transient failures (timeout, connection error, 429/5xx). A permanent failure
  (403/404/...) is not retried.
* **Verified**: every payload is decoded with Pillow before it is stored, so an
  HTML error page or a truncated file never lands in the cache.
* **Small and clean**: the image is resized so its longest side is at most
  ``images.thumbnail_max_side`` (aspect ratio preserved), flattened onto white,
  re-encoded as JPEG and written with no EXIF/ICC metadata.
* **Resumable**: an ASIN whose ``<asin>.jpg`` already exists is skipped unless
  ``--force`` is given; each success flips ``products.image_cached`` to true.
* **Never fatal per item**: a bad URL, a timeout or an unwritable file is recorded
  with a reason and the run continues.

Usage::

    python scripts/download_images.py --limit 5               # smoke test
    python scripts/download_images.py --all                   # the whole subset
    python scripts/download_images.py --asins B00QO4ZX30 B01FQR5RZE
    python scripts/download_images.py --asin-file asins.txt --force

Writes ``reports/image_download_report.json`` with the computed counts only
(attempted / ok / skipped / failed-by-reason).

Exit codes:
  0 - the run completed (individual failures are reported, not fatal).
  1 - the run could not start (bad config, missing database, no selection).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError
from sqlalchemy import Engine, select, update

from munasib.core.config import Settings, load_settings
from munasib.core.constants import ASIN_PATTERN
from munasib.core.exceptions import ConfigError, DataError
from munasib.db.models import Product
from munasib.db.session import create_engine_from_settings, create_session_factory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("munasib.download_images")

#: Name of the JSON summary written into ``paths.reports_dir``.
REPORT_FILENAME = "image_download_report.json"

#: Stored thumbnails are JPEG; quality is a fixed encoder setting, not a metric.
JPEG_QUALITY = 85

#: Pause before the single retry of a transient failure, in seconds.
RETRY_PAUSE_SECONDS = 2.0

#: HTTP status codes worth retrying once (rate limiting / transient server errors).
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

#: Outcome tokens recorded per ASIN.
OK = "ok"
SKIPPED = "skipped"
NOT_IN_CACHE = "not_in_cache"


class ImageDownloadError(Exception):
    """One-item download failure carrying a machine-readable reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download product thumbnails into the local image cache "
            "(display-only; the dashboard never hotlinks)."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration (default: configs/config.yaml).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--asins",
        nargs="+",
        default=None,
        metavar="ASIN",
        help="Download exactly these ASINs (any product, even one without an image URL).",
    )
    source.add_argument(
        "--asin-file",
        default=None,
        help="Text file with one ASIN per line to download.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Download at most N products (applied after selection).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every candidate (required to run the whole subset without --limit).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Pause between downloads in seconds (default: images.download_delay_seconds).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Thumbnail longest side in pixels (default: images.thumbnail_max_side).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite files that already exist in the cache.",
    )
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------
# Selection (which ASINs to download, and from which URL)
# ------------------------------------------------------------------------------


def _read_asin_file(path: Path) -> list[str]:
    """Read ASINs from a text file, one per line.

    Args:
        path: File to read.

    Returns:
        Non-empty stripped lines, in file order.

    Raises:
        DataError: If the file does not exist.
    """
    if not path.exists():
        raise DataError(f"--asin-file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _requested_asins(args: argparse.Namespace) -> list[str] | None:
    """Return the ASINs explicitly requested on the command line, if any.

    Args:
        args: Parsed arguments.

    Returns:
        The requested ASINs in order (deduplicated), or None when no explicit
        selection was given (the whole-image-URL subset is then the candidate set).
    """
    requested: list[str] = []
    if args.asins:
        requested.extend(args.asins)
    if args.asin_file:
        requested.extend(_read_asin_file(Path(args.asin_file)))
    if not requested:
        return None
    # De-duplicate while keeping the requested order.
    return list(dict.fromkeys(requested))


def select_targets(
    engine: Engine, args: argparse.Namespace
) -> tuple[list[tuple[str, str | None]], list[str]]:
    """Resolve the ``(asin, image_url)`` pairs to download.

    Explicit ``--asins``/``--asin-file`` selections are looked up in the cache and
    keep their requested order; an ASIN that is not a cached product cannot have a
    stored image URL and is returned separately (reported as ``not_in_cache``).
    Without an explicit selection, every product with a stored ``image_url`` is a
    candidate, in ASIN order.

    Args:
        engine: Serving-cache engine.
        args: Parsed arguments.

    Returns:
        ``(targets, unknown)`` where ``targets`` is the ordered list of
        ``(asin, image_url)`` pairs (``image_url`` None means "no stored URL") and
        ``unknown`` is the requested ASINs that are not in the serving cache.

    Raises:
        DataError: If nothing at all would be processed (empty cache and no
            explicit selection, or an explicit selection that matches nothing).
    """
    factory = create_session_factory(engine)
    requested = _requested_asins(args)
    unknown: list[str] = []
    with factory() as session:
        if requested is not None:
            rows = session.execute(
                select(Product.asin, Product.image_url).where(Product.asin.in_(requested))
            ).all()
            by_asin = dict(rows)
            unknown = [asin for asin in requested if asin not in by_asin]
            targets: list[tuple[str, str | None]] = [
                (asin, by_asin[asin]) for asin in requested if asin in by_asin
            ]
        else:
            rows = session.execute(
                select(Product.asin, Product.image_url)
                .where(Product.image_url.is_not(None))
                .order_by(Product.asin)
            ).all()
            targets = [(asin, image_url) for asin, image_url in rows]
    if args.limit is not None:
        targets = targets[: args.limit]
    if not targets and not unknown:
        raise DataError(
            "no products selected - the serving cache has no product with an "
            "image_url (run python scripts/build_dataset.py && python scripts/ingest_db.py)"
        )
    if unknown:
        logger.warning("not_in_cache count=%d asins=%s", len(unknown), ",".join(unknown))
    return targets, unknown


# ------------------------------------------------------------------------------
# Fetch, convert and store one image
# ------------------------------------------------------------------------------


def build_client(settings: Settings) -> httpx.Client:
    """Create the HTTP client used for every request.

    Args:
        settings: Loaded project settings.

    Returns:
        A configured ``httpx.Client`` (project User-Agent, timeout, redirects).
    """
    return httpx.Client(
        headers={"User-Agent": settings.images.user_agent},
        timeout=settings.images.timeout_seconds,
        follow_redirects=True,
    )


def fetch_image_bytes(client: httpx.Client, url: str) -> bytes:
    """Download one image URL, retrying a transient failure exactly once.

    Args:
        client: HTTP client.
        url: Absolute ``http(s)`` image URL.

    Returns:
        The raw response body.

    Raises:
        ImageDownloadError: With a reason token (``timeout``, ``network_error``,
            ``not_found``, ``forbidden``, ``gone``, ``http_<status>``).
    """
    last_reason = "network_error"
    for attempt in (1, 2):
        try:
            response = client.get(url)
        except httpx.TimeoutException:
            last_reason = "timeout"
        except httpx.HTTPError:
            last_reason = "network_error"
        else:
            if response.status_code == 200:
                return response.content
            last_reason = {
                404: "not_found",
                403: "forbidden",
                410: "gone",
            }.get(response.status_code, f"http_{response.status_code}")
            if response.status_code not in RETRYABLE_STATUS:
                raise ImageDownloadError(last_reason)
        if attempt == 1:
            time.sleep(RETRY_PAUSE_SECONDS)
    raise ImageDownloadError(last_reason)


def make_thumbnail(payload: bytes, max_side: int) -> bytes:
    """Decode, resize, flatten and re-encode one image as a clean JPEG.

    Args:
        payload: Raw downloaded bytes.
        max_side: Longest side of the stored thumbnail, in pixels.

    Returns:
        JPEG bytes with no EXIF/ICC metadata.

    Raises:
        ImageDownloadError: With reason ``not_an_image`` (undecodable payload) or
            ``decode_error`` (decoded but not re-encodable).
    """
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDownloadError("not_an_image") from exc
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.thumbnail((max_side, max_side))
            rgba = opened.convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDownloadError("decode_error") from exc
    # Flatten alpha onto white so a transparent PNG becomes a normal product photo.
    flat = Image.new("RGB", rgba.size, (255, 255, 255))
    flat.paste(rgba, mask=rgba.getchannel("A"))
    buffer = io.BytesIO()
    flat.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write bytes to ``path`` via a temporary file and an atomic rename.

    Args:
        path: Destination file.
        payload: Bytes to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def download_one(
    client: httpx.Client,
    asin: str,
    image_url: str | None,
    destination: Path,
    *,
    max_side: int,
    force: bool,
) -> str:
    """Download, convert and store one product thumbnail.

    Args:
        client: HTTP client.
        asin: Product ASIN (validated by the caller).
        image_url: Stored image URL, or None when the product has no image URL.
        destination: ``<cache dir>/<asin>.jpg``.
        max_side: Longest side of the stored thumbnail, in pixels.
        force: Overwrite an existing file instead of skipping it.

    Returns:
        :data:`OK` when a new file was written, :data:`SKIPPED` when the file
        already existed and ``force`` was not given.

    Raises:
        ImageDownloadError: With the failure reason (``no_image_url``,
        ``invalid_asin``, ``unwritable``, or a fetch/convert reason).
    """
    if not ASIN_PATTERN.match(asin):
        raise ImageDownloadError("invalid_asin")
    if destination.exists() and not force:
        return SKIPPED
    if image_url is None:
        raise ImageDownloadError("no_image_url")
    payload = fetch_image_bytes(client, image_url)
    thumbnail = make_thumbnail(payload, max_side)
    try:
        _write_atomic(destination, thumbnail)
    except OSError as exc:
        raise ImageDownloadError("unwritable") from exc
    return OK


def _mark_cached(engine: Engine, asin: str) -> None:
    """Set ``products.image_cached = true`` for one ASIN.

    Args:
        engine: Serving-cache engine.
        asin: Product ASIN whose thumbnail now exists locally.
    """
    factory = create_session_factory(engine)
    with factory() as session:
        session.execute(update(Product).where(Product.asin == asin).values(image_cached=True))
        session.commit()


# ------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------


def build_report(outcomes: dict[str, str]) -> dict[str, Any]:
    """Summarise per-ASIN outcomes into the JSON report (computed counts only).

    Args:
        outcomes: ``{asin: outcome}`` where the outcome is :data:`OK`,
            :data:`SKIPPED` or a failure reason.

    Returns:
        ``{attempted, ok, skipped, failed: {reason: count}}``.
    """
    failed: dict[str, int] = {}
    ok = 0
    skipped = 0
    for outcome in outcomes.values():
        if outcome == OK:
            ok += 1
        elif outcome == SKIPPED:
            skipped += 1
        else:
            failed[outcome] = failed.get(outcome, 0) + 1
    return {
        "attempted": len(outcomes),
        "ok": ok,
        "skipped": skipped,
        "failed": dict(sorted(failed.items())),
    }


def write_report(reports_dir: Path, report: dict[str, Any]) -> Path:
    """Write the JSON report into the reports directory.

    Args:
        reports_dir: Destination directory (created when missing).
        report: Report payload from :func:`build_report`.

    Returns:
        The written path.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / REPORT_FILENAME
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def _print_summary(report: dict[str, Any], cache_dir: Path) -> None:
    """Print the computed run summary.

    Args:
        report: Report payload from :func:`build_report`.
        cache_dir: Directory the thumbnails were written to.
    """
    print()
    print(f"Thumbnail cache         : {cache_dir}")
    print(f"Products attempted      : {report['attempted']:,}")
    print(f"  downloaded            : {report['ok']:,}")
    print(f"  already cached        : {report['skipped']:,}")
    failed_total = sum(report["failed"].values())
    print(f"  failed                : {failed_total:,}")
    for reason, count in report["failed"].items():
        print(f"      {reason:<18}: {count:,}")


def main(argv: list[str] | None = None) -> int:
    """Download the selected product thumbnails and write the report.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = completed, 1 = could not start).
    """
    args = parse_args(argv)
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        logger.error("config_error path=%s detail=%s", args.config, exc)
        return 1

    images = settings.images
    cache_dir = Path(images.directory)
    delay = images.download_delay_seconds if args.delay is None else args.delay
    max_side = images.thumbnail_max_side if args.size is None else args.size
    if delay < 0.0:
        logger.error("invalid_delay value=%s minimum=0", delay)
        return 1
    if max_side < 1:
        logger.error("invalid_size value=%s minimum=1", max_side)
        return 1

    explicit_selection = bool(args.asins) or bool(args.asin_file)
    if not explicit_selection and args.limit is None and not args.all:
        logger.error(
            "refusing_full_run detail=pass --all to download the whole subset, or "
            "--limit N / --asins / --asin-file"
        )
        return 1

    engine = create_engine_from_settings(settings)
    try:
        targets, unknown = select_targets(engine, args)
    except DataError as exc:
        logger.error("selection_failed detail=%s", exc)
        return 1

    logger.info(
        "download_start products=%d cache=%s max_side=%d delay=%.2fs force=%s",
        len(targets),
        cache_dir,
        max_side,
        delay,
        args.force,
    )
    # A requested ASIN that is not in the serving cache cannot have a stored URL.
    outcomes: dict[str, str] = dict.fromkeys(unknown, NOT_IN_CACHE)
    with build_client(settings) as client:
        for index, (asin, image_url) in enumerate(targets):
            destination = cache_dir / f"{asin}.jpg"
            try:
                outcome = download_one(
                    client,
                    asin,
                    image_url,
                    destination,
                    max_side=max_side,
                    force=args.force,
                )
            except ImageDownloadError as exc:
                outcome = exc.reason
                logger.warning("image_failed asin=%s reason=%s", asin, outcome)
            else:
                if outcome == OK:
                    _mark_cached(engine, asin)
                    logger.info("image_ok asin=%s file=%s", asin, destination.name)
            outcomes[asin] = outcome
            if delay and index < len(targets) - 1:
                time.sleep(delay)

    report = build_report(outcomes)
    report_path = write_report(Path(settings.paths.reports_dir), report)
    _print_summary(report, cache_dir)
    print(f"Report                  : {report_path}")
    logger.info(
        "download_done attempted=%d ok=%d skipped=%d failed=%d",
        report["attempted"],
        report["ok"],
        report["skipped"],
        sum(report["failed"].values()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
