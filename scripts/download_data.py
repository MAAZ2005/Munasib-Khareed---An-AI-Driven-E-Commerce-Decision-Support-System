"""Resumable downloader for the UCSD Amazon Review Data (2018) per-category files.

Downloads the FULL per-category review file and the per-category metadata file for one
category (default taken from ``dataset.category`` in configs/config.yaml). URLs are read
from ``dataset.sources`` in the same config file, so switching category is a one-line
config change followed by re-running this script.

Features:
  * Resumable: partial downloads are stored as ``<file>.part`` and continued with HTTP
    ``Range`` requests on the next run.
  * Verified: file size is checked against the remote ``Content-Length`` and the gzip
    stream is decompressed end-to-end (CRC check) before the file is accepted.
  * Skippable: an already complete and intact file is not downloaded again (use
    ``--force`` to override).
  * Honest on failure: if the host is unreachable the script prints the exact URLs and
    target paths for manual download instead of fabricating data.

Exit codes:
  0 - all requested files are present and verified (downloaded, resumed, or skipped).
  1 - at least one file could not be fetched or verified; manual instructions printed.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import httpx

from munasib.core.config import Settings, load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("munasib.download_data")

PROGRESS_EVERY_BYTES = 8 * 1024 * 1024  # log a progress line every 8 MB
CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming chunks
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=15.0)


class ManualDownloadRequired(Exception):
    """Raised when a file cannot be fetched automatically and needs a manual download."""


class VerificationFailed(ManualDownloadRequired):
    """Raised when downloaded bytes exist but fail size/gzip verification (retryable)."""


@dataclass(frozen=True)
class DownloadPlan:
    """One remote file and its local destination."""

    url: str
    target: Path
    label: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download the UCSD Amazon Review Data (2018) full per-category review file "
            "and metadata file into data/raw/ (resumable, gzip-verified)."
        )
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Category name registered under dataset.sources in the config "
        "(default: dataset.category from the config file).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when a complete, verified file already exists.",
    )
    return parser.parse_args(argv)


def build_plan(settings: Settings, category: str, raw_dir: Path) -> list[DownloadPlan]:
    """Build the download plan (reviews + metadata) for a category.

    Args:
        settings: Loaded application settings.
        category: Category key inside ``dataset.sources``.
        raw_dir: Directory that receives the raw files.

    Returns:
        List of :class:`DownloadPlan` entries.

    Raises:
        ManualDownloadRequired: If the category has no registered sources.
    """
    source = settings.dataset.sources.get(category)
    if source is None:
        available = ", ".join(sorted(settings.dataset.sources)) or "<none>"
        raise ManualDownloadRequired(
            f"No sources registered for category '{category}'. "
            f"Available categories in dataset.sources: {available}. "
            "Add the category to configs/config.yaml or pass --category."
        )
    return [
        DownloadPlan(
            url=source.reviews_url,
            target=raw_dir / Path(source.reviews_url).name,
            label="reviews",
        ),
        DownloadPlan(
            url=source.metadata_url,
            target=raw_dir / Path(source.metadata_url).name,
            label="metadata",
        ),
    ]


def print_manual_instructions(plan: DownloadPlan, reason: str) -> None:
    """Print exact URL and target path so the file can be fetched by hand.

    Args:
        plan: The file that failed.
        reason: Why automatic download failed.
    """
    logger.warning("MANUAL DOWNLOAD REQUIRED (%s): %s", reason, plan.label)
    print("  URL   : " + plan.url)
    print("  TARGET: " + str(plan.target))
    print(f'  CMD   : curl -L --retry 3 -o "{plan.target}" "{plan.url}"')


def verify_gzip_integrity(path: Path) -> tuple[bool, str]:
    """Decompress a gzip file end-to-end to validate its CRC and trailer.

    Args:
        path: Path to the ``.gz`` file.

    Returns:
        Tuple ``(ok, detail)`` where ``detail`` is the decompressed byte count on
        success or the error message on failure.
    """
    total = 0
    try:
        with gzip.open(path, "rb") as handle:
            while True:
                chunk = handle.read(CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
    except (EOFError, OSError, zlib.error) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"{total:,} decompressed bytes"


def part_path_for(target: Path) -> Path:
    """Return the temporary ``.part`` path used while a download is in flight.

    Args:
        target: Final target file path.

    Returns:
        Path with ``.part`` appended (suffix-safe for ``.json.gz`` names).
    """
    return Path(str(target) + ".part")


def remote_size(client: httpx.Client, url: str) -> int | None:
    """Fetch the remote Content-Length via a HEAD request.

    Args:
        client: HTTP client.
        url: File URL.

    Returns:
        Remote size in bytes, or None when the server does not report it.

    Raises:
        ManualDownloadRequired: When the host cannot be reached at all.
    """
    try:
        response = client.head(url, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise ManualDownloadRequired(f"HEAD request failed: {type(exc).__name__}: {exc}") from exc
    if response.status_code >= 400:
        raise ManualDownloadRequired(f"HEAD returned HTTP {response.status_code}")
    length = response.headers.get("content-length")
    return int(length) if length and length.isdigit() else None


def _log_progress(done: int, total: int | None, next_report: int) -> int:
    """Log download progress and compute the next reporting threshold.

    Args:
        done: Bytes downloaded so far (including a resumed prefix).
        total: Remote total size, when known.
        next_report: Byte count at which the next log line is due.

    Returns:
        Updated next reporting threshold.
    """
    if done >= next_report:
        if total:
            pct = 100.0 * done / total
            logger.info("  progress: %s / %s bytes (%.1f%%)", f"{done:,}", f"{total:,}", pct)
        else:
            logger.info("  progress: %s bytes", f"{done:,}")
        next_report = done + PROGRESS_EVERY_BYTES
    return next_report


def _stream_to_part(client: httpx.Client, plan: DownloadPlan, resume_from: int, total: int | None) -> None:
    """Stream the remote file into the ``.part`` file, optionally resuming.

    Args:
        client: HTTP client.
        plan: Download plan describing URL and target.
        resume_from: Byte offset to resume from (0 for a fresh download).
        total: Remote total size, when known.

    Raises:
        ManualDownloadRequired: On connection/read failures (``.part`` is kept for resume).
    """
    headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else {}
    try:
        with client.stream(
            "GET", plan.url, headers=headers, follow_redirects=True
        ) as response:
            if response.status_code == 416:
                # Range not satisfiable: either the .part is already complete, or stale.
                return
            if resume_from > 0 and response.status_code != 206:
                logger.info("  server did not honour Range (HTTP %d); restarting from 0", response.status_code)
                resume_from = 0
            response.raise_for_status()

            mode = "ab" if (resume_from > 0 and response.status_code == 206) else "wb"
            done = resume_from
            next_report = (done // PROGRESS_EVERY_BYTES + 1) * PROGRESS_EVERY_BYTES
            with open(part_path_for(plan.target), mode) as handle:
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    if chunk:
                        handle.write(chunk)
                        done += len(chunk)
                        next_report = _log_progress(done, total, next_report)
            if total is not None:
                logger.info("  downloaded %s / %s bytes", f"{done:,}", f"{total:,}")
            else:
                logger.info("  downloaded %s bytes", f"{done:,}")
    except httpx.HTTPError as exc:
        raise ManualDownloadRequired(
            f"streaming failed after resume offset {resume_from}: {type(exc).__name__}: {exc}"
        ) from exc


def download_file(client: httpx.Client, plan: DownloadPlan, force: bool) -> str:
    """Ensure a plan's target file exists locally, complete and gzip-verified.

    Resumes from the ``.part`` file when present; if the resumed attempt fails
    verification, one clean restart from byte 0 is attempted automatically.

    Args:
        client: HTTP client.
        plan: File to fetch.
        force: Re-download even if an intact file exists.

    Returns:
        Outcome string: ``'skipped'``, ``'downloaded'``, ``'resumed'`` or ``'failed'``.

    Raises:
        ManualDownloadRequired: When automatic fetching/verification is impossible.
    """
    part_path = part_path_for(plan.target)
    expected = remote_size(client, plan.url)

    # --- Already complete? -------------------------------------------------
    if plan.target.exists() and not force:
        current = plan.target.stat().st_size
        if expected is None or current == expected:
            ok, detail = verify_gzip_integrity(plan.target)
            if ok:
                logger.info(
                    "SKIP (already complete): %s [%s, %s]",
                    plan.target.name,
                    f"{current:,} bytes",
                    detail,
                )
                return "skipped"
            logger.warning("existing file failed gzip check (%s); re-downloading", detail)
        else:
            logger.warning(
                "existing file size %s != remote %s; re-downloading",
                f"{current:,}",
                f"{expected:,}",
            )
        plan.target.unlink(missing_ok=True)

    # --- Choose resume offset, then stream (one clean retry on failure) -----
    resume_from = part_path.stat().st_size if part_path.exists() else 0
    if expected is not None and resume_from > expected:
        logger.warning("stale .part larger than remote file; discarding partial data")
        part_path.unlink(missing_ok=True)
        resume_from = 0

    last_error: ManualDownloadRequired | None = None
    for attempt in range(2):
        if attempt == 1:
            if resume_from == 0:
                break  # nothing to restart from; propagate the original failure
            logger.warning(
                "resumed attempt for %s failed verification; restarting from byte 0",
                part_path.name,
            )
            part_path.unlink(missing_ok=True)
            resume_from = 0
        if resume_from:
            logger.info("resuming %s from %s", part_path.name, f"{resume_from:,} bytes")
        else:
            logger.info("downloading %s -> %s", plan.url, plan.target)
        try:
            _stream_to_part(client, plan, resume_from, expected)
            if not part_path.exists():
                raise VerificationFailed("no data was written (server returned no body)")
            actual = part_path.stat().st_size
            if expected is not None and actual != expected:
                raise VerificationFailed(
                    f"size mismatch after download: got {actual:,} bytes, expected {expected:,}"
                )
            ok, detail = verify_gzip_integrity(part_path)
            if not ok:
                raise VerificationFailed(f"gzip integrity check failed: {detail}")
        except VerificationFailed as exc:
            last_error = exc
            continue  # verification failures get one clean restart from byte 0
        except ManualDownloadRequired as exc:
            last_error = exc
            break  # connection problem; retrying immediately would not help

        part_path.replace(plan.target)
        logger.info("OK: %s [%s bytes, %s]", plan.target.name, f"{actual:,}", detail)
        return "resumed" if resume_from else "downloaded"

    if last_error is None:
        raise ManualDownloadRequired("download failed: no attempt produced data")
    raise last_error


def main(argv: list[str] | None = None) -> int:
    """Run the download plan for the selected category.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = all files present and verified, 1 = manual action needed).
    """
    args = parse_args(argv)
    settings = load_settings(args.config)
    category = args.category or settings.dataset.category
    raw_dir = settings.paths.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        plan_list = build_plan(settings, category, raw_dir)
    except ManualDownloadRequired as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Category: %s (active in config: %s)", category, settings.dataset.category)
    failures = 0
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        for plan in plan_list:
            try:
                download_file(client, plan, force=args.force)
            except ManualDownloadRequired as exc:
                failures += 1
                print_manual_instructions(plan, str(exc))

    if failures:
        logger.error(
            "%d file(s) could not be fetched automatically. Download them manually "
            "with the commands above (files must land at the printed TARGET paths), "
            "then re-run this script to verify.",
            failures,
        )
        return 1
    logger.info("All files for category '%s' are present and verified.", category)
    return 0


if __name__ == "__main__":
    sys.exit(main())
