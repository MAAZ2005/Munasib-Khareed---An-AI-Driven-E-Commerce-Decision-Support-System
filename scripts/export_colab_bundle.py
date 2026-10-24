"""Build the zip bundle that trains the sentiment models on Google Colab.

The bundle is deliberately an **allowlist**, not a directory copy: only the code,
the one config file the training/evaluation path actually loads, the project
metadata, the Colab requirements and a slim review table travel. That keeps the
archive small and makes it structurally impossible to ship ``.env``, ``models/``,
the SQLite cache or ``data/raw/`` (a denylist would silently leak anything new
someone drops in the tree).

The review table is rewritten with only the columns the trainers read
(``review_id``, ``asin``, ``split``, ``review_text_clean``,
``sentiment_label_weak``), which cuts the 113 MB processed parquet to roughly
50 MB before compression.

Usage::

    python scripts/export_colab_bundle.py
    python scripts/export_colab_bundle.py --output data/interim/custom.zip

Exit codes: 0 - the zip was written; 1 - the inputs are missing or unusable.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from munasib.core.exceptions import DataError

#: Default bundle location (``data/interim/*`` is git-ignored).
DEFAULT_OUTPUT = "data/interim/colab_bundle.zip"

#: Directories copied recursively, restricted to Python sources.
SOURCE_DIRS: tuple[str, ...] = ("src", "scripts")

#: The only config file the training and evaluation code loads
#: (``load_settings`` reads one path; ``configs/config.yaml`` is that path).
CONFIG_FILES: tuple[str, ...] = ("configs/config.yaml",)

#: Documentation that travels with the bundle so the Colab run is self-explaining.
DOC_FILES: tuple[str, ...] = ("docs/sentiment.md", "docs/training_on_colab.md")

#: Repository-level metadata installed inside the Colab session.
ROOT_FILES: tuple[str, ...] = ("pyproject.toml", "requirements-colab.txt")

#: Processed review table, rewritten slim before zipping.
REVIEWS_ARCNAME = "data/processed/reviews.parquet"

#: Columns the sentiment training and evaluation code reads.
REVIEW_COLUMNS: tuple[str, ...] = (
    "review_id",
    "asin",
    "split",
    "review_text_clean",
    "sentiment_label_weak",
)

#: Path fragments that must never appear in the archive. The allowlist above
#: already excludes them; this guard makes a future widening of the allowlist
#: fail loudly instead of shipping a secret or a 300 MB cache.
FORBIDDEN_FRAGMENTS: tuple[str, ...] = (
    ".env",
    ".git/",
    ".venv/",
    "__pycache__",
    "data/raw/",
    "data/db/",
    "models/",
    ".sqlite",
    ".joblib",
)


@dataclass(frozen=True)
class BundleFile:
    """One archive member.

    Attributes:
        arcname: Path inside the zip (forward slashes).
        size: Uncompressed size in bytes.
    """

    arcname: str
    size: int


@dataclass(frozen=True)
class BundleReport:
    """Result of one bundle build.

    Attributes:
        path: Written zip path.
        size_bytes: Compressed size on disk.
        uncompressed_bytes: Sum of the member sizes.
        files: Members, sorted by archive name.
    """

    path: Path
    size_bytes: int
    uncompressed_bytes: int
    files: tuple[BundleFile, ...]

    def lines(self) -> list[str]:
        """Render the size summary and the file list.

        Returns:
            Human-readable lines (sizes in MB and bytes).
        """
        lines = [
            f"bundle: {self.path}",
            f"compressed: {self.size_bytes / 1_048_576:.1f} MB ({self.size_bytes:,} bytes)",
            f"uncompressed: {self.uncompressed_bytes / 1_048_576:.1f} MB",
            f"files: {len(self.files)}",
        ]
        for item in self.files:
            lines.append(f"  {item.arcname}  ({item.size / 1024:.1f} KB)")
        return lines


def _is_forbidden(arcname: str) -> bool:
    """Check an archive path against the forbidden fragments.

    Args:
        arcname: Path inside the zip.

    Returns:
        ``True`` when the path must not be shipped.
    """
    lowered = arcname.replace("\\", "/").lower()
    return any(fragment in lowered for fragment in FORBIDDEN_FRAGMENTS)


def collect_files(repo_root: Path) -> list[Path]:
    """List the repository files the bundle should contain (minus the data).

    Args:
        repo_root: Repository root.

    Returns:
        Sorted relative paths.

    Raises:
        DataError: If an allowlisted file is missing or a path is forbidden.
    """
    selected: list[Path] = []
    for directory in SOURCE_DIRS:
        base = repo_root / directory
        if not base.is_dir():
            raise DataError(f"source directory not found: {base}")
        selected.extend(
            path.relative_to(repo_root)
            for path in sorted(base.rglob("*.py"))
            if "__pycache__" not in path.parts
        )
    for name in (*CONFIG_FILES, *DOC_FILES, *ROOT_FILES):
        path = repo_root / name
        if not path.is_file():
            raise DataError(f"bundle input not found: {path}")
        selected.append(Path(name))

    for relative in selected:
        arcname = relative.as_posix()
        if _is_forbidden(arcname):
            raise DataError(f"refusing to bundle forbidden path: {arcname}")
    return selected


def load_slim_reviews(repo_root: Path) -> pd.DataFrame:
    """Read the processed reviews and keep only the columns training needs.

    Args:
        repo_root: Repository root.

    Returns:
        The slim frame, in the original row order.

    Raises:
        DataError: If the parquet is missing, empty or lacks a required column.
    """
    path = repo_root / REVIEWS_ARCNAME
    if not path.is_file():
        raise DataError(f"review table not found: {path}; run scripts/build_dataset.py first")
    frame = pd.read_parquet(path, columns=list(REVIEW_COLUMNS))
    if frame.empty:
        raise DataError(f"review table is empty: {path}")
    return frame


def _write_zip(
    output: Path,
    repo_root: Path,
    relative_files: Sequence[Path],
    reviews: pd.DataFrame,
) -> BundleReport:
    """Write the archive and measure it.

    Args:
        output: Destination zip path (parents created).
        repo_root: Repository root.
        relative_files: Files to add from the repository.
        reviews: Slim review frame to serialise as ``data/processed/reviews.parquet``.

    Returns:
        The build report.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    members: list[BundleFile] = []
    with tempfile.TemporaryDirectory(prefix="munasib-bundle-") as scratch:
        slim_path = Path(scratch) / "reviews.parquet"
        reviews.to_parquet(slim_path, index=False, compression="snappy")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for relative in relative_files:
                arcname = relative.as_posix()
                archive.write(repo_root / relative, arcname=arcname)
                members.append(BundleFile(arcname, (repo_root / relative).stat().st_size))
            archive.write(slim_path, arcname=REVIEWS_ARCNAME)
            members.append(BundleFile(REVIEWS_ARCNAME, slim_path.stat().st_size))
    members.sort(key=lambda item: item.arcname)
    return BundleReport(
        path=output,
        size_bytes=output.stat().st_size,
        uncompressed_bytes=sum(item.size for item in members),
        files=tuple(members),
    )


def build_bundle(repo_root: Path, output: Path) -> BundleReport:
    """Build the Colab bundle from a repository checkout.

    Args:
        repo_root: Repository root containing ``src/``, ``configs/`` and the
            processed review table.
        output: Destination zip path.

    Returns:
        The build report.

    Raises:
        DataError: If a required input is missing or a forbidden path is
            selected.
    """
    files = collect_files(repo_root)
    reviews = load_slim_reviews(repo_root)
    return _write_zip(output, repo_root, files, reviews)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Write the zip bundle used to train the sentiment models on Colab."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Destination zip (default: {DEFAULT_OUTPUT}, git-ignored).",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root to export (default: the checkout this script lives in).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the bundle and print its size and contents.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = written, 1 = failed).
    """
    args = parse_args(argv)
    try:
        report = build_bundle(Path(args.repo_root), Path(args.output))
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for line in report.lines():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
