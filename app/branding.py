"""Brand assets for the dashboard: the local Munasib Khareed logo.

The dashboard must work **offline**, so the logo is a committed PNG
(``app/assets/logo.png``, a copy of ``docs/logo.png``) read from disk - it is never
fetched from a CDN or a web font. Everything here is a small, pure, defensive
helper: a missing or unreadable file returns ``None`` so the header can fall back
to the application name as text, and no code path ever raises for a bad asset.
"""

from __future__ import annotations

from pathlib import Path

#: Application name, used as the text fallback when no logo file is available.
APP_NAME = "Munasib Khareed"

#: Offline one-line description shown under the wordmark fallback.
APP_TAGLINE = "AI-Driven Consumer Decision Support"

#: Directory holding committed local brand assets.
ASSET_DIR = Path(__file__).resolve().parent / "assets"

#: Committed logo filename (the wordmark copy of ``docs/logo.png``).
LOGO_FILENAME = "logo.png"


def logo_path() -> Path:
    """Return the path of the committed logo asset.

    Returns:
        The absolute path of ``app/assets/logo.png`` (which may not exist).
    """
    return ASSET_DIR / LOGO_FILENAME


def load_logo_bytes(path: Path | None = None) -> bytes | None:
    """Read the logo bytes from disk, returning ``None`` on any failure.

    Pure and side-effect free apart from the file read, so a page can prove the
    "missing asset degrades to text, never crashes" rule without Streamlit.

    Args:
        path: Logo file to read, or None to use :func:`logo_path`.

    Returns:
        The non-empty file contents, or ``None`` when the file is missing, is a
        directory, is empty, or cannot be read.
    """
    target = path if path is not None else logo_path()
    try:
        data = target.read_bytes()
    except OSError:
        return None
    return data or None


def logo_is_available(path: Path | None = None) -> bool:
    """Report whether a usable logo file is present.

    Args:
        path: Logo file to check, or None to use :func:`logo_path`.

    Returns:
        True when :func:`load_logo_bytes` returns non-empty bytes.
    """
    return load_logo_bytes(path) is not None


__all__ = [
    "APP_NAME",
    "APP_TAGLINE",
    "ASSET_DIR",
    "LOGO_FILENAME",
    "load_logo_bytes",
    "logo_is_available",
    "logo_path",
]
