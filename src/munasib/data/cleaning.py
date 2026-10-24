"""Field and text cleaning for the Munasib Khareed preprocessing pipeline.

Pure functions only: every function takes one value and returns a cleaned value -
no I/O, no DataFrame state. The pipeline (``preprocessing.py``) keeps the raw values
alongside the cleaned ones so nothing is lost.

Cleaning rules (project rule: *never fabricate*):

* HTML is stripped with the stdlib :class:`html.parser.HTMLParser`, not a regex, so
  markup hidden inside attribute values cannot leak into the text.
* Review text keeps its natural casing and punctuation - the DistilBERT sentiment
  model expects natural language. Only encoding artifacts, HTML entities, URLs and
  whitespace are touched (never lowercasing, never punctuation stripping).
* Missing or empty input returns ``None`` (never an empty string, never a guess).
* ``normalize_brand`` produces a *comparison* key (lower-cased, de-punctuated at the
  edges); the untouched original stays in the ``brand`` column for display.
* ``parse_rank_fields`` extracts ``(rank_value, rank_category)`` from strings such as
  ``">#1,234 in Appliances (See top 100)"``; unparseable input yields ``None`` for
  the missing part rather than a fabricated number.
* ``parse_category_path`` extracts ``(category_path, category_leaf)`` from the
  ``category`` browse breadcrumb (a list of strings, a single string, or missing);
  empty or absent input yields ``(None, None)``.
"""

from __future__ import annotations

import html
import re
import string
import unicodedata
from html.parser import HTMLParser
from typing import Any

from munasib.data.loaders import is_present

# Tags whose textual content is code/styling, not product text.
_SKIP_TAGS = frozenset({"script", "style", "head", "noscript"})
_BLOCK_TAGS = frozenset({"br", "p", "div", "li", "tr", "ul", "ol", "table", "section"})

_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_REPLACEMENT_CHAR = "\ufffd"
# Real Appliances rank strings (verified against data/raw/meta_Appliances.json.gz):
# "►#43,077 in Appliances (See top 100)"-style prefixed forms, truncated forms
# without "#" like "66,593 in Industrial & Scientific (", and HTML entities
# inside the category ("Tools &amp; Home Improvement > ...").
_RANK_PATTERN = re.compile(
    r"^(?:\u25ba|>)?\s*#?\s*(?P<value>\d[\d,]*)(?:\s+in\s+(?P<category>.+))?$"
)
_RANK_TOP_SUFFIX = re.compile(r"\s*\(see\s+top[^)]*\)\s*$", re.IGNORECASE)
_BRAND_EDGE_PUNCTUATION = string.punctuation + " \t\r\n|"
#: Separator between the levels of the metadata ``category`` breadcrumb. One
#: constant shared by :func:`parse_category_path`, the preprocessing schema and
#: the product-intelligence price peer groups, so the three can never drift.
CATEGORY_PATH_SEPARATOR = " > "


class _HTMLTextExtractor(HTMLParser):
    """Extracts human-readable text from HTML, skipping code/styling tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record block-tag spacing and track skip-tag depth."""
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append(" ")

    def handle_endtag(self, tag: str) -> None:
        """Release skip-tag depth and add spacing after block tags."""
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._chunks.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle self-closing tags (``<br/>``) with a single space, not two."""
        if tag in _SKIP_TAGS:
            return
        if tag in _BLOCK_TAGS:
            self._chunks.append(" ")

    def handle_data(self, data: str) -> None:
        """Append textual data unless inside a skipped tag."""
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        """Return the accumulated text."""
        return "".join(self._chunks)


def strip_html(text: str) -> str:
    """Remove HTML tags from a string, keeping the contained text.

    Uses the stdlib HTML parser so tag-like sequences inside attribute values do
    not corrupt the output. When parsing fails, a conservative regex fallback is
    used rather than dropping the text.

    Args:
        text: Possibly-HTML string.

    Returns:
        Text with tags removed and entities decoded; may be empty.
    """
    if "<" not in text:
        return html.unescape(text)
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(text)
        extractor.close()
        return extractor.text()
    except Exception:  # pragma: no cover - parser is tolerant; defensive fallback
        return re.sub(r"<[^>]+>", " ", text)


def fix_encoding_artifacts(text: str) -> str:
    """Repair common text-encoding artifacts without inventing characters.

    Applies NFC unicode normalisation, decodes stray HTML entities (``&amp;``) and
    removes U+FFFD replacement characters that the streaming loaders insert where
    the source bytes were not valid UTF-8.

    Args:
        text: Raw text.

    Returns:
        Normalised text; unchanged content otherwise (case and punctuation kept).
    """
    normalized = unicodedata.normalize("NFC", text)
    unescaped = html.unescape(normalized)
    return unescaped.replace(_REPLACEMENT_CHAR, "")


def remove_urls(text: str) -> str:
    """Remove ``http(s)://`` and ``www.`` URLs from text.

    Args:
        text: Text possibly containing URLs.

    Returns:
        Text with URL tokens removed (spacing is fixed by the caller).
    """
    return _URL_PATTERN.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including newlines/tabs) into single spaces.

    Args:
        text: Text possibly containing repeated whitespace.

    Returns:
        Trimmed text with single-space separators.
    """
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_review_text(value: Any) -> str | None:
    """Clean one review/summary text for model input.

    Encoding artifacts are fixed, HTML entities decoded, URLs removed and
    whitespace collapsed. Casing and punctuation are deliberately preserved
    (DistilBERT consumes natural language).

    Args:
        value: Raw ``reviewText``/``summary`` value (str or missing).

    Returns:
        Cleaned text, or None when the input is absent or empties out to nothing.
    """
    if not is_present(value):
        return None
    cleaned = collapse_whitespace(remove_urls(fix_encoding_artifacts(str(value))))
    return cleaned or None


def clean_title(value: Any) -> str | None:
    """Clean a product title (HTML stripped, whitespace collapsed).

    Args:
        value: Raw title value (str or missing).

    Returns:
        Cleaned title, or None when absent/empty.
    """
    if not is_present(value):
        return None
    cleaned = collapse_whitespace(strip_html(fix_encoding_artifacts(str(value))))
    return cleaned or None


def clean_description(value: Any) -> str | None:
    """Clean a product description, joining list-of-paragraph values.

    Args:
        value: Raw description: list of strings (2018 format), a plain string, or
            missing.

    Returns:
        Cleaned description with paragraphs joined by a space, or None when
        absent/empty.
    """
    if not is_present(value):
        return None
    parts = value if isinstance(value, (list, tuple)) else [value]
    cleaned_parts = [
        cleaned
        for item in parts
        if (cleaned := collapse_whitespace(strip_html(fix_encoding_artifacts(str(item)))))
    ]
    return " ".join(cleaned_parts) or None


def clean_features(value: Any) -> list[str] | None:
    """Clean product feature bullets, keeping them as separate strings.

    Args:
        value: Raw feature list (2018 format), a plain string, or missing.

    Returns:
        List of cleaned feature strings, or None when absent/empty.
    """
    if not is_present(value):
        return None
    parts = value if isinstance(value, (list, tuple)) else [value]
    cleaned = [
        cleaned
        for item in parts
        if (cleaned := collapse_whitespace(strip_html(fix_encoding_artifacts(str(item)))))
    ]
    return cleaned or None


def normalize_brand(value: Any) -> str | None:
    """Normalise a brand into a comparison key.

    Trims, collapses internal whitespace, removes stray punctuation at the edges
    (e.g. ``" Acme, "`` -> ``"acme"``) and lower-cases so the same brand written in
    different casings groups together. The untouched original stays in the raw
    ``brand`` column for display.

    Args:
        value: Raw brand value (str or missing).

    Returns:
        Normalised brand key, or None when the value carries no brand.
    """
    if not is_present(value):
        return None
    trimmed = collapse_whitespace(str(value)).strip(_BRAND_EDGE_PUNCTUATION)
    normalized = collapse_whitespace(trimmed).lower()
    return normalized or None


def parse_rank_fields(value: Any) -> tuple[int | None, str | None]:
    """Parse a sales-rank string into ``(rank_value, rank_category)``.

    Accepted forms: ``">#43,077 in Appliances (See top 100)"`` (case of the
    ``" (See Top 100 ...)"`` suffix varies), ``"#489 in Tools & Home Improvement >
    Appliances > Burner Rings"``, truncated real-world strings without ``"#"``
    (``"66,593 in Industrial & Scientific ("``) and plain numeric strings such as
    ``"1,234"`` (legacy ``salesRank`` values). HTML entities in the category are
    decoded and the ``" (See top ...)"`` suffix is stripped.

    Args:
        value: Rank string from the loader's canonical ``rank`` column (or missing).

    Returns:
        Tuple ``(rank_value, rank_category)``. Either part is None when it cannot be
        derived from the string; both are None when the input is missing or
        unparseable (never a fabricated rank).
    """
    if not is_present(value):
        return None, None
    text = collapse_whitespace(str(value))

    match = _RANK_PATTERN.match(text)
    if match is None:
        return None, None

    try:
        rank_value = int(match.group("value").replace(",", ""))
    except ValueError:
        return None, None

    category = match.group("category")
    rank_category: str | None = None
    if category:
        cleaned = _RANK_TOP_SUFFIX.sub("", category)
        cleaned = html.unescape(cleaned).strip().rstrip(" (").strip()
        rank_category = collapse_whitespace(cleaned) or None
    return rank_value, rank_category


def parse_category_path(value: Any) -> tuple[str | None, str | None]:
    """Parse a metadata ``category`` breadcrumb into ``(category_path, leaf)``.

    The 2018 metadata stores the browse path as a list of node names ordered
    broad -> specific (``["Appliances", "Parts & Accessories"]``), with a single
    string and an absent field as the other two shapes a loader must tolerate. A
    scalar string is split on ``" > "`` exactly like a list item, so both shapes
    end up as the same level structure.

    Every level is HTML-unescaped and whitespace-collapsed before joining. That
    is not cosmetic: the source really mixes ``"Parts &amp; Accessories"`` with
    ``"Parts & Accessories"`` (both spellings occur in
    ``data/raw/meta_Appliances.json.gz``), and the two must group under the same
    peer group instead of silently splitting one browse node in two. Blank
    levels are dropped.

    Args:
        value: Raw ``category`` value: list of strings, single string, or missing.

    Returns:
        Tuple ``(category_path, category_leaf)`` - the levels joined by
        ``CATEGORY_PATH_SEPARATOR`` and the most specific level. ``(None, None)``
        when the field is absent, empty or holds nothing usable; never an empty
        string.
    """
    if not is_present(value):
        return None, None
    raw_levels = list(value) if isinstance(value, (list, tuple)) else [value]
    levels: list[str] = []
    for item in raw_levels:
        if not is_present(item):
            continue
        cleaned = collapse_whitespace(fix_encoding_artifacts(str(item)))
        for part in cleaned.split(CATEGORY_PATH_SEPARATOR):
            part = part.strip()
            if part:
                levels.append(part)
    if not levels:
        return None, None
    return CATEGORY_PATH_SEPARATOR.join(levels), levels[-1]
