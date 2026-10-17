"""Product image component: the cached thumbnail, or a neutral local placeholder.

Images are decoration only - nothing in the analysis looks at one - and the
dashboard works offline, so this component renders either:

* the JPEG bytes the API serves from its local cache (``GET /products/{asin}/image``)
  via :func:`streamlit.image`, or
* a neutral gray tile drawn with inline HTML (no external resource, no broken-image
  icon) carrying the product's first letter and the text ``No image available``.

It requests the image only when the product says ``has_image`` is true, and it
treats a failed fetch like a missing image rather than raising. The Amazon/seller
attribution from :data:`app.theme.IMAGE_ATTRIBUTION` is shown as a caption
underneath every time the component runs, so the source is always visible.
"""

from __future__ import annotations

from html import escape

from app.theme import IMAGE_ATTRIBUTION

#: Text shown on the placeholder tile.
NO_IMAGE_LABEL = "No image available"

#: Placeholder tile fill and border (neutral, offline-safe inline styles).
_PLACEHOLDER_BACKGROUND = "#f1f5f9"
_PLACEHOLDER_BORDER = "#94a3b8"
_PLACEHOLDER_TEXT = "#475569"

#: Default tile/thumbnail edge in pixels.
DEFAULT_WIDTH = 160


def has_renderable_image(has_image: bool, data: bytes | None) -> bool:
    """Decide whether the component draws an image or the placeholder.

    Pure, so the rule (a product claims an image *and* the bytes arrived) is
    unit-tested without Streamlit.

    Args:
        has_image: The product's ``has_image`` flag from the API.
        data: The fetched JPEG bytes, or None when the fetch returned 404/failed.

    Returns:
        True only when the product has an image and the bytes are non-empty.
    """
    return bool(has_image) and bool(data)


def placeholder_initial(title: str | None, asin: str) -> str:
    """Return the letter shown on the placeholder tile.

    Uses the first alphanumeric character of the title, falling back to the ASIN,
    so the tile still carries a hint of which product it stands for.

    Args:
        title: Product title, or None when the source had none.
        asin: Product ASIN (always present).

    Returns:
        A single upper-cased character, or ``"?"`` when neither has one.
    """
    for candidate in (title, asin):
        for char in candidate or "":
            if char.isalnum():
                return char.upper()
    return "?"


def _placeholder_html(initial: str, width: int) -> str:
    """Build the inline-HTML placeholder tile (no external resources).

    Args:
        initial: The single character to show on the tile.
        width: Tile edge length in pixels.

    Returns:
        An HTML fragment safe to render with ``unsafe_allow_html``.
    """
    font_size = max(18, width // 3)
    return (
        f'<div style="width:{width}px;">'
        f'<div style="width:{width}px;height:{width}px;display:flex;'
        f"align-items:center;justify-content:center;"
        f"background-color:{_PLACEHOLDER_BACKGROUND};"
        f"border:1px dashed {_PLACEHOLDER_BORDER};border-radius:8px;"
        f"color:{_PLACEHOLDER_TEXT};font-size:{font_size}px;font-weight:700;"
        f'line-height:1;">{escape(initial)}</div>'
        f'<div style="width:{width}px;text-align:center;'
        f"color:{_PLACEHOLDER_TEXT};font-size:0.72rem;margin-top:4px;"
        f'line-height:1.2;">{escape(NO_IMAGE_LABEL)}</div>'
        f"</div>"
    )


def product_image(
    asin: str,
    has_image: bool,
    width: int = DEFAULT_WIDTH,
    title: str | None = None,
    show_attribution: bool = True,
) -> None:
    """Render a product's cached thumbnail or the neutral placeholder.

    The image is requested only when ``has_image`` is true; a fetch that returns
    ``None`` (no cached thumbnail) or raises (backend down) still renders the
    placeholder, never a broken image and never an exception. A caption with the
    required attribution is shown underneath in every case unless
    ``show_attribution`` is false (grid pages print it once, under the grid).

    Args:
        asin: Product ASIN.
        has_image: ``has_image`` from the product summary/detail.
        width: Rendered edge length in pixels.
        title: Product title, used only for the placeholder's initial letter.
        show_attribution: Whether to print the image attribution caption here.
    """
    import streamlit as st

    from app.cache import get_product_image

    data: bytes | None = None
    if has_image:
        try:
            data = get_product_image(asin)
        except Exception:
            # Backend unreachable or an unexpected error: fall back to the tile.
            data = None

    if has_renderable_image(has_image, data) and data is not None:
        st.image(data, width=width, caption=title, alt=title or asin)
    else:
        st.markdown(_placeholder_html(placeholder_initial(title, asin), width), unsafe_allow_html=True)

    if show_attribution:
        st.caption(IMAGE_ATTRIBUTION)


__all__ = [
    "DEFAULT_WIDTH",
    "NO_IMAGE_LABEL",
    "IMAGE_ATTRIBUTION",
    "has_renderable_image",
    "placeholder_initial",
    "product_image",
]
