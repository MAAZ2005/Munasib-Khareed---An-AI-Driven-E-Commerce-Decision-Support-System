"""Friendly, actionable messages for failed API calls.

A dashboard page never lets an :class:`~app.api_client.ApiClientError` reach
Streamlit's exception handler: it calls :func:`render_api_error`, which shows what
went wrong and how to recover. Wording lives here (one place) so the e2e tests can
assert on it and every page stays consistent.

Rules:
    * A missing product (:class:`~app.api_client.ApiNotFound`) clears the stale
      selection and links back to Product Selection, so the guard on pages 2-7
      cannot keep pointing at a product that is no longer in the cache.
    * An unreachable API (:class:`~app.api_client.ApiUnavailable`) explains how to
      start the backend with ``uvicorn munasib.api.main:app --port 8000``.
    * Any other error shows the request id, which ties the on-screen message to a
      backend log line.
"""

from __future__ import annotations

from app.api_client import ApiClientError, ApiNotFound, ApiUnavailable

#: Headline shown when the requested product is not in the serving cache.
NOT_FOUND_MESSAGE = "Product not found in the serving cache."

#: Headline shown when the API cannot be reached at all.
UNAVAILABLE_MESSAGE = "The analysis API is not reachable."

#: Recovery instructions for an unreachable API (offline wording, no network use).
UNAVAILABLE_HELP = (
    "Start the backend from the repository root with the project virtual "
    "environment active:\n\n"
    "```powershell\n"
    "uvicorn munasib.api.main:app --port 8000\n"
    "```\n\n"
    "Then reload this page. `make run-api` does the same."
)

#: Headline for any other error the API returned.
ERROR_MESSAGE = "The API returned an error."


def describe_api_error(exc: Exception) -> str:
    """Return the one-line headline for a failed API call.

    Pure and side-effect free, so the mapping from exception to message is
    unit-tested directly without running a page.

    Args:
        exc: The exception a page caught around an API call.

    Returns:
        One of :data:`NOT_FOUND_MESSAGE`, :data:`UNAVAILABLE_MESSAGE` or
        :data:`ERROR_MESSAGE`.
    """
    if isinstance(exc, ApiNotFound):
        return NOT_FOUND_MESSAGE
    if isinstance(exc, ApiUnavailable):
        return UNAVAILABLE_MESSAGE
    return ERROR_MESSAGE


def _render_selection_link() -> None:
    """Render a page link back to Product Selection (best effort, never raises)."""
    import streamlit as st

    try:
        st.page_link(
            "pages/1_Product_Selection.py",
            label="Go to Product Selection",
            icon="🔍",
        )
    except Exception:
        try:
            st.page_link("1_Product_Selection.py", label="Go to Product Selection", icon="🔍")
        except Exception:
            pass


def render_api_error(exc: Exception) -> None:
    """Render a friendly, actionable message for a failed API call.

    Never re-raises: a page calls this, then returns, so one failed request shows
    an explanation instead of a traceback.

    Args:
        exc: The exception a page caught around an API call.
    """
    import streamlit as st

    if isinstance(exc, ApiNotFound):
        # Drop the stale selection so pages 2-7 fall back to the guard and the
        # user is sent back to the catalogue rather than a dead product.
        from app.state import set_selected_asin

        set_selected_asin(None)
        st.error(f"⚠️ {NOT_FOUND_MESSAGE}")
        st.caption(str(exc))
        _render_selection_link()
        return

    if isinstance(exc, ApiUnavailable):
        st.error(f"⚠️ {UNAVAILABLE_MESSAGE}")
        st.info(UNAVAILABLE_HELP)
        return

    if isinstance(exc, ApiClientError):
        st.error(f"⚠️ {ERROR_MESSAGE} {exc.message}")
        request_id = getattr(exc, "request_id", None)
        if request_id:
            st.caption(f"request_id: {request_id}")
        return

    st.error(f"⚠️ Unexpected error while contacting the API: {exc}")


def describe_error_detail(exc: Exception) -> str:
    """Return the secondary line for a failed API call.

    Args:
        exc: The exception a page caught.

    Returns:
        The exception's message (its ``str``), used as a caption under the
        headline. Kept separate so a test can check the text without rendering.
    """
    return str(exc) if isinstance(exc, ApiClientError) else f"{type(exc).__name__}: {exc}"


__all__ = [
    "ERROR_MESSAGE",
    "NOT_FOUND_MESSAGE",
    "UNAVAILABLE_HELP",
    "UNAVAILABLE_MESSAGE",
    "describe_api_error",
    "describe_error_detail",
    "render_api_error",
]
