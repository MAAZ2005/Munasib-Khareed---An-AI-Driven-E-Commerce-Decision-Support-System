"""About page: the project, the team and the data behind Munasib Khareed.

All of the text on this page comes from ``configs/about.yaml``, validated by
:class:`app.about.AboutConfig`, so the content can be edited without touching code.
If the file is missing or invalid the page shows a friendly explanation instead of
crashing.

The project deliberately publishes **no** contact details: the page (and the whole
dashboard) contains no email address or phone number.
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

# Ensure repository root is on sys.path so 'app.*' imports resolve
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from app.about import AboutConfigError, format_pipeline, load_about_config  # noqa: E402
from app.components import render_footer, render_header  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from app.theme import apply_theme  # noqa: E402

#: Shown when ``configs/about.yaml`` cannot be read or fails validation.
CONFIG_ERROR_MESSAGE = (
    "The About content could not be loaded from configs/about.yaml. "
    "Check that the file exists and that every field is filled in."
)


def _render_body() -> None:
    """Render the About content loaded from the configuration."""
    try:
        config = load_about_config()
    except AboutConfigError as exc:
        st.error(CONFIG_ERROR_MESSAGE)
        st.caption(str(exc))
        return

    st.title("ℹ️ About Munasib Khareed")
    st.subheader(config.title)
    st.markdown(config.description)

    st.divider()
    st.markdown("### Team")
    st.markdown(f"**Supervisor:** {escape(config.supervisor)}")
    st.markdown("**Members:**")
    for member in config.members:
        role = f" — **{escape(member.role)}**" if member.role else ""
        st.markdown(
            f"- **{escape(member.name)}** ({escape(member.registration)}){role}"
        )
    st.markdown(f"**Department:** {escape(config.department)}")
    st.markdown(f"**Programme:** {escape(config.programme)}")

    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "**Pipeline:** " + escape(format_pipeline(config.pipeline))
    )
    st.markdown(
        "**Technologies:** " + escape(", ".join(config.technologies))
    )

    st.divider()
    st.markdown("### Data")
    st.markdown(escape(config.dataset_citation))

    st.divider()
    st.warning(config.notice)


def main() -> None:
    """Render the About page."""
    apply_theme("About")
    render_header()
    render_sidebar()

    _render_body()
    render_footer()


if __name__ == "__main__":
    main()
