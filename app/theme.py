"""Theming and verdict styling for the Munasib Khareed Streamlit frontend.

Applies offline-safe CSS (system typography, accessible borders, card containers)
and defines colorblind-safe styling for the locked three verdict labels:
- CONSIDER
- CONSIDER WITH CAUTION
- CONSUMER RISK

Rules:
    - Must work 100% offline (no external CDNs, fonts, or tracking scripts).
    - Color is never the sole signal: every verdict presentation couples an
      accessible color with an icon and unambiguous text token.
"""

from __future__ import annotations

import streamlit as st

from munasib.api.schemas import VerdictLabel

#: Colorblind-safe styling dictionary for each valid verdict label.
VERDICT_CONFIGS: dict[str, dict[str, str]] = {
    VerdictLabel.CONSIDER.value: {
        "icon": "✅",
        "text": "CONSIDER",
        "color": "#14532d",
        "background": "#dcfce7",
        "border": "#22c55e",
        "description": "Low assessed risk backed by sufficient review evidence.",
    },
    VerdictLabel.CONSIDER_WITH_CAUTION.value: {
        "icon": "⚠️",
        "text": "CONSIDER WITH CAUTION",
        "color": "#78350f",
        "background": "#fef3c7",
        "border": "#f59e0b",
        "description": "Moderate risk or limited evidence base; review flagged concerns.",
    },
    VerdictLabel.CONSUMER_RISK.value: {
        "icon": "🛑",
        "text": "CONSUMER RISK",
        "color": "#7f1d1d",
        "background": "#fee2e2",
        "border": "#ef4444",
        "description": "Elevated consumer risk or significant negative sentiment patterns.",
    },
}

#: Minimal offline-safe CSS injected into every page.
CUSTOM_CSS = """
<style>
/* Munasib Khareed Offline Styling */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.verdict-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
}

.verdict-card {
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0 20px 0;
}

.verdict-card h3 {
    margin: 0 0 8px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.stat-box {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
</style>
"""


def apply_theme(page_title: str = "Munasib Khareed") -> None:
    """Apply standard page configuration and inject offline styling.

    Safe to call on every page; suppresses duplicate page_config warnings.

    Args:
        page_title: Title suffix for browser tab.
    """
    full_title = f"{page_title} | Munasib Khareed" if page_title != "Munasib Khareed" else page_title
    try:
        st.set_page_config(
            page_title=full_title,
            page_icon="🛒",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        # Streamlit requires set_page_config to be first call; ignore if already called
        pass

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_verdict_style(label: str | VerdictLabel) -> dict[str, str]:
    """Retrieve color, icon, and text attributes for a verdict label.

    Args:
        label: Verdict string or VerdictLabel enum.

    Returns:
        Mapping containing icon, text, color, background, border, and description.
    """
    key = label.value if isinstance(label, VerdictLabel) else str(label).strip()
    return VERDICT_CONFIGS.get(
        key,
        {
            "icon": "ℹ️",
            "text": key,
            "color": "#1e293b",
            "background": "#f1f5f9",
            "border": "#94a3b8",
            "description": "Verdict information.",
        },
    )


def render_verdict_badge(label: str | VerdictLabel) -> str:
    """Generate HTML string for an accessible verdict badge.

    Args:
        label: Verdict label value.

    Returns:
        HTML snippet safe to render via markdown.
    """
    style = get_verdict_style(label)
    return (
        f'<span class="verdict-badge" style="'
        f'color: {style["color"]}; '
        f'background-color: {style["background"]}; '
        f'border: 2px solid {style["border"]};">'
        f'<span>{style["icon"]}</span>'
        f'<span>{style["text"]}</span>'
        f"</span>"
    )


def render_verdict_card(
    label: str | VerdictLabel,
    risk_score: float | None = None,
    evidence_strength: float | None = None,
    rationale: str | None = None,
) -> None:
    """Render a prominent verdict card with supporting risk metrics.

    Args:
        label: One of the 3 verdict labels.
        risk_score: Assessed risk score (0-100).
        evidence_strength: Evidence strength score (0.0-1.0).
        rationale: Consumer-facing narrative explanation.
    """
    style = get_verdict_style(label)
    metrics_html = ""
    if risk_score is not None or evidence_strength is not None:
        parts: list[str] = []
        if risk_score is not None:
            parts.append(f"<b>Risk Score:</b> {risk_score:.1f} / 100")
        if evidence_strength is not None:
            parts.append(f"<b>Evidence Strength:</b> {evidence_strength * 100:.1f}%")
        metrics_html = f"<p style='margin: 8px 0; font-size: 0.95rem;'>{' &nbsp;|&nbsp; '.join(parts)}</p>"

    rationale_html = ""
    if rationale:
        rationale_html = (
            f"<p style='margin: 8px 0 0 0; font-size: 0.95rem; line-height: 1.4;'>"
            f"<b>Rationale:</b> {rationale}</p>"
        )

    card_html = (
        f'<div class="verdict-card" style="'
        f'background-color: {style["background"]}; '
        f'border: 2px solid {style["border"]}; '
        f'color: {style["color"]};">'
        f'<h3 style="color: {style["color"]};">'
        f'<span>{style["icon"]}</span> {style["text"]}'
        f"</h3>"
        f'<p style="margin: 0; font-size: 0.95rem; font-style: italic;">{style["description"]}</p>'
        f"{metrics_html}"
        f"{rationale_html}"
        f"</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)
