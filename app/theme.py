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

from app.branding import logo_path
from munasib.api.schemas import VerdictLabel
from munasib.core.constants import IMAGE_ATTRIBUTION

__all__ = [
    "ACCENT_COLOR",
    "CHART_PALETTE",
    "CUSTOM_CSS",
    "DATASET_CITATION",
    "FONT_FAMILY",
    "FOOTER_CREDIT",
    "HEADER_BACKGROUND",
    "HEADER_TEXT",
    "IMAGE_ATTRIBUTION",
    "VERDICT_CONFIGS",
    "apply_theme",
    "get_verdict_style",
    "render_verdict_badge",
    "render_verdict_card",
]

#: Attribution shown as a caption wherever a cached product image is displayed.
#:
#: Re-exported from :mod:`munasib.core.constants` (the single definition) so the
#: dashboard never retypes the string and cannot drift from ``docs/images.md``.
#: The images belong to Amazon and the individual sellers and are shown only for
#: the academic demo, from the local cache, offline.
IMAGE_ATTRIBUTION = IMAGE_ATTRIBUTION

#: Marketplace header band: a dark navy background with light text. Kept here (not
#: in a page) so every page shares one definition of the app chrome.
HEADER_BACKGROUND = "#0f172a"
HEADER_TEXT = "#f8fafc"

#: The single accent color used for primary actions (the header search submit and
#: any primary button). Documented rule: the accent is deliberately *outside* every
#: verdict and risk color, so an action can never be mistaken for a verdict.
#:
#: The verdict palette is green/amber/red (:data:`VERDICT_CONFIGS`) and the risk
#: bands plus chart hues are the Okabe-Ito palette (:data:`CHART_PALETTE`); the
#: accent is indigo (``#4338ca``), which collides with none of them. It is mirrored
#: by ``primaryColor`` in ``.streamlit/config.toml`` so native buttons match.
ACCENT_COLOR = "#4338ca"

#: Dataset citation shown in the footer (repeated on the About page from config).
DATASET_CITATION = (
    'Dataset: Ni, Li & McAuley, "Justifying recommendations using '
    'distantly-labeled reviews and fine-grained aspects", EMNLP 2019 '
    "(UCSD Amazon Review Data 2018)."
)

#: Footer credit line shown on every page.
FOOTER_CREDIT = "Final Year Project, FAST NU Karachi"

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

#: System font stack used by every Plotly figure. Offline only: no web font is
#: downloaded, the chart simply names fonts the OS already ships.
FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

#: Colorblind-safe chart palette shared by every chart component.
#:
#: Components take their colors from this mapping and never from a literal hex
#: code in a component module, so the dashboard stays readable under the common
#: forms of color-vision deficiency. The hues are the Okabe-Ito palette; the
#: keys are semantic (``positive``, ``cost``, ``caution``, ...) rather than hue
#: names so a component can express meaning instead of picking a shade.
#:
#: Color is never the only channel: every series is also labeled in text
#: (axis tick labels, legend entries, hover text or captions), which is what
#: makes the palette safe rather than merely colorblind-friendly.
CHART_PALETTE: dict[str, str] = {
    # Okabe-Ito hues, available when a chart needs a distinct hue with no
    # semantic weight.
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
    # Review sentiment (weak supervision from star ratings: 1-2 negative,
    # 3 neutral, 4-5 positive).
    "positive": "#009E73",
    "negative": "#D55E00",
    "neutral": "#56B4E9",
    "unlabelled": "#999999",
    # TOPSIS / criterion direction: blue = the more of this the better,
    # orange = the more of this the worse.
    "benefit": "#0072B2",
    "cost": "#E69F00",
    # Alternative relation, concern severity and fuzzy rules.
    "better": "#009E73",
    "comparable": "#56B4E9",
    "worse": "#D55E00",
    "severity_high": "#D55E00",
    "severity_medium": "#E69F00",
    "severity_low": "#56B4E9",
    # The three verdict bands of configs/verdict_thresholds.yaml.
    "band_consider": "#009E73",
    "band_caution": "#E69F00",
    "band_risk": "#D55E00",
    # Documented tints of the band hues above, used as gauge step fills so the
    # darker band boundary and needle stay legible on top of them.
    "band_consider_fill": "#d8efe7",
    "band_caution_fill": "#fdeecd",
    "band_risk_fill": "#f7d6cb",
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

/* Marketplace header band. `st.container(key="mk_header")` adds the
   `.st-key-mk_header` class, which this rule styles. The logo PNG has a white
   background, so it reads as a light chip on the navy band. */
.st-key-mk_header {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 10px 18px;
    margin-bottom: 20px;
}
.st-key-mk_header [data-testid="stImage"] img {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 6px 10px;
    display: block;
}
.mk-wordmark {
    color: #f8fafc;
    font-weight: 800;
    font-size: 1.2rem;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
.mk-tagline {
    color: #94a3b8;
    font-size: 0.74rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.st-key-mk_header a {
    color: #c7d2fe;
    font-weight: 600;
    text-decoration: none;
}
.st-key-mk_header a:hover {
    color: #ffffff;
    text-decoration: underline;
}
.st-key-mk_header label,
.st-key-mk_header p {
    color: #e2e8f0;
}

/* Product card typography (grids on pages 1 and 6). */
.mk-card-title {
    font-weight: 700;
    font-size: 0.95rem;
    line-height: 1.3;
    color: #0f172a;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 2.6em;
}
.mk-price {
    font-size: 1.3rem;
    font-weight: 800;
    color: #0f172a;
}
.mk-price-missing {
    font-size: 0.95rem;
    font-weight: 600;
    color: #64748b;
}
.mk-stars {
    color: #b45309;
    letter-spacing: 2px;
    font-size: 1rem;
}
.mk-brand {
    color: #475569;
    font-size: 0.82rem;
}
.mk-review-count {
    color: #475569;
    font-size: 0.82rem;
}

/* Consumer Advocate buy box (page 2). */
.mk-advocate {
    color: #0f172a;
}

/* Footer block. */
.mk-footer {
    color: #64748b;
    font-size: 0.78rem;
    line-height: 1.5;
}

/* Pipeline sequence & tech chips on About page */
.mk-pipeline-container {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin: 12px 0 16px 0;
}
.mk-pipeline-step {
    background-color: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.88rem;
    font-weight: 600;
    color: #0f172a;
}
.mk-pipeline-arrow {
    color: #6366f1;
    font-weight: 700;
}
.mk-tech-badge {
    display: inline-block;
    background-color: #e0e7ff;
    border: 1px solid #c7d2fe;
    color: #3730a3;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
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
    # Browser page icon: the local logo when present, else a neutral fallback.
    # Never fetched from a CDN, so the tab works offline.
    icon: str = str(logo_path()) if logo_path().is_file() else "🛒"
    try:
        st.set_page_config(
            page_title=full_title,
            page_icon=icon,
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
