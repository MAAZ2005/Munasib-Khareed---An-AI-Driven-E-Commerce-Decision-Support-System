# Munasib Khareed: Frontend Architecture & Streamlit Multipage Guide

This document describes the Streamlit frontend architecture for Munasib Khareed, including page routing, state management, caching policies, environment variables, server execution, and stub banner governance.

---

## 1. Page Routing in Streamlit Multipage Apps

Streamlit natively discovers multipage applications through the directory structure:

```text
app/
├── Home.py                    # Application entry point (Landing / System Status)
├── api_client.py              # Pure HTTP client (no Streamlit dependencies)
├── cache.py                   # Streamlit cache_data wrappers with TTLs
├── state.py                   # st.session_state accessors and page guards
├── theme.py                   # System styling & colorblind-safe verdict themes
├── components/
│   ├── _common.py             # Private: figure scaffolding & number formatting
│   ├── __init__.py            # Exports every reusable component
│   ├── sidebar.py             # Health status, active product, navigation, stub banner
│   ├── metric_card.py         # 1. Labelled number with help text and delta
│   ├── risk_gauge.py          # 2. 0-100 gauge banded by verdict_thresholds.yaml
│   ├── verdict_badge.py       # 3. The three locked verdict labels
│   ├── rating_distribution_chart.py   # 4. Star histogram
│   ├── sentiment_breakdown_chart.py   # 5. Weak-supervision class counts
│   ├── sentiment_over_time_chart.py   # 6. Monthly shares + slope
│   ├── aspect_bar_chart.py    # 7. Positive vs negative aspects
│   ├── evidence_indicator_radar.py    # 8. 0-1 evidence ring (cost direction marked)
│   ├── fuzzy_membership_chart.py      # 9. Input memberships & output degrees
│   ├── topsis_ranking_chart.py        # 10. Per-criterion contributions
│   ├── shap_waterfall_chart.py        # 11. Base value → contributions → prediction
│   ├── concern_list.py        # 12. Severity-tagged concerns
│   └── alternatives_table.py  # 13. Comparison table with deltas
└── pages/
    ├── 1_Product_Selection.py # Search catalogue & select active ASIN
    ├── 2_Product_Overview.py  # Product profile, metadata & rating distributions
    ├── 3_Review_Analysis.py   # Sentiment breakdown & aspect extraction
    ├── 4_Risk_and_Evidence.py # Fuzzy rules, evidence indicators & TOPSIS ranking
    ├── 5_Explanation_SHAP.py  # Surrogate model explanation & feature attribution
    ├── 6_Alternatives.py      # Graph-linked alternatives & delta metrics
    ├── 7_Consumer_Verdict.py  # Three-tier verdict card, concerns & rationale
    └── _component_gallery.py  # Every component on one page (review aid, see §9)
```

### Routing Rules
- The application root is launched using `app/Home.py`.
- Sub-pages live inside `app/pages/` and are automatically ordered alphabetically/numerically by Streamlit based on their filename prefixes (`1_...`, `2_...`, etc.).
- The number prefix dictates the consumer advocacy workflow order.
- Direct navigation between pages can be performed via the sidebar or programmatically using `st.page_link()`.

---

## 2. State Flow (`selection -> session_state -> pages`)

The application avoids global variables and mutates state strictly via encapsulated helper functions in `app/state.py`:

```mermaid
graph TD
    A[Home.py / Landing] -->|Navigate| B[1_Product_Selection.py]
    B -->|Search Query & Offset| C[(st.session_state)]
    B -->|User clicks 'Select'| D[set_selected_asin(asin)]
    D --> C
    C -->|selected_asin| E[Pages 2 through 7]
    E -->|require_product guard| F{ASIN chosen?}
    F -->|No| G[Stop & Prompt Selection Link]
    F -->|Yes| H[Render Page Content]
```

### Stored State Keys
| Key | Type | Default | Description |
|---|---|---|---|
| `selected_asin` | `str \| None` | `None` | ASIN of the actively analyzed product. |
| `last_analysis` | `AnalysisResponse \| None` | `None` | Cached composed analysis object. |
| `search_query` | `str` | `""` | Active text filter in product search. |
| `search_offset` | `int` | `0` | Active pagination offset in product search. |

### Page Guard: `require_product()`
All analytical sub-pages (`2_Product_Overview.py` through `7_Consumer_Verdict.py`) invoke `require_product()` at the start of page execution:
- If `selected_asin` is not set or empty, it renders an informative warning banner, displays a direct `st.page_link("pages/1_Product_Selection.py")`, and calls `st.stop()`.
- If `selected_asin` is present, it returns the ASIN string to the page.

---

## 3. Caching Strategy (`app/cache.py`)

To ensure rapid page navigation and prevent redundant network overhead against the FastAPI backend, API calls are wrapped using `@st.cache_data` with strictly defined Time-To-Live (TTL) horizons:

| Endpoint Wrapper | TTL | Arguments | Rationale |
|---|---|---|---|
| `get_cached_category_meta()` | 10 min (`600s`) | None | Dataset metadata changes only upon full ingestion. |
| `get_cached_search_products()` | 60 s (`60s`) | `query, limit, offset, split` | Allows fast back-and-forth paging while absorbing rapid queries. |
| `get_cached_product()` | 5 min (`300s`) | `asin` | Product details are static within the SQLite serving cache. |
| `get_cached_reviews()` | 60 s (`60s`) | `asin, limit, offset, label, verified` | Balances memory usage with browsing responsiveness. |
| `get_cached_analysis()` | 5 min (`300s`) | `asin` | Pipeline calculations are deterministic per product. |
| `get_cached_alternatives()` | 5 min (`300s`) | `asin` | Product graph comparisons are deterministic per product. |
| `get_cached_health()` | 15 s (`15s`) | None | Short interval for near-realtime backend liveness. |

### Cache Invalidation
A dedicated button in the sidebar (**"Clear cached data"**) invokes `st.cache_data.clear()`, allowing users to manually refresh all cached data across all pages.

---

## 4. Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MUNASIB_API_URL` | `http://127.0.0.1:8000` | Base URL of the FastAPI backend. Used by `ApiClient`. |

---

## 5. How to Run Both Servers

Both the FastAPI backend and Streamlit frontend must run simultaneously in development.

### Option A: Using Makefile (Recommended)
Open two separate PowerShell terminal windows with the virtual environment activated (`.\.venv\Scripts\Activate.ps1`):

**Terminal 1 (Backend API):**
```powershell
make run-api
```
*(Starts uvicorn at `http://127.0.0.1:8000` with hot-reloading)*

**Terminal 2 (Streamlit UI):**
```powershell
make run-ui
```
*(Starts Streamlit at `http://localhost:8501`)*

### Option B: Running Commands Directly
**Terminal 1 (Backend API):**
```powershell
uvicorn munasib.api.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Streamlit UI):**
```powershell
streamlit run app/Home.py --server.port 8501
```

---

## 6. Stub Banner Governance

Because the system uses a deterministic `StubAnalysisService` prior to the full integration of the real fuzzy-logic and TOPSIS engines, strict transparency rules apply:

- Any page displaying analysis data invokes `show_stub_banner(analysis)`.
- The banner triggers if:
  1. `analysis.pipeline_version` begins with `"stub"` (e.g. `"stub-0"`), OR
  2. Any item in `analysis.warnings` starts with `"STUB:"`.
- When triggered, it renders a prominent warning:
  > **⚠️ Placeholder analysis from the stub engine. Risk scores, verdicts and explanations are NOT real results.**
- This guarantees users and evaluators never mistake deterministic test placeholders for authentic machine learning assessments.

---

## 7. Offline Compliance

The frontend is strictly offline-first:
- Zero references to external CDNs, Google Fonts, or remote stylesheets.
- System-default typography stacks (`-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`).
- Colors are curated to be accessible and colorblind-safe across all three verdict states (`CONSIDER`, `CONSIDER WITH CAUTION`, `CONSUMER RISK`), always paired with clear icons (`✅`, `⚠️`, `🛑`) and explicit text labels.

---

## 8. Component Library (`app/components/`)

The dashboard pages are assembled from the reusable components in `app/components/`, each in its own module and re-exported from `app/components/__init__.py`.

### 8.1 The design rule

Every chart component has two halves:

| Half | Shape | Rule |
|---|---|---|
| Builder | `build_<name>_figure(data) -> plotly.graph_objects.Figure` | Pure. Imports no `streamlit`, calls no API, reads no `st.session_state`. Same input → identical figure. |
| Renderer | `render_<name>(data)` | Thin. Adds title, axis labels and captions, then hands the figure to `st.plotly_chart(..., width="stretch")`. |

The builders are the unit-tested half (`tests/unit/test_component_figures.py`); the renderers are exercised end to end by the gallery (`tests/e2e/test_component_gallery.py`).

The four non-chart components expose only a render function: `metric_card`, `verdict_badge`, `concern_list` and `alternatives_table`.

### 8.2 Empty and partial data

A component never crashes and never invents a value:

- A builder handed `None` or an empty payload returns `empty_figure(message)` — a figure with **no traces** and the reason attached as its annotation.
- Its renderer tests `if not figure.data`, reads that reason back with `placeholder_message(figure)` and shows it via `st.info`. Builder and renderer therefore cannot disagree about what counts as empty.
- Missing scalars render as `n/a` (metric card, table deltas), never as `0`.

### 8.3 Colors

All chart colors come from `CHART_PALETTE` in `app/theme.py` — the Okabe-Ito colorblind-safe palette keyed by **semantics** (`positive`, `cost`, `severity_high`, `band_caution`, ...) rather than hue names. A component never hardcodes a hex code, and color is never the only channel: every series is also labeled in text (axis ticks, legend, hover text, captions).

### 8.4 Where the numbers come from

- **Risk gauge bands** are read from `configs/verdict_thresholds.yaml` through `get_settings()` (`app/components/risk_gauge.py`), never hardcoded, so the gauge and the backend verdict logic move together. If the file cannot be read the gauge is replaced by an explanatory `st.info` rather than guessed bands.
- **Sentiment labels** are weak supervision from star ratings (1-2 negative, 3 neutral, 4-5 positive — locked decision #2). Every caption that quotes those numbers repeats the caveat.
- Components accept validated schema objects only (`munasib.api.schemas`); they never import ML modules or the DB layer.

---

## 9. Component Gallery

`app/pages/_component_gallery.py` renders all 13 components on one page from the example payloads already shipped with the API schemas (the same `ANALYSIS_EXAMPLE` / `ALTERNATIVES_EMPTY_EXAMPLE` that `scripts/export_schemas.py` publishes). It calls no endpoint.

### How to open it

1. `streamlit run app/Home.py --server.port 8501`, then pick **`_component_gallery`** from the sidebar navigation.
2. Or run it directly: `streamlit run app/pages/_component_gallery.py --server.port 8501`.

The page always shows a visible banner — *"Sample data: component gallery (illustrative values, not results)"* — and a single checkbox, **"Switch every component to empty data"**, which re-renders every component with missing payloads (`None`, empty lists, zero-count schemas) so the empty states can be checked without breaking anything.

### How to hide it before submission

Streamlit lists every `*.py` in `app/pages/` whose name does not start with `.`, so an underscore prefix is **not** enough to hide it. Move the file out of the pages directory (or give it a non-`.py` suffix):

```powershell
# reversible: rename back to _component_gallery.py to restore it
Rename-Item app\pages\_component_gallery.py _component_gallery.py.disabled
```

The component library itself (`app/components/`) stays in place — only the review page is hidden.
