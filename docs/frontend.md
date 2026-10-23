# Munasib Khareed: Frontend Architecture & Streamlit Multipage Guide

This document describes the Streamlit frontend architecture for Munasib Khareed, including page routing, state management, caching policies, environment variables, server execution, and stub banner governance.

---

## 1. Page Routing in Streamlit Multipage Apps

The application registers its pages explicitly with `st.navigation` / `st.Page` (see `app/Home.py`). Streamlit therefore does **not** treat `app/pages/` as an auto-discovered directory: every page is declared once, with a clean title, an explicit icon and an explicit order, and the numeric filename prefixes are only there to keep the files sorted on disk.

```text
app/
├── Home.py                    # Entry point: st.navigation + the Landing page
├── api_client.py              # Pure HTTP client (no Streamlit dependencies)
├── cache.py                   # Streamlit cache_data wrappers with TTLs
├── state.py                   # st.session_state accessors, page guards, analysis loader
├── formatting.py              # Pure display helpers (price/rank/date, stars, CSV/JSON builders)
├── errors.py                  # Friendly messages for failed API calls
├── theme.py                   # System styling, header/accent tokens & verdict colors
├── branding.py                # Offline logo loader (app/assets/logo.png), text fallback
├── about.py                   # Pydantic model + loader for configs/about.yaml
├── assets/
│   └── logo.png               # Committed copy of docs/logo.png (the only image asset)
├── components/
│   ├── _common.py             # Private: figure scaffolding & number formatting
│   ├── __init__.py            # Exports every reusable component
│   ├── header.py              # Shared header band (brand, search, links) + footer
│   ├── sidebar.py             # Health status, active product, navigation, stub banner
│   ├── product_image.py       # Cached thumbnail or neutral local placeholder tile
│   ├── product_card.py        # Marketplace card builders (escaped, absence-aware)
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
    ├── About.py               # Team, pipeline, technologies & dataset (configs/about.yaml)
    └── _component_gallery.py  # Every component on one page (review aid, see §9)
```

### Routing Rules
- The application root is launched using `app/Home.py`, which is both the entry point and the landing page.
- `Home.py` calls `st.set_page_config(...)` and then `st.navigation(pages, position="sidebar").run()`. Because `st.navigation` is used, auto-discovery of `app/pages/` is switched off and only the declared pages appear.
- The page order, titles (no numeric prefixes) and icons are declared once in `Home._pages()`; the workflow order is Data → Intelligence → Evidence → Risk → Decision → Explanation → Alternatives → Verdict.
- Every page (and the landing page) renders the same shared chrome: `render_header()` at the top and `render_footer()` at the bottom, from `app/components/header.py`.
- Navigation between pages uses native `st.page_link()` (header links and in-page links) and `st.switch_page()` (the header search box, the "View full analysis" button).

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
| `get_product_image()` | 1 hour (`3600s`) | `asin` | A cached thumbnail is immutable for an ASIN; a `404` (no thumbnail) is cached as `None`. |
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

The non-chart components expose a small, focused API: `metric_card`, `verdict_badge`, `concern_list` and `alternatives_table` (render only); `header.render_header` / `header.render_footer` (the shared application chrome); and `product_card` (pure, escaped card builders plus the card layout used by the selection and alternatives grids).

`product_image(asin, has_image, width, title)` is the one component that talks to
the API: it shows the cached thumbnail bytes as an image, or a neutral gray
placeholder tile drawn with inline HTML (no external resource, no broken-image
icon) when the product has no image or the fetch fails. It requests the image
**only** when `has_image` is true, and it always renders the attribution caption.
See `docs/images.md` for the data model and the download step.

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

1. `streamlit run app/Home.py --server.port 8501`, then pick **Component Gallery** from the sidebar navigation.
2. Or run it directly: `streamlit run app/pages/_component_gallery.py --server.port 8501`.

The page always shows a visible banner — *"Sample data: component gallery (illustrative values, not results)"* — and a single checkbox, **"Switch every component to empty data"**, which re-renders every component with missing payloads (`None`, empty lists, zero-count schemas) so the empty states can be checked without breaking anything.

### How to hide it before submission

The page is declared in `Home._pages()` with `visibility="visible"` so it stays reachable exactly as before. To remove it from the sidebar, change its declaration to `visibility="hidden"` (it stays reachable by URL), or delete the `st.Page("pages/_component_gallery.py", ...)` entry:

```python
st.Page("pages/_component_gallery.py", title="Component Gallery", ..., visibility="hidden")
```

The component library itself (`app/components/`) stays in place — only the review page disappears from the navigation.

---

## 10. Dashboard Pages on the Stub API (manual click-through checklist)

The seven pages are wired to the stub API. Work through the list with both servers
running (`make run-api` and `make run-ui`) and confirm every step.

### Shared behaviour to confirm everywhere

- **The header band is on every page.** Top-left shows the committed logo
  (`app/assets/logo.png`, with "Munasib Khareed" as the text fallback if the file
  is missing), the middle is the catalogue search box, and the right has
  **Search** and **About** links. Submitting the search stores the query, resets
  the page offset and jumps to Product Selection.
- **The footer is on every page.** It repeats the image attribution, the dataset
  citation (Ni, Li, McAuley, EMNLP 2019) and *Final Year Project, FAST NU Karachi*.
- **Analysis is fetched once per product.** The first of pages 3-7 you open fetches
  `GET /products/{asin}/analysis` (a spinner is shown) and stores it in
  `last_analysis`; the other pages reuse it. Selecting a different product clears
  it so the next page fetches the new product's analysis.
- **Failure mapping, never a traceback:**
  - `ApiNotFound` (404) shows *"Product not found in the serving cache."*, clears
    `selected_asin` and offers a link back to Product Selection.
  - `ApiUnavailable` shows *"The analysis API is not reachable."* plus the exact
    recovery command `uvicorn munasib.api.main:app --port 8000`.
  - Any other API error shows the request id so it can be matched to a backend log.
- **Nothing is fabricated.** A missing price reads *"Price not listed"*, a missing
  rank *"Rank not listed"*, and any other missing scalar *"n/a"*.

### 1. Product Selection (`1_Product_Selection.py`)

- [ ] Type a brand keyword (e.g. `Range Kleen`) in the **header** search and press
      Enter; the page shows the matches and the "Displaying N–M of T" caption.
- [ ] The grid lays out four cards per row on a wide window and wraps to fewer on a
      narrow one; changing the query resets to page 1.
- [ ] Each card shows a fixed-height image box (the image **or** the neutral
      "No image available" tile), a two-line title, the brand, the star row with the
      numeric rating and review count, and the price in large type (or
      "Price not listed").
- [ ] Page with **Next ➡️** / **⬅️ Previous**; the centred "Page x of y" counter and
      the enabled/disabled buttons follow.
- [ ] Click **Select** on a product: a success banner naming the ASIN appears and
      the button on that card becomes a disabled **Selected ✅**. **Clear search**
      empties the query.

### 2. Product Overview (`2_Product_Overview.py`)

- [ ] The page is split into three zones: image on the left, details in the centre,
      the **Consumer Advocate** panel on the right.
- [ ] The centre shows the title, brand, star row, price, sales rank and the
      **About this item** bullets from the published features.
- [ ] The four metric cards show reviews, mean rating and the first/last review dates.
- [ ] The **Full product description** expander opens.
- [ ] The Consumer Advocate panel shows the stub banner, a **Placeholder** caption,
      the verdict badge, a compact risk gauge, the evidence strength and a
      **View full analysis** button that opens page 7.
- [ ] The "Related products" counts and the "Data basis · category … pipeline …"
      caption are present.

### 3. Review Analysis (`3_Review_Analysis.py`)

- [ ] The stub banner is visible (never a real result without it).
- [ ] The star histogram, sentiment breakdown, sentiment-over-time lines and aspect
      bars all render (or their `st.info` empty state, never an exception).
- [ ] The verified-purchase and helpfulness metric cards show.
- [ ] The review table pages with **⬅️ Newer / Older ➡️** buttons and both filters
      (sentiment label, verified purchase) narrow the rows and its caption count.

### 4. Risk and Evidence (`4_Risk_and_Evidence.py`)

- [ ] The stub banner is visible.
- [ ] The risk gauge is banded by `configs/verdict_thresholds.yaml`, with the band
      legend caption underneath.
- [ ] The composite-evidence metric card shows.
- [ ] The indicator table lists every evidence and risk indicator with value, raw
      value, direction and description.
- [ ] The radar, the fuzzy membership chart and the TOPSIS chart render.

### 5. Explanation (SHAP) (`5_Explanation_SHAP.py`)

- [ ] The stub banner is visible.
- [ ] R², MAE and "meets configured minimum" metric cards show, with the
      plain-language note on how to read them.
- [ ] The waterfall chart renders with the base value and the final prediction.
- [ ] The reasons list quotes real numbers next to each feature.

### 6. Alternatives (`6_Alternatives.py`)

- [ ] The stub banner is visible.
- [ ] The **Customers also considered** heading and the "usable of total related"
      caption show.
- [ ] Each alternative card shows an image or placeholder, a two-line title, its
      verdict badge, its risk score (with the Δ risk delta) and the relation as an
      icon **plus** a word.
- [ ] The comparison table at the bottom matches the cards.
- [ ] Click **Switch to this product**: the active ASIN changes, the analysis is
      re-fetched for the new product, and the page re-renders for it.
- [ ] On a product with no usable alternative, the empty reason and both counts
      are shown instead of cards.

### 7. Consumer Verdict (`7_Consumer_Verdict.py`)

- [ ] The stub banner is visible.
- [ ] The verdict badge, risk gauge, evidence strength and risk score show.
- [ ] The major concerns list, the rationale and the data-basis block
      (category, review count, date range, reference date, pipeline version) show.
- [ ] **Download analysis (JSON)** saves a file that parses back into
      `AnalysisResponse`; **Download summary (CSV)** saves the flat summary.
- [ ] (PDF export arrives in a later prompt.)

### Image behaviour (all pages)

- [ ] With `has_image` false, the page shows the neutral tile and does **not** call
      `GET /products/{asin}/image` at all.
- [ ] With `has_image` true and a thumbnail cached, the real image renders.
- [ ] With `has_image` true but the fetch returning 404 (or failing), the neutral
      tile is shown instead — no broken image, no exception.
- [ ] The caption
      `Images: Amazon product listings, UCSD Amazon Review Data 2018` appears on
      every page that can show an image (1, 2, 6, 7).
- [ ] Nothing is hotlinked: the app works with no internet connection as long as
      the downloader has populated `data/images/` (see `docs/images.md`).

### About (`About.py`)

- [ ] The header **About** link opens it (no product needs to be selected).
- [ ] It shows the exact title, the supervisor, the three members with their
      registration numbers and the **Leader** role, the department and the programme
      line.
- [ ] It shows the pipeline one-liner
      (`Data -> Intelligence -> Evidence -> Risk -> Decision -> Explanation -> Alternatives -> Consumer Verdict`),
      the technology list, the dataset citation and the academic-prototype notice.
- [ ] It contains **no** email address or phone number (search the page text for `@`).

---

## 11. Design notes (marketplace layout)

### Marketplace layout, our own brand only

The dashboard borrows the *layout conventions* of a large online shop — a slim top
bar with the brand at the left, a search box in the middle and links at the right; a
responsive results grid of cards; a buy-box style panel beside the product; a fixed
footer — but every name, logo and asset is the project's own. There is **no**
third-party marketplace name, logo or asset anywhere, and nothing is hotlinked.

### Offline-first chrome

- The header, footer and cards are built from native Streamlit elements
  (`st.container`, `st.columns`, `st.form`, `st.image`, `st.button`, `st.page_link`);
  the only custom HTML/CSS lives in `app/theme.py` (`CUSTOM_CSS`) and is HTML-escaped
  at every point where product data is interpolated (`app/components/product_card.py`).
- No external fonts or CDNs: the type is the system font stack
  (`FONT_FAMILY`), the "icons" are emoji/text characters, and the logo is the one
  committed local PNG (`app/assets/logo.png`).
- The header band is styled with a **key-based CSS class**
  (`st.container(key="mk_header")` → `.st-key-mk_header`), so no fragile DOM
  selectors are needed. The navy band (`HEADER_BACKGROUND`) carries light text
  (`HEADER_TEXT`), and the logo reads as a light chip on it because the PNG is
  dark-on-white.

### The single accent color rule

There is exactly one accent color, `ACCENT_COLOR = "#4338ca"` (indigo) in
`app/theme.py`, mirrored by `primaryColor` in `.streamlit/config.toml` so native
buttons match. It is used only for primary actions (the header **Search** submit,
**Select**, **Switch to this product**, **View full analysis**).

**Rule:** the accent must stay visibly distinct from every verdict and risk color.
Verdict colors are green/amber/red (`VERDICT_CONFIGS`) and the risk bands plus chart
hues are the Okabe-Ito palette (`CHART_PALETTE`); indigo collides with none of them,
so an action can never be mistaken for a verdict. Verdict colors and their icons
(`✅` / `⚠️` / `🛑`) are unchanged.

### How to edit the About content

Edit `configs/about.yaml` — the page reads it at render time. It is validated by
`app.about.AboutConfig`, so a blank member name, an empty member list or a missing
field fails at load time (the page then shows a friendly message) instead of
rendering blanks. Do not add contact details: the project publishes no email address
or phone number.

### How to replace the logo

Replace `app/assets/logo.png` (a committed copy of `docs/logo.png`). The header
shows it top-left on every page and uses it as the browser page icon. If the file is
missing or unreadable the header falls back to the app name as text — the helper in
`app/branding.py` returns `None` rather than raising, so a bad asset can never crash
a page. A dark-on-light logo (like the current one) is displayed on a light chip on
the navy band; a light-on-transparent logo can be dropped on the band directly.

### Manual verification steps (not covered by automated tests)

While automated tests cover state management, headless rendering via `AppTest`,
and API fault tolerance, perform the following visual checks in a live browser
prior to final project defense:

1. **Responsive Viewport Scaling:**
   - **375px (Mobile):** Verify the header wraps gracefully without horizontal overflow, product cards stack cleanly in a single column, and the Product Overview three-zone layout stacks into image → details → Consumer Advocate buy box.
   - **768px (Tablet):** Verify product cards arrange in 2 columns and charts expand to fill container width.
   - **1024px / 1440px (Desktop):** Verify 4-card grid on Product Selection, 3-zone desktop layout on Product Overview, and proper alignment of the shared header band.
2. **Visual Contrast and Logo Legibility:**
   - Confirm the white chip behind `app/assets/logo.png` provides sharp contrast against the midnight navy `#0f172a` header container.
   - Confirm verdict badges and risk gauge colors match their semantic labels (`CONSIDER` green, `CONSIDER WITH CAUTION` amber, `CONSUMER RISK` red) and are never communicated by color alone.
3. **Interactive Browser Downloads:**
   - On page 7 (*Consumer Verdict*), click **Full analysis (JSON)** and **Flat summary (CSV)** in the browser to confirm native browser download prompts trigger and files save with appropriate `.json` and `.csv` extensions.
4. **Interactive Toasts and Transitions:**
   - On page 1 (*Product Selection*), click **Select** on a product card; verify that a confirmation toast pops up and the button immediately turns into a disabled **Selected ✅**.
   - On page 6 (*Alternatives*), click **Switch to this product**; verify the active ASIN switches and new analysis loads with a success toast.
5. **Keyboard Accessibility and Focus Rings:**
   - Tab through the header search input, search submit button, and navigation links to verify standard browser focus rings remain visible.

