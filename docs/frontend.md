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
│   └── sidebar.py             # Health status, active product, navigation, stub banner
└── pages/
    ├── 1_Product_Selection.py # Search catalogue & select active ASIN
    ├── 2_Product_Overview.py  # Product profile, metadata & rating distributions
    ├── 3_Review_Analysis.py   # Sentiment breakdown & aspect extraction
    ├── 4_Risk_and_Evidence.py # Fuzzy rules, evidence indicators & TOPSIS ranking
    ├── 5_Explanation_SHAP.py  # Surrogate model explanation & feature attribution
    ├── 6_Alternatives.py      # Graph-linked alternatives & delta metrics
    └── 7_Consumer_Verdict.py  # Three-tier verdict card, concerns & rationale
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
