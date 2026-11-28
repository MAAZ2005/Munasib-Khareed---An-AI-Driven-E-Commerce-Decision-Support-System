# Running the API

How to start the FastAPI backend, what every route returns, and how to drive it
with PowerShell. The frontend (Streamlit, prompts 8-10) talks to this server and
nothing else - it never imports the analysis modules.

## Prerequisites

The API serves the **SQLite serving cache**, a processed copy of one UCSD Amazon
Review Data (2018) category. Build it once:

```powershell
# from the repository root, with the venv active
python scripts/build_dataset.py     # writes data/processed/*.parquet + meta.json
python scripts/ingest_db.py         # fills data/db/munasib.sqlite
```

If the database is missing, startup fails immediately with a message naming
`scripts/ingest_db.py` rather than letting the first request discover it.

## Starting the server

```powershell
# Option A - Makefile target (recommended)
make run-api

# Option B - uvicorn directly, with auto-reload
uvicorn munasib.api.main:app --reload --port 8000

# Option C - host/port from configs/config.yaml (api.host / api.port)
uvicorn munasib.api.main:app --host 127.0.0.1 --port 8000
```

`make run-api` runs `uvicorn munasib.api.main:app --host 127.0.0.1 --port 8000
--reload`. `src/munasib/api/main.py` exposes both a `create_app()` factory and a
module-level `app`, so `munasib.api.main:app` always works. Thevenv must be
active (`.\.venv\Scripts\Activate.ps1`); never use the system Python.

Interactive docs: <http://127.0.0.1:8000/docs> (Swagger UI) and
<http://127.0.0.1:8000/redoc>.

## Routes

| Method | Route | Response model |
|---|---|---|
| GET | `/health` | `HealthResponse` |
| GET | `/meta/category` | `CategoryMetaResponse` |
| GET | `/products` | `ProductPage` |
| GET | `/products/{asin}` | `ProductDetail` |
| GET | `/products/{asin}/image` | `image/jpeg` (cached thumbnail) |
| GET | `/products/{asin}/reviews` | `ReviewPage` |
| GET | `/products/{asin}/analysis` | `AnalysisResponse` |
| GET | `/products/{asin}/alternatives` | `AlternativesResult` |

Every route documents its query-parameter bounds in the OpenAPI schema; an
out-of-range value is rejected as `422` before any query runs.

### Query parameters

| Route | Parameter | Bounds | Default |
|---|---|---|---|
| `/products` | `query` | any text, literal match | `""` (all products) |
| `/products` | `limit` | 1-100 | `20` |
| `/products` | `offset` | >= 0 | `0` |
| `/products` | `split` | `train`, `val`, `test` | none |
| `/products/{asin}/reviews` | `limit` | 1-100 | `20` |
| `/products/{asin}/reviews` | `offset` | >= 0 | `0` |
| `/products/{asin}/reviews` | `label` | `negative`, `neutral`, `positive` | none |
| `/products/{asin}/reviews` | `verified` | `true`, `false` | none |

## curl examples (PowerShell - use `curl.exe`)

```powershell
# 1. Liveness and cache reachability (also returns the current UTC time)
curl.exe http://127.0.0.1:8000/health

# 2. What the cache contains: counts, splits, date range, reference date
curl.exe http://127.0.0.1:8000/meta/category

# 3. Search products (case-insensitive title/brand match), stable paging
curl.exe "http://127.0.0.1:8000/products?query=filter&limit=3"
curl.exe "http://127.0.0.1:8000/products?query=filter&limit=3&offset=3"
curl.exe "http://127.0.0.1:8000/products?limit=100&split=train"

# 4. One product's full profile
curl.exe http://127.0.0.1:8000/products/B00009V3UF

# 4b. Its cached thumbnail (see "Product images" below; 404 until it is downloaded)
curl.exe -I http://127.0.0.1:8000/products/B00009V3UF/image

# 5. One product's reviews, newest first, with filters
curl.exe "http://127.0.0.1:8000/products/B00009V3UF/reviews?limit=2"
curl.exe "http://127.0.0.1:8000/products/B00009V3UF/reviews?label=negative"
curl.exe "http://127.0.0.1:8000/products/B00009V3UF/reviews?verified=false"

# 6. The full analysis (see "What the analysis contains" below)
curl.exe http://127.0.0.1:8000/products/B00009V3UF/analysis

# 7. Evidence-based alternatives
curl.exe http://127.0.0.1:8000/products/B00009V3UF/alternatives

# 8. Error cases: status codes and the shared ErrorResponse envelope
curl.exe -i http://127.0.0.1:8000/products/NOTANASIN              # 404
curl.exe -i "http://127.0.0.1:8000/products?limit=0"              # 422
curl.exe -i "http://127.0.0.1:8000/products?limit=1000"           # 422
curl.exe -i "http://127.0.0.1:8000/products?offset=-1"            # 422
curl.exe -i "http://127.0.0.1:8000/products/B00009V3UF/reviews?label=angry"  # 422

# 9. Send your own trace id (it is echoed back and logged)
curl.exe -i -H "X-Request-ID: my-trace-1" http://127.0.0.1:8000/health
```

All non-2xx bodies use the same envelope from
`docs/schemas/error_response.json`: `code`, `message`, `detail`,
`request_id`. The `X-Request-ID` response header always matches
`request_id`, so one id ties a dashboard action to a log line.

## What the analysis contains: real data vs placeholders (important)

The `/analysis` route is served by `HybridAnalysisService` by default
(`pipeline_version` `"hybrid-0"`) and marks itself as such on every response:

- `warnings[0]` begins with `"PARTIAL:"` and names exactly which parts are real
  and which are placeholders.

**Real, read from the stored tables:** the star histogram and review counts, the
verified/helpfulness figures, the sentiment breakdown and monthly trend from the
stored **model predictions** (`review_features.predicted_label`; the star-derived
weak shares travel alongside as `star_*_share`), the aspects from
`product_aspects`, the evidence indicators and composite `evidence_strength` from
`product_evidence`, the five decision criteria from `product_risk_indicators`
(when that row exists; otherwise `risk_indicators` is `[]`), the product's own
review date range, `data_basis.reference_date` from `data/processed/meta.json`,
and the alternatives read from the real `related_products` edges.

**Placeholder, seeded from the ASIN with `hashlib`:** the fuzzy assessment, the
TOPSIS ranking, the SHAP explanation, the risk score, the verdict label and the
concerns, and the alternatives' risk scores. Results are byte-identical for
repeated calls of the same ASIN, so a frontend can render deterministically - but
they must never be presented to a user as a real assessment. The real engine
(prompt 22) swaps in behind the same `AnalysisService` contract without any
router change.

The pure `StubAnalysisService` (`analysis.service: stub`, `pipeline_version`
`"stub-0"`, `warnings[0]` beginning `"STUB:"`) still exists for comparison: it
serves the same placeholders but keeps the star-derived weak-label sentiment and
the live evidence composite instead of the stored predictions and stored
composite.

`/products/{asin}/reviews` and `/products/{asin}/alternatives` carry no
placeholder values at all.

## Product images

Images are **display-only** (no image analysis) and the dashboard must work
offline, so the API never hotlinks Amazon. Instead:

1. `python scripts/download_images.py --limit 5` (or `--all`) downloads a small
   JPEG thumbnail per ASIN into `data/images/` (git-ignored) and sets
   `products.image_cached` to true for each success. `scripts/ingest_db.py`
   re-derives that flag from the same files after every run, so re-ingesting the
   cache never un-downloads anything.
2. `GET /products/{asin}/image` streams that local file as `image/jpeg` with a
   `Cache-Control: public, max-age=<images.cache_max_age_seconds>` header.

`has_image` on `/products` and `/products/{asin}` is true **only** when a
thumbnail is cached, so the dashboard can decide between an image and a
placeholder without probing. A product with no usable source image (993 of the
2,066 here) is a normal case: the endpoint answers `404` with the standard
envelope and the UI shows a placeholder.

```powershell
# Fill a few thumbnails, then check one (expect: 200, content-type: image/jpeg)
python scripts/download_images.py --limit 5
curl.exe -I http://127.0.0.1:8000/products/<cached-asin>/image
```

The route validates the ASIN against `ASIN_PATTERN` (exactly 10 alphanumeric
characters) and confirms the resolved file sits inside the cache directory, so a
crafted `asin` can never reach the filesystem. Full details, including image
rights/attribution, live in [`docs/images.md`](images.md).

## Alternatives counting

Both numbers count **distinct related products**, never edges:

- `total_related` is the number of distinct related products of this product
  across `also_bought` **and** `also_viewed`, merged. A target linked by both
  edge types counts once, and the product never counts itself as its own
  alternative.
- `usable` is how many of those distinct related products are themselves in the
  processed subset, so `usable` can never exceed `total_related`.

For `B00009V3UF` the real cache holds **62 edges** (4 `also_bought` + 58
`also_viewed`) which collapse to **60 distinct related products** - two targets
are linked by both edge types - of which 12 edges point back into the subset and
cover **11 distinct alternatives** (one in-subset target is linked twice). So
the endpoint answers `total_related = 60, usable = 11`, not `62` and `12`.

About 18% of this category's products have no usable alternative; those return
`items: []` with an `empty_reason`, which is a valid answer, not a failure.

## Choosing the analysis implementation

`analysis.service` in `configs/config.yaml` selects the implementation:

```yaml
analysis:
  service: "hybrid"   # the default
```

- `"hybrid"` (default) serves the stored sentiment predictions, aspects and
  evidence with clearly-labelled placeholders for the unbuilt decision stages
  (`pipeline_version` `"hybrid-0"`).
- `"stub"` serves the all-placeholder analysis over the live review rows
  (`pipeline_version` `"stub-0"`); useful to compare the two label sets.
- `"real"` (the fuzzy + TOPSIS engine) does not exist yet: selecting it makes
  `/analysis` and `/alternatives` answer `503` with code `MODEL_UNAVAILABLE` and
  logs the actionable message. It never silently falls back to a placeholder
  service, which would present placeholder numbers as a real result.

Override per environment with `MUNASIB_ANALYSIS_SERVICE`:

```powershell
$env:MUNASIB_ANALYSIS_SERVICE = "hybrid"
```

## CORS

`api.cors_origins` in `configs/config.yaml` lists the Streamlit origins
(`http://localhost:8501`, `http://127.0.0.1:8501`). Add an entry there if the
dashboard runs elsewhere; the API never defaults to allowing everything.

## Decisions taken while building this

Where the brief left room, the simplest option was chosen:

1. **`RequestValidationError` is FastAPI's own.** The project's
   `core/exceptions.py` has no such class, so the `422` handler targets
   `fastapi.exceptions.RequestValidationError` (raised for `limit=0`, bad
   `label`, bad `split`, bad `verified`) plus `pydantic.ValidationError`. Field
   details are flattened to `{"errors": [{"field", "type", "message",
   "value", "constraint"}]}`.
2. **An unknown ASIN is 404 on every product route**, including
   `/reviews`, which checks the product first so an empty review list can never
   be mistaken for a missing product.
3. **`generated_at` comes from `meta.json`'s `built_at`**, not the wall clock,
   so two analyses of the same product are byte-identical. `data_basis`
   `date_range_start`/`date_range_end` are the product's own reviews, and
   `reference_date` is the dataset-wide latest review date (2018-10-04).
4. **A missing database fails at startup**, not on the first request, with a
   message naming `scripts/ingest_db.py`.
5. **`/health` answers 200 even when the cache is unreachable**;
   `database` is the field to watch, because a missing cache is an operational
   condition rather than a crash.
6. **The stub compares at most 20 alternatives** (`MAX_STUB_ALTERNATIVES`) to
   stay cheap, while `usable` still reports the true distinct count.
7. **`relation` comes from verdict severity**, compared through the shared
   `VERDICT_LABELS` order, so no comparison threshold is invented.
8. **The examples in `docs/api.md` were corrected, never hand-edited.** The
   documented `total_related` for `B00009V3UF` was first the in-subset edge
   count (`12`), then the edge count (`62`); counting distinct related products
   gives the correct `60` (with `usable = 11`). Both the analysis and
   alternatives examples live in the schema modules and the docs are
   regenerated with `python scripts/export_schemas.py`.
9. **`analysis.service` was added to `configs/config.yaml` and
   `Settings`**, so the switch is configured like every other threshold rather
   than read ad hoc inside the API layer.
10. **`total_related` counts products, not edges** (see "Alternatives
    counting"), so the number a client sees matches the alternatives it can
    iterate over rather than a raw row count from `related_products`.
11. **Images are cached and served locally, never hotlinked.** The image route is
    a binary endpoint (`response_class=FileResponse`) rather than a JSON model, so
    `scripts/export_schemas.py` documents it with its media type instead of a JSON
    block. `has_image` mirrors the serving-cache flag `image_cached`, and a
    malformed ASIN is rejected before any filesystem access.

## Tests

```powershell
python -m pytest tests/unit tests/integration -q
```

- `tests/unit/test_analysis_config.py` - the hybrid default, the env override and
  the rejection of an unknown value.
- `tests/unit/test_hybrid_service.py` - the stored sentiment/aspects/evidence
  values, the reused placeholders and the graceful degradation when a stored row
  is absent.
- `tests/unit/test_stub_service.py` - the stub's seeding (same ASIN, same
  output; different ASIN, different placeholders), the real evidence numbers,
  and config-driven criteria/rules.
- `tests/integration/test_api.py` - every route over a temporary SQLite cache
  (200s, 404s, 422s, paging, request-id, determinism, the hybrid/STUB markers,
  empty alternatives, the `real` service error, the cached-image route and the
  rejection of a malicious ASIN), plus tests against the real
  `data/db/munasib.sqlite` that skip automatically when it is absent (including
  the stored `evidence_strength` equality for `B00009V3UF`).
- `tests/unit/test_download_images.py` - the image downloader against an
  `httpx.MockTransport` (resize, metadata stripping, resume/`--force`, every
  failure reason, the JSON report counts, the cache flag) with no network.
- `tests/unit/test_suggest_demo_products.py` - the demo-product suggester over a
  temporary cache (group thresholds, an overlapping candidate, a candidate with
  too few alternatives, the computed polarization shares, the CLI output).
