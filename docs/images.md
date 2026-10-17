# Product images (local cache, display-only)

Munasib Khareed shows one product photo per ASIN in the dashboard. Images are
**display-only** - nothing in the analysis, risk or verdict pipeline looks at an
image - and the dashboard must keep working **offline**, so the app never
hotlinks Amazon. Instead a one-time script downloads small thumbnails into a
local, git-ignored cache and the API serves them from disk.

Attribution: `Images: Amazon product listings, UCSD Amazon Review Data 2018`
(the constant `munasib.core.constants.IMAGE_ATTRIBUTION`, for the UI to render).

## Data model

| Where | Column | Meaning |
|---|---|---|
| `data/processed/products.parquet` | `image_url` | First `imageURLHighRes` URL, else the first `imageURL` URL, else `null`. Never exposed by the API. |
| `data/processed/products.parquet` | `image_missing` | `True` when the source published no usable image URL. |
| SQLite `products` | `image_url`, `image_missing` | Mirrors of the parquet columns (the cache is a faithful copy of the processed subset). |
| SQLite `products` | `image_cached` | **Cache-only** flag: `True` once a local thumbnail exists. Freshly ingested rows start `False`. |

The raw metadata fields are `imageURL` and `imageURLHighRes`; both are lists of
URLs (a single string and a missing key are also tolerated) and only absolute
`http(s)` URLs are kept. In this category **1,073 of the 2,066** products have an
image URL; the other **993** have none and are handled like any other missing
field.

## Filling the cache

`scripts/download_images.py` is polite, resumable and never fatal per item:

```powershell
# Smoke test: five products, then stop
python scripts/download_images.py --limit 5

# The whole subset (explicit opt-in; several hundred HTTPS requests)
python scripts/download_images.py --all

# Explicit ASINs, or a file with one ASIN per line
python scripts/download_images.py --asins B00QO4ZX30 B01FQR5RZE
python scripts/download_images.py --asin-file asins.txt

# Re-download and overwrite existing files
python scripts/download_images.py --limit 5 --force
```

What it does per product:

1. Reads the stored `image_url` (products with no image URL are skipped - it
   never fabricates one).
2. Downloads with the configured `images.user_agent`, `images.timeout_seconds`,
   one retry on a transient failure and a `images.download_delay_seconds` pause
   between requests.
3. Verifies the bytes with Pillow, resizes so the longest side is at most
   `images.thumbnail_max_side` (aspect ratio preserved), flattens transparency
   onto white and re-encodes as JPEG with no EXIF/ICC metadata.
4. Writes `data/images/<asin>.jpg` atomically, then sets that product's
   `image_cached` to true. An existing file is skipped unless `--force` is given,
   so an interrupted run resumes where it stopped.

Each run writes `reports/image_download_report.json` with the computed counts
only (`attempted` / `ok` / `skipped` / `failed` by reason). A bad URL, a timeout
or an unwritable file is recorded with its reason and the run continues; exit
code `0` means the run finished, not that every image succeeded.

Flags are all overridable per run (`--delay`, `--size`, `--force`); the defaults
come from the `images` block of `configs/config.yaml` - nothing is hardcoded.

## Serving and the offline contract

`GET /products/{asin}/image` streams the cached file as `image/jpeg` with a
long-lived `Cache-Control` header. It answers `404` (standard `ErrorResponse`
envelope, code `PRODUCT_NOT_FOUND`) when:

- the ASIN is not exactly 10 alphanumeric characters (rejected before any
  filesystem access, so a crafted ASIN cannot escape the cache directory),
- the product is not in the serving cache,
- the product has no cached thumbnail (`has_image` is false), or
- the cache file is missing.

There is **no redirect to the upstream Amazon URL** and no remote fallback: if
the file is not in the local cache, the client is told `404`.

`has_image` (`ProductSummary` and `ProductDetail`) is true **only** when a
thumbnail is cached, so the dashboard can pick between an image and a placeholder
without an extra request. It is never a claim about the upstream listing.

## Placeholder rule (for the UI)

A product with `has_image: false`, or an image request that returns `404`, **must
render a placeholder** - a neutral gray tile (for example the product's first
letter or a generic icon) with the text `No image available`. The UI must never:

- hotlink `image_url` or any Amazon URL (it is not exposed by the API anyway),
- show a broken-image icon,
- hide the product or block the analysis because it has no image.

Images are decoration; every product remains fully analysable without one.

## Rights and attribution

The images are the listing photos carried by the UCSD Amazon Review Data (2018)
dataset used for this academic project. They are downloaded once into a local,
git-ignored cache for offline display, are never modified beyond resizing, and
are never redistributed. Display the `IMAGE_ATTRIBUTION` string wherever the
images appear.

## Why this shape

- **Offline and safe:** the dashboard depends on one local file per ASIN, not on
  a third-party host; no user-supplied URL is ever fetched.
- **Faithful:** the cache stores exactly the URLs the dataset published and never
  invents one; a product without an image stays without one.
- **Cheap:** 320 px JPEG thumbnails (a few KB each) instead of full-resolution
  images, so the cache directory stays small.
