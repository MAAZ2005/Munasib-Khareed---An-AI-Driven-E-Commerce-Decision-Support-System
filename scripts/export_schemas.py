"""Export the API contract's JSON Schemas and regenerate docs/api.md.

Two outputs, both derived from the Pydantic v2 schema package (never typed by
hand) so the docs and the frontend cannot drift from the code:

* ``docs/schemas/analysis_response.json`` and ``docs/schemas/error_response.json``
  - the JSON Schema of the two top-level responses, ready to paste into a
  frontend code generator.
* ``docs/api.md`` - endpoint documentation whose example payloads are rendered
  from each model's ``json_schema_extra`` example, so any schema change shows up
  in the docs on the next run.

Usage::

    python scripts/export_schemas.py            # write both outputs
    python scripts/export_schemas.py --check    # verify outputs are current
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from munasib.api.schemas import AnalysisResponse, ErrorResponse
from munasib.api.schemas.alternatives import ALTERNATIVES_EMPTY_EXAMPLE, AlternativesResult
from munasib.api.schemas.analysis import PIPELINE_VERSION
from munasib.api.schemas.common import (
    CategoryMetaResponse,
    HealthResponse,
)
from munasib.api.schemas.products import ProductDetail, ProductPage
from munasib.api.schemas.reviews import ReviewPage

DOCS_DIR = Path("docs")
SCHEMA_DIR = DOCS_DIR / "schemas"
API_DOC = DOCS_DIR / "api.md"

#: Endpoint table: (method, path, query params, description, response model).
ENDPOINTS: list[dict[str, Any]] = [
    {
        "method": "GET",
        "path": "/health",
        "query": "",
        "model": HealthResponse,
        "summary": "Service liveness.",
        "notes": [
            "Returns 200 while the process is serving; `database` tells the client "
            "whether the SQLite serving cache is reachable.",
        ],
    },
    {
        "method": "GET",
        "path": "/meta/category",
        "query": "",
        "model": CategoryMetaResponse,
        "summary": "What the serving cache contains for the analysed category.",
        "notes": [
            "`reference_date` is the latest review timestamp in the processed data "
            "(2018-10-04 for this category). All recency features are anchored on it, "
            "never on today's date.",
        ],
    },
    {
        "method": "GET",
        "path": "/products",
        "query": "query=&limit=&offset=&split=",
        "model": ProductPage,
        "summary": "Search products by title or brand (case-insensitive).",
        "notes": [
            "`query` matches title or brand literally; `%` and `_` are not wildcards.",
            "An empty `query` lists every product.",
            "Rows are ordered by `asin`, so paging with `limit`/`offset` is stable.",
            "`total` counts all matches, not just the returned page.",
            "`split` optionally filters on the product-level split (`train`, `val`, "
            "`test`); anything else is rejected with `422`.",
        ],
    },
    {
        "method": "GET",
        "path": "/products/{asin}",
        "query": "",
        "model": ProductDetail,
        "summary": "Full profile of one product.",
        "notes": [
            "Answer 404 with code `PRODUCT_NOT_FOUND` when the ASIN is not in the cache.",
            "`has_image` is true only when a thumbnail has been cached locally, so "
            "`GET /products/{asin}/image` can serve it; it is never a claim about the "
            "upstream listing.",
            "`has_image` depends on which thumbnails a machine has downloaded (the "
            "examples assume the documented `--limit 5` smoke run, so only `B00009V3UF` "
            "shows `true`); a product without a cached image is a normal case.",
        ],
    },
    {
        "method": "GET",
        "path": "/products/{asin}/image",
        "query": "",
        "model": None,
        "binary": {"media_type": "image/jpeg"},
        "summary": "The locally cached product thumbnail (offline; never hotlinked).",
        "notes": [
            "Returns `image/jpeg` bytes with a long-lived `Cache-Control: public, "
            "max-age=<images.cache_max_age_seconds>` header (cached thumbnails are "
            "immutable for a given ASIN).",
            "Available only when the product's `has_image` is true (a thumbnail exists in "
            "the local cache); otherwise `404` with code `PRODUCT_NOT_FOUND` - the "
            "endpoint never redirects to the upstream Amazon URL.",
            "The ASIN must be exactly 10 alphanumeric characters; anything else is "
            "rejected with `404` before any filesystem access.",
            "`HEAD` is supported and answers with the same headers and no body, so a "
            "client can check availability cheaply.",
        ],
    },
    {
        "method": "GET",
        "path": "/products/{asin}/reviews",
        "query": "limit=&offset=&label=&verified=",
        "model": ReviewPage,
        "summary": "Reviews of one product, newest first.",
        "notes": [
            "`label` filters on the weak-supervision label names `negative`, `neutral`, "
            "`positive`; `verified=true`/`verified=false` filters on the verified-purchase "
            "flag.",
            "`verified=false` excludes reviews whose source did not say (NULL), because "
            "NULL is not False.",
            "Ordering is `review_time` descending, then `review_id`, so paging is stable.",
        ],
    },
    {
        "method": "GET",
        "path": "/products/{asin}/analysis",
        "query": "",
        "model": AnalysisResponse,
        "summary": "The full analysis: review evidence, fuzzy + TOPSIS risk, explanation, "
        "alternatives and verdict.",
        "notes": [
            "`decision` is the fuzzy assessment followed by TOPSIS: TOPSIS ranks on the "
            "six criteria of `configs/criteria_weights.yaml` (`product_quality` 0.25, "
            "`evidence_strength` 0.20, `rating_reliability` 0.15, "
            "`negative_feedback_concentration` 0.15, `price_position` 0.10, "
            "`fuzzy_risk` 0.15 - weights sum to 1.0). The defuzzified fuzzy value is "
            "handed to TOPSIS as the extra cost criterion `fuzzy_risk`, and "
            "`topsis.risk_score` is the TOPSIS result combined with the fuzzy "
            "uncertainty.",
            "**The decision example is illustrative, not a result.** The criteria names, "
            "weights and directions come from the config, but the fuzzy/TOPSIS engine is "
            "not built yet: only `evidence_strength` (0.7173) mirrors a computed value. "
            "Everything else under `decision` - fuzzy term degrees, fired-rule "
            "strengths, the defuzzified score, the criterion values, the normalised "
            "values and the contributions - is a labelled placeholder that is never "
            "presented as a measurement.",
            "When the risk stages cannot run, `decision`, `explanation` and `verdict` are "
            "null and `warnings` explains why - the API never fabricates scores.",
            "`evidence.details` carries extra evidence signals beyond the three fixed "
            "indicators (rating-distribution entropy, helpfulness-weighted sentiment, "
            "negative-severity share, ...), each a 0-1 indicator with its raw "
            "value, direction and description; names are unique within the list and "
            "never shadow the fixed indicators, so clients can key by name.",
            "`risk_indicators` is the risk stage's five criteria (`product_quality`, "
            "`evidence_strength`, `rating_reliability`, "
            "`negative_feedback_concentration`, `price_position`) as top-level 0-1 "
            "indicators with direction and description; it is an empty list until that "
            "stage computes them, and it deliberately lives outside `decision.topsis`.",
            "`data_basis.reference_date` anchors all recency features (2018-10-04 for "
            "this dataset).",
        ],
    },
    {
        "method": "GET",
        "path": "/products/{asin}/alternatives",
        "query": "",
        "model": AlternativesResult,
        "summary": "Evidence-based alternatives for one product.",
        "notes": [
            "`total_related` counts DISTINCT related products merged across "
            "`also_bought` and `also_viewed` (a target under both edge types counts "
            "once, and the product itself is excluded), so it is a product count and "
            "not an edge count; `usable` counts how many of those distinct products "
            "are in the processed subset and can never exceed `total_related`.",
            "About 18% of products in this category have no usable alternative: `items` "
            "is then empty and `empty_reason` says why.",
            "Deltas are `alternative - subject` (a negative risk delta means safer).",
        ],
        "extra_examples": [
            (
                "Empty-alternatives response `200`",
                ALTERNATIVES_EMPTY_EXAMPLE,
            )
        ],
    },
]

#: Error kinds documented with their status codes and example payloads.
ERRORS: list[dict[str, Any]] = [
    {
        "status": 404,
        "code": "PRODUCT_NOT_FOUND",
        "when": "The ASIN is not in the processed subset (products, reviews, analysis or "
        "alternatives endpoints).",
        "payload": {
            "code": "PRODUCT_NOT_FOUND",
            "message": "Product 'B0000000000' was not found in the serving cache.",
            "detail": {"asin": "B0000000000"},
            "request_id": "req-5f0a1c2d",
        },
    },
    {
        "status": 422,
        "code": "VALIDATION_ERROR",
        "when": "A query parameter has an unusable value (for example "
        "`label=angry`, `limit=-1`, `verified=maybe`).",
        "payload": {
            "code": "VALIDATION_ERROR",
            "message": "Unknown query parameter value.",
            "detail": {"field": "label", "value": "angry", "allowed": ["negative", "neutral", "positive"]},
            "request_id": "req-0a41b7e9",
        },
    },
    {
        "status": 500,
        "code": "INTERNAL_ERROR",
        "when": "An unexpected server-side failure (the real reason is only in the "
        "server log, never in the response).",
        "payload": {
            "code": "INTERNAL_ERROR",
            "message": "Internal server error.",
            "detail": None,
            "request_id": "req-77c301aa",
        },
    },
]


def _dumps(payload: Any) -> str:
    """Render a JSON payload with stable formatting for the docs.

    Args:
        payload: JSON-serialisable object.

    Returns:
        Indented JSON text ending with a newline.
    """
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def write_schemas() -> list[Path]:
    """Write the JSON Schemas of the top-level responses.

    Returns:
        Paths written.
    """
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in (AnalysisResponse, ErrorResponse):
        target = SCHEMA_DIR / f"{model.__name__.lower()}.json"
        schema = model.model_json_schema(mode="serialization")
        target.write_text(_dumps(schema), encoding="utf-8")
        written.append(target)
    return written


def _example_of(model: Any) -> dict[str, Any]:
    """Return the realistic example declared on a response model.

    Args:
        model: Pydantic model class with a ``json_schema_extra`` example.

    Returns:
        The example payload.

    Raises:
        SystemExit: If the model declares no example (a contract bug).
    """
    extra = model.model_config.get("json_schema_extra")
    if not isinstance(extra, dict) or "example" not in extra:
        raise SystemExit(f"{model.__name__} declares no json_schema_extra example")
    example = extra["example"]
    if not isinstance(example, dict):
        raise SystemExit(f"{model.__name__} example must be a JSON object")
    return example


def _json_block(payload: Any, schema_name: str) -> list[str]:
    """Render one marked-up JSON example block.

    Args:
        payload: Example payload to render.
        schema_name: Model name the block should be validated against.

    Returns:
        Markdown lines (comment marker plus fenced JSON).
    """
    return [
        f"<!-- schema: {schema_name} -->",
        "",
        "```json",
        _dumps(payload).rstrip("\n"),
        "```",
        "",
    ]


def _endpoint_section(endpoint: dict[str, Any]) -> str:
    """Render one endpoint section (summary, notes, generated examples, markers).

    A JSON response model renders a marked-up example block; an endpoint with
    ``model=None`` and a ``binary`` descriptor (the image route) instead states its
    media type, because the body is not JSON.

    Args:
        endpoint: Endpoint description from :data:`ENDPOINTS`.

    Returns:
        Markdown text for the section.
    """
    model = endpoint["model"]
    method = endpoint["method"]
    path = endpoint["path"]
    query = endpoint["query"]
    lines = [f"### `{method} {path}`", "", endpoint["summary"], ""]
    if query:
        lines.append(f"Query parameters: `{query}`")
        lines.append("")
    for note in endpoint["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    if model is None:
        binary = endpoint.get("binary") or {}
        media_type = binary.get("media_type", "application/octet-stream")
        lines.append(f"Response `200`: `{media_type}` bytes (not a JSON body).")
        return "\n".join(lines) + "\n"
    lines.append(f"Example response `200` (schema `{model.__name__}`):")
    lines.append("")
    lines.extend(_json_block(_example_of(model), model.__name__))
    for label, payload in endpoint.get("extra_examples", []):
        lines.append(f"{label} (schema `{model.__name__}`):")
        lines.append("")
        lines.extend(_json_block(payload, model.__name__))
    return "\n".join(lines) + "\n"


def render_api_doc() -> str:
    """Render the whole ``docs/api.md`` from the schemas and the endpoint table.

    Returns:
        The markdown document text.
    """
    out: list[str] = []
    out.append("# Munasib Khareed REST API")
    out.append("")
    out.append(
        f"Contract version `{PIPELINE_VERSION}`. This document is **generated** by "
        "`python scripts/export_schemas.py` from the Pydantic v2 models in "
        "`src/munasib/api/schemas/` - do not hand-edit it; change the schema and rerun "
        "the script. Every JSON block below is rendered from the example declared on the "
        "model it documents (the `<!-- schema: ... -->` marker), and "
        "`tests/unit/test_schemas.py` re-validates every block against that schema."
    )
    out.append("")
    out.append("## Conventions")
    out.append("")
    out.append(
        "- **All timestamps are UTC.** A naive value is interpreted as UTC; responses "
        "always carry an explicit offset."
    )
    out.append(
        "- **Risk score is 0-100** (higher is riskier). Every indicator, evidence "
        "strength and sub-score is 0-1. These bounds are enforced by the schemas, so an "
        "out-of-range value is rejected at the boundary, not silently clamped. "
        "`percentile` is reported on 0-100 because it is a percentage."
    )
    out.append(
        "- **Missing data stays missing.** Price, rank and brand are `null` when the "
        "source had no usable value, and `*_missing` flags say so; nothing is imputed. "
        "`verified` is `null` when the source did not say, which is not the same as "
        "`false`."
    )
    out.append(
        "- **Images are display-only and served from a local cache.** `has_image` is true "
        "only when a thumbnail has been downloaded for the ASIN; "
        "`GET /products/{asin}/image` streams that file, so the dashboard works offline "
        "and never hotlinks the upstream listing."
    )
    out.append(
        "- **Sentiment is weak supervision.** The API exposes the label names "
        "`negative`, `neutral`, `positive` (never the stored integer codes 0/1/2). They "
        "are derived from star ratings (1-2 negative, 3 neutral, 4-5 positive) and are a "
        "weak signal, not true sentiment."
    )
    out.append(
        "- **No fabricated scores.** When a stage cannot run, its block is `null` and the "
        "reason appears in `warnings`."
    )
    out.append(
        "- **Pagination**: `GET /products` and `GET /products/{asin}/reviews` return "
        "`{total, limit, offset, items}` where `total` is the count matching the query "
        "across all pages."
    )
    out.append("")
    out.append("## Endpoints")
    out.append("")
    for endpoint in ENDPOINTS:
        out.append(_endpoint_section(endpoint))
    out.append("## Errors")
    out.append("")
    out.append(
        "Every non-2xx response uses the same envelope "
        "(`docs/schemas/error_response.json`):"
    )
    out.append("")
    out.append("<!-- schema: ErrorResponse -->")
    out.append("")
    out.append("```json")
    out.append(_dumps(ERRORS[0]["payload"]).rstrip("\n"))
    out.append("```")
    out.append("")
    out.append(
        "`code` is a stable token you can branch on; `detail` carries machine-readable "
        "particulars and may be `null`; `request_id` correlates with the server log when "
        "available."
    )
    out.append("")
    out.append("| Status | `code` | When |")
    out.append("|---:|---|---|")
    for error in ERRORS:
        out.append(f"| {error['status']} | `{error['code']}` | {error['when']} |")
    out.append("")
    out.append("Error payload examples for each status code:")
    out.append("")
    for error in ERRORS:
        out.append(f"- `HTTP {error['status']}` `{error['code']}`:")
        out.append("")
        out.append("<!-- schema: ErrorResponse -->")
        out.append("")
        out.append("```json")
        out.append(_dumps(error["payload"]).rstrip("\n"))
        out.append("```")
        out.append("")
    out.append("## Ambiguity decisions")
    out.append("")
    out.append(
        "Where the task brief left room, the simplest option was chosen and recorded here."
    )
    out.append("")
    out.append(
        "1. **`/health` and `/meta/category` models live in `common.py`.** They describe "
        "the service, not a product, review or analysis, so no dedicated area module owns "
        "them."
    )
    out.append(
        "2. **`fuzzy.output_degrees` is a mapping** keyed by the fuzzy output terms in "
        "`configs/fuzzy_rules.yaml` (`low`, `moderate`, `high`, `critical`) instead of "
        "three hard-coded fields, so the schema cannot drift from the configured rule base."
    )
    out.append(
        "3. **`topsis.percentile` is 0-100**, defined as "
        "`100 * (ranked_count - rank) / ranked_count`: 100 means the safest product in the "
        "ranked set, almost 0 the riskiest. It is the one non-risk statistic on a "
        "percentage scale."
    )
    out.append(
        "4. **`data_basis.date_range_start/end` are per-product** (that product's own oldest "
        "and newest review) while `reference_date` stays the dataset-wide latest review "
        "date, so recency is comparable across products."
    )
    out.append(
        "5. **Both alternatives counts are per distinct product, never per edge.** "
        "`total_related` is the number of distinct related products of the product "
        "across `also_bought` and `also_viewed` - a target linked by both edge types "
        "counts once and the product never counts itself - while `usable` counts how "
        "many of those distinct products are in the processed subset, so `usable` "
        "cannot exceed `total_related`. For `B00009V3UF` that is 60 related products "
        "from 62 edges, 11 of them in the subset."
    )
    out.append(
        "6. **Review text is served raw** (`review_text` reads the stored raw body) while "
        "`summary` is the cleaned headline, exactly as the cache stores them."
    )
    out.append(
        "7. **`evidence.details` and `risk_indicators` are extension points, not fixed "
        "fields.** Evidence signals added later (rating-distribution entropy, "
        "helpfulness-weighted sentiment, ...) and the risk stage's five criteria "
        "(`product_quality`, `evidence_strength`, `rating_reliability`, "
        "`negative_feedback_concentration`, `price_position`) travel as lists of the "
        "same 0-1 indicator shape, so growing the contract never breaks existing "
        "clients. A `details` name must be unique in the list and may not reuse "
        "`verified_purchase`, `helpfulness_ratio` or `review_depth` - those already "
        "have dedicated fields, so a repeat would be ambiguous; both violations are "
        "rejected by the schema. `risk_indicators` defaults to an empty list until "
        "the risk stage computes it."
    )
    out.append(
        "8. **The criterion vocabulary is mirrored in the schema.** The six TOPSIS "
        "criteria (`product_quality`, `evidence_strength`, `rating_reliability`, "
        "`negative_feedback_concentration`, `price_position`, `fuzzy_risk`) are declared "
        "in `configs/criteria_weights.yaml` and mirrored by the `CriterionName` enum, so "
        "an unknown, duplicate or missing criterion name and weights that do not sum to "
        "1.0 are rejected at the boundary; a test keeps config and schema in sync. "
        "`price_position` reads 0 = cheapest relative to peers, and a product without a "
        "usable price must be handled explicitly by the engine later - never imputed."
    )
    out.append(
        "9. **The fuzzy inputs are the proposal's five criteria.** The rule base was "
        "renamed from its three old inputs onto `product_quality`, `rating_reliability` "
        "and `negative_feedback_concentration`, swapping the low/high term labels on the "
        "two axes whose direction flipped, so every rule keeps its original risk "
        "ordering; `evidence_strength` and `price_position` are declared inputs that no "
        "rule references yet, because adding antecedents would change the rule logic. "
        "Output terms (`low`, `moderate`, `high`, `critical`) and all consequents are "
        "unchanged. The SHAP example's features (`negative_sentiment_share`, "
        "`aspect_complaint_intensity`, `verified_purchase_ratio`) are pipeline feature "
        "names - a separate namespace from the criteria - and were deliberately not "
        "renamed."
    )
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Write the schema exports and the generated API doc.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = written, 1 = verification failed).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; fail when the generated files are not current.",
    )
    args = parser.parse_args(argv)

    schemas = write_schemas() if not args.check else []
    doc_text = render_api_doc()
    if args.check:
        current = API_DOC.read_text(encoding="utf-8") if API_DOC.exists() else ""
        if current != doc_text:
            print("docs/api.md is out of date; run: python scripts/export_schemas.py")
            return 1
        for model in (AnalysisResponse, ErrorResponse):
            target = SCHEMA_DIR / f"{model.__name__.lower()}.json"
            if not target.exists():
                print(f"missing {target}; run: python scripts/export_schemas.py")
                return 1
        print("schema exports are current")
        return 0

    API_DOC.write_text(doc_text, encoding="utf-8")
    print("Wrote:")
    for path in (*schemas, API_DOC):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
