# Munasib Khareed REST API

Contract version `0.1.0`. This document is **generated** by `python scripts/export_schemas.py` from the Pydantic v2 models in `src/munasib/api/schemas/` - do not hand-edit it; change the schema and rerun the script. Every JSON block below is rendered from the example declared on the model it documents (the `<!-- schema: ... -->` marker), and `tests/unit/test_schemas.py` re-validates every block against that schema.

## Conventions

- **All timestamps are UTC.** A naive value is interpreted as UTC; responses always carry an explicit offset.
- **Risk score is 0-100** (higher is riskier). Every indicator, evidence strength and sub-score is 0-1. These bounds are enforced by the schemas, so an out-of-range value is rejected at the boundary, not silently clamped. `percentile` is reported on 0-100 because it is a percentage.
- **Missing data stays missing.** Price, rank and brand are `null` when the source had no usable value, and `*_missing` flags say so; nothing is imputed. `verified` is `null` when the source did not say, which is not the same as `false`.
- **Sentiment is weak supervision.** The API exposes the label names `negative`, `neutral`, `positive` (never the stored integer codes 0/1/2). They are derived from star ratings (1-2 negative, 3 neutral, 4-5 positive) and are a weak signal, not true sentiment.
- **No fabricated scores.** When a stage cannot run, its block is `null` and the reason appears in `warnings`.
- **Pagination**: `GET /products` and `GET /products/{asin}/reviews` return `{total, limit, offset, items}` where `total` is the count matching the query across all pages.

## Endpoints

### `GET /health`

Service liveness.

- Returns 200 while the process is serving; `database` tells the client whether the SQLite serving cache is reachable.

Example response `200` (schema `HealthResponse`):

<!-- schema: HealthResponse -->

```json
{
  "status": "ok",
  "service": "Munasib Khareed",
  "version": "0.1.0",
  "database": "reachable"
}
```


### `GET /meta/category`

What the serving cache contains for the analysed category.

- `reference_date` is the latest review timestamp in the processed data (2018-10-04 for this category). All recency features are anchored on it, never on today's date.

Example response `200` (schema `CategoryMetaResponse`):

<!-- schema: CategoryMetaResponse -->

```json
{
  "category": "Appliances",
  "n_products": 2066,
  "n_reviews": 425993,
  "n_related_edges": 84885,
  "n_related_edges_in_subset": 22349,
  "date_range_start": "2001-11-08T00:00:00Z",
  "date_range_end": "2018-10-04T00:00:00Z",
  "reference_date": "2018-10-04T00:00:00Z",
  "split_sizes": {
    "train": {
      "products": 1444,
      "reviews": 303317
    },
    "val": {
      "products": 310,
      "reviews": 62961
    },
    "test": {
      "products": 312,
      "reviews": 59715
    }
  }
}
```


### `GET /products`

Search products by title or brand (case-insensitive).

Query parameters: `query=&limit=&offset=`

- `query` matches title or brand literally; `%` and `_` are not wildcards.
- An empty `query` lists every product.
- Rows are ordered by `asin`, so paging with `limit`/`offset` is stable.
- `total` counts all matches, not just the returned page.

Example response `200` (schema `ProductPage`):

<!-- schema: ProductPage -->

```json
{
  "total": 2,
  "limit": 20,
  "offset": 0,
  "items": [
    {
      "asin": "B00QO4ZX30",
      "title": "FilterBuy Replacement Humidifier Filter Compatible with Holmes HWF62. Replacement for part # HWF62, HWF62D, HWF-62 &amp; Hamilton Beach Part # 05910",
      "brand": "FilterBuy",
      "price": 10.99,
      "n_reviews": 152,
      "mean_rating": 4.473684,
      "split": "train"
    },
    {
      "asin": "B01FQR5RZE",
      "title": "Hamilton Beach PIM-1-1A Portable Ice Maker, Black with Stainless Steel",
      "brand": "Hamilton Beach",
      "price": 18.29,
      "n_reviews": 114,
      "mean_rating": 3.605263,
      "split": "test"
    }
  ]
}
```


### `GET /products/{asin}`

Full profile of one product.

- Answer 404 with code `PRODUCT_NOT_FOUND` when the ASIN is not in the cache.

Example response `200` (schema `ProductDetail`):

<!-- schema: ProductDetail -->

```json
{
  "asin": "B00009V3UF",
  "title": "Range Kleen 8234 Gas Range Knobs &ndash; 4 White Gas Range Safety Knobs Gas Stove Accessories",
  "brand": "Range Kleen",
  "price": 12.66,
  "n_reviews": 52,
  "mean_rating": 2.634615,
  "split": "train",
  "brand_missing": false,
  "price_status": "single",
  "price_missing": false,
  "price_max": null,
  "rank": ">#4,165 in Appliances (See top 100)",
  "rank_value": 4165,
  "rank_category": "Appliances",
  "rank_missing": false,
  "description": "Want a fresh start with your stove burner knobs? Replace them with the Range Kleen 8234 Pack of 4 white gas range replacement knobs! Use these gas stove accessoriesto replace broken gas burner knobsthat have become a nuisance and a safety hazard to you and your loved ones. This pack of 4 gas range safety knobs includes four gas range knobs, eight adapters, and four overlays. The set of gas range knobs replacements are neutral white in color,making them compatible with any kitchen style or motif. No need to do the heavy lifting when it comes to replacing an appliance; just give your stove a facelift with new burner knobs! NOTE: Some gas range knobs replacements may require a longer shaft for replacement. Item 82001 Knob Shaft Extender may be required to replace thestove burner knobs on your gas stove.",
  "features": [
    "SAFE GAS RANGE REPLACEMENT KNOBS: Gas range knobs give you piece of mind; stove burner knobs protect and help gauge stove settings.",
    "PRISTINE WHITE BURNER KNOB: Range Kleen 8234 gas burner knobs are immaculate white color; providing maximum design compatibility.",
    "COMPLETE BURNER KNOBS SET: 4 knobs, 8 adapters and 4 overlays included in this burner knobs for gas stove set; update all of your gas burner knobs with Range Kleen.",
    "REPLACEMENT BURNER KNOB FOR GAS STOVE: No need to buy a new stove, just freshen up the burner knob covers for gas range stoves for a fresh look.",
    "EASY-TURN BURNER KNOB: Gas range knob replacements reset your stoves look; these brand-new burner gas stove accessories restore your stove to that factory setting feel."
  ],
  "first_review_time": "2009-07-20T00:00:00Z",
  "last_review_time": "2018-02-03T00:00:00Z"
}
```


### `GET /products/{asin}/reviews`

Reviews of one product, newest first.

Query parameters: `limit=&offset=&label=&verified=`

- `label` filters on the weak-supervision label names `negative`, `neutral`, `positive`; `verified=true`/`verified=false` filters on the verified-purchase flag.
- `verified=false` excludes reviews whose source did not say (NULL), because NULL is not False.
- Ordering is `review_time` descending, then `review_id`, so paging is stable.

Example response `200` (schema `ReviewPage`):

<!-- schema: ReviewPage -->

```json
{
  "total": 52,
  "limit": 2,
  "offset": 0,
  "items": [
    {
      "review_id": "1b5dbcd03044cba1",
      "asin": "B00009V3UF",
      "reviewer_id": "A2MBG73LAWGRDZ",
      "rating": 1.0,
      "verified": true,
      "helpful_votes": 0,
      "review_time": "2018-02-03T00:00:00Z",
      "review_text": "These knobs are supposed to have rings you can place on them with temperatures so you know where to turn the knob. They were missing the rings, or as I can see from other reviews - they simply don't come with them even though the packaging states that they do. Don't waste your money on these unless the cooktop itself has the temperature markings.",
      "has_text": true,
      "text_length": 348,
      "summary": "Packaging incorrect - no temperature markings are included.",
      "sentiment_label": "negative"
    },
    {
      "review_id": "9ee8b475b1af2971",
      "asin": "B00009V3UF",
      "reviewer_id": "A1W1EHTEKYJIKJ",
      "rating": 5.0,
      "verified": true,
      "helpful_votes": 0,
      "review_time": "2017-12-15T00:00:00Z",
      "review_text": "Exactly as expected",
      "has_text": true,
      "text_length": 19,
      "summary": "Five Stars",
      "sentiment_label": "positive"
    }
  ]
}
```


### `GET /products/{asin}/analysis`

The full analysis: review evidence, fuzzy + TOPSIS risk, explanation, alternatives and verdict.

- `decision` is the fuzzy assessment followed by TOPSIS: the defuzzified fuzzy value is handed to TOPSIS as an extra cost criterion (`risk_score`), and `topsis.risk_score` is the TOPSIS result combined with the fuzzy uncertainty.
- When the risk stages cannot run, `decision`, `explanation` and `verdict` are null and `warnings` explains why - the API never fabricates scores.
- `evidence.details` carries extra evidence signals beyond the three fixed indicators (rating-distribution entropy, helpfulness-weighted sentiment, negative-feedback concentration, ...), each a 0-1 indicator with its raw value, direction and description; names are unique within the list and never shadow the fixed indicators, so clients can key by name.
- `risk_indicators` is the risk stage's five criteria (`product_quality`, `evidence_strength`, `rating_reliability`, `negative_feedback_concentration`, `price_position`) as top-level 0-1 indicators with direction and description; it is an empty list until that stage computes them, and it deliberately lives outside `decision.topsis`.
- `data_basis.reference_date` anchors all recency features (2018-10-04 for this dataset).

Example response `200` (schema `AnalysisResponse`):

<!-- schema: AnalysisResponse -->

```json
{
  "asin": "B00009V3UF",
  "product": {
    "asin": "B00009V3UF",
    "title": "Range Kleen 8234 Gas Range Knobs &ndash; 4 White Gas Range Safety Knobs Gas Stove Accessories",
    "brand": "Range Kleen",
    "price": 12.66,
    "n_reviews": 52,
    "mean_rating": 2.634615,
    "split": "train",
    "brand_missing": false,
    "price_status": "single",
    "price_missing": false,
    "price_max": null,
    "rank": ">#4,165 in Appliances (See top 100)",
    "rank_value": 4165,
    "rank_category": "Appliances",
    "rank_missing": false,
    "description": "Want a fresh start with your stove burner knobs? Replace them with the Range Kleen 8234 Pack of 4 white gas range replacement knobs! Use these gas stove accessoriesto replace broken gas burner knobsthat have become a nuisance and a safety hazard to you and your loved ones. This pack of 4 gas range safety knobs includes four gas range knobs, eight adapters, and four overlays. The set of gas range knobs replacements are neutral white in color,making them compatible with any kitchen style or motif. No need to do the heavy lifting when it comes to replacing an appliance; just give your stove a facelift with new burner knobs! NOTE: Some gas range knobs replacements may require a longer shaft for replacement. Item 82001 Knob Shaft Extender may be required to replace thestove burner knobs on your gas stove.",
    "features": [
      "SAFE GAS RANGE REPLACEMENT KNOBS: Gas range knobs give you piece of mind; stove burner knobs protect and help gauge stove settings.",
      "PRISTINE WHITE BURNER KNOB: Range Kleen 8234 gas burner knobs are immaculate white color; providing maximum design compatibility.",
      "COMPLETE BURNER KNOBS SET: 4 knobs, 8 adapters and 4 overlays included in this burner knobs for gas stove set; update all of your gas burner knobs with Range Kleen.",
      "REPLACEMENT BURNER KNOB FOR GAS STOVE: No need to buy a new stove, just freshen up the burner knob covers for gas range stoves for a fresh look.",
      "EASY-TURN BURNER KNOB: Gas range knob replacements reset your stoves look; these brand-new burner gas stove accessories restore your stove to that factory setting feel."
    ],
    "first_review_time": "2009-07-20T00:00:00Z",
    "last_review_time": "2018-02-03T00:00:00Z"
  },
  "review_analysis": {
    "rating_distribution": {
      "counts": {
        "1": 19,
        "2": 4,
        "3": 14,
        "4": 7,
        "5": 8
      },
      "shares": {
        "1": 0.3654,
        "2": 0.0769,
        "3": 0.2692,
        "4": 0.1346,
        "5": 0.1538
      },
      "mean_rating": 2.634615,
      "total": 52
    },
    "sentiment": {
      "total": 52,
      "negative_count": 23,
      "neutral_count": 14,
      "positive_count": 15,
      "unlabelled_count": 0,
      "negative_share": 0.4423,
      "neutral_share": 0.2692,
      "positive_share": 0.2885
    },
    "temporal_trend": {
      "period": "month",
      "buckets": [
        {
          "period_start": "2009-07-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2010-02-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 1.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2011-11-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 0.0,
          "positive_share": 1.0
        },
        {
          "period_start": "2011-12-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 0.0,
          "positive_share": 1.0
        },
        {
          "period_start": "2012-01-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 1.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2012-08-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2012-10-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2012-11-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 0.0,
          "positive_share": 1.0
        },
        {
          "period_start": "2013-01-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2013-02-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 1.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2013-04-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 0.0,
          "positive_share": 1.0
        },
        {
          "period_start": "2013-05-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 1.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2013-06-01T00:00:00Z",
          "n_reviews": 3,
          "negative_share": 0.0,
          "neutral_share": 0.6667,
          "positive_share": 0.3333
        },
        {
          "period_start": "2013-08-01T00:00:00Z",
          "n_reviews": 2,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2013-09-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 1.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2013-10-01T00:00:00Z",
          "n_reviews": 2,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2013-11-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 0.0,
          "positive_share": 1.0
        },
        {
          "period_start": "2014-01-01T00:00:00Z",
          "n_reviews": 2,
          "negative_share": 0.0,
          "neutral_share": 1.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2014-03-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 1.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2014-04-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 1.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2014-06-01T00:00:00Z",
          "n_reviews": 2,
          "negative_share": 0.0,
          "neutral_share": 0.0,
          "positive_share": 1.0
        },
        {
          "period_start": "2014-08-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2014-09-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2014-10-01T00:00:00Z",
          "n_reviews": 2,
          "negative_share": 0.5,
          "neutral_share": 0.0,
          "positive_share": 0.5
        },
        {
          "period_start": "2014-12-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 1.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2015-01-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 0.0,
          "positive_share": 1.0
        },
        {
          "period_start": "2015-02-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 1.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2015-03-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2015-05-01T00:00:00Z",
          "n_reviews": 2,
          "negative_share": 0.5,
          "neutral_share": 0.5,
          "positive_share": 0.0
        },
        {
          "period_start": "2015-06-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 0.0,
          "positive_share": 1.0
        },
        {
          "period_start": "2015-08-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2015-10-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2015-11-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2016-02-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2016-03-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2016-04-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        },
        {
          "period_start": "2016-08-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 0.0,
          "positive_share": 1.0
        },
        {
          "period_start": "2017-01-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 0.0,
          "neutral_share": 0.0,
          "positive_share": 1.0
        },
        {
          "period_start": "2017-11-01T00:00:00Z",
          "n_reviews": 3,
          "negative_share": 0.6667,
          "neutral_share": 0.0,
          "positive_share": 0.3333
        },
        {
          "period_start": "2017-12-01T00:00:00Z",
          "n_reviews": 2,
          "negative_share": 0.5,
          "neutral_share": 0.0,
          "positive_share": 0.5
        },
        {
          "period_start": "2018-02-01T00:00:00Z",
          "n_reviews": 1,
          "negative_share": 1.0,
          "neutral_share": 0.0,
          "positive_share": 0.0
        }
      ],
      "positive_share_slope": -0.0001,
      "n_reviews_used": 52
    },
    "aspects": [
      {
        "aspect": "fit",
        "polarity": "negative",
        "mention_count": 6,
        "examples": [
          "These do not fit the Maytag Performa.",
          "Have a cheap crapy stove and didn't fit"
        ]
      },
      {
        "aspect": "value",
        "polarity": "positive",
        "mention_count": 4,
        "examples": [
          "good buy",
          "quick and easy"
        ]
      }
    ]
  },
  "evidence": {
    "verified_purchase": {
      "name": "verified_purchase",
      "value": 0.9423,
      "raw_value": 0.9423,
      "direction": "benefit",
      "description": "Share of the product's reviews that are verified purchases (49 of 52 here)."
    },
    "helpfulness": {
      "name": "helpfulness_ratio",
      "value": 0.1538,
      "raw_value": 0.56,
      "direction": "benefit",
      "description": "Share of reviews that at least one other shopper found helpful (8 of 52 here); the raw value is the mean helpful-vote count per review."
    },
    "review_depth": {
      "name": "review_depth",
      "value": 1.0,
      "raw_value": 206.31,
      "direction": "benefit",
      "description": "How substantive the review text is, from the mean character length of the reviews (206.31 characters here)."
    },
    "evidence_strength": 0.7173,
    "details": [
      {
        "name": "rating_distribution_entropy",
        "value": 0.9173,
        "raw_value": 1.4764,
        "direction": "cost",
        "description": "Shannon entropy of the star distribution normalised by log(5) (0 = one shared star, 1 = perfectly spread): 0.9173 here over the 19/4/14/7/8 star split, so opinions span all five stars and the 2.63 mean hides a divided audience."
      },
      {
        "name": "helpfulness_weighted_sentiment",
        "value": 0.5517,
        "raw_value": 1.1034,
        "direction": "benefit",
        "description": "Weak labels (0 negative, 1 neutral, 2 positive) averaged with each review's helpful votes as weight and rescaled to 0-1 (raw 0-2 mean label 1.1034): 0.5517 here from 29 helpful votes across 52 reviews, above the 0.4231 unweighted value - the reviews shoppers voted useful read better than average."
      },
      {
        "name": "negative_feedback_concentration",
        "value": 0.8261,
        "raw_value": 19.0,
        "direction": "cost",
        "description": "Share of the weak-negative reviews that are one-star reviews (19 of 23 here): negative feedback packed into the harshest bucket rather than spread over 2-3 stars."
      }
    ]
  },
  "risk_indicators": [],
  "decision": {
    "fuzzy": {
      "input_memberships": {
        "negative_sentiment": {
          "term_degrees": {
            "low": 0.08,
            "medium": 0.91,
            "high": 0.25
          }
        },
        "aspect_complaint_intensity": {
          "term_degrees": {
            "low": 0.31,
            "medium": 0.62,
            "high": 0.02
          }
        },
        "evidence_volatility": {
          "term_degrees": {
            "low": 0.0,
            "medium": 0.88,
            "high": 0.14
          }
        }
      },
      "fired_rules": [
        {
          "rule_id": 7,
          "antecedents": {
            "negative_sentiment": "medium",
            "aspect_complaint_intensity": "medium",
            "evidence_volatility": "medium"
          },
          "consequent": "high",
          "strength": 0.62
        },
        {
          "rule_id": 8,
          "antecedents": {
            "negative_sentiment": "medium",
            "aspect_complaint_intensity": "high",
            "evidence_volatility": "medium"
          },
          "consequent": "high",
          "strength": 0.25
        },
        {
          "rule_id": 13,
          "antecedents": {
            "negative_sentiment": "low",
            "aspect_complaint_intensity": "medium",
            "evidence_volatility": "high"
          },
          "consequent": "moderate",
          "strength": 0.14
        }
      ],
      "output_degrees": {
        "low": 0.05,
        "moderate": 0.28,
        "high": 0.62,
        "critical": 0.11
      },
      "defuzzified": 62.5,
      "defuzzification_method": "centroid"
    },
    "topsis": {
      "criteria": [
        {
          "name": "sentiment_positivity",
          "description": "Proportion of positive review sentiment",
          "weight": 0.25,
          "direction": "benefit",
          "value": 0.2885,
          "normalized_value": 0.31,
          "contribution": 0.06
        },
        {
          "name": "risk_score",
          "description": "Fuzzy logic consumer risk assessment (0-100 scale, lower is safer)",
          "weight": 0.3,
          "direction": "cost",
          "value": 62.5,
          "normalized_value": 0.71,
          "contribution": 0.19
        },
        {
          "name": "evidence_strength",
          "description": "Review depth, length, and credibility indicator (0-1 scale)",
          "weight": 0.2,
          "direction": "benefit",
          "value": 0.7173,
          "normalized_value": 0.78,
          "contribution": 0.12
        },
        {
          "name": "verified_purchase_ratio",
          "description": "Ratio of reviews originating from verified purchases (0-1 scale)",
          "weight": 0.15,
          "direction": "benefit",
          "value": 0.9423,
          "normalized_value": 0.95,
          "contribution": 0.14
        },
        {
          "name": "average_rating",
          "description": "Normalized star rating average (1-5 scaled to 0-1)",
          "weight": 0.1,
          "direction": "benefit",
          "value": 0.4087,
          "normalized_value": 0.45,
          "contribution": 0.03
        }
      ],
      "closeness": 0.38,
      "rank": 19,
      "ranked_count": 21,
      "peer_count": 20,
      "percentile": 9.5,
      "risk_score": 62.5
    }
  },
  "explanation": {
    "base_value": 41.3,
    "contributions": [
      {
        "feature": "negative_sentiment_share",
        "value": 0.4423,
        "contribution": 16.8
      },
      {
        "feature": "aspect_complaint_intensity",
        "value": 0.37,
        "contribution": 14.9
      },
      {
        "feature": "verified_purchase_ratio",
        "value": 0.9423,
        "contribution": -8.4
      },
      {
        "feature": "evidence_strength",
        "value": 0.7173,
        "contribution": -5.2
      },
      {
        "feature": "review_depth",
        "value": 206.31,
        "contribution": 3.1
      },
      {
        "feature": "mean_rating",
        "value": 2.634615,
        "contribution": 0.4
      },
      {
        "feature": "rating_trend_slope",
        "value": -0.0001,
        "contribution": -0.4
      }
    ],
    "surrogate_fidelity": {
      "r2": 0.91,
      "mae": 2.34,
      "n_training_points": 2066,
      "meets_minimum": true
    },
    "reasons": [
      {
        "text": "23 of 52 reviews (44.2%) carry the weak negative label, the largest single driver of the risk score.",
        "feature": "negative_sentiment_share",
        "raw_value": 0.4423
      },
      {
        "text": "49 of 52 reviews (94.2%) are verified purchases, which lowers the risk score.",
        "feature": "verified_purchase_ratio",
        "raw_value": 0.9423
      },
      {
        "text": "The mean rating is 2.63 out of 5 with 19 of 52 reviews rated one star, which raises the risk score.",
        "feature": "mean_rating",
        "raw_value": 2.634615
      }
    ]
  },
  "alternatives": {
    "items": [
      {
        "summary": {
          "asin": "B0050O1N2U",
          "title": "Whirlpool 8273104 Knob for Range",
          "brand": "Whirlpool",
          "price": 13.99,
          "n_reviews": 108,
          "mean_rating": 4.787037,
          "split": "train"
        },
        "risk_score": 27.9,
        "verdict": "CONSIDER",
        "deltas": {
          "risk_score_delta": -34.6,
          "evidence_strength_delta": 0.06,
          "mean_rating_delta": 2.152422
        },
        "relation": "better",
        "relation_types": [
          "also_viewed"
        ]
      },
      {
        "summary": {
          "asin": "B004W7UWIM",
          "title": "AquaPlumb RKG Not Applicable Gas Range Knob- 5PC/Card",
          "brand": "Aqua Plumb",
          "price": 10.43,
          "n_reviews": 318,
          "mean_rating": 2.216981,
          "split": "train"
        },
        "risk_score": 71.9,
        "verdict": "CONSUMER RISK",
        "deltas": {
          "risk_score_delta": 9.4,
          "evidence_strength_delta": -0.11,
          "mean_rating_delta": -0.417634
        },
        "relation": "worse",
        "relation_types": [
          "also_viewed"
        ]
      }
    ],
    "empty_reason": null,
    "total_related": 12,
    "usable": 11
  },
  "verdict": {
    "label": "CONSIDER WITH CAUTION",
    "risk_score": 62.5,
    "evidence_strength": 0.7173,
    "major_concerns": [
      {
        "code": "high_negative_sentiment",
        "text": "23 of 52 reviews (44.2%) carry the weak negative label, well above the 25% caution band.",
        "severity": "high",
        "supporting_numbers": {
          "negative_share": 0.4423,
          "negative_count": 23.0,
          "total_reviews": 52.0
        }
      },
      {
        "code": "low_mean_rating",
        "text": "The mean rating is 2.63 out of 5, with 19 of 52 reviews rated one star.",
        "severity": "medium",
        "supporting_numbers": {
          "mean_rating": 2.634615,
          "one_star_count": 19.0,
          "one_star_share": 0.3654
        }
      }
    ],
    "rationale": "Consider with caution: 23 of 52 reviews (44.2%) are weakly negative and the mean rating is 2.63 out of 5, but the evidence base is comparatively strong (49 of 52 verified purchases) and a safer alternative exists."
  },
  "risk_score": 62.5,
  "evidence_strength": 0.7173,
  "pipeline_version": "0.1.0",
  "generated_at": "2026-10-10T11:31:52Z",
  "data_basis": {
    "category": "Appliances",
    "n_reviews": 52,
    "date_range_start": "2009-07-20T00:00:00Z",
    "date_range_end": "2018-02-03T00:00:00Z",
    "reference_date": "2018-10-04T00:00:00Z"
  },
  "warnings": [
    "Sentiment labels are weak supervision from star ratings (1-2 negative, 3 neutral, 4-5 positive), not model sentiment.",
    "Evidence base of 52 reviews is below the category median of 111 reviews."
  ]
}
```


### `GET /products/{asin}/alternatives`

Evidence-based alternatives for one product.

- `total_related` counts related-product edges (an ASIN linked by both `also_bought` and `also_viewed` counts twice); `usable` counts distinct alternatives that could actually be compared.
- About 18% of products in this category have no usable alternative: `items` is then empty and `empty_reason` says why.
- Deltas are `alternative - subject` (a negative risk delta means safer).

Example response `200` (schema `AlternativesResult`):

<!-- schema: AlternativesResult -->

```json
{
  "items": [
    {
      "summary": {
        "asin": "B0050O1N2U",
        "title": "Whirlpool 8273104 Knob for Range",
        "brand": "Whirlpool",
        "price": 13.99,
        "n_reviews": 108,
        "mean_rating": 4.787037,
        "split": "train"
      },
      "risk_score": 27.9,
      "verdict": "CONSIDER",
      "deltas": {
        "risk_score_delta": -34.6,
        "evidence_strength_delta": 0.06,
        "mean_rating_delta": 2.152422
      },
      "relation": "better",
      "relation_types": [
        "also_viewed"
      ]
    },
    {
      "summary": {
        "asin": "B004W7UWIM",
        "title": "AquaPlumb RKG Not Applicable Gas Range Knob- 5PC/Card",
        "brand": "Aqua Plumb",
        "price": 10.43,
        "n_reviews": 318,
        "mean_rating": 2.216981,
        "split": "train"
      },
      "risk_score": 71.9,
      "verdict": "CONSUMER RISK",
      "deltas": {
        "risk_score_delta": 9.4,
        "evidence_strength_delta": -0.11,
        "mean_rating_delta": -0.417634
      },
      "relation": "worse",
      "relation_types": [
        "also_viewed"
      ]
    }
  ],
  "empty_reason": null,
  "total_related": 12,
  "usable": 11
}
```

Empty-alternatives response `200` (schema `AlternativesResult`):

<!-- schema: AlternativesResult -->

```json
{
  "items": [],
  "empty_reason": "no related-product edges in the processed subset (0 also_bought / also_viewed links)",
  "total_related": 0,
  "usable": 0
}
```


## Errors

Every non-2xx response uses the same envelope (`docs/schemas/error_response.json`):

<!-- schema: ErrorResponse -->

```json
{
  "code": "PRODUCT_NOT_FOUND",
  "message": "Product 'B0000000000' was not found in the serving cache.",
  "detail": {
    "asin": "B0000000000"
  },
  "request_id": "req-5f0a1c2d"
}
```

`code` is a stable token you can branch on; `detail` carries machine-readable particulars and may be `null`; `request_id` correlates with the server log when available.

| Status | `code` | When |
|---:|---|---|
| 404 | `PRODUCT_NOT_FOUND` | The ASIN is not in the processed subset (products, reviews, analysis or alternatives endpoints). |
| 422 | `VALIDATION_ERROR` | A query parameter has an unusable value (for example `label=angry`, `limit=-1`, `verified=maybe`). |
| 500 | `INTERNAL_ERROR` | An unexpected server-side failure (the real reason is only in the server log, never in the response). |

Error payload examples for each status code:

- `HTTP 404` `PRODUCT_NOT_FOUND`:

<!-- schema: ErrorResponse -->

```json
{
  "code": "PRODUCT_NOT_FOUND",
  "message": "Product 'B0000000000' was not found in the serving cache.",
  "detail": {
    "asin": "B0000000000"
  },
  "request_id": "req-5f0a1c2d"
}
```

- `HTTP 422` `VALIDATION_ERROR`:

<!-- schema: ErrorResponse -->

```json
{
  "code": "VALIDATION_ERROR",
  "message": "Unknown query parameter value.",
  "detail": {
    "field": "label",
    "value": "angry",
    "allowed": [
      "negative",
      "neutral",
      "positive"
    ]
  },
  "request_id": "req-0a41b7e9"
}
```

- `HTTP 500` `INTERNAL_ERROR`:

<!-- schema: ErrorResponse -->

```json
{
  "code": "INTERNAL_ERROR",
  "message": "Internal server error.",
  "detail": null,
  "request_id": "req-77c301aa"
}
```

## Ambiguity decisions

Where the task brief left room, the simplest option was chosen and recorded here.

1. **`/health` and `/meta/category` models live in `common.py`.** They describe the service, not a product, review or analysis, so no dedicated area module owns them.
2. **`fuzzy.output_degrees` is a mapping** keyed by the fuzzy output terms in `configs/fuzzy_rules.yaml` (`low`, `moderate`, `high`, `critical`) instead of three hard-coded fields, so the schema cannot drift from the configured rule base.
3. **`topsis.percentile` is 0-100**, defined as `100 * (ranked_count - rank) / ranked_count`: 100 means the safest product in the ranked set, almost 0 the riskiest. It is the one non-risk statistic on a percentage scale.
4. **`data_basis.date_range_start/end` are per-product** (that product's own oldest and newest review) while `reference_date` stays the dataset-wide latest review date, so recency is comparable across products.
5. **`total_related` counts edges, `usable` counts distinct alternatives.** An ASIN linked by both `also_bought` and `also_viewed` counts twice in `total_related` but once in `usable`.
6. **Review text is served raw** (`review_text` reads the stored raw body) while `summary` is the cleaned headline, exactly as the cache stores them.
7. **`evidence.details` and `risk_indicators` are extension points, not fixed fields.** Evidence signals added later (rating-distribution entropy, helpfulness-weighted sentiment, ...) and the risk stage's five criteria (`product_quality`, `evidence_strength`, `rating_reliability`, `negative_feedback_concentration`, `price_position`) travel as lists of the same 0-1 indicator shape, so growing the contract never breaks existing clients. A `details` name must be unique in the list and may not reuse `verified_purchase`, `helpfulness_ratio` or `review_depth` - those already have dedicated fields, so a repeat would be ambiguous; both violations are rejected by the schema. `risk_indicators` defaults to an empty list until the risk stage computes it.
