# Aspects: extraction, vocabulary and per-product statistics

The aspect stage turns each review into a short list of product attributes
(`noise`, `ice maker`, `delivery`, ...) and then asks whether those attributes
are problems or strengths for a product. Every number in this document is
produced by the scripts below from the processed subset. Nothing is hand-copied
or fabricated.

## Pipeline

```
reviews.review_text_clean
        |
        v
spaCy noun chunks  ->  resumable candidate parts (raw, stoplist not applied)
        |
        v
category vocabulary  (reviews >= min_review_count)
        |
        v
review_features.aspects_json  (filtered per review)
        |
        v
product_aspects  (counts, shares, lift, classification, examples)
```

The expensive spaCy pass is cached. Extending the generic-noun stoplist only
rebuilds the vocabulary and per-review filter; it does not re-run spaCy.
Aggregation additionally drops self-reference rows (see *Self-reference
exclusion* below): only `product_aspects` is filtered, never
`review_features.aspects_json`.

## Commands

```powershell
# 1. Extract candidates (resumable) and print the top 100 / 50 near the cutoff
python scripts/compute_aspects.py --vocabulary

# After an interruption:
python scripts/compute_aspects.py --vocabulary --resume

# 2. Filter cached candidates into review_features.aspects_json
python scripts/compute_aspects.py --reviews

# Development only: process the first N reviews
python scripts/compute_aspects.py --reviews --limit 1000

# 3. Aggregate into product_aspects
python scripts/compute_aspects.py --aggregate

# Or all three, reusing completed candidate parts
python scripts/compute_aspects.py --all

# 4. Manual quality audit: export blank labels, then evaluate later
python scripts/export_aspect_audit_sample.py
python scripts/export_aspect_audit_sample.py --evaluate
```

Exit codes for `compute_aspects.py`:

| Code | Meaning |
| --- | --- |
| 0 | Stage completed |
| 1 | Configuration/data failure |
| 2 | Candidate-cache conflict: missing parts without `--resume`, or a fingerprint mismatch |

`--limit` is accepted only with `--reviews`. `--all --limit` is rejected because
a partial per-review pass cannot support a whole-category aggregation.

## Extraction

spaCy model: `en_core_web_sm`, with NER disabled. The required components are
the tagger, parser, attribute ruler and lemmatizer.

For each parsed review prefix (`max_text_chars`, default 5000):

1. Every spaCy noun chunk is canonicalised.
2. Nominal tokens (`NOUN`, `PROPN`) are collected from the **right** (head) end,
   so a long chunk keeps its most specific tail: `the stainless steel ice maker`
   becomes `ice maker`.
3. At most `max_aspect_tokens` tokens (default 2) are kept. Tokens shorter than
   `min_token_chars` (default 2), non-alphabetic tokens and numerals are dropped.
4. Generic nouns are dropped when the public extractor runs. The candidate cache
   stores raw candidates so a stoplist change is cheap.
5. Nouns the chunker did not claim are added as single-token candidates.
6. Duplicates collapse: a review that says `noise` twice contributes one
   mention.

A mention is therefore one **(review, aspect)** pair, not one occurrence.

### Stoplist

`review_intel.aspects.generic_nouns` excludes vague nouns, person/family words
and review-meta words (`product`, `item`, `review`, `star`, `rating`, `amazon`,
`husband`, `wife`, `son`, `daughter`, `mom`, `dad`, `friend`, `customer`,
`buyer`, `stuff`, `bit`, `way`, `kind`, `issue`, ...).

Service terms are deliberately kept: `delivery`, `shipping`, `seller`,
`packaging`, `box`, `service`.

Because the candidate cache is stoplist-free, the vocabulary build and the
per-review filter both re-canonicalise every cached phrase under the *current*
stoplist (`filter_aspect_phrase`): `customer service` collapses to `service`,
`year warranty` to `warranty`, and phrases left with only generic tokens
(`amazon review`, `star rating`, `owner manual`) are dropped or reduced to
their head exactly as a live extraction would have produced. This is what makes
a stoplist extension complete without re-running spaCy.

## Vocabulary

An aspect enters the category vocabulary when it appears in at least
`min_review_count` **reviews** (default 50). `top_k_aspects` (default 10) is the
dashboard display cap.

The `--vocabulary` run writes:

* `data/interim/aspect_vocabulary.parquet` - machine-readable vocabulary.
* `reports/aspect_vocabulary.csv` - ranked human-readable output
  (`rank, aspect, review_count, product_count`).

It prints the top 100 aspects and the 50 aspects nearest the cutoff, in both
directions, so junk can be spotted before the review/aggregation stages.

## Aspect polarity is review-level

Each aspect inherits the **review's** stored predicted sentiment label
(0 negative, 1 neutral, 2 positive). This is not sentence-level aspect
sentiment. A review that praises the motor in one sentence and complains about
the filter in another gives both aspects the same review label. This limitation
is why the manual audit exists; precision is never claimed without annotations.

Reviews without a stored prediction are excluded from polarity counts rather
than assigned a guessed label. They can still have an aspect list in
`review_features`, but aggregation only uses predicted reviews.

## Per-product rules

For each product and aspect:

* `mention_count` - predicted reviews that mention the aspect.
* `negative_count`, `neutral_count`, `positive_count` - those mentions split by
  the review label.
* `negative_share = negative_count / mention_count`, similarly neutral/positive.
* `net_sentiment = (positive_count - negative_count) / mention_count` in [-1, 1].
* `product_labeled_reviews` - predicted reviews of the product (the baseline
  denominator).
* `product_negative_share = product_negative_count / product_labeled_reviews`.

Only aspects with `mention_count >= min_mentions` (default 5) are stored.

### Negativity lift

Using the product's own negative rate as the baseline and a smoothing
pseudo-count:

```
smoothed_negative_share = (negative_count + lift_smoothing * baseline)
                          / (mention_count + lift_smoothing)
negativity_lift = smoothed_negative_share / baseline
```

`baseline = product_negative_share`, `lift_smoothing` defaults to 5.0.
`negativity_lift` is `NULL` when the product has no predicted-negative reviews
or there are no mentions. It is never replaced with a fabricated value.

Lift above 1 means the aspect appears in negative reviews more often than the
product's overall negative rate; below 1 means less often.

### Classification

All four pain-point conditions must hold:

1. `mention_count >= min_mentions`
2. `negativity_lift >= pain_point_min_lift` (default 1.5)
3. `negative_count >= pain_point_min_negative_mentions` (default 3)
4. `negative_share >= pain_point_min_negative_share` (default 0.25)

Otherwise, a strength requires:

1. `mention_count >= min_mentions`
2. `positive_share >= strength_min_positive_share` (default 0.75)
3. `negativity_lift is NULL or negativity_lift <= strength_max_negative_lift`
   (default 0.75)

Everything else is `neutral` and is still stored.

Consequences pinned by tests:

* A product with a 1% negative rate and an aspect with 5 mentions / 1 negative is
  **not** a pain point (share and count both fail).
* The same product with 6 mentions / 4 negative **is** a pain point.
* A pain point can have more positive than negative mentions; the count/share/
  lift conjunction decides, not the majority label.
* A product with no negative reviews and a high positive share can still be a
  strength (lift is undefined rather than zero).

### Examples

Examples follow the classification:

| Classification | Source reviews |
| --- | --- |
| `pain_point` | Most helpful **negative** reviews |
| `strength` | Most helpful **positive** reviews |
| `neutral` | Most helpful reviews of the dominant label (ties: negative, neutral, positive) |

Helpfulness is `helpful_votes` descending, then `review_id` ascending. Each
example is an **exact prefix** of at most `example_max_chars` (default 240)
characters. No ellipsis is appended. `max_examples` defaults to 3.

### Self-reference exclusion

Aspects that merely restate the product itself are not features: reviewers
name the product when they complain, so `filter` and `mwf filter` for a
water-filter product correlate with complaints without describing anything
about them. When `review_intel.aspects.exclude_self_reference` is `true`
(the default), an aspect is dropped from **that product's**
`product_aspects` rows when **every** token of the aspect appears in the
product's self-reference token set:

1. `title_clean`, `brand` and `category_leaf` are lemmatized with the same
   spaCy pipeline and `token.lemma_` the extractor uses; lowercase
   alphabetic lemmas of at least `min_token_chars` form the set.
2. The aspect is excluded only if all of its tokens are in that set.

Example: a product titled `...Refrigerator Water Filter` yields the set
`{refrigerator, water, filter, ...}` (plus `ge` from the brand and the
category leaf tokens), so `filter`, `water filter` and `mwf filter` are
excluded while `water flow` (`flow` is not in the set) and `fit` are kept.

The lemmatized sets are cached in
`review_intel.aspects.self_reference_cache_path`
(`data/interim/product_self_tokens.json`) under a fingerprint of the
lemmatizer settings and a hash of the product metadata; a re-run reads the
cache instead of loading spaCy. The `--aggregate` report prints whether the
exclusion is on, how many rows it removed, the top 20 excluded aspects, and
before/after counts of pain points, strengths, neutral rows and products
with at least one pain point.

**Limits** - the rule is a token-set test, not semantic understanding:

* A feature word that also occurs in the title is hidden for that product
  (a blender titled `Whisper Quiet Blender` loses `quiet`).
* Brand tokens drop brand discussion (`ge` for GE products); category
  tokens drop category-generic words for products whose category names
  them.
* Only stored product rows are filtered. `review_features.aspects_json`
  and the candidate cache are never modified, so setting the flag to
  `false` and re-running `--aggregate` restores every row.

## Storage

`product_aspects` is keyed by `(asin, aspect)` and references `products.asin`.
It stores counts, shares, net sentiment, the product baseline, lift,
classification, examples JSON and `computed_at`.

`computed_at` is the dataset build timestamp from
`data/processed/meta.json` (`built_at`), not the wall clock, so re-running over
the same cache writes identical rows. `scripts/ingest_db.py` deletes the table
with the rest of the derived cache; it prints the exact recompute commands
before doing so.

## Audit

`scripts/export_aspect_audit_sample.py` draws a deterministic uniform reservoir
sample of `(review, aspect)` mentions and writes
`reports/aspect_audit_sample.csv` in long format. The annotation columns are
`is_correct` and `notes`, both blank.

No precision is printed until a human fills `is_correct` with `yes`/`no` for
every row. Then:

```powershell
python scripts/export_aspect_audit_sample.py --evaluate
```

The evaluation refuses blank or invalid annotations and reports
`correct / labelled` only after that check.
