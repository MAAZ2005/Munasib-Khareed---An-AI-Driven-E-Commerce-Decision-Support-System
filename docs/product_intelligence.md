# Product Intelligence

What this module does, exactly. Every formula, tie rule, threshold and
missing-data rule below is implemented in `src/munasib/product_intel/` and
covered by `tests/unit/test_product_intel*.py`. No number here is illustrative
- each one is either computed from the stored data or declared in
`configs/config.yaml`.

Product Intelligence is the first stage of the analysis pipeline: it turns a
product's **metadata** (brand, price, sales rank, description, feature bullets,
related-product links) into typed indicators that the risk stage
(`src/munasib/risk/`) consumes alongside the review evidence from
`review_intel`. It never reads a review, and it never invents a value.

---

## 1. Files

| File | Responsibility |
|---|---|
| `src/munasib/product_intel/peers.py` | `PeerStats`: peer distributions and the tie-ruled percentile lookups, built once per candidate set; plus the breadcrumb-prefix walk (`price_peer_selection`) that picks each product's price peer group. |
| `src/munasib/product_intel/profile.py` | `ProductProfile` (typed dataclass) and `build_profile()`. |
| `src/munasib/product_intel/features.py` | Persistence into `product_features`, with the dataset build timestamp. |
| `scripts/compute_product_features.py` | CLI: `--limit`, `--asins`, progress output, summary of counts. |
| `configs/config.yaml` → `product_intel` | Every threshold and the metadata field list. |

Two repository methods were added to `src/munasib/db/repositories.py` for this
stage (both keep SQL out of the callers):

* `ProductRepository.list_all_products()` - the whole candidate set, ASIN-ordered.
* `ProductRepository.get_products(asins)` - a batch point-get that never raises
  for an unknown ASIN (the caller decides).
* `RelatedRepository.count_related(asins)` - one grouped query for the distinct
  related-product counts of many products at once.

---

## 2. Thresholds (`configs/config.yaml` → `product_intel`)

| Key | Default | Meaning |
|---|---:|---|
| `target_description_words` | 100 | A description this long (or longer) scores 1.0 completeness. |
| `target_feature_count` | 5 | A feature list with this many bullets (or more) scores 1.0. |
| `min_brand_peers` | 5 | A brand needs at least this many *other* priced products before a brand price percentile is reported. |
| `min_rank_peers` | 5 | A rank category needs at least this many *other* ranked products before a rank percentile is reported. |
| `min_price_peers` | 10 | A category-breadcrumb prefix must hold at least this many *priced* products (the subject among them) to be used as its price peer group; see §4.2.1. |
| `metadata_fields` | `title, brand, price, rank, description, features` | The exact fields whose share of presence is `metadata_completeness`. |

These defaults were chosen against the real subset (Appliances, 2,066
products): the median description is 49 words and the median feature list is 5
bullets, so the two targets sit just above the median and separate "thin
listing" from "normal listing" without rewarding padding. With
`min_brand_peers: 5`, 42 brands (1,252 products) qualify for a brand percentile;
with `min_rank_peers: 5`, 36 rank categories qualify (a category needs at
least 6 ranked products so that 5 of them are *other* peers). With
`min_price_peers: 10`, **2,040 of 2,066 products (98.7%)** are compared inside a
genuine category group (their leaf breadcrumb or a shorter prefix of it) rather
than against the whole 1,678-product price distribution; §4.2.1 gives the rule
and the depth breakdown.

Validation (`ProductIntelConfig`) rejects a non-positive target, an empty field
list, a repeated field, or a field name outside the vocabulary
`title / brand / price / rank / description / features`, so a typo in config
fails loudly instead of silently scoring 0.

---

## 3. The tie rule (one rule, used everywhere)

Every percentile in this module is the same statistic. For a distribution of
*other* products' values, sorted ascending:

```
percentile(subject) = ( #{other values <= subject value} ) / ( #{other values} )
```

Implemented as `(bisect_right(sorted_values, value) - 1) / (len(sorted_values) - 1)`,
where the `- 1` removes the subject's own row from both sides, because **a
product is never its own peer**.

Consequences, all deliberate:

* **Ties resolve towards the subject.** If three products share the lowest
  price, each of them scores `2 / (n - 1)`, not `0`. Equal values get equal
  credit; there is no arbitrary tie-break on ASIN.
* **0 is always the favourable end and 1 the unfavourable one**, for both
  price and rank:
  * `price_percentile_category = 0` → cheapest priced product in its **price
    peer group** (§4.2.1) - the browse-node prefix it was compared against, or
    the whole candidate set when there is no usable prefix.
  * `rank_percentile = 0` → best-selling rank in its category (a *lower*
    `rank_value` is a better seller, and the `<=` counts worse-or-equal ranks).
  * This is exactly the orientation `price_position` expects in
    `configs/criteria_weights.yaml` ("0 means the cheapest price relative to
    peers").
* Percentiles are always in `[0, 1]`; the implementation clamps into that
  range defensively, and a test asserts the bound over the real database.
* The subject must belong to the candidate set that built the `PeerStats`,
  otherwise its own value is not in the array and the `- 1` would be wrong.
  The CLI therefore always builds `PeerStats` from **all** products, even when
  `--limit`/`--asins` profile a slice (a percentile against a truncated peer set
  would silently change meaning).

**Why not a strict `<`?** A strict comparison would make the cheapest of `k`
tied-cheapest products score `0`, which is indistinguishable from "cheapest of
them all" - the tie rule above keeps the distinction visible in the number.

---

## 4. Indicators

All of these land in `product_features.indicators` (JSON, one row per ASIN) via
`ProductProfile.to_indicators()`. Keys are stable and documented in
`PROFILE_INDICATOR_KEYS`.

### 4.1 Brand

* `brand` = **normalized brand key**: `products.brand_clean` when present,
  otherwise `munasib.data.cleaning.normalize_brand(products.brand)` (trim,
  collapse whitespace, strip edge punctuation, lower-case). `None` when there is
  no usable brand. An empty-string brand (`''`, stored with
  `brand_missing=True` on 8 rows) is **not** a brand.
* `brand_missing` = `brand is None`.
* `brand_peer_count` = number of **other** candidate products sharing the same
  normalized brand (`0` if the brand is unique to this product or absent).

### 4.2 Price

* `price` / `price_max` / `price_status` copied from the row; `None` when the
  price is missing or unparseable (`price_status` records which: `missing`,
  `range`, `single`, `unparseable`).
* `price_missing` = `price is None`. In this subset that is **388 of 2,066
  products (18.8%)**, so the `None` path is a first-class case, not a corner.
* `price_percentile_category` = the §3 percentile over the **price peer group**
  chosen for this product (§4.2.1), not over the whole category.
  `0` = cheapest in that group, `1` = most expensive. `None` +
  `price_percentile_category_missing=True` when the product has no price, or when
  its group holds no other priced product.
* `price_peer_group` = the breadcrumb prefix the percentile was measured
  against, or the literal `all products` when the whole candidate set was used.
  A string, never `None`: a product with no breadcrumb still has a comparison
  set, and naming it is what keeps the number interpretable.
* `price_peer_count` = priced products in that group. The subject is one of
  them whenever it has a price, so this is both the group's size and the
  denominator basis of the percentile.
* `price_peer_fallback` = `True` when the product's own full breadcrumb was
  **not** the group it was compared against (a shorter prefix was used, or the
  whole set). This is the honesty flag: a `0.4` measured over a fallback group
  is a weaker statement than the same `0.4` measured inside the leaf.
* `price_percentile_brand` = the §3 percentile over the **same-brand priced**
  products only. `None` + `price_percentile_brand_missing=True` when the
  product has no price, no brand, or the brand has fewer than `min_brand_peers`
  other priced products. Measured on the real subset: **1,252 of 2,066 products
  (60.6%)** carry one; the rest honestly report `None`.

#### 4.2.1 How the price peer group is chosen

The 2018 metadata carries a browse breadcrumb per product - a list of node
names ordered broad to specific, which the build stores as `category_path`
(`" > "`-joined) plus its last level as `category_leaf` (§`docs/dataset_notes.md`).
In this subset:

| Breadth | Products |
|---|---:|
| `category_path` present | 2,040 of 2,066 (98.7%) |
| no breadcrumb published | 26 (1.3%) |
| depth 2 (e.g. `Appliances > Water Filters`) | 172 |
| depth 3 (e.g. `Appliances > Parts & Accessories > Dryer Parts & Accessories`) | 909 |
| depth 4 (e.g. `... > Dryer Parts & Accessories > Vents`) | 959 |

The rule, implemented in `PeerStats.price_peer_selection`:

1. Take the product's own `category_path` and list its **level-boundary
   prefixes**, most specific first: `A > B > C`, then `A > B`, then `A`. Matching
   is on joined levels, so `Appliances > Parts` is never a prefix of
   `Appliances > Parts & Accessories` - the shared leading text is a coincidence
   of the node names, not a shared browse node.
2. Use the first prefix holding at least `min_price_peers` (**10**) *priced*
   products, counting the subject itself when it has a price.
3. If even the root prefix is too small, or the product has no breadcrumb at
   all, fall back to the **whole candidate set** (`price_peer_group = all
   products`, `price_peer_fallback = True`).

Prefix arrays are built by pushing each priced product's price onto the prefix
list of its own breadcrumb, so a prefix group is exactly "the priced products
whose breadcrumb passes through that node" - the same level-boundary rule, on
the build side.

**Why 10, and why the comparability argument.** A percentile over a handful of
products is noise dressed as a number: with 4 peers every product sits at
`0`, `0.33`, `0.67` or `1`, and a `0.75` tells a shopper nothing about whether a
$29 water filter is ordinary. Ten priced products is the smallest group whose
percentile still separates "cheap" from "mid-range" without the steps being
visible, and it is small enough that the browse node stays *comparable* - all
ten are water filters, not a mixture of water filters, refrigerators and dryer
vents. The fallback exists for the opposite failure: refusing to compare a
product at all would leave the risk stage with no price signal, so it widens
one level at a time and reports that it did.

On this subset the rule resolves as:

| Peer group | Products | Share |
|---|---:|---:|
| leaf breadcrumb (no fallback) | 1,826 | 88.4% |
| shorter prefix (fell back one or more levels) | 214 | 10.4% |
| `all products` (no breadcrumb, or nothing small enough) | 26 | 1.3% |

The 214 fallbacks split by the depth of the group actually used: 105 were
compared at depth 3, 85 at depth 2 and 24 at the depth-1 root itself. Falling
back is common because the deep nodes are small - 172 products sit at depth 2,
and a depth-2 node rarely holds 10 priced products once the 388 unpriced rows
are removed. Group sizes confirm the rule does what it says:
`price_peer_count` runs 10 (the configured floor, hit by 33 products) to 1,678
(the whole priced set), median 155. Every fallback is visible as
`price_peer_fallback = True` on the row, so nothing about the comparison is
implicit.

### 4.3 Sales rank

* `rank_value` / `rank_category` copied from the row; `rank_missing` =
  `rank_value is None` (13 products here).
* `rank_percentile` = the §3 percentile over the other ranked products of the
  **same `rank_category`**. `0` = best seller in that category, `1` = worst.
  `None` + `rank_percentile_missing=True` when the product has no rank, or when
  the category has fewer than `min_rank_peers` other ranked products.

> The rank is the **2018 crawl's** rank (the dataset is a 2018 snapshot; review
> dates stop at 2018-10-04). Nothing here uses today's date. A rank category is
> an Amazon browse node, so "peers" means *products sharing that node*, which is
> narrower than the whole category - that is why the category percentile
> (`price_percentile_category`) is the primary price signal and the rank is
> supporting evidence.

### 4.4 Completeness

* `description_completeness` = `min(1, description_word_count / target_description_words)`,
  where the word count is `len(description.split())` over the cleaned
  description. A missing description scores `0.0` (never a partial guess) and
  `description_missing=True`.
* `feature_completeness` = `min(1, feature_bullet_count / target_feature_count)`,
  counting only **present, non-blank** bullets of the JSON list. Missing or
  empty → `0.0` + `features_missing=True`.
* `metadata_completeness` = `present_fields / len(metadata_fields)` - an
  unweighted mean over exactly the fields the config names. Presence per field:
  * `title` - `title_clean` or `title` non-empty;
  * `brand` - the normalized brand is not `None`;
  * `price` - `price is not None`;
  * `rank` - a parsed `rank_value` exists (an unparsable rank string carries no
    rank to reason about);
  * `description` - non-empty;
  * `features` - at least one usable bullet.
* `metadata_missing_fields` = the absent field names, in config order, so a UI
  can say *which* field is missing instead of only *how much*.

### 4.5 Related-product counts

Per **distinct related product**, never per edge (a target linked by both
`also_bought` and `also_viewed` counts once in each field), self-references
excluded - the same contract the alternatives endpoint uses:

* `n_also_bought`, `n_also_viewed` - distinct targets per relation type.
* `n_usable_alternatives` - distinct targets that are themselves in the
  processed subset (so it can never exceed the other two; this is what TOPSIS
  can actually compare against).

A product with no edges gets `0 / 0 / 0` (never an error, never a guess).

---

## 5. Missing data, in one table

| Situation | Value | Flag |
|---|---|---|
| No price | `price = None` | `price_missing = True` |
| No price | `price_percentile_category = None` | `price_percentile_category_missing = True` |
| No price / no brand / brand too small | `price_percentile_brand = None` | `price_percentile_brand_missing = True` |
| No rank / category too small | `rank_percentile = None` | `rank_percentile_missing = True` |
| No rank | `rank_value = None` | `rank_missing = True` |
| No brand | `brand = None` | `brand_missing = True` |
| No breadcrumb published (26 products) | `price_peer_group = "all products"` | `price_peer_fallback = True` |
| No description | `description_completeness = 0.0` | `description_missing = True` |
| No features | `feature_completeness = 0.0` | `features_missing = True` |
| No related rows | `n_* = 0` | - (zero edges is a fact, not missing data) |

A completeness of `0.0` means "nothing to measure", which is why it always
carries its flag: downstream code must read the flag, not the zero. Nothing is
ever imputed from a peer, a median or a default.

---

## 6. Determinism and idempotence

* `PeerStats.from_products` sorts by ASIN and builds sorted arrays, so the same
  rows always yield the same distributions. The price peer group inherits this:
  `price_peer_selection` depends only on the ASIN-sorted breadcrumbs and the
  configured `min_price_peers`, so row order cannot change which group a
  product lands in (asserted by
  `test_price_peer_selection_is_deterministic_across_input_order`).
* `build_profiles` returns profiles ascending by ASIN.
* `write_product_features` stamps every row with `computed_at` =
  `built_at` from `data/processed/meta.json` (**the dataset build timestamp,
  never the wall clock**), converted to naive UTC per the database contract.
  Two runs over the same cache therefore write byte-identical rows.
* Writes are **upserts on the `asin` primary key**, so re-running refreshes
  rather than duplicates.
* The stored JSON is written with sorted keys, so a diff between two runs is
  empty.

`read_build_timestamp` refuses to run when `meta.json` is missing or declares
no `built_at` - it never falls back to the current time, because that would
break the determinism guarantee silently.

---

## 7. Running it

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/compute_product_features.py                 # whole candidate set
python scripts/compute_product_features.py --limit 50      # first 50 by ASIN
python scripts/compute_product_features.py --asins B000050JN4 B00004T8A4
```

Prerequisite: the serving cache must exist
(`python scripts/build_dataset.py && python scripts/ingest_db.py`). Exit code
`0` means profiles were written; `1` means the run could not start (bad config,
unreadable cache, missing `meta.json`, or an `--asins` entry that is not in the
cache).

The summary printed at the end reports **counts and percentile statistics
only** - profiles written, how many lack price/rank/brand, how many carry a
brand percentile, and the min/median/max of each non-`None` percentile. A
percentile that is `None` for every profile is omitted, not reported as zero.

Measured on the current cache ( Appliances, 2,066 products):

```
profiles written            : 2,066
without price               : 388
without rank                : 13
without brand               : 8
with brand percentile       : 1,252
price peer fell back        : 240
peer group depth 1          : 24
peer group depth 2          : 255
peer group depth 3          : 926
peer group depth 4          : 835
peer group all products     : 26
price_percentile_brand      : min=0.0000 median=0.5037 max=1.0000
price_percentile_category   : min=0.0000 median=0.4945 max=1.0000
rank_percentile             : min=0.0000 median=0.5000 max=1.0000
```

The `price peer fell back` line counts products whose `price_peer_fallback`
flag is set - `214` of them fell back to a **shorter breadcrumb prefix** and
`26` to the whole candidate set (the products with no breadcrumb at all). The
four `peer group depth` lines bucket every product by the depth of the group it
was actually compared against, so `24 + 255 + 926 + 835 + 26 = 2,066`; §4.2.1
explains why a product lands in a shallower bucket than its own breadcrumb.

The medians sitting near 0.5 are the expected shape of an ECDF over its own
population; the medians are not asserted by tests (they depend on the data),
only the bounds and the missing-value rules are. Note that
`price_percentile_category`'s median moved from `0.5057` to `0.4945` when p11b
switched it from the whole-category distribution to the per-group one: the
statistic is the same tie-ruled ECDF, but measured against a smaller, genuinely
comparable reference set.

---

## 8. How this feeds the risk stage

`configs/criteria_weights.yaml` declares six TOPSIS criteria. This module
supplies one of them directly:

```yaml
price_position:
  description: "Price relative to peer products, 0-1 where 0 means the cheapest price relative to peers"
  weight: 0.10
  type: "cost"
```

* **`price_percentile_category` is the value fed to `price_position`.** It is
  already in the criterion's exact orientation (0 = cheapest relative to peers,
  1 = priciest, `type: cost`), so no sign flip or rescaling is needed at the
  decision layer. "Peers" now means the product's **price peer group** (§4.2.1)
  rather than the whole category, which is what makes the criterion comparable:
  a $9 water filter is scored against other water filters, not against
  refrigerators. `price_peer_group`, `price_peer_count` and
  `price_peer_fallback` travel with it so the decision layer can tell how
  specific that comparison was - and can treat a fallback-group score as weaker
  evidence than a leaf-group one.
* **A product without a price cannot be scored on this criterion.** The
  criteria file says so explicitly ("a product without a usable price must be
  handled explicitly later... the price is never imputed and never defaulted to
  0"). The decision stage must therefore either exclude the product from
  `price_position` or report it as unknown - it must **not** substitute 0.5, 0
  or a peer median. `price_percentile_category_missing` is the flag that tells
  it which case it is in.
* `metadata_completeness`, `description_completeness` and
  `feature_completeness` feed the **evidence-strength** reasoning as listing
  quality signals (a thin listing is weaker evidence than a full one), and
  `n_usable_alternatives` bounds what the alternatives stage can compare
  against.
* `rank_percentile` and `price_percentile_brand` are supporting indicators: the
  rank speaks to popularity, the brand percentile to whether the product is
  priced oddly *for its brand* (a useful sanity signal when the category
  percentile alone would hide a brand-specific anomaly).

Because every indicator is either a real number in `[0, 1]` or `None` with a
flag, the fuzzy stage can consume the present ones and the narrative can cite
the absent ones explicitly - which is the whole point of the module.
