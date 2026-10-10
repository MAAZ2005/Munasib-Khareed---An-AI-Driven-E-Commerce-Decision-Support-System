# Data Directory

Working data for Munasib Khareed. **Nothing in here is committed to git** (only the
`.gitkeep` placeholders keep the structure) - the raw UCSD Amazon Review Data files are
redistributed by the UCSD Multi-scale Video Learning Lab under their own terms and are
simply too large for the repository.

| Folder | Contents | Produced by |
|---|---|---|
| `raw/` | Downloaded ``.json.gz`` source files (JSON Lines) | `python scripts/download_data.py` |
| `interim/` | Intermediate normalised extracts | future pipeline stages |
| `processed/` | Clean parquet tables (`products`, `reviews`, `related_products`) + `meta.json` | `python scripts/build_dataset.py` |
| `db/` | SQLite serving cache (`serving_cache.sqlite`) | pipeline loader |

## Getting the raw data

The active category comes from `dataset.category` in `configs/config.yaml`
(currently **Appliances**; fallback registered: `Luxury_Beauty`).

```bash
# downloads (resumable, gzip-verified) and skips files that are already complete
python scripts/download_data.py

# explicit category (e.g. after switching dataset.category)
python scripts/download_data.py --category Luxury_Beauty
```

Target files land in `data/raw/`:

- `Appliances.json.gz` + `meta_Appliances.json.gz`
- `Luxury_Beauty.json.gz` + `meta_Luxury_Beauty.json.gz`

If the UCSD host is unreachable, the script prints the exact URLs, target paths and a
ready-made `curl` command for a manual download; re-run it afterwards to verify.

## Building the processed dataset

```bash
python scripts/build_dataset.py   # raw -> data/processed/*.parquet + meta.json
```

Candidate threshold, label boundaries and paths all come from `configs/config.yaml`;
the build is safe to re-run (outputs are rewritten from scratch each time).

Dataset citation, access date, audit summary and processed-dataset facts live in
`docs/dataset_notes.md`.
