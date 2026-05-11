# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FINM 33200 final project: **deriving sector-level sentiment signals from financial news headlines**. Builds on the HW2 replication of Chen, Kelly, and Xiu (2022) "Expected Returns and Large Language Models," but predicts sector-level returns instead of stock-level returns and uses RavenPack's own headline text directly (rather than the independently-scraped headlines required by HW2's licensing constraint).

The pipeline embeds RavenPack headlines with **local** models only (BERT and Gemma) — no headline text is sent to external APIs — then trains rolling-window logistic regressions to predict 3-day return direction, aggregates predictions to the sector level, and compares synthetic sector returns/sentiment against SPDR Select Sector ETF returns.

See `README.md` for the full thesis and methodology comparison; `README_HW2.md` documents the HW2 predecessor pipeline.

## Commands

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # then fill in WRDS_USERNAME, HF_TOKEN, etc.
```

### Running the pipeline
```bash
doit                              # run all tasks in dependency order
doit list                         # list all tasks
doit task_config                  # create _data/ and _output/ directories

# Pulls
doit pull_fred                    # FRED macro data
doit pull_crsp                    # CRSP_MSF_INDEX_INPUTS, CRSP_MSIX, CRSP_DSI, CRSP_daily_stock, market_data
doit pull_sp500_constituents      # historical S&P 500 constituents + names lookup
doit pull_ravenpack               # RavenPack headlines from WRDS
doit pull_sector_etfs             # SPDR Select Sector ETF prices via yfinance

# Linking / cleaning
doit link:link_ravenpack_crsp     # crosswalk RavenPack entity IDs <-> CRSP PERMNOs
doit clean_data:ravenpack         # cleaning funnel (single-entity, has-return, length, dedup)
doit labels                       # 3-day return windows + binary labels
doit merge_sector                 # point-in-time GICS sector merge (S&P 500 only)
doit text_stats                   # headline length / token-count percentiles

# Embeddings + training (per model: bert, gemma)
doit embed:bert
doit embed:gemma
doit train:bert
doit train:gemma

# Sector aggregation
doit build_sector_panel:bert
doit build_sector_panel:gemma
```

`dodo.py` uses a sqlite backend (`./.doit-db.sqlite`); only stale tasks rerun. Notebook execution and Sphinx/chartbook site tasks (`task_run_notebooks`, `task_build_chartbook_site`) are currently commented out in `dodo.py`.

### Tests
```bash
pytest src/test_sector_sentiment.py -v
pytest src/test_sector_sentiment.py::test_run_rolling_returns_predictions_keyed_by_story_id -vv
```

### Linting
```bash
ruff format . && ruff check --select I --fix . && ruff check --fix .
```

## Architecture

### Task runner: `doit` (dodo.py)
PyDoit is used as a Python-based Makefile. Order: `config` → `pull_*` → `link` → `clean_data` → `labels` → `merge_sector` → `text_stats` → `embed` → `train` → `build_sector_panel`. Notebooks (`src/0[1-5]_*_ipynb.py`, jupytext percent format) are not currently wired in.

### Configuration: `src/settings.py`
All scripts do `from settings import config`. Resolution order: CLI args → env vars → `.env` → defaults. Key paths: `BASE_DIR`, `DATA_DIR` (`_data/`, gitignored), `OUTPUT_DIR` (`_output/`, gitignored), `MANUAL_DATA_DIR` (`data_manual/`, tracked). Required `.env` keys: `WRDS_USERNAME`, `HF_TOKEN`.

### `USE_CRSP` toggle (`src/pull_market_data.py`)
`USE_CRSP = True` computes monthly market return/vol from `CRSP_DSI.parquet`; `False` falls back to yfinance `^GSPC`. `dodo.py` imports this flag and adds `CRSP_DSI.parquet` to the market-data task's `file_dep` only when `USE_CRSP` is true — so flipping the flag changes the dependency graph.

### Data flow
1. `pull_ravenpack.py` → RavenPack DJPR headlines from WRDS
2. `pull_CRSP_stock.py` / `pull_CRSP_daily.py` → CRSP monthly + daily files
3. `pull_sp500_constituents.py` → historical S&P 500 membership w/ GICS sectors
4. `link_ravenpack_crsp.py` → entity-ID ↔ PERMNO crosswalk; emits `ravenpack_djpr_with_permno.parquet`
5. `clean_ravenpack.py` → cleaning funnel; emits `ravenpack_clean.parquet` and `ravenpack_stage_counts.json`
6. `clean_labels.py` → 3-day forward compound return + binary `label`; emits `labeled_dataset.parquet`
7. `merge_sector.py` → point-in-time GICS sector join; emits `labeled_dataset_with_sector.parquet` (the canonical training input)
8. `embed_bert.py` / `embed_gemma.py` → chunked parquet under `_data/embeddings_{model}_chunks/`
9. `train_rolling_model.py {model}` → rolling 6+2+1y window logistic regression; emits `rolling_results_{model}.json` and `rolling_predictions_{model}.parquet` (per-headline `p_up` keyed by `rp_story_id`)
10. `pull_sector_etfs.py` + `build_sector_panel.py --model {model}` → per-(sector, day) sentiment + return panel; compares synthetic sector returns vs SPDR sector ETFs

### Embedding model registry
`train_rolling_model.py` has an `EMBEDDING_REGISTRY` dict mapping model name → chunk directory. Currently active: `bert`, `gemma`. `tfidf` and `openai_small` entries are commented out in both `EMBEDDING_REGISTRY` and `dodo.py`'s `task_embed`/`task_train` model lists — re-enable in both places to add a model. The trained logistic regression has `StandardScaler` attached as `model._scaler`.

### Memory management for the multi-GB RavenPack dataset
Three scripts avoid materializing the full dataset in RAM:

- **`pull_ravenpack.py`** — pulls year-by-year from WRDS and appends each batch via PyArrow `ParquetWriter`; each year's DataFrame is `del`'d immediately after writing.
- **`link_ravenpack_crsp.py`** — opens RavenPack as `pl.scan_parquet` (lazy); join → filter → cast is streamed to disk via `sink_parquet()`. Row counts read back from parquet metadata only (`pl.scan_parquet(...).select(pl.len()).collect()`).
- **`clean_ravenpack.py`** / **`merge_sector.py`** — use Polars `LazyFrame` chains, collecting only at the end.

When editing these scripts, preserve the lazy/streaming pattern; do not introduce eager `pl.read_parquet` on the large RavenPack file.

### HW2 leftovers
`src/merge_scraped_headlines.py` and `src/test_homework.py` are holdovers from the HW2 replication pipeline (see `README_HW2.md`) and are not part of the sector-sentiment final-project flow. Do not wire them into new tasks.

### Notebooks
`src/0[1-5]_*_ipynb.py` are jupytext percent-format Python files (`# %%` cells). Edit the `.py` files, not the `.ipynb` files. `05_sector_analysis_ipynb.py` is the sector-sentiment analysis specific to this final project.

### Tests
`src/test_sector_sentiment.py` exercises `train_rolling_model.run_rolling_sentiment_analysis` and related helpers. Per-headline predictions must be keyed by `rp_story_id` with `oos_year` and `p_up` aligned row-wise.

### Conventions
- **Script naming prefixes**: `pull_` (external data fetch), `clean_` (filter/transform), `embed_` (text → vectors), `train_` (model fitting), `merge_` (dataset joins), `build_` (derived panels). Keep new files in this scheme.
- **`docs/`**: build output of the Chartbook/Sphinx site (the build task is currently commented out in `dodo.py`).

### Commits & PRs
- Commit subjects: short imperative (`Implement ...`, `Add ...`, `Modified ...`).
- PR descriptions should state what changed and why, the affected pipeline stage(s) (pull / clean / embed / train / sector / notebooks), evidence of verification (test command + result), and screenshots only when notebook or docs output materially changes.

### Key libraries
- **Polars** (not pandas) for the data pipeline; pandas only appears in market-data computation
- **scikit-learn** — `LogisticRegression` + `StandardScaler`
- **transformers + torch** for BERT (`bert-base-uncased`) and Gemma (`embeddinggemma-300m`); `HF_TOKEN` required for gated Gemma weights
- **wrds** for CRSP / RavenPack / S&P 500 constituents (requires WRDS credentials)
- **yfinance** for SPDR sector ETFs and the optional `^GSPC` market-data fallback
