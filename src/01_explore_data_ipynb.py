# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data Explorer
#
# A quick tour of all datasets assembled for the Chen, Kelly, and Xiu (2022) replication.
# Data sources: RavenPack news analytics, CRSP stock/index data, and FRED macro series.

# %%
from pathlib import Path

import polars as pl

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

# %% [markdown]
# ---
# ## RavenPack
#
# Dow Jones newswire articles scored by RavenPack's NLP analytics engine.

# %% [markdown]
# ### ravenpack_djpr_with_permno.parquet
#
# RavenPack DJ Press Release analytics linked to CRSP PERMNOs via the crosswalk table.

# %%
rp_permno = pl.scan_parquet(DATA_DIR / "ravenpack_djpr_with_permno.parquet")
shape = rp_permno.select(pl.len()).collect().item()
cols = rp_permno.collect_schema().names()
print(f"Rows: {shape:,}  |  Columns: {len(cols)}")
print(f"Columns: {cols}")

# %%
rp_permno.head(5).collect()

# %%
# PERMNO coverage
total = rp_permno.select(pl.len()).collect().item()
matched = rp_permno.filter(pl.col("permno").is_not_null()).select(pl.len()).collect().item()
n_permnos = rp_permno.select(pl.col("permno").n_unique()).collect().item()
print(f"Total rows: {total:,}")
print(f"Rows with PERMNO: {matched:,} ({matched / total * 100:.1f}%)")
print(f"Unique PERMNOs: {n_permnos:,}")

# %% [markdown]
# ### raven_crsp_crosswalk.parquet
#
# Maps RavenPack entity IDs to CRSP PERMNOs.

# %%
crosswalk = pl.read_parquet(DATA_DIR / "raven_crsp_crosswalk.parquet")
print(f"Shape: {crosswalk.shape}")
crosswalk.head(5)

# %%
print(f"Unique RavenPack entities: {crosswalk['rp_entity_id'].n_unique()}")
print(f"Unique PERMNOs: {crosswalk['permno'].n_unique()}")

# %% [markdown]
# ---
# ## CRSP
#
# Stock and index returns from the Center for Research in Security Prices.

# %% [markdown]
# ### CRSP_MSF_INDEX_INPUTS.parquet
#
# Monthly stock file — individual security returns and market cap used as index inputs.

# %%
msf = pl.read_parquet(DATA_DIR / "CRSP_MSF_INDEX_INPUTS.parquet")
print(f"Shape: {msf.shape}")
print(f"Columns: {msf.columns}")
msf.head(5)

# %%
date_col = [c for c in msf.columns if "date" in c.lower()][0]
print(f"Date range: {msf[date_col].min()} to {msf[date_col].max()}")
print(f"Unique PERMNOs: {msf['permno'].n_unique():,}")

# %% [markdown]
# ### CRSP_DSI.parquet
#
# Daily stock index — aggregate market returns and counts.

# %%
dsi = pl.read_parquet(DATA_DIR / "CRSP_DSI.parquet")
print(f"Shape: {dsi.shape}")
dsi.head(5)

# %%
date_col = [c for c in dsi.columns if "date" in c.lower()][0]
print(f"Date range: {dsi[date_col].min()} to {dsi[date_col].max()}")

# %% [markdown]
# ### CRSP_MSIX.parquet
#
# Monthly stock index returns.

# %%
msix = pl.read_parquet(DATA_DIR / "CRSP_MSIX.parquet")
print(f"Shape: {msix.shape}")
msix.head(5)

# %% [markdown]
# ---
# ## FRED
#
# Macroeconomic time series from the Federal Reserve Economic Data service.

# %%
fred = pl.read_parquet(DATA_DIR / "fred.parquet")
print(f"Shape: {fred.shape}")
print(f"Columns: {fred.columns}")
fred.head(5)

# %%
date_col = [c for c in fred.columns if "date" in c.lower()][0]
print(f"Date range: {fred[date_col].min()} to {fred[date_col].max()}")

# %%
# Non-null counts per series
fred.select(pl.all().is_not_null().sum()).unpivot(
    variable_name="series",
    value_name="non_null_count",
).sort("non_null_count")

# %% [markdown]
# ---
# ## Market Data
#
# Computed market-level return and volatility measures.

# %%
market = pl.read_parquet(DATA_DIR / "market_data.parquet")
print(f"Shape: {market.shape}")
market.head(5)

# %%
market.describe()

# %%
