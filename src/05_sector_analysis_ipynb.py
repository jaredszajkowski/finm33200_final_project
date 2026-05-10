# %% [markdown]
# # Sector-level Sentiment vs Sector Returns
#
# For each GICS sector, compares the daily VW sentiment score against:
# - the synthetic VW return (same S&P 500 universe), and
# - the SPDR Select Sector ETF daily return.
#
# Reports Pearson and Spearman correlations same-day and at 1-day lead, and
# overlays z-scored series per sector. Real Estate (~2.8 yrs) and
# Communication Services (~9 mo) have short windows in our 2000-01-01 →
# 2019-06-30 sample; treat their numbers cautiously.

# %%
import sys
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, "../src")
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
MODEL = "bert"  # change to "gemma" to inspect the other model

# %%
panel = pl.read_parquet(DATA_DIR / f"sector_sentiment_panel_{MODEL}.parquet")
print(f"Panel: {len(panel):,} (sector, day) rows, {panel['gsector'].n_unique()} sectors")
panel.head()


# %%
def per_sector_correlations(df: pl.DataFrame) -> pd.DataFrame:
    rows = []
    for sec_key, sub in df.partition_by("gsector", as_dict=True).items():
        sec = sec_key[0] if isinstance(sec_key, tuple) else sec_key
        sub_pd = sub.sort("date").to_pandas()
        name = sub_pd["sector_name"].iloc[0]
        sub_pd["sentiment_lead"] = sub_pd["sentiment"].shift(1)

        for ret_col in ("synthetic_ret", "etf_ret"):
            for sent_col, lag_label in (("sentiment", "same-day"), ("sentiment_lead", "1d-lag")):
                pair = sub_pd[[sent_col, ret_col]].dropna()
                if len(pair) < 30:
                    pearson = spearman = np.nan
                else:
                    pearson = stats.pearsonr(pair[sent_col], pair[ret_col]).statistic
                    spearman = stats.spearmanr(pair[sent_col], pair[ret_col]).statistic
                rows.append({
                    "gsector": sec, "sector_name": name,
                    "return_series": ret_col, "lag": lag_label,
                    "n": len(pair), "pearson": pearson, "spearman": spearman,
                })
    return pd.DataFrame(rows).sort_values(["gsector", "return_series", "lag"]).reset_index(drop=True)


corr_table = per_sector_correlations(panel)
corr_table

# %% [markdown]
# ## Per-sector overlay charts (z-scored)


# %%
def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0


sectors_sorted = (
    panel.group_by(["gsector", "sector_name"])
    .agg(pl.len().alias("n"))
    .sort("gsector").to_pandas()
)
fig, axes = plt.subplots(
    len(sectors_sorted), 1, figsize=(12, 2.5 * len(sectors_sorted)), sharex=False
)
if len(sectors_sorted) == 1:
    axes = [axes]
for ax, (_, row) in zip(axes, sectors_sorted.iterrows()):
    sec = row["gsector"]
    sub = panel.filter(pl.col("gsector") == sec).sort("date").to_pandas()
    ax.plot(sub["date"], zscore(sub["sentiment"]), label="sentiment (z)", lw=1)
    ax.plot(sub["date"], zscore(sub["synthetic_ret"]), label="synthetic ret (z)", lw=1, alpha=0.7)
    ax.plot(sub["date"], zscore(sub["etf_ret"]), label="etf ret (z)", lw=1, alpha=0.7)
    ax.set_title(f"{sec} — {row['sector_name']}  (n={row['n']:,})")
    ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Sample-size caveats
#
# - Real Estate (gsector 60): the GICS sector itself begins 2016-09-01.
# - Communication Services (gsector 50): begins 2018-09-28.
# - XLRE ETF inception 2015-10-08; XLC inception 2018-06-19.
#
# Correlations for these two sectors are reported but should be interpreted
# with the small effective sample size in mind.
