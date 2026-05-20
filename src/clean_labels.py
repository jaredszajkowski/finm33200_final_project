"""
Compute 3-day return labels for news articles.

For each news article, computes the compound return over the window
(-1, +1) trading days relative to the article date, then assigns a
binary label: 1 if return > 0, 0 if return <= 0.

Window convention (from Chen, Kelly, Xiu 2022):
  - Day -1: day before article (pre-event)
  - Day  0: article date
  - Day +1: day after article (post-event)
  Compound return = (1 + r_{t-1}) * (1 + r_t) * (1 + r_{t+1}) - 1

Input:
  - _data/ravenpack_clean.parquet
  - _data/CRSP_daily_stock.parquet

Output: _data/labeled_dataset.parquet
  Columns: rp_story_id, permno, article_date, headline, ret_window, label, market_cap

Usage
-----
    python src/clean_labels.py
"""

from pathlib import Path

import polars as pl

from settings import config

DATA_DIR = Path(config("DATA_DIR"))


def compute_return_window(df_crsp_daily: pl.DataFrame, window=(-1, 1)) -> pl.DataFrame:
    """Pre-compute return windows for every permno-date in CRSP daily.

    Uses polars shift() within groups — no Python-level loops.
    """
    lo, hi = window

    df = df_crsp_daily.select("permno", "date", "ret").sort(["permno", "date"])

    # === HW STEP 1 START ===
    # raise NotImplementedError("TODO: implement compute_return_window")

    # src/clean_labels.py, lines 48-67
    # For each offset k in [lo, hi], shift ret by -k within each permno
    shift_exprs = []
    shift_names = []
    for k in range(lo, hi + 1):
        col_name = f"_ret_k{k}"
        shift_names.append(col_name)
        shift_exprs.append(pl.col("ret").shift(-k).over("permno").alias(col_name))

    df = df.with_columns(shift_exprs)
    df = df.drop_nulls(subset=shift_names)

    # Compound return: prod(1 + r_k) - 1
    compound_expr = pl.lit(1.0)
    for col_name in shift_names:
        compound_expr = compound_expr * (pl.lit(1.0) + pl.col(col_name))
    df = df.with_columns((compound_expr - pl.lit(1.0)).alias("ret_window"))

    # === HW STEP 1 END ===

    return df.select("permno", "date", "ret_window")


def assign_binary_label(df: pl.DataFrame) -> pl.DataFrame:
    """Add a binary 'label' column: 1 if ret_window > 0, else 0."""
    # === HW STEP 2 START ===
    # raise NotImplementedError("TODO: implement assign_binary_label")

    return df.with_columns((pl.col("ret_window") > 0).cast(pl.Int8).alias("label"))

    # === HW STEP 2 END ===


def build_labeled_dataset(
    df_clean: pl.DataFrame, df_crsp_daily: pl.DataFrame, window=(-1, 1)
) -> pl.DataFrame:
    """Build the labeled dataset by merging clean headlines with return windows."""
    # Filter CRSP to only permnos in news (speed improvement)
    # Cast permno to Float64 in both for consistent join types
    permnos_in_news = df_clean.select(
        pl.col("permno").cast(pl.Float64).drop_nulls().unique()
    )
    df_crsp_daily = df_crsp_daily.with_columns(pl.col("permno").cast(pl.Float64))
    df_crsp_news = df_crsp_daily.join(permnos_in_news, on="permno", how="semi")

    n_days = window[1] - window[0] + 1
    print(
        f"Computing {n_days}-day return windows for {len(permnos_in_news):,} permnos..."
    )
    df_windows = compute_return_window(df_crsp_news, window=window)
    df_windows = df_windows.rename({"date": "article_date"})
    df_windows = df_windows.with_columns(pl.col("permno").cast(pl.Float64))

    # Market cap on article date
    df_mc = df_crsp_daily.select(
        pl.col("permno").cast(pl.Float64),
        pl.col("date").cast(pl.Date).alias("article_date"),
        "market_cap",
    )

    # Prepare clean data
    keep_cols = ["rp_story_id", "permno", "article_date", "headline"]
    available = [c for c in keep_cols if c in df_clean.columns]
    df = df_clean.select(available).with_columns(
        pl.col("permno").cast(pl.Float64),
        pl.col("article_date").cast(pl.Date),
    )

    # Merge return windows
    df_windows = df_windows.with_columns(pl.col("article_date").cast(pl.Date))
    df = df.join(df_windows, on=["permno", "article_date"], how="inner")

    # Merge market cap
    df_mc = df_mc.with_columns(pl.col("article_date").cast(pl.Date))
    df = df.join(df_mc, on=["permno", "article_date"], how="left")

    # Binary label
    df = assign_binary_label(df)

    df = df.sort(["article_date", "permno"])

    pct_pos = df["label"].mean()
    print(
        f"Labeled dataset: {len(df):,} obs, "
        f"{pct_pos:.1%} positive, "
        f"date range: {df['article_date'].min()} - {df['article_date'].max()}"
    )
    return df


def load_labeled_dataset(data_dir=DATA_DIR):
    """Load the labeled dataset from disk."""
    return pl.scan_parquet(Path(data_dir) / "labeled_dataset.parquet")


if __name__ == "__main__":
    print("Loading clean data...")
    df_clean = pl.read_parquet(DATA_DIR / "ravenpack_clean.parquet")
    df_crsp_daily = pl.read_parquet(DATA_DIR / "CRSP_daily_stock.parquet")

    print("Building labeled dataset...")
    df_labeled = build_labeled_dataset(df_clean, df_crsp_daily, window=(-1, 1))

    path = DATA_DIR / "labeled_dataset.parquet"
    df_labeled.write_parquet(path)
    print(f"Saved labeled dataset ({len(df_labeled):,} rows): {path}")
