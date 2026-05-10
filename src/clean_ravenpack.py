"""
Clean and filter RavenPack headlines through the filtering funnel.

Implements the multi-stage filtering pipeline from Chen, Kelly, Xiu (2022):
  0. Raw: all RavenPack headlines linked to CRSP PERMNOs
  1. Single-entity: one CRSP permno per rp_story_id
  2. Has return: permno-date has a valid CRSP daily return
  3. Length filter: headline >= 5 characters
  4. Deduplicated: no exact-duplicate headline per permno-date

Also applies the opening-window exclusion: articles published 9:00-9:30 AM
EST are assigned to the next trading day.

Input:
  - _data/ravenpack_djpr_with_permno.parquet
  - _data/CRSP_daily_stock.parquet

Output:
  - _data/ravenpack_clean.parquet
  - _data/ravenpack_stage_counts.json

Usage
-----
    python src/clean_ravenpack.py
"""

import json
from pathlib import Path

import polars as pl
import pandas_market_calendars as mcal

from settings import config

DATA_DIR = Path(config("DATA_DIR"))


def _utc_to_est(ts_col: pl.Expr) -> pl.Expr:
    """Convert UTC timestamp column to US/Eastern, return as naive datetime."""
    return ts_col.dt.convert_time_zone("US/Eastern").dt.replace_time_zone(None)


def apply_opening_window_exclusion(df: pl.DataFrame) -> pl.DataFrame:
    """Shift articles published 9:00-9:30 AM EST to the next trading day.

    Articles in the first 30 minutes of the session reflect pre-open
    information and are treated as next-day news per CKX (2022).
    """
    # Compute EST time
    ts_est = df["timestamp_utc"].dt.convert_time_zone("US/Eastern")
    hour = ts_est.dt.hour()
    minute = ts_est.dt.minute()

    in_opening = (hour == 9) & (minute < 30)

    if not in_opening.any():
        return df

    # Build trading day schedule
    affected_dates = df.filter(in_opening)["article_date"].unique().sort()
    if len(affected_dates) == 0:
        return df

    import pandas as pd

    min_d = affected_dates.min()
    max_d = affected_dates.max()
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=str(min_d),
        end_date=(pd.Timestamp(str(max_d)) + pd.offsets.BDay(10)).strftime("%Y-%m-%d"),
    )
    trading_days = sorted(schedule.index.normalize().date.tolist())

    def _next_trading_day(d):
        for td in trading_days:
            if td > d:
                return td
        return d

    # Build date mapping
    date_map_data = {
        "article_date": affected_dates.to_list(),
        "article_date_new": [_next_trading_day(d) for d in affected_dates.to_list()],
    }
    date_map = pl.DataFrame(date_map_data)

    # Apply the shift
    df = df.join(date_map, on="article_date", how="left")
    df = df.with_columns(
        pl.when(in_opening & pl.col("article_date_new").is_not_null())
        .then(pl.col("article_date_new"))
        .otherwise(pl.col("article_date"))
        .alias("article_date")
    ).drop("article_date_new")

    return df


def build_filtering_funnel(df_headlines: pl.DataFrame, df_crsp_daily: pl.DataFrame, min_chars: int = 5):
    """Apply the full filtering pipeline and return clean data + stage counts."""
    stage_counts = {}

    df = df_headlines.clone()

    # Derive article_date from timestamp_utc (EST date)
    df = df.with_columns(
        df["timestamp_utc"]
        .dt.convert_time_zone("US/Eastern")
        .dt.date()
        .alias("article_date")
    )

    # Stage 0: Raw
    stage_counts["0_raw"] = len(df)
    print(f"Stage 0 (raw):                    {stage_counts['0_raw']:>10,}")

    # Apply opening-window shift before filtering
    df = apply_opening_window_exclusion(df)

    # Stage 1: Single entity — one PERMNO per rp_story_id
    permno_counts = (
        df.filter(pl.col("permno").is_not_null())
        .group_by("rp_story_id")
        .agg(pl.col("permno").n_unique().alias("n_permnos"))
    )
    single_entity_ids = permno_counts.filter(pl.col("n_permnos") == 1)["rp_story_id"].to_list()
    df = df.filter(
        pl.col("rp_story_id").is_in(single_entity_ids) & pl.col("permno").is_not_null()
    )
    stage_counts["1_single_entity"] = len(df)
    print(f"Stage 1 (single entity):          {stage_counts['1_single_entity']:>10,}")

    # Stage 2: Has CRSP return
    crsp_keys = (
        df_crsp_daily
        .filter(pl.col("ret").is_not_null())
        .select(
            pl.col("permno").cast(pl.Float64),
            pl.col("date").cast(pl.Date).alias("article_date"),
        )
        .unique()
    )
    df = df.with_columns(pl.col("permno").cast(pl.Float64))
    df = df.join(crsp_keys, on=["permno", "article_date"], how="semi")
    stage_counts["2_has_return"] = len(df)
    print(f"Stage 2 (has return):             {stage_counts['2_has_return']:>10,}")

    # Stage 3: Length filter
    df = df.filter(
        pl.col("headline").is_not_null()
        & (pl.col("headline").str.strip_chars().str.len_chars() >= min_chars)
    )
    stage_counts["3_length_filter"] = len(df)
    print(f"Stage 3 (len >= {min_chars} chars):        {stage_counts['3_length_filter']:>10,}")

    # Stage 4: Deduplication (exact headline per permno-date, keep earliest)
    df = df.sort(["permno", "article_date", "timestamp_utc"])
    df = df.unique(subset=["permno", "article_date", "headline"], keep="first")
    stage_counts["4_deduplicated"] = len(df)
    print(f"Stage 4 (deduplicated):           {stage_counts['4_deduplicated']:>10,}")

    return df, stage_counts


def load_clean_ravenpack(data_dir=DATA_DIR):
    """Load the cleaned RavenPack dataset from disk."""
    return pl.scan_parquet(Path(data_dir) / "ravenpack_clean.parquet")


if __name__ == "__main__":
    print("Loading RavenPack headlines with PERMNO links...")
    df_scraped = pl.read_parquet(DATA_DIR / "ravenpack_djpr_with_permno.parquet")

    print("Loading CRSP daily returns...")
    df_crsp = pl.read_parquet(DATA_DIR / "CRSP_daily_stock.parquet")

    print("\nApplying filtering funnel...")
    df_clean, stage_counts = build_filtering_funnel(df_scraped, df_crsp)

    path = DATA_DIR / "ravenpack_clean.parquet"
    df_clean.write_parquet(path)
    print(f"\nSaved clean data ({len(df_clean):,} rows): {path}")

    path_counts = DATA_DIR / "ravenpack_stage_counts.json"
    with open(path_counts, "w") as f:
        json.dump(stage_counts, f, indent=2)
    print(f"Saved stage counts: {path_counts}")
