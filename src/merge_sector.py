"""
Merge GICS sector classification onto the labeled headline dataset.

Joins `_data/labeled_dataset.parquet` against `_data/sp500_constituents.parquet`
on `permno` with a point-in-time filter `effstartdt <= article_date <= effenddt`,
so each headline gets the GICS sector that was active for that issuer on the
article date. Coverage is limited to permnos that were S&P 500 constituents.

Input:
  - _data/labeled_dataset.parquet
  - _data/sp500_constituents.parquet

Output: _data/labeled_dataset_with_sector.parquet
  Adds columns: gsector (2-digit GICS code), sector_name (readable label).
"""

from pathlib import Path

import polars as pl

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

# GICS 2-digit sector codes (as strings, since gsector is stored as text).
GICS_SECTOR_NAMES = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}


def merge_sector(df_labeled: pl.DataFrame, df_sp500: pl.DataFrame) -> pl.DataFrame:
    """Attach GICS sector to each headline via a point-in-time range join."""
    df_sectors = (
        df_sp500.select(
            pl.col("permno").cast(pl.Float64),
            pl.col("gsector").cast(pl.Utf8),
            pl.col("effstartdt").cast(pl.Date),
            pl.col("effenddt").cast(pl.Date),
        )
        .drop_nulls(subset=["gsector", "effstartdt", "effenddt"])
        .unique()
    )

    df = df_labeled.with_columns(
        pl.col("permno").cast(pl.Float64),
        pl.col("article_date").cast(pl.Date),
    )

    joined = df.join_where(
        df_sectors,
        pl.col("permno") == pl.col("permno_right"),
        pl.col("article_date") >= pl.col("effstartdt"),
        pl.col("article_date") <= pl.col("effenddt"),
    )

    # Multiple overlapping name spells inside a single GICS window can produce
    # duplicate matches; collapse to one row per headline-permno.
    joined = joined.unique(
        subset=["rp_story_id", "permno", "article_date"], keep="first"
    )
    joined = joined.drop(
        [c for c in ("permno_right", "effstartdt", "effenddt") if c in joined.columns]
    )

    # Left-attach back onto the original frame so non-S&P permnos survive with null sector.
    df = df.join(
        joined.select("rp_story_id", "permno", "article_date", "gsector"),
        on=["rp_story_id", "permno", "article_date"],
        how="left",
    )

    sector_name_expr = pl.col("gsector").replace_strict(
        GICS_SECTOR_NAMES, default=None, return_dtype=pl.Utf8
    )
    df = df.with_columns(sector_name_expr.alias("sector_name"))

    n_total = len(df)
    df = df.filter(pl.col("gsector").is_not_null())
    print(
        f"Sector merge: kept {len(df):,} / {n_total:,} headlines with a GICS sector "
        f"({len(df) / n_total:.1%}); dropped {n_total - len(df):,} unmatched"
    )
    return df


if __name__ == "__main__":
    print("Loading labeled dataset and S&P 500 constituents...")
    df_labeled = pl.read_parquet(DATA_DIR / "labeled_dataset.parquet")
    df_sp500 = pl.read_parquet(DATA_DIR / "sp500_constituents.parquet")

    print("Merging GICS sector...")
    df_out = merge_sector(df_labeled, df_sp500)

    path = DATA_DIR / "labeled_dataset_with_sector.parquet"
    df_out.write_parquet(path)
    print(f"Saved labeled dataset with sector ({len(df_out):,} rows): {path}")
