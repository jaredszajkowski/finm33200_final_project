"""
Load independently scraped headlines from the news_headlines chartbook
pipeline and merge with RavenPack metadata + CRSP PERMNO.

The scraped headlines are TOS-compliant for sending to OpenAI (they are
independently sourced text, not RavenPack's proprietary headline text).

Input:
  - news_headlines chartbook pipeline: scraped_headlines_with_rp_metadata
  - _data/raven_crsp_crosswalk.parquet (rp_entity_id -> permno)

Output: _data/ravenpack_scraped.parquet

Usage
-----
    python src/merge_scraped_headlines.py
"""

from pathlib import Path

import polars as pl
from chartbook import data as cb_data

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")

CROSSWALK_FILE = DATA_DIR / "raven_crsp_crosswalk.parquet"
OUTPUT_FILE = DATA_DIR / "ravenpack_scraped.parquet"

OUTPUT_COLUMNS = [
    "rp_story_id",
    "rp_entity_id",
    "permno",
    "headline",
    "timestamp_utc",
    "event_sentiment_score",
    "css",
    "relevance",
    "event_similarity_key",
    "event_similarity_days",
    "news_type",
    "category",
    "rp_group",
    "entity_name",
]


def load_scraped_headlines() -> pl.LazyFrame:
    """Load the scraped-headlines-with-RP-metadata dataset from the
    news_headlines chartbook pipeline as a LazyFrame (no data read until collect)."""
    print("Loading scraped headlines from chartbook (news_headlines pipeline)...")
    # df_pd = cb_data.load(
    #     pipeline="news_headlines",
    #     dataframe="scraped_headlines_with_rp_metadata",
    #     format="pandas",
    # )
    # df = pl.from_pandas(df_pd)
    # print(f"  Total rows loaded: {len(df):,}")
    # return df
    return cb_data.load(
        pipeline="news_headlines",
        dataframe="scraped_headlines_with_rp_metadata",
    )


def filter_scraped(df: pl.LazyFrame) -> pl.LazyFrame:
    """Apply CKX-compatible filters to scraped headlines."""
    from datetime import datetime

    start = datetime.fromisoformat(START_DATE.strftime("%Y-%m-%d"))
    end = datetime.fromisoformat(END_DATE.strftime("%Y-%m-%d"))

    # Cast timestamp_utc to a consistent type for comparison (strip tz if present)
    return df.with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("us")).alias("timestamp_utc"),
    ).filter(
        pl.col("headline").is_not_null()
        & (pl.col("country_code") == "US")
        & (pl.col("entity_type") == "COMP")
        & (pl.col("relevance") >= 90)
        & (pl.col("timestamp_utc") >= start)
        & (pl.col("timestamp_utc") <= end)
    )


def merge_permno(df: pl.LazyFrame, crosswalk_path: Path) -> pl.LazyFrame:
    """Join PERMNO from the RavenPack-CRSP crosswalk."""
    crosswalk = pl.scan_parquet(crosswalk_path).select("rp_entity_id", "permno")
    # Keep first permno per entity if duplicates
    crosswalk = crosswalk.group_by("rp_entity_id").first()
    return df.join(crosswalk, on="rp_entity_id", how="inner")


def main():
    if not CROSSWALK_FILE.exists():
        raise FileNotFoundError(
            f"Missing crosswalk for PERMNO lookup: {CROSSWALK_FILE}\n"
            "Run: doit pull:link_ravenpack_crsp"
        )

    # Step 1: Load from chartbook (returns LazyFrame, no data read yet)
    df = load_scraped_headlines()

    # Step 2: Filter
    # lf = filter_scraped(df.lazy())
    lf = filter_scraped(df)
    df_filtered = lf.collect()
    print(f"  After filters: {len(df_filtered):,}")

    # Step 3: Merge PERMNO
    lf = merge_permno(df_filtered.lazy(), CROSSWALK_FILE)
    df_merged = lf.collect()
    print(f"  After PERMNO merge: {len(df_merged):,}")

    # Step 4: Select output columns (only those present)
    available = [c for c in OUTPUT_COLUMNS if c in df_merged.columns]
    output = df_merged.select(available)

    print(f"\n=== Final Output ===")
    print(f"  Output rows: {len(output):,}")
    print(f"  Unique permnos: {output['permno'].n_unique():,}")
    ts = output["timestamp_utc"]
    print(f"  Date range: {ts.min()} to {ts.max()}")

    output.write_parquet(OUTPUT_FILE)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
