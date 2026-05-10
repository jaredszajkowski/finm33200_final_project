"""Build per-(sector, day) sentiment + return panel.

Output: _data/sector_sentiment_panel_{model}.parquet with one row per
(gsector, date), with columns:
    gsector, sector_name, date, n_headlines, sentiment, synthetic_ret, etf_ret

Usage
-----
    python src/build_sector_panel.py --model bert
    python src/build_sector_panel.py --model gemma
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from settings import config
from merge_sector import GICS_SECTOR_NAMES

DATA_DIR = Path(config("DATA_DIR"))

# 2-digit GICS sector code -> SPDR Select Sector ETF ticker
SECTOR_ETF_MAP = {
    "10": "XLE", "15": "XLB", "20": "XLI", "25": "XLY",
    "30": "XLP", "35": "XLV", "40": "XLF", "45": "XLK",
    "50": "XLC", "55": "XLU", "60": "XLRE",
}


def join_predictions_to_headlines(
    df_headlines: pl.DataFrame, df_preds: pl.DataFrame
) -> pl.DataFrame:
    """Inner join headlines (with sector + market_cap) to per-story predictions.

    Returns one row per scored headline with `sentiment_score = p_up - 0.5`.
    Headlines without a prediction are dropped.
    """
    return (
        df_headlines
        .join(df_preds.select("rp_story_id", "p_up"), on="rp_story_id", how="inner")
        .with_columns((pl.col("p_up") - 0.5).alias("sentiment_score"))
        .select(
            "rp_story_id", "permno", "article_date", "market_cap",
            "gsector", "sentiment_score",
        )
    )


def aggregate_sentiment(df_per_headline: pl.DataFrame) -> pl.DataFrame:
    """Value-weighted mean of sentiment_score per (gsector, article_date).

    Drops rows with null/zero market_cap before weighting; drops sector-day
    cells whose remaining total weight is zero (no usable headlines).
    """
    df = df_per_headline.filter(
        pl.col("market_cap").is_not_null() & (pl.col("market_cap") > 0)
    )
    grouped = df.group_by(["gsector", "article_date"]).agg(
        (
            (pl.col("sentiment_score") * pl.col("market_cap")).sum()
            / pl.col("market_cap").sum()
        ).alias("sentiment"),
        pl.len().alias("n_headlines"),
    )
    return grouped.rename({"article_date": "date"}).sort(["gsector", "date"])


def compute_synthetic_sector_returns(
    df_crsp: pl.DataFrame, df_sp500: pl.DataFrame
) -> pl.DataFrame:
    """VW return per (gsector, date) over the point-in-time S&P 500 universe.

    Uses the same as-of-date join pattern as `merge_sector.merge_sector` but
    on (permno, CRSP date) instead of (permno, article_date).
    """
    df_sectors = (
        df_sp500
        .select(
            pl.col("permno").cast(pl.Float64),
            pl.col("gsector").cast(pl.Utf8),
            pl.col("effstartdt").cast(pl.Date),
            pl.col("effenddt").cast(pl.Date),
        )
        .drop_nulls(subset=["gsector", "effstartdt", "effenddt"])
        .unique()
    )

    df = df_crsp.with_columns(
        pl.col("permno").cast(pl.Float64),
        pl.col("date").cast(pl.Date),
    ).filter(
        pl.col("ret").is_not_null()
        & pl.col("market_cap").is_not_null()
        & (pl.col("market_cap") > 0)
    )

    joined = df.join_where(
        df_sectors,
        pl.col("permno") == pl.col("permno_right"),
        pl.col("date") >= pl.col("effstartdt"),
        pl.col("date") <= pl.col("effenddt"),
    ).unique(subset=["permno", "date"], keep="first")

    return (
        joined.group_by(["gsector", "date"])
        .agg(
            (
                (pl.col("ret") * pl.col("market_cap")).sum()
                / pl.col("market_cap").sum()
            ).alias("synthetic_ret")
        )
        .sort(["gsector", "date"])
    )


def attach_etf_returns(df_panel: pl.DataFrame, df_etf: pl.DataFrame) -> pl.DataFrame:
    """Join the SPDR ETF return for each (gsector, date) row in df_panel."""
    sector_to_ticker = pl.DataFrame({
        "gsector": list(SECTOR_ETF_MAP.keys()),
        "ticker": list(SECTOR_ETF_MAP.values()),
    })
    df = df_panel.join(sector_to_ticker, on="gsector", how="left")
    df = df.join(
        df_etf.select(pl.col("ticker"), pl.col("date").cast(pl.Date), "ret"),
        on=["ticker", "date"],
        how="left",
    )
    return df.rename({"ret": "etf_ret"}).drop("ticker")


def build_panel(model_name: str, data_dir: Path = None) -> Path:
    """Build the sector panel for the given embedding model and write it.

    Returns the output path.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    df_headlines = pl.read_parquet(data_dir / "labeled_dataset_with_sector.parquet")
    df_preds = pl.read_parquet(data_dir / f"rolling_predictions_{model_name}.parquet")
    df_crsp = pl.read_parquet(data_dir / "CRSP_daily_stock.parquet")
    df_sp500 = pl.read_parquet(data_dir / "sp500_constituents.parquet")
    df_etf = pl.read_parquet(data_dir / "sector_etfs.parquet")

    df_headline_scored = join_predictions_to_headlines(df_headlines, df_preds)
    df_sentiment = aggregate_sentiment(df_headline_scored)
    df_synth = compute_synthetic_sector_returns(df_crsp, df_sp500)

    df_panel = df_sentiment.join(df_synth, on=["gsector", "date"], how="inner")
    df_panel = attach_etf_returns(df_panel, df_etf)

    df_panel = df_panel.with_columns(
        pl.col("gsector").replace_strict(
            GICS_SECTOR_NAMES, default=None, return_dtype=pl.Utf8
        ).alias("sector_name")
    ).select(
        "gsector", "sector_name", "date", "n_headlines",
        "sentiment", "synthetic_ret", "etf_ret",
    ).sort(["gsector", "date"])

    out_path = data_dir / f"sector_sentiment_panel_{model_name}.parquet"
    df_panel.write_parquet(out_path)
    print(
        f"Wrote {len(df_panel):,} (sector, day) rows to {out_path} "
        f"({df_panel['gsector'].n_unique()} sectors)"
    )
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", required=True, choices=["bert", "gemma", "tfidf", "openai_small"]
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_panel(args.model)
