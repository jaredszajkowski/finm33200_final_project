"""Pull SPDR Select Sector ETF daily prices from yfinance.

Output: _data/sector_etfs.parquet with columns:
    ticker (Utf8), date (Date), adj_close (Float64), ret (Float64)

The 11 SPDR Select Sector ETFs map 1:1 to the 11 GICS 2-digit sectors.
XLRE inception 2015-10-08; XLC inception 2018-06-19. Earlier dates
are simply absent for those tickers.

Usage
-----
    python src/pull_sector_etfs.py
"""

from pathlib import Path

import polars as pl

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
START_DATE = "2000-01-01"
END_DATE = "2019-06-30"

SECTOR_ETF_TICKERS = [
    "XLE",
    "XLB",
    "XLI",
    "XLY",
    "XLP",
    "XLV",
    "XLF",
    "XLK",
    "XLC",
    "XLU",
    "XLRE",
]


def download_sector_etfs(
    tickers=SECTOR_ETF_TICKERS, start=START_DATE, end=END_DATE
) -> pl.DataFrame:
    """Download daily adjusted closes for the given tickers from yfinance."""
    import pandas as pd
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    rows = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        sub = raw[t][["Close"]].dropna().reset_index()
        sub = sub.rename(columns={"Date": "date", "Close": "adj_close"})
        sub["ticker"] = t
        rows.append(sub[["ticker", "date", "adj_close"]])

    df = pd.concat(rows, ignore_index=True)
    return pl.from_pandas(df).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("adj_close").cast(pl.Float64),
    )


def compute_etf_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Add per-ticker simple daily returns. First row per ticker is null."""
    return (
        df.sort(["ticker", "date"])
        .with_columns(
            (
                pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1.0
            ).alias("ret")
        )
        .select("ticker", "date", "adj_close", "ret")
    )


if __name__ == "__main__":
    print(f"Downloading {len(SECTOR_ETF_TICKERS)} sector ETFs from yfinance...")
    df = download_sector_etfs()
    df = compute_etf_returns(df)

    path = DATA_DIR / "sector_etfs.parquet"
    df.write_parquet(path)
    n_per = df.group_by("ticker").len().sort("ticker")
    print(f"Saved {len(df):,} rows: {path}")
    print(n_per)
