"""Compute monthly market return and volatility.

Supports two data sources controlled by `USE_CRSP`:
- CRSP: loads daily CRSP value-weighted returns (vwretd) from CRSP_DSI.parquet
- yfinance: downloads daily ^GSPC (S&P 500) prices as a proxy

Both paths compute:
- Monthly log return (sum of daily log returns, in percent)
- Monthly realized volatility (std dev of daily log returns, in percent)

These are used in Table I of Bybee et al. (2024).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

# Set to True if you have CRSP access via WRDS; False uses yfinance ^GSPC proxy
USE_CRSP = True


def _pull_crsp_market_data(data_dir):
    """Compute monthly market data from CRSP daily index (CRSP_DSI.parquet)."""
    dsi = pd.read_parquet(data_dir / "CRSP_DSI.parquet")
    dsi["date"] = pd.to_datetime(dsi["date"])
    dsi = dsi.set_index("date").sort_index()

    dsi["log_return"] = np.log(1 + dsi["vwretd"])
    dsi["year_month"] = dsi.index.to_period("M")

    monthly = dsi.groupby("year_month").agg(
        market_return=("log_return", "sum"),
        market_volatility=("log_return", "std"),
    )

    monthly["market_return"] = monthly["market_return"] * 100
    monthly["market_volatility"] = monthly["market_volatility"] * 100

    monthly.index = monthly.index.to_timestamp(how="start")
    monthly.index.name = "date"

    return monthly


def _pull_yfinance_market_data():
    """Compute monthly market data from yfinance ^GSPC (S&P 500)."""
    import yfinance as yf

    df = yf.download("^GSPC", start="1983-12-01", end="2017-07-01")

    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # yfinance auto_adjust=True (default) folds adjustments into "Close"
    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    daily_log_ret = np.log(df[close_col] / df[close_col].shift(1)).dropna()
    daily_log_ret.index = pd.to_datetime(daily_log_ret.index)
    daily_log_ret = daily_log_ret.to_frame("log_return")
    daily_log_ret["year_month"] = daily_log_ret.index.to_period("M")

    monthly = daily_log_ret.groupby("year_month").agg(
        market_return=("log_return", "sum"),
        market_volatility=("log_return", "std"),
    )

    monthly["market_return"] = monthly["market_return"] * 100
    monthly["market_volatility"] = monthly["market_volatility"] * 100

    monthly.index = monthly.index.to_timestamp(how="start")
    monthly.index.name = "date"

    return monthly


def pull_market_data(data_dir=DATA_DIR):
    """Compute monthly market return and volatility.

    Dispatches to CRSP or yfinance based on USE_CRSP flag.

    Parameters
    ----------
    data_dir : Path
        Directory containing input data and where market_data.parquet
        will be saved.

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame with columns 'market_return' and 'market_volatility',
        indexed by month-start dates.
    """
    data_dir = Path(data_dir)

    if USE_CRSP:
        monthly = _pull_crsp_market_data(data_dir)
    else:
        monthly = _pull_yfinance_market_data()

    # Filter to the sample period
    monthly = monthly.loc["1984-01":"2017-06"]

    # Save
    data_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(data_dir / "market_data.parquet")
    source = "CRSP VWRETD" if USE_CRSP else "yfinance ^GSPC"
    print(f"Saved market_data.parquet ({len(monthly)} months, source: {source})")

    return monthly


def load_market_data(data_dir=DATA_DIR):
    """Load previously saved monthly market data from parquet."""
    return pd.read_parquet(Path(data_dir) / "market_data.parquet")


if __name__ == "__main__":
    pull_market_data()
