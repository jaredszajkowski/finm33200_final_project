"""
Pull CRSP daily stock returns from WRDS.

Tries the CIZ format table (crspd.dsf_v2) first, then falls back to the
legacy SIZ format table (crsp.dsf). Daily returns are needed to compute
3-day return windows around news events for constructing the binary
classification labels (Chen, Kelly, Xiu 2022).

Output: _data/CRSP_daily_stock.parquet

Usage
-----
    python src/pull_CRSP_daily.py
"""

from pathlib import Path

import polars as pl
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


def _pull_ciz(db, start_str, end_str):
    """Try pulling from the CIZ format daily table (crspd.dsf_v2)."""
    query = f"""
        SELECT
            dsf.permno,
            dsf.dlycaldt       AS date,
            dsf.dlyret         AS ret,
            dsf.dlyretx        AS retx,
            dsf.dlyprc         AS prc,
            dsf.dlyshrout      AS shrout,
            dsf.dlyvol         AS vol
        FROM crspd.dsf_v2 AS dsf
        INNER JOIN crspm.stksecurityinfohist AS ssih
            ON dsf.permno = ssih.permno
            AND ssih.secinfostartdt <= dsf.dlycaldt
            AND dsf.dlycaldt <= ssih.secinfoenddt
        WHERE
            dsf.dlycaldt BETWEEN '{start_str}' AND '{end_str}'
            AND ssih.securitytype = 'EQTY'
            AND ssih.securitysubtype = 'COM'
            AND ssih.sharetype = 'NS'
            AND ssih.usincflg = 'Y'
    """
    return db.raw_sql(query, date_cols=["date"])


def _pull_siz(db, start_str, end_str):
    """Pull from the legacy SIZ format daily table (crsp.dsf)."""
    query = f"""
        SELECT
            dsf.permno,
            dsf.date,
            dsf.ret,
            dsf.retx,
            dsf.prc,
            dsf.shrout,
            dsf.vol
        FROM crsp.dsf AS dsf
        INNER JOIN crsp.msenames AS names
            ON dsf.permno = names.permno
            AND names.namedt <= dsf.date
            AND dsf.date <= names.nameendt
        WHERE
            dsf.date BETWEEN '{start_str}' AND '{end_str}'
            AND names.shrcd IN (10, 11)
    """
    return db.raw_sql(query, date_cols=["date"])


def pull_CRSP_daily_file(
    start_date=START_DATE,
    end_date=END_DATE,
    wrds_username=WRDS_USERNAME,
):
    """Pull CRSP daily stock returns from WRDS.

    Pads the date range by 5 business days on each side to allow
    computation of +/-1 day return windows at the sample boundaries.
    """
    import pandas as pd

    start_ts = pd.Timestamp(start_date) - pd.offsets.BDay(5)
    end_ts = pd.Timestamp(end_date) + pd.offsets.BDay(5)
    start_str = start_ts.strftime("%Y-%m-%d")
    end_str = end_ts.strftime("%Y-%m-%d")

    db = wrds.Connection(wrds_username=wrds_username)

    df_pd = None
    for fmt, pull_fn in [
        ("CIZ (crspd.dsf_v2)", _pull_ciz),
        ("SIZ (crsp.dsf)", _pull_siz),
    ]:
        try:
            print(f"Trying {fmt}...")
            df_pd = pull_fn(db, start_str, end_str)
            print(f"  Success: {len(df_pd):,} rows from {fmt}")
            break
        except Exception as e:
            print(f"  {fmt} unavailable: {e}")

    db.close()

    if df_pd is None:
        raise RuntimeError(
            "Could not pull CRSP daily data from either crspd.dsf_v2 or crsp.dsf."
        )

    # Convert to polars, minimal formatting
    df = pl.from_pandas(df_pd.loc[:, ~df_pd.columns.duplicated()])

    # shrout in thousands -> actual shares; market_cap = |prc| * shrout
    df = df.with_columns(
        (pl.col("shrout") * 1000).alias("shrout"),
    ).with_columns(
        (pl.col("prc").abs() * pl.col("shrout")).alias("market_cap"),
    )

    df = df.sort(["permno", "date"])
    return df


def load_CRSP_daily_file(data_dir=DATA_DIR):
    """Load cached CRSP daily returns parquet from disk."""
    return pl.scan_parquet(Path(data_dir) / "CRSP_daily_stock.parquet")


if __name__ == "__main__":
    print("Pulling CRSP daily stock file...")
    df = pull_CRSP_daily_file(start_date=START_DATE, end_date=END_DATE)
    path = DATA_DIR / "CRSP_daily_stock.parquet"
    df.write_parquet(path)
    print(f"Saved {len(df):,} rows to {path}")
