"""
Pull historical S&P 500 constituent membership from WRDS.

Uses the CRSP table `crsp_m_indexes.dsp500list_v2`, which lists every
company that has ever been a member of the S&P 500 along with its
membership start and end dates.

The enriched pull JOINs with `crsp.stocknames` to get company names,
tickers, and other identifiers. This produces multiple rows per
membership spell (one per name period) — intentional so we capture
all historical names (e.g., "APPLE COMPUTER INC" and "APPLE INC").

GICS classification (sector/industry-group/industry/sub-industry) is
pulled point-in-time from Compustat's historical GICS table
(`comp.co_hgic`) via the CRSP/Compustat Merged link table
(`crsp.ccmxpf_linktable`), filtered to primary, valid links (LC/LU
link types and primary link flags P/C).

Because a single membership spell can span multiple GICS revisions,
we expand each `(permno, mbrstartdt, mbrenddt)` spell against
`co_hgic`'s `[indfrom, indthru]` windows so each output row reflects
the GICS classification that was active during that sub-interval.
The effective GICS window is exposed as `gics_startdt` / `gics_enddt`.

A separate names lookup table is built for matching against external
datasets like GDELT.
"""

import re
from pathlib import Path

import pandas as pd
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")

SUFFIX_PATTERN = re.compile(
    r"\b(inc|corp|corporation|co|company|ltd|limited|llc|lp|plc|group|holdings|holding|enterprises|enterprise|intl|international|technologies|technology|systems|industries|services|bancorp|bancshares|financial)\b"
)


def normalize_company_name(name: str) -> str:
    """Normalize a company name for fuzzy matching.

    Lowercase, remove punctuation (& . , '), strip common suffixes,
    and collapse whitespace.
    """
    s = name.lower()
    s = re.sub(r"[&.',]", " ", s)
    s = SUFFIX_PATTERN.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pull_sp500_constituents(wrds_username=WRDS_USERNAME):
    """Pull full S&P 500 constituent history from WRDS, enriched with company names."""
    db = wrds.Connection(wrds_username=wrds_username)

    df = db.raw_sql("""
    SELECT s.permno, s.indno, s.mbrstartdt, s.mbrenddt, s.mbrflg, s.indfam,
           n.comnam, n.ticker, n.ncusip, n.namedt, n.nameenddt, n.siccd, n.exchcd,
           l.gvkey,
           h.gsector, h.ggroup, h.gind, h.gsubind,
           h.indfrom AS gics_startdt,
           h.indthru AS gics_enddt,
           GREATEST(s.mbrstartdt, h.indfrom) AS effstartdt,
           LEAST(s.mbrenddt, COALESCE(h.indthru, s.mbrenddt)) AS effenddt
    FROM crsp_m_indexes.dsp500list_v2 s
    LEFT JOIN crsp.stocknames n
           ON s.permno = n.permno
          AND s.mbrstartdt <= n.nameenddt
          AND s.mbrenddt   >= n.namedt
    LEFT JOIN crsp.ccmxpf_linktable l
           ON s.permno = l.lpermno
          AND l.linktype IN ('LC', 'LU')
          AND l.linkprim IN ('P', 'C')
          AND s.mbrstartdt <= COALESCE(l.linkenddt, s.mbrenddt)
          AND s.mbrenddt   >= l.linkdt
    LEFT JOIN comp.co_hgic h
           ON l.gvkey = h.gvkey
          AND s.mbrstartdt <= COALESCE(h.indthru, s.mbrenddt)
          AND s.mbrenddt   >= h.indfrom
    """)

    db.close()

    df["mbrstartdt"] = pd.to_datetime(df["mbrstartdt"])
    df["mbrenddt"] = pd.to_datetime(df["mbrenddt"])
    df["namedt"] = pd.to_datetime(df["namedt"])
    df["nameenddt"] = pd.to_datetime(df["nameenddt"])
    df["gics_startdt"] = pd.to_datetime(df["gics_startdt"])
    df["gics_enddt"] = pd.to_datetime(df["gics_enddt"])
    df["effstartdt"] = pd.to_datetime(df["effstartdt"])
    df["effenddt"] = pd.to_datetime(df["effenddt"])

    print(f"S&P 500 constituents: {len(df):,} rows (membership spells x name periods)")
    return df


def build_sp500_names_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Build a deduplicated lookup of (permno, comnam, comnam_norm, ticker).

    Used for matching S&P 500 companies against external datasets.
    """
    lookup = df[["permno", "comnam", "ticker"]].dropna(subset=["comnam"]).copy()
    lookup["comnam_norm"] = lookup["comnam"].apply(normalize_company_name)
    lookup = lookup.drop_duplicates().reset_index(drop=True)

    # Drop rows with short normalized names — these cause massive false positives
    # when matching against GDELT organizations (e.g., "gap", "fox", "news", "ball",
    # or empty strings from "LLC CORP" / "LIMITED INC").
    before = len(lookup)
    lookup = lookup[lookup["comnam_norm"].str.len() >= 5].reset_index(drop=True)
    print(
        f"S&P 500 names lookup: {len(lookup):,} rows ({before - len(lookup)} dropped with comnam_norm < 5 chars)"
    )
    return lookup


def load_sp500_constituents(data_dir=DATA_DIR):
    return pd.read_parquet(Path(data_dir) / "sp500_constituents.parquet")


def load_sp500_names_lookup(data_dir=DATA_DIR):
    return pd.read_parquet(Path(data_dir) / "sp500_names_lookup.parquet")


if __name__ == "__main__":
    df = pull_sp500_constituents(wrds_username=WRDS_USERNAME)

    path = DATA_DIR / "sp500_constituents.parquet"
    df.to_parquet(path)
    print(f"Saved to {path}")

    lookup = build_sp500_names_lookup(df)
    lookup_path = DATA_DIR / "sp500_names_lookup.parquet"
    lookup.to_parquet(lookup_path)
    print(f"Saved to {lookup_path}")
