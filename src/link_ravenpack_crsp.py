"""
Build the crosswalk between RavenPack entity IDs and CRSP PERMNOs,
then attach PERMNOs to the RavenPack headlines.

The crosswalk joins CRSP's dse table (which has ncusip, 8 chars) to
RavenPack's wrds_rpa_company_names (which has isin) via:
    ncusip = SUBSTRING(isin FROM 3 FOR 8)

This works because ISINs are structured as:
    {2-char country code}{8-char CUSIP}{1-char check digit}
"""

from pathlib import Path

import pandas as pd
import polars as pl
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")


def build_raven_crsp_crosswalk(wrds_username=WRDS_USERNAME):
    """
    Build a crosswalk from RavenPack rp_entity_id to CRSP permno.

    Joins crsp.dse (ncusip) to rpna.wrds_rpa_company_names (isin)
    via SUBSTRING matching. Returns distinct (permno, rp_entity_id) pairs.
    """
    query = """
    SELECT DISTINCT
        a.permno,
        b.rp_entity_id
    FROM crsp.dse a
    INNER JOIN rpna.wrds_rpa_company_names b
        ON a.ncusip = SUBSTRING(b.isin FROM 3 FOR 8)
    WHERE a.ncusip IS NOT NULL
      AND b.isin IS NOT NULL
      AND LENGTH(b.isin) >= 11
    """

    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query)
    db.close()

    print(f"Crosswalk: {len(df):,} distinct (permno, rp_entity_id) pairs")
    return df


def attach_permno_to_ravenpack(
    data_dir=DATA_DIR,
    wrds_username=WRDS_USERNAME,
    output_path=None,
):
    """
    Left-join the RavenPack-CRSP crosswalk onto RavenPack headlines
    by rp_entity_id. Drops unmatched rows and reports match rate.

    Deduplicates crosswalk by rp_entity_id first (keeping the first permno)
    to prevent row explosion in rare cases where one entity maps to
    multiple PERMNOs.

    Streams directly to parquet via sink_parquet() to avoid materializing
    the full multi-GB dataset in memory.
    """
    crosswalk_path = Path(data_dir) / "raven_crsp_crosswalk.parquet"
    rp_path = Path(data_dir) / "ravenpack_djpr.parquet"
    if output_path is None:
        output_path = Path(data_dir) / "ravenpack_djpr_with_permno.parquet"

    # Crosswalk is small — load eagerly; rp is several GB — scan lazily
    crosswalk = pl.read_parquet(crosswalk_path)
    rp_lazy = pl.scan_parquet(rp_path)
    # crosswalk = pd.read_parquet(crosswalk_path)
    # rp = pd.read_parquet(rp_path)

    # Deduplicate crosswalk by rp_entity_id to prevent row explosion
    crosswalk_dedup = crosswalk.unique(subset=["rp_entity_id"], keep="first")
    # crosswalk_dedup = crosswalk.drop_duplicates(subset="rp_entity_id", keep="first")
    print(
        f"Crosswalk after dedup: {len(crosswalk_dedup):,} unique rp_entity_ids "
        f"(from {len(crosswalk):,} total pairs)"
    )

    # Stream join + filter directly to disk — never materializes full dataset in RAM
    # df = rp.merge(crosswalk_dedup, on="rp_entity_id", how="left")
    # df = df.dropna(subset=["permno"])
    # df["permno"] = df["permno"].astype(int)
    (
        rp_lazy.join(crosswalk_dedup.lazy(), on="rp_entity_id", how="left")
        .filter(pl.col("permno").is_not_null())
        .with_columns(pl.col("permno").cast(pl.Int64))
        .sink_parquet(output_path)
    )

    # Read back row count from metadata only (no data loaded)
    n_matched = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    print(f"Final dataset: {n_matched:,} rows with PERMNO saved to {output_path}")


def load_raven_crsp_crosswalk(data_dir=DATA_DIR):
    path = Path(data_dir) / "raven_crsp_crosswalk.parquet"
    df = pd.read_parquet(path)
    return df


def load_ravenpack_with_permno(data_dir=DATA_DIR):
    path = Path(data_dir) / "ravenpack_djpr_with_permno.parquet"
    df = pd.read_parquet(path)
    return df


if __name__ == "__main__":
    # Step 1: Build crosswalk
    crosswalk = build_raven_crsp_crosswalk()
    crosswalk_path = Path(DATA_DIR) / "raven_crsp_crosswalk.parquet"
    crosswalk.to_parquet(crosswalk_path)
    print(f"Saved crosswalk to {crosswalk_path}")

    # Step 2: Attach PERMNOs to RavenPack headlines (streams directly to parquet)
    output_path = Path(DATA_DIR) / "ravenpack_djpr_with_permno.parquet"
    attach_permno_to_ravenpack(data_dir=DATA_DIR, output_path=output_path)
    # df = attach_permno_to_ravenpack(data_dir=DATA_DIR)
    # df.to_parquet(output_path)

    # # Step 3: Remove intermediate file (data is now in the _with_permno file)
    # intermediate = Path(DATA_DIR) / "ravenpack_djpr.parquet"
    # if intermediate.exists():
    #     intermediate.unlink()
    #     print(f"Removed intermediate file {intermediate}")
