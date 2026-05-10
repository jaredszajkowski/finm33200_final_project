"""
Pull RavenPack DJ Press Release headlines from WRDS.

Tables are partitioned by year: ravenpack_dj.rpa_djpr_equities_YYYY.
Date range covers 2000-01-01 to 2019-06-30 (RavenPack DJ availability window).

Key filters (matching Chen, Kelly, and Xiu 2022):
- entity_type = 'COMP' (companies only)
- country_code = 'US' (US only)
- relevance >= 90 (high relevance)
- Single-firm stories only (one entity per provider story)
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")

# RavenPack DJ availability window
RP_START_DATE = "2000-01-01"
RP_END_DATE = "2019-06-30"


def pull_ravenpack(
    start_date=RP_START_DATE,
    end_date=RP_END_DATE,
    wrds_username=WRDS_USERNAME,
    output_path=None,
):
    """
    Pull RavenPack DJ Press Release headlines from WRDS, year by year.

    Filters for US companies with high relevance (>=90) and single-firm
    stories only (one distinct rp_entity_id per provider story).

    Each year's result is written incrementally to a single parquet file via
    PyArrow's ParquetWriter so that no more than one year's data is held in
    memory at a time.
    """
    if output_path is None:
        output_path = Path(DATA_DIR) / "ravenpack_djpr.parquet"

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    db = wrds.Connection(wrds_username=wrds_username)
    writer = None
    total_rows = 0

    try:
        for year in range(start_year, end_year + 1):
            table_name = f"ravenpack_dj.rpa_djpr_equities_{year}"
            print(f"Pulling {table_name}...")

            query = f"""
            WITH single_firm AS (
                SELECT provider_id, provider_story_id
                FROM {table_name}
                WHERE entity_type = 'COMP'
                  AND country_code = 'US'
                  AND relevance >= 90
                GROUP BY provider_id, provider_story_id
                HAVING COUNT(DISTINCT rp_entity_id) = 1
            )
            SELECT
                a.timestamp_utc,
                a.rp_story_id,
                a.rp_entity_id,
                a.entity_type,
                a.entity_name,
                a.country_code,
                a.relevance,
                a.event_sentiment_score,
                a.event_relevance,
                a.event_similarity_key,
                a.event_similarity_days,
                a.topic,
                a."group" AS rp_group,
                a."type" AS rp_type,
                a.sub_type,
                a.property,
                a.fact_level,
                a.category,
                a.news_type,
                a.rp_source_id,
                a.source_name,
                a.provider_id,
                a.provider_story_id,
                a.headline,
                a.css
            FROM {table_name} a
            INNER JOIN single_firm sf
                ON a.provider_id = sf.provider_id
                AND a.provider_story_id = sf.provider_story_id
            WHERE a.entity_type = 'COMP'
              AND a.country_code = 'US'
              AND a.relevance >= 90
              AND a.timestamp_utc >= '{start_date}'
              AND a.timestamp_utc <= '{end_date}'
            """

            df_year = db.raw_sql(query)
            print(f"  {table_name}: {len(df_year):,} rows")

            arrow_table = pa.Table.from_pandas(df_year, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, arrow_table.schema)
            writer.write_table(arrow_table)
            total_rows += len(df_year)
            del df_year, arrow_table

    finally:
        if writer:
            writer.close()
        db.close()

    print(f"Total RavenPack headlines: {total_rows:,}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    pull_ravenpack(start_date=RP_START_DATE, end_date=RP_END_DATE)
