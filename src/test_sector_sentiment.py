"""Tests for the sector sentiment pipeline."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))


def test_run_rolling_returns_predictions_keyed_by_story_id(tmp_path, monkeypatch):
    """`run_rolling_sentiment_analysis` must return a per-headline predictions
    list, with rp_story_id, oos_year, and p_up. The order of p_up must align
    with the rp_story_id of the same row."""
    import train_rolling_model as trm

    rng = np.random.default_rng(0)
    n = 50
    df_labeled = pl.DataFrame({
        "rp_story_id": [f"s{i:03d}" for i in range(n)],
        "article_date": pl.date_range(
            pl.date(2010, 1, 1),
            pl.date(2010, 1, 1).offset_by(f"{n - 1}d"),
            interval="1d",
            eager=True,
        ),
        "label": rng.integers(0, 2, n).astype(np.int8),
    })

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    df_emb = pl.DataFrame({
        "rp_story_id": [f"s{i:03d}" for i in range(n)],
        **{f"dim_{k}": rng.standard_normal(n).tolist() for k in range(4)},
    })
    df_emb.write_parquet(chunk_dir / "chunk_0.parquet")

    monkeypatch.setattr(trm, "DATA_DIR", tmp_path)

    windows = [{
        "oos_year": 2010,
        "train": ("2010-01-01", "2010-01-30"),
        "val":   ("2010-01-31", "2010-02-09"),
        "test":  ("2010-02-10", "2010-02-19"),
    }]

    results, predictions = trm.run_rolling_sentiment_analysis(
        chunk_dir=chunk_dir,
        df_labeled=df_labeled,
        windows=windows,
        model_name="unit",
        checkpoint_path=tmp_path / "ckpt.json",
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
    )

    assert 2010 in results
    assert "y_proba" in results[2010]

    assert len(predictions) == len(results[2010]["y_proba"])
    assert {"rp_story_id", "oos_year", "p_up"} <= set(predictions[0].keys())
    assert all(p["oos_year"] == 2010 for p in predictions)
    assert [p["p_up"] for p in predictions] == results[2010]["y_proba"]


def test_compute_etf_returns():
    """`compute_etf_returns` turns a long (ticker, date, adj_close) frame into
    one with daily simple returns per ticker, with the first day per ticker
    having a null return."""
    from pull_sector_etfs import compute_etf_returns

    df = pl.DataFrame({
        "ticker": ["XLK", "XLK", "XLK", "XLF", "XLF"],
        "date": [
            pl.date(2020, 1, 2), pl.date(2020, 1, 3), pl.date(2020, 1, 6),
            pl.date(2020, 1, 2), pl.date(2020, 1, 3),
        ],
        "adj_close": [100.0, 101.0, 99.99, 50.0, 50.5],
    })
    out = compute_etf_returns(df)

    assert out.columns == ["ticker", "date", "adj_close", "ret"]

    xlk_d1 = out.filter(
        (pl.col("ticker") == "XLK") & (pl.col("date") == pl.date(2020, 1, 2))
    )
    assert xlk_d1["ret"].item() is None

    xlk_d2 = out.filter(
        (pl.col("ticker") == "XLK") & (pl.col("date") == pl.date(2020, 1, 3))
    )
    assert xlk_d2["ret"].item() == pytest.approx(0.01)


def test_join_predictions_to_headlines():
    """Inner-join headlines with predictions; emit per-headline sentiment_score."""
    from build_sector_panel import join_predictions_to_headlines

    df_headlines = pl.DataFrame({
        "rp_story_id": ["s1", "s2", "s3", "s4"],
        "permno": [10001.0, 10001.0, 10002.0, 10003.0],
        "article_date": [
            pl.date(2017, 1, 5), pl.date(2017, 1, 6),
            pl.date(2017, 1, 5), pl.date(2017, 1, 5),
        ],
        "market_cap": [1.0e9, 1.0e9, 2.0e9, 3.0e9],
        "gsector": ["45", "45", "40", "40"],
    })
    df_preds = pl.DataFrame({
        "rp_story_id": ["s1", "s2", "s3"],
        "oos_year": [2017, 2017, 2017],
        "p_up": [0.7, 0.3, 0.55],
    })

    out = join_predictions_to_headlines(df_headlines, df_preds)

    assert len(out) == 3
    assert "sentiment_score" in out.columns
    s1 = out.filter(pl.col("rp_story_id") == "s1")["sentiment_score"].item()
    assert s1 == pytest.approx(0.2)
    s2 = out.filter(pl.col("rp_story_id") == "s2")["sentiment_score"].item()
    assert s2 == pytest.approx(-0.2)


def test_aggregate_sentiment_value_weighted():
    """VW mean of sentiment_score per (gsector, article_date)."""
    from build_sector_panel import aggregate_sentiment

    df = pl.DataFrame({
        "gsector": ["45", "45", "45", "40"],
        "article_date": [
            pl.date(2017, 1, 5), pl.date(2017, 1, 5),
            pl.date(2017, 1, 6), pl.date(2017, 1, 5),
        ],
        "market_cap": [1.0e9, 3.0e9, 2.0e9, 5.0e9],
        "sentiment_score": [0.2, -0.2, 0.1, 0.4],
    })

    out = aggregate_sentiment(df).sort(["gsector", "date"])

    r = out.filter(
        (pl.col("gsector") == "45") & (pl.col("date") == pl.date(2017, 1, 5))
    )
    assert r["sentiment"].item() == pytest.approx(-0.1)
    assert r["n_headlines"].item() == 2

    r = out.filter(pl.col("gsector") == "40")
    assert r["sentiment"].item() == pytest.approx(0.4)
    assert r["n_headlines"].item() == 1


def test_aggregate_sentiment_skips_zero_weight():
    """Sector-days where total market_cap is 0 (or all null) are dropped."""
    from build_sector_panel import aggregate_sentiment

    df = pl.DataFrame({
        "gsector": ["45"],
        "article_date": [pl.date(2017, 1, 5)],
        "market_cap": [0.0],
        "sentiment_score": [0.2],
    })
    out = aggregate_sentiment(df)
    assert len(out) == 0


def test_compute_synthetic_sector_returns():
    """For each (gsector, date), VW(ret) over S&P 500 stocks classified into
    that sector on that date."""
    from build_sector_panel import compute_synthetic_sector_returns

    df_crsp = pl.DataFrame({
        "permno": [10001.0, 10002.0, 10001.0, 10002.0],
        "date": [
            pl.date(2017, 1, 5), pl.date(2017, 1, 5),
            pl.date(2017, 1, 6), pl.date(2017, 1, 6),
        ],
        "ret": [0.01, -0.02, 0.03, 0.005],
        "market_cap": [1.0e9, 4.0e9, 1.0e9, 4.0e9],
    })
    df_sp500 = pl.DataFrame({
        "permno": [10001.0, 10001.0, 10002.0],
        "gsector": ["45", "50", "45"],
        "effstartdt": [pl.date(2010, 1, 1), pl.date(2017, 1, 6), pl.date(2010, 1, 1)],
        "effenddt":   [pl.date(2017, 1, 5), pl.date(2030, 1, 1), pl.date(2030, 1, 1)],
    })

    out = compute_synthetic_sector_returns(df_crsp, df_sp500).sort(["gsector", "date"])

    r = out.filter(
        (pl.col("gsector") == "45") & (pl.col("date") == pl.date(2017, 1, 5))
    )
    assert r["synthetic_ret"].item() == pytest.approx(-0.014)

    r = out.filter(
        (pl.col("gsector") == "45") & (pl.col("date") == pl.date(2017, 1, 6))
    )
    assert r["synthetic_ret"].item() == pytest.approx(0.005)

    r = out.filter(
        (pl.col("gsector") == "50") & (pl.col("date") == pl.date(2017, 1, 6))
    )
    assert r["synthetic_ret"].item() == pytest.approx(0.03)


def test_attach_etf_returns_uses_sector_map():
    """Each (gsector, date) row gets the matching SPDR ETF return (NaN before
    the ETF's inception)."""
    from build_sector_panel import attach_etf_returns

    df_panel = pl.DataFrame({
        "gsector": ["45", "45", "60"],
        "date": [pl.date(2017, 1, 5), pl.date(2017, 1, 6), pl.date(2014, 1, 5)],
        "sentiment": [0.1, 0.2, 0.05],
        "n_headlines": [3, 2, 1],
        "synthetic_ret": [0.01, 0.02, 0.005],
    })
    df_etf = pl.DataFrame({
        "ticker": ["XLK", "XLK", "XLRE"],
        "date": [pl.date(2017, 1, 5), pl.date(2017, 1, 6), pl.date(2017, 1, 5)],
        "adj_close": [50.0, 50.5, 30.0],
        "ret": [0.005, 0.01, 0.002],
    })

    out = attach_etf_returns(df_panel, df_etf).sort(["gsector", "date"])

    assert (
        out.filter(
            (pl.col("gsector") == "45") & (pl.col("date") == pl.date(2017, 1, 5))
        )["etf_ret"].item()
        == pytest.approx(0.005)
    )
    assert out.filter(pl.col("gsector") == "60")["etf_ret"].item() is None


def test_build_panel_end_to_end_smoke(tmp_path, monkeypatch):
    """End-to-end on tiny in-memory frames: writes the panel parquet with
    expected columns and value bounds."""
    from build_sector_panel import build_panel

    pl.DataFrame({
        "rp_story_id": ["s1"],
        "permno": [10001.0],
        "article_date": [pl.date(2017, 1, 5)],
        "headline": ["news"],
        "ret_window": [0.01],
        "label": [1],
        "market_cap": [1.0e9],
        "gsector": ["45"],
        "sector_name": ["Information Technology"],
    }).write_parquet(tmp_path / "labeled_dataset_with_sector.parquet")

    pl.DataFrame({
        "rp_story_id": ["s1"], "oos_year": [2017], "p_up": [0.7],
    }).write_parquet(tmp_path / "rolling_predictions_unit.parquet")

    pl.DataFrame({
        "permno": [10001.0],
        "date": [pl.date(2017, 1, 5)],
        "ret": [0.01],
        "market_cap": [1.0e9],
    }).write_parquet(tmp_path / "CRSP_daily_stock.parquet")

    pl.DataFrame({
        "permno": [10001.0],
        "gsector": ["45"],
        "effstartdt": [pl.date(2010, 1, 1)],
        "effenddt": [pl.date(2030, 1, 1)],
    }).write_parquet(tmp_path / "sp500_constituents.parquet")

    pl.DataFrame({
        "ticker": ["XLK"], "date": [pl.date(2017, 1, 5)],
        "adj_close": [50.0], "ret": [0.005],
    }).write_parquet(tmp_path / "sector_etfs.parquet")

    import build_sector_panel as bsp
    monkeypatch.setattr(bsp, "DATA_DIR", tmp_path)

    out_path = build_panel("unit")
    assert out_path.exists()
    df = pl.read_parquet(out_path)
    assert set(df.columns) == {
        "gsector", "sector_name", "date", "n_headlines",
        "sentiment", "synthetic_ret", "etf_ret",
    }
    assert len(df) == 1
    row = df.row(0, named=True)
    assert row["sentiment"] == pytest.approx(0.2)
    assert row["synthetic_ret"] == pytest.approx(0.01)
    assert row["etf_ret"] == pytest.approx(0.005)
