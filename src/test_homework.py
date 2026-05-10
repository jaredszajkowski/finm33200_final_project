"""
Unit tests for homework functions.

Run locally with:
    pytest src/test_homework.py -v
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# HW Step 1: compute_return_window
# ---------------------------------------------------------------------------

def test_compute_return_window():
    """Compound return window computation with known values."""
    from clean_labels import compute_return_window

    # 2 stocks, 5 days each
    dates = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 5), eager=True).to_list()
    df = pl.DataFrame({
        "permno": [1] * 5 + [2] * 5,
        "date": dates + dates,
        "ret": [
            # Stock 1: constant 1% daily
            0.01, 0.01, 0.01, 0.01, 0.01,
            # Stock 2: varying returns
            0.02, -0.01, 0.03, 0.00, -0.02,
        ],
    })

    result = compute_return_window(df, window=(-1, 1))

    # Expected columns
    assert set(result.columns) == {"permno", "date", "ret_window"}

    # With window=(-1,1), edge days are dropped (null at boundaries)
    # Each stock keeps 3 interior days
    stock1 = result.filter(pl.col("permno") == 1).sort("date")
    assert len(stock1) == 3
    expected_compound = (1.01) ** 3 - 1
    for val in stock1["ret_window"].to_list():
        assert val == pytest.approx(expected_compound, rel=1e-9)

    # Stock 2, day 2 (2024-01-02): window = days 1,2,3
    #   (1+0.02)*(1-0.01)*(1+0.03) - 1
    stock2 = result.filter(pl.col("permno") == 2).sort("date")
    assert len(stock2) == 3
    assert stock2["ret_window"][0] == pytest.approx(
        (1.02) * (0.99) * (1.03) - 1, rel=1e-9
    )

    # Stock 2, day 3 (2024-01-03): window = days 2,3,4
    #   (1-0.01)*(1+0.03)*(1+0.00) - 1
    assert stock2["ret_window"][1] == pytest.approx(
        (0.99) * (1.03) * (1.00) - 1, rel=1e-9
    )

    # Stock 2, day 4 (2024-01-04): window = days 3,4,5
    #   (1+0.03)*(1+0.00)*(1-0.02) - 1
    assert stock2["ret_window"][2] == pytest.approx(
        (1.03) * (1.00) * (0.98) - 1, rel=1e-9
    )


# ---------------------------------------------------------------------------
# HW Step 2: assign_binary_label
# ---------------------------------------------------------------------------

def test_assign_binary_label():
    """Binary label: positive ret_window -> 1, zero or negative -> 0."""
    from clean_labels import assign_binary_label

    df = pl.DataFrame({
        "ret_window": [0.05, -0.03, 0.0, 0.001, -0.001],
    })

    result = assign_binary_label(df)

    assert "label" in result.columns
    assert result["label"].to_list() == [1, 0, 0, 1, 0]


# ---------------------------------------------------------------------------
# HW Step 3: mean_pool_embeddings
# ---------------------------------------------------------------------------

def test_mean_pool_embeddings():
    """Mean pooling of BERT-like output with attention masking."""
    import torch
    from embed_bert import mean_pool_embeddings

    # batch_size=2, seq_len=4, hidden_dim=3
    hidden = torch.tensor([
        # Sentence 1: 3 real tokens + 1 padding
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0],
         [0.0, 0.0, 0.0]],
        # Sentence 2: 2 real tokens + 2 padding
        [[10.0, 20.0, 30.0],
         [40.0, 50.0, 60.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]],
    ])

    attention_mask = torch.tensor([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
    ])

    class MockOutput:
        def __init__(self, lhs):
            self.last_hidden_state = lhs

    result = mean_pool_embeddings(MockOutput(hidden), attention_mask)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)

    # Sentence 1: mean of 3 tokens = (1+4+7)/3, (2+5+8)/3, (3+6+9)/3
    np.testing.assert_allclose(result[0], [4.0, 5.0, 6.0], atol=1e-5)

    # Sentence 2: mean of 2 tokens = (10+40)/2, (20+50)/2, (30+60)/2
    np.testing.assert_allclose(result[1], [25.0, 35.0, 45.0], atol=1e-5)


# ---------------------------------------------------------------------------
# HW Step 4: train_logistic_sentiment
# ---------------------------------------------------------------------------

def test_train_logistic_sentiment():
    """Logistic regression training with synthetic linearly separable data."""
    from train_rolling_model import train_logistic_sentiment

    rng = np.random.RandomState(42)
    n = 200

    # Linearly separable: class 0 at -2, class 1 at +2
    X_train = np.vstack([rng.randn(n, 5) - 2, rng.randn(n, 5) + 2])
    y_train = np.array([0] * n + [1] * n)

    X_val = np.vstack([rng.randn(50, 5) - 2, rng.randn(50, 5) + 2])
    y_val = np.array([0] * 50 + [1] * 50)

    model, best_C, best_acc = train_logistic_sentiment(
        X_train, y_train, X_val, y_val, c_grid=[0.01, 0.1, 1.0],
    )

    # Returns a fitted model
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    # Scaler attached
    assert hasattr(model, "_scaler")

    # best_C from the grid
    assert best_C in [0.01, 0.1, 1.0]

    # High accuracy on separable data
    assert best_acc > 0.90

    # Generalizes to new data
    X_test = np.vstack([rng.randn(30, 5) - 2, rng.randn(30, 5) + 2])
    y_test = np.array([0] * 30 + [1] * 30)
    X_test_sc = model._scaler.transform(X_test)
    test_acc = (model.predict(X_test_sc) == y_test).mean()
    assert test_acc > 0.85


# ---------------------------------------------------------------------------
# HW Step 5: Self-affirmation
# ---------------------------------------------------------------------------

def test_affirm_replication():
    """Confirm the student has run the full pipeline and reviewed results."""
    from affirm_replication import I_HAVE_COMPLETED_THE_REPLICATION

    assert I_HAVE_COMPLETED_THE_REPLICATION, (
        "Set I_HAVE_COMPLETED_THE_REPLICATION = True in src/affirm_replication.py "
        "after running the full pipeline and verifying predictions."
    )
