"""
Rolling-window logistic regression sentiment model.

Implements the rolling 8-year window scheme from Chen, Kelly, Xiu (2022):
  - 6-year training period
  - 2-year validation period (for L2 penalty tuning)
  - 1-year OOS test period

OOS window is determined dynamically from the labeled dataset date range:
  OOS_START = min_year + TRAIN_YEARS + VAL_YEARS
  OOS_END   = max_year

The logistic regression uses L2 regularization with C tuned on validation.

Input:
  - _data/labeled_dataset_with_sector.parquet
  - _data/embeddings_{model}_chunks/chunk_*.parquet

Output: _data/rolling_results_{model}.json

Usage
-----
    python src/train_rolling_model.py                # all available models
    python src/train_rolling_model.py tfidf          # single model
    python src/train_rolling_model.py tfidf bert     # specific models
"""

import gc
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from settings import config

DATA_DIR = config("DATA_DIR")

# Registry of embedding models: name -> chunk directory
EMBEDDING_REGISTRY = {
    "bert": "embeddings_bert_chunks",
    "gemma": "embeddings_gemma_chunks",
}

# L2 penalty grid (C = inverse regularization strength)
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

TRAIN_YEARS = 3
VAL_YEARS = 1

# Cap rows per split to fit in RAM
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None


def get_rolling_windows(df_labeled: pl.DataFrame):
    """Generate rolling window definitions dynamically from data.

    Determines OOS start/end from the labeled dataset date range.
    """
    years = df_labeled["article_date"].dt.year()
    min_year = years.min()
    max_year = years.max()

    oos_start = min_year + TRAIN_YEARS + VAL_YEARS
    oos_end = max_year

    print(f"Data years: {min_year}-{max_year}")
    print(f"OOS window: {oos_start}-{oos_end} ({oos_end - oos_start + 1} years)")

    windows = []
    for oos_year in range(oos_start, oos_end + 1):
        windows.append(
            {
                "oos_year": oos_year,
                "train": (
                    f"{oos_year - VAL_YEARS - TRAIN_YEARS}-01-01",
                    f"{oos_year - VAL_YEARS - 1}-12-31",
                ),
                "val": (f"{oos_year - VAL_YEARS}-01-01", f"{oos_year - 1}-12-31"),
                "test": (f"{oos_year}-01-01", f"{oos_year}-12-31"),
            }
        )
    return windows


def _get_split(df_labeled: pl.DataFrame, start_str: str, end_str: str) -> pl.DataFrame:
    """Extract rows where article_date is in [start, end]."""
    from datetime import date

    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    return df_labeled.filter(
        (pl.col("article_date") >= start) & (pl.col("article_date") <= end)
    )


def _load_embeddings_for_ids(chunk_dir, story_ids):
    """Load embeddings for specific story IDs from chunk parquets.

    Uses PyArrow for efficient row-level filtering.
    """
    import pyarrow.parquet as pq
    import pyarrow as pa

    chunk_dir = Path(chunk_dir)
    story_ids_set = set(story_ids)
    chunk_files = sorted(chunk_dir.glob("chunk_*.parquet"))

    parts = []
    found = 0
    for f in chunk_files:
        id_col = pq.read_table(f, columns=["rp_story_id"]).column("rp_story_id")
        mask = [sid.as_py() in story_ids_set for sid in id_col]

        if not any(mask):
            del id_col, mask
            continue

        tbl = pq.read_table(f)
        filtered = tbl.filter(mask)
        parts.append(filtered)
        found += filtered.num_rows

        del tbl, id_col, mask, filtered

        if found >= len(story_ids_set):
            break

    if not parts:
        return pl.DataFrame()

    combined = pa.concat_tables(parts)
    del parts

    df = pl.from_arrow(combined)
    del combined

    df = df.unique(subset=["rp_story_id"])
    return df


def train_logistic_sentiment(X_train, y_train, X_val, y_val, c_grid=C_GRID):
    """Train logistic regression with L2 penalty, tuned on validation set."""
    # === HW STEP 4 START ===
    # raise NotImplementedError("TODO: implement train_logistic_sentiment")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    best_acc = -1.0
    best_model = None
    best_C = None

    for C in c_grid:
        model = LogisticRegression(
            C=C,
            l1_ratio=0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train_sc, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val_sc))
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_C = C

    best_model._scaler = scaler
    return best_model, best_C, best_acc

    # === HW STEP 4 END ===


def predict_sentiment_proba(model, X):
    """Predict class probabilities using the fitted model."""
    X_sc = model._scaler.transform(X)
    return model.predict_proba(X_sc)[:, 1]


def run_rolling_sentiment_analysis(
    chunk_dir,
    df_labeled,
    windows,
    model_name="model",
    checkpoint_path=None,
    max_train_samples=MAX_TRAIN_SAMPLES,
    max_val_samples=MAX_VAL_SAMPLES,
    max_test_samples=MAX_TEST_SAMPLES,
):
    """Run rolling sentiment analysis for all OOS years."""
    if checkpoint_path is None:
        checkpoint_path = DATA_DIR / f"rolling_results_{model_name}_checkpoint.json"
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = {int(k): v for k, v in json.load(f).items()}
        print(f"Resuming from checkpoint: {len(results)} years already done")
    else:
        results = {}

    predictions = []

    for w in windows:
        oos_year = w["oos_year"]

        if oos_year in results:
            print(f"OOS {oos_year}: already in checkpoint, skipping")
            continue

        df_train_labels = _get_split(df_labeled, *w["train"])
        df_val_labels = _get_split(df_labeled, *w["val"])
        df_test_labels = _get_split(df_labeled, *w["test"])

        df_n_pre_train, df_n_pre_val, df_n_pre_test = (
            len(df_train_labels),
            len(df_val_labels),
            len(df_test_labels),
        )

        # Subsample if too large
        if max_train_samples and len(df_train_labels) > max_train_samples:
            df_train_labels = df_train_labels.sample(n=max_train_samples, seed=42)
        if max_val_samples and len(df_val_labels) > max_val_samples:
            df_val_labels = df_val_labels.sample(n=max_val_samples, seed=42)
        if max_test_samples and len(df_test_labels) > max_test_samples:
            df_test_labels = df_test_labels.sample(n=max_test_samples, seed=42)

        n_train, n_val, n_test = (
            len(df_train_labels),
            len(df_val_labels),
            len(df_test_labels),
        )

        print(
            f"OOS {oos_year}: "
            f"train {w['train'][0][:4]}-{w['train'][1][:4]} (n={n_train:,}, df={df_n_pre_train:,}), "
            f"val {w['val'][0][:4]}-{w['val'][1][:4]} (n={n_val:,}, df={df_n_pre_val:,}), "
            f"test {oos_year} (n={n_test:,}, df={df_n_pre_test:,})",
            end=" ... ",
            flush=True,
        )

        if n_train < 100 or n_val < 10 or n_test < 10:
            print("Insufficient data. Skipping.")
            continue

        # Load needed embeddings
        all_ids = (
            df_train_labels["rp_story_id"].to_list()
            + df_val_labels["rp_story_id"].to_list()
            + df_test_labels["rp_story_id"].to_list()
        )
        all_ids_unique = list(set(all_ids))
        print(f"loading {len(all_ids_unique):,} embeddings ... ", end="", flush=True)
        df_emb = _load_embeddings_for_ids(chunk_dir, all_ids_unique)

        embed_cols = [c for c in df_emb.columns if c.startswith("dim_")]

        def _build_Xy(df_labels):
            merged = df_labels.join(df_emb, on="rp_story_id", how="inner")
            X = merged.select(embed_cols).to_numpy().astype(np.float32)
            y = merged["label"].to_numpy()
            ids = merged["rp_story_id"].to_list()
            return X, y, ids

        X_train, y_train, _ = _build_Xy(df_train_labels)
        X_val, y_val, _ = _build_Xy(df_val_labels)
        X_test, y_test, test_ids = _build_Xy(df_test_labels)

        model, best_C, val_acc = train_logistic_sentiment(
            X_train, y_train, X_val, y_val
        )

        y_proba = predict_sentiment_proba(model, X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        test_acc = accuracy_score(y_test, y_pred)

        print(f"acc={test_acc:.3f}, C={best_C}")

        results[oos_year] = {
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist(),
            "accuracy": test_acc,
            "best_C": best_C,
            "val_accuracy": val_acc,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        }

        for sid, p in zip(test_ids, y_proba.tolist()):
            predictions.append({"rp_story_id": sid, "oos_year": oos_year, "p_up": p})

        with open(checkpoint_path, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)

        del df_emb, X_train, X_val, X_test
        gc.collect()

    return results, predictions


def save_results(results, model_name, data_dir=DATA_DIR):
    """Save rolling results to JSON."""
    path = Path(data_dir) / f"rolling_results_{model_name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {path}")


def load_results(model_name, data_dir=DATA_DIR):
    """Load rolling results from JSON."""
    path = Path(data_dir) / f"rolling_results_{model_name}.json"
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def _available_models(data_dir=DATA_DIR):
    """Return list of model names that have chunk directories with data."""
    available = []
    for name, chunk_dir_name in EMBEDDING_REGISTRY.items():
        chunk_dir = Path(data_dir) / chunk_dir_name
        if chunk_dir.exists() and any(chunk_dir.glob("chunk_*.parquet")):
            available.append(name)
    return available


if __name__ == "__main__":
    import sys

    cli_models = [a for a in sys.argv[1:] if not a.startswith("-")]
    if cli_models:
        models_to_run = cli_models
    else:
        models_to_run = _available_models()

    if not models_to_run:
        print("No embedding chunk directories found. Run an embed script first.")
        sys.exit(1)

    print(f"Models to train: {', '.join(models_to_run)}")
    print("Loading labeled dataset...")
    df_labeled = pl.read_parquet(DATA_DIR / "labeled_dataset_with_sector.parquet")
    windows = get_rolling_windows(df_labeled)

    for model_name in models_to_run:
        print(f"\n{'=' * 60}")
        print(f"=== Rolling sentiment model: {model_name} ===")
        print(f"{'=' * 60}")

        chunk_dir_name = EMBEDDING_REGISTRY.get(model_name)
        if chunk_dir_name is None:
            print(f"Unknown model: {model_name}")
            continue

        chunk_dir = DATA_DIR / chunk_dir_name
        if not chunk_dir.exists():
            print(f"Skipping {model_name}: {chunk_dir} not found")
            continue

        checkpoint = DATA_DIR / f"rolling_results_{model_name}_checkpoint.json"
        results, predictions = run_rolling_sentiment_analysis(
            chunk_dir,
            df_labeled,
            windows,
            model_name=model_name,
            checkpoint_path=checkpoint,
        )
        save_results(results, model_name)

        pred_path = DATA_DIR / f"rolling_predictions_{model_name}.parquet"
        pl.DataFrame(predictions).write_parquet(pred_path)
        print(f"Saved {len(predictions):,} per-headline predictions: {pred_path}")

        if checkpoint.exists():
            checkpoint.unlink()
            print(f"{model_name} checkpoint cleaned up.")
