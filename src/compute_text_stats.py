"""
Compute headline text statistics for Table 3.

Pre-computes percentile distributions (1st, 25th, 50th, 75th, 99th) and
means for character count, BERT token count, and word count on a random
100K sample of headlines from the labeled training dataset (i.e. the
headlines that survive the return-window join, which is the population
actually used for embedding and modeling).

Input:
  - _data/labeled_dataset_with_sector.parquet

Output:
  - _data/text_stats.json

Usage
-----
    python src/compute_text_stats.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
HF_TOKEN = config("HF_TOKEN")

SAMPLE_SIZE = 100_000
RANDOM_SEED = 42
PERCENTILES = [1, 25, 50, 75, 99]


def count_tokens(texts, model_name, batch_size=512, token=None):
    """Count tokens per text without special tokens.

    Parameters
    ----------
    texts : list of str
    model_name : str
    batch_size : int
    token : str or None
        HuggingFace access token, required for gated repos.

    Returns
    -------
    np.ndarray of int
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    counts = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            return_length=True,
        )
        counts.extend(encoded["length"])
    return np.array(counts)


def _percentile_dict(arr):
    """Compute percentiles and mean, return as dict."""
    pcts = np.percentile(arr, PERCENTILES)
    result = {}
    for p, val in zip(PERCENTILES, pcts):
        result[f"{p}th"] = round(float(val), 1)
    result["mean"] = round(float(arr.mean()), 1)
    return result


def compute_text_stats(data_dir=DATA_DIR):
    """Compute text statistics and save to JSON."""
    df = pl.read_parquet(
        data_dir / "labeled_dataset_with_sector.parquet", columns=["headline"]
    )
    df = df.filter(pl.col("headline").is_not_null())
    total = len(df)

    if total > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, seed=RANDOM_SEED)
        print(f"Sampled {SAMPLE_SIZE:,} of {total:,} headlines")

    headlines = df["headline"].to_list()
    print(f"Computing text stats for {len(headlines):,} headlines...")

    char_counts = np.array([len(h) for h in headlines])
    word_counts = np.array([len(h.split()) for h in headlines])

    print("Counting BERT tokens...")
    bert_counts = count_tokens(headlines, "bert-base-uncased")

    print("Counting Gemma tokens...")
    gemma_counts = count_tokens(headlines, "google/gemma-3-1b-it", token=HF_TOKEN)

    stats = {
        "sample_size": len(headlines),
        "total_headlines": total,
        "random_seed": RANDOM_SEED,
        "percentiles": PERCENTILES,
        "characters": _percentile_dict(char_counts),
        "bert_tokens": _percentile_dict(bert_counts),
        "gemma_tokens": _percentile_dict(gemma_counts),
        "words": _percentile_dict(word_counts),
    }

    path = data_dir / "text_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {path}")

    return stats


if __name__ == "__main__":
    stats = compute_text_stats()
    for measure in ("characters", "bert_tokens", "gemma_tokens", "words"):
        print(f"  {measure}: {stats[measure]}")
