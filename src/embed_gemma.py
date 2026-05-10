"""
Generate EmbeddingGemma embeddings for news headlines.

Uses google/embeddinggemma-300m (768-dim) via sentence-transformers.
Pooling and normalization are handled internally by SentenceTransformer.
The "Classification" task prompt is used since the downstream task is
predicting binary return direction from headlines.

Checkpointing: embeddings are saved in chunks. On resume, already-embedded
story IDs are skipped.

Input: _data/labeled_dataset_with_sector.parquet
Output: _data/embeddings_gemma_chunks/chunk_*.parquet
  Columns: rp_story_id + 768 float32 columns (dim_0 ... dim_767)

Usage
-----
    python src/embed_gemma.py
"""

from pathlib import Path

import numpy as np
import polars as pl

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
HF_TOKEN = config("HF_TOKEN", default=None)

GEMMA_MODEL_NAME = "google/embeddinggemma-300m"
EMBED_DIM = 768
MAX_LENGTH = 64
BATCH_SIZE = 64
PROMPT_NAME = "Classification"


def load_gemma_model(model_name=GEMMA_MODEL_NAME, device=None):
    """Load EmbeddingGemma via sentence-transformers."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device, token=HF_TOKEN)
    model.eval()
    return model, device


def _save_chunk(embeddings_list, story_ids, chunk_dir, chunk_num):
    """Save a chunk of embeddings to the chunk directory."""
    chunk_dir = Path(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.vstack(embeddings_list)
    cols = [f"dim_{i}" for i in range(embeddings.shape[1])]
    df = pl.DataFrame(
        {"rp_story_id": story_ids, **{c: embeddings[:, i].tolist() for i, c in enumerate(cols)}}
    )
    path = chunk_dir / f"chunk_{chunk_num:06d}.parquet"
    df.write_parquet(path)
    print(f"  Chunk saved: {len(story_ids):,} embeddings -> {path.name}")


def _load_chunk_ids(chunk_dir):
    """Load only story IDs from chunks (fast -- skips embedding columns)."""
    chunk_dir = Path(chunk_dir)
    if not chunk_dir.exists():
        return set(), 0

    chunk_files = sorted(chunk_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        return set(), 0

    story_ids = set()
    for f in chunk_files:
        ids = pl.scan_parquet(f).select("rp_story_id").collect()["rp_story_id"].to_list()
        story_ids.update(ids)

    print(f"Scanned {len(chunk_files)} chunks: {len(story_ids):,} story IDs")
    return story_ids, len(chunk_files)


def embed_headlines_gemma(
    headlines, story_ids, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
    model_name=GEMMA_MODEL_NAME, device=None, checkpoint_every=100, chunk_dir=None,
    prompt_name=PROMPT_NAME,
):
    """Embed a list of headlines using EmbeddingGemma."""
    from tqdm import tqdm

    if chunk_dir is None:
        chunk_dir = DATA_DIR / "embeddings_gemma_chunks"
    chunk_dir = Path(chunk_dir)

    model, device = load_gemma_model(model_name=model_name, device=device)
    model.max_seq_length = max_length

    n = len(headlines)
    pending_embeddings = []
    pending_ids = []
    chunk_num = len(list(chunk_dir.glob("chunk_*.parquet"))) if chunk_dir.exists() else 0
    total_embedded = 0

    total_batches = (n + batch_size - 1) // batch_size
    print(f"Starting: {total_batches:,} batches, batch_size={batch_size}")
    batch_iter = tqdm(range(0, n, batch_size), desc="Gemma embedding", unit="batch", total=total_batches)

    try:
        for start in batch_iter:
            batch_texts = headlines[start:start + batch_size]
            batch_emb = model.encode(
                batch_texts,
                prompt_name=prompt_name,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

            pending_embeddings.append(batch_emb)
            pending_ids.extend(story_ids[start:start + batch_size])

            batch_num_in_loop = start // batch_size
            if checkpoint_every and (batch_num_in_loop + 1) % checkpoint_every == 0:
                _save_chunk(pending_embeddings, pending_ids, chunk_dir, chunk_num)
                chunk_num += 1
                total_embedded += len(pending_ids)
                pending_embeddings = []
                pending_ids = []
    except KeyboardInterrupt:
        print("\nInterrupted! Saving progress...")
        if pending_embeddings:
            _save_chunk(pending_embeddings, pending_ids, chunk_dir, chunk_num)
        raise

    if pending_embeddings:
        _save_chunk(pending_embeddings, pending_ids, chunk_dir, chunk_num)
        total_embedded += len(pending_ids)

    print(f"Embedded {total_embedded:,} headlines into chunks.")


if __name__ == "__main__":
    print("Loading clean headlines...")
    df = (
        pl.scan_parquet(DATA_DIR / "labeled_dataset_with_sector.parquet")
        .select("rp_story_id", "headline")
        .unique(subset=["rp_story_id"])
        .collect()
    )

    chunk_dir = DATA_DIR / "embeddings_gemma_chunks"
    chunk_ids, _ = _load_chunk_ids(chunk_dir)

    df_todo = df.filter(~pl.col("rp_story_id").is_in(list(chunk_ids)))
    print(f"Already embedded: {len(chunk_ids):,}. Remaining: {len(df_todo):,}")

    if len(df_todo) > 0:
        headlines = df_todo["headline"].fill_null("").to_list()
        story_ids = df_todo["rp_story_id"].to_list()
        embed_headlines_gemma(headlines, story_ids, chunk_dir=chunk_dir)
    else:
        print("All headlines already embedded.")
