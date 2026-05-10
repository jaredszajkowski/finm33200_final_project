"""
Generate BERT mean-pool embeddings for news headlines.

Uses bert-base-uncased (768-dim) from HuggingFace transformers.
Mean pooling over non-padding token embeddings from the last hidden state.

Checkpointing: embeddings are saved in chunks. On resume, already-embedded
story IDs are skipped.

Input: _data/labeled_dataset_with_sector.parquet
Output: _data/embeddings_bert_chunks/chunk_*.parquet
  Columns: rp_story_id + 768 float32 columns (dim_0 ... dim_767)

Usage
-----
    python src/embed_bert.py
"""

from pathlib import Path

import numpy as np
import polars as pl

from settings import config

DATA_DIR = config("DATA_DIR")

BERT_MODEL_NAME = "bert-base-uncased"
EMBED_DIM = 768
MAX_LENGTH = 64
BATCH_SIZE = 128


def load_bert_model(model_name=BERT_MODEL_NAME, device=None):
    """Load BERT tokenizer and model."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def mean_pool_embeddings(model_output, attention_mask):
    """Mean pool the last hidden state, excluding padding tokens."""
    # === HW STEP 3 START ===
    # raise NotImplementedError("TODO: implement mean_pool_embeddings")
    
    token_embeddings = model_output.last_hidden_state  # (batch, seq_len, hidden_dim)
    mask = attention_mask.unsqueeze(-1).float()        # (batch, seq_len, 1)
    summed = (token_embeddings * mask).sum(dim=1)      # (batch, hidden_dim)
    counts = mask.sum(dim=1).clamp(min=1e-9)           # (batch, 1)
    return (summed / counts).cpu().float().numpy()     # (batch, hidden_dim)
    
    # === HW STEP 3 END ===


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


def embed_headlines_bert(
    headlines, story_ids, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
    model_name=BERT_MODEL_NAME, device=None, checkpoint_every=500, chunk_dir=None,
):
    """Embed a list of headlines using BERT mean pooling."""
    import sys
    import torch
    from tqdm import tqdm

    if chunk_dir is None:
        chunk_dir = DATA_DIR / "embeddings_bert_chunks"
    chunk_dir = Path(chunk_dir)

    tokenizer, model, device = load_bert_model(model_name=model_name, device=device)
    use_autocast = device in ("cuda", "mps")

    n = len(headlines)
    pending_embeddings = []
    pending_ids = []
    chunk_num = len(list(chunk_dir.glob("chunk_*.parquet"))) if chunk_dir.exists() else 0
    total_embedded = 0

    total_batches = (n + batch_size - 1) // batch_size
    print(f"Starting: {total_batches:,} batches, batch_size={batch_size}")
    batch_iter = tqdm(range(0, n, batch_size), desc="BERT embedding", unit="batch", total=total_batches)

    try:
        for start in batch_iter:
            batch_texts = headlines[start:start + batch_size]
            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.inference_mode():
                if use_autocast:
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        output = model(**encoded)
                else:
                    output = model(**encoded)

            batch_emb = mean_pool_embeddings(output, encoded["attention_mask"])
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

    chunk_dir = DATA_DIR / "embeddings_bert_chunks"
    chunk_ids, _ = _load_chunk_ids(chunk_dir)

    df_todo = df.filter(~pl.col("rp_story_id").is_in(list(chunk_ids)))
    print(f"Already embedded: {len(chunk_ids):,}. Remaining: {len(df_todo):,}")

    if len(df_todo) > 0:
        headlines = df_todo["headline"].fill_null("").to_list()
        story_ids = df_todo["rp_story_id"].to_list()
        embed_headlines_bert(headlines, story_ids, chunk_dir=chunk_dir)
    else:
        print("All headlines already embedded.")
