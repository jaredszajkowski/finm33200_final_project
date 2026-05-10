# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Text Embeddings: From Words to Vectors
#
# This notebook builds intuition around **text embeddings** — the core representation
# used in Chen, Kelly, and Xiu (2022) to turn financial news into predictive features.
#
# **Topics covered:**
#
# 1. Why embeddings? From bag-of-words to LLM representations
# 2. Key terminology: embeddings, cosine similarity, dimension
# 3. Your first embedding: inspecting a single vector
# 4. Batching and pairwise similarity
# 5. Word analogies via vector arithmetic
# 6. Embedding real RavenPack headlines and visualizing with PCA

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio
import polars as pl
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from settings import config

pio.renderers.default = "notebook"

client = OpenAI(api_key=config("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"
DATA_DIR = Path(config("DATA_DIR"))

# %% [markdown]
# ---
# ## Why Embeddings?
#
# Text is not numerical. To feed news articles into a statistical model we need a
# mapping **text → dense vector**. The history of that mapping, in brief:
#
# | Era | Method | Key idea |
# |-----|--------|----------|
# | Classical | Bag-of-Words / TF-IDF | Count word frequencies; sparse, high-dimensional |
# | ~2003 | LDA (Blei, Ng, Jordan) | Probabilistic topic model; lower-dimensional but still "topics" |
# | 2013 | Word2Vec (Mikolov et al.) | Neural net learns dense word vectors; analogies emerge |
# | 2018+ | LLM Embeddings (BERT, GPT) | Contextual representations; full-sentence embeddings |
#
# **Connection to CKX (Section 2):** The paper constructs a feature matrix
# $\mathbf{X}$ of dimension $D \times P$, where each of the $D$ news articles
# is represented by a $P$-dimensional embedding vector. This is the input to
# Eq. 2 (ridge regression for return prediction).
#
# **Model choice:** We use `text-embedding-3-small` (\$0.02 / 1M tokens) for
# this demo and for the full replication pipeline.

# %% [markdown]
# ---
# ## Terminology
#
# - **Word embedding:** A fixed-length vector representing a single word.
# - **Text / sentence embedding:** A fixed-length vector representing an entire
#   sentence or document (what we use here).
# - **Cosine similarity:** Measures the angle between two vectors:
#
#   $$\text{cos}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}$$
#   Values range from $-1$ (opposite) to $1$ (identical direction).
# - **Embedding dimension:** The length of the vector. For `text-embedding-3-small`
#   this is **1536**.

# %% [markdown]
# ---
# ## First Embedding
#
# Let's embed a single sentence and inspect what comes back.

# %%
sentence = "Apple reported record quarterly revenue driven by strong iPhone sales."

response = client.embeddings.create(input=[sentence], model=EMBEDDING_MODEL)

vec = np.array(response.data[0].embedding)
print(f"Dimension: {vec.shape[0]}")
print(f"First 10 values: {vec[:10].round(4)}")
print(f"Token usage: {response.usage.total_tokens}")

# %% [markdown]
# The vector has 1536 components — each is a floating-point number, typically
# small in magnitude. On its own one vector is not very informative; embeddings
# become useful when we **compare** vectors.

# %% [markdown]
# ---
# ## Batching and Pairwise Similarity
#
# We embed four sentences in a single API call and compute their pairwise
# cosine similarities.

# %%
sentences = [
    "Apple reported record quarterly revenue driven by strong iPhone sales.",
    "Tesla shares surged after the company beat earnings expectations.",
    "The Federal Reserve held interest rates steady at its latest meeting.",
    "Heavy rainfall is expected across the Midwest this weekend.",
]

response = client.embeddings.create(input=sentences, model=EMBEDDING_MODEL)
embeddings = np.array([d.embedding for d in response.data])
print(f"Embedding matrix shape: {embeddings.shape}")

# %%
sim_matrix = cosine_similarity(embeddings)

labels = ["Apple revenue", "Tesla earnings", "Fed rates", "Midwest rain"]
sim_df = pl.DataFrame(
    {"sentence": labels, **{lab: sim_matrix[:, i].round(3) for i, lab in enumerate(labels)}}
)
sim_df

# %% [markdown]
# **Interpretation:** The two stock-earnings sentences (Apple, Tesla) are most
# similar to each other. The weather sentence is the most distant from
# everything else. The Fed / rates sentence sits in between — financial but
# not equity-earnings news.

# %% [markdown]
# ---
# ## Word Analogies
#
# Mikolov et al. (2013) famously showed that word vectors support analogies via
# arithmetic:
#
# $$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$
#
# Let's test this with OpenAI's text-embedding model. **Caveat:** these models
# are optimized for full sentences, not single words, so results may be
# imperfect — but the intuition still holds.

# %%
vocabulary = [
    "king", "queen", "man", "woman", "prince", "princess", "boy", "girl",
    "doctor", "nurse", "actor", "actress",
    "France", "Paris", "Germany", "Berlin", "Italy", "Rome",
    "cat", "dog",
]

response = client.embeddings.create(input=vocabulary, model=EMBEDDING_MODEL)
word_vecs = {word: np.array(d.embedding) for word, d in zip(vocabulary, response.data)}


def analogy(a, b, c, word_vecs, top_n=5):
    """Solve: a - b + c = ?

    Returns the top_n nearest words (excluding a, b, c) by cosine similarity.
    """
    target = word_vecs[a] - word_vecs[b] + word_vecs[c]
    exclude = {a, b, c}
    candidates = {w: v for w, v in word_vecs.items() if w not in exclude}
    sims = {
        w: cosine_similarity(target.reshape(1, -1), v.reshape(1, -1))[0, 0]
        for w, v in candidates.items()
    }
    ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


# %%
print("king - man + woman = ?")
for word, score in analogy("king", "man", "woman", word_vecs):
    print(f"  {word:12s} {score:.4f}")

# %%
print("France - Paris + Berlin = ?")
for word, score in analogy("France", "Paris", "Berlin", word_vecs):
    print(f"  {word:12s} {score:.4f}")

# %% [markdown]
# Even with a sentence-optimized model, the analogy structure is visible:
# **queen** should rank near the top for the first analogy, and **Germany**
# for the second. Results may not be perfect — single-word embeddings from
# these models carry less structure than dedicated word2vec models — but the
# core idea is sound.

# %% [markdown]
# ---
# ## Transition to Financial News
#
# The toy examples above show that embeddings capture semantic relationships.
# Now let's apply the same idea to **real RavenPack headlines** — the actual
# data used in CKX (2022).
#
# The full dataset has millions of headlines. For this demo we sample **30**
# to keep API costs negligible and output readable.

# %%
rp = (
    pl.scan_parquet(DATA_DIR / "ravenpack_djpr_with_permno.parquet")
    .filter(pl.col("permno").is_not_null())
    .select("headline", "entity_name", "event_sentiment_score")
    .collect()
    .sample(n=30, seed=42)
)
print(f"Sampled {rp.height} headlines")
rp

# %% [markdown]
# ---
# ## Embed Headlines
#
# We embed all 30 headlines in a single API call.

# %%
headlines = rp["headline"].to_list()

response = client.embeddings.create(input=headlines, model=EMBEDDING_MODEL)
headline_embeddings = np.array([d.embedding for d in response.data])
print(f"Embedding matrix shape: {headline_embeddings.shape}")
print(f"This is our X matrix: D={headline_embeddings.shape[0]}, P={headline_embeddings.shape[1]}")

# %% [markdown]
# ---
# ## Pairwise Similarity: Most and Least Similar Headlines
#
# With 30 headlines we get a 30×30 similarity matrix. Let's find the
# most-similar and least-similar pairs.

# %%
sim = cosine_similarity(headline_embeddings)

# Mask the diagonal (self-similarity = 1)
np.fill_diagonal(sim, -1)

# Most similar pair
i_max, j_max = np.unravel_index(sim.argmax(), sim.shape)
print("MOST SIMILAR PAIR")
print(f"  Similarity: {sim[i_max, j_max]:.4f}")
print(f"  1: {headlines[i_max]}")
print(f"  2: {headlines[j_max]}")

# Least similar pair
np.fill_diagonal(sim, 2)  # exclude diagonal from min
i_min, j_min = np.unravel_index(sim.argmin(), sim.shape)
print(f"\nLEAST SIMILAR PAIR")
print(f"  Similarity: {sim[i_min, j_min]:.4f}")
print(f"  1: {headlines[i_min]}")
print(f"  2: {headlines[j_min]}")

# %% [markdown]
# Embeddings group semantically related financial news together: headlines
# about the same company, sector, or event type are close in vector space,
# while unrelated headlines are far apart.

# %% [markdown]
# ---
# ## PCA Visualization
#
# We project the 1536-dimensional embeddings down to 2D with PCA to visualize
# how headlines cluster. Dots are colored by RavenPack's `event_sentiment_score`
# (red = negative, yellow = neutral, green = positive).

# %%
pca = PCA(n_components=2)
coords = pca.fit_transform(headline_embeddings)

entity_names = rp["entity_name"].to_list()
sentiment_scores = rp["event_sentiment_score"].to_list()

fig = px.scatter(
    x=coords[:, 0],
    y=coords[:, 1],
    color=sentiment_scores,
    color_continuous_scale="RdYlGn",
    range_color=[-1, 1],
    hover_data={"Headline": headlines, "Entity": entity_names, "Sentiment": sentiment_scores},
    labels={
        "x": f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        "y": f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        "color": "Sentiment",
    },
    title="RavenPack Headlines in Embedding Space (PCA Projection)",
    opacity=0.7,
)
fig.update_traces(marker=dict(size=12))
fig.update_layout(width=900, height=650)
fig.show()

print(f"Explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

# %% [markdown]
# Headlines about similar topics or events cluster together in embedding space.
# The first two principal components capture only a fraction of the total
# variance — the full 1536-dimensional space contains much richer structure
# that downstream models can exploit.

# %% [markdown]
# ---
# ## Summary and Next Steps
#
# **What we demonstrated:**
#
# - Text embeddings map sentences to dense vectors in $\mathbb{R}^{1536}$
# - Cosine similarity captures semantic relatedness
# - Vector arithmetic supports analogies (king − man + woman ≈ queen)
# - Real financial headlines cluster by topic/entity in embedding space
#
# **Connection to CKX pipeline:**
#
# The embedding matrix $\mathbf{X}$ ($D \times P$) is the input to Eq. 2 in
# the paper — a ridge regression that predicts individual stock returns from
# news embeddings. The key insight is that embeddings capture the semantic
# content of news: similar articles produce similar vectors, and this structure
# is what downstream models exploit to predict returns.
#
# **Next step:** Embed the full headline dataset and construct the return
# prediction model.
