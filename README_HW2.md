[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/1T15YUwT)
Replicating Chen, Kelly, and Xiu (2022): Expected Returns and Large Language Models
====================================================================================

## About

This project replicates Chen, Kelly, and Xiu (2022) "Expected Returns and Large
Language Models," which uses LLM embeddings of financial news text to predict
stock returns. The original paper uses Thomson Reuters news data (RTRS and 3PTY
databases). Since we do not have access to Thomson Reuters, we use RavenPack
news headlines as an alternative text source. Stock data comes from CRSP and
macroeconomic series from FRED.

## Replication Plan

### Original Paper Methodology (Chen, Kelly, and Xiu 2022)

Chen et al. use full-text news articles from Thomson Reuters (RTRS and 3PTY
databases, Jan 1996 -- Jun 2019). Their pipeline is:

1. **Text to tokens.** Each article is tokenized using model-specific tokenizers
   (WordPiece for BERT, BPE for RoBERTa/LLaMA). Articles exceeding the model's
   token limit (512 for BERT/RoBERTa, 2048--4096 for LLaMA) are truncated.
2. **Tokens to embeddings.** The pre-trained LLM produces a contextualized
   embedding vector for each token in the article.
3. **Average across tokens.** Token-level vectors are averaged to produce a
   single article-level embedding vector, $x_{i,t}$.
4. **Merge with stock data.** Each article is tagged with a single stock and
   matched to CRSP returns.
5. **Downstream models.** The embedding vectors are used as features in two
   supervised tasks:
   - *Sentiment analysis* — logistic regression predicting the sign of the
     three-day return around the article (Eq. 1 in the paper).
   - *Return prediction* — ridge regression predicting next-period
     cross-sectional returns (Eq. 2 in the paper).
6. **Rolling-window estimation.** Models are trained on 8-year rolling windows
   (6 years training + 2 years validation). Out-of-sample predictions span
   2004--2019.

### Our Adaptation

We do not have access to Thomson Reuters full-text articles. Instead, we use
**independently scraped news headlines** as our text source, with **RavenPack**
providing only the associated metadata (entity IDs, sentiment scores, relevance,
timestamps, etc.). This separation is necessary because RavenPack's terms of
service prohibit sending their proprietary headline text to external APIs such
as OpenAI. Our scraped headlines are independently sourced text that we are free
to send to embedding APIs.

The scraped headlines are loaded via the [chartbook](https://github.com/backofficedev/chartbook)
pipeline and then merged with RavenPack metadata on story ID
(see `src/merge_scraped_headlines.py`). The result is a dataset where each row
has our own headline text paired with RavenPack's rich metadata and a CRSP
PERMNO for stock matching.

Because headlines are short (typically a single sentence), the
chunking/truncation step is unnecessary --- each headline maps directly to one
embedding vector. This is methodologically equivalent to the original pipeline:
where Chen et al. chunk an article, embed each chunk, and average the resulting
vectors, we simply embed the headline (effectively a single chunk whose average
is itself).

Concretely, our pipeline is:

1. **Pull and merge headlines.** Load independently scraped headlines from the
   chartbook pipeline and merge with RavenPack metadata to obtain company
   identifiers, relevance scores, and timestamps.
2. **Compute headline embeddings.** Pass each headline through an embedding
   model to obtain a single vector per headline (no chunking or averaging
   needed).
3. **Map to CRSP.** Link RavenPack company IDs to CRSP PERMNOs and merge with
   stock return data.
4. **Pull macro data from FRED.** Retrieve macroeconomic series used as
   controls.
5. **Train sentiment and return-prediction models.** Follow the same
   rolling-window supervised learning framework as Chen et al., using headline
   embeddings as features.
6. **Evaluate out-of-sample.** Assess prediction accuracy and portfolio
   performance (long--short quintile spreads, Sharpe ratios).

### Key Differences from the Original

| | Chen, Kelly, and Xiu (2022) | This Replication |
|---|---|---|
| **Text source** | Thomson Reuters full articles + alerts | Independently scraped headlines (RavenPack metadata only) |
| **Text length** | Full article body (median ~450 LLaMA tokens) | Headline only (~10--20 tokens) |
| **Embedding step** | Chunk → embed each chunk → average vectors | Embed headline directly |
| **Stock data** | CRSP (US) + Datastream (international) | CRSP (US only) |
| **Macro data** | — | FRED |

## Your Tasks

Complete the four functions below. Each is marked with a `TODO` comment
and a `raise NotImplementedError(...)` that you must replace with working code.

| Step | File | Function | What to implement |
|------|------|----------|-------------------|
| 1 | `src/clean_labels.py` | `compute_return_window()` | Shift daily returns within each stock and compute the compound multi-day return |
| 2 | `src/clean_labels.py` | `assign_binary_label()` | Convert continuous return to binary label (1 if positive, 0 otherwise) |
| 3 | `src/embed_bert.py` | `mean_pool_embeddings()` | Mean-pool BERT token embeddings, excluding padding tokens |
| 4 | `src/train_rolling_model.py` | `train_logistic_sentiment()` | Standardize features, grid-search over L2 penalty, return best model |
| 5 | `src/affirm_replication.py` | `I_HAVE_COMPLETED_THE_REPLICATION` | Set flag to `True` after running the full pipeline and verifying results |

### Step 1: Compound Return Window

In `src/clean_labels.py`, complete `compute_return_window()`. Given a CRSP
daily returns DataFrame with columns `permno`, `date`, `ret`, compute the
compound return over a window of days around each date. For the default
window `(-1, 1)`:

$$\mathrm{ret\\_window}_t = \prod_{k=-1}^{1} (1 + r_{t+k}) - 1$$

Use `pl.col("ret").shift(-k).over("permno")` to access returns at offset `k`.
Drop any rows where the window extends beyond the data (null values).

### Step 2: Binary Label

In `src/clean_labels.py`, complete `assign_binary_label()`. Add a column
`label` that is `1` when `ret_window > 0` and `0` otherwise.

### Step 3: Mean Pool Embeddings

In `src/embed_bert.py`, complete `mean_pool_embeddings()`. Given BERT model
output and an attention mask, compute the mean of token embeddings *excluding*
padding tokens. Return the result as a numpy array.

Hint: expand the attention mask to match the embedding dimensions, multiply
element-wise to zero out padding, sum along the token dimension, and divide
by the number of real tokens (clamped to avoid division by zero).

### Step 4: Train Logistic Sentiment Model

In `src/train_rolling_model.py`, complete `train_logistic_sentiment()`.
Standardize features using `StandardScaler`, then loop over the provided
`c_grid` values. For each `C`, fit a `LogisticRegression(C=C, penalty="l2",
solver="lbfgs", max_iter=1000, random_state=42)` and evaluate validation
accuracy. Return `(best_model, best_C, best_acc)` and attach the scaler
as `best_model._scaler = scaler`.

### Step 5: Affirm Replication

Once you have completed Steps 1--4, run the full pipeline (`doit`), generate
embeddings for at least one model, train the rolling-window sentiment model,
and review the out-of-sample predictions. Confirm that the model produces
predictions with some directional predictive power (OOS accuracy meaningfully
above 50%). Then set `I_HAVE_COMPLETED_THE_REPLICATION = True` in
`src/affirm_replication.py`.

## Running Tests

Run the homework tests locally:
```bash
pytest src/test_homework.py -v
```

## Grading

Your submission is auto-graded via GitHub Classroom. Each test is worth 1
point (5 points total). The autograder runs `pytest` against each of the
four test functions in `src/test_homework.py`.

## Quick Start

Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the project tasks:
```bash
doit
```

### Other Commands

Lint and format with [Ruff](https://docs.astral.sh/ruff/):
```bash
ruff format . && ruff check --select I --fix . && ruff check --fix .
```


## Directory Structure

- `assets/` — Hand-drawn figures and other non-generated images.
- `_output/` — Generated output (dataframes, charts, rendered notebooks). Safe to delete and regenerate with `doit`.
- `_data/` — Cached data pulled by scripts. Safe to delete and regenerate. Not tracked in Git.
- `data_manual/` — Manually-created data that cannot be regenerated. Tracked in Git.
- `settings.py` — Loads environment variables and paths. All other scripts import configuration from here.
- `.env` — Private per-user paths and credentials. Not tracked in Git.

## Naming Conventions

- **`pull_`** prefix: Functions/files that pull data from an external source (e.g., `pull_fred.py`).
- **`load_`** prefix: Functions that load cached data from the `_data/` folder.

## Acknowledgments

This case study is based on a class project originally developed by
**Andrew Moukabary** and **Reece VanDeWeghe** for FINM 32900.
The current version has been adapted for use as a teaching case study.
Credit for the original pipeline design, data cleaning logic, and
analytical framework belongs to the original authors.
