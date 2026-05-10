# FINM 33200: Generative and Agentic AI for Finance, Final Project: Deriving Sector Sentiment Signals from Financial News Headlines

## Description

In HW2, we produced a modified replication of the Chen, Kelly, and Xiu (2022) paper entitled "Expected Returns and Large Language Models" using independently scraped news headlines instead of Thomson Reuters full articles. The [README for HW2](README_HW2.md) provides detailed context on the original paper's methodology and our replication approach.

In this final project (referred to as sector sentiment, going forward), we build on that foundation with completion of a similar set of headline embedding tasks, but with a focus on deriving *sector-level* sentiment signals rather than individual stock-level return predictions. This involves aggregating headline embeddings at the sector level and evaluating their predictive power for sector returns.

Our thesis is based on the following ideas:

1. Market sector sentiment shifts on a weeks-to-months timescale
2. News headlines are a rich source of information about market sentiment and can be effectively embedded using LLMs
3. Aggregating headline embeddings at the sector level can capture broader sentiment trends that may not be visible at the individual stock level, and these sector-level sentiment signals can have predictive power for future sector returns
4. Implementing a trading/investing strategy based on daily stock-level sentiment signals is impractical due to the magnitude of trades required to capture the signal, but sector-level signals can be actionable for medium-term to long-term portfolio allocation decisions

## Replication/Adaptation Plan

For reference, recall the following summaries of the original paper's methodology and the HW2 adaptation:

> ### Original Paper Methodology (Chen, Kelly, and Xiu 2022)
>
> Chen et al. use full-text news articles from Thomson Reuters (RTRS and 3PTY databases, Jan 1996 -- Jun 2019). Their pipeline is:
>
>1. **Text to tokens.** Each article is tokenized using model-specific tokenizers (WordPiece for BERT, BPE for RoBERTa/LLaMA). Articles exceeding the model's token limit (512 for BERT/RoBERTa, 2048--4096 for LLaMA) are truncated.
>2. **Tokens to embeddings.** The pre-trained LLM produces a contextualized embedding vector for each token in the article.
>3. **Average across tokens.** Token-level vectors are averaged to produce a single article-level embedding vector, $x_{i,t}$.
>4. **Merge with stock data.** Each article is tagged with a single stock and matched to CRSP returns.
>5. **Downstream models.** The embedding vectors are used as features in two supervised tasks:
>
>- *Sentiment analysis* — logistic regression predicting the sign of the three-day return around the article (Eq. 1 in the paper).
>- *Return prediction* — ridge regression predicting next-period cross-sectional returns (Eq. 2 in the paper).
>
>6. **Rolling-window estimation.** Models are trained on 8-year rolling windows (6 years training + 2 years validation). Out-of-sample predictions span 2004--2019.

> ### HW2 Adaptation (Refer to README_HW2.md for details)
>
> We do not have access to Thomson Reuters full-text articles. Instead, we use **independently scraped news headlines** as our text source, with **RavenPack** providing only the associated metadata (entity IDs, sentiment scores, relevance, timestamps, etc.). This separation is necessary because RavenPack's terms of service prohibit sending their proprietary headline text to external APIs such as OpenAI. Our scraped headlines are independently sourced text that we are free to send to embedding APIs.
> 
> The scraped headlines are loaded via the [chartbook](https://github.com/backofficedev/chartbook) pipeline and then merged with RavenPack metadata on story ID (see `src/merge_scraped_headlines.py`). The result is a dataset where each row has our own headline text paired with RavenPack's rich metadata and a CRSP PERMNO for stock matching.
>
> Because headlines are short (typically a single sentence), the chunking/truncation step is unnecessary --- each headline maps directly to one embedding vector. This is methodologically equivalent to the original pipeline: where Chen et al. chunk an article, embed each chunk, and average the resulting vectors, we simply embed the headline (effectively a single chunk whose average is itself).
>
> Concretely, our pipeline is:
>
>1. **Pull and merge headlines.** Load independently scraped headlines from the chartbook pipeline and merge with RavenPack metadata to obtain company identifiers, relevance scores, and timestamps.
>2. **Compute headline embeddings.** Pass each headline through an embedding model to obtain a single vector per headline (no chunking or averaging needed).
>3. **Map to CRSP.** Link RavenPack company IDs to CRSP PERMNOs and merge with stock return data.
>4. **Pull macro data from FRED.** Retrieve macroeconomic series used as controls.
>5. **Train sentiment and return-prediction models.** Follow the same rolling-window supervised learning framework as Chen et al., using headline embeddings as features.
>6. **Evaluate out-of-sample.** Assess prediction accuracy and portfolio performance (long--short quintile spreads, Sharpe ratios).

### Sector Sentiment

As with HW2, we do not have access to Thomson Reuters full-text articles. However, instead of using **independently scraped news headlines** mreged with **RavenPack** metadata (entity IDs, sentiment scores, relevance, timestamps, etc.), as was required in HW2 due to the RavenPack licensing, our approach with sector sentiment is to use the **RavenPack** data in its entirety. We will run local models for tokenization and embedding, and are therefore not sending any of the headline text to OpenAI or any other external API.

Our sector sentiment pipeline is as follows:

1. **Pull Data:** FRED, CRSP (MSF, MSIX, DSI), RavenPack, S&P 500 constituents and sector classifications.
2. **Clean Data:** Raw, single_entity, has_return, length_filter, deduplicated cleaning funnel.
3. **Merge Data:** Return over 3-day window, labels, date, market cap, sector, permno.
4. **Compute headline embeddings:** Pass each headline through BERT and Gemma embedding models to obtain a single vector per headline.
5. **Train sentiment and return-prediction models.** Follow the same rolling-window supervised learning framework as Chen et al., using headline embeddings as features.
6. **Evaluate out-of-sample.** Assess prediction accuracy and portfolio performance (long--short quintile spreads, Sharpe ratios).
7. **Aggregate Returns:** Compute daily synthetic sector-level returns by aggregating stock-level returns using market-cap weights.
8. **Compare Returns:** Assess the relationship between daily synthetic sector-level returns and realized sector ETF returns, including correlation analysis and portfolio backtesting based on sector-level sentiment scores.

### Similarities And Differences Between Chen, Kelly, and Xiu, HW2 Replication, and Sector Sentiment

| | Chen, Kelly, and Xiu (2022) | HW2 Replication | Sector Sentiment |
|---|---|---|---|
| **Text source** | Thomson Reuters full articles + alerts | Independently scraped headlines (RavenPack metadata only) | RavenPack headlines + metadata |
| **Text length** | Full article body (median ~450 LLaMA tokens) | Headline only (~10--20 tokens) | Headline only (~10--20 tokens) |
| **Embedding step** | Chunk → embed each chunk → average vectors | Embed headline directly | Embed headline directly |
| **Stock data** | CRSP (US) + Datastream (international) | CRSP (US only) | CRSP (US only) |
| **Macro data** | — | FRED | FRED |
| **Prediction target** | Stock-level returns | Stock-level returns | Stock-level returns aggregated to sector-level |
| **Prediction training window** | 3-day return around article | 3-day return around headline | 3-day return around headline |
| **Aggregation level** | None (individual articles) | None (individual headlines) | Sector-level aggregation of headline embeddings |
| **Embedding Models (Token Limit)** | (OpenAI) OpenAI-L / `text-embedding-3-large` (3,072), (Google) BERT (512), (Meta) RoBERTa (512), (Meta) LLaMA & LLaMA2 (2048--4096) | TF-IDF + SVD / sklearn (64), (Google) BERT / `bert-base-uncased` (768), (OpenAI) OpenAI-S / `text-embedding-3-small` (1,536) | (Google) BERT / `bert-base-uncased` (768), (Google) Gemma 3 / `embeddinggemma-300m` (768) |

## Quick Start

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt # or uv pip install -r requirements.txt
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

Alternatively, simple formatting with [Black](https://black.readthedocs.io/en/stable/):

```bash
black path/to/file_or_directory.py
```

# Need to review everything below here

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
