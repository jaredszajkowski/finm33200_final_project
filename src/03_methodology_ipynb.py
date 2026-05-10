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
# # Methodology: From Embeddings to Predictions
#
# The previous notebook built intuition for **text embeddings** -- the dense
# vector representations that encode a news headline as a point in
# $\mathbb{R}^P$.  This notebook explains the next step: how those vectors
# are turned into **stock-return predictions** using the methodology of
# Chen, Kelly, and Xiu (2022).
#
# **Outline:**
#
# 1. The two-step framework (text representation + econometric model)
# 2. Constructing sentiment labels from stock returns
# 3. The logistic regression sentiment model
# 4. Rolling-window evaluation scheme
# 5. Prediction, accuracy, and portfolio construction
# 6. End-to-end pipeline summary

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

# %% [markdown]
# ---
# ## 1. The Two-Step Framework
#
# Chen, Kelly, and Xiu (2022) decompose the problem of using text to
# predict returns into two steps:
#
# | Step | Goal | Output |
# |------|------|--------|
# | **Step 1 -- Text Representation** | Convert each article into a numerical vector | Feature matrix $X$ of dimension $D \times P$ |
# | **Step 2 -- Econometric Model** | Map features to return predictions | Fitted model $\hat{\beta}$ |
#
# The key insight is that **Step 1 is delegated to a pre-trained language
# model**.  Whether we use BERT (768 dimensions), OpenAI's
# `text-embedding-3-small` (1,536 dimensions), or even a simple TF-IDF +
# SVD pipeline (64 dimensions), the output is the same: a dense vector
# $x_{i,t} \in \mathbb{R}^P$ for each article $i$ published at time $t$.
#
# Because the LLM has already learned rich language representations from
# massive pre-training corpora, **Step 2 can use a simple linear model** --
# no neural networks or complex architectures are needed on top of the
# embeddings.
#
# ### Where this lives in the codebase
#
# **Step 1** is implemented by three embedding scripts, each producing
# chunked parquet files of the form `(rp_story_id, dim_0, ..., dim_P)`:
#
# | Script | Model | $P$ |
# |--------|-------|-----|
# | `src/embed_tfidf.py` | TF-IDF + Truncated SVD | 64 |
# | `src/embed_bert.py` | BERT (mean-pooled) | 768 |
# | `src/embed_openai_small.py` | `text-embedding-3-small` | 1,536 |
#
# **Step 2** is implemented entirely in `src/train_rolling_model.py`, which
# trains a logistic regression classifier on the embedding features and
# evaluates it out of sample using rolling windows.

# %% [markdown]
# ---
# ## 2. Sentiment Labels from Returns
#
# Before we can train a model, we need labels.  Rather than hiring human
# annotators to read each headline, CKX use **the stock market itself** as
# the labeler: if the stock price went up around the article date, the
# article is labeled *positive*; if it went down, *negative*.
#
# ### The 3-day return window
#
# For each article published on trading day $t$, we compute the compound
# return over the window $(-1, +1)$:
#
# $$
# r_{\text{window}} = \prod_{k=-1}^{1}(1 + r_{i,t+k}) - 1
# $$
#
# where $r_{i,t+k}$ is the CRSP daily return for stock $i$ on day $t+k$.
# The window spans **three trading days**: the day before, the day of, and
# the day after the article.
#
# The binary sentiment label is then:
#
# $$
# y_{i,t} = \mathbf{1}[r_{\text{window}} > 0]
# $$
#
# Using three days rather than a single day improves the signal-to-noise
# ratio in sentiment labeling (Ke et al., 2019).  Although the label
# window includes the day *after* the article, this does not introduce
# look-ahead bias: these labels are only used to train on **past** data,
# and the rolling-window scheme (Section 4) ensures that the model never
# sees labels from the test period.
#
# ### Implementation: `src/clean_labels.py`
#
# The function `compute_return_window()` computes this for every
# permno-date pair using Polars shift operations within groups:
#
# ```python
# # src/clean_labels.py, lines 48-67
# # For each offset k in [lo, hi], shift ret by -k within each permno
# shift_exprs = []
# shift_names = []
# for k in range(lo, hi + 1):
#     col_name = f"_ret_k{k}"
#     shift_names.append(col_name)
#     shift_exprs.append(
#         pl.col("ret").shift(-k).over("permno").alias(col_name)
#     )
#
# df = df.with_columns(shift_exprs)
# df = df.drop_nulls(subset=shift_names)
#
# # Compound return: prod(1 + r_k) - 1
# compound_expr = pl.lit(1.0)
# for col_name in shift_names:
#     compound_expr = compound_expr * (pl.lit(1.0) + pl.col(col_name))
# df = df.with_columns((compound_expr - pl.lit(1.0)).alias("ret_window"))
# ```
#
# The function `build_labeled_dataset()` (line 72) merges these return
# windows with the cleaned headlines and assigns the binary label:
# `label = 1 if ret_window > 0 else 0`.

# %%
df_labeled = pl.read_parquet(DATA_DIR / "labeled_dataset.parquet")

print(f"Labeled dataset: {len(df_labeled):,} observations")
print(f"Date range: {df_labeled['article_date'].min()} to {df_labeled['article_date'].max()}")
print(f"Positive labels: {df_labeled['label'].mean():.1%}")

df_labeled.select("rp_story_id", "permno", "article_date", "headline", "ret_window", "label").head(8).to_pandas()

# %% [markdown]
# ---
# ## 3. The Sentiment Model
#
# ### Logistic regression
#
# CKX model sentiment as a **binary classification** problem.  The
# probability that article $i$ at time $t$ has positive sentiment (i.e.
# is associated with a positive return) is:
#
# $$
# \mathrm{E}(y_{i,t} \mid x_{i,t}) = \sigma(x_{i,t}'\beta)
# \tag{Eq. 1, CKX}
# $$
#
# where $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ is the logistic (sigmoid)
# function and $x_{i,t} \in \mathbb{R}^P$ is the embedding vector.
#
# ### Loss function with L2 regularization
#
# The model is fit by minimizing the **regularized cross-entropy loss**:
#
# $$
# \mathcal{L}(\beta)
# = -\frac{1}{N}\sum_{i=1}^{N}
#   \bigl[y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)\bigr]
# + \frac{1}{2C}\|\beta\|_2^2
# $$
#
# where $\hat{y}_i = \sigma(x_i'\beta)$ and $C > 0$ is the **inverse
# regularization strength**.  Smaller $C$ means stronger regularization
# (more shrinkage of $\beta$ toward zero).
#
# The L2 penalty prevents overfitting when the embedding dimension $P$ is
# large relative to the training sample.  The optimal $C$ is selected on a
# held-out validation set (see Section 4).
#
# ### Implementation: `src/train_rolling_model.py`
#
# The regularization grid and the training function are defined as follows:
#
# ```python
# # src/train_rolling_model.py, line 52
# C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
# ```
#
# ```python
# # src/train_rolling_model.py, lines 145-167
# def train_logistic_sentiment(X_train, y_train, X_val, y_val, c_grid=C_GRID):
#     """Train logistic regression with L2 penalty, tuned on validation set."""
#     scaler = StandardScaler()
#     X_train_sc = scaler.fit_transform(X_train)
#     X_val_sc = scaler.transform(X_val)
#
#     best_acc = -1.0
#     best_model = None
#     best_C = None
#
#     for C in c_grid:
#         model = LogisticRegression(
#             C=C, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42,
#         )
#         model.fit(X_train_sc, y_train)
#         val_acc = accuracy_score(y_val, model.predict(X_val_sc))
#         if val_acc > best_acc:
#             best_acc = val_acc
#             best_model = model
#             best_C = C
#
#     best_model._scaler = scaler
#     return best_model, best_C, best_acc
# ```
#
# Key details:
#
# - **Feature standardization**: `StandardScaler` is fit on the *training
#   set only*.  Validation and test features are transformed with the same
#   mean/variance -- no information leakage.
# - **Grid search**: Each candidate $C$ is evaluated by classification
#   accuracy on the validation set.  The best model is returned.
# - **Solver**: L-BFGS, a quasi-Newton optimizer well-suited for L2-penalized
#   logistic regression.

# %% [markdown]
# ### Quick illustration
#
# To build intuition, here is a toy example: we generate random 10-D
# "embeddings" with a weak signal in the first dimension, then fit the
# same logistic regression used in the pipeline.

# %%
rng = np.random.default_rng(42)

# Toy data: 1000 obs, 10 dims, weak signal in dim 0
N, P = 1000, 10
X_toy = rng.standard_normal((N, P)).astype(np.float32)
signal = 0.3 * X_toy[:, 0]
prob = 1 / (1 + np.exp(-signal))
y_toy = rng.binomial(1, prob)

# Split
X_tr, X_te = X_toy[:800], X_toy[800:]
y_tr, y_te = y_toy[:800], y_toy[800:]

# Standardize
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

# Fit with C=0.1 (moderate regularization)
model = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=1000)
model.fit(X_tr_sc, y_tr)

y_proba = model.predict_proba(X_te_sc)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)
acc = (y_pred == y_te).mean()

print(f"Toy example -- Test accuracy: {acc:.1%}")
print(f"Coefficients (10 dims): {np.round(model.coef_[0], 3)}")
print(f"Note: dim 0 has the largest coefficient, matching the planted signal")

# %% [markdown]
# ---
# ## 4. Rolling-Window Evaluation
#
# To evaluate prediction accuracy **out of sample**, CKX use annually
# updated rolling windows.  Each window consists of an 8-year in-sample
# period split into training and validation, plus a 1-year out-of-sample
# (OOS) test period:
#
# | Split | Period | Purpose |
# |-------|--------|---------|
# | **Train** | $[t-8,\; t-3)$ | 6 years -- fit $\beta$ |
# | **Validation** | $[t-2,\; t-1]$ | 2 years -- select best $C$ |
# | **Test** | year $t$ | 1 year -- evaluate OOS accuracy |
#
# The window advances by one year for each new OOS period, ensuring
# that at every point in time the model only uses **past data** for
# training and tuning.
#
# ### Feature standardization
#
# Before training, all embedding dimensions are standardized to zero mean
# and unit variance.  Crucially, the `StandardScaler` is fit on the
# **training split only** -- validation and test features are transformed
# using the training-set statistics.  This prevents any information from
# future data leaking into the model.
#
# ### Implementation: `src/train_rolling_model.py`
#
# The rolling window constants and the window generator:
#
# ```python
# # src/train_rolling_model.py, lines 54-55
# TRAIN_YEARS = 6
# VAL_YEARS = 2
# ```
#
# ```python
# # src/train_rolling_model.py, lines 63-86
# def get_rolling_windows(df_labeled):
#     years = df_labeled["article_date"].dt.year()
#     min_year = years.min()
#     max_year = years.max()
#
#     oos_start = min_year + TRAIN_YEARS + VAL_YEARS
#     oos_end = max_year
#
#     windows = []
#     for oos_year in range(oos_start, oos_end + 1):
#         windows.append({
#             "oos_year": oos_year,
#             "train": (f"{oos_year - VAL_YEARS - TRAIN_YEARS}-01-01",
#                       f"{oos_year - VAL_YEARS - 1}-12-31"),
#             "val":   (f"{oos_year - VAL_YEARS}-01-01",
#                       f"{oos_year - 1}-12-31"),
#             "test":  (f"{oos_year}-01-01",
#                       f"{oos_year}-12-31"),
#         })
#     return windows
# ```
#
# The main loop in `run_rolling_sentiment_analysis()` (line 176) iterates
# over these windows.  For each OOS year it:
#
# 1. Extracts train / val / test splits by date
# 2. Loads the corresponding embedding vectors from chunk parquets
# 3. Calls `train_logistic_sentiment()` to fit and tune the model
# 4. Calls `predict_sentiment_proba()` to generate OOS predictions
# 5. Computes accuracy and saves results to JSON
#
# Let's inspect the actual window structure from the labeled data:

# %%
from train_rolling_model import get_rolling_windows

windows = get_rolling_windows(df_labeled)

rows = []
for w in windows:
    rows.append({
        "OOS Year": w["oos_year"],
        "Train Start": w["train"][0],
        "Train End": w["train"][1],
        "Val Start": w["val"][0],
        "Val End": w["val"][1],
        "Test Start": w["test"][0],
        "Test End": w["test"][1],
    })

pd.DataFrame(rows).set_index("OOS Year")

# %% [markdown]
# Each row above represents one iteration of the rolling window.  The model
# is retrained from scratch each year using only data from before the test
# period.

# %% [markdown]
# ---
# ## 5. Prediction and Evaluation
#
# ### Generating predictions
#
# Once the model is trained, the predicted sentiment probability for a
# test-set article with embedding $x$ is:
#
# $$
# \hat{p} = \sigma(x'\hat{\beta})
# $$
#
# This is implemented in `predict_sentiment_proba()`:
#
# ```python
# # src/train_rolling_model.py, lines 170-173
# def predict_sentiment_proba(model, X):
#     """Predict class probabilities using the fitted model."""
#     X_sc = model._scaler.transform(X)
#     return model.predict_proba(X_sc)[:, 1]
# ```
#
# The `[:, 1]` selects the probability of the positive class (label = 1,
# i.e. positive return).
#
# ### Classification rule
#
# An article is classified as **positive** if $\hat{p} \geq 0.5$:
#
# ```python
# # src/train_rolling_model.py, lines 251-252
# y_proba = predict_sentiment_proba(model, X_test)
# y_pred = (y_proba >= 0.5).astype(int)
# test_acc = accuracy_score(y_test, y_pred)
# ```
#
# ### Accuracy metric
#
# Classification accuracy is defined as:
#
# $$
# \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
# $$
#
# where TP (true positive) is a predicted-positive article whose 3-day
# return was indeed positive, and so on.
#
# ### Why is accuracy only slightly above 50%?
#
# In a well-functioning market, unpredictable news dominates equity
# returns, so the predictable component is small.  An accuracy of
# 51--54% may seem modest, but it is **statistically and economically
# significant**: CKX show that a long-short portfolio sorted on these
# sentiment scores earns annualized Sharpe ratios of 3.6--4.6 for LLM
# models (Table 6 in the paper).
#
# ### From sentiment scores to portfolios
#
# The predicted probability $\hat{p}_{i,t}$ serves as a **sentiment
# score**.  Each trading day, stocks are sorted by their most recent
# sentiment score into quintiles:
#
# - **Long** the top 20% (most positive sentiment)
# - **Short** the bottom 20% (most negative sentiment)
#
# This produces a zero-net-investment long-short portfolio whose returns
# measure the economic value of the text-based signal.
#
# ### What gets saved per OOS year
#
# The results for each OOS year are stored in
# `_data/rolling_results_{model}.json`:
#
# ```python
# # src/train_rolling_model.py, lines 256-266
# results[oos_year] = {
#     "y_true": y_test.tolist(),
#     "y_pred": y_pred.tolist(),
#     "y_proba": y_proba.tolist(),
#     "accuracy": test_acc,
#     "best_C": best_C,
#     "val_accuracy": val_acc,
#     "n_train": len(X_train),
#     "n_val": len(X_val),
#     "n_test": len(X_test),
# }
# ```

# %% [markdown]
# ---
# ## 6. Putting It All Together
#
# The full pipeline from raw headline to out-of-sample prediction:
#
# $$
# \underbrace{\text{headline}}_{\text{raw text}}
# \;\xrightarrow{\text{embed\_*.py}}\;
# \underbrace{x_{i,t} \in \mathbb{R}^P}_{\text{embedding}}
# \;\xrightarrow{\text{StandardScaler}}\;
# \underbrace{\tilde{x}_{i,t}}_{\text{standardized}}
# \;\xrightarrow{\text{LogisticRegression}}\;
# \underbrace{\hat{p}_{i,t} \in [0,1]}_{\text{sentiment score}}
# \;\xrightarrow{\text{sort}}\;
# \underbrace{\text{portfolio}}_{\text{L/S quintiles}}
# $$
#
# ### Pipeline task dependencies (`dodo.py`)
#
# The `dodo.py` build system encodes this as a dependency chain:
#
# ```
# task_clean_data:labels     (src/clean_labels.py)
#     |
#     v
# task_embed:*               (src/embed_*.py)
#     |
#     v
# task_train:*               (src/train_rolling_model.py)
#     |
#     v
# task_run_notebooks:04_results_ipynb
# ```
#
# Each stage depends on the outputs of the previous one.  The next
# notebook (**04 -- Results**) loads the rolling results JSON files and
# presents the out-of-sample accuracy table and portfolio performance.

# %% [markdown]
# ---
# ## 7. Beyond Classification: Direct Return Prediction
#
# The sentiment model above classifies returns as positive or negative,
# but CKX also present a second exercise that **directly predicts the
# magnitude of returns** using the same embeddings.
#
# ### The return prediction model (Eq. 2, CKX)
#
# Instead of a logistic link, the expected return is modeled as a simple
# linear function of the embedding features:
#
# $$
# \mathrm{E}(r_{i,t+1} \mid x_{i,t}) = x_{i,t}'\,\theta
# $$
#
# where $r_{i,t+1}$ is the **realized next-day open-to-open return** and
# $\theta \in \mathbb{R}^P$ is estimated by **ridge regression** --
# ordinary least squares with an L2 penalty:
#
# $$
# \hat{\theta} = \arg\min_\theta
#   \frac{1}{N}\sum_{i=1}^{N}(r_{i,t+1} - x_{i,t}'\theta)^2
#   + \alpha\|\theta\|_2^2
# $$
#
# The ridge penalty $\alpha$ is tuned on the validation set over a grid
# $\{10^{-5}, 10^{-4}, \dots, 10^{1}, 5\!\times\!10^{1}, 10^{2}\}$.
#
# ### Key differences from the sentiment model
#
# | | Sentiment (Eq. 1) | Return prediction (Eq. 2) |
# |---|---|---|
# | **Model** | Logistic regression | Ridge regression |
# | **Target** | Binary label $y = \mathbf{1}[r_{\text{window}} > 0]$ | Realized return $r_{i,t+1} \in \mathbb{R}$ |
# | **Loss** | Cross-entropy $+ \frac{1}{2C}\|\beta\|_2^2$ | Squared error $+ \alpha\|\theta\|_2^2$ |
# | **Output** | Probability $\hat{p} \in [0, 1]$ | Predicted return $\hat{r} \in \mathbb{R}$ |
# | **Evaluation** | Classification accuracy | Prediction correlation, Sharpe ratio |
#
# ### Evaluation: correlation and portfolios
#
# Because the output is a continuous forecast, accuracy is no longer the
# right metric.  CKX evaluate with:
#
# 1. **Out-of-sample correlation** between predicted and realized returns.
#    LLM-based models achieve OOS correlations of roughly 1.6--2.3%
#    (Table 8 in the paper).
#
# 2. **Quintile long-short portfolios** sorted on $\hat{r}_{i,t+1}$.
#    Stocks are ranked each day by their predicted return; the strategy
#    goes long the top quintile and short the bottom quintile.  The
#    resulting annualized Sharpe ratios range from 3.60 (BERT) to 4.62
#    (ChatGPT/OpenAI), substantially outperforming word-based baselines.
#
# ### Why both exercises matter
#
# The sentiment model asks a simpler question -- *is this news good or
# bad?* -- and discards magnitude information.  The return prediction
# model preserves that information, distinguishing between mildly
# positive and strongly positive news.  CKX find that both exercises
# confirm the same conclusion: LLM embeddings carry economically
# significant information about future stock returns.
#
# Our replication implements the sentiment (classification) exercise.
# Extending it to the return prediction exercise would require replacing
# `LogisticRegression` with `Ridge` from scikit-learn and changing the
# target variable from the binary label to realized returns.

# %%
