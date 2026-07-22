# Site24x7 RAG Hallucination Detection & Risk Cascade Pipeline

This repository implements a production-grade, cost-effective **Hallucination Detection & Gated Cascade Pipeline** for a Retrieval-Augmented Generation (RAG) system mapping natural language queries to Site24x7 API endpoints.

---

## ⚙️ How It Works: The Gated Cascade Architecture

Instead of routing every query through an expensive LLM judge, the pipeline cascades evaluations in three tiers. Two signals are always computed with zero LLM calls; the third (the judge) only fires when either of the first two flags uncertainty:

```mermaid
graph TD
    A[User Query] --> B[RAG Retrieval]
    B --> C[Signal 1: 9-Feature Retrieval Risk Score - lr_prob]
    B --> R[LLM Generates Response]
    R --> S[Signal 2: Embedding Groundedness Check - frac_unsupported_embedding]
    C --> D{Cascade Gate}
    S --> D
    D -- "lr_prob < Low Threshold AND groundedness OK" --> E[Label: Low Risk / Judge Skipped]
    D -- "lr_prob > High Threshold" --> F[Label: High Risk / Judge Skipped]
    D -- "Uncertain band, OR groundedness_bad force-fires the judge even when lr_prob is low" --> G[Signal 3: Gated LLM Judge Fires]
    G --> H[Sentence-Level Verification & Context-Relevance Check]
    H --> I[Final Risk Label - judge verdict always wins when it fired]
```

### Signal 1: Retrieval Risk Score (100% Offline, Zero LLM Calls)
- Every query is evaluated using **9 engineered features** (see [Model Details](#-model-details--feature-engineering) below for the full table and how the classifier itself was selected).
- The classifier predicts `lr_prob`, the probability that **retrieval itself** was wrong. It has no visibility into what the LLM actually generated — that's Signal 2's job.

### Signal 2: Embedding-Based Groundedness (100% Offline, Zero LLM Calls)
- Catches the failure mode Signal 1 is structurally blind to: retrieval found the right document, but the LLM's response still contains a fabricated detail not present in that document.
- The generated response is split into sentences (`split_sentences`), each sentence is embedded, and its **max cosine similarity to any retrieved context chunk** is computed (`embedding_groundedness`).
- A sentence below `--groundedness-threshold` (default `0.5`) is flagged as possibly unsupported; `frac_unsupported_embedding` is the fraction of the response's sentences that fail this bar.
- If `frac_unsupported_embedding` exceeds `--groundedness-gate-threshold` (default `0.10`), the query is force-routed to the judge **regardless of what Signal 1 said** — this is what lets a low-retrieval-risk-but-hallucinated response still get checked.

### Cascade Gate
- `lr_prob < low_thresh` **and** groundedness OK $\rightarrow$ **LOW RISK**, judge skipped.
- `lr_prob > high_thresh` $\rightarrow$ **HIGH RISK**, judge skipped (a bad retrieval is a bad retrieval regardless of groundedness).
- Otherwise — genuinely uncertain `lr_prob`, **or** force-fired by the groundedness gate despite a low `lr_prob` — the judge fires. When the judge fires for either reason, its verdict determines the final label; `lr_prob` is never allowed to override a verdict the judge actually returned.

### Signal 3: Deep Verification (Gated LLM Judge)
- One combined chat call returns `context_relevant` (true/false/partial) plus per-sentence `sentence_verdicts` (supported/unsupported/partial), checked only against the retrieved context.
- `high` if `context_relevant == "false"` or `judge_frac_unsupported >= 0.5`; `medium` if `partial` or any unsupported sentence; `low` otherwise.
- Refines the final risk label (`low`, `medium`, or `high`) for every query the cascade gate routed here.

---

## 🧬 Model Details & Feature Engineering

### The 9 Features

The classifier behind Signal 1 is trained on 9 features — the original 5 plus 4 added later to close blind spots the first 5 couldn't see. Stats below are computed directly from `feature_engineering_v2_dataset.csv` (5,695 queries, 21.4% with wrong top1 retrieval).

| # | Feature | What it measures | Mean | Min | Max |
|---|---|---|---|---|---|
| 1 | `top1_sim` | Cosine similarity of the best-retrieved chunk to the query. | 0.634 | 0.166 | 0.836 |
| 2 | `margin` | `top1_sim - top2_sim` — narrow margin means the retriever is genuinely torn between two candidates. | 0.061 | 0.000 | 0.388 |
| 3 | `avg_pairwise_topk_sim` | Mean pairwise similarity *among* the top-k chunks themselves (cluster tightness, not query relevance). | 0.739 | 0.309 | 0.998 |
| 4 | `query_token_count` | Token count of the query — short fragments are structurally more ambiguous. | 8.83 | 1 | 21 |
| 5 | `n_candidates_within_margin` | Count of top-k candidates within 0.02 similarity of top1 (near-ties with the winner). | 0.43 | 0 | 4 |
| 6 | `sheet_wrong_rate` | Historical wrong-rate of the retrieved API's category sheet. **Leakage-free**: built via out-of-fold CV so a query's own outcome never feeds its own feature value. | 0.215 | 0.056 | 0.571 |
| 7 | `query_endpoint_token_overlap` | Token overlap ratio between the query and the endpoint *path* specifically (not the doc description). | 0.257 | 0.000 | 1.000 |
| 8 | `topk_similarity_entropy` | Normalized Shannon entropy of the top-10 similarity distribution — diffuse vs. one clear winner. **Caveat:** near-constant in this dataset (mean 0.9994, std 0.0006) and lowest-but-one in feature importance below — it's likely contributing little signal here. | 0.9994 | 0.9955 | 1.000 |
| 9 | `knn_neighbor_wrong_rate` | Fraction of the query's k=15 nearest-neighbor queries that historically had wrong retrieval. **Leakage-free**: same out-of-fold design as feature 6 — a query is never its own neighbor. | 0.248 | 0.000 | 1.000 |

### Model Selection

Three candidate classifiers are trained on the same 9-feature matrix and compared by 5-fold stratified cross-validation (out-of-fold predictions, so the comparison is apples-to-apples and not inflated by in-sample fit):

```mermaid
graph TD
    X[9-Feature Matrix] --> CV[5-Fold Stratified CV — Out-of-Fold Predictions]
    CV --> LR["LogisticRegression<br/>class_weight=balanced"]
    CV --> RF["RandomForestClassifier<br/>n_estimators=300, max_depth=6"]
    CV --> GB["GradientBoostingClassifier<br/>n_estimators=200, max_depth=3, lr=0.05"]
    LR --> CMP{Compare ROC-AUC / PR-AUC}
    RF --> CMP
    GB --> CMP
    CMP --> BEST[Best-Scoring Model Selected]
    BEST --> SAVE[Saved via --save-model to .joblib]
```

Reproduced directly against `feature_engineering_v2_dataset.csv` using the same CV split (seed 42) and model configs as `feature_engineering_v2.py`:

| Model | ROC-AUC (9-feat) | PR-AUC (9-feat) | Δ vs. 5-feat baseline |
|---|---|---|---|
| **RandomForest** | **0.9231** | **0.7743** | +0.0293 |
| GradientBoosting | 0.9218 | 0.7748 | +0.0252 |
| LogisticRegression | 0.9156 | 0.7507 | +0.0309 |

All three clear the old 5-feature ceiling (~0.88–0.90) by a meaningful margin, and the three scores are close enough (0.9156–0.9231) that the choice isn't dominated by one model — RandomForest edges out on ROC-AUC, LogisticRegression has the largest *delta* from its own 5-feature baseline.

**Feature importance for the top-scoring model (RandomForest, full-data fit):**

| Feature | Importance |
|---|---|
| `knn_neighbor_wrong_rate` | 0.2753 |
| `margin` | 0.2713 |
| `sheet_wrong_rate` | 0.1401 |
| `n_candidates_within_margin` | 0.1178 |
| `top1_sim` | 0.0894 |
| `avg_pairwise_topk_sim` | 0.0422 |
| `topk_similarity_entropy` | 0.0336 |
| `query_endpoint_token_overlap` | 0.0219 |
| `query_token_count` | 0.0085 |

Two of the three new leakage-free features (`knn_neighbor_wrong_rate`, `sheet_wrong_rate`) rank in the top 3 — they're doing real work, not just adding noise. `topk_similarity_entropy` ranks near the bottom, consistent with the near-zero variance flagged in the feature table above.

**⚠️ Open discrepancy, worth resolving before this is final:** the analysis above shows RandomForest scoring best, which uses `.feature_importances_`. But `check_model.py` — the utility script for inspecting the deployed model — calls `model.coef_`, an attribute that **only exists on a linear model** like LogisticRegression; it will throw an `AttributeError` on a RandomForest or GradientBoosting model. That means either (a) the actually-deployed `hallucination_risk_model_9feat_v2.joblib` is a LogisticRegression, trained on a different data snapshot or `--save-model` run than the one reproduced here, or (b) `check_model.py` is out of date and needs a `hasattr` check for both attribute types. Run `check_model.py` against your real deployed `.joblib` to see which — worth confirming before you present the model comparison numbers above as *the* production model's story.

---

The pipeline was validated against a test set containing **100 honest (supported)** and **100 hallucinated (unsupported)** responses across 3 random seeds. The cascade pipeline consistently outperforms direct LLM verification while reducing LLM calls by roughly **50%**.

### Seed 0
```
========================================================================
  RESULTS (Seed 0)
========================================================================
  100 honest / 100 hallucinated responses

                        Precision     Recall         F1    TP    FN    FP    TN
  DIRECT (2+3)              0.942      0.810      0.871    81    19     5    95
  CASCADE (prod)            0.968      0.900      0.933    90    10     3    97

  Recall gap (direct - cascade) : -0.090
  Cascade judge fire rate on this set : 104/200 (52.0%)
```

### Seed 1
```
========================================================================
  RESULTS (Seed 1)
========================================================================
  100 honest / 100 hallucinated responses

                        Precision     Recall         F1    TP    FN    FP    TN
  DIRECT (2+3)              0.941      0.800      0.865    80    20     5    95
  CASCADE (prod)            0.978      0.910      0.943    91     9     2    98

  Recall gap (direct - cascade) : -0.110
  Cascade judge fire rate on this set : 106/200 (53.0%)
```

### Seed 2
```
========================================================================
  RESULTS (Seed 2)
========================================================================
  100 honest / 100 hallucinated responses

                        Precision     Recall         F1    TP    FN    FP    TN
  DIRECT (2+3)              0.942      0.810      0.871    81    19     5    95
  CASCADE (prod)            0.968      0.900      0.933    90    10     3    97

  Recall gap (direct - cascade) : -0.090
  Cascade judge fire rate on this set : 104/200 (52.0%)
```

---

## 📂 Repository File Index

The following table documents all tracked repository files and their classification:

| File Path | Classification | Description |
| :--- | :--- | :--- |
| **`pipeline/`** | | |
| `hallucination_detect.py` | Production Pipeline | Core classification and gated cascade inference pathway. |
| `hallucination_sim.py` | Production Pipeline | Simulation harness to execute Phase 1 and Phase 2 deep evals. |
| `feature_engineering_v2.py` | Production Pipeline | Signal calculation script (leakage-free cross validation). |
| `rag_eval.py` | Base Retrieval | Original text embedding and baseline RAG accuracy checking. |
| `compare_doc_strategies.py` | Base Retrieval | Strategies evaluation script for text representation format. |
| `build_sheet_wrong_rate_lookup.py` | Setup Utility | Utility script for computing the initial sheet error matrix. |
| `split_holdout.py` | Setup Utility | Splits baseline dataset into stratified train/holdout sets. |
| `synthesize_reports_descriptions.py` | Setup Utility | Pre-generates reports segment description overrides. |
| `threshold_sweep.py` | Tuning Utility | Grid sweeps threshold boundaries over predicted scores. |
| `build_hallucination_testset.py` | Testing Suite | Assembles balanced test sets (100 safe / 100 hallucination). |
| `cascade_gap_analysis.py` | Tuning Utility | Analysis script evaluating cascade boundaries. |
| **`research/`** | | |
| `cross_encoder_rerank_eval.py` | Experimental | Evaluates cross-encoder reranking strategies. |
| `hybrid_rerank_eval.py` | Experimental | Evaluates BM25 + embedding hybrid lookup scores. |
| `hyde_eval.py` | Experimental | Evaluates HyDE (Hypothetical Document Embeddings). |
| `boilerplate_diagnostic.py` | Experimental | Diagnoses boilerplate text noise issues in corpus documents. |
| `boilerplate_rewrite_eval.py` | Experimental | Evaluates rewritten boilerplate context formats. |
| `synthesize_boilerplate_descriptions.py` | Experimental | Pre-generates synthetic descriptions to mask boilerplate noise. |
| `eval_boilerplate_fix.py` | Experimental | Verifies accuracy gains from synthetic descriptions. |
| `diagnose_retrieval.py` | Experimental | Diagnoses specific retrieval path failures. |
| `dataset_check.py` | Experimental | Scans source records for duplication and format errors. |
| **`models/`** | | |
| `hallucination_risk_model_9feat_v2.joblib` | Model Parameter | Current production model (numerical classification weights). |
| `check_model.py` | Utility | Verification check script ensuring joblib models load cleanly. |
| `fix_model_save.py` | Utility | Re-pickling utility script. |
| `sheet_wrong_rate_lookup.json` | Lookup Matrix | Production category error rate map used during live features. |
| `knn_neighbor_index.joblib` | Model Parameter | High-importance NearestNeighbors index fitted under active environment. |
| **`results/`** | | |
| `results_seed0.csv` | Aggregate Results | Results metadata for seed 0 test set. |
| `results_seed1.csv` | Aggregate Results | Results metadata for seed 1 test set. |
| `results_seed2.csv` | Aggregate Results | Results metadata for seed 2 test set. |
| `feature_engineering_v2_model_comparison.csv` | Aggregate Results | Classifier score comparisons table. |
| `cross_encoder_rerank_eval_summary.csv` | Aggregate Results | Reranker performance summaries. |
| `cross_encoder_rerank_eval_per_seed.csv` | Aggregate Results | Reranker seed summaries. |
| `hybrid_rerank_eval_summary.csv` | Aggregate Results | Hybrid retrieval metrics. |
| `hybrid_rerank_eval_per_seed.csv` | Aggregate Results | Hybrid retrieval seed metrics. |
| `eval_boilerplate_fix_summary.csv` | Aggregate Results | Boilerplate evaluation summaries. |
| `eval_boilerplate_fix_per_seed.csv` | Aggregate Results | Boilerplate evaluation seed metrics. |
| `threshold_sweep_results.csv` | Aggregate Results | Sweep precision/recall summaries. |

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dipesh-MK/Zoho_Final_Pipeline.git
   cd Zoho_Final_Pipeline
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment**: Create a `.env` file in the root directory:
   ```env
   PROXY_BASE_URL=https://<your-openai-proxy-endpoint>
   PROXY_API_KEY=your_secret_api_key
   ```

---

## 🚀 How to Run the Pipeline

To evaluate Phase 1 classifier predictions and Phase 2 deep LLM validation over the full dataset:
```bash
python pipeline/hallucination_sim.py \
    Datasets/site24x7_Dataset.csv \
    Datasets/ADMIN_API/site24x7_Admin_API.xlsx \
    --extra-descriptions Datasets/reports_synthetic_descriptions.csv \
    --model-path models/hallucination_risk_model_9feat_v2.joblib \
    --low-risk-threshold 0.40 \
    --high-risk-threshold 0.82 \
    --groundedness-threshold 0.5 \
    --groundedness-gate-threshold 0.10 \
    --base-url YOUR_BASE_URL \
    --api-key YOUR_API_KEY \
    --n-deep 200 \
    --workers 4 \
    --save-log
```
