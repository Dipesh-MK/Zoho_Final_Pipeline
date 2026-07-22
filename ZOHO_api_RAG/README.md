# Site24x7 RAG Hallucination Detection & Risk Cascade Pipeline

This repository implements a production-grade, cost-effective **Hallucination Detection & Gated Cascade Pipeline** for a Retrieval-Augmented Generation (RAG) system mapping natural language queries to Site24x7 API endpoints.

---

## ⚙️ How It Works: The Gated Cascade Architecture

Instead of routing every query through an expensive LLM judge, the pipeline cascades evaluations in two tiers:

```mermaid
graph TD
    A[User Query] --> B[RAG Retrieval & Phase 1 Feature Extraction]
    B --> C[Compute Risk Score via Classifier Model]
    C --> D{Risk Score Thresholds}
    D -- "Score < Low Threshold (Auto-Low)" --> E[Label: Low Risk / Skip LLM Judge]
    D -- "Score > High Threshold (Auto-High)" --> F[Label: High Risk / Skip LLM Judge]
    D -- "Low <= Score <= High" --> G[Gated Phase 2: Fire LLM Judge]
    G --> H[LLM Sentence Verification & Context Check]
    H --> I[Final Refined Risk Label]
```

### 1. Phase 1: Fast Classifier Scoring (100% Offline, Zero LLM Calls)
- Every query is evaluated using **9 engineered features**:
  - `top1_sim`: Cosine similarity of the best-retrieved chunk.
  - `margin`: Similarity gap between top-1 and top-2 candidates (identifies retrieval ambiguity).
  - `avg_pairwise_topk_sim`: Cluster tightness among the top-k chunks.
  - `query_token_count`: Distinguishes short keyword fragments from long questions.
  - `n_candidates_within_margin`: Count of near-tie chunks.
  - `sheet_wrong_rate`: Historical failure rate of the retrieved API's category sheet.
  - `query_endpoint_token_overlap`: Token Jaccard-style overlap of query keywords vs. endpoint path structure.
  - `topk_similarity_entropy`: Softmax-normalized Shannon entropy of top-10 retrieval distribution (measures tail diffusion).
  - `knn_neighbor_wrong_rate`: Historical failure rate of the query's nearest embedding neighbors.
- The 9-feature **GradientBoostingClassifier** predicts a risk probability.
- **Cascade Gate**:
  - If probability $< 0.45$ $\rightarrow$ Automatically flagged **LOW RISK** (LLM judge skipped).
  - If probability $> 0.90$ $\rightarrow$ Automatically flagged **HIGH RISK** (LLM judge skipped).
  - Otherwise (Uncertain Band) $\rightarrow$ Sent to **Phase 2**.

### 2. Phase 2: Deep Verification (Gated LLM Judge)
- Generates a response and triggers the LLM judge.
- The judge assesses context relevance and performs sentence-level verification.
- Refines the final risk label (`low`, `medium`, or `high`).

---

## 📈 Multi-Seed Performance Evaluation Results

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
  DIRECT (2+3)              0.934      0.850      0.890    85    15     6    94
  CASCADE (prod)            0.967      0.870      0.916    87    13     3    97

  Recall gap (direct - cascade) : -0.020
  Cascade judge fire rate on this set : 105/200 (52.5%)
```

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

### 1. Run the Simulation Pipeline
To evaluate Phase 1 classifier predictions and Phase 2 deep LLM validation over the full dataset:
```bash
python pipeline/hallucination_sim.py \
    Datasets/site24x7_Dataset.csv \
    Datasets/ADMIN_API/site24x7_Admin_API.xlsx \
    --extra-descriptions Datasets/reports_synthetic_descriptions.csv \
    --model-path models/hallucination_risk_model_9feat_v2.joblib \
    --low-risk-threshold 0.45 \
    --high-risk-threshold 0.90 \
    --base-url YOUR_BASE_URL \
    --api-key YOUR_API_KEY \
    --n-deep 200 \
    --workers 4 \
    --save-log
```
