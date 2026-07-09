"""
Script 3: relevance_scoring.py
Scores (query, document) pairs using BAAI/bge-reranker-v2-m3 CrossEncoder.
Saves results/relevance_scores.csv.  Prints first 5 rows.
"""
import json
import os
import csv
import time
import sys

os.makedirs("results", exist_ok=True)

# ─── Load all datasets from disk ─────────────────────────────────────────────
DATA_FILES = [
    "data/ragtruth.jsonl",
    "data/ragbench.jsonl",
    "data/delucionqa.jsonl",
    "data/mock.jsonl",
]

records = []
for path in DATA_FILES:
    if not os.path.exists(path):
        print(f"  SKIP (not found): {path}")
        continue
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

print(f"Total records loaded: {len(records)}")
if not records:
    print("No records found. Run load_datasets.py first.")
    sys.exit(1)

# ─── Load BGE reranker (CrossEncoder) ────────────────────────────────────────
print("Loading BAAI/bge-reranker-v2-m3 …")
from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
model = CrossEncoder(MODEL_NAME, max_length=512)
print("Model loaded.")

# ─── Score all pairs ──────────────────────────────────────────────────────────
THRESHOLD = 0.5
print(f"Scoring {len(records)} (query, document) pairs  [threshold={THRESHOLD}] …")

rows = []
pairs = [(r["query"], r["document"][:1000]) for r in records]

t0 = time.time()
scores = model.predict(pairs, batch_size=16, show_progress_bar=True)
elapsed = time.time() - t0
print(f"Scoring complete in {elapsed:.1f}s  ({elapsed/len(records):.3f}s/example)")

for rec, score in zip(records, scores):
    score_f = float(score)
    rows.append({
        "dataset":              rec["dataset"],
        "id":                   rec["id"],
        "query":                rec["query"],
        "doc_snippet":          rec["document"][:200],
        "score":                round(score_f, 6),
        "predicted_relevant":   score_f >= THRESHOLD,
        "ground_truth_relevant": rec["label_relevant"],
    })

# ─── Assert not 100% one class ───────────────────────────────────────────────
pred_true  = sum(1 for r in rows if r["predicted_relevant"])
pred_false = len(rows) - pred_true
print(f"\nPrediction distribution: predicted_relevant=True:{pred_true}  False:{pred_false}")
if pred_true == 0 or pred_false == 0:
    print(f"FAIL: All predictions are one class (True={pred_true}, False={pred_false}). "
          f"Threshold {THRESHOLD} may need adjustment.")
    sys.exit(1)

# ─── Save CSV ─────────────────────────────────────────────────────────────────
OUT_CSV = "results/relevance_scores.csv"
fieldnames = ["dataset", "id", "query", "doc_snippet", "score",
              "predicted_relevant", "ground_truth_relevant"]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved: {OUT_CSV}")

# ─── Print first 5 raw rows ───────────────────────────────────────────────────
print("\n=== First 5 rows ===")
print(f"{'dataset':<12} {'id':<10} {'score':>8}  {'pred_rel':>8}  {'gt_rel':>6}  query[:60]")
print("-" * 90)
for r in rows[:5]:
    print(f"{r['dataset']:<12} {r['id']:<10} {r['score']:>8.4f}  "
          f"{'True' if r['predicted_relevant'] else 'False':>8}  "
          f"{'True' if r['ground_truth_relevant'] else 'False':>6}  "
          f"{r['query'][:60]}")
