"""
Script 5: compute_metrics.py
Reads relevance_scores.csv and hallucination_flags.csv from disk.
Computes precision/recall/F1/accuracy per tool per dataset.
Prints plain table with class balance.  Saves results/final_metrics.json.
"""
import csv
import json
import os
import sys

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score
)

# ─── Load CSVs ────────────────────────────────────────────────────────────────
REL_CSV  = "results/relevance_scores.csv"
HALL_CSV = "results/hallucination_flags.csv"

for path in (REL_CSV, HALL_CSV):
    if not os.path.exists(path):
        print(f"Missing: {path} — run the previous scripts first.")
        sys.exit(1)

def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

rel_rows  = load_csv(REL_CSV)
hall_rows = load_csv(HALL_CSV)

print(f"Loaded {len(rel_rows)} relevance rows, {len(hall_rows)} hallucination rows.")

# ─── Helper ───────────────────────────────────────────────────────────────────
def compute(y_true, y_pred):
    """Returns (precision, recall, f1, accuracy) with zero_division=0."""
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    a = accuracy_score(y_true, y_pred)
    return p, r, f, a

def balance_str(y_true):
    pos = sum(y_true)
    neg = len(y_true) - pos
    return f"{pos}H/{neg}F"

# ─── 1. Relevance metrics (CrossEncoder / BGE reranker) ───────────────────────
print("\n=== RELEVANCE METRICS (BAAI/bge-reranker-v2-m3) ===")
print(f"{'dataset':<14} {'n':>5}  {'balance':>9}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Acc':>6}")
print("-" * 65)

rel_metrics = {}
datasets_rel = sorted(set(r["dataset"] for r in rel_rows))
for ds in datasets_rel + ["ALL"]:
    if ds == "ALL":
        subset = rel_rows
    else:
        subset = [r for r in rel_rows if r["dataset"] == ds]

    if not subset:
        continue

    y_true = [1 if r["ground_truth_relevant"].strip().lower() in ("true", "1") else 0 for r in subset]
    y_pred = [1 if r["predicted_relevant"].strip().lower() in ("true", "1") else 0 for r in subset]

    pos = sum(y_true); neg = len(y_true) - pos
    bal = f"{pos}Rel/{neg}Irr"

    p, r, f, a = compute(y_true, y_pred)
    print(f"{'['+ds+']':<14} {len(subset):>5}  {bal:>9}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  {a:>6.3f}")
    rel_metrics[ds] = {"dataset": ds, "tool": "BGE-reranker", "n": len(subset),
                       "balance": bal, "precision": round(p,4), "recall": round(r,4),
                       "f1": round(f,4), "accuracy": round(a,4)}

# ─── 2. Hallucination metrics per tool per dataset ────────────────────────────
print("\n=== HALLUCINATION METRICS ===")
print(f"{'tool':<14} {'dataset':<14} {'n':>5}  {'balance':>9}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Acc':>6}")
print("-" * 78)

tools   = sorted(set(r["tool"]    for r in hall_rows))
ds_list = sorted(set(r["dataset"] for r in hall_rows))

hall_metrics = []
for tool in tools:
    for ds in ds_list + ["ALL"]:
        if ds == "ALL":
            subset = [r for r in hall_rows if r["tool"] == tool]
        else:
            subset = [r for r in hall_rows if r["tool"] == tool and r["dataset"] == ds]

        if not subset:
            continue

        # Filter out non-scored rows
        scorable = [r for r in subset if r["predicted_label"] not in ("error", "skipped", "")]
        if not scorable:
            print(f"  {tool:<14} {('['+ds+']'):<14} {'--':>5}  {'--':>9}  (all skipped/error)")
            continue

        y_true = [1 if r["ground_truth_label"] == "hallucinated" else 0 for r in scorable]
        y_pred = [1 if r["predicted_label"]     == "hallucinated" else 0 for r in scorable]

        pos = sum(y_true); neg = len(y_true) - pos
        bal = balance_str(y_true)

        p, r, f, a = compute(y_true, y_pred)
        print(f"  {tool:<14} {('['+ds+']'):<14} {len(scorable):>5}  {bal:>9}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  {a:>6.3f}")
        hall_metrics.append({
            "tool": tool, "dataset": ds, "n": len(scorable),
            "balance": bal, "precision": round(p,4), "recall": round(r,4),
            "f1": round(f,4), "accuracy": round(a,4)
        })

# ─── Save JSON ────────────────────────────────────────────────────────────────
out = {
    "relevance":     list(rel_metrics.values()),
    "hallucination": hall_metrics,
}
os.makedirs("results", exist_ok=True)
with open("results/final_metrics.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("\nSaved: results/final_metrics.json")
