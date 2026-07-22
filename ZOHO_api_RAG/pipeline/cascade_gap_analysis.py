# analyze_judge_impact_v2.py
import pandas as pd
from pathlib import Path

p1 = pd.read_csv("Datasets/hallucination_sim_results.csv")
p2 = pd.read_csv("Datasets/hallucination_sim_deep_results.csv") if Path("Datasets/hallucination_sim_deep_results.csv").exists() else None

gt_wrong = ~p1["top1_correct"]
print("=== FALSE NEGATIVES (Missed Risks - LOW confidence) ===")
fns = p1[(p1["phase1_risk_label"] == "low") & gt_wrong].copy()
print(f"Total FN in LOW band: {len(fns)}")
print(fns[["query", "top1_sim", "margin", "lr_risk_probability"]].sort_values("lr_risk_probability", ascending=False).head(10))

print("\n=== FALSE POSITIVES (False Alarms - HIGH confidence) ===")
fps = p1[(p1["phase1_risk_label"] == "high") & ~gt_wrong].copy()
print(f"Total FP in HIGH band: {len(fps)}")
print(fps[["query", "top1_sim", "margin", "lr_risk_probability"]].sort_values("lr_risk_probability", ascending=True).head(10))

if p2 is not None:
    print(f"\nJudge changed { (p2['final_risk_label'] != p1.set_index('query').loc[p2['query']]['phase1_risk_label'].values).sum() } labels in the 200 sample.")