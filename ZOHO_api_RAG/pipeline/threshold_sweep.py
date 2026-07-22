"""
threshold_sweep.py
===================
Sweeps --low-risk-threshold and --high-risk-threshold over the ALREADY-SCORED
lr_risk_probability + top1_correct columns in hallucination_sim_results.csv.

No re-embedding, no re-inference, no API calls - this reuses the probabilities
your last hallucination_sim.py run already computed and saved. Runs in well
under a second, so it's safe to sweep every combination.

KEY INSIGHT (from hallucination_sim.py's own cascade logic):
  Predicted "risky" = phase1_risk_label in ("medium", "high")
  phase1_risk_label = "low"  if p < low_thresh
                       "high" if p > high_thresh
                       "medium" otherwise
  => only LOW_THRESH determines the Phase-1 confusion matrix (TP/FP/FN/TN).
     HIGH_THRESH only determines how the flagged queries split between
     "auto-high" (skipped judge) and "uncertain" (judge fires in Phase 2) -
     i.e. it controls judge/API COST, not Phase-1 accuracy.

This script therefore:
  1. Sweeps low_thresh alone -> full confusion-matrix table (precision,
     recall, F1, accuracy, FP, FN) - this is what actually matters for
     detection quality.
  2. For a chosen low_thresh, sweeps high_thresh -> shows the judge-trigger-
     rate / cost tradeoff (how many queries need the expensive judge call).

USAGE
-----
  python threshold_sweep.py Datasets\\hallucination_sim_results.csv
  python threshold_sweep.py Datasets\\hallucination_sim_results.csv --low-thresh-for-cost 0.35
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def confusion_at(df: pd.DataFrame, low_thresh: float) -> dict:
    gt_risky   = ~df["top1_correct"]
    pred_risky = df["lr_risk_probability"] >= low_thresh

    TP = int((gt_risky & pred_risky).sum())
    FN = int((gt_risky & ~pred_risky).sum())
    FP = int((~gt_risky & pred_risky).sum())
    TN = int((~gt_risky & ~pred_risky).sum())

    n    = TP + FP + FN + TN
    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    acc  = (TP + TN) / max(n, 1)
    fpr  = FP / max(FP + TN, 1)
    fnr  = FN / max(TP + FN, 1)

    return dict(low_thresh=round(low_thresh, 3), TP=TP, FP=FP, FN=FN, TN=TN,
                precision=prec, recall=rec, f1=f1, accuracy=acc, fpr=fpr, fnr=fnr)


def judge_rate_at(df: pd.DataFrame, low_thresh: float, high_thresh: float) -> dict:
    p = df["lr_risk_probability"]
    n = len(df)
    auto_low  = int((p < low_thresh).sum())
    uncertain = int(((p >= low_thresh) & (p <= high_thresh)).sum())
    auto_high = int((p > high_thresh).sum())
    return dict(low_thresh=round(low_thresh, 3), high_thresh=round(high_thresh, 3),
                auto_low=auto_low, uncertain=uncertain, auto_high=auto_high,
                judge_rate=uncertain / max(n, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_csv",
                     help="hallucination_sim_results.csv from your last hallucination_sim.py run "
                          "(needs lr_risk_probability and top1_correct columns)")
    ap.add_argument("--thresh-min", type=float, default=0.05)
    ap.add_argument("--thresh-max", type=float, default=0.95)
    ap.add_argument("--thresh-step", type=float, default=0.05)
    ap.add_argument("--min-recall", type=float, default=0.85,
                     help="Only show/consider combos with recall >= this (default 0.85 - "
                          "missing a real hallucination risk is usually costlier than a "
                          "false alarm, so we filter for recall floor first, then optimize "
                          "precision/F1 among survivors).")
    ap.add_argument("--low-thresh-for-cost", type=float, default=None,
                     help="If set, also sweep high_thresh at this fixed low_thresh to show "
                          "the judge-trigger-rate / cost tradeoff.")
    args = ap.parse_args()

    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("lr_risk_probability", "top1_correct"):
        if col not in df.columns:
            sys.exit(f"'{col}' column not found in {csv_path} - found: {list(df.columns)}")

    if df["top1_correct"].dtype == object:
        df["top1_correct"] = (
            df["top1_correct"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    df["top1_correct"] = df["top1_correct"].astype(bool)

    n_total = len(df)
    n_wrong = int((~df["top1_correct"]).sum())
    print(f"Loaded {n_total:,} scored queries from {csv_path}")
    print(f"  GT wrong (positive class): {n_wrong:,} ({n_wrong/n_total*100:.1f}%)\n")

    # ------------------------------------------------------------------
    # 1. Sweep low_thresh -> full confusion matrix table
    # ------------------------------------------------------------------
    grid = np.arange(args.thresh_min, args.thresh_max + 1e-9, args.thresh_step)
    rows = [confusion_at(df, t) for t in grid]
    sweep_df = pd.DataFrame(rows)

    print("=" * 100)
    print("  LOW-RISK-THRESHOLD SWEEP  (this is what actually drives the Phase 1 confusion matrix)")
    print("=" * 100)
    print(f"  {'thresh':>7}  {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
          f"{'prec':>6} {'recall':>6} {'F1':>6} {'acc':>6} {'FPR':>6} {'FNR':>6}")
    print("  " + "-" * 96)
    for _, r in sweep_df.iterrows():
        print(f"  {r['low_thresh']:>7.2f}  {r['TP']:>5.0f} {r['FP']:>5.0f} "
              f"{r['FN']:>5.0f} {r['TN']:>5.0f}  "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
              f"{r['accuracy']:>6.3f} {r['fpr']:>6.3f} {r['fnr']:>6.3f}")

    # ------------------------------------------------------------------
    # 2. Recommendations
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  RECOMMENDATIONS")
    print(f"{'='*100}\n")

    best_f1 = sweep_df.loc[sweep_df["f1"].idxmax()]
    print(f"  Best F1 overall            : low_thresh={best_f1['low_thresh']:.2f}  "
          f"F1={best_f1['f1']:.3f}  precision={best_f1['precision']:.3f}  "
          f"recall={best_f1['recall']:.3f}  FP={best_f1['FP']:.0f}  FN={best_f1['FN']:.0f}")

    floor_df = sweep_df[sweep_df["recall"] >= args.min_recall]
    if len(floor_df) > 0:
        best_prec_at_floor = floor_df.loc[floor_df["precision"].idxmax()]
        print(f"  Best precision @ recall>={args.min_recall:.2f} : "
              f"low_thresh={best_prec_at_floor['low_thresh']:.2f}  "
              f"precision={best_prec_at_floor['precision']:.3f}  "
              f"recall={best_prec_at_floor['recall']:.3f}  "
              f"FP={best_prec_at_floor['FP']:.0f}  FN={best_prec_at_floor['FN']:.0f}")
    else:
        print(f"  No threshold in the sweep range achieves recall >= {args.min_recall:.2f}")

    best_acc = sweep_df.loc[sweep_df["accuracy"].idxmax()]
    print(f"  Best accuracy overall      : low_thresh={best_acc['low_thresh']:.2f}  "
          f"accuracy={best_acc['accuracy']:.3f}  precision={best_acc['precision']:.3f}  "
          f"recall={best_acc['recall']:.3f}")

    print(f"\n  NOTE: --high-risk-threshold does NOT change any of the numbers above.")
    print(f"  It only controls the auto-high/uncertain split (judge trigger rate, i.e. cost),")
    print(f"  not detection accuracy, since both 'medium' and 'high' count as flagged.")

    # ------------------------------------------------------------------
    # 3. Optional: high_thresh sweep for judge cost, at a fixed low_thresh
    # ------------------------------------------------------------------
    if args.low_thresh_for_cost is not None:
        lt = args.low_thresh_for_cost
        print(f"\n{'='*100}")
        print(f"  HIGH-RISK-THRESHOLD SWEEP @ low_thresh={lt:.2f}  (judge trigger rate / cost)")
        print(f"{'='*100}")
        print(f"  {'high_thresh':>11}  {'auto_low':>9} {'uncertain':>10} {'auto_high':>10}  {'judge_rate':>10}")
        print("  " + "-" * 60)
        hi_grid = np.arange(lt + args.thresh_step, 1.0 + 1e-9, args.thresh_step)
        for ht in hi_grid:
            r = judge_rate_at(df, lt, ht)
            print(f"  {r['high_thresh']:>11.2f}  {r['auto_low']:>9d} {r['uncertain']:>10d} "
                  f"{r['auto_high']:>10d}  {r['judge_rate']:>10.1%}")
        print(f"\n  Pick the high_thresh that keeps judge_rate in a budget you're comfortable")
        print(f"  paying LLM-judge costs for (report recommends targeting 15-25%).")

    # save full sweep for reference
    out_csv = csv_path.parent / "threshold_sweep_results.csv"
    sweep_df.to_csv(out_csv, index=False)
    print(f"\nFull sweep table saved -> {out_csv}")


if __name__ == "__main__":
    main()