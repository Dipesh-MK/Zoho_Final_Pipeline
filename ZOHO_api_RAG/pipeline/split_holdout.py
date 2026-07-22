"""
split_holdout.py
=================
Splits hallucination_sim_results.csv into a TRAIN POOL and a HELD-OUT EVAL
pool, stratified by top1_correct, so that the KNN index and sheet-wrong-rate
lookup used at inference time can be built exclusively from the train pool -
eliminating the self-lookup leak where a query's own label feeds back into
its own knn_neighbor_wrong_rate / sheet_wrong_rate feature at scoring time.

Run this ONCE. Then:

  1. Build the KNN index from the TRAIN POOL only (rebuild_knn_index.py needs
     no changes - just point it at train_pool.csv instead of the full results
     file, since it already only needs 'query' and 'top1_correct'):

       python rebuild_knn_index.py Datasets\train_pool.csv `
           --cache-dir Datasets\.rag_cache `
           --embed-model azure:primary/s247-textembedding-3l `
           --base-url http://20.235.183.15:443/openai/v1 `
           --api-key YOUR_KEY `
           --knn-k 15 `
           --out Datasets\knn_neighbor_index.joblib

  2. Build the sheet-wrong-rate lookup from the TRAIN POOL only:

       python build_sheet_wrong_rate_lookup.py Datasets\train_pool.csv `
           --out Datasets\sheet_wrong_rate_lookup.json

  3. Run hallucination_sim.py Phase 1 restricted to the HELD-OUT pool only
     (requires the --eval-queries-csv patch - see the patch notes you were
     given alongside this script):

       python hallucination_sim.py ... --eval-queries-csv Datasets\holdout_eval.csv

USAGE
-----
  python split_holdout.py Datasets\hallucination_sim_results.csv `
      --holdout-frac 0.2 --seed 42 --out-dir Datasets
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_csv",
                     help="hallucination_sim_results.csv - needs 'query' and "
                          "'top1_correct' columns (plus 'sheet' if you'll also "
                          "run build_sheet_wrong_rate_lookup.py)")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                     help="Fraction of unique queries reserved for held-out "
                          "eval (default 0.2, i.e. an 80/20 split).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("query", "top1_correct"):
        if col not in df.columns:
            sys.exit(f"'{col}' column not found in {csv_path} - "
                      f"found columns: {list(df.columns)}")

    df = df.drop_duplicates(subset="query").reset_index(drop=True)
    n = len(df)
    print(f"Loaded {n:,} unique queries from {csv_path}")

    if df["top1_correct"].dtype == object:
        df["top1_correct"] = (
            df["top1_correct"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    df["top1_correct"] = df["top1_correct"].astype(bool)

    strat = df["top1_correct"]
    train_df, holdout_df = train_test_split(
        df, test_size=args.holdout_frac, random_state=args.seed, stratify=strat
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path   = out_dir / "train_pool.csv"
    holdout_path = out_dir / "holdout_eval.csv"
    train_df.to_csv(train_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)

    def wrong_rate(d):
        return float((~d["top1_correct"]).mean())

    print(f"\nTrain pool   : {len(train_df):,} queries  "
          f"(wrong-rate={wrong_rate(train_df):.3f})  -> {train_path}")
    print(f"Holdout eval : {len(holdout_df):,} queries  "
          f"(wrong-rate={wrong_rate(holdout_df):.3f})  -> {holdout_path}")
    print("\nStratified split by top1_correct - the two wrong-rates above "
          "should be close to each other (sanity check the split worked).")

    print("\nNEXT STEPS:")
    print(f"  1. python rebuild_knn_index.py {train_path} --cache-dir Datasets\\.rag_cache "
          f"--api-key YOUR_KEY --base-url ... --out Datasets\\knn_neighbor_index.joblib")
    print(f"  2. python build_sheet_wrong_rate_lookup.py {train_path} "
          f"--out Datasets\\sheet_wrong_rate_lookup.json")
    print(f"  3. python hallucination_sim.py ... --eval-queries-csv {holdout_path}")
    print("\nZero new embedding API calls needed for steps 1-3 if you've "
          "already run the pipeline once - everything hits the disk cache.")


if __name__ == "__main__":
    main()