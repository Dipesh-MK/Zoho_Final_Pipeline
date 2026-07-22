"""
build_sheet_wrong_rate_lookup.py
=================================
Builds sheet_wrong_rate_lookup.json from a TRAIN POOL csv only (e.g.
Datasets\\train_pool.csv, produced by split_holdout.py) - never from the full
population that will later be scored. hallucination_sim.py's 9-feature path
looks up each query's top1 endpoint's sheet in this table as feature 6
(sheet_wrong_rate). If the table were built from the same queries being
scored, a query's own top1_correct outcome would be baked into the very
sheet-average it's being scored against - this script avoids that by
construction, since it only ever sees the train pool.

Schema written matches what hallucination_sim.py already expects
(dataset_dir / "sheet_wrong_rate_lookup.json"):
  {"sheet_wrong_rate": {"<sheet name>": <float>, ...}, "global_fallback": <float>}

USAGE
-----
  python build_sheet_wrong_rate_lookup.py Datasets\\train_pool.csv `
      --out Datasets\\sheet_wrong_rate_lookup.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train_csv",
                     help="train_pool.csv from split_holdout.py - needs "
                          "'sheet' and 'top1_correct' columns")
    ap.add_argument("--out", required=True,
                     help="Output path, e.g. Datasets\\sheet_wrong_rate_lookup.json")
    args = ap.parse_args()

    csv_path = Path(args.train_csv)
    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("sheet", "top1_correct"):
        if col not in df.columns:
            sys.exit(f"'{col}' column not found in {csv_path} - "
                      f"found columns: {list(df.columns)}")

    if df["top1_correct"].dtype == object:
        df["top1_correct"] = (
            df["top1_correct"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    df["top1_correct"] = df["top1_correct"].astype(bool)
    df["wrong"] = (~df["top1_correct"]).astype(float)

    global_fallback = float(df["wrong"].mean())
    sheet_rates = df.groupby("sheet")["wrong"].mean().to_dict()
    sheet_rates = {str(k): float(v) for k, v in sheet_rates.items()}

    print(f"Loaded {len(df):,} TRAIN POOL queries from {csv_path}")
    print(f"Global fallback wrong-rate: {global_fallback:.4f}\n")
    print(f"Per-sheet wrong-rate ({len(sheet_rates)} sheets), worst first:")
    for sheet, rate in sorted(sheet_rates.items(), key=lambda x: -x[1]):
        n = int((df["sheet"] == sheet).sum())
        print(f"  {sheet:30s}  n={n:5d}  wrong_rate={rate:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"sheet_wrong_rate": sheet_rates, "global_fallback": global_fallback},
            f, indent=2,
        )
    print(f"\nSaved -> {out_path}")
    print("\nBuilt from the TRAIN POOL only - safe to use with a held-out "
          "hallucination_sim.py run over the complementary holdout_eval.csv.")


if __name__ == "__main__":
    main()