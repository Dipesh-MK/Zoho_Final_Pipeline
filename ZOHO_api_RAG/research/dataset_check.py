"""
analyze_dataset.py

Analyze a query -> endpoint labeling CSV for a RAG evaluation task.
Expected columns: query, endpoint, method, sheet, subFeature, markedCorrect, addedAt

Usage:
    python analyze_dataset.py path/to/file.csv
    python analyze_dataset.py path/to/file.csv --n-true 200

What it does:
  1. Parses the CSV into columns and cleans types.
  2. Prints a dataset overview (counts, class balance, breakdown by sheet/subFeature/method).
  3. Builds a sample: first N True rows + all False rows (configurable with --n-true).
  4. Computes the "default" baseline metric: majority-class accuracy
     (i.e. what accuracy you'd get by always predicting the most common label).
  5. Runs a naive keyword-matching baseline: scores each query/endpoint pair by
     lexical overlap and predicts True/False from a threshold, then reports
     accuracy / precision / recall / F1 against your real markedCorrect labels.
  6. Saves the sampled subset and a per-row scored CSV next to the input file.
"""

import argparse
import csv
from email import parser
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ---------- 1. Load & clean ----------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {"query", "endpoint", "method", "sheet", "subFeature", "markedCorrect", "addedAt"}
    missing = expected_cols - set(df.columns)
    if missing:
        print(f"WARNING: missing expected columns: {missing}")

    # normalize markedCorrect to real booleans (handles True/False, "True"/"False", 1/0)
    if df["markedCorrect"].dtype == object:
        df["markedCorrect"] = df["markedCorrect"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    df["markedCorrect"] = df["markedCorrect"].astype(bool)

    df["addedAt"] = pd.to_datetime(df["addedAt"], errors="coerce")
    return df


# ---------- 2. Overview ----------

def print_overview(df: pd.DataFrame) -> None:
    n = len(df)
    n_true = int(df["markedCorrect"].sum())
    n_false = n - n_true

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total rows:        {n}")
    print(f"markedCorrect=True:  {n_true}  ({n_true/n:.1%})")
    print(f"markedCorrect=False: {n_false}  ({n_false/n:.1%})")

    dupes = df.duplicated(subset=["query", "endpoint", "method"]).sum()
    print(f"Duplicate (query,endpoint,method) rows: {dupes}")

    print("\n--- By sheet ---")
    print(df.groupby("sheet")["markedCorrect"].agg(["count", "sum"]).rename(columns={"sum": "true_count"}))

    print("\n--- By subFeature (top 15) ---")
    sub = df.groupby("subFeature")["markedCorrect"].agg(["count", "sum"]).rename(columns={"sum": "true_count"})
    print(sub.sort_values("count", ascending=False).head(15))

    print("\n--- By HTTP method ---")
    print(df.groupby("method")["markedCorrect"].agg(["count", "sum"]).rename(columns={"sum": "true_count"}))
    print()


# ---------- 3. Sampling ----------

def build_sample(df: pd.DataFrame, n_true: int) -> pd.DataFrame:
    true_rows = df[df["markedCorrect"] == True].sort_values("addedAt").head(n_true)
    false_rows = df[df["markedCorrect"] == False]  # keep ALL false rows, they're usually scarcer
    sample = pd.concat([true_rows, false_rows]).sort_values("addedAt").reset_index(drop=True)

    print("=" * 60)
    print(f"SAMPLE BUILT: first {n_true} True rows + all {len(false_rows)} False rows")
    print(f"Sample size: {len(sample)}  (True: {len(true_rows)}, False: {len(false_rows)})")
    print("=" * 60)
    print()
    return sample


# ---------- 4. Majority-class ("default") baseline ----------

def majority_baseline(df: pd.DataFrame) -> None:
    n = len(df)
    n_true = int(df["markedCorrect"].sum())
    majority_label = n_true >= (n - n_true)
    majority_acc = max(n_true, n - n_true) / n

    print("=" * 60)
    print("DEFAULT / MAJORITY-CLASS BASELINE")
    print("=" * 60)
    print(f"Majority label: {majority_label}")
    print(f"Accuracy from always predicting the majority label: {majority_acc:.3f}")
    print("(Any real model/heuristic needs to clearly beat this number to be useful.)")
    print()


# ---------- 5. Keyword-matching baseline ----------

STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "did", "how", "what", "which",
    "i", "to", "for", "of", "in", "on", "list", "get", "view", "full", "used",
    "existing", "specific", "and", "or", "can", "you", "me", "my", "with", "by"
}


def tokenize(text: str) -> set:
    text = str(text).lower()
    # split camelCase / snake_case / path segments / punctuation all into words
    text = re.sub(r"[/_\-]", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def keyword_score(query: str, endpoint: str, sub_feature: str) -> float:
    q_tokens = tokenize(query)
    e_tokens = tokenize(endpoint) | tokenize(sub_feature)
    if not q_tokens or not e_tokens:
        return 0.0
    overlap = q_tokens & e_tokens
    # Jaccard-ish overlap score, biased toward query coverage
    return len(overlap) / len(q_tokens)


def keyword_matching_baseline(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    scored = df.copy()
    scored["kw_score"] = scored.apply(
        lambda r: keyword_score(r["query"], r["endpoint"], r["subFeature"]), axis=1
    )
    scored["kw_predicted"] = scored["kw_score"] >= threshold

    y_true = scored["markedCorrect"]
    y_pred = scored["kw_predicted"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])

    print("=" * 60)
    print(f"KEYWORD-MATCHING BASELINE (threshold = {threshold})")
    print("=" * 60)
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}  (of predicted-correct, how many really were)")
    print(f"Recall:    {rec:.3f}  (of actually-correct, how many it caught)")
    print(f"F1:        {f1:.3f}")
    print("\nConfusion matrix (rows=actual, cols=predicted), order [True, False]:")
    print(pd.DataFrame(cm, index=["actual_True", "actual_False"], columns=["pred_True", "pred_False"]))
    print()
    print("Tip: try --threshold values between 0.1 and 0.5 to see how the")
    print("baseline trades off precision vs recall on your data.")
    print()

    return scored


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Analyze query->endpoint RAG eval CSV")
    # Change this line:
    parser.add_argument("csv_path", nargs="?", default="C:\\Users\\Dipesh\\OneDrive\\Desktop\\Zoho\\ZOHO_api_RAG\\Datasets\\site24x7_Dataset.csv", help="Path to the CSV file")
    parser.add_argument("--n-true", type=int, default=200, help="Number of True rows to sample (default 200)")
    parser.add_argument("--threshold", type=float, default=0.2, help="Keyword-match score threshold (default 0.2)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")

    df = load_data(csv_path)

    print_overview(df)
    majority_baseline(df)
    scored = keyword_matching_baseline(df, threshold=args.threshold)
    sample = build_sample(df, n_true=args.n_true)

    # Save outputs next to the input file
    out_dir = csv_path.parent
    sample_path = out_dir / f"{csv_path.stem}_sample.csv"
    scored_path = out_dir / f"{csv_path.stem}_scored.csv"
    sample.to_csv(sample_path, index=False)
    scored.to_csv(scored_path, index=False)

    print(f"Saved sample subset  -> {sample_path}")
    print(f"Saved scored dataset -> {scored_path}")


if __name__ == "__main__":
    main()