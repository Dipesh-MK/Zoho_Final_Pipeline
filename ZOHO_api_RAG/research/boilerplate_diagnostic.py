"""
boilerplate_diagnostic.py

Checks whether templated/boilerplate description text (e.g. "Fetches data
for: X (endpoint: Y)" with only X/Y substituted per key) is a real, fixable
contributor to your recall@1 misses - as opposed to another ranking-layer
trick, this is about the QUALITY of the text dense embeddings are actually
comparing against.

Two independent parts:

  PART A - static boilerplate detection (no API calls, no embeddings needed).
  For each (endpoint, method) description, strip out the words that are
  clearly just substituted per-key (endpoint path words, method, sub-feature
  name), leaving a "skeleton". If the same skeleton is shared by several
  different keys IN THE SAME SHEET, that's templated boilerplate - a real
  per-endpoint description wouldn't reduce to an identical generic sentence
  once you remove the specific names. Reports skeleton reuse counts per
  sheet and prints the worst offenders so you can see exactly what to fix.

  PART B - retrieval cross-reference (needs embeddings, --mock or real).
  Runs the same doc-covered-only eval methodology as compare_doc_strategies.py
  (doc_description strategy, no fallback), but additionally tags each eval
  query's true key as boilerplate/non-boilerplate using Part A's detection,
  and reports recall@1 split by that tag. If boilerplate-tagged keys have
  meaningfully worse recall@1 than non-boilerplate keys, that's your
  confirmation this is worth fixing before trying anything else.

Usage:
    # Part A only - fast, no API/embeddings needed at all:
    python boilerplate_diagnostic.py site24x7_Dataset.csv site24x7_Admin_API.xlsx --skip-retrieval-check

    # Both parts, offline pipeline test:
    python boilerplate_diagnostic.py site24x7_Dataset.csv site24x7_Admin_API.xlsx --mock

    # Both parts, real:
    python boilerplate_diagnostic.py site24x7_Dataset.csv site24x7_Admin_API.xlsx \
        --base-url http://20.235.183.15:443/openai/v1 --api-key YOUR_KEY \
        --extra-descriptions reports_synthetic_descriptions.csv --seeds 1 2 3 4 5 6 7 8
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from rag_eval import build_chunk_text, compute_recall, embed_texts, get_embeddings_mock
from compare_doc_strategies import (
    build_text_doc_description,
    load_doc_descriptions,
    load_extra_descriptions,
    make_variant_corpus,
)

MIN_SHARED_SKELETON_COUNT = 3  # a skeleton shared by >= this many keys in a sheet is flagged boilerplate


def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def make_skeleton(description: str, endpoint: str, method: str, sub_feature: str) -> str:
    """Strip out words that are clearly just per-key substitutions (drawn from
    the endpoint path, method, sub-feature name) and digits, leaving the
    generic sentence structure underneath. Two descriptions that reduce to
    the same skeleton are templated variants of the same sentence, not
    independently-written descriptions."""
    substituted_words = set(tokenize(endpoint)) | set(tokenize(method)) | set(tokenize(sub_feature or ""))
    words = tokenize(description)
    skeleton_words = [w for w in words if w not in substituted_words and not w.isdigit()]
    return " ".join(skeleton_words)


def detect_boilerplate(doc_map: dict, df: pd.DataFrame, min_shared: int = MIN_SHARED_SKELETON_COUNT) -> tuple:
    """Returns (boilerplate_keys: set, skeleton_report: DataFrame) -
    skeleton_report has one row per (sheet, skeleton) group with count >= min_shared,
    sorted by count descending, so you can see the worst-offending templates first."""
    sheet_of_key = {}
    subfeature_of_key = {}
    for (ep, m), g in df.groupby(["endpoint", "method"]):
        sheet_of_key[(ep, m)] = g["sheet"].iloc[0]
        subfeature_of_key[(ep, m)] = g["subFeature"].iloc[0]

    # skeleton -> sheet -> list of keys
    skeleton_groups = defaultdict(lambda: defaultdict(list))
    for key, desc in doc_map.items():
        sheet = sheet_of_key.get(key)
        if sheet is None:
            continue  # doc_map key not present in this dataset at all
        sub_feature = subfeature_of_key.get(key, "")
        endpoint, method = key
        skeleton = make_skeleton(desc, endpoint, method, sub_feature)
        if len(skeleton) < 15:
            continue  # too short/generic to meaningfully compare, skip rather than false-flag
        skeleton_groups[sheet][skeleton].append(key)

    boilerplate_keys = set()
    rows = []
    for sheet, skeletons in skeleton_groups.items():
        for skeleton, keys in skeletons.items():
            if len(keys) >= min_shared:
                boilerplate_keys.update(keys)
                rows.append({
                    "sheet": sheet,
                    "skeleton_preview": skeleton[:100],
                    "n_keys_sharing_it": len(keys),
                    "example_keys": "; ".join(f"{m} {e}" for e, m in keys[:3]),
                })

    report_df = pd.DataFrame(rows).sort_values("n_keys_sharing_it", ascending=False) if rows else pd.DataFrame(rows)
    return boilerplate_keys, report_df


def build_doc_covered_eval(df: pd.DataFrame, doc_keys: set, n_eval: int, seed: int):
    true_df = df[df["markedCorrect"] == True].copy()
    true_df["key"] = list(zip(true_df["endpoint"], true_df["method"]))
    query_groups = (
        true_df.groupby("query")["key"]
        .apply(lambda keys: sorted(set(keys)))
        .reset_index()
        .rename(columns={"key": "valid_keys"})
    )
    eligible = query_groups[
        query_groups["valid_keys"].apply(lambda ks: len(ks) == 1 and ks[0] in doc_keys)
    ]
    n = min(n_eval, len(eligible))
    eval_df = eligible.sample(n=n, random_state=seed).reset_index(drop=True)
    return eval_df


def embed_dispatch(texts, client, args, cache_dir):
    if args.mock:
        return get_embeddings_mock(texts)
    return embed_texts(client, args.embed_model, texts, cache_dir, not args.no_cache,
                        args.embed_batch_size, request_timeout=args.timeout)


def main():
    parser = argparse.ArgumentParser(description="Detect boilerplate doc_description text and check its effect on recall@1")
    parser.add_argument("csv_path")
    parser.add_argument("xlsx_path")
    parser.add_argument("--extra-descriptions", default=None)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-examples", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--embed-model", default="azure:primary/s247-textembedding-3l")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--skip-retrieval-check", action="store_true",
                         help="only run Part A (static boilerplate detection) - no API calls, no embeddings")
    parser.add_argument("--min-shared", type=int, default=MIN_SHARED_SKELETON_COUNT,
                         help=f"a description skeleton shared by >= this many keys in the same sheet is "
                              f"flagged as boilerplate (default {MIN_SHARED_SKELETON_COUNT})")
    args = parser.parse_args()

    csv_path, xlsx_path = Path(args.csv_path), Path(args.xlsx_path)
    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")
    if not xlsx_path.exists():
        sys.exit(f"File not found: {xlsx_path}")

    df = pd.read_csv(csv_path)
    if df["markedCorrect"].dtype == object:
        df["markedCorrect"] = df["markedCorrect"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    df["markedCorrect"] = df["markedCorrect"].astype(bool)

    print("Loading description file...")
    doc_map = load_doc_descriptions(xlsx_path)
    if args.extra_descriptions:
        extra_map = load_extra_descriptions(Path(args.extra_descriptions))
        for k, v in extra_map.items():
            doc_map.setdefault(k, v)
    all_keys = set(zip(df["endpoint"], df["method"]))
    doc_keys = set(doc_map.keys()) & all_keys
    print(f"  doc_map covers {len(doc_keys)}/{len(all_keys)} pairs\n")

    # ============ PART A: static boilerplate detection ============
    print("=" * 70)
    print(f"PART A: static boilerplate detection (skeleton shared by >= {args.min_shared} keys/sheet)")
    print("=" * 70)
    boilerplate_keys, report_df = detect_boilerplate(doc_map, df, min_shared=args.min_shared)
    boilerplate_keys &= doc_keys  # only count keys actually in this dataset

    sheet_of_key = {}
    for (ep, m), g in df.groupby(["endpoint", "method"]):
        sheet_of_key[(ep, m)] = g["sheet"].iloc[0]

    print(f"\n{len(boilerplate_keys)}/{len(doc_keys)} doc-covered keys "
          f"({len(boilerplate_keys) / max(len(doc_keys), 1) * 100:.1f}%) flagged as boilerplate overall.\n")

    if not report_df.empty:
        by_sheet_total = defaultdict(int)
        by_sheet_boilerplate = defaultdict(int)
        for k in doc_keys:
            sheet = sheet_of_key.get(k)
            by_sheet_total[sheet] += 1
            if k in boilerplate_keys:
                by_sheet_boilerplate[sheet] += 1
        print("Boilerplate % by sheet (doc-covered keys only):")
        sheet_pct_rows = []
        for sheet in by_sheet_total:
            total = by_sheet_total[sheet]
            bp = by_sheet_boilerplate.get(sheet, 0)
            sheet_pct_rows.append({"sheet": sheet, "boilerplate_keys": bp, "total_doc_covered_keys": total,
                                    "pct_boilerplate": round(100 * bp / total, 1)})
        sheet_pct_df = pd.DataFrame(sheet_pct_rows).sort_values("pct_boilerplate", ascending=False)
        print(sheet_pct_df.to_string(index=False))

        print("\nTop shared skeletons (the actual templates to go fix), by how many keys reuse them:")
        print(report_df.head(15).to_string(index=False))
    else:
        print(f"No skeleton was shared by >= {args.min_shared} keys in the same sheet - "
              f"either your descriptions are genuinely distinct, or try --min-shared 2 to loosen the check.")

    out_dir = csv_path.parent
    report_df.to_csv(out_dir / "boilerplate_diagnostic_skeletons.csv", index=False)
    print(f"\nSaved -> {out_dir / 'boilerplate_diagnostic_skeletons.csv'}")

    if args.skip_retrieval_check:
        return

    # ============ PART B: retrieval cross-reference ============
    print("\n" + "=" * 70)
    print("PART B: recall@1 on boilerplate-tagged vs non-boilerplate keys")
    print("        (doc-covered-only eval, doc_description strategy, no fallback)")
    print("=" * 70)

    cache_dir = Path(args.cache_dir) if args.cache_dir else csv_path.parent / ".rag_cache"
    client = None
    if not args.mock:
        if not args.base_url or not args.api_key:
            sys.exit("Provide --base-url and --api-key, or use --mock for an offline test.")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    rows = []
    n_eligible_reported = False
    all_per_query = []

    for seed in args.seeds:
        eval_df = build_doc_covered_eval(df, doc_keys, args.n_eval, seed)
        if not n_eligible_reported:
            print(f"  sampling {len(eval_df)} doc-covered queries per seed\n")
            n_eligible_reported = True
        print(f"--- seed {seed} ---")

        true_df = df[df["markedCorrect"] == True].copy()
        true_df["key"] = list(zip(true_df["endpoint"], true_df["method"]))
        eval_queries = set(eval_df["query"])
        enrichment_pool = true_df[~true_df["query"].isin(eval_queries)]

        # full corpus (all keys), only restricting which queries we evaluate
        corpus_rows = []
        for key, group in df.groupby(["endpoint", "method"]):
            endpoint, method = key
            sheet = group["sheet"].iloc[0]
            sub_feature = group["subFeature"].iloc[0]
            pool = enrichment_pool[enrichment_pool["key"] == key]["query"].tolist()
            examples = pool[:args.n_examples]
            doc_desc = doc_map.get(key)
            text = build_text_doc_description(method, endpoint, sheet, sub_feature, examples, doc_desc)
            corpus_rows.append({"endpoint": endpoint, "method": method, "key": key, "text": text})
        corpus_df = pd.DataFrame(corpus_rows)

        corpus_vecs = embed_dispatch(corpus_df["text"].tolist(), client, args, cache_dir)
        query_vecs = embed_dispatch(eval_df["query"].tolist(), client, args, cache_dir)

        per_query_df, recall = compute_recall(corpus_df, eval_df, corpus_vecs, query_vecs,
                                               ks=(1, 10), variant=f"doc_description_seed{seed}_boilerplate_check")

        # tag each eval query's true key as boilerplate or not
        true_key_of_query = {row["query"]: row["valid_keys"][0] for _, row in eval_df.iterrows()}
        per_query_df["true_key"] = per_query_df["query"].map(true_key_of_query)
        per_query_df["is_boilerplate"] = per_query_df["true_key"].apply(lambda k: k in boilerplate_keys)
        per_query_df["seed"] = seed
        all_per_query.append(per_query_df)

        for tag, group in per_query_df.groupby("is_boilerplate"):
            n = len(group)
            hits1 = group["rank_of_correct"].apply(lambda r: r == 1).sum()
            rows.append({"seed": seed, "is_boilerplate": tag, "recall@1": hits1 / n if n else float("nan"), "n": n})

    combined = pd.DataFrame(rows)
    summary = combined.groupby("is_boilerplate").agg(
        recall_1_mean=("recall@1", "mean"), recall_1_std=("recall@1", "std"), avg_n=("n", "mean"),
    ).round(4)

    print("\n" + "=" * 60)
    print("SUMMARY: recall@1, boilerplate-tagged vs non-boilerplate true-answer keys")
    print("=" * 60)
    print(summary.to_string())

    pq_df = pd.concat(all_per_query, ignore_index=True)
    pq_df.to_csv(out_dir / "boilerplate_diagnostic_per_query.csv", index=False)
    combined.to_csv(out_dir / "boilerplate_diagnostic_retrieval_per_seed.csv", index=False)
    summary.to_csv(out_dir / "boilerplate_diagnostic_retrieval_summary.csv")
    print(f"\nSaved -> {out_dir / 'boilerplate_diagnostic_retrieval_per_seed.csv'}")
    print(f"Saved -> {out_dir / 'boilerplate_diagnostic_retrieval_summary.csv'}")
    print(f"Saved -> {out_dir / 'boilerplate_diagnostic_per_query.csv'}")
    print("\nIf recall@1 is meaningfully lower for is_boilerplate=True than for is_boilerplate=False, "
          "that confirms boilerplate text is a real, fixable drag on recall - worth rewriting those "
          "descriptions (or having an LLM rewrite them with endpoint-specific detail) before trying "
          "anything else, since none of the ranking-layer tricks so far have beaten dense-only.")


if __name__ == "__main__":
    main()