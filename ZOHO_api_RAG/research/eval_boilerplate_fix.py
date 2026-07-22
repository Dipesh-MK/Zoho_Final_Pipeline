"""
eval_boilerplate_fix.py

Paired before/after check: did replacing the boilerplate descriptions
(from synthesize_boilerplate_descriptions.py) actually improve recall@1 on
the SAME keys that were flagged as boilerplate? This is scoped narrowly on
purpose - it only evaluates queries whose true answer is one of the
previously-flagged keys, with the OLD xlsx/synthetic description vs the NEW
replacement description, everything else held constant (same corpus size,
same eval queries, same seeds).

Important merge detail: compare_doc_strategies.py's --extra-descriptions
uses `doc_map.setdefault(k, v)` - the xlsx always wins on collision. That's
correct for ADDING new coverage (e.g. Reports), but wrong here, since we
specifically want to REPLACE the xlsx's boilerplate text. This script
therefore overrides unconditionally: replacement descriptions win over
whatever was already in doc_map for the same key.

Usage:
    python eval_boilerplate_fix.py site24x7_Dataset.csv site24x7_Admin_API.xlsx \
        boilerplate_synthetic_descriptions.csv --mock

    python eval_boilerplate_fix.py site24x7_Dataset.csv site24x7_Admin_API.xlsx \
        boilerplate_synthetic_descriptions.csv \
        --base-url http://20.235.183.15:443/openai/v1 --api-key YOUR_KEY \
        --extra-descriptions reports_synthetic_descriptions.csv \
        --seeds 1 2 3 4 5 6 7 8
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from rag_eval import compute_recall, embed_texts, get_embeddings_mock
from compare_doc_strategies import build_text_doc_description, load_doc_descriptions, load_extra_descriptions


def build_doc_covered_eval_for_keys(df: pd.DataFrame, target_keys: set, n_eval: int, seed: int):
    """Same eligibility rule as compare_doc_strategies.py's doc-covered eval
    (single valid key, must be doc-covered), further restricted to
    target_keys only - so every sampled query's true answer is one of the
    keys we're specifically testing the fix for."""
    true_df = df[df["markedCorrect"] == True].copy()
    true_df["key"] = list(zip(true_df["endpoint"], true_df["method"]))
    query_groups = (
        true_df.groupby("query")["key"]
        .apply(lambda keys: sorted(set(keys)))
        .reset_index()
        .rename(columns={"key": "valid_keys"})
    )
    eligible = query_groups[
        query_groups["valid_keys"].apply(lambda ks: len(ks) == 1 and ks[0] in target_keys)
    ]
    n = min(n_eval, len(eligible))
    if n == 0:
        return eligible.sample(n=0, random_state=seed), 0
    eval_df = eligible.sample(n=n, random_state=seed).reset_index(drop=True)
    return eval_df, len(eligible)


def embed_dispatch(texts, client, args, cache_dir):
    if args.mock:
        return get_embeddings_mock(texts)
    return embed_texts(client, args.embed_model, texts, cache_dir, not args.no_cache,
                        args.embed_batch_size, request_timeout=args.timeout)


def build_full_corpus(df: pd.DataFrame, doc_map: dict, n_examples: int, enrichment_pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(["endpoint", "method"]):
        endpoint, method = key
        sheet = group["sheet"].iloc[0]
        sub_feature = group["subFeature"].iloc[0]
        pool = enrichment_pool[enrichment_pool["key"] == key]["query"].tolist()
        examples = pool[:n_examples]
        doc_desc = doc_map.get(key)
        text = build_text_doc_description(method, endpoint, sheet, sub_feature, examples, doc_desc)
        rows.append({"endpoint": endpoint, "method": method, "key": key, "text": text})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Paired before/after recall@1 check on boilerplate-flagged keys, old vs replacement descriptions")
    parser.add_argument("csv_path")
    parser.add_argument("xlsx_path")
    parser.add_argument("replacements_csv", help="output of synthesize_boilerplate_descriptions.py")
    parser.add_argument("--extra-descriptions", default=None,
                         help="e.g. reports_synthetic_descriptions.csv - merged in the same way as other scripts "
                              "(xlsx wins on collision) BEFORE the replacements_csv override is applied")
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
    args = parser.parse_args()

    csv_path, xlsx_path = Path(args.csv_path), Path(args.xlsx_path)
    repl_path = Path(args.replacements_csv)
    for p in (csv_path, xlsx_path, repl_path):
        if not p.exists():
            sys.exit(f"File not found: {p}")

    df = pd.read_csv(csv_path)
    if df["markedCorrect"].dtype == object:
        df["markedCorrect"] = df["markedCorrect"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    df["markedCorrect"] = df["markedCorrect"].astype(bool)

    print("Loading description file...")
    doc_map_before = load_doc_descriptions(xlsx_path)
    if args.extra_descriptions:
        extra_map = load_extra_descriptions(Path(args.extra_descriptions))
        for k, v in extra_map.items():
            doc_map_before.setdefault(k, v)
    all_keys = set(zip(df["endpoint"], df["method"]))
    print(f"  doc_map (before fix) covers {len(set(doc_map_before.keys()) & all_keys)}/{len(all_keys)} pairs")

    repl_df = pd.read_csv(repl_path)
    required = {"endpoint", "method", "description"}
    missing = required - set(repl_df.columns)
    if missing:
        sys.exit(f"{repl_path} is missing required column(s): {missing}")

    target_keys = set(zip(repl_df["endpoint"].astype(str).str.strip(), repl_df["method"].astype(str).str.strip()))
    target_keys &= all_keys
    print(f"  {len(target_keys)} keys have a replacement description to test\n")

    doc_map_after = dict(doc_map_before)  # copy
    for _, r in repl_df.iterrows():
        key = (str(r["endpoint"]).strip(), str(r["method"]).strip())
        doc_map_after[key] = str(r["description"]).strip()  # unconditional override, unlike setdefault

    cache_dir = Path(args.cache_dir) if args.cache_dir else csv_path.parent / ".rag_cache"
    client = None
    if not args.mock:
        if not args.base_url or not args.api_key:
            sys.exit("Provide --base-url and --api-key, or use --mock for an offline test.")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    before_rows, after_rows = [], []
    n_eligible_reported = False

    for seed in args.seeds:
        eval_df, n_eligible = build_doc_covered_eval_for_keys(df, target_keys, args.n_eval, seed)
        if not n_eligible_reported:
            print(f"  {n_eligible} eligible queries whose true answer is one of the replaced keys "
                  f"(sampling {len(eval_df)} per seed)\n")
            n_eligible_reported = True
        if len(eval_df) == 0:
            print(f"--- seed {seed}: 0 eligible queries, skipping ---")
            continue
        print(f"--- seed {seed} ---")

        true_df = df[df["markedCorrect"] == True].copy()
        true_df["key"] = list(zip(true_df["endpoint"], true_df["method"]))
        eval_queries = set(eval_df["query"])
        enrichment_pool = true_df[~true_df["query"].isin(eval_queries)]

        query_vecs = embed_dispatch(eval_df["query"].tolist(), client, args, cache_dir)

        corpus_before = build_full_corpus(df, doc_map_before, args.n_examples, enrichment_pool)
        corpus_vecs_before = embed_dispatch(corpus_before["text"].tolist(), client, args, cache_dir)
        _, recall_before = compute_recall(corpus_before, eval_df, corpus_vecs_before, query_vecs,
                                           ks=(1, 10), variant=f"before_fix_seed{seed}")

        corpus_after = build_full_corpus(df, doc_map_after, args.n_examples, enrichment_pool)
        corpus_vecs_after = embed_dispatch(corpus_after["text"].tolist(), client, args, cache_dir)
        _, recall_after = compute_recall(corpus_after, eval_df, corpus_vecs_after, query_vecs,
                                          ks=(1, 10), variant=f"after_fix_seed{seed}")

        print(f"  before: recall@1={recall_before[1]:.3f}  recall@10={recall_before[10]:.3f}")
        print(f"  after : recall@1={recall_after[1]:.3f}  recall@10={recall_after[10]:.3f}")

        before_rows.append({"seed": seed, "recall@1": recall_before[1], "recall@10": recall_before[10], "n_eval": len(eval_df)})
        after_rows.append({"seed": seed, "recall@1": recall_after[1], "recall@10": recall_after[10], "n_eval": len(eval_df)})

    if not before_rows:
        sys.exit("No seed had any eligible queries - the replaced keys may have no single-answer "
                  "queries in this dataset. Check target_keys overlap with your eval-eligible query set.")

    before_df = pd.DataFrame(before_rows)
    before_df["variant"] = "before_fix"
    after_df = pd.DataFrame(after_rows)
    after_df["variant"] = "after_fix"
    combined = pd.concat([before_df, after_df], ignore_index=True)

    out_dir = csv_path.parent
    combined.to_csv(out_dir / "eval_boilerplate_fix_per_seed.csv", index=False)

    summary = combined.groupby("variant").agg(
        recall_1_mean=("recall@1", "mean"), recall_1_std=("recall@1", "std"),
        recall_10_mean=("recall@10", "mean"), recall_10_std=("recall@10", "std"),
        avg_n=("n_eval", "mean"),
    ).round(4)

    print("\n" + "=" * 60)
    print(f"SUMMARY: recall@1/@10 on the {len(target_keys)} replaced keys, before vs after")
    print("=" * 60)
    print(summary.to_string())
    summary.to_csv(out_dir / "eval_boilerplate_fix_summary.csv")
    print(f"\nSaved -> {out_dir / 'eval_boilerplate_fix_per_seed.csv'}")
    print(f"Saved -> {out_dir / 'eval_boilerplate_fix_summary.csv'}")
    print("\nIf 'after_fix' recall@1 is meaningfully higher than 'before_fix', the replacement "
          "descriptions are working - worth merging boilerplate_synthetic_descriptions.csv into your "
          "main corpus pipeline (as an override, same as this script does) and re-running your full "
          "compare_doc_strategies.py baseline to confirm the lift holds at the whole-dataset level too.")


if __name__ == "__main__":
    main()