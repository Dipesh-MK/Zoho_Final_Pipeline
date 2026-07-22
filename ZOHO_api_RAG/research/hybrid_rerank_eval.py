"""
hybrid_rerank_eval.py

Approach 2: hybrid retrieval. Combines your existing dense embedding search
with BM25 (sparse/lexical) search via Reciprocal Rank Fusion (RRF), and
checks whether that beats dense-only retrieval.

Why this addresses the synonym worry directly instead of making it worse:
BM25 alone WOULD have the synonym problem you're worried about (exact term
match only). But it's never used alone here - it's fused with dense
similarity, which already handles synonyms fine. RRF just adds points to a
chunk's score for ALSO ranking well on exact terms; if a query is pure
synonym/rephrasing with no lexical overlap, BM25 contributes near-nothing to
that query's fused ranking and dense alone effectively decides it, same as
today. It should only ever help (catching queries where the correct
endpoint's exact terminology - e.g. specific field/report names - didn't
make it into the embedding space as strongly as some other chunk's), not
actively hurt synonym-heavy queries.

RRF formula: score(chunk) = sum over each ranker's full ranking of
w_r / (k + rank_r), k=60 (the standard constant from the original RRF paper).
w_r lets you weight dense higher than BM25 instead of RRF's implicit
equal-weighting - added after equal-weight fusion was found to HURT recall@1
in 8/8 seeds (0.79 -> 0.73), likely because BM25 was voting equally on
queries where dense was already confident and correct.

Fixed to the doc_description corpus-text strategy throughout (already
established as the best of the three tested). Uses the same repeated-seed
methodology as compare_doc_strategies.py: recall@1/@10 mean +/- std across
several seeds, not a single fixed sample.

Can optionally take embeddings from a fine-tuned model instead of the Azure
API, via --embeddings-source local:<path-to-sentence-transformers-model>.

Usage:
    python hybrid_rerank_eval.py site24x7_Dataset.csv site24x7_Admin_API.xlsx --mock
    python hybrid_rerank_eval.py site24x7_Dataset.csv site24x7_Admin_API.xlsx \
        --base-url http://20.235.183.15:443/openai/v1 --api-key YOUR_KEY \
        --extra-descriptions reports_synthetic_descriptions.csv --seeds 1 2 3 4 5 6 7 8

    # weighted fusion sweep, dense weighted much higher than BM25:
    python hybrid_rerank_eval.py site24x7_Dataset.csv site24x7_Admin_API.xlsx \
        --base-url ... --api-key ... --w-dense 0.8 --w-bm25 0.2

    # only fuse BM25 in when dense itself is unsure (top1 vs top2 margin small):
    python hybrid_rerank_eval.py site24x7_Dataset.csv site24x7_Admin_API.xlsx \
        --base-url ... --api-key ... --margin-gate 0.05
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from rag_eval import build_corpus_and_eval, embed_texts, get_embeddings_mock
from compare_doc_strategies import (
    build_text_doc_description,
    load_doc_descriptions,
    load_extra_descriptions,
    make_variant_corpus,
)

RRF_K = 60


def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def rrf_fuse(dense_ranked_idx: np.ndarray, bm25_ranked_idx: np.ndarray,
             w_dense: float = 1.0, w_bm25: float = 1.0, k: int = RRF_K) -> np.ndarray:
    """Both inputs are full corpus-index rankings (best first) for ONE query.
    Returns a fused corpus-index ranking, best first. w_dense/w_bm25 let you
    weight one ranker's vote more than the other instead of RRF's default
    equal weighting."""
    n = len(dense_ranked_idx)
    dense_rank_of = np.empty(n, dtype=int)
    dense_rank_of[dense_ranked_idx] = np.arange(n)
    bm25_rank_of = np.empty(n, dtype=int)
    bm25_rank_of[bm25_ranked_idx] = np.arange(n)
    scores = w_dense / (k + dense_rank_of + 1) + w_bm25 / (k + bm25_rank_of + 1)
    return np.argsort(-scores)


def recall_from_rankings(corpus_keys, eval_df, ranked_idx_per_query, ks=(1, 10)):
    results = {kk: 0 for kk in ks}
    max_k = max(ks)
    rows = []
    for i, row in eval_df.iterrows():
        valid_keys = set(row["valid_keys"])
        top_idx = ranked_idx_per_query[i][:max_k]
        top_keys = [corpus_keys[j] for j in top_idx]
        rank = next((r + 1 for r, kk in enumerate(top_keys) if kk in valid_keys), None)
        for kk in ks:
            if rank is not None and rank <= kk:
                results[kk] += 1
        rows.append({"query": row["query"], "rank_of_correct": rank if rank is not None else f">{max_k}"})
    n = len(eval_df)
    return {kk: results[kk] / n for kk in ks}, pd.DataFrame(rows)


def embed_dispatch(texts, client, args, cache_dir, st_model=None):
    if args.mock:
        return get_embeddings_mock(texts)
    if st_model is not None:
        return np.asarray(st_model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=False))
    return embed_texts(client, args.embed_model, texts, cache_dir, not args.no_cache,
                        args.embed_batch_size, request_timeout=args.timeout)


def main():
    parser = argparse.ArgumentParser(description="Dense vs dense+BM25(RRF) hybrid retrieval eval, doc_description corpus")
    parser.add_argument("csv_path")
    parser.add_argument("xlsx_path")
    parser.add_argument("--extra-descriptions", default=None)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-examples", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--embed-model", default="azure:primary/s247-textembedding-3l")
    parser.add_argument("--embeddings-source", default=None,
                         help="local:<path> to use a local sentence-transformers model instead of the API")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)

    # --- new knobs ---
    parser.add_argument("--w-dense", type=float, default=1.0,
                         help="RRF weight for the dense ranker (default 1.0 = equal weight, RRF's original default)")
    parser.add_argument("--w-bm25", type=float, default=1.0,
                         help="RRF weight for BM25 (default 1.0 = equal weight)")
    parser.add_argument("--margin-gate", type=float, default=None,
                         help="If set, only fuse BM25 in for a query when dense's top1-vs-top2 cosine "
                              "similarity margin is BELOW this value (i.e. dense is unsure). When dense "
                              "is confident (margin >= this value), use dense's ranking as-is, skipping "
                              "fusion entirely for that query. Example: 0.05")
    parser.add_argument("--save-per-query", action="store_true",
                         help="Save per-query rank_of_correct for both dense and hybrid, to "
                              "hybrid_rerank_eval_per_query.csv, so you can see exactly which queries "
                              "hybrid broke (dense rank==1 but hybrid rank>1) and which sheet they're in.")
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
    print(f"  doc_map covers {len(set(doc_map.keys()) & all_keys)}/{len(all_keys)} pairs")
    print(f"  RRF weights: w_dense={args.w_dense}  w_bm25={args.w_bm25}"
          + (f"  margin_gate={args.margin_gate}" if args.margin_gate is not None else "") + "\n")

    st_model = None
    client = None
    cache_dir = Path(args.cache_dir) if args.cache_dir else csv_path.parent / ".rag_cache"

    if args.embeddings_source and args.embeddings_source.startswith("local:"):
        model_path = args.embeddings_source[len("local:"):]
        print(f"Loading local sentence-transformers model from {model_path} ...")
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(model_path)
    elif not args.mock:
        if not args.base_url or not args.api_key:
            sys.exit("Provide --base-url and --api-key, --embeddings-source local:<path>, or --mock.")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    dense_rows, hybrid_rows = [], []
    all_per_query = []

    for seed in args.seeds:
        print(f"\n--- seed {seed} ---")
        base_corpus_df, eval_df = build_corpus_and_eval(df, args.n_eval, args.n_examples, seed)

        true_df = df[df["markedCorrect"] == True].copy()
        true_df["key"] = list(zip(true_df["endpoint"], true_df["method"]))
        eval_queries = set(eval_df["query"])
        enrichment_pool = true_df[~true_df["query"].isin(eval_queries)]

        corpus_df = make_variant_corpus(base_corpus_df, df, doc_map, args.n_examples,
                                         enrichment_pool, build_text_doc_description)
        corpus_keys = list(corpus_df["key"])

        corpus_vecs = embed_dispatch(corpus_df["text"].tolist(), client, args, cache_dir, st_model)
        query_vecs = embed_dispatch(eval_df["query"].tolist(), client, args, cache_dir, st_model)

        corpus_norm = corpus_vecs / np.clip(np.linalg.norm(corpus_vecs, axis=1, keepdims=True), 1e-9, None)
        query_norm = query_vecs / np.clip(np.linalg.norm(query_vecs, axis=1, keepdims=True), 1e-9, None)
        dense_sims = query_norm @ corpus_norm.T
        dense_ranked = np.argsort(-dense_sims, axis=1)

        bm25 = BM25Okapi([tokenize(t) for t in corpus_df["text"].tolist()])
        bm25_ranked = np.array([
            np.argsort(-bm25.get_scores(tokenize(q))) for q in eval_df["query"].tolist()
        ])

        hybrid_ranked = []
        for i in range(len(eval_df)):
            if args.margin_gate is not None:
                sims_sorted = np.sort(dense_sims[i])[::-1]
                margin = sims_sorted[0] - sims_sorted[1] if len(sims_sorted) > 1 else 1.0
                if margin >= args.margin_gate:
                    # dense is confident -> skip fusion, use dense ranking as-is
                    hybrid_ranked.append(dense_ranked[i])
                    continue
            hybrid_ranked.append(
                rrf_fuse(dense_ranked[i], bm25_ranked[i], w_dense=args.w_dense, w_bm25=args.w_bm25)
            )
        hybrid_ranked = np.array(hybrid_ranked)

        dense_recall, dense_pq = recall_from_rankings(corpus_keys, eval_df, dense_ranked, ks=(1, 10))
        hybrid_recall, hybrid_pq = recall_from_rankings(corpus_keys, eval_df, hybrid_ranked, ks=(1, 10))

        print(f"  dense-only : recall@1={dense_recall[1]:.3f}  recall@10={dense_recall[10]:.3f}")
        print(f"  hybrid RRF : recall@1={hybrid_recall[1]:.3f}  recall@10={hybrid_recall[10]:.3f}")

        dense_rows.append({"seed": seed, "recall@1": dense_recall[1], "recall@10": dense_recall[10], "n_eval": len(eval_df)})
        hybrid_rows.append({"seed": seed, "recall@1": hybrid_recall[1], "recall@10": hybrid_recall[10], "n_eval": len(eval_df)})

        if args.save_per_query:
            # merge dense and hybrid per-query rank, flag queries hybrid broke
            merged = dense_pq.merge(hybrid_pq, on="query", suffixes=("_dense", "_hybrid"))
            merged["seed"] = seed
            merged["hybrid_broke_it"] = merged.apply(
                lambda r: (r["rank_of_correct_dense"] == 1 and r["rank_of_correct_hybrid"] != 1), axis=1
            )
            # attach sheet info if available, to see if breakage clusters in boilerplate-heavy sheets
            sheet_lookup = df.drop_duplicates("query").set_index("query")["sheet"].to_dict()
            merged["sheet"] = merged["query"].map(sheet_lookup)
            all_per_query.append(merged)

    dense_df = pd.DataFrame(dense_rows)
    dense_df["method"] = "dense_only"
    hybrid_df = pd.DataFrame(hybrid_rows)
    hybrid_df["method"] = "hybrid_rrf"
    combined = pd.concat([dense_df, hybrid_df], ignore_index=True)

    out_path = csv_path.parent / "hybrid_rerank_eval_per_seed.csv"
    combined.to_csv(out_path, index=False)

    summary = combined.groupby("method").agg(
        recall_1_mean=("recall@1", "mean"), recall_1_std=("recall@1", "std"),
        recall_10_mean=("recall@10", "mean"), recall_10_std=("recall@10", "std"),
    ).round(4)

    print("\n" + "=" * 60)
    print(f"SUMMARY (mean +/- std across seeds, doc_description corpus, "
          f"w_dense={args.w_dense}, w_bm25={args.w_bm25}"
          + (f", margin_gate={args.margin_gate}" if args.margin_gate is not None else "") + ")")
    print("=" * 60)
    print(summary.to_string())
    summary.to_csv(csv_path.parent / "hybrid_rerank_eval_summary.csv")
    print(f"\nSaved -> {out_path}")
    print(f"Saved -> {csv_path.parent / 'hybrid_rerank_eval_summary.csv'}")

    if args.save_per_query and all_per_query:
        pq_df = pd.concat(all_per_query, ignore_index=True)
        pq_path = csv_path.parent / "hybrid_rerank_eval_per_query.csv"
        pq_df.to_csv(pq_path, index=False)
        n_broken = pq_df["hybrid_broke_it"].sum()
        print(f"Saved -> {pq_path}  ({n_broken} query-seed rows where hybrid broke a correct dense rank@1)")
        if n_broken > 0:
            print("\nTop sheets where hybrid broke a correct dense pick:")
            print(pq_df[pq_df["hybrid_broke_it"]]["sheet"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()