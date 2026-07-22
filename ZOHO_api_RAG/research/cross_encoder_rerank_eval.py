"""
cross_encoder_rerank_eval.py

Approach: two-stage retrieve-then-rerank. Dense embeddings retrieve a wide
top-N candidate set (cheap, and you already know recall@10 ~0.93-0.94, so the
correct answer is almost always somewhere in top-N for N>=10-20). A
cross-encoder then re-scores each of those N candidates JOINTLY with the
query - one forward pass per (query, candidate) pair, not two separately-
computed vectors compared by cosine similarity - and re-orders them.

Why this is a structurally different mechanism than the last two things you
tried (hybrid BM25+RRF, HyDE), and worth trying even though both of those
failed: those both worked by changing ONE side of an embedding-similarity
comparison (the query side for HyDE, adding a second independently-computed
ranking for hybrid) while still fundamentally relying on two vectors encoded
separately and compared by geometry. A cross-encoder has no such separation -
it reads the query and candidate text together and outputs one relevance
score directly, so it can pick up on query-candidate interactions (e.g. "this
query's phrasing specifically matches THIS candidate's phrasing, not just
generically similar topic") that two independently-encoded vectors structurally
cannot represent. It also has no analog to HyDE's failure mode (generic LLM
phrasing collapsing distinct candidates together), since there's no LLM
rewrite step here at all.

Cost/latency note: a cross-encoder forward pass is more expensive per
candidate than a cosine similarity lookup, which is why it's only run on the
narrow top-N dense shortlist, not the full 2298-chunk corpus - complexity is
O(n_eval * N), not O(n_eval * corpus_size).

IMPORTANT - this needs a real cross-encoder model, downloaded from
huggingface.co via sentence-transformers' CrossEncoder class, on WHATEVER
MACHINE actually runs this for real. The default model
(cross-encoder/ms-marco-MiniLM-L-6-v2) is a general-purpose passage-reranking
model trained on MS MARCO (web search Q&A pairs) - it has never seen
Site24x7's vocabulary, so treat its first result as a baseline "does generic
cross-encoder reranking help at all" check, not a ceiling. If it helps, a
domain-fine-tuned cross-encoder would very likely help more - but that's a
bigger lift than this script and worth doing only once you've confirmed the
generic model gives you signal to build on.

Usage:
    python cross_encoder_rerank_eval.py site24x7_Dataset.csv site24x7_Admin_API.xlsx --mock

    python cross_encoder_rerank_eval.py site24x7_Dataset.csv site24x7_Admin_API.xlsx \
        --base-url http://20.235.183.15:443/openai/v1 --api-key YOUR_KEY \
        --extra-descriptions reports_synthetic_descriptions.csv \
        --seeds 1 2 3 4 5 6 7 8 --n-eval 100 --top-n 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rag_eval import build_corpus_and_eval, embed_texts, get_embeddings_mock
from compare_doc_strategies import (
    build_text_doc_description,
    load_doc_descriptions,
    load_extra_descriptions,
    make_variant_corpus,
)


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


def embed_dispatch(texts, client, args, cache_dir):
    if args.mock:
        return get_embeddings_mock(texts)
    return embed_texts(client, args.embed_model, texts, cache_dir, not args.no_cache,
                        args.embed_batch_size, request_timeout=args.timeout)


def mock_cross_encoder_score(query: str, candidate_text: str) -> float:
    """Deterministic fake relevance score for offline pipeline testing only -
    plain word-overlap count between query and candidate. NOT a real
    cross-encoder, just exercises the rerank/re-ordering code path."""
    q_words = set(query.lower().split())
    c_words = set(candidate_text.lower().split())
    return len(q_words & c_words)


def main():
    parser = argparse.ArgumentParser(description="Dense top-N retrieval + cross-encoder rerank eval, doc_description corpus")
    parser.add_argument("csv_path")
    parser.add_argument("xlsx_path")
    parser.add_argument("--extra-descriptions", default=None)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-examples", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--embed-model", default="azure:primary/s247-textembedding-3l")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                         help="sentence-transformers CrossEncoder model id. Default is a general-purpose "
                              "MS MARCO passage-reranker, NOT domain-fine-tuned on Site24x7 data - treat "
                              "results as a baseline check, not a ceiling.")
    parser.add_argument("--top-n", type=int, default=20,
                         help="how many dense-retrieved candidates to hand to the cross-encoder for "
                              "reranking per query (default 20; must be >= 10 or recall@10 can't be "
                              "fairly compared, since reranking can only reorder within this shortlist, "
                              "never recover an answer dense missed entirely)")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--save-per-query", action="store_true",
                         help="save per-query rank_of_correct for dense vs reranked, flagging queries "
                              "the cross-encoder broke or fixed, to cross_encoder_rerank_eval_per_query.csv")
    args = parser.parse_args()

    if args.top_n < 10:
        sys.exit("--top-n must be >= 10 so recall@10 is a fair comparison (rerank can't recover an "
                  "answer dense didn't retrieve in the shortlist at all).")

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
    print(f"  cross-encoder model: {args.cross_encoder_model}  top_n={args.top_n}\n")

    cache_dir = Path(args.cache_dir) if args.cache_dir else csv_path.parent / ".rag_cache"

    client = None
    cross_encoder = None
    if not args.mock:
        if not args.base_url or not args.api_key:
            sys.exit("Provide --base-url and --api-key, or use --mock for an offline test.")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
        print(f"Loading cross-encoder model {args.cross_encoder_model} ...")
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder(args.cross_encoder_model)

    dense_rows, rerank_rows = [], []
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
        corpus_texts = corpus_df["text"].tolist()

        corpus_vecs = embed_dispatch(corpus_texts, client, args, cache_dir)
        query_vecs = embed_dispatch(eval_df["query"].tolist(), client, args, cache_dir)

        corpus_norm = corpus_vecs / np.clip(np.linalg.norm(corpus_vecs, axis=1, keepdims=True), 1e-9, None)
        query_norm = query_vecs / np.clip(np.linalg.norm(query_vecs, axis=1, keepdims=True), 1e-9, None)
        dense_sims = query_norm @ corpus_norm.T
        dense_ranked = np.argsort(-dense_sims, axis=1)

        # --- rerank: for each query, take its top-N dense candidates and
        # re-score/re-order them with the cross-encoder. Anything outside
        # top-N keeps its dense position (pushed below the reranked N) so
        # a miss beyond top-N stays a miss either way. ---
        reranked_full = []
        for i in range(len(eval_df)):
            query = eval_df["query"].iloc[i]
            shortlist_idx = dense_ranked[i][:args.top_n]
            if args.mock:
                scores = [mock_cross_encoder_score(query, corpus_texts[j]) for j in shortlist_idx]
            else:
                pairs = [(query, corpus_texts[j]) for j in shortlist_idx]
                scores = cross_encoder.predict(pairs)
            order = np.argsort(-np.asarray(scores))
            reranked_shortlist = shortlist_idx[order]
            rest = dense_ranked[i][args.top_n:]
            reranked_full.append(np.concatenate([reranked_shortlist, rest]))
        reranked_full = np.array(reranked_full)

        dense_recall, dense_pq = recall_from_rankings(corpus_keys, eval_df, dense_ranked, ks=(1, 10))
        rerank_recall, rerank_pq = recall_from_rankings(corpus_keys, eval_df, reranked_full, ks=(1, 10))

        print(f"  dense-only      : recall@1={dense_recall[1]:.3f}  recall@10={dense_recall[10]:.3f}")
        print(f"  cross-enc rerank: recall@1={rerank_recall[1]:.3f}  recall@10={rerank_recall[10]:.3f}")

        dense_rows.append({"seed": seed, "recall@1": dense_recall[1], "recall@10": dense_recall[10], "n_eval": len(eval_df)})
        rerank_rows.append({"seed": seed, "recall@1": rerank_recall[1], "recall@10": rerank_recall[10], "n_eval": len(eval_df)})

        if args.save_per_query:
            merged = dense_pq.merge(rerank_pq, on="query", suffixes=("_dense", "_rerank"))
            merged["seed"] = seed
            merged["rerank_fixed_it"] = merged.apply(
                lambda r: (r["rank_of_correct_dense"] != 1 and r["rank_of_correct_rerank"] == 1), axis=1
            )
            merged["rerank_broke_it"] = merged.apply(
                lambda r: (r["rank_of_correct_dense"] == 1 and r["rank_of_correct_rerank"] != 1), axis=1
            )
            sheet_lookup = df.drop_duplicates("query").set_index("query")["sheet"].to_dict()
            merged["sheet"] = merged["query"].map(sheet_lookup)
            all_per_query.append(merged)

    dense_df = pd.DataFrame(dense_rows)
    dense_df["method"] = "dense_only"
    rerank_df = pd.DataFrame(rerank_rows)
    rerank_df["method"] = "cross_encoder_rerank"
    combined = pd.concat([dense_df, rerank_df], ignore_index=True)

    out_path = csv_path.parent / "cross_encoder_rerank_eval_per_seed.csv"
    combined.to_csv(out_path, index=False)

    summary = combined.groupby("method").agg(
        recall_1_mean=("recall@1", "mean"), recall_1_std=("recall@1", "std"),
        recall_10_mean=("recall@10", "mean"), recall_10_std=("recall@10", "std"),
    ).round(4)

    print("\n" + "=" * 60)
    print(f"SUMMARY (mean +/- std across seeds, doc_description corpus, top_n={args.top_n}, "
          f"model={args.cross_encoder_model})")
    print("=" * 60)
    print(summary.to_string())
    summary.to_csv(csv_path.parent / "cross_encoder_rerank_eval_summary.csv")
    print(f"\nSaved -> {out_path}")
    print(f"Saved -> {csv_path.parent / 'cross_encoder_rerank_eval_summary.csv'}")

    if args.save_per_query and all_per_query:
        pq_df = pd.concat(all_per_query, ignore_index=True)
        pq_path = csv_path.parent / "cross_encoder_rerank_eval_per_query.csv"
        pq_df.to_csv(pq_path, index=False)
        n_fixed = pq_df["rerank_fixed_it"].sum()
        n_broke = pq_df["rerank_broke_it"].sum()
        print(f"Saved -> {pq_path}")
        print(f"  cross-encoder FIXED {n_fixed} query-seed rows (dense had it wrong at rank 1, rerank got it right)")
        print(f"  cross-encoder BROKE {n_broke} query-seed rows (dense had it right at rank 1, rerank got it wrong)")


if __name__ == "__main__":
    main()
    