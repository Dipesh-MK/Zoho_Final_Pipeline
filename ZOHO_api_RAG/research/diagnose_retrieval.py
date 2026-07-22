"""
diagnose_retrieval.py

Detailed per-query diagnostics for the RAG eval: full ranking (not just top-10),
score gaps (ambiguity signal), keyword overlap between query and candidates,
and for misses, the rank/score of the actual ground truth answer.

Reuses build_corpus_and_eval / embed_texts from rag_eval.py so it shares the
same embedding cache - re-running this costs ~nothing if you've already run
rag_eval.py on the same csv/n-eval/seed.

Usage:
    python diagnose_retrieval.py .\Datasets\site24x7_Dataset.csv `
        --base-url http://20.235.183.15:443/openai/v1 --api-key YOUR_KEY --n-eval 100
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rag_eval import build_corpus_and_eval, embed_texts, get_embeddings_mock

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "can",
    "could", "how", "what", "where", "when", "why", "which", "who", "to",
    "of", "in", "on", "for", "and", "or", "i", "my", "me", "it", "this",
    "that", "with", "as", "be", "get", "set", "up", "you", "your", "we",
    "there", "will", "would", "should", "if", "so", "at", "by", "from",
}


def keywords(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def keyword_overlap(query_kw: set, chunk_text: str):
    chunk_kw = keywords(chunk_text)
    if not query_kw:
        return 0, 0.0, ""
    shared = query_kw & chunk_kw
    jaccard = len(shared) / len(query_kw | chunk_kw) if (query_kw | chunk_kw) else 0.0
    return len(shared), jaccard, ", ".join(sorted(shared))


def full_rank_analysis(corpus_df, eval_df, corpus_vecs, query_vecs, top_n_display=10):
    corpus_norm = corpus_vecs / np.clip(np.linalg.norm(corpus_vecs, axis=1, keepdims=True), 1e-9, None)
    query_norm = query_vecs / np.clip(np.linalg.norm(query_vecs, axis=1, keepdims=True), 1e-9, None)
    sims = query_norm @ corpus_norm.T  # (n_queries, n_corpus)
    ranked_idx = np.argsort(-sims, axis=1)  # full ranking, descending, per query

    corpus_keys = list(corpus_df["key"])
    corpus_texts = list(corpus_df["text"])

    rows = []          # summary CSV, one row per query
    report_blocks = [] # human-readable text blocks

    for i, row in eval_df.iterrows():
        query = row["query"]
        true_key = tuple(next(iter(row["valid_keys"]))) if len(row["valid_keys"]) == 1 else None
        valid_keys = set(row["valid_keys"])
        q_kw = keywords(query)

        order = ranked_idx[i]
        scores = sims[i, order]

        # rank (1-indexed) of first valid answer anywhere in the full corpus
        gt_rank = next((r + 1 for r, j in enumerate(order) if corpus_keys[j] in valid_keys), None)
        gt_score = scores[gt_rank - 1] if gt_rank else None

        top1_score = scores[0]
        top2_score = scores[1] if len(scores) > 1 else None
        top10_score = scores[min(9, len(scores) - 1)]
        gap_1_2 = top1_score - top2_score if top2_score is not None else None
        gap_1_10 = top1_score - top10_score

        hit1 = gt_rank == 1
        hit10 = gt_rank is not None and gt_rank <= 10

        # keyword overlap across displayed top-N
        overlaps = []
        for r in range(min(top_n_display, len(order))):
            j = order[r]
            n_shared, jacc, shared_words = keyword_overlap(q_kw, corpus_texts[j])
            overlaps.append({
                "rank": r + 1,
                "endpoint": corpus_keys[j][0],
                "method": corpus_keys[j][1],
                "score": scores[r],
                "is_correct": corpus_keys[j] in valid_keys,
                "kw_shared": n_shared,
                "kw_jaccard": jacc,
                "kw_words": shared_words,
            })
        avg_kw_overlap_top10 = np.mean([o["kw_shared"] for o in overlaps]) if overlaps else 0.0

        rows.append({
            "query": query,
            "valid_endpoints": "; ".join(f"{m} {e}" for e, m in valid_keys),
            "hit@1": hit1,
            "hit@10": hit10,
            "gt_rank": gt_rank if gt_rank else f">{len(order)}",
            "gt_score": round(gt_score, 4) if gt_score is not None else "",
            "top1_endpoint": corpus_keys[order[0]][0],
            "top1_method": corpus_keys[order[0]][1],
            "top1_score": round(top1_score, 4),
            "gap_rank1_vs_rank2": round(gap_1_2, 4) if gap_1_2 is not None else "",
            "gap_rank1_vs_rank10": round(gap_1_10, 4),
            "top1_kw_shared": overlaps[0]["kw_shared"] if overlaps else "",
            "top1_kw_words": overlaps[0]["kw_words"] if overlaps else "",
            "avg_kw_shared_top10": round(avg_kw_overlap_top10, 2),
            "ambiguous_flag": (gap_1_2 is not None and gap_1_2 < 0.02),  # tune threshold after eyeballing
        })

        report_blocks.append(_format_query_block(query, valid_keys, gt_rank, gt_score, overlaps, hit1))

    summary_df = pd.DataFrame(rows)
    return summary_df, report_blocks


def _format_query_block(query, valid_keys, gt_rank, gt_score, overlaps, hit1):
    lines = []
    lines.append("=" * 90)
    status = "HIT (rank 1)" if hit1 else (f"IN TOP-10 (rank {gt_rank})" if gt_rank and gt_rank <= 10
                                            else (f"MISS (found at rank {gt_rank})" if gt_rank else "MISS (not found at all)"))
    lines.append(f"QUERY: {query}")
    lines.append(f"STATUS: {status}")
    lines.append(f"GROUND TRUTH: {'; '.join(f'{m} {e}' for e, m in valid_keys)}"
                  + (f"   (score={gt_score:.4f})" if gt_score is not None else ""))
    lines.append("-" * 90)
    lines.append(f"{'RK':<3} {'SCORE':<8} {'CORRECT':<8} {'SHARED_KW':<10} {'METHOD':<7} ENDPOINT")
    for o in overlaps:
        mark = "✔" if o["is_correct"] else ""
        lines.append(
            f"{o['rank']:<3} {o['score']:.4f}  {mark:<8} {o['kw_shared']:<10} {o['method']:<7} {o['endpoint']}"
            + (f"   [shared: {o['kw_words']}]" if o["kw_words"] else "")
        )
    if not hit1 and overlaps:
        lines.append(f"gap rank1->rank2: {overlaps[0]['score'] - overlaps[1]['score']:.4f}  "
                      f"(small = genuinely ambiguous top choices, large = model confidently picked wrong)")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Per-query retrieval diagnostics")
    parser.add_argument("csv_path")
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-examples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--embed-model", default="azure:primary/s247-textembedding-3l")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--embed-batch-size", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--top-n-display", type=int, default=10, help="how many ranked candidates to show per query")
    args = parser.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    df = pd.read_csv(path)
    if df["markedCorrect"].dtype == object:
        df["markedCorrect"] = df["markedCorrect"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    df["markedCorrect"] = df["markedCorrect"].astype(bool)

    corpus_df, eval_df = build_corpus_and_eval(df, args.n_eval, args.n_examples, args.seed)
    print(f"Corpus: {len(corpus_df)} chunks | Eval queries: {len(eval_df)}\n")

    cache_dir = Path(args.cache_dir) if args.cache_dir else path.parent / ".rag_cache"

    if args.mock:
        corpus_vecs = get_embeddings_mock(corpus_df["text"].tolist())
        query_vecs = get_embeddings_mock(eval_df["query"].tolist())
    else:
        if not args.base_url or not args.api_key:
            sys.exit("Provide --base-url and --api-key, or use --mock.")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
        print("Embedding corpus (should be fully cached if you already ran rag_eval.py)...")
        corpus_vecs = embed_texts(client, args.embed_model, corpus_df["text"].tolist(),
                                   cache_dir, True, args.embed_batch_size, request_timeout=args.timeout)
        print("Embedding eval queries...")
        query_vecs = embed_texts(client, args.embed_model, eval_df["query"].tolist(),
                                  cache_dir, True, args.embed_batch_size, request_timeout=args.timeout)

    summary_df, report_blocks = full_rank_analysis(corpus_df, eval_df, corpus_vecs, query_vecs,
                                                     top_n_display=args.top_n_display)

    # order the readable report: misses first, then ambiguous hits, then clean hits
    misses = [b for b, hit in zip(report_blocks, summary_df["hit@1"]) if not hit]
    hits = [b for b, hit in zip(report_blocks, summary_df["hit@1"]) if hit]

    out_dir = path.parent
    report_path = out_dir / f"{path.stem}_diagnostics_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"MISSES / NOT TOP-1 ({len(misses)} queries)\n")
        f.write("#" * 90 + "\n\n")
        f.write("\n".join(misses))
        f.write(f"\n\nHITS ({len(hits)} queries)\n")
        f.write("#" * 90 + "\n\n")
        f.write("\n".join(hits))

    summary_path = out_dir / f"{path.stem}_diagnostics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # console overview
    print("=" * 60)
    print("QUICK STATS")
    print("=" * 60)
    hit_gaps = summary_df.loc[summary_df["hit@1"], "gap_rank1_vs_rank2"].replace("", np.nan).astype(float)
    miss_gaps = summary_df.loc[~summary_df["hit@1"], "gap_rank1_vs_rank2"].replace("", np.nan).astype(float)
    print(f"Avg rank1-vs-rank2 score gap | hits:  {hit_gaps.mean():.4f}")
    print(f"Avg rank1-vs-rank2 score gap | misses: {miss_gaps.mean():.4f}")
    print(f"  (if misses have a SMALL gap similar to hits -> model is genuinely torn between plausible answers)")
    print(f"  (if misses have a LARGE gap -> model is confidently wrong, not ambiguous - different problem)")
    print(f"Ambiguous flagged queries (gap < 0.02): {summary_df['ambiguous_flag'].sum()}")
    print(f"Avg keyword overlap (top1, hits):  {summary_df.loc[summary_df['hit@1'], 'top1_kw_shared'].mean():.2f}")
    print(f"Avg keyword overlap (top1, misses): {summary_df.loc[~summary_df['hit@1'], 'top1_kw_shared'].mean():.2f}")
    print()
    print(f"Saved readable report -> {report_path}")
    print(f"Saved sortable summary -> {summary_path}")
    print("\nIn the summary CSV, sort by 'gap_rank1_vs_rank2' ascending among misses to find the")
    print("truly-ambiguous cases first (best re-ranking candidates), and check 'top1_kw_shared'")
    print("vs 'avg_kw_shared_top10' to spot cases where a keyword-heavy wrong answer beat semantics.")


if __name__ == "__main__":
    main()