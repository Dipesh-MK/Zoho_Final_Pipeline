"""
hyde_eval.py

Approach 3 (from the "still open" brainstorm): HyDE - Hypothetical Document
Embeddings. Instead of embedding the raw user query, ask an LLM to write a
short hypothetical API-doc-style passage that would answer the query, then
embed THAT and search against the doc_description corpus.

Why this might help specifically here: your corpus text (doc_description
strategy) is written in doc-style prose - "Fetches X for a given Y, filtered
by Z" - while eval queries are short informal user questions - "how do I see
which servers are slow". Dense similarity has to bridge that register gap on
its own today. HyDE bridges it explicitly by moving the QUERY into doc-space
instead of moving the corpus into query-space (which is what n_examples/
query_derived already do on the corpus side).

Three variants are compared per seed, all against the SAME doc_description
corpus (already established as your best corpus-text strategy):

  raw_query   : baseline - embed the query text as-is (this is your current
                production approach, recall@1 ~0.79).
  hyde_always : embed an LLM-generated hypothetical doc for EVERY query,
                regardless of whether the raw query would have worked fine.
  hyde_gated  : two-stage/selective-compute version of the same idea. First
                embed the raw query (cheap, no LLM call) and check dense's
                own top1-vs-top2 similarity margin. Only when that margin is
                below --margin-gate (dense is unsure) do we pay for an LLM
                call to generate + embed a HyDE rewrite for that query, and
                use ITS embedding instead. When dense is already confident,
                skip HyDE entirely and keep the raw-query result. This mirrors
                the margin-gate idea already validated in hybrid_rerank_eval.py
                and should give most of hyde_always's benefit at a fraction
                of the LLM-call cost, since only the ambiguous fraction of
                queries incur the extra call.

Uses the same repeated-seed methodology as compare_doc_strategies.py /
hybrid_rerank_eval.py: recall@1/@10 mean +/- std across seeds, not a single
fixed sample. Fixed to doc_description corpus text throughout.

Usage:
    python hyde_eval.py site24x7_Dataset.csv site24x7_Admin_API.xlsx --mock

    python hyde_eval.py site24x7_Dataset.csv site24x7_Admin_API.xlsx \
        --base-url http://20.235.183.15:443/openai/v1 --api-key YOUR_KEY \
        --extra-descriptions reports_synthetic_descriptions.csv \
        --seeds 1 2 3 4 5 6 7 8 --n-eval 100

    # skip hyde_always (saves LLM calls) if you already expect the gated
    # variant to be the one you'll actually deploy:
    python hyde_eval.py site24x7_Dataset.csv site24x7_Admin_API.xlsx \
        --base-url ... --api-key ... --skip-hyde-always --margin-gate 0.05
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rag_eval import build_corpus_and_eval, compute_recall, embed_texts, get_embeddings_mock
from compare_doc_strategies import (
    build_text_doc_description,
    load_doc_descriptions,
    load_extra_descriptions,
    make_variant_corpus,
)

HYDE_SYSTEM_PROMPT = (
    "You write short hypothetical API-documentation passages. For each numbered "
    "user question below, write ONE short passage (1-2 sentences, doc-style, not "
    "conversational) describing the kind of API feature/endpoint that would answer "
    "it - similar in style to a product's API reference description. Use likely "
    "domain terminology for a monitoring/reporting SaaS product (e.g. uptime, "
    "downtime, SLA, threshold, report, alert, monitor) where natural. Do NOT invent "
    "a specific path, method, or parameter name - describe the FUNCTIONALITY, not "
    "a fake spec. Respond with ONLY a JSON array of strings in the same order as "
    "the input - no markdown, no code fences, no explanation, just the JSON array."
)


def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def generate_hyde_real(client, model: str, queries: list, batch_size: int = 10) -> list:
    out = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        numbered = "\n".join(f"{j + 1}. {q}" for j, q in enumerate(batch))
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                    {"role": "user", "content": numbered},
                ],
                temperature=0.3,
            )
            content = _strip_code_fences(resp.choices[0].message.content)
            parsed = json.loads(content)
            if not isinstance(parsed, list) or len(parsed) != len(batch):
                raise ValueError(f"expected {len(batch)} items, got {parsed!r}")
            out.extend(str(p) for p in parsed)
        except Exception as e:
            print(f"  batch HyDE generation failed ({e}), falling back to one-by-one for this batch")
            for q in batch:
                try:
                    r = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": (
                                "Write one short (1-2 sentence) hypothetical API "
                                "documentation passage describing the feature that "
                                "would answer this user question. Doc-style, not "
                                "conversational. Respond with ONLY the passage."
                            )},
                            {"role": "user", "content": q},
                        ],
                        temperature=0.3,
                    )
                    out.append(r.choices[0].message.content.strip())
                except Exception as e2:
                    print(f"    failed on '{q[:50]}...': {e2} - keeping original query text")
                    out.append(q)
        print(f"  hyde-generated {min(i + batch_size, len(queries))}/{len(queries)}")
    return out


def generate_hyde_mock(queries: list) -> list:
    """Deterministic fake hypothetical doc for offline pipeline testing only -
    just wraps the query in doc-style boilerplate. NOT a real hypothetical
    doc, just exercises the code path so --mock can smoke-test the pipeline."""
    return [f"Provides information related to: {q.rstrip('?')}." for q in queries]


def embed_dispatch(texts, client, args, cache_dir):
    if args.mock:
        return get_embeddings_mock(texts)
    return embed_texts(client, args.embed_model, texts, cache_dir, not args.no_cache,
                        args.embed_batch_size, request_timeout=args.timeout)


def main():
    parser = argparse.ArgumentParser(description="HyDE (hypothetical document embeddings) vs raw-query retrieval, doc_description corpus")
    parser.add_argument("csv_path")
    parser.add_argument("xlsx_path")
    parser.add_argument("--extra-descriptions", default=None)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-examples", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--embed-model", default="azure:primary/s247-textembedding-3l")
    parser.add_argument("--hyde-model", default="azure:primary/gpt-4.1-mini",
                         help="chat model used to generate the hypothetical doc passage")
    parser.add_argument("--margin-gate", type=float, default=0.05,
                         help="for hyde_gated: only generate+use a HyDE rewrite for a query when "
                              "dense's own top1-vs-top2 cosine similarity margin (on the raw query) "
                              "is BELOW this value, i.e. dense is unsure. Default 0.05.")
    parser.add_argument("--skip-hyde-always", action="store_true",
                         help="skip the hyde_always variant (saves LLM calls) - use this once you've "
                              "already confirmed hyde_always's ceiling and just want hyde_gated numbers")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--save-per-query", action="store_true",
                         help="save per-query rank_of_correct for all variants, to hyde_eval_per_query.csv")
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
    print(f"  margin_gate={args.margin_gate}"
          + ("  (hyde_always SKIPPED)" if args.skip_hyde_always else "") + "\n")

    cache_dir = Path(args.cache_dir) if args.cache_dir else csv_path.parent / ".rag_cache"
    hyde_cache_dir = Path(args.cache_dir) if args.cache_dir else csv_path.parent / ".hyde_text_cache"
    hyde_cache_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.mock:
        if not args.base_url or not args.api_key:
            sys.exit("Provide --base-url and --api-key, or use --mock for an offline test.")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    all_rows = []
    all_per_query = []
    n_gated_calls_total = 0
    n_queries_total = 0

    for seed in args.seeds:
        print(f"\n--- seed {seed} ---")
        base_corpus_df, eval_df = build_corpus_and_eval(df, args.n_eval, args.n_examples, seed)

        true_df = df[df["markedCorrect"] == True].copy()
        true_df["key"] = list(zip(true_df["endpoint"], true_df["method"]))
        eval_queries = set(eval_df["query"])
        enrichment_pool = true_df[~true_df["query"].isin(eval_queries)]

        corpus_df = make_variant_corpus(base_corpus_df, df, doc_map, args.n_examples,
                                         enrichment_pool, build_text_doc_description)

        corpus_vecs = embed_dispatch(corpus_df["text"].tolist(), client, args, cache_dir)

        # --- baseline: raw query ---
        raw_query_vecs = embed_dispatch(eval_df["query"].tolist(), client, args, cache_dir)
        per_query_raw, recall_raw = compute_recall(corpus_df, eval_df, corpus_vecs, raw_query_vecs,
                                                     ks=(1, 10), variant=f"raw_query_seed{seed}")
        per_query_raw["variant"] = "raw_query"
        all_per_query.append(per_query_raw)
        all_rows.append({"variant": "raw_query", "seed": seed,
                          "recall@1": recall_raw[1], "recall@10": recall_raw[10], "n_eval": len(eval_df)})

        # dense top1-vs-top2 margin per query, from the raw-query embeddings,
        # used both to gate hyde_gated and to report how "unsure" the corpus
        # naturally is
        corpus_norm = corpus_vecs / np.clip(np.linalg.norm(corpus_vecs, axis=1, keepdims=True), 1e-9, None)
        raw_norm = raw_query_vecs / np.clip(np.linalg.norm(raw_query_vecs, axis=1, keepdims=True), 1e-9, None)
        sims = raw_norm @ corpus_norm.T
        sims_sorted = -np.sort(-sims, axis=1)
        margins = sims_sorted[:, 0] - sims_sorted[:, 1] if sims.shape[1] > 1 else np.ones(len(eval_df))

        # --- hyde_always: generate + embed a HyDE rewrite for every query ---
        if not args.skip_hyde_always:
            if args.mock:
                hyde_texts_all = generate_hyde_mock(eval_df["query"].tolist())
            else:
                hyde_texts_all = generate_hyde_real(client, args.hyde_model, eval_df["query"].tolist())
            hyde_vecs_all = embed_dispatch(hyde_texts_all, client, args, cache_dir)
            per_query_hyde, recall_hyde = compute_recall(corpus_df, eval_df, corpus_vecs, hyde_vecs_all,
                                                           ks=(1, 10), variant=f"hyde_always_seed{seed}")
            per_query_hyde["variant"] = "hyde_always"
            all_per_query.append(per_query_hyde)
            all_rows.append({"variant": "hyde_always", "seed": seed,
                              "recall@1": recall_hyde[1], "recall@10": recall_hyde[10], "n_eval": len(eval_df)})
        else:
            hyde_texts_all = None
            hyde_vecs_all = None

        # --- hyde_gated: only rewrite+embed queries where dense is unsure ---
        gated_idx = [i for i in range(len(eval_df)) if margins[i] < args.margin_gate]
        n_gated_calls_total += len(gated_idx)
        n_queries_total += len(eval_df)
        print(f"  hyde_gated: {len(gated_idx)}/{len(eval_df)} queries below margin_gate={args.margin_gate} "
              f"(these get the LLM rewrite; the rest keep their raw-query result)")

        gated_query_vecs = raw_query_vecs.copy()
        if gated_idx:
            gated_queries = eval_df["query"].iloc[gated_idx].tolist()
            if hyde_texts_all is not None:
                # reuse hyde_always's generations for the same queries, no double LLM spend
                gated_texts = [hyde_texts_all[i] for i in gated_idx]
                gated_vecs = np.array([hyde_vecs_all[i] for i in gated_idx])
            elif args.mock:
                gated_texts = generate_hyde_mock(gated_queries)
                gated_vecs = embed_dispatch(gated_texts, client, args, cache_dir)
            else:
                gated_texts = generate_hyde_real(client, args.hyde_model, gated_queries)
                gated_vecs = embed_dispatch(gated_texts, client, args, cache_dir)
            for local_i, global_i in enumerate(gated_idx):
                gated_query_vecs[global_i] = gated_vecs[local_i]

        per_query_gated, recall_gated = compute_recall(corpus_df, eval_df, corpus_vecs, gated_query_vecs,
                                                         ks=(1, 10), variant=f"hyde_gated_seed{seed}")
        per_query_gated["variant"] = "hyde_gated"
        all_per_query.append(per_query_gated)
        all_rows.append({"variant": "hyde_gated", "seed": seed,
                          "recall@1": recall_gated[1], "recall@10": recall_gated[10],
                          "n_eval": len(eval_df), "n_gated": len(gated_idx)})

        print(f"  raw_query  : recall@1={recall_raw[1]:.3f}  recall@10={recall_raw[10]:.3f}")
        if not args.skip_hyde_always:
            print(f"  hyde_always: recall@1={recall_hyde[1]:.3f}  recall@10={recall_hyde[10]:.3f}")
        print(f"  hyde_gated : recall@1={recall_gated[1]:.3f}  recall@10={recall_gated[10]:.3f}  "
              f"({len(gated_idx)}/{len(eval_df)} queries got an LLM rewrite)")

    combined = pd.DataFrame(all_rows)
    out_dir = csv_path.parent
    combined.to_csv(out_dir / "hyde_eval_per_seed.csv", index=False)

    summary = combined.groupby("variant").agg(
        recall_1_mean=("recall@1", "mean"), recall_1_std=("recall@1", "std"),
        recall_10_mean=("recall@10", "mean"), recall_10_std=("recall@10", "std"),
    ).round(4)

    print("\n" + "=" * 60)
    print("SUMMARY (mean +/- std across seeds, doc_description corpus)")
    print("=" * 60)
    print(summary.to_string())
    summary.to_csv(out_dir / "hyde_eval_summary.csv")

    if n_queries_total:
        pct_gated = 100 * n_gated_calls_total / n_queries_total
        print(f"\nhyde_gated used the LLM rewrite on {n_gated_calls_total}/{n_queries_total} "
              f"query-seed instances ({pct_gated:.1f}%) - that's the fraction of production "
              f"queries that would actually incur the extra LLM call + embed if you deployed "
              f"hyde_gated instead of hyde_always.")

    print(f"\nSaved -> {out_dir / 'hyde_eval_per_seed.csv'}")
    print(f"Saved -> {out_dir / 'hyde_eval_summary.csv'}")

    if args.save_per_query:
        pq_df = pd.concat(all_per_query, ignore_index=True)
        pq_path = out_dir / "hyde_eval_per_query.csv"
        pq_df.to_csv(pq_path, index=False)
        print(f"Saved -> {pq_path}")


if __name__ == "__main__":
    main()