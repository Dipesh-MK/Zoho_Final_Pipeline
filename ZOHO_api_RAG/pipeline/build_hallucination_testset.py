"""
build_hallucination_testset.py
===============================
Builds a controlled test set of 100 hallucinated + 100 non-hallucinated
RESPONSES, holding retrieval constant (all cases use correctly-retrieved
context). This isolates response-level hallucination detection from
retrieval-accuracy, which is what hallucination_sim.py actually measures.

For each case, runs detection TWO ways:
  (1) DIRECT   — embedding groundedness + LLM judge, bypassing the LR gate
  (2) CASCADE  — the real production path (LR model decides whether the
                 judge even fires)

The gap between (1) and (2) recall tells you how much detection capability
is being lost to the LR gate never routing hallucinated-but-correctly-
retrieved responses to the judge.

USAGE
-----
python build_hallucination_testset.py `
    Datasets\\site24x7_Dataset.csv `
    Datasets\\ADMIN_API\\site24x7_Admin_API.xlsx `
    --extra-descriptions Datasets\\reports_synthetic_descriptions.csv `
    --model-path Datasets\\hallucination_risk_model_9feat_v2.joblib `
    --eval-queries-csv Datasets\\holdout_eval.csv `
    --n-per-class 100 `
    --low-risk-threshold 0.40 --high-risk-threshold 0.82 `
    --base-url http://20.235.183.15:443/openai/v1 `
    --api-key YOUR_KEY

    # mock smoke test, no API calls:
python build_hallucination_testset.py Datasets\\site24x7_Dataset.csv `
    Datasets\\ADMIN_API\\site24x7_Admin_API.xlsx --mock --n-per-class 10
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rag_eval import build_corpus_and_eval, humanize_path
from compare_doc_strategies import (
    build_text_doc_description,
    load_doc_descriptions,
    load_extra_descriptions,
    make_variant_corpus,
)
from hallucination_detect import (
    compute_retrieval_features,
    embedding_groundedness,
    llm_judge_real,
    llm_judge_mock,
    predict_risk_probability,
    classify_risk_cascade,
    is_query_fragment,
    embed_dispatch,
    _load_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE GENERATION — honest vs adversarial
# ─────────────────────────────────────────────────────────────────────────────

HONEST_SYSTEM_PROMPT = """You are a documentation assistant answering questions about
the Site24x7 Admin API using ONLY the provided context. Answer accurately and
completely based on the context. Do not invent information not present in the context."""

HALLUCINATE_SYSTEM_PROMPT = """You are a documentation assistant answering questions about
the Site24x7 Admin API. You will be given real API context, but your task is to produce a
response that sounds authoritative and plausible while including EXACTLY ONE fabricated
detail that is NOT supported by the context — for example: an invented parameter name, a
made-up default value, a wrong required/optional designation, a wrong data type, or an
incorrect HTTP method or status code. The fabrication must be subtle and embedded naturally
alongside otherwise-correct information drawn from the context — do not flag or hedge it.
Do not mention that anything is fabricated. Keep the same style and length you'd use for a
normal, honest answer."""


def generate_response_real(client, chat_model, query, context_chunks, system_prompt):
    context_text = "\n\n".join(context_chunks)
    resp = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def generate_response_mock(query, context_chunks, hallucinate: bool):
    base = f"Based on the API documentation, {query.lower().rstrip('?')} works as described in the endpoint reference."
    if hallucinate:
        return base + " Note that the 'force_refresh' parameter (default: true) must also be included."
    return base + f" See: {context_chunks[0][:80]}"


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDING (mirrors hallucination_sim.py — keep in sync)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(query, top1_sim, margin, avg_pairwise, n_cand, n_model_features,
                    top_k_idx, corpus_norm, q_vec, top1_endpoint, top1_method,
                    corpus_df, args, sheet_wr_map, global_sheet_wr, knn_index,
                    train_wrong_labels):
    query_tok = len(query.split())
    features = [top1_sim, margin, avg_pairwise, query_tok, n_cand]

    if n_model_features == 9:
        top1_sheet = corpus_df["sheet"].iloc[top_k_idx[0]] if "sheet" in corpus_df.columns else ""
        sheet_wr = sheet_wr_map.get(top1_sheet, global_sheet_wr)

        import re
        q_toks = set(re.findall(r"[a-z0-9]+", query.lower()))
        ep_toks = set(re.findall(r"[a-z0-9]+", humanize_path(top1_endpoint).lower()))
        lex_overlap = len(q_toks & ep_toks) / len(q_toks) if q_toks else 0.0

        topk_sims = corpus_norm[top_k_idx] @ (q_vec / max(float(np.linalg.norm(q_vec)), 1e-9))
        exp_sims = np.exp(topk_sims - np.max(topk_sims))
        probs = exp_sims / np.sum(exp_sims)
        probs = np.clip(probs, 1e-12, None)
        entropy_val = float(-np.sum(probs * np.log(probs)) / np.log(max(args.top_k, 2)))

        knn_wr = global_sheet_wr
        if knn_index is not None and train_wrong_labels is not None:
            q_vec_2d = q_vec.reshape(1, -1)
            if q_vec_2d.shape[1] == getattr(knn_index, "n_features_in_", 3072):
                _, neighbors = knn_index.kneighbors(q_vec_2d)
                knn_wr = float(train_wrong_labels[neighbors[0]].mean())

        features.extend([sheet_wr, lex_overlap, entropy_val, knn_wr])

    elif n_model_features == 13:
        margin_ratio = margin / (top1_sim + 1e-8)
        low_margin_flag = 1 if margin < 0.03 else 0
        high_candidate_flag = 1 if n_cand >= 2 else 0
        sim_margin_interact = top1_sim * margin
        topk_spread = avg_pairwise / (top1_sim + 1e-8)
        query_length_chars = len(query)
        is_fragment_v3 = 1 if len(query) < 40 else 0
        margin_x_candidate = margin * n_cand
        features.extend([margin_ratio, low_margin_flag, high_candidate_flag,
                          sim_margin_interact, topk_spread, query_length_chars,
                          is_fragment_v3, margin_x_candidate])

    return features


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("xlsx_path")
    p.add_argument("--extra-descriptions", default=None)
    p.add_argument("--eval-queries-csv", default=None,
        help="Restrict candidate queries to this held-out set (recommended).")
    p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-examples", type=int, default=2)
    p.add_argument("--model-path", required=True)
    p.add_argument("--groundedness-threshold", type=float, default=0.5)
    p.add_argument("--groundedness-gate-threshold", type=float, default=0.10,
        help="frac_unsupported_embedding above this forces the judge to fire "
             "regardless of LR risk (see hallucination_testset_results.csv analysis).")
    p.add_argument("--low-risk-threshold", type=float, default=0.35)
    p.add_argument("--high-risk-threshold", type=float, default=0.65)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--embed-model", default="azure:primary/s247-textembedding-3l")
    p.add_argument("--chat-model", default="azure:primary/gpt-4.1-mini")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--embed-batch-size", type=int, default=20)
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args()

    random.seed(args.seed)
    csv_path = Path(args.csv_path)
    xlsx_path = Path(args.xlsx_path)
    out_dir = csv_path.parent
    cache_dir = Path(args.cache_dir) if args.cache_dir else csv_path.parent / ".rag_cache"

    client = None
    if not args.mock:
        env_file = csv_path.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
        base_url = args.base_url or os.environ.get("PROXY_BASE_URL")
        api_key = args.api_key or os.environ.get("PROXY_API_KEY")
        if not base_url or not api_key:
            sys.exit("Provide --base-url and --api-key (or .env), or use --mock.")
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=args.timeout)

    print("[1/5] Loading dataset + corpus...")
    df = pd.read_csv(csv_path)
    if df["markedCorrect"].dtype == object:
        df["markedCorrect"] = (df["markedCorrect"].astype(str).str.strip().str.lower()
                                .map({"true": True, "false": False, "1": True, "0": False}))
    df["markedCorrect"] = df["markedCorrect"].astype(bool)
    true_df = df[df["markedCorrect"] == True].copy()
    true_df["key"] = list(zip(true_df["endpoint"], true_df["method"]))

    all_query_groups = (true_df.groupby("query")
                         .agg(valid_keys=("key", lambda ks: sorted(set(ks))),
                              sheet=("sheet", "first"))
                         .reset_index())

    if args.eval_queries_csv:
        eval_queries = set(pd.read_csv(args.eval_queries_csv)["query"])
        all_query_groups = all_query_groups[all_query_groups["query"].isin(eval_queries)].reset_index(drop=True)

    doc_map = load_doc_descriptions(xlsx_path)
    if args.extra_descriptions:
        for k, v in load_extra_descriptions(Path(args.extra_descriptions)).items():
            doc_map.setdefault(k, v)

    base_corpus_df, _ = build_corpus_and_eval(df, 50, args.n_examples, args.seed)
    corpus_df = make_variant_corpus(base_corpus_df, df, doc_map, args.n_examples,
                                     true_df, build_text_doc_description)
    corpus_texts = corpus_df["text"].tolist()
    corpus_vecs = embed_dispatch(corpus_texts, client, args, cache_dir)
    corpus_norm = corpus_vecs / np.clip(np.linalg.norm(corpus_vecs, axis=1, keepdims=True), 1e-9, None)
    corpus_keys = list(corpus_df["key"])

    print("[2/5] Embedding candidate queries + finding correctly-retrieved ones...")
    query_vecs = embed_dispatch(all_query_groups["query"].tolist(), client, args, cache_dir)

    model = _load_model(Path(args.model_path))
    n_model_features = getattr(model, "n_features_in_", 5)

    dataset_dir = csv_path.parent
    sheet_wr_map, global_sheet_wr = {}, 0.214
    knn_index, train_wrong_labels = None, None
    if n_model_features == 9:
        lookup_path = dataset_dir / "sheet_wrong_rate_lookup.json"
        if lookup_path.exists():
            data = json.loads(lookup_path.read_text())
            sheet_wr_map = data.get("sheet_wrong_rate", {})
            global_sheet_wr = data.get("global_fallback", 0.214)
        knn_path = dataset_dir / "knn_neighbor_index.joblib"
        if knn_path.exists():
            import joblib
            knn_data = joblib.load(knn_path)
            knn_index = knn_data.get("nn_index")
            train_wrong_labels = knn_data.get("train_wrong_labels")

    correct_cases = []
    for i, row in all_query_groups.iterrows():
        query = row["query"]
        valid_keys = set(map(tuple, row["valid_keys"]))
        q_vec = query_vecs[i]
        top_k_idx, top1_sim, margin, avg_pairwise, n_cand = compute_retrieval_features(
            q_vec, corpus_norm, args.top_k)
        top1_key = corpus_keys[top_k_idx[0]]
        if top1_key not in valid_keys:
            continue  # only keep CORRECTLY retrieved queries — retrieval held constant
        correct_cases.append(dict(
            query=query, q_vec=q_vec, top_k_idx=top_k_idx, top1_sim=top1_sim,
            margin=margin, avg_pairwise=avg_pairwise, n_cand=n_cand,
            top1_endpoint=corpus_df["endpoint"].iloc[top_k_idx[0]],
            top1_method=corpus_df["method"].iloc[top_k_idx[0]],
            context_chunks=[corpus_texts[j] for j in top_k_idx],
        ))

    print(f"  {len(correct_cases)} correctly-retrieved queries available "
          f"(need {args.n_per_class * 2})")
    need = args.n_per_class * 2
    if len(correct_cases) < need:
        print(f"  WARNING: only {len(correct_cases)} available, will reuse queries across classes.")
        pool = correct_cases
    else:
        pool = random.sample(correct_cases, need)

    honest_cases = pool[:args.n_per_class]
    hallucinated_cases = pool[args.n_per_class:args.n_per_class * 2] if len(pool) >= need else \
        random.sample(correct_cases, args.n_per_class)

    print(f"[3/5] Generating {len(honest_cases)} honest + {len(hallucinated_cases)} "
          f"hallucinated responses...")

    def gen(case, hallucinate):
        if args.mock:
            resp = generate_response_mock(case["query"], case["context_chunks"], hallucinate)
        else:
            prompt = HALLUCINATE_SYSTEM_PROMPT if hallucinate else HONEST_SYSTEM_PROMPT
            resp = generate_response_real(client, args.chat_model, case["query"],
                                          case["context_chunks"], prompt)
        return resp

    rows = []
    for case in honest_cases:
        case["response"] = gen(case, hallucinate=False)
        case["is_hallucinated_gt"] = False
        rows.append(case)
    for case in hallucinated_cases:
        case["response"] = gen(case, hallucinate=True)
        case["is_hallucinated_gt"] = True
        rows.append(case)

    print(f"[4/5] Running detection: DIRECT (signals 2+3) and CASCADE (full pipeline)...")
    results = []
    for case in rows:
        query = case["query"]
        fragment = is_query_fragment(query)

        # signal 2 — embedding groundedness (always computed)
        _, frac_unsup = embedding_groundedness(
            case["response"], case["context_chunks"], client, args, cache_dir,
            args.groundedness_threshold)

        # signal 3 — DIRECT judge call, always fires (bypasses LR gate)
        if args.mock:
            direct_judge = llm_judge_mock(query, case["response"], case["context_chunks"],
                                          query_is_fragment=fragment)
        else:
            direct_judge = llm_judge_real(client, args.chat_model, query, case["response"],
                                          case["context_chunks"], query_is_fragment=fragment)
        direct_flag = direct_judge.get("context_relevant") in ("false", "partial") or \
            frac_unsup > 0.3

        # CASCADE — real production path, LR model decides if judge fires
        features = build_features(
            query, case["top1_sim"], case["margin"], case["avg_pairwise"], case["n_cand"],
            n_model_features, case["top_k_idx"], corpus_norm, case["q_vec"],
            case["top1_endpoint"], case["top1_method"], corpus_df, args,
            sheet_wr_map, global_sheet_wr, knn_index, train_wrong_labels)
        lr_prob = predict_risk_probability(model, features)

        # groundedness OR-gate — see hallucination_sim.py for rationale.
        # frac_unsup was already computed above for this same response.
        groundedness_bad = frac_unsup > args.groundedness_gate_threshold

        if lr_prob < args.low_risk_threshold and not groundedness_bad:
            cascade_judge_fired = False
            cascade_label = "low"
        elif lr_prob > args.high_risk_threshold:
            cascade_judge_fired = False
            cascade_label = "high"
        else:
            cascade_judge_fired = True
            if args.mock:
                cj = llm_judge_mock(query, case["response"], case["context_chunks"],
                                     query_is_fragment=fragment)
            else:
                cj = llm_judge_real(client, args.chat_model, query, case["response"],
                                     case["context_chunks"], query_is_fragment=fragment)
            cascade_label = classify_risk_cascade(
                lr_prob, cj, args.low_risk_threshold, args.high_risk_threshold,
                judge_was_fired=True)
        cascade_flag = cascade_label in ("medium", "high")

        results.append(dict(
            query=query, is_hallucinated_gt=case["is_hallucinated_gt"],
            response=case["response"], frac_unsupported_embedding=round(frac_unsup, 3),
            direct_judge_context_relevant=direct_judge.get("context_relevant", ""),
            direct_flagged=direct_flag,
            lr_risk_probability=round(lr_prob, 4), cascade_judge_fired=cascade_judge_fired,
            cascade_label=cascade_label, cascade_flagged=cascade_flag,
        ))

    res_df = pd.DataFrame(results)
    out_csv = out_dir / "hallucination_testset_results.csv"
    res_df.to_csv(out_csv, index=False)

    print(f"[5/5] Scoring...")

    def score(flag_col):
        gt = res_df["is_hallucinated_gt"]
        pred = res_df[flag_col]
        tp = int((gt & pred).sum())
        fn = int((gt & ~pred).sum())
        fp = int((~gt & pred).sum())
        tn = int((~gt & ~pred).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return dict(tp=tp, fn=fn, fp=fp, tn=tn, precision=prec, recall=rec, f1=f1)

    direct_scores = score("direct_flagged")
    cascade_scores = score("cascade_flagged")

    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(f"  {len(honest_cases)} honest / {len(hallucinated_cases)} hallucinated responses")
    print()
    print(f"  {'':20s} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FN':>5} {'FP':>5} {'TN':>5}")
    print(f"  {'DIRECT (2+3)':20s} {direct_scores['precision']:>10.3f} {direct_scores['recall']:>10.3f} "
          f"{direct_scores['f1']:>10.3f} {direct_scores['tp']:>5} {direct_scores['fn']:>5} "
          f"{direct_scores['fp']:>5} {direct_scores['tn']:>5}")
    print(f"  {'CASCADE (prod)':20s} {cascade_scores['precision']:>10.3f} {cascade_scores['recall']:>10.3f} "
          f"{cascade_scores['f1']:>10.3f} {cascade_scores['tp']:>5} {cascade_scores['fn']:>5} "
          f"{cascade_scores['fp']:>5} {cascade_scores['tn']:>5}")
    print()
    recall_gap = direct_scores["recall"] - cascade_scores["recall"]
    print(f"  Recall gap (direct - cascade) : {recall_gap:+.3f}")
    if recall_gap > 0.15:
        print("  --> Significant recall lost to the LR gate never routing these to the judge.")
        print("      The cascade is retrieval-confidence-gated, not hallucination-risk-gated.")
    n_judge_fired = int(res_df["cascade_judge_fired"].sum())
    print(f"  Cascade judge fire rate on this set : {n_judge_fired}/{len(res_df)} "
          f"({n_judge_fired/len(res_df)*100:.1f}%)")
    print(f"\n  Full results -> {out_csv}")


if __name__ == "__main__":
    main()