"""
hallucination_sim.py
====================
Full simulation of the hallucination-detection pipeline over ALL (or --n-sim)
unique queries in the dataset. Produces a comprehensive report suitable for
sharing with a manager.

TWO PHASES
----------
Phase 1  BULK (fast, cheap — runs on every query):
  • Embeds all queries (uses disk cache, so most are instant after first run)
  • Computes retrieval features (top1_sim, margin, avg_pairwise) for each query
  • Runs the trained LogisticRegression → risk probability
  • Compares against markedCorrect ground truth → full TP/FP/FN/TN matrix
  • No response generation, no LLM calls at all in this phase

Phase 2  DEEP (optional, expensive — runs on --n-deep sampled queries):
  • Generates responses via the chat model (parallelised with --workers threads)
  • Computes embedding groundedness (signal 2)
  • Runs the LLM judge for uncertain-band queries only (cascade, signal 3)
  • Full per-sentence explainability for every deep-eval query
  • Results merged into the main CSV with full diagnostics

OUTPUT FILES
------------
  hallucination_sim_results.csv          all queries, phase-1 features + label
  hallucination_sim_deep_results.csv     phase-2 queries, full diagnostics
  hallucination_sim_report.txt           manager-ready report
  hallucination_sim.log                  JSON-lines real-time log (one entry/query)

USAGE
-----
  # Phase 1 only (fast, all queries, no LLM calls):
  python hallucination_sim.py Datasets/site24x7_Dataset.csv \\
      Datasets/ADMIN_API/site24x7_Admin_API.xlsx --mock

  # Full run with Phase 2 deep eval on 100 queries:
  python hallucination_sim.py Datasets/site24x7_Dataset.csv \\
      Datasets/ADMIN_API/site24x7_Admin_API.xlsx \\
      --extra-descriptions Datasets/reports_synthetic_descriptions.csv \\
      --n-deep 100 --workers 4 --save-log
"""

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── existing codebase imports (read-only) ──────────────────────────────────
from rag_eval import build_corpus_and_eval, embed_texts, get_embeddings_mock, humanize_path
from compare_doc_strategies import (
    build_text_doc_description,
    load_doc_descriptions,
    load_extra_descriptions,
    make_variant_corpus,
)

# ── import helpers from hallucination_detect (our own pipeline) ────────────
from hallucination_detect import (
    compute_retrieval_features,
    embedding_groundedness,
    llm_judge_real,
    llm_judge_mock,
    generate_response_real,
    generate_response_mock,
    predict_risk_probability,
    classify_risk_cascade,
    make_risk_reason,
    judge_frac_unsupported,
    _load_model,
    embed_dispatch,
    split_sentences,
    is_query_fragment,
    COMBINED_JUDGE_SYSTEM_PROMPT,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("hallucination_sim")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    # Console handler — INFO level, human-readable
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)
    # File handler — DEBUG level, JSON-lines for machine parsing
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    return logger


def log_query_result(logger: logging.Logger, record: dict):
    """Write one JSON-lines entry per query to the log file."""
    logger.debug(json.dumps(record, ensure_ascii=False, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — BULK RETRIEVAL + LR SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _cascade_labels(lr_prob: float,
                    low_thresh: float,
                    high_thresh: float) -> tuple:
    """
    Single source of truth: derive BOTH cascade_bucket AND phase1_risk_label
    from one if/elif/else so the two columns can never disagree.

    Returns (cascade_bucket, phase1_risk_label):
      lr_prob < low_thresh   -> ("auto_low",  "low")
      lr_prob > high_thresh  -> ("auto_high", "high")
      else (uncertain band)  -> ("uncertain", "medium")

    In Phase 1 the LLM judge never fires, so uncertain-band queries get
    'medium' -- the judge (Phase 2) may later refine them, but Phase 1
    never speculatively labels below the low threshold.
    """
    if lr_prob < low_thresh:
        return "auto_low", "low"
    if lr_prob > high_thresh:
        return "auto_high", "high"
    return "uncertain", "medium"


def run_phase1(unique_queries_df: pd.DataFrame,
               corpus_df: pd.DataFrame,
               corpus_norm: np.ndarray,
               query_vecs: np.ndarray,
               model,
               args,
               logger: logging.Logger) -> pd.DataFrame:
    """
    Evaluate every unique query:
      - retrieval features (top1_sim, margin, avg_pairwise, query_token_count,
        n_candidates_within_margin)
      - additional features depending on what the model expects (9 or 13)
      - LR / GB / RF probability + auto-label (low/medium/high)
      - ground-truth comparison (top1_correct)
      - cascade bucket (auto_low / uncertain / auto_high)

    Returns a DataFrame with one row per query.
    """
    corpus_keys = list(corpus_df["key"])
    rows = []
    t_lr_total = 0.0
    print()

    n_model_features = getattr(model, "n_features_in_", 5)

    # Pre-load lookup resources for 9-feature live pipeline if needed
    sheet_wr_map = {}
    global_sheet_wr = 0.214
    knn_index = None
    train_wrong_labels = None
    ep_to_sheet = {}

    dataset_dir = Path(args.csv_path).parent

    if n_model_features == 9:
        lookup_path = dataset_dir / "sheet_wrong_rate_lookup.json"
        if lookup_path.exists():
            with open(lookup_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                sheet_wr_map = data.get("sheet_wrong_rate", {})
                global_sheet_wr = data.get("global_fallback", 0.214)

        knn_path = dataset_dir / "knn_neighbor_index.joblib"
        if knn_path.exists():
            import joblib
            knn_data = joblib.load(knn_path)
            knn_index = knn_data.get("nn_index")
            train_wrong_labels = knn_data.get("train_wrong_labels")

        # Endpoint path -> sheet mapping
        if "sheet" in corpus_df.columns:
            for _, c_row in corpus_df.iterrows():
                ep_to_sheet[(c_row["endpoint"], c_row["method"])] = c_row.get("sheet", "")

    # Pre-compute KNN neighbor wrong rates in ONE batch matrix operation (7119 queries x 3072 dims)
    knn_wr_array = None
    if (n_model_features == 9 and knn_index is not None and train_wrong_labels is not None and
        query_vecs.shape[1] == getattr(knn_index, "n_features_in_", 3072)):
        print("  Pre-computing KNN neighbor wrong rates for all queries in batch...")
        t_knn0 = time.perf_counter()
        # kneighbors on full matrix (7119, 3072)
        _, all_neighbors = knn_index.kneighbors(query_vecs)
        knn_wr_array = train_wrong_labels[all_neighbors].mean(axis=1)
        print(f"  Batch KNN completed in {time.perf_counter() - t_knn0:.2f}s")

    for i, row in unique_queries_df.iterrows():
        query      = row["query"]
        valid_keys = set(map(tuple, row["valid_keys"]))

        t0 = time.perf_counter()
        q_vec = query_vecs[i]
        top_k_idx, top1_sim, margin, avg_pairwise, n_cand = compute_retrieval_features(
            q_vec, corpus_norm, args.top_k)

        top1_key      = corpus_keys[top_k_idx[0]]
        top1_endpoint = corpus_df["endpoint"].iloc[top_k_idx[0]]
        top1_method   = corpus_df["method"].iloc[top_k_idx[0]]
        top1_correct  = top1_key in valid_keys

        query_tok = len(query.split())
        fragment  = is_query_fragment(query)   # still used for the output `is_fragment` column
        features  = [top1_sim, margin, avg_pairwise, query_tok, n_cand]

        if n_model_features == 9:
            # 6. sheet_wrong_rate
            top1_sheet = ep_to_sheet.get((top1_endpoint, top1_method), row.get("sheet", ""))
            sheet_wr = sheet_wr_map.get(top1_sheet, global_sheet_wr)

            # 7. query_endpoint_token_overlap
            q_toks = set(re.findall(r"[a-z0-9]+", query.lower()))
            ep_toks = set(re.findall(r"[a-z0-9]+", humanize_path(top1_endpoint).lower()))
            lex_overlap = len(q_toks & ep_toks) / len(q_toks) if q_toks else 0.0

            # 8. topk_similarity_entropy
            topk_sims = corpus_norm[top_k_idx] @ (q_vec / max(float(np.linalg.norm(q_vec)), 1e-9))
            exp_sims = np.exp(topk_sims - np.max(topk_sims))
            probs = exp_sims / np.sum(exp_sims)
            probs = np.clip(probs, 1e-12, None)
            entropy_val = float(-np.sum(probs * np.log(probs)) / np.log(max(args.top_k, 2)))

            # 9. knn_neighbor_wrong_rate
            if knn_wr_array is not None:
                knn_wr = float(knn_wr_array[i])
            else:
                knn_wr = global_sheet_wr

            features.extend([sheet_wr, lex_overlap, entropy_val, knn_wr])

        elif n_model_features == 13:
            # v3 model: 5 base + 8 engineered features ONLY (no sheet/lex/entropy/knn)
            # Must exactly match improve_model_v3.py's engineer_strong_features().
            margin_ratio        = margin / (top1_sim + 1e-8)
            low_margin_flag     = 1 if margin < 0.03 else 0
            high_candidate_flag = 1 if n_cand >= 2 else 0
            sim_margin_interact = top1_sim * margin
            topk_spread         = avg_pairwise / (top1_sim + 1e-8)
            query_length_chars  = len(query)                    # matches df["query"].str.len()
            is_fragment_v3      = 1 if len(query) < 40 else 0   # matches str.len() < 40
            margin_x_candidate  = margin * n_cand

            features.extend([margin_ratio, low_margin_flag, high_candidate_flag,
                              sim_margin_interact, topk_spread, query_length_chars,
                              is_fragment_v3, margin_x_candidate])

        lr_prob   = predict_risk_probability(model, features)
        t_lr_total += time.perf_counter() - t0

        # ── single source of truth: both columns from one call ────────────
        cascade_bucket, final_risk = _cascade_labels(
            lr_prob, args.low_risk_threshold, args.high_risk_threshold)

        risk_reason = make_risk_reason(
            lr_prob, top1_sim, margin, avg_pairwise,
            judge_fired=False, judge_result=None,
            final_risk=final_risk,
            low_thresh=args.low_risk_threshold,
            high_thresh=args.high_risk_threshold,
        )

        rec = {
            "query":                       query,
            "valid_endpoints":             "; ".join(f"{m} {e}" for e, m in valid_keys),
            "n_valid_answers":             len(valid_keys),
            "top1_endpoint":               top1_endpoint,
            "top1_method":                 top1_method,
            "top1_correct":                top1_correct,
            "top1_sim":                    round(top1_sim, 4),
            "margin":                      round(margin, 4),
            "avg_pairwise_topk_sim":       round(avg_pairwise, 4),
            "query_token_count":           query_tok,
            "n_candidates_within_margin":  n_cand,
            "is_fragment":                 fragment,
            "lr_risk_probability":         round(lr_prob, 4),
            "cascade_bucket":              cascade_bucket,
            "phase1_risk_label":           final_risk,
            "risk_reason":                 risk_reason,
            "sheet":                       row.get("sheet", ""),
        }
        rows.append(rec)

        # log every query
        log_query_result(logger, {
            "phase": 1, "query_idx": i,
            "query": query[:80], "top1_correct": top1_correct,
            "top1_sim": round(top1_sim, 4), "margin": round(margin, 4),
            "query_token_count": query_tok, "is_fragment": fragment,
            "lr_prob": round(lr_prob, 4), "cascade_bucket": cascade_bucket,
            "risk_label": final_risk,
        })

    print(f"  Phase 1 done: {len(rows)} queries  "
          f"(avg LR inference {t_lr_total/max(len(rows),1)*1000:.2f}ms/query)")

    p1 = pd.DataFrame(rows)

    # ── SANITY CHECK: cascade_bucket <-> phase1_risk_label must be 1-to-1 ──
    # If _cascade_labels() is the single source of truth this can never fail,
    # but we keep the guard as a regression detector for future code changes.
    _bucket_to_label = {"auto_low": "low", "auto_high": "high", "uncertain": "medium"}
    _label_to_bucket = {v: k for k, v in _bucket_to_label.items()}
    _mismatches = (
        p1.apply(lambda r: _bucket_to_label.get(r["cascade_bucket"]) != r["phase1_risk_label"],
                 axis=1).sum()
    )
    if _mismatches > 0:
        print(
            f"\nWARNING: risk label / cascade bucket mismatch detected - "
            f"{_mismatches} queries disagree between the two labeling paths.\n"
            f"  cascade_bucket counts : "
            f"{p1['cascade_bucket'].value_counts().to_dict()}\n"
            f"  phase1_risk_label counts: "
            f"{p1['phase1_risk_label'].value_counts().to_dict()}\n"
            "  This is a bug - precision/recall numbers in the report will be wrong."
        )
    else:
        # Confirm counts match (cheap, always runs)
        for bucket, label in _bucket_to_label.items():
            n_b = int((p1["cascade_bucket"] == bucket).sum())
            n_l = int((p1["phase1_risk_label"] == label).sum())
            assert n_b == n_l, (
                f"BUG: cascade_bucket=='{bucket}' has {n_b} rows but "
                f"phase1_risk_label=='{label}' has {n_l} rows"
            )

    return p1


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — DEEP EVAL (response generation + groundedness + LLM judge)
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_deep(query: str, query_vec: np.ndarray,
                      corpus_df: pd.DataFrame,
                      corpus_texts: list, corpus_norm: np.ndarray,
                      model, client, args, cache_dir: Path,
                      valid_keys: set) -> dict:
    """
    Full pipeline for a single query (called in a thread-pool worker).
    Returns a dict of all diagnostics including sentence-level verdicts.
    """
    corpus_keys = list(corpus_df["key"])
    t_start = time.perf_counter()

    # retrieval features (5 features now)
    top_k_idx, top1_sim, margin, avg_pairwise, n_cand = compute_retrieval_features(
        query_vec, corpus_norm, args.top_k)
    context_chunks = [corpus_texts[j] for j in top_k_idx]
    top1_key       = corpus_keys[top_k_idx[0]]
    top1_correct   = top1_key in valid_keys
    fragment       = is_query_fragment(query)

    t_retrieval = time.perf_counter() - t_start

    # response generation
    t_gen = time.perf_counter()
    if args.mock:
        response = generate_response_mock(query, context_chunks)
    else:
        response = generate_response_real(client, args.chat_model, query, context_chunks)
    t_gen = time.perf_counter() - t_gen

    # signal 2: embedding groundedness
    t_grnd = time.perf_counter()
    _, frac_unsup_emb = embedding_groundedness(
        response, context_chunks, client, args, cache_dir, args.groundedness_threshold)
    t_grnd = time.perf_counter() - t_grnd

    # LR / GB / RF probability + cascade (5, 9, or 13 features)
    query_tok = len(query.split())
    features  = [top1_sim, margin, avg_pairwise, query_tok, n_cand]

    n_model_features = getattr(model, "n_features_in_", 5)
    if n_model_features == 9:
        top1_endpoint = corpus_df["endpoint"].iloc[top_k_idx[0]]
        top1_method   = corpus_df["method"].iloc[top_k_idx[0]]

        dataset_dir = Path(args.csv_path).parent
        # 6. sheet_wrong_rate
        lookup_path = dataset_dir / "sheet_wrong_rate_lookup.json"
        sheet_wr_map = {}
        global_sheet_wr = 0.214
        if lookup_path.exists():
            with open(lookup_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                sheet_wr_map = data.get("sheet_wrong_rate", {})
                global_sheet_wr = data.get("global_fallback", 0.214)

        top1_sheet = corpus_df["sheet"].iloc[top_k_idx[0]] if "sheet" in corpus_df.columns else ""
        sheet_wr = sheet_wr_map.get(top1_sheet, global_sheet_wr)

        # 7. query_endpoint_token_overlap
        q_toks = set(re.findall(r"[a-z0-9]+", query.lower()))
        ep_toks = set(re.findall(r"[a-z0-9]+", humanize_path(top1_endpoint).lower()))
        lex_overlap = len(q_toks & ep_toks) / len(q_toks) if q_toks else 0.0

        # 8. topk_similarity_entropy
        topk_sims = corpus_norm[top_k_idx] @ (query_vec / max(float(np.linalg.norm(query_vec)), 1e-9))
        exp_sims = np.exp(topk_sims - np.max(topk_sims))
        probs = exp_sims / np.sum(exp_sims)
        probs = np.clip(probs, 1e-12, None)
        entropy_val = float(-np.sum(probs * np.log(probs)) / np.log(max(args.top_k, 2)))

        # 9. knn_neighbor_wrong_rate
        knn_path = dataset_dir / "knn_neighbor_index.joblib"
        knn_wr = global_sheet_wr
        if knn_path.exists():
            import joblib
            knn_data = joblib.load(knn_path)
            knn_index = knn_data.get("nn_index")
            train_wrong_labels = knn_data.get("train_wrong_labels")
            if (knn_index is not None and train_wrong_labels is not None and
                query_vec.shape[0] == getattr(knn_index, "n_features_in_", 3072)):
                q_vec_2d = query_vec.reshape(1, -1)
                _, neighbors = knn_index.kneighbors(q_vec_2d)
                knn_wr = float(train_wrong_labels[neighbors[0]].mean())

        features.extend([sheet_wr, lex_overlap, entropy_val, knn_wr])

    elif n_model_features == 13:
        # v3 model: 5 base + 8 engineered features ONLY (no sheet/lex/entropy/knn)
        # Must exactly match improve_model_v3.py's engineer_strong_features(),
        # and must exactly match the branch in run_phase1() above.
        margin_ratio        = margin / (top1_sim + 1e-8)
        low_margin_flag     = 1 if margin < 0.03 else 0
        high_candidate_flag = 1 if n_cand >= 2 else 0
        sim_margin_interact = top1_sim * margin
        topk_spread         = avg_pairwise / (top1_sim + 1e-8)
        query_length_chars  = len(query)                    # matches df["query"].str.len()
        is_fragment_v3      = 1 if len(query) < 40 else 0   # matches str.len() < 40
        margin_x_candidate  = margin * n_cand

        features.extend([margin_ratio, low_margin_flag, high_candidate_flag,
                          sim_margin_interact, topk_spread, query_length_chars,
                          is_fragment_v3, margin_x_candidate])

    lr_prob   = predict_risk_probability(model, features)

    judge_result = None
    judge_fired  = False
    t_judge      = 0.0

    # ── groundedness OR-gate ────────────────────────────────────────────
    # LR alone only looks at retrieval features, so a correctly-retrieved
    # query with a fabricated response would otherwise auto-pass as "low
    # risk" without the judge ever seeing it. frac_unsup_emb is cheap
    # (embedding-only) and already computed above, so use it to force the
    # judge to fire even when LR is confident retrieval was fine.
    groundedness_bad = frac_unsup_emb > args.groundedness_gate_threshold

    if lr_prob < args.low_risk_threshold and not groundedness_bad:
        cascade_bucket = "auto_low"
        final_risk     = "low"
    elif lr_prob > args.high_risk_threshold:
        cascade_bucket = "auto_high"
        final_risk     = "high"
    else:
        cascade_bucket = "uncertain"
        judge_fired    = True
        tj = time.perf_counter()
        if args.mock:
            judge_result = llm_judge_mock(query, response, context_chunks,
                                          query_is_fragment=fragment)
        else:
            judge_result = llm_judge_real(
                client, args.chat_model, query, response, context_chunks,
                query_is_fragment=fragment)
        t_judge = time.perf_counter() - tj
        final_risk = classify_risk_cascade(
            lr_prob, judge_result,
            args.low_risk_threshold, args.high_risk_threshold,
            judge_was_fired=True)

    # explainability
    risk_reason = make_risk_reason(
        lr_prob, top1_sim, margin, avg_pairwise,
        judge_fired, judge_result, final_risk,
        args.low_risk_threshold, args.high_risk_threshold)

    # context_relevant + sentence verdicts
    if judge_result:
        context_relevant = judge_result.get("context_relevant", "")
        j_frac           = judge_frac_unsupported(
            judge_result.get("sentence_verdicts", []))
        verdicts_json    = json.dumps(judge_result.get("sentence_verdicts", []))
    else:
        context_relevant = ("auto_likely_true"  if final_risk == "low"
                            else "auto_likely_false" if top1_sim < 0.45
                            else "auto_likely_partial")
        j_frac        = None
        verdicts_json = ""

    t_total = time.perf_counter() - t_start

    return {
        "query":                      query,
        "response":                   response,
        "top1_endpoint":              corpus_df["endpoint"].iloc[top_k_idx[0]],
        "top1_method":                corpus_df["method"].iloc[top_k_idx[0]],
        "top1_correct":               top1_correct,
        "valid_endpoints":            "; ".join(f"{m} {e}" for e, m in valid_keys),
        "top1_sim":                   round(top1_sim, 4),
        "margin":                     round(margin, 4),
        "avg_pairwise_topk_sim":      round(avg_pairwise, 4),
        "frac_unsupported_embedding": round(frac_unsup_emb, 3),
        "lr_risk_probability":        round(lr_prob, 4),
        "cascade_bucket":             cascade_bucket,
        "llm_judge_fired":            judge_fired,
        "context_relevant":           context_relevant,
        "judge_frac_unsupported":     round(j_frac, 3) if j_frac is not None else "",
        "final_risk_label":           final_risk,
        "risk_reason":                risk_reason,
        "sentence_verdicts_json":     verdicts_json,
        # latency breakdown (ms)
        "latency_retrieval_ms":       round(t_retrieval * 1000, 1),
        "latency_generation_ms":      round(t_gen * 1000, 1),
        "latency_groundedness_ms":    round(t_grnd * 1000, 1),
        "latency_judge_ms":           round(t_judge * 1000, 1),
        "latency_total_ms":           round(t_total * 1000, 1),
    }


def run_phase2(deep_queries_df: pd.DataFrame,
               deep_query_vecs: np.ndarray,
               corpus_df: pd.DataFrame,
               corpus_norm: np.ndarray,
               model, client, args, cache_dir: Path,
               logger: logging.Logger) -> pd.DataFrame:
    """
    Run the full pipeline (response generation + groundedness + gated LLM judge)
    on the Phase 2 subset, using a ThreadPoolExecutor for parallelism.
    """
    corpus_texts = corpus_df["text"].tolist()
    n            = len(deep_queries_df)
    results      = [None] * n

    print(f"\n  Phase 2: processing {n} queries with {args.workers} worker(s)...")

    def _worker(idx):
        row        = deep_queries_df.iloc[idx]
        query      = row["query"]
        valid_keys = set(map(tuple, row["valid_keys"]))
        q_vec      = deep_query_vecs[idx]
        result     = _process_one_deep(
            query, q_vec, corpus_df, corpus_texts, corpus_norm,
            model, client, args, cache_dir, valid_keys)
        return idx, result

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, i): i for i in range(n)}
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
                completed += 1
                risk   = result["final_risk_label"]
                gt_tag = "[GT:OK]" if result["top1_correct"] else "[GT:WRONG]"
                judge_tag = "[JUDGE]" if result["llm_judge_fired"] else "       "
                print(f"    {judge_tag} [{risk.upper():6s}] "
                      f"p={result['lr_risk_probability']:.2f} "
                      f"top1={result['top1_sim']:.3f} "
                      f"t={result['latency_total_ms']:.0f}ms  "
                      f"{gt_tag}  \"{result['query'][:50]}\"")
                log_query_result(logger, {"phase": 2, "query_idx": idx, **{
                    k: result[k] for k in [
                        "query", "top1_correct", "top1_sim", "margin",
                        "lr_risk_probability", "final_risk_label",
                        "llm_judge_fired", "context_relevant",
                        "latency_total_ms", "latency_judge_ms",
                    ]
                }})
            except Exception as e:
                idx = futures[future]
                print(f"    ERROR on query {idx}: {e}")
                logger.warning(json.dumps({"phase": 2, "query_idx": idx, "error": str(e)}))

    # filter None (failed) results
    valid_results = [r for r in results if r is not None]
    print(f"  Phase 2 done: {len(valid_results)}/{n} queries processed")
    return pd.DataFrame(valid_results)


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX + METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_confusion(df: pd.DataFrame, risk_col: str) -> dict:
    """
    Ground truth POSITIVE = top1 retrieval was WRONG (risky).
    Predicted  POSITIVE = risk_label in ('medium', 'high').

    Returns dict with TP, FP, FN, TN, precision, recall, f1, accuracy,
    and the separate count for 'high'-only predictions.
    """
    gt_risky    = df["top1_correct"] == False
    pred_risky  = df[risk_col].isin(["medium", "high"])
    pred_high   = df[risk_col] == "high"

    TP  = int((gt_risky  &  pred_risky).sum())
    FN  = int((gt_risky  & ~pred_risky).sum())
    FP  = int((~gt_risky &  pred_risky).sum())
    TN  = int((~gt_risky & ~pred_risky).sum())
    TP_strict = int((gt_risky & pred_high).sum())   # only 'high' counts
    FP_strict = int((~gt_risky & pred_high).sum())

    n   = TP + FP + FN + TN
    prec    = TP / max(TP + FP, 1)
    rec     = TP / max(TP + FN, 1)
    f1      = 2 * prec * rec / max(prec + rec, 1e-9)
    acc     = (TP + TN) / max(n, 1)
    fpr     = FP / max(FP + TN, 1)   # false positive rate
    fnr     = FN / max(TP + FN, 1)   # miss rate

    return dict(
        TP=TP, FP=FP, FN=FN, TN=TN,
        TP_strict=TP_strict, FP_strict=FP_strict,
        n=n, precision=prec, recall=rec, f1=f1,
        accuracy=acc, fpr=fpr, fnr=fnr,
        n_gt_risky=int(gt_risky.sum()),
        n_gt_safe=int((~gt_risky).sum()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _pct(n, total):
    return f"{n/max(total,1)*100:.1f}%"


def _bar(val: float, width: int = 30) -> str:
    """Simple ASCII bar chart."""
    filled = int(round(val * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {val*100:.1f}%"


def build_probability_histogram(p1_df: pd.DataFrame, bins: int = 20) -> list:
    """
    Build a dual-column text histogram of lr_risk_probability
    split by ground truth (correct top1 vs wrong top1).

    Each row covers a 0.05-wide probability bin and shows:
      - count and bar for gt_SAFE  (correct top1 - should be low prob)
      - count and bar for gt_RISKY (wrong top1  - should be high prob)

    The overlap region between the two distributions is where the LR
    model is uncertain. Use this to tune --low-risk-threshold and
    --high-risk-threshold to tighten the uncertain band.
    """
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    safe_df  = p1_df[p1_df["top1_correct"] == True]["lr_risk_probability"]
    risky_df = p1_df[p1_df["top1_correct"] == False]["lr_risk_probability"]

    safe_counts  = np.histogram(safe_df,  bins=bin_edges)[0]
    risky_counts = np.histogram(risky_df, bins=bin_edges)[0]
    max_count    = max(max(safe_counts), max(risky_counts), 1)
    bar_w        = 25

    lines = [
        "  Probability Histogram (lr_risk_probability, 0.05 bins)",
        "  Bin         | GT-SAFE (correct top1)          | GT-RISKY (wrong top1)",
        "  " + "-" * 72,
    ]
    for i in range(bins):
        lo, hi   = bin_edges[i], bin_edges[i + 1]
        sc, rc   = int(safe_counts[i]), int(risky_counts[i])
        s_bar    = "#" * int(sc / max_count * bar_w)
        r_bar    = "#" * int(rc / max_count * bar_w)
        lines.append(
            f"  {lo:.2f}-{hi:.2f}  |"
            f" {s_bar:<{bar_w}} {sc:4d}  |"
            f" {r_bar:<{bar_w}} {rc:4d}"
        )
    lines.append("")
    lines.append(
        "  INTERPRETATION: The overlap zone (where both bars are non-zero) is\n"
        "  your uncertain band. Set --low-risk-threshold to just ABOVE where\n"
        "  GT-SAFE peaks drop off, and --high-risk-threshold to just BELOW where\n"
        "  GT-RISKY bars become dominant. A tighter, well-placed band reduces LLM\n"
        "  judge cost while keeping recall high."
    )
    lines.append("")
    return lines


def build_report(p1_df: pd.DataFrame,
                 p2_df,
                 cm: dict,
                 timings: dict,
                 args,
                 run_ts: str,
                 csv_path: Path) -> list:
    """
    Build a comprehensive manager-ready report as a list of strings.
    """
    sep  = "=" * 72
    sep2 = "-" * 72
    L    = []  # lines

    def add(*lines):
        L.extend(lines)

    n_total = cm["n"]
    n_risky = cm["n_gt_risky"]
    n_safe  = cm["n_gt_safe"]

    # ── HEADER ────────────────────────────────────────────────────────────────
    add(
        sep,
        "  HALLUCINATION DETECTION SIMULATION REPORT",
        f"  Site24x7 RAG Pipeline  |  {run_ts}",
        sep,
        f"  Dataset : {csv_path}",
        f"  Queries evaluated (Phase 1 bulk) : {n_total:,}",
        f"  Phase 2 deep eval queries        : "
        f"{len(p2_df) if p2_df is not None else 0:,}",
        f"  LR model thresholds              : "
        f"low < {args.low_risk_threshold}  |  high > {args.high_risk_threshold}",
        "",
    )

    # ── EXECUTIVE SUMMARY ─────────────────────────────────────────────────────
    add(
        sep, "  1. EXECUTIVE SUMMARY", sep,
        "",
        f"  Out of {n_total:,} queries evaluated against the Site24x7 API corpus:",
        "",
        f"  Retrieval accuracy (top1 correct)  : "
        f"{n_safe:,}/{n_total:,}  ({_pct(n_safe, n_total)})",
        f"  Wrong retrievals (hallucination risk) : "
        f"{n_risky:,}/{n_total:,}  ({_pct(n_risky, n_total)})",
        "",
        f"  Of the {n_risky:,} queries with wrong top1 retrieval:",
        f"    Correctly flagged (TP)  : "
        f"{cm['TP']:,}  ({_pct(cm['TP'], n_risky)}) - caught by the pipeline",
        f"    Missed / not flagged (FN): "
        f"{cm['FN']:,}  ({_pct(cm['FN'], n_risky)}) - slipped through as low risk",
        "",
        f"  Of the {n_safe:,} queries with correct top1 retrieval:",
        f"    Correctly passed (TN)   : "
        f"{cm['TN']:,}  ({_pct(cm['TN'], n_safe)}) - labelled low risk correctly",
        f"    False alarms (FP)       : "
        f"{cm['FP']:,}  ({_pct(cm['FP'], n_safe)}) - incorrectly flagged as risky",
        "",
        f"  Detection Precision : {cm['precision']:.3f}  "
        f"(of flagged queries, {cm['precision']*100:.1f}% were genuinely risky)",
        f"  Detection Recall    : {cm['recall']:.3f}  "
        f"(of risky queries, {cm['recall']*100:.1f}% were caught)",
        f"  F1 Score            : {cm['f1']:.3f}",
        f"  Overall Accuracy    : {cm['accuracy']:.3f}",
        "",
    )

    # ── RETRIEVAL QUALITY ─────────────────────────────────────────────────────
    add(sep, "  2. RETRIEVAL QUALITY", sep, "")

    if "sheet" in p1_df.columns and p1_df["sheet"].notna().any():
        sheet_stats = (
            p1_df.groupby("sheet")["top1_correct"]
            .agg(["count", "sum"])
            .rename(columns={"count": "queries", "sum": "correct"})
        )
        sheet_stats["accuracy"] = sheet_stats["correct"] / sheet_stats["queries"]
        sheet_stats = sheet_stats.sort_values("accuracy")
        add("  Recall@1 by sheet (worst first):", "")
        for sheet, r in sheet_stats.iterrows():
            bar = _bar(r["accuracy"], 25)
            add(f"    {sheet:30s}  {r['correct']:4.0f}/{r['queries']:4.0f}  {bar}")
        add("")

    # overall top1_sim distribution
    add(
        "  top1_sim distribution (all queries):",
        f"    min={p1_df['top1_sim'].min():.4f}  "
        f"p25={p1_df['top1_sim'].quantile(.25):.4f}  "
        f"p50={p1_df['top1_sim'].median():.4f}  "
        f"p75={p1_df['top1_sim'].quantile(.75):.4f}  "
        f"max={p1_df['top1_sim'].max():.4f}",
        "",
        "  margin distribution (top1 - top2 similarity gap):",
        f"    min={p1_df['margin'].min():.4f}  "
        f"p25={p1_df['margin'].quantile(.25):.4f}  "
        f"p50={p1_df['margin'].median():.4f}  "
        f"p75={p1_df['margin'].quantile(.75):.4f}  "
        f"max={p1_df['margin'].max():.4f}",
        "",
    )

    # signal validation: do wrong retrievals actually have lower sim/margin?
    correct_q = p1_df[p1_df["top1_correct"] == True]
    wrong_q   = p1_df[p1_df["top1_correct"] == False]
    add(
        "  Signal validation - do wrong retrievals have worse features?",
        f"  {'Feature':<22}  {'Correct top1 (mean)':>22}  {'Wrong top1 (mean)':>20}",
        f"  {'-'*66}",
    )
    for feat in ["top1_sim", "margin", "avg_pairwise_topk_sim",
                 "query_token_count", "n_candidates_within_margin",
                 "lr_risk_probability"]:
        if feat not in p1_df.columns:
            continue
        c_mean = correct_q[feat].mean() if len(correct_q) else float("nan")
        w_mean = wrong_q[feat].mean()   if len(wrong_q)   else float("nan")
        direction = "< expected" if w_mean < c_mean else "> unexpected"
        add(f"  {feat:<30}  {c_mean:>16.4f}  {w_mean:>16.4f}  [{direction}]")
    add("", "  (Correct retrievals should have higher top1_sim and margin.", "")

    # ── CONFUSION MATRIX ──────────────────────────────────────────────────────
    add(sep, "  3. HALLUCINATION DETECTION PERFORMANCE", sep, "")
    add(
        "  CONFUSION MATRIX  (Predicted: medium+high = detected risk)",
        "",
        "                          Predicted SAFE   Predicted RISKY",
        f"  Actual SAFE  (top1 ok)    {cm['TN']:6d} (TN)      {cm['FP']:6d} (FP)",
        f"  Actual RISKY (top1 wrong) {cm['FN']:6d} (FN)      {cm['TP']:6d} (TP)",
        "",
        "  KEY METRICS:",
        f"    Precision           : {cm['precision']:.4f}  "
        f"({cm['precision']*100:.1f}% of flagged queries were genuinely risky)",
        f"    Recall              : {cm['recall']:.4f}  "
        f"({cm['recall']*100:.1f}% of risky queries were detected)",
        f"    F1 Score            : {cm['f1']:.4f}",
        f"    Overall Accuracy    : {cm['accuracy']:.4f}",
        f"    Miss Rate (FNR)     : {cm['fnr']:.4f}  "
        f"({cm['fnr']*100:.1f}% of risky queries slipped through)",
        f"    False Alarm Rate    : {cm['fpr']:.4f}  "
        f"({cm['fpr']*100:.1f}% of safe queries were over-flagged)",
        "",
        f"  STRICT (only 'high' label counts as flagged):",
        f"    TP (high, gt wrong) : {cm['TP_strict']:,}",
        f"    FP (high, gt ok)    : {cm['FP_strict']:,}",
        "",
    )

    # ── CASCADE EFFICIENCY ────────────────────────────────────────────────────
    n_auto_low  = int((p1_df["cascade_bucket"] == "auto_low").sum())
    n_uncertain = int((p1_df["cascade_bucket"] == "uncertain").sum())
    n_auto_high = int((p1_df["cascade_bucket"] == "auto_high").sum())

    add(
        sep, "  4. CASCADE / LLM COST EFFICIENCY", sep, "",
        "  In production, the LLM judge only fires for 'uncertain' band queries.",
        f"  This simulation shows the cascade distribution over {n_total:,} queries:",
        "",
        f"    Auto-low  (p < {args.low_risk_threshold}, judge skipped) : "
        f"{n_auto_low:6,}  {_bar(n_auto_low/n_total, 25)}",
        f"    Uncertain (judge fires)          : "
        f"{n_uncertain:6,}  {_bar(n_uncertain/n_total, 25)}",
        f"    Auto-high (p > {args.high_risk_threshold}, judge skipped) : "
        f"{n_auto_high:6,}  {_bar(n_auto_high/n_total, 25)}",
        "",
        f"  LLM judge trigger rate   : {_pct(n_uncertain, n_total)}",
        f"  LLM calls saved vs always-judge : "
        f"{n_total - n_uncertain:,}/{n_total:,}  ({_pct(n_total - n_uncertain, n_total)})",
        "",
    )

    # ── LATENCY ───────────────────────────────────────────────────────────────
    add(sep, "  5. LATENCY ANALYSIS", sep, "")

    t_corpus = timings.get("corpus_embedding", 0)
    t_qemb   = timings.get("query_embedding",  0)
    t_p1     = timings.get("phase1",           0)
    t_p2     = timings.get("phase2",           0)
    t_total  = timings.get("total",            0)

    add(
        "  Phase 1 (bulk — all queries):",
        f"    Corpus embedding       : {t_corpus:7.2f}s",
        f"    Query embedding        : {t_qemb:7.2f}s",
        f"    LR inference + scoring : {t_p1:7.2f}s",
        f"    Total Phase 1          : {t_corpus+t_qemb+t_p1:7.2f}s",
        f"    Avg per query          : "
        f"{(t_corpus+t_qemb+t_p1)/max(n_total,1)*1000:.1f}ms",
        "",
    )

    if p2_df is not None and len(p2_df) > 0:
        n_p2 = len(p2_df)
        lat  = p2_df["latency_total_ms"].dropna()
        j_lat = p2_df[p2_df["llm_judge_fired"] == True]["latency_judge_ms"].dropna()
        g_lat = p2_df["latency_generation_ms"].dropna()
        n_judges = int(p2_df["llm_judge_fired"].sum())
        add(
            f"  Phase 2 (deep — {n_p2} queries, {args.workers} parallel worker(s)):",
            f"    Total Phase 2 wall clock   : {t_p2:7.2f}s",
            f"    Avg end-to-end / query     : {lat.mean():.0f}ms",
            f"    p50 / p95 / p99            : "
            f"{lat.quantile(.50):.0f}ms / {lat.quantile(.95):.0f}ms / {lat.quantile(.99):.0f}ms",
            f"    Avg response generation    : {g_lat.mean():.0f}ms",
            f"    LLM judge calls            : {n_judges}/{n_p2}  ({_pct(n_judges, n_p2)})",
            f"    Avg judge latency          : "
            f"{j_lat.mean():.0f}ms" if len(j_lat) > 0 else "    Avg judge latency          : N/A",
            f"    Savings (skipped judges)   : "
            f"{n_p2-n_judges}/{n_p2}  ({_pct(n_p2-n_judges, n_p2)})",
            "",
        )

        # ── PHASE 2 BEFORE vs AFTER LLM JUDGE COMPARISON ──────────────────────
        add(
            sep2,
            f"  PHASE 2 DEEP SAMPLE ({n_p2} QUERIES): BEFORE vs AFTER LLM JUDGE COMPARISON",
            sep2,
            "  (Evaluates the exact impact of the LLM judge on the sampled subset)",
            "",
        )
        # Match each p2 query with its p1 row to get Phase 1 pre-judge risk label
        merged_p2 = p2_df.merge(
            p1_df[["query", "phase1_risk_label", "cascade_bucket"]],
            on="query",
            how="left",
            suffixes=("", "_p1")
        )
        p1_labels = merged_p2["phase1_risk_label"]
        p2_labels = merged_p2["final_risk_label"]
        gt_wrong  = merged_p2["top1_correct"] == False

        # Phase 1 pre-judge metrics on this subset
        p1_pred_risky = p1_labels.isin(["medium", "high"])
        p1_tp = int((gt_wrong & p1_pred_risky).sum())
        p1_fp = int((~gt_wrong & p1_pred_risky).sum())
        p1_fn = int((gt_wrong & ~p1_pred_risky).sum())
        p1_tn = int((~gt_wrong & ~p1_pred_risky).sum())
        p1_prec = p1_tp / max(p1_tp + p1_fp, 1)
        p1_rec  = p1_tp / max(p1_tp + p1_fn, 1)
        p1_f1   = 2 * p1_prec * p1_rec / max(p1_prec + p1_rec, 1e-9)

        # Phase 2 post-judge metrics on this subset
        p2_pred_risky = p2_labels.isin(["medium", "high"])
        p2_tp = int((gt_wrong & p2_pred_risky).sum())
        p2_fp = int((~gt_wrong & p2_pred_risky).sum())
        p2_fn = int((gt_wrong & ~p2_pred_risky).sum())
        p2_tn = int((~gt_wrong & ~p2_pred_risky).sum())
        p2_prec = p2_tp / max(p2_tp + p2_fp, 1)
        p2_rec  = p2_tp / max(p2_tp + p2_fn, 1)
        p2_f1   = 2 * p2_prec * p2_rec / max(p2_prec + p2_rec, 1e-9)

        # Label transitions
        changed_mask = p1_labels != p2_labels
        n_changed = int(changed_mask.sum())

        # Direction of change - compare CORRECTNESS before vs after, not raw
        # label text. "medium" -> "high" is a text change but NOT a
        # correctness change (both count as "risky" in the confusion
        # matrix), so it must NOT be counted as improved/worsened.
        p1_correct = (p1_pred_risky == gt_wrong)
        p2_correct = (p2_pred_risky == gt_wrong)
        improved = int((changed_mask & p2_correct & ~p1_correct).sum())
        worsened = int((changed_mask & p1_correct & ~p2_correct).sum())
        neutral  = int((changed_mask & (p1_correct == p2_correct)).sum())

        add(
            f"  Sub-sample size                    : {n_p2:,} queries",
            f"  Queries triggering LLM judge       : {n_judges:,} ({_pct(n_judges, n_p2)})",
            f"  Queries whose risk label CHANGED    : {n_changed:,} ({_pct(n_changed, n_p2)})",
            f"    - Moved TOWARD ground truth (fixed) : {improved:,}",
            f"    - Moved AWAY from ground truth     : {worsened:,}",
            f"    - Text changed but correctness same (neutral) : {neutral:,}",
            "",
            f"  METRICS ON PHASE 2 SUBSET (N={n_p2}):",
            f"    {'Metric':<20}  {'Phase 1 (Pre-Judge)':>20}  {'Phase 2 (Post-Judge)':>20}  {'Delta':>10}",
            f"    {'-'*75}",
            f"    {'Precision':<20}  {p1_prec:>20.4f}  {p2_prec:>20.4f}  {p2_prec-p1_prec:>+10.4f}",
            f"    {'Recall':<20}  {p1_rec:>20.4f}  {p2_rec:>20.4f}  {p2_rec-p1_rec:>+10.4f}",
            f"    {'F1 Score':<20}  {p1_f1:>20.4f}  {p2_f1:>20.4f}  {p2_f1-p1_f1:>+10.4f}",
            f"    {'False Positives (FP)':<20}  {p1_fp:>20d}  {p2_fp:>20d}  {p2_fp-p1_fp:>+10d}",
            f"    {'False Negatives (FN)':<20}  {p1_fn:>20d}  {p2_fn:>20d}  {p2_fn-p1_fn:>+10d}",
            f"    {'True Positives (TP)':<20}  {p1_tp:>20d}  {p2_tp:>20d}  {p2_tp-p1_tp:>+10d}",
            f"    {'True Negatives (TN)':<20}  {p1_tn:>20d}  {p2_tn:>20d}  {p2_tn-p1_tn:>+10d}",
            "",
        )

    add(f"  Total wall clock (all phases)  : {t_total:.2f}s", "")

    # ── PROBABILITY HISTOGRAM (always in report) ───────────────────────────────
    add(sep, "  5.5  LR PROBABILITY CALIBRATION HISTOGRAM", sep, "")
    hist_lines = build_probability_histogram(p1_df)
    add(*hist_lines)
    if getattr(args, "plot_probability_histogram", False):
        print("\n" + "\n".join(hist_lines))

    # ── RISK DISTRIBUTION ─────────────────────────────────────────────────────
    add(sep, "  6. RISK DISTRIBUTION (Phase 1 labels)", sep, "")
    for lbl in ["low", "medium", "high"]:
        n_lbl = int((p1_df["phase1_risk_label"] == lbl).sum())
        add(f"    {lbl.upper():<8}: {n_lbl:6,}  {_bar(n_lbl/n_total, 30)}")
    add("")

    if p2_df is not None and len(p2_df) > 0:
        add("  Phase 2 final labels (after LLM judge where fired):")
        for lbl in ["low", "medium", "high"]:
            n_lbl = int((p2_df["final_risk_label"] == lbl).sum())
            add(f"    {lbl.upper():<8}: {n_lbl:6,}  {_bar(n_lbl/len(p2_df), 30)}")
        add("")

    # ── FALSE NEGATIVES (MISSED RISKS) ────────────────────────────────────────
    fn_rows = p1_df[(p1_df["top1_correct"] == False) &
                    (p1_df["phase1_risk_label"] == "low")].copy()
    add(sep, f"  7. MISSED RISKS — FALSE NEGATIVES ({len(fn_rows)} queries)", sep,
        "  These queries had WRONG top1 retrieval but were labelled 'low risk'.",
        "  These are the responses most at risk of going undetected.", "")

    if len(fn_rows) > 0:
        fn_show = fn_rows.nlargest(20, "lr_risk_probability")
        add(f"  {'Query':<55}  {'top1_sim':>8}  {'margin':>8}  {'lr_prob':>8}")
        add(f"  {'-'*82}")
        for _, r in fn_show.iterrows():
            add(f"  {r['query'][:55]:<55}  {r['top1_sim']:>8.4f}  "
                f"{r['margin']:>8.4f}  {r['lr_risk_probability']:>8.4f}")
        add("")
        add("  WHY were these missed?",
            f"  Average top1_sim of FN queries : {fn_rows['top1_sim'].mean():.4f}",
            f"  Average margin of FN queries   : {fn_rows['margin'].mean():.4f}",
            "  These queries had similarity / margin values that LOOKED confident",
            "  to the LR model, even though the retrieved endpoint was wrong.",
            "  To catch more: lower --low-risk-threshold (e.g. 0.25) or increase",
            "  --n-train to improve model boundary precision.", "")
    else:
        add("  None — all wrong retrievals were flagged!", "")

    # ── FALSE POSITIVES (FALSE ALARMS) ────────────────────────────────────────
    fp_rows = p1_df[(p1_df["top1_correct"] == True) &
                    (p1_df["phase1_risk_label"].isin(["medium", "high"]))].copy()
    add(sep, f"  8. FALSE ALARMS — FALSE POSITIVES ({len(fp_rows)} queries)", sep,
        "  These queries had CORRECT top1 retrieval but were flagged as risky.",
        "  These cause unnecessary review burden.", "")

    if len(fp_rows) > 0:
        fp_show = fp_rows.nsmallest(20, "lr_risk_probability")
        add(f"  {'Query':<55}  {'top1_sim':>8}  {'margin':>8}  {'lr_prob':>8}")
        add(f"  {'-'*82}")
        for _, r in fp_show.iterrows():
            add(f"  {r['query'][:55]:<55}  {r['top1_sim']:>8.4f}  "
                f"{r['margin']:>8.4f}  {r['lr_risk_probability']:>8.4f}")
        add("")
        add("  WHY were these flagged?",
            f"  Average top1_sim of FP queries : {fp_rows['top1_sim'].mean():.4f}",
            f"  Average margin of FP queries   : {fp_rows['margin'].mean():.4f}",
            "  Even though retrieval was correct, these queries had low-confidence",
            "  features (low margin = narrow top1-top2 gap) that look risky to the",
            "  model. Many may still have correct, well-grounded responses despite",
            "  the retrieval being correct (i.e. not a real hallucination risk).", "")
    else:
        add("  None — zero false alarms!", "")

    # ── HIGH RISK QUERIES CORRECTLY CAUGHT ────────────────────────────────────
    tp_rows = p1_df[(p1_df["top1_correct"] == False) &
                    (p1_df["phase1_risk_label"].isin(["medium", "high"]))].copy()
    add(sep, f"  9. CORRECTLY CAUGHT RISKS — TRUE POSITIVES ({len(tp_rows)} queries)", sep,
        "  Wrong top1 retrieval AND flagged as medium/high risk.", "")

    if len(tp_rows) > 0:
        tp_show = tp_rows.nlargest(20, "lr_risk_probability")
        add(f"  {'Query':<55}  {'top1_sim':>8}  {'margin':>8}  {'lr_prob':>8}  {'label':>8}")
        add(f"  {'-'*92}")
        for _, r in tp_show.iterrows():
            add(f"  {r['query'][:55]:<55}  {r['top1_sim']:>8.4f}  "
                f"{r['margin']:>8.4f}  {r['lr_risk_probability']:>8.4f}  "
                f"{r['phase1_risk_label']:>8}")
        add("")

    # ── PHASE 2 DEEP DETAIL ───────────────────────────────────────────────────
    if p2_df is not None and len(p2_df) > 0:
        p2_high = p2_df[p2_df["final_risk_label"] == "high"]
        add(sep, f"  10. PHASE 2 HIGH-RISK DETAIL ({len(p2_high)} queries)", sep, "")
        for _, r in p2_high.iterrows():
            add(
                f"  Query      : {r['query']}",
                f"  GT correct : {r['top1_correct']}  |  "
                f"Endpoint: {r['top1_method']} {r['top1_endpoint']}",
                f"  Reason     : {r['risk_reason']}",
                f"  Response   : {str(r['response'])[:200]}",
                "",
            )

    # ── RECOMMENDATIONS ───────────────────────────────────────────────────────
    add(sep, "  11. RECOMMENDATIONS", sep, "")

    # Fragment analysis (new feature)
    if "is_fragment" in p1_df.columns:
        frag_df = p1_df[p1_df["is_fragment"] == True]
        frag_fn = frag_df[(frag_df["top1_correct"] == False) &
                          (frag_df["phase1_risk_label"] == "low")]
        frag_fp = frag_df[(frag_df["top1_correct"] == True) &
                          (frag_df["phase1_risk_label"].isin(["medium", "high"]))]
        n_frags = len(frag_df)
        n_frag_fn = len(frag_fn)
        n_frag_fp = len(frag_fp)
        add(
            f"  Fragment query analysis ({n_frags:,} fragment queries detected):",
            f"    FN (missed risks) that are fragments   : {n_frag_fn} "
            f"({_pct(n_frag_fn, max(cm['FN'], 1))} of all FNs)",
            f"    FP (false alarms) that are fragments   : {n_frag_fp}",
            "    Short fragment queries are harder for the LLM judge because",
            "    domain vocabulary overlaps even when the endpoint is wrong.",
            "    The updated judge prompt applies extra scrutiny to these queries.",
            "",
        )

    if cm["fnr"] > 0.4:
        add(
            "  [!] HIGH MISS RATE: The pipeline is missing more than 40% of bad",
            "      retrievals. Recommended actions:",
            "      1. Lower --low-risk-threshold to 0.25 (catches more but increases FP)",
            "      2. Run with --retrain --n-train 2500 for a better LR boundary",
            "      3. Review the Probability Histogram (section 5.5) to spot the",
            "         overlap zone and tune thresholds accordingly.",
            "",
        )
    elif cm["fnr"] > 0.25:
        add(
            "  [~] MODERATE MISS RATE: The pipeline misses ~1 in 4 bad retrievals.",
            "      Consider lowering --low-risk-threshold slightly (e.g. 0.28).",
            "      Check section 5.5 histogram for the optimal boundary.",
            "",
        )
    else:
        add("  [+] Miss rate is within acceptable range.", "")

    if cm["fpr"] > 0.15:
        add(
            "  [!] HIGH FALSE ALARM RATE: More than 15% of safe queries are flagged.",
            "      Raise --high-risk-threshold to 0.70 to reduce unnecessary alerts.",
            "      Also check the histogram to see where safe-query probs peak;",
            "      a well-placed threshold keeps FP rate under 10%.",
            "",
        )
    else:
        add("  [+] False alarm rate is within acceptable range.", "")

    n_uncertain_pct = n_uncertain / max(n_total, 1)
    if n_uncertain_pct > 0.5:
        add(
            "  [!] HIGH LLM JUDGE RATE: More than 50% of queries trigger the expensive",
            "      LLM judge. Run --retrain --n-train 2500 or widen thresholds to",
            "      reduce API cost. Target uncertain band 15-25% of queries.",
            "",
        )
    else:
        add(
            f"  [+] LLM judge rate is {n_uncertain_pct*100:.1f}% - "
            f"{(1-n_uncertain_pct)*100:.1f}% of API cost saved by the cascade.",
            "",
        )

    add(
        sep,
        "  Report generated by hallucination_sim.py",
        f"  Run timestamp: {run_ts}",
        sep,
    )
    return L


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Full simulation of the hallucination-detection pipeline.\n"
            "Phase 1 (bulk) runs on ALL unique queries — cheap, no response generation.\n"
            "Phase 2 (deep) runs the full pipeline on --n-deep sampled queries."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("csv_path",  help="Path to site24x7_Dataset.csv")
    parser.add_argument("xlsx_path", help="Path to site24x7_Admin_API.xlsx")

    parser.add_argument("--extra-descriptions", default=None)
    parser.add_argument("--n-sim",  type=int, default=0,
        help="Limit Phase 1 to N queries (0 = ALL unique queries, default).")
    parser.add_argument("--n-deep", type=int, default=0,
        help="Number of queries to run Phase 2 (full response + judge) on. "
             "0 = Phase 1 only.")
    parser.add_argument("--n-train", type=int, default=2500,
        help="Training queries for the LR model (default 2500; ~21%% wrong-retrieval rate "
             "gives ~500+ risky examples to learn from).")
    parser.add_argument("--n-examples", type=int, default=2)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--top-k",  type=int, default=5)
    parser.add_argument("--workers", type=int, default=3,
        help="ThreadPoolExecutor workers for Phase 2 generation (default 3).")

    parser.add_argument("--no-class-weight", action="store_true",
        help="Disable class_weight='balanced' in LogisticRegression. "
             "By default balanced weighting is used to handle the ~79/21 class imbalance.")

    parser.add_argument("--base-url",  default=None)
    parser.add_argument("--api-key",   default=None)
    parser.add_argument("--embed-model", default="azure:primary/s247-textembedding-3l")
    parser.add_argument("--chat-model",  default="azure:primary/gpt-4.1-mini")

    parser.add_argument("--groundedness-threshold", type=float, default=0.5)
    parser.add_argument("--groundedness-gate-threshold", type=float, default=0.10,
        help="If frac_unsupported_embedding exceeds this, force the LLM judge to "
             "fire even when the LR model says the retrieval-side risk is low. "
             "Set from build_hallucination_testset.py's honest-vs-hallucinated "
             "distribution (default 0.10 catches ~69/83 previously-missed cases "
             "on the validated test set with near-zero false triggers on honest "
             "responses, which sit at 0 in ~75%% of cases).")
    parser.add_argument("--low-risk-threshold",     type=float, default=0.35)
    parser.add_argument("--high-risk-threshold",    type=float, default=0.65)

    parser.add_argument("--model-path", default=None,
        help="Path to trained LR model (default: hallucination_risk_model.joblib "
             "next to csv_path). Must exist — run hallucination_detect.py first.")

    parser.add_argument("--mock",           action="store_true")
    parser.add_argument("--cache-dir",      default=None)
    parser.add_argument("--no-cache",       action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=20)
    parser.add_argument("--timeout",        type=float, default=60.0)
    parser.add_argument("--save-log",       action="store_true",
        help="Write JSON-lines log file (hallucination_sim.log) to the output dir.")

    parser.add_argument("--eval-queries-csv", default=None,
        help="Optional CSV with a 'query' column - restricts Phase 1 (and the "
             "Phase 2 sample) to ONLY these queries. Use with a held-out split "
             "(see split_holdout.py) so the KNN/sheet lookups built from the "
             "complementary train pool can't leak into what's being scored.")

    args = parser.parse_args()

    run_ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    T       = {}
    T["wall_start"] = time.time()

    csv_path  = Path(args.csv_path)
    xlsx_path = Path(args.xlsx_path)
    for p in (csv_path, xlsx_path):
        if not p.exists():
            sys.exit(f"File not found: {p}")

    out_dir    = csv_path.parent
    model_path = Path(args.model_path) if args.model_path \
                 else csv_path.parent / "hallucination_risk_model.joblib"
    log_path   = out_dir / "hallucination_sim.log"
    report_path = out_dir / "hallucination_sim_report.txt"
    p1_csv     = out_dir / "hallucination_sim_results.csv"
    p2_csv     = out_dir / "hallucination_sim_deep_results.csv"

    # ── logging ────────────────────────────────────────────────────────────
    logger = setup_logging(log_path if args.save_log else Path(os.devnull))

    # ── credentials ────────────────────────────────────────────────────────
    env_file = csv_path.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    base_url = args.base_url or os.environ.get("PROXY_BASE_URL")
    api_key  = args.api_key  or os.environ.get("PROXY_API_KEY")

    client = None
    if args.mock:
        print("=" * 72)
        print("MOCK MODE: fake embeddings + fake LLM.  Pipeline test only.")
        print("=" * 72)
    else:
        if not base_url or not api_key:
            sys.exit(
                "Provide --base-url and --api-key (or .env), or use --mock.")
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=args.timeout)

    cache_dir = Path(args.cache_dir) if args.cache_dir \
                else csv_path.parent / ".rag_cache"

    # ── load dataset ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  HALLUCINATION SIMULATION - {run_ts}")
    print(f"{'='*72}")
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(csv_path)
    if df["markedCorrect"].dtype == object:
        df["markedCorrect"] = (
            df["markedCorrect"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    df["markedCorrect"] = df["markedCorrect"].astype(bool)

    true_df = df[df["markedCorrect"] == True].copy()
    true_df["key"]   = list(zip(true_df["endpoint"], true_df["method"]))
    true_df["sheet"] = true_df["sheet"]

    # collapse to unique queries with valid_keys set
    all_query_groups = (
        true_df.groupby("query")
        .agg(
            valid_keys=("key", lambda ks: sorted(set(ks))),
            sheet=("sheet",   "first"),
        )
        .reset_index()
    )
    if args.eval_queries_csv:
        eval_q_path = Path(args.eval_queries_csv)
        if not eval_q_path.exists():
            sys.exit(f"--eval-queries-csv file not found: {eval_q_path}")
        eval_queries = set(pd.read_csv(eval_q_path)["query"])
        before = len(all_query_groups)
        all_query_groups = (all_query_groups[all_query_groups["query"].isin(eval_queries)]
                            .reset_index(drop=True))
        print(f"  --eval-queries-csv: restricted from {before:,} to "
              f"{len(all_query_groups):,} queries (held-out set from {eval_q_path})")

    if args.n_sim > 0:
        all_query_groups = (all_query_groups
                            .sample(min(args.n_sim, len(all_query_groups)),
                                    random_state=args.seed)
                            .reset_index(drop=True))
    n_unique = len(all_query_groups)
    print(f"  {len(df):,} total rows  |  {n_unique:,} unique queries selected for Phase 1")

    # ── load description map ───────────────────────────────────────────────
    print("\n[2/6] Loading description file...")
    doc_map  = load_doc_descriptions(xlsx_path)
    all_keys = set(zip(df["endpoint"], df["method"]))
    if args.extra_descriptions:
        ep = Path(args.extra_descriptions)
        if not ep.exists():
            sys.exit(f"File not found: {ep}")
        for k, v in load_extra_descriptions(ep).items():
            doc_map.setdefault(k, v)
    cov = len(set(doc_map.keys()) & all_keys)
    print(f"  doc_map coverage: {cov}/{len(all_keys)} pairs "
          f"({cov/len(all_keys)*100:.1f}%)")

    # ── build corpus ───────────────────────────────────────────────────────
    print("\n[3/6] Building corpus + embedding...")
    base_corpus_df, _ = build_corpus_and_eval(df, 50, args.n_examples, args.seed)
    enrichment_pool   = true_df  # use all for corpus (no eval exclusion in sim mode)
    corpus_df         = make_variant_corpus(
        base_corpus_df, df, doc_map, args.n_examples,
        enrichment_pool, build_text_doc_description)
    corpus_texts = corpus_df["text"].tolist()

    t0 = time.time()
    corpus_vecs = embed_dispatch(corpus_texts, client, args, cache_dir)
    corpus_norm = corpus_vecs / np.clip(
        np.linalg.norm(corpus_vecs, axis=1, keepdims=True), 1e-9, None)
    T["corpus_embedding"] = time.time() - t0
    print(f"  Corpus: {len(corpus_df):,} chunks  "
          f"(embedding took {T['corpus_embedding']:.1f}s)")

    # ── embed all queries ──────────────────────────────────────────────────
    print(f"\n[4/6] Embedding {n_unique:,} unique queries...")
    t0 = time.time()
    query_vecs = embed_dispatch(
        all_query_groups["query"].tolist(), client, args, cache_dir)
    T["query_embedding"] = time.time() - t0
    print(f"  Done in {T['query_embedding']:.1f}s  "
          f"(avg {T['query_embedding']/max(n_unique,1)*1000:.1f}ms/query)")

    # ── load LR model ──────────────────────────────────────────────────────
    print("\n[5/6] Loading risk model...")
    if not model_path.exists():
        sys.exit(
            f"No trained model at {model_path}.\n"
            "Run  hallucination_detect.py  first to train and save the model.")
    try:
        model = _load_model(model_path)
        print(f"  Model loaded from {model_path}")
    except RuntimeError as e:
        sys.exit(
            f"  {e}\n"
            "  Run hallucination_detect.py --retrain to rebuild the model.")

    # ── PHASE 1 ────────────────────────────────────────────────────────────
    print(f"\n[6/6] PHASE 1: scoring {n_unique:,} queries...")
    t0 = time.time()
    p1_df = run_phase1(
        all_query_groups, corpus_df, corpus_norm, query_vecs,
        model, args, logger)
    T["phase1"] = time.time() - t0

    p1_df.to_csv(p1_csv, index=False)
    print(f"\n  Phase 1 results saved -> {p1_csv}")

    # ── PHASE 2 (optional) ─────────────────────────────────────────────────
    p2_df = None
    T["phase2"] = 0.0
    if args.n_deep > 0:
        print(f"\n{'-'*72}")
        print(f"  PHASE 2: deep eval on {min(args.n_deep, n_unique)} queries "
              f"({args.workers} workers)")
        print(f"{'-'*72}")

        deep_sample = (all_query_groups
                       .sample(min(args.n_deep, n_unique), random_state=args.seed + 1)
                       .reset_index(drop=True))
        deep_vecs   = embed_dispatch(
            deep_sample["query"].tolist(), client, args, cache_dir)

        t0 = time.time()
        p2_df = run_phase2(
            deep_sample, deep_vecs, corpus_df, corpus_norm,
            model, client, args, cache_dir, logger)
        T["phase2"] = time.time() - t0

        p2_df.to_csv(p2_csv, index=False)
        print(f"\n  Phase 2 results saved -> {p2_csv}")

    T["total"] = time.time() - T["wall_start"]

    # ── confusion matrix & metrics ─────────────────────────────────────────
    cm = compute_confusion(p1_df, "phase1_risk_label")

    # ── build & save report ────────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print("  Generating report...")
    report_lines = build_report(p1_df, p2_df, cm, T, args, run_ts, csv_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    # ── print summary to terminal ──────────────────────────────────────────
    sep = "=" * 72
    print()
    print(sep)
    print("  SIMULATION COMPLETE")
    print(sep)
    print(f"  Queries evaluated (Phase 1)   : {cm['n']:,}")
    print(f"  GT correct top1               : {cm['n_gt_safe']:,}  "
          f"({_pct(cm['n_gt_safe'], cm['n'])})")
    print(f"  GT wrong top1                 : {cm['n_gt_risky']:,}  "
          f"({_pct(cm['n_gt_risky'], cm['n'])})")
    print()
    print(f"  +-------------------------------------------------------+")
    print(f"  |  CONFUSION MATRIX         Pred SAFE    Pred RISKY     |")
    print(f"  |  Actual SAFE (top1 ok)    {cm['TN']:7,} TN  {cm['FP']:7,} FP  |")
    print(f"  |  Actual RISKY (wrong top1){cm['FN']:7,} FN  {cm['TP']:7,} TP  |")
    print(f"  +-------------------------------------------------------+")
    print()
    print(f"  Precision  : {cm['precision']:.3f}   (flagged queries that were genuinely risky)")
    print(f"  Recall     : {cm['recall']:.3f}   (risky queries that were caught)")
    print(f"  F1 Score   : {cm['f1']:.3f}")
    print(f"  Accuracy   : {cm['accuracy']:.3f}")
    print(f"  Miss Rate  : {cm['fnr']:.3f}   (fraction of risky queries that slipped through)")
    print(f"  False Alarm: {cm['fpr']:.3f}   (fraction of safe queries over-flagged)")
    print()
    print(f"  LLM judge rate : "
          f"{_pct(int((p1_df['cascade_bucket']=='uncertain').sum()), cm['n'])} "
          f"of queries would trigger the judge in production")
    print(f"  Total wall time: {T['total']:.1f}s")
    print()
    print(f"  Output files:")
    print(f"    Phase 1 CSV   : {p1_csv}")
    if p2_df is not None:
        print(f"    Phase 2 CSV   : {p2_csv}")
    print(f"    Manager report: {report_path}")
    if args.save_log:
        print(f"    JSON-lines log: {log_path}")
    print()

    # ── print PowerShell commands ──────────────────────────────────────────
    print(sep)
    print("  POWERSHELL COMMANDS")
    print(sep)
    print()
    print("# (a) Phase 1 only — fast, all queries, no LLM calls:")
    print("python hallucination_sim.py `")
    print("    Datasets/site24x7_Dataset.csv `")
    print("    Datasets/ADMIN_API/site24x7_Admin_API.xlsx `")
    print("    --extra-descriptions Datasets/reports_synthetic_descriptions.csv `")
    print("    --base-url http://20.235.183.15:443/openai/v1 `")
    print(f"    --api-key {api_key or 'YOUR_API_KEY'} `")
    print("    --save-log")
    print()
    print("# (b) Phase 1 + Phase 2 deep eval on 200 queries (3 workers):")
    print("python hallucination_sim.py `")
    print("    Datasets/site24x7_Dataset.csv `")
    print("    Datasets/ADMIN_API/site24x7_Admin_API.xlsx `")
    print("    --extra-descriptions Datasets/reports_synthetic_descriptions.csv `")
    print("    --base-url http://20.235.183.15:443/openai/v1 `")
    print(f"    --api-key {api_key or 'YOUR_API_KEY'} `")
    print("    --n-deep 200 --workers 4 --save-log")
    print()
    print("# (c) Mock smoke test — zero API calls:")
    print("python hallucination_sim.py `")
    print("    Datasets/site24x7_Dataset.csv `")
    print("    Datasets/ADMIN_API/site24x7_Admin_API.xlsx `")
    print("    --mock --n-sim 100 --n-deep 10 --save-log")


if __name__ == "__main__":
    main()