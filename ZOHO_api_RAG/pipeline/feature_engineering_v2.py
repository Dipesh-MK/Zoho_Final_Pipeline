"""
feature_engineering_v2.py

Extends the existing 5-feature hallucination-detection LR model with four new
features that capture information the current features are structurally blind to:

  Feature 6 - sheet_wrong_rate            (historical wrongness per corpus sheet)
  Feature 7 - query_endpoint_token_overlap (lexical overlap: query vs endpoint path)
  Feature 8 - topk_similarity_entropy     (entropy of top-k similarity distribution)
  Feature 9 - knn_neighbor_wrong_rate     (KNN neighborhood historical wrongness)

ZERO new embedding API calls are needed if hallucination_sim.py was already run
with the same corpus and queries -- embed_texts() has a per-text disk cache keyed
by sha256(model+text). The script prints a cache-hit summary so you can confirm.

Anti-leakage design for features 6 and 9
-----------------------------------------
Both features are "historical outcome" features whose values depend on the ground
truth labels of OTHER queries. A naive implementation would compute them on the full
dataset and then train/evaluate on that same data, leaking test labels into test
features. This script avoids that with a manual K-fold loop:

  Feature 6 (sheet_wrong_rate):
    For each CV fold, the per-sheet wrong-rate map is built ONLY from the TRAINING
    fold rows. The held-out fold queries look up their top1 endpoint's sheet in this
    map. Their own top1_correct value is never part of the computation that produces
    their own feature value.

  Feature 9 (knn_neighbor_wrong_rate):
    For each CV fold, sklearn NearestNeighbors is fitted ONLY on the TRAINING fold
    query vectors + labels. Held-out fold queries find their k nearest neighbors
    within the training fold only, and the fraction of those training-fold neighbors
    that had wrong retrieval becomes the feature. No held-out query can be its own
    neighbor.

  The OOF (out-of-fold) arrays for features 6 and 9 are what drive the AUC numbers.
  Full-dataset (potentially biased) versions are also computed and saved to the CSV
  for reference, labeled clearly as _full suffix columns.

Usage:
    # offline smoke test (fake embeddings, zero API calls):
    python feature_engineering_v2.py ^
        Datasets/site24x7_Dataset.csv ^
        Datasets/ADMIN_API/site24x7_Admin_API.xlsx ^
        Datasets/hallucination_sim_results.csv ^
        --mock

    # real run (100%% cache hits expected from prior hallucination_sim.py run):
    python feature_engineering_v2.py ^
        Datasets/site24x7_Dataset.csv ^
        Datasets/ADMIN_API/site24x7_Admin_API.xlsx ^
        Datasets/hallucination_sim_results.csv ^
        --extra-descriptions Datasets/reports_synthetic_descriptions.csv ^
        --base-url http://20.235.183.15:443/openai/v1 ^
        --api-key YOUR_KEY ^
        --output-dir .
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

try:
    from scipy.special import softmax as _softmax
    from scipy.stats import entropy as _scipy_entropy
    def _softmax_rows(x):
        return _softmax(x, axis=1)
    def _row_entropy(p):
        return _scipy_entropy(p.T)   # scipy entropy over columns -> shape (n_rows,)
except ImportError:
    # Pure-numpy fallbacks if scipy is not installed
    def _softmax_rows(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    def _row_entropy(p):
        p = np.clip(p, 1e-12, None)
        return -(p * np.log(p)).sum(axis=1)

from rag_eval import (
    build_corpus_and_eval,
    embed_texts,
    get_embeddings_mock,
    humanize_path,
)
from compare_doc_strategies import (
    build_text_doc_description,
    load_doc_descriptions,
    load_extra_descriptions,
    make_variant_corpus,
)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

EXISTING_FEATURES = [
    "top1_sim",
    "margin",
    "avg_pairwise_topk_sim",
    "query_token_count",
    "n_candidates_within_margin",
]

NEW_FEATURES = [
    "sheet_wrong_rate",
    "query_endpoint_token_overlap",
    "topk_similarity_entropy",
    "knn_neighbor_wrong_rate",
]

ALL_FEATURES = EXISTING_FEATURES + NEW_FEATURES

# Known 5-feature ceiling from offline_model_tuning.py (5-fold CV, OOF, balanced)
BASELINE_AUC = {
    "logistic_regression": 0.8847,
    "random_forest":       0.8938,   # midpoint of 0.8937-0.8939
    "gradient_boosting":   0.8966,   # midpoint of 0.8963-0.8969
}

MODELS = {
    "logistic_regression": lambda: LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    ),
}


# ---------------------------------------------------------------------------
# EMBEDDING HELPERS
# ---------------------------------------------------------------------------

def embed_dispatch(texts, client, args, cache_dir):
    if args.mock:
        return get_embeddings_mock(texts)
    return embed_texts(client, args.embed_model, texts, cache_dir, use_cache=True)


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return vecs / norms


# ---------------------------------------------------------------------------
# FEATURE 7 -- LEXICAL OVERLAP (no embeddings, no leakage)
# ---------------------------------------------------------------------------

def _tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def compute_lexical_overlap(queries, endpoints):
    """
    Overlap ratio = |query_tokens INTERSECT endpoint_path_tokens| / |query_tokens|.

    Uses simple overlap ratio (not Jaccard) because the endpoint path vocabulary
    is much smaller than a natural-language sentence; we care whether the query's
    specific named resource appears in the endpoint at all.
    humanize_path() is used to tokenize the endpoint path, exactly matching how
    the existing corpus text is built.
    Returns 0.0 when query has no alphanumeric tokens (degenerate edge case).
    """
    out = np.zeros(len(queries), dtype=np.float32)
    for i, (q, ep) in enumerate(zip(queries, endpoints)):
        q_toks  = _tokenize(str(q))
        ep_toks = _tokenize(humanize_path(str(ep)))
        out[i] = len(q_toks & ep_toks) / len(q_toks) if q_toks else 0.0
    return out


# ---------------------------------------------------------------------------
# FEATURE 8 -- TOP-K SIMILARITY ENTROPY (no CV needed, no leakage)
# ---------------------------------------------------------------------------

def compute_topk_entropy(query_vecs_norm, corpus_vecs_norm, top_k=10):
    """
    For each query:
      1. Compute cosine similarity against ALL corpus chunks.
      2. Take the top_k scores.
      3. Softmax-normalize them to a probability distribution.
      4. Compute normalized Shannon entropy: H / log(k), range [0, 1].

    High entropy  -> candidates 1..k look equally likely (diffuse, uncertain).
    Low entropy   -> one clear winner stands out.

    This is complementary to `margin` (which only compares ranks 1 vs 2);
    a query can have a large margin but a still-diffuse tail among ranks 3-10.

    Uses already-cached normalized vectors -- zero new API calls.
    """
    n_q   = query_vecs_norm.shape[0]
    out   = np.zeros(n_q, dtype=np.float32)
    log_k = np.log(max(top_k, 2))
    CHUNK = 512  # process rows in chunks to avoid OOM

    for start in range(0, n_q, CHUNK):
        end  = min(start + CHUNK, n_q)
        sims = query_vecs_norm[start:end] @ corpus_vecs_norm.T   # (chunk, n_corpus)
        # Partition to get top_k indices (unsorted, we only need the values)
        top_idx   = np.argpartition(sims, -top_k, axis=1)[:, -top_k:]
        topk_sims = np.take_along_axis(sims, top_idx, axis=1)    # (chunk, k)
        probs     = _softmax_rows(topk_sims)                      # (chunk, k)
        ent       = _row_entropy(probs)                           # (chunk,)
        out[start:end] = (ent / log_k).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# FEATURES 6 AND 9 -- LEAKAGE-FREE CROSS-VALIDATED HELPERS
# ---------------------------------------------------------------------------

def _sheet_wrong_rate_fold(
    train_endpoints, train_methods, train_wrong,
    test_endpoints, test_methods,
    endpoint_to_sheet, global_fallback
):
    """
    Build per-sheet wrong-rate map from TRAINING fold only.
    Apply to TEST fold queries -- their own outcome is never in the map.
    """
    train_sheets = pd.Series([
        endpoint_to_sheet.get((ep, m), "__unknown__")
        for ep, m in zip(train_endpoints, train_methods)
    ])
    sheet_rate = (pd.Series(train_wrong.astype(float))
                  .groupby(train_sheets).mean().to_dict())

    out = np.zeros(len(test_endpoints), dtype=np.float32)
    for i, (ep, m) in enumerate(zip(test_endpoints, test_methods)):
        sheet = endpoint_to_sheet.get((ep, m), "__unknown__")
        out[i] = sheet_rate.get(sheet, global_fallback)
    return out


def _knn_wrong_rate_fold(train_vecs, train_wrong, test_vecs, knn_k):
    """
    Build KNN index on TRAINING fold only.
    For each TEST fold query, find its knn_k nearest training-fold neighbors
    and return the fraction that had wrong top1 retrieval.
    No test query is ever in the index, so no self-lookup leakage.
    """
    k = min(knn_k, len(train_vecs))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine",
                          algorithm="brute", n_jobs=-1)
    nn.fit(train_vecs)
    _, indices = nn.kneighbors(test_vecs)             # (n_test, k)
    return train_wrong[indices].mean(axis=1).astype(np.float32)


def compute_leakage_free_features(res_df, query_vecs, endpoint_to_sheet, cv, knn_k, y):
    """
    Full CV loop producing leakage-free OOF arrays for features 6 and 9.
    Also produces full-dataset (biased) versions for export-only CSV columns.
    """
    n = len(res_df)
    oof_sheet = np.full(n, np.nan, dtype=np.float32)
    oof_knn   = np.full(n, np.nan, dtype=np.float32)
    global_wr = float(y.mean())

    top1_ep  = res_df["top1_endpoint"].values
    top1_met = res_df["top1_method"].values

    for fold_i, (tr, te) in enumerate(cv.split(query_vecs, y)):
        print(f"    fold {fold_i+1}/{cv.n_splits}: "
              f"train={len(tr)}, test={len(te)}")

        oof_sheet[te] = _sheet_wrong_rate_fold(
            top1_ep[tr], top1_met[tr], y[tr],
            top1_ep[te], top1_met[te],
            endpoint_to_sheet, global_wr,
        )
        oof_knn[te] = _knn_wrong_rate_fold(
            query_vecs[tr], y[tr], query_vecs[te], knn_k,
        )

    # Full-dataset versions (biased, for CSV export reference only)
    all_sheets   = pd.Series([endpoint_to_sheet.get((ep, m), "__unknown__")
                               for ep, m in zip(top1_ep, top1_met)])
    full_sheet   = (pd.Series(y.astype(float))
                    .groupby(all_sheets).transform("mean")
                    .values.astype(np.float32))

    k_full       = min(knn_k + 1, n)
    nn_full      = NearestNeighbors(n_neighbors=k_full, metric="cosine",
                                     algorithm="brute", n_jobs=-1)
    nn_full.fit(query_vecs)
    _, full_idx  = nn_full.kneighbors(query_vecs)
    full_idx     = full_idx[:, 1:]            # drop self (nearest at distance 0)
    full_knn     = y[full_idx].mean(axis=1).astype(np.float32)

    return oof_sheet, oof_knn, full_sheet, full_knn


# ---------------------------------------------------------------------------
# MODEL TRAINING + EVALUATION
# ---------------------------------------------------------------------------

def compare_models_9feat(X, y, cv):
    """
    5-fold stratified CV with manual OOF loop (matches offline_model_tuning.py
    approach exactly for apples-to-apples AUC comparability).
    Returns (summary_df, cv_probs_dict, fitted_models_on_full_data_dict).
    """
    summary_rows = []
    cv_probs     = {}
    fitted_models = {}

    for name, make_model in MODELS.items():
        print(f"  Training {name}...")
        oof_prob = np.zeros(len(y), dtype=np.float64)

        for tr, te in cv.split(X, y):
            m = make_model()
            m.fit(X[tr], y[tr])
            oof_prob[te] = m.predict_proba(X[te])[:, 1]

        cv_probs[name] = oof_prob
        roc_auc = roc_auc_score(y, oof_prob)
        pr_auc  = average_precision_score(y, oof_prob)
        delta   = roc_auc - BASELINE_AUC.get(name, roc_auc)

        # Full-data fit for feature importance
        full_m = make_model()
        full_m.fit(X, y)
        fitted_models[name] = full_m

        summary_rows.append({
            "model":                    name,
            "roc_auc_9feat":            round(roc_auc, 4),
            "pr_auc_9feat":             round(pr_auc,  4),
            "roc_auc_5feat_baseline":   BASELINE_AUC.get(name, float("nan")),
            "roc_auc_delta":            round(delta,   4),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("roc_auc_9feat", ascending=False)
    return summary_df, cv_probs, fitted_models


def print_feature_importance(name, model, features):
    print(f"\n  {name} -- feature importance / coefficients:")
    width = max(len(f) for f in features) + 2
    if hasattr(model, "coef_"):
        ranked = sorted(zip(features, model.coef_[0]), key=lambda x: -abs(x[1]))
        for feat, coef in ranked:
            direction = "-> more risky" if coef > 0 else "-> less risky"
            tag = " [NEW]" if feat in NEW_FEATURES else ""
            print(f"    {feat:{width}s} coef={coef:+.4f}  (higher {direction}){tag}")
    elif hasattr(model, "feature_importances_"):
        ranked = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
        for feat, imp in ranked:
            tag = " [NEW]" if feat in NEW_FEATURES else ""
            print(f"    {feat:{width}s} importance={imp:.4f}{tag}")


def print_histogram(prob, y_true, label, bins=20):
    """Dual-row text histogram identical in style to offline_model_tuning.py."""
    print(f"\n  Probability histogram ({label}), GT-safe vs GT-risky, {bins} bins:")
    edges = np.linspace(0, 1, bins + 1)
    safe_c,  _ = np.histogram(prob[y_true == 0], bins=edges)
    risky_c, _ = np.histogram(prob[y_true == 1], bins=edges)
    scale = 40 / max(safe_c.max(), risky_c.max(), 1)
    for i in range(bins):
        s_bar = "#" * int(safe_c[i]  * scale)
        r_bar = "#" * int(risky_c[i] * scale)
        print(f"    {edges[i]:.2f}-{edges[i+1]:.2f} | safe  {s_bar:<40s} {safe_c[i]:5d}")
        print(f"               | risky {r_bar:<40s} {risky_c[i]:5d}")
    print()
    print("  INTERPRETATION: if the new-feature model's bars separate more cleanly")
    print("  (less overlap in the central bins) vs the old histogram from the prior")
    print("  run, the AUC gain is real and visible.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Engineer 4 new features and compare 9-feature vs 5-feature models.\n"
            "Expects 100%% cache hits -- no new embedding API calls."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csv_path",    help="site24x7_Dataset.csv")
    p.add_argument("xlsx_path",   help="site24x7_Admin_API.xlsx")
    p.add_argument("results_csv", help="hallucination_sim_results.csv (from prior run)")

    p.add_argument("--extra-descriptions", default=None)
    p.add_argument("--base-url",    default=None)
    p.add_argument("--api-key",     default=None)
    p.add_argument("--embed-model", default="azure:primary/s247-textembedding-3l")
    p.add_argument("--cache-dir",   default=None,
                   help="Embedding cache dir. Default: .rag_cache next to csv_path.")
    p.add_argument("--mock",        action="store_true",
                   help="Fake embeddings -- zero API calls, for smoke testing only.")
    p.add_argument("--top-k",    type=int, default=10,
                   help="Top-k chunks for entropy feature (default 10).")
    p.add_argument("--knn-k",    type=int, default=15,
                   help="Number of KNN neighbors for feature 9 (default 15).")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Stratified CV folds (default 5).")
    p.add_argument("--n-examples", type=int, default=2,
                   help="Example queries per corpus chunk -- must match prior run (default 2).")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output-dir",  default=".",
                   help="Directory for output CSVs (default: current directory).")
    p.add_argument("--save-model",  default=None,
                   help="If set, save best 9-feature model to this .joblib path.")
    args = p.parse_args()

    csv_path    = Path(args.csv_path)
    xlsx_path   = Path(args.xlsx_path)
    results_csv = Path(args.results_csv)
    out_dir     = Path(args.output_dir)
    cache_dir   = (Path(args.cache_dir) if args.cache_dir
                   else csv_path.parent / ".rag_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in [csv_path, xlsx_path, results_csv]:
        if not path.exists():
            sys.exit(f"File not found: {path}")

    SEP = "=" * 72
    print(f"\n{SEP}")
    print("  FEATURE ENGINEERING V2  --  4 NEW FEATURES + 9-FEATURE MODEL")
    print(f"{SEP}\n")

    # ------------------------------------------------------------------
    # 1. Load existing simulation results
    # ------------------------------------------------------------------
    print("[1/6] Loading existing simulation results ...")
    res_df = pd.read_csv(results_csv)
    if res_df["top1_correct"].dtype == object:
        res_df["top1_correct"] = (
            res_df["top1_correct"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False})
        )
    res_df["top1_correct"] = res_df["top1_correct"].astype(bool)
    res_df["gt_wrong"]     = (~res_df["top1_correct"]).astype(int)

    missing = set(EXISTING_FEATURES) - set(res_df.columns)
    if missing:
        sys.exit(f"results_csv missing columns: {missing}")

    n_q   = len(res_df)
    n_w   = int(res_df["gt_wrong"].sum())
    y     = res_df["gt_wrong"].values.astype(int)
    X_old = res_df[EXISTING_FEATURES].values
    print(f"  {n_q:,} queries, {n_w:,} ({n_w/n_q*100:.1f}%%) with wrong top1 (positive class)")

    # ------------------------------------------------------------------
    # 2. Load raw dataset + doc_map
    # ------------------------------------------------------------------
    print("\n[2/6] Loading raw dataset and doc_map ...")
    df = pd.read_csv(csv_path)
    if df["markedCorrect"].dtype == object:
        df["markedCorrect"] = (
            df["markedCorrect"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    df["markedCorrect"] = df["markedCorrect"].astype(bool)
    df["key"] = list(zip(df["endpoint"], df["method"]))

    # endpoint->(sheet) map for feature 6
    endpoint_to_sheet = (
        df.groupby(["endpoint", "method"])["sheet"]
        .first().to_dict()
    )

    doc_map = load_doc_descriptions(xlsx_path)
    if args.extra_descriptions:
        ep = Path(args.extra_descriptions)
        if not ep.exists():
            sys.exit(f"--extra-descriptions not found: {ep}")
        for k, v in load_extra_descriptions(ep).items():
            doc_map.setdefault(k, v)
    print(f"  doc_map: {len(doc_map):,} (endpoint, method) entries")

    # ------------------------------------------------------------------
    # 3. Build corpus + embed (should be 100%% cache hits)
    # ------------------------------------------------------------------
    print("\n[3/6] Building corpus and embedding (expecting 100%% cache hits) ...")
    true_df = df[df["markedCorrect"] == True].copy()
    true_df["key"] = list(zip(true_df["endpoint"], true_df["method"]))

    base_corpus_df, _ = build_corpus_and_eval(
        df, n_eval=50, n_examples=args.n_examples, seed=args.seed
    )
    corpus_df = make_variant_corpus(
        base_corpus_df, df, doc_map, args.n_examples,
        enrichment_pool=true_df,
        strategy_fn=build_text_doc_description,
    )
    corpus_texts = corpus_df["text"].tolist()
    print(f"  Corpus: {len(corpus_df):,} chunks")

    if args.mock:
        client      = None
        corpus_vecs = get_embeddings_mock(corpus_texts)
        print("  [MOCK] Fake embeddings -- zero API calls.")
    else:
        if not args.base_url or not args.api_key:
            sys.exit("Provide --base-url and --api-key, or use --mock.")
        from openai import OpenAI
        client      = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=120)
        print(f"  Embedding corpus via cache ({cache_dir}) ...")
        corpus_vecs = embed_texts(client, args.embed_model, corpus_texts,
                                   cache_dir, use_cache=True)

    corpus_vecs_norm = normalize(corpus_vecs)
    print(f"  Corpus vectors: {corpus_vecs_norm.shape}")

    # ------------------------------------------------------------------
    # 4. Embed all queries (should be 100%% cache hits)
    # ------------------------------------------------------------------
    print("\n[4/6] Embedding unique queries (expecting 100%% cache hits) ...")
    queries = res_df["query"].tolist()

    if args.mock:
        query_vecs = get_embeddings_mock(queries)
    else:
        print(f"  {len(queries):,} queries -> embedding (cache lookup) ...")
        query_vecs = embed_texts(client, args.embed_model, queries,
                                  cache_dir, use_cache=True)

    query_vecs_norm = normalize(query_vecs)
    print(f"  Query vectors: {query_vecs_norm.shape}")

    # ------------------------------------------------------------------
    # 5. Compute all 4 new features
    # ------------------------------------------------------------------
    print("\n[5/6] Computing new features ...")
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    # Feature 7: Lexical overlap (no CV, no leakage risk)
    print("  [7] query_endpoint_token_overlap (lexical, no CV needed) ...")
    feat_lexical = compute_lexical_overlap(res_df["query"], res_df["top1_endpoint"])
    print(f"      mean={feat_lexical.mean():.4f}  "
          f"min={feat_lexical.min():.4f}  max={feat_lexical.max():.4f}")

    # Feature 8: Entropy of top-k similarity (no CV, no leakage risk)
    print(f"  [8] topk_similarity_entropy (top_k={args.top_k}) ...")
    feat_entropy = compute_topk_entropy(query_vecs_norm, corpus_vecs_norm, top_k=args.top_k)
    print(f"      mean={feat_entropy.mean():.4f}  "
          f"min={feat_entropy.min():.4f}  max={feat_entropy.max():.4f}")

    # Features 6 and 9: leakage-free OOF via K-fold loop
    print(f"\n  [6,9] sheet_wrong_rate + knn_neighbor_wrong_rate")
    print(f"        Leakage-free {args.cv_folds}-fold CV  "
          f"(knn_k={args.knn_k}, sheet rate from train fold only)")
    (oof_sheet, oof_knn,
     full_sheet, full_knn) = compute_leakage_free_features(
        res_df=res_df,
        query_vecs=query_vecs,      # raw (un-normalized) for cosine NearestNeighbors
        endpoint_to_sheet=endpoint_to_sheet,
        cv=cv, knn_k=args.knn_k, y=y,
    )
    print(f"  [6] sheet_wrong_rate (OOF):        "
          f"mean={oof_sheet.mean():.4f}  min={oof_sheet.min():.4f}  max={oof_sheet.max():.4f}")
    print(f"  [9] knn_neighbor_wrong_rate (OOF): "
          f"mean={oof_knn.mean():.4f}  min={oof_knn.min():.4f}  max={oof_knn.max():.4f}")

    # Assemble 9-feature matrix -- OOF values for features 6 and 9 ensure no leakage
    X_9 = np.column_stack([
        X_old,         # existing 5
        oof_sheet,     # feature 6 (OOF)
        feat_lexical,  # feature 7
        feat_entropy,  # feature 8
        oof_knn,       # feature 9 (OOF)
    ])
    print(f"\n  9-feature matrix: {X_9.shape}")

    # ------------------------------------------------------------------
    # 6. Train 3 models with 5-fold CV
    # ------------------------------------------------------------------
    print(f"\n[6/6] Training {len(MODELS)} models "
          f"({args.cv_folds}-fold stratified CV, OOF predictions) ...")
    summary_df, cv_probs, fitted_models = compare_models_9feat(X_9, y, cv)

    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  9-FEATURE MODEL COMPARISON (out-of-fold, apples-to-apples with baseline)")
    print(f"{SEP}\n")

    hdr  = f"  {'Model':<25}  {'ROC-AUC (9f)':>13}  {'5f Baseline':>12}  {'Delta':>8}  {'PR-AUC (9f)':>12}"
    hrule = f"  {'-'*25}  {'-'*13}  {'-'*12}  {'-'*8}  {'-'*12}"
    print(hdr)
    print(hrule)
    for _, row in summary_df.iterrows():
        b = (f"{row['roc_auc_5feat_baseline']:.4f}"
             if not pd.isna(row['roc_auc_5feat_baseline']) else "     n/a")
        d = f"{row['roc_auc_delta']:+.4f}"
        print(f"  {row['model']:<25}  {row['roc_auc_9feat']:>13.4f}  "
              f"{b:>12}  {d:>8}  {row['pr_auc_9feat']:>12.4f}")

    print()
    best_row   = summary_df.iloc[0]
    best_name  = best_row["model"]
    delta      = best_row["roc_auc_delta"]
    best_auc   = best_row["roc_auc_9feat"]

    if delta > 0.01:
        print(f"  [+] NEW FEATURES ADD SIGNAL: {best_name} gained {delta:+.4f} ROC-AUC.")
        print(f"      The prior 0.88-0.90 ceiling was a FEATURE ceiling -- now broken.")
        print(f"      Compare against {BASELINE_AUC.get(best_name, '?'):.4f} baseline.")
    elif delta > 0.003:
        print(f"  [~] MARGINAL GAIN: {best_name} gained {delta:+.4f} ROC-AUC.")
        print(f"      Check feature importances to see which new feature drove it.")
    else:
        print(f"  [-] NO MEANINGFUL GAIN: delta={delta:+.4f}.")
        print(f"      The 0.88-0.90 is likely an intrinsic ceiling for this retrieval task.")

    print(f"\n  >> KEY COMPARISON NUMBER FOR YOUR MANAGER REPORT:")
    print(f"     Best 5-feature ceiling : {max(BASELINE_AUC.values()):.4f} (GradientBoosting)")
    print(f"     Best 9-feature result  : {best_auc:.4f} ({best_name})")
    print(f"     Gain from new features : {best_auc - max(BASELINE_AUC.values()):+.4f} ROC-AUC")

    # Feature importances
    print(f"\n{SEP}")
    print("  FEATURE IMPORTANCE / COEFFICIENTS (full-dataset fit, all 9 features)")
    print(f"  [NEW] marks features added in this script")
    print(f"{SEP}")
    for name in MODELS:
        print_feature_importance(name, fitted_models[name], ALL_FEATURES)

    # Histogram for best model
    print(f"\n{SEP}")
    print(f"  PROBABILITY HISTOGRAM  --  Best model: {best_name}")
    print(f"{SEP}")
    print_histogram(cv_probs[best_name], y, f"{best_name} (9-feature, OOF)")

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  SAVING OUTPUT FILES")
    print(f"{SEP}")

    feat_df = res_df.copy()
    feat_df["sheet_wrong_rate_oof"]          = oof_sheet
    feat_df["query_endpoint_token_overlap"]  = feat_lexical
    feat_df["topk_similarity_entropy"]       = feat_entropy
    feat_df["knn_neighbor_wrong_rate_oof"]   = oof_knn
    feat_df["sheet_wrong_rate_full"]         = full_sheet   # biased, export-only
    feat_df["knn_neighbor_wrong_rate_full"]  = full_knn     # biased, export-only

    feat_csv = out_dir / "feature_engineering_v2_dataset.csv"
    feat_df.to_csv(feat_csv, index=False)
    print(f"  9-feature dataset       -> {feat_csv}")

    summ_csv = out_dir / "feature_engineering_v2_model_comparison.csv"
    summary_df.to_csv(summ_csv, index=False)
    print(f"  Model comparison summary -> {summ_csv}")

    if args.save_model:
        import joblib
        save_p = Path(args.save_model)
        joblib.dump(fitted_models[best_name], save_p)
        print(f"  Best model ({best_name}) -> {save_p}")
        print(f"  9-feature input order: {ALL_FEATURES}")

    print(f"\n{SEP}")
    print("  DONE")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
