"""
hallucination_detect.py  (final production version)

Full hallucination-detection pipeline for the Site24x7 RAG system.

SIGNALS
-------
  1. Retrieval confidence (always computed, zero LLM calls):
       top1_sim            – cosine similarity of the best-retrieved chunk
       margin              – top1_sim - top2_sim  (narrow gap = ambiguous retrieval)
       avg_pairwise_topk_sim – mean pairwise cosine sim AMONG the top-k chunks
                              themselves (cluster tightness, separate from margin)

  2. Embedding-based groundedness (always computed, zero LLM calls):
       frac_unsupported_embedding – fraction of response sentences whose max
       cosine similarity to any top-k chunk is below --groundedness-threshold

  3. LLM judge – ONE combined call returning context_relevant + sentence_verdicts.
       GATED: only fires for queries where the LogisticRegression probability
       falls in the uncertain band [--low-risk-threshold, --high-risk-threshold].
       Auto-classified high/low rows get a feature-based plain-English explanation
       instead, so the per-query CSV is fully explainable for every row.

TRAINED MODEL
-------------
  LogisticRegression trained ONCE on a weak-label set:
    features = [top1_sim, margin, avg_pairwise_topk_sim]
    label    = 1 if retrieval top1 was WRONG (proxy for hallucination risk)
  Saved to --model-path; reloaded on subsequent runs (--retrain to force rebuild).

OUTPUT
------
  - hallucination_detect_results.csv       : one row per query, all signals + label
  - hallucination_detect_per_query_diag.csv: same + sentence_verdicts_json + risk_reason
  - hallucination_detect_metrics.txt       : full metrics / latency / distributions report
  - hallucination_risk_model.joblib        : trained LR model

USAGE
-----
  # offline smoke test (zero API calls):
  python hallucination_detect.py Datasets/site24x7_Dataset.csv \\
      Datasets/ADMIN_API/site24x7_Admin_API.xlsx --mock --n-eval 20 --n-train 100

  # real run (credentials auto-read from .env):
  python hallucination_detect.py Datasets/site24x7_Dataset.csv \\
      Datasets/ADMIN_API/site24x7_Admin_API.xlsx \\
      --extra-descriptions Datasets/reports_synthetic_descriptions.csv \\
      --n-eval 50 --n-train 2000 --save-per-query
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---- import from existing codebase (read-only) ----
from rag_eval import (
    build_corpus_and_eval,
    embed_texts,
    get_embeddings_mock,
)
from compare_doc_strategies import (
    build_text_doc_description,
    load_doc_descriptions,
    load_extra_descriptions,
    make_variant_corpus,
)


# ---------------------------------------------------------------------------
# PROMPT CONSTANTS
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = (
    "You answer questions about a monitoring/reporting API using ONLY the "
    "provided context passages. If the context does not contain enough "
    "information to answer confidently, say so explicitly rather than "
    "guessing. Keep the answer to 1-3 sentences. Do not invent endpoint "
    "names, parameters, or details not present in the context."
)

# Single combined LLM call: returns BOTH context_relevant and sentence_verdicts
# so we pay for at most one chat completion per gated query, not two.
COMBINED_JUDGE_SYSTEM_PROMPT = (
    "You are a strict fact-checker for API documentation RAG responses.\n"
    "You will be given:\n"
    "  QUERY: the user's original question\n"
    "  CONTEXT: retrieved documentation passages (the ground truth)\n"
    "  RESPONSE: a generated answer to evaluate\n\n"
    "Respond with EXACTLY one JSON object with TWO fields:\n"
    '  "context_relevant": one of "true", "false", or "partial"\n'
    "    true    = the CONTEXT actually answers the QUERY specifically\n"
    "    false   = context is topically related but wrong endpoint/feature\n"
    "    partial = partly answers but is incomplete or mixed\n"
    '  "sentence_verdicts": JSON array, one entry per RESPONSE sentence:\n'
    '    [{"sentence": "...", "verdict": "supported"|"unsupported"|"partial",\n'
    '      "reason": "short phrase"}, ...]\n'
    "    supported   = context explicitly contains or clearly implies this claim\n"
    "    unsupported = claim not found in context (may be true generally but\n"
    "                  is not stated in the provided passages)\n"
    "    partial     = context supports part of the sentence but not all\n\n"
    "IMPORTANT — SHORT FRAGMENT QUERIES:\n"
    "If the query is a short technical fragment (a few words, jargon, not a full "
    "natural-language question), do NOT assume the retrieved endpoint is correct "
    "just because it shares topical or domain vocabulary with the fragment. "
    "Explicitly check whether the RETRIEVED ENDPOINT'S PATH AND SPECIFIC TERMS match "
    "the fragment's specific terms. A shared general topic (e.g. both mention 'tags' "
    "or 'attribute groups') is NOT sufficient — the specific noun/resource referenced "
    "must actually match what the endpoint returns. Apply stricter scrutiny: prefer "
    "'false' or 'partial' over 'true' when the match is only topical.\n\n"
    "Respond with ONLY the JSON object. No markdown, no code fences, no other text."
)


# ---------------------------------------------------------------------------
# UTILITY HELPERS
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


_FRAGMENT_QUESTION_WORDS = re.compile(
    r"\b(how|what|where|when|why|which|who|can i|do i|is there|show me|list|get me)\b",
    re.IGNORECASE,
)

def is_query_fragment(query: str) -> bool:
    """Return True when the query looks like a short technical fragment rather than
    a full natural-language question.
    Heuristics:
      - 4 or fewer whitespace-delimited tokens, OR
      - no recognised question word / verb pattern AND no '?' character.
    Short fragment queries fool the LLM judge because endpoint jargon overlaps
    with the query tokens even when the endpoint is wrong.
    """
    tokens = query.split()
    if len(tokens) <= 4:
        return True
    if '?' not in query and not _FRAGMENT_QUESTION_WORDS.search(query):
        return True
    return False


def split_sentences(text: str) -> list:
    text = str(text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
    return an @ bn.T


def _pct(n: int, total: int) -> str:
    return f"{n/max(total,1)*100:.1f}%"


# ---------------------------------------------------------------------------
# EMBEDDING DISPATCH
# ---------------------------------------------------------------------------

def embed_dispatch(texts: list, client, args, cache_dir: Path) -> np.ndarray:
    if not texts:
        return np.zeros((0, 64), dtype=np.float32)
    if args.mock:
        return get_embeddings_mock(texts)
    return embed_texts(
        client, args.embed_model, texts, cache_dir,
        use_cache=not args.no_cache,
        batch_size=args.embed_batch_size,
        request_timeout=args.timeout,
    )


# ---------------------------------------------------------------------------
# SIGNAL 1 — retrieval confidence
# ---------------------------------------------------------------------------

def compute_retrieval_features(query_vec: np.ndarray,
                                corpus_norm: np.ndarray,
                                top_k: int) -> tuple:
    """
    Returns (top_k_idx, top1_sim, margin, avg_pairwise_topk_sim,
             n_candidates_within_margin).

    top1_sim                 : cosine similarity of the best hit
    margin                   : top1_sim - top2_sim (narrow = ambiguous)
    avg_pairwise             : mean pairwise cosine sim AMONG top-k chunks
                               (cluster tightness)
    n_candidates_within_margin : count of top-k chunks whose similarity is
                               within 0.02 of top1_sim (near-ties with the
                               winner; more near-ties = less certain retrieval
                               even if margin alone looks ok)
    """
    q_norm = query_vec / max(float(np.linalg.norm(query_vec)), 1e-9)
    sims = corpus_norm @ q_norm
    top_k_idx = np.argsort(-sims)[:top_k]
    top_sims = sims[top_k_idx]

    top1_sim = float(top_sims[0]) if len(top_sims) >= 1 else 0.0
    margin   = float(top_sims[0] - top_sims[1]) if len(top_sims) >= 2 else 1.0

    topk_vecs = corpus_norm[top_k_idx]          # already L2-normalised
    pairwise  = topk_vecs @ topk_vecs.T         # (k, k)
    k = len(top_k_idx)
    if k < 2:
        avg_pairwise = 1.0
    else:
        mask = np.triu(np.ones((k, k), dtype=bool), k=1)
        avg_pairwise = float(pairwise[mask].mean())

    # Count how many top-k chunks are within 0.02 of the best match
    # (excludes top1 itself, so range is [1, top_k-1])
    n_candidates_within_margin = int((
        (top_sims[1:] >= top1_sim - 0.02).sum()
    )) if len(top_sims) > 1 else 0

    return top_k_idx, top1_sim, margin, avg_pairwise, n_candidates_within_margin


# ---------------------------------------------------------------------------
# SIGNAL 2 — embedding-based groundedness
# ---------------------------------------------------------------------------

def embedding_groundedness(response: str, context_chunks: list,
                            client, args, cache_dir: Path,
                            threshold: float) -> tuple:
    """
    Returns (per_sentence_rows, frac_unsupported_embedding).
    For each response sentence: max cosine sim to any context chunk.
    Below threshold = flagged as possibly unsupported.
    """
    sentences = split_sentences(response)
    if not sentences or not context_chunks:
        return [], 0.0

    sent_vecs  = embed_dispatch(sentences, client, args, cache_dir)
    chunk_vecs = embed_dispatch(context_chunks, client, args, cache_dir)
    sims       = cosine_sim_matrix(sent_vecs, chunk_vecs)
    max_sims   = sims.max(axis=1)

    rows = [
        {"sentence": s,
         "max_sim_to_context": float(m),
         "embedding_supported": bool(m >= threshold)}
        for s, m in zip(sentences, max_sims)
    ]
    frac_unsup = 1.0 - (sum(r["embedding_supported"] for r in rows) / len(rows))
    return rows, frac_unsup


# ---------------------------------------------------------------------------
# SIGNAL 3 — LLM judge (gated, single combined call)
# ---------------------------------------------------------------------------

def llm_judge_real(client, model: str, query: str,
                   response: str, context_chunks: list,
                   query_is_fragment: bool = False) -> dict:
    """
    ONE combined chat call returning context_relevant + sentence_verdicts.
    When query_is_fragment=True an explicit NOTE is prepended to the user
    message instructing the judge to apply extra path-level scrutiny.
    Falls back to a safe dict on any API or parse error.
    """
    context  = "\n---\n".join(context_chunks)
    numbered = " ".join(split_sentences(response))
    fragment_note = (
        "NOTE: this query is a short technical fragment - apply extra scrutiny "
        "per the system instructions above. Do not accept topical overlap alone; "
        "the specific resource/path in the retrieved endpoint must match the "
        "fragment's specific terms.\n\n"
        if query_is_fragment else ""
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COMBINED_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"{fragment_note}"
                    f"QUERY: {query}\n\n"
                    f"CONTEXT:\n{context}\n\n"
                    f"RESPONSE:\n{numbered}"
                )},
            ],
            temperature=0.0,
        )
        raw    = _strip_code_fences(resp.choices[0].message.content)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"expected JSON object, got: {parsed!r}")
        cr = str(parsed.get("context_relevant", "partial")).lower()
        if cr not in ("true", "false", "partial"):
            cr = "partial"
        return {"context_relevant": cr,
                "sentence_verdicts": parsed.get("sentence_verdicts", [])}
    except Exception as e:
        print(f"    llm-judge failed: {type(e).__name__}: {e}")
        return {
            "context_relevant": "partial",
            "sentence_verdicts": [
                {"sentence": s, "verdict": "unknown", "reason": str(e)[:60]}
                for s in split_sentences(response)
            ],
        }


def llm_judge_mock(query: str, response: str, context_chunks: list,
                   query_is_fragment: bool = False) -> dict:
    """
    Deterministic offline mock — word-overlap heuristics only.
    Not a real judge; exercises the code path for --mock smoke tests.

    Fragment-skepticism: when query_is_fragment=True, a higher overlap
    threshold is required before the mock returns 'true', and 'false' is
    returned more readily — mimicking the tighter scrutiny the real judge
    applies via the updated system prompt.
    """
    ctx_words = set()
    for c in context_chunks:
        ctx_words |= set(re.findall(r"[a-z0-9]+", c.lower()))
    q_words   = set(re.findall(r"[a-z0-9]+", query.lower()))
    cr_ratio  = len(q_words & ctx_words) / max(len(q_words), 1)

    if query_is_fragment:
        # Stricter thresholds: shared vocabulary is not enough for short fragments
        cr = "true" if cr_ratio > 0.65 else ("partial" if cr_ratio > 0.35 else "false")
    else:
        cr = "true" if cr_ratio > 0.4 else ("partial" if cr_ratio > 0.2 else "false")

    verdicts = []
    for s in split_sentences(response):
        s_words  = set(re.findall(r"[a-z0-9]+", s.lower()))
        overlap  = len(s_words & ctx_words) / max(len(s_words), 1)
        verdict  = "supported" if overlap > 0.5 else "unsupported"
        verdicts.append({"sentence": s, "verdict": verdict,
                         "reason": f"{overlap:.0%} word overlap"})
    return {"context_relevant": cr, "sentence_verdicts": verdicts}


def judge_frac_unsupported(sentence_verdicts: list) -> float:
    """Scalar risk score from verdicts. Unsupported=1.0, partial=0.5."""
    if not sentence_verdicts:
        return 0.0
    n_u = sum(1 for v in sentence_verdicts if v.get("verdict") == "unsupported")
    n_p = sum(1 for v in sentence_verdicts if v.get("verdict") == "partial")
    return (n_u + 0.5 * n_p) / len(sentence_verdicts)


# ---------------------------------------------------------------------------
# RESPONSE GENERATION
# ---------------------------------------------------------------------------

def generate_response_real(client, model: str, query: str,
                            context_chunks: list) -> str:
    context = "\n---\n".join(context_chunks)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    generation failed for '{query[:60]}': {e}")
        return ""


def generate_response_mock(query: str, context_chunks: list) -> str:
    """
    Deterministic fake response for --mock mode.
    Appends one made-up unsupported detail so the groundedness
    check has a realistic false-positive sentence to find.
    """
    base = context_chunks[0][:80] if context_chunks else "no information available"
    return (f"Based on the documentation, {base}. "
            f"It also supports real-time CSV export by default.")


# ---------------------------------------------------------------------------
# EXPLAINABILITY — plain-English reason for EVERY row
# ---------------------------------------------------------------------------

def make_risk_reason(lr_prob: float, top1_sim: float, margin: float,
                     avg_pairwise: float, judge_fired: bool,
                     judge_result: dict | None, final_risk: str,
                     low_thresh: float, high_thresh: float) -> str:
    """
    Returns a human-readable one-sentence explanation of the final label.
    Non-judge rows get a feature-based explanation so the per-query CSV
    is fully explainable for every row, not just the ones where the judge fired.
    """
    if not judge_fired:
        if final_risk == "low":
            strong = []
            if top1_sim >= 0.65:
                strong.append(f"top1_sim={top1_sim:.3f} (strong match)")
            if margin >= 0.05:
                strong.append(f"margin={margin:.3f} (clear top1)")
            detail = ("; ".join(strong) if strong
                      else f"top1_sim={top1_sim:.3f}, margin={margin:.3f}")
            return (f"Auto-low (LR p={lr_prob:.2f} < {low_thresh}): {detail}. "
                    f"Model confident no hallucination risk.")
        else:  # auto-high
            weak = []
            if top1_sim < 0.45:
                weak.append(f"top1_sim={top1_sim:.3f} (low — retriever likely off-target)")
            elif top1_sim < 0.60:
                weak.append(f"top1_sim={top1_sim:.3f} (moderate — borderline relevance)")
            if margin < 0.02:
                weak.append(f"margin={margin:.4f} (very narrow — retrieval highly ambiguous)")
            elif margin < 0.05:
                weak.append(f"margin={margin:.3f} (narrow — retrieval ambiguous)")
            detail = ("; ".join(weak) if weak
                      else f"top1_sim={top1_sim:.3f}, margin={margin:.3f}")
            return (f"Auto-high (LR p={lr_prob:.2f} > {high_thresh}): {detail}. "
                    f"Model confident of hallucination risk without needing LLM judge.")

    # Judge fired — explain from judge output
    cr = (judge_result or {}).get("context_relevant", "unknown")
    verdicts = (judge_result or {}).get("sentence_verdicts", [])
    n_total  = len(verdicts)
    n_unsup  = sum(1 for v in verdicts if v.get("verdict") == "unsupported")
    n_part   = sum(1 for v in verdicts if v.get("verdict") == "partial")
    n_supp   = n_total - n_unsup - n_part

    cr_text = {
        "true":    "context answers query specifically",
        "false":   "context is wrong endpoint/feature for this query",
        "partial": "context partially answers query",
    }.get(cr, f"context_relevant={cr}")

    sent_text = (f"{n_supp}/{n_total} supported, "
                 f"{n_part}/{n_total} partial, "
                 f"{n_unsup}/{n_total} unsupported")
    return (f"LLM judge (LR p={lr_prob:.2f} in uncertain band): {cr_text}. "
            f"Sentence verdicts: {sent_text}. Final label: {final_risk}.")


# ---------------------------------------------------------------------------
# LOGISTIC REGRESSION — model training and inference
# ---------------------------------------------------------------------------

FALLBACK_LOW_SIM   = 0.45   # top1_sim below this -> elevated risk (sklearn fallback)
FALLBACK_MARGIN    = 0.03   # margin below this   -> elevated risk (sklearn fallback)


def _try_import_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        try:
            import joblib
            return LogisticRegression, joblib
        except ImportError:
            return LogisticRegression, None
    except ImportError:
        return None, None


def _save_model(model, path: Path):
    _, joblib = _try_import_sklearn()
    if joblib is not None:
        joblib.dump(model, path)
        print(f"  Model saved (joblib) -> {path}")
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Model saved (pickle) -> {path}")


def _load_model(path: Path):
    """Load model and do a quick sanity-check predict to catch sklearn version
    mismatches before they crash mid-run. Automatically detects 5-feature vs 9-feature model."""
    _, joblib = _try_import_sklearn()
    if joblib is not None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress InconsistentVersionWarning
            model = joblib.load(path)
    else:
        with open(path, "rb") as f:
            model = pickle.load(f)
    # Smoke-test the loaded model using its expected n_features_in_
    n_feats = getattr(model, "n_features_in_", 5)
    test_X = np.zeros((1, n_feats), dtype=np.float32)
    test_X[0, 0] = 0.5
    test_X[0, 1] = 0.05
    try:
        _ = model.predict_proba(test_X)
    except (AttributeError, ValueError) as e:
        raise RuntimeError(
            f"Loaded model is incompatible (error: {e}). "
            f"Please check model path and dependencies."
        ) from e
    return model


def build_weak_label_training_set(df: pd.DataFrame,
                                   corpus_df: pd.DataFrame,
                                   corpus_norm: np.ndarray,
                                   n_train: int, top_k: int,
                                   seed: int, exclude_queries: set,
                                   client, args,
                                   cache_dir: Path) -> pd.DataFrame:
    """
    Sample n_train queries (non-overlapping with the flagging eval set),
    compute retrieval features (5 total), label = 1 if top1 was WRONG.

    Features:
      top1_sim, margin, avg_pairwise_topk_sim  -- retrieval confidence signals
      query_token_count                        -- distinguishes fragments from questions
      n_candidates_within_margin               -- near-tie count among top-k

    Retrieval failure is the weak-label proxy for hallucination risk.
    """
    true_df = df[df["markedCorrect"] == True].copy()
    true_df["key"] = list(zip(true_df["endpoint"], true_df["method"]))

    query_groups = (
        true_df.groupby("query")["key"]
        .apply(lambda keys: sorted(set(keys)))
        .reset_index()
        .rename(columns={"key": "valid_keys"})
    )
    query_groups = query_groups[~query_groups["query"].isin(exclude_queries)]

    n = min(n_train, len(query_groups))
    if n < n_train:
        print(f"  WARNING: only {n} non-overlapping training queries available "
              f"(requested {n_train}). Using all.")
    train_df = (query_groups
                .sample(n=n, random_state=seed + 9999)
                .reset_index(drop=True))

    print(f"  Embedding {len(train_df)} training queries...")
    train_vecs  = embed_dispatch(train_df["query"].tolist(), client, args, cache_dir)
    corpus_keys = list(corpus_df["key"])

    rows = []
    for i, row in train_df.iterrows():
        q_vec = train_vecs[i]
        _, top1_sim, margin, avg_pairwise, n_cand = compute_retrieval_features(
            q_vec, corpus_norm, top_k)
        top1_key   = corpus_keys[
            int(np.argmax(corpus_norm @ (q_vec /
                max(float(np.linalg.norm(q_vec)), 1e-9))))]
        valid_keys = set(map(tuple, row["valid_keys"]))
        label      = 0 if top1_key in valid_keys else 1
        query_tok  = len(row["query"].split())

        rows.append({"query": row["query"],
                     "top1_sim": top1_sim,
                     "margin": margin,
                     "avg_pairwise_topk_sim": avg_pairwise,
                     "query_token_count": query_tok,
                     "n_candidates_within_margin": n_cand,
                     "label": label})
    return pd.DataFrame(rows)


def train_logistic_regression(train_feat_df: pd.DataFrame,
                               use_class_weight: bool = True):
    """
    Fit LogisticRegression on 5 features:
      [top1_sim, margin, avg_pairwise_topk_sim,
       query_token_count, n_candidates_within_margin]

    use_class_weight : if True (default) set class_weight='balanced' to
      compensate for the ~79/21 correct/wrong class imbalance. Pass False
      (via --no-class-weight) to disable for comparison runs.

    Returns (model, sklearn_was_available).
    """
    LogisticRegression, _ = _try_import_sklearn()
    if LogisticRegression is None:
        print(
            "  WARNING: scikit-learn not available. Falling back to hand-picked\n"
            "  threshold rules. Install scikit-learn for the trained LR model.\n"
            f"  Fallback: top1_sim < {FALLBACK_LOW_SIM} OR margin < {FALLBACK_MARGIN} -> high risk."
        )
        return None, False

    feature_cols = ["top1_sim", "margin", "avg_pairwise_topk_sim",
                    "query_token_count", "n_candidates_within_margin"]
    X = train_feat_df[feature_cols].values
    y = train_feat_df["label"].values

    pos_rate = y.mean()
    cw = "balanced" if use_class_weight else None
    print(f"  Label distribution: {int(y.sum())}/{len(y)} risky "
          f"({pos_rate:.1%} positive rate)  class_weight={cw!r}")

    model = LogisticRegression(
        class_weight=cw,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    print(f"  In-sample accuracy: {acc:.3f}")
    return model, True


def predict_risk_probability(model, features: list) -> float:
    """
    Return P(label=1 | features).
    features is a list of 5 or 9 feature values depending on model.n_features_in_.
    Falls back to a hand-crafted rule if model=None.
    """
    if model is None:
        top1_sim, margin = features[0], features[1]
        return 0.75 if (top1_sim < FALLBACK_LOW_SIM or margin < FALLBACK_MARGIN) else 0.2

    X = np.array([features], dtype=np.float32)
    proba   = model.predict_proba(X)
    classes = list(model.classes_)
    idx     = classes.index(1) if 1 in classes else -1
    return float(proba[0, idx]) if idx >= 0 else 0.5


def classify_risk_cascade(lr_prob: float, judge_result: dict | None,
                           low_thresh: float, high_thresh: float,
                           judge_was_fired: bool) -> str:
    """
    Three-tier cascade:
      p < low_thresh   -> 'low'  (skip judge)                     [only when judge did NOT fire]
      p > high_thresh  -> 'high' (skip judge)                     [only when judge did NOT fire]
      in between, OR judge was force-fired for any other reason
      (e.g. the groundedness OR-gate)                             -> use judge output

    IMPORTANT: judge_was_fired is checked FIRST. If the judge actually ran —
    for whatever reason, including being force-fired despite lr_prob being
    below low_thresh (see the groundedness OR-gate in main()) — its verdict
    always decides the label. The lr_prob threshold shortcuts below only
    apply when the judge was genuinely never consulted. Previously the
    lr_prob<low_thresh / >high_thresh checks ran unconditionally BEFORE this
    check, which silently discarded a real judge verdict (and the API call
    that produced it) any time the judge was force-fired on a low-lr_prob
    case -- exactly the case the groundedness gate exists to catch.
    """
    if not judge_was_fired:
        if lr_prob < low_thresh:
            return "low"
        if lr_prob > high_thresh:
            return "high"
        # shouldn't normally be reached given how main() routes, but keep a
        # safe fallback rather than raising if this is ever called directly
        return "low" if lr_prob < 0.5 else "medium"

    if judge_result is None:
        return "low" if lr_prob < 0.5 else "medium"

    cr      = judge_result.get("context_relevant", "partial")
    frac_u  = judge_frac_unsupported(judge_result.get("sentence_verdicts", []))

    if cr == "false" or frac_u >= 0.5:
        return "high"
    if cr == "partial" or frac_u > 0.0:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# METRICS & REPORTING HELPERS
# ---------------------------------------------------------------------------

def feature_distribution_table(values: list, label: str) -> str:
    a = np.array(values)
    return (f"  {label:30s}  "
            f"min={a.min():.4f}  "
            f"p25={np.percentile(a,25):.4f}  "
            f"p50={np.percentile(a,50):.4f}  "
            f"p75={np.percentile(a,75):.4f}  "
            f"max={a.max():.4f}")


def write_metrics_report(report_lines: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nFull metrics report saved -> {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Hallucination detection for the Site24x7 RAG pipeline.\n"
            "Three signals (retrieval confidence + embedding groundedness + LLM judge)\n"
            "combined via a trained LogisticRegression with an uncertainty-gated cascade."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # positional
    parser.add_argument("csv_path",  help="Path to site24x7_Dataset.csv")
    parser.add_argument("xlsx_path", help="Path to site24x7_Admin_API.xlsx")

    # data
    parser.add_argument("--extra-descriptions", default=None,
        help="Optional CSV (endpoint, method, description) merged into doc_map.")
    parser.add_argument("--responses-csv", default=None,
        help="CSV with 'query' and 'response' columns to audit. "
             "If omitted, the script generates its own responses.")

    # sizes
    parser.add_argument("--n-eval",     type=int, default=50)
    parser.add_argument("--n-train",    type=int, default=500)
    parser.add_argument("--n-examples", type=int, default=2)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--top-k",      type=int, default=5)

    # API
    parser.add_argument("--base-url",   default=None)
    parser.add_argument("--api-key",    default=None)
    parser.add_argument("--embed-model",default="azure:primary/s247-textembedding-3l")
    parser.add_argument("--chat-model", default="azure:primary/gpt-4.1-mini")

    # thresholds
    parser.add_argument("--groundedness-threshold", type=float, default=0.5,
        help="Per-sentence cosine-sim cutoff for signal 2 (embedding groundedness).")
    parser.add_argument("--groundedness-gate-threshold", type=float, default=0.10,
        help="If frac_unsupported_embedding exceeds this, force the LLM judge to "
             "fire even when lr_risk_probability is below --low-risk-threshold. "
             "This is what catches correctly-retrieved-but-hallucinated responses "
             "that the LR model (retrieval-only features) can't see on its own. "
             "Previously this signal was computed but not wired into routing here.")
    parser.add_argument("--low-risk-threshold",     type=float, default=0.3)
    parser.add_argument("--high-risk-threshold",    type=float, default=0.7)

    # model
    parser.add_argument("--model-path", default=None,
        help="Save/load path for the trained LR model "
             "(default: hallucination_risk_model.joblib next to csv_path).")
    parser.add_argument("--retrain", action="store_true",
        help="Force retraining even if a saved model already exists.")

    # infrastructure
    parser.add_argument("--mock",           action="store_true",
        help="Fully offline: fake embeddings, fake LLM calls. Pipeline test only.")
    parser.add_argument("--cache-dir",      default=None)
    parser.add_argument("--no-cache",       action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=20)
    parser.add_argument("--timeout",        type=float, default=60.0)

    # output
    parser.add_argument("--save-per-query", action="store_true",
        help="Save full per-query diagnostics CSV (all features + sentence verdicts + reason).")

    args = parser.parse_args()

    # ---- wall-clock timer dict ----
    T = {}
    T["pipeline_start"] = time.time()

    # ---- paths ----
    csv_path   = Path(args.csv_path)
    xlsx_path  = Path(args.xlsx_path)
    for p in (csv_path, xlsx_path):
        if not p.exists():
            sys.exit(f"File not found: {p}")

    model_path  = Path(args.model_path) if args.model_path \
                  else csv_path.parent / "hallucination_risk_model.joblib"
    out_dir     = csv_path.parent

    # ---- read credentials from .env ----
    env_file = csv_path.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    base_url = args.base_url or os.environ.get("PROXY_BASE_URL")
    api_key  = args.api_key  or os.environ.get("PROXY_API_KEY")

    # ---- client ----
    client = None
    if args.mock:
        print("=" * 70)
        print("MOCK MODE: deterministic hash-based fake embeddings + fake LLM calls.")
        print("Full pipeline (training + flagging) exercised with zero API calls.")
        print("=" * 70)
        print()
    else:
        if not base_url or not api_key:
            sys.exit(
                "Provide --base-url and --api-key, or set PROXY_BASE_URL / PROXY_API_KEY "
                "in .env, or use --mock for an offline smoke test."
            )
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=args.timeout)

    # ---- load dataset ----
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    if df["markedCorrect"].dtype == object:
        df["markedCorrect"] = (
            df["markedCorrect"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    df["markedCorrect"] = df["markedCorrect"].astype(bool)
    print(f"  {len(df):,} rows, {df['markedCorrect'].sum():,} marked correct, "
          f"{df[['endpoint','method']].drop_duplicates().shape[0]:,} unique (endpoint, method) pairs\n")

    # ---- description map ----
    print("Loading description file (doc_description strategy)...")
    doc_map  = load_doc_descriptions(xlsx_path)
    all_keys = set(zip(df["endpoint"], df["method"]))
    cov      = len(set(doc_map.keys()) & all_keys)
    print(f"  xlsx covers {cov}/{len(all_keys)} pairs ({cov/len(all_keys)*100:.1f}%)")

    if args.extra_descriptions:
        ep = Path(args.extra_descriptions)
        if not ep.exists():
            sys.exit(f"File not found: {ep}")
        extra_map = load_extra_descriptions(ep)
        for k, v in extra_map.items():
            doc_map.setdefault(k, v)
        cov2 = len(set(doc_map.keys()) & all_keys)
        print(f"  + extra: combined coverage {cov2}/{len(all_keys)} "
              f"({cov2/len(all_keys)*100:.1f}%)")
    print()

    cache_dir = Path(args.cache_dir) if args.cache_dir \
                else csv_path.parent / ".rag_cache"

    # ---- build corpus ----
    print("Building corpus (doc_description strategy)...")
    t0 = time.time()
    base_corpus_df, eval_df_from_split = build_corpus_and_eval(
        df, args.n_eval, args.n_examples, args.seed)
    true_df = df[df["markedCorrect"] == True].copy()
    true_df["key"] = list(zip(true_df["endpoint"], true_df["method"]))
    eval_queries_set  = set(eval_df_from_split["query"])
    enrichment_pool   = true_df[~true_df["query"].isin(eval_queries_set)]

    corpus_df    = make_variant_corpus(
        base_corpus_df, df, doc_map, args.n_examples,
        enrichment_pool, build_text_doc_description)
    corpus_texts = corpus_df["text"].tolist()
    corpus_keys  = list(corpus_df["key"])
    print(f"  {len(corpus_df):,} chunks\n")

    print("Embedding corpus...")
    corpus_vecs  = embed_dispatch(corpus_texts, client, args, cache_dir)
    corpus_norm  = corpus_vecs / np.clip(
        np.linalg.norm(corpus_vecs, axis=1, keepdims=True), 1e-9, None)
    T["corpus_embedding"] = time.time() - t0
    print()

    # ---- build ground-truth lookup (query -> set of valid keys) ----
    # Used to compute top1_correct for eval-mode queries (ground-truth alignment)
    gt_lookup: dict[str, set] = {}
    for _, row in eval_df_from_split.iterrows():
        gt_lookup[row["query"]] = set(map(tuple, row["valid_keys"]))

    # ---- determine queries to flag ----
    if args.responses_csv:
        resp_df = pd.read_csv(args.responses_csv)
        missing = {"query", "response"} - set(resp_df.columns)
        if missing:
            sys.exit(f"--responses-csv missing columns: {missing}")
        flag_queries       = resp_df["query"].tolist()
        provided_responses = resp_df["response"].tolist()
        print(f"Auditing {len(flag_queries)} provided (query, response) pairs "
              f"from {args.responses_csv}\n")
    else:
        flag_queries       = eval_df_from_split["query"].tolist()
        provided_responses = None
        print(f"No --responses-csv: generating + flagging {len(flag_queries)} "
              f"responses for the freshly sampled eval set (seed={args.seed})\n")

    # ---- embed flagging queries ----
    print(f"Embedding {len(flag_queries)} flagging queries...")
    t0 = time.time()
    flag_vecs = embed_dispatch(flag_queries, client, args, cache_dir)
    T["query_embedding"] = time.time() - t0
    print()

    # ---- train or load LR model ----
    model        = None
    model_is_new = False

    if model_path.exists() and not args.retrain:
        print(f"Loading saved risk model from {model_path}...")
        try:
            model = _load_model(model_path)
            print("  Model loaded.\n")
        except Exception as e:
            print(f"  WARNING: load failed ({e}). Retraining...\n")
            model = None

    if model is None:
        if args.retrain and model_path.exists():
            print("--retrain set: rebuilding model...\n")
        else:
            print(f"No saved model at {model_path}: training now...\n")

        print(f"Building weak-label training set "
              f"({args.n_train} queries, non-overlapping with flagging set)...")
        t0 = time.time()
        train_feat_df = build_weak_label_training_set(
            df=df, corpus_df=corpus_df, corpus_norm=corpus_norm,
            n_train=args.n_train, top_k=args.top_k, seed=args.seed,
            exclude_queries=set(flag_queries),
            client=client, args=args, cache_dir=cache_dir)
        print(f"Training LogisticRegression on {len(train_feat_df)} examples...")
        use_cw = not getattr(args, 'no_class_weight', False)
        model, sklearn_ok = train_logistic_regression(train_feat_df,
                                                      use_class_weight=use_cw)
        T["model_training"] = time.time() - t0
        model_is_new = True
        if model is not None:
            _save_model(model, model_path)
        print()

    # ---- print coefficients for new model ----
    if model_is_new and model is not None:
        feature_names = ["top1_sim", "margin", "avg_pairwise_topk_sim",
                         "query_token_count", "n_candidates_within_margin"]
        coefs         = model.coef_[0]
        intercept     = model.intercept_[0]
        print("=" * 60)
        print("TRAINED MODEL COEFFICIENTS  (LogisticRegression, label=1 means risky)")
        print("=" * 60)
        for name, c in zip(feature_names, coefs):
            print(f"  {name:30s}: {c:+.4f}")
        print(f"  {'intercept':30s}: {intercept:+.4f}")
        print()
        abs_coefs    = np.abs(coefs)
        dom_idx      = int(np.argmax(abs_coefs))
        dom          = feature_names[dom_idx]
        direction    = ("higher -> LESS risky" if coefs[dom_idx] < 0
                        else "higher -> MORE risky")
        print(f"  Most influential signal: '{dom}' ({direction})")
        print(
            "\n  INTERPRETING COEFFICIENTS:\n"
            "  - Negative coef on top1_sim         : lower similarity = higher risk.\n"
            "  - Negative coef on margin            : narrow top1-top2 gap = higher risk.\n"
            "  - avg_pairwise_topk_sim              : cluster tightness of top-k.\n"
            "  - query_token_count                  : short fragment queries (low count)\n"
            "    tend to be harder to retrieve correctly -> positive coef expected.\n"
            "  - n_candidates_within_margin         : many near-ties with the best result\n"
            "    indicate an ambiguous retrieval even if raw margin looks ok.\n"
        )
        print("=" * 60)
        print()

    # ---- per-query flagging loop ----
    print(f"Flagging {len(flag_queries)} queries...")
    print("-" * 70)

    result_rows           = []
    judge_count           = 0
    judge_forced_count    = 0   # subset of judge_count fired via the groundedness gate, not the uncertain band
    auto_low_count        = 0
    auto_high_count       = 0
    t_judge_total         = 0.0
    t_generation          = 0.0
    t_groundedness        = 0.0
    t_per_query           = []

    for i, query in enumerate(flag_queries):
        tq = time.time()
        q_vec = flag_vecs[i]

        # --- signal 1 ---
        top_k_idx, top1_sim, margin, avg_pairwise, n_cand = compute_retrieval_features(
            q_vec, corpus_norm, args.top_k)
        context_chunks  = [corpus_texts[j] for j in top_k_idx]
        top1_key        = corpus_keys[top_k_idx[0]]
        top1_endpoint   = corpus_df["endpoint"].iloc[top_k_idx[0]]
        top1_method     = corpus_df["method"].iloc[top_k_idx[0]]

        # ground-truth alignment (eval-mode only)
        valid_keys      = gt_lookup.get(query, None)
        top1_correct    = (top1_key in valid_keys) if valid_keys is not None else None

        # fragment heuristic for judge prompt
        fragment        = is_query_fragment(query)

        # --- response ---
        tg = time.time()
        if provided_responses is not None:
            response = str(provided_responses[i])
        elif args.mock:
            response = generate_response_mock(query, context_chunks)
        else:
            response = generate_response_real(
                client, args.chat_model, query, context_chunks)
        t_generation += time.time() - tg

        # --- signal 2: embedding groundedness ---
        tgs = time.time()
        emb_rows, frac_unsup_emb = embedding_groundedness(
            response, context_chunks, client, args,
            cache_dir, args.groundedness_threshold)
        t_groundedness += time.time() - tgs

        # --- LR probability (5 features) ---
        query_tok = len(query.split())
        features  = [top1_sim, margin, avg_pairwise, query_tok, n_cand]
        lr_prob   = predict_risk_probability(model, features)

        # --- cascade ---
        # Two independent reasons the judge fires:
        #   1. lr_prob falls in the uncertain band (LR itself isn't confident)
        #   2. groundedness_bad: frac_unsup_emb crossed the gate threshold --
        #      the response itself looks unsupported even though LR (which
        #      only sees retrieval features) is confident retrieval was fine.
        #      This is what catches correctly-retrieved-but-hallucinated
        #      responses, which LR alone is structurally blind to.
        judge_result      = None
        judge_fired       = False
        judge_forced      = False
        groundedness_bad  = frac_unsup_emb > args.groundedness_gate_threshold

        if lr_prob < args.low_risk_threshold and not groundedness_bad:
            final_risk = "low"
            auto_low_count += 1
        elif lr_prob > args.high_risk_threshold:
            # a bad retrieval is a bad retrieval regardless of groundedness
            final_risk = "high"
            auto_high_count += 1
        else:
            # genuinely uncertain band, OR force-fired by the groundedness
            # gate despite a low lr_prob
            judge_fired = True
            judge_count += 1
            if lr_prob < args.low_risk_threshold and groundedness_bad:
                judge_forced = True
                judge_forced_count += 1
            tj = time.time()
            if args.mock:
                judge_result = llm_judge_mock(query, response, context_chunks,
                                              query_is_fragment=fragment)
            else:
                judge_result = llm_judge_real(
                    client, args.chat_model, query, response, context_chunks,
                    query_is_fragment=fragment)
            t_judge_total += time.time() - tj
            final_risk = classify_risk_cascade(
                lr_prob, judge_result,
                args.low_risk_threshold, args.high_risk_threshold,
                judge_was_fired=True)

        # --- explainability: plain-English reason for EVERY row ---
        risk_reason = make_risk_reason(
            lr_prob, top1_sim, margin, avg_pairwise,
            judge_fired, judge_result, final_risk,
            args.low_risk_threshold, args.high_risk_threshold)

        # --- populate context_relevant for ALL rows ---
        # Judge-fired rows: from judge output
        # Auto-classified rows: feature-based label so CSV is never NaN
        if judge_result is not None:
            context_relevant_out = judge_result.get("context_relevant", "")
            j_frac_unsup         = judge_frac_unsupported(
                judge_result.get("sentence_verdicts", []))
            sentence_verdicts_json = json.dumps(
                judge_result.get("sentence_verdicts", []))
        else:
            # Auto-classified: derive a synthetic context_relevant from features
            if final_risk == "low":
                context_relevant_out = "auto_likely_true"
            else:
                context_relevant_out = (
                    "auto_likely_false" if top1_sim < 0.45
                    else "auto_likely_partial")
            j_frac_unsup           = None
            sentence_verdicts_json = ""

        t_per_query.append(time.time() - tq)

        row = {
            "query":                     query,
            "response":                  response,
            "top1_endpoint":             top1_endpoint,
            "top1_method":               top1_method,
            "top1_correct":              top1_correct,   # None when --responses-csv
            # signal 1
            "top1_sim":                  round(top1_sim, 4),
            "margin":                    round(margin, 4),
            "avg_pairwise_topk_sim":     round(avg_pairwise, 4),
            # signal 2
            "frac_unsupported_embedding": round(frac_unsup_emb, 3),
            "groundedness_gate_forced_judge": judge_forced,
            # LR
            "lr_risk_probability":       round(lr_prob, 4),
            # cascade
            "llm_judge_fired":           judge_fired,
            "context_relevant":          context_relevant_out,
            "judge_frac_unsupported":    (
                round(j_frac_unsup, 3) if j_frac_unsup is not None else ""),
            # final
            "hallucination_risk":        final_risk,
            "risk_reason":               risk_reason,
            # full sentence verdicts (when judge fired)
            "sentence_verdicts_json":    sentence_verdicts_json,
        }
        result_rows.append(row)

        judge_tag = "[JUDGE]" if judge_fired else "       "
        correct_tag = (
            "" if top1_correct is None
            else (" [GT:OK]" if top1_correct else " [GT:WRONG]")
        )
        print(
            f"  {judge_tag} [{final_risk.upper():6s}] "
            f"p={lr_prob:.2f} top1={top1_sim:.3f} margin={margin:.3f}  "
            f"\"{query[:52]}\"{correct_tag}"
        )

    T["total_flagging"] = sum(t_per_query)

    # ---- save CSVs ----
    result_df = pd.DataFrame(result_rows)
    out_csv   = out_dir / "hallucination_detect_results.csv"
    result_df.to_csv(out_csv, index=False)

    diag_csv = None
    if args.save_per_query:
        diag_csv = out_dir / "hallucination_detect_per_query_diag.csv"
        result_df.to_csv(diag_csv, index=False)
        print(f"\nFull diagnostics CSV saved -> {diag_csv}")

    T["pipeline_end"] = time.time()
    T["total_wall"]   = T["pipeline_end"] - T["pipeline_start"]

    # ---- build metrics report ----
    n_total      = len(result_df)
    risk_counts  = result_df["hallucination_risk"].value_counts()
    n_low        = risk_counts.get("low",    0)
    n_med        = risk_counts.get("medium", 0)
    n_high       = risk_counts.get("high",   0)

    # ground-truth alignment (eval mode only, where top1_correct is populated)
    gt_rows = result_df[result_df["top1_correct"].notna()].copy()
    gt_rows["top1_correct"] = gt_rows["top1_correct"].astype(bool)
    has_gt = len(gt_rows) > 0

    lines = []
    sep   = "=" * 70

    lines += [
        sep,
        "HALLUCINATION DETECTION METRICS REPORT",
        f"Run  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"CSV  : {csv_path}",
        f"Eval : {n_total} queries flagged  |  "
        f"n_train={args.n_train}  seed={args.seed}  top_k={args.top_k}",
        sep, "",
    ]

    # --- risk distribution ---
    lines += [
        "RISK DISTRIBUTION",
        "-" * 40,
        f"  Low    : {n_low:4d}  ({_pct(n_low, n_total):>6s})",
        f"  Medium : {n_med:4d}  ({_pct(n_med, n_total):>6s})",
        f"  High   : {n_high:4d}  ({_pct(n_high, n_total):>6s})",
        "",
    ]

    # --- cascade stats ---
    lines += [
        "CASCADE / COST STATS",
        "-" * 40,
        f"  Auto-low  (p < {args.low_risk_threshold})  : "
        f"{auto_low_count:4d}  ({_pct(auto_low_count, n_total):>6s})  [judge skipped]",
        f"  Judge fired (total)           : "
        f"{judge_count:4d}  ({_pct(judge_count, n_total):>6s})  [judge FIRED]",
        f"    - via uncertain band         : {judge_count - judge_forced_count:4d}",
        f"    - via groundedness gate      : {judge_forced_count:4d}  "
        f"(lr_prob was < {args.low_risk_threshold} but frac_unsupported_embedding "
        f"> {args.groundedness_gate_threshold} forced a check anyway)",
        f"  Auto-high (p > {args.high_risk_threshold})  : "
        f"{auto_high_count:4d}  ({_pct(auto_high_count, n_total):>6s})  [judge skipped]",
        f"",
        f"  LLM judge trigger rate        : {_pct(judge_count, n_total)}",
        f"  Cost savings vs always-judge  : "
        f"{_pct(n_total - judge_count, n_total)} fewer LLM calls",
        "",
    ]

    # --- signal feature distributions ---
    lines += ["SIGNAL FEATURE DISTRIBUTIONS  (across all flagged queries)", "-" * 70]
    for col, label in [
        ("top1_sim",              "top1_sim           (signal 1a)"),
        ("margin",                "margin             (signal 1b)"),
        ("avg_pairwise_topk_sim", "avg_pairwise_topk  (signal 1c)"),
        ("frac_unsupported_embedding", "frac_unsup_embed   (signal 2)"),
        ("lr_risk_probability",   "lr_risk_prob       (model out)"),
    ]:
        lines.append(feature_distribution_table(
            result_df[col].tolist(), label))
    lines.append("")

    # --- ground-truth alignment ---
    if has_gt:
        n_gt            = len(gt_rows)
        n_gt_correct    = gt_rows["top1_correct"].sum()
        n_gt_wrong      = n_gt - n_gt_correct
        # false positives: top1 correct but labelled high risk
        fp = ((gt_rows["top1_correct"] == True) &
              (gt_rows["hallucination_risk"] == "high")).sum()
        # false negatives: top1 wrong but labelled low risk
        fn = ((gt_rows["top1_correct"] == False) &
              (gt_rows["hallucination_risk"] == "low")).sum()
        lines += [
            "GROUND-TRUTH ALIGNMENT  (eval mode — comparing retrieved top1 vs markedCorrect)",
            "-" * 70,
            f"  Queries with ground truth          : {n_gt}",
            f"  top1 retrieval correct             : {n_gt_correct}  ({_pct(n_gt_correct, n_gt)})",
            f"  top1 retrieval wrong               : {n_gt_wrong}  ({_pct(n_gt_wrong, n_gt)})",
            f"",
            f"  False positives (correct top1, labelled 'high') : {fp}  "
            f"({_pct(fp, n_gt_correct)}  of correct retrievals over-flagged)",
            f"  False negatives (wrong top1,   labelled 'low')  : {fn}  "
            f"({_pct(fn, n_gt_wrong)}  of wrong retrievals under-flagged)",
            f"",
            f"  NOTE: some correct retrievals may still produce hallucinated responses",
            f"  (wrong parameter values, unsupported claims added by the LLM), so",
            f"  a small false-positive rate on 'high' labels is expected and appropriate.",
            "",
        ]
    else:
        lines += [
            "GROUND-TRUTH ALIGNMENT",
            "  N/A: --responses-csv mode does not have markedCorrect ground truth.",
            "",
        ]

    # --- latency ---
    t_corpus  = T.get("corpus_embedding", 0.0)
    t_queries = T.get("query_embedding",  0.0)
    t_train   = T.get("model_training",   0.0)
    t_gen_s   = t_generation
    t_grnd    = t_groundedness
    t_judge   = t_judge_total
    t_total   = T["total_wall"]
    t_other   = t_total - t_corpus - t_queries - t_train - t_gen_s - t_grnd - t_judge
    avg_query = (sum(t_per_query) / len(t_per_query)) if t_per_query else 0
    avg_judge = (t_judge / judge_count) if judge_count else 0

    lines += [
        "LATENCY  (wall-clock seconds)",
        "-" * 40,
        f"  Corpus embedding            : {t_corpus:7.2f}s",
        f"  Query embedding             : {t_queries:7.2f}s",
        f"  Model training/load         : {t_train:7.2f}s",
        f"  Response generation         : {t_gen_s:7.2f}s",
        f"  Embedding groundedness      : {t_grnd:7.2f}s",
        f"  LLM judge calls ({judge_count:3d} total)  : {t_judge:7.2f}s",
        f"  Other (overhead)            : {t_other:7.2f}s",
        f"  {'─' * 30}",
        f"  TOTAL WALL CLOCK            : {t_total:7.2f}s",
        f"",
        f"  Avg time per flagged query  : {avg_query*1000:.1f}ms",
        f"  Avg LLM judge call          : {avg_judge*1000:.1f}ms  "
        f"({'N/A' if judge_count == 0 else 'per call, when gated trigger fires'})",
        f"  Est. latency without cascade: {(t_total - t_judge + avg_judge * n_total):.2f}s  "
        f"(if judge fired every query)",
        f"  Actual latency saving       : ~{(avg_judge * (n_total - judge_count)):.2f}s  "
        f"from skipping {n_total - judge_count} judge calls",
        "",
    ]

    # --- high-risk detail ----
    high_rows = result_df[result_df["hallucination_risk"] == "high"]
    if len(high_rows) > 0:
        lines += [
            f"HIGH-RISK ROWS  ({len(high_rows)} queries — priority review queue)",
            "-" * 70,
        ]
        for _, r in high_rows.iterrows():
            lines.append(f"  Query     : {r['query']}")
            lines.append(f"  Endpoint  : {r['top1_method']} {r['top1_endpoint']}")
            lines.append(f"  GT correct: {r['top1_correct']}")
            lines.append(f"  Reason    : {r['risk_reason']}")
            lines.append(f"  Response  : {str(r['response'])[:180]}")
            lines.append("")
    else:
        lines += ["HIGH-RISK ROWS", "  None detected.", ""]

    # --- output files ---
    lines += [
        "OUTPUT FILES",
        "-" * 40,
        f"  Results CSV     : {out_csv}",
    ]
    if diag_csv:
        lines.append(f"  Diagnostics CSV : {diag_csv}")
    if model_is_new and model is not None:
        lines.append(f"  Risk model      : {model_path}")
    lines += ["", sep]

    # write metrics file
    metrics_path = out_dir / "hallucination_detect_metrics.txt"
    write_metrics_report(lines, metrics_path)

    # ---- print condensed summary to terminal ----
    print()
    print(sep)
    print("SUMMARY")
    print(sep)
    print(f"  Total queries flagged  : {n_total}")
    print(f"  Low risk               : {n_low:4d}  ({_pct(n_low, n_total)})")
    print(f"  Medium risk            : {n_med:4d}  ({_pct(n_med, n_total)})")
    print(f"  High risk              : {n_high:4d}  ({_pct(n_high, n_total)})")
    print()
    print(f"  LLM judge triggered    : {judge_count}/{n_total}  "
          f"({_pct(judge_count, n_total)}) — "
          f"{_pct(n_total - judge_count, n_total)} of calls saved by cascade")
    print(f"  Total wall clock       : {t_total:.1f}s  "
          f"(avg {avg_query*1000:.0f}ms/query, "
          f"avg judge {avg_judge*1000:.0f}ms/call)")
    if has_gt:
        print(f"  GT retrieval accuracy  : {n_gt_correct}/{n_gt}  "
              f"({_pct(n_gt_correct, n_gt)})")
        print(f"  False positives (FP)   : {fp}  "
              f"(correct top1 labelled 'high')")
        print(f"  False negatives (FN)   : {fn}  "
              f"(wrong top1 labelled 'low')")
    print()
    print(f"  Results CSV            : {out_csv}")
    if diag_csv:
        print(f"  Diagnostics CSV        : {diag_csv}")
    print(f"  Full metrics report    : {metrics_path}")
    if model_is_new and model is not None:
        print(f"  Risk model saved       : {model_path}")
    print()
    print("  'high' rows are your priority review queue.")
    print("  Open hallucination_detect_metrics.txt for the full detailed report.")

    # ---- PowerShell commands ----
    print()
    print(sep)
    print("POWERSHELL COMMANDS  (copy-paste ready)")
    print(sep)
    print()
    print("# (a) Mock smoke test -- fully offline, no API calls:")
    print("python hallucination_detect.py `")
    print("    Datasets/site24x7_Dataset.csv `")
    print("    Datasets/ADMIN_API/site24x7_Admin_API.xlsx `")
    print("    --mock `")
    print("    --n-eval 20 --n-train 100 --save-per-query")
    print()
    print("# (b) Real run (reads .env for credentials):")
    print("python hallucination_detect.py `")
    print("    Datasets/site24x7_Dataset.csv `")
    print("    Datasets/ADMIN_API/site24x7_Admin_API.xlsx `")
    print("    --extra-descriptions Datasets/reports_synthetic_descriptions.csv `")
    print("    --base-url http://20.235.183.15:443/openai/v1 `")
    print(f"    --api-key {api_key or 'YOUR_API_KEY'} `")
    print("    --n-eval 50 --n-train 2000 --top-k 5 `")
    print("    --low-risk-threshold 0.35 --high-risk-threshold 0.65 `")
    print("    --save-per-query")


if __name__ == "__main__":
    main()