"""
rag_eval.py

Build a chunk-per-(endpoint, method) corpus from the full CSV, embed it, sample N
True queries as an eval set, embed those too, and compute Recall@1 / Recall@10:
does ANY correct (endpoint, method) for that query show up in the top-k nearest
chunks?

A single query can legitimately be marked correct for more than one
(endpoint, method) pair (e.g. a query answerable by both GET and POST on the
same or related endpoints). Ground truth per eval query is therefore a SET of
valid keys, not a single key - the model only needs to surface one of them in
the top-k to count as correct.

Usage:
    python rag_eval.py full_dataset.csv --base-url http://20.235.183.15:443/openai/v1 --api-key YOUR_KEY
    python rag_eval.py full_dataset.csv --mock   # offline smoke test, no API calls, fake embeddings
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- text helpers ----------

def humanize_path(path: str) -> str:
    text = re.sub(r"[/_\-]", " ", str(path))
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\d+", " ", text)  # strip numeric IDs, not useful signal
    return re.sub(r"\s+", " ", text).strip()


def build_chunk_text(method: str, endpoint: str, sheet: str, sub_feature: str, examples: list[str]) -> str:
    parts = [
        f"{method} {endpoint}",
        f"Path words: {humanize_path(endpoint)}",
        f"Category: {sheet} > {sub_feature}",
    ]
    if examples:
        parts.append("Example questions this API answers: " + " | ".join(examples))
    return "\n".join(parts)


# ---------- corpus + eval set construction ----------

def build_corpus_and_eval(df: pd.DataFrame, n_eval: int, n_examples: int, seed: int):
    """
    Corpus: one chunk per (endpoint, method) - methods are NOT merged, since
    different methods on the same path are different actions and deserve
    separate vectors.

    Eval set: one row per UNIQUE query string. Because the same query text can
    be marked correct against multiple (endpoint, method) rows in the raw CSV,
    each eval row carries a `valid_keys` set of ALL correct answers for that
    query - not just one. Scoring treats a hit on any member of the set as
    correct.
    """
    df = df.copy()
    df["key"] = list(zip(df["endpoint"], df["method"]))

    true_df = df[df["markedCorrect"] == True]

    # collapse to one row per unique query, collecting every (endpoint, method)
    # that was ever marked correct for that exact query text
    query_groups = (
        true_df.groupby("query")["key"]
        .apply(lambda keys: sorted(set(keys)))
        .reset_index()
        .rename(columns={"key": "valid_keys"})
    )

    eval_df = query_groups.sample(n=min(n_eval, len(query_groups)), random_state=seed).reset_index(drop=True)
    eval_queries = set(eval_df["query"])

    # rows usable for in-chunk example enrichment: True rows whose QUERY TEXT
    # was not selected for eval (excluding by query, not by key, since one
    # eval query can span multiple keys - excluding by key alone could leak
    # an eval query's phrasing into the corpus via a sibling key)
    enrichment_pool = true_df[~true_df["query"].isin(eval_queries)]

    corpus = []
    for key, group in df.groupby("key"):
        endpoint, method = key
        sheet = group["sheet"].iloc[0]
        sub_feature = group["subFeature"].iloc[0]

        pool = enrichment_pool[enrichment_pool["key"] == key]["query"].tolist()
        examples = pool[:n_examples]

        corpus.append({
            "endpoint": endpoint,
            "method": method,
            "key": key,
            "text": build_chunk_text(method, endpoint, sheet, sub_feature, examples),
        })

    corpus_df = pd.DataFrame(corpus)
    return corpus_df, eval_df


# ---------- embeddings ----------

def _text_hash(model: str, text: str) -> str:
    return hashlib.sha256((model + "\x1f" + text).encode("utf-8")).hexdigest()[:20]


def embed_texts(client, model: str, texts: list[str], cache_dir: Path,
                 use_cache: bool = True, batch_size: int = 20,
                 max_retries: int = 5, request_timeout: float = 60.0) -> np.ndarray:
    """Embed a list of texts with a PER-TEXT disk cache: every individual text's
    vector is saved to its own file the moment it's fetched, so if a batch fails
    partway through, everything before it is already saved and won't be re-sent
    on the next run. Small batch size + retry-with-backoff so one slow/failed
    request doesn't kill the whole corpus embedding run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    vecs: list = [None] * n
    to_fetch = []

    if use_cache:
        for i, t in enumerate(texts):
            p = cache_dir / f"{_text_hash(model, t)}.npy"
            if p.exists():
                vecs[i] = np.load(p)
            else:
                to_fetch.append(i)
    else:
        to_fetch = list(range(n))

    print(f"  {n - len(to_fetch)} already cached, {len(to_fetch)} to embed via API")

    for start in range(0, len(to_fetch), batch_size):
        batch_idx = to_fetch[start:start + batch_size]
        batch_texts = [texts[i] for i in batch_idx]

        for attempt in range(1, max_retries + 1):
            try:
                resp = client.embeddings.create(model=model, input=batch_texts, timeout=request_timeout,
                                                 encoding_format="float")
                ordered = sorted(resp.data, key=lambda d: d.index)
                for local_i, d in zip(batch_idx, ordered):
                    vec = np.array(d.embedding, dtype=np.float32)
                    vecs[local_i] = vec
                    if use_cache:
                        np.save(cache_dir / f"{_text_hash(model, texts[local_i])}.npy", vec)
                break
            except Exception as e:
                msg = str(e)
                if "validation error" in msg.lower() or "list_type" in msg:
                    print(f"  batch at offset {start} hit a non-transient schema error - not retrying:")
                    print(f"    {type(e).__name__}: {msg[:300]}")
                    print("  This usually means the server returned embeddings in an unexpected "
                          "format (e.g. base64 instead of float list). If this still happens with "
                          "encoding_format='float' already set, the proxy itself may need a fix.")
                    raise
                if attempt == max_retries:
                    print(f"  batch at offset {start} failed after {max_retries} attempts: {e}")
                    raise
                wait = min(2 ** attempt, 30)
                print(f"  batch at offset {start} failed (attempt {attempt}/{max_retries}): "
                      f"{type(e).__name__}: {e} - retrying in {wait}s")
                time.sleep(wait)

        done = n - len(to_fetch) + min(start + batch_size, len(to_fetch))
        print(f"  progress: {done}/{n}")

    return np.array(vecs, dtype=np.float32)


def get_embeddings_mock(texts: list[str], dim: int = 64) -> np.ndarray:
    """Deterministic fake embeddings for offline pipeline testing only.
    Similar text -> similar vector, via hashed shingles. NOT a real model."""
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        words = re.findall(r"[a-z0-9]+", t.lower())
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            vecs[i, h % dim] += 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


# ---------- paraphrasing ----------

PARAPHRASE_SYSTEM_PROMPT = (
    "You paraphrase user questions. For each numbered question below, write ONE "
    "alternate phrasing that asks the exact same thing in different words. "
    "Preserve the original meaning and intent exactly - do not add, remove, or "
    "guess extra details. Do not mention API names, endpoints, or technical routes. "
    "Respond with ONLY a JSON array of strings in the same order as the input - "
    "no markdown, no code fences, no explanation, just the JSON array."
)


def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def generate_paraphrases_real(client, model: str, queries: list[str], batch_size: int = 10) -> list[str]:
    paraphrases = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        numbered = "\n".join(f"{j + 1}. {q}" for j, q in enumerate(batch))
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                    {"role": "user", "content": numbered},
                ],
                temperature=0.7,
            )
            content = _strip_code_fences(resp.choices[0].message.content)
            parsed = json.loads(content)
            if not isinstance(parsed, list) or len(parsed) != len(batch):
                raise ValueError(f"expected {len(batch)} items, got {parsed!r}")
            paraphrases.extend(str(p) for p in parsed)
        except Exception as e:
            print(f"  batch paraphrase failed ({e}), falling back to one-by-one for this batch")
            for q in batch:
                try:
                    r = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": (
                                "Rephrase this question in different words but keep the "
                                "exact same meaning. Respond with ONLY the rephrased "
                                "question, nothing else."
                            )},
                            {"role": "user", "content": q},
                        ],
                        temperature=0.7,
                    )
                    paraphrases.append(r.choices[0].message.content.strip())
                except Exception as e2:
                    print(f"    failed on '{q[:50]}...': {e2} - keeping original query")
                    paraphrases.append(q)
        print(f"  paraphrased {min(i + batch_size, len(queries))}/{len(queries)}")
    return paraphrases


def generate_paraphrases_mock(queries: list[str]) -> list[str]:
    """Deterministic fake paraphrase for offline pipeline testing only - just
    reverses word order. Not a real paraphrase, just exercises the code path."""
    out = []
    for q in queries:
        words = q.rstrip("?").split()
        out.append(" ".join(reversed(words)) + "?")
    return out


# ---------- recall computation ----------

def compute_recall(corpus_df: pd.DataFrame, eval_df: pd.DataFrame,
                    corpus_vecs: np.ndarray, query_vecs: np.ndarray,
                    ks=(1, 10), variant: str = "original"):
    """
    eval_df must have a `valid_keys` column: a set/list of (endpoint, method)
    tuples, ANY of which counts as a correct retrieval for that query.
    """
    corpus_norm = corpus_vecs / np.clip(np.linalg.norm(corpus_vecs, axis=1, keepdims=True), 1e-9, None)
    query_norm = query_vecs / np.clip(np.linalg.norm(query_vecs, axis=1, keepdims=True), 1e-9, None)

    sims = query_norm @ corpus_norm.T  # (n_queries, n_corpus) cosine similarity
    ranked_idx = np.argsort(-sims, axis=1)  # descending

    corpus_keys = list(corpus_df["key"])
    results = {k: 0 for k in ks}
    max_k = max(ks)
    per_query_rows = []

    for i, row in eval_df.iterrows():
        valid_keys = set(row["valid_keys"])
        top_idx = ranked_idx[i, :max_k]
        top_keys = [corpus_keys[j] for j in top_idx]

        # rank of the FIRST valid answer to appear (1-indexed), None if none in top max_k
        rank = next((r + 1 for r, kk in enumerate(top_keys) if kk in valid_keys), None)

        for k in ks:
            if rank is not None and rank <= k:
                results[k] += 1

        matched_key = top_keys[rank - 1] if rank is not None else None

        per_query_rows.append({
            "variant": variant,
            "query": row["query"],
            "num_valid_answers": len(valid_keys),
            "valid_endpoints": "; ".join(f"{m} {e}" for e, m in valid_keys),
            "rank_of_correct": rank if rank is not None else f">{max_k}",
            "matched_endpoint": matched_key[0] if matched_key else "",
            "matched_method": matched_key[1] if matched_key else "",
            "top1_endpoint": top_keys[0][0],
            "top1_method": top_keys[0][1],
        })

    n = len(eval_df)
    print("=" * 60)
    print(f"RAG RETRIEVAL EVAL [{variant}]  (corpus size: {len(corpus_df)} chunks, eval queries: {n})")
    print("=" * 60)
    for k in sorted(ks):
        print(f"Recall@{k}: {results[k]}/{n} = {results[k]/n:.3f}")
    print()

    return pd.DataFrame(per_query_rows), {k: results[k] / n for k in ks}


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Chunk-per-(endpoint, method) RAG retrieval eval")
    parser.add_argument("csv_path")
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-examples", type=int, default=2, help="example queries embedded per chunk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--embed-model", default="azure:primary/s247-textembedding-3l")
    parser.add_argument("--paraphrase-model", default="azure:primary/gpt-4.1-mini")
    parser.add_argument("--skip-paraphrase", action="store_true", help="only run original queries, skip paraphrase pass")
    parser.add_argument("--mock", action="store_true", help="skip real API calls, use fake hash embeddings/paraphrases")
    parser.add_argument("--tag", default="", help="label appended to output filename, e.g. 'no_examples', so ablation runs don't overwrite each other")
    parser.add_argument("--cache-dir", default=None, help="where to store cached embeddings (default: .rag_cache next to the csv)")
    parser.add_argument("--no-cache", action="store_true", help="disable embedding cache, always call the API")
    parser.add_argument("--embed-batch-size", type=int, default=20, help="texts per embedding API call (default 20, lower if you hit timeouts)")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds (default 60)")
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
    avg_valid = eval_df["valid_keys"].apply(len).mean()
    multi_valid = (eval_df["valid_keys"].apply(len) > 1).sum()
    print(f"Corpus: {len(corpus_df)} (endpoint, method) chunks")
    print(f"Eval set: {len(eval_df)} unique queries "
          f"(avg {avg_valid:.2f} valid answers/query, {multi_valid} queries have >1 valid answer)\n")

    cache_dir = Path(args.cache_dir) if args.cache_dir else path.parent / ".rag_cache"
    use_cache = not args.no_cache

    if args.mock:
        print("MOCK MODE: fake hash-based embeddings and fake word-reversal paraphrases. Pipeline test only.\n")
        client = None
        corpus_vecs = get_embeddings_mock(corpus_df["text"].tolist())
        query_vecs = get_embeddings_mock(eval_df["query"].tolist())
    else:
        if not args.base_url or not args.api_key:
            sys.exit("Provide --base-url and --api-key, or use --mock for an offline test.")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
        print("Embedding corpus...")
        corpus_vecs = embed_texts(client, args.embed_model, corpus_df["text"].tolist(),
                                   cache_dir, use_cache, args.embed_batch_size, request_timeout=args.timeout)
        print("Embedding eval queries (original)...")
        query_vecs = embed_texts(client, args.embed_model, eval_df["query"].tolist(),
                                  cache_dir, use_cache, args.embed_batch_size, request_timeout=args.timeout)

    # --- Pass 1: original queries ---
    per_query_orig, recall_orig = compute_recall(
        corpus_df, eval_df, corpus_vecs, query_vecs, ks=(1, 10), variant="original"
    )

    all_results = [per_query_orig]
    recall_para = None

    # --- Pass 2: paraphrased queries ---
    if not args.skip_paraphrase:
        print("Generating paraphrases...")
        if args.mock:
            paraphrased_queries = generate_paraphrases_mock(eval_df["query"].tolist())
        else:
            paraphrased_queries = generate_paraphrases_real(client, args.paraphrase_model, eval_df["query"].tolist())

        para_df = eval_df.copy()
        para_df["original_query"] = eval_df["query"].values
        para_df["query"] = paraphrased_queries

        if args.mock:
            para_query_vecs = get_embeddings_mock(para_df["query"].tolist())
        else:
            print("Embedding eval queries (paraphrased)...")
            para_query_vecs = embed_texts(client, args.embed_model, para_df["query"].tolist(),
                                           cache_dir, use_cache, args.embed_batch_size, request_timeout=args.timeout)

        per_query_para, recall_para = compute_recall(
            corpus_df, para_df, corpus_vecs, para_query_vecs, ks=(1, 10), variant="paraphrased"
        )
        per_query_para["original_query"] = para_df["original_query"].values
        all_results.append(per_query_para)

        print("=" * 60)
        print("SUMMARY: original vs paraphrased")
        print("=" * 60)
        for k in (1, 10):
            o, p = recall_orig[k], recall_para[k]
            print(f"Recall@{k}:  original={o:.3f}   paraphrased={p:.3f}   drop={o - p:+.3f}")
        print()

    per_query = pd.concat(all_results, ignore_index=True)

    out_dir = path.parent
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"{path.stem}_rag_eval_results{suffix}.csv"
    per_query.to_csv(out_path, index=False)
    print(f"Saved per-query results -> {out_path}")
    print("Rows where rank_of_correct is '>10' are your misses - inspect those first.")
    print("Compare 'variant' == paraphrased rows against original to see which queries")
    print("only fail once reworded - that's your robustness gap.")


if __name__ == "__main__":
    main()