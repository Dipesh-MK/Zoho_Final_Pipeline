"""
boilerplate_rewrite_eval.py

Closes the loop on boilerplate_diagnostic.py's Part A/B: generates specific,
endpoint-aware replacement descriptions for the keys flagged as templated
boilerplate, then re-runs the SAME doc-covered-only recall@1/@10 eval
(doc_description strategy, no fallback), on the SAME sampled eval queries
per seed, once with the original descriptions and once with the rewritten
ones - so you get a measured before/after number instead of a projection.

Imports everything it can directly from your existing modules rather than
reimplementing it:
  - rag_eval.py            : compute_recall, embed_texts, get_embeddings_mock
  - compare_doc_strategies.py : build_text_doc_description, load_doc_descriptions,
                                 load_extra_descriptions
  - boilerplate_diagnostic.py : detect_boilerplate, build_doc_covered_eval,
                                 embed_dispatch, MIN_SHARED_SKELETON_COUNT

The only new logic here is: (1) LLM rewrite generation for flagged keys,
and (2) looping the existing doc-covered eval twice (original vs rewritten
doc_map) instead of once.

Must live in the same folder as rag_eval.py, compare_doc_strategies.py,
and boilerplate_diagnostic.py.

Usage (mock smoke test, no API calls):
    python boilerplate_rewrite_eval.py dataset.csv admin_api.xlsx --mock --seeds 1 2 --n-eval 50

Usage (real):
    python boilerplate_rewrite_eval.py Datasets\\site24x7_Dataset.csv ^
        Datasets\\ADMIN_API\\site24x7_Admin_API.xlsx ^
        --extra-descriptions Datasets\\reports_synthetic_descriptions.csv ^
        --base-url http://20.235.183.15:443/openai/v1 --api-key YOUR_KEY ^
        --seeds 1 2 3 4 5 6 7 8 --n-eval 100
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from rag_eval import compute_recall, _strip_code_fences
from compare_doc_strategies import (
    build_text_doc_description,
    load_doc_descriptions,
    load_extra_descriptions,
)
from boilerplate_diagnostic import (
    detect_boilerplate,
    build_doc_covered_eval,
    embed_dispatch,
    MIN_SHARED_SKELETON_COUNT,
)

REWRITE_SYSTEM_PROMPT = (
    "You rewrite templated API documentation descriptions into specific, "
    "endpoint-aware one-sentence descriptions. You will be given an HTTP "
    "method, an endpoint path, a sheet/sub-feature name, and the current "
    "(templated/boilerplate) description. Write a NEW description that "
    "names the specific resource or data this endpoint actually returns, "
    "inferred from the endpoint path itself. Do not use generic phrasing "
    "like 'fetches data for endpoint' or 'retrieves a specific configuration'. "
    "One sentence, no preamble. Respond with ONLY a JSON array of objects, "
    "each with 'key' (copied exactly as given) and 'new_description'."
)


def generate_rewrites_mock(flagged_keys):
    rewrites = {}
    for key in flagged_keys:
        endpoint, method = key
        last_seg = endpoint.rstrip("/").split("/")[-1] or "resource"
        rewrites[key] = f"{method} returns {last_seg.replace('_', ' ')} data (mock rewrite)."
    return rewrites


def generate_rewrites_real(client, model, flagged_keys, doc_map, sheet_of_key, batch_size=10):
    rewrites = {}
    keys = list(flagged_keys)
    for i in range(0, len(keys), batch_size):
        batch = keys[i:i + batch_size]
        items = [{"key": f"{m} {e}", "endpoint": e, "method": m,
                  "sheet": sheet_of_key.get((e, m), ""),
                  "current_description": doc_map[(e, m)]}
                 for e, m in batch]
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.3,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(items, indent=2)},
                ],
            )
            content = _strip_code_fences(resp.choices[0].message.content)
            parsed = json.loads(content)
            got = set()
            for obj in parsed:
                m, ep = obj["key"].split(" ", 1)
                rewrites[(ep, m)] = obj["new_description"].strip()
                got.add((ep, m))
            missing = set(batch) - got
            if missing:
                raise ValueError(f"missing {len(missing)} keys in batch response")
        except Exception as e:
            print(f"  batch {i // batch_size} failed ({e}), falling back to one-by-one")
            for e_, m_ in batch:
                if (e_, m_) in rewrites:
                    continue
                try:
                    r = client.chat.completions.create(
                        model=model, temperature=0.3,
                        messages=[
                            {"role": "system", "content":
                                "Rewrite this templated API description into a specific, "
                                "endpoint-aware one-sentence description, inferring the "
                                "resource from the endpoint path. Respond with ONLY the "
                                "rewritten sentence, nothing else."},
                            {"role": "user", "content": json.dumps({
                                "endpoint": e_, "method": m_, "sheet": sheet_of_key.get((e_, m_), ""),
                                "current_description": doc_map[(e_, m_)]})},
                        ],
                    )
                    rewrites[(e_, m_)] = r.choices[0].message.content.strip().strip('"')
                except Exception as e2:
                    print(f"    failed on {m_} {e_}: {e2} - keeping original description")
                    rewrites[(e_, m_)] = doc_map[(e_, m_)]
        print(f"  rewrote batch {i // batch_size + 1}/{(len(keys) - 1) // batch_size + 1}")
    return rewrites


def build_doc_description_corpus(df, doc_map, enrichment_pool, n_examples):
    """Same corpus-building loop as boilerplate_diagnostic.py's Part B:
    full corpus (all keys), doc_description strategy, using whichever
    doc_map (original or rewritten) is passed in."""
    corpus_rows = []
    for key, group in df.groupby(["endpoint", "method"]):
        endpoint, method = key
        sheet = group["sheet"].iloc[0]
        sub_feature = group["subFeature"].iloc[0]
        pool = enrichment_pool[enrichment_pool["key"] == key]["query"].tolist()
        examples = pool[:n_examples]
        doc_desc = doc_map.get(key)
        text = build_text_doc_description(method, endpoint, sheet, sub_feature, examples, doc_desc)
        corpus_rows.append({"endpoint": endpoint, "method": method, "key": key, "text": text})
    return pd.DataFrame(corpus_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path")
    parser.add_argument("xlsx_path")
    parser.add_argument("--extra-descriptions", default=None)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-examples", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--embed-model", default="azure:primary/s247-textembedding-3l")
    parser.add_argument("--rewrite-model", default="azure:primary/gpt-4.1-mini")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--min-shared", type=int, default=MIN_SHARED_SKELETON_COUNT)
    args = parser.parse_args()

    csv_path, xlsx_path = Path(args.csv_path), Path(args.xlsx_path)
    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")
    if not xlsx_path.exists():
        sys.exit(f"File not found: {xlsx_path}")

    df = pd.read_csv(csv_path)
    if df["markedCorrect"].dtype == object:
        df["markedCorrect"] = df["markedCorrect"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False})
    df["markedCorrect"] = df["markedCorrect"].astype(bool)

    print("Loading description file...")
    doc_map = load_doc_descriptions(xlsx_path)
    if args.extra_descriptions:
        extra_map = load_extra_descriptions(Path(args.extra_descriptions))
        for k, v in extra_map.items():
            doc_map.setdefault(k, v)
    all_keys = set(zip(df["endpoint"], df["method"]))
    doc_keys = set(doc_map.keys()) & all_keys
    print(f"  doc_map covers {len(doc_keys)}/{len(all_keys)} pairs\n")

    sheet_of_key, subfeature_of_key = {}, {}
    for (ep, m), g in df.groupby(["endpoint", "method"]):
        sheet_of_key[(ep, m)] = g["sheet"].iloc[0]
        subfeature_of_key[(ep, m)] = g["subFeature"].iloc[0]

    # ---------- PART A: static boilerplate detection (unchanged from boilerplate_diagnostic.py) ----------
    print("=" * 70)
    print(f"PART A: static boilerplate detection (skeleton shared by >= {args.min_shared} keys/sheet)")
    print("=" * 70)
    boilerplate_keys, report_df = detect_boilerplate(doc_map, df, min_shared=args.min_shared)
    boilerplate_keys &= doc_keys
    print(f"\n{len(boilerplate_keys)}/{len(doc_keys)} doc-covered keys "
          f"({len(boilerplate_keys) / max(len(doc_keys), 1) * 100:.1f}%) flagged as boilerplate.\n")
    out_dir = csv_path.parent
    report_df.to_csv(out_dir / "boilerplate_rewrite_skeletons.csv", index=False)
    print(f"Saved -> {out_dir / 'boilerplate_rewrite_skeletons.csv'}")

    if not boilerplate_keys:
        sys.exit("No boilerplate keys flagged - nothing to rewrite. Try --min-shared 2 to loosen the check.")

    # ---------- generate rewrites ----------
    print(f"\nGenerating rewrites for {len(boilerplate_keys)} flagged keys "
          f"({'MOCK' if args.mock else f'via {args.rewrite_model}'})...")
    client = None
    if args.mock:
        rewrites = generate_rewrites_mock(boilerplate_keys)
    else:
        if not args.base_url or not args.api_key:
            sys.exit("Provide --base-url and --api-key, or use --mock for an offline test.")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
        rewrites = generate_rewrites_real(client, args.rewrite_model, boilerplate_keys, doc_map, sheet_of_key)

    review_rows = [{"method": k[1], "endpoint": k[0], "sheet": sheet_of_key.get(k, ""),
                     "original_description": doc_map[k], "new_description": v}
                    for k, v in rewrites.items()]
    pd.DataFrame(review_rows).to_csv(out_dir / "boilerplate_rewrites_for_review.csv", index=False)
    print(f"Saved -> {out_dir / 'boilerplate_rewrites_for_review.csv'} (spot-check before trusting these)")

    doc_map_rewritten = dict(doc_map)
    doc_map_rewritten.update(rewrites)

    # ---------- PART B: before vs after, doc-covered-only eval ----------
    print("\n" + "=" * 70)
    print("PART B: recall@1/@10, boilerplate vs non-boilerplate, ORIGINAL vs REWRITTEN descriptions")
    print("        (doc-covered-only eval, doc_description strategy, no fallback)")
    print("=" * 70)

    cache_dir = Path(args.cache_dir) if args.cache_dir else csv_path.parent / ".rag_cache"
    rows = []
    all_per_query = []
    n_eligible_reported = False

    for seed in args.seeds:
        eval_df = build_doc_covered_eval(df, doc_keys, args.n_eval, seed)
        if not n_eligible_reported:
            print(f"  sampling {len(eval_df)} doc-covered queries per seed\n")
            n_eligible_reported = True
        print(f"--- seed {seed} ---")

        true_df = df[df["markedCorrect"] == True].copy()
        true_df["key"] = list(zip(true_df["endpoint"], true_df["method"]))
        eval_queries = set(eval_df["query"])
        enrichment_pool = true_df[~true_df["query"].isin(eval_queries)]

        query_vecs = embed_dispatch(eval_df["query"].tolist(), client, args, cache_dir)
        true_key_of_query = {row["query"]: row["valid_keys"][0] for _, row in eval_df.iterrows()}

        for condition, dmap in [("original", doc_map), ("rewritten", doc_map_rewritten)]:
            corpus_df = build_doc_description_corpus(df, dmap, enrichment_pool, args.n_examples)
            corpus_vecs = embed_dispatch(corpus_df["text"].tolist(), client, args, cache_dir)

            per_query_df, recall = compute_recall(
                corpus_df, eval_df, corpus_vecs, query_vecs, ks=(1, 10),
                variant=f"{condition}_seed{seed}_boilerplate_rewrite_check")
            per_query_df["true_key"] = per_query_df["query"].map(true_key_of_query)
            per_query_df["is_boilerplate"] = per_query_df["true_key"].apply(lambda k: k in boilerplate_keys)
            per_query_df["seed"] = seed
            per_query_df["condition"] = condition
            all_per_query.append(per_query_df)

            for tag, group in per_query_df.groupby("is_boilerplate"):
                n = len(group)
                hits1 = (group["rank_of_correct"].apply(lambda r: r == 1)).sum()
                hits10 = (group["rank_of_correct"].apply(lambda r: r != ">10")).sum()
                rows.append({"seed": seed, "condition": condition, "is_boilerplate": tag,
                             "recall@1": hits1 / n if n else float("nan"),
                             "recall@10": hits10 / n if n else float("nan"), "n": n})

            r1_bp = next((r["recall@1"] for r in rows if r["seed"] == seed and r["condition"] == condition
                          and r["is_boilerplate"] is True), float("nan"))
            r1_nonbp = next((r["recall@1"] for r in rows if r["seed"] == seed and r["condition"] == condition
                              and r["is_boilerplate"] is False), float("nan"))
            print(f"  [{condition:9s}] recall@1 overall={recall[1]:.3f}  "
                  f"boilerplate={r1_bp:.3f}  non-boilerplate={r1_nonbp:.3f}")

    combined = pd.DataFrame(rows)
    summary = combined.groupby(["condition", "is_boilerplate"]).agg(
        recall_1_mean=("recall@1", "mean"), recall_1_std=("recall@1", "std"),
        recall_10_mean=("recall@10", "mean"), avg_n=("n", "mean"),
    ).round(4)

    print("\n" + "=" * 70)
    print("SUMMARY: recall@1, boilerplate vs non-boilerplate, original vs rewritten")
    print("=" * 70)
    print(summary.to_string())

    pq_df = pd.concat(all_per_query, ignore_index=True)
    pq_df.to_csv(out_dir / "boilerplate_rewrite_eval_per_query.csv", index=False)
    combined.to_csv(out_dir / "boilerplate_rewrite_eval_per_seed.csv", index=False)
    summary.to_csv(out_dir / "boilerplate_rewrite_eval_summary.csv")
    print(f"\nSaved -> {out_dir / 'boilerplate_rewrite_eval_per_seed.csv'}")
    print(f"Saved -> {out_dir / 'boilerplate_rewrite_eval_summary.csv'}")
    print(f"Saved -> {out_dir / 'boilerplate_rewrite_eval_per_query.csv'}")
    print(f"Saved -> {out_dir / 'boilerplate_rewrites_for_review.csv'}")
    print("\nCompare rewritten/is_boilerplate=True against original/is_boilerplate=True: if recall@1 moved")
    print("up meaningfully, the rewrite worked - check boilerplate_rewrites_for_review.csv to confirm the")
    print("new descriptions are actually accurate (not just different) before shipping them.")


if __name__ == "__main__":
    main()