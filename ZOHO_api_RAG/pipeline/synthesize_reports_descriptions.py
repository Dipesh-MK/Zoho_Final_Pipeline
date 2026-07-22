"""
synthesize_reports_descriptions.py

The Admin API description file (site24x7_Admin_API.xlsx) has ZERO entries for
the "Reports" sheet - that's 1,803 of the dataset's 2,298 unique
(endpoint, method) pairs (79%). Every strategy tested so far in
compare_doc_strategies.py falls back to identical query_derived text there,
so we have no signal on whether description-style text would help on the
majority of the corpus.

This script closes that gap by asking an LLM to write a short, doc-style
description for every Reports-sheet (endpoint, method) pair, using the same
signal a human doc-writer would have: the endpoint path, its subFeature
label, and a handful of real example queries users asked that map to it.
It does NOT look at the true query text for the pair it's writing about
beyond that - the same information build_chunk_text() already has - so any
retrieval improvement is coming from the LLM restating/enriching that
information in fuller, more descriptive prose, not from leaking extra data.

Mirrors rag_eval.py's generate_paraphrases_real() pattern: batched JSON chat
completions against the same OpenAI-compatible client already used for
embeddings/paraphrasing, with per-item fallback if a batch's JSON doesn't
parse, and a resumable on-disk output (existing rows are skipped on rerun).

Output: reports_synthetic_descriptions.csv with columns
    endpoint, method, sheet, subFeature, description, n_examples_used
This is designed to be fed straight into compare_doc_strategies.py via its
--extra-descriptions flag, which merges it into the doc_map alongside the
real xlsx descriptions (Reports-sheet coverage only - no key collision with
the xlsx, which has none).

Usage:
    python synthesize_reports_descriptions.py site24x7_Dataset.csv \
        --base-url http://20.235.183.15:443/openai/v1 --api-key YOUR_KEY \
        --chat-model azure:primary/gpt-4.1-mini
    python synthesize_reports_descriptions.py site24x7_Dataset.csv --mock   # offline pipeline test
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = (
    "You write short API documentation descriptions for a monitoring/reporting "
    "product (Site24x7). For each numbered API entry below, you are given: the "
    "HTTP method and endpoint path, a category label, and 1-5 real example "
    "questions a user might ask that this API answers. Write ONE concise "
    "description (1-2 sentences, max ~40 words) of what the endpoint DOES, in "
    "the voice of product documentation - e.g. 'Retrieves the weekly uptime "
    "summary for a monitor over a specified date range.' "
    "Rules: base the description ONLY on the path, category, and example "
    "questions given - do not invent specific parameters, response fields, or "
    "behavior you cannot infer from them. Do not repeat the raw path or method "
    "in the description text. Do not use hedging language like 'likely' or "
    "'probably'. Respond with ONLY a JSON array of strings, same order as the "
    "input, no markdown, no code fences, no explanation."
)


def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def _format_entry(i: int, method: str, endpoint: str, sheet: str, sub_feature: str, examples: list[str]) -> str:
    lines = [f"{i}. {method} {endpoint}", f"   Category: {sheet} > {sub_feature}"]
    if examples:
        lines.append("   Example questions: " + " | ".join(examples))
    else:
        lines.append("   Example questions: (none available)")
    return "\n".join(lines)


def synthesize_batch_real(client, model: str, entries: list[dict], temperature: float = 0.3) -> list[str]:
    """entries: list of dicts with keys method, endpoint, sheet, subFeature, examples.
    Returns list of description strings, same order, same length - always,
    falling back to a generic templated description per-item if both the
    batch call and the single-item retry fail, so the run never stalls."""
    numbered = "\n".join(
        _format_entry(i + 1, e["method"], e["endpoint"], e["sheet"], e["subFeature"], e["examples"])
        for i, e in enumerate(entries)
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": numbered},
            ],
            temperature=temperature,
        )
        content = _strip_code_fences(resp.choices[0].message.content)
        parsed = json.loads(content)
        if not isinstance(parsed, list) or len(parsed) != len(entries):
            raise ValueError(f"expected {len(entries)} items, got {parsed!r}")
        return [str(p).strip() for p in parsed]
    except Exception as e:
        print(f"    batch failed ({type(e).__name__}: {e}) - falling back to one-by-one for this batch")
        out = []
        for e_item in entries:
            single = _format_entry(1, e_item["method"], e_item["endpoint"], e_item["sheet"],
                                    e_item["subFeature"], e_item["examples"])
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": single},
                    ],
                    temperature=temperature,
                )
                c = _strip_code_fences(r.choices[0].message.content)
                p = json.loads(c)
                desc = str(p[0]).strip() if isinstance(p, list) else str(p).strip()
            except Exception as e2:
                print(f"      failed on {e_item['method']} {e_item['endpoint']}: {e2} - using generic fallback text")
                desc = f"Handles {e_item['method']} {e_item['endpoint']} in the {e_item['subFeature']} area."
            out.append(desc)
        return out


def synthesize_batch_mock(entries: list[dict]) -> list[str]:
    """Deterministic fake description for offline pipeline testing only."""
    out = []
    for e in entries:
        words = re.sub(r"[/_\-]", " ", e["endpoint"]).split()
        out.append(f"[MOCK] Handles {e['method']} for {' '.join(words[-3:])} under {e['subFeature']}.")
    return out


def load_existing(out_path: Path) -> dict:
    """Resume support: (endpoint, method) -> row dict, for anything already written."""
    if not out_path.exists():
        return {}
    existing_df = pd.read_csv(out_path)
    return {(r["endpoint"], r["method"]): r.to_dict() for _, r in existing_df.iterrows()}


def main():
    parser = argparse.ArgumentParser(description="LLM-synthesize doc-style descriptions for the Reports sheet")
    parser.add_argument("csv_path")
    parser.add_argument("--sheet", default="Reports", help="which 'sheet' value to synthesize descriptions for (default: Reports)")
    parser.add_argument("--n-examples", type=int, default=5, help="max example queries per endpoint shown to the LLM")
    parser.add_argument("--batch-size", type=int, default=8, help="endpoints per chat completion call")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--chat-model", default="azure:primary/gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--mock", action="store_true", help="offline pipeline test, no API calls")
    parser.add_argument("--out", default=None, help="output CSV path (default: <csv_dir>/reports_synthetic_descriptions.csv)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N pairs (for a quick trial run)")
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

    target = df[df["sheet"] == args.sheet]
    if target.empty:
        sys.exit(f"No rows found with sheet == {args.sheet!r}. Check --sheet spelling against the CSV's 'sheet' column.")

    unique_pairs = target.drop_duplicates(["endpoint", "method"])[["endpoint", "method", "sheet", "subFeature"]]
    if args.limit:
        unique_pairs = unique_pairs.head(args.limit)
    print(f"{len(unique_pairs)} unique (endpoint, method) pairs to synthesize descriptions for (sheet={args.sheet!r})")

    true_df = df[df["markedCorrect"] == True]

    out_path = Path(args.out) if args.out else path.parent / "reports_synthetic_descriptions.csv"
    existing = load_existing(out_path)
    if existing:
        print(f"Resuming: {len(existing)} pairs already have descriptions in {out_path}, will skip those")

    client = None
    if not args.mock:
        if not args.base_url or not args.api_key:
            sys.exit("Provide --base-url and --api-key, or use --mock for an offline test.")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=90.0)

    todo = []
    for _, r in unique_pairs.iterrows():
        key = (r["endpoint"], r["method"])
        if key in existing:
            continue
        pool = true_df[(true_df["endpoint"] == r["endpoint"]) & (true_df["method"] == r["method"])]["query"].tolist()
        todo.append({
            "endpoint": r["endpoint"], "method": r["method"], "sheet": r["sheet"], "subFeature": r["subFeature"],
            "examples": pool[:args.n_examples],
        })

    print(f"{len(todo)} pairs left to generate\n")

    results = list(existing.values())
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        if args.mock:
            descs = synthesize_batch_mock(batch)
        else:
            descs = synthesize_batch_real(client, args.chat_model, batch, args.temperature)

        for entry, desc in zip(batch, descs):
            results.append({
                "endpoint": entry["endpoint"], "method": entry["method"],
                "sheet": entry["sheet"], "subFeature": entry["subFeature"],
                "description": desc, "n_examples_used": len(entry["examples"]),
            })

        # write after every batch, so a crash/interrupt only loses the in-flight batch
        pd.DataFrame(results).to_csv(out_path, index=False)
        done = start + len(batch)
        print(f"  {done}/{len(todo)} new pairs done ({len(results)} total in file) -> saved to {out_path}")
        if not args.mock:
            time.sleep(0.2)  # light pacing, avoid hammering the endpoint

    print(f"\nDone. {len(results)} descriptions total -> {out_path}")
    print("Sample:")
    sample_df = pd.DataFrame(results).sample(min(5, len(results)), random_state=1)
    for _, r in sample_df.iterrows():
        print(f"  {r['method']} {r['endpoint']}\n    -> {r['description']}\n")
    print("Next: feed this into compare_doc_strategies.py with --extra-descriptions "
          f"{out_path} to re-run the strategy comparison with Reports-sheet coverage included.")


if __name__ == "__main__":
    main()