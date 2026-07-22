"""
synthesize_boilerplate_descriptions.py

Generates replacement descriptions for the ~227 keys boilerplate_diagnostic.py
flagged as templated ("fetches data for endpoint" and similar generic
sentences repeated across many different endpoints). Same idea as your
existing synthesize_reports_descriptions.py, scoped to the boilerplate keys
instead of the whole Reports sheet.

Grounding strategy: for each flagged key, pull the REAL user queries that
were marked correct for it (the same enrichment-pool data your corpus
chunks already use) and hand those to the LLM alongside the endpoint path/
method/sheet/sub-feature, explicitly instructing it to write from that
evidence rather than inventing detail - this both produces a more useful
description AND reduces the hallucination risk of the generation step
itself (fewer invented specifics = fewer unsupported claims later).

Output CSV columns: endpoint, method, description, sheet, old_description,
n_queries_used - same (endpoint, method, description) shape as
reports_synthetic_descriptions.csv so it plugs into the same --extra-
descriptions argument other scripts already accept. IMPORTANT: those other
scripts merge extra descriptions with `doc_map.setdefault(k, v)` - i.e. the
xlsx wins on collision. Since here we specifically want to REPLACE existing
xlsx entries (they're the boilerplate), use eval_boilerplate_fix.py (companion
script) to test this, which overrides unconditionally instead of setdefault.

Usage:
    python synthesize_boilerplate_descriptions.py site24x7_Dataset.csv site24x7_Admin_API.xlsx --mock

    python synthesize_boilerplate_descriptions.py site24x7_Dataset.csv site24x7_Admin_API.xlsx \
        --base-url http://20.235.183.15:443/openai/v1 --api-key YOUR_KEY \
        --extra-descriptions reports_synthetic_descriptions.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from compare_doc_strategies import load_doc_descriptions, load_extra_descriptions
from boilerplate_diagnostic import detect_boilerplate
from rag_eval import humanize_path

SYNTH_SYSTEM_PROMPT = (
    "You write short API documentation descriptions for a monitoring/reporting "
    "SaaS product. For each numbered item below, you're given an endpoint path, "
    "HTTP method, category, and (when available) real example questions users "
    "asked that this exact endpoint answers. Write ONE specific 1-2 sentence "
    "description of what this endpoint does, grounded in the path structure and "
    "the example questions - do NOT invent field names, parameters, or behavior "
    "not implied by that evidence. Avoid generic boilerplate phrasing like "
    "'fetches data for this endpoint' - be as concrete as the path and examples "
    "allow. If there are no example questions, rely on the path and category "
    "alone, and keep the description more general rather than guessing. "
    "Respond with ONLY a JSON array of strings, same order as input, no "
    "markdown, no code fences, no other text."
)


def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def build_prompt_item(endpoint: str, method: str, sheet: str, sub_feature: str, examples: list) -> str:
    parts = [f"{method} {endpoint}", f"Path words: {humanize_path(endpoint)}", f"Category: {sheet} > {sub_feature}"]
    if examples:
        parts.append("Real example questions this endpoint answers: " + " | ".join(examples))
    else:
        parts.append("(no example questions available for this endpoint)")
    return " || ".join(parts)


def generate_descriptions_real(client, model: str, items: list, batch_size: int = 8) -> list:
    out = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        numbered = "\n".join(f"{j + 1}. {item}" for j, item in enumerate(batch))
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
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
            print(f"  batch synthesis failed ({e}), falling back to one-by-one for this batch")
            for item in batch:
                try:
                    r = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYNTH_SYSTEM_PROMPT.replace(
                                "For each numbered item below, you're", "For the item below, you're"
                            ).replace("Respond with ONLY a JSON array of strings, same order as input",
                                      "Respond with ONLY the description text")},
                            {"role": "user", "content": item},
                        ],
                        temperature=0.3,
                    )
                    out.append(r.choices[0].message.content.strip())
                except Exception as e2:
                    print(f"    failed on '{item[:60]}...': {e2} - keeping a minimal fallback description")
                    out.append(item.split(" || ")[1].replace("Path words: ", "") if " || " in item else item)
        print(f"  synthesized {min(i + batch_size, len(items))}/{len(items)}")
    return out


def generate_descriptions_mock(items: list) -> list:
    """Deterministic fake description for offline pipeline testing only -
    just echoes the path words and category. NOT a real synthesis."""
    out = []
    for item in items:
        parts = item.split(" || ")
        path_words = next((p.replace("Path words: ", "") for p in parts if p.startswith("Path words:")), "")
        category = next((p.replace("Category: ", "") for p in parts if p.startswith("Category:")), "")
        out.append(f"Provides {category.lower()} functionality related to {path_words.lower()}.")
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate replacement descriptions for keys flagged as templated boilerplate")
    parser.add_argument("csv_path")
    parser.add_argument("xlsx_path")
    parser.add_argument("--extra-descriptions", default=None)
    parser.add_argument("--min-shared", type=int, default=3,
                         help="same threshold as boilerplate_diagnostic.py - keep consistent between the two")
    parser.add_argument("--n-examples", type=int, default=4,
                         help="max real example queries per key to ground the generation in (default 4)")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--chat-model", default="azure:primary/gpt-4.1-mini")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", default=None, help="output CSV path (default: boilerplate_synthetic_descriptions.csv next to csv_path)")
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

    boilerplate_keys, _ = detect_boilerplate(doc_map, df, min_shared=args.min_shared)
    boilerplate_keys &= all_keys
    print(f"  {len(boilerplate_keys)} keys flagged as boilerplate - generating replacements for these\n")

    if not boilerplate_keys:
        sys.exit("No boilerplate keys found - nothing to synthesize. Check --min-shared matches what "
                  "boilerplate_diagnostic.py used.")

    sheet_of_key, subfeature_of_key = {}, {}
    for (ep, m), g in df.groupby(["endpoint", "method"]):
        sheet_of_key[(ep, m)] = g["sheet"].iloc[0]
        subfeature_of_key[(ep, m)] = g["subFeature"].iloc[0]

    true_df = df[df["markedCorrect"] == True].copy()
    true_df["key"] = list(zip(true_df["endpoint"], true_df["method"]))

    keys_sorted = sorted(boilerplate_keys)
    items = []
    n_queries_used = []
    for key in keys_sorted:
        endpoint, method = key
        sheet = sheet_of_key.get(key, "")
        sub_feature = subfeature_of_key.get(key, "")
        examples = true_df[true_df["key"] == key]["query"].tolist()[:args.n_examples]
        n_queries_used.append(len(examples))
        items.append(build_prompt_item(endpoint, method, sheet, sub_feature, examples))

    print(f"Generating {len(items)} replacement descriptions "
          f"({sum(1 for n in n_queries_used if n > 0)}/{len(items)} have real example queries to ground on)...")

    if args.mock:
        descriptions = generate_descriptions_mock(items)
    else:
        if not args.base_url or not args.api_key:
            sys.exit("Provide --base-url and --api-key, or use --mock for an offline test.")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=60.0)
        descriptions = generate_descriptions_real(client, args.chat_model, items)

    out_rows = []
    for key, desc, n_q in zip(keys_sorted, descriptions, n_queries_used):
        endpoint, method = key
        out_rows.append({
            "endpoint": endpoint,
            "method": method,
            "description": desc,
            "sheet": sheet_of_key.get(key, ""),
            "old_description": doc_map.get(key, ""),
            "n_queries_used": n_q,
        })

    out_df = pd.DataFrame(out_rows)
    out_path = Path(args.output) if args.output else csv_path.parent / "boilerplate_synthetic_descriptions.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(f"\n{(out_df['n_queries_used'] == 0).sum()}/{len(out_df)} keys had NO real example queries - "
          f"their new descriptions rely on path/category alone, so spot-check those specifically before "
          f"trusting them (they're the ones most likely to still be somewhat generic).")
    print("\nNext: run eval_boilerplate_fix.py with this file to check whether recall@1 on these "
          "specific keys actually improved.")


if __name__ == "__main__":
    main()