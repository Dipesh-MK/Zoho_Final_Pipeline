"""
Script 2: load_datasets.py
Loads RAGTruth, RAGBench, DelucionQA, and the custom 25-example mock set.
Saves each as data/<name>.jsonl.
Prints size and class balance. Aborts if any class has < 3 examples.
"""
import json
import os
import random
import sys

random.seed(42)
os.makedirs("data", exist_ok=True)

# ─── helpers ─────────────────────────────────────────────────────────────────

def shuffle_records(records, seed=42):
    r = random.Random(seed)
    out = list(records)
    r.shuffle(out)
    return out

def stratify_sample(records, n_each, seed=42):
    hall = [r for r in records if r["label_hallucinated"]]
    faith = [r for r in records if not r["label_hallucinated"]]
    # Shuffle each group to get random representatives
    hall = shuffle_records(hall, seed)
    faith = shuffle_records(faith, seed)
    n_h = min(n_each, len(hall))
    n_f = min(n_each, len(faith))
    return shuffle_records(hall[:n_h] + faith[:n_f], seed)

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def print_balance(name, records):
    n  = len(records)
    h  = sum(1 for r in records if r["label_hallucinated"])
    f  = n - h
    rv = sum(1 for r in records if r["label_relevant"])
    ri = n - rv
    print(f"  {name}: n={n}  hallucinated={h}  faithful={f}  relevant={rv}  irrelevant={ri}")
    return h, f, rv, ri

def check_min_class(name, h, f, min_count=3):
    if h < min_count or f < min_count:
        print(f"\nWARNING: {name} has fewer than {min_count} examples in a class "
              f"(hallucinated={h}, faithful={f}). Stopping.")
        sys.exit(1)

# ─── 1. RAGTruth ─────────────────────────────────────────────────────────────
print("Loading RAGTruth (wandb/RAGTruth-processed, QA subset)…")
try:
    from datasets import load_dataset
    ds = load_dataset("wandb/RAGTruth-processed", split="test")
    qa_ds = ds.filter(lambda x: x["task_type"] == "QA")

    records = []
    for idx, item in enumerate(qa_ds):
        lbl = item["hallucination_labels_processed"]
        has_hall = lbl.get("evident_conflict", 0) > 0 or lbl.get("baseless_info", 0) > 0
        records.append({
            "id":                 f"RT_{idx:03d}",
            "dataset":            "ragtruth",
            "query":              item["query"],
            "document":           item["context"],
            "response":           item["output"],
            "label_hallucinated": bool(has_hall),
            "label_relevant":     True,   # RAGTruth context is always retrieved for that query
        })

    records = stratify_sample(records, 3)
    save_jsonl(records, "data/ragtruth.jsonl")
    h, f, rv, ri = print_balance("ragtruth", records)
    check_min_class("ragtruth", h, f)
except Exception as e:
    print(f"  RAGTruth load FAILED: {e}")
    records = []

# ─── 2. RAGBench (hotpotqa + covidqa configs, sample) ────────────────────────
print("Loading RAGBench (rungalileo/ragbench)…")
try:
    from datasets import load_dataset

    all_rb = []
    configs = ["hotpotqa", "covidqa", "techqa", "pubmedqa"]
    for cfg in configs:
        try:
            ds_rb = load_dataset("rungalileo/ragbench", cfg, split="test")
            for item in ds_rb:
                docs = item.get("documents", [])
                doc_text = " ".join(docs) if isinstance(docs, list) else str(docs)
                # Use sentence_support_information to derive hallucination label
                ssi = item.get("sentence_support_information", [])
                any_unsupported = any(
                    not s.get("fully_supported", True) for s in ssi
                ) if ssi else False
                rel_score = item.get("relevance_score", 1.0)
                all_rb.append({
                    "id":                 f"RB_{cfg}_{str(item.get('id', len(all_rb))).zfill(5)}",
                    "dataset":            "ragbench",
                    "query":              item["question"],
                    "document":           doc_text[:3000],
                    "response":           item.get("response", ""),
                    "label_hallucinated": bool(any_unsupported),
                    "label_relevant":     bool(rel_score >= 0.5),
                })
        except Exception as e2:
            print(f"    config {cfg} failed: {e2}")

    sample = stratify_sample(all_rb, 3)
    if not sample:
        raise ValueError("No RAGBench examples loaded from any config")
    save_jsonl(sample, "data/ragbench.jsonl")
    h, f, rv, ri = print_balance("ragbench", sample)
    check_min_class("ragbench", h, f)
except Exception as e:
    print(f"  RAGBench load FAILED: {e}")

# ─── 3. DelucionQA ───────────────────────────────────────────────────────────
print("Loading DelucionQA (rungalileo/ragbench delucionqa config)…")
try:
    from datasets import load_dataset
    ds_dq = load_dataset("rungalileo/ragbench", "delucionqa", split="test")
    dq_records = []
    for item in ds_dq:
        docs = item.get("documents", [])
        doc_text = " ".join(docs) if isinstance(docs, list) else str(docs)
        ssi = item.get("sentence_support_information", [])
        any_unsupported = any(
            not s.get("fully_supported", True) for s in ssi
        ) if ssi else False
        rel_score = item.get("relevance_score", 1.0)
        dq_records.append({
            "id":                 f"DQ_{str(item.get('id', len(dq_records))).zfill(5)}",
            "dataset":            "delucionqa",
            "query":              item["question"],
            "document":           doc_text[:3000],
            "response":           item.get("response", ""),
            "label_hallucinated": bool(any_unsupported),
            "label_relevant":     bool(rel_score >= 0.5),
        })
    sample = stratify_sample(dq_records, 3)
    if not sample:
        raise ValueError("No DelucionQA examples loaded")
    save_jsonl(sample, "data/delucionqa.jsonl")
    h, f, rv, ri = print_balance("delucionqa", sample)
    check_min_class("delucionqa", h, f)
except Exception as e:
    print(f"  DelucionQA load FAILED: {e}")

# ─── 4. Custom 25-example mock set ───────────────────────────────────────────
print("Building custom mock set (25 examples: 13 hallucinated / 12 faithful)…")

MOCK_RAW = [
    # ── Hallucinated (13) ──────────────────────────────────────────────────
    ("H01", "Can I push/pull code via GitHub integration in Site24x7?",
     "Site24x7 GitHub Integration: Monitor your GitHub repositories health, "
     "track commit frequency, pull request metrics, and repository activity. "
     "Set up alerts for unusual patterns in your codebase activity.",
     "Yes, Site24x7 supports full GitHub push/pull integration. You can push "
     "and pull code directly from the Site24x7 dashboard.",
     True, True),

    ("H02", "Does Zoho CRM support video calling with customers?",
     "Zoho CRM Telephony Integration: Make and receive voice calls directly within "
     "Zoho CRM using built-in telephony features. Log call details automatically.",
     "Yes, Zoho CRM supports video calling with customers through its telephony integration. "
     "You can make video calls directly from contact records.",
     True, True),

    ("H03", "Can Zoho Desk automatically translate support tickets?",
     "Zoho Desk Multi-language Support: Create articles in multiple languages and "
     "configure language-specific email templates for the help center portal.",
     "Yes, Zoho Desk automatically translates all incoming support tickets for agents.",
     True, True),

    ("H04", "Does Zoho Books support cryptocurrency payments?",
     "Zoho Books Multi-Currency Support: Handle transactions in 170+ traditional "
     "currencies with automatic exchange rate updates.",
     "Yes, Zoho Books supports cryptocurrency payments like Bitcoin and Ethereum.",
     True, True),

    ("H05", "Can Zoho Projects automatically assign tasks using AI?",
     "Zoho Projects Task Management: Create tasks, deadlines, priorities, "
     "and assign tasks manually or in bulk to team members.",
     "Yes, Zoho Projects uses built-in AI to assign tasks automatically based on workload.",
     True, True),

    ("H06", "Does Zoho Analytics support real-time streaming data ingestion?",
     "Zoho Analytics Data Import: Import data from CSV, databases, and apps, "
     "and schedule automatic syncs at intervals ranging from 30 minutes to daily.",
     "Yes, Zoho Analytics supports real-time streaming data ingestion through live streams.",
     True, True),

    ("H07", "Can Zoho Mail compose AI-generated replies?",
     "Zoho Mail Smart Features: Use canned responses with pre-written templates "
     "organized by category to quickly reply to emails.",
     "Yes, Zoho Mail includes a generative AI helper that writes replies for you.",
     True, True),

    ("H08", "Does Site24x7 support fully automated server self-healing?",
     "Site24x7 IT Automation: Configure automation rules triggered by alerts, "
     "including restarting services and executing custom local scripts on servers.",
     "Yes, Site24x7 detects issue root causes and performs fully autonomous self-healing.",
     True, True),

    ("H09", "Can Zoho Creator deploy apps directly to the Apple App Store?",
     "Zoho Creator Mobile Access: Access Creator apps on iOS and Android via the "
     "official Zoho Creator app available on both mobile stores.",
     "Yes, Zoho Creator can compile and publish apps directly to the Apple App Store.",
     True, True),

    ("H10", "Does Zoho Desk support voice ticket creation with speech-to-text?",
     "Zoho Desk Multi-Channel Support: Create tickets from email, web forms, live chat, "
     "social media, and voice calls where agents log details manually.",
     "Yes, customers can speak, and Zoho Desk automatically transcribes and logs tickets.",
     True, True),

    ("H11", "Can Zoho Books automatically categorize expenses using AI?",
     "Zoho Books Expense Tracking: Upload receipt images, record expenses manually, "
     "and map them into pre-defined categories and projects.",
     "Yes, the built-in AI automatically reads receipts and categorizes expenses for you.",
     True, True),

    ("H12", "Can Zoho Analytics generate natural language insights from charts?",
     "Zoho Analytics Visualizations: Choose from 50+ interactive chart types, "
     "pivot tables, and dashboard builders with dynamic filters.",
     "Yes, the analytics dashboard automatically writes a natural language summary chart.",
     True, True),

    ("H13", "Does Zoho Projects support automatic time tracking based on app usage?",
     "Zoho Projects Time Tracking: Log task time manually or start/stop a timer "
     "directly from the task details page inside the project workspace.",
     "Yes, it tracks background app usage to log timesheet records automatically.",
     True, True),

    # ── Faithful (12) ─────────────────────────────────────────────────────────
    ("F01", "Does Site24x7 support GitHub repository monitoring?",
     "Site24x7 GitHub Integration: Monitor your GitHub repositories health, "
     "track commit frequency, pull request metrics, and repository activity.",
     "Yes, Site24x7 supports GitHub integration to track repository health, commits, and pull requests.",
     False, True),

    ("F02", "How do I set up multi-language support in Zoho Desk help center?",
     "Zoho Desk Multi-language Support: Create articles in multiple languages and "
     "configure language-specific email templates for the help center portal.",
     "You can write article translations and set up customized templates for each language portal.",
     False, True),

    ("F03", "Can I handle foreign currency transactions in Zoho Books?",
     "Zoho Books Multi-Currency Support: Handle transactions in 170+ traditional "
     "currencies with automatic exchange rate updates.",
     "Yes, Zoho Books lets you handle transactions across 170+ foreign currencies with rates synced.",
     False, True),

    ("F04", "What task management features are available in Zoho Projects?",
     "Zoho Projects Task Management: Create tasks, deadlines, priorities, "
     "and assign tasks manually or in bulk to team members.",
     "It supports task creation, deadline setting, priority tagging, and manual or bulk assignments.",
     False, True),

    ("F05", "How does Zoho Analytics handle data imports?",
     "Zoho Analytics Data Import: Import data from CSV, databases, and apps, "
     "and schedule automatic syncs at intervals ranging from 30 minutes to daily.",
     "You can import data from databases, files, or applications and sync them hourly or daily.",
     False, True),

    ("F06", "What are the auto-reply options in Zoho Mail?",
     "Zoho Mail Smart Features: Use canned responses with pre-written templates "
     "organized by category to quickly reply to emails.",
     "You can use canned email templates organized by category to respond to messages.",
     False, True),

    ("F07", "What IT automation actions can I configure in Site24x7?",
     "Site24x7 IT Automation: Configure automation rules triggered by alerts, "
     "including restarting services and executing custom local scripts on servers.",
     "You can set alerts to trigger automated tasks such as script execution and service restarts.",
     False, True),

    ("F08", "Can I access Zoho Creator apps on my phone?",
     "Zoho Creator Mobile Access: Access Creator apps on iOS and Android via the "
     "official Zoho Creator app available on both mobile stores.",
     "Yes, you can run your customized apps by installing the Creator app from the app store.",
     False, True),

    ("F09", "What channels does Zoho Desk support for creating tickets?",
     "Zoho Desk Multi-Channel Support: Create tickets from email, web forms, live chat, "
     "social media, and voice calls where agents log details manually.",
     "Tickets can be created from email, social media, voice calls, forms, or chat feeds.",
     False, True),

    ("F10", "How do I track expenses in Zoho Books?",
     "Zoho Books Expense Tracking: Upload receipt images, record expenses manually, "
     "and map them into pre-defined categories and projects.",
     "Upload invoices/receipts, log costs manually, and map them directly into specific category tags.",
     False, True),

    ("F11", "How do I log time on tasks in Zoho Projects?",
     "Zoho Projects Time Tracking: Log task time manually or start/stop a timer "
     "directly from the task details page inside the project workspace.",
     "You can log hours manually or run the task timer to populate sheets automatically.",
     False, True),

    ("F12", "What visualization options does Zoho Analytics offer?",
     "Zoho Analytics Visualizations: Choose from 50+ interactive chart types, "
     "pivot tables, and dashboard builders with dynamic filters.",
     "It offers more than 50 dashboard chart styles along with pivot grids and filter options.",
     False, True),
]

mock_records = [
    {
        "id":                 eid,
        "dataset":            "mock",
        "query":              q,
        "document":           doc,
        "response":           resp,
        "label_hallucinated": hall,
        "label_relevant":     rel,
    }
    for eid, q, doc, resp, hall, rel in MOCK_RAW
]

mock_records = stratify_sample(mock_records, 3)
save_jsonl(mock_records, "data/mock.jsonl")
h, f, rv, ri = print_balance("mock", mock_records)
check_min_class("mock", h, f)

print("\nAll datasets saved to data/")
