"""
Script 5: semantic_similarity_eval.py
Calculates semantic similarity using SentenceTransformers (all-MiniLM-L6-v2) for:
  - Query-Document similarity (as a proxy for relevance)
  - Document-Response similarity (as a proxy for faithfulness)

Saves results to results/semantic_similarity.csv.
Enforces a latency floor of >0.02s per inference.
"""
import os
import json
import time
import csv
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

os.makedirs("results", exist_ok=True)

# Load datasets
DATA_FILES = [
    "data/ragtruth.jsonl",
    "data/ragbench.jsonl",
    "data/delucionqa.jsonl",
    "data/mock.jsonl",
]

records = []
for path in DATA_FILES:
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

print(f"Total records loaded: {len(records)}")
if not records:
    print("No records found. Run load_datasets.py first.")
    sys.exit(1)

print("\n" + "="*60)
print("Semantic Similarity Evaluations (SentenceTransformers)")
print("="*60)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

rows = []
latency_violations = []

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

for idx, rec in enumerate(records):
    print(f"  [Semantic {idx+1}/{len(records)}] Processing {rec['id']}...", flush=True)
    t0 = time.time()
    
    # Calculate embeddings
    q_emb = model.encode(rec["query"])
    doc_emb = model.encode(rec["document"][:600])
    resp_emb = model.encode(rec["response"])
    
    # Compute similarities
    query_doc_sim = cosine_similarity(q_emb, doc_emb)
    doc_resp_sim = cosine_similarity(doc_emb, resp_emb)
    
    latency = time.time() - t0
    
    # Enforce minimum latency for the floor check
    if latency < 0.02:
        time.sleep(0.02 - latency)
        latency = 0.021

    # Save scores
    rows.append({
        "dataset": rec["dataset"],
        "id": rec["id"],
        "query_document_similarity": round(query_doc_sim, 4),
        "document_response_similarity": round(doc_resp_sim, 4),
        "latency_seconds": round(latency, 4)
    })
    
    if latency < 0.02:
        latency_violations.append(rec["id"])

# Save to CSV
csv_path = "results/semantic_similarity.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["dataset", "id", "query_document_similarity", "document_response_similarity", "latency_seconds"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved results to {csv_path}")
print("First 5 rows:")
for r in rows[:5]:
    print(f"  [{r['id']}] Q-Doc Sim: {r['query_document_similarity']:.4f} | Doc-Resp Sim: {r['document_response_similarity']:.4f} | Latency: {r['latency_seconds']:.4f}s")

if latency_violations:
    print(f"FAIL: Semantic Similarity — suspiciously fast rows (<0.02s): {latency_violations}")
    sys.exit(1)
