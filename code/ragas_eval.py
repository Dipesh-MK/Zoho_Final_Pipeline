"""
Script 6: ragas_eval.py
Calculates Ragas evaluation metrics (Faithfulness, Answer Relevancy, Context Precision)
using ChatOllama and OllamaEmbeddings.

Saves results to results/ragas_scores.csv.
"""
import os
import sys
import json
import time
import csv
import pandas as pd
from datasets import Dataset

# VertexAI Import Workaround to satisfy Ragas imports
import types
try:
    import langchain_google_vertexai
    mod = types.ModuleType("langchain_community.chat_models.vertexai")
    mod.ChatVertexAI = langchain_google_vertexai.ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = mod
except ImportError:
    pass

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate

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
print("Ragas Evaluations (Ollama qwen2.5-coder:7b)")
print("="*60)

# Setup model
local_llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Bind explicitly to metrics to prevent falling back to OpenAI
faithfulness.llm = local_llm
answer_relevancy.llm = local_llm
answer_relevancy.embeddings = local_embeddings
context_precision.llm = local_llm

# Build datasets
data = {
    "user_input":         [r["query"] for r in records],
    "response":           [r["response"] for r in records],
    "retrieved_contexts": [[r["document"][:600]] for r in records],
    "reference":          [r["document"][:600] for r in records],
}
dataset = Dataset.from_dict(data)

t0 = time.time()
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
)
total_latency = time.time() - t0

# Convert results to dataframe and save
res_df = result.to_pandas()
res_df["id"] = [r["id"] for r in records]
res_df["dataset"] = [r["dataset"] for r in records]
res_df["latency_seconds"] = round(total_latency / len(records), 4)

# Keep only desired columns
out_cols = ["dataset", "id", "faithfulness", "answer_relevancy", "context_precision", "latency_seconds"]
res_df = res_df[out_cols]

# Fill NaNs gracefully
res_df = res_df.fillna(0.0)

csv_path = "results/ragas_scores.csv"
res_df.to_csv(csv_path, index=False)

print(f"\nSaved Ragas scores to {csv_path}")
print("First 5 rows:")
for _, row in res_df.head(5).iterrows():
    print(f"  [{row['id']}] Faithfulness: {row['faithfulness']:.4f} | Relevancy: {row['answer_relevancy']:.4f} | Precision: {row['context_precision']:.4f} | Latency: {row['latency_seconds']:.4f}s")
