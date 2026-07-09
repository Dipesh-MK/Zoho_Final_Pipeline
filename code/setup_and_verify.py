"""
Script 1: setup_and_verify.py
Prints Python/torch/CUDA versions and PASS/FAIL per library import.
"""
import sys
import types
try:
    import langchain_google_vertexai
    mod = types.ModuleType("langchain_community.chat_models.vertexai")
    mod.ChatVertexAI = langchain_google_vertexai.ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = mod
except ImportError:
    pass

import importlib

# ── Python version ────────────────────────────────────────────────────────────
print(f"Python: {sys.version}")

# ── torch / CUDA ──────────────────────────────────────────────────────────────
try:
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_version   = torch.version.cuda if cuda_available else "N/A"
    print(f"torch:          {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA version:   {cuda_version}")
except ImportError:
    print("torch:          NOT INSTALLED")
    print("CUDA available: UNKNOWN")

# ── Per-library PASS/FAIL ─────────────────────────────────────────────────────
checks = [
    ("sentence_transformers", "sentence-transformers"),
    ("FlagEmbedding",         "FlagEmbedding"),
    ("lettucedetect",         "lettucedetect"),
    ("deepeval",              "deepeval"),
    ("requests",              "requests"),
    ("sklearn",               "scikit-learn"),
    ("pandas",                "pandas"),
    ("phoenix",               "arize-phoenix"),
    ("litellm",               "litellm"),
    ("datasets",              "datasets"),
    ("nltk",                  "nltk"),
    ("ollama",                "ollama"),
    ("ragas",                 "ragas"),
    ("langchain_community",   "langchain-community"),
    ("minicheck",             "minicheck"),
    ("alignscore",            "alignscore"),
    ("spacy",                 "spacy"),
    ("bitsandbytes",          "bitsandbytes"),
    ("accelerate",            "accelerate"),
]

print()
for module, label in checks:
    try:
        importlib.import_module(module)
        print(f"  PASS  {label}")
    except ImportError as e:
        print(f"  FAIL  {label}  ({e})")
