import os
import sys
import json
import time
import csv
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Dependency check (Bespoke-MiniCheck-7B's remote code + 4-bit quant need these) ---
def _check_deps():
    missing = []
    for pkg, pip_name in [("einops", "einops"), ("bitsandbytes", "bitsandbytes>=0.46.1")]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"Missing required package(s): {', '.join(missing)}")
        print(f"Run: pip install -U {' '.join(missing)}")
        sys.exit(1)
_check_deps()

# Fix transformers dict loading for older compat
class AllTiedWeightsKeysDescriptor:
    def __get__(self, instance, owner): return {}
    def __set__(self, instance, value): pass
transformers.PreTrainedModel.all_tied_weights_keys = AllTiedWeightsKeysDescriptor()

os.makedirs("results", exist_ok=True)

# --- Data Loading ---
# Pointed at the Windows-side data folder via WSL2's /mnt/c mount.
# Update the path below if your Zoho folder lives somewhere else.
DATA_DIR = "/mnt/c/Users/Dipesh/OneDrive/Desktop/Zoho/data"
DATA_FILES = [
    f"{DATA_DIR}/ragtruth.jsonl",
    f"{DATA_DIR}/ragbench.jsonl",
    f"{DATA_DIR}/delucionqa.jsonl",
    f"{DATA_DIR}/mock.jsonl",
]
records = []
for path in DATA_FILES:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip(): records.append(json.loads(line.strip()))

if "--limit" in sys.argv:
    limit_idx = sys.argv.index("--limit")
    records = records[:int(sys.argv[limit_idx + 1])]

print(f"Total records loaded: {len(records)}")

# --- GPU / VRAM diagnostics ---
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free_vram_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    print(f"GPU detected: {gpu_name} | Total VRAM: {total_vram_gb:.1f} GB | Free VRAM: {free_vram_gb:.1f} GB")
    if total_vram_gb < 6.5:
        print(f"    WARNING: Bespoke-MiniCheck-7B in 4-bit needs ~5-6GB VRAM just for weights.")
        print(f"    Your GPU has {total_vram_gb:.1f}GB total, which is tight and may crash silently mid-load.")
        print(f"    This script will fall back to CPU-offload mode automatically if that happens.")
else:
    print("WARNING: No CUDA GPU detected by torch. This will run on CPU only (very slow, may fail).")
    print("If you're in WSL2 and expected a GPU here, check `nvidia-smi` works inside WSL2 first")
    print("and that this environment's torch build actually has CUDA support (torch.__version__ + torch.version.cuda).")

# --- Model Loading ---
# Loading from the already-downloaded local copy in WSL2 -- avoids touching
# the Hugging Face Hub / network entirely (that was the source of hours of
# download flakiness earlier). Update this path if your local copy lives
# somewhere else.
print("Loading Bespoke-MiniCheck-7B via transformers (GPU)...")
model_id = "/home/aracknab/minicheck-model"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
    print(f"    Standard 4-bit GPU load failed ({type(e).__name__}: {e}).")
    print("    Retrying with CPU offload enabled (slower, but avoids VRAM crash)...")
    torch.cuda.empty_cache()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
        max_memory={0: "5GiB", "cpu": "24GiB"},
    )

print("Bespoke-MiniCheck-7B loaded successfully.")

# --- Robust Evaluation Engine ---
def process_minicheck_logits(model, tokenizer, inputs, row_id):
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

    # DEBUG: Print Top 5
    top_k_probs, top_k_ids = torch.topk(probs, 5)
    print(f"    [MiniCheck Debug {row_id}] Top 5 predictions:")
    for rank, (token_id, prob) in enumerate(zip(top_k_ids, top_k_probs)):
        token_str = tokenizer.decode([token_id.item()])
        print(f"      Rank {rank+1}: ID={token_id.item():<5} String={repr(token_str)} Prob={prob.item():.4f}")

    yes_variants = ["Yes", " Yes", "yes", " yes", "1", "true"]
    no_variants = ["No", " No", "no", " no", "0", "false"]

    yes_prob, no_prob = 0.0, 0.0

    for variant in yes_variants:
        try:
            v_id = tokenizer.encode(variant, add_special_tokens=False)[0]
            yes_prob = max(yes_prob, probs[v_id].item())
        except Exception: pass

    for variant in no_variants:
        try:
            v_id = tokenizer.encode(variant, add_special_tokens=False)[0]
            no_prob = max(no_prob, probs[v_id].item())
        except Exception: pass

    if yes_prob == 0.0 and no_prob == 0.0:
        top_token_str = tokenizer.decode([top_k_ids[0].item()]).strip().lower()
        if top_token_str in ["yes", "1", "true", "factual", "faithful"]:
            return "faithful", 1.0
        else:
            return "hallucinated", 0.0

    return ("faithful" if yes_prob >= no_prob else "hallucinated"), yes_prob

system_prompt = (
    "Determine whether the provided claim is consistent with the corresponding document. "
    "Consistency in this context implies that all information presented in the claim is substantiated "
    "by the document. If not, it should be considered inconsistent. Please assess the claim's consistency "
    "with the document by responding with either \"Yes\" or \"No\"."
)

results = []
for idx, rec in enumerate(records):
    print(f"  [MiniCheck {idx+1}/{len(records)}] Processing {rec['id']}...", flush=True)
    t0 = time.time()

    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nDocument: {rec['document'][:1000]}\nClaim: {rec['response']}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    verdict, yes_prob = process_minicheck_logits(model, tokenizer, inputs, rec['id'])
    lat = time.time() - t0
    gt = "hallucinated" if rec["label_hallucinated"] else "faithful"

    print(f"    Result: {verdict} (Yes Prob: {yes_prob:.4f}) in {lat:.2f}s")
    results.append({"id": rec["id"], "pred": verdict, "gt": gt, "lat": lat})

# --- Save Results ---
with open("results/minicheck_only.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "pred", "gt", "lat"])
    writer.writeheader()
    writer.writerows(results)
print(f"Saved: results/minicheck_only.csv ({len(results)} rows)")
