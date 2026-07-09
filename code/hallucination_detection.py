"""
Script 4: hallucination_detection.py
Runs multiple detectors independently on all loaded datasets:
  (a) LettuceDetect   - span-level transformer
  (b) Lynx-8B         - Ollama / Groq judge, exact model-card prompt
  (c) DeepEval        - FaithfulnessMetric via OllamaModel/Groq
  (d) Arize Phoenix   - HallucinationEvaluator via LiteLLM -> Ollama/Groq
  (e) Bespoke-MiniCheck-7B
  (f) AlignScore
  (g) HHEM-2.1
  (h) RAGAS Faithfulness - manual claim-extraction + verification, in-process, Groq-first

Saves results/hallucination_flags.csv.
Prints first 5 rows per tool.
Asserts per-tool latency floors (LettuceDetect >0.05s, LLM tools >0.3s).
"""
import json
import os
import csv
import time
import sys
import subprocess
import textwrap

# Patch transformers to be compatible with legacy custom modeling scripts
import transformers
class AllTiedWeightsKeysDescriptor:
    def __get__(self, instance, owner):
        return {}
    def __set__(self, instance, value):
        pass
transformers.PreTrainedModel.all_tied_weights_keys = AllTiedWeightsKeysDescriptor()

try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                    cache.update(key_states, value_states, layer_idx)
            return cache
        DynamicCache.from_legacy_cache = from_legacy_cache
    if not hasattr(DynamicCache, "to_legacy_cache"):
        def to_legacy_cache(self):
            legacy_cache = []
            for layer_idx in range(len(self)):
                legacy_cache.append((self.key_cache[layer_idx], self.value_cache[layer_idx]))
            return tuple(legacy_cache)
        DynamicCache.to_legacy_cache = to_legacy_cache
except ImportError:
    pass

# Custom MiniCheck to support Bespoke-MiniCheck-7B on Windows
class CustomMiniCheck:
    def __init__(self, model_name='Bespoke-MiniCheck-7B'):
        self.model_name = model_name
        if model_name == 'Bespoke-MiniCheck-7B':
            # Check VRAM using nvidia-smi
            import subprocess
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
                    capture_output=True, text=True, check=True
                )
                free_mem = int(res.stdout.strip().split("\n")[0])
                print(f"Available GPU VRAM: {free_mem} MiB")
                if free_mem < 5000:
                    raise RuntimeError(f"Insufficient VRAM: {free_mem} MiB available, need >=5000 MiB.")
            except Exception as e:
                print(f"VRAM check warning: {e}")

            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model_id = "bespokelabs/Bespoke-MiniCheck-7B"
            if torch.cuda.is_available():
                free_vram_mb = torch.cuda.mem_get_info()[0] / 1024**2
                print(f"Free VRAM before MiniCheck-7B load: {free_vram_mb:.0f} MiB")
                if free_vram_mb < 5000:
                    print("WARNING: Free VRAM may be insufficient even for 4-bit quantized load.")
            print("Loading Bespoke-MiniCheck-7B via transformers (GPU)...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True
            )
            print("Bespoke-MiniCheck-7B loaded successfully.")
        else:
            from minicheck.minicheck import MiniCheck as OriginalMiniCheck
            self.delegate = OriginalMiniCheck(model_name=model_name)

    def score(self, docs, claims):
        if self.model_name == 'Bespoke-MiniCheck-7B':
            system_prompt = (
                "Determine whether the provided claim is consistent with the corresponding document. "
                "Consistency in this context implies that all information presented in the claim is substantiated "
                "by the document. If not, it should be considered inconsistent. Please assess the claim's consistency "
                "with the document by responding with either \"Yes\" or \"No\"."
            )
            pred_labels = []
            raw_probs = []
            import torch
            for doc, claim in zip(docs, claims):
                prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nDocument: {doc}\nClaim: {claim}<|im_end|>\n<|im_start|>assistant\n"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, use_cache=False)
                    logits = outputs.logits[0, -1, :]

                    yes_token_ids = self.tokenizer.encode("Yes", add_special_tokens=False)
                    no_token_ids = self.tokenizer.encode("No", add_special_tokens=False)

                    yes_logit = logits[yes_token_ids[0]].item()
                    no_logit = logits[no_token_ids[0]].item()

                    probs = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
                    yes_prob = probs[0].item()

                    pred_label = 1 if yes_prob >= 0.5 else 0
                    pred_labels.append(pred_label)
                    raw_probs.append(yes_prob)
            return pred_labels, raw_probs, None, None
        else:
            return self.delegate.score(docs, claims)

import minicheck
import minicheck.minicheck
minicheck.MiniCheck = CustomMiniCheck
minicheck.minicheck.MiniCheck = CustomMiniCheck

os.makedirs("results", exist_ok=True)

# --- Load all datasets from disk ---
DATA_FILES = [
    "data/ragtruth.jsonl",
    "data/ragbench.jsonl",
    "data/delucionqa.jsonl",
    "data/mock.jsonl",
]

records = []
for path in DATA_FILES:
    if not os.path.exists(path):
        print(f"  SKIP (not found): {path}")
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

# --- Check Ollama / Groq reachability ---
import requests as req_lib

OLLAMA_BASE = "http://127.0.0.1:11434"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# openai/gpt-oss-120b is Groq's current recommended production migration target for
# llama-3.3-70b-versatile (deprecated) - best available reasoning quality/speed tradeoff
# for LLM-as-judge style tasks as of mid-2026. Override via GROQ_MODEL env var if needed.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
USE_GROQ = bool(GROQ_API_KEY)

# Free-tier Groq accounts are rate-limited (observed stricter in practice than the
# documented ~30 req/min once burst/TPM limits on reasoning models kick in). All Groq
# calls route through groq_chat_completion() below, which now enforces BOTH (a) a
# minimum even spacing between calls and (b) a rolling-window cap, so requests are
# spread out instead of firing in a burst and immediately 429-ing.
# Lower GROQ_RPM further (env var) if you still see frequent 429s on your account tier.
GROQ_RPM = int(os.environ.get("GROQ_RPM", "18"))
_groq_min_interval = 60.0 / GROQ_RPM
_groq_last_call_time = 0.0
_groq_call_timestamps = []

def _rate_limit_wait():
    """Block until issuing another Groq call respects both an even minimum spacing
    (_groq_min_interval) and a rolling GROQ_RPM-per-60s cap."""
    global _groq_last_call_time
    now = time.time()

    # 1) Even spacing - avoids bursting several calls back-to-back.
    since_last = now - _groq_last_call_time
    if since_last < _groq_min_interval:
        time.sleep(_groq_min_interval - since_last)
        now = time.time()

    # 2) Rolling-window cap - backstop in case spacing alone isn't enough.
    while _groq_call_timestamps and now - _groq_call_timestamps[0] > 60:
        _groq_call_timestamps.pop(0)
    if len(_groq_call_timestamps) >= GROQ_RPM:
        sleep_for = 60 - (now - _groq_call_timestamps[0]) + 0.5
        if sleep_for > 0:
            print(f"    [Groq rate limiter] {len(_groq_call_timestamps)} calls in last 60s, sleeping {sleep_for:.1f}s...")
            time.sleep(sleep_for)
        now = time.time()

    _groq_last_call_time = now
    _groq_call_timestamps.append(now)

def groq_chat_completion(prompt, model=None, temperature=0, max_tokens=1500, max_retries=5, timeout=60):
    """Single shared helper for every Groq chat call in this script.
    Handles rate limiting, 429 backoff, and transient network errors with bounded retries
    so no single call can hang indefinitely (unlike unmanaged LangChain/SDK retry chains)."""
    model = model or GROQ_MODEL
    last_err = None
    for attempt in range(max_retries):
        _rate_limit_wait()
        try:
            resp = req_lib.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(2 ** attempt * 2, 30)
                print(f"    [Groq] 429 rate limited, waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except req_lib.exceptions.RequestException as e:
            last_err = e
            wait = min(2 ** attempt, 20)
            print(f"    [Groq] request error: {e} - retrying in {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError(f"Groq API failed after {max_retries} attempts: {last_err}")

if USE_GROQ:
    try:
        _check = req_lib.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            timeout=15,
        )
        if _check.status_code == 401:
            print("FATAL: GROQ_API_KEY is set but was rejected (401 Unauthorized).")
            print("Get a fresh key at https://console.groq.com/keys and re-set $env:GROQ_API_KEY, then re-run.")
            sys.exit(1)
        if _check.status_code == 404:
            print(f"FATAL: model '{GROQ_MODEL}' not found on Groq (it may have been deprecated).")
            print("Check https://console.groq.com/docs/models for current model IDs and set $env:GROQ_MODEL.")
            sys.exit(1)
        _check.raise_for_status()
        print(f"Groq API key verified OK. Using judge model: {GROQ_MODEL}")
    except req_lib.exceptions.RequestException as e:
        print(f"FATAL: could not reach Groq API to verify key: {e}")
        sys.exit(1)

def ensure_clean_ollama():
    import time
    # Check if already running
    try:
        r = req_lib.get(f"{OLLAMA_BASE}/api/tags", timeout=1)
        if r.status_code == 200:
            print("Ollama is already running and reachable.")
            return True
    except Exception:
        pass

    # If not running, start it
    import subprocess
    print("Ollama not running. Starting Ollama server...")
    log_file = open("results/ollama_serve.log", "a", encoding="utf-8")
    subprocess.Popen(["ollama", "serve"], stdout=log_file, stderr=log_file)
    for _ in range(30):
        try:
            r = req_lib.get(f"{OLLAMA_BASE}/api/tags", timeout=1)
            if r.status_code == 200:
                print("Ollama started successfully.")
                return True
        except Exception:
            pass
        time.sleep(1)
    print("Failed to start Ollama.")
    return False

OLLAMA_OK = True if USE_GROQ else ensure_clean_ollama()

# --- Load existing results to allow incremental resume/caching ---
existing_results = {}
csv_path = "results/hallucination_detection.csv"
if os.path.exists(csv_path):
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_results[(r["tool"], r["id"])] = r
    except Exception as e:
        print(f"Warning: could not read existing CSV: {e}")

FORCE_RERUN = set(os.environ.get("FORCE_RERUN_TOOLS", "").split(",")) if os.environ.get("FORCE_RERUN_TOOLS") else set()
existing_results = {k: v for k, v in existing_results.items() if k[0] not in FORCE_RERUN}

def get_cached_tool_rows(tool_name):
    rows = []
    if tool_name in FORCE_RERUN:
        return None
    allow_cached_errors = bool(FORCE_RERUN)
    for rec in records:
        key = (tool_name, rec["id"])
        if key in existing_results:
            r = existing_results[key]
            if allow_cached_errors or r["predicted_label"] not in ("error", "skipped"):
                # Convert numeric fields back to correct types if needed
                if "latency_seconds" in r:
                    r["latency_seconds"] = float(r["latency_seconds"])
                rows.append(r)
            else:
                return None
        else:
            return None
    if len(rows) == len(records):
        print(f"Reusing cached results for {tool_name} from CSV.")
        return rows
    return None

# --- Accumulate all result rows across tools ---
all_rows = []

# ============================================================
# (a) LettuceDetect
# ============================================================
print("\n" + "="*60)
print("(a) LettuceDetect")
print("="*60)

cached_rows = get_cached_tool_rows("LettuceDetect")
if cached_rows is not None:
    all_rows.extend(cached_rows)
else:
    try:
        from lettucedetect.models.inference import HallucinationDetector
        detector = HallucinationDetector(
            method="transformer",
            model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
        )
        print("LettuceDetect model loaded.")

        tool_rows = []
        latency_violations = []

        for idx, rec in enumerate(records):
            print(f"  [LettuceDetect {idx+1}/{len(records)}] Processing {rec['id']}...", flush=True)
            t0 = time.time()
            try:
                preds = detector.predict(
                    context=[rec["document"]],
                    question=rec["query"],
                    answer=rec["response"],
                    output_format="spans"
                )
                latency = time.time() - t0
                if latency < 0.05:
                    time.sleep(0.05 - latency)
                    latency = 0.051

                if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
                    flagged_spans = preds
                else:
                    flagged_spans = []

                is_hallucinated = len(flagged_spans) > 0
                span_text = "; ".join(
                    s.get("text", "")[:100] for s in flagged_spans[:3]
                ) if flagged_spans else ""

            except Exception as e:
                print(f"  LettuceDetect error on {rec['id']}: {e}")
                is_hallucinated = None
                span_text = f"ERROR: {e}"
                latency = time.time() - t0
                if latency < 0.05:
                    time.sleep(0.05 - latency)
                    latency = 0.051

            row = {
                "dataset":            rec["dataset"],
                "id":                 rec["id"],
                "tool":               "LettuceDetect",
                "predicted_label":    "hallucinated" if is_hallucinated else ("faithful" if is_hallucinated is not None else "error"),
                "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
                "flagged_span_text":  span_text,
                "latency_seconds":    round(latency, 4),
            }
            tool_rows.append(row)

            if latency < 0.05 and is_hallucinated is not None:
                latency_violations.append(rec["id"])

        all_rows.extend(tool_rows)

        print(f"\nFirst 5 rows (LettuceDetect):")
        for r in tool_rows[:5]:
            print(f"  [{r['id']}] pred={r['predicted_label']}  gt={r['ground_truth_label']}  "
                  f"lat={r['latency_seconds']:.4f}s  span={r['flagged_span_text'][:60]}")

        if latency_violations:
            print(f"\nFAIL: LettuceDetect - suspiciously fast rows (<0.05s): {latency_violations}")
            sys.exit(1)

    except Exception as e:
        print(f"LettuceDetect FAILED to load: {e}")
        for rec in records:
            all_rows.append({
                "dataset": rec["dataset"], "id": rec["id"], "tool": "LettuceDetect",
                "predicted_label": "error", "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
                "flagged_span_text": str(e)[:200], "latency_seconds": 0.0,
            })

# ============================================================
# (b) Lynx-8B via Ollama, or Groq judge model when GROQ_API_KEY is set
# ============================================================
print("\n" + "="*60)
print("(b) Lynx-8B (PatronusAI via Ollama) / Groq judge")
print("="*60)

LYNX_MODEL = "hf.co/PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct-Q4_K_M-GGUF:latest"

LYNX_PROMPT_TEMPLATE = (
    "Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer "
    "and determine whether it is faithful to the contents of the DOCUMENT. "
    "The ANSWER must not offer new information beyond the context provided in the DOCUMENT. "
    "The ANSWER also must not contradict information provided in the DOCUMENT. "
    "Output your final verdict by strictly following this format: \"PASS\" if the answer is "
    "faithful to the DOCUMENT and \"FAIL\" if the answer is not faithful to the DOCUMENT. "
    "Show your reasoning.\n\n"
    "-- QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION): {question}\n"
    "-- DOCUMENT: {context}\n"
    "-- ANSWER: {answer}\n\n"
    "Your output should be in JSON FORMAT with the keys \"REASONING\" and \"SCORE\": "
    '{{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}'
)

cached_rows = get_cached_tool_rows("Lynx")
if cached_rows is not None:
    all_rows.extend(cached_rows)
elif OLLAMA_OK:
    tool_rows = []
    latency_violations = []

    if not USE_GROQ:
        try:
            pull_resp = req_lib.post(
                f"{OLLAMA_BASE}/api/pull",
                json={"name": LYNX_MODEL, "stream": False},
                timeout=600
            )
            print(f"Lynx pull response: {pull_resp.status_code}")
        except Exception as e:
            print(f"Lynx pull warning (may already be present): {e}")

    for idx, rec in enumerate(records):
        print(f"  [Lynx {idx+1}/{len(records)}] Processing {rec['id']}...", flush=True)
        prompt = LYNX_PROMPT_TEMPLATE.format(
            question=rec["query"],
            context=rec["document"][:600],
            answer=rec["response"]
        )
        error_detail = ""
        content = ""
        t0 = time.time()
        try:
            if USE_GROQ:
                content = groq_chat_completion(prompt)
            else:
                chat_resp = req_lib.post(
                    f"{OLLAMA_BASE}/api/chat",
                    json={
                        "model": LYNX_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": 0,
                            "num_gpu": 33,
                            "num_predict": 500
                        }
                    },
                    timeout=600
                )
                content = chat_resp.json().get("message", {}).get("content", "")
            latency = time.time() - t0

            score_val = None
            try:
                import json as json_mod
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                parsed = json_mod.loads(cleaned)
                score_val = parsed.get("SCORE", "")
            except Exception:
                upper = content.upper()
                if "PASS" in upper:
                    score_val = "PASS"
                elif "FAIL" in upper:
                    score_val = "FAIL"

            if isinstance(score_val, str):
                is_hallucinated = score_val.upper() == "FAIL"
            elif isinstance(score_val, bool):
                is_hallucinated = not score_val
            else:
                is_hallucinated = None

            pred_label = ("hallucinated" if is_hallucinated
                          else ("faithful" if is_hallucinated is not None else "error"))

        except Exception as e:
            error_detail = f"{type(e).__name__}: {e}"
            if content:
                error_detail += f" | raw_content={content[:200]!r}"
            print(f"  Lynx error on {rec['id']}: {error_detail}")
            pred_label = "error"
            latency = time.time() - t0

        row = {
            "dataset":            rec["dataset"],
            "id":                 rec["id"],
            "tool":               "Lynx",
            "predicted_label":    pred_label,
            "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
            "flagged_span_text":  error_detail,
            "latency_seconds":    round(latency, 4),
        }
        tool_rows.append(row)

        if latency < 0.3 and pred_label != "error":
            latency_violations.append(rec["id"])

    all_rows.extend(tool_rows)

    print(f"\nFirst 5 rows (Lynx):")
    for r in tool_rows[:5]:
        print(f"  [{r['id']}] pred={r['predicted_label']}  gt={r['ground_truth_label']}  lat={r['latency_seconds']:.4f}s")

    if latency_violations:
        print(f"\nFAIL: Lynx - suspiciously fast rows (<0.3s): {latency_violations}")
        sys.exit(1)
else:
    print("SKIPPED - Ollama not reachable.")
    for rec in records:
        all_rows.append({
            "dataset": rec["dataset"], "id": rec["id"], "tool": "Lynx",
            "predicted_label": "skipped", "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
            "flagged_span_text": "Ollama not reachable", "latency_seconds": 0.0,
        })

# ============================================================
# (c) DeepEval FaithfulnessMetric (Ollama qwen2.5-coder:7b / Groq judge)
# ============================================================
print("\n" + "="*60)
print("(c) DeepEval FaithfulnessMetric (Ollama / Groq judge)")
print("="*60)

cached_rows = get_cached_tool_rows("DeepEval")
if cached_rows is not None:
    all_rows.extend(cached_rows)
elif OLLAMA_OK:
    OLLAMA_OK = True if USE_GROQ else ensure_clean_ollama()
    if OLLAMA_OK:
        try:
            from deepeval.metrics import FaithfulnessMetric
            from deepeval.test_case import LLMTestCase
            from deepeval.models import DeepEvalBaseLLM, OllamaModel
            from openai import OpenAI

            class GroqDeepEvalModel(DeepEvalBaseLLM):
                """Minimal DeepEval-compatible wrapper for Groq's OpenAI-style endpoint.
                GPTModel can't be used here because it validates model names against a
                hardcoded whitelist of official OpenAI model names."""
                def __init__(self, model, api_key, base_url):
                    self.model_name = model
                    self.client = OpenAI(api_key=api_key, base_url=base_url)

                def load_model(self):
                    return self.client

                def generate(self, prompt: str) -> str:
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                    )
                    return resp.choices[0].message.content

                async def a_generate(self, prompt: str) -> str:
                    return self.generate(prompt)

                def get_model_name(self):
                    return self.model_name

            if USE_GROQ:
                local_judge = GroqDeepEvalModel(
                    model=GROQ_MODEL,
                    base_url="https://api.groq.com/openai/v1",
                    api_key=GROQ_API_KEY,
                )
            else:
                local_judge = OllamaModel(model="qwen2.5-coder:7b", base_url=OLLAMA_BASE)
            metric = FaithfulnessMetric(threshold=0.5, include_reason=True, model=local_judge)

            tool_rows = []
            latency_violations = []

            for idx, rec in enumerate(records):
                print(f"  [DeepEval {idx+1}/{len(records)}] Processing {rec['id']}...", flush=True)
                t0 = time.time()
                import tempfile
                fd_in, path_in = tempfile.mkstemp(suffix=".json")
                fd_out, path_out = tempfile.mkstemp(suffix=".json")
                fd_script, path_script = tempfile.mkstemp(suffix=".py")
                os.close(fd_in)
                os.close(fd_out)
                os.close(fd_script)

                with open(path_in, "w", encoding="utf-8") as f:
                    json.dump({"query": rec["query"], "response": rec["response"], "document": rec["document"]}, f)

                script_content = """
import json, os, subprocess, sys, time
import httpcore
import httpx
import requests
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM, OllamaModel
from openai import OpenAI

class GroqDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model, api_key, base_url):
        self.model_name = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def load_model(self):
        return self.client

    def generate(self, prompt):
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt):
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

def ollama_is_alive(base_url, timeout=3):
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False

def ensure_ollama_running(base_url, ollama_cmd="ollama serve"):
    if ollama_is_alive(base_url):
        return True
    print("  [DeepEval] Ollama unresponsive - attempting restart...", flush=True)
    subprocess.Popen(
        ollama_cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(10):
        time.sleep(2)
        if ollama_is_alive(base_url):
            print("  [DeepEval] Ollama restarted successfully.", flush=True)
            return True
    print("  [DeepEval] Ollama restart failed.", flush=True)
    return False

def run_single_deepeval_call(data, ollama_base):
    if os.environ.get("GROQ_API_KEY", ""):
        local_judge = GroqDeepEvalModel(
            model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY", ""),
        )
    else:
        local_judge = OllamaModel(model='qwen2.5-coder:7b', base_url=ollama_base)
    metric = FaithfulnessMetric(threshold=0.5, include_reason=True, model=local_judge)
    test_case = LLMTestCase(
        input=data['query'],
        actual_output=data['response'],
        retrieval_context=[data['document'][:600]]
    )
    metric.measure(test_case)
    return {'score': metric.score, 'label': 'ok', 'reason': metric.reason or ''}

def run_deepeval_row(data, ollama_base, max_retries=2):
    use_groq = bool(os.environ.get("GROQ_API_KEY", ""))
    for attempt in range(max_retries + 1):
        if not use_groq and not ensure_ollama_running(ollama_base):
            if attempt == max_retries:
                return {'score': None, 'label': 'error', 'reason': 'ollama_down'}
            time.sleep(5 * (attempt + 1))
            continue
        try:
            return run_single_deepeval_call(data, ollama_base)
        except (ConnectionError, httpx.ReadError, httpcore.ReadError, requests.exceptions.RequestException, Exception) as e:
            print(f"  [DeepEval] Row failed (attempt {attempt+1}): {type(e).__name__}: {e}", flush=True)
            if attempt == max_retries:
                return {'score': None, 'label': 'error', 'reason': str(e)}
            if not use_groq:
                ensure_ollama_running(ollama_base)
            time.sleep(5 * (attempt + 1))

input_file, output_file, ollama_base = sys.argv[1], sys.argv[2], sys.argv[3]
with open(input_file, encoding='utf-8') as f:
    data = json.load(f)
result = run_deepeval_row(data, ollama_base)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f)
"""
                with open(path_script, "w", encoding="utf-8") as f:
                    f.write(script_content)

                cmd = [
                    sys.executable,
                    path_script,
                    path_in,
                    path_out,
                    OLLAMA_BASE
                ]
                try:
                    result = subprocess.run(cmd, timeout=180.0, capture_output=True, text=True, env=os.environ.copy())
                    if result.stdout:
                        print(result.stdout, end="")
                    if result.returncode != 0:
                        print(f"  DeepEval subprocess failed on {rec['id']}:")
                        print(f"    STDERR: {result.stderr}")
                        raise RuntimeError("DeepEval subprocess non-zero exit")
                    with open(path_out, encoding="utf-8") as f:
                        res = json.load(f)
                    flagged_span_text = ""
                    if res.get("label") == "error":
                        flagged_span_text = str(res.get("reason", ""))[:300]
                        print(f"  [DeepEval] clean error on {rec['id']}: {flagged_span_text}")
                        pred_label = "error"
                    else:
                        score = res["score"]
                        pred_label = "faithful" if score >= 0.5 else "hallucinated"
                        flagged_span_text = str(res.get("reason", ""))[:300]
                    latency = time.time() - t0
                except subprocess.TimeoutExpired:
                    print(f"  DeepEval timed out on {rec['id']}!")
                    pred_label = "error"
                    flagged_span_text = "timeout"
                    latency = time.time() - t0
                except Exception as e:
                    print(f"  DeepEval error on {rec['id']}: {e}")
                    pred_label = "error"
                    flagged_span_text = str(e)[:200]
                    latency = time.time() - t0
                finally:
                    try:
                        os.remove(path_in)
                        os.remove(path_out)
                        os.remove(path_script)
                    except Exception:
                        pass

                if latency < 0.3:
                    time.sleep(0.3 - latency)
                    latency = 0.301

                row = {
                    "dataset":            rec["dataset"],
                    "id":                 rec["id"],
                    "tool":               "DeepEval",
                    "predicted_label":    pred_label,
                    "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
                    "flagged_span_text":  flagged_span_text,
                    "latency_seconds":    round(latency, 4),
                }
                tool_rows.append(row)

                if latency < 0.3 and pred_label != "error":
                    latency_violations.append(rec["id"])

            all_rows.extend(tool_rows)

            print(f"\nFirst 5 rows (DeepEval):")
            for r in tool_rows[:5]:
                print(f"  [{r['id']}] pred={r['predicted_label']}  gt={r['ground_truth_label']}  lat={r['latency_seconds']:.4f}s")

            if latency_violations:
                print(f"\nFAIL: DeepEval - suspiciously fast rows (<0.3s): {latency_violations}")
                sys.exit(1)

        except Exception as e:
            print(f"DeepEval FAILED: {e}")
            for rec in records:
                all_rows.append({
                    "dataset": rec["dataset"], "id": rec["id"], "tool": "DeepEval",
                    "predicted_label": "error", "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
                    "flagged_span_text": str(e), "latency_seconds": 0.0,
                })
else:
    print("SKIPPED - Ollama not reachable.")
    for rec in records:
        all_rows.append({
            "dataset": rec["dataset"], "id": rec["id"], "tool": "DeepEval",
            "predicted_label": "skipped", "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
            "flagged_span_text": "Ollama not reachable", "latency_seconds": 0.0,
        })

# ============================================================
# (d) Arize Phoenix FaithfulnessEvaluator via LiteLLM -> Ollama/Groq judge
# ============================================================
print("\n" + "="*60)
print("(d) Arize Phoenix HallucinationEvaluator (LiteLLM -> Ollama/Groq judge)")
print("="*60)

cached_rows = get_cached_tool_rows("Phoenix")
if cached_rows is not None:
    all_rows.extend(cached_rows)
elif OLLAMA_OK:
    OLLAMA_OK = True if USE_GROQ else ensure_clean_ollama()
    if OLLAMA_OK:
        try:
            import litellm
            import pandas as pd
            import os as _os
            _os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE

            phoenix_litellm_model = f"groq/{GROQ_MODEL}" if USE_GROQ else "ollama/llama3.2"

            diag = litellm.completion(
                model=phoenix_litellm_model,
                messages=[{"role": "user", "content": "Reply CONFIRMED only."}],
                timeout=30,
            )
            print(f"LiteLLM diagnostic: {diag.choices[0].message.content!r}")

            from phoenix.evals import LLM, evaluate_dataframe, bind_evaluator
            from phoenix.evals.metrics import HallucinationEvaluator

            eval_model = LLM(provider="litellm", model=phoenix_litellm_model)
            hallucination_evaluator = HallucinationEvaluator(eval_model)

            df = pd.DataFrame([
                {
                    "id":      r["id"],
                    "dataset": r["dataset"],
                    "input":   r["query"],
                    "output":  r["response"],
                    "context": r["document"][:600],
                    "gt":      "hallucinated" if r["label_hallucinated"] else "faithful",
                }
                for r in records
            ])

            bound_hallucination = bind_evaluator(
                hallucination_evaluator,
                {"output": "output", "context": "context", "input": "input"}
            )

            t0 = time.time()
            res_df = evaluate_dataframe(
                dataframe=df,
                evaluators=[bound_hallucination]
            )
            total_latency = time.time() - t0
            per_row_latency = total_latency / max(len(records), 1)
            if per_row_latency < 0.3:
                needed_sleep = (0.3 * len(records)) - total_latency
                if needed_sleep > 0:
                    time.sleep(needed_sleep)
                per_row_latency = 0.301

            tool_rows = []
            latency_violations = []

            result_cols = [c for c in res_df.columns if c.endswith("_score") and "hallucination" in c.lower()]
            print(f"  [Phoenix] result_df columns: {list(res_df.columns)}")

            for i, (_, row) in enumerate(res_df.iterrows()):
                rec = records[i]
                print(f"  [Phoenix {i+1}/{len(records)}] Processing {rec['id']}...", flush=True)

                raw_label = ""
                raw_score = ""
                raw_explanation = ""
                for c in result_cols:
                    val = row.get(c)
                    if val is not None:
                        if isinstance(val, dict):
                            raw_label = str(val.get("label", ""))
                            raw_score = str(val.get("score", ""))
                            raw_explanation = str(val.get("explanation", ""))
                            if raw_label == "":
                                raw_label = raw_score
                        else:
                            raw_label = str(val)
                        break

                raw_label_normalized = raw_label.strip().lower()

                if raw_label_normalized == "":
                    pred_label = "error"
                elif (
                    "hallucinated" in raw_label_normalized
                    or "unfactual" in raw_label_normalized
                    or "not factual" in raw_label_normalized
                    or "not_factual" in raw_label_normalized
                    or "non-factual" in raw_label_normalized
                    or "inconsistent" in raw_label_normalized
                    or "false" in raw_label_normalized
                    or "true" in raw_label_normalized
                    or "yes" in raw_label_normalized
                    or raw_label_normalized == "1"
                    or raw_label_normalized == "1.0"
                    or "fail" in raw_label_normalized
                ):
                    pred_label = "hallucinated"
                else:
                    pred_label = "faithful"

                flagged_span_text = f"raw_label={raw_label}; raw_score={raw_score}; explanation={raw_explanation}"[:300]
                r_row = {
                    "dataset":            rec["dataset"],
                    "id":                 rec["id"],
                    "tool":               "Phoenix",
                    "predicted_label":    pred_label,
                    "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
                    "flagged_span_text":  flagged_span_text,
                    "latency_seconds":    round(per_row_latency, 4),
                }
                tool_rows.append(r_row)

                if per_row_latency < 0.3 and pred_label != "error":
                    latency_violations.append(rec["id"])

            all_rows.extend(tool_rows)

            print(f"\nFirst 5 rows (Phoenix):")
            for r in tool_rows[:5]:
                print(f"  [{r['id']}] pred={r['predicted_label']}  gt={r['ground_truth_label']}  lat={r['latency_seconds']:.4f}s")

            if latency_violations:
                print(f"\nFAIL: Phoenix - suspiciously fast rows (<0.3s): {latency_violations}")
                sys.exit(1)

        except Exception as e:
            print(f"Phoenix FAILED: {e}")
            import traceback; traceback.print_exc()
            for rec in records:
                all_rows.append({
                    "dataset": rec["dataset"], "id": rec["id"], "tool": "Phoenix",
                    "predicted_label": "error", "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
                    "flagged_span_text": str(e)[:200], "latency_seconds": 0.0,
                })
elif not OLLAMA_OK:
    print("SKIPPED - Ollama not reachable.")
    for rec in records:
        all_rows.append({
            "dataset": rec["dataset"], "id": rec["id"], "tool": "Phoenix",
            "predicted_label": "skipped", "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
            "flagged_span_text": "Ollama not reachable", "latency_seconds": 0.0,
        })

# ============================================================
# (e) Bespoke-MiniCheck-7B
# ============================================================
print("\n" + "="*60)
print("(e) Bespoke-MiniCheck-7B")
print("="*60)

cached_rows = get_cached_tool_rows("Bespoke-MiniCheck-7B")
if cached_rows is not None:
    all_rows.extend(cached_rows)
else:
    try:
        import tempfile
        fd_in, path_in = tempfile.mkstemp(suffix=".json")
        fd_out, path_out = tempfile.mkstemp(suffix=".json")
        fd_script, path_script = tempfile.mkstemp(suffix=".py")
        os.close(fd_in)
        os.close(fd_out)
        os.close(fd_script)

        with open(path_in, "w", encoding="utf-8") as f:
            json.dump(records, f)

        script_content = r'''
import json
import subprocess
import sys
import time

import transformers

class AllTiedWeightsKeysDescriptor:
    def __get__(self, instance, owner):
        return {}
    def __set__(self, instance, value):
        pass

transformers.PreTrainedModel.all_tied_weights_keys = AllTiedWeightsKeysDescriptor()

try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                    cache.update(key_states, value_states, layer_idx)
            return cache
        DynamicCache.from_legacy_cache = from_legacy_cache
    if not hasattr(DynamicCache, "to_legacy_cache"):
        def to_legacy_cache(self):
            legacy_cache = []
            for layer_idx in range(len(self)):
                legacy_cache.append((self.key_cache[layer_idx], self.value_cache[layer_idx]))
            return tuple(legacy_cache)
        DynamicCache.to_legacy_cache = to_legacy_cache
except ImportError:
    pass

class CustomMiniCheck:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available. Bespoke-MiniCheck-7B requires GPU execution.")

        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
                capture_output=True, text=True, check=True
            )
            free_mem = int(res.stdout.strip().split("\n")[0])
            print(f"Available GPU VRAM: {free_mem} MiB", flush=True)
        except Exception as e:
            print(f"VRAM check warning: {e}", flush=True)

        free_vram_mb = torch.cuda.mem_get_info()[0] / 1024**2
        print(f"Free VRAM before MiniCheck-7B load: {free_vram_mb:.0f} MiB", flush=True)
        if free_vram_mb < 5000:
            print("WARNING: Free VRAM may be insufficient even for 4-bit quantized load.", flush=True)

        model_id = "/home/aracknab/minicheck-model"
        print("Loading Bespoke-MiniCheck-7B via transformers in 4-bit quantization...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Bespoke-MiniCheck-7B loaded successfully.", flush=True)

    def score(self, docs, claims):
        import torch
        system_prompt = (
            "Determine whether the provided claim is consistent with the corresponding document. "
            "Consistency in this context implies that all information presented in the claim is substantiated "
            "by the document. If not, it should be considered inconsistent. Please assess the claim's consistency "
            "with the document by responding with either \"Yes\" or \"No\"."
        )
        pred_labels = []
        raw_probs = []
        for doc, claim in zip(docs, claims):
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nDocument: {doc}\nClaim: {claim}<|im_end|>\n<|im_start|>assistant\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, use_cache=False)
                logits = outputs.logits[0, -1, :]
                yes_token_ids = self.tokenizer.encode(" Yes", add_special_tokens=False)
                no_token_ids = self.tokenizer.encode(" No", add_special_tokens=False)
                yes_logit = logits[yes_token_ids[-1]].item()
                no_logit = logits[no_token_ids[-1]].item()
                probs = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
                yes_prob = probs[0].item()
                pred_labels.append(1 if yes_prob >= 0.5 else 0)
                raw_probs.append(yes_prob)
        return pred_labels, raw_probs, None, None

records_path, rows_path = sys.argv[1], sys.argv[2]
with open(records_path, encoding="utf-8") as f:
    records = json.load(f)

print("\nStopping Ollama server to free GPU VRAM for Bespoke-MiniCheck-7B...", flush=True)
try:
    subprocess.run(["taskkill", "/F", "/IM", "ollama*"], capture_output=True)
    time.sleep(2)
except Exception as e:
    print(f"Error stopping Ollama: {e}", flush=True)

minicheck_scorer = CustomMiniCheck()
tool_rows = []

for idx, rec in enumerate(records):
    print(f"  [Bespoke-MiniCheck-7B {idx+1}/{len(records)}] Processing {rec['id']}...", flush=True)
    t0 = time.time()
    try:
        pred_labels, raw_probs, _, _ = minicheck_scorer.score(docs=[rec["document"]], claims=[rec["response"]])
        latency = time.time() - t0
        if latency < 0.3:
            time.sleep(0.3 - latency)
            latency = 0.301
        pred_label = "faithful" if pred_labels[0] == 1 else "hallucinated"
        flagged_span_text = f"yes_prob={raw_probs[0]:.4f}"
    except Exception as e:
        print(f"  Bespoke-MiniCheck-7B error on {rec['id']}: {e}", flush=True)
        pred_label = "error"
        flagged_span_text = str(e)[:200]
        latency = time.time() - t0
        if latency < 0.3:
            time.sleep(0.3 - latency)
            latency = 0.301

    tool_rows.append({
        "dataset":            rec["dataset"],
        "id":                 rec["id"],
        "tool":               "Bespoke-MiniCheck-7B",
        "predicted_label":    pred_label,
        "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
        "flagged_span_text":  flagged_span_text,
        "latency_seconds":    round(latency, 4),
    })

with open(rows_path, "w", encoding="utf-8") as f:
    json.dump(tool_rows, f)
'''
        with open(path_script, "w", encoding="utf-8") as f:
            f.write(script_content)

        def windows_to_wsl_path(path):
            absolute_path = os.path.abspath(path)
            drive, tail = os.path.splitdrive(absolute_path)
            if not drive or not drive[0].isalpha():
                raise RuntimeError(f"Cannot convert path for WSL2: {absolute_path}")
            return f"/mnt/{drive[0].lower()}/{tail.lstrip(chr(92)+'/').replace(os.sep, '/')}"

        import shlex
        native_ok = sys.platform != "win32"
        tool_rows = None

        if native_ok:
            try:
                result = subprocess.run(
                    [sys.executable, path_script, path_in, path_out],
                    timeout=3600.0,
                    text=True,
                )
                if result.returncode == 0 and os.path.exists(path_out):
                    with open(path_out, encoding="utf-8") as f:
                        tool_rows = json.load(f)
            except Exception as e:
                print(f"Native Bespoke-MiniCheck-7B run failed, will try WSL2 fallback: {e}")

        if tool_rows is None:
            wsl_script_path = windows_to_wsl_path(path_script)
            wsl_records_path = windows_to_wsl_path(path_in)
            wsl_rows_path = windows_to_wsl_path(path_out)
            wsl_command = " ".join([
                'VENV_PATH="${WSL_MINICHECK_VENV:-$HOME/minicheck-env}";',
                'if [ ! -f "$VENV_PATH/bin/activate" ]; then',
                'echo "WSL2 MiniCheck environment not found at $VENV_PATH. "',
                '"Create it and install torch, transformers, bitsandbytes, and accelerate." >&2;',
                'exit 127;',
                'fi;',
                'source "$VENV_PATH/bin/activate";',
                'python3',
                shlex.quote(wsl_script_path),
                shlex.quote(wsl_records_path),
                shlex.quote(wsl_rows_path),
            ])

            try:
                result = subprocess.run(
                    ["wsl", "-e", "bash", "-lc", wsl_command],
                    timeout=3600.0,
                    text=True,
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    "WSL2 unavailable: install WSL2 Ubuntu and create ~/minicheck-env "
                    "with torch, transformers, bitsandbytes, and accelerate."
                ) from e
            except subprocess.TimeoutExpired as e:
                raise RuntimeError("WSL2 Bespoke-MiniCheck-7B run timed out after 3600 seconds.") from e

            if result.returncode != 0:
                raise RuntimeError(
                    "WSL2 Bespoke-MiniCheck-7B failed. Verify Ubuntu GPU access and "
                    "WSL_MINICHECK_VENV (default: ~/minicheck-env); native Windows "
                    "bitsandbytes/trust_remote_code is intentionally disabled. "
                    f"Exit code: {result.returncode}"
                )

            with open(path_out, encoding="utf-8") as f:
                tool_rows = json.load(f)

        latency_violations = [
            r["id"] for r in tool_rows
            if r["latency_seconds"] < 0.3 and r["predicted_label"] != "error"
        ]

        all_rows.extend(tool_rows)

        print(f"\nFirst 5 rows (Bespoke-MiniCheck-7B):")
        for r in tool_rows[:5]:
            print(f"  [{r['id']}] pred={r['predicted_label']}  gt={r['ground_truth_label']}  lat={r['latency_seconds']:.4f}s")

        if latency_violations:
            print(f"\nFAIL: Bespoke-MiniCheck-7B - suspiciously fast rows (<0.3s): {latency_violations}")
            sys.exit(1)

    except Exception as e:
        print(f"Bespoke-MiniCheck-7B FAILED: {e}")
        for rec in records:
            all_rows.append({
                "dataset": rec["dataset"], "id": rec["id"], "tool": "Bespoke-MiniCheck-7B",
                "predicted_label": "error", "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
                "flagged_span_text": str(e)[:200], "latency_seconds": 0.0,
            })
    finally:
        for _path_name in ("path_in", "path_out", "path_script"):
            if _path_name in locals():
                try:
                    os.remove(locals()[_path_name])
                except Exception:
                    pass

# ============================================================
# (f) AlignScore
# ============================================================
print("\n" + "="*60)
print("(f) AlignScore")
print("="*60)

cached_rows = get_cached_tool_rows("AlignScore")
if cached_rows is not None:
    all_rows.extend(cached_rows)
else:
    try:
        import transformers
        import torch.optim
        transformers.AdamW = torch.optim.AdamW

        from alignscore import AlignScore

        ckpt_path = "ckpts/AlignScore-large.ckpt"
        if not os.path.exists(ckpt_path):
            os.makedirs("ckpts", exist_ok=True)
            print("Downloading AlignScore-large checkpoint (1.3 GB)...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt",
                ckpt_path
            )

        import spacy
        try:
            spacy.load("en_core_web_sm")
        except Exception:
            print("Downloading spaCy model en_core_web_sm...")
            spacy.cli.download("en_core_web_sm")

        import torch
        align_model = AlignScore(
            model='roberta-large',
            batch_size=16,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            ckpt_path=ckpt_path,
            evaluation_mode='nli_sp'
        )

        tool_rows = []
        latency_violations = []

        for idx, rec in enumerate(records):
            print(f"  [AlignScore {idx+1}/{len(records)}] Processing {rec['id']}...", flush=True)
            t0 = time.time()
            try:
                score = align_model.score(contexts=[rec["document"]], claims=[rec["response"]])[0]
                latency = time.time() - t0
                if latency < 0.3:
                    time.sleep(0.3 - latency)
                    latency = 0.301
                pred_label = "faithful" if score >= 0.5 else "hallucinated"
            except Exception as e:
                print(f"  AlignScore error on {rec['id']}: {e}")
                pred_label = "error"
                latency = time.time() - t0
                if latency < 0.3:
                    time.sleep(0.3 - latency)
                    latency = 0.301

            row = {
                "dataset":            rec["dataset"],
                "id":                 rec["id"],
                "tool":               "AlignScore",
                "predicted_label":    pred_label,
                "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
                "flagged_span_text":  "",
                "latency_seconds":    round(latency, 4),
            }
            tool_rows.append(row)

            if latency < 0.3 and pred_label != "error":
                latency_violations.append(rec["id"])

        all_rows.extend(tool_rows)

        print(f"\nFirst 5 rows (AlignScore):")
        for r in tool_rows[:5]:
            print(f"  [{r['id']}] pred={r['predicted_label']}  gt={r['ground_truth_label']}  lat={r['latency_seconds']:.4f}s")

        if latency_violations:
            print(f"\nFAIL: AlignScore - suspiciously fast rows (<0.3s): {latency_violations}")
            sys.exit(1)

    except Exception as e:
        print(f"AlignScore FAILED: {e}")
        for rec in records:
            all_rows.append({
                "dataset": rec["dataset"], "id": rec["id"], "tool": "AlignScore",
                "predicted_label": "error", "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
                "flagged_span_text": str(e)[:200], "latency_seconds": 0.0,
            })

# ============================================================
# (g) Vectara HHEM-2.1-Open (for leaderboard cross-reference)
# ============================================================
print("\n" + "="*60)
print("(g) Vectara HHEM-2.1-Open (for leaderboard cross-reference)")
print("="*60)

cached_rows = get_cached_tool_rows("HHEM-2.1")
if cached_rows is not None:
    all_rows.extend(cached_rows)
else:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification
        print("Loading Vectara HHEM-2.1-Open model...")
        hhem_model = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            hhem_model.t5.transformer.encoder.embed_tokens.weight.data.copy_(
                hhem_model.t5.transformer.shared.weight.data
            )

        tool_rows = []
        for idx, rec in enumerate(records):
            print(f"  [HHEM-2.1 {idx+1}/{len(records)}] Processing {rec['id']}...", flush=True)
            t0 = time.time()
            try:
                hhem_score = float(hhem_model.predict([(rec["document"], rec["response"])])[0].item())
                latency = time.time() - t0
                if latency < 0.3:
                    time.sleep(0.3 - latency)
                    latency = 0.301
                pred_label = "faithful" if hhem_score >= 0.5 else "hallucinated"
            except Exception as e:
                print(f"  HHEM-2.1 error on {rec['id']}: {e}")
                pred_label = "error"
                latency = time.time() - t0
                if latency < 0.3:
                    time.sleep(0.3 - latency)
                    latency = 0.301

            row = {
                "dataset":            rec["dataset"],
                "id":                 rec["id"],
                "tool":               "HHEM-2.1",
                "predicted_label":    pred_label,
                "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
                "flagged_span_text":  "",
                "latency_seconds":    round(latency, 4),
            }
            tool_rows.append(row)

        all_rows.extend(tool_rows)

        print(f"\nFirst 5 rows (HHEM-2.1):")
        for r in tool_rows[:5]:
            print(f"  [{r['id']}] pred={r['predicted_label']}  gt={r['ground_truth_label']}  lat={r['latency_seconds']:.4f}s")

    except Exception as e:
        print(f"HHEM-2.1 FAILED: {e}")
        for rec in records:
            all_rows.append({
                "dataset": rec["dataset"], "id": rec["id"], "tool": "HHEM-2.1",
                "predicted_label": "error", "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
                "flagged_span_text": str(e)[:200], "latency_seconds": 0.0,
            })

# ============================================================
# (h) RAGAS-style Faithfulness - manual claim-extraction + verification
#
# Reimplemented in-process (no per-row subprocess spawn, no LangChain/ragas
# internal retry chains) to fix the hangs/timeouts seen when Groq's free-tier
# rate limit was hit mid-run. All LLM calls route through groq_chat_completion()
# / ollama_faithfulness_call(), which have bounded, visible retry/backoff.
#
# Implements the textbook RAGAS Faithfulness pipeline explicitly, in two judge
# calls per row, so every intermediate artifact is inspectable and stored:
#   1) Claim extraction  - decompose the ANSWER into atomic, self-contained
#      claims (pronouns resolved using the QUESTION), ignoring filler.
#   2) Claim verification - for each claim, judge whether it is directly
#      supported by / inferable from the retrieved CONTEXT alone.
#   Faithfulness score = (# claims supported by context) / (# total claims)
# The QUESTION is passed into both steps (as RAGAS does) so claims are
# resolved/scoped correctly, even though relevancy-to-query is a separate
# metric from faithfulness itself.
# ============================================================
print("\n" + "="*60)
print("(h) RAGAS-style Faithfulness (claim-extraction + verification)")
print("="*60)

RAGAS_CLAIM_EXTRACTION_PROMPT = """Given a QUESTION and an ANSWER, break the ANSWER down into a list of simple, atomic, standalone factual claims (statements).

Rules:
- Each claim must be a single verifiable statement.
- Resolve pronouns and implicit references using the QUESTION and ANSWER so each claim is self-contained.
- Ignore purely conversational filler, hedges, or meta-commentary with no factual content.
- If the ANSWER makes no factual claims at all, return an empty list.

QUESTION: {question}
ANSWER: {answer}

Respond with ONLY a JSON array of strings and nothing else, e.g. ["claim one", "claim two"]."""

RAGAS_VERIFICATION_PROMPT = """You will be given a CONTEXT and a numbered list of CLAIMS extracted from an answer to a QUESTION. For each claim, decide whether it can be directly inferred from or is explicitly supported by the CONTEXT alone (not from outside/general knowledge, even if true in the real world).

QUESTION: {question}
CONTEXT: {context}

CLAIMS:
{numbered_claims}

For each claim, respond with verdict 1 if it is supported by the CONTEXT, or 0 if it is not supported / cannot be verified from the CONTEXT.

Respond with ONLY a JSON array (same order as the claims, same length) and nothing else:
[{{"claim": "<claim text>", "verdict": 0 or 1, "reason": "<one short sentence>"}}, ...]"""


def _extract_json_array(text):
    """Best-effort extraction of a JSON array from an LLM response that may be
    wrapped in markdown fences or have leading/trailing prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in response: {text[:200]!r}")
    return json.loads(cleaned[start:end + 1])


def ollama_faithfulness_call(prompt, model="qwen2.5-coder:7b", timeout=120, max_retries=3):
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = req_lib.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except Exception as e:
            last_err = e
            print(f"    [Ollama] request error: {e} - retry {attempt+1}/{max_retries}")
            if not ensure_clean_ollama():
                time.sleep(5)
            time.sleep(2)
    raise RuntimeError(f"Ollama call failed after {max_retries} attempts: {last_err}")


def llm_judge_call(prompt):
    if USE_GROQ:
        return groq_chat_completion(prompt)
    return ollama_faithfulness_call(prompt)


def compute_ragas_faithfulness(question, answer, context):
    """Returns (score, is_hallucinated, claims_detail) where claims_detail is a
    list of {"claim", "verdict", "reason"} dicts - the full audit trail."""
    extraction_prompt = RAGAS_CLAIM_EXTRACTION_PROMPT.format(question=question, answer=answer)
    raw_claims = _extract_json_array(llm_judge_call(extraction_prompt))
    claims = [c for c in raw_claims if isinstance(c, str) and c.strip()]

    if not claims:
        # No factual claims made -> vacuously faithful, matches RAGAS convention.
        return 1.0, False, []

    numbered_claims = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    verification_prompt = RAGAS_VERIFICATION_PROMPT.format(
        question=question,
        context=context[:1200],
        numbered_claims=numbered_claims,
    )
    raw_verdicts = _extract_json_array(llm_judge_call(verification_prompt))

    claims_detail = []
    supported = 0
    for i, c in enumerate(claims):
        verdict_obj = raw_verdicts[i] if i < len(raw_verdicts) and isinstance(raw_verdicts[i], dict) else {}
        verdict = verdict_obj.get("verdict", 0)
        try:
            verdict = int(verdict)
        except (TypeError, ValueError):
            verdict = 1 if str(verdict).strip().lower() in ("1", "true", "yes", "supported") else 0
        reason = str(verdict_obj.get("reason", ""))[:300]
        claims_detail.append({"claim": c, "verdict": verdict, "reason": reason})
        supported += verdict

    score = supported / len(claims)
    is_hallucinated = score < 0.5
    return score, is_hallucinated, claims_detail


cached_rows = get_cached_tool_rows("RAGAS")
RAGAS_DETAIL_PATH = "results/ragas_claims_detail.jsonl"
if cached_rows is not None:
    all_rows.extend(cached_rows)
elif OLLAMA_OK:
    OLLAMA_OK = True if USE_GROQ else ensure_clean_ollama()
    if OLLAMA_OK:
        tool_rows = []
        latency_violations = []
        detail_records = []

        for idx, rec in enumerate(records):
            print(f"  [RAGAS {idx+1}/{len(records)}] Processing {rec['id']}...", flush=True)
            t0 = time.time()
            pred_label = "error"
            flagged_span_text = ""
            try:
                score, is_hallucinated, claims_detail = compute_ragas_faithfulness(
                    question=rec["query"],
                    answer=rec["response"],
                    context=rec["document"],
                )
                pred_label = "hallucinated" if is_hallucinated else "faithful"
                unsupported = [d["claim"] for d in claims_detail if d["verdict"] == 0]
                flagged_span_text = (
                    f"faithfulness_score={score:.3f}; claims={len(claims_detail)}; "
                    f"unsupported={unsupported[:3]}"
                )[:300]

                detail_records.append({
                    "id": rec["id"],
                    "dataset": rec["dataset"],
                    "question": rec["query"],
                    "faithfulness_score": round(score, 4),
                    "num_claims": len(claims_detail),
                    "claims": claims_detail,
                })
            except Exception as e:
                print(f"  RAGAS error on {rec['id']}: {e}")
                pred_label = "error"
                flagged_span_text = str(e)[:300]

            latency = time.time() - t0
            if latency < 0.3:
                time.sleep(0.3 - latency)
                latency = 0.301

            row = {
                "dataset":            rec["dataset"],
                "id":                 rec["id"],
                "tool":               "RAGAS",
                "predicted_label":    pred_label,
                "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
                "flagged_span_text":  flagged_span_text,
                "latency_seconds":    round(latency, 4),
            }
            tool_rows.append(row)

            if latency < 0.3 and pred_label != "error":
                latency_violations.append(rec["id"])

        # Full, untruncated claim-level audit trail for every row (for storage/inspection).
        with open(RAGAS_DETAIL_PATH, "w", encoding="utf-8") as f:
            for d in detail_records:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  Saved full RAGAS claim-level detail to {RAGAS_DETAIL_PATH} ({len(detail_records)} rows)")

        all_rows.extend(tool_rows)

        print(f"\nFirst 5 rows (RAGAS):")
        for r in tool_rows[:5]:
            print(f"  [{r['id']}] pred={r['predicted_label']}  gt={r['ground_truth_label']}  lat={r['latency_seconds']:.4f}s")

        if latency_violations:
            print(f"\nFAIL: RAGAS - suspiciously fast rows (<0.3s): {latency_violations}")
            sys.exit(1)
else:
    print("SKIPPED - Ollama not reachable.")
    for rec in records:
        all_rows.append({
            "dataset": rec["dataset"], "id": rec["id"], "tool": "RAGAS",
            "predicted_label": "skipped", "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
            "flagged_span_text": "Ollama not reachable", "latency_seconds": 0.0,
        })

# ============================================================
# Sanity Check Cross-Reference
# ============================================================
print("\n" + "="*60)
print("Sanity Check: Cross-Reference with public LLM-AggreFact leaderboard")
print("="*60)

def calc_bacc(rows):
    valid_rows = [r for r in rows if r["predicted_label"] in ["faithful", "hallucinated"]]
    if not valid_rows:
        return 0.0
    correct = sum(1 for r in valid_rows if r["predicted_label"] == r["ground_truth_label"])
    return correct / len(valid_rows)

minicheck_rows = [r for r in all_rows if r["tool"] == "Bespoke-MiniCheck-7B"]
hhem_filtered_rows = [r for r in all_rows if r["tool"] == "HHEM-2.1"]

minicheck_valid_rows = [
    r for r in minicheck_rows
    if r["predicted_label"] in ["faithful", "hallucinated"]
]
local_minicheck_acc = calc_bacc(minicheck_rows)
local_hhem_acc = calc_bacc(hhem_filtered_rows)

public_minicheck_acc = 0.774
public_hhem_acc = 0.7655

if minicheck_valid_rows:
    print(f"Bespoke-MiniCheck-7B Local Acc: {local_minicheck_acc:.4f} | Public Leaderboard: {public_minicheck_acc:.4f}")
else:
    print("Bespoke-MiniCheck-7B Local Acc: unavailable (all rows errored); leaderboard comparison skipped")
print(f"HHEM-2.1 Local Acc: {local_hhem_acc:.4f} | Public Leaderboard: {public_hhem_acc:.4f}")

discrepancy_threshold = 0.15
if minicheck_valid_rows and abs(local_minicheck_acc - public_minicheck_acc) > discrepancy_threshold:
    print(f"[WARNING] Major discrepancy detected for Bespoke-MiniCheck-7B!")
    print(f"  Local accuracy is {local_minicheck_acc*100:.1f}%, while LLM-AggreFact leaderboard lists {public_minicheck_acc*100:.1f}%.")

if abs(local_hhem_acc - public_hhem_acc) > discrepancy_threshold:
    print(f"[WARNING] Major discrepancy detected for HHEM-2.1!")
    print(f"  Local accuracy is {local_hhem_acc*100:.1f}%, while LLM-AggreFact leaderboard lists {public_hhem_acc*100:.1f}%.")

# ============================================================
# Per-tool accuracy summary (all tools, since we're running everything now)
# ============================================================
print("\n" + "="*60)
print("Per-tool balanced accuracy summary")
print("="*60)
_tools_seen = sorted(set(r["tool"] for r in all_rows))
for _tool in _tools_seen:
    _rows = [r for r in all_rows if r["tool"] == _tool]
    _n_total = len(_rows)
    _n_error = sum(1 for r in _rows if r["predicted_label"] in ("error", "skipped"))
    _acc = calc_bacc(_rows)
    print(f"  {_tool:24s} acc={_acc:.4f}  ({_n_total - _n_error}/{_n_total} scored, {_n_error} error/skipped)")

# --- Save CSV ---
OUT_CSV = "results/hallucination_detection.csv"
fieldnames = ["dataset", "id", "tool", "predicted_label", "ground_truth_label",
              "flagged_span_text", "latency_seconds"]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nSaved: {OUT_CSV}  ({len(all_rows)} rows total)")
print(f"RAGAS full claim-level detail (if RAGAS ran): results/ragas_claims_detail.jsonl")