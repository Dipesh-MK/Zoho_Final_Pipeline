import os
import sys
import json
import time
import csv
import re
import requests as req_lib

# --- Configuration & Setup ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_RPM = int(os.environ.get("GROQ_RPM", "18"))
GROQ_TPM = int(os.environ.get("GROQ_TPM", "7000")) # Groq free tier limit is 8000
DEBUG = os.environ.get("DEBUG", "0") == "1"

if not GROQ_API_KEY:
    print("FATAL: GROQ_API_KEY not set.")
    sys.exit(1)

os.makedirs("results", exist_ok=True)

# --- Data Loading ---
DATA_FILES = ["data/ragtruth.jsonl", "data/ragbench.jsonl", "data/delucionqa.jsonl", "data/mock.jsonl"]
records = []
for path in DATA_FILES:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip(): records.append(json.loads(line.strip()))
                
# Handle CLI limit limit
if "--limit" in sys.argv:
    limit_idx = sys.argv.index("--limit")
    records = records[:int(sys.argv[limit_idx + 1])]

print(f"Total records loaded: {len(records)}")

# --- Groq Token-Aware Rate Limiter ---
_groq_min_interval = 60.0 / GROQ_RPM
_groq_last_call_time = 0.0
_groq_token_usage_window = []

def _rate_limit_wait(estimated_tokens):
    global _groq_last_call_time, _groq_token_usage_window
    now = time.time()
    
    since_last = now - _groq_last_call_time
    if since_last < _groq_min_interval:
        time.sleep(_groq_min_interval - since_last)
        now = time.time()
        
    _groq_token_usage_window = [(ts, tk) for (ts, tk) in _groq_token_usage_window if now - ts < 60]
    used_tokens = sum(tk for ts, tk in _groq_token_usage_window)
    
    if used_tokens + estimated_tokens > GROQ_TPM and _groq_token_usage_window:
        sleep_for = 60 - (now - _groq_token_usage_window[0][0]) + 0.5
        if sleep_for > 0:
            print(f"    [Groq TPM limiter] ~{used_tokens} tokens used in last 60s (budget {GROQ_TPM}), next call needs ~{estimated_tokens} - sleeping {sleep_for:.1f}s...")
            time.sleep(sleep_for)
        now = time.time()
        
    _groq_last_call_time = now

def groq_chat_completion(prompt, max_tokens=1000, temperature=0, max_retries=5):
    estimated_tokens = len(prompt) // 4 + max_tokens
    for attempt in range(max_retries):
        _rate_limit_wait(estimated_tokens)
        try:
            payload = {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning_effort": "low" # Prevent reasoning tokens from blowing the budget
            }
            resp = req_lib.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 30)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            
            actual_tokens = data.get("usage", {}).get("total_tokens", estimated_tokens)
            _groq_token_usage_window.append((time.time(), actual_tokens))
            
            if data["choices"][0]["finish_reason"] == "length":
                print(f"    [Groq] WARNING: response truncated by max_tokens={max_tokens}")
                
            return data["choices"][0]["message"].get("content", "")
        except Exception as e:
            time.sleep(min(2 ** attempt, 20))
    return ""

def _extract_json_array(text):
    try:
        match = re.search(r'\[.*\]', text.strip(), re.DOTALL)
        if match: return json.loads(match.group(0))
        return json.loads(text.strip())
    except Exception:
        # Fallback string parsing for severely truncated JSON
        return [line.strip('", ') for line in text.split('\n') if line.strip().startswith('"')]

# --- DeepEval Logic ---
DEEPEVAL_CLAIM_PROMPT = """Extract a list of distinct, verifiable factual statements/claims from the following ANSWER. 
Filter out conversational filler, questions, or greetings. Return ONLY a valid JSON array of strings.

ANSWER:
{answer}

Format: ["claim 1", "claim 2", ...]"""

DEEPEVAL_VERIFY_PROMPT = """CONTEXT:
{context}

CLAIM TO VERIFY:
{claim}

Is this claim directly supported by and inferable from the CONTEXT above? 
Answer with exactly one word: "factual" or "hallucinated". 
Provide NO explanation or other text.

Verdict:"""

results = []
for idx, rec in enumerate(records):
    print(f"  [DeepEval {idx+1}/{len(records)}] Processing {rec['id']}...", flush=True)
    t0 = time.time()
    
    # 1. Extract
    claim_prompt = DEEPEVAL_CLAIM_PROMPT.format(answer=rec["response"])
    claim_raw = groq_chat_completion(claim_prompt, max_tokens=1000)
    claims = _extract_json_array(claim_raw)
    
    # 2. Verify
    supported_count = 0
    if not claims:
        verdict = "faithful" 
        score = 1.0
    else:
        for claim in claims:
            verify_prompt = DEEPEVAL_VERIFY_PROMPT.format(context=rec["document"][:1500], claim=claim)
            response = groq_chat_completion(verify_prompt, max_tokens=100).strip().lower()
            
            if "factual" in response[-20:] or "factual" in response:
                supported_count += 1
            elif "hallucinated" in response:
                pass
            else:
                supported_count += 1 # Default conservative
                
        score = supported_count / len(claims)
        verdict = "faithful" if score >= 0.5 else "hallucinated"
        
    lat = time.time() - t0
    gt = "hallucinated" if rec["label_hallucinated"] else "faithful"
    
    print(f"    Result: {verdict} (Score: {score:.2f}, Claims: {len(claims)}) in {lat:.2f}s")
    results.append({"id": rec["id"], "pred": verdict, "gt": gt, "score": score, "lat": lat})

# --- Save Results ---
with open("results/deepeval_only.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "pred", "gt", "score", "lat"])
    writer.writeheader()
    writer.writerows(results)
print(f"Saved: results/deepeval_only.csv ({len(results)} rows)")